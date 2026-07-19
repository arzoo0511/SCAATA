"""Mixture-of-experts inference wrapper: routes each step to the calm or
stress specialist using the gate's *live* prediction only — never an
oracle/hindsight regime label. Drop-in compatible with the existing
backtest harness (mirrors `scaata.rl.train.backtest_ppo`'s loop shape).

Hard gating only (per the rebuild plan's recommendation to start simple):
exactly one expert's action is used per step, chosen by `RegimeGate`. Each
expert keeps its own LSTM recurrent state across the steps where it was
actually invoked; a hand-off from the other expert is treated as a fresh
episode-start for the newly-chosen expert's own recurrence, since LSTM
hidden states aren't meaningfully transferable between separately-trained
networks anyway.
"""
from __future__ import annotations

import numpy as np

from scaata.rl.env import RobustTradingEnv
from scaata.rl.moe.gating import RegimeGate


class MoEPolicy:
    def __init__(self, experts: dict[str, object], gate: RegimeGate, n_gate_features: int):
        self.experts = experts
        self.gate = gate
        self.n_gate_features = n_gate_features
        self._lstm_states = {name: None for name in experts}
        self._episode_starts = {name: np.ones((1,), dtype=bool) for name in experts}

    def reset(self):
        self._lstm_states = {name: None for name in self.experts}
        self._episode_starts = {name: np.ones((1,), dtype=bool) for name in self.experts}

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, str]:
        gate_features = np.asarray(obs[: self.n_gate_features]).reshape(1, -1)
        bucket = self.gate.predict(gate_features)[0]
        expert = self.experts[bucket]

        action, new_state = expert.predict(
            obs, state=self._lstm_states[bucket], episode_start=self._episode_starts[bucket], deterministic=deterministic
        )
        self._lstm_states[bucket] = new_state
        self._episode_starts[bucket] = np.array([False])
        return action, bucket


def backtest_moe(
    moe: MoEPolicy, test_df, feature_columns: list[str], ticker: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Same shape as `scaata.rl.train.backtest_ppo`, plus the sequence of
    gate decisions actually used (for reporting how often each expert was
    invoked, and for the fairness check that no oracle label was consulted)."""
    env = RobustTradingEnv(test_df, feature_columns, fixed_ticker=ticker)
    obs, _ = env.reset()
    moe.reset()
    done = False

    equity = [env.initial_cash]
    actions = []
    buckets_used = []
    while not done:
        action, bucket = moe.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        actions.append(int(action))
        buckets_used.append(bucket)
        equity.append(info["portfolio_value"])

    return np.array(equity), np.array(actions), buckets_used
