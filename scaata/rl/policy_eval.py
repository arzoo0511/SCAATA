"""Runs a frozen, already-trained RL policy over historical data to get its
own implied action sequence — the piece needed to make the RL policy
accountable within the same Hedge/critique bookkeeping every strategy-pool
expert already gets (Phase 16, closing the critique -> RL feedback loop
that was previously one-way: critique's weights fed the meta-selector's
next training run, but the RL policy itself was trained once and never
scored by critique in return).

Deliberately does not touch `RobustTradingEnv` or run a real backtest here
— this only asks "what would the policy DO at each historical step," the
same "implied action" role `scaata.strategies.meta_selector.
attach_meta_implied_action` already plays for the meta-selector, not a
portfolio simulation (that's what `simulate_trajectory_for_choice` is for,
once the signal exists).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.rl.env import BUY, HOLD, SELL

ACTION_TO_SIGNAL = {HOLD: 0, BUY: 1, SELL: -1}


def compute_rl_implied_signals(model, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Steps a frozen recurrent policy through `df`'s rows in order (LSTM
    state carried forward across the whole sequence, `episode_start=True`
    only on the first row — matches the exact inference pattern
    `scaata.live.forward_test.run_daily_decision` and every ablation's test-
    time rollout already use), returning a `{-1, 0, 1}` signal array in the
    same convention as `scaata.strategies.pool`'s strategy signals, so the
    result can be used anywhere a pooled strategy's `signals` array is
    used (an `expert_losses_from_critique`/`simulate_trajectory_for_choice`
    input, or attached to a DataFrame for `critique_node` to pick up as an
    expert via `RL_IMPLIED_SIGNAL_COLUMN`).

    `model` only needs a `.predict(obs, state, episode_start, deterministic)
    -> (action, new_state)` method — any stable-baselines3-compatible
    policy (or a test double) works, no import of a concrete model class
    needed here.
    """
    states = df[feature_columns].values.astype(np.float32)
    lstm_state = None
    episode_start = np.array([True])

    actions = np.zeros(len(states), dtype=int)
    for i in range(len(states)):
        action, lstm_state = model.predict(states[i], state=lstm_state, episode_start=episode_start, deterministic=True)
        actions[i] = int(action)
        episode_start = np.array([False])

    return np.array([ACTION_TO_SIGNAL[a] for a in actions], dtype=int)


def attach_rl_implied_signal(df: pd.DataFrame, model, feature_columns: list[str], column: str | None = None) -> pd.DataFrame:
    """Adds the RL policy's implied signal as a column on `df`, the same
    "compute once, attach as a column, let downstream nodes pick it up"
    pattern already used for `meta_confidence`/`sentiment_score`/
    `novelty_score`."""
    from scaata.config import RL_IMPLIED_SIGNAL_COLUMN

    column = column or RL_IMPLIED_SIGNAL_COLUMN
    out = df.copy()
    out[column] = compute_rl_implied_signals(model, df, feature_columns)
    return out
