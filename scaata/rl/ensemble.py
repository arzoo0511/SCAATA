"""Ensemble uncertainty-aware sizing (Phase 14) — activates a hook that was
defined but never consumed anywhere in the codebase before this phase:
`config.DEFAULT_SEEDS = [0, 1, 2, 3, 4]`.

Trains `RecurrentPPO` across all 5 seeds (reusing `scaata.rl.train.train_ppo`
unchanged — no signature change needed, just a loop), then at inference
decomposes the ensemble's *disagreement* into a genuine "I don't know"
signal: below an agreement threshold, the system abstains (forces HOLD)
rather than manufacturing confidence from a coin-flip majority; otherwise
it sizes its position down proportional to measured uncertainty via
`RobustTradingEnv.step`'s `size_multiplier` kwarg (Phase 14's one required
env change).

Scope note, stated plainly: uncertainty here is the entropy of the
*empirical vote distribution* across ensemble members (a standard,
robust "variation ratio"-style disagreement measure that works with any
object exposing `.predict()`, including simple test doubles) — not the
full Lakshminarayanan-style decomposition into each member's own predictive
distribution, which would require reaching into `RecurrentPPO`'s internal
policy/LSTM-state API and is both more fragile and harder to verify
offline. Vote entropy is a legitimate, simpler proxy for the same
underlying idea (members disagreeing on the *action*, not just its
probability), and is what's actually implemented and tested here.
"""
from __future__ import annotations

import numpy as np

from scaata.config import (
    DEFAULT_SEEDS,
    ENSEMBLE_ABSTAIN_AGREEMENT,
    ENSEMBLE_KAPPA,
    ENSEMBLE_MIN_SIZE,
    PPO_TOTAL_TIMESTEPS,
)
from scaata.rl.env import RobustTradingEnv
from scaata.rl.train import train_ppo

HOLD, BUY, SELL = 0, 1, 2
N_ACTIONS = 3


def train_ppo_ensemble(
    train_df, feature_columns: list[str], seeds: list[int] = DEFAULT_SEEDS, total_timesteps: int = PPO_TOTAL_TIMESTEPS
) -> list:
    """Trains one `RecurrentPPO` per seed, identical hyperparameters except
    the seed. This is the concrete activation of `DEFAULT_SEEDS` — every
    other consumer of `DEFAULT_SEEDS` in this codebase (walk-forward,
    ablations) only loops seeds for reporting variance, never trains a
    genuine multi-member ensemble used together at inference time."""
    return [train_ppo(train_df, feature_columns, seed=seed, total_timesteps=total_timesteps) for seed in seeds]


def _entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


class EnsemblePolicy:
    """Wraps a list of trained policies (or any object exposing
    `.predict(obs, state=..., episode_start=..., deterministic=...)`) and
    decomposes their combined prediction into a majority action, an
    agreement fraction, and a normalized (`[0, 1]`) vote-entropy
    uncertainty term.
    """

    def __init__(self, members: list):
        if not members:
            raise ValueError("EnsemblePolicy requires at least one member")
        self.members = members

    def predict_with_uncertainty(self, obs, states: list | None = None, episode_start=None, deterministic: bool = True) -> dict:
        n = len(self.members)
        states = states if states is not None else [None] * n

        actions, new_states = [], []
        for member, member_state in zip(self.members, states):
            action, new_state = member.predict(obs, state=member_state, episode_start=episode_start, deterministic=deterministic)
            actions.append(int(np.asarray(action).reshape(-1)[0]))
            new_states.append(new_state)

        actions_arr = np.array(actions)
        counts = np.bincount(actions_arr, minlength=N_ACTIONS).astype(float)
        vote_probs = counts / counts.sum()
        majority_action = int(np.argmax(counts))
        agreement = float(counts[majority_action] / n)

        max_entropy = np.log(N_ACTIONS)
        epistemic_uncertainty = _entropy(vote_probs) / max_entropy

        return {
            "action": majority_action,
            "action_agreement": agreement,
            "epistemic_uncertainty": epistemic_uncertainty,
            "vote_probs": vote_probs,
            "member_actions": actions_arr,
            "new_states": new_states,
        }


def backtest_ensemble(
    ensemble: EnsemblePolicy,
    test_df,
    feature_columns: list[str],
    ticker: str,
    min_agreement: float = ENSEMBLE_ABSTAIN_AGREEMENT,
    kappa: float = ENSEMBLE_KAPPA,
    min_size: float = ENSEMBLE_MIN_SIZE,
):
    """Rolls the ensemble through `RobustTradingEnv`, same shape as
    `scaata.rl.train.backtest_ppo` but driven by `EnsemblePolicy` instead
    of a single model: each step's majority vote gets sized (or replaced
    with an abstain-HOLD) via `size_from_uncertainty` before being applied.
    The channel that matters most here isn't the entry-day sizing (see
    `scaata.rl.vol_targeting`'s docstring on why single-shot entry sizing
    is a weak lever given this env's buy-once-and-hold position model) --
    it's abstain-on-disagreement changing *entry timing*: a low-agreement
    day gets skipped instead of entered on, which single-seed training has
    no way to express at all. Returns `(equity, actions, agreements)`.
    """
    env = RobustTradingEnv(test_df, feature_columns, fixed_ticker=ticker)
    obs, _ = env.reset()
    done = False
    states = [None] * len(ensemble.members)
    episode_starts = np.ones((1,), dtype=bool)

    equity = [env.initial_cash]
    actions_taken = []
    agreements = []
    while not done:
        pred = ensemble.predict_with_uncertainty(obs, states=states, episode_start=episode_starts, deterministic=True)
        states = pred["new_states"]
        action, size = size_from_uncertainty(
            pred["action"], pred["epistemic_uncertainty"], pred["action_agreement"], min_agreement, kappa, min_size
        )
        obs, reward, done, _, info = env.step(action, size_multiplier=size)
        episode_starts = np.array([done], dtype=bool)
        actions_taken.append(action)
        agreements.append(pred["action_agreement"])
        equity.append(info["portfolio_value"])

    return np.array(equity), np.array(actions_taken), np.array(agreements)


def size_from_uncertainty(
    base_action: int,
    epistemic_uncertainty: float,
    action_agreement: float,
    min_agreement: float = ENSEMBLE_ABSTAIN_AGREEMENT,
    kappa: float = ENSEMBLE_KAPPA,
    min_size: float = ENSEMBLE_MIN_SIZE,
) -> tuple[int, float]:
    """Below `min_agreement`, abstain — force HOLD regardless of what the
    majority action was, a genuine "I don't know" rather than manufactured
    confidence from a bare-majority vote. Otherwise, size down
    proportional to `epistemic_uncertainty` (already normalized to
    `[0, 1]`), floored at `min_size` so a small amount of disagreement
    doesn't fully zero out an otherwise well-agreed-upon trade.
    """
    if action_agreement < min_agreement:
        return HOLD, 0.0
    size_multiplier = max(min_size, 1.0 - kappa * epistemic_uncertainty)
    return base_action, size_multiplier
