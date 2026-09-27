"""Hedge / multiplicative-weights signal combiner (Phase 10) — replaces
`critique_node`'s single binary rule (`if net-negative window: halve
whichever strategy was picked most`) with the standard "prediction with
expert advice" update (Cesa-Bianchi & Lugosi): every pooled strategy is
treated as a forecasting expert, and its weight is updated multiplicatively
by its own realized loss each round.

This is a genuinely different mechanism from a black-box neural ensemble:
it comes with a provable worst-case regret bound relative to the single
best expert in hindsight, and it's fully auditable — `hedge_regret_report`
lets you state each expert's cumulative regret as a number, rather than
trusting an opaque learned weight.

Loss is intentionally NOT just "did the expert call direction right"
(`expert_losses_from_returns`) — it also incorporates the same
drawdown/volatility/holding-time penalty vocabulary already used by the RL
reward and the critique loop (`expert_losses_from_critique`), so an expert
that called direction right via a reckless, high-drawdown path is still
penalized the same way the rest of this project scores risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.config import (
    DEFAULT_LAMBDA_DD,
    DEFAULT_LAMBDA_HOLD,
    DEFAULT_LAMBDA_VOL,
    HEDGE_LOSS_RHO,
    NOVELTY_ETA_KAPPA,
    NOVELTY_SENTIMENT_PRIOR_GAMMA,
)
from scaata.rl.reward import window_critique_report
from scaata.strategies.backtest_signal import simulate_trajectory_for_choice


def eta_from_horizon(n_experts: int, horizon: int) -> float:
    """Standard Hedge/Exponential-Weights tuning for losses in [0, 1]:
    `eta = sqrt(8 * ln(N) / T)`, which gives the textbook `O(sqrt(T ln N))`
    worst-case regret bound relative to the best expert in hindsight."""
    n_experts = max(int(n_experts), 2)
    horizon = max(int(horizon), 1)
    return float(np.sqrt(8 * np.log(n_experts) / horizon))


def _normalize_losses(losses: np.ndarray) -> np.ndarray:
    """Min-max normalizes losses to [0, 1] across experts for a single
    round — Hedge's regret bound assumes bounded losses, and experts here
    can have wildly different raw loss scales (a return-based loss vs. a
    penalty-based loss), so only the *relative ranking* within a round is
    meaningful, not the absolute magnitude."""
    losses = np.asarray(losses, dtype=float)
    lo, hi = np.nanmin(losses), np.nanmax(losses)
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-12:
        return np.zeros_like(losses)
    return np.clip((losses - lo) / (hi - lo), 0.0, 1.0)


class HedgeWeights:
    """Maintains a probability-simplex weight vector over `n_experts`,
    updated multiplicatively by `update()`. Stateless across calls by
    design — construct with the weights carried over from the previous
    round (e.g. from `AgentState["strategy_pool_weights"]`) rather than
    holding long-lived object identity, so it composes cleanly with the
    existing LangGraph state-dict pattern.
    """

    def __init__(self, n_experts: int, eta: float, weights: np.ndarray | None = None):
        self.n_experts = n_experts
        self.eta = eta
        if weights is None:
            self.weights = np.ones(n_experts) / n_experts
        else:
            weights = np.asarray(weights, dtype=float)
            total = weights.sum()
            self.weights = weights / total if total > 0 else np.ones(n_experts) / n_experts

    def update(self, losses: np.ndarray, eta_override: float | None = None, log_prior: np.ndarray | None = None) -> np.ndarray:
        """Applies one multiplicative-weights round given per-expert
        `losses` (any real scale — normalized internally to [0, 1]).
        `log_prior` (optional, same shape as `losses`) is an additive term
        in log-space applied *before* renormalization — e.g. Phase 10c's
        novelty-driven sentiment prior — kept separable from the
        data-driven multiplicative-weights term so each contribution stays
        auditable rather than blended into a single opaque number.
        """
        eta = eta_override if eta_override is not None else self.eta
        normalized = _normalize_losses(losses)
        log_weights = np.log(self.weights + 1e-300) - eta * normalized
        if log_prior is not None:
            log_weights = log_weights + np.asarray(log_prior, dtype=float)
        log_weights -= log_weights.max()  # numerical stability before exp
        raw = np.exp(log_weights)
        self.weights = raw / raw.sum()
        return self.weights


def expert_losses_from_returns(signal_matrix: np.ndarray, realized_next_return: np.ndarray) -> np.ndarray:
    """Per-expert mean loss from "did the expert call direction right":
    `signal_matrix` is `(n_experts, T)` of {-1, 0, 1}, `realized_next_return`
    is `(T,)`. Loss is `-mean_t(signal_i(t) * next_return(t))` — an expert
    that was long ahead of positive returns (or short ahead of negative
    ones) accrues low/negative raw loss.
    """
    signal_matrix = np.asarray(signal_matrix, dtype=float)
    realized_next_return = np.asarray(realized_next_return, dtype=float)
    per_step_loss = -(signal_matrix * realized_next_return[np.newaxis, :])
    return per_step_loss.mean(axis=1)


def expert_losses_from_critique(
    train_df: pd.DataFrame,
    strategy_signals: list[dict],
    window: int,
    lam_dd: float = DEFAULT_LAMBDA_DD,
    lam_vol: float = DEFAULT_LAMBDA_VOL,
    lam_hold: float = DEFAULT_LAMBDA_HOLD,
) -> np.ndarray:
    """Per-expert loss from re-simulating "what if we'd only ever followed
    expert i" over the trailing `window`, scored by the same
    `window_critique_report` the RL reward and the original critique loop
    both use. Loss combines the window's realized return *and* its risk
    penalties (`-(window_return + total_penalty)`), so an expert can't
    score well just by avoiding all penalty triggers while also losing
    money (e.g. never trading at all).
    """
    n_experts = len(strategy_signals)
    losses = np.zeros(n_experts)
    for i in range(n_experts):
        portfolio_values, actions = simulate_trajectory_for_choice(train_df, strategy_signals, i, window)
        report = window_critique_report(portfolio_values, actions, window, lam_dd, lam_vol, lam_hold)
        losses[i] = -(report["window_return"] + report["total_penalty"])
    return losses


def blended_expert_losses(
    critique_losses: np.ndarray,
    return_losses: np.ndarray | None = None,
    rho: float = HEDGE_LOSS_RHO,
) -> np.ndarray:
    """Blends the critique-based loss with the direction-call loss:
    `rho * return_loss + (1 - rho) * critique_loss` (both min-max
    normalized first so neither dominates purely by scale). Falls back to
    critique-only when no return-based loss is available (e.g. the
    coarse, per-LangGraph-iteration path that only has window-level data).
    """
    if return_losses is None:
        return critique_losses
    return rho * _normalize_losses(return_losses) + (1 - rho) * _normalize_losses(critique_losses)


def novelty_modulated_eta(eta_base: float, novelty_score: float, kappa: float = NOVELTY_ETA_KAPPA) -> float:
    """`eta_t = eta_base * (1 + kappa * novelty_score)` — a continuous
    multiplier on the Hedge learning rate, not an `if regime == "crisis"`
    branch. High novelty means Hedge forgets stale expert performance
    faster exactly when the environment is shifting; `novelty_score` in
    [0, 1] (see `scaata.regimes.novelty`) so eta ranges from `eta_base` (no
    novelty) up to `eta_base * (1 + kappa)` (maximal novelty).
    """
    return float(eta_base * (1.0 + kappa * novelty_score))


def sentiment_log_prior(
    n_experts: int,
    sentiment_expert_idx: int,
    novelty_score: float,
    gamma: float = NOVELTY_SENTIMENT_PRIOR_GAMMA,
) -> np.ndarray:
    """An explicit, auditable additive log-prior favoring the sentiment
    expert in proportion to measured novelty — separable from the
    data-driven multiplicative-weights term (`HedgeWeights.update`'s
    `log_prior` argument is added after the loss-based update, before
    renormalization), so "how much of this round's weight shift came from
    the prior vs. from realized performance" is always inspectable rather
    than blended into one opaque number. Operationalizes "sentiment reacts
    fastest to unseen shocks" without hardcoding a war/crisis if-branch.
    """
    prior = np.zeros(n_experts)
    prior[sentiment_expert_idx] = gamma * novelty_score
    return prior


def hedge_update_with_novelty(
    hedge: HedgeWeights,
    losses: np.ndarray,
    novelty_score: float,
    sentiment_expert_idx: int | None = None,
    kappa: float = NOVELTY_ETA_KAPPA,
    gamma: float = NOVELTY_SENTIMENT_PRIOR_GAMMA,
) -> np.ndarray:
    """Convenience wrapper combining both novelty-modulation mechanisms in
    one call — the learning-rate speed-up (`novelty_modulated_eta`) and,
    when a sentiment expert index is given, the sentiment log-prior
    (`sentiment_log_prior`). Passing `sentiment_expert_idx=None` (e.g. when
    no sentiment expert is in the pool yet) applies only the eta
    modulation.
    """
    eta_t = novelty_modulated_eta(hedge.eta, novelty_score, kappa)
    log_prior = (
        sentiment_log_prior(hedge.n_experts, sentiment_expert_idx, novelty_score, gamma)
        if sentiment_expert_idx is not None
        else None
    )
    return hedge.update(losses, eta_override=eta_t, log_prior=log_prior)


def floor_weights_on_simplex(weights: np.ndarray, floor: float) -> np.ndarray:
    """Projects `weights` (assumed to already sum to 1) onto the probability
    simplex subject to every component being `>= floor`, preserving the
    relative ranking of whichever components exceed the floor.

    A naive `np.maximum(weights, floor)` followed by dividing by the new
    sum does *not* guarantee the floor post-renormalization — e.g. with
    `floor=0.05` and `weights=[0.98, 0.01, 0.01]`, that naive approach
    produces `[0.907, 0.0463, 0.0463]`, silently violating the floor it was
    supposed to enforce. This does genuine "water-filling": every
    component gets exactly `floor`, and the remaining `1 - n*floor` budget
    is distributed across components in proportion to how much each
    exceeded `floor` in the original distribution.
    """
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    floor_total = floor * n
    if floor_total >= 1.0:
        return np.full(n, 1.0 / n)

    remaining_budget = 1.0 - floor_total
    excess = np.maximum(weights - floor, 0.0)
    excess_sum = excess.sum()
    if excess_sum < 1e-12:
        return np.full(n, floor) + remaining_budget / n
    return floor + (excess / excess_sum) * remaining_budget


def hedge_regret_report(loss_history: np.ndarray, weight_history: np.ndarray | None = None) -> dict:
    """Reports each expert's cumulative (normalized) loss, the best expert
    in hindsight, and — if `weight_history` (the weights actually used at
    each round) is provided — Hedge's own realized cumulative loss and its
    empirical regret relative to that best-in-hindsight expert. Regret
    should scale roughly as `O(sqrt(T ln N))`, not linearly in T — a
    falsifiable check on the mechanism itself, not just on downstream
    trading performance.
    """
    loss_history = np.asarray(loss_history, dtype=float)
    cumulative_per_expert = loss_history.sum(axis=0)
    best_in_hindsight = float(cumulative_per_expert.min())

    hedge_cumulative = float("nan")
    regret = float("nan")
    if weight_history is not None:
        weight_history = np.asarray(weight_history, dtype=float)
        hedge_loss_per_round = (weight_history * loss_history).sum(axis=1)
        hedge_cumulative = float(hedge_loss_per_round.sum())
        regret = hedge_cumulative - best_in_hindsight

    return {
        "cumulative_loss_per_expert": cumulative_per_expert,
        "best_in_hindsight_loss": best_in_hindsight,
        "hedge_cumulative_loss": hedge_cumulative,
        "regret": regret,
        "n_rounds": loss_history.shape[0],
    }
