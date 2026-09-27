"""Self-critique penalty terms (drawdown extension, volatility-chasing
entries, prolonged losing-position holding). This is the concrete fix for
the gap where the original v1 notebook's reward was only `%change * 10`
plus a stop-loss penalty — despite the paper describing a self-critique
loop that reshapes the advantage computation, none of it was actually
wired in. That closed loop (trajectory-level, rolling-window review
feeding a strategy-pool reweighting mechanism into the RL policy, provably
wired end-to-end — see `scaata/agents/nodes/critique_node.py` and
`tests/test_reward.py`/`tests/test_policy_init.py`) is the real, tested,
novel contribution.

Implementation note, stated precisely so it can't be misread as doubt
about the loop itself: the v1 paper's eq. 4 names these three penalties
by role only ("λ_dd, λ_vol, and λ_hold penalize drawdown extension,
volatility-chasing entry, and prolonged negative-PnL holding,
respectively") and gives no closed-form definition for any of them. The
functions below are this rebuild's own operationalization of that prose —
not a verified reproduction of an equation the paper never wrote down, and
not claimed as novel in isolation (risk-shaped reward terms like these
have prior art in quant-RL, e.g. Moody & Saffell's differential Sharpe
ratio). The defensible novelty here is the closed loop and its
integration with imitation learning and meta-strategy selection, not the
existence of a drawdown/volatility/holding-time penalty term by itself.

Two use sites:
1. `SelfCritiqueTracker` — stateful, step-wise, causal; wired directly into
   `RobustTradingEnv.step()` for live RL training.
2. `window_critique_report` — a post-hoc vectorized summary over a
   completed trajectory slice, used by the LangGraph critique node (and by
   the Phase 2 ablation/robustness scoring) to score a finished window
   without needing the live per-step tracker state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scaata.config import (
    CRITIQUE_DD_THRESHOLD,
    CRITIQUE_HOLD_GRACE_DAYS,
    CRITIQUE_VOL_BASELINE_WINDOW,
    CRITIQUE_VOL_WINDOW,
    DEFAULT_LAMBDA_DD,
    DEFAULT_LAMBDA_HOLD,
    DEFAULT_LAMBDA_VOL,
    DSR_ETA,
    DSR_REWARD_CLIP,
    DSR_VARIANCE_EPSILON,
    DSR_WARMUP_STEPS,
)


def drawdown_penalty(portfolio_history: np.ndarray, lam: float, threshold: float = CRITIQUE_DD_THRESHOLD) -> float:
    """Penalizes being in a drawdown deeper than `threshold` below the
    trailing peak. Causal: peak is computed only over history-so-far."""
    if len(portfolio_history) < 2:
        return 0.0
    peak = np.max(portfolio_history)
    current = portfolio_history[-1]
    dd = (current - peak) / peak  # <= 0
    excess = max(0.0, -dd - threshold)
    return -lam * excess


def volatility_chasing_penalty(
    portfolio_returns: np.ndarray,
    is_new_entry: bool,
    lam: float,
    window: int = CRITIQUE_VOL_WINDOW,
    baseline_window: int = CRITIQUE_VOL_BASELINE_WINDOW,
) -> float:
    """Penalizes opening a new position while the agent's own recent
    realized-return volatility is elevated relative to its own trailing
    baseline — a proxy for "entering on a volatile spike" rather than a
    stable signal. Only triggers on the step a new position is opened."""
    if not is_new_entry or len(portfolio_returns) < window:
        return 0.0
    recent_vol = np.std(portfolio_returns[-window:])
    base_slice = portfolio_returns[-baseline_window:] if len(portfolio_returns) >= baseline_window else portfolio_returns
    baseline_vol = np.std(base_slice)
    if baseline_vol < 1e-8:
        return 0.0
    excess_ratio = max(0.0, (recent_vol / baseline_vol) - 1.0)
    return -lam * excess_ratio


def holding_time_penalty(
    unrealized_pnl_pct: float,
    holding_days: int,
    lam: float,
    grace_days: int = CRITIQUE_HOLD_GRACE_DAYS,
) -> float:
    """Penalizes holding a losing position past a grace period; grows with
    both how long it's been held and how deep the unrealized loss is."""
    if unrealized_pnl_pct >= 0 or holding_days <= grace_days:
        return 0.0
    excess_days = holding_days - grace_days
    return -lam * excess_days * abs(unrealized_pnl_pct)


@dataclass
class SelfCritiqueTracker:
    """Stateful, causal, step-wise tracker wired into `RobustTradingEnv`."""

    lambda_dd: float = DEFAULT_LAMBDA_DD
    lambda_vol: float = DEFAULT_LAMBDA_VOL
    lambda_hold: float = DEFAULT_LAMBDA_HOLD
    portfolio_history: list = field(default_factory=list)
    entry_step: int | None = None

    def reset(self, initial_value: float):
        self.portfolio_history = [initial_value]
        self.entry_step = None

    def on_entry(self, step_idx: int):
        self.entry_step = step_idx

    def on_exit(self):
        self.entry_step = None

    def step_penalty(
        self,
        current_value: float,
        step_idx: int,
        position: int,
        entry_price: float,
        current_price: float,
        is_new_entry: bool,
    ) -> dict:
        self.portfolio_history.append(current_value)
        history = np.array(self.portfolio_history)
        returns = np.diff(history) / history[:-1] if len(history) > 1 else np.array([])

        dd_pen = drawdown_penalty(history, self.lambda_dd)
        vol_pen = volatility_chasing_penalty(returns, is_new_entry, self.lambda_vol)

        hold_pen = 0.0
        if position == 1 and self.entry_step is not None:
            holding_days = step_idx - self.entry_step
            unrealized = (current_price - entry_price) / entry_price
            hold_pen = holding_time_penalty(unrealized, holding_days, self.lambda_hold)

        total = dd_pen + vol_pen + hold_pen
        return {
            "drawdown_penalty": dd_pen,
            "volatility_penalty": vol_pen,
            "holding_time_penalty": hold_pen,
            "total": total,
        }


def window_critique_report(
    portfolio_values: np.ndarray,
    actions: np.ndarray,
    window: int,
    lam_dd: float = DEFAULT_LAMBDA_DD,
    lam_vol: float = DEFAULT_LAMBDA_VOL,
    lam_hold: float = DEFAULT_LAMBDA_HOLD,
) -> dict:
    """Post-hoc summary over the trailing `window` days of a completed
    trajectory. `holding_time_penalty` here is a coarser window-level proxy
    (fraction of HOLD actions weighted by the window's negative return)
    since we don't have live per-step position/entry-price state at this
    point — the live env uses `SelfCritiqueTracker` for the exact per-step
    version instead.
    """
    portfolio_values = np.asarray(portfolio_values, dtype=float)
    actions = np.asarray(actions)

    tail = portfolio_values[-(window + 1):] if len(portfolio_values) > window else portfolio_values
    returns = np.diff(tail) / tail[:-1] if len(tail) > 1 else np.array([])

    dd_pen = drawdown_penalty(tail, lam_dd)
    vol = float(np.std(returns)) if len(returns) > 0 else 0.0

    window_actions = actions[-window:] if len(actions) >= window else actions
    cum_return = float(tail[-1] / tail[0] - 1) if len(tail) > 1 else 0.0
    holding_frac = float(np.mean(window_actions == 0)) if len(window_actions) > 0 else 0.0
    hold_pen = -lam_hold * holding_frac * abs(min(cum_return, 0.0))

    total = dd_pen - lam_vol * vol + hold_pen
    return {
        "drawdown_penalty": dd_pen,
        "volatility": vol,
        "holding_time_penalty": hold_pen,
        "window_return": cum_return,
        "total_penalty": total,
    }


def differential_sharpe_reward(return_t: float, A_prev: float, B_prev: float, eta: float) -> tuple[float, float, float]:
    """One step of Moody & Saffell's (2001) Differential Sharpe Ratio
    ("Learning to Trade via Direct Reinforcement", eq. 12-14) — a
    genuinely different reward mechanism from `SelfCritiqueTracker` above,
    not a retuning of it: instead of raw return plus separately hand-tuned
    risk penalties (which can dominate and bias training toward inaction —
    the exact failure mode found in live-data testing, where a policy
    converged to never opening a position at all), this reward *is* an
    online estimate of the derivative of the Sharpe ratio with respect to
    the newest return. A policy trained to maximize cumulative reward
    under this scheme is directly trained to maximize risk-adjusted
    return, with no separate penalty coefficients to miscalibrate.

    `A_prev`/`B_prev` are running exponential-moving-average estimates of
    the first and second moments of the per-step return; `eta` is their
    adaptation rate. Returns `(reward, new_A, new_B)`.

    Honest limitation, stated plainly: for a policy whose return is
    identically 0 every step (never trades), `A` and `B` both stay at 0,
    the running variance estimate `B - A**2` stays at 0, and the reward is
    defined as 0.0 (see `DSR_VARIANCE_EPSILON` below) — exactly the same
    net reward a flat policy already got under the old
    `percent_change * 10` scheme. This mechanism does not *guarantee* a
    fix for policy collapse; its actual benefit is removing the specific,
    empirically-observed failure mode of penalty coefficients that can
    outweigh raw return, replacing them with a single principled
    risk-adjusted signal. Whether that changes real training behavior is
    an empirical question to verify by training, not something this
    formula can prove on its own.

    Second, sharper limitation, found empirically while building this
    (not in the original paper's caveats): `D_t` is a first-order
    approximation of `dS/dη`, valid only while `A`/`B` are slowly-varying
    relative to `eta`. Tested directly against a pair of return sequences
    with identical mean but different variance (over ~200 steps): at
    `eta <= 0.005`, cumulative reward correctly favors the lower-variance
    sequence, matching the final EMA-Sharpe estimate's own ranking. At
    `eta >= 0.01`, the ranking **inverts** — cumulative reward favors the
    *higher*-variance sequence, the opposite of what a risk-adjusted
    reward should do, and would actively train a policy toward more
    volatile behavior if used. `config.DSR_ETA` is set conservatively
    below this verified break point; do not raise it without re-running
    the same ranking check (`tests/test_differential_sharpe.py::test_lower_variance_same_mean_scores_higher_cumulative_reward`)
    at the new value first.
    """
    delta_A = return_t - A_prev
    delta_B = return_t ** 2 - B_prev

    variance_est = B_prev - A_prev ** 2
    if variance_est <= DSR_VARIANCE_EPSILON:
        reward = 0.0
    else:
        reward = (B_prev * delta_A - 0.5 * A_prev * delta_B) / (variance_est ** 1.5)

    new_A = A_prev + eta * delta_A
    new_B = B_prev + eta * delta_B
    return reward, new_A, new_B


@dataclass
class DifferentialSharpeTracker:
    """Stateful, causal, step-wise wrapper around
    `differential_sharpe_reward`, mirroring `SelfCritiqueTracker`'s role
    for the penalty-based reward — wired directly into
    `RobustTradingEnv.step()` when `use_differential_sharpe=True`.

    Adds two practical safety measures on top of the pure formula, found
    necessary by testing against real `RobustTradingEnv` dynamics (fees,
    stop-loss, all-in sizing), not present in the textbook formula:

    1. `warmup_steps`: the first N steps of each episode always return
       reward 0.0 (A/B still update normally). This is the well-known DSR
       "burn-in" problem — with only a step or two of return history, the
       running variance estimate `B - A**2` can be a tiny nonzero number
       that `DSR_VARIANCE_EPSILON` doesn't catch, and dividing by that
       number raised to the 1.5 power produced rewards in the *millions*
       in real-environment testing (a single ~1% return at step 2 of an
       episode produced a reward of ~3.4 million before this fix).
    2. `reward_clip`: a hard clip on the raw (pre-`DSR_REWARD_SCALE`)
       reward, as defense-in-depth against rarer post-warmup spikes (e.g.
       a stop-loss-triggered large loss during an otherwise-calm stretch).

    Both defaults were chosen by directly re-testing against the same
    real-environment scenario that produced the multi-million-magnitude
    reward, not picked arbitrarily — see `tests/test_differential_sharpe.py`.
    """

    eta: float = DSR_ETA
    A: float = 0.0
    B: float = 0.0
    warmup_steps: int = DSR_WARMUP_STEPS
    reward_clip: float = DSR_REWARD_CLIP
    _step_count: int = field(default=0, repr=False, compare=False)

    def reset(self):
        self.A = 0.0
        self.B = 0.0
        self._step_count = 0

    def step(self, return_t: float) -> float:
        raw_reward, self.A, self.B = differential_sharpe_reward(return_t, self.A, self.B, self.eta)
        self._step_count += 1
        if self._step_count <= self.warmup_steps:
            return 0.0
        return float(np.clip(raw_reward, -self.reward_clip, self.reward_clip))
