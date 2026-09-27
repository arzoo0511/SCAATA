"""Volatility-targeting position sizing — a mechanical, non-RL lever for
raising risk-adjusted return: size positions inversely to recent realized
volatility, so the return stream has roughly constant risk over time,
rather than trying to predict market direction better. This is one of the
most reliable, well-documented ways real quant funds/CTAs raise Sharpe
(risk-parity/vol-scaling sizing), and it's a genuinely different lever
from everything else in this codebase — it doesn't require retraining any
model. It composes directly with `RobustTradingEnv.step()`'s
`size_multiplier` kwarg (Phase 14) on top of ANY existing signal source:
a rule-based strategy, an evolved strategy, a trained RL policy's own
actions, or even a constant "always long" signal like Buy&Hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.config import (
    VOL_TARGET_DAILY,
    VOL_TARGET_MAX_SIZE,
    VOL_TARGET_MIN_SIZE,
    VOL_TARGET_WINDOW,
)
from scaata.rl.env import BUY, HOLD, SELL, RobustTradingEnv


def compute_realized_volatility(returns: pd.Series, window: int = VOL_TARGET_WINDOW) -> pd.Series:
    """Trailing (causal) realized volatility — rolling std of returns.
    Matches `scaata.features.technical.add_features`'s own `volatility`
    column convention (same window, same rolling-std definition), so
    either can be used interchangeably.
    """
    return returns.rolling(window, min_periods=window).std()


def volatility_target_size(
    realized_vol: float,
    target_vol: float = VOL_TARGET_DAILY,
    min_size: float = VOL_TARGET_MIN_SIZE,
    max_size: float = VOL_TARGET_MAX_SIZE,
) -> float:
    """Position size that would keep realized risk near `target_vol`,
    capped at `max_size` (long-only, no leverage — a realized vol far
    below target does not mean "go above 100%") and floored at `min_size`
    (elevated volatility shrinks the position but never fully zeroes it
    out). Returns `max_size` when there isn't yet a meaningful volatility
    estimate (NaN, zero, or negative) — no information to size down on.
    """
    if realized_vol is None or not np.isfinite(realized_vol) or realized_vol <= 0:
        return max_size
    raw_size = target_vol / realized_vol
    return float(np.clip(raw_size, min_size, max_size))


def backtest_ppo_with_vol_targeting(
    model,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    ticker: str,
    window: int = VOL_TARGET_WINDOW,
    target_vol: float = VOL_TARGET_DAILY,
    min_size: float = VOL_TARGET_MIN_SIZE,
    max_size: float = VOL_TARGET_MAX_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same rollout as `scaata.rl.train.backtest_ppo`, except every BUY/SELL
    the live policy takes is sized by trailing realized volatility instead
    of always going to 100%/0% -- the model-driven counterpart to
    `simulate_with_vol_targeting` (which takes a pre-computed signal array
    instead of asking a model). This is what a real deployment overlay
    looks like: the policy still decides direction, sizing is a mechanical
    layer on top. Returns `(equity_curve, actions, size_multipliers_used)`.
    """
    ticker_df = test_df[test_df["Ticker"] == ticker]
    realized_vol = compute_realized_volatility(ticker_df["Close"].pct_change(), window)

    env = RobustTradingEnv(test_df, feature_columns, fixed_ticker=ticker)
    obs, _ = env.reset()
    done = False
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    equity = [env.initial_cash]
    actions = []
    sizes_used = []
    step_idx = 0
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        size = volatility_target_size(realized_vol.iloc[step_idx], target_vol, min_size, max_size)
        obs, reward, done, _, info = env.step(action, size_multiplier=size)
        episode_starts = np.array([done], dtype=bool)
        actions.append(int(action))
        sizes_used.append(size)
        equity.append(info["portfolio_value"])
        step_idx += 1

    return np.array(equity), np.array(actions), np.array(sizes_used)


def simulate_with_vol_targeting(
    df: pd.DataFrame,
    signals: np.ndarray,
    ticker: str,
    window: int = VOL_TARGET_WINDOW,
    target_vol: float = VOL_TARGET_DAILY,
    min_size: float = VOL_TARGET_MIN_SIZE,
    max_size: float = VOL_TARGET_MAX_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Steps a fixed, pre-computed `signals` array (`{-1, 0, 1}`, same
    length as `df`, e.g. from `scaata.strategies.pool.run_strategy_safely`,
    a trained policy's frozen greedy actions, or a constant "always long"
    array for Buy&Hold) through `RobustTradingEnv`, sizing each BUY by the
    trailing realized volatility at that day — a real-data test of the
    sizing overlay's effect, independent of how the underlying signal was
    generated (no RL training involved here).

    Returns `(equity_curve, size_multipliers_used)`.
    """
    returns = df["Close"].pct_change()
    realized_vol = compute_realized_volatility(returns, window)

    signal_to_action = {-1: SELL, 0: HOLD, 1: BUY}
    env = RobustTradingEnv(df, feature_columns=["returns"], fixed_ticker=ticker)
    env.reset()

    equity = [env.initial_cash]
    sizes_used = []
    done = False
    step_idx = 0
    while not done:
        action = signal_to_action[int(signals[step_idx])]
        size = volatility_target_size(realized_vol.iloc[step_idx], target_vol, min_size, max_size)
        _, _, done, _, info = env.step(action, size_multiplier=size)
        equity.append(info["portfolio_value"])
        sizes_used.append(size)
        step_idx += 1

    return np.array(equity), np.array(sizes_used)
