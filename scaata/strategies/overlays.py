"""Two mechanical overlays on a portfolio you already hold.

Neither predicts anything. They are the two levers available to someone who
owns a handful of stocks and wants a better outcome than leaving them alone:

1. **Volatility targeting** -- hold the same stock, but smaller when it is
   moving violently. Realized volatility is strongly autocorrelated (calm
   follows calm, wild follows wild) in a way returns are not, which is why
   this can help when direction forecasting does not.
2. **Rebalancing** -- hold all five, and periodically trim whatever ran up
   back toward an equal share, buying whatever lagged. Mechanically sells
   high and buys low between correlated assets.

Everything is causal (decisions from data up to day t are applied at t+1),
costs are charged on every change in position, and idle cash can be given a
yield, since in India it would sit in a liquid fund rather than under a
mattress.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Trailing daily volatility -- only past returns, shifted so day t's
    value is knowable before day t's move."""
    return returns.rolling(window, min_periods=window).std().shift(1)


def vol_targeted_exposure(prices: pd.Series, target_daily_vol: float = 0.015, window: int = 20,
                          max_exposure: float = 1.0) -> pd.Series:
    """Fraction of the sleeve to hold each day: target vol over recent
    realized vol, capped (no leverage). Before there is enough history to
    measure, hold the full position."""
    returns = prices.pct_change()
    vol = realized_volatility(returns, window)
    exposure = (target_daily_vol / vol).clip(upper=max_exposure)
    return exposure.fillna(max_exposure)


def simulate(exposure: pd.Series, prices: pd.Series, cost_per_side: float,
             cash_yield: float = 0.0, initial: float = 10_000.0) -> np.ndarray:
    """Equity curve for a daily exposure series in [0, 1]; uninvested cash
    earns `cash_yield` a year."""
    returns = prices.pct_change().values[1:]
    held = exposure.values[:-1]
    turnover = np.abs(np.diff(np.r_[0.0, held]))
    daily_cash_rate = (1 + cash_yield) ** (1 / TRADING_DAYS) - 1
    daily = held * returns + (1 - held) * daily_cash_rate - turnover * cost_per_side
    return initial * np.cumprod(np.r_[1.0, 1 + daily])


def rebalanced_portfolio(closes: pd.DataFrame, cost_per_side: float, schedule: str = "none",
                         drift_threshold: float = 0.2, initial: float = 10_000.0) -> tuple[np.ndarray, int]:
    """Equal-weight portfolio under a rebalancing rule.

    `schedule`: "none" (buy and hold, weights drift), "M" (month end),
    "Q" (quarter end), or "threshold" (whenever any holding drifts more than
    `drift_threshold` away from its equal share). Returns the equity curve
    and how many rebalances happened.
    """
    prices = closes.dropna()
    returns = prices.pct_change().fillna(0.0).values
    n_assets = prices.shape[1]
    target = np.full(n_assets, 1 / n_assets)

    if schedule in ("M", "Q"):
        period = prices.index.to_period("M" if schedule == "M" else "Q")
        rebalance_days = set(np.where(period[:-1] != period[1:])[0] + 1)
    else:
        rebalance_days = set()

    weights, equity, curve, rebalances = target.copy(), initial, [initial], 0
    for t in range(1, len(prices)):
        weights = weights * (1 + returns[t])
        total = weights.sum()
        equity *= total
        weights /= total                                   # drifted weights
        due = t in rebalance_days or (
            schedule == "threshold" and np.max(np.abs(weights - target) / target) > drift_threshold)
        if due:
            traded = np.abs(target - weights).sum() / 2     # one side of the swap
            equity *= 1 - traded * 2 * cost_per_side        # sell one, buy the other
            weights = target.copy()
            rebalances += 1
        curve.append(equity)
    return np.array(curve), rebalances
