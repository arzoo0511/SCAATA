"""Cross-sectional momentum: hold the strongest few names out of many.

The one classic effect with decades of published evidence across markets,
including India: rank a universe by its trailing return, hold the top slice,
refresh periodically. It is not a forecast of any single stock -- it is a bet
that relative strength persists for months at a time.

This is the only idea in this project that needs an agent rather than a
person: ranking 60 stocks every month and trading the changes is mechanical
work at a scale a human would not do by hand.

Two honesty rules are built into how this is measured:

1. **Momentum skips the most recent month** (the standard 12-1 window).
   Last month's return reverses on average, and including it turns a
   momentum test into a short-term-reversal test.
2. **The benchmark is the same universe, equally weighted.** Picking today's
   index members and testing them over the past six years is survivorship
   bias; comparing against an equal-weight hold of those same names cancels
   most of it, because both legs own the same survivors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_scores(closes: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Trailing return from `lookback` days ago to `skip` days ago, per day."""
    return closes.shift(skip) / closes.shift(lookback) - 1


def month_end_positions(index: pd.DatetimeIndex, every: str = "M") -> list[int]:
    """Row positions of the last trading day of each month (or quarter)."""
    period = index.to_period("M" if every == "M" else "Q")
    return list(np.where(period[:-1] != period[1:])[0])


def backtest_cross_sectional(closes: pd.DataFrame, cost_per_side: float, top_n: int = 10,
                             lookback: int = 252, skip: int = 21, every: str = "M",
                             initial: float = 10_000.0) -> dict:
    """Equal-weight the `top_n` highest-momentum names, refreshed monthly.

    Weights chosen on a rebalance day are applied from the next day, so the
    ranking never uses a return it then earns. Costs are charged on the
    weight actually traded.
    """
    prices = closes.dropna(how="all")
    returns = prices.pct_change().fillna(0.0)
    scores = momentum_scores(prices, lookback, skip)
    rebalance_days = set(month_end_positions(prices.index, every))

    weights = pd.Series(0.0, index=prices.columns)
    equity, curve, turnover_total, rebalances = initial, [initial], 0.0, 0
    holdings_log = []
    for t in range(1, len(prices)):
        equity *= 1 + float((weights * returns.iloc[t]).sum())
        drifted = weights * (1 + returns.iloc[t])
        total = drifted.sum()
        weights = drifted / total if total > 0 else drifted

        if t in rebalance_days:
            ranked = scores.iloc[t].dropna()
            if len(ranked) >= top_n:
                winners = ranked.nlargest(top_n).index
                target = pd.Series(0.0, index=prices.columns)
                target[winners] = 1 / top_n
                traded = float((target - weights).abs().sum())
                equity *= 1 - traded * cost_per_side
                turnover_total += traded
                weights, rebalances = target, rebalances + 1
                holdings_log.append({"date": prices.index[t].date().isoformat(),
                                     "holdings": [w.replace(".NS", "") for w in winners]})
        curve.append(equity)

    return {"equity": np.array(curve), "index": prices.index, "rebalances": rebalances,
            "turnover_per_year": turnover_total / max((prices.index[-1] - prices.index[0]).days / 365.25, 1e-9),
            "holdings_log": holdings_log}


def equal_weight_benchmark(closes: pd.DataFrame, cost_per_side: float, every: str = "M",
                           initial: float = 10_000.0) -> np.ndarray:
    """Hold the whole universe, equally weighted, refreshed on the same
    schedule -- the fair comparison for a strategy that picks from it."""
    prices = closes.dropna(how="all")
    returns = prices.pct_change().fillna(0.0)
    rebalance_days = set(month_end_positions(prices.index, every))
    available = prices.notna()

    weights = (available.iloc[0] / available.iloc[0].sum()).fillna(0.0)
    equity, curve = initial, [initial]
    for t in range(1, len(prices)):
        equity *= 1 + float((weights * returns.iloc[t]).sum())
        drifted = weights * (1 + returns.iloc[t])
        total = drifted.sum()
        weights = drifted / total if total > 0 else drifted
        if t in rebalance_days:
            target = available.iloc[t] / available.iloc[t].sum()
            equity *= 1 - float((target - weights).abs().sum()) * cost_per_side
            weights = target.fillna(0.0)
        curve.append(equity)
    return np.array(curve)
