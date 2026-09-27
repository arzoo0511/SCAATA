"""The council: run every known strategy every day, and keep shifting money
toward the ones that are actually working.

This is the "taught first, then learns from its own trades" piece. Each
strategy in the library votes long or flat on a stock each day. The council
holds a weight per strategy, takes the weighted vote as its exposure, and
after the day's return is known, updates those weights by how each strategy
actually did (multiplicative weights / Hedge). A strategy that keeps calling
it right compounds weight; one that keeps being wrong fades. Nothing is
retrained and nothing is fitted in advance -- the learning is the weight
update, and it happens once per trading day.

Everything here is causal: the weights used on day t come only from returns
up to day t-1, and a signal for day t is computed from data up to t, acted
on at t+1's price. Costs are charged on every change in exposure.

Why Hedge rather than "pick the best strategy so far": Hedge has a proven
bound -- over any run, it cannot do much worse than the single best strategy
chosen with hindsight -- whereas chasing the recent winner has no such
guarantee and flips at the worst moments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.strategies.hedge import HedgeWeights, eta_from_horizon
from scaata.strategies.pool import run_strategy_safely


def strategy_exposures(df: pd.DataFrame, strategies: list[dict]) -> tuple[np.ndarray, list[str]]:
    """(n_strategies, T) matrix of long/flat exposure per strategy per day.

    Signals are {-1, 0, 1}; this book never shorts, so -1 and 0 both mean
    flat. A strategy whose code fails or never changes its mind is dropped.
    """
    rows, names = [], []
    for strategy in strategies:
        code = strategy.get("clean_code") or strategy.get("code")
        signals = run_strategy_safely(code, df)
        if signals is None or len(np.unique(signals)) <= 1:
            continue
        rows.append((np.asarray(signals) > 0).astype(float))
        names.append(strategy["source"])
    return (np.vstack(rows) if rows else np.zeros((0, len(df)))), names


def simulate_exposure(exposure: np.ndarray, prices: np.ndarray, cost_per_side: float,
                      initial_cash: float = 10_000.0) -> np.ndarray:
    """Equity curve for a daily exposure series in [0, 1].

    Exposure decided on day t is acted on at day t+1's price, so the day-t
    signal never earns the day-t move. Costs are charged on the size of each
    change in exposure.
    """
    returns = np.diff(prices) / prices[:-1]
    applied = exposure[:-1]                      # yesterday's decision, today's return
    turnover = np.abs(np.diff(np.r_[0.0, applied]))
    daily = applied * returns - turnover * cost_per_side
    return initial_cash * np.cumprod(np.r_[1.0, 1 + daily])


def run_council(df: pd.DataFrame, strategies: list[dict], cost_per_side: float,
                initial_cash: float = 10_000.0, eta: float | None = None,
                lookback: int = 60) -> dict:
    """Runs the council over one stock's history.

    Returns the equity curve, the daily exposure it chose, the final and
    historical strategy weights, and the per-strategy equity curves for
    comparison.
    """
    exposures, names = strategy_exposures(df, strategies)
    prices = df["Close"].values
    if exposures.size == 0:
        flat = np.full(len(prices), initial_cash)
        return {"equity": flat, "exposure": np.zeros(len(prices)), "names": [], "weights": np.zeros(0),
                "weight_history": np.zeros((0, 0)), "per_strategy": {}}

    n_strategies = len(names)
    hedge = HedgeWeights(n_strategies, eta if eta is not None else eta_from_horizon(n_strategies, lookback))
    returns = np.diff(prices) / prices[:-1]

    equity, position, weights_history, council_exposure = initial_cash, 0.0, [], np.zeros(len(prices))
    curve = [initial_cash]
    for t in range(len(prices) - 1):
        target = float(hedge.weights @ exposures[:, t])   # weighted vote -> exposure for tomorrow
        council_exposure[t] = target
        turnover = abs(target - position)
        equity *= 1 + target * returns[t] - turnover * cost_per_side
        position = target
        curve.append(equity)
        weights_history.append(hedge.weights.copy())
        # Learn from what just happened: a strategy that was long into a
        # gain took a low loss, one that was long into a fall took a high one.
        hedge.update(-exposures[:, t] * returns[t])

    per_strategy = {name: simulate_exposure(exposures[i], prices, cost_per_side, initial_cash)
                    for i, name in enumerate(names)}
    return {"equity": np.array(curve), "exposure": council_exposure, "names": names,
            "weights": hedge.weights, "weight_history": np.array(weights_history),
            "per_strategy": per_strategy}
