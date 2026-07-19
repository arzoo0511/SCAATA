"""Critique Agent — evaluates the meta-selector-implied trajectory over the
trailing review window using the same drawdown/volatility/holding-time
penalty terms the RL reward uses (`scaata.rl.reward`), then down-weights
whichever strategy was most responsible for the worst days. This node's
continue/done decision is what makes the Critique -> Meta-Selector feedback
edge in `agents/graph.py` a real, executing cycle.
"""
from __future__ import annotations

import numpy as np

from scaata.agents.state import AgentState
from scaata.config import (
    CONVERGENCE_EPSILON,
    MIN_STRATEGY_WEIGHT,
    REVIEW_WINDOW_DAYS,
    STRATEGY_DOWNWEIGHT_FACTOR,
)
from scaata.rl.reward import window_critique_report

SIGNAL_TO_ACTION = {-1: 2, 0: 0, 1: 1}


def _simulate_implied_trajectory(train_df, strategy_signals, weight_matrix, window, initial_cash: float = 10_000.0):
    """Simple, fee-free portfolio simulation following the meta-selector's
    top-weighted strategy at each row of the trailing `window` days — a
    coarse internal estimate for the critique loop's own bookkeeping, not
    the actual RL backtest (which uses the real env with fees/stop-loss).
    """
    n = len(train_df)
    start = max(0, n - window)
    prices = train_df["Close"].values[start:]
    top_strategy_idx = weight_matrix[start:].argmax(axis=1)

    cash, shares, position = initial_cash, 0.0, 0
    portfolio_values = [initial_cash]
    actions = []
    for i, price in enumerate(prices[:-1]):
        strat_idx = int(top_strategy_idx[i])
        signal = int(strategy_signals[strat_idx]["signals"][start + i]) if strat_idx < len(strategy_signals) else 0
        action = SIGNAL_TO_ACTION[signal]
        if action == 1 and position == 0:
            shares = cash / price
            cash = 0.0
            position = 1
        elif action == 2 and position == 1:
            cash = shares * price
            shares = 0.0
            position = 0
        portfolio_values.append(cash + shares * price)
        actions.append(action)

    return np.array(portfolio_values), np.array(actions), top_strategy_idx[: len(actions)]


def critique_node(state: AgentState) -> dict:
    train_df = state["train_df"]
    window = state.get("review_window", REVIEW_WINDOW_DAYS)
    weight_matrix = state["strategy_weight_matrix"]
    strategy_signals = state["strategy_signals"]
    pool_weights = list(state["strategy_pool_weights"])

    portfolio_values, actions, top_strategy_per_step = _simulate_implied_trajectory(
        train_df, strategy_signals, weight_matrix, window
    )
    report = window_critique_report(portfolio_values, actions, window)

    # Down-weight whichever strategy was most often "top choice" during a
    # window that scored net-negative on the self-critique penalty terms.
    if report["total_penalty"] < 0 and len(top_strategy_per_step) > 0:
        culprit = int(np.bincount(top_strategy_per_step).argmax())
        pool_weights[culprit] = max(MIN_STRATEGY_WEIGHT, pool_weights[culprit] * STRATEGY_DOWNWEIGHT_FACTOR)

    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", 4)

    history = list(state.get("history", []))
    prev_weights = history[-1]["strategy_pool_weights"] if history else None
    weight_delta = (
        max(abs(a - b) for a, b in zip(pool_weights, prev_weights)) if prev_weights else float("inf")
    )
    converged = weight_delta < CONVERGENCE_EPSILON

    history.append({"iteration": iteration, "strategy_pool_weights": list(pool_weights), "critique_report": report})

    return {
        "strategy_pool_weights": pool_weights,
        "critique_report": report,
        "iteration": iteration,
        "converged": converged or iteration >= max_iterations,
        "history": history,
    }
