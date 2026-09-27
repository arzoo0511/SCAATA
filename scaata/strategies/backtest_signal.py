"""Shared trajectory-simulation core (Phase 10), factored out of
`scaata.agents.nodes.critique_node`'s original private
`_simulate_implied_trajectory` so the same fee-free, trailing-window
portfolio simulation can be reused by: the critique loop itself (Phase 2),
the Hedge combiner's per-expert loss (Phase 10, `scaata.strategies.hedge`),
strategy evolution's fitness function (Phase 11), and the devil's-advocate
counterfactual re-simulation (Phase 12) — one mechanism, several consumers,
instead of four copies of the same loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_TO_ACTION = {-1: 2, 0: 0, 1: 1}  # sell, hold, buy -> RobustTradingEnv's action ids


def simulate_trajectory_for_choice(
    train_df: pd.DataFrame,
    strategy_signals: list[dict],
    choice_idx_per_step,
    window: int,
    initial_cash: float = 10_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates a fee-free portfolio over the trailing `window` days of
    `train_df`, following at each step whichever pooled strategy
    `choice_idx_per_step` selects for that step. `choice_idx_per_step` may
    be a per-step array (e.g. `weight_matrix.argmax(axis=1)`, the top-
    weighted strategy's path) or a single int/0-d value, broadcast across
    every step (e.g. "what if we'd only ever followed expert i").

    Returns `(portfolio_values, actions)` — `actions` uses
    `RobustTradingEnv`'s action ids (0=hold, 1=buy, 2=sell) so the result
    can be scored directly by `scaata.rl.reward.window_critique_report`.
    """
    n = len(train_df)
    start = max(0, n - window)
    prices = train_df["Close"].values[start:]

    choice_idx_per_step = np.broadcast_to(np.asarray(choice_idx_per_step), (len(prices),))

    cash, shares, position = initial_cash, 0.0, 0
    portfolio_values = [initial_cash]
    actions = []
    for i, price in enumerate(prices[:-1]):
        strat_idx = int(choice_idx_per_step[i])
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

    return np.array(portfolio_values), np.array(actions)
