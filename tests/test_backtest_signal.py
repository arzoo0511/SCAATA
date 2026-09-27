"""Regression + behavior tests for the shared trajectory-simulation core
(Phase 10 refactor out of `critique_node`'s original private
`_simulate_implied_trajectory`): the top-weighted-path mode must reproduce
the original critique-loop behavior exactly, and the constant-expert-index
broadcast mode (needed by the Hedge per-expert loss) must simulate "as if
we'd only ever followed expert i" correctly.
"""
import numpy as np
import pandas as pd

from scaata.strategies.backtest_signal import simulate_trajectory_for_choice


def _make_synthetic_train_df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close}, index=dates)


def _make_strategy_signals(n, seed=1):
    rng = np.random.default_rng(seed)
    return [
        {"source": "a", "signals": rng.choice([-1, 0, 1], size=n)},
        {"source": "b", "signals": rng.choice([-1, 0, 1], size=n)},
        {"source": "c", "signals": rng.choice([-1, 0, 1], size=n)},
    ]


def test_top_weighted_path_matches_original_critique_node_logic():
    """The original `_simulate_implied_trajectory` (before the Phase 10
    refactor) always followed `weight_matrix.argmax(axis=1)` per step —
    reproduced here manually and compared against the shared function to
    guard against a behavior change during the refactor.
    """
    train_df = _make_synthetic_train_df()
    strategy_signals = _make_strategy_signals(len(train_df))
    window = 20
    rng = np.random.default_rng(2)
    weight_matrix = rng.random((len(train_df), 3))

    start = max(0, len(train_df) - window)
    top_strategy_idx = weight_matrix[start:].argmax(axis=1)

    portfolio_values, actions = simulate_trajectory_for_choice(train_df, strategy_signals, top_strategy_idx, window)

    # Manual re-derivation of the original logic for comparison.
    prices = train_df["Close"].values[start:]
    signal_to_action = {-1: 2, 0: 0, 1: 1}
    cash, shares, position = 10_000.0, 0.0, 0
    expected_values = [10_000.0]
    expected_actions = []
    for i, price in enumerate(prices[:-1]):
        strat_idx = int(top_strategy_idx[i])
        signal = int(strategy_signals[strat_idx]["signals"][start + i])
        action = signal_to_action[signal]
        if action == 1 and position == 0:
            shares, cash, position = cash / price, 0.0, 1
        elif action == 2 and position == 1:
            cash, shares, position = shares * price, 0.0, 0
        expected_values.append(cash + shares * price)
        expected_actions.append(action)

    np.testing.assert_allclose(portfolio_values, expected_values)
    np.testing.assert_array_equal(actions, expected_actions)


def test_constant_expert_broadcast_simulates_a_single_strategy_throughout():
    train_df = _make_synthetic_train_df()
    strategy_signals = _make_strategy_signals(len(train_df))
    window = 20

    portfolio_values, actions = simulate_trajectory_for_choice(train_df, strategy_signals, 1, window)

    start = max(0, len(train_df) - window)
    signal_to_action = {-1: 2, 0: 0, 1: 1}
    expected_actions = [signal_to_action[int(s)] for s in strategy_signals[1]["signals"][start : start + len(actions)]]

    np.testing.assert_array_equal(actions, expected_actions)
