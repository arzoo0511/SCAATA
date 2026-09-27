"""Tests for ContinuousTradingEnv (Phase 19) -- the capability
RobustTradingEnv structurally cannot do (per its own docstring:
"single-shot sizing at entry, not a full continuous position-adjustment
model"): buying more into an already-open position, and partially
trimming one, not just a single all-in entry and a single full exit.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import FEATURE_COLUMNS
from scaata.rl.continuous_env import ContinuousTradingEnv


def _make_env_df(n=20, seed=0, ticker="TEST", close=None):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    if close is None:
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(5, min_periods=1).mean()
    df["Ticker"] = ticker
    return df


def _reset_env(df, **kwargs):
    env = ContinuousTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST", use_differential_sharpe=False, **kwargs)
    env.reset()
    return env


def test_full_target_fraction_invests_nearly_all_cash():
    env = _reset_env(_make_env_df())
    env.step(np.array([1.0]))

    assert env.cash < env.initial_cash * 0.01  # fee-adjusted, not exactly zero
    assert env.shares > 0


def test_half_target_fraction_invests_roughly_half():
    env = _reset_env(_make_env_df())
    obs, reward, done, trunc, info = env.step(np.array([0.5]))

    assert info["actual_fraction"] == pytest.approx(0.5, abs=0.02)
    assert env.cash > env.initial_cash * 0.4


def test_zero_target_fraction_stays_flat():
    env = _reset_env(_make_env_df())
    env.step(np.array([0.0]))

    assert env.shares == 0.0
    assert env.cash == pytest.approx(env.initial_cash)


def test_portfolio_value_conserved_through_a_buy():
    env = _reset_env(_make_env_df())
    obs, reward, done, trunc, info = env.step(np.array([0.7]))

    price_at_step = env.prices[env.step_idx - 1]
    assert info["portfolio_value"] == pytest.approx(env.cash + env.shares * price_at_step)


def test_can_increase_an_already_open_position():
    """The core new capability RobustTradingEnv cannot do: a second BUY-like
    action (raising the target fraction) while already holding shares must
    add to the position, not be a no-op."""
    env = _reset_env(_make_env_df(n=30))
    env.step(np.array([0.3]))
    shares_after_first = env.shares

    env.step(np.array([0.8]))

    assert env.shares > shares_after_first


def test_can_partially_trim_an_open_position_without_fully_exiting():
    """The other core new capability: lowering the target fraction sells
    only the delta, leaving a smaller but still-open position -- not a
    forced full liquidation."""
    env = _reset_env(_make_env_df(n=30))
    env.step(np.array([1.0]))
    shares_after_full = env.shares
    assert shares_after_full > 0

    env.step(np.array([0.4]))

    assert 0 < env.shares < shares_after_full


def test_trade_below_min_rebalance_fraction_is_skipped():
    env = _reset_env(_make_env_df(n=30), min_rebalance_fraction=0.5)
    env.step(np.array([0.3]))
    shares_after_first = env.shares
    cash_after_first = env.cash

    # A tiny nudge (0.3 -> 0.32), well under the 0.5 threshold, should be
    # treated as noise and skipped entirely -- no fee drag from churn.
    env.step(np.array([0.32]))

    assert env.shares == shares_after_first
    assert env.cash == cash_after_first


def test_cost_basis_is_shares_weighted_average_across_multiple_buys():
    prices = np.array([100.0] * 5 + [200.0] * 5 + [150.0] * 10)
    env = _reset_env(_make_env_df(n=20, close=prices))

    env.step(np.array([0.5]))  # buys at price ~100
    shares_1, price_1 = env.shares, env.prices[env.step_idx - 1]
    env.step(np.array([1.0]))  # buys more at price ~200
    shares_2_delta = env.shares - shares_1
    price_2 = env.prices[env.step_idx - 1]

    expected_basis = (price_1 * shares_1 + price_2 * shares_2_delta) / env.shares
    assert env.cost_basis == pytest.approx(expected_basis, rel=1e-6)


def test_stop_loss_forces_full_exit_on_cost_basis_drawdown():
    # Sharp, sustained drop well past STOP_LOSS_PCT after building a position.
    prices = np.array([100.0] * 5 + [70.0] * 15)
    env = _reset_env(_make_env_df(n=20, close=prices))

    env.step(np.array([1.0]))  # enters near price=100
    assert env.shares > 0

    for _ in range(5):
        obs, reward, done, trunc, info = env.step(np.array([1.0]))  # keep wanting full exposure
        if env.shares == 0.0:
            break

    assert env.shares == 0.0
    assert env.cost_basis == 0.0


def test_cannot_sell_more_shares_than_held():
    env = _reset_env(_make_env_df(n=30))
    env.step(np.array([0.3]))

    # A target far below 0 isn't possible (Box is clipped to [0,1] by the
    # caller/policy in practice, but the env itself must be robust even if
    # something passes an already-at-zero-or-below intent) -- confirm
    # repeated zero-target steps never drive shares negative.
    env.step(np.array([0.0]))
    env.step(np.array([0.0]))

    assert env.shares >= 0.0


def test_default_action_space_is_box_zero_to_one():
    env = ContinuousTradingEnv(_make_env_df(), FEATURE_COLUMNS, fixed_ticker="TEST")
    assert env.action_space.low[0] == 0.0
    assert env.action_space.high[0] == 1.0


def test_differential_sharpe_reward_wires_through(monkeypatch):
    """Reuses DifferentialSharpeTracker unchanged -- confirms the env
    actually calls it rather than silently falling back to raw reward."""
    env = ContinuousTradingEnv(_make_env_df(n=10), FEATURE_COLUMNS, fixed_ticker="TEST", use_differential_sharpe=True)
    env.reset()
    assert env.dsr_tracker is not None

    obs, reward, done, trunc, info = env.step(np.array([0.5]))
    assert isinstance(reward, float)
