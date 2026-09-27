"""Tests for `RobustTradingEnv`'s `use_differential_sharpe` wiring: default
behavior must stay bit-identical to before (regression guard, since every
existing training call site never passes this flag), and enabling it must
actually change the reward to the DSR value instead of `percent_change*10`,
using the tracker directly as the source of truth for what to expect.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import FEATURE_COLUMNS
from scaata.rl.env import BUY, HOLD, RobustTradingEnv
from scaata.rl.reward import DifferentialSharpeTracker


def _make_env_df(n=30, seed=0, ticker="TEST"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(5, min_periods=1).mean()
    df["Ticker"] = ticker
    return df


def test_default_reproduces_percent_change_reward_exactly():
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST")
    env.reset()

    prev_value = env.portfolio_value
    _, reward, _, _, info = env.step(BUY)
    percent_change = (info["portfolio_value"] - prev_value) / prev_value

    assert reward == pytest.approx(percent_change * 10)


def test_dsr_mode_uses_tracker_value_instead_of_percent_change():
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST", use_differential_sharpe=True, dsr_eta=0.005)
    env.reset()

    reference_tracker = DifferentialSharpeTracker(eta=0.005)

    prev_value = env.portfolio_value
    _, reward, _, _, info = env.step(BUY)
    percent_change = (info["portfolio_value"] - prev_value) / prev_value
    expected_reward = reference_tracker.step(percent_change) * env.dsr_reward_scale

    assert reward == pytest.approx(expected_reward)
    # And it must NOT equal the old percent_change*10 reward (first step is
    # 0 either way here since variance isn't established yet -- step twice
    # to get a case where they'd differ if DSR weren't actually wired in).
    _, reward2, _, _, info2 = env.step(BUY)
    assert reward2 != pytest.approx(((info2["portfolio_value"] - info["portfolio_value"]) / info["portfolio_value"]) * 10)


def test_dsr_tracker_resets_on_episode_reset():
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST", use_differential_sharpe=True)
    env.reset()
    env.step(BUY)
    env.step(HOLD)
    assert env.dsr_tracker.A != 0.0 or env.dsr_tracker.B != 0.0

    env.reset()
    assert env.dsr_tracker.A == 0.0
    assert env.dsr_tracker.B == 0.0


def test_never_trading_gives_zero_dsr_contribution():
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST", use_differential_sharpe=True)
    env.reset()

    total_reward = 0.0
    done = False
    while not done:
        _, reward, done, _, _ = env.step(HOLD)
        total_reward += reward

    assert total_reward == 0.0
