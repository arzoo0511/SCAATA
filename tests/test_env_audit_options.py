"""Tests for the opt-in environment fixes from the 2026-09-14 audit:
position state in the observation, scale-free price/volume features, and
random episode start points. Every option defaults off, so existing
policies and results are unaffected."""
import numpy as np
import pandas as pd
import pytest

from scaata.config import FEATURE_COLUMNS, STATIONARY_FEATURE_COLUMNS
from scaata.features.technical import add_features
from scaata.rl.env import BUY, HOLD, HOLDING_DAYS_SCALE, POSITION_OBS_SIZE, SELL, RobustTradingEnv


def _env_df(close, ticker="AAA"):
    close = np.asarray(close, dtype=float)
    n = len(close)
    df = pd.DataFrame({
        "Close": close, "Volume": np.full(n, 1e6), "volume_ma_30": np.full(n, 1e6), "Ticker": ticker,
    }, index=pd.bdate_range("2024-01-01", periods=n))
    for col in FEATURE_COLUMNS:
        if col not in df:
            df[col] = np.linspace(-1, 1, n)
    return df


def _raw(n=200, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)) * scale
    df = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n) * scale, "Ticker": "AAA",
    }, index=pd.bdate_range("2024-01-01", periods=n))
    df.index.name = "Date"
    return df


def test_default_observation_is_unchanged():
    env = RobustTradingEnv(_env_df(np.linspace(100, 110, 30)), FEATURE_COLUMNS, fixed_ticker="AAA")
    obs, _ = env.reset()
    assert obs.shape == (len(FEATURE_COLUMNS),)
    assert env.observation_space.shape == (len(FEATURE_COLUMNS),)


def test_position_observation_tracks_entry_pnl_and_holding_time():
    close = np.linspace(100, 130, 30)
    env = RobustTradingEnv(_env_df(close), FEATURE_COLUMNS, fixed_ticker="AAA",
                           include_position_obs=True, stop_loss_pct=-1.0)
    obs, _ = env.reset()
    assert obs.shape == (len(FEATURE_COLUMNS) + POSITION_OBS_SIZE,)
    assert env.observation_space.shape == obs.shape
    assert obs[-POSITION_OBS_SIZE:].tolist() == [0.0, 0.0, 0.0]

    obs, *_ = env.step(BUY)
    obs, *_ = env.step(HOLD)
    position, unrealized, held = obs[-POSITION_OBS_SIZE:]
    assert position == 1.0
    assert unrealized == pytest.approx(close[2] / close[0] - 1, rel=1e-5)
    assert held == pytest.approx(2 / HOLDING_DAYS_SCALE)

    obs, *_ = env.step(SELL)
    assert obs[-POSITION_OBS_SIZE:].tolist() == [0.0, 0.0, 0.0]


def test_stop_loss_exit_clears_the_position_features():
    close = np.r_[100.0, 90.0, np.full(10, 90.0)]  # -10% the day after entry trips the -2% stop
    env = RobustTradingEnv(_env_df(close), FEATURE_COLUMNS, fixed_ticker="AAA", include_position_obs=True)
    env.reset()

    env.step(BUY)
    obs, *_ = env.step(HOLD)

    assert env.position == 0
    assert obs[-POSITION_OBS_SIZE:].tolist() == [0.0, 0.0, 0.0]


def test_terminal_observation_has_the_same_shape():
    env = RobustTradingEnv(_env_df(np.linspace(100, 110, 5)), FEATURE_COLUMNS, fixed_ticker="AAA",
                           include_position_obs=True)
    env.reset()
    done = False
    while not done:
        obs, _, done, _, _ = env.step(BUY)
    assert obs.shape == (len(FEATURE_COLUMNS) + POSITION_OBS_SIZE,)


def test_random_episode_start_varies_the_start_and_keeps_a_minimum_length():
    df = _env_df(np.linspace(100, 120, 300))
    env = RobustTradingEnv(df, FEATURE_COLUMNS, random_episode_start_min_steps=100)
    lengths = set()
    for seed in range(20):
        env.reset(seed=seed)
        assert env.n_steps >= 100
        lengths.add(env.n_steps)
    assert len(lengths) > 1


def test_random_start_never_applies_to_fixed_ticker_backtests():
    df = _env_df(np.linspace(100, 120, 300))
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="AAA", random_episode_start_min_steps=100)
    env.reset(seed=3)
    assert env.n_steps == len(df)


def test_stationary_features_do_not_depend_on_price_or_volume_scale():
    small = add_features(_raw(scale=1.0))
    large = add_features(_raw(scale=10.0))

    np.testing.assert_allclose(small[STATIONARY_FEATURE_COLUMNS].values, large[STATIONARY_FEATURE_COLUMNS].values, rtol=1e-6)
    assert not np.allclose(small["ma_50"], large["ma_50"])  # the raw-level features the default set uses do


def test_train_and_backtest_accept_the_options():
    from scaata.rl.train import backtest_ppo, train_ppo

    df = add_features(_raw(n=260))
    model = train_ppo(df, STATIONARY_FEATURE_COLUMNS, total_timesteps=10,
                      include_position_obs=True, random_episode_start_min_steps=100)
    assert model.observation_space.shape == (len(STATIONARY_FEATURE_COLUMNS) + POSITION_OBS_SIZE,)

    equity, actions = backtest_ppo(model, df, STATIONARY_FEATURE_COLUMNS, "AAA", include_position_obs=True)
    assert len(equity) == len(df)
    assert len(actions) == len(df) - 1


def test_fixed_fee_is_charged_on_every_trade_regardless_of_liquidity():
    """India's STT/stamp duty don't shrink on busy days; the liquidity-scaled
    fee can drop to half its base, the fixed fee must not."""
    from scaata.config import INDIA_STATUTORY_COST_PER_SIDE

    df = _env_df(np.full(10, 100.0))
    df["Volume"] = 1e7  # very liquid day: liquidity scaling hits its floor
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="AAA", transaction_fee=0.0,
                           fixed_fee=INDIA_STATUTORY_COST_PER_SIDE, stop_loss_pct=-1.0)
    env.reset()
    env.step(BUY)
    _, _, _, _, info = env.step(SELL)

    expected = 10_000 * (1 - INDIA_STATUTORY_COST_PER_SIDE) ** 2
    assert info["portfolio_value"] == pytest.approx(expected, rel=1e-9)


def test_india_statutory_cost_matches_its_components():
    from scaata.config import INDIA_STATUTORY_COST_BUY, INDIA_STATUTORY_COST_PER_SIDE, INDIA_STATUTORY_COST_SELL

    assert INDIA_STATUTORY_COST_BUY == pytest.approx(0.0011862, abs=1e-7)   # STT + stamp + exchange + SEBI + GST
    assert INDIA_STATUTORY_COST_SELL == pytest.approx(0.0010362, abs=1e-7)  # no stamp duty on sells
    assert INDIA_STATUTORY_COST_PER_SIDE == pytest.approx(0.0011112, abs=1e-7)
