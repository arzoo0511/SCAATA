"""Tests for `RobustTradingEnv`'s `size_multiplier` kwarg (Phase 14):
default behavior must remain bit-identical to the pre-Phase-14 all-in
logic (a regression guard, since every existing SB3 training call site
never passes this kwarg and must be unaffected), and partial sizing must
correctly invest only a fraction of cash while still fully accounting for
portfolio value.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import FEATURE_COLUMNS
from scaata.rl.env import BUY, HOLD, SELL, RobustTradingEnv


def _make_env_df(n=20, seed=0, ticker="TEST"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(5, min_periods=1).mean()
    df["Ticker"] = ticker
    return df


def test_default_size_multiplier_reproduces_original_all_in_behavior():
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST")
    env.reset()

    obs, reward, done, truncated, info = env.step(BUY)  # no size_multiplier passed -- must default to 1.0

    assert env.cash == 0.0  # fully invested, exactly like before this phase
    assert env.shares > 0
    assert info["portfolio_value"] == pytest.approx(env.cash + env.shares * env.prices[env.step_idx - 1])


def test_partial_size_multiplier_leaves_leftover_cash():
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST")
    env.reset()
    initial_cash = env.cash

    env.step(BUY, size_multiplier=0.4)

    assert env.cash == pytest.approx(initial_cash * 0.6)
    assert env.shares > 0
    # portfolio value must still account for 100% of capital (cash + invested value)
    current_price = env.prices[env.step_idx - 1]
    assert (env.cash + env.shares * current_price) == pytest.approx(env.portfolio_value)


def test_zero_size_multiplier_is_equivalent_to_not_buying():
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST")
    env.reset()
    initial_cash = env.cash

    env.step(BUY, size_multiplier=0.0)

    assert env.shares == 0.0
    assert env.cash == pytest.approx(initial_cash)
    # position is still marked "open" at zero size -- a known, documented
    # narrow-scope simplification (single-shot sizing, no position
    # top-ups) -- but a subsequent SELL must not crash or misbehave.
    env.step(SELL)
    assert env.position == 0


def test_full_liquidation_on_sell_regardless_of_entry_size():
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST")
    env.reset()

    env.step(BUY, size_multiplier=0.5)
    assert env.position == 1
    shares_held = env.shares

    env.step(SELL)

    assert env.position == 0
    assert env.shares == 0.0
    assert shares_held > 0  # sanity: there was actually something to liquidate


def test_partial_buy_then_sell_preserves_leftover_cash():
    """Regression test for a real bug caught via real-strategy testing: a
    partial-size BUY (size_multiplier < 1.0) leaves a nonzero `self.cash`
    remainder sitting alongside the position. The SELL branch used to
    *overwrite* `self.cash` with just the liquidation proceeds instead of
    adding to it, silently destroying that leftover cash on every
    partial-buy-then-sell cycle -- invisible in any test that only ever
    bought once and held (size_multiplier=1.0 always left exactly 0
    leftover cash, masking the bug), but catastrophic (60-99%+ drawdowns)
    for any strategy that actually cycles in and out of positions with
    partial sizing, e.g. volatility-targeted sizing on a real trading
    signal. Portfolio value must be conserved (up to fees) across the
    whole cycle, not just "shares went to zero".
    """
    df = _make_env_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST")
    env.reset()
    initial_cash = env.cash

    env.step(BUY, size_multiplier=0.3)  # 70% of cash should remain uninvested
    leftover_cash_after_buy = env.cash
    assert leftover_cash_after_buy == pytest.approx(initial_cash * 0.7)

    env.step(HOLD)  # let one day pass with the position open
    env.step(SELL)

    assert env.position == 0
    assert env.shares == 0.0
    # the leftover 70% must still be there -- portfolio value should only
    # be down by the round-trip transaction fees, not by ~30% of assets
    # (which is what the bug did: it discarded the leftover cash entirely).
    assert env.cash > leftover_cash_after_buy  # sold shares added TO the leftover, not replaced it
    assert env.portfolio_value > initial_cash * 0.9  # fees only, nowhere near the ~30%+ the bug destroyed


def test_partial_buy_then_stop_loss_preserves_leftover_cash():
    """Same bug, same fix, for the stop-loss exit branch -- it has the
    identical overwrite-instead-of-add pattern as the SELL branch above.
    """
    n = 20
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # a sharp, immediate drop to guarantee the stop-loss actually triggers
    close = np.concatenate([[100.0], np.full(n - 1, 90.0)])
    df = pd.DataFrame({c: 0.0 for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["Volume"] = 1_000_000.0
    df["volume_ma_30"] = 1_000_000.0
    df["Ticker"] = "TEST"

    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST", stop_loss_pct=-0.02)
    env.reset()
    initial_cash = env.cash

    env.step(BUY, size_multiplier=0.3)
    leftover_cash_after_buy = env.cash

    env.step(HOLD)  # price drops 10% here, well past the -2% stop-loss

    assert env.position == 0  # stop-loss should have fired
    assert env.cash > leftover_cash_after_buy  # liquidation proceeds added to leftover, not replacing it
    assert env.portfolio_value > initial_cash * 0.75  # a real ~3% loss on the invested 30%, not a ~70% wipeout


def test_size_multiplier_scales_shares_purchased_proportionally():
    df = _make_env_df()

    env_full = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST")
    env_full.reset()
    env_full.step(BUY, size_multiplier=1.0)

    env_half = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST")
    env_half.reset()
    env_half.step(BUY, size_multiplier=0.5)

    assert env_half.shares == pytest.approx(env_full.shares * 0.5)
