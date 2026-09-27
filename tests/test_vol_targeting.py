"""Tests for volatility-targeting position sizing: arithmetic/edge-case
correctness on synthetic data, and the actual falsifiable hypothesis --
applying vol-targeting to a fixed "always long" signal on REAL AAPL data
should reduce max drawdown relative to full-size Buy&Hold, without
requiring any RL training (this overlay works on any signal source).
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import TEST_START, TEST_END
from scaata.data.loaders import load_market_data
from scaata.evaluation.metrics import max_drawdown, sharpe_ratio, sortino_ratio
from scaata.features.technical import add_features
from scaata.rl.env import BUY
from scaata.rl.train import buy_and_hold_equity
from scaata.rl.vol_targeting import (
    backtest_ppo_with_vol_targeting,
    compute_realized_volatility,
    simulate_with_vol_targeting,
    volatility_target_size,
)

# load_market_data caches by the EXACT (start, end) string pair -- a
# request for a sub-range of an already-cached larger range is still a
# cache miss and falls through to a live (rate-limited-prone) download.
# Every real-data script this session loads the full cached range
# (2020-01-01 to 2026-07-19) and filters to the test window in-memory
# instead; this test follows the same pattern.
_FULL_RANGE_START = "2020-01-01"
_FULL_RANGE_END = "2026-07-19"


def test_compute_realized_volatility_matches_manual_rolling_std():
    returns = pd.Series([0.01, -0.02, 0.015, 0.03, -0.01, 0.02, 0.005, -0.015, 0.01, 0.02])
    vol = compute_realized_volatility(returns, window=5)
    expected_at_5 = returns.iloc[1:6].std()  # rolling window ending at index 5
    assert vol.iloc[5] == pytest.approx(expected_at_5)
    assert pd.isna(vol.iloc[3])  # not enough history yet (min_periods=window)


def test_volatility_target_size_shrinks_under_high_vol():
    size = volatility_target_size(realized_vol=0.06, target_vol=0.015, min_size=0.1, max_size=1.0)
    assert size == pytest.approx(0.25)  # 0.015/0.06


def test_volatility_target_size_caps_at_max_under_low_vol():
    size = volatility_target_size(realized_vol=0.001, target_vol=0.015, min_size=0.1, max_size=1.0)
    assert size == 1.0  # 0.015/0.001 = 15, clipped down to max_size


def test_volatility_target_size_floors_at_min_under_extreme_vol():
    size = volatility_target_size(realized_vol=1.0, target_vol=0.015, min_size=0.1, max_size=1.0)
    assert size == 0.1


def test_volatility_target_size_defaults_to_max_when_vol_unknown():
    assert volatility_target_size(realized_vol=float("nan")) == 1.0
    assert volatility_target_size(realized_vol=0.0) == 1.0
    assert volatility_target_size(realized_vol=-0.01) == 1.0


def test_vol_targeting_reduces_drawdown_on_real_aapl_data():
    """The actual hypothesis: applying vol-targeting to a constant
    'always long' signal (the same signal Buy&Hold implicitly follows) on
    REAL AAPL test-period data should reduce max drawdown relative to
    always being fully sized -- no RL training involved, this tests the
    sizing overlay in isolation.
    """
    aapl_raw = load_market_data(["AAPL"], _FULL_RANGE_START, _FULL_RANGE_END, use_cache=True)
    aapl_full = add_features(aapl_raw)  # simulate_with_vol_targeting needs the "returns" feature column
    aapl_test = aapl_full[(aapl_full.index >= TEST_START) & (aapl_full.index <= TEST_END)]
    always_long_signal = np.ones(len(aapl_test), dtype=int)

    vol_targeted_equity, sizes_used = simulate_with_vol_targeting(aapl_test, always_long_signal, "AAPL")
    plain_bh_equity = buy_and_hold_equity(aapl_test, "AAPL")

    vol_targeted_dd = max_drawdown(vol_targeted_equity)
    plain_dd = max_drawdown(plain_bh_equity)

    assert vol_targeted_dd > plain_dd, (
        f"vol-targeted MaxDD ({vol_targeted_dd:.3%}) should be less severe than full-size Buy&Hold's "
        f"MaxDD ({plain_dd:.3%}) -- otherwise sizing down during high-volatility stretches isn't doing "
        "what it's supposed to"
    )
    # sanity: sizing must have actually varied (not just always capped at max)
    assert sizes_used.min() < 1.0
    assert sizes_used.max() == pytest.approx(1.0)

    print(f"\nReal AAPL vol-targeting result: vol_targeted_sharpe={sharpe_ratio(vol_targeted_equity):.3f} "
          f"vs plain_bh_sharpe={sharpe_ratio(plain_bh_equity):.3f}; "
          f"vol_targeted_dd={vol_targeted_dd:.3%} vs plain_dd={plain_dd:.3%}")


class _AlwaysBuyModel:
    """Minimal .predict() double -- deterministic BUY every step, so the
    resulting equity curve is directly comparable to always_long_signal in
    the test above without needing a real trained policy."""

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return BUY, None


def test_backtest_ppo_with_vol_targeting_sizes_down_in_high_vol(monkeypatch):
    """The model-driven variant must produce the same kind of sizing
    behavior as simulate_with_vol_targeting (varying, capped at max) when
    driven by a policy that always says BUY -- confirms the env-stepping
    plumbing (obs/lstm_state handling) doesn't silently break the sizing."""
    aapl_raw = load_market_data(["AAPL"], _FULL_RANGE_START, _FULL_RANGE_END, use_cache=True)
    aapl_full = add_features(aapl_raw)
    aapl_test = aapl_full[(aapl_full.index >= TEST_START) & (aapl_full.index <= TEST_END)]

    from scaata.config import FEATURE_COLUMNS

    equity, actions, sizes = backtest_ppo_with_vol_targeting(_AlwaysBuyModel(), aapl_test, FEATURE_COLUMNS, "AAPL")

    assert (actions == BUY).all()
    assert sizes.min() < 1.0
    assert sizes.max() == pytest.approx(1.0)
    assert len(equity) == len(sizes) + 1
