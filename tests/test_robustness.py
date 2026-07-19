"""Regression test: the strategy-pool robustness check must show that
deliberately poisoned (sign-reversed) strategies get down-weighted relative
to good ones by the Critique <-> Meta-Selector feedback loop. Uses a fully
synthetic price series (no network access) for determinism.
"""
import numpy as np
import pandas as pd

from scaata.config import FEATURE_COLUMNS
from scaata.evaluation.robustness import run_robustness_test
from scaata.strategies.scraper import mock_strategies
from scaata.strategies.normalizer import validate_mock_strategies


def _make_synthetic_train_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["returns"] = pd.Series(close, index=dates).pct_change().fillna(0)
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(30, min_periods=1).mean()
    df["Ticker"] = "TEST"
    return df


def test_poisoned_strategies_are_down_weighted_relative_to_good_ones():
    train_df = _make_synthetic_train_df()
    good = validate_mock_strategies(mock_strategies())
    assert len(good) >= 2

    report = run_robustness_test(train_df, good, n_bad_strategies=2, max_iterations=4)

    assert report["down_weighted_as_expected"] is not None
    assert report["mean_good_weight"] >= report["mean_bad_weight"]
