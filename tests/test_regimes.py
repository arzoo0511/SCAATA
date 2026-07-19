"""Causal-correctness tests for regime detection.

The core property under test: a regime label at time t must depend only on
data available up to time t. We verify this by computing labels twice —
once on the full series, once on a series truncated right after time t —
and confirming time t's label is identical in both runs. If the detector
were leaking future information (e.g. via a centered rolling window or a
full-sample percentile), truncating the series would change earlier labels.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.regimes.detector import (
    compute_rolling_drawdown,
    compute_rolling_vol,
    threshold_regime_labels,
)


def _make_synthetic_ticker_df(n=400, seed=0, ticker="TEST"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # Inject a clear "crash" segment in the middle so we can sanity-check
    # the detector actually flags it, plus generic noise elsewhere.
    returns = rng.normal(0.0005, 0.01, size=n)
    returns[150:170] = rng.normal(-0.04, 0.05, size=20)  # crash segment
    close = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({"Close": close, "returns": returns, "Ticker": ticker}, index=dates)
    return df


def test_rolling_vol_and_drawdown_are_causal_under_truncation():
    df = _make_synthetic_ticker_df()
    t = 250

    vol_full = compute_rolling_vol(df["returns"])
    dd_full = compute_rolling_drawdown(df["Close"])

    truncated = df.iloc[: t + 1]
    vol_trunc = compute_rolling_vol(truncated["returns"])
    dd_trunc = compute_rolling_drawdown(truncated["Close"])

    assert vol_full.iloc[t] == pytest.approx(vol_trunc.iloc[-1], nan_ok=True)
    assert dd_full.iloc[t] == pytest.approx(dd_trunc.iloc[-1])


def test_threshold_regime_labels_are_causal_under_truncation():
    df = _make_synthetic_ticker_df()
    t = 300

    full_labels = threshold_regime_labels(df)
    truncated_labels = threshold_regime_labels(df.iloc[: t + 1])

    assert full_labels["regime"].iloc[t] == truncated_labels["regime"].iloc[-1]
    assert full_labels["vol_regime"].iloc[t] == truncated_labels["vol_regime"].iloc[-1]
    assert full_labels["drawdown_regime"].iloc[t] == truncated_labels["drawdown_regime"].iloc[-1]


def test_threshold_detector_flags_the_injected_crash():
    df = _make_synthetic_ticker_df()
    labels = threshold_regime_labels(df)
    crash_window = labels.iloc[170:185]  # shortly after the injected crash
    assert (crash_window["drawdown_regime"] == "stress").any(), (
        "detector failed to flag an obvious, deliberately injected drawdown"
    )
