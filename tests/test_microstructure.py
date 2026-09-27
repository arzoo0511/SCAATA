"""Tests for the OHLC-only microstructure proxies (Phase 9c): the
Corwin-Schultz spread estimator must clamp negative artifacts at 0, and
per-ticker computation must not leak values across tickers that share the
same Date index — a real risk here since `add_microstructure_features`
assigns computed columns back onto a DataFrame whose Date index repeats
across tickers.
"""
import numpy as np
import pandas as pd

from scaata.features.microstructure import (
    add_microstructure_features,
    amihud_illiquidity,
    corwin_schultz_spread,
)


def _make_ticker_df(ticker, n, seed, price_scale=100):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = price_scale * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    df = pd.DataFrame({"High": high, "Low": low, "Close": close, "Volume": volume, "Ticker": ticker}, index=dates)
    df.index.name = "Date"
    return df


def test_corwin_schultz_spread_is_nonnegative():
    df = _make_ticker_df("TEST", 100, seed=0)
    spread = corwin_schultz_spread(df["High"], df["Low"])
    assert (spread.dropna() >= 0).all()
    assert pd.isna(spread.iloc[0])  # first row has no prior day to pair with


def test_amihud_illiquidity_higher_for_low_volume_days():
    returns = pd.Series([0.01] * 25)
    high_volume = pd.Series([1_000_000] * 25)
    low_volume = pd.Series([10_000] * 25)

    illiq_high_vol = amihud_illiquidity(returns, high_volume * returns.abs().replace(0, 1), window=20)
    illiq_low_vol = amihud_illiquidity(returns, low_volume * returns.abs().replace(0, 1), window=20)

    assert illiq_low_vol.iloc[-1] > illiq_high_vol.iloc[-1]


def test_add_microstructure_features_does_not_cross_contaminate_tickers():
    # Two tickers sharing the exact same Date index (the normal shape of
    # this project's multi-ticker frames) but very different price/volume
    # scales, so a cross-ticker mixup would be numerically obvious.
    df_a = _make_ticker_df("AAA", 80, seed=1, price_scale=50)
    df_b = _make_ticker_df("BBB", 80, seed=2, price_scale=5000)
    combined = pd.concat([df_a, df_b])

    result = add_microstructure_features(combined, amihud_window=20)

    expected_a = add_microstructure_features(df_a, amihud_window=20)
    expected_b = add_microstructure_features(df_b, amihud_window=20)

    got_a = result[result["Ticker"] == "AAA"]["cs_spread"]
    got_b = result[result["Ticker"] == "BBB"]["cs_spread"]

    np.testing.assert_allclose(got_a.values, expected_a["cs_spread"].values, equal_nan=True)
    np.testing.assert_allclose(got_b.values, expected_b["cs_spread"].values, equal_nan=True)
