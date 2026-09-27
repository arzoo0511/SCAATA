"""Causal-correctness test for sentiment wiring (Phase 9a): merging
sentiment into the feature set must not let a future-dated sentiment
observation leak backward, and missing coverage must be neutral-filled,
never forward-filled — following the same truncation-test ethos as
`tests/test_regimes.py`.
"""
import numpy as np
import pandas as pd

from scaata.features.sentiment_features import merge_sentiment_into_features


def _make_synthetic_feature_df(n=50, ticker="TEST"):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Ticker": ticker, "returns": np.zeros(n)}, index=dates)
    df.index.name = "Date"
    return df


def _make_synthetic_sentiment_df(dates, ticker="TEST", missing_idx=None):
    missing_idx = missing_idx or set()
    rows = []
    for i, date in enumerate(dates):
        if i in missing_idx:
            continue
        rows.append({"date": date, "ticker": ticker, "sentiment_score": float(i)})
    return pd.DataFrame(rows)


def test_merge_is_causal_under_truncation():
    feature_df = _make_synthetic_feature_df()
    dates = feature_df.index
    sentiment_df = _make_synthetic_sentiment_df(dates)

    t = 30
    merged_full = merge_sentiment_into_features(feature_df, sentiment_df)
    merged_trunc = merge_sentiment_into_features(
        feature_df.iloc[: t + 1], sentiment_df[sentiment_df["date"] <= dates[t]]
    )

    assert merged_full["sentiment_score"].iloc[t] == merged_trunc["sentiment_score"].iloc[-1]


def test_missing_coverage_is_neutral_filled_not_forward_filled():
    feature_df = _make_synthetic_feature_df(n=10)
    dates = feature_df.index
    sentiment_df = _make_synthetic_sentiment_df(dates, missing_idx={5})

    merged = merge_sentiment_into_features(feature_df, sentiment_df)

    assert merged["sentiment_score"].iloc[5] == 0.0
    # neighboring days keep their real (nonzero) values — no ffill/bfill smearing
    assert merged["sentiment_score"].iloc[4] == 4.0
    assert merged["sentiment_score"].iloc[6] == 6.0


def test_missing_ticker_entirely_gets_neutral_fill():
    feature_df = _make_synthetic_feature_df(n=10, ticker="OTHER")
    sentiment_df = _make_synthetic_sentiment_df(feature_df.index, ticker="SOME_TICKER_NOT_IN_FEATURES")

    merged = merge_sentiment_into_features(feature_df, sentiment_df)

    assert (merged["sentiment_score"] == 0.0).all()


def test_merging_twice_is_idempotent_not_a_crash():
    """Regression test for a real bug caught while running the pipeline
    live: if `feature_df` already has a `sentiment_score` column (e.g. a
    caller re-merges, or passes an already-wired df into a second stage
    that merges again), pandas' merge silently renames both copies to
    `sentiment_score_x`/`_y`, and the old code raised a KeyError looking
    for the plain column name. Merging twice must produce the same result
    as merging once, not crash.
    """
    feature_df = _make_synthetic_feature_df(n=20)
    sentiment_df = _make_synthetic_sentiment_df(feature_df.index)

    once = merge_sentiment_into_features(feature_df, sentiment_df)
    twice = merge_sentiment_into_features(once, sentiment_df)

    pd.testing.assert_series_equal(once["sentiment_score"], twice["sentiment_score"])
