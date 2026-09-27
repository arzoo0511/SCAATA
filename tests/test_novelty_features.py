"""Causal-correctness tests for novelty wiring (Phase 10 gap fix): merging
the continuous novelty score into the feature set must not let a future
row leak backward, and insufficient trailing history must be
neutral-filled, never forward-filled -- same ethos as
`tests/test_sentiment_wiring.py`, which this mirrors.
"""
import numpy as np
import pandas as pd

from scaata.regimes.novelty_features import merge_novelty_into_features

FEATURE_COLUMNS = ["returns", "ma_10"]


def _make_synthetic_feature_df(n=80, ticker="TEST", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Ticker": ticker,
            "returns": rng.normal(0, 0.01, n),
            "ma_10": rng.normal(100, 1, n),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_merge_is_causal_under_truncation():
    feature_df = _make_synthetic_feature_df(n=80)

    t = 70
    merged_full = merge_novelty_into_features(feature_df, FEATURE_COLUMNS)
    merged_trunc = merge_novelty_into_features(feature_df.iloc[: t + 1], FEATURE_COLUMNS)

    assert merged_full["novelty_score"].iloc[t] == merged_trunc["novelty_score"].iloc[-1]


def test_insufficient_history_is_neutral_filled():
    # Fewer rows than NOVELTY_REF_WINDOW (60): every row lacks enough
    # trailing history for either novelty method, so all should neutral-fill.
    feature_df = _make_synthetic_feature_df(n=30)

    merged = merge_novelty_into_features(feature_df, FEATURE_COLUMNS)

    assert (merged["novelty_score"] == 0.0).all()


def test_no_nans_leak_through_to_the_output_column():
    feature_df = _make_synthetic_feature_df(n=80)
    merged = merge_novelty_into_features(feature_df, FEATURE_COLUMNS)
    assert not merged["novelty_score"].isna().any()


def test_merging_twice_is_idempotent_not_a_crash():
    feature_df = _make_synthetic_feature_df(n=80)

    once = merge_novelty_into_features(feature_df, FEATURE_COLUMNS)
    twice = merge_novelty_into_features(once, FEATURE_COLUMNS)

    pd.testing.assert_series_equal(once["novelty_score"], twice["novelty_score"])


def test_missing_feature_columns_gives_neutral_fill_not_a_crash():
    feature_df = _make_synthetic_feature_df(n=80)
    merged = merge_novelty_into_features(feature_df, ["column_that_does_not_exist"])
    assert (merged["novelty_score"] == 0.0).all()
