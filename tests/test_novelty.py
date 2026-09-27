"""Causal-correctness + sanity tests for the continuous novelty detector
(Phase 10), following the same truncation-test ethos as
`tests/test_regimes.py`: a novelty score at time t must depend only on data
available up to time t, and an injected distributional shift should
actually produce a higher novelty score than the surrounding calm segment.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.regimes.novelty import (
    combined_novelty_score,
    rolling_discriminator_novelty,
    rolling_mahalanobis_d2,
    rolling_mahalanobis_novelty,
)

FEATURE_COLUMNS = ["f1", "f2", "f3"]


def _make_synthetic_feature_df(n=300, seed=0, ticker="TEST", shift_start=150, shift_end=200):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0, 1, size=(n, len(FEATURE_COLUMNS)))
    # Inject a clear distributional shift (much higher variance and a mean
    # offset) so we can sanity-check novelty actually flags it.
    data[shift_start:shift_end] = rng.normal(6, 4, size=(shift_end - shift_start, len(FEATURE_COLUMNS)))
    df = pd.DataFrame(data, columns=FEATURE_COLUMNS, index=dates)
    df["Ticker"] = ticker
    return df


def test_mahalanobis_d2_is_causal_under_truncation():
    df = _make_synthetic_feature_df()
    t = 250

    d2_full = rolling_mahalanobis_d2(df, FEATURE_COLUMNS, ref_window=60)
    d2_trunc = rolling_mahalanobis_d2(df.iloc[: t + 1], FEATURE_COLUMNS, ref_window=60)

    assert d2_full.iloc[t] == pytest.approx(d2_trunc.iloc[-1], nan_ok=True)


def test_mahalanobis_novelty_flags_the_injected_shift():
    df = _make_synthetic_feature_df()
    novelty = rolling_mahalanobis_novelty(df, FEATURE_COLUMNS, ref_window=60)

    shift_window = novelty.iloc[150:170]
    calm_window = novelty.iloc[60:140]

    assert shift_window.mean() > calm_window.mean(), (
        "Mahalanobis novelty failed to flag an obvious, deliberately injected distributional shift"
    )


def test_discriminator_novelty_is_causal_under_truncation():
    df = _make_synthetic_feature_df(n=200)
    t = 150

    result_full = rolling_discriminator_novelty(
        df, FEATURE_COLUMNS, recent_window=20, historical_window=60, refit_every_days=5
    )
    result_trunc = rolling_discriminator_novelty(
        df.iloc[: t + 1], FEATURE_COLUMNS, recent_window=20, historical_window=60, refit_every_days=5
    )

    full_val = result_full["discriminator_novelty"].iloc[t]
    trunc_val = result_trunc["discriminator_novelty"].iloc[-1]
    assert full_val == pytest.approx(trunc_val, nan_ok=True)


def test_discriminator_novelty_flags_the_injected_shift():
    df = _make_synthetic_feature_df(n=250, shift_start=150, shift_end=200)
    result = rolling_discriminator_novelty(
        df, FEATURE_COLUMNS, recent_window=20, historical_window=60, refit_every_days=5
    )

    # Just after the shift begins, the "recent" window starts overlapping
    # the shifted segment while "historical" is still mostly calm data —
    # the discriminator should find that easy to separate (high novelty).
    post_shift_scores = result["discriminator_novelty"].iloc[160:180].dropna()
    pre_shift_scores = result["discriminator_novelty"].iloc[80:140].dropna()

    assert len(post_shift_scores) > 0 and len(pre_shift_scores) > 0
    assert post_shift_scores.mean() > pre_shift_scores.mean()


def test_combined_novelty_score_falls_back_to_whichever_is_available():
    mahalanobis = pd.Series([0.2, np.nan, 0.6], index=[0, 1, 2])
    discriminator = pd.Series([np.nan, 0.9, 0.4], index=[0, 1, 2])

    combined = combined_novelty_score(mahalanobis, discriminator)

    assert combined.iloc[0] == pytest.approx(0.2)
    assert combined.iloc[1] == pytest.approx(0.9)
    assert combined.iloc[2] == pytest.approx(0.5)
