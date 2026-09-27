"""Tests for sentiment velocity/mention-volume velocity (Phase 9c): both
must be trailing (diff/pct_change look only backward) and must never mix
values across a ticker boundary.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.features.sentiment import (
    attach_sentiment_velocity,
    mention_volume_velocity,
    sentiment_velocity,
)


def test_sentiment_velocity_is_trailing_diff():
    df = pd.DataFrame({"sentiment_score": [0.0, 0.1, 0.2, 0.5, 0.9]})
    vel = sentiment_velocity(df, window=2)
    assert np.isnan(vel.iloc[0])
    assert np.isnan(vel.iloc[1])
    assert vel.iloc[2] == pytest.approx(0.2)
    assert vel.iloc[4] == pytest.approx(0.7)


def test_mention_volume_velocity_is_pct_change():
    df = pd.DataFrame({"mention_volume": [10, 10, 10, 20, 40]})
    vel = mention_volume_velocity(df, window=1)
    assert vel.iloc[3] == 1.0  # 20 vs 10 -> +100%
    assert vel.iloc[4] == 1.0  # 40 vs 20 -> +100%


def test_velocity_does_not_cross_ticker_boundary():
    df = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
        "sentiment_score": [0.0, 0.5, 1.0, 10.0, 10.5, 11.0],
        "mention_volume": [5, 10, 15, 100, 110, 120],
    })
    out = attach_sentiment_velocity(df, tone_window=1, volume_window=1)

    # first row of each ticker group has no prior row within that group
    aaa_first = out[out["ticker"] == "AAA"].iloc[0]
    bbb_first = out[out["ticker"] == "BBB"].iloc[0]
    assert np.isnan(aaa_first["sentiment_velocity"])
    assert np.isnan(bbb_first["sentiment_velocity"])

    bbb_second = out[out["ticker"] == "BBB"].iloc[1]
    assert bbb_second["sentiment_velocity"] == pytest.approx(0.5)  # 10.5 - 10.0, not 10.5 - 1.0
