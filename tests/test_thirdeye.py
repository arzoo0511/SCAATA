"""Sanity tests for the 3rd Eye correlator and narrative faithfulness check."""
import numpy as np
import pandas as pd

from scaata.thirdeye.correlator import era_correlation_report, granger_causality_test
from scaata.thirdeye.narrative import faithfulness_check


def _make_market_and_sentiment(n=200, seed=0):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    market_df = pd.DataFrame({"returns": returns}, index=dates)
    market_df["rolling_vol"] = market_df["returns"].rolling(20).std()

    sentiment_df = pd.DataFrame({
        "date": dates,
        "sentiment_score": rng.normal(0, 1, n),
        "mention_volume": rng.integers(1, 50, n),
    })
    return market_df, sentiment_df


def test_era_correlation_report_produces_both_eras():
    market_df, sentiment_df = _make_market_and_sentiment()
    report = era_correlation_report(
        market_df, sentiment_df, ("2020-01-01", "2020-06-01"), ("2020-06-02", "2020-12-31")
    )
    assert "era_early" in report and "era_late" in report
    assert report["era_early"]["n_days"] > 0
    assert report["era_late"]["n_days"] > 0


def test_granger_causality_reports_error_on_insufficient_data():
    tiny_sentiment = pd.Series([0.1, 0.2, 0.3])
    tiny_target = pd.Series([0.01, 0.02, 0.03])
    result = granger_causality_test(tiny_sentiment, tiny_target, max_lag=5)
    assert "error" in result


def test_granger_causality_runs_on_adequate_data():
    rng = np.random.default_rng(0)
    n = 100
    sentiment = pd.Series(rng.normal(0, 1, n))
    target = pd.Series(rng.normal(0, 1, n))
    result = granger_causality_test(sentiment, target, max_lag=3)
    assert "p_values_by_lag" in result
    assert len(result["p_values_by_lag"]) == 3


def test_faithfulness_check_flags_matching_direction():
    report = {"era_early": {"corr_vol": -0.2}, "era_late": {"corr_vol": 0.3}}
    narrative = "The correlation increased between the two periods."
    result = faithfulness_check(narrative, report)
    assert result["checked"] is True
    assert result["faithful"] is True


def test_faithfulness_check_flags_contradicting_direction():
    report = {"era_early": {"corr_vol": -0.2}, "era_late": {"corr_vol": 0.3}}
    narrative = "The correlation decreased between the two periods."
    result = faithfulness_check(narrative, report)
    assert result["checked"] is True
    assert result["faithful"] is False
