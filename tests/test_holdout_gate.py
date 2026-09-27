"""Tests for the final honesty/holdout gate (Phase 15): the holdout and
development ticker sets must be genuinely disjoint (and the gate must
refuse to run if they aren't), and the gate must refuse tickers outside
the designated holdout set -- both of which would otherwise silently
defeat the entire point of a holdout check.
"""
import numpy as np
import pytest

from scaata.config import CORE_TICKERS, NON_SURVIVOR_TICKERS
from scaata.evaluation import holdout_gate
from scaata.evaluation.holdout_gate import (
    DEVELOPMENT_TICKERS,
    HOLDOUT_TICKERS,
    assert_no_ticker_overlap,
    run_holdout_gate,
)


def test_holdout_and_development_sets_match_config_and_are_disjoint():
    assert set(HOLDOUT_TICKERS) == set(NON_SURVIVOR_TICKERS)
    assert set(DEVELOPMENT_TICKERS) == set(CORE_TICKERS)
    assert set(HOLDOUT_TICKERS).isdisjoint(DEVELOPMENT_TICKERS)


def test_assert_no_ticker_overlap_passes_normally():
    assert_no_ticker_overlap()  # should not raise given the real config


def test_assert_no_ticker_overlap_raises_if_sets_are_corrupted(monkeypatch):
    monkeypatch.setattr(holdout_gate, "HOLDOUT_TICKERS", ["AAPL", "XOM"])
    monkeypatch.setattr(holdout_gate, "DEVELOPMENT_TICKERS", ["AAPL", "MSFT"])
    with pytest.raises(ValueError, match="overlap"):
        assert_no_ticker_overlap()


def _make_curve(n=60, seed=0, drift=0.0005):
    rng = np.random.default_rng(seed)
    return 10_000 * np.cumprod(1 + rng.normal(drift, 0.01, n))


def test_run_holdout_gate_rejects_a_development_ticker():
    curves = {"AAPL": _make_curve()}  # AAPL is a development ticker, not holdout
    with pytest.raises(ValueError, match="outside HOLDOUT_TICKERS"):
        run_holdout_gate(curves, curves)


def test_run_holdout_gate_rejects_missing_baseline():
    ticker = HOLDOUT_TICKERS[0]
    with pytest.raises(ValueError, match="missing baseline"):
        run_holdout_gate({ticker: _make_curve()}, {})


def test_run_holdout_gate_happy_path_reports_expected_shape():
    system_curves = {t: _make_curve(seed=i, drift=0.001) for i, t in enumerate(HOLDOUT_TICKERS)}
    baseline_curves = {t: _make_curve(seed=i + 100, drift=0.0002) for i, t in enumerate(HOLDOUT_TICKERS)}

    report = run_holdout_gate(system_curves, baseline_curves)

    assert set(report["holdout_tickers"]) == set(HOLDOUT_TICKERS)
    assert set(report["per_ticker"].keys()) == set(HOLDOUT_TICKERS)
    for ticker_report in report["per_ticker"].values():
        assert "system_sharpe" in ticker_report
        assert "sharpe_bootstrap_ci" in ticker_report
    assert "p_value" in report["vs_baseline_significance"]
    assert np.isfinite(report["mean_system_sharpe"])
