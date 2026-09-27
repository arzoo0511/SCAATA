"""Tests for the weekly retrain entry point's reporting: failures must be
loud, not silent (six weeks of all-skipped runs plus a failing email went
unnoticed)."""
import pytest

import scaata.live.run_scheduled_retrain as job


def _evaluated(**overrides):
    row = {
        "ticker": "AAPL", "status": "evaluated", "deployed": False,
        "candidate_sharpe": 0.5, "incumbent_sharpe": 0.7, "buy_and_hold_sharpe": 0.9,
        "prob_beats_incumbent": 0.31, "prob_beats_buy_and_hold": 0.25,
        "data_source": "live", "staleness_days": 1, "reason": "keeping the incumbent live",
    }
    row.update(overrides)
    return row


def test_all_skipped_run_carries_a_warning():
    summary = job._format_summary([
        {"ticker": "AAPL", "status": "skipped", "deployed": False, "reason": "too few rows"},
        {"ticker": "MSFT", "status": "error", "deployed": False, "reason": "rate limited"},
    ])
    assert "WARNING: no ticker was evaluated" in summary


def test_evaluated_run_reports_gate_probabilities_and_no_warning():
    summary = job._format_summary([_evaluated()])
    assert "WARNING" not in summary
    assert "vs incumbent 0.31" in summary
    assert "vs buy&hold 0.25" in summary


def test_first_run_and_stale_data_are_labelled():
    summary = job._format_summary([_evaluated(incumbent_sharpe=None, prob_beats_incumbent=None,
                                              data_source="stale_cache:2026-07-19", staleness_days=57)])
    assert "n/a (first run)" in summary
    assert "stale_cache:2026-07-19, 57d old" in summary


def test_email_failure_exits_nonzero(monkeypatch):
    monkeypatch.setattr(job, "run_scheduled_retrain", lambda: [_evaluated()])
    monkeypatch.setenv("GMAIL_SENDER_ADDRESS", "someone@example.com")

    def _fail(*a, **k):
        raise RuntimeError("535 bad credentials")

    monkeypatch.setattr(job, "send_email", _fail)

    with pytest.raises(SystemExit) as exc:
        job.main()
    assert exc.value.code == 1
