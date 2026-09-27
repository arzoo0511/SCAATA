"""Tests for the equity cash-and-carry email content builder (Phase 17):
a pure function with no side effects -- must raise on a non-opportunity
row, and must include the key numbers (strike, expiration, implied vs
risk-free rate) a human needs to evaluate the opportunity before acting.
"""
import pytest

from scaata.notify.equity_email_draft import build_equity_opportunity_email


def _opportunity_row(**overrides):
    row = {
        "opportunity": True,
        "symbol": "AAPL",
        "direction": "conversion",
        "implied_rate": 0.10,
        "risk_free_rate": 0.05,
        "strike": 220.0,
        "expiration": "2026-08-15",
        "timestamp_utc": "2026-07-25T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_raises_on_non_opportunity_row():
    row = _opportunity_row(opportunity=False)
    with pytest.raises(ValueError):
        build_equity_opportunity_email(row)


def test_subject_includes_symbol_strike_and_direction():
    result = build_equity_opportunity_email(_opportunity_row())
    assert "AAPL" in result["subject"]
    assert "220.00" in result["subject"]
    assert "conversion" in result["subject"]


def test_body_includes_rates_and_gap():
    result = build_equity_opportunity_email(_opportunity_row())
    assert "10.00%" in result["body"]
    assert "5.00%" in result["body"]
    assert "5.00%" in result["body"].split("Gap:")[1]


def test_reversal_direction_is_described_correctly():
    row = _opportunity_row(direction="reversal", implied_rate=0.01, risk_free_rate=0.05)
    result = build_equity_opportunity_email(row)
    assert "reversal" in result["body"].lower()
    assert "sell stock short" in result["body"]


def test_has_no_side_effects_beyond_return_value(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("build_equity_opportunity_email must not perform any network call")

    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(requests, "post", _boom)

    build_equity_opportunity_email(_opportunity_row())
