"""Tests for scaata.notify.equity_opportunity_alert -- the equity-options
scan-to-draft-content wiring for cash-and-carry alerts. Parallel to
test_opportunity_alert.py (the crypto version): confirms an empty scan
produces no drafts, a real detection produces genuine email content built
from that exact row, and multiple simultaneous opportunities each get
their own draft.
"""
import scaata.notify.equity_opportunity_alert as equity_alert_module
from scaata.notify.equity_opportunity_alert import check_for_equity_opportunities, send_equity_opportunity_alerts


def _make_scan_row(symbol: str, opportunity: bool, implied_rate: float = 0.10, direction: str = "conversion"):
    return {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "symbol": symbol,
        "spot_price": 200.0,
        "strike": 200.0,
        "expiration": "2026-08-15",
        "implied_rate": implied_rate,
        "risk_free_rate": 0.05,
        "opportunity": opportunity,
        "direction": direction if opportunity else None,
        "data_complete": True,
        "price_source": "live",
        "chain_source": "live",
        "rate_source": "live",
    }


def test_no_drafts_when_no_opportunity_detected(monkeypatch):
    monkeypatch.setattr(
        equity_alert_module, "run_equity_carry_scan",
        lambda symbols, threshold: [_make_scan_row("AAPL", opportunity=False)],
    )

    drafts = check_for_equity_opportunities()
    assert drafts == []


def test_draft_produced_for_a_genuine_opportunity(monkeypatch):
    row = _make_scan_row("AAPL", opportunity=True, implied_rate=0.12)
    monkeypatch.setattr(equity_alert_module, "run_equity_carry_scan", lambda symbols, threshold: [row])

    drafts = check_for_equity_opportunities()

    assert len(drafts) == 1
    assert drafts[0]["scan_row"] == row
    assert "AAPL" in drafts[0]["subject"]
    assert "conversion" in drafts[0]["subject"]


def test_multiple_opportunities_each_get_their_own_draft(monkeypatch):
    rows = [
        _make_scan_row("AAPL", opportunity=True, implied_rate=0.12),
        _make_scan_row("MSFT", opportunity=False),
        _make_scan_row("GOOGL", opportunity=True, implied_rate=0.01, direction="reversal"),
    ]
    monkeypatch.setattr(equity_alert_module, "run_equity_carry_scan", lambda symbols, threshold: rows)

    drafts = check_for_equity_opportunities()

    assert len(drafts) == 2
    subjects = [d["subject"] for d in drafts]
    assert any("AAPL" in s for s in subjects)
    assert any("GOOGL" in s for s in subjects)


def test_check_for_equity_opportunities_passes_through_symbols_and_threshold(monkeypatch):
    captured = {}

    def _fake_scan(symbols, threshold):
        captured["symbols"] = symbols
        captured["threshold"] = threshold
        return []

    monkeypatch.setattr(equity_alert_module, "run_equity_carry_scan", _fake_scan)

    check_for_equity_opportunities(symbols=["NVDA"], threshold=0.05)
    assert captured["symbols"] == ["NVDA"]
    assert captured["threshold"] == 0.05


def test_send_equity_opportunity_alerts_sends_nothing_when_no_opportunity(monkeypatch):
    monkeypatch.setattr(
        equity_alert_module, "run_equity_carry_scan",
        lambda symbols, threshold: [_make_scan_row("AAPL", opportunity=False)],
    )
    send_calls = []
    monkeypatch.setattr(equity_alert_module, "send_email", lambda *a, **k: send_calls.append(a) or ({}, "live"))

    results = send_equity_opportunity_alerts("me@example.com")

    assert results == []
    assert send_calls == []


def test_send_equity_opportunity_alerts_sends_real_email_for_genuine_opportunity(monkeypatch):
    row = _make_scan_row("AAPL", opportunity=True, implied_rate=0.12)
    monkeypatch.setattr(equity_alert_module, "run_equity_carry_scan", lambda symbols, threshold: [row])

    send_calls = []

    def _fake_send(to, subject, body):
        send_calls.append((to, subject, body))
        return {"to": to, "subject": subject}, "live"

    monkeypatch.setattr(equity_alert_module, "send_email", _fake_send)

    results = send_equity_opportunity_alerts("me@example.com")

    assert len(results) == 1
    assert results[0]["send_source"] == "live"
    assert results[0]["scan_row"] == row
    assert len(send_calls) == 1
    to, subject, body = send_calls[0]
    assert to == "me@example.com"
    assert subject == results[0]["subject"]
    assert body == results[0]["body"]


def test_send_equity_opportunity_alerts_reports_mock_source_without_credentials(monkeypatch):
    row = _make_scan_row("AAPL", opportunity=True)
    monkeypatch.setattr(equity_alert_module, "run_equity_carry_scan", lambda symbols, threshold: [row])
    monkeypatch.setattr(equity_alert_module, "send_email", lambda to, subject, body: ({"to": to}, "mock"))

    results = send_equity_opportunity_alerts("me@example.com")

    assert results[0]["send_source"] == "mock"
