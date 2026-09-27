"""Tests for the daily subscriber fan-out job (Phase 19) -- the piece that
turns the single-hardcoded-recipient notify convention into something a
real subscriber list can use. Monkeypatches `read_signal_log`/
`send_email` as bound inside `scaata.live.distribute_signals` (that
module imported them by name, so patching the origin module wouldn't
affect the reference this module already holds).
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

import scaata.live.distribute_signals as distribute_module
from scaata.product.db import create_subscriber, get_connection, has_been_delivered, init_db, set_subscriber_active

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
NOW_ISO = datetime.now(timezone.utc).isoformat()


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test_product.db"
    conn = get_connection(path)
    init_db(conn)
    conn.close()
    return path


def _active_subscriber(db_path, email, tickers):
    conn = get_connection(db_path)
    sub = create_subscriber(conn, email, tickers)
    set_subscriber_active(conn, sub["id"], True)
    conn.close()
    return sub


def _fresh_log(ticker="MSFT", action="BUY", price=420.5, timestamp=NOW_ISO):
    return pd.DataFrame([{
        "timestamp_utc": timestamp, "ticker": ticker, "action": action, "current_price": price,
        "position_qty": 5.0, "actionable": True, "suggested_qty": 10,
        "account_cash": 12345.0, "account_equity": 54321.0, "data_source": "live",
    }])


def _patch_logs(monkeypatch, logs: dict):
    monkeypatch.setattr(distribute_module, "read_signal_log", lambda ticker: logs.get(ticker, pd.DataFrame()))


def _patch_send_email(monkeypatch, calls: list, raise_for: set = frozenset()):
    def _fake_send(to, subject, body):
        if to in raise_for:
            raise RuntimeError("simulated SMTP failure")
        calls.append({"to": to, "subject": subject, "body": body})
        return {"to": to, "subject": subject}, "mock"
    monkeypatch.setattr(distribute_module, "send_email", _fake_send)


def test_sends_to_active_subscriber_with_fresh_signal(db_path, monkeypatch):
    sub = _active_subscriber(db_path, "client@example.com", ["MSFT"])
    _patch_logs(monkeypatch, {"MSFT": _fresh_log()})
    calls = []
    _patch_send_email(monkeypatch, calls)

    results = distribute_module.distribute_todays_signals(db_path)

    assert len(results) == 1
    assert results[0]["email"] == "client@example.com"
    assert results[0]["tickers"] == ["MSFT"]
    assert len(calls) == 1
    assert "MSFT" in calls[0]["body"] and "BUY" in calls[0]["body"]


def test_email_body_never_contains_account_fields(db_path, monkeypatch):
    _active_subscriber(db_path, "client@example.com", ["MSFT"])
    _patch_logs(monkeypatch, {"MSFT": _fresh_log()})
    calls = []
    _patch_send_email(monkeypatch, calls)

    distribute_module.distribute_todays_signals(db_path)

    assert "12345" not in calls[0]["body"]  # account_cash must never appear in a subscriber email
    assert "54321" not in calls[0]["body"]  # account_equity


def test_skips_inactive_subscriber(db_path, monkeypatch):
    conn = get_connection(db_path)
    create_subscriber(conn, "client@example.com", ["MSFT"])  # never activated
    conn.close()
    _patch_logs(monkeypatch, {"MSFT": _fresh_log()})
    calls = []
    _patch_send_email(monkeypatch, calls)

    results = distribute_module.distribute_todays_signals(db_path)

    assert results == []
    assert calls == []


def test_skips_subscriber_when_ticker_has_no_signal_at_all(db_path, monkeypatch):
    _active_subscriber(db_path, "client@example.com", ["MSFT"])
    _patch_logs(monkeypatch, {})  # MSFT has no log rows at all
    calls = []
    _patch_send_email(monkeypatch, calls)

    results = distribute_module.distribute_todays_signals(db_path)

    assert results == []
    assert calls == []


def test_skips_ticker_with_only_a_stale_prior_day_signal(db_path, monkeypatch):
    _active_subscriber(db_path, "client@example.com", ["MSFT"])
    stale = _fresh_log(timestamp="2020-01-01T12:00:00+00:00")
    _patch_logs(monkeypatch, {"MSFT": stale})
    calls = []
    _patch_send_email(monkeypatch, calls)

    results = distribute_module.distribute_todays_signals(db_path)

    assert results == []
    assert calls == []


def test_skips_ticker_not_on_this_deployments_subscribable_list(db_path, monkeypatch):
    """A subscriber row can technically hold any ticker (the db layer is
    generic; the API layer is what enforces SUBSCRIBABLE_TICKERS) -- the
    distribution job must defend against sending on a ticker this
    deployment doesn't actually trade regardless of how it got there."""
    _active_subscriber(db_path, "client@example.com", ["NOTATICKER"])
    calls = []
    _patch_send_email(monkeypatch, calls)

    results = distribute_module.distribute_todays_signals(db_path)

    assert results == []
    assert calls == []


def test_second_run_same_day_does_not_resend(db_path, monkeypatch):
    _active_subscriber(db_path, "client@example.com", ["MSFT"])
    _patch_logs(monkeypatch, {"MSFT": _fresh_log()})
    calls = []
    _patch_send_email(monkeypatch, calls)

    first = distribute_module.distribute_todays_signals(db_path)
    second = distribute_module.distribute_todays_signals(db_path)

    assert len(first) == 1
    assert second == []  # nothing left to send -- already delivered today
    assert len(calls) == 1  # only one real send happened


def test_only_watched_tickers_are_sent_not_the_whole_universe(db_path, monkeypatch):
    _active_subscriber(db_path, "client@example.com", ["MSFT"])  # not watching AAPL
    _patch_logs(monkeypatch, {"MSFT": _fresh_log("MSFT"), "AAPL": _fresh_log("AAPL")})
    calls = []
    _patch_send_email(monkeypatch, calls)

    results = distribute_module.distribute_todays_signals(db_path)

    assert results[0]["tickers"] == ["MSFT"]
    assert "AAPL" not in calls[0]["body"]


def test_failed_send_leaves_delivery_unrecorded_for_retry(db_path, monkeypatch):
    sub = _active_subscriber(db_path, "client@example.com", ["MSFT"])
    _patch_logs(monkeypatch, {"MSFT": _fresh_log()})
    calls = []
    _patch_send_email(monkeypatch, calls, raise_for={"client@example.com"})

    results = distribute_module.distribute_todays_signals(db_path)

    assert results == []  # the attempted send failed, so nothing to report as sent
    conn = get_connection(db_path)
    assert has_been_delivered(conn, sub["id"], "MSFT", TODAY) is False
    conn.close()


def test_retry_succeeds_after_a_prior_failed_send(db_path, monkeypatch):
    sub = _active_subscriber(db_path, "client@example.com", ["MSFT"])
    _patch_logs(monkeypatch, {"MSFT": _fresh_log()})

    failing_calls = []
    _patch_send_email(monkeypatch, failing_calls, raise_for={"client@example.com"})
    distribute_module.distribute_todays_signals(db_path)  # fails, leaves undelivered

    working_calls = []
    _patch_send_email(monkeypatch, working_calls)  # simulate SMTP recovering
    results = distribute_module.distribute_todays_signals(db_path)

    assert len(results) == 1
    assert len(working_calls) == 1
