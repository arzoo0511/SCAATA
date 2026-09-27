"""Tests for the client-facing product API (Phase 19). Uses a real
file-backed sqlite DB per test (not `:memory:`) because FastAPI's `_db`
dependency opens a fresh connection per request -- an in-memory DB would
reset to empty on every single call, making a signup in one request
invisible to the very next request in the same test.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import scaata.product.api as api_module
from scaata.product.api import _db, app
from scaata.product.db import get_connection, init_db, set_subscriber_active


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_product.db"


@pytest.fixture
def client(db_path):
    def _override_db():
        conn = get_connection(db_path)
        init_db(conn)
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[_db] = _override_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def _activate(db_path, subscriber_id):
    conn = get_connection(db_path)
    set_subscriber_active(conn, subscriber_id, True)
    conn.close()


# --- signup ---

def test_signup_creates_inactive_subscriber_with_mock_checkout(client):
    resp = client.post("/subscribers", json={"email": "client@example.com", "tickers": ["MSFT", "AAPL"]})
    assert resp.status_code == 201
    body = resp.json()
    assert body["active"] is False
    assert body["checkout_source"] == "mock"  # no Stripe configured in tests
    assert body["checkout_url"] is None
    assert sorted(body["tickers"]) == ["AAPL", "MSFT"]
    assert len(body["api_key"]) > 20


def test_signup_rejects_unknown_ticker(client):
    resp = client.post("/subscribers", json={"email": "client@example.com", "tickers": ["NOTATICKER"]})
    assert resp.status_code == 422


def test_signup_rejects_empty_ticker_list(client):
    resp = client.post("/subscribers", json={"email": "client@example.com", "tickers": []})
    assert resp.status_code == 422


def test_signup_rejects_duplicate_email(client):
    client.post("/subscribers", json={"email": "client@example.com", "tickers": ["MSFT"]})
    resp = client.post("/subscribers", json={"email": "client@example.com", "tickers": ["AAPL"]})
    assert resp.status_code == 409


def test_signup_rejects_invalid_email(client):
    resp = client.post("/subscribers", json={"email": "not-an-email", "tickers": ["MSFT"]})
    assert resp.status_code == 422


# --- authenticated signal reads ---

def test_latest_signal_requires_api_key_header(client):
    resp = client.get("/signals/MSFT/latest")
    assert resp.status_code == 422  # FastAPI's own required-header validation


def test_latest_signal_rejects_invalid_api_key(client):
    resp = client.get("/signals/MSFT/latest", headers={"X-API-Key": "not-a-real-key"})
    assert resp.status_code == 401


def test_latest_signal_rejects_inactive_subscriber(client):
    signup = client.post("/subscribers", json={"email": "client@example.com", "tickers": ["MSFT"]}).json()
    resp = client.get("/signals/MSFT/latest", headers={"X-API-Key": signup["api_key"]})
    assert resp.status_code == 402


def test_latest_signal_rejects_ticker_not_on_watchlist(client, db_path):
    signup = client.post("/subscribers", json={"email": "client@example.com", "tickers": ["MSFT"]}).json()
    _activate(db_path, get_connection(db_path).execute(
        "SELECT id FROM subscribers WHERE email = 'client@example.com'"
    ).fetchone()[0])

    resp = client.get("/signals/AAPL/latest", headers={"X-API-Key": signup["api_key"]})
    assert resp.status_code == 403


def test_latest_signal_404_when_nothing_computed_yet(client, db_path, monkeypatch):
    signup = client.post("/subscribers", json={"email": "client@example.com", "tickers": ["MSFT"]}).json()
    subscriber_id = get_connection(db_path).execute(
        "SELECT id FROM subscribers WHERE email = 'client@example.com'"
    ).fetchone()[0]
    _activate(db_path, subscriber_id)

    monkeypatch.setattr(api_module, "read_signal_log", lambda ticker: pd.DataFrame())

    resp = client.get("/signals/MSFT/latest", headers={"X-API-Key": signup["api_key"]})
    assert resp.status_code == 404


def test_latest_signal_happy_path_has_no_account_fields(client, db_path, monkeypatch):
    signup = client.post("/subscribers", json={"email": "client@example.com", "tickers": ["MSFT"]}).json()
    subscriber_id = get_connection(db_path).execute(
        "SELECT id FROM subscribers WHERE email = 'client@example.com'"
    ).fetchone()[0]
    _activate(db_path, subscriber_id)

    fake_log = pd.DataFrame([{
        "timestamp_utc": "2026-08-08T12:00:00+00:00", "ticker": "MSFT", "action": "BUY",
        "current_price": 420.5, "position_qty": 5.0, "actionable": True, "suggested_qty": 10,
        "account_cash": 12345.0, "account_equity": 54321.0, "data_source": "live",
    }])
    monkeypatch.setattr(api_module, "read_signal_log", lambda ticker: fake_log)

    resp = client.get("/signals/MSFT/latest", headers={"X-API-Key": signup["api_key"]})
    assert resp.status_code == 200
    body = resp.json()

    assert body["ticker"] == "MSFT"
    assert body["action"] == "BUY"
    assert body["current_price"] == 420.5
    assert "disclaimer" in body and len(body["disclaimer"]) > 0
    # The whole point of the impersonal view -- another subscriber's
    # dashboard must never be able to see this account's real numbers.
    for leaked_field in ("account_cash", "account_equity", "position_qty", "suggested_qty"):
        assert leaked_field not in body


# --- Stripe webhook ---

def test_stripe_webhook_rejects_unverified_request(client):
    resp = client.post("/webhooks/stripe", content=b"{}", headers={"stripe-signature": "bogus"})
    assert resp.status_code == 400


def test_stripe_webhook_activates_subscriber_when_verified(client, db_path, monkeypatch):
    signup = client.post("/subscribers", json={"email": "client@example.com", "tickers": ["MSFT"]}).json()
    subscriber_id = get_connection(db_path).execute(
        "SELECT id FROM subscribers WHERE email = 'client@example.com'"
    ).fetchone()[0]

    fake_event = {
        "type": "checkout.session.completed",
        "data": {"object": {"client_reference_id": str(subscriber_id), "customer": "cus_1", "subscription": "sub_1"}},
    }
    monkeypatch.setattr(api_module, "verify_and_parse_webhook", lambda payload, sig: fake_event)

    resp = client.post("/webhooks/stripe", content=b"irrelevant-once-mocked", headers={"stripe-signature": "sig"})
    assert resp.status_code == 200
    assert resp.json()["handled_as"] == "checkout.session.completed"

    # The subscriber can now actually read a signal.
    activated_check = client.get("/signals/MSFT/latest", headers={"X-API-Key": signup["api_key"]})
    assert activated_check.status_code in (200, 404)  # 402 (inactive) must no longer be the answer
