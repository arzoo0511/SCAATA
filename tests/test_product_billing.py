"""Tests for the Stripe billing scaffold (Phase 19). conftest.py's
autouse fixture already clears STRIPE_SECRET_KEY/STRIPE_WEBHOOK_SECRET/
STRIPE_PRICE_ID for every test, so the "unconfigured" path below is what
every test gets by default without any monkeypatching of its own -- the
same mock-first convention as `scaata.live.alpaca_broker` and
`scaata.notify.email_sender`.
"""
import pytest

import scaata.product.billing as billing
from scaata.product.db import create_subscriber, get_connection, init_db


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    init_db(c)
    yield c
    c.close()


def test_create_checkout_session_is_mock_when_unconfigured():
    url, source = billing.create_checkout_session("client@example.com", subscriber_id=1)
    assert url is None
    assert source == "mock"


def test_create_checkout_session_is_mock_when_key_set_but_no_price_id(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fake")
    # STRIPE_PRICE_ID deliberately left unset
    url, source = billing.create_checkout_session("client@example.com", subscriber_id=1)
    assert url is None
    assert source == "mock"


def test_verify_and_parse_webhook_returns_none_when_unconfigured():
    assert billing.verify_and_parse_webhook(b"{}", "t=1,v1=deadbeef") is None


def test_verify_and_parse_webhook_returns_none_without_webhook_secret(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_fake")
    # STRIPE_WEBHOOK_SECRET deliberately left unset
    assert billing.verify_and_parse_webhook(b"{}", "t=1,v1=deadbeef") is None


def test_apply_webhook_event_activates_subscriber_on_checkout_completed(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    event = {
        "type": "checkout.session.completed",
        "data": {"object": {
            "client_reference_id": str(sub["id"]),
            "customer": "cus_123",
            "subscription": "sub_123",
        }},
    }

    handled = billing.apply_webhook_event(conn, event)

    assert handled == "checkout.session.completed"
    from scaata.product.db import get_subscriber_by_id
    updated = get_subscriber_by_id(conn, sub["id"])
    assert updated["active"] is True
    assert updated["stripe_customer_id"] == "cus_123"
    assert updated["stripe_subscription_id"] == "sub_123"


def test_apply_webhook_event_ignores_checkout_without_client_reference_id(conn):
    """A checkout session that isn't ours (no client_reference_id we set)
    must never activate an arbitrary subscriber."""
    event = {"type": "checkout.session.completed", "data": {"object": {"customer": "cus_123"}}}
    assert billing.apply_webhook_event(conn, event) is None


def test_apply_webhook_event_deactivates_on_subscription_deleted(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    from scaata.product.db import set_subscriber_active
    set_subscriber_active(conn, sub["id"], True, stripe_subscription_id="sub_123")

    event = {"type": "customer.subscription.deleted", "data": {"object": {"id": "sub_123"}}}
    handled = billing.apply_webhook_event(conn, event)

    assert handled == "customer.subscription.deleted"
    from scaata.product.db import get_subscriber_by_id
    assert get_subscriber_by_id(conn, sub["id"])["active"] is False


def test_apply_webhook_event_deactivates_on_subscription_updated_to_past_due(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    from scaata.product.db import set_subscriber_active
    set_subscriber_active(conn, sub["id"], True, stripe_subscription_id="sub_123")

    event = {"type": "customer.subscription.updated", "data": {"object": {"id": "sub_123", "status": "past_due"}}}
    billing.apply_webhook_event(conn, event)

    from scaata.product.db import get_subscriber_by_id
    assert get_subscriber_by_id(conn, sub["id"])["active"] is False


def test_apply_webhook_event_ignores_subscription_updated_while_still_active(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    from scaata.product.db import set_subscriber_active
    set_subscriber_active(conn, sub["id"], True, stripe_subscription_id="sub_123")

    event = {"type": "customer.subscription.updated", "data": {"object": {"id": "sub_123", "status": "active"}}}
    handled = billing.apply_webhook_event(conn, event)

    assert handled is None
    from scaata.product.db import get_subscriber_by_id
    assert get_subscriber_by_id(conn, sub["id"])["active"] is True


def test_apply_webhook_event_ignores_unrelated_event_types(conn):
    event = {"type": "invoice.payment_succeeded", "data": {"object": {}}}
    assert billing.apply_webhook_event(conn, event) is None
