"""Tests for the subscriber store (Phase 19) -- the multi-tenant
replacement for the single-hardcoded-recipient convention elsewhere in
this codebase. Uses an in-memory sqlite DB per test (`:memory:`) so tests
never touch the real `product.db` and never see each other's state.
"""
import pytest

from scaata.product.db import (
    DuplicateEmailError,
    InvalidEmailError,
    create_subscriber,
    deactivate_by_stripe_subscription_id,
    get_connection,
    get_subscriber_by_api_key,
    get_subscriber_by_email,
    get_subscriber_by_id,
    has_been_delivered,
    init_db,
    list_active_subscribers,
    record_delivery,
    set_subscriber_active,
)


@pytest.fixture
def conn():
    c = get_connection(":memory:")
    init_db(c)
    yield c
    c.close()


def test_create_subscriber_starts_inactive(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT", "AAPL"])
    assert sub["active"] is False
    assert sub["tickers"] == ["AAPL", "MSFT"]  # stored/returned sorted, not insertion order
    assert sub["email"] == "client@example.com"
    assert len(sub["api_key"]) > 20


def test_api_key_is_unique_per_subscriber(conn):
    a = create_subscriber(conn, "a@example.com", ["MSFT"])
    b = create_subscriber(conn, "b@example.com", ["MSFT"])
    assert a["api_key"] != b["api_key"]


def test_email_is_normalized_to_lowercase_and_stripped(conn):
    sub = create_subscriber(conn, "  Client@Example.COM  ", ["MSFT"])
    assert sub["email"] == "client@example.com"


def test_duplicate_email_is_rejected(conn):
    create_subscriber(conn, "dup@example.com", ["MSFT"])
    with pytest.raises(DuplicateEmailError):
        create_subscriber(conn, "dup@example.com", ["AAPL"])


def test_duplicate_email_case_insensitive(conn):
    create_subscriber(conn, "dup@example.com", ["MSFT"])
    with pytest.raises(DuplicateEmailError):
        create_subscriber(conn, "DUP@EXAMPLE.COM", ["AAPL"])


def test_invalid_email_is_rejected(conn):
    with pytest.raises(InvalidEmailError):
        create_subscriber(conn, "not-an-email", ["MSFT"])


def test_ticker_watchlist_is_deduplicated(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT", "msft", "MSFT"])
    assert sub["tickers"] == ["MSFT"]


def test_get_subscriber_by_api_key_returns_none_for_unknown_key(conn):
    assert get_subscriber_by_api_key(conn, "not-a-real-key") is None


def test_get_subscriber_by_api_key_finds_the_right_one(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    found = get_subscriber_by_api_key(conn, sub["api_key"])
    assert found["id"] == sub["id"]


def test_get_subscriber_by_email(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    assert get_subscriber_by_email(conn, "CLIENT@example.com")["id"] == sub["id"]
    assert get_subscriber_by_email(conn, "nobody@example.com") is None


def test_get_subscriber_by_id_unknown_returns_none(conn):
    assert get_subscriber_by_id(conn, 999) is None


def test_new_subscriber_excluded_from_active_list(conn):
    create_subscriber(conn, "client@example.com", ["MSFT"])
    assert list_active_subscribers(conn) == []


def test_set_subscriber_active_true_includes_in_active_list(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    set_subscriber_active(conn, sub["id"], True, stripe_customer_id="cus_1", stripe_subscription_id="sub_1")

    active = list_active_subscribers(conn)
    assert len(active) == 1
    assert active[0]["id"] == sub["id"]
    assert active[0]["stripe_customer_id"] == "cus_1"
    assert active[0]["stripe_subscription_id"] == "sub_1"


def test_set_subscriber_active_false_removes_from_active_list(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    set_subscriber_active(conn, sub["id"], True)
    set_subscriber_active(conn, sub["id"], False)
    assert list_active_subscribers(conn) == []


def test_set_subscriber_active_preserves_stripe_ids_when_not_passed(conn):
    """A later call that only flips `active` (no stripe_customer_id passed)
    must not null out ids recorded by an earlier call -- COALESCE, not a
    blind overwrite."""
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    set_subscriber_active(conn, sub["id"], True, stripe_customer_id="cus_1", stripe_subscription_id="sub_1")
    set_subscriber_active(conn, sub["id"], False)

    row = get_subscriber_by_id(conn, sub["id"])
    assert row["stripe_customer_id"] == "cus_1"
    assert row["stripe_subscription_id"] == "sub_1"


def test_deactivate_by_stripe_subscription_id(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    set_subscriber_active(conn, sub["id"], True, stripe_subscription_id="sub_123")

    deactivate_by_stripe_subscription_id(conn, "sub_123")

    assert get_subscriber_by_id(conn, sub["id"])["active"] is False


def test_deactivate_by_unknown_stripe_subscription_id_is_a_no_op(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    set_subscriber_active(conn, sub["id"], True, stripe_subscription_id="sub_123")

    deactivate_by_stripe_subscription_id(conn, "sub_does_not_exist")

    assert get_subscriber_by_id(conn, sub["id"])["active"] is True


def test_delivery_not_recorded_until_record_delivery_is_called(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    assert has_been_delivered(conn, sub["id"], "MSFT", "2026-08-08") is False


def test_record_delivery_marks_it_delivered(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    record_delivery(conn, sub["id"], "MSFT", "2026-08-08")
    assert has_been_delivered(conn, sub["id"], "MSFT", "2026-08-08") is True


def test_record_delivery_is_idempotent(conn):
    """Regression guard: re-running the daily distribution job (e.g. after
    a crash mid-run, see the machine-sleeps-mid-job history) must not
    crash on a primary-key collision when the same (subscriber, ticker,
    day) is recorded twice."""
    sub = create_subscriber(conn, "client@example.com", ["MSFT"])
    record_delivery(conn, sub["id"], "MSFT", "2026-08-08")
    record_delivery(conn, sub["id"], "MSFT", "2026-08-08")  # must not raise
    assert has_been_delivered(conn, sub["id"], "MSFT", "2026-08-08") is True


def test_delivery_is_scoped_to_ticker_and_day(conn):
    sub = create_subscriber(conn, "client@example.com", ["MSFT", "AAPL"])
    record_delivery(conn, sub["id"], "MSFT", "2026-08-08")

    assert has_been_delivered(conn, sub["id"], "AAPL", "2026-08-08") is False  # different ticker
    assert has_been_delivered(conn, sub["id"], "MSFT", "2026-08-09") is False  # different day
