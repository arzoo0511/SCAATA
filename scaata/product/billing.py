"""Stripe billing (Phase 19) -- same mock-fallback convention as
`scaata.live.alpaca_broker` and `scaata.notify.email_sender`: every
function here returns a `source`/degrades to a safe no-op when Stripe
isn't configured, so this module is safe to import and call with zero
Stripe account anywhere in the loop (dev machine, CI, tests).

This is scaffolding, not a finished billing integration. Creating a real
Stripe account, a subscription Product/Price, and populating
`STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` / `STRIPE_PRICE_ID` in the
environment is something only the account owner can do -- Claude does not
create financial accounts or enter payment credentials on a user's
behalf. See `.env.example` for the exact variables this reads.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

try:
    import stripe as _stripe_sdk
except ImportError:  # pragma: no cover - exercised by test_billing's mock-mode tests
    _stripe_sdk = None


def _configured() -> bool:
    return _stripe_sdk is not None and bool(os.environ.get("STRIPE_SECRET_KEY"))


def create_checkout_session(email: str, subscriber_id: int) -> tuple[str | None, str]:
    """Returns `(checkout_url, source)`. `source="mock"` (`checkout_url=
    None`) whenever Stripe isn't fully configured (no `stripe` package,
    no `STRIPE_SECRET_KEY`, or no `STRIPE_PRICE_ID`) -- the caller
    (`scaata.product.api`) still returns a usable signup response in that
    case, just without a real payment link, so local development and
    tests never need a live Stripe account.

    `client_reference_id` carries our own subscriber id through the
    checkout flow -- Stripe echoes it back on the `checkout.session.
    completed` webhook event, which is how `handle_checkout_completed`
    below knows which subscriber to activate without guessing from email
    (a subscriber could theoretically pay with a different email than
    they signed up with).
    """
    if not _configured():
        return None, "mock"

    price_id = os.environ.get("STRIPE_PRICE_ID")
    if not price_id:
        return None, "mock"

    session = _stripe_sdk.checkout.Session.create(
        mode="subscription",
        payment_method_types=["card"],
        customer_email=email,
        line_items=[{"price": price_id, "quantity": 1}],
        client_reference_id=str(subscriber_id),
        success_url=os.environ.get("STRIPE_SUCCESS_URL", "https://example.com/signup/success"),
        cancel_url=os.environ.get("STRIPE_CANCEL_URL", "https://example.com/signup/cancelled"),
    )
    return session.url, "live"


def verify_and_parse_webhook(payload: bytes, signature_header: str):
    """Verifies the Stripe webhook signature and returns the parsed event,
    or `None` on anything that isn't a genuine, verified Stripe event:
    Stripe not configured, no `STRIPE_WEBHOOK_SECRET`, or a signature that
    doesn't verify. A forged POST to `/webhooks/stripe` claiming
    "payment completed" must never be able to activate a free subscriber
    -- returning `None` here and letting the caller respond 400 is the
    entire defense against that, so this must fail closed, not open, on
    every unconfigured/unverifiable case.
    """
    if not _configured():
        return None
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        return None
    try:
        return _stripe_sdk.Webhook.construct_event(payload, signature_header, webhook_secret)
    except (ValueError, _stripe_sdk.error.SignatureVerificationError):
        return None


def apply_webhook_event(conn, event: dict) -> str | None:
    """Applies a verified Stripe event to the subscriber store. Returns
    the event type handled, or `None` for an event type this product
    doesn't act on (Stripe sends many event types; silently ignoring the
    ones we don't care about is correct, not a gap).

    Imports the db functions locally (not at module top) to keep this
    module importable without circularity concerns if `scaata.product.db`
    ever needs billing state in the future.
    """
    from scaata.product.db import deactivate_by_stripe_subscription_id, set_subscriber_active

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        subscriber_id = data.get("client_reference_id")
        if subscriber_id is None:
            return None  # not one of our checkout sessions -- nothing to activate
        set_subscriber_active(
            conn, int(subscriber_id), True,
            stripe_customer_id=data.get("customer"),
            stripe_subscription_id=data.get("subscription"),
        )
        return event_type

    if event_type == "customer.subscription.deleted":
        deactivate_by_stripe_subscription_id(conn, data["id"])
        return event_type

    if event_type == "customer.subscription.updated" and data.get("status") != "active":
        # Covers past_due/unpaid/canceled transitions that arrive as an
        # "updated" event rather than "deleted" -- a lapsed card should
        # stop delivery just as surely as an explicit cancellation.
        deactivate_by_stripe_subscription_id(conn, data["id"])
        return event_type

    return None
