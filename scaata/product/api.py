"""Client-facing product API (Phase 19) -- the surface a real website or
mobile client would call. Two deliberately different trust levels:

- `POST /subscribers` is public (anyone can sign up), but a new subscriber
  starts **inactive** (`scaata.product.db.create_subscriber`) and receives
  nothing until a genuine Stripe payment event flips them active via
  `POST /webhooks/stripe` -- a signup alone can never result in free
  delivery.
- `GET /signals/{ticker}/latest` requires a valid, active subscriber
  `X-API-Key` and only ever returns tickers on THAT key's own watchlist.
  There is no endpoint that lists or enumerates other subscribers.

Every response on the signals path carries `scaata.config.SIGNAL_
DISCLAIMER` -- see that constant's docstring for what this product design
does and does not establish legally.

Run with: `uvicorn scaata.product.api:app --port 8000`
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from scaata.config import SUBSCRIBABLE_TICKERS
from scaata.live.daily_signal import impersonal_view, read_signal_log
from scaata.product.billing import apply_webhook_event, create_checkout_session, verify_and_parse_webhook
from scaata.product.db import (
    DuplicateEmailError,
    InvalidEmailError,
    create_subscriber,
    get_connection,
    get_subscriber_by_api_key,
    init_db,
)

app = FastAPI(
    title="SCAATA Signals API",
    version="0.1.0",
    description="Impersonal, per-ticker model signals for subscribed clients. " + "See /subscribers to sign up.",
)


def _db():
    """Per-request connection -- this is a low-traffic signal API, not a
    high-throughput service, so a fresh sqlite3 connection per request is
    simpler than pool management and cheap enough to not matter.
    """
    conn = get_connection()
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


def _authenticated_subscriber(x_api_key: str = Header(...), conn=Depends(_db)) -> dict:
    subscriber = get_subscriber_by_api_key(conn, x_api_key)
    if subscriber is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not subscriber["active"]:
        raise HTTPException(status_code=402, detail="Subscription inactive — complete checkout to activate")
    return subscriber


class SignupRequest(BaseModel):
    email: str
    tickers: list[str]


@app.post("/subscribers", status_code=201)
def signup(request: SignupRequest, conn=Depends(_db)):
    """Creates an inactive subscriber and (if Stripe is configured) a real
    Stripe Checkout session. `checkout_source="mock"` means no live
    Stripe account is wired up yet — see `scaata.product.billing`; the
    subscriber row still exists and the api_key still works, it just
    never activates until something (Stripe webhook, or a manual operator
    call to `set_subscriber_active`) flips it.
    """
    email, tickers = request.email, request.tickers
    if not tickers:
        raise HTTPException(status_code=422, detail="at least one ticker required")
    unknown = set(t.strip().upper() for t in tickers) - set(SUBSCRIBABLE_TICKERS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"not tradeable on this deployment: {sorted(unknown)}")

    try:
        subscriber = create_subscriber(conn, email, tickers)
    except InvalidEmailError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except DuplicateEmailError as e:
        raise HTTPException(status_code=409, detail=str(e))

    checkout_url, checkout_source = create_checkout_session(subscriber["email"], subscriber["id"])
    return {
        "api_key": subscriber["api_key"],
        "tickers": subscriber["tickers"],
        "active": subscriber["active"],
        "checkout_url": checkout_url,
        "checkout_source": checkout_source,
        "note": "Save the api_key now — it is not shown again by this endpoint. "
                "Inactive until checkout completes." if checkout_source == "live"
                else "checkout_source='mock': no live Stripe account configured on this deployment yet.",
    }


@app.get("/signals/{ticker}/latest")
def latest_signal(ticker: str, subscriber: dict = Depends(_authenticated_subscriber)):
    ticker = ticker.strip().upper()
    if ticker not in subscriber["tickers"]:
        raise HTTPException(status_code=403, detail="Ticker is not on your watchlist")

    log = read_signal_log(ticker)
    if log.empty:
        raise HTTPException(status_code=404, detail="No signal computed yet for this ticker")

    return impersonal_view(log.iloc[-1].to_dict())


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request, conn=Depends(_db)):
    """Stripe calls this directly (not a browser), so authentication is
    the signature check inside `verify_and_parse_webhook`, not an API
    key. A request that doesn't verify -- wrong secret, no secret
    configured, or an outright forgery -- is rejected with 400 and never
    reaches `apply_webhook_event`; see that function's docstring for why
    this must fail closed.
    """
    payload = await request.body()
    event = verify_and_parse_webhook(payload, request.headers.get("stripe-signature", ""))
    if event is None:
        raise HTTPException(status_code=400, detail="Invalid or unconfigured Stripe webhook")

    handled = apply_webhook_event(conn, event)
    return {"received": True, "handled_as": handled}
