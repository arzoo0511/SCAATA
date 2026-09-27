"""Zerodha Kite Connect wrapper (Personal app) -- the Indian-market
counterpart of `scaata.live.alpaca_broker`.

Read-only by design for now: profile, holdings, positions and funds. No
order placement until a strategy has shown an edge in the walk-forward
evaluation. Every call returns `(data, source)` where source is "live" or
"unavailable", so a missing login can never be mistaken for an empty
account.

Kite access tokens expire every day at 06:00 IST, so someone has to log in
once a day: `python -m scaata.live.kite_login` opens the Kite login page,
catches the redirect on http://127.0.0.1:5000/, exchanges the
request_token, and saves the session to `.kite_session.json` (gitignored).
The Personal app has no market-data APIs, so prices still come from
yfinance.

Talks to the REST API directly with `requests` rather than adding the
`kiteconnect` package: https://kite.trade/docs/connect/v3/
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from scaata.config import ROOT_DIR

load_dotenv()

API_ROOT = "https://api.kite.trade"
LOGIN_URL = "https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
SESSION_PATH = ROOT_DIR / ".kite_session.json"
IST = timezone(timedelta(hours=5, minutes=30))
TOKEN_RESET_HOUR_IST = 6
TIMEOUT = 15


class KiteError(RuntimeError):
    """The Kite API answered with an error (bad credentials, expired token, ...)."""


def credentials() -> tuple[str | None, str | None]:
    """(api_key, api_secret). Accepts KITE_CONNECT_* (as in this project's
    .env) or the shorter KITE_* names."""
    key = os.environ.get("KITE_CONNECT_API_KEY") or os.environ.get("KITE_API_KEY")
    secret = os.environ.get("KITE_CONNECT_API_SECRET") or os.environ.get("KITE_API_SECRET")
    return (key.strip() if key else None), (secret.strip() if secret else None)


def login_url(api_key: str) -> str:
    return LOGIN_URL.format(api_key=api_key)


def checksum(api_key: str, request_token: str, api_secret: str) -> str:
    return hashlib.sha256((api_key + request_token + api_secret).encode()).hexdigest()


def session_expiry(created_at: datetime) -> datetime:
    """Kite tokens die at the next 06:00 IST after they were issued."""
    ist = created_at.astimezone(IST)
    reset = ist.replace(hour=TOKEN_RESET_HOUR_IST, minute=0, second=0, microsecond=0)
    return reset if ist < reset else reset + timedelta(days=1)


def _unwrap(response):
    try:
        body = response.json()
    except ValueError:
        raise KiteError(f"HTTP {response.status_code}: non-JSON response from Kite")
    if response.status_code != 200 or body.get("status") != "success":
        raise KiteError(f"{body.get('error_type', 'HTTP ' + str(response.status_code))}: {body.get('message', '')}")
    return body["data"]


def create_session(request_token: str, http=requests, path: Path | None = None) -> dict:
    """Exchanges a one-time request_token (from the login redirect) for the
    day's access token and saves it. Returns the user's id and name; the
    token itself is only written to the session file."""
    api_key, api_secret = credentials()
    if not api_key or not api_secret:
        raise KiteError("KITE_CONNECT_API_KEY / KITE_CONNECT_API_SECRET are not set in .env")
    data = _unwrap(http.post(
        f"{API_ROOT}/session/token",
        data={"api_key": api_key, "request_token": request_token,
              "checksum": checksum(api_key, request_token, api_secret)},
        headers={"X-Kite-Version": "3"}, timeout=TIMEOUT,
    ))
    session = {
        "access_token": data["access_token"], "user_id": data.get("user_id"),
        "user_name": data.get("user_name"), "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (path or SESSION_PATH).write_text(json.dumps(session), encoding="utf-8")
    return {"user_id": session["user_id"], "user_name": session["user_name"]}


def load_access_token(path: Path | None = None, now: datetime | None = None) -> str | None:
    """Today's access token, or None if there is no session or it has expired."""
    path = path or SESSION_PATH
    if not path.exists():
        return None
    session = json.loads(path.read_text(encoding="utf-8"))
    created = datetime.fromisoformat(session["created_at_utc"])
    if (now or datetime.now(timezone.utc)) >= session_expiry(created):
        return None
    return session["access_token"]


def _get(endpoint: str, http=requests, path: Path | None = None):
    """(data, "live"), or (None, "unavailable") when not logged in today."""
    api_key, _ = credentials()
    token = load_access_token(path)
    if not api_key or not token:
        return None, "unavailable"
    data = _unwrap(http.get(
        f"{API_ROOT}{endpoint}",
        headers={"X-Kite-Version": "3", "Authorization": f"token {api_key}:{token}"}, timeout=TIMEOUT,
    ))
    return data, "live"


def get_profile(http=requests, path: Path | None = None) -> tuple[dict | None, str]:
    data, source = _get("/user/profile", http, path)
    if data is None:
        return None, source
    return {"user_id": data.get("user_id"), "user_name": data.get("user_name"),
            "exchanges": data.get("exchanges", [])}, source


def get_holdings(http=requests, path: Path | None = None) -> tuple[list[dict], str]:
    """Delivery holdings (settled plus T1). `symbol` is NSE's tradingsymbol,
    e.g. "HDFCBANK" -- the yfinance symbol is that plus ".NS"."""
    data, source = _get("/portfolio/holdings", http, path)
    if data is None:
        return [], source
    return [{
        "symbol": h["tradingsymbol"], "exchange": h.get("exchange"),
        "qty": float(h.get("quantity", 0)) + float(h.get("t1_quantity", 0)),
        "avg_price": float(h.get("average_price", 0.0)), "last_price": float(h.get("last_price", 0.0)),
        "pnl": float(h.get("pnl", 0.0)),
    } for h in data], source


def get_positions(http=requests, path: Path | None = None) -> tuple[list[dict], str]:
    """Net open positions (intraday and carry-forward), zero-quantity rows dropped."""
    data, source = _get("/portfolio/positions", http, path)
    if data is None:
        return [], source
    return [{
        "symbol": p["tradingsymbol"], "exchange": p.get("exchange"), "product": p.get("product"),
        "qty": float(p.get("quantity", 0)), "avg_price": float(p.get("average_price", 0.0)),
        "pnl": float(p.get("pnl", 0.0)),
    } for p in data.get("net", []) if float(p.get("quantity", 0)) != 0], source


def get_funds(http=requests, path: Path | None = None) -> tuple[dict | None, str]:
    """Equity-segment funds: `net` (usable margin) and cash balances."""
    data, source = _get("/user/margins/equity", http, path)
    if data is None:
        return None, source
    available = data.get("available", {})
    return {"net": float(data.get("net", 0.0)), "cash": float(available.get("cash", 0.0)),
            "live_balance": float(available.get("live_balance", 0.0))}, source
