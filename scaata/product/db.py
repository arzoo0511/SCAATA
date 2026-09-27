"""Subscriber store for the client-facing signal product (Phase 19) --
plain stdlib `sqlite3`, no new dependency, replacing the "one hardcoded
email address" convention every other notify module in this codebase
still uses. Three tables:

- `subscribers`: one row per paying (or trying-to-pay) client. Starts
  `active=0` on signup -- nothing is ever sent until a real Stripe payment
  event flips it (see `scaata.product.billing`), so a signup alone can
  never result in a subscriber receiving signals for free.
- `watchlist`: which tickers each subscriber gets. A subscriber only ever
  receives tickers on their own watchlist, and only tickers this
  deployment actually trades (`scaata.config.SUBSCRIBABLE_TICKERS`).
- `deliveries`: one row per (subscriber, ticker, day) that's already been
  sent -- the idempotency guard `scaata.live.distribute_signals` checks
  before emailing, so re-running the daily job (e.g. after a crash) never
  double-sends the same day's signal to the same subscriber.

Every write commits immediately -- this runs from a single daily batch
job and a low-traffic API, not a high-concurrency service, so the
simplicity of "no held-open transaction" outweighs the throughput cost.
"""
from __future__ import annotations

import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from scaata.config import PRODUCT_DB_PATH

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class InvalidEmailError(ValueError):
    pass


class DuplicateEmailError(ValueError):
    pass


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """One connection per caller (test fixtures pass `:memory:` or a
    tmp_path file; the API/distribution job use the real product.db).
    `row_factory` makes every query return dict-like `sqlite3.Row`s
    instead of positional tuples, so column-order changes here can't
    silently scramble a caller reading by index.

    `check_same_thread=False`: FastAPI resolves a sync dependency (see
    `scaata.product.api._db`) in a worker thread that can differ from the
    thread the `async def` endpoint body itself runs on -- caught live,
    the Stripe webhook route hit `sqlite3.ProgrammingError: SQLite
    objects created in a thread can only be used in that same thread`
    with the default. Safe here because each request/job run gets its
    own fresh connection via `_db`/`distribute_todays_signals` (never
    shared or reused concurrently across requests) -- this opts out of
    sqlite3's same-thread check, not out of any actual isolation.
    """
    conn = sqlite3.connect(str(db_path or PRODUCT_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Idempotent -- safe to call on every connection open (the API layer
    does exactly that per-request via a dependency), not just once at
    deploy time.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS subscribers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            api_key TEXT NOT NULL UNIQUE,
            active INTEGER NOT NULL DEFAULT 0,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            created_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            subscriber_id INTEGER NOT NULL REFERENCES subscribers(id),
            ticker TEXT NOT NULL,
            PRIMARY KEY (subscriber_id, ticker)
        );

        CREATE TABLE IF NOT EXISTS deliveries (
            subscriber_id INTEGER NOT NULL REFERENCES subscribers(id),
            ticker TEXT NOT NULL,
            delivery_date TEXT NOT NULL,
            sent_at_utc TEXT NOT NULL,
            PRIMARY KEY (subscriber_id, ticker, delivery_date)
        );
        """
    )
    conn.commit()


def _row_to_subscriber(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    tickers = [
        r["ticker"] for r in conn.execute(
            "SELECT ticker FROM watchlist WHERE subscriber_id = ? ORDER BY ticker", (row["id"],)
        )
    ]
    return {
        "id": row["id"],
        "email": row["email"],
        "api_key": row["api_key"],
        "active": bool(row["active"]),
        "stripe_customer_id": row["stripe_customer_id"],
        "stripe_subscription_id": row["stripe_subscription_id"],
        "created_at_utc": row["created_at_utc"],
        "tickers": tickers,
    }


def create_subscriber(conn: sqlite3.Connection, email: str, tickers: list[str]) -> dict:
    """Inserts a new, inactive subscriber with the given watchlist.
    Raises `InvalidEmailError`/`DuplicateEmailError` rather than letting a
    malformed or repeat signup hit sqlite's raw `IntegrityError` -- the
    API layer turns these into a 400/409, not a 500.

    Generates a fresh `api_key` via `secrets.token_urlsafe` (cryptographic
    randomness, not `random` -- this key is the only thing that gates
    reading a subscriber's own signals, so it needs to be unguessable, the
    same bar Alpaca/Stripe/every real API holds its own keys to).
    """
    email = email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise InvalidEmailError(f"not a valid email address: {email!r}")
    if conn.execute("SELECT 1 FROM subscribers WHERE email = ?", (email,)).fetchone():
        raise DuplicateEmailError(f"already a subscriber: {email!r}")

    api_key = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc).isoformat()
    try:
        cur = conn.execute(
            "INSERT INTO subscribers (email, api_key, active, created_at_utc) VALUES (?, ?, 0, ?)",
            (email, api_key, now),
        )
    except sqlite3.IntegrityError:
        # Defense-in-depth against the check-then-insert race above: two
        # signups for the same email arriving close enough together could
        # both pass the SELECT above before either INSERTs. The UNIQUE
        # constraint on `email` is the real guard; this just turns its
        # raw IntegrityError into the same typed error the pre-check
        # already gives callers, so the API layer's except block (-> 409)
        # doesn't need a second, different case to handle.
        raise DuplicateEmailError(f"already a subscriber: {email!r}")
    subscriber_id = cur.lastrowid
    for ticker in dict.fromkeys(t.strip().upper() for t in tickers):  # de-dup, preserve order
        conn.execute("INSERT OR IGNORE INTO watchlist (subscriber_id, ticker) VALUES (?, ?)", (subscriber_id, ticker))
    conn.commit()

    return get_subscriber_by_id(conn, subscriber_id)


def get_subscriber_by_id(conn: sqlite3.Connection, subscriber_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM subscribers WHERE id = ?", (subscriber_id,)).fetchone()
    return None if row is None else _row_to_subscriber(conn, row)


def get_subscriber_by_api_key(conn: sqlite3.Connection, api_key: str) -> dict | None:
    row = conn.execute("SELECT * FROM subscribers WHERE api_key = ?", (api_key,)).fetchone()
    return None if row is None else _row_to_subscriber(conn, row)


def get_subscriber_by_email(conn: sqlite3.Connection, email: str) -> dict | None:
    row = conn.execute("SELECT * FROM subscribers WHERE email = ?", (email.strip().lower(),)).fetchone()
    return None if row is None else _row_to_subscriber(conn, row)


def set_subscriber_active(
    conn: sqlite3.Connection, subscriber_id: int, active: bool,
    stripe_customer_id: str | None = None, stripe_subscription_id: str | None = None,
) -> None:
    """Flips billing status. Only `scaata.product.billing`'s Stripe-
    webhook handling and (in a genuine support/manual-comp case) an
    operator should ever call this with `active=True` -- there is no
    public API path that does.
    """
    conn.execute(
        "UPDATE subscribers SET active = ?, stripe_customer_id = COALESCE(?, stripe_customer_id), "
        "stripe_subscription_id = COALESCE(?, stripe_subscription_id) WHERE id = ?",
        (1 if active else 0, stripe_customer_id, stripe_subscription_id, subscriber_id),
    )
    conn.commit()


def deactivate_by_stripe_subscription_id(conn: sqlite3.Connection, stripe_subscription_id: str) -> None:
    """Used when a Stripe subscription is cancelled/lapses -- looked up by
    Stripe's own subscription id rather than our subscriber id, since
    that's the identifier the webhook event actually carries.
    """
    conn.execute(
        "UPDATE subscribers SET active = 0 WHERE stripe_subscription_id = ?", (stripe_subscription_id,)
    )
    conn.commit()


def list_active_subscribers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM subscribers WHERE active = 1 ORDER BY id").fetchall()
    return [_row_to_subscriber(conn, row) for row in rows]


def has_been_delivered(conn: sqlite3.Connection, subscriber_id: int, ticker: str, delivery_date: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM deliveries WHERE subscriber_id = ? AND ticker = ? AND delivery_date = ?",
        (subscriber_id, ticker, delivery_date),
    ).fetchone()
    return row is not None


def record_delivery(conn: sqlite3.Connection, subscriber_id: int, ticker: str, delivery_date: str) -> None:
    """`INSERT OR IGNORE` so calling this twice for the same (subscriber,
    ticker, day) -- e.g. a retried job -- is a harmless no-op rather than
    a primary-key crash.
    """
    conn.execute(
        "INSERT OR IGNORE INTO deliveries (subscriber_id, ticker, delivery_date, sent_at_utc) VALUES (?, ?, ?, ?)",
        (subscriber_id, ticker, delivery_date, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
