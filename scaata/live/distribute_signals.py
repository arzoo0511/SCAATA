"""Daily fan-out of today's already-computed signals to every active,
paying subscriber (Phase 19) -- the multi-tenant replacement for the
single hardcoded-recipient convention every other notify module in this
codebase still uses (`scaata.notify.email_sender`'s `to` argument used to
always be the project owner's own address).

Reads what `scaata.live.run_all_daily_signals` already wrote to each
ticker's `signal_log_{TICKER}.csv` today -- does **not** recompute
anything or touch a policy's recurrent LSTM state a second time (see
`scaata.live.daily_signal.impersonal_view`). Intended to run once a day,
right after `run_all_daily_signals`, from the same Windows Task Scheduler
job or a new one immediately after it.

Idempotency: `scaata.product.db`'s `deliveries` table is checked before
sending and only updated after a send actually succeeds, so re-running
this job the same day (e.g. after the machine slept mid-run, see the
[[scaata-machine-constraints]] history) never double-sends a ticker that
already went out, and never silently drops one that failed to send.
"""
from __future__ import annotations

from datetime import datetime, timezone

from scaata.config import SUBSCRIBABLE_TICKERS
from scaata.live.daily_signal import impersonal_view, read_signal_log
from scaata.notify.email_sender import send_email
from scaata.product.db import get_connection, has_been_delivered, init_db, list_active_subscribers, record_delivery


def _format_subscriber_email(rows: list[dict]) -> tuple[str, str]:
    lines = [f"{r['ticker']}: {r['action']} at ${r['current_price']:.2f} (as of {r['as_of_utc']})" for r in rows]
    body = "\n".join(lines) + "\n\n" + rows[0]["disclaimer"]
    subject = f"SCAATA daily signals — {datetime.now(timezone.utc):%Y-%m-%d}"
    return subject, body


def distribute_todays_signals(db_path=None) -> list[dict]:
    """Sends one email per active subscriber containing today's signal for
    every ticker on their watchlist that has one logged today, skipping
    tickers already delivered today and skipping a subscriber entirely if
    none of their watched tickers has anything new to send (no empty
    emails). Returns a list of `{subscriber_id, email, tickers, source}`
    for whatever was actually attempted, so a caller (a test, or a future
    admin dashboard) gets a record without re-querying the DB itself.
    """
    conn = get_connection(db_path)
    init_db(conn)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Cache each ticker's latest log row across the whole run: with N
    # subscribers all plausibly watching the same handful of mega-caps,
    # reading signal_log_{ticker}.csv once per ticker (not once per
    # subscriber-ticker pair) turns an O(subscribers x tickers) file-read
    # count into O(tickers).
    latest_by_ticker: dict[str, dict | None] = {}

    def _latest(ticker: str) -> dict | None:
        if ticker not in latest_by_ticker:
            log = read_signal_log(ticker)
            latest_by_ticker[ticker] = None if log.empty else log.iloc[-1].to_dict()
        return latest_by_ticker[ticker]

    results = []
    for subscriber in list_active_subscribers(conn):
        rows, pending_tickers = [], []
        for ticker in subscriber["tickers"]:
            if ticker not in SUBSCRIBABLE_TICKERS:
                continue  # a subscriber can only watch a ticker this deployment actually trades
            latest = _latest(ticker)
            if latest is None:
                continue  # nothing computed for this ticker at all yet
            if latest["timestamp_utc"][:10] != today:
                continue  # today's job hasn't produced a fresh row for this ticker yet -- don't resend a stale prior day
            if has_been_delivered(conn, subscriber["id"], ticker, today):
                continue  # already sent today -- a rerun of this job is a no-op for this pair
            rows.append(impersonal_view(latest))
            pending_tickers.append(ticker)

        if not rows:
            continue

        subject, body = _format_subscriber_email(rows)
        try:
            _, source = send_email(subscriber["email"], subject, body)
        except Exception:
            # Leave undelivered so the NEXT run retries this subscriber --
            # marking it sent here (before a confirmed send) would lose a
            # paying subscriber's signal permanently on a transient SMTP
            # failure, which is the wrong failure mode for a paid product.
            continue

        for ticker in pending_tickers:
            record_delivery(conn, subscriber["id"], ticker, today)

        results.append({
            "subscriber_id": subscriber["id"], "email": subscriber["email"],
            "tickers": pending_tickers, "source": source,
        })

    return results


if __name__ == "__main__":
    sent = distribute_todays_signals()
    print(f"Distributed to {len(sent)} subscriber(s).")
    for row in sent:
        print(f"  {row['email']}: {', '.join(row['tickers'])} ({row['source']})")
