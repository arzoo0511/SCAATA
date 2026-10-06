"""HANSEI's daily run -- the whole agent, once per trading day, no dashboard
needed.

    python -m scaata.agent.daily

After each NSE session it:

1. credits liquid-fund interest on idle cash, then fills yesterday's
   decisions at today's open (never at the close that
   produced them), and starts the buy-and-hold comparison on day one;
2. marks the book to today's close;
3. learns: views it formed 20 sessions ago are now judged against what the
   stocks actually did, and its trust in each advisor shifts accordingly;
4. reads today's news for each stock and scores it;
5. asks every advisor for a view, lets the brain decide, and queues any
   trades for tomorrow's open -- only where it has a real reason to act;
6. writes a journal entry saying what it did and why.

A day that has already been processed is skipped, so running it twice, or
catching up after the PC was asleep, never double-trades. Fake money, real
prices, real NSE costs; no broker is ever contacted.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scaata.agent import advisors, brain, memory as mem
from scaata.agent.news import gather, news_signal, stories
from scaata.config import INDIA_TICKERS, ROOT_DIR
from scaata.live.paper_book import (
    BOOK_PATH, accrue_cash_yield, execute_pending, load_book, mark_to_market, new_book, queue_target, save_book,
    start_benchmark, summary,
)

JOURNAL_PATH = ROOT_DIR / "forward_test" / "hansei_journal.json"
HISTORY_START = "2019-01-01"   # enough for the 200-day and 252-day windows
IST = timezone(timedelta(hours=5, minutes=30))
SETTLED = (15, 45)             # NSE closes 15:30 IST; after this, today's bar is final
NEXT_OPEN = (9, 15)            # NSE opens 09:15 IST
WATCHDOG_SECONDS = 20 * 60     # a stalled download must not hang the scheduled run forever
STALE_WEEKDAYS = 4             # more weekdays than this unprocessed means HANSEI is stuck


def load_journal(path: Path | None = None) -> list[dict]:
    path = path or JOURNAL_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _save_journal(entries: list[dict], path: Path | None = None) -> None:
    path = path or JOURNAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries[-500:], indent=2), encoding="utf-8")


def _fetch_prices(today: date) -> pd.DataFrame:
    from scaata.data.loaders import load_market_data

    # yfinance's `end` is exclusive: ask through tomorrow so today's bar is included.
    return load_market_data(list(INDIA_TICKERS), HISTORY_START, (today + timedelta(days=1)).isoformat(),
                            use_cache=False)


def _sessions_since_trade(book: dict, symbol: str, sessions: pd.DatetimeIndex) -> int | None:
    dates = [t["date"] for t in book["trades"] if t["symbol"] == symbol]
    if not dates:
        return None
    return int((sessions > pd.Timestamp(max(dates))).sum())


def _drop_unsettled_bar(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Before the close, today's "close" is just the latest trade. Decide on
    finished sessions only."""
    local = now.astimezone(IST)
    if (local.hour, local.minute) < SETTLED:
        frame = frame.loc[frame.index.date < local.date()]
    return frame


def _next_open(session_day: date) -> datetime:
    """When the orders decided on `session_day` fill: the next weekday's open."""
    day = session_day + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return datetime(day.year, day.month, day.day, *NEXT_OPEN, tzinfo=IST)


def _news_known_by(news: dict, symbol: str, cutoff: datetime, now: datetime) -> dict:
    """A run that lands after the next open must not use news from after it:
    those orders fill at that open. Keep only what was known then, aged to then."""
    if now <= cutoff:
        return news
    kept = [h for h in news["headlines"] if datetime.fromisoformat(h["published_utc"]) <= cutoff]
    return {**news, "headlines": kept, "signal": news_signal(kept, now=cutoff, symbol=symbol)}


def run_day(today: date | None = None, prices: pd.DataFrame | None = None, news_fn=gather,
            book_path: Path | None = None, memory_path: Path | None = None,
            journal_path: Path | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    today = today or now.astimezone(IST).date()
    raw = prices if prices is not None else _fetch_prices(today)
    raw = raw[raw["Volume"] > 0]   # Yahoo sometimes adds a flat, zero-volume bar on an NSE holiday
    closes = raw.reset_index().pivot_table(index="Date", columns="Ticker", values="Close").dropna(how="all")
    opens = raw.reset_index().pivot_table(index="Date", columns="Ticker", values="Open").dropna(how="all")
    closes, opens = _drop_unsettled_bar(closes, now), _drop_unsettled_bar(opens, now)
    session = closes.index.max()
    session_day = session.date()

    book = load_book(book_path) or new_book(list(INDIA_TICKERS))
    # Never step backwards: after midnight IST Yahoo can briefly drop the last
    # finished bar, which would replay an older session over a newer book.
    if book.get("last_session") and book["last_session"] >= session_day.isoformat():
        return {"skipped": True, "session": session_day.isoformat(),
                "reason": "this session was already processed", "summary": summary(book)}

    # 1-2: yesterday's decisions fill at today's open; mark to today's close
    today_opens = {s: float(opens.loc[session, s]) for s in INDIA_TICKERS if pd.notna(opens.loc[session].get(s))}
    today_closes = {s: float(closes.loc[session, s]) for s in INDIA_TICKERS if pd.notna(closes.loc[session].get(s))}
    accrue_cash_yield(book, session_day)
    filled = execute_pending(book, today_opens, session_day)
    start_benchmark(book, today_opens, session_day)
    marked = mark_to_market(book, today_closes, session_day)
    equity = marked["equity"]
    share = equity / len(INDIA_TICKERS)

    # 3: learn from views that are now old enough to judge
    memory = mem.load_memory(memory_path)
    lessons = mem.learn(memory, {s: closes[s].dropna() for s in INDIA_TICKERS}, session_day)
    weights = memory["weights"]

    # 4-5: news, views, decisions
    decisions, news_digest = [], {}
    news_cutoff = _next_open(session_day)
    for symbol in INDIA_TICKERS:
        history = closes[symbol].dropna()
        try:
            news = _news_known_by(news_fn(symbol, on=today), symbol, news_cutoff, now)
        except Exception as e:  # news must never stop the run
            print(f"{symbol}: news unavailable ({type(e).__name__}); deciding without it")
            news = {"signal": {"view": 0.0, "score": 0.0, "strength": 0.0, "material_count": 0}, "headlines": []}
        view = advisors.views(history, news["signal"])
        mem.remember(memory, session_day, symbol, view, float(history.iloc[-1]))

        position = book["positions"].get(symbol, {"qty": 0})
        value = position["qty"] * today_closes.get(symbol, 0.0)
        exposure = value / share if share > 0 else 0.0
        decision = brain.decide(symbol, view, weights, exposure,
                                _sessions_since_trade(book, symbol, closes.index),
                                value / equity if equity > 0 else 0.0, 1 / len(INDIA_TICKERS))
        if decision.action != "HOLD":
            queue_target(book, symbol, share * decision.target_exposure, "; ".join(decision.reasons), session_day)

        top = [s for s in stories(news["headlines"], symbol) if s["material"] >= 0.5][:4]
        news_digest[symbol] = {"signal": news["signal"], "top": top}
        decisions.append({**decision.as_dict(), "views": view, "value": round(value, 2)})

    book["last_session"] = session_day.isoformat()
    book["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    save_book(book, book_path)
    mem.save_memory(memory, memory_path)

    result = {"skipped": False, "session": session_day.isoformat(), "filled": filled, "marked": marked,
              "decisions": decisions, "queued": list(book["pending"]), "weights": weights,
              "lessons": lessons, "news": news_digest, "summary": summary(book)}
    journal = load_journal(journal_path)
    journal = [e for e in journal if e["session"] != result["session"]] + [{
        "session": result["session"], "ran_utc": book["last_run_utc"], "equity": marked["equity"],
        "benchmark_equity": marked["benchmark_equity"], "filled": filled, "decisions": decisions,
        "weights": weights, "lessons_learned": len(lessons), "news": news_digest}]
    _save_journal(journal, journal_path)
    return result


def weekdays_behind(last_session: str, today: date) -> int:
    """Weekdays strictly between the last processed session and today."""
    day, count = date.fromisoformat(last_session) + timedelta(days=1), 0
    while day < today:
        count += day.weekday() < 5
        day += timedelta(days=1)
    return count


def main() -> None:
    import faulthandler

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)  # nothing is saved; next run retries
    result = run_day()
    s = result["summary"]
    print(f"HANSEI — session {result['session']}  (fake money, real prices, no broker)")
    if result["skipped"]:
        print(f"  skipped: {result['reason']}")
    else:
        for t in result["filled"]:
            print(f"  FILLED {t['action']:<4} {t['symbol'].replace('.NS', ''):<10} {t['qty']:>4} @ ₹{t['price']:,.2f}"
                  f"  (cost ₹{t['cost']:.2f})")
        for d in result["decisions"]:
            print(f"  {d['action']:<9} {d['symbol'].replace('.NS', ''):<10} {d['reasons'][0]}")
        print(f"  trust: " + ", ".join(f"{k} {v:.0%}" for k, v in result["weights"].items()))
    print(f"  HANSEI   ₹{s['equity']:,.2f} ({s['return']:+.2%})   buy&hold ₹{s['benchmark_equity']:,.2f} "
          f"({s['benchmark_return']:+.2%})   difference {s['vs_benchmark']:+.2%}")

    behind = weekdays_behind(result["session"], datetime.now(IST).date())
    if behind > STALE_WEEKDAYS:   # fail the scheduled run so GitHub emails about it
        sys.exit(f"HANSEI is stuck: {behind} weekdays since session {result['session']} were never processed.")


if __name__ == "__main__":
    main()
