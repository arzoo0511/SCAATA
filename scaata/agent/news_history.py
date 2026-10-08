"""A news archive for backtesting HANSEI's news advisor.

    python -m scaata.agent.news_history fetch     # headlines, week by week (Google News)
    python -m scaata.agent.news_history score     # LLM-score them, month by month (Groq)
    python -m scaata.agent.news_history status
    python -m scaata.agent.news_history run       # both, waiting out the daily Groq budget until done

The live agent reads Google News, so the archive does too: the same
"company name" query, bounded with `after:`/`before:` to one week at a
time. Each stock-month is then scored by the same prompt and models as the
live agent (`news.llm_scores`), so the backtest's news views are built the
way the live ones are.

Both steps are resumable (every week and month is its own file) and polite:
Google is asked once every few seconds, and Groq scoring stops for the day
at `DAILY_REQUEST_BUDGET`, well under the free tier's 1,000 requests, so the
live agent's own scoring never runs out.

Caveats, both real:
- the archive is thinner than the live feed (a handful of headlines a week
  for older years, against up to 40 every three days live);
- the LLM was trained on data up to about mid-2024 and may "remember" how
  an older story ended. Views before `MODEL_KNOWLEDGE_CUTOFF` can look
  better than they would have been at the time; the honest test is after it.
"""
from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from scaata.agent import news
from scaata.config import INDIA_TICKERS, ROOT_DIR

ARCHIVE_DIR = ROOT_DIR / "data_cache" / "news_history"
LEDGER_PATH = ARCHIVE_DIR / "groq_ledger.json"
START = date(2012, 1, 2)                 # a Monday
MODEL_KNOWLEDGE_CUTOFF = date(2024, 7, 1)
GOOGLE_PAUSE_SECONDS = 3.0
DAILY_REQUEST_BUDGET = 800               # of Groq's 1,000/day free tier; the rest is the live agent's
BATCH = news.MAX_HEADLINES               # headlines per scoring call, as live
TOKENS_PER_MINUTE_PAUSE = 9.0            # free tier: 8,000 tokens/min; a call is ~1,000-1,500


def _symbol_dir(symbol: str) -> Path:
    return ARCHIVE_DIR / symbol.replace(".NS", "")


def weeks(start: date = START, end: date | None = None) -> list[date]:
    end = end or date.today()
    out, day = [], start - timedelta(days=start.weekday())
    while day + timedelta(days=7) <= end:
        out.append(day)
        day += timedelta(days=7)
    return out


def _parse(content: bytes) -> list[dict]:
    items = []
    for node in ET.fromstring(content).findall(".//item"):
        title = (node.findtext("title") or "").strip()
        source = (node.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()
        try:
            published = parsedate_to_datetime(node.findtext("pubDate")).astimezone(timezone.utc)
        except Exception:
            continue
        items.append({"title": title, "source": source or "Google News", "published_utc": published.isoformat()})
    return items


def fetch_week(symbol: str, monday: date, http=requests) -> list[dict]:
    query = f'"{news.COMPANY_NAMES[symbol][0]}"'
    sunday = monday + timedelta(days=7)
    url = (f"https://news.google.com/rss/search?q={quote(query)}%20after:{monday.isoformat()}"
           f"%20before:{sunday.isoformat()}&hl=en-IN&gl=IN&ceid=IN:en")
    response = http.get(url, timeout=30)
    response.raise_for_status()
    lo = datetime(monday.year, monday.month, monday.day, tzinfo=timezone.utc) - timedelta(days=1)
    hi = lo + timedelta(days=9)
    return [h for h in _parse(response.content) if lo <= datetime.fromisoformat(h["published_utc"]) < hi]


def fetch_all(symbols=INDIA_TICKERS, end: date | None = None) -> None:
    for symbol in symbols:
        folder = _symbol_dir(symbol)
        folder.mkdir(parents=True, exist_ok=True)
        todo = [w for w in weeks(end=end) if not (folder / f"{w.isoformat()}.json").exists()]
        print(f"{symbol}: {len(todo)} weeks to fetch", flush=True)
        for i, monday in enumerate(todo):
            for attempt in range(4):
                try:
                    items = fetch_week(symbol, monday)
                    break
                except Exception as e:
                    wait = 60 * (attempt + 1)
                    print(f"  {monday}: {type(e).__name__}; waiting {wait}s", flush=True)
                    time.sleep(wait)
            else:
                print(f"  {monday}: giving up for now; rerun to retry", flush=True)
                continue
            (folder / f"{monday.isoformat()}.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
            if i % 50 == 0:
                print(f"  {monday}: {len(items)} headlines", flush=True)
            time.sleep(GOOGLE_PAUSE_SECONDS)


def month_headlines(symbol: str, month: str) -> list[dict]:
    """A stock-month's archived headlines (published in that month), de-duplicated by title."""
    seen, out = set(), []
    for f in sorted(_symbol_dir(symbol).glob("????-??-??.json")):
        for h in json.loads(f.read_text(encoding="utf-8")):
            key = re.sub(r"[^a-z0-9]", "", h["title"].lower())[:80]
            if h["published_utc"][:7] == month and key not in seen:
                seen.add(key)
                out.append(h)
    return sorted(out, key=lambda h: h["published_utc"])


def _ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8")) if LEDGER_PATH.exists() else {}


def _spend(n: int = 1) -> int:
    ledger = _ledger()
    today = datetime.now(timezone.utc).date().isoformat()
    ledger[today] = ledger.get(today, 0) + n
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    return ledger[today]


def score_all(symbols=INDIA_TICKERS, client=None, budget: int = DAILY_REQUEST_BUDGET) -> bool:
    """Scores every fully-fetched stock-month not yet scored. Returns True when
    nothing is left, False if it stopped at the daily budget."""
    client = client if client is not None else news._llm_client()
    if client is None:
        sys.exit("GROQ_API_KEY is not set (in the environment or .env)")
    # month by month across all stocks, the months the model can't know first:
    # they are the honest test, and partial results are usable along the way
    cutoff = MODEL_KNOWLEDGE_CUTOFF.isoformat()[:7]
    months = sorted({w.isoformat()[:7] for w in weeks()}, key=lambda m: (m < cutoff, m))
    for month in months:
        for symbol in symbols:
            folder = _symbol_dir(symbol)
            out = folder / f"scored_{month}.json"
            if out.exists():
                continue
            headlines = month_headlines(symbol, month)
            scored = []
            for k in range(0, len(headlines), BATCH):
                if _ledger().get(datetime.now(timezone.utc).date().isoformat(), 0) >= budget:
                    print(f"daily Groq budget of {budget} reached; rerun tomorrow", flush=True)
                    return False
                chunk = headlines[k:k + BATCH]
                scores = news.llm_scores(chunk, news.COMPANY_NAMES[symbol][0], client)
                _spend()
                time.sleep(TOKENS_PER_MINUTE_PAUSE)
                if scores is None:   # transient failure or rate limit: leave the month for the next run
                    print(f"{symbol} {month}: scoring failed; will retry", flush=True)
                    time.sleep(60)
                    break
                scored += [{**h, **s} for h, s in zip(chunk, scores)]
            else:
                out.write_text(json.dumps(scored, indent=1), encoding="utf-8")
                print(f"{symbol} {month}: {len(scored)} headlines scored", flush=True)
    return True


def load_scored(symbol: str) -> list[dict]:
    out = []
    for f in sorted(_symbol_dir(symbol).glob("scored_*.json")):
        out += json.loads(f.read_text(encoding="utf-8"))
    return out


def scored_months(symbol: str) -> set[str]:
    return {f.stem.removeprefix("scored_") for f in _symbol_dir(symbol).glob("scored_*.json")}


def news_views(symbol: str, sessions: pd.DatetimeIndex, scored: list[dict] | None = None,
               months: set[str] | None = None) -> pd.Series:
    """The news advisor's view on each session, as the live agent would have
    formed it: headlines from the last `news.LOOKBACK_DAYS` days up to that
    day's evening run (16:15 IST), aged to then. NaN where the month isn't
    scored yet."""
    scored = scored if scored is not None else load_scored(symbol)
    months = months if months is not None else scored_months(symbol)
    stamps = sorted(scored, key=lambda h: h["published_utc"])
    times = [datetime.fromisoformat(h["published_utc"]) for h in stamps]
    out = {}
    lo = 0
    for t in sessions:
        run = datetime(t.year, t.month, t.day, 10, 45, tzinfo=timezone.utc)   # 16:15 IST
        if t.strftime("%Y-%m") not in months:
            out[t] = float("nan")
            continue
        start = run - timedelta(days=news.LOOKBACK_DAYS)
        while lo < len(times) and times[lo] < start:
            lo += 1
        hi = lo
        while hi < len(times) and times[hi] <= run:
            hi += 1
        out[t] = news.news_signal(stamps[lo:hi], now=run, symbol=symbol)["view"]
    return pd.Series(out, dtype=float)


def status() -> None:
    for symbol in INDIA_TICKERS:
        folder = _symbol_dir(symbol)
        fetched = len(list(folder.glob("????-??-??.json"))) if folder.exists() else 0
        scored = len(list(folder.glob("scored_*.json"))) if folder.exists() else 0
        count = sum(len(json.loads(f.read_text(encoding="utf-8"))) for f in folder.glob("????-??-??.json")) \
            if folder.exists() else 0
        print(f"{symbol:<14} weeks fetched {fetched}/{len(weeks())}  headlines {count:,}  "
              f"months scored {scored}/{len({w.isoformat()[:7] for w in weeks()})}")
    print(f"Groq requests today: {_ledger().get(datetime.now(timezone.utc).date().isoformat(), 0)}"
          f"/{DAILY_REQUEST_BUDGET}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "fetch":
        fetch_all()
    elif command == "score":
        score_all()
    elif command == "run":   # fetch, then score day after day until done (resumable; safe to kill)
        fetch_all()
        while not score_all():
            now = datetime.now(timezone.utc)
            tomorrow = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) + timedelta(days=1, minutes=5)
            print(f"sleeping until {tomorrow.isoformat()}", flush=True)
            time.sleep(max(60.0, (tomorrow - now).total_seconds()))
    status()


if __name__ == "__main__":
    main()
