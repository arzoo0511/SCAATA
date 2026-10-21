"""Results surprises: how a company's quarterly EPS compared with what analysts expected.

    python -m scaata.agent.earnings fetch      # history for the large-cap universe (Yahoo)
    python -m scaata.agent.earnings study      # does a surprise predict the next weeks' returns?

A well-documented pattern (post-earnings-announcement drift) is that stocks
keep moving in the direction of a results surprise for weeks afterwards:
markets underreact to news about earnings. This module tests whether that
holds for large NSE companies before HANSEI is allowed to use it.

Yahoo keeps, for each NSE company, every results date since about 2005 with
the consensus EPS estimate, the reported EPS and the surprise. Its
announcement times are unreliable (Indian companies report during market
hours, after the close, or at night), so a surprise is only treated as known
from the close of the first session *after* the announcement's date in IST.
That gives up a day of drift to rule out look-ahead.

Surprise percentages explode when the estimate is near zero, so each
surprise becomes a bounded score: tanh(surprise% / SCALE), with SCALE fixed
in advance at 10% (a 10% beat scores +0.76).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from scaata.config import INDIA_TICKERS, ROOT_DIR

EARNINGS_DIR = ROOT_DIR / "data_cache" / "earnings"
RESULTS_PATH = ROOT_DIR / "results" / "hansei_earnings_study.json"
IST = timezone(timedelta(hours=5, minutes=30))
SCALE = 10.0                    # surprise % that maps to tanh(1) ~ 0.76; fixed before testing
YAHOO_PAUSE_SECONDS = 2.0
HORIZONS = (5, 20, 60)
SPLITS = (("Tune 2007-2019", "2007-01-01", "2019-12-31"), ("Holdout 2020-", "2020-01-01", None),
          ("Final 2024H2-", "2024-07-01", None))


def fetch_one(symbol: str) -> list[dict]:
    """Every past results event Yahoo has for `symbol`, oldest first."""
    import yfinance as yf

    frame = yf.Ticker(symbol).get_earnings_dates(limit=100)
    if frame is None or frame.empty:
        return []
    events = []
    for stamp, row in frame.iterrows():
        reported, estimate = row.get("Reported EPS"), row.get("EPS Estimate")
        if pd.isna(reported) or pd.isna(estimate):
            continue                         # a future date, or no consensus to compare with
        events.append({"announced_utc": stamp.tz_convert("UTC").isoformat(),
                       "estimate": float(estimate), "reported": float(reported),
                       "surprise_pct": None if pd.isna(row.get("Surprise(%)")) else float(row["Surprise(%)"])})
    return sorted(events, key=lambda e: e["announced_utc"])


def fetch_all(symbols, refresh: bool = False) -> list[str]:
    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for symbol in symbols:
        path = EARNINGS_DIR / f"{symbol.replace('.NS', '')}.json"
        if path.exists() and not refresh:
            continue
        try:
            path.write_text(json.dumps(fetch_one(symbol), indent=1), encoding="utf-8")
        except Exception as e:
            failed.append(f"{symbol} ({type(e).__name__})")
        time.sleep(YAHOO_PAUSE_SECONDS)
    return failed


def load_events(symbol: str) -> list[dict]:
    path = EARNINGS_DIR / f"{symbol.replace('.NS', '')}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def surprise_score(event: dict) -> float:
    """Bounded surprise in [-1, 1]. Uses the reported and estimated EPS (not
    Yahoo's percentage) with the estimate's size floored, so a near-zero
    estimate can't turn a small miss into an enormous number."""
    estimate, reported = event["estimate"], event["reported"]
    pct = 100 * (reported - estimate) / max(abs(estimate), 1e-9)
    return float(np.tanh(pct / SCALE))


def first_usable_session(announced_utc: str, sessions: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The first session strictly after the announcement's IST date: the
    earliest close at which the surprise is certainly public."""
    day = datetime.fromisoformat(announced_utc).astimezone(IST).date()
    later = sessions[sessions > pd.Timestamp(day)]
    return later[0] if len(later) else None


def event_table(closes: pd.DataFrame, symbols=None) -> pd.DataFrame:
    """One row per results event: the stock, the session the surprise becomes
    usable, and its score."""
    rows = []
    for symbol in symbols or closes.columns:
        series = closes[symbol].dropna()
        for e in load_events(symbol):
            t0 = first_usable_session(e["announced_utc"], series.index)
            if t0 is not None:
                rows.append({"symbol": symbol, "usable": t0, "score": surprise_score(e)})
    return pd.DataFrame(rows)


def results_views(closes: pd.DataFrame, events: pd.DataFrame, hold: int = 60) -> pd.DataFrame:
    """The advisor's view on every stock and session: the latest surprise's
    score from its usable session for `hold` sessions, fading linearly to
    zero, and silent before or after."""
    out = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    position = {t: i for i, t in enumerate(closes.index)}
    for row in events.sort_values("usable").itertuples():
        start = position[row.usable]
        span = closes.index[start:start + hold]
        fade = np.linspace(1.0, 0.0, hold, endpoint=False)[:len(span)]
        out.loc[span, row.symbol] = row.score * fade
    return out.fillna(0.0).where(closes.notna())


def study(closes: pd.DataFrame, seed: int = 0) -> dict:
    """Event study, scored like the other advisors (signal_study.score):
    liked (beat) minus disliked (miss) forward return over cash, with a
    month-block bootstrap. Each event counts once, on its usable session."""
    from scaata.agent.signal_study import forward_excess, score

    rng = np.random.default_rng(seed)
    events = event_table(closes)
    on_event = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    for row in events.itertuples():
        on_event.at[row.usable, row.symbol] = row.score
    out = {"events": int(len(events)), "stocks_with_events": int(events["symbol"].nunique()) if len(events) else 0}
    for label, start, end in SPLITS:
        window = slice(pd.Timestamp(start), pd.Timestamp(end) if end else None)
        out[label] = {f"{h}_sessions": score(on_event.loc[window], forward_excess(closes, h).loc[window], rng)
                      for h in HORIZONS}
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from scaata.agent.signal_study import UNIVERSE, load_universe

    command = sys.argv[1] if len(sys.argv) > 1 else "study"
    if command == "fetch":
        failed = fetch_all(UNIVERSE, refresh="--refresh" in sys.argv)
        print(f"fetched {len(UNIVERSE) - len(failed)}/{len(UNIVERSE)}; failed: {', '.join(failed) or 'none'}")
        return
    closes, _ = load_universe(date.today())
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "scale": SCALE,
              "large_caps": study(closes), "paper_book_stocks": study(closes[list(INDIA_TICKERS)])}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for group in ("large_caps", "paper_book_stocks"):
        r = report[group]
        print(f"\n{group.replace('_', ' ')}: {r['events']} results events, {r['stocks_with_events']} stocks; "
              f"beat minus miss, forward return over cash")
        for split, _, _ in SPLITS:
            for horizon, c in r[split].items():
                if not c.get("opinions"):
                    continue
                lo, hi = c["spread_ci95"]
                print(f"  {split:<16}{horizon.replace('_', ' '):<13}spread {c['spread']:+.2%} [{lo:+.2%}, {hi:+.2%}]"
                      f"  P(>0) {c['prob_spread_positive']:.0%}  hit {c['hit_rate']:.0%}"
                      f"  ({c['opinions']:,} events, {c['liked_share']:.0%} beats)")
    print(f"\nsaved {RESULTS_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
