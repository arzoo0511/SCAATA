"""What did each stock do after each kind of news?

For every scored headline in the archive (2012 onward) that was about the
company and material, this takes its event type (`scaata.agent.events`) and
its tone, and measures the stock's return over the next 5 and 20 sessions
minus the equal-weight basket of the five stocks over the same sessions.

How it avoids fooling itself:

- The clock starts at the close of the first session strictly after the
  headline's date, so the reaction on the day itself is not counted as
  something that could have been traded.
- A stock with many headlines of one kind and tone in one week counts once
  for that week, and the uncertainty is a bootstrap over weeks, because
  news comes in clumps.
- Headlines the scoring model could have "remembered" (published before its
  knowledge cutoff, mid-2024) are reported separately from those it could
  not. Only the later group is a clean test of the scorer; it is small.

This is a description of the past, not a signal. An earlier, stricter test
found the news view as a whole had no reliable edge; nothing here changes
how the agent trades.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scaata.agent.events import LABELS, event_type
from scaata.config import INDIA_TICKERS, ROOT_DIR

STUDY_PATH = ROOT_DIR / "research" / "event_study.json"
HORIZONS = (5, 20)
MATERIAL, TONE = 0.5, 0.15
CLEAN_FROM = "2024-07-01"      # headlines after this could not have been in the scoring model's training
MIN_EVENTS = 15
DRAWS = 1000


def events_frame(scored_by_symbol: dict[str, list[dict]], closes: pd.DataFrame) -> pd.DataFrame:
    """One row per (stock, week, event type, tone): the first material headline of that kind that week,
    with the stock's excess return over the next sessions."""
    basket = np.log(closes).diff().mean(axis=1)
    own = np.log(closes).diff()
    sessions = closes.index
    rows = []
    for symbol, headlines in scored_by_symbol.items():
        if symbol not in closes.columns:
            continue
        for h in headlines:
            if not h.get("relevant") or h.get("material", 0) < MATERIAL or abs(h.get("sentiment", 0)) < TONE:
                continue
            published = pd.Timestamp(h["published_utc"]).tz_localize(None).normalize()
            start = sessions.searchsorted(published, side="right")     # first session strictly after the date
            if start + max(HORIZONS) >= len(sessions):
                continue
            row = {"symbol": symbol, "date": published, "week": published.to_period("W").start_time,
                   "event": h.get("event") or event_type(h.get("title", "")),
                   "tone": "good" if h["sentiment"] > 0 else "bad", "title": h.get("title", "")}
            for horizon in HORIZONS:
                window = slice(start + 1, start + 1 + horizon)
                row[f"excess_{horizon}"] = float(np.exp(own[symbol].iloc[window].sum() - basket.iloc[window].sum()) - 1)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("date").drop_duplicates(["symbol", "week", "event", "tone"], keep="first")


def _interval(values: np.ndarray, weeks: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval for the mean, resampling whole weeks."""
    unique = np.unique(weeks)
    by_week = [values[weeks == w] for w in unique]
    means = np.empty(DRAWS)
    for k in range(DRAWS):
        picked = rng.integers(0, len(by_week), len(by_week))
        means[k] = np.concatenate([by_week[i] for i in picked]).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarise(frame: pd.DataFrame, seed: int = 0) -> list[dict]:
    """Mean excess return after each (event, tone), with its interval, for the whole archive and
    for the part the scorer could not have remembered."""
    rng = np.random.default_rng(seed)
    out = []
    if frame.empty:
        return out
    for (event, tone), part in frame.groupby(["event", "tone"]):
        for span, rows in (("all", part), ("clean", part[part["date"] >= pd.Timestamp(CLEAN_FROM)])):
            if len(rows) < MIN_EVENTS:
                continue
            record = {"event": event, "label": LABELS.get(event, event), "tone": tone, "span": span,
                      "events": int(len(rows)), "weeks": int(rows["week"].nunique())}
            for horizon in HORIZONS:
                values = rows[f"excess_{horizon}"].values
                low, high = _interval(values, rows["week"].values.astype("datetime64[D]").astype(int), rng)
                expected = low > 0 if tone == "good" else high < 0
                record[f"h{horizon}"] = {"mean": float(values.mean()), "low": low, "high": high,
                                         "reliable": bool(low > 0 or high < 0), "as_expected": bool(expected)}
            out.append(record)
    return out


def run(path: Path | None = None) -> dict:
    """Builds the study from the scored archive and the research price snapshot, and saves it."""
    from scaata.agent.news_history import load_scored
    from scaata.research import prices

    _, closes, meta = prices.load()
    scored = {s: load_scored(s) for s in INDIA_TICKERS}
    frame = events_frame(scored, closes)
    table = summarise(frame)
    study = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "headlines_scored": int(sum(len(v) for v in scored.values())), "events": int(len(frame)),
             "first": str(frame["date"].min().date()) if len(frame) else None,
             "last": str(frame["date"].max().date()) if len(frame) else None,
             "clean_from": CLEAN_FROM, "horizons": list(HORIZONS), "min_events": MIN_EVENTS, "table": table,
             "prices_sha256": meta.get("sha256"),
             "method": "Excess return over the equal-weight basket of the five stocks, starting at the close of "
                       "the first session after the headline; one event per stock, week, type and tone; 95% "
                       "intervals from resampling weeks."}
    path = path or STUDY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(study, indent=1, ensure_ascii=False), encoding="utf-8")
    return study
