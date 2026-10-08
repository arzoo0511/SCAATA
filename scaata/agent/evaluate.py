"""HANSEI's report card, rebuilt after every daily run.

Answers four questions from the files the agent already keeps (paper book,
journal, memory, news cache), with no downloads:

1. Is it beating holding? Return, drawdown and the gap, day by day.
2. Why? Splits the gap into *exposure* (sitting partly in cash while the
   market moved) and *selection* (which stocks it held, and how much).
3. Which advisors are actually right? Each view against the stock's return
   over the next 5 and 20 sessions.
4. Is the machinery healthy? Missed sessions, news scored by keywords
   instead of the LLM, orders too small to buy a single share.

It also says how far the live record is from meaning anything: how many
sessions it would take for the backtest's edge to stand out from noise.
The live record is for watching, not for tuning: change the agent only on
what the backtest shows across both halves and the settings grid.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from scaata.agent.advisors import ADVISORS
from scaata.config import ROOT_DIR
from scaata.live.paper_book import liquid_yield

BOOK_PATH = ROOT_DIR / "paper_book_india.json"
JOURNAL_PATH = ROOT_DIR / "forward_test" / "hansei_journal.json"
MEMORY_PATH = ROOT_DIR / "forward_test" / "hansei_memory.json"
NEWS_DIR = ROOT_DIR / "data_cache" / "news_india"
BACKTEST_PATH = ROOT_DIR / "results" / "hansei_backtest.json"
EVALUATION_PATH = ROOT_DIR / "forward_test" / "hansei_evaluation.json"
HORIZONS = (5, 20)              # sessions ahead an advisor's view is judged on
TRADING_DAYS = 248              # NSE sessions a year


def _max_drawdown(values: list[float]) -> float:
    curve = np.asarray(values, dtype=float)
    return float((curve / np.maximum.accumulate(curve) - 1).min()) if len(curve) else 0.0


def performance(book: dict) -> dict:
    """HANSEI against holding, and the gap split into exposure and selection.

    The *same-exposure* line is what HANSEI would have made holding all five
    names equally with the stock fraction it actually had each day, and cash
    for the rest. HANSEI minus that line is selection; that line minus
    holding is exposure."""
    history = book["equity_history"]
    start = book["initial_cash"]
    equity = [h["equity"] for h in history]
    held = [h["benchmark_equity"] for h in history]
    same_exposure, prev = [start], None
    for h in history:
        if prev is not None:
            invested = 1 - prev["cash"] / prev["equity"] if prev["equity"] > 0 else 0.0
            days = (date.fromisoformat(h["date"]) - date.fromisoformat(prev["date"])).days
            cash_return = liquid_yield(date.fromisoformat(h["date"]).year) * days / 365
            market_return = h["benchmark_equity"] / prev["benchmark_equity"] - 1
            same_exposure.append(same_exposure[-1] * (1 + invested * market_return + (1 - invested) * cash_return))
        prev = h
    ret = equity[-1] / start - 1 if equity else 0.0
    held_ret = held[-1] / start - 1 if held else 0.0
    same_ret = same_exposure[-1] / start - 1 if history else 0.0
    gaps = np.diff(np.log(equity)) - np.diff(np.log(held)) if len(history) > 1 else np.array([])
    invested = [1 - h["cash"] / h["equity"] for h in history if h["equity"] > 0]
    return {
        "sessions": len(history),
        "first": history[0]["date"] if history else None,
        "last": history[-1]["date"] if history else None,
        "hansei_return": ret, "hold_return": held_ret, "gap": ret - held_ret,
        "hansei_max_drawdown": _max_drawdown([start, *equity]),
        "hold_max_drawdown": _max_drawdown([start, *held]),
        "sessions_ahead": int(sum(e > b for e, b in zip(equity, held))),
        "gap_from_exposure": same_ret - held_ret,
        "gap_from_selection": ret - same_ret,
        "average_invested": float(np.mean(invested)) if invested else 0.0,
        "daily_gap_volatility": float(gaps.std(ddof=1)) if len(gaps) > 1 else None,
    }


def advisor_scorecard(memory: dict, horizons=HORIZONS) -> dict:
    """For each advisor and horizon: how often its view had the right sign
    (views smaller than 0.05 are no opinion and are skipped), and the
    correlation between view and the stock's forward return."""
    by_symbol: dict[str, list[dict]] = {}
    for obs in sorted(memory["observations"], key=lambda o: o["date"]):
        by_symbol.setdefault(obs["symbol"], []).append(obs)
    card = {}
    for horizon in horizons:
        pairs = {a: [] for a in ADVISORS}
        for series in by_symbol.values():
            for i, obs in enumerate(series[:-horizon] if horizon < len(series) else []):
                forward = series[i + horizon]["close"] / obs["close"] - 1
                for a in ADVISORS:
                    pairs[a].append((obs["views"].get(a, 0.0), forward))
        card[f"{horizon}_sessions"] = {}
        for a, xs in pairs.items():
            opinions = [(v, r) for v, r in xs if abs(v) >= 0.05]
            views, rets = (np.array([p[0] for p in xs]), np.array([p[1] for p in xs])) if xs else (None, None)
            corr = (float(np.corrcoef(views, rets)[0, 1])
                    if xs and len(xs) > 2 and views.std() > 0 and rets.std() > 0 else None)
            card[f"{horizon}_sessions"][a] = {
                "judged": len(opinions),
                "hit_rate": float(np.mean([np.sign(v) == np.sign(r) for v, r in opinions])) if opinions else None,
                "correlation": corr,
            }
    return card


def health(book: dict, journal: list[dict], news_dir: Path = NEWS_DIR, today: date | None = None) -> dict:
    """Things that mean the machinery, not the market, is the problem."""
    today = today or datetime.now(timezone.utc).date()
    sessions = [e["session"] for e in journal]
    weekdays_missing = []
    if sessions:
        day = date.fromisoformat(sessions[0])
        while day.isoformat() < sessions[-1]:
            if day.weekday() < 5 and day.isoformat() not in sessions:
                weekdays_missing.append(day.isoformat())   # an NSE holiday, or a missed session
            day += timedelta(days=1)
    scorers: dict[str, dict[str, int]] = {}
    for f in sorted(news_dir.glob("*.json")) if news_dir.exists() else []:
        day = f.name[:10]
        for name in re.findall(r'"scorer": "([^"]+)"', f.read_text(encoding="utf-8")):
            scorers.setdefault(day, {}).setdefault(name, 0)
            scorers[day][name] += 1
    keyword_days = [d for d, c in scorers.items() if c.get("keywords", 0) > sum(c.values()) / 2]
    # an order that was queued but filled nothing at the next session
    unfilled = []
    for prev, nxt in zip(journal, journal[1:]):
        queued = {d["symbol"] for d in prev["decisions"] if d["action"] != "HOLD"}
        filled = {t["symbol"] for t in nxt["filled"]}
        unfilled += [{"queued_on": prev["session"], "symbol": s} for s in sorted(queued - filled)]
    return {
        "last_session": book.get("last_session"),
        "weekdays_without_a_session": weekdays_missing,
        "sessions_out_of_order": sessions != sorted(sessions),
        "news_days_scored_by_keywords": keyword_days,
        "news_scored_by_keywords_latest": bool(scorers) and max(scorers) in keyword_days,
        "orders_that_filled_nothing": unfilled,
    }


def significance(perf: dict, backtest: dict | None) -> dict:
    """How many sessions until the backtest's edge would stand out (2 standard
    errors) from the day-to-day noise in the live gap seen so far."""
    sigma = perf.get("daily_gap_volatility")
    edge = (backtest or {}).get("periods", {}).get("Full period", {}).get("cagr_diff")
    out = {"backtest_cagr_edge": edge, "daily_gap_volatility": sigma}
    if sigma and edge and edge > 0:
        daily_edge = np.log1p(edge) / TRADING_DAYS
        needed = int(np.ceil((2 * sigma / daily_edge) ** 2))
        out.update({"sessions_needed": needed, "years_needed": round(needed / TRADING_DAYS, 1),
                    "sessions_so_far": perf["sessions"]})
    return out


def evaluate(book: dict, journal: list[dict], memory: dict, backtest: dict | None = None,
             news_dir: Path = NEWS_DIR, today: date | None = None) -> dict:
    perf = performance(book)
    return {
        "as_of": perf["last"],   # no wall-clock stamp: an unchanged book gives an unchanged report, so no empty commits
        "performance": perf,
        "advisors": advisor_scorecard(memory),
        "trust": memory["weights"],
        "health": health(book, journal, news_dir, today),
        "significance": significance(perf, backtest),
        "backtest_reference": {k: {"cagr_diff": v.get("cagr_diff"), "hansei_max_drawdown": v["hansei"]["max_drawdown"],
                                   "hold_max_drawdown": v["buy_and_hold"]["max_drawdown"]}
                               for k, v in (backtest or {}).get("periods", {}).items()},
    }


def _load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = evaluate(_load(BOOK_PATH, None), _load(JOURNAL_PATH, []), _load(MEMORY_PATH, None),
                      _load(BACKTEST_PATH, None))
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    p, h, s = report["performance"], report["health"], report["significance"]
    print(f"HANSEI report card -- {p['sessions']} sessions, {p['first']} to {p['last']}")
    print(f"  return   HANSEI {p['hansei_return']:+.2%}   holding {p['hold_return']:+.2%}   gap {p['gap']:+.2%}"
          f"   (ahead on {p['sessions_ahead']}/{p['sessions']} sessions)")
    print(f"  gap from exposure {p['gap_from_exposure']:+.2%}, from selection {p['gap_from_selection']:+.2%}"
          f"   (average invested {p['average_invested']:.0%})")
    print(f"  max drawdown  HANSEI {p['hansei_max_drawdown']:.2%}   holding {p['hold_max_drawdown']:.2%}")
    for horizon, card in report["advisors"].items():
        cells = [f"{a} {c['hit_rate']:.0%} of {c['judged']}" if c["hit_rate"] is not None else f"{a} --"
                 for a, c in card.items()]
        print(f"  advisors right over {horizon.replace('_', ' ')}: " + ", ".join(cells))
    if s.get("sessions_needed"):
        print(f"  significance: ~{s['sessions_needed']} sessions (~{s['years_needed']} years) needed for the "
              f"backtest edge to stand out; {s['sessions_so_far']} so far")
    problems = []
    if h["sessions_out_of_order"]:
        problems.append("sessions out of order")
    if h["news_scored_by_keywords_latest"]:
        problems.append("the latest news was scored by keywords, not the LLM (is GROQ_API_KEY set?)")
    if h["orders_that_filled_nothing"]:
        problems.append(f"{len(h['orders_that_filled_nothing'])} order(s) filled nothing (too small for one share)")
    print("  health: " + ("; ".join(problems) if problems else "ok"))


if __name__ == "__main__":
    main()
