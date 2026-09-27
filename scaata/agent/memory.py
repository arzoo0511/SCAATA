"""Memory: how the agent learns which advisors to trust.

Every day it records each advisor's view on each stock and the price. Once
`HORIZON` sessions have passed it knows what actually happened, and scores
each view: an advisor that said "own it" before a rise, or "get out" before
a fall, did well; one that said the opposite did badly. Those scores feed a
multiplicative-weights (Hedge) update, so trust shifts toward advisors that
keep being right and away from ones that keep being wrong.

Trust never drops to zero (`MIN_WEIGHT`): markets change, and an advisor
that was wrong for a month may be right for the next.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from scaata.agent.advisors import ADVISORS
from scaata.config import ROOT_DIR
from scaata.live.paper_book import liquid_yield
from scaata.strategies.hedge import HedgeWeights, floor_weights_on_simplex

MEMORY_PATH = ROOT_DIR / "forward_test" / "hansei_memory.json"
HORIZON = 20                    # sessions after which a view is judged
LEARNING_RATE = 0.05            # Hedge eta on per-day normalized losses. Slow on purpose:
                                # faster settings chased 10-day noise and traded more in backtests
MIN_WEIGHT = 0.05
SILENT = 0.05                   # a view smaller than this counts as no opinion


def new_memory() -> dict:
    return {"weights": {a: 1 / len(ADVISORS) for a in ADVISORS}, "observations": [],
            "weight_history": [], "lessons": []}


def load_memory(path: Path | None = None) -> dict:
    path = path or MEMORY_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else new_memory()


def save_memory(memory: dict, path: Path | None = None) -> None:
    path = path or MEMORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def remember(memory: dict, on: date, symbol: str, views: dict[str, float], close: float) -> None:
    """Records today's views (one entry per stock per day)."""
    key = (on.isoformat(), symbol)
    memory["observations"] = [o for o in memory["observations"] if (o["date"], o["symbol"]) != key]
    memory["observations"].append({"date": on.isoformat(), "symbol": symbol, "views": views,
                                   "close": close, "judged": False})


def learn(memory: dict, closes_by_symbol: dict[str, "pd.Series"], on: date,
          learning_rate: float = LEARNING_RATE) -> list[dict]:
    """Judges every observation that is now `HORIZON` sessions old, and
    updates trust. Returns the lessons learned today."""
    import pandas as pd

    lessons = []
    pending = [o for o in memory["observations"] if not o["judged"]]
    by_day: dict[str, list[dict]] = {}
    for obs in pending:
        closes = closes_by_symbol.get(obs["symbol"])
        if closes is None:
            continue
        after = closes.loc[closes.index > pd.Timestamp(obs["date"])]
        if len(after) < HORIZON:
            continue
        # judged against what the money would have earned in a liquid fund instead
        hurdle = liquid_yield(int(obs["date"][:4])) * HORIZON / 252
        forward = float(after.iloc[HORIZON - 1] / obs["close"] - 1) - hurdle
        obs["judged"] = True
        obs["forward_return"] = round(forward, 4)
        by_day.setdefault(obs["date"], []).append(obs)

    weights = memory["weights"]
    for day in sorted(by_day):
        observations = by_day[day]
        # An advisor with no opinion that day (e.g. news on a quiet day) is
        # neither rewarded nor punished: silence is not the same as being wrong.
        active = [a for a in ADVISORS if any(abs(o["views"].get(a, 0.0)) >= SILENT for o in observations)]
        if len(active) < 2:
            continue
        # loss per advisor: -view x excess return, averaged over the day's stocks
        losses = np.array([np.mean([-o["views"].get(a, 0.0) * o["forward_return"] for o in observations])
                           for a in active])
        mass = sum(weights[a] for a in active)
        hedge = HedgeWeights(len(active), learning_rate, np.array([weights[a] / mass for a in active]))
        updated_active = hedge.update(losses) * mass
        before = dict(weights)
        merged = {**weights, **{a: float(w) for a, w in zip(active, updated_active)}}
        floored = floor_weights_on_simplex(np.array([merged[a] for a in ADVISORS]), MIN_WEIGHT)
        weights = {a: round(float(w), 4) for a, w in zip(ADVISORS, floored)}
        best = active[int(np.argmin(losses))]
        worst = active[int(np.argmax(losses))]
        lessons.append({"learned_on": on.isoformat(), "about_day": day, "best": best, "worst": worst,
                        "weights_before": before, "weights_after": weights,
                        "stocks": {o["symbol"]: o["forward_return"] for o in observations}})
    memory["weights"] = weights
    if lessons:
        memory["weight_history"].append({"date": on.isoformat(), **weights})
        memory["lessons"] = (memory["lessons"] + lessons)[-200:]
    # keep memory bounded: judged observations older than a year are no longer needed
    memory["observations"] = [o for o in memory["observations"] if not o["judged"]
                              or o["date"] >= f"{on.year - 1}-{on.month:02d}-{on.day:02d}"]
    return lessons
