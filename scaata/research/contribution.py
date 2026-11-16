"""The contribution ledger: what each advisor adds to the *whole system*, risk rules included.

The skill ledger (`learners.py`) scores an advisor's forecast on its own. That ignores that the brain
holds by default, caps its exposure moves and waits between trades, so a good forecaster can add nothing
to the portfolio and a mediocre one can matter. This ledger asks the portfolio question directly:

    replay the real brain with every possible subset of advisors allowed to speak,
    and give each advisor its Shapley value: its average marginal effect over all
    the orders in which advisors could have been added.

With nothing speaking the brain simply holds all five stocks, so the empty subset is the
"hold by default" baseline and the full set is the live system. The values add up exactly to the
difference between the two.

The news advisor is left out of the game: there is no scored news before 2012, so in this history it
never speaks and its value is zero by construction. That is a limit of the data, not a finding.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scaata.agent import backtest as bt
from scaata.config import ROOT_DIR
from scaata.research.engine import Engine, silence

CONTRIBUTION_PATH = ROOT_DIR / "research" / "contribution.json"
PLAYERS = ("trend", "volatility", "momentum")
METRICS = ("sharpe", "cagr", "max_drawdown")


def shapley(values: dict[frozenset, float], players: tuple[str, ...]) -> dict[str, float]:
    """Exact Shapley value of each player given the value of every coalition."""
    n = len(players)
    out = {}
    for p in players:
        total = 0.0
        others = [q for q in players if q != p]
        for k in range(len(others) + 1):
            weight = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for subset in itertools.combinations(others, k):
                s = frozenset(subset)
                total += weight * (values[s | {p}] - values[s])
        out[p] = total
    return out


def run(path: Path | None = None) -> dict:
    """Replays every subset on both spans and saves the result. About ten minutes."""
    started = time.time()
    engine = Engine()
    subsets = [frozenset(c) for k in range(len(PLAYERS) + 1) for c in itertools.combinations(PLAYERS, k)]
    results = {span: {} for span in engine.periods}
    for subset in subsets:
        silent = [a for a in ("trend", "volatility", "momentum", "news") if a not in subset]
        views = engine.views
        for a in silent:
            views = silence(views, a)
        for span in engine.periods:
            stats = bt.evaluate(engine._run(span, views))
            results[span][subset] = {"sharpe": stats["hansei"]["sharpe"], "cagr": stats["hansei"]["cagr"],
                                     "max_drawdown": stats["hansei"]["max_drawdown"], "holding": stats["buy_and_hold"],
                                     "trades": stats["trades"]}
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "players": list(PLAYERS),
           "prices_sha256": engine.meta.get("sha256"), "prices_through": engine.meta.get("last"),
           "no_borrowing": engine.no_borrowing, "spans": {}}
    for span, table in results.items():
        values = {m: {s: v[m] for s, v in table.items()} for m in METRICS}
        out["spans"][span] = {
            "coalitions": [{"advisors": sorted(s), **{m: v[m] for m in METRICS}, "trades": v["trades"]}
                           for s, v in table.items()],
            "shapley": {m: shapley(values[m], PLAYERS) for m in METRICS},
            "baseline": {m: values[m][frozenset()] for m in METRICS},       # hold by default
            "full": {m: values[m][frozenset(PLAYERS)] for m in METRICS},
            "holding": next(iter(table.values()))["holding"]}
    out["runtime_seconds"] = round(time.time() - started, 1)
    path = path or CONTRIBUTION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
