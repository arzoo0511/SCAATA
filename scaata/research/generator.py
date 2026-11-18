"""The system proposes its own experiments, and cannot over-search.

Hand-written questions (director.py) come from what the live record raises. This generator adds the other kind:
a family of changes to the brain's existing knobs, tried together, judged as a family.

Rules, all fixed before any candidate runs:

  * Every candidate carries its written reason (the mechanism it is supposed to exploit) and its cost assumption
    (costs are always charged by the replay; a candidate that trades much more must beat the baseline by enough
    to cover them) before it runs. The batch is written to `research/generated.jsonl` first, then the results.
  * The holdout years (2020 on) are SEALED. They have been looked at in every earlier experiment and the evidence
    budget (`review.budget_ledger`) is spent, so a search is not allowed to touch them. Candidates run on the
    years the brain was tuned on (2007-2019) only. Nothing found here can be confirmed here.
  * Many candidates were tried, so the best of them is judged by White's Reality Check (2000): the chance that
    the best of the whole family would look this good if none of them had any real edge. The check is run on
    the daily return differences against the live rule, resampled in blocks.
  * At most one champion per quarter, and only if the family's reality-check p is below 0.05 and the champion
    met its own criteria. A champion is a proposal for forward testing (a shadow book, Phase F), never a change.
  * One batch per quarter. A re-run is refused.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scaata.config import ROOT_DIR
from scaata.research import review

GENERATED_LOG = ROOT_DIR / "research" / "generated.jsonl"
GENERATED_RESULT = ROOT_DIR / "research" / "generated.json"
SPAN = ("2007-01-01", "2019-12-31")
DRAWS, BLOCK = 4000, 20
ALPHA = 0.05
MAX_DRAWDOWN_WORSE = 0.02          # a champion's worst fall may be at most 2 points worse than the live rule's
MAX_EXTRA_TRADES = 0.25            # and it may not trade more than 25% more often

CANDIDATES = [
    {"id": "act-0.20", "params": {"act_threshold": 0.20},
     "reason": "Acting on smaller gaps between wanted and held exposure tracks the views more closely.",
     "cost": "More trades; each pays NSE statutory costs and spread."},
    {"id": "act-0.50", "params": {"act_threshold": 0.50},
     "reason": "Acting only on large gaps avoids paying costs to chase small changes in the views.",
     "cost": "Fewer trades; exposure drifts further from the target between trades."},
    {"id": "hold-5", "params": {"min_hold_days": 5},
     "reason": "A shorter wait lets the brain re-enter sooner after a fall (the known weakness).",
     "cost": "More trades and more whipsaw."},
    {"id": "hold-20", "params": {"min_hold_days": 20},
     "reason": "A longer wait cuts whipsaw trades, which cost money without adding information.",
     "cost": "Slower to exit when a fall starts."},
    {"id": "tolerance-0.10", "params": {"tolerance": 0.10},
     "reason": "Staying fully invested through mildly negative scores reduces time out of the market.",
     "cost": "Bigger falls when mild scores are the start of a crash."},
    {"id": "tolerance-0.20", "params": {"tolerance": 0.20},
     "reason": "A wider version of the same: only strongly negative scores cut exposure.",
     "cost": "Bigger falls; this is the direction that gives up the brain's one proven benefit."},
    {"id": "floor-0.25", "params": {"floor": 0.25},
     "reason": "Never being fully out keeps some of every rebound.",
     "cost": "A quarter of the book rides every fall."},
    {"id": "floor-0.50", "params": {"floor": 0.50},
     "reason": "Half-invested at least: most of holding's return with some of the protection.",
     "cost": "Half the book rides every fall."},
    {"id": "learn-0.02", "params": {"learning_rate": 0.02},
     "reason": "Slower trust updates are less affected by noisy 20-day judgements.",
     "cost": "Slower to follow a real change in which advisor is best."},
    {"id": "learn-0.15", "params": {"learning_rate": 0.15},
     "reason": "Faster trust updates follow a changing best advisor sooner.",
     "cost": "Chases noise; Phase A found no learner beat equal trust."},
    {"id": "share-0.01", "params": {"fixed_share": 0.01},
     "reason": "A larger Fixed-Share pool keeps every advisor able to recover trust quickly.",
     "cost": "Pulls trust back toward equal, blunting what was learned."},
    {"id": "share-0.05", "params": {"fixed_share": 0.05},
     "reason": "A much larger pool: trust stays close to equal, a hedge against learning the wrong thing.",
     "cost": "Almost equal trust, i.e. learning is largely switched off."},
]


def quarter(day: datetime | None = None) -> str:
    day = day or datetime.now(timezone.utc)
    return f"{day.year}Q{(day.month - 1) // 3 + 1}"


def stationary_indices(n: int, rng: np.random.Generator, block: int = BLOCK) -> np.ndarray:
    """Politis-Romano stationary bootstrap: blocks of geometric length (mean `block`), wrapping around."""
    out = np.empty(n, dtype=int)
    i, pos = 0, int(rng.integers(n))
    p = 1.0 / block
    while i < n:
        out[i] = pos
        i += 1
        pos = int(rng.integers(n)) if rng.random() < p else (pos + 1) % n
    return out


def reality_check(diffs: np.ndarray, seed: int = 0, draws: int = DRAWS) -> dict:
    """White's Reality Check. `diffs` is (days, candidates): each candidate's daily return minus the live rule's.
    Statistic: the largest mean difference (scaled). Null: no candidate has an edge, so each series is re-centred to
    mean zero and resampled. p = share of resamples whose best candidate beats the observed best."""
    n, k = diffs.shape
    mean = diffs.mean(axis=0)
    centred = diffs - mean
    observed = float(np.sqrt(n) * mean.max())
    rng = np.random.default_rng(seed)
    best = np.empty(draws)
    for d in range(draws):
        rows = stationary_indices(n, rng)
        best[d] = np.sqrt(n) * centred[rows].mean(axis=0).max()
    return {"p": float((best >= observed).mean()), "observed": observed, "best_index": int(mean.argmax()),
            "luck_only_median": float(np.median(best)), "luck_only_95": float(np.percentile(best, 95)),
            "candidates": int(k), "days": int(n)}


def criteria_for(candidate: dict) -> list[dict]:
    return [{"metric": "sharpe_vs_baseline", "op": ">", "value": 0.0, "label": "higher Sharpe than the live rule, 2007-2019"},
            {"metric": "drawdown_vs_baseline", "op": ">=", "value": -MAX_DRAWDOWN_WORSE,
             "label": "worst fall no more than 2 points worse"},
            {"metric": "extra_trades", "op": "<=", "value": MAX_EXTRA_TRADES, "label": "no more than 25% more trades"},
            {"metric": "family_reality_check_p", "op": "<", "value": ALPHA,
             "label": "the best of the whole family beats chance (White's Reality Check)"}]


def budget_state(records: list[dict]) -> dict:
    """The evidence budget as the holdout stands. Searching on the holdout is allowed only if the offered level is
    meaningful; otherwise the holdout is sealed (this is the case today, and a search never needs the holdout)."""
    rows = review.budget_ledger(records)
    wealth = rows[-1]["wealth_after"] if rows else 0.10
    level = review.rstats.offered_level(wealth)
    return {"wealth": wealth, "level": level, "holdout_open": level >= 0.01}


def _events(path: Path | None = None) -> list[dict]:
    path = path or GENERATED_LOG
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(event: dict, path: Path | None = None) -> None:
    path = path or GENERATED_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def already_run(batch: str, path: Path | None = None) -> bool:
    return any(e.get("event") == "results" and e.get("batch") == batch for e in _events(path))


def run(engine, records: list[dict], batch: str | None = None, candidates: list[dict] | None = None,
        log: Path | None = None, result: Path | None = None, seed: int = 0) -> dict:
    """Registers a batch, runs every candidate on the tuning years, and judges the family. `engine` is a
    `research.engine.Engine`; the holdout is never opened."""
    from scaata.agent import backtest as bt

    batch, candidates = batch or quarter(), candidates or CANDIDATES
    if already_run(batch, log):
        raise RuntimeError(f"batch {batch} was already run; one batch per quarter")
    budget = budget_state(records)
    registered = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _append({"event": "register", "batch": batch, "at": registered, "span": list(SPAN), "holdout_sealed": True,
             "budget": budget, "candidates": [{**c, "criteria": criteria_for(c)} for c in candidates]}, log)

    start, end = SPAN
    base_run = engine._run("tune", engine.views)
    base = bt.evaluate(base_run)
    base_ret = np.diff(base_run["curve"]) / base_run["curve"][:-1]
    rows, diffs = [], []
    for c in candidates:
        r = engine._run("tune", engine.views, **c["params"])
        s = bt.evaluate(r)
        ret = np.diff(r["curve"]) / r["curve"][:-1]
        diffs.append(ret - base_ret)
        rows.append({"id": c["id"], "params": c["params"], "reason": c["reason"], "cost": c["cost"],
                     "sharpe": s["hansei"]["sharpe"], "cagr": s["hansei"]["cagr"], "max_drawdown": s["hansei"]["max_drawdown"],
                     "trades": s["trades"], "costs": s["costs"],
                     "sharpe_vs_baseline": s["hansei"]["sharpe"] - base["hansei"]["sharpe"],
                     "cagr_vs_baseline": s["hansei"]["cagr"] - base["hansei"]["cagr"],
                     "drawdown_vs_baseline": s["hansei"]["max_drawdown"] - base["hansei"]["max_drawdown"],
                     "extra_trades": s["trades"] / max(base["trades"], 1) - 1,
                     "took_effect": bool(not np.allclose(r["curve"], base_run["curve"]))})
    rc = reality_check(np.column_stack(diffs), seed=seed)
    for row in rows:
        row["family_reality_check_p"] = rc["p"]
        row["criteria_met"] = [bool(review.OPS[k["op"]](row[k["metric"]], k["value"])) for k in criteria_for(row)]
        row["passes"] = all(row["criteria_met"]) and row["took_effect"]
    champions = sorted((r for r in rows if r["passes"]), key=lambda r: -r["sharpe_vs_baseline"])[:1]
    out = {"batch": batch, "registered_at": registered, "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "span": list(SPAN), "holdout_sealed": True, "budget": budget, "tried": len(rows),
           "baseline": {"sharpe": base["hansei"]["sharpe"], "cagr": base["hansei"]["cagr"],
                        "max_drawdown": base["hansei"]["max_drawdown"], "trades": base["trades"]},
           "reality_check": rc, "best_by_sharpe": max(rows, key=lambda r: r["sharpe_vs_baseline"])["id"],
           "candidates": rows, "champion": champions[0]["id"] if champions else None,
           "data": {k: engine.meta.get(k) for k in ("last", "sha256")}}
    _append({"event": "results", "batch": batch, "at": out["ran_at"], "champion": out["champion"],
             "reality_check_p": rc["p"], "tried": len(rows)}, log)
    path = result or GENERATED_RESULT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
