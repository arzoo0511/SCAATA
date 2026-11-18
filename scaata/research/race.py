"""The promotion race: the frozen baseline, the live rule and challengers, side by side on the same prices.

    python -m scaata.research race        ->  research/race.json

Every entrant is replayed by the real brain (`backtest.simulate`) from the same start day, on the same freshly
downloaded prices, from equal trust. Each challenger is compared with the FROZEN baseline: the brain settings as
they stood on 16 Nov 2026, written out below so that a later change to the live code cannot move the yardstick.

Two races are reported:

    forward     starts on the day the baseline was frozen. This is the only one that is evidence.
    rehearsal   starts 1 Jan 2024. The years were seen during earlier tuning, so it proves the machinery works and
                shows what a race looks like; it is never used to promote anything.

The meter for each challenger is an anytime-valid e-process on the daily return difference to the baseline
(`stats.e_process`; 20 = 5% false-alarm rate however often it is read). Promotion needs ALL of:

    1. the e-process against the baseline above 20 (and at least MIN_SESSIONS sessions of forward data);
    2. the worst fall no more than 2 points worse than the baseline's;
    3. the poison test (Phase A) still passing for the live learner;
    4. the challenger is promotable: it is a registered hypothesis or a generator champion, not an exploratory entry.

Demotion is the mirror image: a promoted setting whose down-meter crosses 20 is reverted. Nothing here changes the
live agent; "promote" is a recommendation that still needs a person and a tagged commit.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scaata.agent import backtest as bt
from scaata.config import ROOT_DIR
from scaata.research import engine as E, stats as rstats

RACE_PATH = ROOT_DIR / "research" / "race.json"
PHASE_A_PATH = ROOT_DIR / "research" / "phase_a.json"
FORWARD_START = "2026-11-16"          # the day the baseline was frozen
REHEARSAL_START = "2024-01-01"
WARMUP_FROM = "2022-06-01"            # prices downloaded from here so the advisors have history on day one
THRESHOLD = rstats.THRESHOLD
MIN_SESSIONS = 60
MAX_DRAWDOWN_WORSE = 0.02

FROZEN = {"act_threshold": 1 / 3, "min_hold_days": 10, "learning_rate": 0.05, "fixed_share": round(1 / 252, 5)}
ENTRANTS = [
    {"id": "frozen", "label": "Frozen baseline (16 Nov 2026)", "params": dict(FROZEN), "view": None, "role": "baseline"},
    {"id": "live", "label": "Live rule (current code)", "params": {}, "view": None, "role": "live"},
    {"id": "rebound", "label": "Rebound rule (Phase B)", "params": dict(FROZEN), "view": "rebound", "role": "challenger",
     "promotable": True, "why": "A registered hypothesis; the Lab found it not supported over 19 years, so it needs forward proof."},
    {"id": "learn-0.02", "label": "Slower trust updates", "params": {**FROZEN, "learning_rate": 0.02}, "view": None,
     "role": "challenger", "promotable": False,
     "why": "Best of the self-proposed batch by Sharpe, but the family failed its Reality Check: exploratory only."},
    {"id": "share-0.05", "label": "Larger Fixed-Share pool", "params": {**FROZEN, "fixed_share": 0.05}, "view": None,
     "role": "challenger", "promotable": False,
     "why": "Second best of the self-proposed batch; exploratory only."},
]


def _poison_passes(path: Path | None = None) -> bool | None:
    path = path or PHASE_A_PATH
    if not path.exists():
        return None
    live = (json.loads(path.read_text(encoding="utf-8")).get("poison") or {}).get("live") or {}
    passes = live.get("passes")
    return bool(all(passes.values())) if passes else None


def promotion(row: dict, poison: bool | None, baseline: bool = False) -> dict:
    """Applies the rule to one challenger's race row."""
    checks = [
        {"name": "e-process above 20", "passed": bool(row["e_up"] >= THRESHOLD), "detail": f"{row['e_up']:.2f} (needs {THRESHOLD:g})"},
        {"name": f"at least {MIN_SESSIONS} sessions", "passed": bool(row["sessions"] >= MIN_SESSIONS), "detail": f"{row['sessions']} sessions"},
        {"name": "worst fall no more than 2 points worse", "passed": bool(row["drawdown_vs_baseline"] >= -MAX_DRAWDOWN_WORSE),
         "detail": f"{row['drawdown_vs_baseline']:+.1%}"},
        {"name": "poison test passing", "passed": poison, "detail": "Phase A: planted bad/random advisors demoted" if poison else "not available"},
        {"name": "promotable entry", "passed": bool(row.get("promotable")), "detail": row.get("why", "")},
    ]
    verdict = ("promote" if all(c["passed"] for c in checks) else
               "demote" if row["e_down"] >= THRESHOLD else "keep racing")
    return {"verdict": verdict, "checks": checks}


def _sessions_to_threshold(diffs: np.ndarray) -> int | None:
    """Rough sessions an e-process needs to reach 20 if the daily difference keeps this mean and spread: the best
    bet grows the log-wealth by about mean^2 / (2 variance) a session, and ln(20) is needed. None if the mean is
    not positive. An estimate from one noisy sample, shown only to say how long a verdict could take."""
    mean, var = float(np.mean(diffs)), float(np.var(diffs))
    return int(round(2 * np.log(THRESHOLD) * var / mean ** 2)) if mean > 0 and var > 0 else None


def _entrant_run(entrant: dict, opens, closes, views, start: str) -> dict:
    v = views
    if entrant["view"] == "rebound":
        v = E.rebound_transform(views, closes)
    return bt.simulate(opens, closes, v, start, None, no_borrowing=True, **entrant["params"])


def run_race(opens: pd.DataFrame, closes: pd.DataFrame, start: str, poison: bool | None, entrants: list[dict] | None = None) -> dict:
    entrants = entrants or ENTRANTS
    views = bt.precompute_views(closes, start)
    runs = {e["id"]: _entrant_run(e, opens, closes, views, start) for e in entrants}
    base = runs["frozen"]
    days = [str(d.date()) for d in base["days"]]
    base_ret = np.diff(base["curve"]) / base["curve"][:-1]
    base_stats = bt.evaluate(base)
    rows = []
    for e in entrants:
        r, curve = runs[e["id"]], runs[e["id"]]["curve"]
        s = bt.evaluate(r)
        ret = np.diff(curve) / curve[:-1]
        diffs = ret - base_ret
        ep = rstats.e_process(diffs)
        row = {"id": e["id"], "label": e["label"], "role": e["role"], "sessions": int(len(curve) - 1),
               "return": float(curve[-1] / curve[0] - 1), "sharpe": s["hansei"]["sharpe"],
               "max_drawdown": s["hansei"]["max_drawdown"], "trades": s["trades"],
               "drawdown_vs_baseline": s["hansei"]["max_drawdown"] - base_stats["hansei"]["max_drawdown"],
               "return_vs_baseline": float(curve[-1] / curve[0] - base["curve"][-1] / base["curve"][0]),
               "sessions_to_20": _sessions_to_threshold(diffs),
               "e_up": ep["up_final"], "e_down": ep["down_final"],
               "curve": np.round(curve / curve[0] * 100, 3).tolist(),
               "e_path_up": np.round(ep["up"], 4).tolist(), "e_path_down": np.round(ep["down"], 4).tolist(),
               "promotable": e.get("promotable", False), "why": e.get("why", "")}
        if e["role"] == "challenger":
            row["promotion"] = promotion(row, poison)
        rows.append(row)
    return {"start": start, "end": days[-1], "days": days, "holding": np.round(np.array(base["bh_curve"]) / base["bh_curve"][0] * 100, 3).tolist(),
            "entrants": rows}


def run(path: Path | None = None) -> dict:
    opens, closes = bt.load_prices(date.today(), WARMUP_FROM)
    poison = _poison_passes()
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "threshold": THRESHOLD,
           "min_sessions": MIN_SESSIONS, "frozen_params": FROZEN, "poison_passes": poison,
           "prices_through": str(closes.index[-1].date()), "baseline_frozen_on": FORWARD_START}
    for name, start in (("forward", FORWARD_START), ("rehearsal", REHEARSAL_START)):
        out[name] = run_race(opens, closes, start, poison) if (closes.index >= pd.Timestamp(start)).sum() > 1 else None
    path = path or RACE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
