"""Runs Phase A end to end on the frozen price snapshot and saves `research/phase_a.json`.

    python -m scaata.research phase-a

Takes a few minutes: the advisors' views are computed once for every day since 2007, then the skill
ledger, calibration, seven learners and the poison test (20 seeds) all run on those arrays. Nothing here
touches the live agent.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scaata.agent import backtest as bt
from scaata.config import ROOT_DIR
from scaata.research import learners as L, prices

PHASE_A_PATH = ROOT_DIR / "research" / "phase_a.json"


def _clean(obj):
    """JSON-safe: numpy numbers to Python, NaN to null."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run(path: Path | None = None, seeds: int = 20) -> dict:
    started = time.time()
    opens, closes, meta = prices.load()
    views = bt.precompute_views(closes, "2007-01-01")
    arrays = L.build_arrays(views, closes)
    study = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prices_sha256": meta.get("sha256"), "prices_through": meta.get("last"),
        "days": int(len(arrays["days"])), "first": str(arrays["days"][0].date()),
        "judged_days": int(np.isfinite(arrays["fwd"]).all(axis=1).sum()),
        "note": "The news advisor is silent in this history (no scored headlines exist before 2012 and the "
                "scorer could remember later years), so everything here is about trend, volatility and momentum.",
        "skill": L.skill_ledger(arrays),
        "calibration": {s: L.calibration(arrays, s) for s in ("equal blend", "trend", "volatility", "momentum")},
        "learners": L.compare_learners(arrays),
        "poison": {k: L.poison_test(arrays, seeds=seeds, kind=k) for k in ("live", "hedge")},
        "learner_labels": L.LEARNERS, "p_scale": L.P_SCALE, "block_days": L.BLOCK,
    }
    study["runtime_seconds"] = round(time.time() - started, 1)
    study = _clean(study)
    path = path or PHASE_A_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(study, indent=1, ensure_ascii=False), encoding="utf-8")
    return study
