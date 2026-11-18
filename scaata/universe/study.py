"""The universe study: did a model that learned from scratch beat the plain average of the features?

    python -m scaata.universe study      ->  research/universe_study.json

Pre-registered before it was run (and unchanged since):

  * Primary test (goal G2): mean weekly rank correlation of the learned ridge model's scores with the next
    20 sessions' relative returns, minus the same for the equal-weight average of the signed features. The model
    "learned something" if that paired difference is positive with a block-bootstrap p below 0.05, on 2010-2019
    and on 2020 onward, and shows its learning by choosing coefficient signs the literature agrees with
    without having been told them.
  * A gradient-boosted tree model is used only if it beats ridge under the same test. Otherwise ridge stays.
  * Everything is walk-forward: a week's score comes from a model fitted only on weeks whose outcomes had
    finished before that week.

Caveat printed with every result: the stock list is today's large and mid caps (survivorship bias).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scaata.config import ROOT_DIR
from scaata.universe import data, features as F, model as M

STUDY_PATH = ROOT_DIR / "research" / "universe_study.json"
SPANS = {"2010-2019": ("2010-01-01", "2019-12-31"), "2020 onward": ("2020-01-01", None)}


def _mask(days, span: str) -> np.ndarray:
    start, end = SPANS[span]
    m = days >= np.datetime64(start)
    return m & (days <= np.datetime64(end)) if end else m


def _gbt_walk_forward(panel: dict, first_test: str = M.FIRST_TEST) -> dict | None:
    """Gradient-boosted trees under the same walk-forward rules, refit yearly. None if scikit-learn is absent."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return None
    days = panel["days"]
    start = int(np.searchsorted(days, np.datetime64(first_test)))
    scores = np.full(panel["y"].shape, np.nan)
    gbt = None
    for w in range(start, len(days)):
        if gbt is None or (w - start) % 52 == 0:
            xs, ys = M._training_rows(panel, w - M.PURGE_WEEKS)
            if len(xs) > 2000:
                gbt = HistGradientBoostingRegressor(max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=500,
                                                    random_state=0).fit(xs, ys)
        ok = panel["usable"][w]
        if gbt is not None and ok.any():
            scores[w, ok] = gbt.predict(panel["x"][w][ok])
    return {"scores": scores, "fits": [], "kind": "gbt", "start": start}


def run(path: Path | None = None) -> dict:
    closes, meta = data.load()
    panel = F.build_panel(closes)
    runs = {k: M.walk_forward(panel, k) for k in ("ridge", "equal", "mom_12_1")}
    gbt = _gbt_walk_forward(panel)
    if gbt:
        runs["gbt"] = gbt
    ic = {k: M.weekly_ic(r["scores"], panel["rank"]) for k, r in runs.items()}
    spreads = {k: M.spread(r["scores"], panel["forward"]) for k, r in runs.items()}
    days = panel["days"]
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prices_sha256": meta.get("sha256"),
           "prices_through": meta.get("last"), "stocks": meta.get("stocks"), "bias": meta.get("bias"),
           "weeks_scored": int(np.isfinite(ic["ridge"]).sum()), "first_test": M.FIRST_TEST,
           "ridge_penalty": M.RIDGE, "refit_every_weeks": M.REFIT_EVERY, "features": list(F.FEATURES),
           "assumed_signs": F.SIGN, "ic": {}, "versus": {}, "spreads": {}}
    for span in ("all", *SPANS):
        sel = np.ones(len(days), bool) if span == "all" else _mask(days, span)
        out["ic"][span] = {k: M.block_bootstrap(np.where(sel, v, np.nan)) for k, v in ic.items()}
        out["versus"][span] = {}
        for rival in ("equal", "mom_12_1") + (("ridge",) if "gbt" in runs else ()):
            for k in ("ridge", "gbt"):
                if k not in runs or k == rival:
                    continue
                out["versus"][span][f"{k}_minus_{rival}"] = M.block_bootstrap(np.where(sel, ic[k] - ic[rival], np.nan))
        out["spreads"][span] = {}
        for k, s in spreads.items():
            top, bottom, allr = (np.where(sel, s[n], np.nan) for n in ("top", "bottom", "all"))
            out["spreads"][span][k] = {"top_minus_all": M.block_bootstrap(top - allr), "all_minus_bottom": M.block_bootstrap(allr - bottom),
                                      "top_minus_bottom": M.block_bootstrap(top - bottom)}
    fits = runs["ridge"]["fits"]
    out["coefficients"] = {"final": fits[-1] if fits else None, "first": fits[0] if fits else None,
                           "path": [{"day": f["day"], **f["coef"]} for f in fits[::4]]}
    out["coefficient_signs_agree_with_literature"] = (
        {f: bool(np.sign(fits[-1]["coef"][f]) == np.sign(F.SIGN[f])) for f in F.FEATURES} if fits else {})
    out["weekly"] = {"days": [str(d.date()) for d in days[runs["ridge"]["start"]:]],
                     **{f"ic_{k}": [None if not np.isfinite(x) else round(float(x), 4) for x in v[runs["ridge"]["start"]:]]
                        for k, v in ic.items()}}
    path = path or STUDY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
