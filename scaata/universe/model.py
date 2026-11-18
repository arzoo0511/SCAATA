"""A model that learns from scratch, and the test that decides whether it learned anything.

Learning: ridge regression of each stock's ranked 20-session relative return on its eight features, refit every
quarter on every week old enough to be fully known. A week is used for training only if its whole 20-session
outcome had finished before the day being predicted (the "purge"), so no outcome leaks into a prediction.
Nothing is told the sign of any feature. The penalty is fixed in advance (RIDGE).

The test: each week, the rank correlation (information coefficient, IC) between the model's scores and what the
stocks then did, across ~200 stocks. The model has to beat the baseline that simply averages the signed features
(`features.equal_weight_score`). The comparison is the paired weekly difference in IC, with a block bootstrap over
runs of four weeks, because outcomes 20 sessions long overlap.
"""
from __future__ import annotations

import numpy as np

from scaata.universe import features as F

RIDGE = 100.0            # fixed in advance; the standardised features make this a mild penalty at ~60,000 rows
REFIT_EVERY = 13         # weeks
FIRST_TEST = "2010-01-01"
BLOCK_WEEKS = 4
DRAWS = 2000
PURGE_WEEKS = int(np.ceil(F.HORIZON / F.STEP)) + 1


def ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float = RIDGE) -> np.ndarray:
    """Coefficients (plus an intercept in the last slot) of ridge regression by the normal equations."""
    design = np.column_stack([x, np.ones(len(x))])
    reg = penalty * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + reg, design.T @ y)


def ridge_predict(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ coef[:-1] + coef[-1]


def _training_rows(panel: dict, last_day: int) -> tuple[np.ndarray, np.ndarray]:
    """Every usable (stock, week) whose outcome was complete before day index `last_day` of the weekly grid."""
    usable = panel["usable"][:last_day] & np.isfinite(panel["rank"][:last_day])
    return panel["x"][:last_day][usable], panel["rank"][:last_day][usable]


def walk_forward(panel: dict, kind: str = "ridge", first_test: str = FIRST_TEST) -> dict:
    """Out-of-sample scores for every week from `first_test` on. `kind`: "ridge" (learned), "equal" (the baseline
    average of signed features) or "mom_12_1" (plain momentum, for reference)."""
    days = panel["days"]
    start = int(np.searchsorted(days, np.datetime64(first_test)))
    scores = np.full(panel["y"].shape, np.nan)
    coefs, fits = [], []
    coef = None
    for w in range(start, len(days)):
        if kind == "ridge" and (coef is None or (w - start) % REFIT_EVERY == 0):
            xs, ys = _training_rows(panel, w - PURGE_WEEKS)
            if len(xs) > 2000:
                coef = ridge_fit(xs, ys)
                fits.append({"day": str(days[w].date()), "rows": int(len(xs)),
                             "coef": {f: float(c) for f, c in zip(F.FEATURES, coef[:-1])}})
        ok = panel["usable"][w]
        if not ok.any():
            continue
        if kind == "ridge":
            if coef is None:
                continue
            scores[w, ok] = ridge_predict(coef, panel["x"][w][ok])
        elif kind == "equal":
            scores[w, ok] = F.equal_weight_score(panel["x"][w][ok][None])[0]
        elif kind == "mom_12_1":
            scores[w, ok] = panel["x"][w][ok][:, F.FEATURES.index("mom_12_1")]
        else:
            raise ValueError(kind)
    return {"scores": scores, "fits": fits, "kind": kind, "start": start}


def weekly_ic(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rank correlation between scores and outcomes, one number per week (NaN where fewer than 30 stocks)."""
    out = np.full(len(scores), np.nan)
    for w in range(len(scores)):
        ok = np.isfinite(scores[w]) & np.isfinite(target[w])
        if ok.sum() < F.MIN_STOCKS:
            continue
        a = scores[w][ok].argsort().argsort().astype(float)
        b = target[w][ok].argsort().argsort().astype(float)
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        out[w] = (a * b).sum() / denom if denom > 0 else np.nan
    return out


def block_bootstrap(series: np.ndarray, seed: int = 0) -> dict:
    """Mean, 95% interval and one-sided p (that the mean is not above zero) over blocks of consecutive weeks."""
    x = series[np.isfinite(series)]
    n = len(x)
    if n < 3 * BLOCK_WEEKS:
        return {"mean": float(x.mean()) if n else None, "low": None, "high": None, "p": None, "weeks": int(n)}
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / BLOCK_WEEKS))
    starts = rng.integers(0, n - BLOCK_WEEKS + 1, (DRAWS, blocks))
    means = np.array([np.concatenate([x[s:s + BLOCK_WEEKS] for s in row])[:n].mean() for row in starts])
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(x.mean()), "low": float(low), "high": float(high), "p": float((means <= 0).mean()), "weeks": int(n)}


def spread(scores: np.ndarray, forward: np.ndarray, fraction: float = 0.2) -> dict:
    """Average 20-session return of the top fifth by score, the bottom fifth and everything, per week (gross)."""
    top, bottom, everything = [], [], []
    for w in range(len(scores)):
        ok = np.isfinite(scores[w]) & np.isfinite(forward[w])
        if ok.sum() < F.MIN_STOCKS:
            top.append(np.nan); bottom.append(np.nan); everything.append(np.nan)
            continue
        s, r = scores[w][ok], forward[w][ok]
        k = max(1, int(len(s) * fraction))
        order = s.argsort()
        top.append(r[order[-k:]].mean()); bottom.append(r[order[:k]].mean()); everything.append(r.mean())
    return {"top": np.array(top), "bottom": np.array(bottom), "all": np.array(everything)}
