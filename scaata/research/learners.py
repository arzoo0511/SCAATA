"""Phase A: can the system's own critique be shown to work?

Everything here runs on arrays, not on the live agent: the advisors' views
for every stock on every day since 2007, and what each stock then did over the
next 20 sessions against cash. That is 19 years of judged predictions where the
live record has 14 days, and an online learner's forecasts are out of sample by
construction (the weights used on a day were learned only from days at least 20
sessions earlier), so the whole history is a fair test of a learner whose
settings were fixed in advance.

What it answers:

    skill ledger    how good is each advisor's forecast, on a proper score
    calibration     does a stronger view mean a more reliable view
    learners        does any way of learning trust beat plain equal trust
    poison          does the learner demote an advisor that is wrong on purpose,
                    and refuse to promote one that is random

The learners reproduce the live learner exactly (a test checks it against
`scaata.agent.memory.learn`), so a result about "the live learner" is about the
real thing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.agent.advisors import ADVISORS
from scaata.agent.memory import HORIZON, LEARNING_RATE, MIN_WEIGHT, SHARE, SILENT
from scaata.live.paper_book import liquid_yield
from scaata.strategies.hedge import floor_weights_on_simplex

BLOCK = 20                      # days per resampling block: overlapping 20-session outcomes are not independent
DRAWS = 2000
SPANS = {"tune": ("2007-01-01", "2019-12-31"), "holdout": ("2020-01-01", None)}
DELAY_KAPPA = 240               # the shrinkage prior is worth one year of independent judgements
P_SCALE = 0.25                  # a view of +1 means 75% that the stock beats cash, 0 means 50%; fixed in advance


def _nanmean(a: np.ndarray, axis=None):
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a, axis=axis)


# ------------------------------------------------------------------ arrays
def build_arrays(views_by_day: dict, closes: pd.DataFrame) -> dict:
    """Views (days x stocks x advisors) and 20-session forward returns net of cash (days x stocks)."""
    days = pd.DatetimeIndex(sorted(views_by_day))
    stocks = list(closes.columns)
    v = np.zeros((len(days), len(stocks), len(ADVISORS)))
    for i, day in enumerate(days):
        for j, s in enumerate(stocks):
            for a, name in enumerate(ADVISORS):
                v[i, j, a] = views_by_day[day][s].get(name, 0.0)
    index = closes.index.get_indexer(days)
    values = closes.values
    fwd = np.full((len(days), len(stocks)), np.nan)
    ok = index + HORIZON < len(closes)
    fwd[ok] = values[index[ok] + HORIZON] / values[index[ok]] - 1
    hurdle = np.array([liquid_yield(d.year) * HORIZON / 252 for d in days])
    # rounded to four decimals, as the live memory stores it, so the learners see exactly what the live one saw
    return {"days": days, "views": v, "fwd": np.round(fwd - hurdle[:, None], 4), "advisors": list(ADVISORS),
            "stocks": stocks}


def span_mask(days: pd.DatetimeIndex, span: str) -> np.ndarray:
    start, end = SPANS[span]
    mask = days >= pd.Timestamp(start)
    return mask & (days <= pd.Timestamp(end)) if end else mask


# ---------------------------------------------------------------- learners
def _floor(weights: np.ndarray) -> np.ndarray:
    return np.asarray(floor_weights_on_simplex(weights, MIN_WEIGHT), dtype=float)


def _normalise(losses: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(losses), np.nanmax(losses)
    return np.zeros_like(losses) if (not np.isfinite(hi - lo) or hi - lo < 1e-12) else np.clip((losses - lo) / (hi - lo), 0, 1)


def _losses(views: np.ndarray, fwd: np.ndarray, sleeping: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-advisor loss for one judged day (-view x excess return, averaged over stocks) and who was awake.
    An advisor is awake for the day if it spoke on any stock. In `sleeping` mode it is averaged only over the
    stocks it spoke on, so silence on some stocks does not dilute it with zeros."""
    awake = (np.abs(views) >= SILENT).any(axis=0)
    raw = -views * fwd[:, None]
    if sleeping:
        spoke = np.abs(views) >= SILENT
        loss = np.where(spoke.any(axis=0), (raw * spoke).sum(axis=0) / np.maximum(spoke.sum(axis=0), 1), 0.0)
    else:
        loss = raw.mean(axis=0)
    return loss, awake


def _hedge_step(weights: np.ndarray, loss: np.ndarray, awake: np.ndarray, eta: float, share: float,
                round4: bool) -> np.ndarray:
    """One update, exactly as `memory.learn` does it: Hedge among the advisors that spoke, the others untouched,
    then the Fixed-Share pooling, then the trust floor."""
    if awake.sum() < 2:
        return weights
    mass = weights[awake].sum()
    sub = weights[awake] / mass
    logw = np.log(sub + 1e-300) - eta * _normalise(loss[awake])
    logw -= logw.max()
    new = np.exp(logw)
    merged = weights.copy()
    merged[awake] = new / new.sum() * mass
    if share > 0:
        merged = (1 - share) * merged + share * merged.sum() / len(merged)
    out = _floor(merged)
    return np.round(out, 4) if round4 else out


def _adahedge_step(state: dict, loss: np.ndarray, awake: np.ndarray) -> np.ndarray:
    """AdaHedge (de Rooij, van Erven, Grunwald and Koolen 2014): no learning rate to set. An advisor that did
    not speak is charged the average loss of those that did, so silence neither helps nor hurts it."""
    if awake.sum() < 2:
        return state["w"]
    norm = np.zeros_like(loss)
    norm[awake] = _normalise(loss[awake])
    norm[~awake] = norm[awake].mean()
    n = len(loss)
    eta = np.log(n) / state["gap"] if state["gap"] > 1e-9 else 50.0
    eta = min(eta, 50.0)
    w = state["w"]
    mix = -np.log(np.sum(w * np.exp(-eta * norm))) / eta        # mixability loss
    state["gap"] += float(w @ norm - mix)
    state["cum"] += norm
    eta = min(np.log(n) / state["gap"], 50.0) if state["gap"] > 1e-9 else 50.0
    z = -eta * (state["cum"] - state["cum"].min())
    new = np.exp(z) / np.exp(z).sum()
    state["w"] = _floor(new)
    return state["w"]


LEARNERS = {
    "equal": "Equal trust, never changed",
    "hedge": "Plain Hedge (rate 0.05)",
    "live": "The live learner (Hedge with Fixed-Share)",
    "sleeping": "Sleeping advisors: each judged only on the stocks it spoke about",
    "adahedge": "AdaHedge: no learning rate to set",
    "delay": "Delay-aware: 21 interleaved learners, rate from theory",
    "shrunk": "The live learner, shrunk toward equal until there is a year of evidence",
}


def run_learner(arrays: dict, kind: str, advisor_cols: list[int] | None = None) -> dict:
    """Trust weights used on each day (after learning from everything 20 sessions old or older), and a
    record of how hard they were shrunk. `advisor_cols` limits the pool (used by the poison test)."""
    v, f = arrays["views"], arrays["fwd"]
    if advisor_cols is not None:
        v = v[:, :, advisor_cols]
    k, _, n = v.shape
    w = np.full(n, 1 / n)
    path = np.zeros((k, n))
    shrink = np.ones(k)
    judged = 0
    ada = {"w": w.copy(), "gap": 0.0, "cum": np.zeros(n)}
    cohort = HORIZON + 1
    cohort_w = [np.full(n, 1 / n) for _ in range(cohort)]
    cohort_eta = float(np.sqrt(8 * np.log(n) / max(k / cohort, 1)))        # fixed from theory, not tuned
    for t in range(k):
        j = t - HORIZON                                    # the day whose outcome has just become known
        if kind != "equal" and j >= 0 and np.isfinite(f[j]).all():
            loss, awake = _losses(v[j], f[j], sleeping=(kind == "sleeping"))
            if kind == "hedge":
                w = _hedge_step(w, loss, awake, LEARNING_RATE, 0.0, True)
            elif kind in ("live", "shrunk"):
                w = _hedge_step(w, loss, awake, LEARNING_RATE, SHARE, True)
            elif kind == "sleeping":
                w = _hedge_step(w, loss, awake, LEARNING_RATE, SHARE, True)
            elif kind == "adahedge":
                w = _adahedge_step(ada, loss, awake)
            elif kind == "delay":
                r = j % cohort
                cohort_w[r] = _hedge_step(cohort_w[r], loss, awake, cohort_eta, 0.0, False)
            judged += awake.sum() >= 2
        if kind == "equal":
            path[t] = 1 / n
        elif kind == "delay":
            path[t] = cohort_w[t % cohort]
        elif kind == "shrunk":
            lam = judged / (judged + DELAY_KAPPA)
            path[t] = lam * w + (1 - lam) / n
            shrink[t] = lam
        else:
            path[t] = w if kind != "adahedge" else ada["w"]
    return {"weights": path, "shrink": shrink, "kind": kind}


# -------------------------------------------------------------- forecasts
def forecast_scores(arrays: dict, weights: np.ndarray, cols: list[int] | None = None) -> dict:
    """The combined view for each stock-day, turned into a probability that the stock beats cash over the
    next 20 sessions (p = 0.5 + 0.25 x view), and its Brier score against what happened."""
    v = arrays["views"] if cols is None else arrays["views"][:, :, cols]
    s = np.clip(np.einsum("tsa,ta->ts", v, weights), -1, 1)
    p = 0.5 + P_SCALE * s
    y = (arrays["fwd"] > 0).astype(float)
    valid = np.isfinite(arrays["fwd"]) & (arrays["fwd"] != 0)
    brier = np.where(valid, (p - y) ** 2, np.nan)
    return {"score": s, "p": p, "brier_day": _nanmean(brier, axis=1), "valid": valid}


def payoff_day(arrays: dict, weights: np.ndarray) -> np.ndarray:
    """What following the combined view's sign would have earned each day (average over stocks, zero where
    the view is silent): long when it is positive, short when negative, over the next 20 sessions against
    cash. Unlike the Brier score it does not care how *large* the view is, only which way it points."""
    s = np.einsum("tsa,ta->ts", arrays["views"], weights)
    f = np.where(np.isfinite(arrays["fwd"]), arrays["fwd"], np.nan)
    pay = np.where(np.abs(s) >= SILENT, np.sign(s) * f, 0.0)
    pay[~np.isfinite(f).all(axis=1)] = np.nan
    return _nanmean(pay, axis=1)


def _block_bootstrap(series: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """(mean, 95% low, 95% high, one-sided p that the mean is not above zero) for a daily series, resampling
    blocks of consecutive days; NaN days are dropped first."""
    x = series[np.isfinite(series)]
    n = len(x)
    if n < 3 * BLOCK:
        return float(x.mean()) if n else float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, (DRAWS, blocks))
    means = np.array([np.concatenate([x[s:s + BLOCK] for s in row])[:n].mean() for row in starts])
    low, high = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(low), float(high), float((means <= 0).mean())


def compare_learners(arrays: dict, kinds: list[str] | None = None) -> list[dict]:
    """Every learner against equal trust, on two scores of the combined forecast, over the whole history and
    each span, with block-bootstrap intervals (resampling runs of consecutive days).

    `brier`  the proper score of the forecast (positive = better than equal trust). This is the pre-registered
             test. It punishes a forecast for being more confident than it is entitled to be.
    `payoff` what following the combined view's sign would have earned (positive = better than equal trust).
             It ignores how confident the view is, so it separates "picks the right direction" from
             "is over-confident".

    Each p-value is corrected for the number of learners compared (Bonferroni)."""
    kinds = kinds or [k for k in LEARNERS if k != "equal"]
    equal = run_learner(arrays, "equal")["weights"]
    base_brier = forecast_scores(arrays, equal)["brier_day"]
    base_pay = payoff_day(arrays, equal)
    rows = []
    for kind in kinds:
        run = run_learner(arrays, kind)
        diffs = {"brier": base_brier - forecast_scores(arrays, run["weights"])["brier_day"],
                 "payoff": payoff_day(arrays, run["weights"]) - base_pay}
        row = {"kind": kind, "label": LEARNERS[kind], "final_weights": [round(float(x), 3) for x in run["weights"][-1]],
               "mean_shrink": float(run["shrink"].mean())}
        for metric, diff in diffs.items():
            cell = {}
            for name in ("all", *SPANS):
                part = diff if name == "all" else np.where(span_mask(arrays["days"], name), diff, np.nan)
                mean, low, high, p = _block_bootstrap(part)
                cell[name] = {"improvement": mean, "low": low, "high": high, "p": p,
                              "p_corrected": min(1.0, p * len(kinds)) if np.isfinite(p) else None}
            row[metric] = cell
        row["all"] = row["brier"]["all"]            # the pre-registered test, kept at the top level
        rows.append(row)
    return rows


# ------------------------------------------------------------ skill ledger
def skill_ledger(arrays: dict) -> list[dict]:
    """How good each advisor's own forecast is: Brier skill against a coin flip (positive = better), the share
    of its opinions that pointed the right way, and the rank correlation between its views and outcomes."""
    out = []
    y = (arrays["fwd"] > 0).astype(float)
    valid = np.isfinite(arrays["fwd"]) & (arrays["fwd"] != 0)
    sources = {name: arrays["views"][:, :, a] for a, name in enumerate(arrays["advisors"])}
    sources["equal blend"] = arrays["views"].mean(axis=2)
    for name, view in sources.items():
        for span in ("all", *SPANS):
            days = np.ones(len(arrays["days"]), bool) if span == "all" else span_mask(arrays["days"], span)
            spoke = valid & (np.abs(view) >= SILENT) & days[:, None]
            if spoke.sum() < 30:
                out.append({"source": name, "span": span, "opinions": int(spoke.sum()), "skill": None})
                continue
            p = 0.5 + P_SCALE * np.clip(view, -1, 1)
            daily = np.where(spoke, 1 - (p - y) ** 2 / 0.25, np.nan)           # Brier skill per stock-day
            right = np.where(spoke, ((view > 0) == (arrays["fwd"] > 0)).astype(float), np.nan)
            mean, low, high, _ = _block_bootstrap(_nanmean(daily, axis=1))
            hit, hlow, hhigh, _ = _block_bootstrap(_nanmean(right, axis=1) - 0.5)
            out.append({"source": name, "span": span, "opinions": int(spoke.sum()), "skill": mean, "low": low,
                        "high": high, "hit_rate": hit + 0.5, "hit_low": hlow + 0.5, "hit_high": hhigh + 0.5})
    return out


# ------------------------------------------------------------- calibration
def isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotone (non-decreasing) fit of y on x by pooling adjacent violators. Returns the fitted levels and
    the x value each level ends at, to look new values up."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    level, weight, end = [], [], []
    for i in range(len(xs)):
        level.append(ys[i]); weight.append(1.0); end.append(xs[i])
        while len(level) > 1 and level[-2] > level[-1]:
            total = weight[-2] + weight[-1]
            level[-2] = (level[-2] * weight[-2] + level[-1] * weight[-1]) / total
            weight[-2], end[-2] = total, end[-1]
            level.pop(); weight.pop(); end.pop()
    return np.array(level), np.array(end)


def apply_isotonic(fit: tuple[np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    level, end = fit
    return level[np.minimum(np.searchsorted(end, x, side="left"), len(level) - 1)]


def calibration(arrays: dict, source: str = "equal blend") -> dict:
    """Is a stronger view a more reliable one, and does recalibrating on 2007-2019 help on 2020 onward?

    Reliability: the combined view is cut into ten equal-sized groups; for each, the probability the view
    implied against how often the stock really beat cash. Strong-versus-weak: among views that pointed
    one way, how often the stronger half was right against the weaker half, per span.
    Recalibration: an isotonic map fitted on the tune years only, scored on the holdout years."""
    view = arrays["views"].mean(axis=2) if source == "equal blend" else arrays["views"][:, :, arrays["advisors"].index(source)]
    y = (arrays["fwd"] > 0).astype(float)
    valid = np.isfinite(arrays["fwd"]) & (arrays["fwd"] != 0)
    p = 0.5 + P_SCALE * np.clip(view, -1, 1)
    out = {"source": source, "reliability": {}, "strong_vs_weak": {}}
    tune = valid & span_mask(arrays["days"], "tune")[:, None]
    hold = valid & span_mask(arrays["days"], "holdout")[:, None]
    for name, mask in (("tune", tune), ("holdout", hold)):
        x, o = p[mask], y[mask]
        if len(x) < 30:
            out["reliability"][name] = []
            continue
        edges = np.quantile(x, np.linspace(0, 1, 11))
        bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
        out["reliability"][name] = [{"predicted": float(x[bins == b].mean()), "observed": float(o[bins == b].mean()),
                                     "n": int((bins == b).sum())} for b in range(10) if (bins == b).any()]
    spoke_tune = tune & (np.abs(view) >= SILENT)
    if spoke_tune.sum() < 30:
        out["recalibration"] = None
        return out
    cut = np.median(np.abs(view[spoke_tune]))
    for name, mask in (("tune", tune), ("holdout", hold)):
        spoke = mask & (np.abs(view) >= SILENT)
        right = np.where(spoke, ((view > 0) == (arrays["fwd"] > 0)).astype(float), np.nan)
        weak = np.where(spoke & (np.abs(view) < cut), right, np.nan)
        strong = np.where(spoke & (np.abs(view) >= cut), right, np.nan)
        d_mean, d_low, d_high, _ = _block_bootstrap(_daily_gap(strong, weak))
        out["strong_vs_weak"][name] = {"weak_right": float(_nanmean(weak)), "strong_right": float(_nanmean(strong)),
                                       "gap": d_mean, "low": d_low, "high": d_high, "cut": float(cut),
                                       "n_weak": int(np.isfinite(weak).sum()), "n_strong": int(np.isfinite(strong).sum())}
    if tune.sum() < 100 or hold.sum() < 100:
        out["recalibration"] = None
        return out
    fit = isotonic(p[tune], y[tune])
    raw = (p[hold] - y[hold]) ** 2
    recal = (apply_isotonic(fit, p[hold]) - y[hold]) ** 2
    day_idx = np.nonzero(hold)[0]
    gap = np.full(len(arrays["days"]), np.nan)
    for t in np.unique(day_idx):
        sel = day_idx == t
        gap[t] = raw[sel].mean() - recal[sel].mean()
    mean, low, high, _ = _block_bootstrap(gap)
    out["recalibration"] = {"brier_raw": float(raw.mean()), "brier_recalibrated": float(recal.mean()),
                            "improvement": mean, "low": low, "high": high, "fitted_on": int(tune.sum())}
    return out


def _daily_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per day: share right among strong views minus among weak views (NaN where either has none)."""
    return _nanmean(a, axis=1) - _nanmean(b, axis=1)


# ------------------------------------------------------------------ poison
POISON_RULES = {
    "adversary": "Wrong on purpose (the opposite of what happened): trust must end within 2 points of the floor",
    "random": "Pure noise: trust must not end more than 5 points above an equal share",
    "informative": "Right 60% of the time: trust must end in the top two of eight",
    "weak_adversary": "Right only 40% of the time (a subtle failure, not an obvious one): trust must end below an equal share",
}
POISON_PASS = 0.95      # share of seeds in which a rule must hold; fixed before the test was run


def add_poison(arrays: dict, seed: int) -> dict:
    """The real advisors plus three planted ones, built with knowledge of the future. They exist only to test
    the learner; nothing here can reach the live agent."""
    rng = np.random.default_rng(seed)
    f = arrays["fwd"]
    truth = np.where(np.isfinite(f), np.sign(f), 0.0)
    adversary = -0.8 * truth
    informative = 0.6 * truth * np.where(rng.random(f.shape) < 0.6, 1.0, -1.0)
    weak_adversary = 0.6 * truth * np.where(rng.random(f.shape) < 0.4, 1.0, -1.0)
    noise = rng.uniform(-0.6, 0.6, f.shape)
    views = np.concatenate([arrays["views"], adversary[:, :, None], noise[:, :, None], informative[:, :, None],
                            weak_adversary[:, :, None]], axis=2)
    return {**arrays, "views": views,
            "advisors": [*arrays["advisors"], "adversary", "random", "informative", "weak_adversary"]}


def poison_test(arrays: dict, seeds: int = 20, kind: str = "live") -> dict:
    """Does the learner demote the adversary, leave the noise alone, and favour the informative advisor?"""
    n = len(arrays["advisors"])
    equal = 1 / (n + 4)
    runs = []
    for seed in range(seeds):
        extended = add_poison(arrays, seed)
        final = run_learner(extended, kind)["weights"][-1]
        trust = dict(zip(extended["advisors"], final))
        ranked = sorted(trust, key=trust.get, reverse=True)
        runs.append({"seed": seed, "trust": {k: round(float(v), 4) for k, v in trust.items()},
                     "adversary_demoted": bool(trust["adversary"] <= MIN_WEIGHT + 0.02),
                     "random_not_promoted": bool(trust["random"] <= equal + 0.05),
                     "informative_top2": bool("informative" in ranked[:2]),
                     "weak_adversary_below_equal": bool(trust["weak_adversary"] < equal)})
    rates = {k: float(np.mean([r[k] for r in runs])) for k in
             ("adversary_demoted", "random_not_promoted", "informative_top2", "weak_adversary_below_equal")}
    mean = {k: float(np.mean([r["trust"][k] for r in runs])) for k in runs[0]["trust"]}
    return {"learner": kind, "seeds": seeds, "rates": rates, "mean_trust": mean, "required": POISON_PASS,
            "passes": {k: bool(v >= POISON_PASS) for k, v in rates.items()}, "equal_share": equal, "rules": POISON_RULES}
