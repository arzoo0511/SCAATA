"""The wide paper book: four portfolios over ~200 stocks, run forward only, judged by a live evidence meter.

Fake money, real prices, no broker. Separate from the five-stock agent in every way: its own state file
(`paper_book_wide.json`), its own data download, no shared memory.

Sleeves (all long-only, equal-weighted, rebalanced monthly):

    learned     the fifth of stocks the from-scratch ridge model ranks highest
    features    the fifth ranked highest by the plain average of the signed features (the baseline to beat)
    momentum    the fifth with the best 12-1 month momentum (the simplest signal that works)
    universe    every stock, equal-weighted (the benchmark all three are judged against)

Timing, using closes only: on the first session of a month the sleeves decide from closes up to and including
that session; the new weights take effect at the *next* session's close, with NSE costs on the turnover. So no
decision ever trades at the close that produced it, and nothing is back-filled: the book starts on the day it is
first run.

Every week it also logs each method's score for every stock; 20 sessions later it records the rank correlation
of those scores with what the stocks did. That is the forward test of whether the model learned anything.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scaata.config import INDIA_TICKERS, ROOT_DIR
from scaata.live.paper_book import COST_PER_SIDE
from scaata.universe import features as F, model as M

BOOK_PATH = ROOT_DIR / "paper_book_wide.json"
CAPITAL = 1_000_000.0
FRACTION = 0.2
SLEEVES = ("learned", "features", "momentum", "universe")
KEEP_SIGNALS = 60          # weekly score logs kept (about a year)


def _new_state(session: str) -> dict:
    return {"currency": "INR", "initial": CAPITAL, "started": session, "last_session": session, "pending": None,
            "coef": None, "coef_day": None, "model_rows": 0,
            "sleeves": {n: {"equity": CAPITAL, "weights": {}, "history": [], "rebalances": []} for n in SLEEVES},
            "signals": [], "forward_ic": [], "five_stock": {}}


def load(path: Path | None = None) -> dict | None:
    path = path or BOOK_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _decision_inputs(closes: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Standardised features on the last row for every stock that can be scored (features exist, price exists)."""
    raw = {k: F.standardise(v) for k, v in F._raw_features(closes).items()}
    last = np.stack([raw[f].iloc[-1].values for f in F.FEATURES], axis=1)
    ok = np.isfinite(last).all(axis=1) & closes.iloc[-1].notna().values
    return last[ok], list(closes.columns[ok])


def _scores(closes: pd.DataFrame, coef: np.ndarray | None) -> dict[str, dict[str, float]]:
    x, names = _decision_inputs(closes)
    out = {"features": dict(zip(names, F.equal_weight_score(x[None])[0])),
           "momentum": dict(zip(names, x[:, F.FEATURES.index("mom_12_1")]))}
    if coef is not None:
        out["learned"] = dict(zip(names, M.ridge_predict(coef, x)))
    return out


def _fit(closes: pd.DataFrame) -> tuple[np.ndarray | None, int]:
    """Ridge fitted on every week whose outcome has finished, using closes only up to the last row."""
    panel = F.build_panel(closes)
    xs, ys = M._training_rows(panel, len(panel["days"]))
    return (M.ridge_fit(xs, ys), int(len(xs))) if len(xs) > 2000 else (None, int(len(xs)))


def _weights(scores: dict[str, float], fraction: float = FRACTION) -> dict[str, float]:
    k = max(1, int(round(len(scores) * fraction)))
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return {s: 1.0 / k for s in top}


def _target_weights(closes: pd.DataFrame, coef: np.ndarray | None) -> dict[str, dict[str, float]]:
    scores = _scores(closes, coef)
    names = list(scores["features"])
    out = {"features": _weights(scores["features"]), "momentum": _weights(scores["momentum"]),
           "universe": {s: 1.0 / len(names) for s in names}}
    if "learned" in scores:
        out["learned"] = _weights(scores["learned"])
    return out


def _step_sleeve(sleeve: dict, returns: pd.Series, day: str, pending: dict | None) -> None:
    """One session: earn the day's return on the weights held, then (if a rebalance is due) trade to the new
    weights at the close and pay costs on the turnover."""
    weights = sleeve["weights"]
    r = sum(w * float(returns.get(s, 0.0) if np.isfinite(returns.get(s, 0.0)) else 0.0) for s, w in weights.items())
    sleeve["equity"] *= 1 + r
    if weights and abs(1 + r) > 1e-12:
        sleeve["weights"] = {s: w * (1 + float(returns.get(s, 0.0) if np.isfinite(returns.get(s, 0.0)) else 0.0)) / (1 + r)
                             for s, w in weights.items()}
    cost = 0.0
    if pending is not None:
        old, new = sleeve["weights"], pending
        turnover = sum(abs(new.get(s, 0.0) - old.get(s, 0.0)) for s in set(old) | set(new))
        cost = turnover * COST_PER_SIDE
        sleeve["equity"] *= 1 - cost
        sleeve["weights"] = dict(new)
        sleeve["rebalances"].append({"day": day, "holdings": len(new), "turnover": round(turnover, 4)})
        sleeve["rebalances"] = sleeve["rebalances"][-24:]
    sleeve["history"].append({"date": day, "equity": round(sleeve["equity"], 2), "return": r - cost,
                              "holdings": len(sleeve["weights"])})


def advance(closes: pd.DataFrame | None = None, path: Path | None = None) -> dict:
    """Brings the book up to the latest session in `closes` (downloaded fresh when not given). Idempotent: sessions
    already processed are skipped. Returns a one-line summary."""
    if closes is None:
        from scaata.universe import data

        closes = data.download()
    path = path or BOOK_PATH
    closes = closes.dropna(how="all")
    if closes.empty:
        return {"status": "no prices"}
    state = load(path)
    sessions = closes.index
    latest = str(sessions[-1].date())
    if state is None:
        state = _new_state(latest)
        history = closes
        state["coef"], state["model_rows"] = (lambda c, n: (None if c is None else [float(v) for v in c], n))(*_fit(history))
        state["coef_day"] = latest
        state["pending"] = {n: w for n, w in _target_weights(history, None if state["coef"] is None
                                                             else np.array(state["coef"])).items()}
        _save(state, path)
        return {"status": "started", "session": latest, "pending": {n: len(w) for n, w in state["pending"].items()}}
    last = state["last_session"]
    todo = [s for s in sessions if str(s.date()) > last]
    for s in todo:
        day = str(s.date())
        pos = sessions.get_loc(s)
        returns = closes.iloc[pos] / closes.iloc[pos - 1] - 1
        for name, sleeve in state["sleeves"].items():
            if state["pending"] is not None and name in state["pending"]:
                _step_sleeve(sleeve, returns, day, state["pending"][name])
            else:
                _step_sleeve(sleeve, returns, day, None)
        state["pending"] = None
        history = closes.iloc[:pos + 1]
        first_of_month = pos > 0 and sessions[pos].month != sessions[pos - 1].month
        if first_of_month:
            coef, rows = _fit(history)
            state["coef"], state["model_rows"], state["coef_day"] = (None if coef is None else [float(v) for v in coef]), rows, day
            state["pending"] = _target_weights(history, coef)
        if pos % F.STEP == 0:
            _log_signals(state, history, day)
        state["last_session"] = day
    _forward_ic(state, closes)
    state["five_stock"] = _five_stock(closes, state)
    state["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _save(state, path)
    return {"status": "advanced" if todo else "up to date", "sessions": [str(s.date()) for s in todo], "last": state["last_session"]}


def _log_signals(state: dict, history: pd.DataFrame, day: str) -> None:
    coef = None if state["coef"] is None else np.array(state["coef"])
    scores = _scores(history, coef)
    state["signals"].append({"day": day, **{k: {s: round(float(v), 3) for s, v in d.items()} for k, d in scores.items()}})
    state["signals"] = state["signals"][-KEEP_SIGNALS:]


def _forward_ic(state: dict, closes: pd.DataFrame) -> None:
    """Rank correlation of each logged score set with the stocks' relative return over the 20 sessions that followed."""
    done = {r["day"] for r in state["forward_ic"]}
    sessions = closes.index
    for entry in state["signals"]:
        day = entry["day"]
        if day in done or pd.Timestamp(day) not in sessions:
            continue
        pos = sessions.get_loc(pd.Timestamp(day))
        if pos + F.HORIZON >= len(sessions):
            continue
        forward = closes.iloc[pos + F.HORIZON] / closes.iloc[pos] - 1
        row = {"day": day}
        for method in ("learned", "features", "momentum"):
            if method not in entry:
                continue
            names = [s for s in entry[method] if np.isfinite(forward.get(s, np.nan))]
            if len(names) < F.MIN_STOCKS:
                continue
            a = np.array([entry[method][s] for s in names]).argsort().argsort().astype(float)
            b = forward[names].values.argsort().argsort().astype(float)
            a, b = a - a.mean(), b - b.mean()
            row[method] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
        if len(row) > 1:
            state["forward_ic"].append(row)


def _five_stock(closes: pd.DataFrame, state: dict) -> dict:
    """What each method currently says about the five stocks the main agent trades (shown, never acted on)."""
    if state["coef"] is None:
        return {}
    scores = _scores(closes, np.array(state["coef"]))
    out = {}
    for name in ("learned", "features", "momentum"):
        ranked = sorted(scores[name], key=scores[name].get, reverse=True)
        out[name] = {s.replace(".NS", ""): {"rank": ranked.index(s) + 1, "of": len(ranked), "score": round(float(scores[name][s]), 3)}
                     for s in INDIA_TICKERS if s in scores[name]}
    return out


def _save(state: dict, path: Path) -> None:
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")
