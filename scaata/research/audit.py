"""How much is each Lab result worth? Tables built from the registry alone.

- `dsr_table`: every experiment's Sharpe, discounted for the number of variants tried in the same span.
- `pbo_table`: how often "pick whichever variant did best on part of the history" fails on the rest.
- `selection_meter`: the live record's daily evidence, for the paper book against a benchmark that has
  the same exposure, so that simply being less invested does not count as skill.

It reads records and the paper book only; it writes nothing and needs no scipy.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from scaata.research import stats as rstats

SPANS = ("tune", "holdout")
SPAN_NAMES = {"tune": "2007–2019", "holdout": "2020 onward"}


def _trials(records: list[dict], span: str) -> list[dict]:
    """One row per distinct variant run in `span`. Runs that failed review, and legacy results with no
    replay behind them, are not trials; two runs with the identical result are one trial."""
    seen, rows = set(), []
    for r in sorted((r for r in records if r.get("origin") == "lab" and r.get("results")),
                    key=lambda r: r.get("registered_at") or ""):
        if (r.get("review") or {}).get("verdict") == "failed_review":
            continue
        p = (r["results"].get("periods") or {}).get(span)
        if not p or r.get("kind") == "regime_study":
            continue
        key = round(p["variant"]["sharpe"], 6), round(p["variant"]["cagr"], 6)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": r["id"], "title": r["title"], "span": span, "period": p})
    return rows


def dsr_table(records: list[dict]) -> list[dict]:
    """For each distinct variant in each span: the chance its true Sharpe really exceeds *holding's*, after
    discounting for the variants tried (the Deflated Sharpe Ratio with holding as the benchmark). Daily-return
    skew and kurtosis are used where the run recorded them, and a normal distribution is assumed (and flagged)
    where it did not. Holding's own Sharpe is treated as known, which overstates the uncertainty a little
    (the two are strongly correlated), so these are cautious figures."""
    out = []
    for span in SPANS:
        trials = _trials(records, span)
        per_period = [t["period"]["variant"]["sharpe"] / np.sqrt(rstats.TRADING_DAYS) for t in trials]
        for t, sr in zip(trials, per_period):
            p = t["period"]
            moments = p.get("moments")
            result = rstats.deflated_sharpe(sr, p["sessions"], per_period,
                                            benchmark=p["holding"]["sharpe"] / np.sqrt(rstats.TRADING_DAYS),
                                            **({"skew": moments["skew"], "kurt": moments["kurt"]} if moments else {}))
            out.append({"id": t["id"], "title": t["title"], "span": span, "sessions": p["sessions"],
                        "tails": "measured" if moments else "assumed normal", **result})
    return out


def pbo_table(records: list[dict]) -> list[dict]:
    """Probability of backtest overfitting over the variants that have stored weekly curves."""
    out = []
    for span in SPANS:
        curves = {}
        for r in records:
            if r.get("origin") != "lab" or not r.get("results") or \
                    (r.get("review") or {}).get("verdict") == "failed_review":
                continue
            c = (r["results"].get("curves") or {}).get(span)
            if c and c.get("variant"):
                curves[tuple(np.round(c["variant"], 2))] = c["variant"]      # identical curves are one variant
        if len(curves) < 2:
            out.append({"span": span, "pbo": None, "variants": len(curves), "splits": 0, "reason": "fewer than two variants"})
            continue
        length = max(set(map(len, curves.values())), key=list(map(len, curves.values())).count)
        series = [np.array(v) for v in curves.values() if len(v) == length]
        matrix = np.column_stack([np.diff(v) / v[:-1] for v in series])
        out.append({"span": span, **rstats.pbo(matrix)})
    return out


# ------------------------------------------------------------- live meter
def selection_meter(book: dict | None, cash_rate=None) -> dict | None:
    """Daily evidence from the paper book that HANSEI differs from holding *by more than its exposure*.

    Each day HANSEI's return is compared with a benchmark that holds the same five stocks in the same
    proportion HANSEI was invested at the start of that day, and the rest in the liquid fund. Two
    differences are tested, each by its own e-process:

        selection  HANSEI's return minus the matched benchmark's
        downside   the matched benchmark's loss that day (zero on up days) minus HANSEI's

    Approximation: exposure is read at the previous close, but orders fill at the next open."""
    history = (book or {}).get("equity_history") or []
    if len(history) < 3:
        return None
    from scaata.live.paper_book import liquid_yield

    rate = cash_rate or (lambda year: liquid_yield(year) / rstats.TRADING_DAYS)
    equity = np.array([h["equity"] for h in history], dtype=float)
    hold = np.array([h["benchmark_equity"] for h in history], dtype=float)
    cash = np.array([h["cash"] for h in history], dtype=float)
    r_h, r_b = equity[1:] / equity[:-1] - 1, hold[1:] / hold[:-1] - 1
    exposure = np.clip(1 - cash[:-1] / equity[:-1], 0.0, 1.0)
    daily_cash = np.array([rate(int(h["date"][:4])) for h in history[1:]])
    matched = exposure * r_b + (1 - exposure) * daily_cash
    selection = r_h - matched
    downside = np.maximum(-matched, 0.0) - np.maximum(-r_h, 0.0)
    dates = [h["date"] for h in history[1:]]
    return {"dates": dates, "selection": selection, "downside": downside, "exposure": exposure,
            "e_selection": rstats.e_process(selection), "e_downside": rstats.e_process(downside),
            "threshold": rstats.THRESHOLD, "sessions": len(dates),
            "average_exposure": float(exposure.mean()),
            "total_selection": float(np.prod(1 + r_h) - np.prod(1 + matched)),
            "first": dates[0], "last": dates[-1]}
