"""Runs experiments through HANSEI's real decision code.

Every experiment here calls `scaata.agent.backtest.simulate`, which calls the
same `brain.decide` and `memory.learn` the daily agent uses. There is no
simplified copy. Each kind of experiment changes exactly one thing and is
compared with the live rule over the same days:

    baseline          the live rule against simply holding
    brain_variant     different brain or memory settings
    advisor_ablation  one advisor made permanently silent
    regime_study      the live rule's edge split by market condition

All run on the frozen price snapshot (`research.prices`), on two fixed
spans: the years the rule was tuned on, and the years it was not.
"""
from __future__ import annotations

import subprocess
import time

import numpy as np
import pandas as pd

from scaata.agent import backtest as bt
from scaata.agent.advisors import ADVISORS
from scaata.config import INDIA_TICKERS, ROOT_DIR
from scaata.evaluation.stats import paired_block_bootstrap_prob_better
from scaata.research import prices

PERIODS = {"tune": ("2007-01-01", "2019-12-31"), "holdout": ("2020-01-01", None)}
KINDS = ("baseline", "brain_variant", "advisor_ablation", "regime_study")
BRAIN_KEYS = ("act_threshold", "min_hold_days", "learning_rate", "tolerance", "floor")
TREND_WINDOW, VOL_WINDOW, VOL_BASELINE = 126, 20, 252
TRUNCATION_SESSIONS = 260      # length of the stretch replayed for the look-ahead check
VIEW_SAMPLES = 16              # days on which each advisor's view is recomputed without the future


def _commit() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "scaata"))}
    except OSError:
        return {"commit": None, "dirty": None}


def silence(views_by_day: dict, advisor: str) -> dict:
    """The same views with one advisor saying nothing, every day."""
    return {t: {s: {**v, advisor: 0.0} for s, v in by_stock.items()} for t, by_stock in views_by_day.items()}


def _annualised(returns: np.ndarray) -> float:
    return float(np.prod(1 + returns) ** (252 / len(returns)) - 1) if len(returns) else float("nan")


def regimes(closes: pd.DataFrame) -> pd.DataFrame:
    """Each session's market condition, known at the previous close: is the equal-weight
    basket of the five stocks above where it was six months ago (trend up or down), and is
    it moving more than its own one-year norm (calm or choppy)."""
    basket = (closes / closes.iloc[0]).mean(axis=1)
    daily = basket.pct_change()
    recent = daily.rolling(VOL_WINDOW).std()
    frame = pd.DataFrame({"trend": np.where(basket / basket.shift(TREND_WINDOW) - 1 >= 0, "up", "down"),
                          "mood": np.where(recent > recent.rolling(VOL_BASELINE).median(), "choppy", "calm")},
                         index=closes.index)
    unknown = basket.shift(TREND_WINDOW).isna().values | recent.rolling(VOL_BASELINE).median().isna().values
    frame.loc[unknown, :] = None
    return frame.shift(1)      # yesterday's label decides which bucket today's return goes in


class Engine:
    """Holds the snapshot, the advisors' views and the live rule's runs, so a batch of
    experiments does not recompute them."""

    def __init__(self, opens: pd.DataFrame | None = None, closes: pd.DataFrame | None = None,
                 meta: dict | None = None, views_fn=bt.precompute_views, periods: dict | None = None) -> None:
        if opens is None or closes is None:
            opens, closes, meta = prices.load()
        self.opens, self.closes, self.meta = opens, closes, meta or {}
        self.periods = periods or PERIODS
        self.views_fn = views_fn
        self._views: dict | None = None
        self._baseline: dict[str, dict] = {}

    @property
    def views(self) -> dict:
        if self._views is None:
            self._views = self.views_fn(self.closes, min(start for start, _ in self.periods.values()))
        return self._views

    def _run(self, period: str, views: dict, **brain) -> dict:
        start, end = self.periods[period]
        return bt.simulate(self.opens, self.closes, views, start, end, **brain)

    def baseline(self, period: str) -> dict:
        if period not in self._baseline:
            self._baseline[period] = self._run(period, self.views)
        return self._baseline[period]

    # ------------------------------------------------------------ integrity
    def truncation_invariant(self, transform=None, **brain) -> dict:
        """The look-ahead test, in two parts.

        Views: on a spread of days, each advisor's view is recomputed with every price after
        that day deleted. It must equal the view the experiment actually used that day.

        Replay: a stretch is run twice, once with all of history available and once with
        every later price deleted. The two equity curves must be identical.

        If any decision used a later price, one of the two differs."""
        apply = transform or (lambda v: v)
        days = sorted(self.views)
        sample = [days[i] for i in np.linspace(0, len(days) - 1, min(VIEW_SAMPLES, len(days))).astype(int)]
        leaked = []
        for t in sample:
            alone = apply(self.views_fn(self.closes.loc[:t], str(t.date()))).get(t, {})
            used = apply({t: self.views[t]})[t]
            if any(abs(alone.get(s, {}).get(a, 0.0) - v) > 1e-9 for s, by in used.items() for a, v in by.items()):
                leaked.append(str(t.date()))

        start = pd.Timestamp(self.periods["holdout"][0] if "holdout" in self.periods
                             else next(iter(self.periods.values()))[0])
        stretch = self.closes.index[self.closes.index >= start][:TRUNCATION_SESSIONS]
        if len(stretch) < 30:
            return {"checked": False, "invariant": None, "sessions": len(stretch)}
        end = stretch[-1]
        full_views = apply({t: v for t, v in self.views.items() if start <= t <= end})
        cut_opens, cut_closes = self.opens.loc[:end], self.closes.loc[:end]
        cut_views = apply(self.views_fn(cut_closes, str(start.date())))
        full = bt.simulate(self.opens, self.closes, full_views, str(start.date()), str(end.date()), **brain)
        cut = bt.simulate(cut_opens, cut_closes, cut_views, str(start.date()), str(end.date()), **brain)
        same = len(full["curve"]) == len(cut["curve"]) and bool(np.allclose(full["curve"], cut["curve"], rtol=1e-9))
        return {"checked": True, "invariant": same and not leaked, "sessions": len(stretch),
                "days_sampled": len(sample), "days_with_leaked_views": leaked, "replay_identical": same}

    # --------------------------------------------------------------- kinds
    def run(self, spec: dict) -> dict:
        """Runs one registered experiment and returns its results."""
        kind, params = spec["kind"], spec.get("params", {})
        if kind not in KINDS:
            raise ValueError(f"unknown experiment kind '{kind}'")
        started = time.time()
        if kind == "regime_study":
            out = self._regime_study()
        else:
            brain = {k: params[k] for k in BRAIN_KEYS if k in params} if kind == "brain_variant" else {}
            transform = None
            if kind == "advisor_ablation":
                if params.get("drop") not in ADVISORS:
                    raise ValueError(f"advisor_ablation needs params.drop to be one of {ADVISORS}")
                transform = lambda views: silence(views, params["drop"])   # noqa: E731
            out = self._compare(kind, transform, brain)
        out["data"] = {k: self.meta.get(k) for k in ("source", "retrieved_utc", "first", "last", "sha256")}
        out["engine"] = {**_commit(), "runtime_seconds": round(time.time() - started, 1),
                         "periods": {k: list(v) for k, v in self.periods.items()}}
        return out

    def _compare(self, kind: str, transform, brain: dict) -> dict:
        views = transform(self.views) if transform else self.views
        metrics, periods, curves = {}, {}, {}
        for period in self.periods:
            base = self.baseline(period)
            run = base if kind == "baseline" else self._run(period, views, **brain)
            stats, base_stats = bt.evaluate(run), bt.evaluate(base)
            hold = stats["buy_and_hold"]
            periods[period] = {"start": stats["start"], "end": stats["end"], "sessions": len(run["days"]),
                               "variant": stats["hansei"], "baseline": base_stats["hansei"], "holding": hold,
                               "trades": stats["trades"], "costs": stats["costs"],
                               "baseline_trades": base_stats["trades"]}
            metrics[f"{period}.sharpe_vs_holding"] = stats["hansei"]["sharpe"] - hold["sharpe"]
            metrics[f"{period}.cagr_vs_holding"] = stats["hansei"]["cagr"] - hold["cagr"]
            metrics[f"{period}.drawdown_vs_holding"] = stats["hansei"]["max_drawdown"] - hold["max_drawdown"]
            metrics[f"{period}.prob_sharpe_beats_holding"] = float(
                paired_block_bootstrap_prob_better(run["curve"], run["bh_curve"]))
            if kind != "baseline":
                metrics[f"{period}.sharpe_vs_baseline"] = stats["hansei"]["sharpe"] - base_stats["hansei"]["sharpe"]
                metrics[f"{period}.cagr_vs_baseline"] = stats["hansei"]["cagr"] - base_stats["hansei"]["cagr"]
                metrics[f"{period}.drawdown_vs_baseline"] = (stats["hansei"]["max_drawdown"]
                                                             - base_stats["hansei"]["max_drawdown"])
                metrics[f"{period}.prob_sharpe_beats_baseline"] = float(
                    paired_block_bootstrap_prob_better(run["curve"], base["curve"]))
            step = max(1, len(run["days"]) // 260)       # about weekly: enough to draw, small enough to store
            curves[period] = {"dates": [d.strftime("%Y-%m-%d") for d in run["days"][::step]],
                              "variant": np.round(run["curve"][::step], 2).tolist(),
                              "baseline": np.round(base["curve"][::step], 2).tolist(),
                              "holding": np.round(run["bh_curve"][::step], 2).tolist()}
        against = "holding" if kind == "baseline" else "baseline"
        primary = {p: metrics[f"{p}.sharpe_vs_{against}"] for p in self.periods}
        prob = {p: metrics[f"{p}.prob_sharpe_beats_{against}"] for p in self.periods}
        return {"metrics": metrics, "periods": periods, "curves": curves,
                "primary": {"label": f"Sharpe against {'holding' if kind == 'baseline' else 'the live rule'}",
                            "values": primary, "prob_better": prob},
                "integrity": self.truncation_invariant(transform, **brain)}

    def _regime_study(self) -> dict:
        labels = regimes(self.closes)
        metrics, periods, table = {}, {}, []
        for period in self.periods:
            run = self.baseline(period)
            days = pd.DatetimeIndex(run["days"])
            agent = pd.Series(run["curve"], index=days).pct_change().iloc[1:]
            hold = pd.Series(run["bh_curve"], index=days).pct_change().iloc[1:]
            label = labels.reindex(agent.index)
            stats = bt.evaluate(run)
            periods[period] = {"start": stats["start"], "end": stats["end"], "sessions": len(days),
                               "variant": stats["hansei"], "baseline": stats["hansei"],
                               "holding": stats["buy_and_hold"], "trades": stats["trades"], "costs": stats["costs"],
                               "baseline_trades": stats["trades"]}
            for column, values in (("trend", ("up", "down")), ("mood", ("calm", "choppy"))):
                for value in values:
                    mask = (label[column] == value).values
                    a, h = _annualised(agent.values[mask]), _annualised(hold.values[mask])
                    metrics[f"{period}.{value}.hansei_minus_holding"] = a - h
                    metrics[f"{period}.{value}.sessions"] = int(mask.sum())
                    table.append({"period": period, "regime": value, "sessions": int(mask.sum()),
                                  "hansei": a, "holding": h, "difference": a - h})
        return {"metrics": metrics, "periods": periods, "regimes": table, "curves": {},
                "primary": {"label": "Yearly return against holding when the basket is falling",
                            "values": {p: metrics[f"{p}.down.hansei_minus_holding"] for p in self.periods},
                            "prob_better": {}},
                "integrity": self.truncation_invariant()}
