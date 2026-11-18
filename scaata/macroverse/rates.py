"""Interest rates: how much each of the five stocks moves when bond yields move, and the "rates +200 bp" what-if.

Two yield measures, because no free daily Indian 10-year history exists:

    india   change in an Indian 5-year government bond yield, backed out of the price of a gilt ETF
            (GILT5YBEES, since April 2021). A bond's price moves about -duration x the change in its yield, so
            d(yield, in points) = -100 x d(log price) / DURATION. DURATION is an assumption (4.0 years); if the true
            figure is 3.5 or 4.5 every sensitivity below scales by that ratio. The ETF's steady accrual of interest
            (its average weekly drift) is removed first.
    us      the US 10-year Treasury yield, in points, daily since 2006. A global rates factor.

Method (same as the other Macroverse sensitivities): weekly non-overlapping returns, a block bootstrap over weeks
for a 95% interval, and a held-out check (is the later half's estimate inside what the earlier half alone would
have given?). The number reported is the stock's response to a 1-point (100 bp) rise in yield, with and without
the market held fixed: "total" is what a rates shock does to the stock including its effect on the whole market,
"beyond the market" is what is left after the market's move is accounted for.

The what-if "Indian yields +200 bp" pushes the total sensitivity through the same simulator and the real brain as the
other scenarios. 200 bp is far bigger than any week in the sample, so it is a straight-line extrapolation; the
page says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.macroverse import factors, shock
from scaata.macroverse.replay import NAMES

DURATION = 4.0
MEASURES = {"india": "Indian 5-year gilt yield (from a gilt ETF)", "us": "US 10-year Treasury yield"}
SCENARIO = {"id": "rbi-200bp", "title": "Indian bond yields up 2 points",
            "shock": {"india_yield_pp": 2.0},
            "summary": "Indian government bond yields rise 2 percentage points over a month and stay there, the size of "
                       "the move a 200-basis-point round of rate rises would aim for. The index, crude and the rupee "
                       "are not moved separately; whatever the yield does to the stocks (including through the market) "
                       "is what the sensitivities below say."}


def yield_changes(rates: pd.DataFrame, sessions: pd.DatetimeIndex, measure: str, step: int = 1) -> pd.Series:
    """Change in yield, in percentage points, on NSE sessions, over `step` sessions (no overlap when sampled by step)."""
    aligned = rates.reindex(rates.index.union(sessions)).ffill().reindex(sessions)
    if measure == "us":
        level = aligned["us10y"]
        return level.iloc[::step].diff().dropna()
    log_price = np.log(aligned["gilt"]).iloc[::step]         # NaN before the ETF existed; same grid as the stocks
    d = log_price.diff().dropna()
    d = d - d.mean()                                        # remove the fund's steady accrual of interest
    return -100.0 * d / DURATION


def _stock_returns(closes: pd.DataFrame, step: int) -> pd.DataFrame:
    return np.log(closes.iloc[::step]).diff().dropna()


def estimate(closes: pd.DataFrame, rates: pd.DataFrame, market: pd.Series, measure: str, seed: int = 0) -> dict:
    """Sensitivities to a 1-point rise in `measure`, with and without the market held fixed."""
    step = shock.WEEK
    dy = yield_changes(rates, closes.index, measure, step)
    stocks = _stock_returns(closes, step)
    mkt = np.log(market.reindex(closes.index).ffill().iloc[::step]).diff().dropna()
    frame = pd.concat([stocks, dy.rename("dy"), mkt.rename("mkt")], axis=1).dropna()
    cols = list(closes.columns)
    y, dyv, mk = frame[cols].values, frame[["dy"]].values, frame[["dy", "mkt"]].values
    rng = np.random.default_rng(seed)
    n = len(frame)
    total = shock._ols(y, dyv)                                       # (1, stocks)
    draws = shock._bootstrap(y, dyv, rng)
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    beyond = shock._ols(y, mk)[0:1]
    half = n // 2
    first, second = np.arange(n) < half, np.arange(n) >= half
    before, after = shock._ols(y[first], dyv[first])[0], shock._ols(y[second], dyv[second])[0]
    e_low, e_high = np.percentile(shock._bootstrap(y[first], dyv[first], rng), [2.5, 97.5], axis=0)
    table = []
    for j, s in enumerate(cols):
        table.append({"symbol": s.replace(".NS", ""), "name": NAMES.get(s, s), "measure": measure,
                      "beta": float(total[0, j]), "low": float(low[0, j]), "high": float(high[0, j]),
                      "reliable": bool(low[0, j] > 0 or high[0, j] < 0), "beyond_market": float(beyond[0, j]),
                      "before": float(before[j]), "after": float(after[j]),
                      "held_out_ok": bool(e_low[0, j] <= after[j] <= e_high[0, j])})
    return {"table": table, "weeks": int(n), "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()),
            "split": str(frame.index[half].date()), "point": total, "draws": draws,
            "yield_sd_weekly_pp": float(frame["dy"].std()), "yield_range_weekly_pp": [float(frame["dy"].min()), float(frame["dy"].max())]}


def noise(closes: pd.DataFrame, rates: pd.DataFrame, point: np.ndarray) -> np.ndarray:
    """Daily stock returns the yield does not explain, de-meaned, on days the yield measure exists."""
    dy = yield_changes(rates, closes.index, "india", 1)
    daily = np.log(closes).diff().dropna()
    both = pd.concat([daily, dy.rename("dy")], axis=1).dropna()
    left = both[list(closes.columns)].values - both[["dy"]].values @ point
    return left - left.mean(axis=0)


def run(opens: pd.DataFrame, closes: pd.DataFrame, rates: pd.DataFrame, market: pd.Series, paths: int = 30) -> tuple[dict, dict]:
    """(sensitivity study for both measures, the +200 bp scenario)."""
    studies = {m: estimate(closes, rates, market, m) for m in MEASURES}
    india = studies["india"]
    path = np.zeros((shock.HORIZON, 1))
    path[:shock.RAMP, 0] = SCENARIO["shock"]["india_yield_pp"] / shock.RAMP
    sens = {"point": india["point"], "draws": india["draws"], "table": [], "weeks": india["weeks"]}
    out = shock.run_scenario(SCENARIO, opens, closes, pd.DataFrame(), paths=paths, sens=sens,
                             noise=noise(closes, rates, india["point"]), path=path)
    out["kind"] = "rates-scenario"
    out["notes"] += [f"The shock is +{SCENARIO['shock']['india_yield_pp']:.0f} points in an Indian 5-year yield, backed out of a "
                     f"gilt ETF assuming a duration of {DURATION:g} years; the sample only has {india['weeks']} weeks "
                     f"(largest weekly move {max(abs(x) for x in india['yield_range_weekly_pp']):.2f} points), so +2 is "
                     "a long straight-line extrapolation.",
                     "The stocks' response is the TOTAL sensitivity, including the yield's effect on the whole market."]
    for r in out["stocks"]:
        row = next(t for t in india["table"] if t["symbol"] == r["symbol"])
        r["per_point"] = {"beta": row["beta"], "low": row["low"], "high": row["high"], "reliable": row["reliable"]}
    study = {"kind": "rates", "measures": MEASURES, "duration_assumed": DURATION,
             "studies": {m: {k: v for k, v in s.items() if k not in ("point", "draws")} for m, s in studies.items()}}
    return study, out
