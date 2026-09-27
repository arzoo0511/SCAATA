"""Replays HANSEI's daily loop over history, against buy-and-hold.

    python -m scaata.agent.backtest

Mirrors `scaata.agent.daily` step for step: starts from ₹10,000 cash,
decides after each close, fills at the next open, pays NSE costs on every
rupee traded, parks idle cash in a liquid fund at the rate of the time,
learns every day from views that are 20 sessions old. The
only difference is news: there is no archive of what the headlines said on
each past day, so the news advisor stays silent here and its value can only
be measured going forward (the live agent's trust in it is that measure).

Beyond the headline numbers it asks the questions that matter before
believing them:

- **is it luck?** a paired block bootstrap gives the probability that the
  agent's Sharpe really beats holding over the same days;
- **is it one lucky setting?** the same run over a grid of the brain's
  tuning knobs -- a result that only holds at one setting is not a result;
- **does learning help?** the same run with learning switched off.

Writes `results/hansei_backtest.json` for the dashboard. Fractional shares
(the live book trades whole shares); otherwise the same mechanics.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from scaata.agent import advisors, brain, memory as mem
from scaata.config import INDIA_TICKERS, ROOT_DIR
from scaata.evaluation.metrics import max_drawdown, sharpe_ratio
from scaata.evaluation.stats import paired_block_bootstrap_prob_better
from scaata.live.paper_book import COST_PER_SIDE, liquid_yield

RESULTS_PATH = ROOT_DIR / "results" / "hansei_backtest.json"
START_CASH = 10_000.0
VIEW_WINDOW = 400          # every advisor looks back at most ~272 sessions
PERIODS = (("Full period", "2020-01-01", None), ("First half", "2020-01-01", "2023-06-30"),
           ("Second half", "2023-07-01", None))
DEFAULTS = {"act_threshold": brain.ACT_THRESHOLD, "min_hold_days": brain.MIN_HOLD_DAYS,
            "learning_rate": mem.LEARNING_RATE}


def load_prices(end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scaata.data.loaders import load_market_data

    raw = load_market_data(list(INDIA_TICKERS), "2019-01-01", (end + pd.Timedelta(days=1)).isoformat(),
                           use_cache=False).reset_index()
    closes = raw.pivot_table(index="Date", columns="Ticker", values="Close").dropna()
    opens = raw.pivot_table(index="Date", columns="Ticker", values="Open").reindex(closes.index)
    return opens.fillna(closes), closes


def precompute_views(closes: pd.DataFrame, start: str) -> dict[pd.Timestamp, dict[str, dict]]:
    """Technical views for every decision day. They do not depend on the
    brain's settings, so the parameter sweep reuses them."""
    out = {}
    for k in np.flatnonzero(closes.index >= pd.Timestamp(start)):
        window = closes.iloc[max(0, k - VIEW_WINDOW):k + 1]
        out[closes.index[k]] = {s: advisors.views(window[s], None) for s in INDIA_TICKERS}
    return out


def simulate(opens: pd.DataFrame, closes: pd.DataFrame, views_by_day: dict, start: str, end: str | None,
             act_threshold: float = brain.ACT_THRESHOLD, min_hold_days: int = brain.MIN_HOLD_DAYS,
             learning_rate: float = mem.LEARNING_RATE) -> dict:
    days = [t for t in views_by_day if t >= pd.Timestamp(start) and (end is None or t <= pd.Timestamp(end))]
    n = len(INDIA_TICKERS)
    memory = mem.new_memory()
    value = {s: 0.0 for s in INDIA_TICKERS}
    cash, costs, trades = START_CASH, 0.0, []
    last_trade: dict[str, int | None] = {s: None for s in INDIA_TICKERS}
    pending: dict[str, float] = {}
    curve, bh_curve, bh_value = [], [], None

    for i, t in enumerate(days):
        if i > 0:
            prev = days[i - 1]
            for s in INDIA_TICKERS:
                value[s] *= opens.at[t, s] / closes.at[prev, s]
            cash *= (1 + liquid_yield(t.year)) ** ((t - prev).days / 365)
            equity = cash + sum(value.values())
            for s, target in pending.items():          # yesterday's decisions, at today's open
                delta = equity / n * target - value[s]
                cash -= delta + abs(delta) * COST_PER_SIDE
                costs += abs(delta) * COST_PER_SIDE
                value[s] += delta
                last_trade[s] = i
                trades.append({"date": t.date().isoformat(), "symbol": s, "rupees": round(delta, 2)})
            pending = {}
            for s in INDIA_TICKERS:
                value[s] *= closes.at[t, s] / opens.at[t, s]
            if bh_value is None:                        # buy-and-hold buys at the same first open
                bh_value = {s: START_CASH / n * (1 - COST_PER_SIDE) * closes.at[t, s] / opens.at[t, s]
                            for s in INDIA_TICKERS}
            else:
                bh_value = {s: bh_value[s] * closes.at[t, s] / closes.at[prev, s] for s in INDIA_TICKERS}
        equity = cash + sum(value.values())
        curve.append(equity)
        bh_curve.append(sum(bh_value.values()) if bh_value else START_CASH)

        if learning_rate > 0:
            mem.learn(memory, {s: closes[s].loc[:t] for s in INDIA_TICKERS}, t.date(), learning_rate)
        share = equity / n
        for s in INDIA_TICKERS:
            views = views_by_day[t][s]
            mem.remember(memory, t.date(), s, views, float(closes.at[t, s]))
            since = None if last_trade[s] is None else i - last_trade[s]
            decision = brain.decide(s, views, memory["weights"], value[s] / share, since, value[s] / equity,
                                    1 / n, act_threshold, min_hold_days)
            if decision.action != "HOLD":
                pending[s] = decision.target_exposure

    return {"days": days, "curve": np.array(curve), "bh_curve": np.array(bh_curve), "trades": trades,
            "costs": costs, "weights": memory["weights"], "weight_history": memory["weight_history"]}


def _stats(curve: np.ndarray, days: list[pd.Timestamp]) -> dict:
    years = (days[-1] - days[0]).days / 365.25
    return {"cagr": float((curve[-1] / curve[0]) ** (1 / years) - 1), "sharpe": float(sharpe_ratio(curve)),
            "max_drawdown": float(max_drawdown(curve)), "final": float(curve[-1])}


def evaluate(run: dict, bootstrap: bool = False) -> dict:
    agent, held = _stats(run["curve"], run["days"]), _stats(run["bh_curve"], run["days"])
    out = {"start": run["days"][0].date().isoformat(), "end": run["days"][-1].date().isoformat(),
           "hansei": agent, "buy_and_hold": held, "trades": len(run["trades"]), "costs": round(run["costs"], 2),
           "sharpe_diff": agent["sharpe"] - held["sharpe"], "cagr_diff": agent["cagr"] - held["cagr"],
           "weights": run["weights"]}
    if bootstrap:
        out["prob_sharpe_better"] = paired_block_bootstrap_prob_better(run["curve"], run["bh_curve"])
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    opens, closes = load_prices(date.today())
    views = precompute_views(closes, PERIODS[0][1])

    periods = {}
    for label, start, end in PERIODS:
        run = simulate(opens, closes, views, start, end)
        periods[label] = evaluate(run, bootstrap=True)
        if label == "Full period":
            full = run
    grid = []
    for threshold in (0.25, 1 / 3, 0.5):
        for hold in (5, 10, 20):
            result = evaluate(simulate(opens, closes, views, PERIODS[0][1], None, threshold, hold))
            grid.append({"act_threshold": round(threshold, 3), "min_hold_days": hold, **{
                k: result[k] for k in ("sharpe_diff", "cagr_diff", "trades")},
                "max_drawdown": result["hansei"]["max_drawdown"]})
    learning = []
    for eta in (0.0, 0.02, 0.05, 0.1):
        result = evaluate(simulate(opens, closes, views, PERIODS[0][1], None, learning_rate=eta))
        learning.append({"learning_rate": eta, **{k: result[k] for k in ("sharpe_diff", "cagr_diff", "trades")},
                         "sharpe": result["hansei"]["sharpe"], "weights": result["weights"]})

    weekly = pd.DataFrame({"hansei": full["curve"], "buy_and_hold": full["bh_curve"]},
                          index=pd.DatetimeIndex(full["days"])).resample("W-FRI").last()
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "defaults": DEFAULTS,
              "cost_per_side": COST_PER_SIDE, "news": "silent (no historical archive)",
              "periods": periods, "grid": grid, "learning": learning,
              "curve": {"dates": [d.date().isoformat() for d in weekly.index],
                        "hansei": weekly["hansei"].round(2).tolist(),
                        "buy_and_hold": weekly["buy_and_hold"].round(2).tolist()},
              "weight_history": full["weight_history"][::5]}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'':<13}{'HANSEI CAGR/Sharpe/DD':>26}{'Buy&hold CAGR/Sharpe/DD':>28}{'trades':>8}{'P(better)':>11}")
    for label, r in periods.items():
        a, b = r["hansei"], r["buy_and_hold"]
        print(f"{label:<13}{a['cagr']:>9.2%}{a['sharpe']:>8.2f}{a['max_drawdown']:>9.1%}"
              f"{b['cagr']:>11.2%}{b['sharpe']:>8.2f}{b['max_drawdown']:>9.1%}{r['trades']:>8}"
              f"{r['prob_sharpe_better']:>11.0%}")
    beat = sum(g["sharpe_diff"] > 0 for g in grid)
    print(f"settings grid: {beat}/{len(grid)} beat buy-and-hold on Sharpe; "
          f"Sharpe diff range {min(g['sharpe_diff'] for g in grid):+.2f} .. {max(g['sharpe_diff'] for g in grid):+.2f}")
    for row in learning:
        print(f"learning rate {row['learning_rate']:<5} Sharpe diff {row['sharpe_diff']:+.2f}  "
              f"CAGR diff {row['cagr_diff']:+.2%}  trades {row['trades']}")
    print(f"saved {RESULTS_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
