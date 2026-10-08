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

    python -m scaata.agent.backtest --long

runs the same from 2007, which adds the 2008 crash and five more years, and
splits it the honest way: settings may only be chosen on 2007-2019 (*tune*);
2020 onward (*holdout*) is looked at once, to check them. It also reports
every three-year block, so one good stretch can't carry the result. Writes
`results/hansei_backtest_long.json`.

    python -m scaata.agent.backtest --news

runs the agent with and without the archived news (`scaata.agent.news_history`)
over the same days and scores the news view on its own. Writes
`results/hansei_backtest_news.json`.
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
LONG_HISTORY_START = "2006-01-01"   # Yahoo's IOC series has a broken print in Jul 2005
LONG_PERIODS = (("Tune 2007-2019", "2007-01-01", "2019-12-31"), ("Holdout 2020-", "2020-01-01", None),
                ("2007-2009", "2007-01-01", "2009-12-31"), ("2010-2012", "2010-01-01", "2012-12-31"),
                ("2013-2015", "2013-01-01", "2015-12-31"), ("2016-2019", "2016-01-01", "2019-12-31"),
                ("2020-2022", "2020-01-01", "2022-12-31"), ("2023-", "2023-01-01", None))
LONG_RESULTS_PATH = ROOT_DIR / "results" / "hansei_backtest_long.json"
BAD_OPEN = 0.10            # an open this far from both the previous close and its own close...
FLAT_DAY = 0.05            # ...on a day that otherwise barely moved is a bad print
DEFAULTS = {"act_threshold": brain.ACT_THRESHOLD, "min_hold_days": brain.MIN_HOLD_DAYS,
            "learning_rate": mem.LEARNING_RATE}


def repair_opens(opens: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Yahoo's older NSE opens contain bad prints: a stock "opens" 18% down and
    closes flat. Fills happen at the open, so those would be fake wins or
    losses. Where the open is far from the previous close while the close
    is near it, use the previous close as the open."""
    prev = closes.shift()
    gap, net = np.log(opens / prev), np.log(closes / prev)
    bad = (gap.abs() > BAD_OPEN) & (net.abs() < FLAT_DAY)
    return opens.mask(bad, prev)


def load_prices(end: date, start: str = "2019-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    from scaata.data.loaders import load_market_data

    raw = load_market_data(list(INDIA_TICKERS), start, (end + pd.Timedelta(days=1)).isoformat(),
                           use_cache=False)
    raw = raw[raw["Volume"] > 0].reset_index()   # Yahoo's flat, zero-volume bars on NSE holidays
    closes = raw.pivot_table(index="Date", columns="Ticker", values="Close").dropna()
    opens = raw.pivot_table(index="Date", columns="Ticker", values="Open").reindex(closes.index)
    return repair_opens(opens.fillna(closes), closes), closes


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
             learning_rate: float = mem.LEARNING_RATE, tolerance: float = 0.0, floor: float = 0.0) -> dict:
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
                                    1 / n, act_threshold, min_hold_days, tolerance, floor)
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


def _print_periods(periods: dict) -> None:
    print(f"{'':<16}{'HANSEI CAGR/Sharpe/DD':>26}{'Buy&hold CAGR/Sharpe/DD':>28}{'trades':>8}{'P(better)':>11}")
    for label, r in periods.items():
        a, b = r["hansei"], r["buy_and_hold"]
        print(f"{label:<16}{a['cagr']:>9.2%}{a['sharpe']:>8.2f}{a['max_drawdown']:>9.1%}"
              f"{b['cagr']:>11.2%}{b['sharpe']:>8.2f}{b['max_drawdown']:>9.1%}{r['trades']:>8}"
              f"{r['prob_sharpe_better']:>11.0%}")


def _grid(opens, closes, views, start: str, end: str | None) -> list[dict]:
    grid = []
    for threshold in (0.25, 1 / 3, 0.5):
        for hold in (5, 10, 20):
            result = evaluate(simulate(opens, closes, views, start, end, threshold, hold))
            grid.append({"act_threshold": round(threshold, 3), "min_hold_days": hold, **{
                k: result[k] for k in ("sharpe_diff", "cagr_diff", "trades")},
                "max_drawdown": result["hansei"]["max_drawdown"],
                "hold_max_drawdown": result["buy_and_hold"]["max_drawdown"]})
    return grid


def main_long() -> None:
    """2007 onward: tune on 2007-2019, check once on 2020-, and every 3-year block."""
    opens, closes = load_prices(date.today(), LONG_HISTORY_START)
    views = precompute_views(closes, LONG_PERIODS[0][1])
    periods = {label: evaluate(simulate(opens, closes, views, start, end), bootstrap=True)
               for label, start, end in LONG_PERIODS}
    grids = {label: _grid(opens, closes, views, start, end) for label, start, end in LONG_PERIODS[:2]}
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "defaults": DEFAULTS,
              "cost_per_side": COST_PER_SIDE, "news": "silent (no historical archive)",
              "history_start": LONG_HISTORY_START, "periods": periods, "grids": grids}
    LONG_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LONG_RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _print_periods(periods)
    for label, grid in grids.items():
        beat_sharpe = sum(g["sharpe_diff"] > 0 for g in grid)
        beat_cagr = sum(g["cagr_diff"] > 0 for g in grid)
        shallower = sum(g["max_drawdown"] > g["hold_max_drawdown"] for g in grid)
        print(f"grid on {label}: Sharpe better {beat_sharpe}/{len(grid)}, CAGR better {beat_cagr}/{len(grid)}, "
              f"smaller drawdown {shallower}/{len(grid)}; CAGR diff "
              f"{min(g['cagr_diff'] for g in grid):+.2%} .. {max(g['cagr_diff'] for g in grid):+.2%}")
    print(f"saved {LONG_RESULTS_PATH.relative_to(ROOT_DIR)}")


EXPOSURE_RESULTS_PATH = ROOT_DIR / "results" / "hansei_backtest_exposure.json"
TOLERANCES = (0.0, 0.1, 0.2, 0.3)
FLOORS = (0.0, 0.25, 0.5)
DRAWDOWN_MARGIN = 0.05     # a variant must keep its worst fall at least 5 points smaller than holding's


def main_exposure() -> None:
    """Can HANSEI spend its risk edge on being invested more? Sweeps how
    tolerant the brain is of negative evidence and the least it ever holds.
    The rule for choosing, fixed before running: the best 2007-2019 CAGR
    among variants whose 2007-2019 drawdown stays DRAWDOWN_MARGIN smaller
    than holding's. Then that one choice is checked once on 2020-."""
    opens, closes = load_prices(date.today(), LONG_HISTORY_START)
    views = precompute_views(closes, LONG_PERIODS[0][1])
    (tune_label, ts, te), (hold_label, hs, he) = LONG_PERIODS[:2]
    rows = []
    for tolerance in TOLERANCES:
        for floor in FLOORS:
            tune = evaluate(simulate(opens, closes, views, ts, te, tolerance=tolerance, floor=floor))
            rows.append({"tolerance": tolerance, "floor": floor, "tune": tune})
            print(f"tolerance {tolerance:.1f} floor {floor:.2f}: tune CAGR {tune['hansei']['cagr']:.2%} "
                  f"(hold {tune['buy_and_hold']['cagr']:.2%})  DD {tune['hansei']['max_drawdown']:.1%} "
                  f"(hold {tune['buy_and_hold']['max_drawdown']:.1%})  Sharpe {tune['hansei']['sharpe']:.2f}"
                  f"  trades {tune['trades']}", flush=True)
    eligible = [r for r in rows
                if r["tune"]["hansei"]["max_drawdown"] >= r["tune"]["buy_and_hold"]["max_drawdown"] + DRAWDOWN_MARGIN]
    chosen = max(eligible, key=lambda r: r["tune"]["hansei"]["cagr"])
    checks = {}
    for name, r in (("default", rows[0]), ("chosen", chosen)):
        checks[name] = {"tolerance": r["tolerance"], "floor": r["floor"],
                        "holdout": evaluate(simulate(opens, closes, views, hs, he, tolerance=r["tolerance"],
                                                     floor=r["floor"]), bootstrap=True),
                        "holdout_grid": [{"act_threshold": round(th, 3), "min_hold_days": hold, **{
                            k: v for k, v in evaluate(simulate(opens, closes, views, hs, he, th, hold,
                                                               tolerance=r["tolerance"], floor=r["floor"])).items()
                            if k in ("cagr_diff", "sharpe_diff")}}
                            for th in (0.25, 1 / 3, 0.5) for hold in (5, 10, 20)]}
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "rule": (
        f"best {tune_label} CAGR with drawdown at least {DRAWDOWN_MARGIN:.0%} smaller than holding's"),
        "sweep": rows, "checks": checks}
    EXPOSURE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPOSURE_RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"chosen on {tune_label}: tolerance {chosen['tolerance']}, floor {chosen['floor']}")
    for name, c in checks.items():
        h = c["holdout"]
        print(f"{name:<8} {hold_label}: CAGR {h['hansei']['cagr']:.2%} vs hold {h['buy_and_hold']['cagr']:.2%}  "
              f"DD {h['hansei']['max_drawdown']:.1%} vs {h['buy_and_hold']['max_drawdown']:.1%}  "
              f"Sharpe {h['hansei']['sharpe']:.2f} vs {h['buy_and_hold']['sharpe']:.2f}  "
              f"CAGR better in {sum(g['cagr_diff'] > 0 for g in c['holdout_grid'])}/9 settings")
    print(f"saved {EXPOSURE_RESULTS_PATH.relative_to(ROOT_DIR)}")


NEWS_HISTORY_START = "2011-01-01"
NEWS_PERIODS = (("2012-2024H1", "2012-01-01", "2024-06-30"), ("2024H2- (true test)", "2024-07-01", None))
NEWS_RESULTS_PATH = ROOT_DIR / "results" / "hansei_backtest_news.json"


def with_news(views_by_day: dict, news_by_symbol: dict[str, pd.Series]) -> dict:
    """The same technical views, with the news advisor's archived view filled in."""
    out = {}
    for t, by_symbol in views_by_day.items():
        out[t] = {}
        for s, v in by_symbol.items():
            n = news_by_symbol[s].get(t, np.nan)
            out[t][s] = {**v, "news": round(float(n), 3) if np.isfinite(n) else 0.0}
    return out


def main_news() -> None:
    """Does the news advisor earn its vote? The same agent, with and without
    archived news, over the same days. Before mid-2024 the scoring model may
    remember how stories ended; after it, it cannot."""
    from scaata.agent import news_history
    from scaata.agent.signal_study import forward_excess, score

    opens, closes = load_prices(date.today(), NEWS_HISTORY_START)
    news = {s: news_history.news_views(s, closes.index) for s in INDIA_TICKERS}
    covered = pd.concat(news, axis=1).dropna().index
    if covered.empty:
        sys.exit("no scored news yet: run `python -m scaata.agent.news_history run` first")
    last = covered.max().date().isoformat()
    views = precompute_views(closes, NEWS_PERIODS[0][1])
    views_news = with_news(views, news)
    periods, signal = {}, {}
    rng = np.random.default_rng(0)
    news_frame = pd.DataFrame(news)
    for label, start, end in NEWS_PERIODS:
        inside = covered[(covered >= pd.Timestamp(start)) & (end is None or covered <= pd.Timestamp(end))]
        if len(inside) < 60:      # under ~3 months of scored news says nothing
            continue
        start, end = inside.min().date().isoformat(), inside.max().date().isoformat()
        base, plus = (simulate(opens, closes, v, start, end) for v in (views, views_news))
        a, b = evaluate(plus), evaluate(base)
        periods[label] = {"with_news": a, "without_news": b,
                          "cagr_diff": a["hansei"]["cagr"] - b["hansei"]["cagr"],
                          "sharpe_diff": a["hansei"]["sharpe"] - b["hansei"]["sharpe"],
                          "prob_news_better": paired_block_bootstrap_prob_better(plus["curve"], base["curve"]),
                          "grid": [{"act_threshold": round(th, 3), "min_hold_days": hold,
                                    "cagr_diff": evaluate(simulate(opens, closes, views_news, start, end, th, hold))
                                    ["hansei"]["cagr"] - evaluate(simulate(opens, closes, views, start, end, th, hold))
                                    ["hansei"]["cagr"]}
                                   for th in (0.25, 1 / 3, 0.5) for hold in (5, 10, 20)]}
        window = slice(pd.Timestamp(start), pd.Timestamp(end))
        signal[label] = {f"{h}_sessions": score(news_frame.loc[window], forward_excess(closes, h).loc[window], rng)
                         for h in (5, 20)}
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "news_covered_through": last,
              "spans": {k: [v["with_news"]["start"], v["with_news"]["end"]] for k, v in periods.items()},
              "periods": periods, "news_signal": signal}
    NEWS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEWS_RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for label, r in periods.items():
        a, b = r["with_news"]["hansei"], r["without_news"]["hansei"]
        better = sum(g["cagr_diff"] > 0 for g in r["grid"])
        print(f"{label:<22} with news {a['cagr']:.2%}/{a['sharpe']:.2f}/{a['max_drawdown']:.1%}   "
              f"without {b['cagr']:.2%}/{b['sharpe']:.2f}/{b['max_drawdown']:.1%}   "
              f"P(news better) {r['prob_news_better']:.0%}   CAGR better in {better}/9 settings")
        for h, c in signal[label].items():
            if c.get("opinions"):
                print(f"{'':<22} news view, {h.replace('_', ' ')}: spread {c['spread']:+.2%} "
                      f"[{c['spread_ci95'][0]:+.2%}, {c['spread_ci95'][1]:+.2%}]  hit {c['hit_rate']:.0%}"
                      f"  ({c['opinions']:,} opinions)")
    print(f"news covered through {last}; saved {NEWS_RESULTS_PATH.relative_to(ROOT_DIR)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--long" in sys.argv[1:]:
        return main_long()
    if "--news" in sys.argv[1:]:
        return main_news()
    if "--exposure" in sys.argv[1:]:
        return main_exposure()
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
