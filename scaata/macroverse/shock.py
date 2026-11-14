"""Synthetic what-if scenarios: a shock to the market, crude or the rupee,
passed through measured sensitivities, traded by HANSEI's real brain.

How a scenario is built, and what is assumed at each step:

1. Sensitivities. Each stock's weekly return is regressed on the weekly
   returns of the three factors together. Weekly, because Brent and the
   rupee close hours after the NSE and daily returns would not line up.
   The uncertainty is kept: the regression is repeated on resampled blocks
   of weeks, and every simulated path draws one of those resampled sets.
2. The shock. The stated move in each factor arrives evenly over the first
   `RAMP` sessions and then stops. Nothing else in the world moves.
3. Everything else about a stock. Its daily returns that the factors do not
   explain are resampled from history in blocks (all five stocks together,
   so their co-movement survives) and de-meaned, so no stock is assumed to
   drift up or down on its own.
4. Trading. Each simulated price path is appended to the real history and
   traded by `backtest.simulate`: the real brain and memory, deciding each
   day on prices up to that day only. The agent is started half a year
   before the shock so it arrives with positions.

A scenario is a model output. It is not a forecast and not history, and
every result carries `kind: "scenario"` so it cannot be mistaken for one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.agent import backtest as bt
from scaata.config import INDIA_TICKERS
from scaata.macroverse.replay import NAMES, summarise

FACTORS = ("market", "crude", "rupee")
WEEK, BLOCK_WEEKS, BLOCK_DAYS = 5, 8, 20
BETA_DRAWS = 400
HORIZON, RAMP, WARMUP = 126, 21, 130
SPLIT = "2017-01-01"

SCENARIOS = [
    {"id": "crude-up-60", "title": "Crude oil up 60%", "shock": {"crude": 0.60},
     "summary": "Brent rises 60% over a month and stays there. The market index and the rupee are held still, "
                "so this isolates the part of an oil shock that is not already in the index."},
    {"id": "market-down-20", "title": "Market down 20%", "shock": {"market": -0.20},
     "summary": "The Nifty falls 20% over a month and stays there. Crude and the rupee are held still."},
    {"id": "rupee-down-10", "title": "Rupee weakens 10%", "shock": {"rupee": 0.10},
     "summary": "The dollar gains 10% against the rupee over a month. The index and crude are held still."},
    {"id": "all-three", "title": "All three at once", "shock": {"market": -0.20, "crude": 0.60, "rupee": 0.10},
     "summary": "The market falls 20% while crude rises 60% and the rupee weakens 10%, all over one month."},
]
UNAVAILABLE = [
    ("RBI raises rates by 200 basis points", "No reliable free daily history of Indian bond yields or the policy "
     "rate path is in the data, so a rate sensitivity cannot be measured."),
    ("Inflation spike, GDP slowdown, unemployment", "These are monthly or quarterly series. Their effect on five "
     "stocks' daily prices cannot be estimated without inventing it."),
    ("Banking or liquidity crisis", "No measurable daily factor for it. The closest honest answer is the 2008 and "
     "2018 replays."),
]


def weekly_returns(closes: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-overlapping five-session log returns of the stocks and the factors, on shared sessions."""
    both = pd.concat([closes, factors[list(FACTORS)]], axis=1).dropna()
    weekly = np.log(both.iloc[::WEEK]).diff().dropna()
    return weekly[list(closes.columns)], weekly[list(FACTORS)]


def _ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0][1:]      # one row per factor, one column per stock


def sensitivities(closes: pd.DataFrame, factors: pd.DataFrame, seed: int = 0) -> dict:
    """Each stock's sensitivity to each factor, with a 95% interval from a block bootstrap over
    weeks, and the same estimate on the two halves of history to show whether it is stable."""
    stocks, moves = weekly_returns(closes, factors)
    y, x = stocks.values, moves.values
    point = _ols(y, x)
    rng = np.random.default_rng(seed)
    n = len(x)
    starts_needed = int(np.ceil(n / BLOCK_WEEKS))
    draws = np.empty((BETA_DRAWS, *point.shape))
    for k in range(BETA_DRAWS):
        starts = rng.integers(0, n - BLOCK_WEEKS + 1, starts_needed)
        rows = np.concatenate([np.arange(s, s + BLOCK_WEEKS) for s in starts])[:n]
        draws[k] = _ols(y[rows], x[rows])
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    early, late = stocks.index < pd.Timestamp(SPLIT), stocks.index >= pd.Timestamp(SPLIT)
    halves = {"before": _ols(y[early], x[early]), "after": _ols(y[late], x[late])} if early.sum() > 60 and late.sum() > 60 else {}
    table = []
    for j, symbol in enumerate(stocks.columns):
        for i, factor in enumerate(FACTORS):
            table.append({"symbol": symbol.replace(".NS", ""), "name": NAMES.get(symbol, symbol), "factor": factor,
                          "beta": float(point[i, j]), "low": float(low[i, j]), "high": float(high[i, j]),
                          "reliable": bool(low[i, j] > 0 or high[i, j] < 0),
                          "before": float(halves["before"][i, j]) if halves else None,
                          "after": float(halves["after"][i, j]) if halves else None})
    return {"table": table, "draws": draws, "point": point, "weeks": int(n),
            "first": str(stocks.index[0].date()), "last": str(stocks.index[-1].date()), "split": SPLIT}


def residuals(closes: pd.DataFrame, factors: pd.DataFrame, point: np.ndarray) -> np.ndarray:
    """Daily stock returns the factors do not explain, de-meaned: the noise a scenario resamples."""
    both = pd.concat([closes, factors[list(FACTORS)]], axis=1).dropna()
    daily = np.log(both).diff().dropna()
    left = daily[list(closes.columns)].values - daily[list(FACTORS)].values @ point
    return left - left.mean(axis=0)


def factor_path(shock: dict) -> np.ndarray:
    """Daily log moves of each factor: the whole shock, spread evenly over the first RAMP sessions."""
    path = np.zeros((HORIZON, len(FACTORS)))
    for i, factor in enumerate(FACTORS):
        if shock.get(factor):
            path[:RAMP, i] = np.log1p(shock[factor]) / RAMP
    return path


def simulate_prices(last_close: pd.Series, path: np.ndarray, beta: np.ndarray, noise: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """One simulated close path for the five stocks over the horizon."""
    blocks = int(np.ceil(HORIZON / BLOCK_DAYS))
    starts = rng.integers(0, len(noise) - BLOCK_DAYS + 1, blocks)
    sampled = np.concatenate([noise[s:s + BLOCK_DAYS] for s in starts])[:HORIZON]
    returns = path @ beta + sampled
    return last_close.values * np.exp(np.cumsum(returns, axis=0))


def run_scenario(scenario: dict, opens: pd.DataFrame, closes: pd.DataFrame, factors: pd.DataFrame,
                 paths: int = 30, seed: int = 0, sens: dict | None = None) -> dict:
    """Runs one scenario: `paths` simulated futures from the last real day, each traded by the real brain."""
    sens = sens or sensitivities(closes, factors, seed)
    noise = residuals(closes, factors, sens["point"])
    path = factor_path(scenario["shock"])
    rng = np.random.default_rng(seed + 1)
    anchor = closes.index[-1]
    future = pd.bdate_range(anchor + pd.Timedelta(days=1), periods=HORIZON)
    warm = closes.index[-WARMUP]
    history_opens, history_closes = opens.iloc[-(bt.VIEW_WINDOW + WARMUP):], closes.iloc[-(bt.VIEW_WINDOW + WARMUP):]
    agent_paths, hold_paths, invested_end, stock_paths = [], [], [], []
    for _ in range(paths):
        beta = sens["draws"][rng.integers(len(sens["draws"]))]
        sim = pd.DataFrame(simulate_prices(closes.iloc[-1], path, beta, noise, rng), index=future,
                           columns=closes.columns)
        all_closes = pd.concat([history_closes, sim])
        sim_opens = all_closes.shift().loc[future]            # assumption: each day opens at the previous close
        all_opens = pd.concat([history_opens, sim_opens])
        views = bt.precompute_views(all_closes, str(warm.date()))
        run = bt.simulate(all_opens, all_closes, views, str(warm.date()), None)
        days = pd.DatetimeIndex(run["days"])
        inside = days >= anchor
        agent, hold = run["curve"][inside], run["bh_curve"][inside]
        agent_paths.append(agent / agent[0] * 100)
        hold_paths.append(hold / hold[0] * 100)
        stock_paths.append(np.vstack([np.full(len(closes.columns), 100.0), sim.values / closes.iloc[-1].values * 100]))
    agent_paths, hold_paths, stock_paths = np.array(agent_paths), np.array(hold_paths), np.array(stock_paths)
    band = lambda a: {k: np.round(np.percentile(a, q, axis=0), 2).tolist()   # noqa: E731
                      for k, q in (("p10", 10), ("p50", 50), ("p90", 90))}
    agent_stats = [summarise(p) for p in agent_paths]
    hold_stats = [summarise(p) for p in hold_paths]
    expected = np.exp(path.sum(axis=0) @ sens["point"]) - 1
    return {
        "id": scenario["id"], "kind": "scenario", "title": scenario["title"], "summary": scenario["summary"],
        "shock": scenario["shock"], "paths": paths, "horizon": HORIZON, "ramp": RAMP, "seed": seed,
        "anchor": str(anchor.date()), "days": list(range(HORIZON + 1)),
        "hansei": band(agent_paths), "holding": band(hold_paths),
        "sample_paths": {"hansei": np.round(agent_paths[:12], 2).tolist(),
                         "holding": np.round(hold_paths[:12], 2).tolist()},
        "stocks": [{"symbol": s.replace(".NS", ""), "name": NAMES.get(s, s),
                    "expected_move": float(expected[j]), **band(stock_paths[:, :, j])}
                   for j, s in enumerate(closes.columns)],
        "stats": {
            "hansei": {"median_return": float(np.median([s["return"] for s in agent_stats])),
                       "median_drawdown": float(np.median([s["max_drawdown"] for s in agent_stats])),
                       "worst_drawdown": float(min(s["max_drawdown"] for s in agent_stats))},
            "holding": {"median_return": float(np.median([s["return"] for s in hold_stats])),
                        "median_drawdown": float(np.median([s["max_drawdown"] for s in hold_stats])),
                        "worst_drawdown": float(min(s["max_drawdown"] for s in hold_stats))},
            "share_hansei_ends_ahead": float(np.mean(agent_paths[:, -1] > hold_paths[:, -1])),
            "share_hansei_falls_less": float(np.mean([a["max_drawdown"] > h["max_drawdown"]
                                                      for a, h in zip(agent_stats, hold_stats)]))},
        "notes": [f"Synthetic: {paths} simulated paths over {HORIZON} sessions from {anchor.date()}. Not a forecast.",
                  f"The shock arrives evenly over the first {RAMP} sessions; no other factor moves.",
                  "Sensitivities come from weekly regressions and are uncertain; each path draws a different "
                  "plausible set.",
                  "Unexplained daily moves are resampled from history with no drift of their own.",
                  "Each simulated day opens at the previous close; the news advisor is silent.",
                  f"With {paths} paths the 10th and 90th percentile bands are rough."],
    }
