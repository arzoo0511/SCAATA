"""Historical replay: real episodes, real prices, HANSEI's real brain.

Each episode is replayed with `scaata.agent.backtest.simulate`, the same
loop the backtests use, which calls the live `brain.decide` and
`memory.learn`. The agent is started a year before the episode so that it
arrives with positions and learned trust, as a running agent would; both
portfolios are then rebased to 100 on the episode's first day.

The replay only ever sees prices up to the day it is deciding on: the
history handed to it is cut at the episode's last day, and each view uses
closes up to its own day.

What a replay is not: the news advisor is silent (there is no scored news
that far back), and the agent trades rupee amounts, not whole shares.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.agent import backtest as bt
from scaata.config import INDIA_TICKERS
from scaata.live.paper_book import COST_PER_SIDE

WARMUP_SESSIONS = 260
NAMES = {"HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank", "PETRONET.NS": "Petronet LNG",
         "IOC.NS": "Indian Oil", "ITC.NS": "ITC"}

# Dates are of the events themselves; each is shown on the first session on or after it.
EPISODES = [
    {"id": "gfc-2008", "title": "Global financial crisis", "start": "2008-01-01", "end": "2009-06-30",
     "summary": "A US housing and credit collapse became a global banking crisis. Indian equities fell by more "
                "than half within a year and then rebounded sharply in 2009.",
     "events": [("2008-01-21", "Global sell-off; Indian markets hit their lower circuit the next morning"),
                ("2008-09-15", "Lehman Brothers files for bankruptcy"),
                ("2008-10-24", "Indian equities' worst single day of the crisis"),
                ("2009-05-18", "Election result: markets hit the upper circuit")]},
    {"id": "taper-2013", "title": "Taper tantrum", "start": "2013-05-01", "end": "2013-12-31",
     "summary": "The US Federal Reserve signalled it would slow its bond buying. Money left emerging markets and "
                "the rupee fell to a then-record low.",
     "events": [("2013-05-22", "Fed chair Bernanke signals tapering"),
                ("2013-08-28", "Rupee hits a then-record low near 68.8 to the dollar"),
                ("2013-09-04", "Raghuram Rajan takes over as RBI governor")]},
    {"id": "demonetisation-2016", "title": "Demonetisation", "start": "2016-10-03", "end": "2017-03-31",
     "summary": "India withdrew its ₹500 and ₹1,000 notes overnight, the same night the US election result "
                "came in.",
     "events": [("2016-11-08", "₹500 and ₹1,000 notes withdrawn; US election"),
                ("2016-12-30", "Deadline to deposit the old notes")]},
    {"id": "ilfs-2018", "title": "IL&FS and the shadow-bank scare", "start": "2018-08-01", "end": "2019-03-29",
     "summary": "A large infrastructure lender defaulted, credit to non-bank lenders froze, and crude was "
                "rising into October.",
     "events": [("2018-09-21", "IL&FS defaults surface; housing-finance stocks crash intraday"),
                ("2018-10-01", "Government takes control of IL&FS's board"),
                ("2018-10-03", "Brent crude peaks above $85")]},
    {"id": "covid-2020", "title": "COVID crash", "start": "2020-01-01", "end": "2020-12-31",
     "summary": "A pandemic shut economies down. Indian equities lost over a third in five weeks, then "
                "recovered the whole fall by November.",
     "events": [("2020-03-11", "WHO declares a pandemic"),
                ("2020-03-23", "Indian markets' biggest one-day fall; the low of the crash"),
                ("2020-03-25", "India's nationwide lockdown begins"),
                ("2020-11-09", "Vaccine trial results; markets at new highs")]},
    {"id": "crude-2022", "title": "Ukraine war and the crude spike", "start": "2022-01-03", "end": "2022-12-30",
     "summary": "Russia invaded Ukraine, Brent crude spiked, inflation rose and the RBI began raising rates.",
     "events": [("2022-02-24", "Russia invades Ukraine"),
                ("2022-03-08", "Brent crude spikes to its highest since 2008"),
                ("2022-05-04", "RBI raises rates in an unscheduled meeting"),
                ("2022-06-17", "Indian equities' low for the year")]},
]


def stock_values(run: dict, opens: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """Rupees held in each stock at every close, rebuilt from the run's own trades and the
    prices. Pure book-keeping: no decision is recomputed."""
    days = run["days"]
    trades: dict[str, dict[str, float]] = {}
    for t in run["trades"]:
        trades.setdefault(t["date"], {})[t["symbol"]] = t["rupees"]
    value = {s: 0.0 for s in INDIA_TICKERS}
    rows = [dict(value)]
    for i in range(1, len(days)):
        t, prev = days[i], days[i - 1]
        today = trades.get(t.date().isoformat(), {})
        for s in INDIA_TICKERS:
            value[s] = (value[s] * opens.at[t, s] / closes.at[prev, s] + today.get(s, 0.0)) \
                       * closes.at[t, s] / opens.at[t, s]
        rows.append(dict(value))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(days))


def _drawdown(curve: np.ndarray) -> tuple[float, int]:
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1
    return float(dd.min()), int(dd.argmin())


def _recovery(curve: np.ndarray) -> int | None:
    """Sessions from the deepest point back to the level it fell from; None if it never got back."""
    _, trough = _drawdown(curve)
    peak = curve[:trough + 1].max()
    back = np.flatnonzero(curve[trough:] >= peak)
    return int(back[0]) if len(back) else None


def summarise(curve: np.ndarray) -> dict:
    daily = np.diff(curve) / curve[:-1]
    dd, trough = _drawdown(curve)
    return {"return": float(curve[-1] / curve[0] - 1), "max_drawdown": dd, "trough_index": trough,
            "worst_day": float(daily.min()) if len(daily) else 0.0, "recovery_sessions": _recovery(curve),
            "volatility": float(daily.std() * np.sqrt(252)) if len(daily) else 0.0}


def run_episode(episode: dict, opens: pd.DataFrame, closes: pd.DataFrame,
                factor_closes: pd.DataFrame | None = None) -> dict:
    """Replays one episode and returns everything the page draws."""
    start, end = pd.Timestamp(episode["start"]), pd.Timestamp(episode["end"])
    sessions = closes.index[(closes.index >= start) & (closes.index <= end)]
    if len(sessions) < 20:
        raise ValueError(f"episode {episode['id']}: only {len(sessions)} sessions of price history")
    first = closes.index.get_loc(sessions[0])
    warm = closes.index[max(0, first - WARMUP_SESSIONS)]
    cut_opens, cut_closes = opens.loc[:sessions[-1]], closes.loc[:sessions[-1]]   # nothing after the episode
    views = bt.precompute_views(cut_closes, str(warm.date()))
    run = bt.simulate(cut_opens, cut_closes, views, str(warm.date()), str(sessions[-1].date()))
    days = pd.DatetimeIndex(run["days"])
    inside = days >= sessions[0]
    agent, hold = run["curve"][inside], run["bh_curve"][inside]
    values = stock_values(run, cut_opens, cut_closes).loc[inside]
    slot = pd.Series(agent, index=values.index) / len(INDIA_TICKERS)
    exposure = values.div(slot, axis=0)
    window = days[inside]
    prices = cut_closes.loc[window] / cut_closes.loc[window].iloc[0] * 100
    trades = [t for t in run["trades"] if t["date"] >= str(window[0].date())]
    events = []
    for when, text in episode.get("events", []):
        at = window.searchsorted(pd.Timestamp(when))
        if at < len(window):
            events.append({"index": int(at), "date": str(window[at].date()), "text": text})
    out = {
        "id": episode["id"], "kind": "replay", "title": episode["title"], "summary": episode["summary"],
        "start": str(window[0].date()), "end": str(window[-1].date()), "sessions": int(len(window)),
        "dates": [str(d.date()) for d in window],
        "hansei": np.round(agent / agent[0] * 100, 3).tolist(), "holding": np.round(hold / hold[0] * 100, 3).tolist(),
        "invested": np.round(values.sum(axis=1).values / agent, 4).tolist(),
        "stocks": [{"symbol": s.replace(".NS", ""), "name": NAMES.get(s, s),
                    "price": np.round(prices[s].values, 2).tolist(),
                    "exposure": np.round(exposure[s].values, 3).tolist()} for s in INDIA_TICKERS],
        "trades": [{"index": int(window.searchsorted(pd.Timestamp(t["date"]))), "date": t["date"],
                    "symbol": t["symbol"].replace(".NS", ""), "name": NAMES.get(t["symbol"], t["symbol"]),
                    "rupees": t["rupees"]} for t in trades],
        "events": events,
        "stats": {"hansei": summarise(agent), "holding": summarise(hold), "trades": len(trades),
                  "costs": round(sum(abs(t["rupees"]) for t in trades) * COST_PER_SIDE, 2),
                  "average_invested": float((values.sum(axis=1).values / agent).mean())},
        "factors": {},
        "notes": ["Real prices, replayed day by day through the same brain the live agent uses.",
                  "The agent starts a year earlier, so it arrives with positions and learned trust.",
                  "The news advisor is silent: no scored headlines exist for these years.",
                  "It trades rupee amounts here; the live ₹10,000 book trades whole shares."],
    }
    peak_invested = float((values.sum(axis=1).values / agent).max())
    out["stats"]["peak_invested"] = peak_invested
    if peak_invested > 1.005:
        out["notes"].append(f"At its most invested the replay held {peak_invested:.0%} of the book in stocks: the "
                            "replay engine lets cash dip below zero when holdings have outgrown their slots. The "
                            "live paper book cannot do this, so the replay is slightly more aggressive than live.")
    if factor_closes is not None:
        for name in factor_closes:
            series = factor_closes[name].reindex(window)
            if series.notna().all():
                out["factors"][name] = np.round(series.values / series.values[0] * 100, 2).tolist()
    return out
