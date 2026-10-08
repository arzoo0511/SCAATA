"""Do HANSEI's advisors actually predict anything? Tested on ~100 large NSE stocks.

    python -m scaata.agent.signal_study

The paper book has five stocks, which gives five data points a day: far too
few to tell a real signal from luck. The advisors don't care which stock
they look at, so this scores them on a large-cap universe (roughly the
Nifty 100) over 2007 onward. That is twenty times the evidence per day.

For every stock and day, each advisor's view is set against the stock's
return over the next 20 (and 5) sessions *in excess of the liquid-fund rate*:
the agent's real choice is between owning a stock and parking the money.
An advisor earns its place if stocks it likes go on to beat cash more than
stocks it dislikes. That gap is the **spread**. The volatility advisor is a
brake that never says "buy", so it is scored as warned days against all
others, and on what it is for: whether the swings that follow are larger.

Days within a horizon overlap and all stocks fall together in a crash, so
the uncertainty comes from a block bootstrap over whole months, not from
counting rows. Settings may only be judged on 2007-2019 (*tune*); 2020
onward (*holdout*) is a check.

Caveats: the universe is today's large caps (survivorship bias flatters
absolute returns, much less a spread between views); news has no history
here, so it is not scored.

Writes `results/hansei_signal_study.json`.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from scaata.config import INDIA_TICKERS, ROOT_DIR
from scaata.live.paper_book import liquid_yield

RESULTS_PATH = ROOT_DIR / "results" / "hansei_signal_study.json"
HISTORY_START = "2006-01-01"
HORIZONS = (5, 20)
SPLITS = (("Tune 2007-2019", "2007-01-01", "2019-12-31"), ("Holdout 2020-", "2020-01-01", None))
SILENT = 0.05                  # same as memory.SILENT: smaller views are no opinion
BOOTSTRAP_DRAWS = 1000

# Large NSE companies, roughly the Nifty 100 plus the paper book's names.
# Tickers Yahoo can't serve are skipped and reported.
UNIVERSE = sorted(set(INDIA_TICKERS) | {f"{s}.NS" for s in """
ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE BAJAJFINSV BEL BHARTIARTL
CIPLA COALINDIA DRREDDY EICHERMOT GRASIM HCLTECH HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDUNILVR
ICICIBANK INDUSINDBK INFY ITC JSWSTEEL KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE
SBILIFE SBIN SHRIRAMFIN SUNPHARMA TATACONSUM TATASTEEL TCS TECHM TITAN TRENT ULTRACEMCO WIPRO
ABB ADANIGREEN ADANIPOWER AMBUJACEM BAJAJHLDNG BANKBARODA BOSCHLTD BRITANNIA CANBK CHOLAFIN DABUR
DIVISLAB DLF DMART GAIL GODREJCP HAVELLS HAL ICICIGI ICICIPRULI INDHOTEL IOC INDIGO IRFC JINDALSTEL
LICI LODHA NAUKRI PIDILITIND PFC PNB RECLTD SIEMENS SHREECEM TATAPOWER TORNTPHARM TVSMOTOR UNITDSPR
VBL VEDL ZYDUSLIFE MOTHERSON BPCL CGPOWER HINDPETRO PETRONET COLPAL MARICO LUPIN AUROPHARMA
BHARATFORG CUMMINSIND MUTHOOTFIN SRF PAGEIND BERGEPAINT ICICIBANK ASHOKLEY TATACHEM
""".split()})


# ------------------------------------------------------- vectorized advisors
# The same formulas as scaata.agent.advisors, for every day at once
# (tests check they agree with the live functions).
def trend_views(closes: pd.DataFrame) -> pd.DataFrame:
    return np.tanh((closes / closes.rolling(200).mean() - 1) / 0.08)


def volatility_views(closes: pd.DataFrame, window: int = 20, baseline: int = 252) -> pd.DataFrame:
    returns = closes.pct_change(fill_method=None)
    rolling = returns.rolling(window).std()
    normal = rolling.rolling(baseline, min_periods=baseline - window + 1).median()
    ratio = rolling / normal
    return -np.tanh((ratio - 1.2).clip(lower=0.0) / 0.5)


def momentum_views(closes: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    return np.tanh((closes.shift(skip) / closes.shift(lookback) - 1) / 0.15)


def all_views(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"trend": trend_views(closes), "volatility": volatility_views(closes),
            "momentum": momentum_views(closes)}


# ------------------------------------------------------------ the scoring
def forward_excess(closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Each stock's log return over the next `horizon` sessions, minus what
    the liquid fund paid over the same calendar days."""
    forward = np.log(closes.shift(-horizon) / closes)
    days = pd.Series(closes.index, index=closes.index)
    span = (days.shift(-horizon) - days).dt.days
    cash = np.log1p(pd.Series([liquid_yield(d.year) for d in closes.index], index=closes.index)) * span / 365
    return forward.sub(cash, axis=0)


def forward_volatility(closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Annualized volatility of each stock's daily returns over the next `horizon` sessions."""
    returns = np.log(closes / closes.shift())
    return returns.rolling(horizon).std().shift(-horizon) * np.sqrt(248)


def _monthly_spreads(view: pd.DataFrame, excess: pd.DataFrame, brake: bool = False) -> pd.DataFrame:
    """Per calendar month: sums and counts of forward excess return for liked
    and disliked stock-days. A brake (the volatility advisor, never positive)
    is scored as warned versus every other day instead."""
    v, r = view.stack(), excess.stack()
    frame = pd.concat({"view": v, "ret": r}, axis=1).dropna()
    if not brake:
        frame = frame[frame["view"].abs() >= SILENT]
    frame["month"] = frame.index.get_level_values(0).to_period("M")
    frame["liked"] = frame["view"] > (-SILENT if brake else 0)
    g = frame.groupby(["month", "liked"])["ret"].agg(["sum", "count"]).unstack(fill_value=0)
    return g


def score(view: pd.DataFrame, excess: pd.DataFrame, rng: np.random.Generator, brake: bool = False,
          future_vol: pd.DataFrame | None = None) -> dict:
    """Spread (liked minus disliked, per horizon, in % of forward excess
    return), hit rate, rank correlation, and a month-block bootstrap of the
    spread. With `future_vol`, also how volatile liked and disliked stocks
    were over the horizon."""
    g = _monthly_spreads(view, excess, brake)
    if g.empty or ("count", True) not in g or ("count", False) not in g:
        return {"opinions": 0}

    def spread(rows) -> float:
        s = rows.sum()
        liked = s[("sum", True)] / s[("count", True)] if s[("count", True)] else np.nan
        disliked = s[("sum", False)] / s[("count", False)] if s[("count", False)] else np.nan
        return float(liked - disliked)

    point = spread(g)
    months = len(g)
    draws = np.array([spread(g.iloc[rng.integers(0, months, months)]) for _ in range(BOOTSTRAP_DRAWS)])
    draws = draws[np.isfinite(draws)]
    v, r = view.stack(), excess.stack()
    both = pd.concat({"view": v, "ret": r}, axis=1).dropna()
    opinions = both[both["view"].abs() >= SILENT]
    s = g.sum()
    swings = {}
    if future_vol is not None:
        fv = pd.concat({"view": view.stack(), "vol": future_vol.stack()}, axis=1).dropna()
        liked = fv["view"] > (-SILENT if brake else SILENT)
        disliked = fv["view"] <= -SILENT
        swings = {"liked_future_volatility": float(fv.loc[liked, "vol"].mean()),
                  "disliked_future_volatility": float(fv.loc[disliked, "vol"].mean())}
    return {**swings,
        "opinions": int(len(opinions)),
        "liked_share": float(s[("count", True)] / (s[("count", True)] + s[("count", False)])),
        "liked_excess": float(s[("sum", True)] / s[("count", True)]),
        "disliked_excess": float(s[("sum", False)] / s[("count", False)]),
        "spread": point,
        "spread_ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "prob_spread_positive": float((draws > 0).mean()),
        "hit_rate": float((np.sign(opinions["view"]) == np.sign(opinions["ret"])).mean()),
        "rank_correlation": float(both["view"].rank().corr(both["ret"].rank())),
        "months": months,
    }


def study(closes: pd.DataFrame, splits=SPLITS, horizons=HORIZONS, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    views = all_views(closes)
    out = {}
    for label, start, end in splits:
        window = slice(pd.Timestamp(start), pd.Timestamp(end) if end else None)
        out[label] = {}
        for horizon in horizons:
            excess = forward_excess(closes, horizon).loc[window]
            future_vol = forward_volatility(closes, horizon).loc[window]
            out[label][f"{horizon}_sessions"] = {name: score(v.loc[window], excess, rng, name == "volatility",
                                                             future_vol)
                                                 for name, v in views.items()}
    return out


def load_universe(end: date) -> tuple[pd.DataFrame, list[str]]:
    from scaata.data.loaders import load_market_data

    raw = load_market_data(UNIVERSE, HISTORY_START, (end + pd.Timedelta(days=1)).isoformat(), use_cache=True)
    raw = raw[raw["Volume"] > 0].reset_index()
    closes = raw.pivot_table(index="Date", columns="Ticker", values="Close").sort_index()
    # Yahoo's oldest NSE prints include broken days (IOC jumped 179% and back in
    # Jul 2005): a one-day round trip of 40% or more is a bad print, not a price.
    r = np.log(closes / closes.shift())
    broken = (r.abs() > 0.4) & (r.shift(-1).abs() > 0.4) & (np.sign(r) != np.sign(r.shift(-1)))
    closes = closes.mask(broken)
    missing = sorted(set(UNIVERSE) - set(closes.columns))
    return closes, missing


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    closes, missing = load_universe(date.today())
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "stocks": int(closes.shape[1]),
              "missing": missing, "history_start": HISTORY_START,
              "large_caps": study(closes), "paper_book_stocks": study(closes[list(INDIA_TICKERS)])}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{closes.shape[1]} stocks ({len(missing)} unavailable: {', '.join(m[:-3] for m in missing) or 'none'})")
    for group in ("large_caps", "paper_book_stocks"):
        print(f"\n{group.replace('_', ' ')}: liked minus disliked, forward return over cash")
        for split, horizons in report[group].items():
            for horizon, cards in horizons.items():
                for name, c in cards.items():
                    if not c.get("opinions"):
                        continue
                    lo, hi = c["spread_ci95"]
                    print(f"  {split:<15}{horizon.replace('_', ' '):<12}{name:<11}"
                          f"spread {c['spread']:+.2%} [{lo:+.2%}, {hi:+.2%}]  P(>0) {c['prob_spread_positive']:.0%}"
                          f"  hit {c['hit_rate']:.0%}  rank corr {c['rank_correlation']:+.3f}"
                          f"  ({c['opinions']:,} opinions, {c['liked_share']:.0%} liked)"
                          f"  swings after: liked {c['liked_future_volatility']:.0%}"
                          f" disliked {c['disliked_future_volatility']:.0%}")
    print(f"\nsaved {RESULTS_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
