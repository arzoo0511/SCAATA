"""Equity cash-and-carry monitor (Phase 17) -- scans a stock's nearest-
expiry, near-the-money options for a genuine conversion/reversal
opportunity via put-call parity. Logs every scan to a permanent,
source-tagged CSV. Replaces an earlier crypto funding-rate version, removed
per explicit user request (equity-only going forward, not crypto).

Never auto-emails: `scaata.notify.equity_opportunity_alert` is the
drafting/sending layer, invoked separately (by a human, a dashboard
button, or the standalone Task-Scheduler script), never from inside this
scan itself.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scaata.config import (
    DATA_CACHE_DIR,
    EQUITY_CARRY_MAX_DAYS_TO_EXPIRY,
    EQUITY_CARRY_MIN_DAYS_TO_EXPIRY,
    EQUITY_CARRY_RATE_THRESHOLD,
    EQUITY_CARRY_SYMBOLS,
)
from scaata.data.options import fetch_option_chain, fetch_risk_free_rate
from scaata.live.alpaca_broker import get_latest_daily_bars
from scaata.signals.equity_cash_and_carry import detect_conversion_opportunity, implied_financing_rate

EQUITY_CARRY_DIR = DATA_CACHE_DIR.parent / "equity_carry_monitor"
EQUITY_CARRY_DIR.mkdir(exist_ok=True)

LOG_COLUMNS = [
    "timestamp_utc", "symbol", "spot_price", "strike", "expiration", "implied_rate",
    "risk_free_rate", "opportunity", "direction", "data_complete",
    "price_source", "chain_source", "rate_source",
]


def _log_path(symbol: str) -> Path:
    return EQUITY_CARRY_DIR / f"equity_carry_log_{symbol}.csv"


def _best_near_the_money_pair(chain: pd.DataFrame, spot: float, min_days_to_expiry: int = EQUITY_CARRY_MIN_DAYS_TO_EXPIRY):
    """Picks the nearest expiry present in `chain` that is still at least
    `min_days_to_expiry` out, then within that expiry the strike closest
    to spot among strikes that have BOTH a call and a put quote (put-call
    parity needs the matched pair) -- returns None if no such pair exists
    (e.g. an entirely one-sided chain, or every expiry is too near-dated).

    The floor on tenor matters: `implied_financing_rate` divides by time-
    to-expiry in years, so a same-week expiry (T ~ 0.005 years) turns an
    ordinary few-cent bid-ask spread into several *points* of annualized
    rate noise -- caught live when a real scan flagged 5 of 6 mega-caps as
    "opportunities" purely from this, not genuine mispricing.
    """
    if chain.empty:
        return None

    min_expiry = date.today() + timedelta(days=min_days_to_expiry)
    eligible = chain[chain["expiration"] >= min_expiry]
    if eligible.empty:
        return None

    nearest_expiry = eligible["expiration"].min()
    same_expiry = eligible[eligible["expiration"] == nearest_expiry]
    calls = same_expiry[same_expiry["option_type"] == "call"].set_index("strike")
    puts = same_expiry[same_expiry["option_type"] == "put"].set_index("strike")
    common_strikes = calls.index.intersection(puts.index)
    if len(common_strikes) == 0:
        return None

    best_strike = min(common_strikes, key=lambda k: abs(k - spot))
    return calls.loc[best_strike], puts.loc[best_strike], float(best_strike), nearest_expiry


def scan_symbol_for_carry(symbol: str, threshold: float = EQUITY_CARRY_RATE_THRESHOLD) -> dict:
    """One symbol's worth of the scan. Always returns a fully-shaped row --
    any data-availability gap (no spot price, empty chain, no matched
    call/put pair, already-expired nearest date, or a failed/rate-limited
    risk-free-rate fetch) shows up as `data_complete=False` rather than
    the scan silently skipping the symbol, raising, or -- the real bug
    this guards against -- comparing a genuine implied rate against a
    fake mock risk-free rate and calling the result an "opportunity".
    `implied_rate`/`strike`/`expiration` are still populated even when the
    risk-free leg fails, since they're independently real; only the
    opportunity *comparison* is withheld.
    """
    bars, price_source = get_latest_daily_bars(symbol, lookback_days=5)
    spot = float(bars["Close"].iloc[-1]) if not bars.empty else None

    chain, chain_source = fetch_option_chain(symbol, max_days_to_expiry=EQUITY_CARRY_MAX_DAYS_TO_EXPIRY)
    risk_free_rate, rate_source = fetch_risk_free_rate()

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "spot_price": spot,
        "strike": None,
        "expiration": None,
        "implied_rate": None,
        "risk_free_rate": risk_free_rate,
        "opportunity": False,
        "direction": None,
        "data_complete": False,
        "price_source": price_source,
        "chain_source": chain_source,
        "rate_source": rate_source,
    }

    if spot is None or chain.empty:
        return row

    picked = _best_near_the_money_pair(chain, spot)
    if picked is None:
        return row
    call_row, put_row, strike, expiration = picked

    days_to_expiry = (expiration - date.today()).days
    if days_to_expiry <= 0:
        return row
    time_to_expiry_years = days_to_expiry / 365.0

    implied_rate = implied_financing_rate(spot, call_row["mid"], put_row["mid"], strike, time_to_expiry_years)
    if implied_rate is None:
        return row

    # implied_rate itself is real (from live spot + a live options chain)
    # and worth recording even if the risk-free-rate leg failed -- but
    # never call an "opportunity" against a fake baseline. Caught live:
    # a yfinance rate-limit on ^IRX silently fell back to a mock 0.0%
    # rate, which then made every ordinary positive implied rate look
    # like a huge false-positive "conversion opportunity" -- comparing
    # against an unreliable rate is worse than not comparing at all.
    row["implied_rate"] = implied_rate
    row["strike"] = strike
    row["expiration"] = expiration.isoformat()
    if rate_source != "live":
        row["data_complete"] = False
        return row

    detection = detect_conversion_opportunity(implied_rate, risk_free_rate, threshold, symbol, strike, expiration)
    row.update({
        "opportunity": detection["opportunity"],
        "direction": detection["direction"],
        "data_complete": True,
    })
    return row


def run_equity_carry_scan(
    symbols: list[str] = EQUITY_CARRY_SYMBOLS, threshold: float = EQUITY_CARRY_RATE_THRESHOLD
) -> list[dict]:
    """Scans every symbol, appends each result to its own permanent CSV
    log, and returns the list of scan-result rows."""
    results = []
    for symbol in symbols:
        row = scan_symbol_for_carry(symbol, threshold=threshold)
        results.append(row)
        log_path = _log_path(symbol)
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    opportunities = [r for r in results if r["opportunity"]]
    if opportunities:
        print(
            f"Equity cash-and-carry: {len(opportunities)} opportunity(ies) detected this scan "
            f"({', '.join(o['symbol'] for o in opportunities)})."
        )
    return results


def read_equity_carry_log(symbol: str) -> pd.DataFrame:
    path = _log_path(symbol)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path)
