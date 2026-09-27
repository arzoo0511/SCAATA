"""Real options-chain data (Phase 17) for equity cash-and-carry detection
via put-call parity -- uses Alpaca's own options market data API (the
account already connected for stock trading/quotes, no new account or key
needed) rather than a new external dependency. Same mock-fallback
convention as `scaata.live.alpaca_broker`: no keys means every function
returns an empty/zero placeholder tagged source="mock", never silently
mistaken for real data.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv

from scaata.config import DATA_CACHE_DIR

load_dotenv()

RISK_FREE_RATE_CACHE_PATH = DATA_CACHE_DIR / "risk_free_rate_cache.json"
RISK_FREE_RATE_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # T-bill yields move slowly; a day-old rate is still a real rate

# OCC option symbol format: {root}{YYMMDD}{C|P}{strike, 8 digits, *1000}
# e.g. "AAPL260729P00240000" -> AAPL, 2026-07-29, Put, strike 240.000
OCC_SYMBOL_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$")

OPTION_CHAIN_COLUMNS = ["occ_symbol", "underlying", "expiration", "option_type", "strike", "bid", "ask", "mid"]


def parse_occ_symbol(symbol: str) -> dict:
    """Parses an OCC option symbol into {underlying, expiration (date),
    option_type ('call'/'put'), strike (float)}. Raises ValueError on a
    symbol that doesn't match the expected format, rather than silently
    returning a partially-wrong parse."""
    m = OCC_SYMBOL_RE.match(symbol)
    if not m:
        raise ValueError(f"not a recognized OCC option symbol: {symbol!r}")
    expiration = date(2000 + int(m["yy"]), int(m["mm"]), int(m["dd"]))
    option_type = "call" if m["cp"] == "C" else "put"
    strike = int(m["strike"]) / 1000.0
    return {"underlying": m["root"], "expiration": expiration, "option_type": option_type, "strike": strike}


def _options_client():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    from alpaca.data.historical.option import OptionHistoricalDataClient

    return OptionHistoricalDataClient(api_key, secret_key)


def fetch_option_chain(symbol: str, max_days_to_expiry: int, client=None) -> tuple[pd.DataFrame, str]:
    """Returns (DataFrame[occ_symbol, underlying, expiration, option_type,
    strike, bid, ask, mid], source). Only contracts with a real two-sided
    quote (bid and ask both > 0) are kept -- an unquoted/illiquid contract
    contributes no usable price, not a zero.

    `client` is injectable for testing; production callers should omit it
    (falls back to a real `OptionHistoricalDataClient` built from
    `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`, or mock mode if those aren't set).
    """
    client = client if client is not None else _options_client()
    if client is None:
        return pd.DataFrame(columns=OPTION_CHAIN_COLUMNS), "mock"

    from alpaca.data.requests import OptionChainRequest

    max_date = (date.today() + timedelta(days=max_days_to_expiry)).isoformat()
    request = OptionChainRequest(underlying_symbol=symbol, expiration_date_lte=max_date)
    chain = client.get_option_chain(request)

    rows = []
    for occ_symbol, contract in chain.items():
        quote = contract.latest_quote
        if quote is None or not quote.bid_price or not quote.ask_price or quote.bid_price <= 0 or quote.ask_price <= 0:
            continue
        parsed = parse_occ_symbol(occ_symbol)
        bid, ask = float(quote.bid_price), float(quote.ask_price)
        rows.append({
            "occ_symbol": occ_symbol,
            "underlying": parsed["underlying"],
            "expiration": parsed["expiration"],
            "option_type": parsed["option_type"],
            "strike": parsed["strike"],
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
        })
    return pd.DataFrame(rows, columns=OPTION_CHAIN_COLUMNS), "live"


def _load_cached_risk_free_rate() -> float | None:
    if not RISK_FREE_RATE_CACHE_PATH.exists():
        return None
    with open(RISK_FREE_RATE_CACHE_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    age_seconds = time.time() - payload["cached_at_unix"]
    if age_seconds > RISK_FREE_RATE_CACHE_MAX_AGE_SECONDS:
        return None
    return payload["rate"]


def _save_cached_risk_free_rate(rate: float) -> None:
    with open(RISK_FREE_RATE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"rate": rate, "cached_at_unix": time.time()}, f)


def fetch_risk_free_rate() -> tuple[float, str]:
    """Free proxy for the short-term risk-free rate: the 13-week Treasury
    bill yield (`^IRX` via yfinance, already a project dependency -- no
    new data source). Returns (rate as a decimal, e.g. 0.0525, source).

    Cached for `RISK_FREE_RATE_CACHE_MAX_AGE_SECONDS` (24h): T-bill yields
    move slowly, so a day-old rate is a real, usable rate, not a stale
    guess -- and this directly fixes a real bug caught live, where a
    yfinance rate-limit (this session hit yfinance very heavily) silently
    fell back to a 0.0% mock rate on *every single scan*, which then made
    every ordinary implied rate look like a false-positive "opportunity"
    (see the regression test in scaata.live.equity_carry_monitor's test
    suite). Only falls through to (0.0, "mock") -- a real, visible zero,
    not a silent wrong answer -- when there's no usable cache AND the live
    fetch also fails.
    """
    cached = _load_cached_risk_free_rate()
    if cached is not None:
        return cached, "live"

    import yfinance as yf

    try:
        history = yf.Ticker("^IRX").history(period="5d")
        if history.empty:
            return 0.0, "mock"
        rate = float(history["Close"].iloc[-1]) / 100.0
        _save_cached_risk_free_rate(rate)
        return rate, "live"
    except Exception:
        return 0.0, "mock"
