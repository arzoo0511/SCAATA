"""Thin wrapper around Alpaca's paper-trading API for the live forward-test
demo (Phase 8). Follows the same mock-fallback convention as
`scaata/rl/llm_baseline.py`: every function returns a `source` ("live" or
"mock") alongside its result, so a placeholder can never be mistaken for a
real broker response downstream. No `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`
means every call here runs in mock mode using the most recent cached
market data, not a live order.

This only ever targets Alpaca's *paper* trading environment (paper=True is
hardcoded, not a caller-supplied option) — there is no code path in this
module that can place a live order with real capital.
"""
from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv

from scaata.config import DATA_CACHE_DIR, INITIAL_CASH
from scaata.data.loaders import load_market_data

load_dotenv()


def _trading_client():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    from alpaca.trading.client import TradingClient

    return TradingClient(api_key, secret_key, paper=True)


def _data_client():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(api_key, secret_key)


def get_account_snapshot(client=None) -> tuple[dict, str]:
    """Returns ({equity, cash, buying_power}, source)."""
    client = client if client is not None else _trading_client()
    if client is None:
        return {"equity": INITIAL_CASH, "cash": INITIAL_CASH, "buying_power": INITIAL_CASH}, "mock"

    account = client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
    }, "live"


def get_latest_daily_bars(symbol: str, lookback_days: int = 90, data_client=None) -> tuple[pd.DataFrame, str]:
    """Returns (OHLCV DataFrame indexed by Date, source). Mock mode reuses
    the most recent cached market data for `symbol` (via the same loader
    Phase 1 uses) rather than synthetic noise, since the point here is
    proving the harness runs end-to-end, not fabricating a price series."""
    data_client = data_client if data_client is not None else _data_client()
    if data_client is None:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=lookback_days * 2)
        cached = load_market_data([symbol], start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), use_cache=True)
        return cached[cached["Ticker"] == symbol].tail(lookback_days), "mock"

    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end = pd.Timestamp.today(tz="UTC")
    start = end - pd.Timedelta(days=lookback_days * 2)
    # IEX, not the default SIP feed -- SIP's recent/real-time data requires a
    # paid market-data subscription; IEX is what free accounts are authorized
    # to query for recent bars.
    request = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end, feed=DataFeed.IEX,
    )
    bars = data_client.get_stock_bars(request).df
    bars = bars.reset_index().rename(
        columns={"timestamp": "Date", "open": "Open", "high": "High", "low": "Low",
                 "close": "Close", "volume": "Volume", "symbol": "Ticker"}
    ).set_index("Date")
    return bars.tail(lookback_days), "live"


def submit_paper_order(symbol: str, action_desc: str, qty: float, client=None) -> tuple[dict, str]:
    """`action_desc` is "buy" or "sell". Returns ({order_id, status}, source).
    Mock mode logs the intended order without contacting any broker."""
    client = client if client is not None else _trading_client()
    if client is None:
        return {"order_id": "mock-order", "status": "accepted (mock, not submitted)"}, "mock"

    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    side = OrderSide.BUY if action_desc == "buy" else OrderSide.SELL
    order_request = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
    order = client.submit_order(order_request)
    return {"order_id": str(order.id), "status": str(order.status)}, "live"
