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


def get_position_qty(symbol: str, client=None) -> tuple[float, str]:
    """Returns (shares currently held, source). 0.0 (mock) when no keys are
    configured -- a fresh/empty account is the only sensible mock default,
    since there is no cached "position" data the way there is cached price
    history."""
    client = client if client is not None else _trading_client()
    if client is None:
        return 0.0, "mock"

    from alpaca.common.exceptions import APIError

    try:
        position = client.get_open_position(symbol)
        return float(position.qty), "live"
    except APIError:
        return 0.0, "live"  # a real, confirmed "no position" answer, not a fallback


def get_all_positions(client=None) -> tuple[list[dict], str]:
    """Returns ([{symbol, qty, avg_entry_price, market_value, unrealized_pl}, ...], source).
    Empty list (mock) when no keys are configured."""
    client = client if client is not None else _trading_client()
    if client is None:
        return [], "mock"

    positions = client.get_all_positions()
    return [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        }
        for p in positions
    ], "live"


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


# Free Alpaca data plans may only query SIP bars older than 15 minutes.
SIP_DELAY = pd.Timedelta(minutes=16)


def daily_bars_request(symbol: str, lookback_days: int, now: pd.Timestamp | None = None):
    """Consolidated (SIP), split- and dividend-adjusted daily bars -- the
    same kind of series yfinance's `auto_adjust=True` gives the training
    pipeline. The live path previously used the IEX feed, unadjusted: IEX
    volume measured at 2.8% of consolidated volume for MSFT (May-Jul 2026),
    so `volume_ma_30` was nowhere near anything the policies trained on.
    SIP with `adjustment=all` matched yfinance within 0.4% on volume and
    0.2% on close over the same window."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    end = now - SIP_DELAY
    start = end - pd.Timedelta(days=lookback_days * 2)
    return StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end,
        feed=DataFeed.SIP, adjustment=Adjustment.ALL,
    )


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

    bars = _reshape_bars(data_client.get_stock_bars(daily_bars_request(symbol, lookback_days)).df)
    return bars.tail(lookback_days), "live"


def _reshape_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Alpaca's (symbol, timestamp)-indexed bars -> the loaders' shape:
    naive US/Eastern trading-date index named Date, OHLCV + Ticker."""
    bars = raw.reset_index().rename(
        columns={"timestamp": "Date", "open": "Open", "high": "High", "low": "Low",
                 "close": "Close", "volume": "Volume", "symbol": "Ticker"}
    )
    dates = pd.to_datetime(bars["Date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("America/New_York").dt.tz_localize(None)
    bars["Date"] = dates.dt.normalize()
    return bars.set_index("Date")[["Open", "High", "Low", "Close", "Volume", "Ticker"]]


def get_daily_bars_range(symbol: str, start: str, end: str, data_client=None) -> tuple[pd.DataFrame | None, str]:
    """Adjusted consolidated (SIP) daily bars from `start` up to `end` (or
    the newest bar a free plan may query), shaped like
    `scaata.data.loaders.collect_data` output. Lets the weekly retrain get
    fresh data when yfinance rate-limits, from the same kind of series live
    inference uses. Returns (None, "unavailable") without credentials."""
    data_client = data_client if data_client is not None else _data_client()
    if data_client is None:
        return None, "unavailable"

    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end_ts = min(pd.Timestamp(end, tz="UTC"), pd.Timestamp.now(tz="UTC") - SIP_DELAY)
    request = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=pd.Timestamp(start, tz="UTC"), end=end_ts,
        feed=DataFeed.SIP, adjustment=Adjustment.ALL,
    )
    return _reshape_bars(data_client.get_stock_bars(request).df), "alpaca_sip"


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
