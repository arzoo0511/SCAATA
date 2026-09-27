"""Tests for the Alpaca paper-trading wrapper: every function must run
end-to-end in mock mode (no ALPACA_API_KEY/ALPACA_SECRET_KEY) without
touching a real broker or network, and must always report source="mock"
in that mode so a placeholder can never be mistaken for a live fill.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import INITIAL_CASH
from scaata.live import alpaca_broker


def _synthetic_bars(ticker="TEST", n=30):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n), "Ticker": ticker,
    }, index=dates)
    df.index.name = "Date"
    return df


def test_get_account_snapshot_uses_mock_when_no_keys_configured(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    account, source = alpaca_broker.get_account_snapshot()

    assert source == "mock"
    assert account["equity"] == INITIAL_CASH


def test_get_latest_daily_bars_falls_back_to_cached_data(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(alpaca_broker, "load_market_data", lambda *a, **k: _synthetic_bars())

    bars, source = alpaca_broker.get_latest_daily_bars("TEST", lookback_days=10)

    assert source == "mock"
    assert len(bars) == 10
    assert "Close" in bars.columns


def test_live_bars_use_consolidated_adjusted_feed_like_the_training_data():
    """Regression test: IEX volume is ~3% of the consolidated volume the
    policies were trained on (yfinance), and IEX bars were unadjusted. Live
    bars must be SIP + fully adjusted, and old enough (>15 min) for a free
    data plan to serve."""
    from alpaca.data.enums import Adjustment, DataFeed

    now = pd.Timestamp("2026-09-14T03:00:00", tz="UTC")
    request = alpaca_broker.daily_bars_request("TEST", lookback_days=90, now=now)

    assert request.feed == DataFeed.SIP
    assert request.adjustment == Adjustment.ALL
    end = pd.Timestamp(request.end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end  # alpaca-py stores UTC as naive
    assert end <= now - pd.Timedelta(minutes=15)


def test_live_bars_are_reshaped_to_the_training_column_names():
    dates = pd.date_range("2026-01-01", periods=5, freq="B", tz="UTC")
    raw = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0},
        index=pd.MultiIndex.from_product([["TEST"], dates], names=["symbol", "timestamp"]),
    )
    captured = {}

    class _FakeDataClient:
        def get_stock_bars(self, request):
            captured["request"] = request
            return type("Resp", (), {"df": raw})()

    bars, source = alpaca_broker.get_latest_daily_bars("TEST", lookback_days=3, data_client=_FakeDataClient())

    assert source == "live"
    assert len(bars) == 3
    assert {"Open", "High", "Low", "Close", "Volume", "Ticker"} <= set(bars.columns)


def test_bar_dates_match_the_yfinance_training_index():
    """Training/holdout splits compare against naive trading dates; Alpaca
    stamps daily bars at 04:00/05:00 UTC, which must map to the NY date."""
    stamps = pd.DatetimeIndex(["2026-07-17T04:00:00Z", "2026-12-18T05:00:00Z"])
    raw = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0, "trade_count": 3, "vwap": 1.2},
        index=pd.MultiIndex.from_product([["TEST"], stamps], names=["symbol", "timestamp"]),
    )

    class _FakeDataClient:
        def get_stock_bars(self, request):
            return type("Resp", (), {"df": raw})()

    bars, source = alpaca_broker.get_daily_bars_range("TEST", "2026-01-01", "2026-12-31", data_client=_FakeDataClient())

    assert source == "alpaca_sip"
    assert list(bars.index) == [pd.Timestamp("2026-07-17"), pd.Timestamp("2026-12-18")]
    assert list(bars.columns) == ["Open", "High", "Low", "Close", "Volume", "Ticker"]


def test_daily_bars_range_is_unavailable_without_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert alpaca_broker.get_daily_bars_range("TEST", "2026-01-01", "2026-02-01") == (None, "unavailable")


def test_submit_paper_order_does_not_contact_a_broker_in_mock_mode(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    result, source = alpaca_broker.submit_paper_order("TEST", "buy", 1.0)

    assert source == "mock"
    assert "mock" in result["status"]


def test_trading_client_is_none_without_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert alpaca_broker._trading_client() is None
    assert alpaca_broker._data_client() is None


def test_get_position_qty_is_zero_mock_without_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    qty, source = alpaca_broker.get_position_qty("TEST")

    assert qty == 0.0
    assert source == "mock"


def test_get_all_positions_is_empty_mock_without_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    positions, source = alpaca_broker.get_all_positions()

    assert positions == []
    assert source == "mock"
