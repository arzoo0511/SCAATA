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
