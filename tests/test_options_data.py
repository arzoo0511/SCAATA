"""Tests for scaata.data.options: OCC symbol parsing is covered in
test_equity_cash_and_carry.py; this covers the risk-free-rate caching
(the fix for a real bug -- a yfinance rate-limit silently produced a fake
0.0% rate on every single scan) and the option-chain fetch's mock/live
behavior.
"""
import json
import time

import pandas as pd
import pytest

import scaata.data.options as options_module
from scaata.data.options import fetch_option_chain, fetch_risk_free_rate


# --- fetch_risk_free_rate caching ---

def test_uses_cached_rate_without_calling_yfinance_when_cache_is_fresh(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(options_module, "RISK_FREE_RATE_CACHE_PATH", cache_path)
    with open(cache_path, "w") as f:
        json.dump({"rate": 0.0521, "cached_at_unix": time.time()}, f)

    def _fail_if_called(*a, **k):
        raise AssertionError("yfinance should not be called when the cache is fresh")

    monkeypatch.setattr("yfinance.Ticker", _fail_if_called)

    rate, source = fetch_risk_free_rate()
    assert rate == 0.0521
    assert source == "live"


def test_refetches_when_cache_is_stale(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(options_module, "RISK_FREE_RATE_CACHE_PATH", cache_path)
    monkeypatch.setattr(options_module, "RISK_FREE_RATE_CACHE_MAX_AGE_SECONDS", 60)
    stale_time = time.time() - 3600  # 1 hour old, cache max age is 60s
    with open(cache_path, "w") as f:
        json.dump({"rate": 0.01, "cached_at_unix": stale_time}, f)

    class _FakeHistory:
        empty = False
        def __getitem__(self, key):
            return pd.Series([5.30])

    class _FakeTicker:
        def __init__(self, symbol):
            pass
        def history(self, period):
            return _FakeHistory()

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)

    rate, source = fetch_risk_free_rate()
    assert rate == pytest.approx(0.053)
    assert source == "live"

    # the fresh rate was cached for next time
    with open(cache_path) as f:
        saved = json.load(f)
    assert saved["rate"] == pytest.approx(0.053)


def test_refetches_when_no_cache_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(options_module, "RISK_FREE_RATE_CACHE_PATH", tmp_path / "does_not_exist.json")

    class _FakeHistory:
        empty = False
        def __getitem__(self, key):
            return pd.Series([4.80])

    class _FakeTicker:
        def __init__(self, symbol):
            pass
        def history(self, period):
            return _FakeHistory()

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)

    rate, source = fetch_risk_free_rate()
    assert rate == pytest.approx(0.048)
    assert source == "live"


def test_falls_back_to_mock_zero_when_cache_missing_and_fetch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(options_module, "RISK_FREE_RATE_CACHE_PATH", tmp_path / "does_not_exist.json")

    class _FailingTicker:
        def __init__(self, symbol):
            pass
        def history(self, period):
            raise RuntimeError("rate limited")

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", _FailingTicker)

    rate, source = fetch_risk_free_rate()
    assert rate == 0.0
    assert source == "mock"


# --- fetch_option_chain ---

def test_fetch_option_chain_is_mock_without_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    chain, source = fetch_option_chain("AAPL", max_days_to_expiry=30)
    assert source == "mock"
    assert chain.empty


class _FakeQuote:
    def __init__(self, bid_price, ask_price):
        self.bid_price = bid_price
        self.ask_price = ask_price


class _FakeContract:
    def __init__(self, bid_price, ask_price):
        self.latest_quote = _FakeQuote(bid_price, ask_price) if bid_price is not None else None


class _FakeOptionsClient:
    def __init__(self, chain_dict):
        self._chain_dict = chain_dict

    def get_option_chain(self, request):
        return self._chain_dict


def test_fetch_option_chain_parses_contracts_and_skips_unquoted(monkeypatch):
    chain_dict = {
        "AAPL260805C00220000": _FakeContract(112.36, 116.56),
        "AAPL260805P00220000": _FakeContract(0.0, 0.0),  # unquoted, must be skipped
    }
    client = _FakeOptionsClient(chain_dict)

    chain, source = fetch_option_chain("AAPL", max_days_to_expiry=30, client=client)

    assert source == "live"
    assert len(chain) == 1
    row = chain.iloc[0]
    assert row["underlying"] == "AAPL"
    assert row["option_type"] == "call"
    assert row["strike"] == 220.0
    assert row["mid"] == pytest.approx((112.36 + 116.56) / 2)
