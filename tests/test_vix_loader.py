"""Test for the VIX loader (Phase 9c): must reuse the existing
download/cache/clean pipeline and rename `Close` to `vix_close`, without
touching the network in this test (network calls are mocked the same way
`tests/test_alpaca_broker.py` mocks `load_market_data`).
"""
import pandas as pd

from scaata.data import loaders


def _fake_vix_raw():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {"Open": 15.0, "High": 16.0, "Low": 14.0, "Close": [15.0, 16.0, 14.0, 20.0, 18.0], "Volume": 0, "Ticker": "^VIX"},
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_load_vix_renames_close_and_reuses_pipeline(monkeypatch):
    monkeypatch.setattr(loaders, "collect_data", lambda symbols, start, end, use_cache=True: _fake_vix_raw())

    result = loaders.load_vix("2024-01-01", "2024-01-08")

    assert "vix_close" in result.columns
    assert "Close" not in result.columns
    assert list(result["vix_close"]) == [15.0, 16.0, 14.0, 20.0, 18.0]
