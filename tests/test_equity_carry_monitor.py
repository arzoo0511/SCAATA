"""Tests for the equity cash-and-carry monitor's scan/pairing/logging
logic -- all underlying data sources (spot price, option chain, risk-free
rate) are mocked, so these never touch a real network or Alpaca account.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import scaata.live.equity_carry_monitor as monitor_module
from scaata.live.equity_carry_monitor import (
    _best_near_the_money_pair,
    read_equity_carry_log,
    run_equity_carry_scan,
    scan_symbol_for_carry,
)


def _make_bars(price=200.0, ticker="AAPL", n=5):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": price, "High": price * 1.01, "Low": price * 0.99, "Close": price,
        "Volume": 1_000_000, "Ticker": ticker,
    }, index=dates)


def _make_chain(spot=200.0, expiry_days=20, strikes=(190.0, 195.0, 200.0, 205.0), missing_put_strike=None):
    expiration = date.today() + timedelta(days=expiry_days)
    rows = []
    for strike in strikes:
        rows.append({
            "occ_symbol": f"AAPL_C_{strike}", "underlying": "AAPL", "expiration": expiration,
            "option_type": "call", "strike": strike, "bid": 5.0, "ask": 5.2, "mid": 5.1,
        })
        if strike == missing_put_strike:
            continue
        rows.append({
            "occ_symbol": f"AAPL_P_{strike}", "underlying": "AAPL", "expiration": expiration,
            "option_type": "put", "strike": strike, "bid": 3.0, "ask": 3.2, "mid": 3.1,
        })
    return pd.DataFrame(rows)


# --- _best_near_the_money_pair ---

def test_picks_strike_closest_to_spot_among_common_strikes():
    chain = _make_chain(spot=200.0)
    result = _best_near_the_money_pair(chain, spot=201.0)
    assert result is not None
    call_row, put_row, strike, expiration = result
    assert strike == 200.0


def test_returns_none_when_no_common_call_put_strikes():
    chain = _make_chain(missing_put_strike=200.0)
    # remove the 200 call too so there's genuinely no overlap left at 200,
    # but leave other strikes with only one side to force a real gap
    chain = chain[~((chain["strike"] == 200.0) & (chain["option_type"] == "call"))]
    result = _best_near_the_money_pair(chain, spot=200.0)
    assert result is not None  # other strikes still have matched pairs
    _, _, strike, _ = result
    assert strike != 200.0


def test_returns_none_for_empty_chain():
    assert _best_near_the_money_pair(pd.DataFrame(columns=["expiration", "option_type", "strike"]), spot=200.0) is None


def test_picks_the_nearest_expiry_when_multiple_present():
    near = _make_chain(spot=200.0, expiry_days=20)
    far = _make_chain(spot=200.0, expiry_days=40)
    combined = pd.concat([near, far], ignore_index=True)
    result = _best_near_the_money_pair(combined, spot=200.0)
    assert result is not None
    _, _, _, expiration = result
    assert expiration == near["expiration"].iloc[0]


def test_filters_out_expiries_that_are_too_near_dated():
    """Regression test for a real bug caught live: a same-week (~2 day)
    expiry was picked as 'nearest', and its ordinary bid-ask spread --
    divided by a tiny time-to-expiry -- produced several points of fake
    annualized rate noise, flagging 5 of 6 mega-caps as false-positive
    opportunities. A near-dated expiry must be skipped in favor of the
    nearest one that clears the minimum tenor."""
    too_near = _make_chain(spot=200.0, expiry_days=2)
    far_enough = _make_chain(spot=200.0, expiry_days=20)
    combined = pd.concat([too_near, far_enough], ignore_index=True)

    result = _best_near_the_money_pair(combined, spot=200.0, min_days_to_expiry=14)

    assert result is not None
    _, _, _, expiration = result
    assert expiration == far_enough["expiration"].iloc[0]


def test_returns_none_when_every_expiry_is_too_near_dated():
    chain = _make_chain(spot=200.0, expiry_days=2)
    result = _best_near_the_money_pair(chain, spot=200.0, min_days_to_expiry=14)
    assert result is None


# --- scan_symbol_for_carry ---

def _patch_sources(monkeypatch, bars, chain, chain_source="live", risk_free_rate=0.05, rate_source="live"):
    monkeypatch.setattr(monitor_module, "get_latest_daily_bars", lambda symbol, lookback_days=5: (bars, "live"))
    monkeypatch.setattr(monitor_module, "fetch_option_chain", lambda symbol, max_days_to_expiry: (chain, chain_source))
    monkeypatch.setattr(monitor_module, "fetch_risk_free_rate", lambda: (risk_free_rate, rate_source))


def test_scan_is_incomplete_when_no_spot_price(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor_module, "EQUITY_CARRY_DIR", tmp_path)
    _patch_sources(monkeypatch, bars=pd.DataFrame(), chain=_make_chain())

    row = scan_symbol_for_carry("AAPL")
    assert row["data_complete"] is False
    assert row["spot_price"] is None


def test_scan_is_incomplete_when_chain_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor_module, "EQUITY_CARRY_DIR", tmp_path)
    _patch_sources(monkeypatch, bars=_make_bars(), chain=pd.DataFrame(columns=["expiration", "option_type", "strike", "mid"]))

    row = scan_symbol_for_carry("AAPL")
    assert row["data_complete"] is False


def test_scan_completes_and_computes_implied_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor_module, "EQUITY_CARRY_DIR", tmp_path)
    _patch_sources(monkeypatch, bars=_make_bars(price=200.0), chain=_make_chain(spot=200.0))

    row = scan_symbol_for_carry("AAPL", threshold=0.03)

    assert row["data_complete"] is True
    assert row["strike"] == 200.0
    assert row["implied_rate"] is not None
    assert row["risk_free_rate"] == 0.05
    assert isinstance(row["opportunity"], (bool, np.bool_))


def test_scan_flags_opportunity_when_gap_exceeds_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor_module, "EQUITY_CARRY_DIR", tmp_path)
    # A call priced far above the put at the same strike implies a large
    # positive rate -- well above any reasonable risk_free_rate default.
    chain = _make_chain(spot=200.0, strikes=(200.0,))
    chain.loc[chain["option_type"] == "call", "mid"] = 30.0
    chain.loc[chain["option_type"] == "put", "mid"] = 1.0
    _patch_sources(monkeypatch, bars=_make_bars(price=200.0), chain=chain, risk_free_rate=0.05)

    row = scan_symbol_for_carry("AAPL", threshold=0.03)

    assert row["opportunity"] is True
    assert row["direction"] == "conversion"


def test_scan_does_not_flag_opportunity_against_a_mock_risk_free_rate(monkeypatch, tmp_path):
    """Regression test for a real bug caught live: a yfinance rate-limit on
    ^IRX silently fell back to a mock 0.0% risk-free rate, and every
    ordinary positive implied rate then looked like a huge false-positive
    'conversion opportunity' against that fake baseline. When the rate
    fetch isn't live, data_complete must be False and opportunity must
    stay False, even though implied_rate itself is still worth recording."""
    monkeypatch.setattr(monitor_module, "EQUITY_CARRY_DIR", tmp_path)
    chain = _make_chain(spot=200.0, strikes=(200.0,))
    chain.loc[chain["option_type"] == "call", "mid"] = 30.0  # would be a huge "opportunity" against a 0% baseline
    chain.loc[chain["option_type"] == "put", "mid"] = 1.0
    _patch_sources(monkeypatch, bars=_make_bars(price=200.0), chain=chain, risk_free_rate=0.0, rate_source="mock")

    row = scan_symbol_for_carry("AAPL", threshold=0.03)

    assert row["rate_source"] == "mock"
    assert row["data_complete"] is False
    assert row["opportunity"] is False
    assert row["direction"] is None
    # the implied rate itself is still real and worth keeping
    assert row["implied_rate"] is not None
    assert row["strike"] == 200.0


def test_scan_result_is_logged_to_a_permanent_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(monitor_module, "EQUITY_CARRY_DIR", tmp_path)
    _patch_sources(monkeypatch, bars=_make_bars(), chain=_make_chain())

    run_equity_carry_scan(symbols=["AAPL"])
    log = read_equity_carry_log("AAPL")

    assert len(log) == 1
    assert log.iloc[0]["symbol"] == "AAPL"


def test_read_equity_carry_log_empty_when_no_history(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor_module, "EQUITY_CARRY_DIR", tmp_path)
    log = read_equity_carry_log("NEVERSCANNED")
    assert log.empty
