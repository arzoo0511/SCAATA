"""Tests for the put-call-parity math (scaata.signals.equity_cash_and_carry)
and OCC symbol parsing (scaata.data.options) -- the two pure-function
layers everything else in Phase 17 builds on.
"""
import math
from datetime import date

import pytest

from scaata.data.options import parse_occ_symbol
from scaata.signals.equity_cash_and_carry import detect_conversion_opportunity, implied_financing_rate


# --- OCC symbol parsing ---

def test_parse_occ_symbol_extracts_all_fields_correctly():
    parsed = parse_occ_symbol("AAPL260729P00240000")
    assert parsed["underlying"] == "AAPL"
    assert parsed["expiration"] == date(2026, 7, 29)
    assert parsed["option_type"] == "put"
    assert parsed["strike"] == 240.0


def test_parse_occ_symbol_handles_calls_and_fractional_strikes():
    parsed = parse_occ_symbol("MSFT251003C00382500")
    assert parsed["underlying"] == "MSFT"
    assert parsed["option_type"] == "call"
    assert parsed["strike"] == 382.5


def test_parse_occ_symbol_raises_on_malformed_input():
    with pytest.raises(ValueError):
        parse_occ_symbol("not-a-real-symbol")


# --- implied_financing_rate ---

def test_implied_rate_recovers_a_known_rate_by_construction():
    """Build a call/put pair that's exactly consistent with a known 5%
    rate via put-call parity, then confirm the function recovers it."""
    spot = 100.0
    strike = 100.0
    T = 0.25  # 3 months
    true_rate = 0.05
    # C - P = S - K*e^(-rT)  =>  pick P freely, derive C
    put_mid = 3.0
    call_mid = spot - strike * math.exp(-true_rate * T) + put_mid

    recovered = implied_financing_rate(spot, call_mid, put_mid, strike, T)
    assert recovered == pytest.approx(true_rate, abs=1e-9)


def test_implied_rate_none_for_non_positive_time_to_expiry():
    assert implied_financing_rate(100.0, 5.0, 5.0, 100.0, 0.0) is None
    assert implied_financing_rate(100.0, 5.0, 5.0, 100.0, -0.1) is None


def test_implied_rate_none_for_non_positive_strike():
    assert implied_financing_rate(100.0, 5.0, 5.0, 0.0, 0.25) is None


def test_implied_rate_none_for_degenerate_quote_combination():
    """A call price far below intrinsic combined with a tiny put would
    make (S - C + P) <= 0 -- not a real rate, must return None rather than
    raise on log of a non-positive number."""
    assert implied_financing_rate(100.0, 500.0, 0.0, 50.0, 0.25) is None


# --- detect_conversion_opportunity ---

def test_no_opportunity_when_gap_below_threshold():
    result = detect_conversion_opportunity(
        implied_rate=0.052, risk_free_rate=0.05, threshold=0.03,
        symbol="AAPL", strike=200.0, expiration=date(2026, 8, 1),
    )
    assert result["opportunity"] is False
    assert result["direction"] is None


def test_conversion_flagged_when_implied_rate_too_high():
    result = detect_conversion_opportunity(
        implied_rate=0.10, risk_free_rate=0.05, threshold=0.03,
        symbol="AAPL", strike=200.0, expiration=date(2026, 8, 1),
    )
    assert result["opportunity"] is True
    assert result["direction"] == "conversion"
    assert result["gap"] == pytest.approx(0.05)


def test_reversal_flagged_when_implied_rate_too_low():
    result = detect_conversion_opportunity(
        implied_rate=0.01, risk_free_rate=0.05, threshold=0.03,
        symbol="AAPL", strike=200.0, expiration=date(2026, 8, 1),
    )
    assert result["opportunity"] is True
    assert result["direction"] == "reversal"
    assert result["gap"] == pytest.approx(-0.04)


def test_expiration_serialized_to_iso_string():
    result = detect_conversion_opportunity(
        implied_rate=0.10, risk_free_rate=0.05, threshold=0.03,
        symbol="AAPL", strike=200.0, expiration=date(2026, 8, 1),
    )
    assert result["expiration"] == "2026-08-01"
