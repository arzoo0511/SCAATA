"""Equity cash-and-carry via put-call parity -- conversion/reversal
arbitrage (Phase 17). A stock's call and put prices at the same strike and
expiry, together with the spot price, imply a market financing rate:

    C - P = S - K * e^(-rT)   =>   r_implied = -ln((S - C + P) / K) / T

When the options market's implied rate diverges meaningfully from the
actual risk-free rate, a genuinely (near-)riskless spread exists:
- **Conversion** (buy stock + buy put + sell call, same strike/expiry):
  profits when r_implied > actual rate -- the options market is paying you
  to effectively lend money above the going rate.
- **Reversal** (short stock + sell put + buy call): profits the other way.

Honest, stated simplification: this ignores dividends. The full relation
with dividends is `C - P = (S - PV(dividends)) - K*e^(-rT)`; a stock
paying a dividend before expiry will show a persistent negative bias in
`r_implied` that this doesn't separate from a genuine mispricing.
Near-term expiries (see `scaata.data.options.fetch_option_chain`'s
`max_days_to_expiry`) partially limit how much dividend exposure can
accumulate, but don't eliminate it -- this is flagged, not hidden, the
same way the crypto module's own "always correct side" simplification was
flagged rather than quietly presented as a clean result.
"""
from __future__ import annotations

import math
from datetime import date


def implied_financing_rate(spot: float, call_mid: float, put_mid: float, strike: float, time_to_expiry_years: float) -> float | None:
    """Solves put-call parity for the market-implied risk-free rate.
    Returns None for a degenerate input (non-positive strike/expiry, or a
    quote combination that would require taking the log of a non-positive
    number) rather than raising or returning a nonsensical rate.
    """
    if time_to_expiry_years <= 0 or strike <= 0:
        return None
    discounted_strike_proxy = spot - call_mid + put_mid
    if discounted_strike_proxy <= 0:
        return None
    return -math.log(discounted_strike_proxy / strike) / time_to_expiry_years


def detect_conversion_opportunity(
    implied_rate: float,
    risk_free_rate: float,
    threshold: float,
    symbol: str,
    strike: float,
    expiration: date | str,
) -> dict:
    """Flags an opportunity only when the implied-vs-actual gap (annualized)
    exceeds `threshold` -- small gaps are ordinary bid-ask-spread noise on
    illiquid strikes, not a real edge. Returns
    {opportunity, direction, symbol, strike, expiration, implied_rate,
    risk_free_rate, gap}; `direction` is "conversion" (implied rate too
    high) or "reversal" (implied rate too low), None when no opportunity.
    """
    gap = implied_rate - risk_free_rate
    opportunity = abs(gap) >= threshold
    direction = ("conversion" if gap > 0 else "reversal") if opportunity else None
    expiration_str = expiration.isoformat() if hasattr(expiration, "isoformat") else expiration

    return {
        "opportunity": opportunity,
        "direction": direction,
        "symbol": symbol,
        "strike": strike,
        "expiration": expiration_str,
        "implied_rate": implied_rate,
        "risk_free_rate": risk_free_rate,
        "gap": gap,
    }
