"""Equity cash-and-carry opportunity notification content (Phase 17) --
parallel to `scaata.notify.email_draft` but for the options-based
conversion/reversal opportunities `scaata.live.equity_carry_monitor`
detects, rather than crypto funding-rate opportunities. Same scope
boundary: a pure function, no side effects, no send.
"""
from __future__ import annotations


def build_equity_opportunity_email(opportunity: dict) -> dict:
    """Builds `{"subject", "body"}` from an equity cash-and-carry
    opportunity row (the shape `scaata.live.equity_carry_monitor.
    scan_symbol_for_carry` produces). Raises `ValueError` if
    `opportunity["opportunity"]` isn't truthy.
    """
    if not opportunity.get("opportunity"):
        raise ValueError("build_equity_opportunity_email called on a row that isn't a detected opportunity")

    symbol = opportunity.get("symbol", "UNKNOWN")
    direction = opportunity.get("direction")
    implied_rate = opportunity.get("implied_rate")
    risk_free_rate = opportunity.get("risk_free_rate")
    strike = opportunity.get("strike")
    expiration = opportunity.get("expiration")
    timestamp = opportunity.get("timestamp_utc", "unknown time")

    direction_desc = {
        "conversion": "conversion (buy stock + buy put, sell call) -- implied financing rate is too high",
        "reversal": "reversal (sell stock short + sell put, buy call) -- implied financing rate is too low",
    }.get(direction, "direction unavailable")

    gap = None
    if implied_rate is not None and risk_free_rate is not None:
        gap = implied_rate - risk_free_rate

    subject = f"Cash-and-carry opportunity: {symbol} {strike:.2f} strike, exp {expiration} ({direction})"

    body_lines = [
        f"An options-based cash-and-carry opportunity was detected for {symbol} at {timestamp}.",
        "",
        f"Suggested trade: {direction_desc}",
        f"Strike: {strike}",
        f"Expiration: {expiration}",
        f"Implied financing rate: {implied_rate:+.2%}" if implied_rate is not None else "Implied financing rate: n/a",
        f"Risk-free rate (T-bill proxy): {risk_free_rate:+.2%}" if risk_free_rate is not None else "Risk-free rate: n/a",
    ]
    if gap is not None:
        body_lines.append(f"Gap: {gap:+.2%}")
    body_lines += [
        "",
        "This is derived from put-call parity (C - P = S - K*e^(-rT)) and ignores dividends -- "
        "review the actual dividend schedule, current bid/ask spreads, and borrow costs before acting; "
        "this is a computed signal from a snapshot quote, not a guaranteed executable price.",
    ]

    return {"subject": subject, "body": "\n".join(body_lines)}
