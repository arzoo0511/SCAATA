"""Ties `scaata.live.equity_carry_monitor.run_equity_carry_scan` (detection)
to `scaata.notify.equity_email_draft.build_equity_opportunity_email`
(content) into one "check now, hand back a draft-ready email for every real
opportunity found" step -- the equity-options counterpart to
`scaata.notify.opportunity_alert`, per explicit user request to move
cash-and-carry from crypto funding to normal stocks.

Same two-tier shape as the crypto version: `check_for_equity_opportunities`
never sends anything (dashboard's "Run scan now" button uses this);
`send_equity_opportunity_alerts` is the real-send counterpart, used only by
the standalone Task-Scheduler job (`scaata.notify.run_cash_and_carry_check`).
"""
from __future__ import annotations

from scaata.config import EQUITY_CARRY_RATE_THRESHOLD, EQUITY_CARRY_SYMBOLS
from scaata.live.equity_carry_monitor import run_equity_carry_scan
from scaata.notify.email_sender import send_email
from scaata.notify.equity_email_draft import build_equity_opportunity_email


def check_for_equity_opportunities(
    symbols: list[str] = EQUITY_CARRY_SYMBOLS, threshold: float = EQUITY_CARRY_RATE_THRESHOLD
) -> list[dict]:
    """Runs one real equity carry scan (also logs every symbol's result to
    its permanent CSV, same as calling `run_equity_carry_scan` directly)
    and returns pre-built `{subject, body}` email content only for symbols
    where a real opportunity was detected this call -- an empty list means
    no genuine opportunity exists right now, not a failure.
    """
    results = run_equity_carry_scan(symbols=symbols, threshold=threshold)
    drafts = []
    for row in results:
        if row["opportunity"]:
            email = build_equity_opportunity_email(row)
            drafts.append({"scan_row": row, "subject": email["subject"], "body": email["body"]})
    return drafts


def send_equity_opportunity_alerts(
    recipient: str, symbols: list[str] = EQUITY_CARRY_SYMBOLS, threshold: float = EQUITY_CARRY_RATE_THRESHOLD
) -> list[dict]:
    """Real-send counterpart to `check_for_equity_opportunities`: for every
    genuinely detected opportunity this call, actually emails `recipient`
    via `scaata.notify.email_sender.send_email` (source="mock", nothing
    sent, if Gmail credentials aren't configured). Returns one dict per
    opportunity: `{"scan_row", "subject", "body", "send_source"}`. An
    empty list means no opportunity was detected -- the common/expected
    outcome most calls, since a genuine >=300bps mispricing on liquid
    mega-cap options is rare by construction.
    """
    opportunities = check_for_equity_opportunities(symbols=symbols, threshold=threshold)
    results = []
    for opp in opportunities:
        _, send_source = send_email(recipient, opp["subject"], opp["body"])
        results.append({**opp, "send_source": send_source})
    return results
