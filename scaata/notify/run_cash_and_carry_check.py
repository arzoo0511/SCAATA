"""Standalone entry point (Phase 16f, repointed to equities in Phase 17)
for the fully autonomous cash-and-carry check -- meant to be launched by
Windows Task Scheduler, not Claude. Pure Python: real Alpaca options-chain
data, real Gmail SMTP send when a genuine opportunity is detected, zero
MCP/Claude tool dependency. This is what "live, no Claude, no button
presses" actually means for this alert.

Runs the equity (options conversion/reversal) scan per explicit user
request to move cash-and-carry from crypto funding to normal stocks -- the
crypto scan (`scaata.notify.opportunity_alert`) still exists and still
works, it's just no longer what this scheduled job checks.

Sends to GMAIL_SENDER_ADDRESS itself (a self-notification), since that's
the account authorized to send. Falls back to a no-op mock send (nothing
actually goes out) if GMAIL_APP_PASSWORD/GMAIL_SENDER_ADDRESS aren't
configured -- printed clearly rather than failing silently.

Usage: python -m scaata.notify.run_cash_and_carry_check
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from scaata.notify.equity_opportunity_alert import send_equity_opportunity_alerts

load_dotenv()


def main() -> None:
    recipient = os.environ.get("GMAIL_SENDER_ADDRESS")
    if not recipient:
        print("GMAIL_SENDER_ADDRESS not configured -- nothing to send to, exiting.", file=sys.stderr)
        return

    results = send_equity_opportunity_alerts(recipient)
    if not results:
        print("No cash-and-carry opportunity above threshold this check.")
        return

    for r in results:
        status = "SENT" if r["send_source"] == "live" else "MOCK (credentials missing, not actually sent)"
        print(f"{status}: {r['subject']}")


if __name__ == "__main__":
    main()
