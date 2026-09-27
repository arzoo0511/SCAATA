"""Standalone entry point (Phase 18) for the fully autonomous weekly
retraining loop -- meant to be launched by Windows Task Scheduler, not
Claude. Retrains every ticker in ALL_TICKERS, deploys only what clears the
safety gate in `scaata.rl.retrain`, and emails a summary either way so a
candidate that got correctly REJECTED is just as visible as one that got
deployed -- silence here would defeat the point of having a gate at all.

Fails loudly (exit code 1, so Task Scheduler's LastTaskResult shows it) when
the summary email can't be sent. Caught in the logs: from August 2026 every
run skipped all 9 tickers and the email failed with SMTP 535 -- nobody saw
either, for six weeks.

Usage: python -m scaata.live.run_scheduled_retrain
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from scaata.notify.email_sender import send_email
from scaata.rl.retrain import run_scheduled_retrain

load_dotenv()


def _format_summary(results: list[dict]) -> str:
    lines = ["Weekly SCAATA retrain summary:", ""]
    if not any(r["status"] == "evaluated" for r in results):
        lines += ["WARNING: no ticker was evaluated this run -- live policies were not checked or updated.", ""]
    for r in results:
        if r["status"] in ("skipped", "error"):
            lines.append(f"{r['ticker']}: {r['status'].upper()} -- {r['reason']}")
            continue
        status = "DEPLOYED" if r["deployed"] else "KEPT INCUMBENT"
        incumbent = f"{r['incumbent_sharpe']:.3f}" if r["incumbent_sharpe"] is not None else "n/a (first run)"
        p_inc = r.get("prob_beats_incumbent")
        p_inc_text = f"{p_inc:.2f}" if p_inc is not None else "n/a"
        data_note = "" if r.get("data_source") == "live" else f" [data: {r.get('data_source')}, {r.get('staleness_days')}d old]"
        lines.append(
            f"{r['ticker']}: {status} -- holdout Sharpe candidate {r['candidate_sharpe']:.3f}, "
            f"incumbent {incumbent}, buy&hold {r['buy_and_hold_sharpe']:.3f}; "
            f"P(candidate better) vs incumbent {p_inc_text}, vs buy&hold {r['prob_beats_buy_and_hold']:.2f}."
            f"{data_note} {r['reason']}"
        )
    return "\n".join(lines)


def main() -> None:
    results = run_scheduled_retrain()
    summary = _format_summary(results)
    print(summary)

    recipient = os.environ.get("GMAIL_SENDER_ADDRESS")
    if not recipient:
        print("\nGMAIL_SENDER_ADDRESS not configured -- summary printed above only, nothing emailed.", file=sys.stderr)
        return

    deployed_count = sum(1 for r in results if r.get("deployed"))
    subject = f"SCAATA weekly retrain: {deployed_count}/{len(results)} ticker(s) updated"
    try:
        _, send_source = send_email(recipient, subject, summary)
    except Exception as e:
        print(f"\nEMAIL FAILED -- nobody was notified of this run: {e!r}. Check the Gmail app password in .env.", file=sys.stderr)
        sys.exit(1)
    print(f"\nEmail send_source: {send_source}")


if __name__ == "__main__":
    main()
