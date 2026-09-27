"""Standalone entry point (Phase 16e) for computing the daily signal for
every trained ticker and logging the result — meant to be launched by
Windows Task Scheduler, not by Claude. Pure Python, no MCP/Claude tool
dependency: it only touches the Alpaca API and local files, exactly like
running it by hand would. The dashboard (scaata/dashboard/app.py) reads
the resulting logs and displays them, so "compute" and "display" are
decoupled -- this script can run on a schedule with nobody watching, and
the dashboard just shows whatever it last wrote.

Never places an order (same guarantee as scaata.live.daily_signal itself:
no submit_paper_order import exists on this path).

Two-phase run (Phase 16e-fix), fixing a real bug caught live: sizing each
ticker's signal independently at ~100% of cash meant that on a day with
several actionable BUYs at once (a real, observed case: GOOGL, AMZN, NVDA,
and META all BUY the same day), acting on more than one needed far more
capital than the account actually has. Phase 1 previews every ticker's
action (no state saved, no log written) to count today's actionable BUYs;
phase 2 does the real, logged computation with capital split evenly across
that count, so the suggestions are coherent as a set.

Usage: python -m scaata.live.run_all_daily_signals
"""
from __future__ import annotations

import sys
from pathlib import Path

from sb3_contrib import RecurrentPPO

from scaata.config import ALL_TICKERS
from scaata.live.daily_signal import compute_daily_signal, format_signal_message, preview_daily_signal

FORWARD_TEST_DIR = Path(__file__).resolve().parent.parent.parent / "forward_test"


def main() -> None:
    models = {}
    for ticker in ALL_TICKERS:
        model_path = FORWARD_TEST_DIR / f"policy_{ticker}.zip"
        if not model_path.exists():
            print(f"{ticker}: no trained model yet, skipping.")
            continue
        try:
            models[ticker] = RecurrentPPO.load(str(model_path))
        except Exception as e:
            print(f"{ticker}: FAILED to load model -- {e}", file=sys.stderr)

    actionable_buy_count = 0
    for ticker, model in models.items():
        try:
            preview = preview_daily_signal(ticker, model)
            if preview["actionable"] and preview["action"] == "BUY":
                actionable_buy_count += 1
        except Exception as e:
            # A single ticker's preview failure must not stop the rest --
            # it just won't count toward today's split; its real pass below
            # will hit (and report) the same error independently.
            print(f"{ticker}: FAILED during preview -- {e}", file=sys.stderr)

    capital_fraction = 1.0 / actionable_buy_count if actionable_buy_count > 0 else 1.0
    if actionable_buy_count > 1:
        print(f"{actionable_buy_count} actionable BUYs today -- splitting available cash {capital_fraction:.0%} each.\n")

    for ticker, model in models.items():
        try:
            row = compute_daily_signal(ticker, model, capital_fraction=capital_fraction)
            print(format_signal_message(row))
        except Exception as e:
            # A single ticker's failure (e.g. a transient data-fetch error)
            # must not stop the rest of the run -- print and move on rather
            # than let one bad ticker silently kill the whole scheduled job.
            print(f"{ticker}: FAILED -- {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
