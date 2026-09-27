"""SCAATA live dashboard (Phase 16d) — a local Streamlit app so the project
can be checked without going through Claude for every look. Pulls live data
directly from the same functions the scheduled tasks use
(scaata.live.daily_signal, scaata.live.alpaca_broker, scaata.notify.
opportunity_alert) — nothing here is a separate reimplementation.

Deliberately read-only: no button on this page ever submits an order or
sends an email. It shows the same notify-only signals the scheduled tasks
already produce, for the same reason those tasks never auto-execute --
placing a trade or sending a message is a real-world action a human
should take deliberately, not something a dashboard button should do for
you. Run with: streamlit run scaata/dashboard/app.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from scaata.config import ALL_TICKERS, EQUITY_CARRY_SYMBOLS, SIGNAL_PRICE_GAP_BUFFER, STOP_LOSS_PCT
from scaata.live.alpaca_broker import get_account_snapshot, get_all_positions
from scaata.live.daily_signal import compute_daily_signal, format_signal_message, preview_daily_signal, read_signal_log
from scaata.live.equity_carry_monitor import read_equity_carry_log
from scaata.notify.equity_opportunity_alert import check_for_equity_opportunities
from scaata.rl.retrain import load_retrain_metadata

st.set_page_config(page_title="SCAATA Dashboard", page_icon="📈", layout="wide")

# Re-renders the page every 60s so it stays current with whatever the
# Task-Scheduler-launched background scripts (scaata.live.run_all_daily_signals,
# the cash-and-carry monitor) have written to the log files since the page
# loaded -- no button press needed. This only re-reads local CSVs on each
# tick (cheap); it does NOT re-hit the Alpaca/Binance APIs, which stays a
# deliberate, on-demand "Refresh now"/"Run scan now" action.
st_autorefresh(interval=60_000, key="dashboard_autorefresh")

# Ticker -> saved policy path. All 9 ALL_TICKERS are listed; a ticker whose
# .zip doesn't exist yet (still training) shows as "not trained yet" below
# rather than being omitted, so the dashboard's shape doesn't shift as
# training completes for each one.
TICKER_MODELS = {t: f"forward_test/policy_{t}.zip" for t in ALL_TICKERS}

st.title("📈 SCAATA Dashboard")
st.caption(f"Local, read-only, live-data view — no trade or email is ever sent from this page. Loaded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# --- Account ---
st.header("Alpaca Account")
account, account_source = get_account_snapshot()
positions, positions_source = get_all_positions()

if account_source == "mock":
    st.warning("ALPACA_API_KEY/ALPACA_SECRET_KEY not detected — showing mock data.")

c1, c2, c3 = st.columns(3)
c1.metric("Equity", f"${account['equity']:,.2f}")
c2.metric("Cash", f"${account['cash']:,.2f}")
c3.metric("Buying Power", f"${account['buying_power']:,.2f}")

if positions:
    st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
else:
    st.info("No open positions.")

st.divider()

# --- Daily trading signals ---
st.header("Daily Trading Signals")
st.caption("Notify-only — tells you what the frozen policy would do. Never places an order. Placing it is up to you.")

# Portfolio-level summary, computed from each ticker's latest logged
# signal -- caught live: every actionable BUY used to be sized
# independently at ~100% of cash, so a day with several actionable BUYs
# at once looked fine card-by-card but needed far more capital in total
# than the account actually has. This adds up what today's suggestions
# actually require in total, at a glance, instead of making you total up
# 9 separate cards yourself.
actionable_buys = []
for ticker in TICKER_MODELS:
    log = read_signal_log(ticker)
    if log.empty:
        continue
    latest = log.iloc[-1]
    if bool(latest.get("actionable", False)) and latest["action"] == "BUY":
        # Cost at a gap-cushioned price, not the logged close: current_price
        # is the PREVIOUS session's close (signals run pre-market) while the
        # fill is hours later, so costing at the raw close would understate
        # what these orders actually need and could show a reassuring "fits
        # within your cash" for a basket that doesn't.
        worst_case_price = latest["current_price"] * (1 + SIGNAL_PRICE_GAP_BUFFER)
        actionable_buys.append((ticker, latest["suggested_qty"] * worst_case_price))

if actionable_buys:
    total_needed = sum(cost for _, cost in actionable_buys)
    over_budget = total_needed > account["cash"]
    # Streamlit's markdown renders $...$ as LaTeX -- two dollar amounts in
    # one string got silently swallowed into a broken math block. Escaped
    # \$ here, and st.metric (never LaTeX-parsed) for the actual numbers.
    tickers_str = ", ".join(t for t, _ in actionable_buys)
    if over_budget:
        st.warning(f"Today's actionable BUYs ({tickers_str}) need more cash than you have — the per-ticker suggestions below are already sized to split what's available, not the full amount each.")
    else:
        st.success(f"Today's actionable BUYs ({tickers_str}) fit within your available cash.")
    m1, m2 = st.columns(2)
    m1.metric("Total needed for today's BUYs", f"${total_needed:,.0f}")
    m2.metric("Cash available", f"${account['cash']:,.0f}")
    st.caption(
        f"Cost estimated at last close +{SIGNAL_PRICE_GAP_BUFFER:.0%}. Signals are computed pre-market, "
        "so the prices shown are the previous session's close, not your fill price."
    )

for ticker, model_path in TICKER_MODELS.items():
    with st.container(border=True):
        col_header, col_button = st.columns([4, 1])
        col_header.subheader(ticker)

        if not Path(model_path).exists():
            col_button.button("Refresh now", key=f"refresh_{ticker}", disabled=True)
            st.caption("Not trained yet.")
            continue

        refresh = col_button.button("Refresh now", key=f"refresh_{ticker}")

        if refresh:
            with st.spinner(f"Computing live signal for {ticker}..."):
                from sb3_contrib import RecurrentPPO
                model = RecurrentPPO.load(model_path)

                # Portfolio-coherent sizing for a single-ticker refresh too --
                # caught live: clicking Refresh now for one ticker used the
                # default all-in fraction, silently overwriting that
                # ticker's log with a stale, non-coherent 100%-of-cash
                # suggestion even while several OTHER tickers' logged
                # signals were still actionable BUYs the same day. Preview
                # this ticker, count it against every other ticker's most
                # recently *logged* actionable-BUY status (cheap -- no live
                # inference needed for the other 8), and split accordingly.
                preview = preview_daily_signal(ticker, model)
                other_actionable_buys = 0
                for other_ticker in TICKER_MODELS:
                    if other_ticker == ticker:
                        continue
                    other_log = read_signal_log(other_ticker)
                    if not other_log.empty:
                        other_latest = other_log.iloc[-1]
                        if bool(other_latest.get("actionable", False)) and other_latest["action"] == "BUY":
                            other_actionable_buys += 1
                this_is_actionable_buy = preview["actionable"] and preview["action"] == "BUY"
                total_actionable_buys = other_actionable_buys + (1 if this_is_actionable_buy else 0)
                capital_fraction = 1.0 / total_actionable_buys if total_actionable_buys > 0 else 1.0

                row = compute_daily_signal(ticker, model, capital_fraction=capital_fraction)
            st.code(format_signal_message(row), language=None)
        else:
            log = read_signal_log(ticker)
            if log.empty:
                st.info("No signal computed yet — click Refresh now.")
            else:
                latest = log.iloc[-1]
                action_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}.get(latest["action"], "gray")
                st.markdown(f"Last computed: **{latest['timestamp_utc']}** — :{action_color}[**{latest['action']}**] at ${latest['current_price']:.2f}")
                if latest.get("norm_source") == "missing":
                    st.warning("Signal withheld: this policy has no saved training normalization stats, so the model was not run.")
                elif not bool(latest.get("actionable", False)):
                    st.caption("Not actionable given current position — no action needed.")
                    if pd.notna(latest.get("fallback_action")):
                        st.info(
                            f"**Fallback (buy-and-hold, not a DSR signal):** {latest['fallback_action']} "
                            f"{latest['fallback_qty']:.0f} shares — this policy isn't trading {ticker} at all, "
                            "and research this session found plain buy-and-hold beats a collapsed policy here."
                        )
                elif latest["action"] == "BUY":
                    fraction = latest.get("capital_fraction", 1.0)
                    fraction_note = "" if pd.isna(fraction) or fraction >= 1.0 else f" ({fraction:.0%} of cash — split across today's other actionable BUYs)"
                    st.caption(f"Suggested: {latest['suggested_qty']:.0f} shares{fraction_note}, stop-loss ${latest['stop_loss_price']:.2f}")
                elif latest["action"] == "SELL":
                    st.caption(f"Suggested: close {latest['suggested_qty']:.0f} shares")

            if len(log) > 1:
                with st.expander(f"History ({len(log)} checks)"):
                    st.dataframe(log.tail(30), use_container_width=True, hide_index=True)

st.divider()

# --- Scheduled retraining ---
st.header("Scheduled Retraining")
st.caption(
    "SCAATA-WeeklyRetrain retrains every ticker on fresh data, but only ever replaces the live policy if the "
    "fresh candidate does not measurably regress against what's currently deployed on a real recent holdout "
    "window neither model trained on -- a regression is never deployed silently. This section is read-only."
)
retrain_rows = []
for ticker in ALL_TICKERS:
    meta = load_retrain_metadata(ticker)
    if meta is None:
        retrain_rows.append({"ticker": ticker, "status": "never retrained (still on the original policy)"})
        continue
    retrain_rows.append({
        "ticker": ticker,
        "retrained_at_utc": meta["retrained_at_utc"],
        "deployed": meta["deployed"],
        "candidate_sharpe": meta["candidate_sharpe"],
        "incumbent_sharpe": meta["incumbent_sharpe"],
        "buy_and_hold_sharpe": meta["buy_and_hold_sharpe"],
        "data_source": meta.get("data_source", "live"),  # older metadata predates this field
    })
st.dataframe(pd.DataFrame(retrain_rows), use_container_width=True, hide_index=True)

st.divider()

# --- Cash-and-carry ---
st.header("Cash-and-Carry (Equity Options Conversion/Reversal) Monitor")
st.caption(
    "Put-call parity on liquid mega-cap options -- flags when the implied financing rate "
    "(from live call/put quotes) diverges from the real risk-free rate (13-week T-bill) by more than "
    "the threshold. This on-screen check is read-only — it never sends anything, same as every other "
    "button on this page. Real alerts are sent automatically (real SMTP email, no draft/review step) "
    "by the SCAATA-CashAndCarry Windows Task Scheduler job every 4 hours, independent of this dashboard "
    "and independent of Claude."
)

if st.button("Run scan now"):
    with st.spinner("Fetching live option chains and risk-free rate..."):
        drafts = check_for_equity_opportunities()
    if drafts:
        st.success(f"{len(drafts)} real opportunity(ies) detected this scan — the scheduled job will have emailed you about these already.")
        for d in drafts:
            st.markdown(f"**{d['subject']}**")
            st.text(d["body"])
    else:
        st.info("No opportunity above threshold right now.")

cols = st.columns(len(EQUITY_CARRY_SYMBOLS))
for col, symbol in zip(cols, EQUITY_CARRY_SYMBOLS):
    log = read_equity_carry_log(symbol)
    if log.empty:
        col.info(f"{symbol}: no scan history yet.")
        continue
    latest = log.iloc[-1]
    implied_rate = latest.get("implied_rate")
    rate_source = latest.get("rate_source")
    col.metric(
        f"{symbol} implied rate",
        f"{float(implied_rate):.2%}" if pd.notna(implied_rate) else "n/a",
        delta="Opportunity!" if bool(latest["opportunity"]) else None,
    )
    if rate_source != "live":
        col.caption(f"⚠️ risk-free rate source: {rate_source} — opportunity check withheld until a live rate is available")
    col.caption(f"Last checked: {latest['timestamp_utc']}")
    with col.expander("History"):
        st.dataframe(
            log.tail(30)[["timestamp_utc", "implied_rate", "risk_free_rate", "opportunity", "rate_source"]],
            use_container_width=True, hide_index=True,
        )

st.divider()
st.caption(
    "All fully autonomous — Windows Task Scheduler, not Claude (check via the Task Scheduler app): "
    "SCAATA-DailySignals (weekday mornings) keeps the trading-signal logs updated, "
    "SCAATA-Dashboard relaunches this page at logon, "
    "SCAATA-CashAndCarry (every 4h) checks equity options for a conversion/reversal opportunity and sends "
    "a real email on a genuine one, no draft/review step, "
    "SCAATA-WeeklyRetrain (weekly) retrains every ticker on fresh data and deploys only non-regressing "
    "candidates, emailing a summary either way. Nothing on this page ever places a trade or sends anything itself."
)
