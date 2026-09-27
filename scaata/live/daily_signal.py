"""Notify-only daily trading signal (Phase 16b) — computes what a frozen,
already-trained policy would do on the latest available data and returns a
fully-described recommendation for a human to act on themselves. Never
submits an order.

This is a deliberate alternative to `scaata.live.forward_test.run_daily_decision`
(Phase 8), which auto-submits a real paper order to Alpaca the moment
`action != HOLD` and API keys are configured — no confirmation step. That
auto-submit behavior is a real, functioning code path (confirmed: two live
orders were actually placed via it on a prior run), but it isn't what
should run unattended: placing any order, paper or real, is an action
Claude does not take on a user's behalf. This module produces the same
underlying signal without ever calling `submit_paper_order`.
"""
from __future__ import annotations

import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scaata.config import DATA_CACHE_DIR, FEATURE_COLUMNS, SIGNAL_DISCLAIMER, SIGNAL_PRICE_GAP_BUFFER, STOP_LOSS_PCT
from scaata.features.technical import add_features
from scaata.live.alpaca_broker import get_account_snapshot, get_latest_daily_bars, get_position_qty
from scaata.live.forward_test import latest_observation, load_policy_norm_stats
from scaata.rl.env import BUY, HOLD, SELL

NORM_MISSING = "missing"

SIGNAL_DIR = DATA_CACHE_DIR.parent / "daily_signal"
SIGNAL_DIR.mkdir(exist_ok=True)

ACTION_DESC = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}


def _lstm_state_path(ticker: str) -> Path:
    return SIGNAL_DIR / f"lstm_state_{ticker}.pkl"


def _signal_log_path(ticker: str) -> Path:
    return SIGNAL_DIR / f"signal_log_{ticker}.csv"


def _append_to_log(log_path: Path, row: dict) -> None:
    """Appends `row` to `log_path`, archiving the existing file first if its
    header doesn't match `row`'s columns. Caught live: editing this
    module's row schema (adding position_qty/actionable/position_source)
    after a log file already existed with the old schema silently produced
    a column-count-mismatched CSV that crashed on the next read -- pandas
    doesn't error on the *write* side (append just adds a wider row), only
    much later when something tries to read it back. Checking the header
    before appending catches this at write time instead.
    """
    if log_path.exists():
        existing_header = pd.read_csv(log_path, nrows=0).columns.tolist()
        if existing_header != list(row.keys()):
            archive_path = log_path.with_suffix(f".schema_changed_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.csv")
            log_path.rename(archive_path)

    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def _load_lstm_state(ticker: str):
    path = _lstm_state_path(ticker)
    if not path.exists():
        return None, np.array([True])
    with open(path, "rb") as f:
        state = pickle.load(f)
    return state, np.array([False])


def _save_lstm_state(ticker: str, state) -> None:
    with open(_lstm_state_path(ticker), "wb") as f:
        pickle.dump(state, f)


def _load_norm_stats(ticker: str, feature_columns: list[str]):
    return load_policy_norm_stats(ticker, feature_columns)


def _decide_action(ticker: str, model, feature_columns: list[str]) -> tuple[int, float, str, object, str]:
    """Runs the frozen policy once and returns `(action, current_price,
    data_source, new_lstm_state, norm_source)` -- the pure decision logic,
    with no side effects (no saved state, no log write). Factored out so
    `preview_daily_signal` can ask "what would today's action be" for every
    ticker before anything commits, without corrupting the recurrent state
    a real `compute_daily_signal` call would advance. Calling this multiple
    times with the same saved `lstm_state` is safe and deterministic
    (`deterministic=True`); only `compute_daily_signal` actually persists
    the resulting state.

    Features are scaled with the stats the policy was trained on. With no
    saved stats the policy is not run at all: the result is HOLD with
    `norm_source == NORM_MISSING` and `new_lstm_state` None, so nothing
    downstream treats it as a real signal or advances the recurrent state.
    """
    bars, data_source = get_latest_daily_bars(ticker, lookback_days=90)
    current_price = float(bars["Close"].iloc[-1])

    stats = _load_norm_stats(ticker, feature_columns)
    if stats is None:
        return HOLD, current_price, data_source, None, NORM_MISSING
    norm_mean, norm_std, provenance = stats

    featured = add_features(bars.reset_index())
    obs = latest_observation(featured, feature_columns, norm_mean, norm_std)

    lstm_state, episode_start = _load_lstm_state(ticker)
    action, new_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)
    return int(action), current_price, data_source, new_state, provenance.get("quality", "training")


def _gap_safe_qty(cash: float, capital_fraction: float, current_price: float) -> int:
    """Shares to suggest, sized against a price cushioned by
    `SIGNAL_PRICE_GAP_BUFFER` rather than the raw quote.

    `current_price` is always the previous session's close (the signal job
    runs pre-market, so that's the newest complete daily bar), but the fill
    happens hours later. Dividing the cash allocation by the stale price
    yields an order that only fits if the stock doesn't gap -- see
    `SIGNAL_PRICE_GAP_BUFFER` in config for the live case that motivated
    this. Erring toward slightly fewer shares is the safe direction: too
    few means leftover cash, too many means an order you can't fund.

    Floors at 0. Found live: once an earlier over-sized order pushed the
    account to NEGATIVE cash (-$1,954), `math.floor` on a negative
    numerator rounded *away* from zero and the dashboard cheerfully
    advised "BUY -5 shares" on several tickers. There is no such thing as
    buying a negative number of shares -- no spare cash means no suggested
    buy at all.
    """
    if cash <= 0 or current_price <= 0:
        return 0
    return max(0, math.floor(cash * capital_fraction / (current_price * (1 + SIGNAL_PRICE_GAP_BUFFER))))


def preview_daily_signal(ticker: str, model, feature_columns: list[str] = FEATURE_COLUMNS) -> dict:
    """Read-only preview of what `compute_daily_signal` would decide today:
    same action logic, but never saves LSTM state and never logs anything,
    so it's safe to call before the real pass for every ticker without
    corrupting the policy's recurrent state or writing a duplicate row.

    Exists to fix a real bug caught live: `compute_daily_signal` sized
    every actionable BUY independently at ~100% of available cash, so on a
    day with several actionable BUYs (a real, observed case: GOOGL, AMZN,
    NVDA, and META all BUY on the same day), acting on more than one would
    need far more capital than the account actually has. Callers (see
    `scaata.live.run_all_daily_signals`) use this to count today's
    actionable BUYs first, then pass a fair `capital_fraction` into the
    real call.

    A BUY the account cannot fund at ALL (no cash, or not even one share
    at 100% of it) is reported as not actionable here, matching what
    `compute_daily_signal` will decide. Without this the count is
    inflated: observed live with negative cash, three BUYs were counted
    as "actionable", capital was split 33% three ways, and then all three
    were demoted as unaffordable -- and in the mixed case (one fundable,
    two not) the fundable one would be sized at a third of what it should
    get. This deliberately only screens out the definitely-unfundable
    case; it can't resolve the general one, since affordability depends on
    the fraction and the fraction depends on this count.
    """
    action, current_price, _, _, norm_source = _decide_action(ticker, model, feature_columns)
    position_qty, _ = get_position_qty(ticker)
    is_flat = position_qty <= 0
    actionable = norm_source != NORM_MISSING and ((action == BUY and is_flat) or (action == SELL and not is_flat))

    if action == BUY and actionable:
        account, _ = get_account_snapshot()
        if _gap_safe_qty(account["cash"], 1.0, current_price) <= 0:
            actionable = False

    return {"ticker": ticker, "action": ACTION_DESC[action], "actionable": actionable}


def compute_daily_signal(
    ticker: str, model, feature_columns: list[str] = FEATURE_COLUMNS, capital_fraction: float = 1.0,
) -> dict:
    """Fetches the latest bars, runs the frozen policy, and returns a
    recommendation dict — `action` (HOLD/BUY/SELL), `current_price`,
    `position_qty` (shares actually held right now, per the real account —
    not tracked internally the way `RobustTradingEnv` tracks it during
    backtesting), `actionable` (whether this action means anything given
    the real position: BUY only means something new while flat, SELL only
    means something while holding shares — the policy's own training
    assumes exactly this gating, since `RobustTradingEnv` only opens a
    position on BUY-while-flat and only closes on SELL-while-holding),
    `suggested_qty` (BUY only, and only when actionable:
    floor(available_cash * capital_fraction / price) -- `capital_fraction`
    defaults to 1.0 (the original all-in sizing convention the backtests
    this policy was evaluated under actually use) for a single ad-hoc call
    like the dashboard's "Refresh now" button, which has no visibility into
    what other tickers are suggesting today. `scaata.live.
    run_all_daily_signals` -- the actual source of what the dashboard
    displays day to day -- passes `1 / (number of actionable BUYs today)`
    instead, so the suggestions are coherent as a set, not just
    individually. A human is free to size smaller regardless.),
    `stop_loss_price` (BUY only), and the account snapshot for context.
    Appends the same row to a permanent, source-tagged CSV log. No order
    is ever submitted.

    `fallback_action`/`fallback_qty`: when the policy has nothing
    actionable to say AND the ticker is currently flat, this session's own
    research found that a collapsed ("never trades") policy is reliably
    worse than simply buying and holding on that ticker -- so rather than
    the dashboard going silent, this surfaces the buy-and-hold-implied
    action (always BUY, since buy-and-hold is never a seller) as an
    explicitly-labeled fallback, not a DSR recommendation. Only set when
    there's nothing else to say: a real actionable DSR signal, or an
    existing position, always takes precedence and leaves this None.
    """
    action, current_price, data_source, new_state, norm_source = _decide_action(ticker, model, feature_columns)
    withheld = norm_source == NORM_MISSING
    if not withheld:
        _save_lstm_state(ticker, new_state)

    action_desc = ACTION_DESC[action]
    account, account_source = get_account_snapshot()
    position_qty, position_source = get_position_qty(ticker)

    is_flat = position_qty <= 0
    actionable = not withheld and ((action == BUY and is_flat) or (action == SELL and not is_flat))

    suggested_qty = None
    stop_loss_price = None
    unaffordable = False
    if action == BUY and actionable and current_price > 0:
        suggested_qty = _gap_safe_qty(account["cash"], capital_fraction, current_price)
        if suggested_qty <= 0:
            # A BUY you can't fund isn't something to act on. Demote it
            # rather than showing an actionable signal with a 0/negative
            # size, which is what surfaced live once cash went negative.
            actionable = False
            suggested_qty = None
            unaffordable = True
        else:
            stop_loss_price = round(current_price * (1 + STOP_LOSS_PCT), 2)
    elif action == SELL and actionable:
        suggested_qty = position_qty  # close what's actually held, not an undefined amount

    fallback_action = None
    fallback_qty = None
    if not withheld and not actionable and is_flat and current_price > 0:
        candidate_qty = _gap_safe_qty(account["cash"], capital_fraction, current_price)
        # Same rule as above: with no spare cash there is no buy-and-hold
        # fallback to suggest either, so leave it unset instead of
        # advertising a 0-share "BUY".
        if candidate_qty > 0:
            fallback_action = "BUY"
            fallback_qty = candidate_qty

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "action": action_desc,
        "current_price": current_price,
        "position_qty": position_qty,
        "actionable": actionable,
        "unaffordable": unaffordable,
        "suggested_qty": suggested_qty,
        "capital_fraction": capital_fraction,
        "stop_loss_price": stop_loss_price,
        "fallback_action": fallback_action,
        "fallback_qty": fallback_qty,
        "account_cash": account["cash"],
        "account_equity": account["equity"],
        "data_source": data_source,
        "norm_source": norm_source,
        "account_source": account_source,
        "position_source": position_source,
        "overall_source": "live" if "mock" not in (data_source, account_source, position_source) else "mock",
    }

    _append_to_log(_signal_log_path(ticker), row)
    return row


def format_signal_message(row: dict) -> str:
    """Human-readable one-block summary of a `compute_daily_signal` row —
    what to tell the user, not what to execute. Explicitly says when an
    action isn't actionable given the real position (a SELL signal while
    holding nothing, or a BUY signal while already holding, means "stay as
    you are" — the policy's own training treats it as a no-op) rather than
    implying a trade that wouldn't match what the model actually learned.
    """
    tag = "" if row["overall_source"] == "live" else " [MOCK DATA — not a live account/price]"
    lines = [f"{row['ticker']}: {row['action']} at ${row['current_price']:.2f}{tag}"]
    lines.append(f"  Current position: {row['position_qty']:g} shares")

    if row.get("norm_source") == NORM_MISSING:
        lines.append("  Signal withheld: no training normalization stats saved for this policy -- model not run.")
    elif not row["actionable"]:
        if row["action"] == "SELL":
            lines.append("  Not actionable: no open position to close -- no action needed.")
        elif row["action"] == "BUY" and row.get("unaffordable"):
            # Distinct from "already holding": the policy DOES want to buy,
            # there just isn't cash to do it. Saying "already holding"
            # here would be plainly wrong, and staying silent would hide a
            # real signal.
            lines.append("  Not actionable: no available cash to fund a buy -- signal noted, not sized.")
        elif row["action"] == "BUY":
            lines.append("  Not actionable: already holding a position -- no action needed.")
        else:
            lines.append("  No action needed.")
        if row.get("fallback_action"):
            lines.append(
                f"  Fallback (buy-and-hold, not a DSR signal): {row['fallback_action']} "
                f"{row['fallback_qty']:g} shares -- this policy isn't trading this ticker at all, "
                f"and research found plain buy-and-hold beats a collapsed policy here."
            )
    elif row["action"] == "BUY":
        fraction = row.get("capital_fraction", 1.0)
        fraction_desc = "your available cash" if fraction >= 1.0 else f"a {fraction:.0%} share of available cash (split across today's other actionable BUYs)"
        lines.append(f"  Suggested size: {row['suggested_qty']} shares (~${row['suggested_qty'] * row['current_price']:,.0f}, using {fraction_desc})")
        # current_price is the previous close; the fill is hours later, so
        # say so rather than implying the quoted price is what you'd pay.
        lines.append(f"  (sized with a {SIGNAL_PRICE_GAP_BUFFER:.0%} buffer -- quoted price is the prior close, not your fill price)")
        lines.append(f"  Suggested stop-loss: ${row['stop_loss_price']:.2f}")
    elif row["action"] == "SELL":
        lines.append(f"  Suggested: close the full position ({row['suggested_qty']:g} shares)")

    lines.append(f"  Account: ${row['account_cash']:,.0f} cash / ${row['account_equity']:,.0f} equity")
    return "\n".join(lines)


def impersonal_view(row: dict) -> dict:
    """Strips a `compute_daily_signal`-style row down to what's safe to
    publish to every subscriber watching this ticker (Phase 19) -- same
    signal for everyone, on purpose. `row` carries fields sized against
    ONE account's real cash and position (`suggested_qty`, `account_cash`,
    `position_qty`, ...); none of that means anything for a subscriber
    whose account is a different account entirely, and leaking it would
    hand a stranger a real balance that isn't theirs. This is what
    `scaata.product.distribute_signals` actually sends -- it reads what
    `scaata.live.run_all_daily_signals` already logged today rather than
    re-running the policy (and re-advancing its recurrent LSTM state) a
    second time per ticker per day.
    """
    return {
        "ticker": row["ticker"],
        "action": row["action"],
        "current_price": row["current_price"],
        "as_of_utc": row["timestamp_utc"],
        "data_source": row["data_source"],
        "norm_source": row.get("norm_source"),
        "disclaimer": SIGNAL_DISCLAIMER,
    }


def compute_impersonal_signal(ticker: str, model, feature_columns: list[str] = FEATURE_COLUMNS) -> dict:
    """Read-only, account-independent signal (Phase 19) -- the same model
    decision `compute_daily_signal` makes, reported with no account cash,
    no position, no suggested share count. Exists so the product layer can
    be exercised/tested without any account-aware caller at all; the
    actual daily distribution job reads today's already-logged row via
    `impersonal_view(read_signal_log(...))` instead of calling this, since
    `scaata.live.run_all_daily_signals` already owns the one canonical,
    state-advancing call per ticker per day. Calling this a second time in
    the same day is safe (same no-side-effects contract as
    `preview_daily_signal`: no saved state, no log write) but redundant
    for that path.
    """
    action, current_price, data_source, _, norm_source = _decide_action(ticker, model, feature_columns)
    return {
        "ticker": ticker,
        "action": ACTION_DESC[action],
        "current_price": current_price,
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "norm_source": norm_source,
        "disclaimer": SIGNAL_DISCLAIMER,
    }


def read_signal_log(ticker: str) -> pd.DataFrame:
    path = _signal_log_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=[
            "timestamp_utc", "ticker", "action", "current_price", "position_qty", "actionable",
            "unaffordable", "suggested_qty", "capital_fraction", "stop_loss_price", "fallback_action", "fallback_qty",
            "account_cash", "account_equity", "data_source", "norm_source", "account_source", "position_source", "overall_source",
        ])
    return pd.read_csv(path)
