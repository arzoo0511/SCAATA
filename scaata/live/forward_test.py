"""Phase 8 — live paper-trading forward test.

This is the strongest available evidence against look-ahead leakage: a
policy trained entirely on historical data makes a real decision on a day
it had zero opportunity to have seen during development, and that
decision (plus its outcome) is logged permanently, not just printed once.

Safe to call as many times as you want, whenever you want -- once a real
(source="live") decision has been placed for a ticker on the current US
market day, every further call that same day is a no-op that just returns
the existing decision instead of submitting a second real order (a
double-run submitting two independent sell orders is exactly how you end
up accidentally short 2 shares instead of 1). Mock-mode calls are never
rate-limited this way, since no real order is ever at stake in mock mode.
Every logged row is source-tagged "live"/"mock" (see
`scaata/live/alpaca_broker.py`) so a forward-test result can never be
silently confused with a harness smoke-test.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from scaata.config import DATA_CACHE_DIR, FEATURE_COLUMNS
from scaata.features.normalize import normalize_data
from scaata.features.technical import add_features
from scaata.live.alpaca_broker import get_account_snapshot, get_latest_daily_bars, submit_paper_order
from scaata.rl.env import BUY, HOLD, SELL

FORWARD_TEST_DIR = DATA_CACHE_DIR.parent / "forward_test"
FORWARD_TEST_DIR.mkdir(exist_ok=True)


def _model_path(ticker: str) -> Path:
    return FORWARD_TEST_DIR / f"policy_{ticker}.zip"


def _state_path(ticker: str) -> Path:
    return FORWARD_TEST_DIR / f"lstm_state_{ticker}.pkl"


def _log_path(ticker: str) -> Path:
    return FORWARD_TEST_DIR / f"decision_log_{ticker}.csv"


def train_or_load_policy(
    ticker: str, train_df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS,
    seed: int = 0, total_timesteps: int = 100_000,
) -> RecurrentPPO:
    """Trains once and persists to disk; every later call just loads the
    saved policy, so the forward test's daily decisions come from one
    fixed, frozen model rather than silently retraining on data the model
    would then have "seen" before deciding."""
    path = _model_path(ticker)
    if path.exists():
        return RecurrentPPO.load(str(path))

    from scaata.rl.train import train_ppo
    model = train_ppo(train_df, feature_columns, seed=seed, total_timesteps=total_timesteps)
    model.save(str(path))
    return model


def _load_lstm_state(ticker: str):
    path = _state_path(ticker)
    if not path.exists():
        return None, np.array([True])
    with open(path, "rb") as f:
        state = pickle.load(f)
    return state, np.array([False])


def _save_lstm_state(ticker: str, state) -> None:
    with open(_state_path(ticker), "wb") as f:
        pickle.dump(state, f)


def run_daily_decision(
    ticker: str, model: RecurrentPPO, feature_columns: list[str] = FEATURE_COLUMNS, qty: float = 1.0,
) -> dict:
    """Runs one forward-test decision: fetch latest data -> compute
    features -> ask the frozen policy for an action -> submit a paper
    order -> append a permanent, source-tagged row to this ticker's log.
    Returns the logged row as a dict.

    If a real (source="live") decision was already placed for `ticker`
    today, returns that existing row unchanged instead of submitting a
    second real order -- see the module docstring."""
    existing = _already_decided_today_live(ticker)
    if existing is not None:
        print(
            f"Already placed a live decision for {ticker} today "
            f"({existing['timestamp_utc']}, action={existing['action']}) -- "
            "returning that instead of submitting a duplicate order. Re-run after "
            "the next US market day begins for a new one."
        )
        return existing

    bars, data_source = get_latest_daily_bars(ticker, lookback_days=90)
    featured = add_features(bars.reset_index())
    # Normalize against the same trailing window used at inference time
    # (no separate train split here -- this is live data, not a backtest).
    normed, _, _, _ = normalize_data(featured, featured, feature_columns)
    latest_row = normed.iloc[-1]
    obs = latest_row[feature_columns].values.astype(np.float32)

    lstm_state, episode_start = _load_lstm_state(ticker)
    action, new_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)
    _save_lstm_state(ticker, new_state)

    action = int(action)
    action_desc = {BUY: "buy", SELL: "sell", HOLD: "hold"}[action]

    order_result = {"order_id": None, "status": "no action (hold)"}
    order_source = data_source
    if action != HOLD:
        order_result, order_source = submit_paper_order(ticker, action_desc, qty)

    account, account_source = get_account_snapshot()

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "action": action_desc,
        "close_price": float(bars["Close"].iloc[-1]),
        "order_id": order_result.get("order_id"),
        "order_status": order_result.get("status"),
        "account_equity": account["equity"],
        "data_source": data_source,
        "order_source": order_source,
        "account_source": account_source,
        "overall_source": "live" if {data_source, order_source, account_source} == {"live"} else "mock",
    }

    log_path = _log_path(ticker)
    log_df = pd.DataFrame([row])
    log_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return row


def read_decision_log(ticker: str) -> pd.DataFrame:
    path = _log_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=[
            "timestamp_utc", "ticker", "action", "close_price", "order_id", "order_status",
            "account_equity", "data_source", "order_source", "account_source", "overall_source",
        ])
    return pd.read_csv(path)


def _already_decided_today_live(ticker: str) -> dict | None:
    """Returns today's already-logged live decision for `ticker`, if one
    exists (US/Eastern trading day, matching how the rest of this project
    aligns to US market days) -- so a second call the same day can return
    it instead of submitting a duplicate real order."""
    log = read_decision_log(ticker)
    live_log = log[log["overall_source"] == "live"]
    if live_log.empty:
        return None

    eastern_dates = pd.to_datetime(live_log["timestamp_utc"], utc=True).dt.tz_convert("America/New_York").dt.date
    today = pd.Timestamp.now(tz="America/New_York").date()
    todays_rows = live_log[eastern_dates == today]
    if todays_rows.empty:
        return None
    return todays_rows.iloc[-1].to_dict()
