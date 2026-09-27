"""The daily paper-trading run for the Indian book.

One command per trading day:

    python -m scaata.live.india_paper

Each run, in this order:

1. **Fill yesterday's queued orders at today's open** -- never at the close
   that produced them.
2. **Start the buy-and-hold leg** on the same morning, at the same opens and
   costs, so the comparison is fair from day one.
3. **Mark both legs to today's close.**
4. **Ask the policy what to do next**, and queue those orders for tomorrow's
   open.

No broker is contacted and no order is placed: this is fake money on real
prices. The policy's recurrent state is carried across days, the same way it
would be in a live deployment.
"""
from __future__ import annotations

import pickle
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scaata.config import INDIA_TICKERS, ROOT_DIR, START_DATE
from scaata.features.normalize import apply_normalizer
from scaata.features.technical import add_features
from scaata.live.india_policy import FEATURE_COLUMNS, INCLUDE_POSITION_OBS, load_india_policy
from scaata.live.paper_book import (
    BOOK_PATH, BUY, SELL, execute_pending, load_book, mark_to_market, new_book, queue_order,
    save_book, start_benchmark, summary,
)
from scaata.rl.env import HOLD, POSITION_OBS_SIZE, HOLDING_DAYS_SCALE
from scaata.rl.env import BUY as ENV_BUY, SELL as ENV_SELL

STATE_PATH = ROOT_DIR / "forward_test" / "india_paper_lstm.pkl"
ACTION_NAME = {HOLD: "HOLD", ENV_BUY: BUY, ENV_SELL: SELL}


def _load_states() -> dict:
    return pickle.loads(STATE_PATH.read_bytes()) if STATE_PATH.exists() else {}


def _observation(featured: pd.DataFrame, mean, std, position: dict | None, last_close: float) -> np.ndarray:
    """Latest row, scaled with the policy's training stats, plus the position
    block the policy was trained with."""
    scaled = apply_normalizer(featured.iloc[[-1]], FEATURE_COLUMNS, mean, std)
    features = scaled[FEATURE_COLUMNS].values[0].astype(np.float32)
    if not INCLUDE_POSITION_OBS:
        return features
    if position:
        held_days = position.get("days_held", 0)
        block = [1.0, last_close / position["avg_price"] - 1.0, held_days / HOLDING_DAYS_SCALE]
    else:
        block = [0.0] * POSITION_OBS_SIZE
    return np.concatenate([features, np.asarray(block, dtype=np.float32)])


def run_paper_day(today: date | None = None, book_path: Path | None = None) -> dict:
    from scaata.data.loaders import load_market_data

    today = today or date.today()
    loaded = load_india_policy()
    if loaded is None:
        raise RuntimeError("No India policy yet -- run `python -m scaata.live.india_policy` first.")
    model, mean, std, _ = loaded

    # yfinance's `end` is exclusive, so ask through tomorrow to include today's bar.
    end = (today + timedelta(days=1)).isoformat()
    raw = load_market_data(list(INDIA_TICKERS), START_DATE, end, use_cache=False)
    featured = add_features(raw)
    latest_date = featured.index.max().date()

    opens, closes = {}, {}
    for symbol in INDIA_TICKERS:
        rows = featured[featured["Ticker"] == symbol]
        if rows.empty:
            continue
        opens[symbol] = float(rows["Open"].iloc[-1])
        closes[symbol] = float(rows["Close"].iloc[-1])

    book = load_book(book_path) or new_book(list(INDIA_TICKERS))
    filled = execute_pending(book, opens, latest_date)
    start_benchmark(book, opens, latest_date)
    marked = mark_to_market(book, closes, latest_date)

    # Ask the policy for tomorrow's orders.
    states = _load_states()
    decisions = {}
    for symbol in INDIA_TICKERS:
        rows = featured[featured["Ticker"] == symbol]
        if rows.empty:
            continue
        position = book["positions"].get(symbol)
        state, episode_start = states.get(symbol), np.array([symbol not in states])
        observation = _observation(rows, mean, std, position, closes[symbol])
        action, new_state = model.predict(observation, state=state, episode_start=episode_start, deterministic=True)
        states[symbol] = new_state
        name = ACTION_NAME[int(action)]
        decisions[symbol] = name
        actionable = (name == BUY and not position) or (name == SELL and position)
        if actionable:
            queue_order(book, symbol, name, f"policy {name} on {latest_date}", latest_date)
        if position:
            position["days_held"] = position.get("days_held", 0) + 1

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_bytes(pickle.dumps(states))
    book["last_run_utc"] = datetime.now(timezone.utc).isoformat()
    save_book(book, book_path)
    return {"date": str(latest_date), "filled": filled, "marked": marked, "decisions": decisions,
            "queued": list(book["pending"]), "summary": summary(book)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    result = run_paper_day()
    s = result["summary"]
    print(f"HANSEI paper book — {result['date']}  (fake money, real prices, no broker)")
    for trade in result["filled"]:
        print(f"  FILLED {trade['action']:<4} {trade['symbol']:<14} {trade['qty']:>4} @ ₹{trade['price']:,.2f} "
              f"(cost ₹{trade['cost']:.2f})")
    if not result["filled"]:
        print("  no fills today")
    print(f"  decisions: " + ", ".join(f"{k.replace('.NS','')}={v}" for k, v in result["decisions"].items()))
    if result["queued"]:
        print("  queued for tomorrow's open: " +
              ", ".join(f"{o['action']} {o['symbol'].replace('.NS','')}" for o in result["queued"]))
    print(f"  agent     ₹{s['equity']:,.2f}  ({s['return']:+.2%})   cash ₹{s['cash']:,.2f}, {s['holdings']} holdings")
    print(f"  buy&hold  ₹{s['benchmark_equity']:,.2f}  ({s['benchmark_return']:+.2%})")
    print(f"  agent vs buy&hold: {s['vs_benchmark']:+.2%} over {s['days']} day(s), {s['trades']} trade(s)")
    print(f"  book: {BOOK_PATH}")


if __name__ == "__main__":
    main()
