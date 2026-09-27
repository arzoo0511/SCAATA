"""A simulated (paper) book for the Indian market.

Fake money, real prices, real costs. Nothing here talks to a broker: Zerodha
has no paper-trading mode, and this project does not place orders. The point
is to learn what the agent does in live market conditions at zero risk --
and, just as importantly, to run an equal-weight buy-and-hold leg of the
same size beside it, because "did it make money" is the wrong question when
simply holding the same stocks is the thing to beat.

Two rules keep this honest:

1. **Orders fill at the NEXT session's open, never at the close that
   produced the signal.** A decision made after Wednesday's close can only
   be filled at Thursday's open. Filling at the deciding close is the
   look-ahead that flatters every naive backtest.
2. **Every fill pays real NSE costs** -- statutory charges (STT, stamp duty,
   exchange and SEBI fees, GST) plus spread, the same
   `INDIA_STATUTORY_COST_PER_SIDE + INDIA_SPREAD_COST` the walk-forward
   evaluation charges, on both legs.

The book is one JSON file so a run that dies mid-way never leaves half a
trade behind.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scaata.config import INDIA_SPREAD_COST, INDIA_STATUTORY_COST_PER_SIDE, ROOT_DIR

BOOK_PATH = ROOT_DIR / "paper_book_india.json"
COST_PER_SIDE = INDIA_STATUTORY_COST_PER_SIDE + INDIA_SPREAD_COST
BUY, SELL, TARGET = "BUY", "SELL", "TARGET"

# Idle cash is assumed parked in a liquid ETF (LIQUIDBEES-style) -- what an
# Indian investor does with money that is not in stocks -- so stepping out
# of a stock earns something rather than nothing. Approximate annual yields,
# net of the fund's ~0.6% expense ratio, tracking the RBI policy rate; the
# latest year's figure is used for later years. Parking and unparking costs
# are negligible for a debt ETF (no STT) and are ignored.
LIQUID_FUND_YIELD = {2019: 0.053, 2020: 0.028, 2021: 0.028, 2022: 0.042, 2023: 0.059,
                     2024: 0.060, 2025: 0.052, 2026: 0.047}


def liquid_yield(year: int) -> float:
    year = min(max(year, min(LIQUID_FUND_YIELD)), max(LIQUID_FUND_YIELD))
    return LIQUID_FUND_YIELD[year]


def new_book(symbols: list[str], initial_cash: float = 10_000.0) -> dict:
    """A fresh book. Each symbol gets an equal sleeve of the starting cash,
    so one name can never quietly consume the whole account."""
    return {
        "currency": "INR", "initial_cash": initial_cash, "symbols": list(symbols),
        "sleeve": initial_cash / len(symbols),
        "cash": initial_cash, "positions": {}, "pending": [], "trades": [], "equity_history": [],
        "benchmark": {"cash": initial_cash, "positions": {}, "started": False},
    }


def load_book(path: Path | None = None) -> dict | None:
    path = path or BOOK_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_book(book: dict, path: Path | None = None) -> None:
    (path or BOOK_PATH).write_text(json.dumps(book, indent=2), encoding="utf-8")


def queue_order(book: dict, symbol: str, action: str, reason: str, on: date) -> None:
    """Queues an order for the next session's open. Replaces any earlier
    unfilled order for the same symbol -- the newest signal wins."""
    book["pending"] = [p for p in book["pending"] if p["symbol"] != symbol]
    book["pending"].append({"symbol": symbol, "action": action, "reason": reason, "queued_on": on.isoformat()})


def queue_target(book: dict, symbol: str, target_value: float, reason: str, on: date) -> None:
    """Queues "move this stock to about `target_value` rupees" for the next
    open -- a partial buy or sell, sized at fill time from the actual price."""
    book["pending"] = [p for p in book["pending"] if p["symbol"] != symbol]
    book["pending"].append({"symbol": symbol, "action": TARGET, "target_value": round(target_value, 2),
                            "reason": reason, "queued_on": on.isoformat()})


def _fill_target(book: dict, order: dict, price: float) -> tuple[str, int, float] | None:
    held = book["positions"].get(order["symbol"], {"qty": 0, "avg_price": price})
    wanted = int(order["target_value"] // (price * (1 + COST_PER_SIDE)))
    change = wanted - held["qty"]
    if change > 0:
        change = min(change, int(book["cash"] // (price * (1 + COST_PER_SIDE))))
    if change == 0:
        return None
    qty = abs(change)
    cost = qty * price * COST_PER_SIDE
    if change > 0:
        book["cash"] -= qty * price + cost
        new_qty = held["qty"] + qty
        held["avg_price"] = (held["avg_price"] * held["qty"] + price * qty) / new_qty
        held["qty"] = new_qty
        book["positions"][order["symbol"]] = held
        return BUY, qty, cost
    book["cash"] += qty * price - cost
    held["qty"] -= qty
    if held["qty"] <= 0:
        book["positions"].pop(order["symbol"], None)
    else:
        book["positions"][order["symbol"]] = held
    return SELL, qty, cost


def execute_pending(book: dict, opens: dict[str, float], on: date) -> list[dict]:
    """Fills queued orders at today's open, paying costs. Orders whose symbol
    has no open price today stay queued."""
    filled, still_pending = [], []
    for order in book["pending"]:
        symbol, action = order["symbol"], order["action"]
        price = opens.get(symbol)
        if price is None or price <= 0:
            still_pending.append(order)
            continue
        position = book["positions"].get(symbol)
        if action == TARGET:
            result = _fill_target(book, order, price)
            if result is None:
                continue
            action, qty, cost = result
        elif action == BUY and not position:
            budget = min(book["sleeve"], book["cash"])
            qty = int(budget / (price * (1 + COST_PER_SIDE)))
            if qty <= 0:
                continue  # sleeve can't afford a single share; drop the order
            cost = qty * price * COST_PER_SIDE
            book["cash"] -= qty * price + cost
            book["positions"][symbol] = {"qty": qty, "avg_price": price}
        elif action == SELL and position:
            qty = position["qty"]
            cost = qty * price * COST_PER_SIDE
            book["cash"] += qty * price - cost
            del book["positions"][symbol]
        else:
            continue  # BUY while already holding, or SELL while flat: a no-op
        trade = {"date": on.isoformat(), "symbol": symbol, "action": action, "qty": qty,
                 "price": price, "cost": round(cost, 2), "reason": order["reason"],
                 "cash_after": round(book["cash"], 2)}
        book["trades"].append(trade)
        filled.append(trade)
    book["pending"] = still_pending
    return filled


def start_benchmark(book: dict, opens: dict[str, float], on: date) -> None:
    """Buys the equal-weight buy-and-hold leg once, at the same opens and
    costs the agent pays, so both legs start from the same ₹10,000 on the
    same morning."""
    benchmark = book["benchmark"]
    if benchmark["started"]:
        return
    usable = {s: p for s, p in opens.items() if p and p > 0}
    if not usable:
        return
    sleeve = book["initial_cash"] / len(book["symbols"])
    for symbol, price in usable.items():
        qty = int(sleeve / (price * (1 + COST_PER_SIDE)))
        if qty <= 0:
            continue
        benchmark["cash"] -= qty * price * (1 + COST_PER_SIDE)
        benchmark["positions"][symbol] = {"qty": qty, "avg_price": price}

    # Whole-share rounding can strand a lot of cash (a Rs 2,000 sleeve buys
    # one Rs 1,337 share), which would quietly hand the agent an easier
    # benchmark. Spend what's left, one share at a time, on whichever
    # holding sits furthest below its equal-weight target.
    while True:
        affordable = {s: p for s, p in usable.items() if p * (1 + COST_PER_SIDE) <= benchmark["cash"]}
        if not affordable:
            break
        gap = {s: sleeve - benchmark["positions"].get(s, {"qty": 0})["qty"] * p for s, p in affordable.items()}
        symbol = max(gap, key=gap.get)
        price = affordable[symbol]
        benchmark["cash"] -= price * (1 + COST_PER_SIDE)
        holding = benchmark["positions"].setdefault(symbol, {"qty": 0, "avg_price": price})
        holding["qty"] += 1
    benchmark["started"] = True
    benchmark["started_on"] = on.isoformat()


def accrue_cash_yield(book: dict, on: date) -> float:
    """Credits liquid-fund interest on idle cash, in both legs, for the
    calendar days since it was last credited. Idempotent within a day.
    Returns what the agent's cash earned."""
    last = book.get("cash_accrued_through")
    book["cash_accrued_through"] = on.isoformat()
    if last is None:
        return 0.0
    days = (on - date.fromisoformat(last)).days
    if days <= 0:
        return 0.0
    growth = (1 + liquid_yield(on.year)) ** (days / 365) - 1
    earned = book["cash"] * growth
    book["cash"] += earned
    if book["benchmark"]["started"]:
        book["benchmark"]["cash"] *= 1 + growth
    return earned


def _equity(cash: float, positions: dict, closes: dict[str, float]) -> float:
    return cash + sum(p["qty"] * closes.get(s, p["avg_price"]) for s, p in positions.items())


def mark_to_market(book: dict, closes: dict[str, float], on: date) -> dict:
    """Records today's value of both legs. One row per date; re-running the
    same day overwrites rather than duplicating."""
    row = {
        "date": on.isoformat(),
        "equity": round(_equity(book["cash"], book["positions"], closes), 2),
        "benchmark_equity": round(_equity(book["benchmark"]["cash"], book["benchmark"]["positions"], closes), 2),
        "cash": round(book["cash"], 2), "holdings": len(book["positions"]),
    }
    book["equity_history"] = [r for r in book["equity_history"] if r["date"] != row["date"]] + [row]
    book["equity_history"].sort(key=lambda r: r["date"])
    return row


def summary(book: dict) -> dict:
    start = book["initial_cash"]
    history = book["equity_history"]
    if not history:
        return {"equity": start, "return": 0.0, "benchmark_equity": start, "benchmark_return": 0.0,
                "trades": 0, "holdings": 0, "days": 0, "vs_benchmark": 0.0}
    latest = history[-1]
    agent_return = latest["equity"] / start - 1
    benchmark_return = latest["benchmark_equity"] / start - 1 if book["benchmark"]["started"] else 0.0
    return {
        "equity": latest["equity"], "return": agent_return,
        "benchmark_equity": latest["benchmark_equity"], "benchmark_return": benchmark_return,
        "vs_benchmark": agent_return - benchmark_return,
        "trades": len(book["trades"]), "holdings": len(book["positions"]), "days": len(history),
        "cash": latest["cash"],
    }
