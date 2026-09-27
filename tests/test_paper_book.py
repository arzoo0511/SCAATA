"""Tests for the India paper book: next-open fills, real costs on both legs,
sleeve limits, and an honest agent-vs-buy-and-hold comparison."""
from datetime import date

import pytest

from scaata.live.paper_book import (
    BUY, COST_PER_SIDE, SELL, execute_pending, mark_to_market, new_book, queue_order, start_benchmark, summary,
)

SYMBOLS = ["A.NS", "B.NS"]
DAY1, DAY2, DAY3 = date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 25)


def _book():
    return new_book(SYMBOLS, initial_cash=10_000.0)


def test_each_symbol_gets_an_equal_sleeve():
    book = _book()
    assert book["sleeve"] == 5_000.0
    assert book["cash"] == 10_000.0


def test_order_fills_at_the_next_open_not_the_signal_close():
    book = _book()
    queue_order(book, "A.NS", BUY, "signal", DAY1)
    # Nothing is filled on the day the signal was produced...
    assert book["positions"] == {}
    # ...it fills at the next session's open.
    filled = execute_pending(book, {"A.NS": 100.0}, DAY2)

    assert len(filled) == 1
    assert filled[0]["price"] == 100.0
    assert book["positions"]["A.NS"]["qty"] == int(5_000 / (100.0 * (1 + COST_PER_SIDE)))


def test_buy_pays_costs_and_never_exceeds_its_sleeve():
    book = _book()
    queue_order(book, "A.NS", BUY, "signal", DAY1)
    execute_pending(book, {"A.NS": 100.0}, DAY2)

    position = book["positions"]["A.NS"]
    spent = 10_000.0 - book["cash"]
    assert spent <= book["sleeve"]
    assert spent == pytest.approx(position["qty"] * 100.0 * (1 + COST_PER_SIDE))


def test_sell_closes_the_position_and_pays_costs():
    book = _book()
    queue_order(book, "A.NS", BUY, "signal", DAY1)
    execute_pending(book, {"A.NS": 100.0}, DAY2)
    cash_after_buy = book["cash"]
    qty = book["positions"]["A.NS"]["qty"]

    queue_order(book, "A.NS", SELL, "signal", DAY2)
    execute_pending(book, {"A.NS": 110.0}, DAY3)

    assert book["positions"] == {}
    assert book["cash"] == pytest.approx(cash_after_buy + qty * 110.0 * (1 - COST_PER_SIDE))


def test_redundant_orders_are_no_ops():
    book = _book()
    queue_order(book, "A.NS", SELL, "nothing held", DAY1)
    assert execute_pending(book, {"A.NS": 100.0}, DAY2) == []  # SELL while flat

    queue_order(book, "A.NS", BUY, "signal", DAY2)
    execute_pending(book, {"A.NS": 100.0}, DAY3)
    queue_order(book, "A.NS", BUY, "again", DAY3)
    assert execute_pending(book, {"A.NS": 100.0}, DAY3) == []  # BUY while already holding


def test_a_newer_signal_replaces_an_unfilled_one():
    book = _book()
    queue_order(book, "A.NS", BUY, "first", DAY1)
    queue_order(book, "A.NS", SELL, "changed mind", DAY1)
    assert [o["action"] for o in book["pending"]] == [SELL]


def test_missing_price_keeps_the_order_queued():
    book = _book()
    queue_order(book, "A.NS", BUY, "signal", DAY1)
    assert execute_pending(book, {"B.NS": 50.0}, DAY2) == []
    assert len(book["pending"]) == 1


def test_benchmark_buys_everything_once_with_the_same_costs():
    book = _book()
    start_benchmark(book, {"A.NS": 100.0, "B.NS": 200.0}, DAY2)
    first = dict(book["benchmark"]["positions"])

    start_benchmark(book, {"A.NS": 999.0, "B.NS": 999.0}, DAY3)  # must not buy again

    assert book["benchmark"]["positions"] == first
    assert set(first) == set(SYMBOLS)
    spent = 10_000.0 - book["benchmark"]["cash"]
    expected = sum(p["qty"] * p["avg_price"] * (1 + COST_PER_SIDE) for p in first.values())
    assert spent == pytest.approx(expected)


def test_marking_twice_in_a_day_does_not_duplicate_history():
    book = _book()
    mark_to_market(book, {"A.NS": 100.0}, DAY2)
    mark_to_market(book, {"A.NS": 101.0}, DAY2)

    assert len(book["equity_history"]) == 1
    assert book["equity_history"][0]["equity"] == 10_000.0  # all cash, nothing held


def test_summary_compares_the_agent_against_buy_and_hold():
    book = _book()
    queue_order(book, "A.NS", BUY, "signal", DAY1)
    execute_pending(book, {"A.NS": 100.0, "B.NS": 100.0}, DAY2)
    start_benchmark(book, {"A.NS": 100.0, "B.NS": 100.0}, DAY2)

    # A doubles, B is flat: the agent holds only A, buy-and-hold holds both.
    mark_to_market(book, {"A.NS": 200.0, "B.NS": 100.0}, DAY3)
    result = summary(book)

    assert result["return"] > 0
    assert result["benchmark_return"] > 0
    assert result["vs_benchmark"] == pytest.approx(result["return"] - result["benchmark_return"])
    assert result["trades"] == 1
    assert result["days"] == 1


def test_benchmark_invests_almost_all_of_the_cash():
    """Whole-share rounding must not leave a big idle cash pile -- that would
    make buy-and-hold an artificially easy bar for the agent to clear."""
    book = _book()
    start_benchmark(book, {"A.NS": 1337.0, "B.NS": 137.0}, DAY2)

    leftover = book["benchmark"]["cash"]
    cheapest = 137.0 * (1 + COST_PER_SIDE)
    assert leftover < cheapest          # nothing affordable is left unbought
    assert leftover / 10_000 < 0.02     # and the drag is small


def test_target_order_buys_up_to_the_target_value():
    from scaata.live.paper_book import queue_target

    book = _book()
    queue_target(book, "A.NS", 3_000.0, "brain", DAY1)
    filled = execute_pending(book, {"A.NS": 100.0}, DAY2)

    assert filled[0]["action"] == BUY
    assert book["positions"]["A.NS"]["qty"] == int(3_000 // (100.0 * (1 + COST_PER_SIDE)))


def test_target_order_trims_partially_and_keeps_the_rest():
    from scaata.live.paper_book import queue_target

    book = _book()
    queue_target(book, "A.NS", 4_000.0, "brain", DAY1)
    execute_pending(book, {"A.NS": 100.0}, DAY2)
    held = book["positions"]["A.NS"]["qty"]

    queue_target(book, "A.NS", 2_000.0, "trim", DAY2)
    filled = execute_pending(book, {"A.NS": 100.0}, DAY3)

    assert filled[0]["action"] == SELL
    assert 0 < book["positions"]["A.NS"]["qty"] < held


def test_target_of_zero_exits_completely():
    from scaata.live.paper_book import queue_target

    book = _book()
    queue_target(book, "A.NS", 2_000.0, "brain", DAY1)
    execute_pending(book, {"A.NS": 100.0}, DAY2)
    queue_target(book, "A.NS", 0.0, "exit", DAY2)
    execute_pending(book, {"A.NS": 90.0}, DAY3)

    assert "A.NS" not in book["positions"]


def test_target_buy_never_spends_more_cash_than_the_book_has():
    from scaata.live.paper_book import queue_target

    book = _book()
    queue_target(book, "A.NS", 50_000.0, "too big", DAY1)
    execute_pending(book, {"A.NS": 100.0}, DAY2)

    assert book["cash"] >= 0
