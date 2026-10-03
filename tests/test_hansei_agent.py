"""Tests for the HANSEI agent: advisors, the brain's decision rules, learning,
news scoring, and a full daily run on synthetic data (no network)."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scaata.agent import advisors, brain, memory as mem, news
from scaata.agent.daily import run_day
from scaata.config import INDIA_TICKERS

EQUAL = {a: 0.25 for a in advisors.ADVISORS}


def _series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# ------------------------------------------------------------ advisors
def test_trend_is_positive_in_an_uptrend_and_negative_in_a_downtrend():
    assert advisors.trend_view(_series(np.linspace(100, 200, 300))) > 0.5
    assert advisors.trend_view(_series(np.linspace(200, 100, 300))) < -0.5


def test_volatility_is_only_ever_a_brake():
    rng = np.random.default_rng(0)
    calm = _series(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)))
    assert advisors.volatility_view(calm) <= 0.0
    wild = np.r_[100 * np.cumprod(1 + rng.normal(0, 0.01, 380)), 0]
    wild[-20:] = wild[-21] * np.cumprod(1 + rng.normal(0, 0.06, 20))
    assert advisors.volatility_view(_series(wild[:400])) < -0.5


def test_views_include_news_and_default_it_to_neutral():
    closes = _series(np.linspace(100, 120, 300))
    assert advisors.views(closes, None)["news"] == 0.0
    assert advisors.views(closes, {"view": -0.7})["news"] == -0.7


# ------------------------------------------------------------ brain
def test_default_is_to_hold_fully():
    views = {"trend": 0.1, "volatility": 0.0, "momentum": 0.1, "news": 0.0}
    decision = brain.decide("ITC.NS", views, EQUAL, 1.0, None)
    assert decision.action == "HOLD"
    assert decision.target_exposure == 1.0


def test_starts_by_buying_when_it_holds_nothing():
    views = {a: 0.0 for a in advisors.ADVISORS}
    decision = brain.decide("ITC.NS", views, EQUAL, 0.0, None)
    assert decision.action == "BUY"
    assert decision.target_exposure == 1.0


def test_sells_when_the_evidence_is_clearly_against():
    views = {"trend": -0.9, "volatility": -0.8, "momentum": -0.7, "news": -0.6}
    decision = brain.decide("ITC.NS", views, EQUAL, 1.0, None)
    assert decision.action == "SELL"
    assert decision.target_exposure < 0.5
    assert any("Exiting" in r or "Trimming" in r for r in decision.reasons)


def test_small_changes_in_opinion_do_not_trade():
    views = {"trend": -0.2, "volatility": 0.0, "momentum": 0.0, "news": 0.0}   # score -0.05 -> target 0.9
    assert brain.decide("ITC.NS", views, EQUAL, 1.0, None).action == "HOLD"


def test_waits_after_a_recent_trade_unless_news_is_strong():
    bearish = {"trend": -0.9, "volatility": -0.8, "momentum": -0.7, "news": 0.0}
    assert brain.decide("ITC.NS", bearish, EQUAL, 1.0, days_since_trade=3).action == "HOLD"
    with_news = {**bearish, "news": -0.8}
    decision = brain.decide("ITC.NS", with_news, EQUAL, 1.0, days_since_trade=3)
    assert decision.action == "SELL"
    assert any("news overrode" in r for r in decision.reasons)


def test_rebalances_a_holding_that_drifted_too_far():
    views = {a: 0.1 for a in advisors.ADVISORS}
    decision = brain.decide("ITC.NS", views, EQUAL, 1.0, None, weight_in_book=0.30, equal_weight=0.20)
    assert decision.action == "REBALANCE"
    assert decision.target_exposure == 1.0


# ------------------------------------------------------------ memory
def _closes_after(start_close, forward_return, n=25):
    path = np.linspace(start_close, start_close * (1 + forward_return), n)
    return _series(np.r_[start_close, path], start="2026-01-01")


def test_learning_shifts_trust_toward_the_advisor_that_was_right():
    memory = mem.new_memory()
    day = date(2026, 1, 1)
    # trend said "own it", momentum said "get out"; the stock then rose 10%
    mem.remember(memory, day, "ITC.NS", {"trend": 0.8, "volatility": 0.0, "momentum": -0.8, "news": 0.0}, 100.0)
    lessons = mem.learn(memory, {"ITC.NS": _closes_after(100.0, 0.10)}, date(2026, 2, 15))

    assert lessons and lessons[0]["best"] == "trend"
    assert memory["weights"]["trend"] > 0.25 > memory["weights"]["momentum"]


def test_an_advisor_with_no_opinion_is_not_punished():
    memory = mem.new_memory()
    mem.remember(memory, date(2026, 1, 1), "ITC.NS", {"trend": 0.8, "volatility": 0.0, "momentum": -0.8, "news": 0.0}, 100.0)
    mem.learn(memory, {"ITC.NS": _closes_after(100.0, 0.10)}, date(2026, 2, 15))

    assert memory["weights"]["news"] == pytest.approx(0.25, abs=1e-3)
    assert memory["weights"]["volatility"] == pytest.approx(0.25, abs=1e-3)


def test_views_are_not_judged_before_the_horizon():
    memory = mem.new_memory()
    mem.remember(memory, date(2026, 1, 1), "ITC.NS", {"trend": 0.8, "volatility": 0.0, "momentum": -0.8, "news": 0.0}, 100.0)
    assert mem.learn(memory, {"ITC.NS": _closes_after(100.0, 0.10, n=5)}, date(2026, 1, 8)) == []


# ------------------------------------------------------------ news
def test_keyword_fallback_separates_material_news_from_noise():
    material = news.keyword_score("ICICI Bank Q2 profit jumps, beats estimates", "ICICIBANK.NS")
    noise = news.keyword_score("ICICI Bank share price today live updates", "ICICIBANK.NS")
    other = news.keyword_score("Axis Bank profit jumps", "ICICIBANK.NS")
    assert material["sentiment"] > 0 and material["material"] >= 0.5
    assert noise["material"] == 0.0
    assert other["relevant"] is False


def test_news_signal_weights_fresh_material_headlines_over_stale_noise():
    now = datetime(2026, 9, 23, 12, tzinfo=timezone.utc)
    fresh_bad = {"relevant": True, "sentiment": -0.8, "material": 0.9, "published_utc": now.isoformat()}
    stale_good = {"relevant": True, "sentiment": 0.8, "material": 0.2,
                  "published_utc": (now - timedelta(days=3)).isoformat()}
    irrelevant = {"relevant": False, "sentiment": 1.0, "material": 1.0, "published_utc": now.isoformat()}
    signal = news.news_signal([fresh_bad, stale_good, irrelevant], now=now)
    assert signal["score"] < -0.5
    assert signal["material_count"] == 1


def test_llm_failure_returns_none_so_the_caller_falls_back():
    class Broken:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("provider down")
    assert news.llm_scores([{"title": "x"}], "ITC", client=Broken()) is None


# ------------------------------------------------------------ daily run
def _prices(days, falling=()):
    """Steadily rising, calm prices (every advisor supportive), except any
    symbol in `falling`, which declines steadily."""
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2025-01-01", periods=days)
    frames = []
    for i, symbol in enumerate(INDIA_TICKERS):
        drift = -0.002 if symbol in falling else 0.001
        close = (100 + 20 * i) * np.cumprod(1 + rng.normal(drift, 0.004, days))
        frames.append(pd.DataFrame({"Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99,
                                    "Close": close, "Volume": 1e6, "Ticker": symbol}, index=index))
    raw = pd.concat(frames)
    raw.index.name = "Date"
    return raw


def _quiet_news(symbol, on=None):
    return {"signal": {"view": 0.0, "score": 0.0, "strength": 0.0, "material_count": 0}, "headlines": []}


@pytest.fixture
def paths(tmp_path):
    return {"book_path": tmp_path / "book.json", "memory_path": tmp_path / "memory.json",
            "journal_path": tmp_path / "journal.json"}


def test_first_run_queues_buys_for_the_next_open_and_trades_nothing_today(paths):
    result = run_day(date(2026, 1, 1), prices=_prices(300), news_fn=_quiet_news, **paths)

    assert result["filled"] == []                                  # nothing fills on the decision day
    assert {q["symbol"] for q in result["queued"]} == set(INDIA_TICKERS)
    assert all(q["action"] == "TARGET" for q in result["queued"])
    assert result["summary"]["benchmark_return"] != 0 or result["marked"]["benchmark_equity"] > 0


def test_same_session_is_never_processed_twice(paths):
    prices = _prices(300)
    run_day(date(2026, 1, 1), prices=prices, news_fn=_quiet_news, **paths)
    again = run_day(date(2026, 1, 1), prices=prices, news_fn=_quiet_news, **paths)
    assert again["skipped"] is True


def test_an_older_session_never_replays_over_a_newer_book(paths):
    from scaata.agent.daily import load_journal

    run_day(date(2026, 1, 2), prices=_prices(301), news_fn=_quiet_news, **paths)
    stale = run_day(date(2026, 1, 2), prices=_prices(300), news_fn=_quiet_news, **paths)
    assert stale["skipped"] is True
    assert len(load_journal(paths["journal_path"])) == 1


def test_next_session_fills_at_the_open_and_writes_the_journal(paths):
    from scaata.agent.daily import load_journal

    run_day(date(2026, 1, 1), prices=_prices(300), news_fn=_quiet_news, **paths)
    result = run_day(date(2026, 1, 2), prices=_prices(301), news_fn=_quiet_news, **paths)

    assert len(result["filled"]) == len(INDIA_TICKERS)
    assert all(t["action"] == "BUY" for t in result["filled"])
    journal = load_journal(paths["journal_path"])
    assert len(journal) == 2
    assert journal[-1]["decisions"][0]["reasons"]


def test_news_failure_does_not_stop_the_run(paths):
    def broken(symbol, on=None):
        raise ConnectionError("offline")
    result = run_day(date(2026, 1, 1), prices=_prices(300), news_fn=broken, **paths)
    assert len(result["decisions"]) == len(INDIA_TICKERS)


def test_a_clearly_falling_stock_is_not_bought_on_day_one(paths):
    result = run_day(date(2026, 1, 1), prices=_prices(300, falling=("ITC.NS",)), news_fn=_quiet_news, **paths)
    queued = {q["symbol"] for q in result["queued"]}
    assert "ITC.NS" not in queued
    assert queued == set(INDIA_TICKERS) - {"ITC.NS"}


def test_a_bar_from_a_session_still_in_progress_is_ignored(paths):
    from scaata.agent.daily import IST
    prices = _prices(301)
    last = prices.index.max().date()
    during = datetime(last.year, last.month, last.day, 11, 0, tzinfo=IST)
    after = datetime(last.year, last.month, last.day, 16, 0, tzinfo=IST)
    assert run_day(prices=prices, news_fn=_quiet_news, now=during, **paths)["session"] < last.isoformat()
    assert run_day(prices=prices, news_fn=_quiet_news, now=after, **paths)["session"] == last.isoformat()


def test_a_syndicated_story_counts_once():
    now = datetime(2026, 9, 23, 12, tzinfo=timezone.utc)
    def h(title, sentiment):
        return {"title": title, "relevant": True, "sentiment": sentiment, "material": 0.8,
                "published_utc": now.isoformat()}
    copies = [h("Oil firms face Rs 530 crore daily fuel losses as crude surges: ICRA", -1.0),
              h("OMCs losing Rs 530 crore daily as crude surge squeezes fuel margins: ICRA", -1.0),
              h("Oil firms face Rs 530 cr daily fuel losses as crude surges: ICRA", -1.0)]
    other = h("Indian Oil board approves Kochi-Thoothukudi gas pipeline", 1.0)

    grouped = news.stories(copies + [other], "IOC.NS")
    assert sorted(s["count"] for s in grouped) == [1, 3]
    # three copies of one bad story no longer outvote one good story 3-to-1
    assert news.news_signal(copies + [other], now=now, symbol="IOC.NS")["score"] > -0.5


def test_idle_cash_earns_a_liquid_fund_rate_in_both_legs_once_per_day():
    from scaata.live.paper_book import accrue_cash_yield, liquid_yield, new_book

    book = new_book(list(INDIA_TICKERS))
    book["benchmark"].update(started=True, cash=100.0)
    assert accrue_cash_yield(book, date(2026, 1, 1)) == 0.0          # first call only starts the clock
    earned = accrue_cash_yield(book, date(2026, 1, 2))
    assert earned == pytest.approx(10_000 * ((1 + liquid_yield(2026)) ** (1 / 365) - 1))
    assert book["benchmark"]["cash"] > 100.0
    assert accrue_cash_yield(book, date(2026, 1, 2)) == 0.0          # same day again: nothing


def test_liquid_yield_uses_the_nearest_known_year():
    from scaata.live.paper_book import LIQUID_FUND_YIELD, liquid_yield
    assert liquid_yield(2031) == LIQUID_FUND_YIELD[max(LIQUID_FUND_YIELD)]
    assert liquid_yield(2010) == LIQUID_FUND_YIELD[min(LIQUID_FUND_YIELD)]
