"""Tests for scaata.data.alpaca_news -- the real-news alternative to GDELT
(which this project's dev sandbox has confirmed hard rate-limited). Mocked
throughout: no real Alpaca/FinBERT calls, just the grouping/capping/caching
logic and the trading-day attribution reuse.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

import scaata.data.alpaca_news as alpaca_news_module
from scaata.data.alpaca_news import (
    MAX_HEADLINES_PER_DAY,
    _to_naive_utc,
    compute_daily_sentiment,
    fetch_headlines_by_trading_day,
)


def test_to_naive_utc_strips_tzinfo_from_aware_timestamp():
    aware = pd.Timestamp("2026-07-25 12:00:00", tz="UTC")
    naive = _to_naive_utc(aware)
    assert naive.tzinfo is None
    assert naive == pd.Timestamp("2026-07-25 12:00:00")


def test_to_naive_utc_passes_through_already_naive_timestamp():
    naive_in = pd.Timestamp("2026-07-25 12:00:00")
    assert _to_naive_utc(naive_in) == naive_in


class _FakeArticle:
    def __init__(self, created_at, headline):
        self.created_at = created_at
        self.headline = headline


class _FakeNewsResult:
    def __init__(self, articles):
        self.data = {"news": articles}


class _FakeNewsClient:
    def __init__(self, articles):
        self._articles = articles

    def get_news(self, request):
        return _FakeNewsResult(self._articles)


def test_fetch_headlines_is_mock_without_credentials(monkeypatch):
    monkeypatch.setattr(alpaca_news_module, "_news_client", lambda: None)
    monkeypatch.setattr(alpaca_news_module, "NEWS_CACHE_DIR", alpaca_news_module.NEWS_CACHE_DIR)
    by_day, source = fetch_headlines_by_trading_day("AAPL", "2026-07-01", "2026-07-25", use_cache=False)
    assert source == "mock"
    assert by_day == {}


def test_fetch_headlines_groups_by_trading_day_and_caps_volume(monkeypatch, tmp_path):
    monkeypatch.setattr(alpaca_news_module, "NEWS_CACHE_DIR", tmp_path)
    # 3 articles the same pre-close trading day, plus enough extra same-day
    # articles to exceed MAX_HEADLINES_PER_DAY, plus 1 post-close article
    # that must roll to the next trading day.
    same_day = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)  # 10am ET, pre-close
    articles = [_FakeArticle(same_day, f"headline {i}") for i in range(MAX_HEADLINES_PER_DAY + 3)]
    post_close = datetime(2026, 7, 20, 21, 0, tzinfo=timezone.utc)  # 5pm ET, post-close
    articles.append(_FakeArticle(post_close, "after-hours headline"))

    client = _FakeNewsClient(articles)
    by_day, source = fetch_headlines_by_trading_day("AAPL", "2026-07-01", "2026-07-25", client=client, use_cache=False)

    assert source == "live"
    same_day_key = pd.Timestamp("2026-07-20")
    next_day_key = pd.Timestamp("2026-07-21")
    assert len(by_day[same_day_key]) == MAX_HEADLINES_PER_DAY  # capped, not all 8
    assert by_day[next_day_key] == ["after-hours headline"]


def test_fetch_headlines_uses_cache_on_second_call(monkeypatch, tmp_path):
    monkeypatch.setattr(alpaca_news_module, "NEWS_CACHE_DIR", tmp_path)
    articles = [_FakeArticle(datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc), "cached headline")]
    client = _FakeNewsClient(articles)

    fetch_headlines_by_trading_day("AAPL", "2026-07-01", "2026-07-25", client=client, use_cache=True)

    def _fail_if_called(request):
        raise AssertionError("client.get_news should not be called when a cache hit exists")

    client.get_news = _fail_if_called
    by_day, source = fetch_headlines_by_trading_day("AAPL", "2026-07-01", "2026-07-25", client=client, use_cache=True)

    assert source == "cached"
    assert by_day[pd.Timestamp("2026-07-20")] == ["cached headline"]


def test_compute_daily_sentiment_scores_and_averages_per_day(monkeypatch):
    monkeypatch.setattr(
        "scaata.features.sentiment.score_headlines_finbert",
        lambda headlines: [0.5, -0.1][: len(headlines)],
    )
    headlines_by_day = {
        pd.Timestamp("2026-07-20"): ["good news", "bad news"],
        pd.Timestamp("2026-07-21"): [],  # no coverage -- must be skipped, not zero-filled
    }
    result = compute_daily_sentiment("AAPL", headlines_by_day)

    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2026-07-20")
    assert result.iloc[0]["ticker"] == "AAPL"
    assert result.iloc[0]["sentiment_score"] == pytest.approx(0.2)  # mean(0.5, -0.1)


def test_compute_daily_sentiment_empty_input_returns_empty_frame():
    result = compute_daily_sentiment("AAPL", {})
    assert result.empty
    assert list(result.columns) == ["date", "ticker", "sentiment_score"]
