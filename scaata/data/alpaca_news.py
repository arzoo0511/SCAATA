"""Real news headlines via Alpaca's own News API (already connected, no new
account/key needed) -- a working alternative to GDELT, which this project's
dev sandbox has confirmed rate-limited (HTTP 429, "please limit requests to
one every 5 seconds") regardless of client-side backoff, almost certainly a
shared-IP limit on GDELT's side rather than anything this code controls
(see `scaata.data.gdelt`'s module docstring, written independently and
reproduced live again this session).

A single high-profile ticker (AAPL) returned 10,000+ articles in under 3
years even with article content excluded -- real coverage goes back through
2020, but running FinBERT over every single article isn't practical.
Headlines are capped per trading day (`MAX_HEADLINES_PER_DAY`) so scoring
cost stays bounded regardless of how heavily a ticker is covered.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from scaata.config import DATA_CACHE_DIR
from scaata.data.gdelt import attribute_trading_date

load_dotenv()

MAX_HEADLINES_PER_DAY = 5
NEWS_CACHE_DIR = DATA_CACHE_DIR / "alpaca_news"
NEWS_CACHE_DIR.mkdir(exist_ok=True)


def _news_client():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    from alpaca.data.historical.news import NewsClient

    return NewsClient(api_key, secret_key)


def _cache_path(ticker: str, start: str, end: str) -> Path:
    return NEWS_CACHE_DIR / f"news_{ticker}_{start}_{end}.json"


def _to_naive_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def fetch_headlines_by_trading_day(
    ticker: str, start: str, end: str, client=None, use_cache: bool = True
) -> tuple[dict, str]:
    """Returns ({trading_date: [headline, ...]}, source). Groups each
    headline under the trading day its publish time is actually
    attributable to, reusing `scaata.data.gdelt.attribute_trading_date`'s
    post-close-rolls-forward rule -- the look-ahead-safety logic is
    identical regardless of which news source produced the timestamp.
    Capped at `MAX_HEADLINES_PER_DAY` per day. `source` is `"live"`,
    `"cached"` (an earlier live fetch, same exact date range), or
    `"mock"` (no Alpaca credentials configured).
    """
    cache_path = _cache_path(ticker, start, end)
    if use_cache and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {pd.Timestamp(k): v for k, v in raw.items()}, "cached"

    client = client if client is not None else _news_client()
    if client is None:
        return {}, "mock"

    from alpaca.data.requests import NewsRequest

    request = NewsRequest(
        symbols=ticker, start=pd.Timestamp(start), end=pd.Timestamp(end),
        limit=10000, include_content=False,
    )
    news = client.get_news(request)
    articles = news.data.get("news", [])

    by_day: dict = {}
    for article in articles:
        trading_date = attribute_trading_date(_to_naive_utc(article.created_at))
        by_day.setdefault(trading_date, []).append(article.headline)

    capped = {day: headlines[:MAX_HEADLINES_PER_DAY] for day, headlines in by_day.items()}

    if use_cache:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({k.isoformat(): v for k, v in capped.items()}, f)

    return capped, "live"


def compute_daily_sentiment(ticker: str, headlines_by_day: dict) -> pd.DataFrame:
    """Scores each day's capped headline list with FinBERT and averages to
    one `sentiment_score` per day. Returns columns `date`/`ticker`/
    `sentiment_score` -- exactly what
    `scaata.features.sentiment_features.merge_sentiment_into_features`
    expects. Days with no headlines are simply absent (not zero-filled) --
    the merge step's `neutral_fill` handles that gap.
    """
    from scaata.features.sentiment import score_headlines_finbert

    rows = []
    for day, headlines in headlines_by_day.items():
        if not headlines:
            continue
        scores = score_headlines_finbert(headlines)
        rows.append({"date": day, "ticker": ticker, "sentiment_score": float(np.mean(scores))})

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "sentiment_score"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
