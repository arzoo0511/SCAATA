"""GDELT sentiment ingestion (Phase 3 "3rd eye" data source).

Uses the GDELT DOC 2.0 API (`api.gdeltproject.org/api/v2/doc/doc`) rather
than the BigQuery GKG dataset, since it needs no account/API key and fits
the "publicly available data" ethics framing already used in the v1 paper.
GDELT indexes organizations/documents, not tickers, so a ticker-to-entity
name mapping plus a market-context keyword filter is required to reduce
false positives (e.g. "Apple" the fruit vs Apple Inc).

Known limitation: GDELT's crawl coverage and tone-scoring consistency
improved over 2020-2026 independent of any real market change — any
2020-vs-2026 comparison built on this data (see `scaata/thirdeye/`) should
report mention-volume alongside tone so coverage drift isn't mistaken for
a genuine change in the news-market relationship.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
MIN_REQUEST_INTERVAL_SECONDS = 6  # GDELT asks for >= 1 request per 5s; pad slightly
MARKET_CONTEXT_KEYWORDS = ["stock", "shares", "Nasdaq", "earnings", "investor"]

TICKER_ENTITY_MAP = {
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "Nvidia",
    "META": "Meta Platforms",
    "XOM": "Exxon Mobil",
    "INTC": "Intel",
    "BABA": "Alibaba",
}

US_MARKET_CLOSE_HOUR_ET = 16  # 4:00pm US/Eastern
_last_request_time = 0.0


def _rate_limited_get(params: dict) -> requests.Response:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    response = requests.get(GDELT_DOC_API, params=params, timeout=30)
    _last_request_time = time.time()
    return response


def build_query(ticker: str) -> str:
    entity = TICKER_ENTITY_MAP.get(ticker, ticker)
    context = " OR ".join(MARKET_CONTEXT_KEYWORDS)
    return f'"{entity}" ({context})'


def fetch_daily_tone_timeline(ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """Queries GDELT's `timelinetone` mode for a per-ticker daily average
    tone series between `start` and `end` (YYYY-MM-DD). Returns an empty
    DataFrame (not an exception) on repeated failure, so callers can decide
    how to handle missing coverage rather than crash a whole pipeline run.
    """
    params = {
        "query": build_query(ticker),
        "mode": "timelinetone",
        "format": "json",
        "startdatetime": pd.Timestamp(start).strftime("%Y%m%d000000"),
        "enddatetime": pd.Timestamp(end).strftime("%Y%m%d000000"),
    }

    for attempt in range(1, max_retries + 1):
        response = _rate_limited_get(params)
        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                print(f"GDELT returned non-JSON for {ticker} (likely rate-limited): {response.text[:200]}")
                time.sleep(MIN_REQUEST_INTERVAL_SECONDS * attempt)
                continue
            return _parse_timeline_tone(data, ticker)
        time.sleep(MIN_REQUEST_INTERVAL_SECONDS * attempt)

    print(f"Warning: GDELT fetch failed for {ticker} after {max_retries} attempts — returning empty coverage.")
    return pd.DataFrame(columns=["date", "ticker", "tone", "mention_volume"])


def _parse_timeline_tone(data: dict, ticker: str) -> pd.DataFrame:
    timelines = data.get("timeline", [])
    tone_series = next((t for t in timelines if t.get("series") == "GDELT Tone (Avg)"), None)
    if tone_series is None or not tone_series.get("data"):
        return pd.DataFrame(columns=["date", "ticker", "tone", "mention_volume"])

    rows = []
    for point in tone_series["data"]:
        rows.append({
            "date": pd.Timestamp(point["date"]).normalize(),
            "ticker": ticker,
            "tone": float(point["value"]),
        })
    df = pd.DataFrame(rows)
    df["mention_volume"] = 1  # DOC API's timelinetone mode doesn't expose per-point volume; see fetch_volume_timeline
    return df


def attribute_trading_date(utc_timestamp: pd.Timestamp) -> pd.Timestamp:
    """Maps a UTC-timestamped record to the trading date its information
    was actually available for: records at/after 4:00pm US/Eastern (market
    close) roll forward to the next calendar date, so post-close news never
    leaks into a same-day feature used at that day's close. This does not
    account for market holidays/half-days — a documented simplification.
    """
    eastern = utc_timestamp.tz_localize("UTC").tz_convert(ZoneInfo("America/New_York"))
    trading_date = eastern.normalize()
    if eastern.hour >= US_MARKET_CLOSE_HOUR_ET:
        trading_date += timedelta(days=1)
    return trading_date.tz_localize(None)


def align_sentiment_to_trading_days(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Applies `attribute_trading_date` and re-aggregates same-day-after-
    reattribution records (e.g. two records both roll onto the same next
    trading day) by averaging tone and summing mention volume.
    """
    if sentiment_df.empty:
        return sentiment_df
    out = sentiment_df.copy()
    out["trading_date"] = out["date"].apply(attribute_trading_date)
    aggregated = out.groupby(["trading_date", "ticker"], as_index=False).agg(
        tone=("tone", "mean"), mention_volume=("mention_volume", "sum")
    )
    return aggregated.rename(columns={"trading_date": "date"})


def mock_daily_sentiment(ticker: str, dates: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily tone/volume series for offline testing when live
    GDELT access isn't available (e.g. this project's dev sandbox is
    rate-limited/blocked by GDELT's shared-IP throttling regardless of
    backoff) — mirrors `scaata.strategies.scraper.mock_strategies`'s role
    for the strategy pool. Real runs should use `fetch_daily_tone_timeline`.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    tone = rng.normal(0, 2.0, len(dates))
    volume = rng.integers(1, 50, len(dates))
    return pd.DataFrame({"date": dates, "ticker": ticker, "tone": tone, "mention_volume": volume})
