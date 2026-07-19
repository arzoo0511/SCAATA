"""Market data loading with per-ticker parquet caching (ported/extended from
the v1 notebook, which cached the whole combined download as one blob).

Caching per-ticker (rather than per requested-symbol-list) means a rate
limit hit on one symbol doesn't cost you the tickers you already have, and
running the pipeline with a different subset of tickers reuses whatever's
already cached instead of re-downloading everything.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from scaata.config import DATA_CACHE_DIR

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


class DataDownloadError(RuntimeError):
    """Raised when a ticker could not be downloaded and no cache exists for it."""


def _ticker_cache_path(symbol: str, start: str, end: str) -> Path:
    safe = f"{symbol}_{start}_{end}".replace("/", "-")
    return DATA_CACHE_DIR / f"raw_{safe}.parquet"


def _download_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df["Ticker"] = symbol
            df.index.name = "Date"
            return df
        last_error = f"empty response on attempt {attempt}"
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise DataDownloadError(
        f"Failed to download {symbol} ({start} to {end}) after {MAX_RETRIES} attempts: {last_error}. "
        "This is commonly yfinance rate-limiting from repeated calls in a short window — "
        "wait a bit and retry, or use a cached parquet if one exists in data_cache/."
    )


def collect_data(symbols: list[str], start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Download daily OHLCV for `symbols` between `start` and `end`, one
    ticker at a time, each cached to its own parquet file. Raises
    `DataDownloadError` (rather than silently returning partial/empty data)
    if a ticker can't be fetched and has no cache to fall back on.
    """
    df_list = []
    missing = []
    for sym in symbols:
        cache_path = _ticker_cache_path(sym, start, end)
        if use_cache and cache_path.exists():
            df_list.append(pd.read_parquet(cache_path))
            continue
        try:
            df = _download_one(sym, start, end)
        except DataDownloadError as e:
            missing.append(str(e))
            continue
        if use_cache:
            df.to_parquet(cache_path)
        df_list.append(df)

    if not df_list:
        raise DataDownloadError(
            "No tickers could be loaded (all downloads failed and no cache existed). "
            "Details:\n" + "\n".join(missing)
        )
    if missing:
        print(f"Warning: {len(missing)}/{len(symbols)} ticker(s) failed and were skipped:\n" + "\n".join(missing))

    raw_df = pd.concat(df_list)
    raw_df.index.name = "Date"
    return raw_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns, drop residual gaps, keep only the columns we need."""
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[c for c in numeric_cols if c in data.columns])
    return data[[*numeric_cols, "Ticker"]]


def load_market_data(
    symbols: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Full load: download + clean. Returns one row per (Date, Ticker)."""
    raw = collect_data(symbols, start, end, use_cache=use_cache)
    return clean_data(raw)
