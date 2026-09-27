"""CBOE total put/call ratio (Phase 9c) — a free, publicly-published,
market-wide options-positioning signal. Unlike CBOE/OPRA-quality options
chain data (skew, open interest by strike), the *aggregate* daily put/call
ratio has been published as a free CSV for years and needs no account or
key, so it's a genuinely obtainable "richer signal" rather than a wishlist
item.

Treated with the same fail-soft discipline as `scaata.data.gdelt`: CBOE's
exact CSV endpoint/layout has moved before and may move again, so any
fetch failure returns an empty DataFrame (never raises) and callers must
be able to run without it, exactly like GDELT's rate-limited fallback.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from scaata.config import CBOE_PUTCALL_URL, DATA_CACHE_DIR, PUTCALL_LAG_DAYS

REQUEST_TIMEOUT_SECONDS = 30
EMPTY_COLUMNS = ["date", "total_putcall_ratio"]


def _cache_path(start: str, end: str) -> Path:
    safe = f"putcall_{start}_{end}".replace("/", "-")
    return DATA_CACHE_DIR / f"{safe}.parquet"


def _parse_putcall_csv(raw_text: str) -> pd.DataFrame:
    """CBOE's totalpc.csv historically ships a few descriptive lines before
    the real header row (`DATE,CALL,PUT,TOTAL,P/C Ratio`). Scan for that
    header rather than hardcoding a skiprows count, since the exact preamble
    length has changed before.
    """
    lines = raw_text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.upper().startswith("DATE")), None)
    if header_idx is None:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    ratio_col = next((c for c in df.columns if "ratio" in c.lower()), None)
    date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
    if ratio_col is None or date_col is None:
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "total_putcall_ratio": pd.to_numeric(df[ratio_col], errors="coerce"),
    })
    return out.dropna(subset=["date"]).reset_index(drop=True)


def fetch_total_putcall_ratio(start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetches CBOE's daily total put/call ratio between `start` and `end`
    (YYYY-MM-DD). Returns an empty `["date", "total_putcall_ratio"]`
    DataFrame (not an exception) on any failure — the endpoint moving, a
    network error, or an unexpected CSV layout — so callers can decide how
    to handle missing coverage rather than crash a whole pipeline run, the
    same discipline `scaata.data.gdelt.fetch_daily_tone_timeline` uses.
    """
    cache_path = _cache_path(start, end)
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    try:
        response = requests.get(CBOE_PUTCALL_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            print(f"Warning: CBOE put/call fetch returned {response.status_code} — returning empty coverage.")
            return pd.DataFrame(columns=EMPTY_COLUMNS)
        parsed = _parse_putcall_csv(response.text)
    except requests.RequestException as e:
        print(f"Warning: CBOE put/call fetch failed ({e}) — returning empty coverage.")
        return pd.DataFrame(columns=EMPTY_COLUMNS)

    mask = (parsed["date"] >= pd.Timestamp(start)) & (parsed["date"] <= pd.Timestamp(end))
    out = parsed.loc[mask].reset_index(drop=True)

    if use_cache and not out.empty:
        out.to_parquet(cache_path)
    return out


def attach_putcall_feature(
    feature_df: pd.DataFrame,
    putcall_df: pd.DataFrame,
    lag_days: int = PUTCALL_LAG_DAYS,
    neutral_fill: float | None = None,
) -> pd.DataFrame:
    """Left-joins the put/call ratio onto `feature_df` (DatetimeIndex named
    "Date"), lagged by `lag_days` (default 1) before use — a market-wide
    macro series is an easy off-by-one lookahead leak, since the published
    ratio for day t typically isn't knowable until after day t's close.
    Missing coverage (no data, or before CBOE's publication start) falls
    back to `neutral_fill` (the series' own mean if not given, since unlike
    sentiment there's no natural "zero" for a ratio).
    """
    if putcall_df.empty:
        out = feature_df.copy()
        out["total_putcall_ratio"] = neutral_fill if neutral_fill is not None else 1.0
        return out

    lagged = putcall_df.copy()
    lagged["date"] = lagged["date"] + pd.Timedelta(days=lag_days)
    lagged = lagged.rename(columns={"date": "Date"}).drop_duplicates(subset=["Date"])

    fill_value = neutral_fill if neutral_fill is not None else float(lagged["total_putcall_ratio"].mean())

    index_name = feature_df.index.name or "Date"
    out = feature_df.reset_index().rename(columns={index_name: "Date"})
    out = out.merge(lagged[["Date", "total_putcall_ratio"]], on="Date", how="left")
    out["total_putcall_ratio"] = out["total_putcall_ratio"].fillna(fill_value)
    return out.set_index("Date")
