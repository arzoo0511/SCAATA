"""Merges the sentiment pipeline's per-day `sentiment_score` into the
technical feature set so it can actually reach the RL policy / meta-selector
(Phase 9a) — closing the gap where GDELT+FinBERT sentiment was fully
computed but only ever consumed by the standalone `scaata.thirdeye`
narrative/correlation modules, never by any trading decision.
"""
from __future__ import annotations

import pandas as pd


def merge_sentiment_into_features(
    feature_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    sentiment_col: str = "sentiment_score",
    neutral_fill: float = 0.0,
) -> pd.DataFrame:
    """Left-joins `sentiment_df` (columns `date`, `ticker`, `sentiment_col`;
    `date` already causally attributed to a trading day via
    `scaata.data.gdelt.align_sentiment_to_trading_days`) onto `feature_df`
    (DatetimeIndex named "Date", `Ticker` column — the shape
    `scaata.features.technical.add_features` produces).

    Days/tickers with no sentiment coverage get `neutral_fill` (0.0, i.e.
    "no signal") rather than being dropped or forward-filled from a prior
    day. Forward-filling would manufacture apparent sentiment persistence
    that was never actually observed on that day — a subtle,
    lookahead-adjacent distortion, not just a cosmetic gap-filling choice.

    Idempotent: if `feature_df` already has a `sentiment_col` column (e.g.
    a caller re-merging, or a df that was already sentiment-wired being
    passed through a second pipeline stage that merges again), that
    existing column is dropped and replaced rather than left in place —
    otherwise pandas' merge silently suffixes both into
    `{sentiment_col}_x`/`_y`, and the line below would raise a confusing
    `KeyError` for a column that visibly exists in the input (a real bug
    caught by running this against a live pipeline, not a hypothetical).
    """
    if sentiment_col not in sentiment_df.columns:
        raise ValueError(f"sentiment_df is missing required column '{sentiment_col}'")

    sentiment_slim = (
        sentiment_df[["date", "ticker", sentiment_col]]
        .drop_duplicates(subset=["date", "ticker"])
        .rename(columns={"date": "Date", "ticker": "Ticker"})
    )

    index_name = feature_df.index.name or "Date"
    out = feature_df.reset_index().rename(columns={index_name: "Date"})
    if sentiment_col in out.columns:
        out = out.drop(columns=[sentiment_col])
    out = out.merge(sentiment_slim, on=["Date", "Ticker"], how="left")
    out[sentiment_col] = out[sentiment_col].fillna(neutral_fill)
    return out.set_index("Date")
