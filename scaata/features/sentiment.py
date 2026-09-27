"""FinBERT sentiment scoring, combined with GDELT's own tone score as a
second, independent sentiment signal (Phase 3's "3rd eye" feature).

Kept as two separate scores rather than replacing GDELT's tone outright —
this gives the meta-selector/RL agent two independently-derived sentiment
signals to cross-check against each other, and lets `thirdeye/correlator.py`
report whether they agree (a sanity signal on data quality, not just a
model feature).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FINBERT_MODEL_NAME = "ProsusAI/finbert"
_pipeline_cache = None


def _get_finbert_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        from transformers import pipeline

        # token=False forces anonymous access to this public model — avoids
        # an unrelated cached HuggingFace credential on this machine causing
        # spurious 401s when it's expired/invalid for this repo.
        _pipeline_cache = pipeline("sentiment-analysis", model=FINBERT_MODEL_NAME, truncation=True, token=False)
    return _pipeline_cache


def score_headlines_finbert(headlines: list[str]) -> list[float]:
    """Scores each headline/snippet as a signed sentiment value in
    [-1, 1] (positive score - negative score, neutral treated as 0
    contribution), using FinBERT run locally. Batches all headlines in one
    call rather than the full underlying article set, to keep inference
    cost bounded (loaded once, cached across calls via `_pipeline_cache`).
    """
    if not headlines:
        return []
    clf = _get_finbert_pipeline()
    results = clf(headlines)
    scores = []
    for r in results:
        label, score = r["label"].lower(), r["score"]
        if label == "positive":
            scores.append(score)
        elif label == "negative":
            scores.append(-score)
        else:
            scores.append(0.0)
    return scores


def attach_finbert_daily_score(sentiment_df: pd.DataFrame, headlines_by_date: dict[pd.Timestamp, list[str]]) -> pd.DataFrame:
    """Adds a `finbert_score` column (mean FinBERT score of that day's
    headlines) to a GDELT-derived `sentiment_df` (with a `date` column).
    Days with no headlines get NaN, not 0, so downstream correlation code
    can distinguish "no coverage" from "neutral coverage".
    """
    out = sentiment_df.copy()
    scores = {}
    for date, headlines in headlines_by_date.items():
        headline_scores = score_headlines_finbert(headlines)
        scores[date] = float(np.mean(headline_scores)) if headline_scores else np.nan
    out["finbert_score"] = out["date"].map(scores)
    return out


def combine_sentiment_signals(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Produces a single `sentiment_score` combining GDELT `tone` (rescaled
    to roughly [-1, 1], since GDELT tone is typically in [-10, 10]) and
    `finbert_score` where both are available; falls back to whichever one
    exists if only one is present for a given day.
    """
    out = sentiment_df.copy()
    gdelt_scaled = out["tone"] / 10.0 if "tone" in out.columns else pd.Series(np.nan, index=out.index)
    finbert = out["finbert_score"] if "finbert_score" in out.columns else pd.Series(np.nan, index=out.index)

    combined = pd.concat([gdelt_scaled, finbert], axis=1).mean(axis=1, skipna=True)
    out["sentiment_score"] = combined
    return out


def sentiment_velocity(sentiment_df: pd.DataFrame, score_col: str = "sentiment_score", window: int = 3) -> pd.Series:
    """Rate of change of sentiment tone over `window` days, grouped by
    `ticker` if present so velocity is never computed across a ticker
    boundary. This is deliberately distinct from the tone *level*: a story
    "blowing up" shows up as acceleration before the average tone itself
    necessarily moves (Phase 9c)."""
    if "ticker" in sentiment_df.columns:
        return sentiment_df.groupby("ticker")[score_col].diff(window)
    return sentiment_df[score_col].diff(window)


def mention_volume_velocity(sentiment_df: pd.DataFrame, volume_col: str = "mention_volume", window: int = 3) -> pd.Series:
    """Rate of change of news coverage volume over `window` days — coverage
    acceleration is arguably a more causally-honest "something is happening"
    signal than tone level alone (Phase 9c)."""
    if "ticker" in sentiment_df.columns:
        return sentiment_df.groupby("ticker")[volume_col].pct_change(window)
    return sentiment_df[volume_col].pct_change(window)


def attach_sentiment_velocity(
    sentiment_df: pd.DataFrame,
    score_col: str = "sentiment_score",
    volume_col: str = "mention_volume",
    tone_window: int = 3,
    volume_window: int = 3,
) -> pd.DataFrame:
    """Adds `sentiment_velocity`/`mention_volume_velocity` columns. Assumes
    `sentiment_df` is already sorted by date within each ticker (the same
    ordering assumption `combine_sentiment_signals` makes)."""
    out = sentiment_df.copy()
    out["sentiment_velocity"] = sentiment_velocity(out, score_col, tone_window)
    out["mention_volume_velocity"] = mention_volume_velocity(out, volume_col, volume_window)
    return out


def mock_headlines_by_date(dates: pd.DatetimeIndex, seed: int = 0) -> dict[pd.Timestamp, list[str]]:
    """Synthetic headlines for offline testing — real runs should source
    headlines from GDELT's article-level `artlist` mode or another news API."""
    rng = np.random.default_rng(seed)
    templates = [
        "Company beats earnings expectations, shares rally",
        "Stock slides after disappointing guidance",
        "Analysts remain neutral on outlook amid mixed signals",
        "Shares surge on strong quarterly revenue growth",
        "Investors sell off amid regulatory concerns",
    ]
    return {date: [templates[rng.integers(0, len(templates))]] for date in dates}
