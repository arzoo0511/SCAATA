"""3rd Eye correlator: quantifies the relationship between sentiment and
market behavior (realized volatility / forward returns), separately for an
early ("2020-era") and late ("2026-era") slice of the backtest, plus a
Granger-causality/lead-lag test for a directional claim stronger than
plain correlation.

Known caveat (carried through to the narrative and charts): GDELT's crawl
coverage likely improved 2020->2026 independent of any real change in the
news-market relationship, so `mean_coverage_volume` is reported alongside
every correlation number to make that distinguishable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def rolling_sentiment_correlation(sentiment: pd.Series, target: pd.Series, window: int = 20) -> pd.Series:
    """Rolling correlation between an already-causal sentiment score series
    and an already-causal target series (e.g. trailing realized
    volatility), aligned by index. No further shifting is applied here."""
    aligned = pd.concat([sentiment, target], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])


def _era_stats(window_df: pd.DataFrame, sentiment_col: str, vol_col: str, return_col: str) -> dict:
    if len(window_df) < 5:
        return {"n_days": len(window_df), "corr_vol": np.nan, "corr_next_day_return": np.nan, "mean_coverage_volume": np.nan}
    return {
        "n_days": len(window_df),
        "corr_vol": float(window_df[sentiment_col].corr(window_df[vol_col])),
        "corr_next_day_return": float(window_df[sentiment_col].corr(window_df[return_col].shift(-1))),
        "mean_coverage_volume": float(window_df["mention_volume"].mean()) if "mention_volume" in window_df else float("nan"),
    }


def era_correlation_report(
    market_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    era_early: tuple[str, str],
    era_late: tuple[str, str],
    sentiment_col: str = "sentiment_score",
    vol_col: str = "rolling_vol",
    return_col: str = "returns",
) -> dict:
    """`market_df` must be date-indexed with `vol_col`/`return_col` columns
    (e.g. Phase 1's regime-detector output). `sentiment_df` has a `date`
    column plus `sentiment_col`/`mention_volume`."""
    sentiment_indexed = sentiment_df.set_index("date")
    merged = market_df.join(sentiment_indexed[[c for c in [sentiment_col, "mention_volume"] if c in sentiment_indexed.columns]], how="left")

    early = merged.loc[era_early[0]:era_early[1]].dropna(subset=[sentiment_col]) if sentiment_col in merged.columns else pd.DataFrame()
    late = merged.loc[era_late[0]:era_late[1]].dropna(subset=[sentiment_col]) if sentiment_col in merged.columns else pd.DataFrame()

    return {
        "era_early": _era_stats(early, sentiment_col, vol_col, return_col),
        "era_late": _era_stats(late, sentiment_col, vol_col, return_col),
    }


def granger_causality_test(sentiment: pd.Series, target: pd.Series, max_lag: int = 5) -> dict:
    """Does sentiment help predict `target` beyond target's own past?
    Reports p-values per lag (lower = stronger evidence of predictive lead).
    Granger causality tests predictive content, not structural causation —
    treat results as indicative, especially given typically small daily
    samples, not as proof of a causal mechanism.
    """
    aligned = pd.concat([target, sentiment], axis=1).dropna()
    aligned.columns = ["target", "sentiment"]
    if len(aligned) < max_lag * 3:
        return {"error": "insufficient data for Granger test", "n": len(aligned)}

    try:
        result = grangercausalitytests(aligned[["target", "sentiment"]], maxlag=max_lag)
    except Exception as e:
        return {"error": str(e), "n": len(aligned)}

    p_values = {lag: round(result[lag][0]["ssr_ftest"][1], 4) for lag in result}
    best_lag = min(p_values, key=p_values.get)
    return {"p_values_by_lag": p_values, "best_lag": best_lag, "best_p_value": p_values[best_lag], "n": len(aligned)}
