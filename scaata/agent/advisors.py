"""Advisors: each gives a view on each stock, from -1 (get out) to +1 (own it).

Each one looks at a single kind of evidence and nothing else, so the brain
can learn separately how far to trust each kind. All are causal: a view for
day t uses only data up to and including day t's close.

- **trend**: price against its 200-day average. The slow, well-documented
  tendency for long uptrends to keep going.
- **volatility**: a brake, never an accelerator. Negative only when a stock
  is moving much more violently than its own normal -- the one thing the
  tests showed reliably reduces drawdowns.
- **momentum**: six-month return, skipping the latest month.
- **news**: the materiality-weighted sentiment from `scaata.agent.news`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ADVISORS = ("trend", "volatility", "momentum", "news")


def trend_view(closes: pd.Series) -> float:
    if len(closes) < 200:
        return 0.0
    gap = closes.iloc[-1] / closes.iloc[-200:].mean() - 1
    return float(np.tanh(gap / 0.08))


def volatility_view(closes: pd.Series, window: int = 20, baseline: int = 252) -> float:
    returns = closes.pct_change().dropna()
    if len(returns) < baseline:
        return 0.0
    recent = returns.iloc[-window:].std()
    normal = returns.rolling(window).std().iloc[-baseline:].median()
    ratio = recent / normal if normal > 0 else 1.0
    return float(-np.tanh(max(ratio - 1.2, 0.0) / 0.5))


def momentum_view(closes: pd.Series, lookback: int = 126, skip: int = 21) -> float:
    if len(closes) <= lookback:
        return 0.0
    move = closes.iloc[-1 - skip] / closes.iloc[-1 - lookback] - 1
    return float(np.tanh(move / 0.15))


def views(closes: pd.Series, news_signal: dict | None = None) -> dict[str, float]:
    """All advisors' views for one stock, as of the last row of `closes`."""
    return {
        "trend": round(trend_view(closes), 3),
        "volatility": round(volatility_view(closes), 3),
        "momentum": round(momentum_view(closes), 3),
        "news": round(float(news_signal["view"]) if news_signal else 0.0, 3),
    }
