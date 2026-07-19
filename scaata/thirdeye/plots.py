"""Phase 3 sentiment graphs — the visual core of the "3rd eye" thesis."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scaata.evaluation.plots import REGIME_COLORS
from scaata.thirdeye.correlator import rolling_sentiment_correlation


def plot_sentiment_overlaid_price(price: pd.Series, sentiment: pd.Series, title: str, ax=None):
    """Price on the primary axis, sentiment as a shaded band on a twin axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))
    ax.plot(price.index, price.values, color="black", lw=1.5, label="Price")
    ax.set_ylabel("Price ($)")

    ax2 = ax.twinx()
    aligned_sentiment = sentiment.reindex(price.index)
    ax2.fill_between(
        aligned_sentiment.index, aligned_sentiment.values, 0,
        color="steelblue", alpha=0.3, label="Sentiment",
    )
    ax2.set_ylabel("Sentiment score")
    ax2.axhline(0, color="grey", lw=0.5)

    ax.set_title(title, weight="bold")
    ax.set_xlabel("Date")
    return ax


def plot_event_study(
    returns: pd.Series, sentiment: pd.Series, shock_zscore_threshold: float = 1.5,
    window_before: int = 5, window_after: int = 10, ax=None,
):
    """Average abnormal return in a window around detected sentiment-shock
    days (|sentiment z-score| exceeds `shock_zscore_threshold`)."""
    aligned = pd.concat([returns, sentiment], axis=1).dropna()
    aligned.columns = ["returns", "sentiment"]
    z = (aligned["sentiment"] - aligned["sentiment"].mean()) / (aligned["sentiment"].std() + 1e-8)
    shock_days = aligned.index[np.abs(z) > shock_zscore_threshold]

    windows = []
    positions = {d: i for i, d in enumerate(aligned.index)}
    for day in shock_days:
        pos = positions[day]
        start, end = pos - window_before, pos + window_after + 1
        if start < 0 or end > len(aligned):
            continue
        windows.append(aligned["returns"].iloc[start:end].values)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if not windows:
        ax.text(0.5, 0.5, "No sentiment shocks detected at this threshold.", ha="center", va="center")
        return ax

    avg_window = np.mean(windows, axis=0)
    x = np.arange(-window_before, window_after + 1)
    ax.bar(x, avg_window, color=["#e76f51" if v < 0 else "#2a9d8f" for v in avg_window])
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_title(f"Event Study: Avg Return Around Sentiment Shocks (n={len(windows)})", weight="bold")
    ax.set_xlabel("Days relative to shock")
    ax.set_ylabel("Average return")
    return ax


def plot_rolling_sentiment_volatility_correlation(
    sentiment: pd.Series, volatility: pd.Series, regime_labels: pd.Series | None = None,
    window: int = 20, ax=None,
):
    """The core '3rd eye' chart: rolling correlation between sentiment and
    realized volatility across the full date range, with regime bands
    overlaid if provided — shows how the news-market relationship changes
    over time (e.g. 2020 vs 2026)."""
    corr = rolling_sentiment_correlation(sentiment, volatility, window=window)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    if regime_labels is not None:
        aligned_regimes = regime_labels.reindex(corr.index).ffill()
        current_regime, band_start = None, None
        for date, regime in aligned_regimes.items():
            if regime != current_regime:
                if current_regime is not None:
                    ax.axvspan(band_start, date, color=REGIME_COLORS.get(current_regime, "grey"), alpha=0.12)
                current_regime, band_start = regime, date
        if current_regime is not None and len(aligned_regimes) > 0:
            ax.axvspan(band_start, aligned_regimes.index[-1], color=REGIME_COLORS.get(current_regime, "grey"), alpha=0.12)

    ax.plot(corr.index, corr.values, color="darkblue", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Rolling {window}-Day Sentiment-vs-Volatility Correlation", weight="bold")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    ax.set_ylim(-1, 1)
    return ax
