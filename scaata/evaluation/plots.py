"""Regime-aware plotting suite (Phase 1).

Extends the v1 notebook's plots (equity curve, drawdown, monthly heatmap,
returns distribution, ACF) with regime shading/annotation and adds the
per-regime, 2020-block vs 2026-block comparison the user asked for.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REGIME_COLORS = {
    "calm_bull": "#2a9d8f",
    "volatile_bull": "#e9c46a",
    "bear": "#f4a261",
    "crisis": "#e76f51",
}


def plot_regime_equity_curve(dates: pd.DatetimeIndex, equity: np.ndarray, regime_labels: pd.Series, title: str, ax=None):
    """Equity curve with vertical regime bands shaded behind it."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    ax.plot(dates, equity[: len(dates)], color="black", lw=1.8, zorder=5, label="Equity")

    regime_labels = regime_labels.reindex(dates).ffill()
    current_regime = None
    band_start = None
    for date, regime in regime_labels.items():
        if regime != current_regime:
            if current_regime is not None:
                ax.axvspan(band_start, date, color=REGIME_COLORS.get(current_regime, "grey"), alpha=0.15)
            current_regime = regime
            band_start = date
    if current_regime is not None:
        ax.axvspan(band_start, dates[-1], color=REGIME_COLORS.get(current_regime, "grey"), alpha=0.15)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in REGIME_COLORS.values()]
    ax.legend(handles + [ax.lines[0]], list(REGIME_COLORS.keys()) + ["Equity"], loc="upper left", fontsize=8)
    ax.set_title(title, weight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    return ax


def plot_macro_event_annotations(dates: pd.DatetimeIndex, ax):
    """Overlay macro-event boundary lines + text labels on an existing axis."""
    from scaata.regimes.calendar_labels import MACRO_EVENTS

    for event in MACRO_EVENTS:
        start = pd.Timestamp(event.start)
        if dates[0] <= start <= dates[-1]:
            ax.axvline(start, color="grey", linestyle="--", alpha=0.5, lw=0.8)
            ax.text(start, ax.get_ylim()[1] * 0.98, event.name, rotation=90, fontsize=7, va="top", alpha=0.7)


def plot_regime_performance_comparison(regime_table_2020: pd.DataFrame, regime_table_2026: pd.DataFrame, metric: str = "sharpe"):
    """Grouped bar chart: per-regime metric, 2020-era block vs 2026-era block side by side."""
    regimes = sorted(set(regime_table_2020.index) | set(regime_table_2026.index))
    vals_2020 = [regime_table_2020[metric].get(r, np.nan) for r in regimes]
    vals_2026 = [regime_table_2026[metric].get(r, np.nan) for r in regimes]

    x = np.arange(len(regimes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, vals_2020, width, label="2020-era", color="#2a9d8f")
    ax.bar(x + width / 2, vals_2026, width, label="2026-era", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Per-Regime {metric.title()}: 2020-era vs 2026-era", weight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_returns_distribution(equity: np.ndarray, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    returns = pd.Series(equity).pct_change().dropna()
    sns.histplot(returns, bins=50, kde=True, color="teal", ax=ax)
    ax.axvline(0, color="red", linestyle="dashed", lw=2)
    ax.set_title("Daily Returns Distribution", weight="bold")
    ax.set_xlabel("Daily Percent Return")
    return ax


def plot_drawdown(dates: pd.DatetimeIndex, equity: np.ndarray, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    equity_series = pd.Series(equity[: len(dates)], index=dates)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    ax.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    ax.plot(drawdown.index, drawdown, color="darkred", lw=1.2)
    ax.set_title("Portfolio Drawdown Over Time", weight="bold")
    ax.set_ylabel("Drop from Peak (%)")
    return ax
