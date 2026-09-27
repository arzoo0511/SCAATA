"""Microstructure proxies computable from OHLCV alone (Phase 9c) — zero new
data source, pure functions of columns `scaata.data.loaders` already
returns. These measure *liquidity stress / cost-of-trading*, a genuinely
different axis from `scaata.features.technical`'s `volatility` feature
(dispersion of returns): a low-volume, wide-spread day can spike these
without moving realized volatility at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin & Schultz (2012) bid-ask spread estimator from consecutive
    daily high/low pairs alone — purely trailing (today's high/low paired
    with yesterday's), so no look-ahead risk. Negative raw estimates (a
    known artifact of this estimator when high approx-equals low on a quiet
    day) are clamped to 0, since a negative "spread" isn't economically
    meaningful.
    """
    h1, l1 = high.shift(1), low.shift(1)
    h2, l2 = high, low

    beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
    h_max = np.maximum(h1, h2)
    l_min = np.minimum(l1, l2)
    gamma = np.log(h_max / l_min) ** 2

    denom = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


def amihud_illiquidity(returns: pd.Series, dollar_volume: pd.Series, window: int) -> pd.Series:
    """Amihud (2002) illiquidity ratio: rolling mean of |return| / dollar
    volume, i.e. the price-impact cost of trading — distinct from
    `volatility`, which only measures how much price moved, not how much
    volume it took to move it.
    """
    daily_ratio = returns.abs() / (dollar_volume + 1e-8)
    return daily_ratio.rolling(window).mean()


def add_microstructure_features(df: pd.DataFrame, amihud_window: int) -> pd.DataFrame:
    """Adds `cs_spread` and `amihud_illiq` to a (Date-indexed, `Ticker`-
    columned) OHLCV DataFrame. Computed per-ticker on a temporary unique
    positional index before reassembly, since the Date index repeats across
    tickers and label-based assignment would otherwise risk cross-ticker
    misalignment.
    """
    index_name = df.index.name or "Date"
    data = df.copy()
    if "returns" not in data.columns:
        data["returns"] = data.groupby("Ticker")["Close"].pct_change()

    data = data.reset_index(drop=False)
    if index_name not in data.columns:
        data = data.rename(columns={"index": index_name})

    cs = pd.Series(index=data.index, dtype=float)
    illiq = pd.Series(index=data.index, dtype=float)
    for _, g in data.groupby("Ticker"):
        cs.loc[g.index] = corwin_schultz_spread(g["High"], g["Low"]).values
        dollar_volume = g["Close"] * g["Volume"]
        illiq.loc[g.index] = amihud_illiquidity(g["returns"], dollar_volume, amihud_window).values

    data["cs_spread"] = cs
    data["amihud_illiq"] = illiq
    return data.set_index(index_name)
