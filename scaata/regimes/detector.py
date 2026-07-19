"""Causal regime/stress-zone detection (Phase 1).

Two complementary methods, both causal (only ever use data available up to
time t — the single highest-value correctness property in this module):

1. Rolling volatility + drawdown thresholds (primary) — transparent, cheap.
2. Gaussian HMM (secondary cross-check) — fit once on the training slice
   only, then decoded using a rolling trailing context window so no future
   observation (train or test) can inform an earlier day's label.

`regime_label_agreement` reports Cohen's kappa between the two so the
threshold method's choice of buckets isn't just an arbitrary pick.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import cohen_kappa_score

from scaata.config import (
    HMM_N_STATES,
    REGIME_DD_THRESHOLD,
    REGIME_DD_WINDOW,
    REGIME_VOL_MEDIAN_WINDOW,
    REGIME_VOL_MULTIPLIER,
    REGIME_VOL_WINDOW,
    TRADING_DAYS_PER_YEAR,
)


def compute_rolling_vol(returns: pd.Series, window: int = REGIME_VOL_WINDOW) -> pd.Series:
    """Trailing annualized realized volatility. Rolling (not centered) => causal."""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_rolling_drawdown(close: pd.Series, window: int = REGIME_DD_WINDOW) -> pd.Series:
    """Drawdown from the trailing `window`-day peak. Peak only looks backward => causal."""
    trailing_peak = close.rolling(window, min_periods=1).max()
    return (close - trailing_peak) / trailing_peak


def threshold_regime_labels(
    df: pd.DataFrame, price_col: str = "Close", returns_col: str = "returns"
) -> pd.DataFrame:
    """Per-ticker causal regime labels from rolling vol + drawdown thresholds.

    Adds: rolling_vol, vol_regime {normal, high_vol}, drawdown,
    drawdown_regime {normal, stress}, and a combined `regime` label in
    {calm_bull, volatile_bull, bear, crisis}.
    """
    out_frames = []
    for ticker, group in df.groupby("Ticker"):
        group = group.sort_index().copy()
        vol = compute_rolling_vol(group[returns_col])
        # Trailing median vol baseline, also causal (bounded rolling window,
        # not an expanding/full-sample percentile).
        vol_baseline = vol.rolling(REGIME_VOL_MEDIAN_WINDOW, min_periods=REGIME_VOL_WINDOW).median()
        dd = compute_rolling_drawdown(group[price_col])

        vol_regime = np.where(vol > REGIME_VOL_MULTIPLIER * vol_baseline, "high_vol", "normal")
        dd_regime = np.where(dd < REGIME_DD_THRESHOLD, "stress", "normal")

        combined = np.select(
            [
                (vol_regime == "high_vol") & (dd_regime == "stress"),
                dd_regime == "stress",
                vol_regime == "high_vol",
            ],
            ["crisis", "bear", "volatile_bull"],
            default="calm_bull",
        )

        group["rolling_vol"] = vol
        group["vol_regime"] = vol_regime
        group["drawdown"] = dd
        group["drawdown_regime"] = dd_regime
        group["regime"] = combined
        out_frames.append(group)

    return pd.concat(out_frames).sort_index()


def _causal_hmm_decode(model: GaussianHMM, features: np.ndarray, context_window: int) -> np.ndarray:
    """Decode each row from only a trailing `context_window` of history, so no
    later observation (even later within this same slice) can affect an
    earlier day's decoded state.
    """
    n = len(features)
    labels = np.full(n, -1, dtype=int)
    for t in range(n):
        start = max(0, t - context_window + 1)
        window = features[start : t + 1]
        if len(window) < 2:
            labels[t] = labels[t - 1] if t > 0 else 0
            continue
        labels[t] = model.predict(window)[-1]
    return labels


def hmm_regime_labels(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    returns_col: str = "returns",
    n_states: int = HMM_N_STATES,
    context_window: int = REGIME_VOL_MEDIAN_WINDOW,
    random_state: int = 0,
) -> pd.DataFrame:
    """Fit a GaussianHMM on the training slice only, then causally decode
    hidden states across the full series using a rolling context window.

    Also maps the fitted states to a binary `hmm_vol_regime` (comparable to
    the threshold method's `vol_regime`) by ranking states on their mean
    volatility feature — the state(s) with the highest mean vol become
    "high_vol", matching the threshold method's semantics for the kappa
    comparison in `regime_label_agreement`.
    """
    out_frames = []
    for ticker, group in full_df.groupby("Ticker"):
        group = group.sort_index().copy()
        vol = compute_rolling_vol(group[returns_col])
        feat = pd.DataFrame({"returns": group[returns_col], "vol": vol}).dropna()

        train_index = train_df.loc[train_df["Ticker"] == ticker].index
        train_feat = feat.loc[feat.index.isin(train_index)].values

        if len(train_feat) < n_states * 10:
            group["hmm_state"] = np.nan
            group["hmm_vol_regime"] = np.nan
            out_frames.append(group)
            continue

        model = GaussianHMM(
            n_components=n_states, covariance_type="diag", random_state=random_state, n_iter=100
        )
        model.fit(train_feat)

        decoded = _causal_hmm_decode(model, feat.values, context_window)
        vol_feature_idx = feat.columns.get_loc("vol")
        vol_rank = np.argsort(model.means_[:, vol_feature_idx])
        high_vol_state = vol_rank[-1]

        state_series = pd.Series(decoded, index=feat.index, name="hmm_state")
        vol_regime_series = pd.Series(
            np.where(decoded == high_vol_state, "high_vol", "normal"),
            index=feat.index,
            name="hmm_vol_regime",
        )
        group = group.join(state_series).join(vol_regime_series)
        out_frames.append(group)

    return pd.concat(out_frames).sort_index()


def regime_label_agreement(threshold_vol_regime: pd.Series, hmm_vol_regime: pd.Series) -> float:
    """Cohen's kappa between the threshold and HMM binary vol-regime labels —
    a robustness check that the threshold method's buckets aren't arbitrary.
    """
    mask = threshold_vol_regime.notna() & hmm_vol_regime.notna()
    if mask.sum() == 0:
        return float("nan")
    return cohen_kappa_score(threshold_vol_regime[mask], hmm_vol_regime[mask])
