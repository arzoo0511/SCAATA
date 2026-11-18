"""Price-only features for every stock on every week, and what each stock then did.

Eight features, each computed from closes up to and including the decision day and nothing later, then put on a
common scale *within each day across stocks* (so the model learns which stocks to prefer on a day, not what the
market did that day). The target is the stock's return over the next 20 sessions minus the average stock's, ranked
across stocks, so a crash or a rally that moves everything together carries no signal.

The signs in `SIGN` are fixed in advance from the literature and are only used by the equal-weight baseline the
learned model has to beat: momentum and trend are good, one-month and one-week returns revert, calm stocks beat
wild ones, and being near a 52-week high is good. The learned model is not told them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ("mom_1m", "mom_3m", "mom_6_1", "mom_12_1", "rev_1w", "calm_3m", "near_high", "trend_200")
SIGN = {"mom_1m": -1.0, "mom_3m": 1.0, "mom_6_1": 1.0, "mom_12_1": 1.0, "rev_1w": 1.0, "calm_3m": 1.0,
        "near_high": 1.0, "trend_200": 1.0}
HORIZON = 20
STEP = 5                   # one decision day a week
MIN_STOCKS = 30            # a day needs at least this many scored stocks to be used


def _raw_features(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each feature for every day and stock, using only that day's and earlier closes."""
    log_ret = np.log(closes).diff()
    return {
        "mom_1m": closes / closes.shift(21) - 1,
        "mom_3m": closes.shift(21) / closes.shift(63) - 1,
        "mom_6_1": closes.shift(21) / closes.shift(126) - 1,
        "mom_12_1": closes.shift(21) / closes.shift(252) - 1,
        "rev_1w": -(closes / closes.shift(5) - 1),
        "calm_3m": -log_ret.rolling(63, min_periods=60).std(),
        "near_high": closes / closes.rolling(252, min_periods=240).max() - 1,
        "trend_200": closes / closes.rolling(200, min_periods=190).mean() - 1,
    }


def standardise(frame: pd.DataFrame) -> pd.DataFrame:
    """Across stocks on each day: winsorise at 3 robust standard deviations, then z-score. NaN stays NaN."""
    med = frame.median(axis=1)
    mad = (frame.sub(med, axis=0)).abs().median(axis=1) * 1.4826
    clipped = frame.sub(med, axis=0).clip(lower=-3 * mad, upper=3 * mad, axis=0)
    sd = clipped.std(axis=1).replace(0, np.nan)
    return clipped.sub(clipped.mean(axis=1), axis=0).div(sd, axis=0)


def rank_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Each day's cross-section as ranks scaled to [-0.5, 0.5]; NaN stays NaN."""
    ranks = frame.rank(axis=1)
    n = frame.notna().sum(axis=1)
    return ranks.sub(0.5 * (n + 1), axis=0).div(n, axis=0)


def build_panel(closes: pd.DataFrame, step: int = STEP) -> dict:
    """Weekly rows: features (days x stocks x features), the 20-session relative return, and its rank."""
    raw = {k: standardise(v) for k, v in _raw_features(closes).items()}
    forward = closes.shift(-HORIZON) / closes - 1
    relative = forward.sub(forward.mean(axis=1), axis=0)
    positions = np.arange(252, len(closes), step)
    days = closes.index[positions]
    x = np.stack([raw[f].iloc[positions].values for f in FEATURES], axis=2)
    usable = np.isfinite(x).all(axis=2)
    day_ok = usable.sum(axis=1) >= MIN_STOCKS
    y = relative.iloc[positions].values
    return {"days": days, "symbols": list(closes.columns), "x": x, "y": y, "rank": rank_target(relative).iloc[positions].values,
            "forward": forward.iloc[positions].values, "positions": positions, "usable": usable & day_ok[:, None],
            "closes_index": closes.index}


def equal_weight_score(x: np.ndarray) -> np.ndarray:
    """The baseline: the average of the signed, standardised features. No learning."""
    sign = np.array([SIGN[f] for f in FEATURES])
    return np.nanmean(x * sign, axis=2)
