"""Performance metrics, extended to report per-regime, not just blended.

Ported from the v1 notebook's compute_sharpe/compute_drawdown and extended
with Sortino, turnover, and an optional regime mask so each metric can be
computed for the full series and for each regime bucket separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.config import TRADING_DAYS_PER_YEAR


def _returns(equity_curve: np.ndarray | pd.Series) -> pd.Series:
    return pd.Series(equity_curve).pct_change().dropna()


def sharpe_ratio(equity_curve, annualize: bool = True) -> float:
    r = _returns(equity_curve)
    if r.std() == 0 or len(r) == 0:
        return float("nan")
    ratio = r.mean() / (r.std() + 1e-8)
    return ratio * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else ratio


def sortino_ratio(equity_curve, annualize: bool = True) -> float:
    r = _returns(equity_curve)
    downside = r[r < 0]
    if len(r) == 0 or downside.std() == 0:
        return float("nan")
    ratio = r.mean() / (downside.std() + 1e-8)
    return ratio * np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else ratio


def max_drawdown(equity_curve) -> float:
    equity = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    return float(drawdowns.min())


def turnover(actions: np.ndarray) -> float:
    """Fraction of steps where the agent traded (action != hold=0)."""
    actions = np.asarray(actions)
    if len(actions) == 0:
        return float("nan")
    return float(np.mean(actions != 0))


def total_return(equity_curve) -> float:
    equity = np.asarray(equity_curve, dtype=float)
    return float(equity[-1] / equity[0] - 1)


SMALL_SAMPLE_ANNUALIZE_WARN_DAYS = 40


def metrics_for_mask(
    equity_curve: np.ndarray,
    actions: np.ndarray | None,
    mask: np.ndarray | None = None,
) -> dict:
    """Compute all metrics over the (optionally masked) sub-slice of the
    equity curve. `mask` selects which *return* indices belong to a regime;
    equity_curve is one longer than returns, so we reconstruct the sub-curve
    from cumulative returns of just the masked days rather than slicing the
    raw equity level (avoids mixing in unrelated-day price gaps).
    """
    equity = np.asarray(equity_curve, dtype=float)
    full_returns = pd.Series(equity).pct_change().dropna().reset_index(drop=True)

    if mask is not None:
        mask = np.asarray(mask)[: len(full_returns)]
        sub_returns = full_returns[mask]
    else:
        sub_returns = full_returns

    n_days = len(sub_returns)
    if n_days == 0:
        return {
            "sharpe": float("nan"), "sortino": float("nan"),
            "max_drawdown": float("nan"), "turnover": float("nan"),
            "total_return": float("nan"), "n_days": 0, "small_sample": True,
        }

    sub_equity = (1 + sub_returns).cumprod().values
    sub_equity = np.insert(sub_equity, 0, 1.0)

    result = {
        "sharpe": sharpe_ratio(sub_equity),
        "sortino": sortino_ratio(sub_equity),
        "max_drawdown": max_drawdown(sub_equity),
        "total_return": total_return(sub_equity),
        "n_days": n_days,
        "small_sample": n_days < SMALL_SAMPLE_ANNUALIZE_WARN_DAYS,
    }
    if actions is not None:
        actions = np.asarray(actions)
        if mask is not None:
            mask_a = np.asarray(mask)[: len(actions)]
            result["turnover"] = turnover(actions[mask_a])
        else:
            result["turnover"] = turnover(actions)
    else:
        result["turnover"] = float("nan")
    return result


def regime_metrics_table(
    equity_curve: np.ndarray,
    actions: np.ndarray,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """Per-regime metrics table plus an 'ALL' aggregate row.

    `regime_labels` must be aligned to the equity curve's *return* index
    (i.e. len(regime_labels) == len(equity_curve) - 1), one label per day.
    Small-sample regimes (< SMALL_SAMPLE_ANNUALIZE_WARN_DAYS days) are
    flagged so annualized Sharpe isn't over-trusted on a handful of days.
    """
    rows = {"ALL": metrics_for_mask(equity_curve, actions, mask=None)}
    for regime in pd.unique(regime_labels.dropna()):
        mask = (regime_labels == regime).values
        rows[regime] = metrics_for_mask(equity_curve, actions, mask=mask)
    return pd.DataFrame(rows).T
