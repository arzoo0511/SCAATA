"""Statistical rigor additions: block-bootstrap CIs and paired significance
testing, since the v1 paper reported single point estimates with no spread
and no test for whether SCAATA's improvement over baselines was significant
versus just PPO/seed variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scaata.config import BOOTSTRAP_BLOCK_SIZE, BOOTSTRAP_N_RESAMPLES, CONFIDENCE_LEVEL


def _block_bootstrap_resample(returns: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Resample a returns series in contiguous blocks (not i.i.d. points) so
    the autocorrelation structure of daily returns is preserved."""
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
    blocks = [returns[s : s + block_size] for s in starts]
    resampled = np.concatenate(blocks)[:n]
    return resampled


def block_bootstrap_ci(
    equity_curve: np.ndarray,
    metric_fn,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = 0,
) -> dict:
    """Block-bootstrap confidence interval for a metric computed from a
    returns series (metric_fn takes an equity-curve-like array and returns
    a scalar, e.g. `sharpe_ratio`).

    Returns {point_estimate, ci_low, ci_high, resamples}.
    """
    equity = np.asarray(equity_curve, dtype=float)
    returns = pd.Series(equity).pct_change().dropna().values
    rng = np.random.default_rng(seed)

    point_estimate = metric_fn(equity)

    boot_values = np.empty(n_resamples)
    for i in range(n_resamples):
        resampled_returns = _block_bootstrap_resample(returns, block_size, rng)
        resampled_equity = np.insert((1 + resampled_returns).cumprod(), 0, 1.0)
        boot_values[i] = metric_fn(resampled_equity)

    alpha = 1 - confidence
    ci_low, ci_high = np.nanpercentile(boot_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "point_estimate": point_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
        "n_resamples": n_resamples,
    }


def paired_wilcoxon(
    scores_a: list[float], scores_b: list[float]
) -> dict:
    """Paired Wilcoxon signed-rank test between two methods' per-ticker (or
    per-fold) metric values — e.g. SCAATA's Sharpe vs vanilla-PPO's Sharpe on
    the same 6-9 tickers. Non-parametric, appropriate for the very small n
    typical of a per-ticker comparison.

    Returns {statistic, p_value, n, note} — `note` flags when n is too small
    (<6) for the test to be meaningful at all, since Wilcoxon needs a
    reasonable sample to have any power.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diffs = a - b
    diffs = diffs[~np.isnan(diffs)]

    if len(diffs) < 6:
        return {
            "statistic": float("nan"), "p_value": float("nan"), "n": len(diffs),
            "note": "n < 6: Wilcoxon has little power at this sample size; "
                    "treat any p-value here as indicative, not confirmatory.",
        }
    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0, "n": len(diffs), "note": "no difference between methods"}

    stat, p_value = wilcoxon(diffs)
    return {"statistic": float(stat), "p_value": float(p_value), "n": len(diffs), "note": ""}


def across_seed_summary(seed_results: list[float]) -> dict:
    """Mean/std/min/max of a metric across multiple PPO training seeds, so a
    single lucky/unlucky seed isn't mistaken for a real effect."""
    values = np.asarray(seed_results, dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "n_seeds": len(values),
    }
