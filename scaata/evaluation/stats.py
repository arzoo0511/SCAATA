"""Statistical rigor additions: block-bootstrap CIs and paired significance
testing, since the v1 paper reported single point estimates with no spread
and no test for whether SCAATA's improvement over baselines was significant
versus just PPO/seed variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scaata.config import BOOTSTRAP_BLOCK_SIZE, BOOTSTRAP_N_RESAMPLES, CONFIDENCE_LEVEL, TRADING_DAYS_PER_YEAR


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


def _annualized_sharpe_rows(returns: np.ndarray) -> np.ndarray:
    """Row-wise annualized Sharpe; a zero-variance row (flat equity, e.g. a
    policy that never trades) scores 0 -- the same as holding cash -- rather
    than NaN, so it can still be compared."""
    sd = returns.std(axis=1, ddof=1)
    mean = returns.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sd > 1e-12, mean / sd, 0.0)
    return ratio * np.sqrt(TRADING_DAYS_PER_YEAR)


def paired_block_bootstrap_prob_better(
    equity_a,
    equity_b,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    seed: int = 0,
) -> float:
    """Probability that strategy A's annualized Sharpe beats B's over the
    same days. Both return series are resampled with the SAME blocks, so a
    market-wide move lands on both at once and only the difference between
    the strategies drives the answer -- two point Sharpes on ~40 days (SE of
    ~2.5 each) cannot make that call. Curves must cover the same days;
    returns are aligned from the start and truncated to the shorter one."""
    ra = pd.Series(np.asarray(equity_a, dtype=float)).pct_change().dropna().values
    rb = pd.Series(np.asarray(equity_b, dtype=float)).pct_change().dropna().values
    n = min(len(ra), len(rb))
    if n < 2:
        return float("nan")
    ra, rb = ra[:n], rb[:n]

    rng = np.random.default_rng(seed)
    block = min(block_size, n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_resamples, -1)[:, :n]
    return float(np.mean(_annualized_sharpe_rows(ra[idx]) > _annualized_sharpe_rows(rb[idx])))


def fold_clustered_comparison(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    fold_col: str = "fold_id",
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = 0,
) -> dict:
    """Directional comparison of `method_a` vs `method_b` Sharpe, treating
    the walk-forward fold -- not the (ticker, fold) pair -- as the unit.

    `df` has one row per (seed, fold, ticker) with a Sharpe column per
    method. Tickers in the same fold share one six-month market window and
    seeds share the same data, so they are not independent observations:
    `paired_wilcoxon` over 79 such pairs claimed p=0.002 for a difference
    whose mean pointed the other way. Here each fold contributes one number,
    the mean of (a - b) across its tickers and seeds.

    NaN Sharpe (a flat curve: a policy that never traded) counts as 0, the
    Sharpe of holding cash, instead of being dropped; the counts are
    reported. Returns the mean and median fold difference, a bootstrap CI
    over folds, the share of folds where A beat B, and a one-sided sign-flip
    permutation p-value for "A beats B" (exact for up to 16 folds).
    """
    a = df[method_a].astype(float)
    b = df[method_b].astype(float)
    collapsed_a, collapsed_b = int(a.isna().sum()), int(b.isna().sum())
    diffs = (a.fillna(0.0) - b.fillna(0.0)).groupby(df[fold_col]).mean().values
    k = len(diffs)
    if k == 0:
        return {"n_folds": 0, "n_rows": len(df), "mean_diff": float("nan"), "p_one_sided_a_better": float("nan")}

    observed = diffs.mean()
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, k, size=(n_resamples, k))].mean(axis=1)
    alpha = 1 - confidence
    ci_low, ci_high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    if k <= 16:
        signs = ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1) * 2 - 1
    else:
        signs = rng.choice([-1, 1], size=(20_000, k))
    flipped_means = (signs * np.abs(diffs)).mean(axis=1)
    p_value = float(np.mean(flipped_means >= observed - 1e-12))

    return {
        "method_a": method_a, "method_b": method_b,
        "n_folds": k, "n_rows": len(df),
        "mean_diff": float(observed), "median_fold_diff": float(np.median(diffs)),
        "ci_low": float(ci_low), "ci_high": float(ci_high), "confidence": confidence,
        "share_folds_a_better": float(np.mean(diffs > 0)),
        "p_one_sided_a_better": p_value,
        "fold_diffs": diffs.tolist(),
        "collapsed_a": collapsed_a, "collapsed_b": collapsed_b,
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
