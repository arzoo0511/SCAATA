"""The statistics that keep the Lab honest about how much a result is worth.

Four tools, each answering one question, all using only the standard
library and numpy so they run on the hosted dashboard and in the daily job:

    Deflated Sharpe Ratio     "this strategy has Sharpe 1.0 -- but I tried 40
                               variants. How likely is it that the best of 40
                               noise strategies looks this good?"
    Backtest overfitting      "if I always picked whichever variant did best on
    probability (CSCV)         part of the history, how often would my pick land
                               below the median on the rest?"
    E-process                 "has the live record shown a real difference yet?"
                               Valid to look at every single day: a fixed-sample
                               p-value is not, and peeking at it inflates errors.
    Evidence budget           "how much false-alarm risk am I still allowed to
                               spend on new ideas?" (alpha-investing)

References: Bailey and Lopez de Prado, "The Deflated Sharpe Ratio" (2014);
Bailey, Borwein, Lopez de Prado and Zhu, "The probability of backtest
overfitting" (2017); Howard, Ramdas, McAuliffe and Sekhon, "Time-uniform,
nonparametric, nonasymptotic confidence sequences" (2021); Shafer, "Testing
by betting" (2021); Foster and Stine, "Alpha-investing" (2008).
"""
from __future__ import annotations

import itertools
from statistics import NormalDist

import numpy as np

NORMAL = NormalDist()
EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


# ------------------------------------------------------------ Sharpe, deflated
def sharpe_per_period(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    sd = returns.std(ddof=1) if len(returns) > 1 else 0.0
    return float(returns.mean() / sd) if sd > 0 else 0.0


def moments(curve: np.ndarray) -> dict:
    """Skewness and (raw) kurtosis of the daily returns of an equity curve; a normal return has 0 and 3."""
    curve = np.asarray(curve, dtype=float)
    r = np.diff(curve) / curve[:-1]
    if len(r) < 8 or r.std() == 0:
        return {"skew": 0.0, "kurt": 3.0}
    z = (r - r.mean()) / r.std()
    return {"skew": float((z ** 3).mean()), "kurt": float((z ** 4).mean())}


def psr(sr: float, benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio: the chance the true Sharpe exceeds `benchmark`, given a Sharpe of `sr`
    measured over `n_obs` periods. All Sharpe values are per period (not annualised)."""
    if n_obs < 3:
        return 0.5
    denominator = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denominator <= 0:
        return 0.5
    return NORMAL.cdf((sr - benchmark) * np.sqrt(n_obs - 1) / np.sqrt(denominator))


def expected_max_sharpe(n_trials: int, variance_of_trials: float) -> float:
    """The Sharpe the best of `n_trials` zero-skill strategies is expected to show, per period."""
    if n_trials < 2 or variance_of_trials <= 0:
        return 0.0
    first = NORMAL.inv_cdf(1 - 1 / n_trials)
    second = NORMAL.inv_cdf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(variance_of_trials) * ((1 - EULER_GAMMA) * first + EULER_GAMMA * second))


def deflated_sharpe(sr: float, n_obs: int, trial_srs: list[float], skew: float = 0.0, kurt: float = 3.0,
                    benchmark: float = 0.0) -> dict:
    """The Deflated Sharpe Ratio: PSR against the Sharpe that luck alone would produce after the number of
    variants tried, on top of `benchmark` (zero, or the Sharpe of the thing being beaten, e.g. holding).
    `trial_srs` are the per-period Sharpe ratios of every variant tried (including this one).

    The dispersion of `trial_srs` sets how much luck to expect. If the variants are near-copies of one
    another that dispersion is small and the deflation is mild: the number of *independent* tries is
    then much smaller than the number of tries, and the result should be read with that in mind."""
    n = len(trial_srs)
    variance = float(np.var(trial_srs, ddof=1)) if n > 1 else 0.0
    luck = expected_max_sharpe(n, variance)
    return {"dsr": psr(sr, benchmark + luck, n_obs, skew, kurt),
            "psr_vs_benchmark": psr(sr, benchmark, n_obs, skew, kurt),
            "trials": n, "luck_threshold_annual": luck * np.sqrt(TRADING_DAYS),
            "sharpe_annual": sr * np.sqrt(TRADING_DAYS), "benchmark_annual": benchmark * np.sqrt(TRADING_DAYS)}


# ----------------------------------------------------- probability of overfit
def pbo(matrix: np.ndarray, blocks: int = 8) -> dict:
    """Probability of Backtest Overfitting by combinatorially symmetric cross-validation.

    `matrix` is periods x variants of per-period returns. The history is cut into `blocks` contiguous
    blocks; for every way of choosing half of them as "train", the variant that did best on train is found
    and its rank on the other half is recorded. If the pick is usually below the median out of sample,
    choosing by past performance is not working."""
    matrix = np.asarray(matrix, dtype=float)
    periods, variants = matrix.shape
    if variants < 2 or periods < blocks * 5:
        return {"pbo": None, "splits": 0, "variants": variants, "reason": "too few variants or too short"}
    edges = np.linspace(0, periods, blocks + 1).astype(int)
    parts = [matrix[edges[i]:edges[i + 1]] for i in range(blocks)]
    logits = []
    for train in itertools.combinations(range(blocks), blocks // 2):
        test = [b for b in range(blocks) if b not in train]
        a, b = np.vstack([parts[i] for i in train]), np.vstack([parts[i] for i in test])
        sr_a = a.mean(axis=0) / np.where(a.std(axis=0) > 0, a.std(axis=0), np.inf)
        sr_b = b.mean(axis=0) / np.where(b.std(axis=0) > 0, b.std(axis=0), np.inf)
        pick = int(np.argmax(sr_a))
        rank = 1 + int((sr_b < sr_b[pick]).sum())          # 1 = worst, variants = best
        omega = rank / (variants + 1)
        logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "splits": len(logits), "variants": variants,
            "median_logit": float(np.median(logits))}


# ----------------------------------------------------------------- e-process
SCALE = 0.02            # a daily difference of 2% or more counts as the largest it can be; fixed in advance
LAMBDA_MAX = 0.5
PRIOR_VARIANCE = 0.1    # prior guess for the variance of a scaled difference, so early bets stay small
THRESHOLD = 20.0        # evidence of 20 to 1 = 5% error however often it is checked


def _bets(x: np.ndarray, sign: float) -> np.ndarray:
    """E_t = product of (1 + lambda_t * sign * x_t), with lambda_t decided from the past only."""
    e, path, total, total_sq = 1.0, np.empty(len(x)), 0.0, 0.0
    for t, value in enumerate(x):
        mean = total / (t + 1)
        variance = (PRIOR_VARIANCE + total_sq) / (t + 1)
        lam = min(max(mean / (variance + mean * mean), 0.0), LAMBDA_MAX)
        e *= 1 + lam * sign * value
        path[t] = e
        total += sign * value
        total_sq += value * value
    return path


def e_process(diffs: np.ndarray, scale: float = SCALE) -> dict:
    """Running evidence that the mean of `diffs` is above zero (`up`) or below zero (`down`).

    Each path starts at 1 and is a fair-bet wealth process under "no difference": it exceeds 1/alpha
    with probability at most alpha, however many times it is read. 20 means 5% error. This makes a daily
    live meter honest; it does not make a small effect easy to find."""
    x = np.clip(np.asarray(diffs, dtype=float) / scale, -1.0, 1.0)
    up, down = _bets(x, 1.0), _bets(x, -1.0)
    return {"up": up, "down": down, "up_final": float(up[-1]) if len(up) else 1.0,
            "down_final": float(down[-1]) if len(down) else 1.0, "n": int(len(x)),
            "mean": float(np.mean(diffs)) if len(diffs) else 0.0}


# ------------------------------------------------------------ evidence budget
INITIAL_WEALTH = 0.10
PAYOUT = 0.05
SPEND_SHARE = 0.5


def offered_level(wealth: float) -> float:
    """The false-alarm level the next test may use: half the remaining wealth."""
    return max(wealth, 0.0) * SPEND_SHARE


def spend(wealth: float, p_value: float | None) -> tuple[float, float, bool]:
    """Runs one test through the budget. Returns (level used, wealth afterwards, whether it was a discovery).
    A failure costs alpha/(1-alpha); a discovery pays back a fixed amount (Foster and Stine 2008)."""
    alpha = offered_level(wealth)
    if p_value is None:
        return alpha, wealth, False
    if p_value <= alpha:
        return alpha, wealth + PAYOUT, True
    return alpha, wealth - alpha / (1 - alpha), False


def ledger(p_values: list[tuple[str, float | None]]) -> list[dict]:
    """Replays the budget over experiments in the order they were run: [(id, p-value or None), ...]."""
    wealth, rows = INITIAL_WEALTH, []
    for name, p in p_values:
        before = wealth
        alpha, wealth, discovery = spend(wealth, p)
        rows.append({"id": name, "p": p, "level": alpha, "discovery": discovery, "wealth_before": before,
                     "wealth_after": wealth if p is not None else before})
    return rows
