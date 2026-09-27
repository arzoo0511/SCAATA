"""Multi-seed feature validation (Phase 20) -- fixes a methodology gap
caught live: single-seed comparisons this session (microstructure/novelty
features, VIX) kept showing a "helps one ticker, badly hurts another"
pattern. VIX specifically: AAPL 1.803 -> 2.255 Sharpe (real improvement),
GOOGL 1.541 -> -2.982 (severe regression) from the exact same feature
addition. At ~1500 training rows and a single PPO seed, that swing is at
least as likely to be pure stochastic-optimization variance as it is a
genuine effect of the feature -- a single seed cannot tell the two apart.

This reuses `DEFAULT_SEEDS` (already defined for `scaata.rl.ensemble`,
just never used for validation purposes before) to train N seeds per
configuration and report the seed-averaged Sharpe with its spread, so a
"config A beat config B" claim means something across the noise floor,
not "config A happened to get a luckier seed."
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from scaata.config import DEFAULT_SEEDS
from scaata.evaluation.metrics import sharpe_ratio
from scaata.rl.train import backtest_ppo, train_ppo

# Conservative default: each worker is a full Python process with its own
# torch runtime and env copy (~700MB observed). Running one per seed on a
# 16GB machine reproduced the memory exhaustion that killed the dashboard
# task earlier, so cap the pool rather than defaulting to len(seeds).
DEFAULT_MAX_WORKERS = 3


def _train_and_score_one_seed(args: tuple) -> float:
    """One seed's train+backtest, as a module-level function so it can be
    pickled to a worker process (Windows uses spawn, not fork).

    Each PPO run is fully determined by its own `seed` (both
    `env.reset(seed=...)` and `RecurrentPPO(seed=...)`), so running the
    seeds concurrently instead of sequentially cannot change any
    individual seed's result -- this is purely a wall-clock change.
    Separate processes additionally isolate global torch/RNG state, which
    a sequential loop shares.
    """
    train_df, holdout_df, feature_columns, ticker, seed, total_timesteps, train_ppo_kwargs = args

    # Each worker gets one thread: with several processes running tiny
    # networks, torch's intra-op threading costs more in synchronization
    # than it returns, and oversubscribes the CPU N-fold.
    import torch
    torch.set_num_threads(1)

    model = train_ppo(train_df, feature_columns, seed=seed, total_timesteps=total_timesteps, **train_ppo_kwargs)
    equity, _ = backtest_ppo(model, holdout_df, feature_columns, ticker)
    return sharpe_ratio(equity)


def multi_seed_sharpe(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_columns: list[str],
    ticker: str,
    seeds: list[int] = DEFAULT_SEEDS,
    total_timesteps: int = 100_000,
    max_workers: int | None = None,
    **train_ppo_kwargs,
) -> dict:
    """Trains one policy per seed on `train_df` (identical config except
    the seed) and backtests each on `holdout_df`, returning per-seed
    Sharpes plus the mean/std/min/max -- the std is the important number:
    if it's larger than the effect you're trying to detect, a single-seed
    "before vs after" comparison for this config is not trustworthy.
    `**train_ppo_kwargs` passes through to `train_ppo` unchanged (e.g.
    `use_differential_sharpe=True`, `dsr_benchmark_relative=True`), so
    this works for validating any config, not just feature additions.

    Seeds are trained in parallel worker processes (`max_workers`,
    default `DEFAULT_MAX_WORKERS`) rather than sequentially. Each seed's
    run is fully determined by its own seed, so this changes wall-clock
    time only, never the numbers. Pass `max_workers=1` to force the old
    sequential behavior (useful when memory is tight).
    """
    workers = max_workers if max_workers is not None else min(len(seeds), DEFAULT_MAX_WORKERS, os.cpu_count() or 1)

    payloads = [
        (train_df, holdout_df, feature_columns, ticker, seed, total_timesteps, train_ppo_kwargs)
        for seed in seeds
    ]

    if workers <= 1:
        sharpes = [_train_and_score_one_seed(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # map preserves input order, so sharpes[i] stays aligned with seeds[i].
            sharpes = list(pool.map(_train_and_score_one_seed, payloads))

    sharpes_arr = np.array(sharpes, dtype=float)
    finite = sharpes_arr[np.isfinite(sharpes_arr)]
    return {
        "ticker": ticker,
        "seeds": list(seeds),
        "sharpes": sharpes,
        "mean_sharpe": float(finite.mean()) if len(finite) else float("nan"),
        "std_sharpe": float(finite.std()) if len(finite) else float("nan"),
        "min_sharpe": float(finite.min()) if len(finite) else float("nan"),
        "max_sharpe": float(finite.max()) if len(finite) else float("nan"),
        "n_collapsed": int((~np.isfinite(sharpes_arr)).sum()),
    }


def compare_configs_multi_seed(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_columns_a: list[str],
    feature_columns_b: list[str],
    ticker: str,
    seeds: list[int] = DEFAULT_SEEDS,
    total_timesteps: int = 100_000,
    label_a: str = "baseline",
    label_b: str = "candidate",
    max_workers: int | None = None,
    **train_ppo_kwargs,
) -> dict:
    """Runs `multi_seed_sharpe` for two feature-column configs (e.g. base
    features vs. base+VIX) on the same ticker/train/holdout split, and
    reports whether B's mean Sharpe advantage over A is bigger than
    either config's own seed-to-seed spread -- a "different feature sets,
    same everything else" comparison isn't trustworthy if the difference
    between A and B is smaller than the noise within A or within B alone.
    """
    result_a = multi_seed_sharpe(train_df, holdout_df, feature_columns_a, ticker, seeds, total_timesteps, max_workers, **train_ppo_kwargs)
    result_b = multi_seed_sharpe(train_df, holdout_df, feature_columns_b, ticker, seeds, total_timesteps, max_workers, **train_ppo_kwargs)

    mean_diff = result_b["mean_sharpe"] - result_a["mean_sharpe"]
    noise_floor = max(result_a["std_sharpe"], result_b["std_sharpe"])
    exceeds_noise_floor = bool(np.isfinite(mean_diff) and np.isfinite(noise_floor) and abs(mean_diff) > noise_floor)

    return {
        "ticker": ticker,
        label_a: result_a,
        label_b: result_b,
        "mean_sharpe_diff": mean_diff,
        "noise_floor": noise_floor,
        "exceeds_noise_floor": exceeds_noise_floor,
    }
