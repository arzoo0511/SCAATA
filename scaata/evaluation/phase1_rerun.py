"""Honest re-run of the Phase 1 walk-forward comparison (audit, 2026-09-14).

`scaata.evaluation.phase1_pipeline` produced the result previously reported
as "PPO beats buy-and-hold, Wilcoxon p=0.0021", which was not valid evidence
(seed 0 only, direction-less test, non-independent pairs, fee-free
baselines). This re-runs the same experiment -- same RecurrentPPO, folds,
tickers and timesteps -- with those problems fixed:

- every seed is evaluated and saved, not just seed 0;
- every method pays `BASE_TRANSACTION_FEE` per trade side (PPO keeps the
  env's liquidity scaling and stop-loss, as before);
- the rule-based baseline uses moving averages computed over full history
  instead of sitting in cash for the first 50 days of every test window;
- the old "LLM agent" column is the deterministic momentum fallback, named
  for what it is -- no LLM is called;
- a policy that never trades counts as Sharpe 0 (cash), not a dropped row;
- the comparison is directional and clustered by fold
  (`scaata.evaluation.stats.fold_clustered_comparison`).

Two arms share folds, seeds and baselines: `vanilla` is the original setup;
`env_fixes` adds the audit's environment options (position state in the
observation, scale-free features, random episode starts).
`compare_arms` compares the two policies the same fold-clustered way.

Each (fold, seed) job saves its equity curves as soon as it finishes, so a
run interrupted by the laptop sleeping resumes where it left off, and the
statistics can be recomputed without retraining. A manifest records the git
commit, config and a data fingerprint; resuming with a different config
refuses to mix results. `--fold`/`--seed` run a subset of the full run's jobs
(e.g. one job to time it) without changing its manifest.

Usage:
    python -m scaata.evaluation.phase1_rerun run [--market us|india] [--arm ARM] [--fold N ...] [--seed N ...] [--workers N] [--smoke]
    python -m scaata.evaluation.phase1_rerun summarize [--market us|india] [--arm ARM] [--smoke]
    python -m scaata.evaluation.phase1_rerun compare [--market us|india] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scaata.config import (
    ALL_TICKERS,
    BASE_TRANSACTION_FEE,
    DEFAULT_SEEDS,
    END_DATE,
    FEATURE_COLUMNS,
    INDIA_SPREAD_COST,
    INDIA_STATUTORY_COST_PER_SIDE,
    INDIA_TICKERS,
    PPO_BATCH_SIZE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_TOTAL_TIMESTEPS,
    ROOT_DIR,
    START_DATE,
    STATIONARY_FEATURE_COLUMNS,
)
from scaata.data.loaders import load_market_data
from scaata.evaluation.metrics import sharpe_ratio
from scaata.evaluation.stats import fold_clustered_comparison
from scaata.evaluation.walkforward import Fold, generate_folds, split_fold
from scaata.features.normalize import normalize_data
from scaata.features.technical import add_features
from scaata.rl.llm_baseline import backtest_llm_agent
from scaata.rl.train import backtest_ppo, buy_and_hold_equity, rule_based_equity, train_ppo

RESULTS_ROOT = ROOT_DIR / "results" / "phase1_rerun"
SMOKE_RESULTS_ROOT = ROOT_DIR / "results" / "phase1_rerun_smoke"
METHODS = ["ppo", "buy_and_hold", "rule_based", "momentum_fallback"]
DEFAULT_MAX_WORKERS = 3  # ~700MB per worker; more has crashed other jobs on the 16GB dev laptop

# Per market: which tickers, and what a trade costs. Baselines pay
# `spread_fee + fixed_fee` per side; PPO's env scales `spread_fee` by
# liquidity and always adds `fixed_fee` (statutory charges don't shrink on
# busy days). The US settings reproduce the original run exactly.
MARKETS = {
    "us": {
        "tickers": list(ALL_TICKERS), "spread_fee": BASE_TRANSACTION_FEE, "fixed_fee": 0.0,
        "results_root": RESULTS_ROOT, "smoke_root": SMOKE_RESULTS_ROOT,
    },
    "india": {
        "tickers": list(INDIA_TICKERS), "spread_fee": INDIA_SPREAD_COST, "fixed_fee": INDIA_STATUTORY_COST_PER_SIDE,
        "results_root": ROOT_DIR / "results" / "phase1_rerun_india",
        "smoke_root": ROOT_DIR / "results" / "phase1_rerun_india_smoke",
    },
}

ARMS = {
    "vanilla": {
        "feature_columns": list(FEATURE_COLUMNS),
        "include_position_obs": False,
        "random_episode_start_min_steps": None,
    },
    "env_fixes": {
        "feature_columns": list(STATIONARY_FEATURE_COLUMNS),
        "include_position_obs": True,
        "random_episode_start_min_steps": 126,  # every training episode still covers ~6 months
    },
}


def _job_path(results_dir: Path, fold_id: int, seed: int) -> Path:
    return Path(results_dir) / f"fold{fold_id}_seed{seed}.pkl"


def run_fold_seed(
    featured_df: pd.DataFrame, fold: Fold, tickers: list[str], seed: int,
    ppo_timesteps: int, results_dir: Path, fee: float, arm: str = "vanilla",
    ppo_transaction_fee: float = BASE_TRANSACTION_FEE, ppo_fixed_fee: float = 0.0,
) -> Path:
    """Trains one policy pooled across `tickers` for `fold` with `seed`
    (exactly as `phase1_pipeline.run_phase1_backtest` does, plus `arm`'s env
    options), backtests it and every baseline on each ticker's test window,
    and saves the curves. A job whose file already exists is skipped."""
    path = _job_path(results_dir, fold.fold_id, seed)
    if path.exists():
        return path

    spec = ARMS[arm]
    columns = spec["feature_columns"]
    started = time.time()

    train_df, test_df = split_fold(featured_df, fold)
    train_df = train_df[train_df["Ticker"].isin(tickers)]
    test_df = test_df[test_df["Ticker"].isin(tickers)]
    train_norm, test_norm, _, _ = normalize_data(train_df, test_df, columns)
    model = train_ppo(
        train_norm, columns, seed=seed, total_timesteps=ppo_timesteps,
        include_position_obs=spec["include_position_obs"],
        random_episode_start_min_steps=spec["random_episode_start_min_steps"],
        transaction_fee=ppo_transaction_fee, fixed_fee=ppo_fixed_fee,
    )

    curves = {}
    present = set(test_df["Ticker"])
    for ticker in tickers:
        if ticker not in present:
            continue
        ppo_equity, ppo_actions = backtest_ppo(
            model, test_norm, columns, ticker, include_position_obs=spec["include_position_obs"],
            transaction_fee=ppo_transaction_fee, fixed_fee=ppo_fixed_fee,
        )
        rule_equity, _ = rule_based_equity(test_df, ticker, fee=fee, use_precomputed_mas=True)
        momentum_equity, _, _ = backtest_llm_agent(test_norm, columns, ticker, fee=fee, use_llm=False)
        curves[ticker] = {
            "ppo": np.asarray(ppo_equity),
            "buy_and_hold": buy_and_hold_equity(test_df, ticker, fee=fee),
            "rule_based": np.asarray(rule_equity),
            "momentum_fallback": np.asarray(momentum_equity),
            "ppo_actions": np.asarray(ppo_actions),
        }

    payload = {"fold": fold, "seed": seed, "arm": arm, "curves": curves, "elapsed_seconds": time.time() - started}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)  # atomic: an interrupted job never leaves a half-written result
    return path


def _run_job(args: tuple) -> Path:
    import torch
    torch.set_num_threads(1)  # several small-network workers: intra-op threads only oversubscribe the CPU
    return run_fold_seed(*args)


def _git_state() -> dict:
    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}


def _run_config(arm, tickers, seeds, ppo_timesteps, fee, folds, start, end, market="us",
                ppo_transaction_fee=BASE_TRANSACTION_FEE, ppo_fixed_fee=0.0) -> dict:
    return {
        "market": market, "arm": arm, "arm_spec": ARMS[arm],
        "ppo_fees": {"transaction_fee": ppo_transaction_fee, "fixed_fee": ppo_fixed_fee},
        "tickers": list(tickers), "seeds": list(seeds), "ppo_timesteps": ppo_timesteps, "fee": fee,
        "start": start, "end": end,
        "folds": [[f.fold_id, str(f.train_start.date()), str(f.test_start.date()), str(f.test_end.date())] for f in folds],
        "ppo": {"learning_rate": PPO_LEARNING_RATE, "n_steps": PPO_N_STEPS, "batch_size": PPO_BATCH_SIZE,
                "gamma": PPO_GAMMA, "ent_coef": PPO_ENT_COEF},
    }


def _write_or_check_manifest(results_dir: Path, config: dict, data_fingerprint: str) -> None:
    path = results_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["config"] != config or existing["data_fingerprint"] != data_fingerprint:
            raise RuntimeError(
                f"{path} was written for a different config or dataset -- use a fresh results dir "
                "rather than mixing runs."
            )
        return
    path.write_text(json.dumps({
        "config": config, "data_fingerprint": data_fingerprint, "git": _git_state(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")


def run_rerun(
    arm: str = "vanilla",
    tickers: list[str] | None = None,
    seeds: list[int] = DEFAULT_SEEDS,
    ppo_timesteps: int = PPO_TOTAL_TIMESTEPS,
    results_dir: Path | None = None,
    fee: float | None = None,
    max_folds: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    featured_df: pd.DataFrame | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
    only_folds: list[int] | None = None,
    only_seeds: list[int] | None = None,
    market: str = "us",
) -> list[Path]:
    """Runs (or resumes) every (fold, seed) job for `arm`. `only_folds` /
    `only_seeds` restrict which jobs run now; the manifest still describes
    the full run, so those jobs count toward it."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}; choose from {sorted(MARKETS)}")
    mkt = MARKETS[market]
    tickers = list(tickers) if tickers is not None else mkt["tickers"]
    fee = fee if fee is not None else mkt["spread_fee"] + mkt["fixed_fee"]
    ppo_transaction_fee, ppo_fixed_fee = mkt["spread_fee"], mkt["fixed_fee"]
    results_dir = Path(results_dir) if results_dir is not None else mkt["results_root"] / arm
    results_dir.mkdir(parents=True, exist_ok=True)
    if featured_df is None:
        featured_df = add_features(load_market_data(list(tickers), start, end, use_cache=True))
    folds = generate_folds(start, end)
    if max_folds:
        folds = folds[:max_folds]

    fingerprint = str(int(pd.util.hash_pandas_object(featured_df, index=True).sum()))
    config = json.loads(json.dumps(_run_config(arm, tickers, seeds, ppo_timesteps, fee, folds, start, end,
                                               market, ppo_transaction_fee, ppo_fixed_fee)))
    _write_or_check_manifest(results_dir, config, fingerprint)

    jobs = [
        (featured_df, fold, list(tickers), seed, ppo_timesteps, results_dir, fee, arm,
         ppo_transaction_fee, ppo_fixed_fee)
        for fold in folds for seed in seeds
        if (only_folds is None or fold.fold_id in only_folds) and (only_seeds is None or seed in only_seeds)
    ]
    pending = [j for j in jobs if not _job_path(results_dir, j[1].fold_id, j[3]).exists()]
    print(f"[{arm}] {len(jobs) - len(pending)}/{len(jobs)} selected jobs already done; running {len(pending)} "
          f"with {max_workers} worker(s)", flush=True)

    failures = []

    def _report(job, started):
        print(f"[{arm}] done: fold {job[1].fold_id} seed {job[3]} at {(time.time() - started) / 60:.1f} min elapsed", flush=True)

    if max_workers <= 1:
        for job in pending:
            started = time.time()
            try:
                run_fold_seed(*job)
                _report(job, started)
            except Exception as e:
                failures.append((job[1].fold_id, job[3], repr(e)))
                print(f"[{arm}] FAILED: fold {job[1].fold_id} seed {job[3]}: {e!r}", flush=True)
    else:
        started = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_job, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                    _report(job, started)
                except Exception as e:
                    failures.append((job[1].fold_id, job[3], repr(e)))
                    print(f"[{arm}] FAILED: fold {job[1].fold_id} seed {job[3]}: {e!r}", flush=True)

    if failures:
        raise RuntimeError(f"{len(failures)} job(s) failed (re-run to retry just those): {failures}")
    return [_job_path(results_dir, j[1].fold_id, j[3]) for j in jobs]


def _sharpe_table(results_dir: Path) -> tuple[pd.DataFrame, list[float]]:
    rows, elapsed = [], []
    for path in sorted(Path(results_dir).glob("fold*_seed*.pkl")):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if "elapsed_seconds" in payload:
            elapsed.append(payload["elapsed_seconds"])
        for ticker, curves in payload["curves"].items():
            row = {"seed": payload["seed"], "fold_id": payload["fold"].fold_id, "ticker": ticker}
            for method in METHODS:
                row[method] = sharpe_ratio(curves[method])
            row["ppo_never_traded"] = bool(np.allclose(np.diff(curves["ppo"]), 0.0))
            rows.append(row)
    return pd.DataFrame(rows), elapsed


def summarize_rerun(results_dir: Path) -> dict:
    """Recomputes every statistic from the saved curves (no retraining) and
    writes `sharpe_table.csv` and `summary.json` next to them."""
    results_dir = Path(results_dir)
    table, elapsed = _sharpe_table(results_dir)
    job_files = sorted(results_dir.glob("fold*_seed*.pkl"))

    manifest_path = results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    expected = len(manifest["config"]["folds"]) * len(manifest["config"]["seeds"]) if manifest else None

    summary = {
        "arm": manifest["config"]["arm"] if manifest else None,
        "jobs_found": len(job_files),
        "jobs_expected": expected,
        "complete": expected is not None and len(job_files) == expected,
        "median_job_minutes": float(np.median(elapsed) / 60) if elapsed else None,
        "rows": len(table),
        "mean_sharpe_never_traded_as_zero": {m: float(table[m].fillna(0.0).mean()) for m in METHODS} if len(table) else {},
        "median_sharpe_never_traded_as_zero": {m: float(table[m].fillna(0.0).median()) for m in METHODS} if len(table) else {},
        "ppo_never_traded_rows": int(table["ppo_never_traded"].sum()) if len(table) else 0,
        "comparisons": {f"ppo_vs_{m}": fold_clustered_comparison(table, "ppo", m) for m in METHODS[1:]} if len(table) else {},
        "git_at_start": manifest["git"] if manifest else None,
    }
    table.to_csv(results_dir / "sharpe_table.csv", index=False)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def compare_arms(baseline_dir: Path, candidate_dir: Path) -> dict:
    """Fold-clustered comparison of the candidate arm's PPO vs the baseline
    arm's PPO, over the (seed, fold, ticker) rows both have finished."""
    base, _ = _sharpe_table(baseline_dir)
    cand, _ = _sharpe_table(candidate_dir)
    if base.empty or cand.empty:
        return {"matched_rows": 0}
    keys = ["seed", "fold_id", "ticker"]
    merged = base[keys + ["ppo"]].merge(cand[keys + ["ppo"]], on=keys, suffixes=("_baseline", "_candidate"))
    result = fold_clustered_comparison(merged, "ppo_candidate", "ppo_baseline") if len(merged) else {}
    result["matched_rows"] = len(merged)
    return result


def _print_summary(summary: dict) -> None:
    print(f"[{summary['arm']}] jobs: {summary['jobs_found']}/{summary['jobs_expected']}  rows: {summary['rows']}  "
          f"PPO never traded: {summary['ppo_never_traded_rows']}  median job: {summary['median_job_minutes']} min")
    for method, value in summary["mean_sharpe_never_traded_as_zero"].items():
        print(f"  mean Sharpe {method:18} {value:+.3f}   median {summary['median_sharpe_never_traded_as_zero'][method]:+.3f}")
    for name, c in summary["comparisons"].items():
        print(f"  {name:28} mean fold diff {c['mean_diff']:+.3f}  95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  "
              f"PPO better in {c['share_folds_a_better']:.0%} of {c['n_folds']} folds  one-sided p={c['p_one_sided_a_better']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run", "summarize", "compare"])
    parser.add_argument("--arm", choices=sorted(ARMS), default="vanilla")
    parser.add_argument("--market", choices=sorted(MARKETS), default="us")
    parser.add_argument("--fold", type=int, nargs="*", help="run only these fold ids")
    parser.add_argument("--seed", type=int, nargs="*", help="run only these seeds")
    parser.add_argument("--smoke", action="store_true", help="3 tickers, 2 folds, seed 0, 2048 timesteps")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()

    mkt = MARKETS[args.market]
    root = mkt["smoke_root"] if args.smoke else mkt["results_root"]
    if args.command == "compare":
        result = compare_arms(root / "vanilla", root / "env_fixes")
        if not result.get("matched_rows"):
            print("no (seed, fold, ticker) rows finished in both arms yet")
            return
        print(f"env_fixes PPO vs vanilla PPO over {result['matched_rows']} matched rows: "
              f"mean fold diff {result['mean_diff']:+.3f}  95% CI [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]  "
              f"better in {result['share_folds_a_better']:.0%} of {result['n_folds']} folds  one-sided p={result['p_one_sided_a_better']:.3f}")
        return

    results_dir = root / args.arm
    if args.command == "run":
        common = dict(arm=args.arm, results_dir=results_dir, max_workers=args.workers,
                      only_folds=args.fold, only_seeds=args.seed, market=args.market)
        if args.smoke:
            run_rerun(tickers=mkt["tickers"][:3], seeds=[0], ppo_timesteps=PPO_N_STEPS, max_folds=2, **common)
        else:
            run_rerun(**common)
    _print_summary(summarize_rerun(results_dir))


if __name__ == "__main__":
    main()
