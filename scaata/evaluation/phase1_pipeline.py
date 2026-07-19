"""Phase 1 end-to-end orchestration: data -> features -> regimes -> walk-forward
backtest of buy-and-hold / rule-based / PPO baselines, with statistics.

Kept in the package (not the notebook) so it's reusable and testable; the
notebook should only call `run_phase1_backtest` and render the results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scaata.config import FEATURE_COLUMNS, PPO_TOTAL_TIMESTEPS
from scaata.evaluation.metrics import regime_metrics_table
from scaata.evaluation.stats import paired_wilcoxon
from scaata.evaluation.walkforward import Fold, split_fold
from scaata.features.normalize import normalize_data
from scaata.regimes.detector import threshold_regime_labels
from scaata.rl.train import backtest_ppo, buy_and_hold_equity, rule_based_equity, train_ppo


@dataclass
class TickerFoldResult:
    fold_id: int
    ticker: str
    equity: dict = field(default_factory=dict)   # method -> equity curve
    actions: dict = field(default_factory=dict)  # method -> actions array
    regimes: pd.Series = None


def run_single_fold_ticker(
    fold: Fold,
    ticker: str,
    train_norm: pd.DataFrame,
    test_norm: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_columns: list[str],
    ppo_model,
) -> TickerFoldResult:
    result = TickerFoldResult(fold_id=fold.fold_id, ticker=ticker)

    ppo_equity, ppo_actions = backtest_ppo(ppo_model, test_norm, feature_columns, ticker)
    result.equity["ppo"] = ppo_equity
    result.actions["ppo"] = ppo_actions

    bah_equity = buy_and_hold_equity(test_raw, ticker)
    result.equity["buy_and_hold"] = bah_equity
    result.actions["buy_and_hold"] = np.zeros(len(bah_equity) - 1)

    rb_equity, rb_actions = rule_based_equity(test_raw, ticker)
    result.equity["rule_based"] = rb_equity
    result.actions["rule_based"] = rb_actions

    ticker_test = test_raw[test_raw["Ticker"] == ticker]
    thresh = threshold_regime_labels(ticker_test)
    result.regimes = thresh["regime"]

    return result


def run_phase1_backtest(
    tickers: list[str],
    featured_df: pd.DataFrame,
    folds: list[Fold],
    feature_columns: list[str] = FEATURE_COLUMNS,
    ppo_timesteps: int = PPO_TOTAL_TIMESTEPS,
    seed: int = 0,
) -> list[TickerFoldResult]:
    """Runs one PPO training per fold (sampling across all `tickers`), then
    backtests + baselines per ticker within that fold's test window.
    """
    results: list[TickerFoldResult] = []

    for fold in folds:
        train_df, test_df = split_fold(featured_df, fold)
        train_df = train_df[train_df["Ticker"].isin(tickers)]
        test_df = test_df[test_df["Ticker"].isin(tickers)]
        if train_df.empty or test_df.empty:
            continue

        train_norm, test_norm, _, _ = normalize_data(train_df, test_df, feature_columns)
        model = train_ppo(train_norm, feature_columns, seed=seed, total_timesteps=ppo_timesteps)

        for ticker in tickers:
            if ticker not in test_df["Ticker"].unique():
                continue
            fold_result = run_single_fold_ticker(
                fold, ticker, train_norm, test_norm, test_df, feature_columns, model
            )
            results.append(fold_result)

    return results


def summarize_results(results: list[TickerFoldResult], method: str, metric: str = "sharpe") -> dict:
    """Per-regime metrics table pooled across all folds/tickers for one
    method (ppo / buy_and_hold / rule_based), plus a bootstrap CI on the
    concatenated-fold Sharpe and a cross-ticker Wilcoxon vs the other
    methods.
    """
    per_ticker_metric = {}
    all_tables = []
    for r in results:
        equity = r.equity[method]
        actions = r.actions[method]
        table = regime_metrics_table(equity, actions, r.regimes)
        table["ticker"] = r.ticker
        table["fold_id"] = r.fold_id
        all_tables.append(table)
        per_ticker_metric[(r.ticker, r.fold_id)] = table.loc["ALL", metric]

    pooled = pd.concat(all_tables)
    return {"pooled_table": pooled, "per_ticker_metric": per_ticker_metric}


def compare_methods_across_tickers(
    results: list[TickerFoldResult], metric: str = "sharpe"
) -> pd.DataFrame:
    """One row per (ticker, fold), one column per method — the basis for the
    paired Wilcoxon test between methods."""
    rows = []
    for r in results:
        row = {"ticker": r.ticker, "fold_id": r.fold_id}
        for method in ["buy_and_hold", "rule_based", "ppo"]:
            table = regime_metrics_table(r.equity[method], r.actions[method], r.regimes)
            row[method] = table.loc["ALL", metric]
        rows.append(row)
    return pd.DataFrame(rows)


def significance_report(comparison_df: pd.DataFrame) -> dict:
    return {
        "ppo_vs_buy_and_hold": paired_wilcoxon(comparison_df["ppo"], comparison_df["buy_and_hold"]),
        "ppo_vs_rule_based": paired_wilcoxon(comparison_df["ppo"], comparison_df["rule_based"]),
    }
