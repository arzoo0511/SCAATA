"""Phase 5: does the decomposed LangGraph pipeline's strategy weighting
track actual profitability better than a single monolithic LLM call's
one-shot self-assessed confidence? Both start from the same raw strategy
pool, so total LLM call count is directly comparable too.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from scaata.agents.monolithic import mock_monolithic_pass, monolithic_normalize_and_weight
from scaata.agents.orchestrator import run_inner_loop
from scaata.strategies.pool import build_strategy_pool_signals


def actual_strategy_profitability(strategy_signals: list[dict], train_df: pd.DataFrame) -> dict[str, float]:
    """Ground truth: each strategy's realized signal-weighted return sum
    over the training window — the numeric anchor both weighting schemes
    are compared against."""
    future_returns = train_df["returns"].shift(-1).fillna(0).values
    profitability = {}
    for s in strategy_signals:
        signals = np.asarray(s["signals"])
        n = min(len(signals), len(future_returns))
        profitability[s["source"]] = float(np.sum(signals[:n] * future_returns[:n]))
    return profitability


def _weight_vs_profit_correlation(weights: dict[str, float], profits: dict[str, float]) -> float:
    common = [k for k in weights if k in profits]
    if len(common) < 3:
        return float("nan")
    w = np.array([weights[k] for k in common])
    p = np.array([profits[k] for k in common])
    if w.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(w, p)[0, 1])


def compare_monolithic_vs_decomposed(
    train_df: pd.DataFrame,
    raw_strategies: list[dict],
    feature_columns: list[str],
    max_iterations: int = 4,
) -> dict:
    """Runs both the decomposed LangGraph pipeline and a monolithic-LLM
    pass over the same `raw_strategies`, then compares how well each
    scheme's confidence/weight tracks the strategies' actual realized
    profitability on `train_df`.
    """
    final_state = run_inner_loop(train_df, feature_columns=feature_columns, max_iterations=max_iterations)
    decomposed_signals = final_state["strategy_signals"]
    decomposed_weights = {
        s["source"]: w for s, w in zip(decomposed_signals, final_state["strategy_pool_weights"])
    }

    if os.environ.get("GROQ_API_KEY"):
        mono_raw_results = [monolithic_normalize_and_weight(s["code"]) for s in raw_strategies]
        mono_results = [
            {**r, "source": s["source"]} for r, s in zip(mono_raw_results, raw_strategies) if r is not None
        ]
    else:
        mono_results = mock_monolithic_pass(raw_strategies)

    mono_pool = [{"source": r["source"], "clean_code": r["clean_code"]} for r in mono_results]
    mono_signals = build_strategy_pool_signals(mono_pool, train_df)
    mono_weights = {r["source"]: r["weight"] for r in mono_results}

    actual_profit_decomposed = actual_strategy_profitability(decomposed_signals, train_df)
    actual_profit_mono = actual_strategy_profitability(mono_signals, train_df)

    return {
        "decomposed_weight_vs_profit_corr": _weight_vs_profit_correlation(decomposed_weights, actual_profit_decomposed),
        "monolithic_weight_vs_profit_corr": _weight_vs_profit_correlation(mono_weights, actual_profit_mono),
        "n_decomposed_llm_calls": len(raw_strategies),  # normalizer_node: 1 call per raw strategy
        "n_monolithic_llm_calls": len(raw_strategies),  # 1 combined call per raw strategy
        "decomposed_weights": decomposed_weights,
        "monolithic_weights": mono_weights,
        "decomposed_iterations": final_state["iteration"],
    }
