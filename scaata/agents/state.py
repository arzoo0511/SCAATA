"""Shared state threaded through the LangGraph inner-loop pipeline.

This is what makes the paper's diagram (Critique -> Meta-Selector feedback
edge) real executing code instead of a description: every node reads and
writes this single state object as the graph cycles.
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # --- config threaded through, set once at graph invocation ---
    train_df: Any            # pd.DataFrame with technical features
    feature_columns: list[str]
    review_window: int
    max_iterations: int

    # --- scraper -> normalizer ---
    raw_scripts: list[dict]              # [{"source", "code"}]
    normalized_strategies: list[dict]    # [{"source", "clean_code"}]

    # --- strategy pool + per-strategy down-weighting (the feedback target) ---
    strategy_signals: list[dict]         # [{"source", "signals"}]
    strategy_pool_weights: list[float]   # multiplier per strategy, same order as strategy_signals

    # --- regime classifier (thin wrapper reusing Phase 1, no recompute) ---
    regime_labels: Any                   # pd.Series aligned to train_df

    # --- meta-selector / BC outputs ---
    bc_model: Any
    meta_model: Any
    strategy_weight_matrix: Any          # np.ndarray (n_states, K) softmax
    num_strategies: int

    # --- critique / feedback loop control ---
    critique_report: dict
    iteration: int
    converged: bool
    history: list[dict]                  # per-iteration snapshot, for inspection/tests
