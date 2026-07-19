"""Entry point that builds the initial shared state, invokes the compiled
LangGraph, and hands the results off to RL training: `bc_model` for
`rl/policy_init.py`, and `meta_model`/`strategy_signals` for computing the
meta-confidence and meta-implied-action columns (`strategies/meta_selector.py`)
that get merged into the training/test DataFrame before `RobustTradingEnv`
is constructed.
"""
from __future__ import annotations

from scaata.agents.graph import build_graph
from scaata.config import FEATURE_COLUMNS, MAX_GRAPH_ITERATIONS, REVIEW_WINDOW_DAYS


def run_inner_loop(
    train_df,
    feature_columns: list[str] = FEATURE_COLUMNS,
    review_window: int = REVIEW_WINDOW_DAYS,
    max_iterations: int = MAX_GRAPH_ITERATIONS,
) -> dict:
    """Runs the full Scraper -> Normalizer -> Regime-Classifier ->
    Meta-Selector <-> Critique graph to completion and returns the final
    state, ready for `scaata.rl.policy_init` and
    `scaata.strategies.meta_selector.attach_meta_confidence/attach_meta_implied_action`.
    """
    graph = build_graph()

    initial_state = {
        "train_df": train_df,
        "feature_columns": feature_columns,
        "review_window": review_window,
        "max_iterations": max_iterations,
        "iteration": 0,
        "strategy_pool_weights": None,
        "history": [],
    }

    # recursion_limit bounds total node visits; each loop iteration revisits
    # meta_selector + critique (2 nodes), plus the 3 one-shot upstream nodes.
    final_state = graph.invoke(initial_state, config={"recursion_limit": 3 + max_iterations * 2 + 5})
    return final_state
