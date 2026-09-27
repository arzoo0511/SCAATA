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
    ensemble_disagreement: float | None = None,
) -> dict:
    """Runs the full Scraper -> Normalizer -> Regime-Classifier ->
    Meta-Selector <-> Critique graph to completion and returns the final
    state, ready for `scaata.rl.policy_init` and
    `scaata.strategies.meta_selector.attach_meta_confidence/attach_meta_implied_action`.

    `ensemble_disagreement` (Phase 16, optional): pass in Phase 14's 5-seed
    ensemble uncertainty measure if a caller has already computed it --
    `devils_advocate_node` uses it to decide whether a single runner-up
    comparison is trustworthy enough, or whether the full all-experts
    ranking is worth computing this call. An RL policy's own accountability
    within `critique_node`'s Hedge weights is wired via `train_df` carrying
    an `RL_IMPLIED_SIGNAL_COLUMN` column instead (see
    `scaata.rl.policy_eval.attach_rl_implied_signal`), not a parameter here.
    """
    graph = build_graph()

    initial_state = {
        # Copy: nodes run untrusted/generated strategy code against this
        # DataFrame (scaata.strategies.pool.run_strategy_safely); without
        # this copy, any in-place mutation on the caller's side would leak
        # back out and corrupt whatever the caller does with `train_df`
        # after this call returns.
        "train_df": train_df.copy(),
        "feature_columns": feature_columns,
        "review_window": review_window,
        "max_iterations": max_iterations,
        "ensemble_disagreement": ensemble_disagreement,
        "iteration": 0,
        "strategy_pool_weights": None,
        "history": [],
    }

    # recursion_limit bounds total node visits; each loop iteration revisits
    # meta_selector + devils_advocate + critique (3 nodes, Phase 12 added
    # devils_advocate to the cycle), plus the one-shot upstream nodes
    # (scraper, normalizer, evolver, regime_classifier).
    final_state = graph.invoke(initial_state, config={"recursion_limit": 4 + max_iterations * 3 + 5})
    return final_state
