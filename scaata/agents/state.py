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

    # --- Phase 10: continuous novelty + Hedge multiplicative-weights
    # combiner (replaces the old binary "halve the single most-picked
    # culprit" rule in critique_node) ---
    novelty_score: float          # continuous distributional-shift score for train_df's most recent row, in [0, 1]
    expert_names: list[str]       # source labels for strategy_pool_weights, same order/length (incl. "sentiment_expert" if present)
    loss_history: list            # per-iteration per-expert loss vectors, for the running regret report
    weight_history: list          # per-iteration per-expert weight vectors (weights "in play" that round, before the round's update)
    cumulative_regret: dict       # scaata.strategies.hedge.hedge_regret_report(...) output, recomputed each iteration

    # --- Phase 12: devil's-advocate adversarial critique -- the actual
    # decision-reasoning trace ("what is its thinking, what alternative was
    # considered"), a real numeric counterfactual re-simulation, not
    # templated text ---
    devils_advocate_report: dict | None  # scaata.agents.nodes.devils_advocate_node.devils_advocate_report(...) output

    # --- Phase 16: closes the critique -> RL feedback loop (previously
    # one-way) and extends devil's-advocate. `ensemble_disagreement` is set
    # by an external caller that already ran Phase 14's 5-seed ensemble
    # (not computed inside the graph itself); when present and above
    # ENSEMBLE_DISAGREEMENT_DEEP_DIVE_THRESHOLD, devils_advocate_node
    # computes a full all-experts ranking instead of just the top-2. The RL
    # policy itself becomes an accountable Hedge expert in critique_node
    # when train_df carries an RL_IMPLIED_SIGNAL_COLUMN column (see
    # scaata.rl.policy_eval.attach_rl_implied_signal) -- no separate state
    # field needed for that since it's carried on train_df like
    # sentiment_score/novelty_score already are ---
    ensemble_disagreement: float | None
