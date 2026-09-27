"""Critique Agent — evaluates the meta-selector-implied trajectory over the
trailing review window using the same drawdown/volatility/holding-time
penalty terms the RL reward uses (`scaata.rl.reward`), then reweights the
strategy pool via the Hedge multiplicative-weights combiner (Phase 10).
This node's continue/done decision is what makes the Critique ->
Meta-Selector feedback edge in `agents/graph.py` a real, executing cycle.

Phase 10 change, precisely: the original mechanism was a single binary rule
("if the trailing window scored net-negative, halve whichever strategy was
picked most often") applied only within the strategy pool. That's replaced
here with a Hedge/multiplicative-weights update (`scaata.strategies.hedge`)
over every pooled strategy — plus a sentiment "expert" when `train_df` has
a `sentiment_score` column (Phase 9's wiring) — with the update's learning
rate continuously modulated by a real, causal novelty score
(`scaata.regimes.novelty`) rather than a hardcoded regime if-branch. This
is a strictly more general mechanism: with one expert and zero novelty it
degenerates to a single-expert weight of 1.0, and the sentiment expert only
participates when sentiment data actually exists in `train_df`.

Phase 16: an RL policy expert, exactly the same way, when `train_df` has an
`rl_implied_signal` column (see `scaata.rl.policy_eval`). This is the
concrete fix for the previously one-way critique -> RL relationship — the
RL policy was trained once and frozen, never made accountable within the
same regret-tracked Hedge weight system every strategy-pool expert already
is. Once wired in, its Hedge weight/cumulative regret is a real, auditable
"how much should we trust the RL policy's suggestion right now relative to
the strategy pool" signal, usable to gate a live deployment decision.
"""
from __future__ import annotations

import numpy as np

from scaata.agents.state import AgentState
from scaata.config import (
    CONVERGENCE_EPSILON,
    FEATURE_COLUMNS,
    MIN_STRATEGY_WEIGHT,
    REVIEW_WINDOW_DAYS,
    RL_IMPLIED_SIGNAL_COLUMN,
)
from scaata.regimes.novelty import rolling_mahalanobis_novelty
from scaata.rl.reward import window_critique_report
from scaata.strategies.backtest_signal import simulate_trajectory_for_choice
from scaata.strategies.hedge import (
    HedgeWeights,
    eta_from_horizon,
    expert_losses_from_critique,
    floor_weights_on_simplex,
    hedge_regret_report,
    hedge_update_with_novelty,
)

SENTIMENT_EXPERT_SOURCE = "sentiment_expert"
SENTIMENT_EXPERT_THRESHOLD = 0.1  # |sentiment_score| below this maps to "hold", not a directional call
RL_POLICY_EXPERT_SOURCE = "rl_policy_expert"


def _sentiment_expert_entry(train_df) -> dict:
    scores = train_df["sentiment_score"].values
    signals = np.where(scores > SENTIMENT_EXPERT_THRESHOLD, 1, np.where(scores < -SENTIMENT_EXPERT_THRESHOLD, -1, 0))
    return {"source": SENTIMENT_EXPERT_SOURCE, "signals": signals.astype(int)}


def _rl_policy_expert_entry(train_df) -> dict:
    signals = train_df[RL_IMPLIED_SIGNAL_COLUMN].values
    return {"source": RL_POLICY_EXPERT_SOURCE, "signals": signals.astype(int)}


def _current_novelty_score(train_df, feature_columns: list[str]) -> float:
    """Mahalanobis-only novelty (cheap: no per-row classifier refitting) is
    enough for the live Hedge-modulation hot path; the discriminator
    cross-check (`scaata.regimes.novelty.rolling_discriminator_novelty`) is
    reserved for offline evaluation/plots rather than run on every
    LangGraph iteration."""
    available_columns = [c for c in feature_columns if c in train_df.columns]
    if not available_columns:
        return 0.0
    novelty_series = rolling_mahalanobis_novelty(train_df, available_columns)
    latest = novelty_series.iloc[-1]
    return float(latest) if np.isfinite(latest) else 0.0


def critique_node(state: AgentState) -> dict:
    train_df = state["train_df"]
    feature_columns = state.get("feature_columns", FEATURE_COLUMNS)
    window = state.get("review_window", REVIEW_WINDOW_DAYS)
    weight_matrix = state["strategy_weight_matrix"]
    strategy_signals = state["strategy_signals"]
    pool_weights_before = list(state["strategy_pool_weights"])

    has_sentiment_expert = "sentiment_score" in train_df.columns
    has_rl_policy_expert = RL_IMPLIED_SIGNAL_COLUMN in train_df.columns
    experts = (
        list(strategy_signals)
        + ([_sentiment_expert_entry(train_df)] if has_sentiment_expert else [])
        + ([_rl_policy_expert_entry(train_df)] if has_rl_policy_expert else [])
    )
    expert_names = [s["source"] for s in experts]

    # Extend pool_weights with a uniform starting weight for any newly-
    # appeared expert (sentiment and/or RL policy, each only happens once,
    # the first iteration that expert is present) rather than dropping it.
    if len(pool_weights_before) < len(experts):
        pool_weights_before = pool_weights_before + [1.0] * (len(experts) - len(pool_weights_before))

    # Top-weighted-path report: what the meta-selector's current pick
    # actually did over the trailing window (unchanged role from before
    # Phase 10 — used for the convergence/logging report, not the weight
    # update itself, which now comes from per-expert Hedge losses below).
    n = len(train_df)
    start = max(0, n - window)
    top_strategy_idx = weight_matrix[start:].argmax(axis=1)
    top_values, top_actions = simulate_trajectory_for_choice(train_df, strategy_signals, top_strategy_idx, window)
    report = window_critique_report(top_values, top_actions, window)

    # Phase 10: Hedge multiplicative-weights update, novelty-modulated.
    losses = expert_losses_from_critique(train_df, experts, window)
    novelty_score = state.get("novelty_score")
    if novelty_score is None:
        novelty_score = _current_novelty_score(train_df, feature_columns)

    eta = eta_from_horizon(len(experts), window)
    hedge = HedgeWeights(n_experts=len(experts), eta=eta, weights=np.array(pool_weights_before))
    sentiment_expert_idx = expert_names.index(SENTIMENT_EXPERT_SOURCE) if has_sentiment_expert else None
    updated_weights = hedge_update_with_novelty(
        hedge, losses, novelty_score, sentiment_expert_idx=sentiment_expert_idx
    )

    pool_weights = floor_weights_on_simplex(updated_weights, MIN_STRATEGY_WEIGHT).tolist()

    loss_history = list(state.get("loss_history", [])) + [losses.tolist()]
    weight_history = list(state.get("weight_history", [])) + [pool_weights_before]
    cumulative_regret = hedge_regret_report(np.array(loss_history), np.array(weight_history))

    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", 4)

    history = list(state.get("history", []))
    prev_weights = history[-1]["strategy_pool_weights"] if history else None
    weight_delta = (
        max(abs(a - b) for a, b in zip(pool_weights, prev_weights)) if prev_weights else float("inf")
    )
    converged = weight_delta < CONVERGENCE_EPSILON

    history.append({
        "iteration": iteration,
        "strategy_pool_weights": list(pool_weights),
        "critique_report": report,
        "novelty_score": novelty_score,
        "expert_names": expert_names,
        "loss_per_expert": losses.tolist(),
        # Phase 12: the decision-reasoning trace -- what alternative was
        # considered and why the chosen path did or didn't beat it, purely
        # from re-simulated numeric evidence. Read-only here (does not
        # affect this iteration's weight update); see devils_advocate_node.
        "devils_advocate_report": state.get("devils_advocate_report"),
    })

    return {
        "strategy_pool_weights": pool_weights,
        "critique_report": report,
        "iteration": iteration,
        "converged": converged or iteration >= max_iterations,
        "history": history,
        "novelty_score": novelty_score,
        "expert_names": expert_names,
        "loss_history": loss_history,
        "weight_history": weight_history,
        "cumulative_regret": cumulative_regret,
    }
