"""Strategy-pool robustness test: injects deliberately bad (sign-reversed)
strategies into the pool and checks whether the self-critique/meta-selector
feedback loop actually down-weights them relative to the good strategies —
a direct, quantitative test of the "self-critiquing" claim the whole
project is named after, rather than just trusting the mechanism exists.

Poisoning happens at the *signal* level (negating an already-computed
signal array), not by generating reversed Python source and re-executing
it — `scaata.strategies.pool.run_strategy_safely` executes code with split
globals/locals dicts (class-body semantics), so a generated wrapper
function referencing another name defined earlier in the same exec'd
source can't see it as a free variable. Negating signals directly sidesteps
that entirely and is simpler regardless.
"""
from __future__ import annotations

import numpy as np

from scaata.agents.graph import build_graph
from scaata.strategies.pool import build_strategy_pool_signals


def make_reversed_signal_entries(good_signal_entries: list[dict], n_bad: int) -> list[dict]:
    """Builds `n_bad` deliberately-bad {"source", "signals"} entries by
    negating a good strategy's signal array (buy becomes sell and vice
    versa) — guaranteed anti-correlated with whatever made the original
    profitable."""
    bad = []
    for i in range(n_bad):
        source_entry = good_signal_entries[i % len(good_signal_entries)]
        bad.append({
            "source": f"poisoned:{source_entry['source']}",
            "signals": -np.asarray(source_entry["signals"]),
        })
    return bad


def run_robustness_test(
    train_df,
    good_strategies: list[dict],
    n_bad_strategies: int = 2,
    max_iterations: int = 4,
    feature_columns: list[str] | None = None,
) -> dict:
    """Injects `n_bad_strategies` sign-reversed strategies alongside
    `good_strategies`, runs the LangGraph inner loop's Meta-Selector <->
    Critique cycle on the combined pool, and reports whether the final
    pool weights ended up lower, on average, for the poisoned strategies
    than for the good ones.
    """
    from scaata.config import FEATURE_COLUMNS

    feature_columns = feature_columns or FEATURE_COLUMNS

    good_signal_entries = build_strategy_pool_signals(good_strategies, train_df)
    bad_signal_entries = make_reversed_signal_entries(good_signal_entries, n_bad_strategies)
    combined_signals = good_signal_entries + bad_signal_entries

    good_indices = list(range(len(good_signal_entries)))
    bad_indices = list(range(len(good_signal_entries), len(combined_signals)))

    graph = build_graph()
    initial_state = {
        "train_df": train_df,
        "feature_columns": feature_columns,
        "review_window": 20,
        "max_iterations": max_iterations,
        "iteration": 0,
        "strategy_pool_weights": None,
        "history": [],
        # Pre-populated so meta_selector_node uses this exact combined pool
        # (with known good/bad indices) instead of recomputing its own from
        # `normalized_strategies`.
        "strategy_signals": combined_signals,
    }
    # 3 nodes per loop iteration (meta_selector, devils_advocate, critique
    # -- Phase 12 added devils_advocate to the cycle) plus one-shot upstream nodes.
    final_state = graph.invoke(initial_state, config={"recursion_limit": 4 + max_iterations * 3 + 5})

    weights = np.array(final_state["strategy_pool_weights"])
    good_weights = weights[good_indices] if good_indices else np.array([])
    bad_weights = weights[bad_indices] if bad_indices else np.array([])

    return {
        "final_weights": weights,
        "good_indices": good_indices,
        "bad_indices": bad_indices,
        "mean_good_weight": float(good_weights.mean()) if len(good_weights) else float("nan"),
        "mean_bad_weight": float(bad_weights.mean()) if len(bad_weights) else float("nan"),
        "down_weighted_as_expected": (
            bool(good_weights.mean() >= bad_weights.mean()) if len(good_weights) and len(bad_weights) else None
        ),
        "iterations_run": final_state["iteration"],
        "history": final_state["history"],
    }
