"""Regime-Classifier Agent — a thin wrapper that reuses Phase 1's causal
regime detector. Deliberately does not recompute the (expensive, train-only)
HMM cross-check here; that lives in Phase 1's notebook/tests. This node
just attaches a `regime` label per row of `train_df` so the Critique node
can condition on it.
"""
from __future__ import annotations

from scaata.agents.state import AgentState
from scaata.regimes.detector import threshold_regime_labels


def regime_classifier_node(state: AgentState) -> dict:
    labeled = threshold_regime_labels(state["train_df"])
    return {"regime_labels": labeled["regime"]}
