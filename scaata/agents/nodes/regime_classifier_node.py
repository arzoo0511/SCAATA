"""Regime-Classifier Agent — a thin wrapper that reuses Phase 1's causal
regime detector. Deliberately does not recompute the (expensive, train-only)
HMM cross-check here; that lives in Phase 1's notebook/tests. This node
attaches a `regime` label per row of `train_df` (the discrete buckets, kept
for backward compatibility/reporting), and also computes the Phase 10
continuous novelty score here rather than leaving it as critique_node's own
ad hoc, inline computation -- this is the node that's actually about regime/
novelty detection, so it should be the one producing the real signal, not
just the decorative discrete label. critique_node already prefers
`state["novelty_score"]` when present (falling back to computing it itself
only if this node is skipped/state is missing it), so this is a pure
promotion of where the computation lives, not a behavior change for
critique_node's own logic.
"""
from __future__ import annotations

import numpy as np

from scaata.agents.state import AgentState
from scaata.config import FEATURE_COLUMNS
from scaata.regimes.detector import threshold_regime_labels
from scaata.regimes.novelty import rolling_mahalanobis_novelty


def regime_classifier_node(state: AgentState) -> dict:
    train_df = state["train_df"]
    feature_columns = state.get("feature_columns", FEATURE_COLUMNS)

    labeled = threshold_regime_labels(train_df)

    available_columns = [c for c in feature_columns if c in train_df.columns]
    novelty_score = 0.0
    if available_columns:
        novelty_series = rolling_mahalanobis_novelty(train_df, available_columns)
        latest = novelty_series.iloc[-1]
        novelty_score = float(latest) if np.isfinite(latest) else 0.0

    return {"regime_labels": labeled["regime"], "novelty_score": novelty_score}
