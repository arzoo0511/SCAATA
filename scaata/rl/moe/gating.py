"""Regime gate for Phase 4's mixture-of-experts: a lightweight classifier
predicting a 2-bucket regime {calm, stress} from technical features alone,
fit once on the training split and applied to unseen data purely via its
own `.predict()` calls — never given the true regime label at evaluation
time. This is the "live, non-oracle" gating decision the rebuild plan
requires: using ground-truth regime labels to pick which expert acts at
eval time would make the MoE's numbers look artificially good relative to
the single-policy Phase 1/2 baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scaata.regimes.detector import threshold_regime_labels

STRESS_REGIMES = {"volatile_bull", "bear", "crisis"}
CALM_REGIME = "calm_bull"


def bucket_regime_labels(regime_series: pd.Series) -> pd.Series:
    """Maps the 4-way threshold regime label down to the 2 buckets Phase 4
    trains specialist experts for."""
    return regime_series.apply(lambda r: "stress" if r in STRESS_REGIMES else "calm")


class RegimeGate:
    """Fit once on train_df; at inference, call `predict`/`predict_one` on
    already-available feature rows only — never re-fit on test data, and
    never given test_df's true regime label."""

    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)
        self.fitted = False

    def fit(self, train_df: pd.DataFrame, feature_columns: list[str]) -> "RegimeGate":
        labeled = threshold_regime_labels(train_df)
        buckets = bucket_regime_labels(labeled["regime"])
        X = train_df[feature_columns].values
        self.clf.fit(X, buckets)
        self.fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RegimeGate must be fit before predict()")
        return self.clf.predict(features)

    def predict_one(self, feature_row: np.ndarray) -> str:
        return self.predict(feature_row.reshape(1, -1))[0]


def gate_accuracy_report(gate: RegimeGate, eval_df: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Diagnostic only (not used by the MoE itself): how often the gate's
    live prediction matches the causal threshold-detector's own label on
    held-out data. Useful for sanity-checking the gate, never for routing.
    """
    true_labels = bucket_regime_labels(threshold_regime_labels(eval_df)["regime"])
    predicted = gate.predict(eval_df[feature_columns].values)
    accuracy = float(np.mean(predicted == true_labels.values))
    return {"accuracy": accuracy, "n": len(true_labels), "true_stress_frac": float((true_labels == "stress").mean())}
