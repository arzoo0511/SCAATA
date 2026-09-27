"""Leakage-safe rolling z-score normalization, ported from the v1 notebook.

Mean/std are fit on the training slice only and applied unchanged to the
test slice, so no future information leaks into either split's features.

`save_norm_stats`/`load_norm_stats` persist a policy's training mean/std
next to the policy itself, so live inference applies exactly the scaling
the policy was trained on. Re-fitting on the live window instead (what the
live path used to do) was measured to shift MSFT's `ma_50` z-score from
+0.81 to -1.06 on the same day -- the policy was seeing inputs from a
different distribution than anything it trained on.
"""
import json
from pathlib import Path

import pandas as pd


def fit_normalizer(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.Series, pd.Series]:
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std() + 1e-8
    return mean, std


def apply_normalizer(
    df: pd.DataFrame, feature_cols: list[str], mean: pd.Series, std: pd.Series
) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = (out[feature_cols] - mean) / std
    return out


def normalize_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mean, std = fit_normalizer(train_df, feature_cols)
    train_norm = apply_normalizer(train_df, feature_cols, mean, std)
    test_norm = apply_normalizer(test_df, feature_cols, mean, std)
    return train_norm, test_norm, mean, std


def save_norm_stats(path, mean: pd.Series, std: pd.Series, provenance: dict | None = None) -> None:
    """Writes training mean/std (as returned by `fit_normalizer`, epsilon
    already included in `std`) plus a free-form `provenance` dict saying
    how these stats were obtained, so a reconstructed file can never be
    mistaken for one saved at training time."""
    payload = {
        "feature_columns": list(mean.index),
        "mean": {k: float(v) for k, v in mean.items()},
        "std": {k: float(v) for k, v in std.items()},
        "provenance": provenance or {},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_norm_stats(path, feature_cols: list[str]) -> tuple[pd.Series, pd.Series, dict] | None:
    """Returns `(mean, std, provenance)` ordered as `feature_cols`, or None
    if no stats file exists. Raises if the file lacks any requested column
    -- silently scaling a column with the wrong stats is the bug this
    exists to prevent."""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [c for c in feature_cols if c not in payload["mean"] or c not in payload["std"]]
    if missing:
        raise ValueError(f"{path.name} has no normalization stats for column(s) {missing}")
    mean = pd.Series({c: payload["mean"][c] for c in feature_cols})
    std = pd.Series({c: payload["std"][c] for c in feature_cols})
    return mean, std, payload.get("provenance", {})
