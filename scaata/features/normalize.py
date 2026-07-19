"""Leakage-safe rolling z-score normalization, ported from the v1 notebook.

Mean/std are fit on the training slice only and applied unchanged to the
test slice, so no future information leaks into either split's features.
"""
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
