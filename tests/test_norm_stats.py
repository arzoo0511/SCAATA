"""Tests for persisting a policy's training normalization stats -- the
fix for live inference re-fitting z-scores on the live window instead of
using the scaling the policy was trained on."""
import numpy as np
import pandas as pd
import pytest

from scaata.features.normalize import apply_normalizer, fit_normalizer, load_norm_stats, save_norm_stats


def _frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(5, 2, n), "b": rng.normal(-1, 0.5, n)})


def test_round_trip_reproduces_the_same_scaling(tmp_path):
    df = _frame()
    mean, std = fit_normalizer(df, ["a", "b"])
    path = tmp_path / "stats.json"

    save_norm_stats(path, mean, std, {"quality": "exact"})
    loaded_mean, loaded_std, provenance = load_norm_stats(path, ["a", "b"])

    pd.testing.assert_frame_equal(
        apply_normalizer(df, ["a", "b"], mean, std),
        apply_normalizer(df, ["a", "b"], loaded_mean, loaded_std),
    )
    assert provenance == {"quality": "exact"}


def test_load_returns_columns_in_requested_order(tmp_path):
    mean, std = fit_normalizer(_frame(), ["a", "b"])
    path = tmp_path / "stats.json"
    save_norm_stats(path, mean, std)

    loaded_mean, _, _ = load_norm_stats(path, ["b", "a"])

    assert list(loaded_mean.index) == ["b", "a"]


def test_missing_file_returns_none(tmp_path):
    assert load_norm_stats(tmp_path / "nope.json", ["a"]) is None


def test_missing_column_raises_instead_of_silently_misscaling(tmp_path):
    mean, std = fit_normalizer(_frame(), ["a"])
    path = tmp_path / "stats.json"
    save_norm_stats(path, mean, std)

    with pytest.raises(ValueError, match="b"):
        load_norm_stats(path, ["a", "b"])
