"""Tests for the fold-clustered directional comparison that replaces the
(ticker, fold)-pair Wilcoxon for walk-forward results."""
import numpy as np
import pandas as pd

from scaata.evaluation.stats import fold_clustered_comparison


def _grid(folds=9, tickers=9, seeds=5, a_fn=None, b_fn=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for fold in range(folds):
        market = rng.normal(0, 1.5)  # shared by every ticker in the fold
        for t in range(tickers):
            for s in range(seeds):
                rows.append({
                    "seed": s, "fold_id": fold, "ticker": f"T{t}",
                    "a": a_fn(market, rng) if a_fn else market + rng.normal(0, 0.5),
                    "b": b_fn(market, rng) if b_fn else market + rng.normal(0, 0.5),
                })
    return pd.DataFrame(rows)


def test_consistent_edge_is_detected_in_the_right_direction():
    df = _grid(a_fn=lambda m, r: m + 0.8 + r.normal(0, 0.3), b_fn=lambda m, r: m + r.normal(0, 0.3))

    result = fold_clustered_comparison(df, "a", "b")

    assert result["mean_diff"] > 0.5
    assert result["ci_low"] > 0
    assert result["share_folds_a_better"] == 1.0
    assert result["p_one_sided_a_better"] < 0.01  # 1/512 is the smallest possible with 9 folds


def test_direction_matters_the_loser_does_not_get_a_small_p():
    df = _grid(a_fn=lambda m, r: m - 0.8 + r.normal(0, 0.3), b_fn=lambda m, r: m + r.normal(0, 0.3))

    result = fold_clustered_comparison(df, "a", "b")

    assert result["mean_diff"] < 0
    assert result["p_one_sided_a_better"] > 0.99


def test_unit_is_the_fold_not_the_row():
    df = _grid()
    result = fold_clustered_comparison(df, "a", "b")
    assert result["n_folds"] == 9
    assert result["n_rows"] == 9 * 9 * 5
    assert len(result["fold_diffs"]) == 9


def test_never_traded_counts_as_cash_not_dropped():
    df = pd.DataFrame({
        "fold_id": [0, 0, 1, 1],
        "a": [np.nan, 1.0, np.nan, 1.0],
        "b": [-1.0, -1.0, -1.0, -1.0],
    })

    result = fold_clustered_comparison(df, "a", "b")

    assert result["collapsed_a"] == 2
    assert result["fold_diffs"] == [1.5, 1.5]  # (0 - -1 + 1 - -1) / 2 per fold
