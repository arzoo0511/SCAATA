"""Unit tests for scaata.strategies.meta_selector -- there was no dedicated
test file for this module before now, which is part of why the
classification target's collapse to a constant, zero-variance prediction
(see overnight_findings.md section 13) went undetected since Phase 2.
These test mechanics/shapes/no-crash on synthetic data; the real-data
collapse finding itself is an empirical result, not something to hardcode
as a synthetic-data assertion here.
"""
import numpy as np
import pytest

from scaata.strategies.meta_selector import (
    StrategySelector,
    build_meta_dataset,
    build_meta_regression_dataset,
    predict_expected_profits,
    predict_strategy_weights,
    train_meta_selector,
    train_meta_selector_regression,
)


def _make_synthetic_data(n=200, num_strategies=4, input_dim=5, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.normal(0, 1, (n, input_dim))
    future_returns = rng.normal(0, 0.01, n)
    strategy_signals = [rng.choice([-1, 0, 1], n) for _ in range(num_strategies)]
    return states, strategy_signals, future_returns


def test_build_meta_dataset_label_matches_highest_profit_strategy():
    states, signals, returns = _make_synthetic_data(n=50, num_strategies=3)
    meta_X, meta_y = build_meta_dataset(states, signals, returns)

    assert meta_X.shape == (49, 5)
    assert meta_y.shape == (49,)
    # Directly recompute the expected label for one arbitrary step and confirm.
    t = 10
    profits = [signals[i][t] * returns[t] for i in range(3)]
    assert meta_y[t] == int(np.argmax(profits))


def test_build_meta_regression_dataset_preserves_full_profit_vector():
    states, signals, returns = _make_synthetic_data(n=50, num_strategies=3)
    meta_X, meta_Y = build_meta_regression_dataset(states, signals, returns)

    assert meta_Y.shape == (49, 3)
    t = 10
    expected = np.array([signals[i][t] * returns[t] for i in range(3)])
    np.testing.assert_allclose(meta_Y[t], expected)


def test_build_meta_regression_dataset_matches_classification_argmax():
    """Both datasets are built from the same underlying profit computation
    -- the regression target's per-row argmax must equal the classification
    target's label, confirming they encode the same information at
    different levels of granularity."""
    states, signals, returns = _make_synthetic_data(n=80, num_strategies=5, seed=1)
    _, meta_y = build_meta_dataset(states, signals, returns)
    _, meta_Y = build_meta_regression_dataset(states, signals, returns)

    np.testing.assert_array_equal(meta_y, meta_Y.argmax(axis=1))


def test_train_meta_selector_val_split_returns_metrics():
    states, signals, returns = _make_synthetic_data(n=200)
    model, num_strategies, metrics = train_meta_selector(
        states, signals, returns, epochs=2, val_frac=0.2, seed=0,
    )
    assert isinstance(model, StrategySelector)
    assert num_strategies == 4
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert 0.0 <= metrics["val_accuracy"] <= 1.0
    assert metrics["n_train"] + metrics["n_val"] == 199  # len(states) - 1


def test_train_meta_selector_default_val_frac_returns_2_tuple_unchanged():
    """Backward-compatibility guard: every existing caller unpacks a
    2-tuple (model, num_strategies); val_frac=0.0 must keep doing exactly
    that, not the 3-tuple."""
    states, signals, returns = _make_synthetic_data(n=100)
    result = train_meta_selector(states, signals, returns, epochs=2, seed=0)
    assert len(result) == 2


def test_train_meta_selector_class_balanced_loss_runs_without_crashing():
    states, signals, returns = _make_synthetic_data(n=200)
    model, num_strategies, metrics = train_meta_selector(
        states, signals, returns, epochs=2, val_frac=0.2, seed=0, class_balanced_loss=True,
    )
    assert isinstance(model, StrategySelector)
    assert 0.0 <= metrics["val_accuracy"] <= 1.0


def test_strategy_selector_hidden_dims_configurable():
    model_shallow = StrategySelector(input_dim=5, num_strategies=3, hidden_dims=[128])
    model_deep = StrategySelector(input_dim=5, num_strategies=3, hidden_dims=[256, 128, 64])
    # Deep model must have strictly more parameters than the default-depth one.
    n_params_shallow = sum(p.numel() for p in model_shallow.parameters())
    n_params_deep = sum(p.numel() for p in model_deep.parameters())
    assert n_params_deep > n_params_shallow


def test_predict_strategy_weights_rows_sum_to_one():
    states, signals, returns = _make_synthetic_data(n=100)
    model, _ = train_meta_selector(states, signals, returns, epochs=2, seed=0)
    weights = predict_strategy_weights(model, states)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)


def test_train_meta_selector_regression_val_split_returns_metrics():
    states, signals, returns = _make_synthetic_data(n=200)
    model, num_strategies, metrics = train_meta_selector_regression(
        states, signals, returns, epochs=2, val_frac=0.2, seed=0,
    )
    assert isinstance(model, StrategySelector)
    assert num_strategies == 4
    assert metrics["train_mse"] >= 0.0
    assert metrics["val_mse"] >= 0.0
    assert 0.0 <= metrics["train_rank_accuracy"] <= 1.0
    assert 0.0 <= metrics["val_rank_accuracy"] <= 1.0


def test_train_meta_selector_regression_default_val_frac_returns_2_tuple():
    states, signals, returns = _make_synthetic_data(n=100)
    result = train_meta_selector_regression(states, signals, returns, epochs=2, seed=0)
    assert len(result) == 2


def test_predict_expected_profits_shape_and_no_softmax_applied():
    """Regression outputs are raw profit estimates, not probabilities --
    unlike predict_strategy_weights, rows must NOT be constrained to sum to
    1 (that would silently reintroduce a softmax on a regression head)."""
    states, signals, returns = _make_synthetic_data(n=100)
    model, _ = train_meta_selector_regression(states, signals, returns, epochs=2, seed=0)
    profits = predict_expected_profits(model, states)
    assert profits.shape == (100, 4)
    row_sums = profits.sum(axis=1)
    assert not np.allclose(row_sums, 1.0)
