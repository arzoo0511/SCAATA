"""Tests for genetic-programming strategy evolution (Phase 11): compiled
trees must satisfy the existing `strategy(df)` contract unchanged,
absolute-scale features (`ma_10`/`ma_50`) must never appear in a
fixed-threshold condition, fitness must reward correct directional calls,
the cross-ticker/val-split fitness gap must actually flag an overfit
candidate (not just claim to), and the full evolution loop must be able to
discover a genuinely generalizable planted signal.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import FEATURE_COLUMNS
from scaata.strategies.evolve import (
    PairCondition,
    RuleNode,
    RuleTree,
    ThresholdCondition,
    compile_rule_tree,
    crossover,
    evaluate_fitness,
    evaluate_fitness_across_tickers,
    mutate,
    random_condition,
    random_rule_tree,
    run_evolution,
)
from scaata.strategies.pool import run_strategy_safely


def _make_synthetic_df(n=150, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["rsi"] = rng.uniform(0, 100, n)  # keep rsi in its real bounded range
    df["ma_10"] = 100 + rng.normal(0, 2, n)
    df["ma_50"] = 100 + rng.normal(0, 2, n)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    df["Close"] = close
    return df


def test_compile_rule_tree_produces_valid_executable_strategy():
    tree = RuleTree(
        nodes=[
            RuleNode(ThresholdCondition("rsi", "<", 30.0), 1),
            RuleNode(ThresholdCondition("rsi", ">", 70.0), -1),
        ],
        else_action=0,
    )
    code = compile_rule_tree(tree)
    df = _make_synthetic_df()

    signals = run_strategy_safely(code, df)

    assert signals is not None
    assert len(signals) == len(df)
    assert set(np.unique(signals)).issubset({-1, 0, 1})
    # rsi is uniform(0,100) over 150 rows, so both the <30 and >70 bands should fire.
    assert len(np.unique(signals)) > 1


def test_pair_condition_compiles_and_executes():
    tree = RuleTree(nodes=[RuleNode(PairCondition("ma_10", "ma_50", ">"), 1)], else_action=-1)
    code = compile_rule_tree(tree)
    df = _make_synthetic_df()

    signals = run_strategy_safely(code, df)

    assert signals is not None
    assert len(np.unique(signals)) > 1


def test_random_condition_never_puts_price_scale_features_on_a_fixed_threshold():
    rng = np.random.default_rng(0)
    for _ in range(500):
        cond = random_condition(rng, FEATURE_COLUMNS)
        if isinstance(cond, ThresholdCondition):
            assert cond.feature not in ("ma_10", "ma_50"), (
                "a price-scale feature must never be compared to a fixed numeric threshold "
                "-- it isn't comparable across tickers with different price levels"
            )


def test_mutate_and_crossover_respect_the_depth_cap():
    rng = np.random.default_rng(0)
    max_depth = 3
    tree = random_rule_tree(rng, FEATURE_COLUMNS, max_depth=max_depth)

    for _ in range(50):
        tree = mutate(tree, rng, FEATURE_COLUMNS, max_depth=max_depth)
        assert 1 <= len(tree.nodes) <= max_depth

    tree_b = random_rule_tree(rng, FEATURE_COLUMNS, max_depth=max_depth)
    for _ in range(50):
        child = crossover(tree, tree_b, rng, max_depth=max_depth)
        assert 1 <= len(child.nodes) <= max_depth


def test_evaluate_fitness_rewards_correct_directional_calls():
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = np.full(n, 0.01)  # steady uptrend
    close = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({"Close": close}, index=dates)

    correct_signals = np.ones(n, dtype=int)   # always buy into a steady uptrend
    wrong_signals = -np.ones(n, dtype=int)    # always sell into a steady uptrend

    correct_fitness = evaluate_fitness(correct_signals, df)["fitness"]
    wrong_fitness = evaluate_fitness(wrong_signals, df)["fitness"]

    assert correct_fitness > wrong_fitness


def test_fitness_gap_flags_a_regime_flip_overfit_candidate():
    """Construct a df where momentum > 0 predicts positive returns in the
    in-sample half but negative returns in the held-out half (a regime
    flip) -- a strategy tuned to "momentum > 0 -> buy" should look great
    in-sample and fail out-of-sample, which is exactly what `fitness_gap`
    is supposed to catch.

    A signal decided at day t (from momentum[t]) is only realized in the
    simulation as the move from price[t] to price[t+1] (you buy at today's
    close, profit is tomorrow's move) -- so the planted relationship must
    correlate momentum[t] with `returns[t+1]`, one day ahead, not the same
    day's return, or the "signal" would just be describing a move that
    already happened by the time it's acted on.
    """
    n = 200
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    momentum = rng.uniform(-0.05, 0.05, n)
    split = n // 2

    returns = np.zeros(n)
    t = np.arange(1, n)
    sign = np.where(t <= split, 1.0, -1.0)  # in-sample: momentum[t-1] correctly predicts returns[t]; val: reversed
    returns[1:] = sign * np.sign(momentum[:-1]) * 0.01

    close = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["momentum"] = momentum
    df["Close"] = close

    tree = RuleTree(nodes=[RuleNode(ThresholdCondition("momentum", ">", 0.0), 1)], else_action=-1)
    code = compile_rule_tree(tree)

    report = evaluate_fitness_across_tickers(code, {"REGIME_FLIP": df}, val_frac=0.5)

    assert report["in_sample_fitness"] > 0
    assert report["val_fitness"] < report["in_sample_fitness"]
    assert report["fitness_gap"] > 0


def test_run_evolution_discovers_a_genuinely_generalizable_signal():
    """A planted signal that holds consistently across BOTH the in-sample
    and held-out slices of two different tickers -- if evolution can't
    consistently beat a random baseline here, something in the search loop
    itself (not just "GP is hard") is broken.
    """
    def _make_consistent_signal_df(seed):
        n = 150
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        momentum = rng.uniform(-0.05, 0.05, n)
        returns = np.sign(momentum) * 0.015  # momentum > 0 always precedes a positive move, in both halves
        close = 100 * np.cumprod(1 + returns)
        df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
        df["momentum"] = momentum
        df["Close"] = close
        return df

    ticker_dfs = {"A": _make_consistent_signal_df(1), "B": _make_consistent_signal_df(2)}

    results = run_evolution(
        ticker_dfs, population_size=12, n_generations=6, elite_frac=0.25,
        mutation_rate=0.3, random_immigrant_frac=0.15, val_frac=0.3, max_depth=2, seed=0,
    )

    assert len(results) == 12
    best = results[0]
    assert best["val_fitness"] > 0, "evolution failed to find a strategy that generalizes to held-out data at all"
    # results should be sorted by held-out (not in-sample) fitness, descending
    assert all(results[i]["val_fitness"] >= results[i + 1]["val_fitness"] for i in range(len(results) - 1))
