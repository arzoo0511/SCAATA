"""Non-degeneracy check for the curated strategy library (Phase 9b): every
strategy must actually produce more than one distinct signal value on a
representative synthetic OHLCV sample — the same bar
`scaata.strategies.pool.build_strategy_pool_signals` already applies before
trusting a strategy into the pool — and must pass the same execute-to-
validate check `scaata.strategies.normalizer` applies to mock strategies.
"""
import numpy as np
import pandas as pd

from scaata.strategies.library import library_strategies
from scaata.strategies.normalizer import validate_mock_strategies
from scaata.strategies.pool import run_strategy_safely


def _make_synthetic_ohlcv(n=200, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


def test_library_strategies_are_valid_and_non_degenerate():
    df = _make_synthetic_ohlcv()
    strategies = library_strategies()
    assert len(strategies) >= 5

    for strat in strategies:
        signals = run_strategy_safely(strat["code"], df)
        assert signals is not None, f"{strat['source']} failed to execute"
        assert len(signals) == len(df)
        assert len(np.unique(signals)) > 1, f"{strat['source']} produced a degenerate (constant) signal"


def test_library_strategies_pass_execute_to_validate_check():
    strategies = library_strategies()
    validated = validate_mock_strategies(strategies)
    assert len(validated) == len(strategies)
    for v in validated:
        assert "clean_code" in v
