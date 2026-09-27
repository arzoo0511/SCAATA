import numpy as np
import pandas as pd

from scaata.strategies.pool import build_strategy_pool_signals, run_strategy_safely

MUTATING_STRATEGY_CODE = """
def strategy(df):
    df['rsi'] = np.nan
    return [1] * len(df)
"""

WELL_BEHAVED_STRATEGY_CODE = """
def strategy(df):
    return (df['close'] > df['close'].mean()).astype(int).tolist()
"""


def _make_df():
    return pd.DataFrame({
        "close": np.linspace(100, 110, 20),
        "rsi": np.linspace(30, 70, 20),
    })


def test_run_strategy_safely_does_not_mutate_callers_dataframe():
    df = _make_df()
    original_rsi = df["rsi"].copy()

    signals = run_strategy_safely(MUTATING_STRATEGY_CODE, df)

    assert signals is not None
    assert df["rsi"].equals(original_rsi)
    assert not df["rsi"].isna().any()


def test_build_strategy_pool_signals_does_not_mutate_callers_dataframe():
    df = _make_df()
    original_rsi = df["rsi"].copy()
    strategies = [
        {"source": "mutating", "clean_code": MUTATING_STRATEGY_CODE},
        {"source": "well_behaved", "clean_code": WELL_BEHAVED_STRATEGY_CODE},
    ]

    build_strategy_pool_signals(strategies, df)

    assert df["rsi"].equals(original_rsi)


def test_run_strategy_safely_still_returns_correct_signals():
    df = _make_df()
    signals = run_strategy_safely(WELL_BEHAVED_STRATEGY_CODE, df)
    assert signals is not None
    assert len(signals) == len(df)
    assert set(np.unique(signals)) <= {0, 1}
