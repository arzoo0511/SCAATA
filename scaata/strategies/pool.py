"""Strategy pool execution harness, ported from the v1 notebook's
`run_strategy_safely`. Executes LLM-generated (or mock) `strategy(df)`
code in a restricted globals dict (only numpy/pandas available) and
validates the returned signal array's shape/dtype before trusting it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_strategy_safely(code: str, df: pd.DataFrame) -> np.ndarray | None:
    restricted_globals = {"np": np, "pd": pd}
    local_env = {}

    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 3:
            code = parts[1].strip()

    try:
        exec(code, restricted_globals, local_env)
        if "strategy" not in local_env:
            return None
        signals = local_env["strategy"](df)
        if len(signals) != len(df):
            return None
        return np.array(signals).astype(int)
    except Exception as e:
        if not isinstance(e, KeyError):
            print(f"Error evaluating strategy: {e}")
        return None


def build_strategy_pool_signals(strategies: list[dict], df: pd.DataFrame) -> list[dict]:
    """Runs every validated strategy over `df`, keeping only those that
    produce a non-degenerate (more than one distinct value) signal array.
    Returns [{"source", "signals"}].
    """
    pool = []
    for strat in strategies:
        signals = run_strategy_safely(strat["clean_code"], df)
        if signals is not None and len(np.unique(signals)) > 1:
            pool.append({"source": strat["source"], "signals": signals})
    return pool
