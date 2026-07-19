"""Per-regime performance report, including the explicit 2020-era vs
2026-era comparison the user asked for (also reused by Phase 3's 3rd Eye
agent for the same comparison against sentiment data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.evaluation.metrics import regime_metrics_table, sharpe_ratio
from scaata.evaluation.stats import block_bootstrap_ci


def per_regime_report(equity: np.ndarray, actions: np.ndarray, regime_labels: pd.Series) -> pd.DataFrame:
    return regime_metrics_table(equity, actions, regime_labels)


def era_comparison_report(
    equity_2020_era: np.ndarray,
    actions_2020_era: np.ndarray,
    regimes_2020_era: pd.Series,
    equity_2026_era: np.ndarray,
    actions_2026_era: np.ndarray,
    regimes_2026_era: pd.Series,
) -> dict:
    """Side-by-side per-regime tables for an early (2020-era) slice and a
    late (2026-era) slice of the backtest, each with bootstrap CIs on Sharpe.
    """
    table_2020 = per_regime_report(equity_2020_era, actions_2020_era, regimes_2020_era)
    table_2026 = per_regime_report(equity_2026_era, actions_2026_era, regimes_2026_era)

    ci_2020 = block_bootstrap_ci(equity_2020_era, sharpe_ratio)
    ci_2026 = block_bootstrap_ci(equity_2026_era, sharpe_ratio)

    return {
        "table_2020": table_2020,
        "table_2026": table_2026,
        "sharpe_ci_2020": ci_2020,
        "sharpe_ci_2026": ci_2026,
    }
