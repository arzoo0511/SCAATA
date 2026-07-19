"""Walk-forward (rolling-origin) backtest splitting.

Replaces the v1 notebook's single 80/20 split, which is one sample of
market luck, with many independent train/test folds — each fold's test
window immediately follows its train window in time, so no fold ever
trains on data from after its own test period.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from dateutil.relativedelta import relativedelta

from scaata.config import (
    WALKFORWARD_STEP_MONTHS,
    WALKFORWARD_TEST_MONTHS,
    WALKFORWARD_TRAIN_YEARS,
)


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(
    start: str,
    end: str,
    train_years: float = WALKFORWARD_TRAIN_YEARS,
    test_months: int = WALKFORWARD_TEST_MONTHS,
    step_months: int = WALKFORWARD_STEP_MONTHS,
) -> list[Fold]:
    """Generate rolling-origin folds. Each fold's train window is
    `train_years` long, immediately followed by a `test_months`-long test
    window; the origin advances by `step_months` each fold.
    """
    overall_start = pd.Timestamp(start)
    overall_end = pd.Timestamp(end)

    folds = []
    fold_id = 0
    train_start = overall_start
    while True:
        train_end = train_start + relativedelta(years=train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - pd.Timedelta(days=1)

        if test_end > overall_end:
            break

        folds.append(Fold(fold_id, train_start, train_end, test_start, test_end))
        fold_id += 1
        train_start = train_start + relativedelta(months=step_months)

    return folds


def split_fold(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice a (Date, Ticker)-indexed DataFrame into this fold's train/test."""
    idx = df.index
    train = df[(idx >= fold.train_start) & (idx <= fold.train_end)]
    test = df[(idx >= fold.test_start) & (idx <= fold.test_end)]
    return train, test


def assert_no_overlap(folds: list[Fold]) -> None:
    """Sanity check: no fold's test window overlaps its own train window,
    and train never extends past its own test start."""
    for f in folds:
        assert f.train_end < f.test_start, f"fold {f.fold_id}: train/test overlap"
        assert f.test_start <= f.test_end, f"fold {f.fold_id}: empty test window"
