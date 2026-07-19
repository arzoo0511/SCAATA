import pandas as pd

from scaata.evaluation.walkforward import assert_no_overlap, generate_folds, split_fold


def test_folds_have_no_train_test_overlap():
    folds = generate_folds("2020-01-01", "2026-07-19", train_years=2.0, test_months=6, step_months=6)
    assert len(folds) > 0
    assert_no_overlap(folds)


def test_folds_progress_forward_in_time():
    folds = generate_folds("2020-01-01", "2026-07-19", train_years=2.0, test_months=6, step_months=6)
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.train_start > prev.train_start


def test_split_fold_respects_boundaries():
    dates = pd.date_range("2020-01-01", "2022-01-01", freq="B")
    df = pd.DataFrame({"Close": range(len(dates)), "Ticker": "TEST"}, index=dates)
    folds = generate_folds("2020-01-01", "2022-01-01", train_years=1.0, test_months=6, step_months=6)
    assert len(folds) > 0
    fold = folds[0]
    train, test = split_fold(df, fold)
    assert train.index.max() <= fold.train_end
    assert test.index.min() >= fold.test_start
    assert train.index.max() < test.index.min()
