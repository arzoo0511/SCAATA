"""Phase 4 mixture-of-experts tests: regime segments must be genuinely
contiguous (no artificial jumps between unrelated days), and the gate must
be usable at evaluation time from features alone — never given the true
regime label, which would be an oracle/hindsight shortcut per the rebuild
plan's fairness requirement.
"""
import numpy as np
import pandas as pd

from scaata.config import FEATURE_COLUMNS
from scaata.rl.moe.gating import RegimeGate, bucket_regime_labels
from scaata.rl.moe.moe_policy import MoEPolicy, backtest_moe
from scaata.rl.moe.train_experts import build_regime_segments, train_moe_experts


def _make_synthetic_df(n=300, seed=0, ticker="TEST"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.normal(0.0005, 0.01, n)
    returns[100:130] = rng.normal(-0.03, 0.04, 30)  # a stress segment
    close = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["returns"] = pd.Series(close, index=dates).pct_change().fillna(0)
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(30, min_periods=1).mean()
    df["Ticker"] = ticker
    return df


def test_regime_segments_are_contiguous_slices():
    df = _make_synthetic_df()
    for bucket in ["calm", "stress"]:
        segments = build_regime_segments(df, bucket, min_length=5)
        for seg in segments:
            assert seg.index.is_monotonic_increasing
            # a genuinely contiguous business-day slice has no unexpected
            # multi-week gaps (weekends/holidays aside)
            gaps = seg.index.to_series().diff().dropna()
            assert (gaps <= pd.Timedelta(days=4)).all()


def test_gate_predicts_from_features_only_no_label_access():
    df = _make_synthetic_df()
    train_df, test_df = df.iloc[:200], df.iloc[200:]

    gate = RegimeGate().fit(train_df, FEATURE_COLUMNS)
    # predict() only accepts a raw feature array -- structurally cannot
    # receive the true regime label, since that column is never passed in.
    features_only = test_df[FEATURE_COLUMNS].values
    predictions = gate.predict(features_only)

    assert len(predictions) == len(test_df)
    assert set(predictions) <= {"calm", "stress"}


def test_moe_backtest_runs_end_to_end_with_live_gating():
    df = _make_synthetic_df()
    train_df, test_df = df.iloc[:200], df.iloc[200:]

    result = train_moe_experts(train_df, FEATURE_COLUMNS, seed=0, total_timesteps=500)
    gate = RegimeGate().fit(train_df, FEATURE_COLUMNS)
    moe = MoEPolicy(result["experts"], gate, n_gate_features=len(FEATURE_COLUMNS))

    equity, actions, buckets_used = backtest_moe(moe, test_df, FEATURE_COLUMNS, "TEST")

    assert len(equity) == len(test_df)
    assert len(buckets_used) == len(test_df) - 1
    assert set(buckets_used) <= {"calm", "stress"}
