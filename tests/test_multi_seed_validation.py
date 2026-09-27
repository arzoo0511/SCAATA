"""Tests for scaata.evaluation.multi_seed_validation -- fast, mocked
(no real training): confirms the seed-averaging/spread math and the
noise-floor comparison logic are correct, which is what actually matters
here (the training itself is already covered by test_train_ppo_dsr.py).
"""
import numpy as np
import pandas as pd
import pytest

import scaata.evaluation.multi_seed_validation as msv_module
from scaata.evaluation.multi_seed_validation import (
    _train_and_score_one_seed,
    compare_configs_multi_seed,
    multi_seed_sharpe,
)


class _FakeModel:
    def __init__(self, seed):
        self.seed = seed


def _patch_simple(monkeypatch, sharpe_by_seed: dict):
    """Simpler patch: backtest_ppo returns the model's seed as a sentinel,
    sharpe_ratio maps sentinel -> the configured sharpe for that seed."""
    def _fake_train_ppo(train_df, feature_columns, seed=0, total_timesteps=100_000, **kwargs):
        return _FakeModel(seed)

    def _fake_backtest_ppo(model, df, feature_columns, ticker):
        return model.seed, None  # pass the seed through as the "equity curve"

    def _fake_sharpe_ratio(equity_curve_sentinel):
        return sharpe_by_seed[equity_curve_sentinel]

    monkeypatch.setattr(msv_module, "train_ppo", _fake_train_ppo)
    monkeypatch.setattr(msv_module, "backtest_ppo", _fake_backtest_ppo)
    monkeypatch.setattr(msv_module, "sharpe_ratio", _fake_sharpe_ratio)


def _dummy_df():
    return pd.DataFrame({"Close": [100.0, 101.0], "Ticker": ["TEST", "TEST"]})


def test_multi_seed_sharpe_reports_mean_and_spread(monkeypatch):
    _patch_simple(monkeypatch, sharpe_by_seed={0: 1.0, 1: 2.0, 2: 3.0})

    result = multi_seed_sharpe(_dummy_df(), _dummy_df(), ["f1"], "TEST", seeds=[0, 1, 2], max_workers=1)

    assert result["sharpes"] == [1.0, 2.0, 3.0]
    assert result["mean_sharpe"] == pytest.approx(2.0)
    assert result["std_sharpe"] == pytest.approx(np.std([1.0, 2.0, 3.0]))
    assert result["min_sharpe"] == 1.0
    assert result["max_sharpe"] == 3.0
    assert result["n_collapsed"] == 0


def test_multi_seed_sharpe_handles_collapsed_nan_seeds(monkeypatch):
    _patch_simple(monkeypatch, sharpe_by_seed={0: 1.0, 1: float("nan"), 2: 3.0})

    result = multi_seed_sharpe(_dummy_df(), _dummy_df(), ["f1"], "TEST", seeds=[0, 1, 2], max_workers=1)

    assert result["mean_sharpe"] == pytest.approx(2.0)  # nan excluded, mean of [1.0, 3.0]
    assert result["n_collapsed"] == 1


def test_compare_configs_detects_a_real_difference_above_noise_floor(monkeypatch):
    # Config A (baseline): tightly clustered around 1.0. Config B
    # (candidate): tightly clustered around 3.0 -- a real, low-noise gap.
    call_count = {"n": 0}

    def _fake_train_ppo(train_df, feature_columns, seed=0, total_timesteps=100_000, **kwargs):
        call_count["n"] += 1
        return _FakeModel((tuple(feature_columns), seed))

    def _fake_backtest_ppo(model, df, feature_columns, ticker):
        return model.seed, None

    def _fake_sharpe_ratio(sentinel):
        (features, seed) = sentinel
        base = 1.0 if features == ("base",) else 3.0
        return base + (seed * 0.01)  # tiny, low spread

    monkeypatch.setattr(msv_module, "train_ppo", _fake_train_ppo)
    monkeypatch.setattr(msv_module, "backtest_ppo", _fake_backtest_ppo)
    monkeypatch.setattr(msv_module, "sharpe_ratio", _fake_sharpe_ratio)

    result = compare_configs_multi_seed(
        _dummy_df(), _dummy_df(), ["base"], ["base_plus_feature"], "TEST", seeds=[0, 1, 2], max_workers=1,
    )

    assert result["exceeds_noise_floor"] is True
    assert result["mean_sharpe_diff"] == pytest.approx(2.0, abs=0.05)


def test_compare_configs_flags_a_difference_within_noise_floor_as_untrustworthy(monkeypatch):
    """Regression test for the exact real problem this module fixes: VIX
    showed AAPL improving and GOOGL badly regressing from a single seed
    each -- a difference that size is easily within normal seed-to-seed
    spread and shouldn't be reported as a real effect."""
    def _fake_train_ppo(train_df, feature_columns, seed=0, total_timesteps=100_000, **kwargs):
        return _FakeModel((tuple(feature_columns), seed))

    def _fake_backtest_ppo(model, df, feature_columns, ticker):
        return model.seed, None

    # Both configs have real Sharpes scattered over a wide range (std ~2),
    # and their MEANS only differ by 0.3 -- much smaller than the spread.
    scattered = {0: 1.8, 1: -2.9, 2: 1.5, 3: 0.2, 4: -0.5}

    def _fake_sharpe_ratio(sentinel):
        (features, seed) = sentinel
        return scattered[seed] + (0.3 if features == ("base_plus_feature",) else 0.0)

    monkeypatch.setattr(msv_module, "train_ppo", _fake_train_ppo)
    monkeypatch.setattr(msv_module, "backtest_ppo", _fake_backtest_ppo)
    monkeypatch.setattr(msv_module, "sharpe_ratio", _fake_sharpe_ratio)

    result = compare_configs_multi_seed(
        _dummy_df(), _dummy_df(), ["base"], ["base_plus_feature"], "TEST", seeds=[0, 1, 2, 3, 4], max_workers=1,
    )

    assert result["exceeds_noise_floor"] is False


def test_compare_configs_passes_through_train_ppo_kwargs(monkeypatch):
    captured = []

    def _fake_train_ppo(train_df, feature_columns, seed=0, total_timesteps=100_000, **kwargs):
        captured.append(kwargs)
        return _FakeModel(seed)

    monkeypatch.setattr(msv_module, "train_ppo", _fake_train_ppo)
    monkeypatch.setattr(msv_module, "backtest_ppo", lambda model, df, fc, ticker: (model.seed, None))
    monkeypatch.setattr(msv_module, "sharpe_ratio", lambda seed: 1.0)

    compare_configs_multi_seed(
        _dummy_df(), _dummy_df(), ["base"], ["base2"], "TEST", seeds=[0], max_workers=1,
        use_differential_sharpe=True, dsr_benchmark_relative=True,
    )

    assert all(k.get("use_differential_sharpe") is True for k in captured)
    assert all(k.get("dsr_benchmark_relative") is True for k in captured)


def test_worker_trains_with_its_own_seed_and_returns_that_seeds_sharpe(monkeypatch):
    """The parallel worker must use the seed it was handed -- if payloads
    and seeds got misaligned, every reported per-seed Sharpe would be
    attributed to the wrong seed and the whole comparison would be junk.
    """
    _patch_simple(monkeypatch, sharpe_by_seed={0: 1.0, 7: 9.0})

    payload = (_dummy_df(), _dummy_df(), ["f1"], "TEST", 7, 100, {})
    assert _train_and_score_one_seed(payload) == 9.0


def test_worker_forwards_train_ppo_kwargs(monkeypatch):
    captured = {}

    def _fake_train_ppo(train_df, feature_columns, seed=0, total_timesteps=100_000, **kwargs):
        captured.update(kwargs)
        return _FakeModel(seed)

    monkeypatch.setattr(msv_module, "train_ppo", _fake_train_ppo)
    monkeypatch.setattr(msv_module, "backtest_ppo", lambda model, df, fc, ticker: (model.seed, None))
    monkeypatch.setattr(msv_module, "sharpe_ratio", lambda seed: 1.0)

    payload = (_dummy_df(), _dummy_df(), ["f1"], "TEST", 0, 100, {"use_differential_sharpe": True})
    _train_and_score_one_seed(payload)

    assert captured.get("use_differential_sharpe") is True


def test_multi_seed_sharpe_builds_one_payload_per_seed_in_order(monkeypatch):
    """`multi_seed_sharpe` must hand each worker the right seed, in the
    same order as `seeds` -- `sharpes[i]` is reported as seed `seeds[i]`'s
    result, so any misordering here silently mislabels every per-seed
    number in the output.
    """
    seen_seeds = []

    def _fake_worker(payload):
        train_df, holdout_df, feature_columns, ticker, seed, total_timesteps, kwargs = payload
        seen_seeds.append(seed)
        return float(seed)

    monkeypatch.setattr(msv_module, "_train_and_score_one_seed", _fake_worker)

    result = multi_seed_sharpe(
        _dummy_df(), _dummy_df(), ["f1"], "TEST", seeds=[3, 1, 4], max_workers=1,
    )

    assert seen_seeds == [3, 1, 4]
    assert result["sharpes"] == [3.0, 1.0, 4.0]


def test_multi_seed_sharpe_defaults_to_a_bounded_worker_pool():
    """Guards the memory ceiling: one worker per seed exhausted RAM and
    killed the dashboard task earlier in this project's history, so the
    default pool size must stay capped rather than scaling with seeds.
    """
    assert msv_module.DEFAULT_MAX_WORKERS <= 4
