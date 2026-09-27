"""Plumbing tests for scaata.rl.train_continuous (Phase 19) -- fake
env/model, no real training, confirming train_continuous_ppo wires its
parameters into ContinuousTradingEnv/RecurrentPPO correctly and
backtest_continuous_ppo correctly drives a fixed-ticker rollout to
completion. Mirrors test_train_ppo_dsr.py's pattern for the existing
discrete-action train_ppo.
"""
import numpy as np
import pandas as pd
import pytest

import scaata.rl.train_continuous as train_continuous_module
from scaata.config import FEATURE_COLUMNS


class _FakeEnv:
    captured_kwargs = None

    def __init__(self, df, feature_columns, fixed_ticker=None, **kwargs):
        _FakeEnv.captured_kwargs = kwargs
        self.df = df
        self.fixed_ticker = fixed_ticker
        self.initial_cash = 10_000.0
        self._step_idx = 0
        self._n = len(df)

    def reset(self, seed=None):
        self._step_idx = 0
        return np.zeros(len(FEATURE_COLUMNS)), {}

    def step(self, action):
        self._step_idx += 1
        done = self._step_idx >= self._n - 1
        fraction = float(np.clip(action[0], 0.0, 1.0))
        info = {"portfolio_value": 10_000.0 * (1 + 0.001 * self._step_idx), "target_fraction": fraction}
        return np.zeros(len(FEATURE_COLUMNS)), 0.0, done, False, info


class _FakeModel:
    def __init__(self, *args, **kwargs):
        pass

    def learn(self, total_timesteps):
        pass

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.array([0.5]), None


def _make_df(n=10, ticker="TEST"):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({c: 0.0 for c in FEATURE_COLUMNS} | {"Ticker": ticker, "Close": 100.0}, index=dates)


@pytest.fixture(autouse=True)
def _patch_env_and_model(monkeypatch):
    monkeypatch.setattr(train_continuous_module, "ContinuousTradingEnv", _FakeEnv)
    monkeypatch.setattr(train_continuous_module, "RecurrentPPO", _FakeModel)
    _FakeEnv.captured_kwargs = None


def test_train_continuous_ppo_defaults_dsr_on():
    """Unlike the discrete train_ppo (DSR default off, for backward
    compatibility with existing callers), this is a brand-new module with
    no prior callers to preserve -- defaults straight to the
    already-validated DSR mechanism rather than reproducing the same
    opt-in trap that caused a real regression in scaata.rl.retrain."""
    train_continuous_module.train_continuous_ppo(_make_df(), FEATURE_COLUMNS, total_timesteps=10)
    assert _FakeEnv.captured_kwargs["use_differential_sharpe"] is True


def test_train_continuous_ppo_passes_through_benchmark_relative_and_entropy():
    train_continuous_module.train_continuous_ppo(
        _make_df(), FEATURE_COLUMNS, total_timesteps=10,
        dsr_benchmark_relative=True, ent_coef=0.05,
    )
    assert _FakeEnv.captured_kwargs["dsr_benchmark_relative"] is True


def test_backtest_continuous_ppo_runs_to_completion_and_collects_fractions():
    model = _FakeModel()
    equity, fractions = train_continuous_module.backtest_continuous_ppo(model, _make_df(n=6), FEATURE_COLUMNS, "TEST")

    assert len(equity) == len(fractions) + 1
    assert all(f == pytest.approx(0.5) for f in fractions)


def test_backtest_continuous_ppo_first_equity_value_is_initial_cash():
    model = _FakeModel()
    equity, _ = train_continuous_module.backtest_continuous_ppo(model, _make_df(n=6), FEATURE_COLUMNS, "TEST")
    assert equity[0] == 10_000.0
