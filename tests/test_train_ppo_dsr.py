"""Tests for train_ppo's optional Differential Sharpe Ratio wiring (used
to train the per-ticker models the Alpaca daily-signal dashboard loads) --
fast plumbing tests (fake env/model, no real training) confirming the flag
reaches RobustTradingEnv correctly and defaults preserve Phase 1 behavior
for existing callers.
"""
import numpy as np
import pandas as pd
import pytest

import scaata.rl.train as train_module
from scaata.config import DSR_ETA, DSR_REWARD_SCALE, FEATURE_COLUMNS


class _FakeEnv:
    captured_use_differential_sharpe = None
    captured_dsr_eta = None
    captured_dsr_reward_scale = None
    captured_dsr_benchmark_relative = None

    def __init__(
        self, df, feature_columns, use_differential_sharpe=False, dsr_eta=None, dsr_reward_scale=None,
        dsr_benchmark_relative=False, **kwargs,
    ):
        _FakeEnv.captured_use_differential_sharpe = use_differential_sharpe
        _FakeEnv.captured_dsr_eta = dsr_eta
        _FakeEnv.captured_dsr_reward_scale = dsr_reward_scale
        _FakeEnv.captured_dsr_benchmark_relative = dsr_benchmark_relative
        self.df = df

    def reset(self, seed=None):
        return np.zeros(len(FEATURE_COLUMNS)), {}


class _FakeModel:
    captured_ent_coef = None

    def __init__(self, *args, **kwargs):
        _FakeModel.captured_ent_coef = kwargs.get("ent_coef")
        self.policy = object()

    def learn(self, total_timesteps):
        pass


def _make_df(n=30, ticker="TEST"):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({c: 0.0 for c in FEATURE_COLUMNS} | {"Ticker": ticker}, index=dates)


@pytest.fixture(autouse=True)
def _patch_env_and_model(monkeypatch):
    monkeypatch.setattr(train_module, "RobustTradingEnv", _FakeEnv)
    monkeypatch.setattr(train_module, "RecurrentPPO", _FakeModel)
    _FakeEnv.captured_use_differential_sharpe = None
    _FakeEnv.captured_dsr_eta = None
    _FakeEnv.captured_dsr_reward_scale = None
    _FakeEnv.captured_dsr_benchmark_relative = None
    _FakeModel.captured_ent_coef = None


def test_train_ppo_defaults_to_no_dsr():
    train_module.train_ppo(_make_df(), FEATURE_COLUMNS, total_timesteps=10)
    assert _FakeEnv.captured_use_differential_sharpe is False


def test_train_ppo_passes_through_dsr_flag_and_defaults():
    train_module.train_ppo(_make_df(), FEATURE_COLUMNS, total_timesteps=10, use_differential_sharpe=True)
    assert _FakeEnv.captured_use_differential_sharpe is True
    assert _FakeEnv.captured_dsr_eta == DSR_ETA
    assert _FakeEnv.captured_dsr_reward_scale == DSR_REWARD_SCALE


def test_train_ppo_passes_through_custom_dsr_params():
    train_module.train_ppo(
        _make_df(), FEATURE_COLUMNS, total_timesteps=10,
        use_differential_sharpe=True, dsr_eta=0.01, dsr_reward_scale=50.0,
    )
    assert _FakeEnv.captured_dsr_eta == 0.01
    assert _FakeEnv.captured_dsr_reward_scale == 50.0


def test_train_ppo_defaults_to_no_benchmark_relative():
    train_module.train_ppo(_make_df(), FEATURE_COLUMNS, total_timesteps=10)
    assert _FakeEnv.captured_dsr_benchmark_relative is False


def test_train_ppo_passes_through_benchmark_relative_flag():
    train_module.train_ppo(
        _make_df(), FEATURE_COLUMNS, total_timesteps=10,
        use_differential_sharpe=True, dsr_benchmark_relative=True,
    )
    assert _FakeEnv.captured_dsr_benchmark_relative is True


def test_train_ppo_defaults_ent_coef_to_config_value():
    from scaata.config import PPO_ENT_COEF

    train_module.train_ppo(_make_df(), FEATURE_COLUMNS, total_timesteps=10)
    assert _FakeModel.captured_ent_coef == PPO_ENT_COEF


def test_train_ppo_passes_through_custom_ent_coef():
    train_module.train_ppo(_make_df(), FEATURE_COLUMNS, total_timesteps=10, ent_coef=0.05)
    assert _FakeModel.captured_ent_coef == 0.05


def test_train_ppo_does_not_warm_start_by_default(monkeypatch):
    """Regression guard: bc_model=None (the default, matching every
    existing caller) must never even attempt a warm-start call."""
    calls = []
    monkeypatch.setattr("scaata.rl.policy_init.load_bc_weights_into_policy", lambda bc, policy: calls.append((bc, policy)))

    train_module.train_ppo(_make_df(), FEATURE_COLUMNS, total_timesteps=10)

    assert calls == []


def test_train_ppo_warm_starts_when_bc_model_given(monkeypatch):
    """The actual new wiring: passing a bc_model must trigger
    load_bc_weights_into_policy with that model and the constructed
    policy, before training -- this is what makes BC warm-start (already
    validated in the ablation suite) reachable from production training
    for the first time."""
    calls = []
    monkeypatch.setattr("scaata.rl.policy_init.load_bc_weights_into_policy", lambda bc, policy: calls.append((bc, policy)))
    sentinel_bc_model = object()

    train_module.train_ppo(_make_df(), FEATURE_COLUMNS, total_timesteps=10, bc_model=sentinel_bc_model)

    assert len(calls) == 1
    assert calls[0][0] is sentinel_bc_model
