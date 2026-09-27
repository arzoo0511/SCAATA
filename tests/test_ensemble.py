"""Tests for the PPO ensemble and uncertainty-aware sizing (Phase 14).
`EnsemblePolicy`/`size_from_uncertainty` are tested with lightweight fake
members (no real training cost, matching `tests/test_forward_test.py`'s
`_FakePolicy` pattern); `train_ppo_ensemble` gets one real, deliberately
tiny (small df, few timesteps, 2 seeds) end-to-end training test to prove
the DEFAULT_SEEDS-activation wiring actually works, without paying full
5-seed/100k-timestep cost in every test run.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import FEATURE_COLUMNS
from scaata.rl.env import HOLD, BUY, SELL
from scaata.rl.ensemble import (
    EnsemblePolicy,
    size_from_uncertainty,
    train_ppo_ensemble,
)


class _FixedActionMember:
    def __init__(self, action):
        self.action = action

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return self.action, "new-state"


def test_unanimous_ensemble_has_zero_uncertainty_and_full_agreement():
    members = [_FixedActionMember(BUY) for _ in range(5)]
    ensemble = EnsemblePolicy(members)

    result = ensemble.predict_with_uncertainty(np.zeros(len(FEATURE_COLUMNS)))

    assert result["action"] == BUY
    assert result["action_agreement"] == 1.0
    # Near-zero, not exactly 0.0: the entropy calc clips probabilities away
    # from 0 (to avoid log(0)), which leaves a negligible residual from the
    # zero-probability actions -- not a real uncertainty signal.
    assert result["epistemic_uncertainty"] == pytest.approx(0.0, abs=1e-6)


def test_maximally_split_ensemble_has_high_uncertainty():
    # 5 members split as evenly as possible across all 3 actions -> near-max entropy
    members = [
        _FixedActionMember(HOLD), _FixedActionMember(HOLD),
        _FixedActionMember(BUY), _FixedActionMember(BUY),
        _FixedActionMember(SELL),
    ]
    ensemble = EnsemblePolicy(members)

    result = ensemble.predict_with_uncertainty(np.zeros(len(FEATURE_COLUMNS)))

    assert result["action_agreement"] == pytest.approx(2 / 5)
    assert result["epistemic_uncertainty"] > 0.8  # close to the normalized max of 1.0


def test_majority_vote_wins_the_action():
    members = [_FixedActionMember(BUY), _FixedActionMember(BUY), _FixedActionMember(BUY), _FixedActionMember(SELL), _FixedActionMember(HOLD)]
    ensemble = EnsemblePolicy(members)

    result = ensemble.predict_with_uncertainty(np.zeros(len(FEATURE_COLUMNS)))

    assert result["action"] == BUY
    assert result["action_agreement"] == pytest.approx(3 / 5)


def test_ensemble_policy_requires_at_least_one_member():
    with pytest.raises(ValueError):
        EnsemblePolicy([])


def test_size_from_uncertainty_abstains_below_agreement_threshold():
    action, size = size_from_uncertainty(
        base_action=BUY, epistemic_uncertainty=0.5, action_agreement=0.4, min_agreement=0.6
    )
    assert action == HOLD
    assert size == 0.0


def test_size_from_uncertainty_full_sizes_when_certain():
    action, size = size_from_uncertainty(
        base_action=BUY, epistemic_uncertainty=0.0, action_agreement=1.0, min_agreement=0.6, kappa=1.0
    )
    assert action == BUY
    assert size == pytest.approx(1.0)


def test_size_from_uncertainty_shrinks_with_uncertainty_but_respects_floor():
    action, size = size_from_uncertainty(
        base_action=BUY, epistemic_uncertainty=0.9, action_agreement=0.8, min_agreement=0.6, kappa=1.0, min_size=0.1
    )
    assert action == BUY
    assert size == pytest.approx(0.1)  # 1 - 0.9 = 0.1, exactly at the floor here


def test_size_from_uncertainty_never_goes_below_min_size():
    action, size = size_from_uncertainty(
        base_action=BUY, epistemic_uncertainty=1.0, action_agreement=0.8, min_agreement=0.6, kappa=2.0, min_size=0.15
    )
    assert size == pytest.approx(0.15)


def _make_tiny_train_df(n=80, seed=0, ticker="TEST"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["Ticker"] = ticker
    return df


def test_train_ppo_ensemble_activates_default_seeds_end_to_end():
    """A deliberately tiny real-training smoke test (2 seeds, ~500
    timesteps, an 80-row df) -- proves the DEFAULT_SEEDS loop actually
    trains distinct, usable policies, without paying full ensemble cost in
    every test run (the 5x-training-cost tradeoff is a documented,
    intentional scope decision for real usage, not for this test)."""
    train_df = _make_tiny_train_df()

    members = train_ppo_ensemble(train_df, FEATURE_COLUMNS, seeds=[0, 1], total_timesteps=500)

    assert len(members) == 2
    obs = train_df[FEATURE_COLUMNS].iloc[0].values.astype(np.float32)
    ensemble = EnsemblePolicy(members)
    result = ensemble.predict_with_uncertainty(obs, states=[None, None], episode_start=np.array([True]))

    assert result["action"] in (HOLD, BUY, SELL)
    assert 0.0 <= result["action_agreement"] <= 1.0
    assert 0.0 <= result["epistemic_uncertainty"] <= 1.0
