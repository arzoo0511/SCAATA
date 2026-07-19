"""Regression test for the v1 gap where the BC model was trained, saved,
and never loaded into the RL policy. This must keep passing after any
future refactor of `rl/policy_init.py` or `imitation/model.py`.
"""
import copy

import numpy as np
from sb3_contrib import RecurrentPPO

from scaata.config import FEATURE_COLUMNS
from scaata.imitation.train import train_imitation_model
from scaata.rl.env import RobustTradingEnv
from scaata.rl.policy_init import load_bc_weights_into_policy, verify_transfer


def _make_synthetic_df(n=200):
    rng = np.random.default_rng(0)
    dates = None
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    import pandas as pd
    df = pd.DataFrame(
        {c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    df["Close"] = close
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(30, min_periods=1).mean()
    df["Ticker"] = "TEST"
    return df


def test_bc_weights_actually_load_into_ppo_policy():
    df = _make_synthetic_df()
    env = RobustTradingEnv(df, FEATURE_COLUMNS, fixed_ticker="TEST")
    model = RecurrentPPO("MlpLstmPolicy", env, n_steps=64, verbose=0, seed=0)
    fresh_state = copy.deepcopy(model.policy.state_dict())

    rng = np.random.default_rng(1)
    states = rng.normal(0, 1, (300, len(FEATURE_COLUMNS)))
    signals = [rng.choice([-1, 0, 1], 300) for _ in range(2)]
    bc_model = train_imitation_model(states, signals, epochs=2)

    transferred = load_bc_weights_into_policy(bc_model, model.policy)
    assert all(transferred.values()), "expected all mapped keys to transfer successfully"

    report = verify_transfer(bc_model, model.policy, fresh_state)
    assert report["weights_match_bc"], "loaded policy weights must exactly match BC's trained weights"
    assert report["differs_from_fresh_init"], (
        "loaded policy weights must differ from a fresh init — a silent no-op "
        "(the original v1 bug) would leave this False"
    )
