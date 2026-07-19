"""Phase 2 ablation suite: isolates the contribution of each mechanism
(BC init, meta-selector feature, self-critique reward) against the Phase 1
baseline and the full combined pipeline — the evidence that closing the
v1 wiring gaps actually mattered, not just that the code runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from scaata.config import (
    FEATURE_COLUMNS,
    META_CONFIDENCE_COLUMN,
    PPO_BATCH_SIZE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_TOTAL_TIMESTEPS,
)
from scaata.imitation.train import train_imitation_model
from scaata.rl.env import RobustTradingEnv
from scaata.rl.policy_init import load_bc_weights_into_policy
from scaata.strategies.meta_selector import attach_meta_confidence, train_meta_selector


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_bc_init: bool
    use_meta_feature: bool
    use_self_critique: bool


ABLATION_CONFIGS = [
    AblationConfig("phase1_baseline", use_bc_init=False, use_meta_feature=False, use_self_critique=False),
    AblationConfig("bc_only", use_bc_init=True, use_meta_feature=False, use_self_critique=False),
    AblationConfig("meta_only", use_bc_init=False, use_meta_feature=True, use_self_critique=False),
    AblationConfig("self_critique_only", use_bc_init=False, use_meta_feature=False, use_self_critique=True),
    AblationConfig("full_pipeline", use_bc_init=True, use_meta_feature=True, use_self_critique=True),
]


def _prepare_bc_and_meta(train_df: pd.DataFrame, strategy_signals: list[dict], feature_columns: list[str]):
    states = train_df[feature_columns].values
    signals_list = [s["signals"] for s in strategy_signals]
    bc_model = train_imitation_model(states, signals_list, epochs=15, seed=0)

    future_returns = train_df["returns"].shift(-1).fillna(0).values
    meta_model, _ = train_meta_selector(states, signals_list, future_returns, epochs=15, seed=0)
    return bc_model, meta_model


def run_ablation_config(
    config: AblationConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ticker: str,
    strategy_signals: list[dict],
    bc_model=None,
    meta_model=None,
    seed: int = 0,
    ppo_timesteps: int = PPO_TOTAL_TIMESTEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Trains + backtests one ablation config. `bc_model`/`meta_model` are
    precomputed once by `run_ablation_suite` and reused across configs that
    need them, so training cost isn't duplicated across the 5 configs."""
    feature_columns = list(FEATURE_COLUMNS)
    train_env_df, test_env_df = train_df, test_df

    if config.use_meta_feature:
        train_env_df = attach_meta_confidence(train_df, meta_model, FEATURE_COLUMNS)
        test_env_df = attach_meta_confidence(test_df, meta_model, FEATURE_COLUMNS)
        feature_columns = feature_columns + [META_CONFIDENCE_COLUMN]

    env = RobustTradingEnv(train_env_df, feature_columns, enable_self_critique=config.use_self_critique)
    env.reset(seed=seed)
    model = RecurrentPPO(
        "MlpLstmPolicy", env,
        learning_rate=PPO_LEARNING_RATE, n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE,
        gamma=PPO_GAMMA, ent_coef=PPO_ENT_COEF, seed=seed, verbose=0,
    )

    if config.use_bc_init:
        load_bc_weights_into_policy(bc_model, model.policy)

    model.learn(total_timesteps=ppo_timesteps)

    test_env = RobustTradingEnv(
        test_env_df, feature_columns, fixed_ticker=ticker, enable_self_critique=config.use_self_critique
    )
    obs, _ = test_env.reset()
    done = False
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    equity = [test_env.initial_cash]
    actions = []
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        episode_starts = np.array([done], dtype=bool)
        actions.append(int(action))
        equity.append(info["portfolio_value"])

    return np.array(equity), np.array(actions)


def run_ablation_suite(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ticker: str,
    strategy_signals: list[dict],
    seed: int = 0,
    ppo_timesteps: int = PPO_TOTAL_TIMESTEPS,
    configs: list[AblationConfig] = ABLATION_CONFIGS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    needs_bc = any(c.use_bc_init for c in configs)
    needs_meta = any(c.use_meta_feature for c in configs)

    bc_model, meta_model = (None, None)
    if needs_bc or needs_meta:
        bc_model, meta_model = _prepare_bc_and_meta(train_df, strategy_signals, FEATURE_COLUMNS)

    results = {}
    for config in configs:
        equity, actions = run_ablation_config(
            config, train_df, test_df, ticker, strategy_signals,
            bc_model=bc_model, meta_model=meta_model, seed=seed, ppo_timesteps=ppo_timesteps,
        )
        results[config.name] = (equity, actions)
    return results
