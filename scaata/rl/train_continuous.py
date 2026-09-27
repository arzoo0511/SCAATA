"""Training/backtest harness for `ContinuousTradingEnv` (Phase 19) --
mirrors `scaata.rl.train.train_ppo`/`backtest_ppo`'s shape exactly so a
continuous-action-space policy can be trained and evaluated the same way
as the existing discrete one, for a fair, apples-to-apples comparison.
`RecurrentPPO` (sb3_contrib) supports continuous (Box) action spaces
natively via a Gaussian policy head -- no new RL algorithm/dependency
needed, just a different environment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from scaata.config import (
    DSR_ETA,
    DSR_REWARD_SCALE,
    PPO_BATCH_SIZE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_TOTAL_TIMESTEPS,
)
from scaata.rl.continuous_env import ContinuousTradingEnv


def train_continuous_ppo(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int = 0,
    total_timesteps: int = PPO_TOTAL_TIMESTEPS,
    use_differential_sharpe: bool = True,
    dsr_eta: float = DSR_ETA,
    dsr_reward_scale: float = DSR_REWARD_SCALE,
    dsr_benchmark_relative: bool = False,
    ent_coef: float = PPO_ENT_COEF,
) -> RecurrentPPO:
    env = ContinuousTradingEnv(
        train_df, feature_columns,
        use_differential_sharpe=use_differential_sharpe,
        dsr_eta=dsr_eta, dsr_reward_scale=dsr_reward_scale,
        dsr_benchmark_relative=dsr_benchmark_relative,
    )
    env.reset(seed=seed)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        gamma=PPO_GAMMA,
        ent_coef=ent_coef,
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def backtest_continuous_ppo(
    model: RecurrentPPO, test_df: pd.DataFrame, feature_columns: list[str], ticker: str,
) -> tuple[np.ndarray, np.ndarray]:
    env = ContinuousTradingEnv(test_df, feature_columns, fixed_ticker=ticker)
    obs, _ = env.reset()
    done = False
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    equity = [env.initial_cash]
    fractions = []
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        episode_starts = np.array([done], dtype=bool)
        fractions.append(info["target_fraction"])
        equity.append(info["portfolio_value"])
    return np.array(equity), np.array(fractions)
