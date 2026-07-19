"""Baseline runners for Phase 1: buy-and-hold, rule-based, vanilla PPO.

These are the comparison points every later phase gets measured against —
kept intentionally simple/unmodified from the v1 notebook's baselines so
the comparison stays fair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from scaata.config import (
    INITIAL_CASH,
    PPO_BATCH_SIZE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_TOTAL_TIMESTEPS,
)
from scaata.rl.env import HOLD, BUY, SELL, RobustTradingEnv


def buy_and_hold_equity(test_df: pd.DataFrame, ticker: str, initial_cash: float = INITIAL_CASH) -> np.ndarray:
    prices = test_df[test_df["Ticker"] == ticker]["Close"].values
    shares = initial_cash / prices[0]
    return shares * prices


def rule_based_signals(df: pd.DataFrame) -> np.ndarray:
    """Simple MA10/MA50 crossover, matching the paper's description of the
    mechanical rule-based baseline (no adaptive weighting)."""
    ma_fast = df["Close"].rolling(10).mean()
    ma_slow = df["Close"].rolling(50).mean()
    signal = np.where(ma_fast > ma_slow, BUY, np.where(ma_fast < ma_slow, SELL, HOLD))
    signal[: 50] = HOLD  # no signal until both MAs are warmed up
    return signal


def rule_based_equity(test_df: pd.DataFrame, ticker: str, initial_cash: float = INITIAL_CASH) -> tuple[np.ndarray, np.ndarray]:
    ticker_df = test_df[test_df["Ticker"] == ticker].reset_index(drop=True)
    signals = rule_based_signals(ticker_df)
    prices = ticker_df["Close"].values

    cash, shares, position = initial_cash, 0.0, 0
    equity = [cash]
    actions = []
    for i in range(len(prices) - 1):
        action = signals[i]
        price = prices[i]
        if action == BUY and position == 0:
            shares = cash / price
            cash = 0.0
            position = 1
        elif action == SELL and position == 1:
            cash = shares * price
            shares = 0.0
            position = 0
        equity.append(cash + shares * price)
        actions.append(action)
    return np.array(equity), np.array(actions)


def train_ppo(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int = 0,
    total_timesteps: int = PPO_TOTAL_TIMESTEPS,
) -> RecurrentPPO:
    """Train a single RecurrentPPO policy on episodes sampling randomly
    across all tickers in `train_df` (unchanged v1 approach for Phase 1)."""
    env = RobustTradingEnv(train_df, feature_columns)
    env.reset(seed=seed)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=PPO_LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        gamma=PPO_GAMMA,
        ent_coef=PPO_ENT_COEF,
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def backtest_ppo(
    model: RecurrentPPO, test_df: pd.DataFrame, feature_columns: list[str], ticker: str
) -> tuple[np.ndarray, np.ndarray]:
    env = RobustTradingEnv(test_df, feature_columns, fixed_ticker=ticker)
    obs, _ = env.reset()
    done = False
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    equity = [env.initial_cash]
    actions = []
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        episode_starts = np.array([done], dtype=bool)
        actions.append(int(action))
        equity.append(info["portfolio_value"])
    return np.array(equity), np.array(actions)
