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
    BASE_TRANSACTION_FEE,
    DSR_ETA,
    DSR_REWARD_SCALE,
    INITIAL_CASH,
    PPO_BATCH_SIZE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_TOTAL_TIMESTEPS,
)
from scaata.rl.env import HOLD, BUY, SELL, RobustTradingEnv


def buy_and_hold_equity(
    test_df: pd.DataFrame, ticker: str, initial_cash: float = INITIAL_CASH, fee: float = 0.0,
) -> np.ndarray:
    """`fee` (default 0, the original fee-free baseline) is charged once on
    entry as a fraction of the amount invested; day 0 shows the cash before
    it, so the first return includes the fee. Pass the env's base fee when
    comparing against a policy that pays fees."""
    prices = test_df[test_df["Ticker"] == ticker]["Close"].values
    shares = initial_cash * (1 - fee) / prices[0]
    equity = shares * prices
    if fee:
        equity[0] = initial_cash
    return equity


def rule_based_signals(df: pd.DataFrame, use_precomputed_mas: bool = False) -> np.ndarray:
    """Simple MA10/MA50 crossover, matching the paper's description of the
    mechanical rule-based baseline (no adaptive weighting).

    `use_precomputed_mas` uses the frame's own `ma_10`/`ma_50` columns
    (`add_features` computes them over full history) instead of recomputing
    on this slice -- recomputing on a six-month test slice leaves the first
    50 days, 40% of the window, forced to HOLD."""
    if use_precomputed_mas:
        ma_fast, ma_slow = df["ma_10"], df["ma_50"]
    else:
        ma_fast = df["Close"].rolling(10).mean()
        ma_slow = df["Close"].rolling(50).mean()
    signal = np.where(ma_fast > ma_slow, BUY, np.where(ma_fast < ma_slow, SELL, HOLD))
    if not use_precomputed_mas:
        signal[: 50] = HOLD  # no signal until both MAs are warmed up
    return signal


def rule_based_equity(
    test_df: pd.DataFrame, ticker: str, initial_cash: float = INITIAL_CASH,
    fee: float = 0.0, use_precomputed_mas: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """`fee` is charged per trade side as a fraction of the traded amount
    (default 0, the original fee-free baseline)."""
    ticker_df = test_df[test_df["Ticker"] == ticker].reset_index(drop=True)
    signals = rule_based_signals(ticker_df, use_precomputed_mas=use_precomputed_mas)
    prices = ticker_df["Close"].values

    cash, shares, position = initial_cash, 0.0, 0
    equity = [cash]
    actions = []
    for i in range(len(prices) - 1):
        action = signals[i]
        price = prices[i]
        if action == BUY and position == 0:
            shares = cash * (1 - fee) / price
            cash = 0.0
            position = 1
        elif action == SELL and position == 1:
            cash = shares * price * (1 - fee)
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
    use_differential_sharpe: bool = False,
    dsr_eta: float = DSR_ETA,
    dsr_reward_scale: float = DSR_REWARD_SCALE,
    dsr_benchmark_relative: bool = False,
    ent_coef: float = PPO_ENT_COEF,
    bc_model=None,
    include_position_obs: bool = False,
    random_episode_start_min_steps: int | None = None,
    transaction_fee: float = BASE_TRANSACTION_FEE,
    fixed_fee: float = 0.0,
) -> RecurrentPPO:
    """Train a single RecurrentPPO policy on episodes sampling randomly
    across all tickers in `train_df` (unchanged v1 approach for Phase 1).

    `use_differential_sharpe` (default off, preserves the original Phase 1
    behavior): trains against the Differential Sharpe Ratio reward instead
    of raw return -- this is the one reward mechanism found, this session,
    to have a real (if ticker-dependent) win over the original raw-return-
    plus-penalties reward, which reliably collapses to "never trade" at
    full training scale on several real tickers. Matches exactly the
    `dsr_only` ablation config's env construction
    (`scaata.evaluation.ablations.run_ablation_config`), so a model trained
    here reproduces that already-validated result, not a new untested
    variant.

    `dsr_benchmark_relative` (default off, matches existing behavior):
    computes the DSR input as the policy's return *minus* the underlying
    ticker's own same-step return, rather than the policy's raw return
    alone -- so sitting flat is only "safe" in reward terms during a
    falling market, not unconditionally. Already exists as
    `dsr_benchmark_relative_only` in the ablation suite, tried once and
    found not to fix the collapse pattern in isolation. Published research
    on this exact "no-trade artificial attractor" failure mode
    (arxiv.org/abs/2107.08083, arxiv.org/pdf/2605.30896) recommends both
    a benchmark-relative reward *and* higher entropy regularization
    together -- this project only ever tried the first alone.

    `ent_coef` (default `PPO_ENT_COEF`, matches existing behavior): PPO's
    entropy bonus coefficient. Left at its original 0.01 all session
    (never revisited specifically for the collapse problem), which sits at
    the *low* end of what that same research recommends (0.01-0.1) for
    preventing premature convergence to a deterministic no-action policy.

    `bc_model` (default None, matches existing behavior): an optional
    pre-trained `scaata.imitation.model.ImitationModel` (e.g. from
    `scaata.agents.orchestrator.run_inner_loop`'s output) to warm-start
    the policy's final layers from before training -- see
    `scaata.rl.policy_init.load_bc_weights_into_policy` for exactly which
    layers transfer and why only those. Already validated in
    `scaata.evaluation.ablations`'s `use_bc_init` configs; this is the
    same mechanism, just newly reachable from production training, which
    never called it at all before now.

    `include_position_obs` / `random_episode_start_min_steps` (default off):
    the audit's environment options, see `RobustTradingEnv`. A policy
    trained with `include_position_obs` must be backtested with it too.
    """
    env = RobustTradingEnv(
        train_df, feature_columns,
        use_differential_sharpe=use_differential_sharpe,
        dsr_eta=dsr_eta, dsr_reward_scale=dsr_reward_scale,
        dsr_benchmark_relative=dsr_benchmark_relative,
        include_position_obs=include_position_obs,
        random_episode_start_min_steps=random_episode_start_min_steps,
        transaction_fee=transaction_fee, fixed_fee=fixed_fee,
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
    if bc_model is not None:
        from scaata.rl.policy_init import load_bc_weights_into_policy
        load_bc_weights_into_policy(bc_model, model.policy)
    model.learn(total_timesteps=total_timesteps)
    return model


def backtest_ppo(
    model: RecurrentPPO, test_df: pd.DataFrame, feature_columns: list[str], ticker: str,
    include_position_obs: bool = False,
    transaction_fee: float = BASE_TRANSACTION_FEE,
    fixed_fee: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    env = RobustTradingEnv(
        test_df, feature_columns, fixed_ticker=ticker, include_position_obs=include_position_obs,
        transaction_fee=transaction_fee, fixed_fee=fixed_fee,
    )
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
