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
    DSR_ETA,
    DSR_REWARD_SCALE,
    FEATURE_COLUMNS,
    META_CONFIDENCE_COLUMN,
    NOVELTY_SCORE_COLUMN,
    PPO_BATCH_SIZE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_TOTAL_TIMESTEPS,
    SENTIMENT_SCORE_COLUMN,
)
from scaata.features.sentiment_features import merge_sentiment_into_features
from scaata.imitation.train import train_imitation_model
from scaata.regimes.novelty_features import merge_novelty_into_features
from scaata.rl.env import RobustTradingEnv
from scaata.rl.policy_init import load_bc_weights_into_policy
from scaata.strategies.meta_selector import attach_meta_confidence, train_meta_selector


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_bc_init: bool
    use_meta_feature: bool
    use_self_critique: bool
    use_sentiment_feature: bool = False
    use_differential_sharpe: bool = False
    dsr_benchmark_relative: bool = False
    use_novelty_feature: bool = False


ABLATION_CONFIGS = [
    AblationConfig("phase1_baseline", use_bc_init=False, use_meta_feature=False, use_self_critique=False),
    AblationConfig("bc_only", use_bc_init=True, use_meta_feature=False, use_self_critique=False),
    AblationConfig("meta_only", use_bc_init=False, use_meta_feature=True, use_self_critique=False),
    AblationConfig("self_critique_only", use_bc_init=False, use_meta_feature=False, use_self_critique=True),
    AblationConfig("full_pipeline", use_bc_init=True, use_meta_feature=True, use_self_critique=True),
    # Phase 9: isolates whether wiring sentiment in (previously computed but
    # never fed to any decision — see scaata/features/sentiment_features.py)
    # moves Sharpe/Sortino/MaxDD at all, on its own and combined with the
    # rest of the Phase 2 pipeline. A null result here is a legitimate
    # finding, not a bug — Phase 10's regime-conditional Hedge combiner is
    # the mechanism expected to unlock sentiment's value in specific regimes
    # a flat ablation can't isolate.
    AblationConfig(
        "sentiment_only", use_bc_init=False, use_meta_feature=False, use_self_critique=False,
        use_sentiment_feature=True,
    ),
    AblationConfig(
        "full_pipeline_with_sentiment", use_bc_init=True, use_meta_feature=True, use_self_critique=True,
        use_sentiment_feature=True,
    ),
    # Differential Sharpe Ratio reward: found via live-data testing that
    # `self_critique_only`/`full_pipeline` (raw-return + hand-tuned
    # penalties) collapse to a "never trade" policy at full (100k-timestep)
    # training scale. These two arms swap that reward for the Differential
    # Sharpe Ratio (scaata.rl.reward.differential_sharpe_reward) -- a
    # principled risk-adjusted signal with no separate penalty coefficients
    # to miscalibrate -- to test directly whether that's what avoids the
    # collapse. `use_self_critique` and `use_differential_sharpe` are kept
    # mutually exclusive in these two configs (not enforced by the
    # mechanism itself) since combining two different risk-adjustment
    # schemes in the same reward would confound which one is responsible
    # for any observed effect.
    AblationConfig(
        "dsr_only", use_bc_init=False, use_meta_feature=False, use_self_critique=False,
        use_differential_sharpe=True,
    ),
    AblationConfig(
        "full_pipeline_with_dsr", use_bc_init=True, use_meta_feature=True, use_self_critique=False,
        use_differential_sharpe=True,
    ),
    # Benchmark-relative DSR: found necessary because plain dsr_only only
    # avoids the "never trade" collapse when buy-and-hold happens to be a
    # strong bet on that ticker's own training data (verified: works on
    # AAPL, collapses on MSFT, same failure mode as self_critique_only just
    # with a different trigger). Feeds the tracker excess return over that
    # ticker's own buy-and-hold instead of raw return, removing "replicate
    # buy-and-hold" as a free-lunch attractor -- see
    # RobustTradingEnv.dsr_benchmark_relative in scaata/rl/env.py.
    AblationConfig(
        "dsr_benchmark_relative_only", use_bc_init=False, use_meta_feature=False, use_self_critique=False,
        use_differential_sharpe=True, dsr_benchmark_relative=True,
    ),
    # Novelty feature: novelty_score existed since Phase 10 but was only
    # ever consumed ad hoc inside critique_node to modulate Hedge's eta --
    # never fed to the RL policy's own observation space. Isolates whether
    # giving the policy this signal directly moves anything on its own
    # (novelty_only) and specifically whether it reduces how often dsr_only
    # collapses to "never trade" (dsr_only_with_novelty) -- the collapse
    # was found to correlate with the policy having no demonstrated edge on
    # a given ticker; richer, more informative features are the untested
    # remaining lever for that, as opposed to further reward-shaping
    # variants (which were tried twice and both fell short — see
    # dsr_benchmark_relative_only above).
    AblationConfig(
        "novelty_only", use_bc_init=False, use_meta_feature=False, use_self_critique=False,
        use_novelty_feature=True,
    ),
    AblationConfig(
        "dsr_only_with_novelty", use_bc_init=False, use_meta_feature=False, use_self_critique=False,
        use_differential_sharpe=True, use_novelty_feature=True,
    ),
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
    train_sentiment_df: pd.DataFrame | None = None,
    test_sentiment_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trains + backtests one ablation config. `bc_model`/`meta_model` are
    precomputed once by `run_ablation_suite` and reused across configs that
    need them, so training cost isn't duplicated across configs. Sentiment
    (Phase 9) is merged in only for configs with `use_sentiment_feature`;
    `train_sentiment_df`/`test_sentiment_df` are required for those configs
    (each with `date`/`ticker`/`sentiment_score` columns, see
    `scaata.features.sentiment_features.merge_sentiment_into_features`).
    Novelty (Phase 10, wired for the first time here) is merged in for
    configs with `use_novelty_feature` -- no extra data required, since
    `scaata.regimes.novelty_features.merge_novelty_into_features` derives
    it from whatever feature_columns are already present."""
    feature_columns = list(FEATURE_COLUMNS)
    train_env_df, test_env_df = train_df, test_df

    if config.use_meta_feature:
        train_env_df = attach_meta_confidence(train_env_df, meta_model, FEATURE_COLUMNS)
        test_env_df = attach_meta_confidence(test_env_df, meta_model, FEATURE_COLUMNS)
        feature_columns = feature_columns + [META_CONFIDENCE_COLUMN]

    if config.use_sentiment_feature:
        if train_sentiment_df is None or test_sentiment_df is None:
            raise ValueError(f"config '{config.name}' requires train_sentiment_df/test_sentiment_df")
        train_env_df = merge_sentiment_into_features(train_env_df, train_sentiment_df)
        test_env_df = merge_sentiment_into_features(test_env_df, test_sentiment_df)
        feature_columns = feature_columns + [SENTIMENT_SCORE_COLUMN]

    if config.use_novelty_feature:
        train_env_df = merge_novelty_into_features(train_env_df, feature_columns)
        test_env_df = merge_novelty_into_features(test_env_df, feature_columns)
        feature_columns = feature_columns + [NOVELTY_SCORE_COLUMN]

    env = RobustTradingEnv(
        train_env_df, feature_columns,
        enable_self_critique=config.use_self_critique,
        use_differential_sharpe=config.use_differential_sharpe,
        dsr_eta=DSR_ETA, dsr_reward_scale=DSR_REWARD_SCALE,
        dsr_benchmark_relative=config.dsr_benchmark_relative,
    )
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
        test_env_df, feature_columns, fixed_ticker=ticker,
        enable_self_critique=config.use_self_critique,
        use_differential_sharpe=config.use_differential_sharpe,
        dsr_eta=DSR_ETA, dsr_reward_scale=DSR_REWARD_SCALE,
        dsr_benchmark_relative=config.dsr_benchmark_relative,
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
    train_sentiment_df: pd.DataFrame | None = None,
    test_sentiment_df: pd.DataFrame | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    needs_bc = any(c.use_bc_init for c in configs)
    needs_meta = any(c.use_meta_feature for c in configs)
    needs_sentiment = any(c.use_sentiment_feature for c in configs)
    if needs_sentiment and (train_sentiment_df is None or test_sentiment_df is None):
        raise ValueError(
            "one or more configs set use_sentiment_feature=True but "
            "train_sentiment_df/test_sentiment_df were not provided"
        )

    bc_model, meta_model = (None, None)
    if needs_bc or needs_meta:
        bc_model, meta_model = _prepare_bc_and_meta(train_df, strategy_signals, FEATURE_COLUMNS)

    results = {}
    for config in configs:
        equity, actions = run_ablation_config(
            config, train_df, test_df, ticker, strategy_signals,
            bc_model=bc_model, meta_model=meta_model, seed=seed, ppo_timesteps=ppo_timesteps,
            train_sentiment_df=train_sentiment_df, test_sentiment_df=test_sentiment_df,
        )
        results[config.name] = (equity, actions)
    return results
