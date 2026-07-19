"""Trains Phase 4's regime-specialist experts.

Each expert must only ever see genuinely adjacent trading days — naively
filtering `train_df` down to "all stress-labeled rows" would splice
together non-adjacent dates, corrupting `RobustTradingEnv`'s day-to-day
percent-change reward with artificial jumps between unrelated days. Instead,
`build_regime_segments` finds contiguous runs of the same causal regime
bucket, and `SegmentedTradingEnv` samples one whole contiguous segment per
episode (same mechanics as `RobustTradingEnv` otherwise).
"""
from __future__ import annotations

import gymnasium as gym
import pandas as pd
from sb3_contrib import RecurrentPPO

from scaata.config import (
    PPO_BATCH_SIZE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_TOTAL_TIMESTEPS,
)
from scaata.regimes.detector import threshold_regime_labels
from scaata.rl.env import RobustTradingEnv
from scaata.rl.moe.gating import bucket_regime_labels


class SegmentedTradingEnv(RobustTradingEnv):
    """Same mechanics as `RobustTradingEnv`; each `reset()` samples one
    whole contiguous same-regime-bucket segment instead of a full
    multi-year ticker series."""

    def __init__(self, segments: list[pd.DataFrame], feature_columns, **kwargs):
        if not segments:
            raise ValueError("SegmentedTradingEnv requires at least one non-empty segment")
        super().__init__(segments[0], feature_columns, **kwargs)
        self.segments = segments

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        chosen = self.segments[self.np_random.integers(0, len(self.segments))]
        return self._init_episode(chosen)


def build_regime_segments(train_df: pd.DataFrame, bucket: str, min_length: int = 10) -> list[pd.DataFrame]:
    """Splits `train_df` (per ticker) into contiguous runs where the causal
    threshold-detector's regime bucket matches `bucket` ('calm'/'stress'),
    dropping runs shorter than `min_length` days (too short for a
    meaningful episode)."""
    labeled = threshold_regime_labels(train_df)
    labeled["bucket"] = bucket_regime_labels(labeled["regime"])

    segments = []
    for _, group in labeled.groupby("Ticker"):
        group = group.sort_index()
        is_target = (group["bucket"] == bucket).values

        run_start = None
        for i, flag in enumerate(is_target):
            if flag and run_start is None:
                run_start = i
            elif not flag and run_start is not None:
                if i - run_start >= min_length:
                    segments.append(group.iloc[run_start:i])
                run_start = None
        if run_start is not None and len(group) - run_start >= min_length:
            segments.append(group.iloc[run_start:])

    return segments


def train_regime_expert(
    train_df: pd.DataFrame,
    bucket: str,
    feature_columns: list[str],
    seed: int = 0,
    total_timesteps: int = PPO_TOTAL_TIMESTEPS,
    min_segment_length: int = 10,
) -> tuple[RecurrentPPO, int]:
    segments = build_regime_segments(train_df, bucket, min_length=min_segment_length)
    if not segments:
        raise ValueError(f"No segments of length >= {min_segment_length} found for bucket '{bucket}'")

    env = SegmentedTradingEnv(segments, feature_columns)
    env.reset(seed=seed)
    model = RecurrentPPO(
        "MlpLstmPolicy", env,
        learning_rate=PPO_LEARNING_RATE, n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE,
        gamma=PPO_GAMMA, ent_coef=PPO_ENT_COEF, seed=seed, verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return model, len(segments)


def train_moe_experts(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int = 0,
    total_timesteps: int = PPO_TOTAL_TIMESTEPS,
    min_segment_length: int = 10,
) -> dict:
    """Trains one expert per bucket ('calm', 'stress'). Returns
    {"experts": {bucket: model}, "segment_counts": {bucket: n}} — the
    segment counts matter because rare regimes (e.g. a 2-month crash) yield
    few, short segments and a correspondingly higher overfitting risk,
    which should be reported alongside any evaluation result, not hidden.
    """
    experts, segment_counts = {}, {}
    for bucket in ["calm", "stress"]:
        model, n_segments = train_regime_expert(
            train_df, bucket, feature_columns, seed=seed,
            total_timesteps=total_timesteps, min_segment_length=min_segment_length,
        )
        experts[bucket] = model
        segment_counts[bucket] = n_segments
    return {"experts": experts, "segment_counts": segment_counts}
