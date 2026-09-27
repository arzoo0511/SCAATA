"""Trains (once) the India policy the paper book trades on, and loads it back.

Config is deliberately identical to the `env_fixes` arm of the walk-forward
evaluation (`scaata.evaluation.phase1_rerun.ARMS`): scale-free features,
position state in the observation, random episode starts, and real NSE costs
in the environment. Same thing being measured, same thing being traded --
otherwise the evaluation says nothing about what the book is doing.

Unlike the evaluation, this trains on ALL available history (there is no
holdout to protect: the paper book's out-of-sample data is the future,
arriving one day at a time). The training normalization statistics are saved
next to the policy, because live inference must scale features exactly the
way training did -- the audit found the old live path re-fitting them on the
last ~41 bars, which put the policy's inputs outside anything it trained on.

    python -m scaata.live.india_policy          # train if missing, then report
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from scaata.config import (
    INDIA_SPREAD_COST,
    INDIA_STATUTORY_COST_PER_SIDE,
    INDIA_TICKERS,
    PPO_TOTAL_TIMESTEPS,
    ROOT_DIR,
    START_DATE,
    STATIONARY_FEATURE_COLUMNS,
)
from scaata.features.normalize import fit_normalizer, load_norm_stats, save_norm_stats

POLICY_PATH = ROOT_DIR / "forward_test" / "india_policy.zip"
NORM_PATH = ROOT_DIR / "forward_test" / "india_policy_norm.json"
FEATURE_COLUMNS = list(STATIONARY_FEATURE_COLUMNS)
INCLUDE_POSITION_OBS = True
RANDOM_EPISODE_START_MIN_STEPS = 126


def train_india_policy(total_timesteps: int = PPO_TOTAL_TIMESTEPS, seed: int = 0, end: str | None = None) -> dict:
    """Trains on every available bar and saves the policy plus its
    normalization stats. Returns what it trained on."""
    from scaata.data.loaders import load_market_data
    from scaata.features.normalize import apply_normalizer
    from scaata.features.technical import add_features
    from scaata.rl.train import train_ppo

    end = end or date.today().isoformat()
    featured = add_features(load_market_data(list(INDIA_TICKERS), START_DATE, end, use_cache=True))
    mean, std = fit_normalizer(featured, FEATURE_COLUMNS)
    train_norm = apply_normalizer(featured, FEATURE_COLUMNS, mean, std)

    model = train_ppo(
        train_norm, FEATURE_COLUMNS, seed=seed, total_timesteps=total_timesteps,
        include_position_obs=INCLUDE_POSITION_OBS,
        random_episode_start_min_steps=RANDOM_EPISODE_START_MIN_STEPS,
        transaction_fee=INDIA_SPREAD_COST, fixed_fee=INDIA_STATUTORY_COST_PER_SIDE,
    )
    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(POLICY_PATH))
    save_norm_stats(NORM_PATH, mean, std, {
        "quality": "exact", "source": "india_policy", "trained_through": str(featured.index.max().date()),
        "train_rows": len(featured), "arm": "env_fixes", "seed": seed, "timesteps": total_timesteps,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return {"rows": len(featured), "trained_through": str(featured.index.max().date()),
            "tickers": list(INDIA_TICKERS), "timesteps": total_timesteps}


def load_india_policy():
    """(model, mean, std, provenance), or None if it hasn't been trained."""
    if not POLICY_PATH.exists():
        return None
    stats = load_norm_stats(NORM_PATH, FEATURE_COLUMNS)
    if stats is None:
        return None
    from sb3_contrib import RecurrentPPO

    mean, std, provenance = stats
    return RecurrentPPO.load(str(POLICY_PATH)), mean, std, provenance


if __name__ == "__main__":
    if POLICY_PATH.exists():
        print(f"{POLICY_PATH.name} already exists -- delete it to retrain.")
    else:
        print("Training the India policy on all available history; this takes ~20 minutes.", flush=True)
        print(train_india_policy())
