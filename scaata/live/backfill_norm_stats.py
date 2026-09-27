"""One-off backfill of training normalization stats for the policies that
were deployed before stats were saved alongside them.

Each entry rebuilds the exact training slice the policy saw (the retrain
rule in force when these were trained: cached raw data, `add_features`,
rows dated on or before `trained_through - HISTORICAL_HOLDOUT_DAYS`).
Where a holdout Sharpe for that training run was recorded, the policy is
re-backtested on the matching holdout using the rebuilt stats, and the file
is only written if it reproduces that Sharpe -- a deterministic backtest
matching to the recorded precision is proof the stats are the ones used.

Verified 2026-09-14:
- MSFT/META/BABA: `policy_*_meta.json` candidate_sharpe from the 2026-08-01
  retrain, reproduced to 6 decimals.
- AAPL/AMZN/GOOGL: the 2026-07-26 retrain summary ("AAPL: DEPLOYED --
  candidate Sharpe 2.226", AMZN -0.496, GOOGL 0.115), reproduced to 3.
- NVDA/XOM: trained in an unlogged 2026-07-25 batch, so nothing to verify
  against. Stats use the same rule with trained_through 2026-07-25; moving
  the cutoff anywhere from 2026-04-26 to 2026-07-19 shifts the latest row's
  z-scores by at most 0.16 (NVDA) / 0.12 (XOM). Marked "approximate".
- INTC: same unlogged batch, but the same cutoff range shifts its z-scores
  by up to 1.76 -- not reconstructable. Deliberately left without stats, so
  its live signal stays withheld until it is retrained.

Usage: python -m scaata.live.backfill_norm_stats [--force]
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import pandas as pd

from scaata.config import DATA_CACHE_DIR, FEATURE_COLUMNS
from scaata.evaluation.metrics import sharpe_ratio
from scaata.features.normalize import apply_normalizer, fit_normalizer, save_norm_stats
from scaata.features.technical import add_features
from scaata.live.forward_test import _model_path, _norm_stats_path

# The retrain holdout rule when these policies were trained (calendar days
# before trained_through). The current rule is RETRAIN_GATE_HOLDOUT_ROWS.
HISTORICAL_HOLDOUT_DAYS = 60

RECONSTRUCTIONS = {
    "AAPL": {"cache_end": "2026-07-26", "trained_through": "2026-07-26", "recorded_sharpe": 2.226, "decimals": 3},
    "AMZN": {"cache_end": "2026-07-19", "trained_through": "2026-07-26", "recorded_sharpe": -0.496, "decimals": 3},
    "GOOGL": {"cache_end": "2026-07-19", "trained_through": "2026-07-26", "recorded_sharpe": 0.115, "decimals": 3},
    "MSFT": {"cache_end": "2026-07-19", "trained_through": "2026-08-02", "recorded_sharpe": -0.577089, "decimals": 6},
    "META": {"cache_end": "2026-07-27", "trained_through": "2026-08-02", "recorded_sharpe": 1.254276, "decimals": 6},
    "BABA": {"cache_end": "2026-07-19", "trained_through": "2026-08-02", "recorded_sharpe": -0.891632, "decimals": 6},
    "NVDA": {"cache_end": "2026-07-19", "trained_through": "2026-07-25", "recorded_sharpe": None, "max_z_shift": 0.16},
    "XOM": {"cache_end": "2026-07-19", "trained_through": "2026-07-25", "recorded_sharpe": None, "max_z_shift": 0.12},
}


def rebuild_and_verify(ticker: str, spec: dict) -> dict:
    from sb3_contrib import RecurrentPPO
    from scaata.rl.train import backtest_ppo

    raw = pd.read_parquet(DATA_CACHE_DIR / f"raw_{ticker}_2020-01-01_{spec['cache_end']}.parquet")
    featured = add_features(raw)
    cutoff = pd.Timestamp(spec["trained_through"]) - pd.Timedelta(days=HISTORICAL_HOLDOUT_DAYS)
    train_part = featured[featured.index <= cutoff]
    holdout_part = featured[featured.index > cutoff]
    mean, std = fit_normalizer(train_part, FEATURE_COLUMNS)

    provenance = {
        "source": "backfill_norm_stats", "cache_end": spec["cache_end"],
        "trained_through": spec["trained_through"], "train_rows": len(train_part),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if spec["recorded_sharpe"] is None:
        provenance.update(quality="approximate", max_latest_row_z_shift=spec["max_z_shift"])
        return {"verified": True, "mean": mean, "std": std, "provenance": provenance}

    model = RecurrentPPO.load(str(_model_path(ticker)))
    equity, _ = backtest_ppo(model, apply_normalizer(holdout_part, FEATURE_COLUMNS, mean, std), FEATURE_COLUMNS, ticker)
    reproduced = sharpe_ratio(equity)
    verified = round(reproduced, spec["decimals"]) == spec["recorded_sharpe"]
    provenance.update(quality="exact", recorded_holdout_sharpe=spec["recorded_sharpe"], reproduced_holdout_sharpe=reproduced)
    return {"verified": verified, "mean": mean, "std": std, "provenance": provenance}


def main(force: bool = False) -> None:
    for ticker, spec in RECONSTRUCTIONS.items():
        path = _norm_stats_path(ticker)
        if path.exists() and not force:
            print(f"{ticker}: {path.name} already exists -- skipped (pass --force to overwrite)")
            continue
        result = rebuild_and_verify(ticker, spec)
        prov = result["provenance"]
        if not result["verified"]:
            print(f"{ticker}: NOT written -- reproduced Sharpe {prov['reproduced_holdout_sharpe']:.6f} "
                  f"does not match recorded {spec['recorded_sharpe']}")
            continue
        save_norm_stats(path, result["mean"], result["std"], prov)
        detail = (f"reproduced {prov['reproduced_holdout_sharpe']:.6f} = recorded {spec['recorded_sharpe']}"
                  if prov["quality"] == "exact" else f"max z shift {spec['max_z_shift']}")
        print(f"{ticker}: wrote {path.name} ({prov['quality']}; {detail})")
    print("INTC: no stats written by design -- retrain it to get a live signal.")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
