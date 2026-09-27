"""Scheduled retraining (Phase 18) -- closes a gap found during a status
review: the live daily signal (`scaata.live.daily_signal`) just calls
`.predict()` on a policy trained once and frozen forever ("does critique
learn for the future" was answered honestly as "no" -- nothing in the live
system ever re-trains on new data). This module is the fix: periodically
retrain each ticker's RL policy on fresh data, and only ever replace the
live policy if the fresh candidate is measurably better than what's
currently deployed, checked on a recent holdout window neither model was
allowed to train on.

Deliberately scoped to the SAME feature set the live policies already use
(`scaata.config.FEATURE_COLUMNS`) -- not the richer meta-confidence/
sentiment/novelty feature sets used elsewhere in this project's ablation
studies. Keeping the feature set unchanged means a deployed candidate is a
drop-in replacement: same observation shape, zero other code changes
required in `scaata.live.daily_signal`/`forward_test`.

The gate, and why it looks like this (rewritten 2026-09-14 after an audit):
- The holdout is the last `RETRAIN_GATE_HOLDOUT_ROWS` (~1 year) of the data
  actually fetched. It used to be the last 60 calendar days before *today*:
  ~39 rows, an annualized-Sharpe standard error of ~2.5, and -- once
  yfinance rate-limited and the job fell back to a weeks-old cache -- fewer
  than the minimum rows, so every run from August on was skipped.
- Candidate vs. incumbent and candidate vs. buy-and-hold are compared with a
  paired block bootstrap over the same holdout days
  (`scaata.evaluation.stats.paired_block_bootstrap_prob_better`), not by
  subtracting two point Sharpes with a 0.05 tolerance.
- The incumbent is scored on holdout features scaled with its own saved
  training stats, not the candidate's.
- A candidate that never trades on the holdout is never deployed, and
  nothing is deployed from data older than `RETRAIN_MAX_DATA_STALENESS_DAYS`.
- Training uses the reward config multi-seed validation tested
  (benchmark-relative DSR, `ent_coef=0.05`); retraining previously passed
  neither setting.

First-ever run for a ticker (no incumbent on disk) deploys if the candidate
trades and isn't confidently worse than buy-and-hold.
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO

from scaata.config import (
    ALL_TICKERS,
    DATA_CACHE_DIR,
    FEATURE_COLUMNS,
    PPO_TOTAL_TIMESTEPS,
    RETRAIN_DSR_BENCHMARK_RELATIVE,
    RETRAIN_ENT_COEF,
    RETRAIN_GATE_HOLDOUT_ROWS,
    RETRAIN_GATE_MIN_PROB_VS_BUY_AND_HOLD,
    RETRAIN_GATE_MIN_PROB_VS_INCUMBENT,
    RETRAIN_MAX_DATA_STALENESS_DAYS,
    RETRAIN_MIN_TRAIN_ROWS,
    START_DATE,
)
from scaata.agents.orchestrator import run_inner_loop
from scaata.data.loaders import DataDownloadError, collect_data
from scaata.evaluation.metrics import sharpe_ratio
from scaata.evaluation.stats import paired_block_bootstrap_prob_better
from scaata.features.normalize import apply_normalizer, load_norm_stats, normalize_data, save_norm_stats
from scaata.features.technical import add_features
from scaata.live.alpaca_broker import get_daily_bars_range
from scaata.live.forward_test import FORWARD_TEST_DIR, _model_path, _norm_stats_path, _state_path
from scaata.rl.train import backtest_ppo, buy_and_hold_equity, train_ppo

ARCHIVE_DIR = FORWARD_TEST_DIR / "retrain_archive"
ARCHIVE_DIR.mkdir(exist_ok=True)


def _metadata_path(ticker: str) -> Path:
    return FORWARD_TEST_DIR / f"policy_{ticker}_meta.json"


def load_retrain_metadata(ticker: str) -> dict | None:
    path = _metadata_path(ticker)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_metadata(ticker: str, metadata: dict) -> None:
    with open(_metadata_path(ticker), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _most_recent_cached_parquet(ticker: str) -> Path | None:
    """Finds the most recently-dated cached raw parquet for `ticker`,
    regardless of the exact (start, end) it was cached under -- last-resort
    fallback only. Cache filenames are `raw_{ticker}_{start}_{end}.parquet`;
    sorting by the trailing end-date token picks the freshest one available.
    """
    candidates = list(DATA_CACHE_DIR.glob(f"raw_{ticker}_*.parquet"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stem.split("_")[-1])


def _fetch_training_data(ticker: str, end: str) -> tuple:
    """yfinance first (what every backtest used); then Alpaca's adjusted SIP
    bars, the same kind of series live inference uses (yfinance
    rate-limiting took down every ticker on this job's first scheduled run);
    then the freshest cached parquet, however stale. Returns
    (raw_df, source): "live", "alpaca_sip", or "stale_cache:{end_date}".
    Raises `DataDownloadError` only when all three come up empty.
    """
    try:
        return collect_data([ticker], START_DATE, end), "live"
    except DataDownloadError:
        pass

    try:
        bars, source = get_daily_bars_range(ticker, START_DATE, end)
        if bars is not None and not bars.empty:
            return bars, source
    except Exception as e:
        print(f"{ticker}: Alpaca fallback failed ({e})")

    cached_path = _most_recent_cached_parquet(ticker)
    if cached_path is None:
        raise DataDownloadError(f"{ticker}: yfinance and Alpaca both failed and no cached parquet exists")
    cached_end = cached_path.stem.split("_")[-1]
    return pd.read_parquet(cached_path), f"stale_cache:{cached_end}"


def _archive_incumbent(ticker: str) -> None:
    """Moves (not deletes) the currently-deployed policy, LSTM state and
    normalization stats aside before a candidate replaces them."""
    model_path = _model_path(ticker)
    if not model_path.exists():
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.move(str(model_path), str(ARCHIVE_DIR / f"policy_{ticker}_{ts}.zip"))
    state_path = _state_path(ticker)
    if state_path.exists():
        shutil.move(str(state_path), str(ARCHIVE_DIR / f"lstm_state_{ticker}_{ts}.pkl"))
    norm_path = _norm_stats_path(ticker)
    if norm_path.exists():
        shutil.move(str(norm_path), str(ARCHIVE_DIR / f"policy_{ticker}_norm_{ts}.json"))


def _build_bc_model(train_df: pd.DataFrame, feature_columns: list[str]):
    """Runs the full inner-loop graph to get a Hedge-weighted
    behavior-cloning model for warm-starting the policy. Returns `None`
    (rather than raising) on any failure, so a warm-start problem degrades
    to "train from scratch," never blocks a retrain outright.
    """
    try:
        final_state = run_inner_loop(train_df, feature_columns=feature_columns)
        return final_state.get("bc_model")
    except Exception as e:
        print(f"bc_model warm-start unavailable this run ({e}) -- training from scratch instead.")
        return None


def _never_trades(equity: np.ndarray) -> bool:
    return bool(np.allclose(np.diff(np.asarray(equity, dtype=float)), 0.0))


def retrain_ticker(
    ticker: str,
    feature_columns: list[str] = FEATURE_COLUMNS,
    total_timesteps: int = PPO_TOTAL_TIMESTEPS,
    holdout_rows: int = RETRAIN_GATE_HOLDOUT_ROWS,
    min_prob_vs_incumbent: float = RETRAIN_GATE_MIN_PROB_VS_INCUMBENT,
    min_prob_vs_buy_and_hold: float = RETRAIN_GATE_MIN_PROB_VS_BUY_AND_HOLD,
    use_differential_sharpe: bool = True,
    dsr_benchmark_relative: bool = RETRAIN_DSR_BENCHMARK_RELATIVE,
    ent_coef: float = RETRAIN_ENT_COEF,
    use_bc_warm_start: bool = False,
    today: date | None = None,
) -> dict:
    """Retrains one ticker's policy, gates the candidate against the current
    incumbent and buy-and-hold on a holdout the candidate never trained on,
    and deploys only if the gate passes. Always returns a full result dict
    describing what happened.

    `use_bc_warm_start` defaults to False: multi-seed-style checks found it
    made Sharpe worse (MSFT 0.36 -> -0.14). Still available opt-in.
    """
    today = today or date.today()
    raw, data_source = _fetch_training_data(ticker, today.isoformat())
    featured = add_features(raw).sort_index()

    if len(featured) < holdout_rows + RETRAIN_MIN_TRAIN_ROWS:
        return {
            "ticker": ticker, "status": "skipped", "deployed": False, "data_source": data_source,
            "reason": (f"only {len(featured)} usable rows; need {holdout_rows} holdout + "
                       f"{RETRAIN_MIN_TRAIN_ROWS} training"),
        }

    train_part = featured.iloc[:-holdout_rows]
    holdout_part = featured.iloc[-holdout_rows:]
    data_end = pd.Timestamp(featured.index.max()).normalize()
    staleness_days = int((pd.Timestamp(today) - data_end).days)

    train_norm, holdout_norm, norm_mean, norm_std = normalize_data(train_part, holdout_part, feature_columns)

    bc_model = _build_bc_model(train_norm, feature_columns) if use_bc_warm_start else None

    candidate = train_ppo(
        train_norm, feature_columns, total_timesteps=total_timesteps,
        use_differential_sharpe=use_differential_sharpe, dsr_benchmark_relative=dsr_benchmark_relative,
        ent_coef=ent_coef, bc_model=bc_model,
    )
    candidate_equity, _ = backtest_ppo(candidate, holdout_norm, feature_columns, ticker)
    candidate_sharpe = sharpe_ratio(candidate_equity)
    candidate_collapsed = _never_trades(candidate_equity)

    bah_equity = buy_and_hold_equity(holdout_part, ticker)
    bah_sharpe = sharpe_ratio(bah_equity)
    prob_vs_bah = paired_block_bootstrap_prob_better(candidate_equity, bah_equity)

    incumbent_sharpe = None
    prob_vs_incumbent = None
    incumbent_norm_source = None
    if _model_path(ticker).exists():
        incumbent = RecurrentPPO.load(str(_model_path(ticker)))
        # Score the incumbent on inputs scaled the way IT was trained, not
        # with the candidate's stats.
        incumbent_stats = load_norm_stats(_norm_stats_path(ticker), feature_columns)
        if incumbent_stats is not None:
            inc_mean, inc_std, _ = incumbent_stats
            incumbent_holdout = apply_normalizer(holdout_part, feature_columns, inc_mean, inc_std)
            incumbent_norm_source = "own_training_stats"
        else:
            incumbent_holdout = holdout_norm
            incumbent_norm_source = "candidate_stats_fallback"
        incumbent_equity, _ = backtest_ppo(incumbent, incumbent_holdout, feature_columns, ticker)
        incumbent_sharpe = sharpe_ratio(incumbent_equity)
        prob_vs_incumbent = paired_block_bootstrap_prob_better(candidate_equity, incumbent_equity)

    deploy = False
    if candidate_collapsed:
        reason = "candidate never traded on the holdout (collapsed policy) -- not deploying"
    elif staleness_days > RETRAIN_MAX_DATA_STALENESS_DAYS:
        reason = (f"newest bar is {staleness_days} days old ({data_source}) -- evaluated only, "
                  "not deploying a policy trained on stale data")
    elif prob_vs_incumbent is not None and not prob_vs_incumbent >= min_prob_vs_incumbent:
        reason = (f"candidate beats the incumbent with probability {prob_vs_incumbent:.2f} "
                  f"(needs {min_prob_vs_incumbent}) -- keeping the incumbent live")
    elif not prob_vs_bah >= min_prob_vs_buy_and_hold:
        reason = (f"candidate is confidently worse than buy-and-hold (beats it with probability "
                  f"{prob_vs_bah:.2f}, needs {min_prob_vs_buy_and_hold}) -- not deploying")
    else:
        deploy = True
        if prob_vs_incumbent is None:
            reason = "no incumbent on disk -- first deploy for this ticker"
        else:
            reason = f"candidate beats the incumbent with probability {prob_vs_incumbent:.2f} -- deploying"

    trained_through = pd.Timestamp(train_part.index.max()).date().isoformat()
    reward_config = {
        "use_differential_sharpe": use_differential_sharpe,
        "dsr_benchmark_relative": dsr_benchmark_relative,
        "ent_coef": ent_coef,
    }
    result = {
        "ticker": ticker, "status": "evaluated", "deployed": False,
        "candidate_sharpe": candidate_sharpe, "incumbent_sharpe": incumbent_sharpe,
        "buy_and_hold_sharpe": bah_sharpe,
        "prob_beats_incumbent": prob_vs_incumbent, "prob_beats_buy_and_hold": prob_vs_bah,
        "candidate_collapsed": candidate_collapsed,
        "holdout_rows": len(holdout_part), "train_rows": len(train_part),
        "trained_through": trained_through, "data_end": data_end.date().isoformat(),
        "staleness_days": staleness_days, "reason": reason,
        "data_source": data_source, "bc_warm_start_used": bc_model is not None,
        "incumbent_norm_source": incumbent_norm_source, "reward_config": reward_config,
    }

    if deploy:
        _archive_incumbent(ticker)
        candidate.save(str(_model_path(ticker)))
        save_norm_stats(_norm_stats_path(ticker), norm_mean, norm_std, {
            "quality": "exact", "source": "retrain", "trained_through": trained_through,
            "train_rows": len(train_part), "data_source": data_source,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        state_path = _state_path(ticker)
        if state_path.exists():
            state_path.unlink()  # a new policy shouldn't reuse the old one's recurrent hidden state
        result["deployed"] = True

    _save_metadata(ticker, {
        "retrained_at_utc": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in result.items() if k not in ("ticker", "status")},
    })
    return result


def run_scheduled_retrain(tickers: list[str] = ALL_TICKERS) -> list[dict]:
    """Retrains every ticker, one at a time. A single ticker's failure must
    not stop the rest of the run, same convention as
    `scaata.live.run_all_daily_signals`."""
    results = []
    for ticker in tickers:
        try:
            results.append(retrain_ticker(ticker))
        except Exception as e:
            results.append({
                "ticker": ticker, "status": "error", "deployed": False, "reason": str(e),
            })
    return results
