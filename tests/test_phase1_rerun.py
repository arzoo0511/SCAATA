"""Tests for the Phase 1 re-run harness: resumable per-(fold, seed) jobs,
a manifest that refuses to mix configs, fee-paying baselines, the env-fixes
arm, running a subset of a full run, and summaries recomputed from saved
curves. PPO training/backtesting are faked."""
import json

import numpy as np
import pandas as pd
import pytest

import scaata.evaluation.phase1_rerun as rerun
from scaata.config import STATIONARY_FEATURE_COLUMNS
from scaata.features.technical import add_features
from scaata.rl.env import POSITION_OBS_SIZE
from scaata.rl.train import buy_and_hold_equity

START, END = "2020-01-01", "2023-01-15"  # exactly two walk-forward folds


def _featured(tickers=("AAA", "BBB")):
    dates = pd.bdate_range(START, END)
    rng = np.random.default_rng(0)
    frames = []
    for ticker in tickers:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(dates)))
        frames.append(pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
            "Volume": 1e6 + rng.integers(0, 1000, len(dates)), "Ticker": ticker,
        }, index=dates))
    raw = pd.concat(frames)
    raw.index.name = "Date"
    return add_features(raw)


class _FakeModel:
    pass


@pytest.fixture
def fakes(monkeypatch):
    calls = {"train": 0, "train_kwargs": [], "backtest_kwargs": []}

    def _train(train_df, feature_columns, seed=0, total_timesteps=0, **kwargs):
        calls["train"] += 1
        calls["train_kwargs"].append({"feature_columns": list(feature_columns), **kwargs})
        return _FakeModel()

    def _backtest(model, test_df, feature_columns, ticker, **kwargs):
        calls["backtest_kwargs"].append(kwargs)
        n = int((test_df["Ticker"] == ticker).sum())
        rng = np.random.default_rng(n)
        return 10_000 * np.cumprod(np.r_[1.0, 1 + rng.normal(0.0002, 0.01, n - 1)]), np.zeros(n - 1, dtype=int)

    monkeypatch.setattr(rerun, "train_ppo", _train)
    monkeypatch.setattr(rerun, "backtest_ppo", _backtest)
    monkeypatch.setattr(rerun, "_git_state", lambda: {"commit": "test", "dirty": False})
    return calls


def _run(results_dir, featured, **kwargs):
    params = dict(tickers=["AAA", "BBB"], seeds=[0, 1], ppo_timesteps=10, results_dir=results_dir,
                  max_workers=1, featured_df=featured, start=START, end=END)
    params.update(kwargs)
    return rerun.run_rerun(**params)


def test_every_fold_and_seed_gets_its_own_saved_job(tmp_path, fakes):
    paths = _run(tmp_path, _featured())

    assert sorted(p.name for p in paths) == ["fold0_seed0.pkl", "fold0_seed1.pkl", "fold1_seed0.pkl", "fold1_seed1.pkl"]
    assert all(p.exists() for p in paths)
    assert fakes["train"] == 4
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["config"]["seeds"] == [0, 1]
    assert manifest["config"]["arm"] == "vanilla"
    assert manifest["git"]["commit"] == "test"


def test_rerun_resumes_without_retraining_finished_jobs(tmp_path, fakes):
    featured = _featured()
    _run(tmp_path, featured)
    _run(tmp_path, featured)

    assert fakes["train"] == 4


def test_a_subset_run_counts_toward_the_full_run(tmp_path, fakes):
    """Timing one job first must not waste it or fork the manifest."""
    featured = _featured()
    subset = _run(tmp_path, featured, only_folds=[0], only_seeds=[0])
    assert [p.name for p in subset] == ["fold0_seed0.pkl"]
    assert fakes["train"] == 1

    _run(tmp_path, featured)  # the full run, same config

    assert fakes["train"] == 4  # the subset job was not retrained
    assert rerun.summarize_rerun(tmp_path)["complete"] is True


def test_refuses_to_mix_results_from_a_different_config(tmp_path, fakes):
    featured = _featured()
    _run(tmp_path, featured)

    with pytest.raises(RuntimeError, match="different config"):
        _run(tmp_path, featured, fee=0.0)


def test_env_fixes_arm_trains_and_backtests_with_the_audit_options(tmp_path, fakes):
    _run(tmp_path, _featured(), arm="env_fixes", only_folds=[0], only_seeds=[0])

    kwargs = fakes["train_kwargs"][0]
    assert kwargs["feature_columns"] == list(STATIONARY_FEATURE_COLUMNS)
    assert kwargs["include_position_obs"] is True
    assert kwargs["random_episode_start_min_steps"] == 126
    assert all(k["include_position_obs"] is True for k in fakes["backtest_kwargs"])


def test_unknown_arm_is_rejected(tmp_path, fakes):
    with pytest.raises(ValueError, match="unknown arm"):
        _run(tmp_path, _featured(), arm="nope")


def test_baselines_in_saved_jobs_pay_the_fee(tmp_path, fakes):
    featured = _featured()
    paths = _run(tmp_path, featured, only_folds=[0], only_seeds=[0])

    payload = pd.read_pickle(paths[0])
    _, test_df = rerun.split_fold(featured, payload["fold"])
    fee_free = buy_and_hold_equity(test_df, "AAA")
    assert payload["curves"]["AAA"]["buy_and_hold"][-1] < fee_free[-1]
    assert payload["elapsed_seconds"] >= 0


def test_summary_is_recomputed_from_saved_curves(tmp_path, fakes):
    _run(tmp_path, _featured())

    summary = rerun.summarize_rerun(tmp_path)

    assert summary["jobs_found"] == summary["jobs_expected"] == 4
    assert summary["complete"] is True
    assert summary["rows"] == 8  # 2 folds x 2 seeds x 2 tickers
    assert set(summary["comparisons"]) == {"ppo_vs_buy_and_hold", "ppo_vs_rule_based", "ppo_vs_momentum_fallback"}
    assert summary["comparisons"]["ppo_vs_buy_and_hold"]["n_folds"] == 2
    assert summary["median_job_minutes"] is not None
    assert (tmp_path / "sharpe_table.csv").exists()
    assert (tmp_path / "summary.json").exists()


def test_compare_arms_matches_rows_both_arms_finished(tmp_path, fakes):
    featured = _featured()
    _run(tmp_path / "vanilla", featured)
    _run(tmp_path / "env_fixes", featured, arm="env_fixes", only_folds=[1])

    result = rerun.compare_arms(tmp_path / "vanilla", tmp_path / "env_fixes")

    assert result["matched_rows"] == 4  # fold 1 only: 2 seeds x 2 tickers
    assert result["n_folds"] == 1


def test_position_obs_size_is_what_the_env_arm_adds():
    assert POSITION_OBS_SIZE == 3


def test_india_market_uses_its_tickers_and_statutory_costs(tmp_path, fakes):
    from scaata.config import INDIA_SPREAD_COST, INDIA_STATUTORY_COST_PER_SIDE

    featured = _featured(tickers=("HDFCBANK.NS", "ITC.NS"))
    rerun.run_rerun(market="india", tickers=["HDFCBANK.NS", "ITC.NS"], seeds=[0], ppo_timesteps=10,
                    results_dir=tmp_path, max_workers=1, featured_df=featured, start=START, end=END,
                    only_folds=[0])

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["config"]["market"] == "india"
    assert manifest["config"]["fee"] == pytest.approx(INDIA_SPREAD_COST + INDIA_STATUTORY_COST_PER_SIDE)
    assert fakes["train_kwargs"][0]["fixed_fee"] == pytest.approx(INDIA_STATUTORY_COST_PER_SIDE)
    assert fakes["train_kwargs"][0]["transaction_fee"] == pytest.approx(INDIA_SPREAD_COST)
    assert all(k["fixed_fee"] == pytest.approx(INDIA_STATUTORY_COST_PER_SIDE) for k in fakes["backtest_kwargs"])


def test_unknown_market_is_rejected(tmp_path, fakes):
    with pytest.raises(ValueError, match="unknown market"):
        _run(tmp_path, _featured(), market="mars")
