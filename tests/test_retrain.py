"""Tests for scaata.rl.retrain -- the scheduled-retraining safety gate.
Everything expensive (data download, actual PPO training/backtesting) is
mocked; these tests are about the gate LOGIC: deploy on first run, deploy
a clearly better candidate, keep the incumbent when the difference is
noise, refuse collapsed or stale candidates, skip cleanly when there isn't
enough data, and never lose the incumbent's files when a deploy is rejected.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scaata.rl.retrain as retrain_module
from scaata.config import RETRAIN_GATE_HOLDOUT_ROWS
from scaata.data.loaders import DataDownloadError
from scaata.rl.retrain import (
    _build_bc_model,
    _fetch_training_data,
    _most_recent_cached_parquet,
    load_retrain_metadata,
    retrain_ticker,
    run_scheduled_retrain,
)


class _FakeModel:
    def __init__(self, name: str):
        self.name = name

    def save(self, path):
        Path(path).write_text(self.name)


class _FakeRecurrentPPO:
    @staticmethod
    def load(path):
        return _FakeModel(f"incumbent-loaded-from:{path}")


def _make_raw_df(ticker="AAPL", n=600, end=None):
    # Anchored to "today" by default so the data is fresh (not stale) no
    # matter what day the suite runs.
    end = end if end is not None else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n)
    close = 100 + np.cumsum(np.full(n, 0.05))
    df = pd.DataFrame({
        "Close": close, "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Volume": 1_000_000.0, "Ticker": ticker,
    }, index=dates)
    df.index.name = "Date"
    return df


def _identity_add_features(df):
    # Substitutes for the real add_features -- attaches placeholder feature
    # columns so RobustTradingEnv-shaped code downstream never sees a
    # missing-column KeyError, without re-testing indicator math here.
    out = df.copy()
    for col in ["returns", "ma_10", "ma_50", "volatility", "momentum", "rsi"]:
        out[col] = 0.0
    out["volume_ma_30"] = out["Volume"]
    return out


def _patch_common(monkeypatch, tmp_path, ticker="AAPL", n=600):
    forward_test_dir = tmp_path / "forward_test"
    forward_test_dir.mkdir()
    archive_dir = forward_test_dir / "retrain_archive"
    archive_dir.mkdir()

    monkeypatch.setattr(retrain_module, "FORWARD_TEST_DIR", forward_test_dir)
    monkeypatch.setattr(retrain_module, "ARCHIVE_DIR", archive_dir)
    monkeypatch.setattr(retrain_module, "_model_path", lambda t: forward_test_dir / f"policy_{t}.zip")
    monkeypatch.setattr(retrain_module, "_state_path", lambda t: forward_test_dir / f"lstm_state_{t}.pkl")
    monkeypatch.setattr(retrain_module, "_metadata_path", lambda t: forward_test_dir / f"policy_{t}_meta.json")
    monkeypatch.setattr(retrain_module, "_norm_stats_path", lambda t: forward_test_dir / f"policy_{t}_norm.json")

    monkeypatch.setattr(retrain_module, "collect_data", lambda symbols, start, end: _make_raw_df(ticker, n))
    # Never reach the real Alpaca API from tests, even with keys in .env.
    monkeypatch.setattr(retrain_module, "get_daily_bars_range", lambda symbol, start, end: (None, "unavailable"))
    monkeypatch.setattr(retrain_module, "add_features", _identity_add_features)
    monkeypatch.setattr(retrain_module, "RecurrentPPO", _FakeRecurrentPPO)
    # Default: no bc_model available -- run_inner_loop itself would
    # otherwise do real scraping/LLM/training work inside these tests.
    monkeypatch.setattr(retrain_module, "run_inner_loop", lambda train_df, feature_columns: {"bc_model": None})

    return forward_test_dir


def _patch_equity(monkeypatch, candidate_sharpe_curve, incumbent_sharpe_curve=None, bah_curve=None):
    monkeypatch.setattr(retrain_module, "train_ppo", lambda *a, **k: _FakeModel("candidate"))

    def _fake_backtest(model, df, feature_columns, ticker):
        if model.name == "candidate":
            return candidate_sharpe_curve, None
        return incumbent_sharpe_curve, None

    monkeypatch.setattr(retrain_module, "backtest_ppo", _fake_backtest)
    monkeypatch.setattr(retrain_module, "buy_and_hold_equity", lambda df, ticker: bah_curve)


def _curve(drift, seed):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(np.r_[1.0, 1 + rng.normal(drift, 0.01, RETRAIN_GATE_HOLDOUT_ROWS - 1)])


_RISING = _curve(0.003, seed=1)
_FALLING = _curve(-0.003, seed=2)
_FLAT = np.full(RETRAIN_GATE_HOLDOUT_ROWS, 100.0)


def test_retrain_uses_the_validated_reward_config_by_default(monkeypatch, tmp_path):
    """Regression test for two real misses: retrain_ticker first omitted
    use_differential_sharpe, then omitted dsr_benchmark_relative/ent_coef --
    so deployed policies were trained on a config multi-seed validation had
    rejected (all 9 deployed zips had ent_coef 0.01)."""
    _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    captured = {}
    monkeypatch.setattr(
        retrain_module, "train_ppo",
        lambda *a, **k: captured.update(k) or _FakeModel("candidate"),
    )

    result = retrain_ticker("AAPL")

    assert captured.get("use_differential_sharpe") is True
    assert captured.get("dsr_benchmark_relative") is True
    assert captured.get("ent_coef") == 0.05
    assert result["reward_config"] == {"use_differential_sharpe": True, "dsr_benchmark_relative": True, "ent_coef": 0.05}


def test_retrain_use_differential_sharpe_can_be_overridden(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    captured = {}
    monkeypatch.setattr(
        retrain_module, "train_ppo",
        lambda *a, **k: captured.update(k) or _FakeModel("candidate"),
    )

    retrain_ticker("AAPL", use_differential_sharpe=False)

    assert captured.get("use_differential_sharpe") is False


# --- BC warm-start wiring ---

def test_build_bc_model_returns_run_inner_loops_bc_model(monkeypatch):
    sentinel_bc_model = object()
    monkeypatch.setattr(retrain_module, "run_inner_loop", lambda train_df, feature_columns: {"bc_model": sentinel_bc_model})

    result = _build_bc_model(pd.DataFrame(), ["f1"])

    assert result is sentinel_bc_model


def test_build_bc_model_returns_none_on_failure_instead_of_raising(monkeypatch, capsys):
    def _fail(train_df, feature_columns):
        raise RuntimeError("simulated graph failure")

    monkeypatch.setattr(retrain_module, "run_inner_loop", _fail)

    result = _build_bc_model(pd.DataFrame(), ["f1"])

    assert result is None
    assert "warm-start unavailable" in capsys.readouterr().out


def test_retrain_ticker_does_not_warm_start_by_default(monkeypatch, tmp_path):
    """`use_bc_warm_start` defaults to False: a real single-seed MSFT test
    found warm-start made Sharpe worse (0.36 -> -0.14)."""
    _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)
    monkeypatch.setattr(retrain_module, "run_inner_loop", lambda train_df, feature_columns: {"bc_model": object()})

    captured = {}
    monkeypatch.setattr(
        retrain_module, "train_ppo",
        lambda *a, **k: captured.update(k) or _FakeModel("candidate"),
    )

    result = retrain_ticker("AAPL")

    assert captured.get("bc_model") is None
    assert result["bc_warm_start_used"] is False


def test_retrain_ticker_passes_bc_model_to_train_ppo_when_enabled(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)
    sentinel_bc_model = object()
    monkeypatch.setattr(retrain_module, "run_inner_loop", lambda train_df, feature_columns: {"bc_model": sentinel_bc_model})

    captured = {}
    monkeypatch.setattr(
        retrain_module, "train_ppo",
        lambda *a, **k: captured.update(k) or _FakeModel("candidate"),
    )

    result = retrain_ticker("AAPL", use_bc_warm_start=True)

    assert captured.get("bc_model") is sentinel_bc_model
    assert result["bc_warm_start_used"] is True


def test_retrain_ticker_still_trains_when_bc_model_unavailable(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)
    monkeypatch.setattr(retrain_module, "run_inner_loop", lambda train_df, feature_columns: (_ for _ in ()).throw(RuntimeError("no GITHUB_PAT")))

    result = retrain_ticker("AAPL", use_bc_warm_start=True)

    assert result["status"] == "evaluated"
    assert result["bc_warm_start_used"] is False


# --- the gate ---

def test_first_ever_run_deploys_a_trading_candidate(monkeypatch, tmp_path):
    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    result = retrain_ticker("AAPL")

    assert result["status"] == "evaluated"
    assert result["deployed"] is True
    assert result["incumbent_sharpe"] is None
    assert "first deploy" in result["reason"]
    assert (forward_test_dir / "policy_AAPL.zip").exists()


def test_holdout_is_the_last_year_of_rows(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, n=600)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    result = retrain_ticker("AAPL")

    assert result["holdout_rows"] == RETRAIN_GATE_HOLDOUT_ROWS
    assert result["train_rows"] == 600 - RETRAIN_GATE_HOLDOUT_ROWS


def test_clearly_better_candidate_deploys_and_archives_incumbent(monkeypatch, tmp_path):
    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    (forward_test_dir / "policy_AAPL.zip").write_text("old-incumbent")
    (forward_test_dir / "lstm_state_AAPL.pkl").write_text("old-state")

    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, incumbent_sharpe_curve=_FALLING, bah_curve=_FLAT)

    result = retrain_ticker("AAPL")

    assert result["deployed"] is True
    assert result["prob_beats_incumbent"] > 0.95
    assert (forward_test_dir / "policy_AAPL.zip").read_text() == "candidate"
    assert not (forward_test_dir / "lstm_state_AAPL.pkl").exists()
    archived = list((forward_test_dir / "retrain_archive").glob("policy_AAPL_*.zip"))
    assert len(archived) == 1
    assert archived[0].read_text() == "old-incumbent"


def test_noise_level_difference_keeps_the_incumbent(monkeypatch, tmp_path):
    """The old gate replaced the incumbent whenever the candidate's point
    Sharpe on ~39 days was within 0.05 -- pure noise decided deployments.
    A candidate that isn't better with real probability keeps the incumbent."""
    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    (forward_test_dir / "policy_AAPL.zip").write_text("old-incumbent")
    same = _curve(0.001, seed=7)
    _patch_equity(monkeypatch, candidate_sharpe_curve=same, incumbent_sharpe_curve=same.copy(), bah_curve=_FLAT)

    result = retrain_ticker("AAPL")

    assert result["deployed"] is False
    assert "keeping the incumbent" in result["reason"]
    assert (forward_test_dir / "policy_AAPL.zip").read_text() == "old-incumbent"


def test_deploys_over_a_collapsed_flat_incumbent(monkeypatch, tmp_path):
    """A flat incumbent (never trades) counts as cash, not NaN -- otherwise
    a collapsed policy could never be replaced."""
    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    (forward_test_dir / "policy_AAPL.zip").write_text("old-incumbent")

    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, incumbent_sharpe_curve=_FLAT, bah_curve=_FLAT)

    result = retrain_ticker("AAPL")

    assert result["deployed"] is True
    assert np.isnan(result["incumbent_sharpe"])
    assert result["prob_beats_incumbent"] > 0.95


def test_collapsed_candidate_is_never_deployed(monkeypatch, tmp_path):
    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_FLAT, bah_curve=_FALLING)

    result = retrain_ticker("AAPL")

    assert result["deployed"] is False
    assert result["candidate_collapsed"] is True
    assert "collapsed" in result["reason"]
    assert not (forward_test_dir / "policy_AAPL.zip").exists()


def test_candidate_confidently_worse_than_buy_and_hold_is_not_deployed(monkeypatch, tmp_path):
    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_FALLING, bah_curve=_RISING)

    result = retrain_ticker("AAPL")

    assert result["deployed"] is False
    assert "worse than buy-and-hold" in result["reason"]
    assert not (forward_test_dir / "policy_AAPL.zip").exists()


def test_worse_candidate_is_rejected_and_incumbent_untouched(monkeypatch, tmp_path):
    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    (forward_test_dir / "policy_AAPL.zip").write_text("old-incumbent")

    _patch_equity(monkeypatch, candidate_sharpe_curve=_FALLING, incumbent_sharpe_curve=_RISING, bah_curve=_FLAT)

    result = retrain_ticker("AAPL")

    assert result["deployed"] is False
    assert "keeping the incumbent live" in result["reason"]
    assert (forward_test_dir / "policy_AAPL.zip").read_text() == "old-incumbent"
    assert list((forward_test_dir / "retrain_archive").glob("*")) == []


def test_stale_data_still_gets_a_full_holdout_but_is_never_deployed(monkeypatch, tmp_path):
    """Regression test for the six-week outage: with yfinance rate-limited
    the job fell back to a cache ending mid-July, the holdout was anchored
    to today's date, and so it had too few rows -- every ticker SKIPPED.
    The holdout must come from the end of the data, and stale data must be
    reported, not deployed."""
    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    stale_end = pd.Timestamp("2026-07-17")
    monkeypatch.setattr(retrain_module, "collect_data", lambda symbols, start, end: _make_raw_df("AAPL", 600, end=stale_end))
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    result = retrain_ticker("AAPL", today=date(2026, 9, 14))

    assert result["status"] == "evaluated"
    assert result["holdout_rows"] == RETRAIN_GATE_HOLDOUT_ROWS
    assert result["staleness_days"] == 59
    assert result["deployed"] is False
    assert "days old" in result["reason"]
    assert not (forward_test_dir / "policy_AAPL.zip").exists()


def test_skips_cleanly_when_not_enough_data(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, n=RETRAIN_GATE_HOLDOUT_ROWS + 10)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    result = retrain_ticker("AAPL")

    assert result["status"] == "skipped"
    assert result["deployed"] is False


def test_metadata_is_saved_after_evaluation(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    retrain_ticker("AAPL")

    meta = load_retrain_metadata("AAPL")
    assert meta is not None
    assert meta["deployed"] is True
    assert "retrained_at_utc" in meta
    assert meta["prob_beats_buy_and_hold"] > 0.95
    assert meta["reward_config"]["ent_coef"] == 0.05


# --- normalization stats ---

def test_deploy_saves_the_candidates_training_norm_stats(monkeypatch, tmp_path):
    from scaata.config import FEATURE_COLUMNS
    from scaata.features.normalize import load_norm_stats

    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    result = retrain_ticker("AAPL")

    assert result["deployed"] is True
    stats = load_norm_stats(forward_test_dir / "policy_AAPL_norm.json", FEATURE_COLUMNS)
    assert stats is not None
    assert stats[2]["quality"] == "exact"
    assert stats[2]["source"] == "retrain"
    assert stats[2]["trained_through"] == result["trained_through"]


def test_rejected_candidate_does_not_touch_the_incumbents_norm_stats(monkeypatch, tmp_path):
    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    (forward_test_dir / "policy_AAPL.zip").write_text("old-incumbent")
    (forward_test_dir / "policy_AAPL_norm.json").write_text("incumbent-stats")
    _patch_equity(monkeypatch, candidate_sharpe_curve=_FALLING, incumbent_sharpe_curve=_RISING, bah_curve=_FLAT)
    monkeypatch.setattr(retrain_module, "load_norm_stats", lambda path, cols: None)

    result = retrain_ticker("AAPL")

    assert result["deployed"] is False
    assert (forward_test_dir / "policy_AAPL_norm.json").read_text() == "incumbent-stats"


def test_incumbent_is_scored_with_its_own_training_stats(monkeypatch, tmp_path):
    """The incumbent was trained on inputs scaled with ITS stats; scoring it
    on data scaled with the candidate's stats is the same live skew, just
    inside the gate."""
    from scaata.config import FEATURE_COLUMNS
    from scaata.features.normalize import save_norm_stats

    forward_test_dir = _patch_common(monkeypatch, tmp_path)
    (forward_test_dir / "policy_AAPL.zip").write_text("old-incumbent")
    incumbent_mean = pd.Series(7.0, index=FEATURE_COLUMNS)
    incumbent_std = pd.Series(2.0, index=FEATURE_COLUMNS)
    save_norm_stats(forward_test_dir / "policy_AAPL_norm.json", incumbent_mean, incumbent_std, {"quality": "exact"})

    monkeypatch.setattr(retrain_module, "train_ppo", lambda *a, **k: _FakeModel("candidate"))
    monkeypatch.setattr(retrain_module, "buy_and_hold_equity", lambda df, ticker: _FLAT)
    seen = {}

    def _capturing_backtest(model, df, feature_columns, ticker):
        seen["candidate" if model.name == "candidate" else "incumbent"] = df.copy()
        return (_RISING if model.name == "candidate" else _FALLING), None

    monkeypatch.setattr(retrain_module, "backtest_ppo", _capturing_backtest)

    result = retrain_ticker("AAPL")

    raw_volume = seen["incumbent"]["Volume"].iloc[0]
    assert seen["incumbent"]["volume_ma_30"].iloc[0] == pytest.approx((raw_volume - 7.0) / 2.0)
    assert not seen["incumbent"]["volume_ma_30"].equals(seen["candidate"]["volume_ma_30"])
    assert result["incumbent_norm_source"] == "own_training_stats"
    # deploy archived the incumbent's stats alongside its policy
    assert list((forward_test_dir / "retrain_archive").glob("policy_AAPL_norm_*.json"))


# --- data-source fallback (the real 9/9 yfinance-rate-limit failure) ---

def test_most_recent_cached_parquet_picks_the_latest_end_date(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain_module, "DATA_CACHE_DIR", tmp_path)
    (tmp_path / "raw_AAPL_2020-01-01_2026-07-05.parquet").write_text("older")
    (tmp_path / "raw_AAPL_2020-01-01_2026-07-19.parquet").write_text("newer")
    (tmp_path / "raw_MSFT_2020-01-01_2026-07-19.parquet").write_text("different ticker")

    result = _most_recent_cached_parquet("AAPL")

    assert result.name == "raw_AAPL_2020-01-01_2026-07-19.parquet"


def test_most_recent_cached_parquet_returns_none_when_nothing_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain_module, "DATA_CACHE_DIR", tmp_path)
    assert _most_recent_cached_parquet("AAPL") is None


def _fail_yfinance(symbols, start, end):
    raise DataDownloadError("simulated yfinance rate limit")


def test_fetch_training_data_uses_alpaca_when_yfinance_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain_module, "DATA_CACHE_DIR", tmp_path)
    monkeypatch.setattr(retrain_module, "collect_data", _fail_yfinance)
    alpaca_df = pd.DataFrame({"Close": [300.0], "Ticker": ["AAPL"]})
    monkeypatch.setattr(retrain_module, "get_daily_bars_range", lambda symbol, start, end: (alpaca_df, "alpaca_sip"))

    df, source = _fetch_training_data("AAPL", "2026-09-14")

    assert source == "alpaca_sip"
    assert df is alpaca_df


def test_fetch_training_data_falls_back_to_stale_cache_when_live_sources_fail(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain_module, "DATA_CACHE_DIR", tmp_path)
    cached_df = pd.DataFrame({"Close": [100.0, 101.0], "Ticker": ["AAPL", "AAPL"]})
    cached_df.to_parquet(tmp_path / "raw_AAPL_2020-01-01_2026-07-19.parquet")
    monkeypatch.setattr(retrain_module, "collect_data", _fail_yfinance)
    monkeypatch.setattr(retrain_module, "get_daily_bars_range", lambda symbol, start, end: (None, "unavailable"))

    df, source = _fetch_training_data("AAPL", "2026-07-26")

    assert source == "stale_cache:2026-07-19"
    assert len(df) == 2


def test_fetch_training_data_raises_when_no_source_or_cache_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain_module, "DATA_CACHE_DIR", tmp_path)
    monkeypatch.setattr(retrain_module, "collect_data", _fail_yfinance)

    def _alpaca_down(symbol, start, end):
        raise RuntimeError("simulated Alpaca outage")

    monkeypatch.setattr(retrain_module, "get_daily_bars_range", _alpaca_down)

    with pytest.raises(DataDownloadError):
        _fetch_training_data("AAPL", "2026-07-26")


def test_fetch_training_data_prefers_yfinance_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(retrain_module, "DATA_CACHE_DIR", tmp_path)
    live_df = pd.DataFrame({"Close": [200.0], "Ticker": ["AAPL"]})
    monkeypatch.setattr(retrain_module, "collect_data", lambda symbols, start, end: live_df)

    df, source = _fetch_training_data("AAPL", "2026-07-26")

    assert source == "live"
    assert df is live_df


def test_run_scheduled_retrain_loops_every_ticker(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    results = run_scheduled_retrain(tickers=["AAPL", "MSFT"])

    assert [r["ticker"] for r in results] == ["AAPL", "MSFT"]
    assert all(r["deployed"] for r in results)


def test_one_tickers_failure_does_not_stop_the_rest(monkeypatch, tmp_path):
    """Regression test caught live: a real yfinance rate-limit made
    collect_data raise for AAPL. Without per-ticker isolation, that single
    exception would have killed the whole weekly job."""
    _patch_common(monkeypatch, tmp_path)
    _patch_equity(monkeypatch, candidate_sharpe_curve=_RISING, bah_curve=_FLAT)

    real_collect_data = retrain_module.collect_data

    def _flaky_collect_data(symbols, start, end):
        if symbols == ["AAPL"]:
            raise RuntimeError("simulated yfinance rate limit")
        return real_collect_data(symbols, start, end)

    monkeypatch.setattr(retrain_module, "collect_data", _flaky_collect_data)

    results = run_scheduled_retrain(tickers=["AAPL", "MSFT"])

    assert [r["ticker"] for r in results] == ["AAPL", "MSFT"]
    assert results[0]["status"] == "error"
    assert "simulated yfinance rate limit" in results[0]["reason"]
    assert results[1]["status"] == "evaluated"
    assert results[1]["deployed"] is True
