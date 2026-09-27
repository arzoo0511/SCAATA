"""Integration-plumbing tests for the Phase 9 sentiment ablation wiring in
`scaata.evaluation.ablations`: a config with `use_sentiment_feature=True`
must (a) refuse to run without sentiment data rather than silently ignoring
it, and (b) actually merge `sentiment_score` into the observation's feature
columns. Real PPO training is stubbed out with lightweight fakes so this
stays a fast unit test of the wiring, not a real training run.
"""
import numpy as np
import pandas as pd
import pytest

from scaata.config import NOVELTY_SCORE_COLUMN, SENTIMENT_SCORE_COLUMN
from scaata.evaluation import ablations
from scaata.evaluation.ablations import ABLATION_CONFIGS, AblationConfig, run_ablation_config


class _FakeEnv:
    captured_feature_columns = None
    captured_use_differential_sharpe = None
    captured_dsr_benchmark_relative = None

    def __init__(
        self, df, feature_columns, fixed_ticker=None, enable_self_critique=False,
        use_differential_sharpe=False, dsr_eta=None, dsr_reward_scale=None,
        dsr_benchmark_relative=False,
    ):
        _FakeEnv.captured_feature_columns = feature_columns
        _FakeEnv.captured_use_differential_sharpe = use_differential_sharpe
        _FakeEnv.captured_dsr_benchmark_relative = dsr_benchmark_relative
        self.df = df
        self.feature_columns = feature_columns
        self.initial_cash = 10_000.0

    def reset(self, seed=None):
        return np.zeros(len(self.feature_columns)), {}

    def step(self, action):
        return np.zeros(len(self.feature_columns)), 0.0, True, False, {"portfolio_value": self.initial_cash}


class _FakeModel:
    def __init__(self, *args, **kwargs):
        self.policy = None

    def learn(self, total_timesteps):
        pass

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        return np.array([0]), None


def _make_df(n=60, ticker="TEST"):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        "Close": close, "returns": rng.normal(0, 0.01, n), "ma_10": close, "ma_50": close,
        "volatility": 0.01, "momentum": 0.0, "rsi": 50.0, "volume_ma_30": 1_000_000.0, "Ticker": ticker,
    }, index=dates)
    df.index.name = "Date"
    return df


def _make_sentiment_df(dates, ticker="TEST"):
    return pd.DataFrame({"date": dates, "ticker": ticker, "sentiment_score": np.linspace(-1, 1, len(dates))})


@pytest.fixture(autouse=True)
def _patch_env_and_model(monkeypatch):
    monkeypatch.setattr(ablations, "RobustTradingEnv", _FakeEnv)
    monkeypatch.setattr(ablations, "RecurrentPPO", _FakeModel)
    _FakeEnv.captured_feature_columns = None
    _FakeEnv.captured_use_differential_sharpe = None


def test_sentiment_only_config_exists_in_default_suite():
    names = {c.name for c in ABLATION_CONFIGS}
    assert "sentiment_only" in names
    assert "full_pipeline_with_sentiment" in names


def test_sentiment_config_raises_without_sentiment_data():
    df = _make_df()
    config = AblationConfig(
        "sentiment_only", use_bc_init=False, use_meta_feature=False, use_self_critique=False,
        use_sentiment_feature=True,
    )
    with pytest.raises(ValueError, match="sentiment"):
        run_ablation_config(config, df, df, "TEST", strategy_signals=[])


def test_sentiment_config_merges_sentiment_score_into_feature_columns():
    df = _make_df()
    sentiment_df = _make_sentiment_df(df.index)
    config = AblationConfig(
        "sentiment_only", use_bc_init=False, use_meta_feature=False, use_self_critique=False,
        use_sentiment_feature=True,
    )

    run_ablation_config(
        config, df, df, "TEST", strategy_signals=[],
        train_sentiment_df=sentiment_df, test_sentiment_df=sentiment_df,
    )

    assert SENTIMENT_SCORE_COLUMN in _FakeEnv.captured_feature_columns


def test_non_sentiment_config_is_unaffected_by_new_wiring():
    df = _make_df()
    config = AblationConfig("phase1_baseline", use_bc_init=False, use_meta_feature=False, use_self_critique=False)

    run_ablation_config(config, df, df, "TEST", strategy_signals=[])

    assert SENTIMENT_SCORE_COLUMN not in _FakeEnv.captured_feature_columns


def test_dsr_configs_exist_in_default_suite():
    names = {c.name for c in ABLATION_CONFIGS}
    assert "dsr_only" in names
    assert "full_pipeline_with_dsr" in names


def test_dsr_config_passes_use_differential_sharpe_to_env():
    df = _make_df()
    config = AblationConfig(
        "dsr_only", use_bc_init=False, use_meta_feature=False, use_self_critique=False,
        use_differential_sharpe=True,
    )

    run_ablation_config(config, df, df, "TEST", strategy_signals=[])

    assert _FakeEnv.captured_use_differential_sharpe is True


def test_non_dsr_config_does_not_enable_differential_sharpe():
    df = _make_df()
    config = AblationConfig("phase1_baseline", use_bc_init=False, use_meta_feature=False, use_self_critique=False)

    run_ablation_config(config, df, df, "TEST", strategy_signals=[])

    assert _FakeEnv.captured_use_differential_sharpe is False


def test_dsr_benchmark_relative_config_exists_in_default_suite():
    names = {c.name for c in ABLATION_CONFIGS}
    assert "dsr_benchmark_relative_only" in names


def test_dsr_benchmark_relative_config_passes_flag_to_env():
    df = _make_df()
    config = AblationConfig(
        "dsr_benchmark_relative_only", use_bc_init=False, use_meta_feature=False, use_self_critique=False,
        use_differential_sharpe=True, dsr_benchmark_relative=True,
    )

    run_ablation_config(config, df, df, "TEST", strategy_signals=[])

    assert _FakeEnv.captured_use_differential_sharpe is True
    assert _FakeEnv.captured_dsr_benchmark_relative is True


def test_non_dsr_benchmark_relative_config_defaults_to_false():
    df = _make_df()
    config = AblationConfig("phase1_baseline", use_bc_init=False, use_meta_feature=False, use_self_critique=False)

    run_ablation_config(config, df, df, "TEST", strategy_signals=[])

    assert _FakeEnv.captured_dsr_benchmark_relative is False
