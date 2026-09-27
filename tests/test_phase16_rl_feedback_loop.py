"""Tests for Phase 16: closing the critique -> RL feedback loop (the RL
policy becomes an accountable Hedge expert in critique_node when train_df
carries an RL_IMPLIED_SIGNAL_COLUMN column) and the devil's-advocate
extensions (full all-experts ranking gated on ensemble disagreement, and
an always-on RL-policy-vs-best-alternative comparison).
"""
import numpy as np
import pandas as pd

from scaata.agents.nodes.critique_node import RL_POLICY_EXPERT_SOURCE
from scaata.agents.nodes.devils_advocate_node import devils_advocate_report
from scaata.agents.orchestrator import run_inner_loop
from scaata.config import ENSEMBLE_DISAGREEMENT_DEEP_DIVE_THRESHOLD, FEATURE_COLUMNS, RL_IMPLIED_SIGNAL_COLUMN
from scaata.rl.policy_eval import attach_rl_implied_signal, compute_rl_implied_signals


def _make_synthetic_train_df(n=300, seed=0, with_rl_signal=False):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}, index=dates)
    df["Close"] = close
    df["returns"] = pd.Series(close, index=dates).pct_change().fillna(0)
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(30, min_periods=1).mean()
    df["Ticker"] = "TEST"
    if with_rl_signal:
        df[RL_IMPLIED_SIGNAL_COLUMN] = rng.choice([-1, 0, 1], n)
    return df


# --- critique_node: RL policy as an accountable Hedge expert ---

def test_rl_policy_expert_included_when_column_present():
    train_df = _make_synthetic_train_df(with_rl_signal=True)
    final_state = run_inner_loop(train_df, max_iterations=3)

    assert RL_POLICY_EXPERT_SOURCE in final_state["expert_names"]
    assert len(final_state["strategy_pool_weights"]) == len(final_state["expert_names"])


def test_rl_policy_expert_absent_without_column():
    train_df = _make_synthetic_train_df(with_rl_signal=False)
    final_state = run_inner_loop(train_df, max_iterations=3)

    assert RL_POLICY_EXPERT_SOURCE not in final_state["expert_names"]


# --- devils_advocate: rl_vs_best_alternative (always-on when RL expert exists) ---

def _make_deterministic_price_df(n=150):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = np.where(np.arange(n) % 2 == 0, 0.02, -0.02)
    close = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({"Close": close}, index=dates)
    future_returns = np.roll(returns, -1)
    future_returns[-1] = 0.0
    return df, future_returns


def test_rl_vs_best_alternative_is_none_without_rl_expert():
    train_df, _ = _make_deterministic_price_df()
    n = len(train_df)
    strategy_signals = [
        {"source": "a", "signals": np.ones(n, dtype=int)},
        {"source": "b", "signals": np.zeros(n, dtype=int)},
    ]
    weight_matrix = np.tile([0.7, 0.3], (n, 1))

    result = devils_advocate_report(train_df, strategy_signals, weight_matrix, window=60)
    assert result["rl_vs_best_alternative"] is None


def test_rl_vs_best_alternative_correctly_flags_a_good_rl_policy():
    """Same deterministic good/poisoned construction as
    test_devils_advocate.py's ground-truth check: the RL policy's implied
    signal has perfect one-step foresight, the strategy pool is poisoned --
    the comparison must report the RL policy beat the best alternative."""
    train_df, future_returns = _make_deterministic_price_df()
    good_signal = np.sign(future_returns).astype(int)
    train_df[RL_IMPLIED_SIGNAL_COLUMN] = good_signal

    n = len(train_df)
    strategy_signals = [{"source": "poisoned", "signals": -good_signal}]
    weight_matrix = np.ones((n, 1))  # only 1 pool strategy -> no runner-up, but rl_vs_best_alternative doesn't need one

    result = devils_advocate_report(train_df, strategy_signals, weight_matrix, window=60)
    assert result is None  # weight_matrix.shape[1] < 2 -> the whole report short-circuits to None

    # Give it 2 pool strategies so the top-2 mechanism has a runner-up too,
    # while still keeping both of them poisoned relative to the RL signal.
    strategy_signals_2 = [
        {"source": "poisoned_a", "signals": -good_signal},
        {"source": "poisoned_b", "signals": np.zeros(n, dtype=int)},
    ]
    weight_matrix_2 = np.tile([0.7, 0.3], (n, 1))
    result_2 = devils_advocate_report(train_df, strategy_signals_2, weight_matrix_2, window=60)

    assert result_2["rl_vs_best_alternative"] is not None
    assert result_2["rl_vs_best_alternative"]["rl_beat_best_alternative"] is True
    assert result_2["rl_vs_best_alternative"]["best_alternative_source"] != RL_POLICY_EXPERT_SOURCE


def test_rl_vs_best_alternative_correctly_flags_a_bad_rl_policy():
    """Mirror of the above: RL's implied signal IS the poisoned one, the
    pool has the good foresight strategy -- rl_beat_best_alternative must
    be False."""
    train_df, future_returns = _make_deterministic_price_df()
    good_signal = np.sign(future_returns).astype(int)
    train_df[RL_IMPLIED_SIGNAL_COLUMN] = -good_signal  # RL is poisoned

    n = len(train_df)
    strategy_signals = [
        {"source": "good", "signals": good_signal},
        {"source": "neutral", "signals": np.zeros(n, dtype=int)},
    ]
    weight_matrix = np.tile([0.7, 0.3], (n, 1))

    result = devils_advocate_report(train_df, strategy_signals, weight_matrix, window=60)
    assert result["rl_vs_best_alternative"]["rl_beat_best_alternative"] is False
    assert result["rl_vs_best_alternative"]["best_alternative_source"] == "good"


# --- devils_advocate: full_ranking gated on ensemble_disagreement ---

def test_full_ranking_absent_when_no_ensemble_disagreement_given():
    train_df, _ = _make_deterministic_price_df()
    n = len(train_df)
    strategy_signals = [
        {"source": "a", "signals": np.ones(n, dtype=int)},
        {"source": "b", "signals": np.zeros(n, dtype=int)},
    ]
    weight_matrix = np.tile([0.7, 0.3], (n, 1))

    result = devils_advocate_report(train_df, strategy_signals, weight_matrix, window=60)
    assert result["full_ranking"] is None
    assert result["deep_dive_triggered"] is False


def test_full_ranking_absent_when_disagreement_below_threshold():
    train_df, _ = _make_deterministic_price_df()
    n = len(train_df)
    strategy_signals = [
        {"source": "a", "signals": np.ones(n, dtype=int)},
        {"source": "b", "signals": np.zeros(n, dtype=int)},
    ]
    weight_matrix = np.tile([0.7, 0.3], (n, 1))
    below = ENSEMBLE_DISAGREEMENT_DEEP_DIVE_THRESHOLD - 0.05

    result = devils_advocate_report(train_df, strategy_signals, weight_matrix, window=60, ensemble_disagreement=below)
    assert result["full_ranking"] is None
    assert result["deep_dive_triggered"] is False


def test_full_ranking_present_and_sorted_when_disagreement_above_threshold():
    train_df, future_returns = _make_deterministic_price_df()
    good_signal = np.sign(future_returns).astype(int)
    n = len(train_df)
    strategy_signals = [
        {"source": "good", "signals": good_signal},
        {"source": "poisoned", "signals": -good_signal},
        {"source": "neutral", "signals": np.zeros(n, dtype=int)},
    ]
    weight_matrix = np.tile([0.5, 0.3, 0.2], (n, 1))
    above = ENSEMBLE_DISAGREEMENT_DEEP_DIVE_THRESHOLD + 0.1

    result = devils_advocate_report(train_df, strategy_signals, weight_matrix, window=60, ensemble_disagreement=above)

    assert result["deep_dive_triggered"] is True
    ranking = result["full_ranking"]
    assert ranking is not None
    assert len(ranking) == 3
    # best-to-worst: scores must be monotonically non-increasing
    scores = [r["score"] for r in ranking]
    assert scores == sorted(scores, reverse=True)
    assert ranking[0]["source"] == "good"  # the perfect-foresight strategy must rank first


# --- scaata.rl.policy_eval ---

class _FakeRecurrentModel:
    """Deterministic stub matching stable-baselines3's .predict() signature
    closely enough to exercise compute_rl_implied_signals without a real
    trained policy: alternates BUY/SELL/HOLD by call count."""

    def __init__(self):
        self.call_count = 0

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        action = self.call_count % 3  # HOLD, BUY, SELL, HOLD, BUY, SELL, ...
        self.call_count += 1
        return action, {"calls": self.call_count}


def test_compute_rl_implied_signals_maps_actions_to_signal_convention():
    df = pd.DataFrame({c: np.zeros(6) for c in FEATURE_COLUMNS})
    model = _FakeRecurrentModel()

    signals = compute_rl_implied_signals(model, df, FEATURE_COLUMNS)

    assert signals.shape == (6,)
    # actions cycle 0,1,2,0,1,2 (HOLD,BUY,SELL) -> signals 0,1,-1,0,1,-1
    np.testing.assert_array_equal(signals, [0, 1, -1, 0, 1, -1])


def test_compute_rl_implied_signals_carries_lstm_state_forward():
    df = pd.DataFrame({c: np.zeros(3) for c in FEATURE_COLUMNS})

    class _StateTrackingModel:
        def __init__(self):
            self.seen_states = []
            self.seen_episode_starts = []

        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            self.seen_states.append(state)
            self.seen_episode_starts.append(bool(episode_start[0]))
            return 0, {"step": len(self.seen_states)}

    model = _StateTrackingModel()
    compute_rl_implied_signals(model, df, FEATURE_COLUMNS)

    assert model.seen_episode_starts == [True, False, False]
    assert model.seen_states[0] is None  # no prior state on the first call
    assert model.seen_states[1] == {"step": 1}  # carried forward from the previous call's returned state


def test_attach_rl_implied_signal_adds_the_configured_column():
    df = pd.DataFrame({c: np.zeros(4) for c in FEATURE_COLUMNS})
    model = _FakeRecurrentModel()

    out = attach_rl_implied_signal(df, model, FEATURE_COLUMNS)

    assert RL_IMPLIED_SIGNAL_COLUMN in out.columns
    assert len(out) == len(df)
