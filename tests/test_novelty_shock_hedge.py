"""Capstone Phase 10 test: does novelty-modulated Hedge weighting actually
shift trust toward the sentiment expert during an injected, genuinely
novel shock — the direct, falsifiable test of the "news should dominate
during a shock" hypothesis, as opposed to trusting the mechanism just
because it's structurally similar to code that already works elsewhere.

Construction: a synthetic `train_df` where two "noise" strategies are
uncorrelated with returns throughout, but `sentiment_score` correctly
predicts the direction of a clear price trend specifically in the final
`window` days — and those same final days carry a large distributional
shift in the technical feature columns, so the causal novelty detector
should register elevated novelty right at the point being scored.

Calling `critique_node` with `novelty_score` forced to 0 isolates the
"fixed eta, no sentiment prior" ablation arm from the real, computed-novelty
arm — both see identical per-expert losses (loss computation doesn't depend
on novelty_score), so any weight difference between the two runs is
attributable specifically to novelty-modulation, not to "any" reweighting.
"""
import numpy as np
import pandas as pd

from scaata.agents.nodes.critique_node import SENTIMENT_EXPERT_SOURCE, critique_node
from scaata.config import FEATURE_COLUMNS, REVIEW_WINDOW_DAYS


def _make_shock_train_df(n=200, window=REVIEW_WINDOW_DAYS, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Calm segment: flat-ish noisy price, feature columns are unit-normal noise.
    returns = rng.normal(0.0, 0.005, n)
    features = {c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS}

    # Shock segment: the final `window` days get a clear directional trend
    # AND a large mean/variance shift in the feature columns (so the
    # causal novelty detector should register this as unusual relative to
    # the preceding calm history).
    shock_start = n - window
    trend = np.linspace(0.01, 0.03, window)  # strong, steadily increasing daily returns
    returns[shock_start:] = trend
    for c in FEATURE_COLUMNS:
        features[c][shock_start:] = rng.normal(8, 3, window)  # far outside the calm N(0,1) reference

    close = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame(features, index=dates)
    df["Close"] = close
    df["returns"] = returns
    df["Volume"] = rng.integers(1_000_000, 5_000_000, n)
    df["volume_ma_30"] = df["Volume"].rolling(30, min_periods=1).mean()
    df["Ticker"] = "SHOCK"

    # sentiment_score: pure noise in the calm segment (uncorrelated with
    # returns), but a strong, correctly-signed signal during the shock --
    # sentiment "sees" the trend the noise strategies can't.
    sentiment = rng.normal(0, 0.05, n)
    sentiment[shock_start:] = 1.0  # unambiguously bullish, matching the injected uptrend
    df["sentiment_score"] = sentiment

    return df


def _make_noise_strategy_signals(n, seed=1):
    rng = np.random.default_rng(seed)
    return [
        {"source": "noise_a", "signals": rng.choice([-1, 0, 1], size=n)},
        {"source": "noise_b", "signals": rng.choice([-1, 0, 1], size=n)},
    ]


def _base_state(train_df, strategy_signals, window, novelty_score=None):
    n_strategies = len(strategy_signals)
    state = {
        "train_df": train_df,
        "feature_columns": FEATURE_COLUMNS,
        "review_window": window,
        "strategy_signals": strategy_signals,
        "strategy_pool_weights": [1.0] * n_strategies,
        "strategy_weight_matrix": np.full((len(train_df), n_strategies), 1.0 / n_strategies),
        "iteration": 0,
        "max_iterations": 4,
        "history": [],
    }
    if novelty_score is not None:
        state["novelty_score"] = novelty_score
    return state


def test_computed_novelty_is_elevated_during_the_injected_shock():
    window = REVIEW_WINDOW_DAYS
    train_df = _make_shock_train_df(window=window)
    strategy_signals = _make_noise_strategy_signals(len(train_df))

    result = critique_node(_base_state(train_df, strategy_signals, window))

    assert result["novelty_score"] > 0.5, (
        "novelty detector failed to flag an obvious, deliberately injected distributional shift"
    )


def test_sentiment_expert_gets_more_weight_with_novelty_modulation_than_without():
    window = REVIEW_WINDOW_DAYS
    train_df = _make_shock_train_df(window=window)
    strategy_signals = _make_noise_strategy_signals(len(train_df))

    # Ablation arm A: novelty forced to 0 -- fixed eta, no sentiment prior.
    result_fixed = critique_node(_base_state(train_df, strategy_signals, window, novelty_score=0.0))
    # Ablation arm B: real, computed novelty (high, per the test above).
    result_novelty = critique_node(_base_state(train_df, strategy_signals, window, novelty_score=None))

    sentiment_idx_fixed = result_fixed["expert_names"].index(SENTIMENT_EXPERT_SOURCE)
    sentiment_idx_novelty = result_novelty["expert_names"].index(SENTIMENT_EXPERT_SOURCE)

    weight_fixed = result_fixed["strategy_pool_weights"][sentiment_idx_fixed]
    weight_novelty = result_novelty["strategy_pool_weights"][sentiment_idx_novelty]

    assert weight_novelty > weight_fixed, (
        "novelty-modulated Hedge should give the sentiment expert strictly more weight "
        "than the fixed-eta (novelty-blind) ablation arm, given identical per-expert losses"
    )
    # And in both arms, sentiment should already be doing well on pure loss
    # grounds (it correctly read the shock, the noise strategies didn't) --
    # novelty-modulation should amplify an existing advantage, not manufacture one.
    assert weight_fixed > 1.0 / len(result_fixed["expert_names"])
