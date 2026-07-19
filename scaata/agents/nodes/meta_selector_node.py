"""Meta-Selector Agent — trains/updates the BC model and meta-selector
classifier on the current (down-)weighted strategy pool, and emits the
per-state strategy weight matrix that feeds `strategy_weights` in shared
state and, downstream, the RL env's observation (gap 3).
"""
from __future__ import annotations

from scaata.agents.state import AgentState
from scaata.config import FEATURE_COLUMNS
from scaata.imitation.train import train_imitation_model
from scaata.strategies.meta_selector import predict_strategy_weights, train_meta_selector
from scaata.strategies.pool import build_strategy_pool_signals


def meta_selector_node(state: AgentState) -> dict:
    train_df = state["train_df"]
    feature_columns = state.get("feature_columns", FEATURE_COLUMNS)

    # Compute the master strategy-signal pool once; kept stable across
    # iterations so strategy indices (and their down-weights) stay meaningful.
    strategy_signals = state.get("strategy_signals")
    if not strategy_signals:
        strategy_signals = build_strategy_pool_signals(state["normalized_strategies"], train_df)

    num_strategies = max(len(strategy_signals), 1)
    weights = state.get("strategy_pool_weights") or [1.0] * num_strategies

    states_arr = train_df[feature_columns].values
    signals_list = [s["signals"] for s in strategy_signals]
    seed = state.get("iteration", 0)

    bc_model = train_imitation_model(
        states_arr, signals_list, strategy_weights=weights, epochs=5, seed=seed
    )

    future_returns = train_df["returns"].shift(-1).fillna(0).values
    meta_model, _ = train_meta_selector(
        states_arr, signals_list, future_returns, strategy_weights=weights, epochs=5, seed=seed
    )

    weight_matrix = predict_strategy_weights(meta_model, states_arr)

    return {
        "strategy_signals": strategy_signals,
        "strategy_pool_weights": weights,
        "bc_model": bc_model,
        "meta_model": meta_model,
        "strategy_weight_matrix": weight_matrix,
        "num_strategies": num_strategies,
    }
