"""Strategy-Evolution Agent (Phase 11, optional) — when
`config.ENABLE_STRATEGY_EVOLUTION` is set, evolves additional candidate
strategies directly from `train_df`'s own feature columns
(`scaata.strategies.evolve.run_evolution`) and appends the best few (by
held-out `val_fitness`, never `in_sample_fitness`) to `normalized_strategies`
alongside whatever the scraper/normalizer produced, so `meta_selector_node`'s
pool includes both.

Honest scope note: this node only has access to the single ticker's
`train_df` carried in graph state, so evolution run this way is
single-ticker fitness, not the cross-ticker-averaged fitness
`run_evolution` is built to support (and that the Phase 11 ablation
verification actually exercises by calling `run_evolution` directly across
multiple tickers). Evolving a shared, genuinely cross-ticker pool once,
offline, and feeding it in as an additional strategy source — the same way
`scaata.strategies.library`'s strategies are added — is the more rigorous
path; this node exists for convenience when a caller wants per-invocation
evolution and accepts the narrower single-ticker scope.

Default-off (`ENABLE_STRATEGY_EVOLUTION=False`): returns `{}` unchanged, an
exact no-op that preserves every existing test's/pipeline's behavior.
"""
from __future__ import annotations

from scaata.agents.state import AgentState
from scaata.config import (
    ENABLE_STRATEGY_EVOLUTION,
    EVOLUTION_ELITE_FRAC,
    EVOLUTION_MAX_TREE_DEPTH,
    EVOLUTION_MUTATION_RATE,
    EVOLUTION_N_GENERATIONS,
    EVOLUTION_POPULATION_SIZE,
    EVOLUTION_RANDOM_IMMIGRANT_FRAC,
    EVOLUTION_TOP_K_TO_POOL,
    EVOLUTION_VAL_FRAC,
    FEATURE_COLUMNS,
)
from scaata.strategies.evolve import run_evolution


def evolver_node(state: AgentState) -> dict:
    if not ENABLE_STRATEGY_EVOLUTION:
        return {}

    train_df = state["train_df"]
    feature_columns = state.get("feature_columns", FEATURE_COLUMNS)
    ticker = train_df["Ticker"].iloc[0] if "Ticker" in train_df.columns else "UNKNOWN"

    evolved = run_evolution(
        {ticker: train_df},
        feature_columns=feature_columns,
        population_size=EVOLUTION_POPULATION_SIZE,
        n_generations=EVOLUTION_N_GENERATIONS,
        elite_frac=EVOLUTION_ELITE_FRAC,
        mutation_rate=EVOLUTION_MUTATION_RATE,
        random_immigrant_frac=EVOLUTION_RANDOM_IMMIGRANT_FRAC,
        val_frac=EVOLUTION_VAL_FRAC,
        max_depth=EVOLUTION_MAX_TREE_DEPTH,
        seed=state.get("iteration", 0),
    )

    accepted = [e for e in evolved if e["val_fitness"] > float("-inf")][:EVOLUTION_TOP_K_TO_POOL]
    normalized_strategies = list(state.get("normalized_strategies", []))
    normalized_strategies += [{"source": e["source"], "clean_code": e["clean_code"]} for e in accepted]

    return {"normalized_strategies": normalized_strategies}
