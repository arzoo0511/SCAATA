"""Genetic-programming strategy evolution (Phase 11) — the actual answer to
"are the scraped/mock strategies enough": instead of relying only on
GitHub-scraped code or `scaata.strategies.library`'s hand-written set,
evolve new candidate strategies directly from `FEATURE_COLUMNS` via
mutation/crossover of shallow rule-trees, fitness-scored by the same
`window_critique_report` the RL reward and critique loop already use.

Every evolved candidate compiles to a literal `def strategy(df): ...`
source string that satisfies `scaata.strategies.pool.run_strategy_safely`'s
existing contract unchanged — evolution needs no new backtest engine and no
per-candidate PPO/LLM calls.

Deliberate scale discipline: `ma_10`/`ma_50` are absolute, price-scale
features (a $150 stock and a $3000 stock aren't comparable via the same
fixed threshold), so they may only appear in *pairwise* conditions against
each other (`ma_10 > ma_50`, a scale-invariant crossover check) — never
against a fixed numeric threshold. Only genuinely scale-invariant/bounded
features (`returns`, `momentum`, `rsi`, `volatility`) get threshold
conditions. This isn't just an overfitting mitigation, it's a category
error to skip: a threshold tuned against one ticker's price level is
meaningless on another's.

Depth is capped at `EVOLUTION_MAX_TREE_DEPTH` (3) deliberately: a 3-condition
tree already has enough free thresholds to fit noise in a few thousand
rows, and overfitting risk grows faster than this project's ~2.5-year test
window can validate away. Fitness is averaged across all available tickers
(not per-ticker) specifically to discourage a rule that only fits one
idiosyncratic price path, and the in-sample-vs-held-out-val fitness gap is
reported per candidate as the honest overfitting tell — a large gap should
be reported, not quietly dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scaata.config import (
    EVOLUTION_ELITE_FRAC,
    EVOLUTION_MAX_TREE_DEPTH,
    EVOLUTION_MUTATION_RATE,
    EVOLUTION_N_GENERATIONS,
    EVOLUTION_POPULATION_SIZE,
    EVOLUTION_RANDOM_IMMIGRANT_FRAC,
    EVOLUTION_VAL_FRAC,
    FEATURE_COLUMNS,
)
from scaata.rl.reward import window_critique_report
from scaata.strategies.backtest_signal import simulate_trajectory_for_choice
from scaata.strategies.pool import run_strategy_safely

# Only these are safe to compare against a fixed numeric threshold across
# tickers with different price/volume scales; (lo, hi) bounds the random
# threshold sampling range to plausible values for each feature.
THRESHOLD_FEATURE_RANGES = {
    "returns": (-0.05, 0.05),
    "momentum": (-0.10, 0.10),
    "rsi": (20.0, 80.0),
    "volatility": (0.0, 0.05),
}
PAIR_FEATURES = ("ma_10", "ma_50")  # the only scale-invariant pairwise comparison among FEATURE_COLUMNS


@dataclass(frozen=True)
class ThresholdCondition:
    feature: str
    op: str  # ">" or "<"
    threshold: float

    def to_expr(self) -> str:
        return f"df['{self.feature}'] {self.op} {self.threshold!r}"


@dataclass(frozen=True)
class PairCondition:
    feature_a: str
    feature_b: str
    op: str  # ">" or "<"

    def to_expr(self) -> str:
        return f"df['{self.feature_a}'] {self.op} df['{self.feature_b}']"


@dataclass(frozen=True)
class RuleNode:
    condition: ThresholdCondition | PairCondition
    action: int  # -1 (sell), 1 (buy) -- 0/hold is reserved for else_action


@dataclass
class RuleTree:
    nodes: list[RuleNode] = field(default_factory=list)  # evaluated in order; first true condition wins (if/elif)
    else_action: int = 0


def random_condition(rng: np.random.Generator, feature_columns: list[str]):
    pair_available = PAIR_FEATURES[0] in feature_columns and PAIR_FEATURES[1] in feature_columns
    threshold_candidates = [f for f in feature_columns if f in THRESHOLD_FEATURE_RANGES]

    use_pair = pair_available and (not threshold_candidates or rng.random() < 0.3)
    if use_pair:
        op = str(rng.choice([">", "<"]))
        return PairCondition(PAIR_FEATURES[0], PAIR_FEATURES[1], op)

    feature = str(rng.choice(threshold_candidates))
    lo, hi = THRESHOLD_FEATURE_RANGES[feature]
    threshold = float(rng.uniform(lo, hi))
    op = str(rng.choice([">", "<"]))
    return ThresholdCondition(feature, op, threshold)


def random_rule_tree(
    rng: np.random.Generator,
    feature_columns: list[str] = FEATURE_COLUMNS,
    max_depth: int = EVOLUTION_MAX_TREE_DEPTH,
) -> RuleTree:
    depth = int(rng.integers(1, max_depth + 1))
    nodes = [RuleNode(random_condition(rng, feature_columns), int(rng.choice([-1, 1]))) for _ in range(depth)]
    else_action = int(rng.choice([-1, 0, 1]))
    return RuleTree(nodes=nodes, else_action=else_action)


def mutate(
    tree: RuleTree,
    rng: np.random.Generator,
    feature_columns: list[str] = FEATURE_COLUMNS,
    max_depth: int = EVOLUTION_MAX_TREE_DEPTH,
) -> RuleTree:
    """Applies exactly one random structural or parametric change: perturb
    a node's condition, add/remove a node (within the depth cap), flip a
    node's action, or change the else-action."""
    nodes = list(tree.nodes)
    options = ["perturb", "swap_action", "else_action"]
    if len(nodes) < max_depth:
        options.append("add")
    if len(nodes) > 1:
        options.append("remove")

    kind = str(rng.choice(options))
    else_action = tree.else_action

    if kind == "perturb" and nodes:
        idx = int(rng.integers(0, len(nodes)))
        nodes[idx] = RuleNode(random_condition(rng, feature_columns), nodes[idx].action)
    elif kind == "add":
        nodes.append(RuleNode(random_condition(rng, feature_columns), int(rng.choice([-1, 1]))))
    elif kind == "remove" and nodes:
        idx = int(rng.integers(0, len(nodes)))
        nodes.pop(idx)
    elif kind == "swap_action" and nodes:
        idx = int(rng.integers(0, len(nodes)))
        nodes[idx] = RuleNode(nodes[idx].condition, int(rng.choice([-1, 1])))
    elif kind == "else_action":
        else_action = int(rng.choice([-1, 0, 1]))

    if not nodes:  # never produce a tree with zero conditions
        nodes = [RuleNode(random_condition(rng, feature_columns), int(rng.choice([-1, 1])))]

    return RuleTree(nodes=nodes, else_action=else_action)


def crossover(
    tree_a: RuleTree, tree_b: RuleTree, rng: np.random.Generator, max_depth: int = EVOLUTION_MAX_TREE_DEPTH
) -> RuleTree:
    """Pools both parents' nodes and samples a fresh subset (capped at
    `max_depth`), rather than a single-point splice — with such shallow
    trees a single crossover point wouldn't meaningfully mix genetic
    material."""
    pool = list(tree_a.nodes) + list(tree_b.nodes)
    rng.shuffle(pool)
    depth = int(rng.integers(1, max_depth + 1))
    nodes = pool[:depth] if pool else []
    if not nodes:
        nodes = [tree_a.nodes[0]] if tree_a.nodes else [tree_b.nodes[0]]
    else_action = tree_a.else_action if rng.random() < 0.5 else tree_b.else_action
    return RuleTree(nodes=nodes, else_action=else_action)


def compile_rule_tree(tree: RuleTree) -> str:
    """Emits a literal `def strategy(df): ...` source string implementing
    the tree's if/elif priority order (first matching condition wins) —
    satisfies `scaata.strategies.pool.run_strategy_safely`'s exact contract
    unchanged."""
    lines = [
        "def strategy(df):",
        "    import numpy as np",
        "    n = len(df)",
        f"    sig = np.full(n, {tree.else_action}, dtype=int)",
        "    matched = np.zeros(n, dtype=bool)",
    ]
    for node in tree.nodes:
        lines.append(f"    cond = ({node.condition.to_expr()}).values & (~matched)")
        lines.append(f"    sig = np.where(cond, {node.action}, sig)")
        lines.append("    matched = matched | cond")
    lines.append("    return np.nan_to_num(sig).astype(int)")
    return "\n".join(lines) + "\n"


def evaluate_fitness(signals: np.ndarray, df: pd.DataFrame) -> dict:
    """Scores a single already-computed signal array over the full span of
    `df` by reusing `simulate_trajectory_for_choice` (wrapping the single
    signal array as a one-expert pool, broadcasting its constant index 0
    across every step) and `window_critique_report` — the same mechanism
    the RL reward and the Hedge combiner's per-expert loss (Phase 10) use,
    not a separate backtest engine. Fitness is `window_return +
    total_penalty` (higher is better) — a strategy that avoids every
    penalty trigger by never trading also never scores well, since
    penalties alone can't compensate for zero return.
    """
    window = len(df)
    wrapped = [{"source": "candidate", "signals": signals}]
    values, actions = simulate_trajectory_for_choice(df, wrapped, 0, window)
    report = window_critique_report(values, actions, window)
    fitness = report["window_return"] + report["total_penalty"]
    return {"fitness": fitness, "window_return": report["window_return"], "total_penalty": report["total_penalty"]}


def evaluate_fitness_across_tickers(
    code: str, ticker_dfs: dict[str, pd.DataFrame], val_frac: float = EVOLUTION_VAL_FRAC
) -> dict:
    """Averages fitness across every ticker in `ticker_dfs` (not per-ticker)
    to discourage a rule that only fits one idiosyncratic price path, using
    a time-based (not random) in-sample/held-out split per ticker so the
    validation slice is genuinely out-of-sample, not just held-out rows
    interleaved with training rows. A strategy that fails to execute or
    produces a constant (degenerate) signal on a ticker is scored as
    maximally bad on that ticker rather than silently skipped.
    """
    in_sample_scores, val_scores = [], []
    for _, df in ticker_dfs.items():
        signals = run_strategy_safely(code, df)
        if signals is None or len(np.unique(signals)) <= 1:
            in_sample_scores.append(-np.inf)
            val_scores.append(-np.inf)
            continue

        split_idx = int(len(df) * (1 - val_frac))
        in_df, in_signals = df.iloc[:split_idx], signals[:split_idx]
        val_df, val_signals = df.iloc[split_idx:], signals[split_idx:]

        if len(in_df) > 5:
            in_sample_scores.append(evaluate_fitness(in_signals, in_df)["fitness"])
        if len(val_df) > 5:
            val_scores.append(evaluate_fitness(val_signals, val_df)["fitness"])

    mean_in_sample = float(np.mean(in_sample_scores)) if in_sample_scores else -np.inf
    mean_val = float(np.mean(val_scores)) if val_scores else -np.inf
    return {
        "in_sample_fitness": mean_in_sample,
        "val_fitness": mean_val,
        "fitness_gap": mean_in_sample - mean_val,
    }


def run_evolution(
    ticker_dfs: dict[str, pd.DataFrame],
    feature_columns: list[str] = FEATURE_COLUMNS,
    population_size: int = EVOLUTION_POPULATION_SIZE,
    n_generations: int = EVOLUTION_N_GENERATIONS,
    elite_frac: float = EVOLUTION_ELITE_FRAC,
    mutation_rate: float = EVOLUTION_MUTATION_RATE,
    random_immigrant_frac: float = EVOLUTION_RANDOM_IMMIGRANT_FRAC,
    val_frac: float = EVOLUTION_VAL_FRAC,
    max_depth: int = EVOLUTION_MAX_TREE_DEPTH,
    seed: int = 0,
) -> list[dict]:
    """Evolves `population_size` rule-trees for `n_generations`, selecting
    on in-sample fitness (averaged across `ticker_dfs`) each generation,
    and returns the final population ranked by *held-out* validation
    fitness (not in-sample) — selection pressure during evolution uses
    in-sample performance since that's the only fair "did this generation's
    change help" signal available without leaking the validation split into
    search, but the final ranking presented to callers uses val_fitness so
    an overfit-but-lucky-in-sample candidate doesn't get promoted.
    """
    rng = np.random.default_rng(seed)
    population = [random_rule_tree(rng, feature_columns, max_depth) for _ in range(population_size)]

    n_elite = max(1, int(population_size * elite_frac))
    n_immigrants = max(0, int(population_size * random_immigrant_frac))

    for _ in range(n_generations):
        scored = []
        for tree in population:
            code = compile_rule_tree(tree)
            report = evaluate_fitness_across_tickers(code, ticker_dfs, val_frac)
            scored.append((tree, report["in_sample_fitness"]))
        scored.sort(key=lambda item: item[1], reverse=True)
        elites = [tree for tree, _ in scored[:n_elite]]

        next_population = list(elites)
        while len(next_population) < population_size - n_immigrants:
            parent_a = elites[rng.integers(0, len(elites))]
            parent_b = elites[rng.integers(0, len(elites))]
            child = crossover(parent_a, parent_b, rng, max_depth)
            if rng.random() < mutation_rate:
                child = mutate(child, rng, feature_columns, max_depth)
            next_population.append(child)
        while len(next_population) < population_size:
            next_population.append(random_rule_tree(rng, feature_columns, max_depth))

        population = next_population[:population_size]

    final = []
    for i, tree in enumerate(population):
        code = compile_rule_tree(tree)
        report = evaluate_fitness_across_tickers(code, ticker_dfs, val_frac)
        final.append({
            "source": f"evolved:gen{n_generations}_ind{i}",
            "clean_code": code,
            "in_sample_fitness": report["in_sample_fitness"],
            "val_fitness": report["val_fitness"],
            "fitness_gap": report["fitness_gap"],
        })
    final.sort(key=lambda d: d["val_fitness"], reverse=True)
    return final
