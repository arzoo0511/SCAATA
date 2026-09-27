"""Devil's-Advocate Agent (Phase 12) — the actual decision-reasoning trace
mechanism ("what is its thinking, what alternative was considered"): not
templated rationale text, but a second, real re-simulation of the runner-up
(second-highest-weighted) strategy over the exact same trailing window the
meta-selector's top pick was simulated over, diffed against the same
`window_critique_report` vocabulary already used everywhere else in this
pipeline (the RL reward, the Hedge combiner's per-expert loss, strategy
evolution's fitness function).

Deliberately does NOT feed back into `critique_node`'s weight update in
this phase — `state["devils_advocate_report"]` is attached to `history`
purely as an inspectable trace. Wiring its `delta` into the down-weight
decision is a natural future extension (the report already contains
everything needed to do so), left out here to avoid destabilizing the
Hedge mechanism `critique_node` already validates via
`tests/test_robustness.py`.

Phase 16 additions (three concrete extensions, not a rewrite):
1. `rank_all_experts` — every pooled expert's counterfactual, not just the
   runner-up. Exposed as `full_ranking` when an external caller has
   already computed ensemble disagreement (Phase 14's 5-seed uncertainty)
   and passed it into state above `ENSEMBLE_DISAGREEMENT_DEEP_DIVE_THRESHOLD`
   -- exactly when a single runner-up comparison is least trustworthy, a
   full ranking is worth the extra compute.
2. `rl_vs_best_alternative` — when an RL-policy expert is present (see
   `scaata.rl.policy_eval`/`critique_node.RL_POLICY_EXPERT_SOURCE`), always
   diffs the RL policy's own realized trajectory against whichever
   non-RL expert scored best this window. This is the literal "critique
   the RL policy's actual action, not just the strategy pool's pick" this
   phase was built for.
3. Both reuse `critique_node`'s expert-assembly helpers (sentiment/RL
   entries) rather than duplicating that logic, so "which experts exist"
   is defined in exactly one place.
"""
from __future__ import annotations

import numpy as np

from scaata.agents.nodes.critique_node import (
    RL_POLICY_EXPERT_SOURCE,
    _rl_policy_expert_entry,
    _sentiment_expert_entry,
)
from scaata.agents.state import AgentState
from scaata.config import ENSEMBLE_DISAGREEMENT_DEEP_DIVE_THRESHOLD, REVIEW_WINDOW_DAYS, RL_IMPLIED_SIGNAL_COLUMN
from scaata.rl.reward import window_critique_report
from scaata.strategies.backtest_signal import simulate_trajectory_for_choice

DELTA_KEYS = ("drawdown_penalty", "volatility", "holding_time_penalty", "window_return", "total_penalty")
NEGLIGIBLE_EPSILON = 1e-6


def _build_full_experts(train_df, strategy_signals: list[dict]) -> list[dict]:
    """Same expert-assembly `critique_node` uses (strategy pool + optional
    sentiment/RL-policy experts, gated on whether their source columns are
    present on `train_df`) -- kept in one place there and reused here so
    devil's-advocate and critique never disagree about which experts
    exist."""
    has_sentiment = "sentiment_score" in train_df.columns
    has_rl_policy = RL_IMPLIED_SIGNAL_COLUMN in train_df.columns
    return (
        list(strategy_signals)
        + ([_sentiment_expert_entry(train_df)] if has_sentiment else [])
        + ([_rl_policy_expert_entry(train_df)] if has_rl_policy else [])
    )


def rank_all_experts(train_df, experts: list[dict], window: int) -> list[dict]:
    """Every expert's counterfactual trajectory over the trailing window,
    ranked best-to-worst by the same `window_return + total_penalty` score
    `scaata.strategies.hedge.expert_losses_from_critique` uses (as a loss;
    here reported as a score, so higher is better) -- the full picture
    behind a single runner-up comparison.
    """
    ranked = []
    for idx, expert in enumerate(experts):
        values, actions = simulate_trajectory_for_choice(train_df, experts, idx, window)
        report = window_critique_report(values, actions, window)
        ranked.append({
            "expert_idx": idx,
            "source": expert["source"],
            "report": report,
            "score": report["window_return"] + report["total_penalty"],
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def _rl_vs_best_alternative_report(train_df, experts: list[dict], window: int) -> dict | None:
    """Diffs the RL policy's own realized trajectory this window against
    whichever non-RL expert scored best -- returns None when no RL-policy
    expert is present (nothing to critique yet)."""
    rl_idx = next((i for i, e in enumerate(experts) if e["source"] == RL_POLICY_EXPERT_SOURCE), None)
    if rl_idx is None:
        return None

    ranking = rank_all_experts(train_df, experts, window)
    rl_entry = next(r for r in ranking if r["expert_idx"] == rl_idx)
    best_alternative = next((r for r in ranking if r["source"] != RL_POLICY_EXPERT_SOURCE), None)
    if best_alternative is None:
        return None

    return {
        "rl_report": rl_entry["report"],
        "best_alternative_source": best_alternative["source"],
        "best_alternative_report": best_alternative["report"],
        "rl_beat_best_alternative": bool(rl_entry["score"] > best_alternative["score"]),
    }


def _runner_up_choice_per_step(weight_matrix: np.ndarray) -> np.ndarray:
    """Second-highest-weighted strategy index per row: `argsort` ascending,
    so `[:, -1]` is the top pick (again) and `[:, -2]` is the runner-up."""
    return np.argsort(weight_matrix, axis=1)[:, -2]


def _verdict_from_delta(delta: dict) -> str:
    """A small, deterministic lookup over sign combinations of
    `total_penalty` (risk) and `window_return` (raw return), both computed
    as chosen-minus-runner-up so a positive value means "chosen did better
    on this metric". Genuine structured reasoning generated from real
    numbers — not a mail-merge template, not an LLM call.
    """
    risk_delta = delta["total_penalty"]
    return_delta = delta["window_return"]

    if abs(risk_delta) < NEGLIGIBLE_EPSILON and abs(return_delta) < NEGLIGIBLE_EPSILON:
        return "negligible difference between the chosen strategy and the runner-up this window"
    if risk_delta >= 0 and return_delta >= 0:
        return "chosen strategy dominates the runner-up on both risk and return this window"
    if risk_delta < 0 and return_delta < 0:
        return (
            "runner-up would have scored better on both risk and return this window "
            "-- the chosen strategy is the weaker pick"
        )
    if risk_delta >= 0 and return_delta < 0:
        return "chosen strategy avoided deeper risk penalties at the cost of lower raw return this window"
    return (
        "chosen strategy achieved a better raw return this window, but only by incurring "
        "higher risk penalties than the runner-up would have"
    )


def devils_advocate_report(
    train_df,
    strategy_signals: list[dict],
    weight_matrix: np.ndarray,
    window: int,
    ensemble_disagreement: float | None = None,
) -> dict | None:
    """Returns `{chosen_report, alt_report, delta, verdict,
    chosen_strategy_idx, runner_up_strategy_idx, full_ranking,
    deep_dive_triggered, rl_vs_best_alternative}`, or `None` if the pool
    has fewer than 2 strategies (no runner-up exists to argue for).

    The original top-2 (chosen vs. runner-up, by the meta-selector's
    per-day `weight_matrix`) comparison is unchanged. Three Phase 16
    additions layer on top, all optional/additive:
    - `full_ranking`: every expert (strategy pool + sentiment + RL-policy,
      whichever are present) ranked by this window's counterfactual score,
      computed when `ensemble_disagreement` is given and crosses
      `ENSEMBLE_DISAGREEMENT_DEEP_DIVE_THRESHOLD` -- exactly when a single
      runner-up comparison is least trustworthy.
    - `rl_vs_best_alternative`: always computed (regardless of
      `ensemble_disagreement`) whenever an RL-policy expert is present --
      the RL policy's own realized trajectory vs. whichever non-RL expert
      scored best, the literal "critique the RL policy's actual action"
      this phase exists for.
    - `deep_dive_triggered`: whether the disagreement threshold was
      actually crossed this call, so a caller can tell an always-None
      `full_ranking` (no ensemble info was ever passed) apart from a
      not-triggered-this-time one.
    """
    if weight_matrix.shape[1] < 2:
        return None

    n = len(train_df)
    start = max(0, n - window)
    windowed_weights = weight_matrix[start:]
    chosen_idx = windowed_weights.argmax(axis=1)
    runner_up_idx = _runner_up_choice_per_step(windowed_weights)

    chosen_values, chosen_actions = simulate_trajectory_for_choice(train_df, strategy_signals, chosen_idx, window)
    alt_values, alt_actions = simulate_trajectory_for_choice(train_df, strategy_signals, runner_up_idx, window)

    chosen_report = window_critique_report(chosen_values, chosen_actions, window)
    alt_report = window_critique_report(alt_values, alt_actions, window)
    delta = {k: chosen_report[k] - alt_report[k] for k in DELTA_KEYS}
    verdict = _verdict_from_delta(delta)

    full_experts = _build_full_experts(train_df, strategy_signals)
    deep_dive_triggered = (
        ensemble_disagreement is not None and ensemble_disagreement >= ENSEMBLE_DISAGREEMENT_DEEP_DIVE_THRESHOLD
    )
    full_ranking = rank_all_experts(train_df, full_experts, window) if deep_dive_triggered else None
    rl_vs_best_alternative = _rl_vs_best_alternative_report(train_df, full_experts, window)

    return {
        "chosen_report": chosen_report,
        "alt_report": alt_report,
        "delta": delta,
        "verdict": verdict,
        "chosen_strategy_idx": int(chosen_idx[-1]) if len(chosen_idx) else None,
        "runner_up_strategy_idx": int(runner_up_idx[-1]) if len(runner_up_idx) else None,
        "ensemble_disagreement": ensemble_disagreement,
        "deep_dive_triggered": deep_dive_triggered,
        "full_ranking": full_ranking,
        "rl_vs_best_alternative": rl_vs_best_alternative,
    }


def devils_advocate_node(state: AgentState) -> dict:
    train_df = state["train_df"]
    window = state.get("review_window", REVIEW_WINDOW_DAYS)
    weight_matrix = state["strategy_weight_matrix"]
    strategy_signals = state["strategy_signals"]
    ensemble_disagreement = state.get("ensemble_disagreement")

    report = devils_advocate_report(train_df, strategy_signals, weight_matrix, window, ensemble_disagreement)
    return {"devils_advocate_report": report}
