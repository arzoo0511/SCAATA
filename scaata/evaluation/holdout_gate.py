"""Final honesty/holdout gate (Phase 15) — the last check before claiming
any headline Sharpe/drawdown number from Phases 9–15's mechanisms.

`HOLDOUT_TICKERS` (the project's existing `NON_SURVIVOR_TICKERS` — XOM,
INTC, BABA, added specifically to counter survivorship bias) are reserved
for this gate and must never be used to pick a hyperparameter, debug a
threshold, or otherwise be looked at during Phase 9–15 development.
`DEVELOPMENT_TICKERS` (`CORE_TICKERS`) are everything else — fine to use,
tune against, and iterate on freely.

Run this LAST, once, and report whatever it says — including a
disappointing result. If Sharpe/drawdown on `HOLDOUT_TICKERS` looks
meaningfully worse than on `DEVELOPMENT_TICKERS`, that gap IS the honest
signal that something was tuned to the development set rather than to a
real, generalizable edge — the entire reason this gate exists.

Operational note: actually executing this against real market data
requires live data access and full walk-forward retraining, both outside
what this development session can run. `run_holdout_gate` accepts
already-backtested equity curves (the same dependency-injection pattern
`scaata.strategies.evolve.evaluate_fitness_across_tickers` uses) so the
gate's logic is fully testable offline; producing real curves for
`HOLDOUT_TICKERS` via the existing walk-forward + ablation harness is an
action item for whoever runs the final validation pass, not something this
module does itself.
"""
from __future__ import annotations

import numpy as np

from scaata.config import CORE_TICKERS, NON_SURVIVOR_TICKERS
from scaata.evaluation.metrics import max_drawdown, sharpe_ratio, sortino_ratio
from scaata.evaluation.stats import block_bootstrap_ci, paired_wilcoxon

HOLDOUT_TICKERS = list(NON_SURVIVOR_TICKERS)
DEVELOPMENT_TICKERS = list(CORE_TICKERS)


def assert_no_ticker_overlap() -> None:
    """A cheap but real sanity check: if a ticker were ever added to both
    sets (e.g. a future config edit), the whole point of this gate — that
    `HOLDOUT_TICKERS` were genuinely unseen — would be silently violated.
    Called by `run_holdout_gate` before doing anything else.
    """
    overlap = set(HOLDOUT_TICKERS) & set(DEVELOPMENT_TICKERS)
    if overlap:
        raise ValueError(
            f"HOLDOUT_TICKERS and DEVELOPMENT_TICKERS overlap on {overlap} -- "
            "the holdout set must be strictly disjoint from the development set."
        )


def run_holdout_gate(
    ticker_equity_curves: dict[str, np.ndarray],
    baseline_equity_curves: dict[str, np.ndarray],
) -> dict:
    """Given already-backtested equity curves for each `HOLDOUT_TICKERS`
    ticker (the system under test) and a same-keyed baseline (e.g.
    Buy&Hold), reports Sharpe/Sortino/MaxDD with a bootstrap CI per
    ticker, and a paired Wilcoxon significance test of the system vs. the
    baseline across the holdout tickers — the same rigor already used for
    every other headline number in this project
    (`scaata.evaluation.stats`), applied one final time to data that was
    never used to make any development decision.

    Raises if either argument references any ticker outside
    `HOLDOUT_TICKERS`, and refuses to run at all if
    `HOLDOUT_TICKERS`/`DEVELOPMENT_TICKERS` have somehow come to overlap
    — this gate is only meaningful if the tickers it's run against were
    genuinely held out.
    """
    assert_no_ticker_overlap()

    unexpected = (set(ticker_equity_curves) | set(baseline_equity_curves)) - set(HOLDOUT_TICKERS)
    if unexpected:
        raise ValueError(
            f"run_holdout_gate was given ticker(s) outside HOLDOUT_TICKERS: {unexpected}. "
            "This gate exists specifically to check performance on tickers untouched by "
            "development -- passing a development ticker here defeats its purpose."
        )
    missing_baseline = set(ticker_equity_curves) - set(baseline_equity_curves)
    if missing_baseline:
        raise ValueError(f"missing baseline_equity_curves for ticker(s): {missing_baseline}")

    per_ticker_report = {}
    system_sharpes, baseline_sharpes = [], []
    for ticker, system_eq in ticker_equity_curves.items():
        baseline_eq = baseline_equity_curves[ticker]

        system_sharpe = sharpe_ratio(system_eq)
        baseline_sharpe = sharpe_ratio(baseline_eq)
        system_sharpes.append(system_sharpe)
        baseline_sharpes.append(baseline_sharpe)

        per_ticker_report[ticker] = {
            "system_sharpe": system_sharpe,
            "system_sortino": sortino_ratio(system_eq),
            "system_max_drawdown": max_drawdown(system_eq),
            "baseline_sharpe": baseline_sharpe,
            "sharpe_bootstrap_ci": block_bootstrap_ci(system_eq, sharpe_ratio),
        }

    significance = paired_wilcoxon(system_sharpes, baseline_sharpes)

    print(
        f"Holdout gate run against {list(ticker_equity_curves.keys())} -- report this result "
        "as-is, including if it looks worse than development-set numbers. This is a one-shot "
        "check: re-running it after seeing a disappointing result and then changing something "
        "defeats the entire purpose."
    )

    return {
        "holdout_tickers": list(ticker_equity_curves.keys()),
        "per_ticker": per_ticker_report,
        "vs_baseline_significance": significance,
        "mean_system_sharpe": float(np.mean(system_sharpes)),
        "mean_baseline_sharpe": float(np.mean(baseline_sharpes)),
    }
