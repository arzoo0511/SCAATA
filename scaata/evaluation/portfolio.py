"""Portfolio-level evaluation (Phase 15) — extends the existing
walk-forward harness (`scaata.evaluation.walkforward`) to a JOINTLY held
portfolio across multiple tickers, closing the paper's own named blind
spot: every prior phase's Sharpe/MaxDD is single-asset, so a shock
correlated across several simultaneously-held tickers never shows up in
any individual per-ticker number.

Honest scope: this is a portfolio-level EVALUATION tool, not a new
joint-action RL training target. Training an actual multi-asset PPO policy
(shared observation/action space across tickers, real capital reallocation
between them) would be a substantially larger undertaking than combining
already-independent single-asset backtests into a joint view — and
combining is enough to answer the specific question the paper's conclusion
raised ("a shock that hits multiple holdings at once is not something a
per-asset Sharpe ratio can reveal"). Given a set of already-simulated
per-ticker equity curves (e.g. from N independent `RobustTradingEnv`/PPO
backtests, one per ticker, each already accounting for its own fees/stop-
loss), `combine_into_portfolio` allocates each ticker a fixed share of one
shared notional capital base and sums them into ONE portfolio equity curve
— summing genuinely correlated series reveals joint drawdown in a way
that averaging each series' own max drawdown cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaata.config import INITIAL_CASH
from scaata.evaluation.metrics import max_drawdown, sharpe_ratio, sortino_ratio


def normalize_equity_curve(
    equity_curve: np.ndarray, allocation_fraction: float, initial_cash: float = INITIAL_CASH
) -> np.ndarray:
    """Rescales an already-simulated equity curve (which starts at its own
    full `initial_cash`) to reflect only `allocation_fraction` of one
    shared, larger capital base — e.g. an equal-weight 1/N share — while
    preserving the curve's own percentage-return path unchanged.
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    returns_multiplier = equity_curve / equity_curve[0]
    return returns_multiplier * (initial_cash * allocation_fraction)


def slice_equity_curves_to_common_length(ticker_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Trims each ticker's (test-fold) DataFrame to the shortest common
    length across all tickers, so their equity curves — once separately
    backtested — can be combined. Walk-forward test windows
    (`scaata.evaluation.walkforward.generate_folds`) are date-based, not
    row-count-based, and different tickers can have slightly different
    numbers of trading rows within the same calendar window (e.g. a
    listing gap), so an exact length match isn't guaranteed for free.
    """
    min_len = min(len(df) for df in ticker_dfs.values())
    return {ticker: df.iloc[:min_len] for ticker, df in ticker_dfs.items()}


def combine_into_portfolio(
    ticker_equity_curves: dict[str, np.ndarray],
    allocations: dict[str, float] | None = None,
    initial_cash: float = INITIAL_CASH,
) -> dict:
    """Combines N independently-simulated per-ticker equity curves (each
    already fee/stop-loss-accounted, each starting from its own full
    `initial_cash`) into one portfolio-level equity curve, allocating each
    ticker `allocations[ticker]` (default: equal-weight `1/N`) of one
    shared notional capital base. All curves must have the same length
    (same aligned date range) — see `slice_equity_curves_to_common_length`.

    Reports both the portfolio-level risk metrics and, critically, the
    "naive" average of each ticker's own max drawdown, plus a
    `diversification_effect` (`portfolio_max_drawdown -
    naive_average_max_drawdown`): positive means the portfolio's actual
    drawdown was *less* severe than a per-ticker average would suggest
    (real diversification benefit); negative means it was *more* severe
    — tickers crashed together, exactly the blind spot a per-asset Sharpe
    ratio can't reveal.
    """
    tickers = list(ticker_equity_curves.keys())
    n = len(tickers)
    if n == 0:
        raise ValueError("combine_into_portfolio requires at least one ticker")

    lengths = {len(v) for v in ticker_equity_curves.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"all per-ticker equity curves must have the same length (same aligned date range); "
            f"got lengths {sorted(lengths)}. Use slice_equity_curves_to_common_length first."
        )

    allocations = allocations or {t: 1.0 / n for t in tickers}
    if abs(sum(allocations.values()) - 1.0) > 1e-6:
        raise ValueError(f"allocations must sum to 1.0, got {sum(allocations.values())}")

    scaled_curves = {
        t: normalize_equity_curve(curve, allocations[t], initial_cash) for t, curve in ticker_equity_curves.items()
    }
    portfolio_equity = np.sum(list(scaled_curves.values()), axis=0)

    per_ticker_max_dd = {t: max_drawdown(curve) for t, curve in ticker_equity_curves.items()}
    naive_average_max_dd = float(np.mean(list(per_ticker_max_dd.values())))
    portfolio_max_dd = max_drawdown(portfolio_equity)

    return {
        "portfolio_equity_curve": portfolio_equity,
        "scaled_ticker_curves": scaled_curves,
        "portfolio_sharpe": sharpe_ratio(portfolio_equity),
        "portfolio_sortino": sortino_ratio(portfolio_equity),
        "portfolio_max_drawdown": portfolio_max_dd,
        "per_ticker_max_drawdown": per_ticker_max_dd,
        "naive_average_max_drawdown": naive_average_max_dd,
        "diversification_effect": portfolio_max_dd - naive_average_max_dd,
    }
