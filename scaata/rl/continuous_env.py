"""ContinuousTradingEnv (Phase 19) -- addresses a structural limitation
found this session in `RobustTradingEnv`: `size_multiplier` only ever
matters at the single entry moment ("single-shot sizing at entry, not a
full continuous position-adjustment model", per that env's own docstring).
This is the concrete reason two separate sizing overlays tested this
session (volatility-targeting, ensemble-uncertainty sizing) both
underperformed -- neither could ever do more than bet once and freeze.

Published research on continuous action spaces for trading
(arxiv.org/abs/2210.03469) reports real improvements to both return and
Sharpe from giving the agent precise, adjustable position control instead
of discrete BUY/SELL/HOLD -- a genuinely different lever from "add another
overlay on top of a single-shot entry," which this session already tried
twice.

Action space: `Box(low=0.0, high=1.0, shape=(1,))` -- the target fraction
of total portfolio value to hold in the asset, long-only (matching every
other env in this project; no short-selling anywhere in this codebase).
Every step, the position is adjusted *toward* that target (partial buys
and sells both possible, not just a single all-in entry and a single
full exit), paying the same dynamic liquidity-scaled fee on whatever
dollar amount actually trades that step.

Deliberately reuses `scaata.rl.reward`'s DSR/critique machinery unchanged
-- both operate on portfolio percent-change, which is agnostic to how the
position got to its current size, so no reward-side changes were needed
to combine this with the DSR mechanism already validated this session.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from scaata.config import (
    BASE_TRANSACTION_FEE,
    DSR_ETA,
    DSR_REWARD_SCALE,
    INITIAL_CASH,
    LIQUIDITY_FEE_CEIL,
    LIQUIDITY_FEE_FLOOR,
    STOP_LOSS_PCT,
)
from scaata.rl.reward import DifferentialSharpeTracker

# A trade smaller than this fraction of portfolio value is skipped entirely
# -- otherwise a continuous policy that outputs e.g. 0.5001 vs 0.4999 on
# consecutive steps churns the account on fee-losing noise trades that
# don't reflect any real intent to change exposure.
MIN_REBALANCE_FRACTION = 0.02


class ContinuousTradingEnv(gym.Env):
    def __init__(
        self,
        df,
        feature_columns: list[str],
        initial_cash: float = INITIAL_CASH,
        transaction_fee: float = BASE_TRANSACTION_FEE,
        stop_loss_pct: float = STOP_LOSS_PCT,
        fixed_ticker: str | None = None,
        use_differential_sharpe: bool = True,
        dsr_eta: float = DSR_ETA,
        dsr_reward_scale: float = DSR_REWARD_SCALE,
        dsr_benchmark_relative: bool = False,
        min_rebalance_fraction: float = MIN_REBALANCE_FRACTION,
    ):
        super().__init__()
        self.full_df = df.copy()
        self.feature_columns = feature_columns
        self.initial_cash = initial_cash
        self.base_fee = transaction_fee
        self.stop_loss = stop_loss_pct
        self.fixed_ticker = fixed_ticker
        self.min_rebalance_fraction = min_rebalance_fraction

        self.use_differential_sharpe = use_differential_sharpe
        self.dsr_reward_scale = dsr_reward_scale
        self.dsr_tracker = DifferentialSharpeTracker(eta=dsr_eta) if use_differential_sharpe else None
        self.dsr_benchmark_relative = dsr_benchmark_relative

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_columns),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fixed_ticker is not None:
            chosen_ticker = self.fixed_ticker
        else:
            tickers = self.full_df["Ticker"].unique()
            chosen_ticker = self.np_random.choice(tickers)
        chosen_df = self.full_df[self.full_df["Ticker"] == chosen_ticker]
        return self._init_episode(chosen_df)

    def _init_episode(self, df):
        self.df = df.copy()
        self.features_array = self.df[self.feature_columns].values
        self.prices = self.df["Close"].values
        self.volumes_ma = self.df["volume_ma_30"].values
        self.volumes = self.df["Volume"].values
        self.n_steps = len(self.df)

        self.step_idx = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        # Weighted-average cost basis across all buys since the position
        # was last fully flat -- the continuous analogue of `entry_price`,
        # since there's no longer a single moment of "entry" to anchor a
        # stop-loss to once partial buys/sells are possible.
        self.cost_basis = 0.0
        self.portfolio_value = self.cash

        if self.dsr_tracker is not None:
            self.dsr_tracker.reset()

        return self._get_obs(), {}

    def _get_obs(self):
        return self.features_array[self.step_idx].astype(np.float32)

    def step(self, action):
        target_fraction = float(np.clip(action[0] if hasattr(action, "__len__") else action, 0.0, 1.0))
        current_price = self.prices[self.step_idx]
        prev_portfolio_value = self.portfolio_value

        vol_ma = self.volumes_ma[self.step_idx]
        current_vol = self.volumes[self.step_idx] + 1e-8
        liquidity_ratio = min(max(vol_ma / current_vol, LIQUIDITY_FEE_FLOOR), LIQUIDITY_FEE_CEIL)
        dynamic_fee = self.base_fee * liquidity_ratio

        pre_trade_value = self.cash + self.shares * current_price
        current_fraction = (self.shares * current_price) / pre_trade_value if pre_trade_value > 0 else 0.0
        rebalance_amount = target_fraction * pre_trade_value - self.shares * current_price

        if abs(rebalance_amount) / max(pre_trade_value, 1e-8) >= self.min_rebalance_fraction:
            if rebalance_amount > 0:
                # Buying more: can't spend more cash than we have.
                buy_amount = min(rebalance_amount, self.cash)
                new_shares = (buy_amount * (1 - dynamic_fee)) / current_price
                total_shares = self.shares + new_shares
                # Roll the cost basis forward as a shares-weighted average
                # of the old basis and this buy's price.
                self.cost_basis = (
                    (self.cost_basis * self.shares + current_price * new_shares) / total_shares
                    if total_shares > 0 else 0.0
                )
                self.shares = total_shares
                self.cash -= buy_amount
            else:
                sell_amount_shares = min(-rebalance_amount / current_price, self.shares)
                self.cash += sell_amount_shares * current_price * (1 - dynamic_fee)
                self.shares -= sell_amount_shares
                if self.shares <= 1e-8:
                    self.shares = 0.0
                    self.cost_basis = 0.0

        reward = 0.0
        # Portfolio-level stop-loss on the rolling cost basis -- the
        # continuous analogue of RobustTradingEnv's fixed-entry stop-loss:
        # forces a full exit (not just a partial trim) if the position's
        # blended cost basis has drawn down past the threshold, so a
        # continuous policy can't quietly ride a large loss down just
        # because it never crossed a single fixed entry/exit boundary.
        if self.shares > 0 and self.cost_basis > 0:
            drawdown = (current_price - self.cost_basis) / self.cost_basis
            if drawdown <= self.stop_loss:
                self.cash += self.shares * current_price * (1 - dynamic_fee)
                self.shares = 0.0
                self.cost_basis = 0.0
                reward -= 0.05

        current_portfolio_value = self.cash + self.shares * current_price

        percent_change = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        if self.dsr_tracker is not None:
            dsr_input = percent_change
            if self.dsr_benchmark_relative and self.step_idx > 0:
                benchmark_return = (current_price - self.prices[self.step_idx - 1]) / self.prices[self.step_idx - 1]
                dsr_input = percent_change - benchmark_return
            reward += self.dsr_tracker.step(dsr_input) * self.dsr_reward_scale
        else:
            reward += percent_change * 10

        self.portfolio_value = current_portfolio_value

        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1
        obs = self._get_obs() if not done else self.features_array[-1].astype(np.float32)

        info = {
            "portfolio_value": current_portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "target_fraction": target_fraction,
            "actual_fraction": (self.shares * current_price) / current_portfolio_value if current_portfolio_value > 0 else 0.0,
            "ticker": self.df["Ticker"].iloc[0],
        }
        return obs, reward, done, False, info
