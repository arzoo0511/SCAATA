"""RobustTradingEnv, ported from the v1 notebook, extended in Phase 2 to
close the self-critique wiring gap.

Base mechanics are unchanged from v1: dynamic liquidity-scaled transaction
fees and a hard stop-loss. `enable_self_critique` (default False) adds the
drawdown/volatility-chasing/holding-time penalty terms from
`scaata.rl.reward` into the reward — defaulting to off keeps Phase 1
baselines exactly reproducible for a fair ablation/before-after comparison.
The meta-selector signal (gap 3) needs no structural change here: it's
merged into the DataFrame as an ordinary column upstream and simply added
to `feature_columns`, since the observation space already generalizes to
`len(feature_columns)`.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from scaata.config import (
    BASE_TRANSACTION_FEE,
    DEFAULT_LAMBDA_DD,
    DEFAULT_LAMBDA_HOLD,
    DEFAULT_LAMBDA_VOL,
    INITIAL_CASH,
    LIQUIDITY_FEE_CEIL,
    LIQUIDITY_FEE_FLOOR,
    META_ACTION_BONUS,
    STOP_LOSS_PCT,
)
from scaata.rl.reward import SelfCritiqueTracker

HOLD, BUY, SELL = 0, 1, 2


class RobustTradingEnv(gym.Env):
    def __init__(
        self,
        df,
        feature_columns: list[str],
        initial_cash: float = INITIAL_CASH,
        transaction_fee: float = BASE_TRANSACTION_FEE,
        stop_loss_pct: float = STOP_LOSS_PCT,
        fixed_ticker: str | None = None,
        enable_self_critique: bool = False,
        lambda_dd: float = DEFAULT_LAMBDA_DD,
        lambda_vol: float = DEFAULT_LAMBDA_VOL,
        lambda_hold: float = DEFAULT_LAMBDA_HOLD,
        meta_action_bonus_col: str | None = None,
        meta_action_bonus: float = META_ACTION_BONUS,
    ):
        super().__init__()
        self.full_df = df.copy()
        self.feature_columns = feature_columns
        self.initial_cash = initial_cash
        self.base_fee = transaction_fee
        self.stop_loss = stop_loss_pct
        self.fixed_ticker = fixed_ticker
        self.enable_self_critique = enable_self_critique
        self.critique_tracker = (
            SelfCritiqueTracker(lambda_dd=lambda_dd, lambda_vol=lambda_vol, lambda_hold=lambda_hold)
            if enable_self_critique
            else None
        )
        self.meta_action_bonus_col = meta_action_bonus_col
        self.meta_action_bonus = meta_action_bonus

        self.action_space = spaces.Discrete(3)
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
        """Sets up all per-episode state from a chosen (already-selected)
        contiguous DataFrame slice. Factored out so subclasses that pick
        episodes differently (e.g. `SegmentedTradingEnv` in
        `scaata.rl.moe.train_experts`, which samples a contiguous
        same-regime segment instead of a full ticker series) only need to
        override *which* slice is chosen, not re-duplicate this setup —
        duplicating it previously caused a real bug (a subclass missing the
        `meta_actions` line and crashing at the first `step()` call).
        """
        self.df = df.copy()
        self.features_array = self.df[self.feature_columns].values
        self.prices = self.df["Close"].values
        self.volumes_ma = self.df["volume_ma_30"].values
        self.volumes = self.df["Volume"].values
        self.n_steps = len(self.df)
        self.meta_actions = (
            self.df[self.meta_action_bonus_col].values if self.meta_action_bonus_col else None
        )

        self.step_idx = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.portfolio_value = self.cash

        if self.critique_tracker is not None:
            self.critique_tracker.reset(self.cash)

        return self._get_obs(), {}

    def _get_obs(self):
        return self.features_array[self.step_idx].astype(np.float32)

    def step(self, action):
        current_price = self.prices[self.step_idx]
        prev_portfolio_value = self.portfolio_value

        vol_ma = self.volumes_ma[self.step_idx]
        current_vol = self.volumes[self.step_idx] + 1e-8
        liquidity_ratio = min(max(vol_ma / current_vol, LIQUIDITY_FEE_FLOOR), LIQUIDITY_FEE_CEIL)
        dynamic_fee = self.base_fee * liquidity_ratio

        reward = 0.0
        is_new_entry = action == BUY and self.position == 0
        if is_new_entry:
            self.shares = (self.cash * (1 - dynamic_fee)) / current_price
            self.cash = 0.0
            self.position = 1
            self.entry_price = current_price
            if self.critique_tracker is not None:
                self.critique_tracker.on_entry(self.step_idx)
        elif action == SELL and self.position == 1:
            self.cash = (self.shares * current_price) * (1 - dynamic_fee)
            self.shares = 0.0
            self.position = 0
            if self.critique_tracker is not None:
                self.critique_tracker.on_exit()

        current_portfolio_value = self.cash + (self.shares * current_price)

        if self.position == 1:
            drawdown = (current_price - self.entry_price) / self.entry_price
            if drawdown <= self.stop_loss:
                self.cash = (self.shares * current_price) * (1 - dynamic_fee)
                self.shares = 0.0
                self.position = 0
                reward -= 0.05
                current_portfolio_value = self.cash
                if self.critique_tracker is not None:
                    self.critique_tracker.on_exit()

        percent_change = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward += percent_change * 10

        critique_breakdown = None
        if self.critique_tracker is not None:
            critique_breakdown = self.critique_tracker.step_penalty(
                current_value=current_portfolio_value,
                step_idx=self.step_idx,
                position=self.position,
                entry_price=self.entry_price,
                current_price=current_price,
                is_new_entry=is_new_entry,
            )
            reward += critique_breakdown["total"]

        if self.meta_actions is not None and int(action) == int(self.meta_actions[self.step_idx]):
            reward += self.meta_action_bonus

        self.portfolio_value = current_portfolio_value

        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1
        obs = self._get_obs() if not done else self.features_array[-1].astype(np.float32)

        info = {
            "portfolio_value": current_portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "ticker": self.df["Ticker"].iloc[0],
        }
        if critique_breakdown is not None:
            info["critique"] = critique_breakdown
        return obs, reward, done, False, info
