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
    DSR_ETA,
    DSR_REWARD_SCALE,
    INITIAL_CASH,
    LIQUIDITY_FEE_CEIL,
    LIQUIDITY_FEE_FLOOR,
    META_ACTION_BONUS,
    STOP_LOSS_PCT,
)
from scaata.rl.reward import DifferentialSharpeTracker, SelfCritiqueTracker

HOLD, BUY, SELL = 0, 1, 2

# `include_position_obs` appends [in_position, unrealized_return, days_held / HOLDING_DAYS_SCALE].
POSITION_OBS_SIZE = 3
HOLDING_DAYS_SCALE = 20.0


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
        use_differential_sharpe: bool = False,
        dsr_eta: float = DSR_ETA,
        dsr_reward_scale: float = DSR_REWARD_SCALE,
        dsr_benchmark_relative: bool = False,
        include_position_obs: bool = False,
        random_episode_start_min_steps: int | None = None,
        fixed_fee: float = 0.0,
    ):
        super().__init__()
        self.full_df = df.copy()
        self.feature_columns = feature_columns
        self.initial_cash = initial_cash
        self.base_fee = transaction_fee
        # Charged on every trade regardless of liquidity (e.g. India's STT and
        # stamp duty), on top of the liquidity-scaled `transaction_fee`.
        self.fixed_fee = fixed_fee
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
        # Differential Sharpe Ratio reward (opt-in, default off preserves
        # exact prior behavior): replaces `percent_change * 10` below with
        # a principled risk-adjusted signal instead of raw return, since
        # raw-return-plus-hand-tuned-penalties was found (in live-data
        # testing) to collapse to a "never trade" policy at full training
        # scale. See scaata.rl.reward.differential_sharpe_reward for the
        # eta-sensitivity constraint before changing dsr_eta.
        self.use_differential_sharpe = use_differential_sharpe
        self.dsr_reward_scale = dsr_reward_scale
        self.dsr_tracker = DifferentialSharpeTracker(eta=dsr_eta) if use_differential_sharpe else None
        # Experimental (found necessary via real-data testing, not in the
        # original design): feeding the DSR tracker raw portfolio return
        # makes "buy and hold forever" a reward-maximizing attractor
        # whenever buy-and-hold happens to be a strong bet on that
        # ticker's training data (verified on AAPL) -- but on tickers
        # where buy-and-hold ISN'T strong (verified on MSFT), "never
        # trade" remains competitive and the same collapse resurfaces.
        # Feeding EXCESS return over that ticker's own buy-and-hold
        # instead removes buy-and-hold as a free-lunch attractor entirely
        # -- the policy has to find something that actually beats the
        # benchmark, on every ticker, not just replicate whatever the
        # benchmark's own average outcome was.
        self.dsr_benchmark_relative = dsr_benchmark_relative
        # Audit options (2026-09-14), both off by default so existing
        # policies are unaffected:
        # - include_position_obs: BUY and SELL only do anything depending on
        #   whether a position is open, yet the observation never said
        #   whether one was -- the LSTM had to infer it. Appends
        #   POSITION_OBS_SIZE values describing the open position.
        # - random_episode_start_min_steps: every training episode replayed
        #   a ticker's history from its first row (~64 passes over the same
        #   path at 100k steps). Starts each training episode at a random
        #   row, leaving at least this many steps. Never applies with
        #   `fixed_ticker` (backtests).
        self.include_position_obs = include_position_obs
        self.random_episode_start_min_steps = random_episode_start_min_steps

        self.action_space = spaces.Discrete(3)
        obs_size = len(feature_columns) + (POSITION_OBS_SIZE if include_position_obs else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.fixed_ticker is not None:
            chosen_ticker = self.fixed_ticker
        else:
            tickers = self.full_df["Ticker"].unique()
            chosen_ticker = self.np_random.choice(tickers)

        chosen_df = self.full_df[self.full_df["Ticker"] == chosen_ticker]
        if self.random_episode_start_min_steps and self.fixed_ticker is None:
            max_start = len(chosen_df) - self.random_episode_start_min_steps
            if max_start > 0:
                chosen_df = chosen_df.iloc[int(self.np_random.integers(0, max_start + 1)):]
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
        self.entry_step = None
        self.portfolio_value = self.cash

        if self.critique_tracker is not None:
            self.critique_tracker.reset(self.cash)

        if self.dsr_tracker is not None:
            self.dsr_tracker.reset()

        return self._get_obs(), {}

    def _get_obs(self):
        return self._obs_at(self.step_idx)

    def _obs_at(self, idx: int):
        features = self.features_array[idx].astype(np.float32)
        if not self.include_position_obs:
            return features
        if self.position == 1 and self.entry_step is not None:
            state = [1.0, self.prices[idx] / self.entry_price - 1.0, (idx - self.entry_step) / HOLDING_DAYS_SCALE]
        else:
            state = [0.0, 0.0, 0.0]
        return np.concatenate([features, np.asarray(state, dtype=np.float32)])

    def step(self, action, size_multiplier: float = 1.0):
        """`size_multiplier` (Phase 14) controls what fraction of available
        cash a BUY commits — default `1.0` reproduces the original
        all-in-on-buy behavior exactly (every existing caller, including
        SB3's own training loop which only ever passes `action`, gets
        this default). Deliberately narrow scope: this is single-shot
        sizing at entry, not a full continuous position-adjustment model —
        `self.position` stays a binary open/closed flag and a SELL still
        fully liquidates whatever `self.shares` holds, however it was
        sized at entry; you can't add to an already-open position with a
        second BUY (same as before this change). Leftover, uninvested cash
        (when `size_multiplier < 1.0`) simply stays in `self.cash` rather
        than being zeroed out, so `portfolio_value` (cash + shares*price)
        still accounts for 100% of capital at every step.
        """
        current_price = self.prices[self.step_idx]
        prev_portfolio_value = self.portfolio_value

        vol_ma = self.volumes_ma[self.step_idx]
        current_vol = self.volumes[self.step_idx] + 1e-8
        liquidity_ratio = min(max(vol_ma / current_vol, LIQUIDITY_FEE_FLOOR), LIQUIDITY_FEE_CEIL)
        dynamic_fee = self.base_fee * liquidity_ratio + self.fixed_fee

        reward = 0.0
        is_new_entry = action == BUY and self.position == 0
        if is_new_entry:
            invest_amount = self.cash * size_multiplier
            self.shares = (invest_amount * (1 - dynamic_fee)) / current_price
            self.cash = self.cash - invest_amount
            self.position = 1
            self.entry_price = current_price
            self.entry_step = self.step_idx
            if self.critique_tracker is not None:
                self.critique_tracker.on_entry(self.step_idx)
        elif action == SELL and self.position == 1:
            # += , not = : a partial-size BUY (Phase 14's size_multiplier)
            # leaves a nonzero self.cash remainder sitting alongside the
            # position -- overwriting it here would silently destroy that
            # leftover cash every time a partially-sized position gets
            # closed. Harmless no-op for the default size_multiplier=1.0
            # case (self.cash is already 0 going into this branch there).
            self.cash += (self.shares * current_price) * (1 - dynamic_fee)
            self.shares = 0.0
            self.position = 0
            self.entry_step = None
            if self.critique_tracker is not None:
                self.critique_tracker.on_exit()

        current_portfolio_value = self.cash + (self.shares * current_price)

        if self.position == 1:
            drawdown = (current_price - self.entry_price) / self.entry_price
            if drawdown <= self.stop_loss:
                self.cash += (self.shares * current_price) * (1 - dynamic_fee)  # same += fix as the SELL branch above
                self.shares = 0.0
                self.position = 0
                self.entry_step = None
                reward -= 0.05
                current_portfolio_value = self.cash
                if self.critique_tracker is not None:
                    self.critique_tracker.on_exit()

        percent_change = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        if self.dsr_tracker is not None:
            dsr_input = percent_change
            if self.dsr_benchmark_relative and self.step_idx > 0:
                benchmark_return = (current_price - self.prices[self.step_idx - 1]) / self.prices[self.step_idx - 1]
                dsr_input = percent_change - benchmark_return
            reward += self.dsr_tracker.step(dsr_input) * self.dsr_reward_scale
        else:
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
        obs = self._get_obs() if not done else self._obs_at(len(self.features_array) - 1)

        info = {
            "portfolio_value": current_portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "ticker": self.df["Ticker"].iloc[0],
        }
        if critique_breakdown is not None:
            info["critique"] = critique_breakdown
        return obs, reward, done, False, info
