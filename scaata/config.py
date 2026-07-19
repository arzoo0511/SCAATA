"""Central configuration for the SCAATA v2 pipeline."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = ROOT_DIR / "data_cache"
DATA_CACHE_DIR.mkdir(exist_ok=True)

# Original 6 mega-cap tickers from the v1 paper — all big winners of 2020-2026,
# so results on these alone mostly reflect survivorship bias, not skill.
CORE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]

# Added to counter survivorship bias: a name that had a rough multi-year stretch
# in this window, a struggling retailer, and a non-US/small-cap name.
NON_SURVIVOR_TICKERS = ["XOM", "INTC", "BABA"]

ALL_TICKERS = CORE_TICKERS + NON_SURVIVOR_TICKERS

START_DATE = "2020-01-01"
END_DATE = "2026-07-19"  # today; 2026 data will be partial-year

# Train covers COVID crash + ZIRP bull + 2022 bear (all regime types for
# normalization stats); test isolates the most recent, least-seen regime.
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = END_DATE

FEATURE_COLUMNS = [
    "returns", "ma_10", "ma_50", "volatility", "momentum", "rsi", "volume_ma_30",
]

# Phase 2: adds the meta-selector's top-strategy confidence as an extra
# observation dimension (closes the v1 gap where the meta-selector's output
# was trained but never fed to the RL policy). No env code change needed
# for this — RobustTradingEnv already generalizes to len(feature_columns).
META_CONFIDENCE_COLUMN = "meta_confidence"
PHASE2_FEATURE_COLUMNS = FEATURE_COLUMNS + [META_CONFIDENCE_COLUMN]
META_ACTION_BONUS = 0.1  # small reward bonus when PPO's action agrees with the meta-selector's top strategy, ablatable

# --- Regime detection thresholds (Phase 1) ---
REGIME_VOL_WINDOW = 20          # rolling realized-vol window (days)
REGIME_VOL_MEDIAN_WINDOW = 252  # trailing window for the vol median baseline
REGIME_VOL_MULTIPLIER = 1.5     # "high-vol" if vol > 1.5x trailing median
REGIME_DD_WINDOW = 60           # rolling drawdown lookback (days)
REGIME_DD_THRESHOLD = -0.15     # "stress/bear" if drawdown < -15%
HMM_N_STATES = 3

# --- RL environment (ported from v1 notebook) ---
INITIAL_CASH = 10_000.0
BASE_TRANSACTION_FEE = 0.001
STOP_LOSS_PCT = -0.02
LIQUIDITY_FEE_FLOOR = 0.5
LIQUIDITY_FEE_CEIL = 5.0

# --- PPO training config (ported, kept as v1 defaults; notebooks can override
# with smaller values for smoke-testing) ---
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_GAMMA = 0.99
PPO_ENT_COEF = 0.01
PPO_TOTAL_TIMESTEPS = 100_000

# --- Walk-forward backtest config ---
WALKFORWARD_TRAIN_YEARS = 2.0
WALKFORWARD_TEST_MONTHS = 6
WALKFORWARD_STEP_MONTHS = 6

# --- Statistics ---
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_BLOCK_SIZE = 20  # days, for block-bootstrap of autocorrelated returns
CONFIDENCE_LEVEL = 0.95
DEFAULT_SEEDS = [0, 1, 2]

TRADING_DAYS_PER_YEAR = 252

# --- Self-critique reward shaping (Phase 2, closes the v1 wiring gap) ---
DEFAULT_LAMBDA_DD = 1.0     # drawdown-extension penalty weight
DEFAULT_LAMBDA_VOL = 0.5    # volatility-chasing-entry penalty weight
DEFAULT_LAMBDA_HOLD = 0.5   # prolonged-losing-hold penalty weight
CRITIQUE_DD_THRESHOLD = 0.05        # only penalize drawdown beyond 5%
CRITIQUE_VOL_WINDOW = 5              # recent-volatility window (days) for entry-timing penalty
CRITIQUE_VOL_BASELINE_WINDOW = 30    # trailing baseline-volatility window (days)
CRITIQUE_HOLD_GRACE_DAYS = 5         # days a losing position can be held before the penalty kicks in
REVIEW_WINDOW_DAYS = 20              # self-critique review window (matches v1 paper)

# --- LangGraph inner-loop config (Phase 2) ---
MAX_GRAPH_ITERATIONS = 4    # bounded Critique -> Meta-Selector feedback cycles (LLM call cost)
STRATEGY_DOWNWEIGHT_FACTOR = 0.5   # per-iteration multiplier applied to a poorly-performing strategy's pool weight
MIN_STRATEGY_WEIGHT = 0.05         # floor so a strategy is deprioritized, never fully deleted mid-run
CONVERGENCE_EPSILON = 0.02         # stop looping early if strategy-pool weights barely change between iterations
