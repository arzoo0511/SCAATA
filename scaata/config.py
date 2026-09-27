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

# --- Indian market (NSE, via yfinance ".NS" symbols) ---
INDIA_TICKERS = ["HDFCBANK.NS", "ICICIBANK.NS", "PETRONET.NS", "IOC.NS", "ITC.NS"]
INDIA_BENCHMARK = "^NSEI"  # Nifty 50
# Statutory cost of an NSE equity-delivery trade, as a fraction of traded
# value, assuming a zero-brokerage delivery broker: STT 0.1% on each side,
# stamp duty 0.015% on buys, NSE transaction charge 0.00297%, SEBI fee
# 0.0001%, GST 18% on (transaction charge + SEBI fee). Buy side ~0.1186%,
# sell side ~0.1036%. The flat per-sell DP charge (~Rs 15.9) is not modelled.
INDIA_STATUTORY_COST_BUY = 0.001 + 0.00015 + 0.0000297 + 0.000001 + 0.18 * (0.0000297 + 0.000001)
INDIA_STATUTORY_COST_SELL = 0.001 + 0.0000297 + 0.000001 + 0.18 * (0.0000297 + 0.000001)
INDIA_STATUTORY_COST_PER_SIDE = (INDIA_STATUTORY_COST_BUY + INDIA_STATUTORY_COST_SELL) / 2
INDIA_SPREAD_COST = 0.0005  # spread/impact on liquid large caps, scaled by liquidity in the env like BASE_TRANSACTION_FEE

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

# Scale-free alternative to FEATURE_COLUMNS (audit, 2026-09-14): the same seven
# signals with the three raw price/volume levels replaced by ratios. Opt-in;
# every deployed policy was trained on FEATURE_COLUMNS.
STATIONARY_FEATURE_COLUMNS = [
    "returns", "close_to_ma_10", "close_to_ma_50", "volatility", "momentum", "rsi", "volume_ratio_30",
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
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

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

# --- Phase 9: wiring sentiment into the actual decision path (was previously
# computed but only ever read by scaata.thirdeye's standalone narrative/
# correlation modules) + richer raw signals feeding Phase 10's novelty
# detector ---
SENTIMENT_SCORE_COLUMN = "sentiment_score"
PHASE9_FEATURE_COLUMNS = PHASE2_FEATURE_COLUMNS + [SENTIMENT_SCORE_COLUMN]

VIX_TICKER = "^VIX"

PUTCALL_LAG_DAYS = 1        # lag the CBOE put/call ratio by 1 day before use — a macro series is an easy off-by-one leak
CBOE_PUTCALL_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/totalpc.csv"

SENTIMENT_VELOCITY_WINDOW = 3        # rolling window (days) for rate-of-change of sentiment tone
MENTION_VOLUME_VELOCITY_WINDOW = 3   # rolling window (days) for rate-of-change of news coverage volume

MICROSTRUCTURE_CS_WINDOW = 1         # Corwin-Schultz spread uses a 2-day (t-1,t) high/low pair per estimate
AMIHUD_ILLIQUIDITY_WINDOW = 20       # rolling window (days) for the Amihud illiquidity ratio

# --- Phase 10: continuous novelty detector (replaces/complements the 4
# discrete regime buckets as the thing that modulates signal weighting) ---
NOVELTY_REF_WINDOW = 60              # trailing days used to build the Mahalanobis reference distribution
NOVELTY_RIDGE_EPS = 1e-6             # covariance-matrix ridge regularization to keep it invertible
NOVELTY_RECENT_WINDOW = REVIEW_WINDOW_DAYS   # "recent" window for the recent-vs-historical discriminator
NOVELTY_HISTORICAL_WINDOW = REGIME_VOL_MEDIAN_WINDOW  # "historical" window for the same discriminator
NOVELTY_REFIT_EVERY_DAYS = 5         # how often the discriminator is refit
NOVELTY_AUC_TRUST_THRESHOLD = 0.55   # below this, the discriminator's score isn't trusted (near coin-flip)

# Phase 10 wiring fix: novelty_score existed since Phase 10 but was only
# ever consumed ad hoc inside critique_node to modulate Hedge's eta -- never
# fed to the RL policy's own observation space. This closes that gap the
# same way Phase 9 did for sentiment.
NOVELTY_SCORE_COLUMN = "novelty_score"
PHASE10_FEATURE_COLUMNS = PHASE9_FEATURE_COLUMNS + [NOVELTY_SCORE_COLUMN]

# Meta-selector retrain: feeds it the richest feature set that's actually
# live and real right now (microstructure is self-contained, novelty is
# already causal and computed) -- sentiment is deliberately excluded here
# since no live GDELT/FinBERT source has ever been configured this session,
# so a "sentiment_score" column would just be synthetic mock noise, not a
# genuine enrichment. VIX/put-call were tried too but are excluded from the
# default set below since both external sources are currently unreachable
# (VIX: yfinance rate-limited; CBOE put/call: endpoint now returns 403/
# AccessDenied) -- the wiring degrades gracefully rather than crashing, but
# a constant neutral-filled column adds zero information, so it's left out
# of the default rather than silently padding the input with a no-op.
MICROSTRUCTURE_FEATURE_COLUMNS = ["cs_spread", "amihud_illiq"]
META_SELECTOR_RICH_FEATURE_COLUMNS = FEATURE_COLUMNS + MICROSTRUCTURE_FEATURE_COLUMNS + [NOVELTY_SCORE_COLUMN]
META_SELECTOR_HIDDEN_DIMS = [256, 128, 64]  # up from the original single 128-unit layer
META_SELECTOR_EPOCHS = 40                   # up from 15
META_SELECTOR_VAL_FRAC = 0.2                # time-based held-out split for the retrain's honest accuracy check

# --- Phase 10: Hedge / multiplicative-weights signal combiner (replaces
# critique_node's single binary "if net-negative, halve the culprit's
# weight" rule with an auditable per-expert regret-minimizing update) ---
HEDGE_LOSS_RHO = 0.3                 # blend weight between direction-call loss and self-critique-penalty loss
NOVELTY_ETA_KAPPA = 1.0              # how much novelty accelerates Hedge's learning rate (eta_t = eta*(1+kappa*novelty))
NOVELTY_SENTIMENT_PRIOR_GAMMA = 0.5  # log-prior strength nudging weight toward sentiment when novelty is high
HEDGE_COMBINE_COLUMN = "hedge_combined_signal"

# Phase 16: closes the critique -> RL feedback loop (previously one-way --
# critique's Hedge weights fed the meta-selector's next training run, but
# the RL policy itself was trained once and frozen, never made accountable
# within the same regret-tracked Hedge system). A trained policy's own
# implied action sequence (scaata.rl.policy_eval.compute_rl_implied_signals)
# can be attached to train_df under this column, at which point
# critique_node treats it as one more expert -- its realized performance
# becomes part of the same auditable weight/regret bookkeeping every
# strategy-pool expert already gets.
RL_IMPLIED_SIGNAL_COLUMN = "rl_implied_signal"

# Devil's-advocate (Phase 12) extension: normally only diffs the chosen
# pick against the runner-up. When an external caller has already computed
# ensemble disagreement (Phase 14's 5-seed uncertainty) and passes it into
# state, a full ranking of every expert's counterfactual is worth the extra
# compute specifically because that's when a single runner-up comparison is
# least trustworthy -- above this threshold, devils_advocate_node computes
# the full ranking instead of just top-2.
ENSEMBLE_DISAGREEMENT_DEEP_DIVE_THRESHOLD = 0.3

# --- Phase 11: genetic-programming strategy evolution (grows the strategy
# pool beyond the fixed scraped/hand-written set) ---
ENABLE_STRATEGY_EVOLUTION = True     # turned on once the reward-mechanism work (DSR, benchmark-relative)
                                      # was exhausted and real-data testing pointed at strategy-pool
                                      # diversity, not the reward function, as the remaining bottleneck
EVOLUTION_POPULATION_SIZE = 40
EVOLUTION_N_GENERATIONS = 15
EVOLUTION_ELITE_FRAC = 0.2
EVOLUTION_MUTATION_RATE = 0.3
EVOLUTION_VAL_FRAC = 0.3             # time-based (not random) in-sample/held-out split for the overfitting-gap check
EVOLUTION_MAX_TREE_DEPTH = 3         # capped deliberately: overfitting risk grows faster than the test window can validate away
EVOLUTION_RANDOM_IMMIGRANT_FRAC = 0.1
EVOLUTION_TOP_K_TO_POOL = 5          # only the K best (by held-out val_fitness) evolved candidates join the live pool

# --- Phase 17: equity cash-and-carry via put-call parity (conversion/
# reversal arbitrage). An earlier crypto perp-spot funding-rate version
# (Phase 13) was removed per explicit user request -- equity-only, not
# crypto. Uses Alpaca's own options-chain data (already connected, no new
# account) and yfinance's ^IRX (13-week T-bill yield, already a project
# dependency) as the risk-free-rate reference.
EQUITY_CARRY_SYMBOLS = CORE_TICKERS   # mega-caps: most liquid options chains, tightest bid-ask spreads
EQUITY_CARRY_MIN_DAYS_TO_EXPIRY = 14   # caught live: at T=2 days, implied_financing_rate divides by a ~0.005-year T,
                                       # so a routine few-cent bid-ask spread blows up into several *points* of
                                       # annualized rate noise -- a real scan showed 5/6 mega-caps "flagged" purely
                                       # from this, not genuine mispricing. A floor on tenor keeps quote noise from
                                       # dominating the signal.
EQUITY_CARRY_MAX_DAYS_TO_EXPIRY = 45  # near-term only: most liquid, and minimizes (not eliminates) dividend distortion
EQUITY_CARRY_RATE_THRESHOLD = 0.03    # 300bps annualized implied-vs-actual gap before flagging an "opportunity" --
                                       # a starting estimate, not backtested (no free historical options data source
                                       # found yet); options bid-ask spreads alone can create noise on this order, so
                                       # treat this as a first, conservative guess to revisit once real scans accumulate

# --- Phase 18: scheduled retraining -- the live daily signal was found to
# call .predict() on a policy trained once, this session, and frozen
# forever; nothing in the live path ever re-trains on new data. This closes
# that gap: periodically retrain each ticker on fresh data, but only ever
# replace the live policy if the fresh candidate doesn't measurably regress
# against what's currently deployed, checked on a real recent window
# neither model trained on.
RETRAIN_GATE_HOLDOUT_ROWS = 252       # trading rows (~1 year) at the END OF THE DATA, never trained on by the candidate;
                                       # candidate, incumbent and buy-and-hold are all backtested on this same window.
                                       # Was 60 calendar days (~39 rows): annualized-Sharpe standard error ~2.5, so
                                       # every deploy decision was noise. 252 rows brings that to ~1.0.
RETRAIN_MIN_TRAIN_ROWS = 252          # skip (touch nothing) if less than a year is left to train on
RETRAIN_GATE_MIN_PROB_VS_INCUMBENT = 0.6     # replace the incumbent only if a paired block bootstrap says the candidate's
                                             # holdout Sharpe beats it with at least this probability; ties keep the
                                             # incumbent, so noise can't churn the live policy week to week
RETRAIN_GATE_MIN_PROB_VS_BUY_AND_HOLD = 0.2  # ...and only if it isn't confidently worse than buy-and-hold
RETRAIN_MAX_DATA_STALENESS_DAYS = 10  # newest bar older than this (every data source failed) -> evaluate and report,
                                       # never deploy
# Reward config for retraining: the one multi-seed validation actually tested
# (MSFT, 5 seeds: mean Sharpe 0.08 vs -1.47 for plain DSR on one holdout window
# -- weak evidence, but the plain-DSR config it replaces did worse). Deployed
# policies before 2026-09-14 were trained without these two settings.
RETRAIN_DSR_BENCHMARK_RELATIVE = True
RETRAIN_ENT_COEF = 0.05

# --- Phase 14: ensemble uncertainty-aware sizing + continuous position
# sizing. Activates DEFAULT_SEEDS (defined above, Statistics section),
# which is otherwise unused anywhere in the codebase before this phase. ---
ENSEMBLE_ABSTAIN_AGREEMENT = 0.6   # below this fraction of members agreeing, abstain (force HOLD) rather than guess
ENSEMBLE_KAPPA = 1.0               # how strongly epistemic uncertainty shrinks position size
ENSEMBLE_MIN_SIZE = 0.1            # floor so nonzero uncertainty never fully zeroes out an agreed-upon trade

# --- Differential Sharpe Ratio reward (Moody & Saffell 2001) -- an
# alternative to the raw-return-plus-hand-tuned-penalties reward, directly
# optimizing risk-adjusted return step by step instead of proxying it via
# separately-tuned penalty coefficients that can (and empirically did, in
# live-data testing) dominate raw return and push training toward a
# degenerate "never trade" policy. Opt-in (RobustTradingEnv defaults to the
# original reward) so this is a genuine ablation arm, not a silent change. ---
DSR_ETA = 0.005               # adaptation rate for the running return-moment estimates -- MUST stay small (empirically
                               # verified correct below eta~0.005 for ~200-step sequences; verified WRONG-SIGNED, i.e.
                               # rewarding higher variance at the same mean, for eta>=0.01) -- see the eta-sensitivity
                               # note in scaata/rl/reward.py's differential_sharpe_reward docstring before changing this.
DSR_REWARD_SCALE = 100.0      # scales the raw DSR value to a magnitude comparable to the existing percent_change*10 reward
DSR_VARIANCE_EPSILON = 1e-12  # below this running-variance estimate, the ratio is undefined; reward defined as 0 ("no information yet")
DSR_WARMUP_STEPS = 20         # first N steps of each episode return reward 0 (A/B still update) -- found empirically
                               # via real-env testing that a few steps into an episode, the EMA variance estimate can
                               # be a tiny nonzero number (not caught by DSR_VARIANCE_EPSILON) that still explodes the
                               # 1/variance**1.5 term into a multi-million-magnitude reward. This is the well-known
                               # DSR "burn-in" instability, not a formula bug -- the estimate just isn't reliable yet.
DSR_REWARD_CLIP = 10.0        # hard clip on the raw (pre-scale) DSR reward -- defense-in-depth against rarer
                               # post-warmup spikes (e.g. a stop-loss-triggered large loss during an otherwise-calm
                               # stretch), since warmup alone only guards the start of each episode

# --- Volatility-targeting position sizing -- a mechanical, non-RL lever:
# size positions inversely to recent realized volatility so the return
# stream has roughly constant risk over time, instead of trying to predict
# direction better. Composes with RobustTradingEnv's size_multiplier
# (Phase 14) on top of ANY existing signal source (rule-based, evolved,
# or a trained RL policy's actions) -- doesn't require retraining anything. ---
VOL_TARGET_WINDOW = 10        # trailing window (days) for realized volatility -- matches FEATURE_COLUMNS' own "volatility"
VOL_TARGET_DAILY = 0.015      # target daily volatility (~1.5%, roughly a "calm" range for a typical large-cap)
VOL_TARGET_MIN_SIZE = 0.1     # floor so elevated volatility never fully zeroes out a position
VOL_TARGET_MAX_SIZE = 1.0     # ceiling -- long-only, no leverage

# --- Live daily-signal sizing safety ---
# The signal job runs pre-market, so the newest COMPLETE daily bar is the
# previous session's -- `daily_signal.current_price` is therefore always a
# stale close, while the fill happens hours later. Sizing straight off that
# stale price means an overnight gap up can push the suggested order past
# its allocated share of cash (observed live: a GOOGL BUY sized at $33,251
# against a $33,333 allocation -- a ~1% gap would already overflow it, and
# several simultaneous BUYs gapping together can overflow the account).
# Sizing against `price * (1 + this)` reserves headroom for that move.
#
# This is a buffer, NOT a live re-quote: pre-market there is no fresh trade
# price to re-quote against, so it deliberately trades a slightly smaller
# position for not overspending. 3% comfortably covers ordinary overnight
# moves on mega-caps; it does not pretend to cover an earnings gap.
SIGNAL_PRICE_GAP_BUFFER = 0.03

# --- Phase 19: client-facing signal product -- distributing the same
# model signal to paying subscribers, decoupled from any single account's
# cash/position (see scaata.live.daily_signal.impersonal_view,
# scaata.product). Every subscriber-facing surface (email, API response)
# carries this disclaimer verbatim.
#
# This is NOT a substitute for actual legal review. It's a structural
# nudge toward the "impersonal, non-tailored to any individual's
# circumstances" signal shape that publisher's-exemption-style analyses
# look for under the U.S. Investment Advisers Act -- it is not a legal
# conclusion that this product qualifies, and other jurisdictions have
# their own rules entirely. Get a real securities lawyer's sign-off before
# taking a single paying client's money on this. Flagged, not solved.
SIGNAL_DISCLAIMER = (
    "Informational/educational only. Not personalized investment advice, "
    "not a recommendation to buy or sell any security, and not issued by "
    "a registered investment adviser or broker-dealer. Model output can "
    "be wrong; trading involves risk of loss. You are solely responsible "
    "for your own trading decisions."
)

# Same subscriber-tickers-only shape for every user of this deployment --
# a subscriber cannot ask to watch a ticker this deployment doesn't
# actually trade, so the API validates against this rather than trusting
# arbitrary client input.
SUBSCRIBABLE_TICKERS = ALL_TICKERS

PRODUCT_DB_PATH = ROOT_DIR / "product.db"
