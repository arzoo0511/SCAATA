"""Curated, deterministic strategy library (Phase 9b) — a diversity floor
for the pool so real (not purely mock) strategies exist in every run
regardless of whether GITHUB_PAT/GROQ_API_KEY are configured. Every run to
date has used the 3-strategy `scaata.strategies.scraper.MOCK_STRATEGIES`
fallback (MA-crossover, momentum, mean-reversion) because those keys were
never set live — this module adds distinct underlying logic (volatility-band
mean-reversion, breakout, oscillator-based, trend-with-volatility-gate,
volume-confirmed momentum) rather than minor variations on the same idea.

Genuinely *new* strategies beyond this fixed list are the job of
`scaata.strategies.evolve` (Phase 11) — this module intentionally stops at
hand-written textbook techniques.

Each entry follows the exact `{"source", "code"}` shape already used by
`scaata.strategies.scraper.MOCK_STRATEGIES`, so it flows unchanged through
`scaata.strategies.normalizer.validate_mock_strategies` and
`scaata.strategies.pool.run_strategy_safely`.
"""
from __future__ import annotations

LIBRARY_STRATEGIES = [
    {
        "source": "lib:bollinger_mean_reversion",
        "code": (
            "def strategy(df):\n"
            "    import numpy as np\n"
            "    mid = df['Close'].rolling(20).mean()\n"
            "    std = df['Close'].rolling(20).std()\n"
            "    upper = mid + 2 * std\n"
            "    lower = mid - 2 * std\n"
            "    sig = np.where(df['Close'] < lower, 1, np.where(df['Close'] > upper, -1, 0))\n"
            "    return np.nan_to_num(sig).astype(int)\n"
        ),
    },
    {
        "source": "lib:donchian_breakout",
        "code": (
            "def strategy(df):\n"
            "    import numpy as np\n"
            "    upper = df['Close'].rolling(20).max().shift(1)\n"
            "    lower = df['Close'].rolling(20).min().shift(1)\n"
            "    sig = np.where(df['Close'] > upper, 1, np.where(df['Close'] < lower, -1, 0))\n"
            "    return np.nan_to_num(sig).astype(int)\n"
        ),
    },
    {
        "source": "lib:rsi_overbought_oversold",
        "code": (
            "def strategy(df):\n"
            "    import numpy as np\n"
            "    delta = df['Close'].diff()\n"
            "    gain = delta.clip(lower=0).rolling(14).mean()\n"
            "    loss = (-delta.clip(upper=0)).rolling(14).mean()\n"
            "    rs = gain / (loss + 1e-8)\n"
            "    rsi = 100 - (100 / (1 + rs))\n"
            "    sig = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))\n"
            "    return np.nan_to_num(sig).astype(int)\n"
        ),
    },
    {
        "source": "lib:macd_crossover",
        "code": (
            "def strategy(df):\n"
            "    import numpy as np\n"
            "    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()\n"
            "    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()\n"
            "    macd = ema_fast - ema_slow\n"
            "    signal_line = macd.ewm(span=9, adjust=False).mean()\n"
            "    sig = np.where(macd > signal_line, 1, np.where(macd < signal_line, -1, 0))\n"
            "    return np.nan_to_num(sig).astype(int)\n"
        ),
    },
    {
        "source": "lib:atr_trend_filter",
        "code": (
            "def strategy(df):\n"
            "    import numpy as np\n"
            "    high, low, close = df['High'], df['Low'], df['Close']\n"
            "    prev_close = close.shift(1)\n"
            "    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))\n"
            "    atr = tr.rolling(14).mean()\n"
            "    trend = close - close.shift(20)\n"
            "    calm = atr < atr.rolling(60).mean()\n"
            "    sig = np.where(calm & (trend > 0), 1, np.where(calm & (trend < 0), -1, 0))\n"
            "    return np.nan_to_num(sig).astype(int)\n"
        ),
    },
    {
        "source": "lib:volume_weighted_momentum",
        "code": (
            "def strategy(df):\n"
            "    import numpy as np\n"
            "    mom = df['Close'].pct_change(10)\n"
            "    vol_ratio = df['Volume'] / (df['Volume'].rolling(30).mean() + 1e-8)\n"
            "    confirmed = vol_ratio > 1.2\n"
            "    sig = np.where((mom > 0.01) & confirmed, 1, np.where((mom < -0.01) & confirmed, -1, 0))\n"
            "    return np.nan_to_num(sig).astype(int)\n"
        ),
    },
]


def library_strategies() -> list[dict]:
    """Returns the curated strategy library — used as the *default* real
    pool for live/eval runs (unlike `scraper.mock_strategies`, which exists
    purely as an offline test fixture). GitHub-scraped strategies are
    additive on top of this when GITHUB_PAT/GROQ_API_KEY are configured."""
    return list(LIBRARY_STRATEGIES)
