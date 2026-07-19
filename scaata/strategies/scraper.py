"""GitHub strategy scraper, ported from the v1 notebook.

Reads GITHUB_PAT from the environment (never hardcoded — the v1 notebook's
hardcoded token was one of the two exposed secrets flagged during the
rebuild). If no token is configured, `scrape_strategies` returns an empty
list rather than making unauthenticated (heavily rate-limited) calls, so
callers should fall back to `mock_strategies` for offline testing.
"""
from __future__ import annotations

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_API = "https://api.github.com/search/code"

DEFAULT_QUERIES = [
    "backtrader strategy",
    "trading strategy pandas signal",
    "algorithmic trading strategy python",
    "macd crossover strategy",
    "mean reversion python",
    "statistical arbitrage strategy",
]


def _github_token() -> str | None:
    return os.environ.get("GITHUB_PAT") or None


def search_github_strategies(query: str, max_results: int = 10) -> list[dict]:
    token = _github_token()
    if not token:
        print("Warning: GITHUB_PAT not set — skipping live GitHub search.")
        return []

    headers = {"Accept": "application/vnd.github.v3.text-match+json", "Authorization": f"token {token}"}
    params = {"q": f"{query} language:python", "per_page": min(max_results, 100)}

    response = requests.get(GITHUB_API, headers=headers, params=params, timeout=30)
    if response.status_code != 200:
        print(f"GitHub API error for query '{query}': {response.status_code} {response.text[:200]}")
        return []
    return response.json().get("items", [])


def fetch_raw_file(html_url: str) -> str | None:
    raw_url = html_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    response = requests.get(raw_url, timeout=30)
    return response.text if response.status_code == 200 else None


def scrape_strategies(queries: list[str] = DEFAULT_QUERIES, max_per_query: int = 15) -> list[dict]:
    """Returns a list of {"source": html_url, "code": raw_code} dicts."""
    all_codes = []
    for query in queries:
        results = search_github_strategies(query, max_per_query)
        time.sleep(2)
        for item in results:
            html_url = item.get("html_url")
            code = fetch_raw_file(html_url)
            if code and len(code) > 200:
                all_codes.append({"source": html_url, "code": code})
    return all_codes


def filter_and_dedup(codes: list[dict]) -> list[dict]:
    """Keyword-density filter + hash dedup, ported unchanged from v1."""
    keywords = ["strategy", "signal", "buy", "sell", "position", "return", "indicator"]
    filtered = [c for c in codes if sum(k in c["code"].lower() for k in keywords) >= 3]

    seen = set()
    unique = []
    for item in filtered:
        h = hash(item["code"])
        if h not in seen:
            seen.add(h)
            unique.append(item)
    return unique


MOCK_STRATEGIES = [
    {
        "source": "mock:ma_crossover",
        "code": (
            "def strategy(df):\n"
            "    import numpy as np\n"
            "    fast = df['Close'].rolling(10).mean()\n"
            "    slow = df['Close'].rolling(50).mean()\n"
            "    sig = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))\n"
            "    return np.nan_to_num(sig).astype(int)\n"
        ),
    },
    {
        "source": "mock:momentum",
        "code": (
            "def strategy(df):\n"
            "    import numpy as np\n"
            "    mom = df['Close'].pct_change(10)\n"
            "    sig = np.where(mom > 0.02, 1, np.where(mom < -0.02, -1, 0))\n"
            "    return np.nan_to_num(sig).astype(int)\n"
        ),
    },
    {
        "source": "mock:mean_reversion",
        "code": (
            "def strategy(df):\n"
            "    import numpy as np\n"
            "    z = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).std() + 1e-8)\n"
            "    sig = np.where(z < -1, 1, np.where(z > 1, -1, 0))\n"
            "    return np.nan_to_num(sig).astype(int)\n"
        ),
    },
]


def mock_strategies() -> list[dict]:
    """Small hand-written strategy set used for offline testing when no
    GITHUB_PAT/GROQ_API_KEY is configured — lets the LangGraph pipeline and
    tests run without any live API access."""
    return list(MOCK_STRATEGIES)
