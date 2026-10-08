"""The brain: turns advisor views into a decision, and decides whether it is
worth acting on.

Its default is to hold. Every test run on these five stocks found that
simply holding them beat actively trading them, so the brain starts fully
invested and only steps away when the weighted evidence is clearly against
a stock. Positive evidence brings it back to fully invested; it never
leverages.

It acts only when it has to:

- the gap between what it holds and what it now wants must be large (at
  least a third of the stock's share), so small wobbles in opinion never
  turn into trades that pay STT;
- after trading a stock it waits `MIN_HOLD_DAYS` sessions before trading it
  again, unless material news is strong enough to override that;
- if a holding drifts more than 20% away from an equal share of the book,
  it rebalances -- the one mechanical rule that edged out holding in tests.

Every decision carries its reasons in plain words, so the journal says why,
not just what.
"""
from __future__ import annotations

from dataclasses import dataclass

ACT_THRESHOLD = 1 / 3          # minimum change in exposure worth a trade
MIN_HOLD_DAYS = 10             # sessions between trades in one stock, unless news overrides
NEWS_OVERRIDE = 0.5            # |news view| at or above this may override the waiting period
DRIFT_LIMIT = 0.20             # rebalance when a holding is 20% away from its equal share
LABELS = {"trend": "trend", "volatility": "volatility", "momentum": "momentum", "news": "news"}


@dataclass
class Decision:
    symbol: str
    score: float
    target_exposure: float
    current_exposure: float
    action: str               # "BUY", "SELL", "REBALANCE", "HOLD"
    reasons: list[str]

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "score": round(self.score, 3),
                "target_exposure": round(self.target_exposure, 3),
                "current_exposure": round(self.current_exposure, 3),
                "action": self.action, "reasons": self.reasons}


def combined_score(views: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in views.items())


def target_from_score(score: float, tolerance: float = 0.0, floor: float = 0.0) -> float:
    """Hold fully unless the evidence is net negative; at -0.5 or worse, be out.
    `tolerance` starts cutting only below -tolerance, and `floor` is the least
    of its share ever held (both 0 live; the backtest's exposure sweep varies them)."""
    return max(floor, min(1.0, 1.0 + 2.0 * (score + tolerance)))


def _explain(views: dict[str, float], weights: dict[str, float]) -> list[str]:
    """The two biggest contributions to the score, in words."""
    contributions = sorted(((weights.get(k, 0) * v, k, v) for k, v in views.items()), key=lambda c: -abs(c[0]))
    reasons = []
    for contribution, name, value in contributions[:2]:
        if abs(contribution) < 0.02:
            continue
        tone = "supportive" if value > 0 else "against"
        reasons.append(f"{LABELS[name]} {tone} ({value:+.2f}, trusted {weights.get(name, 0):.0%})")
    return reasons


def decide(symbol: str, views: dict[str, float], weights: dict[str, float], current_exposure: float,
           days_since_trade: int | None, weight_in_book: float | None = None,
           equal_weight: float | None = None, act_threshold: float = ACT_THRESHOLD,
           min_hold_days: int = MIN_HOLD_DAYS, tolerance: float = 0.0, floor: float = 0.0) -> Decision:
    score = combined_score(views, weights)
    target = target_from_score(score, tolerance, floor)
    reasons = _explain(views, weights)
    gap = target - current_exposure

    strong_news = abs(views.get("news", 0.0)) >= NEWS_OVERRIDE
    waiting = days_since_trade is not None and days_since_trade < min_hold_days and not strong_news

    if abs(gap) >= act_threshold and not waiting:
        action = "BUY" if gap > 0 else "SELL"
        verb = "Adding to" if gap > 0 and current_exposure > 0 else ("Buying" if gap > 0 else
               ("Trimming" if target > 0 else "Exiting"))
        reasons.insert(0, f"{verb} {symbol.replace('.NS', '')}: {current_exposure:.0%} -> {target:.0%} of its share")
        if strong_news and days_since_trade is not None and days_since_trade < min_hold_days:
            reasons.append("material news overrode the waiting period")
        return Decision(symbol, score, target, current_exposure, action, reasons)

    if (weight_in_book is not None and equal_weight and current_exposure > 0 and target >= 0.99
            and abs(weight_in_book / equal_weight - 1) > DRIFT_LIMIT and not waiting):
        direction = "over" if weight_in_book > equal_weight else "under"
        reasons.insert(0, f"Rebalancing {symbol.replace('.NS', '')}: {direction}weight at "
                          f"{weight_in_book:.0%} of the book vs {equal_weight:.0%} target")
        return Decision(symbol, score, 1.0, current_exposure, "REBALANCE", reasons)

    if waiting and abs(gap) >= act_threshold:
        reasons.insert(0, f"Would move to {target:.0%} but traded {days_since_trade} session(s) ago -- waiting")
    else:
        reasons.insert(0, f"Holding at {current_exposure:.0%}: no strong reason to change")
    return Decision(symbol, score, target, current_exposure, "HOLD", reasons)
