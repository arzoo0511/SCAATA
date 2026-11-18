"""Phase B, episode by episode: does the rebound rule bring HANSEI back sooner after a fall?

The Lab experiment `rebound-reentry` scores the rule over two long spans. This looks at the falls themselves:
the six crises Macroverse replays plus the 2011 and 2015-16 drawdowns. For each, the live rule, the live rule
with the rebound change, and simply holding, and one number for the question asked: of the gain holding made
from its own low to the end of the episode, how much did each capture?

Eight episodes is few. The rule is one rule with settings taken from the paper and the live agent, none
searched, and the result is described, not tested.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scaata.config import ROOT_DIR
from scaata.macroverse import replay
from scaata.research import prices
from scaata.research.engine import rebound_transform

REBOUND_PATH = ROOT_DIR / "research" / "rebound.json"
EXTRA_EPISODES = [
    {"id": "euro-2011", "title": "2011 sell-off", "start": "2011-07-01", "end": "2012-03-30", "summary": "", "events": []},
    {"id": "china-2015", "title": "2015-16 China scare", "start": "2015-08-03", "end": "2016-03-31", "summary": "",
     "events": []},
]


def capture(strategy: list[float], holding: list[float]) -> float | None:
    """Of the gain holding made from its lowest point to the end, the share the strategy made over the same
    stretch. None if holding made no real gain (a stretch with nothing to recapture)."""
    s, h = np.asarray(strategy), np.asarray(holding)
    low = int(h.argmin())
    gain = h[-1] / h[low] - 1
    if gain < 0.02:
        return None
    return float((s[-1] / s[low] - 1) / gain)


def run(path: Path | None = None) -> dict:
    opens, closes, meta = prices.load()
    rows = []
    for episode in [*replay.EPISODES, *EXTRA_EPISODES]:
        base = replay.run_episode(episode, opens, closes)
        variant = replay.run_episode(episode, opens, closes, views_transform=rebound_transform)
        stats = lambda r: {"return": r["stats"]["hansei"]["return"], "worst_fall": r["stats"]["hansei"]["max_drawdown"],   # noqa: E731
                           "trades": r["stats"]["trades"]}
        rows.append({
            "id": episode["id"], "title": episode["title"], "start": base["start"], "end": base["end"],
            "live_rule": stats(base), "rebound": stats(variant),
            "holding": {"return": base["stats"]["holding"]["return"], "worst_fall": base["stats"]["holding"]["max_drawdown"]},
            "capture_live": capture(base["hansei"], base["holding"]),
            "capture_rebound": capture(variant["hansei"], variant["holding"]),
            "curves": {"dates": base["dates"][::3], "holding": base["holding"][::3], "live_rule": base["hansei"][::3],
                       "rebound": variant["hansei"][::3]}})
    usable = [r for r in rows if r["capture_live"] is not None]
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "prices_sha256": meta.get("sha256"), "episodes": rows,
           "summary": {"episodes": len(rows), "with_a_rebound": len(usable),
                       "captures_more": sum(r["capture_rebound"] > r["capture_live"] for r in usable),
                       "falls_no_worse": sum(r["rebound"]["worst_fall"] >= r["live_rule"]["worst_fall"] - 0.02 for r in rows),
                       "mean_capture_live": float(np.mean([r["capture_live"] for r in usable])) if usable else None,
                       "mean_capture_rebound": float(np.mean([r["capture_rebound"] for r in usable])) if usable else None}}
    path = path or REBOUND_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
