"""Experiments run before the Lab existed, entered so they are not forgotten
or repeated.

Their numbers are read from the result files those runs wrote; nothing is
typed in by hand. They are marked `origin: "legacy"`: their criteria were not
filed in advance, so the reviewer's first integrity check does not apply and
they carry the conclusion reached at the time instead of a Lab verdict.
"""
from __future__ import annotations

import json

from scaata.config import ROOT_DIR
from scaata.research import registry

RESULTS = ROOT_DIR / "results"


def _read(name: str):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ppo() -> dict | None:
    summary = _read("phase1_rerun/env_fixes/summary.json")
    if not summary:
        return None
    c, means = summary["comparisons"]["ppo_vs_buy_and_hold"], summary["mean_sharpe_never_traded_as_zero"]
    return {"id": "legacy-ppo-vs-holding", "title": "A deep-RL policy (PPO) against holding",
            "at": "2026-09-16T00:00:00+00:00",
            "question": "Can a recurrent PPO policy beat buy-and-hold after costs?",
            "hypothesis": "PPO earns a higher Sharpe than buy-and-hold across walk-forward folds.",
            "method": f"{c['n_folds']} walk-forward folds x 5 seeds x 9 tickers ({summary['rows']} runs), fees on "
                      "every method, differences clustered by fold.",
            "metrics": {"ppo_mean_sharpe": means["ppo"], "holding_mean_sharpe": means["buy_and_hold"],
                        "mean_difference": c["mean_diff"], "ci_low": c["ci_low"], "ci_high": c["ci_high"],
                        "share_of_folds_ahead": c["share_folds_a_better"]},
            "verdict": "not_supported",
            "conclusion": f"PPO trailed holding by {c['mean_diff']:+.2f} Sharpe (95% CI {c['ci_low']:+.2f} to "
                          f"{c['ci_high']:+.2f}) and was ahead in {c['share_folds_a_better']:.0%} of folds. This is "
                          "why HANSEI is a simple, explainable agent.",
            "source": "results/phase1_rerun/env_fixes/summary.json"}


def _signals() -> dict | None:
    study = _read("hansei_signal_study.json")
    if not study:
        return None
    metrics, crossing, total = {}, 0, 0
    for split, horizons in study["large_caps"].items():
        for advisor, r in horizons.get("20_sessions", {}).items():
            low, high = r["spread_ci95"]
            metrics[f"{split}.{advisor}.spread"] = r["spread"]
            metrics[f"{split}.{advisor}.ci_low"], metrics[f"{split}.{advisor}.ci_high"] = low, high
            total += 1
            crossing += low <= 0 <= high
    return {"id": "legacy-advisors-as-return-signals", "title": "Do the advisors predict returns?",
            "at": study["generated_utc"],
            "question": "Across NSE large caps, do stocks an advisor likes outperform the ones it dislikes?",
            "hypothesis": "Each advisor's liked-minus-disliked return spread is positive, 20 sessions ahead.",
            "method": f"{study['stocks']} large caps, 2007 onward, tune and holdout splits, monthly-clustered "
                      "confidence intervals.",
            "metrics": metrics, "verdict": "not_supported",
            "conclusion": f"{crossing} of {total} spreads have a 95% interval that includes zero. No advisor is a "
                          "statistically reliable return signal; the volatility brake predicts bigger swings, "
                          "not lower returns.",
            "source": "results/hansei_signal_study.json"}


def _exposure() -> dict | None:
    sweep = _read("hansei_backtest_exposure.json")
    if not sweep:
        return None
    chosen, default = sweep["checks"]["chosen"], sweep["checks"]["default"]
    return {"id": "legacy-staying-more-invested", "at": sweep["generated_utc"],
            "title": f"Staying more invested (tolerance {chosen['tolerance']}, floor {chosen['floor']})",
            "question": "Would a rule that cuts exposure later, and never below half, beat the live rule?",
            "hypothesis": "A more-invested rule keeps most of the drawdown protection and earns more.",
            "method": f"{len(sweep['sweep'])} tolerance/floor settings swept on 2007-2019 with a rule fixed in "
                      f"advance ({sweep['rule']}); the chosen setting checked once on 2020 onward.",
            "metrics": {"chosen.holdout.cagr_vs_holding": chosen["holdout"]["cagr_diff"],
                        "live.holdout.cagr_vs_holding": default["holdout"]["cagr_diff"],
                        "chosen.holdout.sharpe_vs_holding": chosen["holdout"]["sharpe_diff"],
                        "live.holdout.sharpe_vs_holding": default["holdout"]["sharpe_diff"]},
            "verdict": "inconclusive",
            "conclusion": f"On 2020 onward it beat holding by {chosen['holdout']['cagr_diff']:+.2%} a year against "
                          f"the live rule's {default['holdout']['cagr_diff']:+.2%}, with a lower Sharpe edge "
                          f"({chosen['holdout']['sharpe_diff']:+.2f} vs {default['holdout']['sharpe_diff']:+.2f}). "
                          "A later check on the falling Jul 2024 onward window had it behind the live rule. "
                          "Not switched on.",
            "source": "results/hansei_backtest_exposure.json"}


def _news() -> dict | None:
    report = _read("hansei_backtest_news.json")
    if not report:
        return None
    key = next((k for k in report["periods"] if "true test" in k), None)
    if not key:
        return None
    test, signal = report["periods"][key], report["news_signal"][key]["20_sessions"]
    return {"id": "legacy-news-as-an-advisor", "title": "Does reading the news help?", "at": report["generated_utc"],
            "question": "Does adding scored headlines improve the agent, on news the scoring model could not "
                        "have seen in training?",
            "hypothesis": "With news, HANSEI earns more than without it over Jul 2024 onward.",
            "method": "The agent replayed with and without archived, model-scored headlines over the same days; "
                      "the news view also scored on its own.",
            "metrics": {"cagr_with_minus_without": test["cagr_diff"], "sharpe_with_minus_without": test["sharpe_diff"],
                        "prob_news_better": test["prob_news_better"], "news_view_hit_rate": signal["hit_rate"],
                        "news_opinions": signal["opinions"]},
            "verdict": "not_supported",
            "conclusion": f"News added {test['cagr_diff']:+.2%} a year, with a {test['prob_news_better']:.0%} chance "
                          f"of being better than no news: noise. On its own the news view was right "
                          f"{signal['hit_rate']:.0%} of the time over {signal['opinions']:,} opinions. It stays "
                          "live because it is cheap and its live record is short.",
            "source": "results/hansei_backtest_news.json"}


def _earnings() -> dict | None:
    report = _read("hansei_backtest_results.json")
    if not report:
        return None
    tune, hold = report["tune"], report["checks"]["Holdout 2020-"]
    chosen = tune[f"hold_{report['chosen_hold']}"]
    return {"id": "legacy-earnings-surprise-advisor", "title": "An earnings-surprise advisor",
            "at": report["generated_utc"],
            "question": "Should a fifth advisor react to results that beat or miss estimates?",
            "hypothesis": "Adding it raises returns against the four-advisor rule in tune and holdout.",
            "method": f"{report['events']} results announcements for the five stocks; the hold period chosen on "
                      "2007-2019, then checked on 2020 onward across the settings grid.",
            "metrics": {"tune.cagr_with_minus_without": chosen["cagr_diff"] - tune["without"]["cagr_diff"],
                        "holdout.cagr_with_minus_without": hold["with"]["cagr_diff"] - hold["without"]["cagr_diff"],
                        "holdout.prob_better": hold["prob_better"],
                        "holdout.grid_settings_better": sum(v > 0 for v in hold["grid_cagr_diff"]),
                        "holdout.grid_settings": len(hold["grid_cagr_diff"])},
            "verdict": "not_supported",
            "conclusion": f"It lowered 2007-2019 returns by "
                          f"{chosen['cagr_diff'] - tune['without']['cagr_diff']:+.2%} a year and helped in "
                          f"{sum(v > 0 for v in hold['grid_cagr_diff'])} of {len(hold['grid_cagr_diff'])} holdout "
                          "settings. Small cross-sectional edges do not survive a hold-by-default design. Not added.",
            "source": "results/hansei_backtest_results.json"}


def entries() -> list[dict]:
    return [e for e in (_ppo(), _signals(), _exposure(), _news(), _earnings()) if e]


def seed(path=None) -> list[str]:
    """Enters every legacy experiment that is not already on file. Returns the ids added."""
    known = {r["id"] for r in registry.load(path)}
    added = []
    for e in entries():
        if e["id"] in known:
            continue
        registry.register({"id": e["id"], "title": e["title"], "question": e["question"],
                           "hypothesis": e["hypothesis"], "kind": "legacy", "method": e["method"],
                           "criteria": [{"metric": "n/a", "op": ">", "value": 0, "label": "not filed in advance"}],
                           "limitations": ["Run before the Lab existed: its criteria were not filed in advance, "
                                           "and it did not go through the reviewer."],
                           "origin": "legacy", "source": e["source"]}, path, at=e["at"])
        registry.record_result(e["id"], {"metrics": e["metrics"], "source": e["source"]}, path, at=e["at"])
        registry.record_review(e["id"], {"verdict": e["verdict"], "summary": e["conclusion"], "checks": [],
                                         "criteria": [], "legacy": True}, path, at=e["at"])
        added.append(e["id"])
    return added
