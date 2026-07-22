# SCAATA v2

Self-Critiquing Autonomous Algorithmic Trading Agent — a v2 rebuild of the original SCAATA research notebook and paper.

## The actual contribution, in plain terms

The original v1 notebook trained a behavior-cloning (BC) model and a meta-strategy-selector classifier, and the accompanying paper describes a self-critique loop that reshapes the PPO agent's reward based on realized portfolio risk. **None of that was actually wired in.** The BC model and meta-selector were trained, saved to disk, and never loaded again; the reward function was just `%change * 10` plus a stop-loss penalty — no self-critique terms at all. The self-critiquing mechanism the paper is named after was described, not implemented.

This rebuild closes that gap for real:
- Self-critique penalty terms (drawdown extension, volatility-chasing entries, prolonged losing-position holding) are computed causally, step-by-step, and actually added to the RL reward.
- The BC model's trained weights are actually loaded into the RL policy before fine-tuning.
- The meta-selector's output actually reaches the RL agent's observation.
- All three fixes are regression-tested (`tests/test_reward.py`, `tests/test_policy_init.py`), so this can't silently regress back into "described but not wired in."

That's the core, verified contribution. Everything else below (regime segmentation, a real multi-agent pipeline, news sentiment, a mixture-of-experts variant) is a scoped extension on top of it — some core to the rebuild, some explicitly a lower-priority stretch. See the status table below for which is which and what's real vs. placeholder right now.

## Vocabulary decoder

The architecture diagrams use a few informal names. Plain technical translation, first use:

| Diagram name | What it actually is |
|---|---|
| Outer loop | The reinforcement-learning environment and training loop (market data → features → regime detection → RL policy → evaluation) |
| Inner loop | A LangGraph multi-agent pipeline that scrapes candidate trading strategies from GitHub, normalizes them via an LLM, trains a meta-selector to weight them, and down-weights poor performers via a critique step |
| 3rd eye | A news-sentiment analysis module (GDELT + FinBERT) that both feeds a sentiment feature into the RL agent and produces a standalone narrative comparing the sentiment-market relationship across time |
| Knowledge base | A shared store of strategy weights, regime labels, critique history, and trained model artifacts, referenced by all three pieces above |

## Status per phase

| Phase | Scope | Status |
|---|---|---|
| 1 — Data, regimes, honest baselines | Core | **Full-scale run complete.** 9 tickers × 9 walk-forward folds × 5 seeds × 100k PPO timesteps/seed. PPO beats buy-and-hold significantly (Wilcoxon p=0.0021); not significantly different from rule-based (p=0.833) or the LLM-agent baseline (p=0.209, which ran on the mock fallback — no live `GROQ_API_KEY`). |
| 2 — Multi-agent inner loop + closed wiring gaps | Core | **Full-scale run complete.** Mechanism fully implemented and unit-tested. Robustness test confirms the self-critique loop down-weights poisoned strategies (mean weight 0.667 vs. 0.500). The ablation table is a single seed/ticker result — read as evidence the mechanism changes behavior, not as the performance verdict (Phase 1 is). Strategy pool uses 3 mock strategies (no live `GITHUB_PAT`/`GROQ_API_KEY` configured). |
| 3 — 3rd eye sentiment | Core | Query construction and causal trading-day alignment are implemented and unit-tested. Live GDELT access is rate-limited from this project's dev sandbox (confirmed live via direct API calls returning HTTP 429, even with 6+ second spacing) — the notebook runs on synthetic sentiment data as a result. Narrative generation uses a template fallback (no live `GROQ_API_KEY`). |
| 4 — Mixture of experts (calm/stress) | Stretch, lower priority | **Full-scale run complete.** Gate accuracy 0.884 vs. causal labels, and is structurally confirmed never to see the true regime label at eval time. MoE finished at 9,486 equity vs. the single-policy baseline's 10,700 — a real result, but every regime row except `calm_bull` is small-sample (≤34 days), so read it as illustrating the mechanism, not a validated performance claim. |
| 5 — Research rigor (threats doc, agent comparison) | Core deliverable, no training | Threats-to-validity document complete. Monolithic-vs-decomposed agent comparison harness works end-to-end but currently runs on a mock LLM weight (no `GROQ_API_KEY`) — not yet a real finding. |
| 8 — Live paper-trading forward test | Stretch | **Real, live, and running.** Connected to a real Alpaca paper-trading account (simulated money, real market data, real broker order acceptance). A frozen policy has placed real orders confirmed accepted by Alpaca (`OrderStatus.ACCEPTED`, real order IDs). Idempotency-guarded so re-running the notebook never places a second real order for the same market day. This is what used to be listed under Future Work below — it is no longer a future item. |

This table reflects the most recent full-scale runs. Check each notebook's own status banner if you re-run it yourself, since re-running with a live `GROQ_API_KEY`/`GITHUB_PAT` would change what's marked mock above.

## Running it yourself

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m pytest tests/ -v
```

Open any notebook in `notebooks/` with the `scaata-venv` Jupyter kernel. Each has a `SMOKE_TEST` (and, for Phase 3, `USE_MOCK_DATA`) flag near the top — `True` runs a fast sanity check in a few minutes; `False` runs the real, compute-heavy version. Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`/`GITHUB_PAT` for live strategy scraping and LLM narratives.

## Future work

One piece remains genuinely out of reach in this environment rather than overlooked:

- **Live GDELT validation.** GDELT is rate-limited/blocked from this project's development sandbox regardless of backoff strategy. Re-running `notebooks/phase3_gdelt_sentiment_thirdeye.ipynb` with `USE_MOCK_DATA = False` from a network without that restriction would replace the synthetic sentiment data with real news-market analysis.

The live/paper-trading forward test that used to be listed here is done — see Phase 8 in the status table above and `notebooks/phase8_live_paper_trading.ipynb`.

See `docs/THREATS_TO_VALIDITY.md` for the full list of risks and mitigations across every phase.
