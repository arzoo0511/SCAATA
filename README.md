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
| 1 — Data, regimes, honest baselines | Core | Full-scale run in progress (9 tickers, all walk-forward folds, 100k PPO timesteps, multiple seeds). Not yet complete as of this writing — see the notebook's own status banner for the latest. |
| 2 — Multi-agent inner loop + closed wiring gaps | Core | Mechanism fully implemented and unit-tested. Full-scale run in progress. Strategy pool uses 3 mock strategies (no live `GITHUB_PAT`/`GROQ_API_KEY` configured) — the LangGraph mechanics, ablations, and RL training are real; the input strategies are not live-scraped. |
| 3 — 3rd eye sentiment | Core | Query construction and causal trading-day alignment are implemented and unit-tested. Live GDELT access is rate-limited/blocked from this project's dev sandbox regardless of retry/backoff — the notebook runs on synthetic sentiment data as a result. Narrative generation uses a template fallback (no live `GROQ_API_KEY`). |
| 4 — Mixture of experts (calm/stress) | Stretch, lower priority | Full-scale run in progress. Explicitly the most compute-hungry, most caveated phase — read its small-sample and gating-fairness notes before trusting any single number. |
| 5 — Research rigor (threats doc, agent comparison) | Core deliverable, no training | Threats-to-validity document complete. Monolithic-vs-decomposed agent comparison harness works end-to-end but currently runs on a mock LLM weight (no `GROQ_API_KEY`) — not yet a real finding. |

"Full-scale run in progress" means: real ticker/fold/timestep counts, not the fast smoke-test defaults each notebook also supports. Check each notebook's own status banner for the most current state — this table is a snapshot, not a live feed.

## Running it yourself

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m pytest tests/ -v
```

Open any notebook in `notebooks/` with the `scaata-venv` Jupyter kernel. Each has a `SMOKE_TEST` (and, for Phase 3, `USE_MOCK_DATA`) flag near the top — `True` runs a fast sanity check in a few minutes; `False` runs the real, compute-heavy version. Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`/`GITHUB_PAT` for live strategy scraping and LLM narratives.

## Future work

Two pieces would be the strongest possible additional evidence, and are deliberately not attempted in this environment rather than overlooked:

- **Live GDELT validation.** GDELT is rate-limited/blocked from this project's development sandbox regardless of backoff strategy. Re-running `notebooks/phase3_gdelt_sentiment_thirdeye.ipynb` with `USE_MOCK_DATA = False` from a network without that restriction would replace the synthetic sentiment data with real news-market analysis.
- **A live/paper-trading forward test.** The strongest available evidence against look-ahead leakage, since the model would have zero opportunity to have seen the data during development. This needs a broker sandbox account and credentials that should be set up by the project owner directly, not automated.

See `docs/THREATS_TO_VALIDITY.md` for the full list of risks and mitigations across every phase.
