# Threats to Validity

This document exists so a skeptical reviewer finds these caveats stated explicitly rather than discovering them. Each item names the risk, why it matters, what in this codebase mitigates it, and what residual risk remains after mitigation.

## 1. Survivorship bias

**Risk**: the original 6 tickers (AAPL, MSFT, GOOGL, AMZN, NVDA, META) are among the best-performing mega-caps of exactly the 2020-2026 window studied. Any result on these alone conflates genuine trading skill with simply holding assets that happened to go up a lot.

**Mitigation**: `scaata/config.py`'s `NON_SURVIVOR_TICKERS` (XOM, INTC, BABA) adds names with a materially rougher multi-year stretch over this period, and `ALL_TICKERS` includes both sets in Phase 1's data pull and evaluation.

**Residual risk**: 9 tickers is still a small, hand-picked universe, not a random or market-cap-weighted sample. A claim of "generalizes across the market" would still be overreaching; a claim of "shows some robustness beyond the original 6 mega-cap winners" is what the evidence supports.

## 2. Look-ahead bias in regime labeling

**Risk**: if regime boundaries are computed using future information (e.g. full-sample percentiles, centered rolling windows), both the reported per-regime metrics and any train/test or expert-training split built on those boundaries become invalid — a regime label "knowing" a crash is coming before it happens is a severe, easy-to-miss bug.

**Mitigation**: `scaata/regimes/detector.py`'s `threshold_regime_labels` and `hmm_regime_labels` are both built from strictly trailing windows (`rolling(..., min_periods=...)`, never `.expanding()` over the full sample or centered windows), and the HMM is fit only on the training slice, then decoded causally via a trailing context window (`_causal_hmm_decode`). This is directly unit-tested: `tests/test_regimes.py` verifies a label at time t is unchanged when the series is truncated immediately after t (the causality property, stated as a testable invariant, not just a design intent).

**Residual risk**: the threshold constants themselves (`REGIME_VOL_MULTIPLIER`, `REGIME_DD_THRESHOLD`, etc. in `config.py`) were chosen by inspection during development, which is a mild "researcher degrees of freedom" risk distinct from *computational* look-ahead — the numbers were not tuned to future data, but they also weren't derived from a fully pre-registered procedure.

## 3. GDELT coverage drift, 2020 → 2026

**Risk**: GDELT's crawl coverage and tone-scoring consistency have likely improved over this six-year window independent of any real change in the news-market relationship. An apparent shift in the sentiment-volatility correlation (the core "3rd eye" comparison) could be substantially a data-coverage artifact rather than a genuine market-structure change.

**Mitigation**: `scaata/thirdeye/correlator.py`'s `era_correlation_report` reports `mean_coverage_volume` alongside every correlation number for both eras, specifically so a reader can check whether a correlation change coincides with a coverage change. `scaata/thirdeye/narrative.py`'s narrative generation explicitly instructs the model (or the offline template fallback) to flag a coverage confound when volumes differ substantially between eras.

**Residual risk**: reporting coverage alongside the correlation doesn't *correct* for the confound statistically — a rigorous treatment would need a coverage-normalized tone metric or a matched-coverage subsample, neither of which is implemented here.

## 4. GDELT entity-matching precision

**Risk**: GDELT indexes organizations and documents, not ticker symbols, so a keyword/name match (e.g. "Apple") can pick up irrelevant context (the fruit, unrelated "Apple" entities) rather than the company.

**Mitigation**: `scaata/data/gdelt.py`'s `build_query` requires co-occurrence with a market-context keyword (stock/shares/Nasdaq/earnings/investor) alongside the entity name, reducing (not eliminating) false positives.

**Residual risk**: this is a precision heuristic, not a validated entity-linking system. No recall/precision measurement against a labeled ground truth was performed — live GDELT access was rate-limited/blocked from the development sandbox's IP (documented in `notebooks/phase3_gdelt_sentiment_thirdeye.ipynb`), so this heuristic is untested against real query results as of this rebuild; only the offline query-construction and causal-alignment logic were verified.

## 5. LLM non-determinism and model version drift

**Risk**: strategy normalization (`scaata/strategies/normalizer.py`) and 3rd-eye narrative generation (`scaata/thirdeye/narrative.py`) call a hosted Groq model. Even at `temperature=0.0`, hosted model behavior can drift silently when the provider updates the underlying model version, meaning a re-run months later may not reproduce identical strategy code or narratives.

**Mitigation**: every LLM call site has a fully offline, deterministic fallback (`validate_mock_strategies`, `_template_narrative`) so the surrounding pipeline (regime detection, RL training, evaluation) does not depend on LLM determinism to be tested or reproduced. The execute-to-validate step (`normalizer.py`'s `validate_strategy_code`) means a drifted LLM output either still produces a working strategy or is rejected outright — it cannot silently corrupt the pool with broken code.

**Residual risk**: the *substantive content* of live-scraped, LLM-normalized strategies and generated narratives is not guaranteed reproducible across time or provider-side model updates. This should be disclosed in any writeup that reports specific numbers derived from a live LLM run, and the exact model identifier/date of the run recorded.

## 6. Small-sample risk in regime-specialist training (Phase 4)

**Risk**: rare regimes (e.g. a multi-week crash) yield few, short contiguous training segments for that expert, risking overfitting to a handful of episodes.

**Mitigation**: `scaata/rl/moe/train_experts.py`'s `train_moe_experts` returns `segment_counts` alongside the trained experts specifically so this is visible, not hidden; `notebooks/phase4_moe_experts.ipynb` prints segment lengths before training and repeats the small-sample caveat next to the results.

**Residual risk**: no fix is applied (no data augmentation, no adjacent-regime blending) — Phase 4 is explicitly the lowest-priority, most compute-hungry, most caveated phase in the rebuild plan, and its numbers should be read as illustrative of the mechanism working, not as a validated performance claim.

## 7. Evaluation-fairness risk specific to Phase 4 (oracle leakage)

**Risk**: if the mixture-of-experts' evaluation used the *true* regime label to decide which expert should have acted at each step, the comparison against the single-policy baseline would be unfairly favorable — the MoE would effectively get to see the future relative to what a live system could know.

**Mitigation**: `scaata/rl/moe/gating.py`'s `RegimeGate` is fit once on the training split and, at evaluation time, is called only via `.predict()` on already-available feature rows (`scaata/rl/moe/moe_policy.py`'s `MoEPolicy.predict`) — never given `test_df`'s true regime column. This is a structural test, not just a stated intent: `tests/test_moe.py::test_gate_predicts_from_features_only_no_label_access` confirms the gate's prediction call cannot receive the label column at all.

## 8. Reproducibility and data versioning

**Risk**: `yfinance` data can be revised (splits, dividend adjustments) after the fact, so a re-run of this pipeline months later may not exactly reproduce a past run's numbers even with the same code and date range.

**Mitigation**: `scaata/data/loaders.py` caches each ticker's raw pull to a per-ticker parquet file in `data_cache/` (gitignored), so a given run's exact input data can be preserved and re-used even if a later live pull would differ.

**Residual risk**: the cache is local and gitignored by design (avoiding committing large binary data to the repo) — it is not itself a versioned, shareable artifact. Anyone needing to exactly reproduce a specific reported result should archive the relevant `data_cache/` parquet files alongside that result, not rely on cache presence alone.

## 9. Ethics / public-data framing

Both data sources used in this rebuild are public and require no private credentials: `yfinance` (public market data, as in the original v1 paper) and GDELT (explicitly open, documented, no-API-key dataset). This preserves the v1 paper's existing "public data only" ethics statement. The Groq and GitHub API keys used for strategy scraping/normalization are read from environment variables (`.env`, gitignored) rather than hardcoded, correcting the credential-exposure issue found in the original v1 notebook.
