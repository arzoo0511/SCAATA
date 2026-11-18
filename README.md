<p align="center"><img src="docs/logo/hansei_wordmark.png" alt="HANSEI" width="420"></p>

# HANSEI (反省)

**Live dashboard: [hansei.streamlit.app](https://hansei.streamlit.app/)** · [Architecture and roadmap](docs/HANSEI_ARCHITECTURE.md) · [Threats to validity](docs/THREATS_TO_VALIDITY.md)

**An AI agent that manages the stocks you already own: it reads the news and the charts, trades only when it has a reason to, and is judged on one thing — beating what you'd have made by just holding.**

*Hansei* is Japanese for reflecting on your own mistakes in order to improve. The agent keeps score of which of its advisors were right and trusts them accordingly.

> Paper trading only. HANSEI trades ₹10,000 of fake money. Nothing here is investment advice.

## Status

- **Live paper trading since 24 Sep 2026** on five NSE stocks (HDFCBANK, ICICIBANK, ITC, IOC, PETRONET), running on its own after every market close.
  The live record so far is on the dashboard, including when it is behind holding. A few weeks of five stocks proves nothing either way yet.
- **Backtest, 2020 – Sep 2026** (`python -m scaata.agent.backtest` → `results/hansei_backtest.json`):

| Period | | CAGR | Sharpe | Max drawdown |
|---|---|---|---|---|
| Full (2020 – 2026) | **HANSEI** | **10.5%** | **0.82** | **−28.3%** |
| | Buy & hold | 9.8% | 0.60 | −38.8% |
| First half (2020 – mid-2023) | **HANSEI** | 12.5% | **0.89** | **−28.3%** |
| | Buy & hold | **13.6%** | 0.71 | −38.8% |
| Second half (mid-2023 – 2026) | **HANSEI** | **8.1%** | **0.68** | **−13.5%** |
| | Buy & hold | 6.8% | 0.48 | −20.5% |

The edge is mainly **smaller drawdowns and a higher Sharpe**. The return edge is thin (about +0.7%/yr overall) and HANSEI trailed holding on return in the first half. The backtest has no historical news, so the news advisor is silent there. HANSEI only counts as working once it beats holding **live**.

## How it works

1. **Four advisors** score each stock every day (`scaata/agent/advisors.py`, `news.py`):
   - **Trend** — price vs its 200-day average
   - **Volatility brake** — cuts exposure when the market gets choppy
   - **Momentum** — 6-month return, skipping the latest month
   - **News** — an LLM (gpt-oss-120b via Groq) scores recent headlines from Google News and Yahoo Finance; syndicated copies of one story count once; keyword fallback if the API is down
2. **A cautious brain** (`brain.py`) holds by default. It acts only when the advisors disagree with the current position by at least a third, waits 10 sessions between trades unless the news is strong, and rebalances on 20% drift.
3. **Memory** (`memory.py`) learns from every trade with Hedge and the Fixed-Share step: advisors that were right gain trust, wrong ones lose it, and a small share of all trust is handed back equally each time so an advisor that fell behind can recover when the market changes.
4. **Paper book** (`scaata/live/paper_book.py`) fills orders at the next day's open with NSE costs; idle cash earns a liquid-fund rate.
5. **Daily report card** (`evaluate.py`) splits the gap to holding into "how much it was invested" and "which stocks it held", and tracks each advisor's hit rate.
6. **Dashboard** (`scaata/dashboard/hansei_app.py`, pages in `scaata/dashboard/ui/`) is read-only and interactive:
   - **Now** — HANSEI vs holding, why the gap, where the money is
   - **Mind** — replay any session: which advisor pulled which stock which way, and what it decided
   - **Beliefs** — everything it currently believes, each with its status and evidence; its own predictions scored
   - **Learning** — trust in each advisor over time, every lesson, the ledger of beliefs and what happened next
   - **News** — every story it judged material
   - **Evidence** — the backtests and studies, including what was tested and rejected
   - **Lab** — experiments filed in advance, run through the real brain, and judged by a code reviewer
   - **Macroverse** — real crises replayed day by day, and synthetic shocks, through the real brain
   - **System** — what is live, what is in testing, what is only planned

## Research Lab

`scaata/research/` is where ideas get tested before they can go anywhere near the agent. An experiment's
pass/fail criteria are filed first (`research/experiments.jsonl`, append-only), it is run through the same
`brain.decide` and `memory.learn` the daily agent uses on a frozen price snapshot, and a reviewer made of
code checks it: no look-ahead, costs charged, same direction in the tuning years and the holdout years,
bigger than bootstrap noise, with the bar raised for every extra look at the holdout.

```
python -m scaata.research refresh          # freeze a new price snapshot
python -m scaata.research agenda           # questions the live record raises
python -m scaata.research run --proposed   # file, run and review them
python -m scaata.research list
```

Phase A (`scaata/research/learners.py`, `contribution.py`) tests the self-critique itself on 19 years of
judged predictions: a poison test that plants known-bad advisors (the live learner passes in 20 of 20 draws),
a comparison of six ways of learning trust against equal trust (none beats it), a skill ledger, a calibration
study, and each advisor's Shapley value in the whole system (all three make the worst fall smaller). The honest
reading: the advisors are risk tools, not forecasters.

Phase B (`scaata/research/rebound.py`) tried the one published fix for returning late after a fall (the "rebound"
rule of Goulding, Harvey and Mazzoleni, 2023). It recovers a little more of each rebound but trades that for more
whipsaw: not supported over the long spans, so it is not switched on. It runs as a shadow advisor in the journal,
scored daily and never acted on.

Phase C (`scaata/universe/`) widened the view to about 200 NSE stocks (price-only, survivorship-biased) and trained a
model from scratch to rank them. It finds a small real signal, but the learning adds nothing over a plain average of
the signals or plain momentum. A separate fake-money "wide book" (`paper_book_wide.json`) now tracks the learned
model, the plain average, momentum and owning everything, forward only. `python -m scaata.universe refresh|study|book`.

Phase D added interest rates (`scaata/macroverse/rates.py`): each stock's measured response to Indian and US bond
yields, the "Indian yields up 2 points" what-if, and the links as beliefs. In the sample, every stock falls when
Indian yields rise (HDFC Bank about 13% per point); US yields show no pattern. The language-model event records in
the plan are not built (they need a model service and hand-labelled headlines).

How much a result is worth is itself measured (`scaata/research/stats.py`): a Deflated Sharpe Ratio against
holding's Sharpe, the probability that picking the past winner fails, an evidence budget that shrinks with every
test, and a live e-process on the paper book that compares HANSEI with a holder of the same exposure and can be
read every day without fooling itself.

First results (13 Nov 2026): the live rule falls less than holding in both spans, but its Sharpe edge is
just inside noise (89% against a 90% bar), so the verdict is *inconclusive*. Silencing the trend advisor,
and switching learning off, did not make it measurably worse. The Lab cannot change the agent: a supported
finding is only a proposal.

### Learner: Fixed-Share (live since 16 Nov 2026)

Plain Hedge is only promised to match the single best advisor over the whole record. Which advisor is best
changes with the market, so the learner now uses Fixed-Share (Herbster and Warmuth, 1998): after each update
1/252 of all trust is pooled and returned equally. The fraction was fixed from theory, as one expected change
of best advisor a year, and not tuned. This was the owner's decision on the strength of the theory.

What the replay made of it (experiment `fixed-share-corrected`): Sharpe +0.03 and return +1.1% a year against
plain Hedge on 2007-2019, with a quarter fewer trades; Sharpe -0.05 and return -0.7% a year on 2020 onward.
Both are inside noise, so the verdict on file is *not supported*: the replay does not show it helps, and does
not show it hurts.

The first run of that experiment was invalid: a variable-name collision in the replay engine meant the setting
never reached the learner. It came out identical to the baseline to the last digit, which is how it was caught.
The invalid run stays on file marked as failed, and the reviewer now fails any variant identical to its baseline
and any run whose baseline no longer reproduces.

### Other Lab results since

| Experiment | Verdict | What it showed |
|---|---|---|
| A book like the live one (whole shares, no borrowing) | Not supported | Worse than the rupee replay on 2007-19 (Sharpe -0.04, return -1.9% a year), better on 2020 on (+0.08, +0.5%). The whole-share handicap is not consistent. |
| The replay without borrowing | Not supported | Borrowing barely matters: return -0.3% a year on 2007-19, +0.1% on 2020 on. |
| Waiting 5 sessions instead of 10 | Not supported | No measurable difference either way. |

A study of 3,962 news events since 2012 (`python -m scaata.research events`) found that of 22 kinds of good or
bad news, three were followed by a move that can be told from zero, about what chance alone would give at this
many comparisons. Results stories, good or bad, were followed by the stock lagging the other four.

### What runs on its own

- **After every close** (`hansei-daily.yml`): the agent, its report card, a belief snapshot, and the Lab's agenda.
- **Every Saturday** (`hansei-research.yml`): fresh price and factor snapshots, Macroverse, the news event study,
  any experiment the agenda proposes, and a belief snapshot.

Neither can change how the agent trades. A supported finding becomes a proposal on the Lab page; a person decides.

## Macroverse

`scaata/macroverse/` asks what happens to HANSEI when the world moves, in two clearly separated ways.

- **Replay history** (real): six episodes — 2008, the 2013 taper tantrum, demonetisation, the 2018 IL&FS scare,
  COVID, the 2022 crude spike — replayed on real prices through the same `brain.decide` and `memory.learn` the
  daily agent uses. HANSEI's worst fall was smaller than holding's in four of the six and never larger; it
  finished ahead of holding only in 2008.
- **What if** (synthetic, eight scenarios): a shock to the Nifty, Brent or USD/INR is passed to each stock through sensitivities
  measured from weekly returns, with their uncertainty kept; the real brain then trades 30 simulated paths.
  Only 7 of the 15 sensitivities can be told apart from zero, and the page shows which.

Rate, inflation, GDP and credit shocks are not offered: there is no daily data to measure them with.

```
python -m scaata.macroverse refresh    # freeze Nifty, Brent and USD/INR
python -m scaata.macroverse all        # replays, sensitivities and scenarios -> macroverse/runs/
```

Everything is computed offline and saved; the site only displays it. Macroverse never reads or writes the live
paper book, journal or memory.

## Cognitive layer

`scaata/cognition/` assembles one inspectable picture of what HANSEI believes from the records the other
parts already keep: the journal, its memory, the daily report card, the Lab's registry and Macroverse's
results. It generates no text of its own and cannot change the agent.

Every belief has a kind (fact, relationship, finding, hypothesis, assumption) and a status worked out from
its evidence by fixed rules: a measured relationship is *established* only if its 95% interval excludes zero;
a Lab finding takes the reviewer's verdict; a live hit rate is judged over days (not opinions), with a Wilson
interval, and never on fewer than 20 judged days.

```
python -m scaata.cognition            # what it currently believes
python -m scaata.cognition snapshot   # append today's beliefs to cognition/history.jsonl
```

As of 16 Nov 2026: 42 beliefs. 7 established, 7 leaning, 12 where the data cannot tell, 13 contradicted by
testing, 3 untested. After 14 judged days no advisor, and not the brain itself, can yet be told apart from a coin flip.

## Tech stack

Python · pandas / NumPy · yfinance · Google News + Yahoo headlines · Groq LLM · Hedge online learning · Zerodha Kite Connect (read-only) · Streamlit + Plotly · GitHub Actions (daily run) · pytest

## Run it

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env        # add GROQ_API_KEY (optional: KITE_API_KEY / KITE_API_SECRET)

python -m scaata.agent.daily                                        # today's decisions
python -m scaata.agent.backtest                                     # 2020–2026 backtest
streamlit run scaata/dashboard/hansei_app.py --server.port 8502     # dashboard
python -m pytest tests/
```

The daily run lives on GitHub Actions (`.github/workflows/hansei-daily.yml`): it runs the agent after each NSE close and commits the new paper book, journal and memory, which the public dashboard then picks up. On Windows, `run_hansei_dashboard.bat` starts the dashboard and `run_hansei_daily.bat` only pulls the latest state. Don't run the agent locally while the workflow is active, or the two paper books will diverge.

The dashboard needs `streamlit` and `plotly` (`pip install -r requirements.txt`).

---

## Research history: SCAATA v2

HANSEI grew out of **SCAATA** (Self-Critiquing Autonomous Algorithmic Trading Agent), a research system built on deep reinforcement learning, regime detection and a multi-agent LLM pipeline across US and Indian markets. Tested across multiple seeds and walk-forward folds, it did **not** beat buy-and-hold, so it was dropped for the simpler, explainable agent above. The package is still named `scaata`. The original research notes follow unchanged.

### The actual contribution, in plain terms

The original v1 notebook trained a behavior-cloning (BC) model and a meta-strategy-selector classifier, and the accompanying paper describes a self-critique loop that reshapes the PPO agent's reward based on realized portfolio risk. **None of that was actually wired in.** The BC model and meta-selector were trained, saved to disk, and never loaded again; the reward function was just `%change * 10` plus a stop-loss penalty — no self-critique terms at all. The self-critiquing mechanism the paper is named after was described, not implemented.

This rebuild closes that gap for real:
- Self-critique penalty terms (drawdown extension, volatility-chasing entries, prolonged losing-position holding) are computed causally, step-by-step, and actually added to the RL reward.
- The BC model's trained weights are actually loaded into the RL policy before fine-tuning.
- The meta-selector's output actually reaches the RL agent's observation.
- All three fixes are regression-tested (`tests/test_reward.py`, `tests/test_policy_init.py`), so this can't silently regress back into "described but not wired in."

That's the core, verified contribution: the mechanisms exist and are tested. Everything else below (regime segmentation, a real multi-agent pipeline, news sentiment, a mixture-of-experts variant) is a scoped extension on top of it — some core to the rebuild, some explicitly a lower-priority stretch. See the status table below for which is which and what's real vs. placeholder right now.

### What is and isn't established (audit, 2026-09-14)

Read this before quoting any result from this repo.

- **Re-run result (2026-09-16): buy-and-hold wins.** A full re-run (9 folds x 5 seeds x 9 tickers, 405 rows per arm, fees on every method, saved in `results/phase1_rerun/`) puts mean Sharpe at **0.17 for PPO vs 0.68 for buy-and-hold**; rule-based 0.19, momentum fallback 0.32. Clustered by fold, PPO trails buy-and-hold by **-0.51 (95% CI [-0.81, -0.24])** and is ahead in 1 of 9 folds. Against rule-based and the momentum fallback it is a tie (-0.02 and -0.15, both intervals spanning zero). Repeating it with the audit's environment fixes (position in the observation, scale-free features, random episode starts) gives PPO 0.24 and stops the policy collapsing (0 never-traded runs vs 3), but still trails buy-and-hold by -0.44 (CI [-0.70, -0.17]); the two arms differ by +0.07 (CI [-0.08, +0.21]), i.e. not measurably.
- **No trading edge has been shown.** The walk-forward result previously described as "PPO beats buy-and-hold, Wilcoxon p=0.0021" does not support that claim:
  - it is **vanilla RecurrentPPO** (`scaata/evaluation/phase1_pipeline.py` trains with the default reward — no self-critique, behaviour cloning, meta-selector or DSR);
  - the test used **seed 0 only**, dropped runs that never traded, treated 79 (ticker, fold) pairs sharing the same 6-month windows as independent (mean cross-ticker return correlation 0.33), and `paired_wilcoxon` is **two-sided and reports no direction**;
  - the same run's own summary has mean Sharpe **0.22 for PPO vs 0.69 for buy-and-hold** (rule-based 0.21), so if anything the difference runs the other way;
  - the test output was never saved in the notebook.
- **The self-critique loop is not in the traded policies.** Critique reweights the strategy pool on the last 20 training days; that reaches the RL policy only through behaviour-cloning warm start (off by default, rejected on MSFT) or `meta_confidence` (not a live feature).
- **The poisoned-strategy robustness test is weak evidence.** Final weights were `[0.5, 1, 0.5, 0.5, 0.5]` — only one of three good strategies was separated from the poisoned ones, in one run, under the pre-Hedge rule.
- **There is no forward-test track record.** Auto-execution ran for one day (two duplicate SELL orders); since then signals are notify-only and were not consistently acted on, so the paper account's P&L reflects discretion, not the model.
- **Fixed in the live system on 2026-09-14:** live inference now scales features with each policy's saved training statistics (it was re-fitting z-scores on the last ~41 bars — `ma_50` came out +0.81 in training scale vs -1.06 live) and reads consolidated, adjusted Alpaca bars (IEX volume was 2.8% of the volume the policies trained on). INTC's signal is withheld: its training statistics couldn't be reconstructed. The weekly retrain now gates on a one-year holdout with a paired bootstrap, trains with the reward config multi-seed validation tested, and fails loudly when nothing is evaluated or the email can't be sent.

### Vocabulary decoder

The architecture diagrams use a few informal names. Plain technical translation, first use:

| Diagram name | What it actually is |
|---|---|
| Outer loop | The reinforcement-learning environment and training loop (market data → features → regime detection → RL policy → evaluation) |
| Inner loop | A LangGraph multi-agent pipeline that scrapes candidate trading strategies from GitHub, normalizes them via an LLM, trains a meta-selector to weight them, and down-weights poor performers via a critique step |
| 3rd eye | A news-sentiment analysis module (GDELT + FinBERT) that both feeds a sentiment feature into the RL agent and produces a standalone narrative comparing the sentiment-market relationship across time |
| Knowledge base | A shared store of strategy weights, regime labels, critique history, and trained model artifacts, referenced by all three pieces above |

### Status per phase

| Phase | Scope | Status |
|---|---|---|
| 1 — Data, regimes, honest baselines | Core | **Full-scale run complete; no edge shown.** 9 tickers × 9 walk-forward folds × 5 seeds × 100k PPO timesteps/seed, vanilla RecurrentPPO with the default reward. Re-run 2026-09-16 over all 5 seeds (405 rows): mean Sharpe PPO 0.17, buy-and-hold 0.68, rule-based 0.19, momentum fallback 0.32; fold-clustered, PPO trails buy-and-hold by -0.51 (95% CI [-0.81, -0.24]). The previously reported Wilcoxon p-values (0.0021 vs buy-and-hold, 0.833 vs rule-based, 0.209 vs the mock LLM baseline) came from a seed-0-only, direction-less test on non-independent pairs and are not evidence of outperformance — see the audit section above. |
| 2 — Multi-agent inner loop + closed wiring gaps | Core | **Full-scale run complete.** Mechanism fully implemented and unit-tested. Robustness test: final weights `[0.5, 1, 0.5, 0.5, 0.5]` (good mean 0.667 vs poisoned 0.500) — only one good strategy separated, one run. The ablation table is a single seed/ticker result on a 56-day test window — it shows the mechanisms change behaviour, nothing about performance. Strategy pool uses 3 mock strategies (no live `GITHUB_PAT`/`GROQ_API_KEY` configured). |
| 3 — 3rd eye sentiment | Core | Query construction and causal trading-day alignment are implemented and unit-tested. Live GDELT access is rate-limited from this project's dev sandbox (confirmed live via direct API calls returning HTTP 429, even with 6+ second spacing) — the notebook runs on synthetic sentiment data as a result. Narrative generation uses a template fallback (no live `GROQ_API_KEY`). |
| 4 — Mixture of experts (calm/stress) | Stretch, lower priority | **Full-scale run complete.** Gate accuracy 0.884 vs. causal labels, and is structurally confirmed never to see the true regime label at eval time. MoE finished at 9,486 equity vs. the single-policy baseline's 10,700 — a real result, but every regime row except `calm_bull` is small-sample (≤34 days), so read it as illustrating the mechanism, not a validated performance claim. |
| 5 — Research rigor (threats doc, agent comparison) | Core deliverable, no training | Threats-to-validity document complete. Monolithic-vs-decomposed agent comparison harness works end-to-end but currently runs on a mock LLM weight (no `GROQ_API_KEY`) — not yet a real finding. |
| 8 — Live paper-trading forward test | Stretch | **Harness works; no track record.** Connected to a real Alpaca paper-trading account. Auto-submitted orders ran for one day (2026-07-22, two duplicate SELL orders before the idempotency guard existed); the scheduled job has since been notify-only, with signals not consistently acted on. Until 2026-09-14 live features were scaled differently from training (see the audit section), so earlier live signals don't reflect the trained policies. |
| 19 — Client-facing signal product | Productization | **Multi-tenant plumbing done and tested (94 tests); not yet actually serving a paying client.** `scaata/product/` turns the single-hardcoded-recipient signal pipeline into a real subscriber store (SQLite), an impersonal per-ticker signal (`daily_signal.impersonal_view` — no account cash/position ever leaves this deployment), a FastAPI signup+auth+webhook layer, and Stripe billing scaffolding on the same mock-fallback convention as `alpaca_broker`/`email_sender`. What's still outstanding, and whose to do: **you** need a real Stripe account/product/price and a lawyer's sign-off before taking a client's money (see "Shipping this to real clients" below); **infra** still runs on one Windows laptop that sleeps — this needs a real host before a second person can depend on it; **the strategy itself** — see Phase 1 above — has no demonstrated edge over buy-and-hold, so there is nothing yet that a subscriber should be paying for. |

This table reflects the most recent full-scale runs. Check each notebook's own status banner if you re-run it yourself, since re-running with a live `GROQ_API_KEY`/`GITHUB_PAT` would change what's marked mock above.

### Running it yourself

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m pytest tests/ -v
```

Open any notebook in `notebooks/` with the `scaata-venv` Jupyter kernel. Each has a `SMOKE_TEST` (and, for Phase 3, `USE_MOCK_DATA`) flag near the top — `True` runs a fast sanity check in a few minutes; `False` runs the real, compute-heavy version. Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`/`GITHUB_PAT` for live strategy scraping and LLM narratives.

### Shipping this to real clients (Phase 19)

`scaata/product/` (subscriber store, API, billing scaffold) and
`scaata/live/distribute_signals.py` (the daily fan-out job) are real,
tested code — not a mockup. What they don't do is make this a legal or
operationally ready business on their own. In order:

1. **Get real legal sign-off before taking anyone's money.** Sending
   trading signals to paying third parties is regulated (Investment
   Advisers Act in the U.S.; other rules elsewhere). This product is
   deliberately *impersonal* — the same signal to every subscriber
   watching a ticker, never sized against an individual's account, always
   carrying `config.SIGNAL_DISCLAIMER` — because that's the shape a
   publisher's-exemption-style analysis looks for. That is a design
   choice, not a legal conclusion. Get an actual securities lawyer to
   confirm before Step 2.
2. **Create a real Stripe account**, a subscription Product and Price,
   and set `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` / `STRIPE_PRICE_ID`
   in `.env` (see `.env.example`). Until these are set, signups work and
   `checkout_source` in the response is `"mock"` — no real payment link,
   no real activation path other than manually calling
   `scaata.product.db.set_subscriber_active`.
3. **Run the API and point Stripe's webhook at it**: `run_product_api.bat`
   (or `uvicorn scaata.product.api:app`), then register
   `https://<your-host>/webhooks/stripe` in the Stripe dashboard for the
   `checkout.session.completed`, `customer.subscription.deleted`, and
   `customer.subscription.updated` events.
4. **Wire the daily fan-out into the existing schedule**: add a step
   after `SCAATA-DailySignals` (Task Scheduler) that runs
   `python -m scaata.live.distribute_signals` — it only ever sends
   today's already-computed, already-logged signal, once per subscriber
   per ticker per day (see its module docstring for the idempotency
   guarantee).
5. **Move off this laptop.** Everything above still assumes the same
   16GB Windows machine that sleeps and skips scheduled tasks on battery
   (see the machine-constraints notes) — fine for one person's own paper
   account, not something a paying client's signal delivery should depend
   on. A small always-on VM or a scheduled cloud job is the next real
   piece of work, not built here.
6. **There is no edge to sell yet.** Phase 1's full-scale run does not show
   PPO beating buy-and-hold (mean Sharpe 0.22 vs 0.69 on the same windows;
   the old "p=0.0021" claim is withdrawn — see the audit section). Don't take
   a subscriber's money until a walk-forward evaluation with a directional,
   fold-clustered test says otherwise.

### Future work

One piece remains genuinely out of reach in this environment rather than overlooked:

- **Live GDELT validation.** GDELT is rate-limited/blocked from this project's development sandbox regardless of backoff strategy. Re-running `notebooks/phase3_gdelt_sentiment_thirdeye.ipynb` with `USE_MOCK_DATA = False` from a network without that restriction would replace the synthetic sentiment data with real news-market analysis.

The live/paper-trading forward test that used to be listed here is done — see Phase 8 in the status table above and `notebooks/phase8_live_paper_trading.ipynb`.

See `docs/THREATS_TO_VALIDITY.md` for the full list of risks and mitigations across every phase.
