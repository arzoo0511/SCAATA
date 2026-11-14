# HANSEI: audit, architecture and roadmap

Written 13 Nov 2026 from a read of the code, the saved state and the live site.
It covers what exists, what the Research Lab and Macroverse should be, how the
pieces connect, and the order to build them in. Phase 1 (the dashboard) is done;
Phases 2 (Research Lab) and 3 (Macroverse) each have a working first version.

## 1. The aim

HANSEI should grow from an agent that reacts to indicators into one that can
connect events to companies, form a testable expectation, act on it, and check
afterwards whether it was right. "Learning" has to mean a measured improvement
against simply holding the five stocks, on data the change was not tuned on. A
reflection written in fluent text is not learning.

## 2. What exists and works (verified)

| Piece | Where | State |
|---|---|---|
| Daily run | `.github/workflows/hansei-daily.yml` → `scaata/agent/daily.py` | Live. Idempotent, catches up missed sessions, cuts news off at the fill time. |
| Advisors | `agent/advisors.py`, `agent/news.py` | Live: trend, volatility brake, momentum, news. `relstr` is logged as a shadow advisor and never traded on. |
| Brain | `agent/brain.py` | Live. Holds by default; acts on a gap of a third; waits 10 sessions; rebalances on 20% drift. |
| Paper book | `live/paper_book.py` | Live. Next-open fills, NSE costs, whole shares, liquid-fund yield on cash. |
| Memory | `agent/memory.py` | Live. Beliefs judged after 20 sessions, Hedge update, lessons kept. |
| Report card | `agent/evaluate.py` | Live. Gap split into exposure and selection, hit rates, data health, significance. |
| Backtests | `agent/backtest.py` (`--long`, `--news`), `signal_study.py`, `earnings.py` | Run by hand. Tune 2007–19, holdout 2020–, settings grid, block bootstrap. |
| Critique pieces | `agents/nodes/critique_node.py`, `devils_advocate_node.py`, `strategies/hedge.py`, `rl/reward.py` | Built and tested for the older SCAATA pipeline. Not in HANSEI's daily path. |
| Dashboard | `dashboard/hansei_app.py`, `dashboard/ui/` | Rebuilt in Phase 1. Read-only. |

What the evidence says, without rounding up:

- The robust edge is **smaller drawdowns**, in every block since 2007. The
  return edge is **not established**: HANSEI lost to holding on return in
  2007–2019 and won after 2020.
- Live (34 sessions): about 5 points behind holding, almost all of it from
  being on average 46% invested during a rally.
- **Learning has not been shown to help.** Switching it off barely changes the
  2020–26 replay. Three alternative learners tested as zero. The
  trajectory-based self-critique passed its wiring test but has not been through
  the settings grid, and is not live.
- News, earnings surprise, a 50% floor, and cash rotation were each tested and
  did not earn a place.

## 3. What is incomplete

- No registry of experiments. The findings above live in commit messages,
  result files and one person's memory.
- No structured link from an event (oil up, RBI hike) to a stock. News is scored
  good or bad per company; it is not tagged by what kind of event it is.
- No scenario or stress view.
- The critique loop is not in the live path.

## 4. Research Lab (Phase 2)

Build it as a **registry plus a runner plus a reviewer**, not as a cast of
chat agents. Roles are stages a record passes through.

- **Registry** — `research/experiments.jsonl`, append-only. One record:
  question, hypothesis, data and date range, method, the pass/fail criteria
  *written before the run*, the code commit, results, review verdict, status
  (`open` / `supported` / `rejected` / `inconclusive`). Seed it with the
  experiments already done, so they are never rediscovered.
- **Runner** — a thin wrapper over `backtest.py`, `signal_study.py` and
  `evaluate.py`. Same splits, costs, grid and bootstrap every time, so results
  are comparable. Deterministic given the commit and the cached data.
- **Devil's advocate** — mostly code, not a model. Checks: decision uses no
  data after its timestamp (a shift test), costs applied, sample size, both
  splits reported, robust across the grid, benchmark is holding, and how many
  variants were tried (multiple testing). An experiment can fail review. An LLM
  may add written objections, stored as opinion and never as a verdict.
- **Researcher** — rule-based triggers on the daily report card (an advisor's
  hit rate drifting, a widening gap, a regime the backtest handled badly)
  raise candidate questions. A person or a scheduled job picks which to run.
- **Director** — a weekly job that lists open questions, stale findings and
  what to test next. It writes proposals. It has no write access to the agent.
- **Promotion** — a `supported` finding becomes a proposal; it reaches the
  live agent only as a reviewed code change that passed the existing gate
  (beats holding in both halves and across the grid).

Why not autonomous LLM agents: with five stocks and a few thousand days of
data, an agent that can generate and test strategies freely will find something
that fits by chance. The scarce thing is disciplined testing, not ideas.

### What the first Lab runs found (13 Nov 2026, snapshot through 13 Nov)

| Experiment | Verdict | What it showed |
|---|---|---|
| Live rule against holding | Inconclusive | Worst fall smaller by 17 points (2007-19) and 10 points (2020 on); Sharpe higher in both. The Sharpe edge agrees in 89% of bootstrap resamples; the bar is 90%. Return edge on 2020 onward is now +0.2% a year. |
| Where the edge comes from | Not supported | The hypothesis was backwards. HANSEI is *behind* holding while the basket is below its six-month-ago level (-5.9% and -2.2% a year) and level or slightly ahead while it is above. It saves money as a fall begins and gives some back by sitting out the rebound. Descriptive only. |
| Without the trend advisor | Not supported | Sharpe unchanged on 2007-19 (+0.00), lower on 2020 on (-0.09, within noise). Trend is not shown to earn its place. |
| Learning switched off | Not supported | Slightly better without learning on 2007-19 (+0.03), slightly worse on 2020 on (-0.02). Learning is not shown to help. |

None of these changes the agent. The second and third are candidates for a follow-up experiment (a
faster way back in after a fall; an agent without trend), each of which must be filed with its own criteria.

Still missing from Phase 2: a whole-share mode in the replay (needed for the rounding question), experiments
for new strategies rather than ablations, a schedule for the director, and event tags on headlines.

## 5. Macroverse (Phase 3)

Start with what can be defended, in this order:

1. **Historical replay.** Run the real brain (`brain.decide`, the backtest's
   own loop) through named episodes: 2008, the 2013 taper tantrum, 2016
   demonetisation, March 2020, the 2022 crude spike. Animated day by day, with
   HANSEI against holding. This is real data, and most of the engine exists.
2. **Factor what-ifs.** Estimate each stock's sensitivity to four observable
   daily factors: Nifty, Brent in rupees, USD/INR, the 10-year yield. Show the
   estimates with their intervals and how they change across regimes; some will
   be weak, and the page must say so. A scenario is a path for the factors; a
   stock's path is sensitivity × shock plus block-bootstrapped residuals; many
   draws give a fan, not a line. The real brain trades each path.
3. **Compound shocks and recovery**, once 1 and 2 are trusted.

Not recommended: a simulated economy with GDP, unemployment, government
spending and so on. Those are monthly or quarterly series; their effect on five
stocks cannot be calibrated, so the numbers would be invented and the animation
would hide that. Company-specific channels (IOC's marketing margins under oil
shocks, bank margins under rate moves) should be written down as labelled
assumptions with a sensitivity slider, never as fitted facts.

Limits to state on the page: in simulated paths the news advisor is silent;
every synthetic result is watermarked; Macroverse writes only under
`macroverse/runs/` and never to `forward_test/` or the paper book.

### What the first Macroverse runs found (13-14 Nov 2026)

Replays, real prices (HANSEI / holding):

| Episode | Return | Worst fall |
|---|---|---|
| Global financial crisis, 2008-09 | -14.3% / -26.5% | -40.2% / -58.5% |
| Taper tantrum, 2013 | -9.6% / -8.9% | -13.6% / -22.7% |
| Demonetisation, 2016-17 | +16.0% / +17.7% | -9.4% / -9.4% |
| IL&FS scare, 2018-19 | +6.7% / +10.2% | -10.1% / -12.8% |
| COVID crash, 2020 | -6.8% / -3.1% | -28.7% / -39.3% |
| Crude spike, 2022 | +14.4% / +19.3% | -9.8% / -9.8% |

The pattern matches the Lab: it falls less, then gives the difference back by returning late. In 2020 it
avoided the worst of March and was still under-invested for the rebound.

Sensitivities (weekly, 2007-2026): every stock's market sensitivity is reliable (0.6 for ITC to 1.5 for ICICI
Bank). To crude, only Indian Oil (-0.11) and HDFC Bank (-0.03) can be told from zero. To the rupee, none can.
So "crude +60%" alone moves the book very little; the damage in a real oil shock comes through the market.

Scenarios (synthetic, 30 paths, 126 sessions; HANSEI / holding, middle outcome): market -20%: -12.4% / -18.1%;
all three shocks at once: -13.4% / -19.9%; crude +60% alone: +0.2% / +0.5%; rupee -10% alone: 0.0% / +0.7%.

Limits of this first version: three factors only; a shock is a one-month ramp with nothing else moving; 30
paths; each simulated day opens at the previous close; the news advisor is silent; no custom scenarios on
the site (each takes about two minutes of compute, so they are run offline).

## 6. Cognitive layer and how the pieces connect

Memory maps onto files that mostly exist:

| Memory | Holds | Store |
|---|---|---|
| Episodic | what it saw, believed and did each day | journal (exists; append-only) |
| Working | today's views and queued orders | latest journal entry, paper book (exists) |
| Procedural | trust weights, the brain's rules | `hansei_memory.json`, `brain.py` (exists) |
| Semantic | factor sensitivities, event→company links | new, produced by Macroverse calibration |
| Research | experiments and verdicts | new, the Lab registry |

Flat JSON in git is the right store at this size: every change is dated,
diffable and reversible. A vector store or knowledge graph adds nothing until
there are far more than five companies.

Data flows one way: **live agent → journal → Lab and Macroverse → proposals →
gate → live agent.** The Lab and Macroverse read production state and never
write it. That one rule prevents circular feedback and keeps synthetic data out
of the real record.

Where language models belong: classifying headlines (today) and tagging them
with an event type (next), so the Lab can ask "what did IOC do after oil-up
news?". Where they do not: deciding trades, or judging whether an experiment
passed.

## 7. Risks

- **Overfitting by enthusiasm.** Every new advisor or rule is another chance to
  fit noise. Count the variants tried; keep the holdout honest.
- **Theatre.** A convincing reasoning trace or simulation can be wrong. Each
  feature needs a number that could embarrass it.
- **Small sample.** Five correlated stocks. Most live questions will stay
  unanswered for years; the dashboard says so.
- **Free-tier infrastructure.** GitHub's scheduler runs hours late; Groq's free
  quota caps news scoring; Yahoo revises and rate-limits.
- **Dashboard coupling.** The site redeploys on every push to `main`, including
  the daily bot commit. The dashboard is pinned to a tested Streamlit release.

## 8. Suggested improvements to the daily run (not made; each needs a decision)

1. **Whole-share rounding keeps it under-invested.** Over the live record the
   brain *wanted* about 62% invested on average and *held* about 46%. With
   ₹10,000 across five slots, one ICICI Bank share is most of a slot and two do
   not fit, so the same rebalance is queued day after day and fills nothing (23
   such orders so far). Options: fractional paper shares, or a larger paper
   book. Either changes the live experiment, so it should restart the
   comparison rather than silently continue it. The backtest appears to trade
   rupee amounts rather than whole shares (`agent/backtest.py`), so it does not
   carry this drag and its numbers are not a like-for-like forecast of the
   ₹10,000 live book. Worth confirming before choosing.
2. **Stop re-queuing orders that cannot fill.** Cosmetic, but it muddies the
   journal.
3. **Tag each headline with an event type** when it is scored. No effect on
   trading; it starts the dataset Phase 2 needs.
4. **Put the trajectory self-critique through the grid.** It is the one
   learning mechanism with a positive result, and the result is a single run.

5. **The replay engine can borrow.** `backtest.simulate` lets cash go below zero when holdings have outgrown
   their slots and another stock is bought back to its full slot: the 2008 replay was up to 109% invested.
   The live paper book cannot do this. Every backtest, Lab and Macroverse result therefore comes from an agent
   slightly more aggressive than the live one. Fixing it changes every recorded backtest number, so it should
   be a deliberate change with the results re-run, not a quiet patch.

## 9. Build order

| Phase | Milestone | Done when |
|---|---|---|
| 1 | Dashboard rebuilt: six pages, interactive charts, session replay | Done. Tests in `tests/test_hansei_dashboard.py`. |
| 2a | Registry and runner; past experiments entered | Done. `scaata/research/registry.py`, `engine.py`, `legacy.py`; five earlier experiments entered from their result files. |
| 2b | Devil's-advocate checks; Lab page on the site | Done. `review.py`; a leaky advisor fails review in `tests/test_hansei_research.py`. |
| 2c | Triggers and weekly director report | Triggers done (`director.py`, `research/agenda.json`). Not scheduled: experiments are run by hand. |
| 3a | Historical replay through the real brain | Done. `scaata/macroverse/replay.py`; a test checks a replay equals the backtest engine's own run and cannot see prices after the episode. |
| 3b | Factor sensitivities with intervals; what-if fan | First version done (`shock.py`): three factors, bootstrap intervals, before/after-2017 estimates shown, 30-path fans. Not yet checked on a held-out period. |
| 3c | Compound scenarios, post-run report into the registry | A scenario run appears as a Lab record. |

No extra paid services are needed for any of this. Heavy runs belong on GitHub
Actions or run one at a time on the laptop.
