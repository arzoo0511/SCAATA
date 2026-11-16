# HANSEI master plan: from here to a system that critiques itself, learns, and gets better

Written 16 Nov 2026. This is a plan only; nothing in it has been built. It draws on the two research
reports already in `reports/`, on two literature reviews run for this plan (online learning and
self-critique; validation, discovery, LLM agents, Indian data), and on the code and results in this
repo. A third review, an independent audit of the repo, did not finish; the state described below is
from the code and records directly.

Citations marked † were quoted from memory by the reviewers and not re-checked; verify before quoting
them elsewhere. Unmarked ones were confirmed during the review.

---

## 1. The honest starting point

**What is built and working:** a daily paper agent (four advisors, a hold-by-default brain, Fixed-Share
learning); a Research Lab with a registry, a code reviewer and 13 reviewed experiments; Macroverse (six
crisis replays, eight shock scenarios); a belief store; event labels on headlines; a nine-page site;
daily and weekly automation.

**What the evidence says:**

| Claim | Status |
|---|---|
| HANSEI falls less than holding | Robust. Every multi-year block since 2007; four of six crisis replays; never worse. |
| HANSEI earns more than holding | Not shown. Lost on return 2007–19, roughly level since 2020. Live: 5 points behind after 34 sessions. |
| Learning makes it better | Not shown. Learning on vs off, Fixed-Share vs plain Hedge, each advisor in vs out: all within noise. |
| Its confidence means something | Not shown, and possibly inverted: strong views right less often than weak ones so far (14 judged days). |

**Why nothing has shown up, and why more of the same will not help.** Two results from the reviews
change the plan:

1. *With this setup, "no measurable benefit from learning" is the expected result.* For 4 advisors over
   about 4,800 rounds with feedback 20 sessions late, the regret bound is about 0.055 per round in
   normalised-loss units (Weinberger & Ordentlich 2002†; Joulani, György & Szepesvári 2013). Learning
   can only beat equal weights if the best advisor beats the average by more than that, consistently.
   Overlapping 20-session judgements leave about 240 independent observations. And because the brain
   holds by default, a change in trust rarely changes a trade. This is the forecast-combination
   puzzle: equal weights are hard to beat when skills are close and samples short (Claeskens, Magnus,
   Vasnev & Wang 2016; Timmermann 2006†; DeMiguel, Garlappi & Uppal 2009†).
2. *A return-based live proof on five stocks is out of reach.* To show a Sharpe improvement of 0.2 at
   10% tracking error takes about 68 years one-sided, about 196 years at 80% power (Bailey & López de
   Prado 2012†; Lo 2002†). Daily data does not help; power comes from calendar time or from breadth.

So the end goal cannot be reached by adding advisors or changing the learner on five stocks and
watching the paper book. It needs three changes of approach: **prove different things, learn where
learning is detectable, and widen what it learns from.**

---

## 2. The end goal, stated so it can be checked

"A system that critiques itself, learns from scratch, and becomes a better model" is met when all
three of these hold on data the system was not tuned on:

| Test | Passes when |
|---|---|
| **G1. It critiques itself correctly** | A planted bad advisor and a planted random advisor are both demoted, and a planted good one is promoted, in at least 19 of 20 seeds; and its stated confidence is calibrated (reliability error below a pre-set bound) on forward data. |
| **G2. It learns** | The learned combination beats the equal-weight combination on a proper forecast score (log or Brier), on forward data, by an anytime-valid test at 5%. |
| **G3. Learning makes it a better model** | A version chosen by the system's own process beats the frozen 16 Nov 2026 baseline on the pre-registered primary endpoint (risk-adjusted, see 4.2), on forward data, by an anytime-valid test at 5%. |

G1 is achievable in weeks. G2 needs the wider universe (Phase D) to have power. G3 is the long one:
on a risk endpoint it is checkable in one to three years; on return alone it is not checkable at all,
and the plan says so rather than pretending.

---

## 3. Rules that hold throughout

- **The live agent changes only by a reviewed, deliberate edit.** No component writes to the agent's
  settings. (Unchanged.)
- **2007–2025 is development data, permanently.** The holdout from 2020 has been looked at ten times;
  it is spent. Only data after a change is filed can confirm that change.
- **Every trial is logged, including the ones that fail.** The registry stays append-only.
- **A language model extracts and classifies; it does not decide and it does not judge.** Its outputs
  are features that pass the same tests as any other (Huang et al. 2023, "LLMs cannot self-correct
  reasoning yet"; Beel et al. 2025 on the AI Scientist).
- **When a result is exactly zero or suspiciously good, suspect the wiring first.**
- **One heavy job at a time on the laptop; long jobs detached or on GitHub Actions.**

---

## 4. Phase 0 — Stabilise and make the proof honest (1–2 weeks)

Nothing new is learned in this phase. It makes everything after it trustworthy.

### 4.1 Verify what was just shipped

| Item | Check | File |
|---|---|---|
| First daily run on the new code | A "HANSEI daily update" commit appears; `cognition/history.jsonl` and `research/agenda.json` are in it; trust weights move by the share step. | `.github/workflows/hansei-daily.yml` |
| First weekly research run | Trigger by hand once (`workflow_dispatch`); it finishes inside 120 minutes and commits. | `.github/workflows/hansei-research.yml` |
| Headlines carry `event` | The day's `data_cache/news_india/*.json` rows have the field. | `scaata/agent/news.py` |

### 4.2 Replace the proof machinery

1. **Freeze the baseline.** Tag the 16 Nov 2026 agent (`baseline-2026-11-16`). Every later comparison
   is against this exact code and settings, replayed alongside the live agent as a shadow.
2. **Primary endpoint, filed once.** *Paired daily difference in risk-adjusted outcome between a
   candidate and the frozen baseline*: the difference in realised downside (daily loss beyond a fixed
   threshold) at matched average exposure. Secondary: return, Sharpe, turnover. Rationale: a volatility
   or drawdown estimate has relative error about 1/√(2T), far tighter than a mean, and pairing removes
   the market noise both share (Diebold & Mariano 1995†).
3. **Anytime-valid monitoring.** Add an e-process on the paired daily difference,
   Eₜ = ∏(1 + λ·Dₜ) with λ fixed in advance; declare a result only when Eₜ ≥ 20 (Howard, Ramdas,
   McAuliffe & Sekhon 2021†; Shafer 2021†). This lets the site show a live "evidence meter" that can be
   read every day without invalidating the test. It does not create power; it makes looking free.
4. **Honest backtest statistics.** Add the Deflated Sharpe Ratio to every registry result, with the
   trial count and variance taken from the registry itself (Bailey & López de Prado 2014†), and report
   the Probability of Backtest Overfitting over the settings grid (Bailey, Borwein, López de Prado & Zhu
   2017†). These go on the Lab page as "how much to discount this".
5. **Replace the rising Bonferroni bar with an evidence budget.** For screening ideas, use
   alpha-investing: a fixed pot of "evidence budget" that each test spends and a pass refills (Foster &
   Stine 2008†; Javanmard & Montanari 2018†). Confirmatory claims (G2, G3) keep strict family-wise
   control.
6. **Stop judging new work on 2020 onward.** The reviewer's `same_direction_in_both_spans` check
   becomes: tune on purged walk-forward folds inside 2007–2025 (López de Prado 2018†, ch. 7), confirm
   only on forward data.

Files: new `scaata/research/stats.py` (DSR, PBO, e-process, alpha-wealth); `scaata/research/review.py`;
`scaata/research/registry.py` (budget events); `scaata/dashboard/ui/lab.py`.
Done when: every existing record shows a DSR; a planted "lucky" strategy among 50 random ones is
discounted to insignificance in a test; the e-process rejects on a synthetic stream with a real effect
and does not on one without, across 1,000 simulations at the stated error rate.

### 4.3 Known defects to clear

| Defect | Fix |
|---|---|
| Replay can borrow; live cannot | Make `no_borrowing=True` the replay default and re-run the registry's baseline as a new dated record. The experiment says the effect is small, so do it for correctness. |
| Evidence page shows a September backtest | Regenerate from the frozen snapshot weekly, in the research workflow. |
| Two beliefs with the same text (invalid and corrected Fixed-Share runs) | Belief store skips records whose review is `failed_review` and superseded. |
| "Page not found" flash on a direct link after a restart | Register pages before the first render, or replace deep links with a query parameter. |
| Hosted site sleeps | A scheduled ping from the daily workflow. |
| Keyword-scored news days are mixed with model-scored ones | Mark them in the journal and exclude them from the news advisor's scorecard. |
| Whole-share drag on the ₹10,000 book | Decision for the owner (section 12). |

---

### 4.4 Phase 0 status (built 16 Nov 2026, not yet published)

| Item | State |
|---|---|
| Frozen baseline | Tagged `baseline-2026-11-16` at the Fixed-Share commit (local tag; push it with the next publish). A running shadow copy of the baseline beside the live agent is **not** built: see "What Phase 0 did not do". |
| Deflated Sharpe, overfitting probability | Done: `scaata/research/stats.py`, `scaata/research/audit.py`, shown on the Lab page. The Deflated Sharpe is taken against *holding's* Sharpe (against zero it is 100% for everything, which says nothing). |
| E-process and live meter | Done. The Now page shows "Is the edge real yet?": HANSEI against a holder with the **same share in stocks**, so being less invested earns no credit. After 33 sessions: no evidence either way (best case 0.9 to 1; 20 to 1 is needed). |
| Evidence budget replacing the rising bar | Done in the reviewer. Alpha-investing: starts at 0.10, each test uses half of what is left, a pass earns 0.05 back. The standing question re-asked on new data spends nothing. Existing verdicts are unchanged (the registry is append-only). |
| No-borrowing replay | Experiments and Macroverse now cap buys at the cash on hand, as the live book does. The standing baseline was re-run in that mode as `live-rule-vs-holding-no-borrowing`. Older records keep their original mode, recorded as such. |
| Evidence page stays current | The weekly workflow now re-runs the backtests behind it. |
| Belief store | Superseded invalid runs no longer appear as a second belief; the news advisor's scorecard skips days the keyword fallback was used. |
| First daily run on the new code | Not yet seen: the last daily commit on `main` is from 13 Nov. Check after the next run. |
| First weekly research run | Not yet run. Needs a manual trigger from the Actions tab. |

**What the new numbers say** (on the 2007-2019 and 2020-onward replays):

- The chance HANSEI's Sharpe truly beats holding's is 60-72% taken alone and 55-65% after discounting for the
  variants tried. Nothing reaches 95%.
- The overfitting probability is 64% (2007-2019) and 39% (2020 on) over seven variants, several of them
  near-copies: a smoke alarm, not a measurement.
- The evidence budget is down to 0.001. The 2020-onward years cannot confirm anything further.
- Live meter: no evidence either way after 33 sessions, on return or on loss avoided.

**What Phase 0 did not do, and why**

- *A shadow copy of the frozen baseline.* It needs a second paper book running old code on the same prices every
  day. That belongs with Phase F (shadow books), where the same machinery serves candidates.
- *Fixing "Page not found" on a direct link.* It does not occur on a normal visit; it appears only on the first
  request after the server restarts, and the page loads anyway.
- *Keeping the hosted site awake.* A scheduled request does not count as a visitor, so it would not work, and
  I will not claim it does. A first visit after idle shows a "wake it up" button.
- *Whole-share drag on the ₹10,000 book.* Still the owner's decision (section 12).

## 5. Phase A — Self-critique that can be proved (2–4 weeks) → G1

The current critique is "trust moved". This phase makes critique a measured thing.

### 5.1 Two ledgers instead of one

HANSEI is a price-taker: a stock's next 20 sessions do not depend on what HANSEI did. That gives two
clean, separate scores (difference rewards, Wolpert & Tumer 2002†; Shapley values for ensembles,
Rozemberczki & Sarkar 2021†):

- **Skill ledger.** Each advisor's view scored against the stock's own forward return over cash, with a
  proper score. The brain's risk rules cannot contaminate it.
- **Contribution ledger.** The whole pipeline replayed with each subset of advisors silenced. With four
  advisors that is 2⁴ = 16 replays, so the Shapley value of each advisor is computed exactly, not
  estimated. This replaces "replay each advisor alone, fully invested", which the earlier report flagged
  as biased because it ignores interaction and the brain's own rules.

Files: `scaata/cognition/ledgers.py`; reuse `scaata/research/engine.py::silence`. About 16 × 70 s per
weekly run.

### 5.2 The poison test, as a permanent check

Add three synthetic advisors to a replay: one adversarial (the sign of the future return, inverted),
one random, one informative (the future return's sign plus noise). Pass = the learner demotes the
first two and promotes the third in 19 of 20 seeds, and the contribution ledger gives the adversary a
negative Shapley value. The earlier version caught only a coin-flip; "buy and never sell" slipped
through. Runs weekly; a failure blocks any promotion.

Files: `scaata/research/poison.py`; a new experiment kind in `engine.py`.

### 5.3 Calibration: make confidence mean something

- Turn each view into a probability that the stock beats cash over 20 sessions, with a monotone map
  fitted on past data only (isotonic or Platt; Zadrozny & Elkan 2002†). Fit on pooled advisors and
  stocks: this is the one target with enough data.
- Score with Brier and log score; publish the reliability diagram and the Murphy decomposition
  (reliability, resolution, uncertainty; Murphy 1973†).
- Wrap forecasts in adaptive conformal intervals so stated coverage holds under drift
  (Gibbs & Candès 2021): αₜ₊₁ = αₜ + γ(α − errₜ).
- If strong views stay less reliable than weak ones, shrink them (Grinold's refinement: forecast =
  IC × volatility × score†). This is the direct response to the inverted-confidence finding.

Files: `scaata/agent/calibration.py` (fitted offline, applied as a shadow first); Beliefs page.

### 5.4 Learner corrections the theory supports

In order; each is one registry experiment, scored on the **skill ledger**, not on P&L:

1. **News as a sleeping expert.** Today a silent advisor keeps its trust but its zero view still dilutes
   the score. The specialists update renormalises among advisors that spoke (Freund, Schapire, Singer &
   Warmuth 1997†; Kleinberg et al. 2010†).
2. **Delay handled properly.** Run 21 interleaved learners, one per phase of the 20-session delay
   (Weinberger & Ordentlich 2002†), instead of one learner fed late.
3. **AdaHedge in place of a hand-set rate** (de Rooij et al. 2014†). The first research report's top
   recommendation; it removes a knob. Expect equivalence, and say so if that is the result.
4. **Shrink learned weights toward equal, and report how hard.** A shrinkage near 1 is the honest
   reading "there is nothing to learn here yet".
5. **Mixing past posteriors** for recurring regimes (Bousquet & Warmuth 2002). Last; unlikely to be
   detectable with four advisors.

Done when (G1): the poison test passes; reliability error on the next 60 live sessions is inside the
pre-set bound; each of the five changes has a filed verdict.

---

## 6. Phase B — Fix the one weakness the evidence identified (2–3 weeks)

The Lab and Macroverse agree: HANSEI saves money as a fall begins and gives it back by returning late
(COVID replay, the market-condition study). This is the documented cost of trend-following at turning
points (Daniel & Moskowitz 2016†).

**The published fix:** Goulding, Harvey & Mazzoleni 2023, "Momentum turning points". A slow and a fast
trend signal define four states: Bull, Correction, Bear, Rebound. In **Rebound** (slow says out, fast
says in) the right move is to re-enter.

- Add a fast trend view (one month) alongside the 200-day one, **used only to permit re-entry**.
- Parameters taken from the paper, not searched. One experiment, one rule.
- Judge it on the six replayed episodes plus the 2011 and 2015–16 drawdowns, on recovery captured and on
  the primary endpoint. With so few episodes one tuned rule would "work"; that is why nothing is tuned.
- A shorter wait between trades was already tested and did nothing, so the wait is not the lever.

Then, lower priority: **meta-label the cut decision** (López de Prado 2018†; Joubert 2022). A two- or
three-feature logistic model answers "was cutting worthwhile, 20 sessions on?". Low capacity on purpose:
there are only a few hundred overlapping cut events, and no independent replication of meta-labelling's
out-of-sample gains exists.

Files: `scaata/agent/advisors.py` (shadow advisor first, as `relstr` is today); `scaata/agent/brain.py`
only after a pass; `scaata/research/director.py`.

---

## 7. Phase C — Breadth: give it enough to learn from (4–8 weeks) → G2

This is the largest lever in both reviews. Information ratio ≈ skill × √breadth (Grinold 1989†); five
correlated large caps give an effective breadth of two or three. The successes of machine learning in
return prediction all rest on huge cross-sections (Gu, Kelly & Xiu 2020†: about 30,000 stocks) and
finance is otherwise a low-signal, small-sample problem (Israel, Kelly & Moskowitz 2020†).

### 7.1 The data

- Universe: Nifty 200 members. The signal study already pulls about 105 large caps through Yahoo;
  extend and freeze it as a snapshot like `research/data/prices.csv.gz`.
- **Survivorship:** free sources give today's members only. Record the bias on the page, and use
  historical index membership where it can be found.
- Do not redistribute NSE archive data; snapshots in the repo stay to Yahoo-sourced adjusted prices.

### 7.2 A learner that starts from nothing

- A pooled cross-sectional model: for each stock each week, predict the next 20 sessions' return
  relative to the universe from price-only features (momentum at several horizons, reversal,
  volatility, distance from highs, liquidity).
- Start with ridge regression; add gradient-boosted trees only if they beat ridge under purged
  walk-forward. Complexity for its own sake is contested (Kelly, Malamud & Zhou 2024†; Nagel 2025).
- Train on expanding windows, never on the future. This is the part of the system that genuinely
  "learns from scratch": it is given features and outcomes, not rules.
- Its output for the five traded stocks becomes a fifth advisor, in shadow first.

### 7.3 A wide paper book for proof

Run a second paper book over the wide universe (for example the top and bottom fifths by the learner's
score, equal weight, monthly). It never mixes with the five-stock book. Its purpose is statistical:
with a hundred or more positions the same e-process reaches a verdict in months to a couple of years,
where the five-stock book needs decades.

Files: `scaata/universe/` (data, features, model); `scaata/live/paper_book.py` reused for a second book
file; a new workflow step; a "Universe" section on the site.
Done when (G2): on forward data the learned model beats equal-weight features on rank-correlation loss,
by the e-process at 5%.

---

## 8. Phase D — Perception: connect events to companies (4–6 weeks)

The directive's "crude rises → which companies, why, how much" is still missing. The event study found
almost nothing in keyword labels, which is a useful null: labels alone are too coarse.

1. **Better events.** Have the language model output a structured record per headline: event type,
   direction, magnitude, and the entities involved. Validate against a hand-labelled sample of 300
   headlines before using it; report its accuracy on the site.
2. **Kill look-ahead.** Language models can "remember" what happened next (Sarkar & Vafa 2024;
   Glasserman & Lin 2023). Rules: backtest language-model features only on text after the model's
   training cutoff, or on anonymised text; mark everything earlier as contaminated.
3. **Macro factors, including rates.** Add a daily Indian 10-year yield. Best free route: the Nifty
   10-year benchmark G-Sec index, backing the yield out of the price index if no yield column is
   offered; anchor against RBI weekly figures and the monthly OECD series on FRED; save CCIL's daily
   indicative yield going forward for non-commercial use. This unlocks the "RBI +200 basis points"
   scenario that Macroverse cannot run today. Check each source's terms first; do not scrape sites that
   forbid it.
4. **Relationships as beliefs.** Each measured link (event type → stock, factor → stock) enters the
   belief store with its interval and its out-of-sample check, as sensitivities do now.

Files: `scaata/agent/events.py` (structured path beside the keyword one); `scaata/macroverse/factors.py`;
`scaata/cognition/beliefs.py`.

---

## 9. Phase E — It proposes its own experiments (3–5 weeks)

Today the director's questions are hand-written rules. This phase lets the system generate hypotheses,
under a budget that makes over-searching impossible.

- **Generators:** (a) the rule-based director; (b) a parameter proposer over existing knobs; (c) a
  language model that reads the belief store and proposes a hypothesis with a stated mechanism.
- **Every proposal must carry** a written economic reason, a cost assumption, and criteria, before it
  runs (Arnott, Harvey & Markowitz 2019†).
- **Every test spends evidence budget.** When the pot is empty, testing stops until forward data
  refills it. Published anomalies lose 26–58% out of sample (McLean & Pontiff 2016†); mined ones lose
  more.
- **At most one champion per quarter** goes forward to live confirmation, chosen by a test that
  accounts for everything tried that quarter (White's Reality Check 2000†; Hansen 2005†).

Files: `scaata/research/generator.py`; `scaata/research/director.py`; Lab page.

---

## 10. Phase F — Promotion and the long proof (continuous) → G3

1. **Shadow first.** A champion runs as a shadow paper book beside the live agent and the frozen
   baseline, on the same prices.
2. **Promotion rule.** Promote when the e-process against the baseline crosses 20 on the primary
   endpoint, the poison test passes, and calibration holds. Demote on the mirror-image rule.
3. **One change at a time,** each a tagged release, each reversible in one commit.
4. **The site shows it:** baseline, live, and shadows as an evidence race, with the meter for each.

---

## 11. Order, dependencies and effort

| # | Phase | Depends on | Effort | Unlocks |
|---|---|---|---|---|
| 0 | Stabilise; new proof machinery | — | 1–2 weeks | everything |
| A | Provable self-critique | 0 | 2–4 weeks | **G1** |
| B | Rebound re-entry; cut meta-label | 0 | 2–3 weeks | the one known weakness |
| C | Wide universe; from-scratch learner; wide book | 0 | 4–8 weeks | **G2**, real power |
| D | Structured events; rates; relationships | 0 (C helps) | 4–6 weeks | the "understand the world" goal |
| E | Self-generated experiments on a budget | 0, A | 3–5 weeks | autonomy without overfitting |
| F | Promotion and live confirmation | A, C | continuous | **G3** |

A and B can run in parallel after 0. C is the long pole and should start as soon as 0 is done.
All of it fits the free tier: heavy runs move to the weekly workflow; nothing needs paid data.

---

## 12. Decisions that are the owner's

1. **Primary endpoint.** The plan proposes risk-adjusted outcome against the frozen baseline. Return
   alone cannot be proved on this book in a lifetime.
2. **A second, wide paper book.** It is the only route to a verdict within a couple of years. It is a
   different book from "my five stocks".
3. **The ₹10,000 whole-share book.** Keep it as is (faithful to a small real account), allow fractional
   shares, or raise the paper capital. Any change restarts the live comparison.
4. **Fixed-Share.** It is live by decision; the replay shows no effect either way. Phase A re-tests it
   on the skill ledger, where an effect could be seen.
5. **Language-model spend.** Structured event extraction on the free Groq tier is about 65 calls a day;
   backfilling the archive would take months at that rate.

---

## 13. What not to do

- Do not add advisors or learners and judge them on the five-stock paper book's return.
- Do not keep testing on 2020 onward and call it a holdout.
- Do not report a best backtest without its Deflated Sharpe and the trial count.
- Do not treat a language model's reflection or verdict as evidence.
- Do not backtest language-model features inside the model's training window.
- Do not use GAN-style synthetic data, MAML, Thompson-sampling trust or Bayesian model averaging: the
  first report found their support is single unreplicated runs.
- Do not sell volatility targeting as a return lever; its out-of-sample Sharpe gain did not replicate
  (Cederburg, O'Doherty, Wang & Yan 2020). Its drawdown benefit is the safe claim.
- Do not tune the rebound rule on six episodes.

---

## 14. Risks

| Risk | Guard |
|---|---|
| The wide-universe learner finds nothing either | Then the honest result is "no learnable edge in free price data"; G1 still stands, and the plan says so. |
| Survivorship bias flatters the wide universe | Forward-only confirmation; bias stated on the page. |
| Free data sources change or block | Snapshots with hashes; each source's terms checked before use. |
| More automation, more silent failure | Every workflow step may fail alone; a weekly health line on the site. |
| A bug that looks like a result | The two reviewer checks added on 16 Nov; poison test as a standing check. |
| The proof takes years | Stated up front; the evidence meter shows progress honestly rather than a verdict early. |
