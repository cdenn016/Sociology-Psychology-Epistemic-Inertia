# Real-World Experiments: Epistemic Inertia Framework

**Generated:** 2026-03-15
**Framework:** "The Inertia of Belief" (Dennis, 2025)
**Goal:** Firm, executable experiments using open public datasets

---

## Mass Formula Under Test

```
M_i = Λ_p + Λ_o + Σ_k β_ik Λ̃_qk + Σ_j β_ji Λ_qi
      ───   ───   ──────────────   ──────────────
      prior  obs   incoming social   outgoing social
```

**Core predictions:**
1. High M → smaller belief updates (|Δμ|)
2. High M → longer relaxation times (τ = M/γ)
3. Outgoing attention accumulates as rigidity (influence → inertia)
4. Underdamped regime → non-monotonic belief trajectories (overshoot, oscillation)
5. Resonant persuasion frequency ω_res = √(K/M)

---

## Experiment Overview

| # | Dataset | Hypothesis | Data Access | Status |
|---|---------|-----------|-------------|--------|
| 1 | Wikipedia Edit Histories | Influence → Rigidity (H3.1) | Public dumps | Ready |
| 2 | Fed Survey of Professional Forecasters | Belief Oscillation (H2.1) | Public CSV | Ready |
| 3 | Stack Overflow Data Dump | Reputation → Update Inertia | Public archive | Ready |
| 4 | OpenAlex Citation Network | Scientific Belief Inertia | Public API | Ready |
| 5 | ANES Panel Studies | Political Belief Persistence (H4.1) | Public download | Ready |
| 6 | Yahoo Finance + Analyst Forecasts | Financial Forecast Rigidity | Public API | Ready |
| 7 | Manifold Markets | 4-Term Mass Formula | Public API | Built |
| 8 | Reddit Comment Archives | Echo Chamber Threshold (H3.2) | Academic Torrents | Ready |

---

## Experiment 1: Wikipedia Editor Belief Inertia

**Directory:** `experiments/wikipedia_inertia/`
**Hypothesis tested:** H3.1 (Influence → Rigidity / Adams-Solzhenitsyn Effect)

### Theory Connection

The outgoing social inertia term Σ_j β_ji Λ_qi predicts that editors whose
contributions are widely watched/reverted/built-upon develop greater epistemic
mass and become more resistant to changing their positions.

### Dataset

- **Source:** Wikipedia complete edit history dumps
- **URL:** `https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-stub-meta-history.xml.gz`
- **Lighter alternative:** Wikipedia API (`https://en.wikipedia.org/w/api.php`)
- **Size:** Individual article histories via API (no bulk download needed)
- **Access:** Fully public, no authentication

### Mass Proxy Mapping

| Mass Component | Wikipedia Proxy | Measurement |
|---------------|----------------|-------------|
| Λ_p (prior precision) | Editor tenure + edit count | Days active × log(total_edits) |
| Λ_o (observation precision) | Domain expertise | Edits in topic category / total edits |
| Σ_k β_ik Λ̃_qk (incoming social) | Editors they watch/follow | Watchlist overlap proxy |
| **Σ_j β_ji Λ_qi (outgoing social)** | **Editors watching their pages** | **Page watchers count** |

### Primary Hypothesis

**H_wiki:** Editors with more page watchers (high outgoing social mass) show
smaller revert-recovery rates—they are less likely to accept reverts of their
edits and more likely to re-revert.

### Operationalization

1. Select contentious articles (identified by high revert rates)
2. For each editor, measure:
   - `outgoing_mass`: sum of watchers across pages they primarily edit
   - `revert_acceptance_rate`: fraction of reverts the editor does NOT re-revert
   - `edit_persistence`: fraction of their edits surviving 30+ days
3. Test: Spearman correlation between outgoing_mass and revert_acceptance_rate
4. Prediction: ρ < 0 (more watched editors accept fewer reverts)

### Statistical Design

- **Primary test:** Spearman rank correlation
- **Controls:** Editor tenure, admin status, topic area (fixed effects)
- **Expected effect size:** ρ ≈ -0.15 to -0.30
- **Power:** N ≥ 500 editors for 80% power at α = 0.05 with ρ = -0.15
- **Multiple comparisons:** Bonferroni correction across 4 mass components

---

## Experiment 2: Survey of Professional Forecasters — Belief Oscillation

**Directory:** `experiments/spf_inertia/`
**Hypothesis tested:** H2.1 (Belief Oscillation) + H1.2 (Relaxation Time)

### Theory Connection

The SPF provides quarterly probability forecasts from identified panelists over
decades. If beliefs follow second-order dynamics (M·μ̈ + γ·μ̇ + ∇F = 0),
forecast revisions should show non-monotonic trajectories following shocks—
overshoot and oscillation rather than monotonic convergence.

### Dataset

- **Source:** Federal Reserve Bank of Philadelphia, Survey of Professional Forecasters
- **URL:** `https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters`
- **Direct CSV:** `https://www.philadelphiafed.org/surveys-and-data/spf-individual-level-data`
- **Format:** Individual-level panel data, quarterly, 1968–present
- **Access:** Fully public, direct download, no authentication

### Mass Proxy Mapping

| Mass Component | SPF Proxy | Measurement |
|---------------|----------|-------------|
| Λ_p (prior precision) | Forecaster experience | Quarters in panel |
| Λ_o (observation precision) | Historical accuracy | 1 / RMSE of past forecasts |
| Σ_k β_ik Λ̃_qk (incoming) | Consensus proximity | 1 / |forecast - median| |
| Σ_j β_ji Λ_qi (outgoing) | Forecast influence | Correlation of their forecast with subsequent consensus shift |

### Primary Hypotheses

**H_spf_1 (Oscillation):** Following macroeconomic shocks (identified ex post),
individual forecast revision sequences show non-monotonic trajectories
(sign changes in Δforecast) more often than predicted by AR(1) noise.

**H_spf_2 (Relaxation time):** Experienced forecasters (high Λ_p) take longer
to converge to post-shock consensus than inexperienced forecasters.
τ_experienced / τ_inexperienced > 1, scaling with experience ratio.

**H_spf_3 (Overshoot):** High-precision forecasters (low historical RMSE)
overshoot the eventual consensus more than low-precision forecasters when
they do eventually update. d_overshoot ∝ √(Λ_p).

### Operationalization

1. Identify macroeconomic shocks: quarters where consensus GDP/inflation
   forecast shifts by > 1 SD from prior quarter
2. Track individual forecaster trajectories for 4-8 quarters post-shock
3. Classify trajectories: monotonic, single-overshoot, oscillatory
4. Test oscillation prevalence against AR(1) null model
5. Test relaxation time scaling with experience

### Statistical Design

- **Primary test:** χ² test for oscillation prevalence vs. null model
- **Secondary:** OLS regression of relaxation time on experience
- **Expected effect size:** 15-25% of trajectories oscillatory (vs. ~5% under null)
- **Power:** ~200 forecaster-shock episodes available from 1990-present
- **Shock identification:** Pre-registered using NBER recession dates + ≥1 SD consensus shifts

---

## Experiment 3: Stack Overflow Reputation → Answer Rigidity

**Directory:** `experiments/stackoverflow_inertia/`
**Hypothesis tested:** H3.1 (Influence → Rigidity) + H4.2 (Confirmation Bias)

### Theory Connection

Stack Overflow reputation is a direct, quantified measure of social influence.
The outgoing social inertia term predicts that high-reputation users should
be less likely to edit their own answers in response to comments, competing
answers, or downvotes.

### Dataset

- **Source:** Stack Exchange Data Dump (quarterly)
- **URL:** `https://archive.org/details/stackexchange`
- **Direct:** `https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z`
- **Format:** XML/CSV, ~60GB compressed
- **Access:** Fully public under CC BY-SA, no authentication
- **Lighter alternative:** Stack Exchange Data Explorer (SQL queries, `https://data.stackexchange.com/`)

### Mass Proxy Mapping

| Mass Component | SO Proxy | Measurement |
|---------------|---------|-------------|
| Λ_p (prior precision) | User reputation score | log(reputation) |
| Λ_o (observation precision) | Tag expertise score | Answers in tag / total answers |
| Σ_k β_ik Λ̃_qk (incoming) | Users they follow/interact with | Comment network in-degree |
| **Σ_j β_ji Λ_qi (outgoing)** | **Answer view count + upvotes** | **Σ views × score** |

### Primary Hypothesis

**H_so:** High-reputation users edit their answers less frequently in response
to critical comments or competing higher-voted answers, controlling for
answer quality.

### Operationalization

1. Sample 10,000+ answers that received critical comments (identified by
   question marks, disagreement keywords, or competing answers)
2. For each answer, measure:
   - `author_reputation`: reputation at time of posting
   - `edit_response`: whether author edited within 48h of criticism
   - `edit_magnitude`: character-level diff if edited
3. Control for: answer score, question difficulty, tag, time since posting

### Statistical Design

- **Primary test:** Logistic regression: P(edit | criticism) ~ reputation + controls
- **Expected effect:** OR ≈ 0.85 per SD increase in log(reputation)
- **Power:** N = 10,000 answers, ~80% power for OR = 0.85
- **Data Explorer query** (immediately executable, no download needed):

```sql
-- Stack Exchange Data Explorer query
SELECT
    p.Id AS answer_id,
    p.OwnerUserId,
    u.Reputation,
    p.Score AS answer_score,
    p.CreationDate,
    p.LastEditDate,
    COUNT(c.Id) AS comment_count,
    CASE WHEN p.LastEditDate > MIN(c.CreationDate) THEN 1 ELSE 0 END AS edited_after_comment
FROM Posts p
JOIN Users u ON p.OwnerUserId = u.Id
LEFT JOIN Comments c ON c.PostId = p.Id
WHERE p.PostTypeId = 2  -- Answers only
    AND p.Score >= 0     -- Non-negative score (not spam)
GROUP BY p.Id, p.OwnerUserId, u.Reputation, p.Score, p.CreationDate, p.LastEditDate
HAVING COUNT(c.Id) >= 1  -- At least one comment
ORDER BY u.Reputation DESC
```

---

## Experiment 4: OpenAlex Scientific Citation Inertia

**Directory:** `experiments/openalex_inertia/`
**Hypothesis tested:** H1.2 (Relaxation Time) + H3.1 (Influence → Rigidity)

### Theory Connection

When a paper is retracted, citing authors face a "debunking" event. The epistemic
inertia framework predicts that highly-cited authors (high outgoing social mass)
take longer to stop citing the retracted work, and their subsequent papers show
slower belief revision.

### Dataset

- **Source:** OpenAlex (free, open citation database)
- **API:** `https://api.openalex.org/` (no key required, polite pool)
- **Retraction Watch:** `https://retractionwatch.com/retraction-watch-database-user-guide/`
- **CrossRef retraction data:** `https://api.crossref.org/` (open)
- **Access:** Fully public, no authentication (polite email header only)

### Mass Proxy Mapping

| Mass Component | OpenAlex Proxy | Measurement |
|---------------|---------------|-------------|
| Λ_p (prior precision) | Author h-index / career citations | log(citation_count) |
| Λ_o (observation precision) | Publications in field | works_count in concept |
| Σ_k β_ik Λ̃_qk (incoming) | Co-author network | cited_by_count of co-authors |
| **Σ_j β_ji Λ_qi (outgoing)** | **Author's own cited_by_count** | **Total citations received** |

### Primary Hypotheses

**H_retract_1 (Relaxation time):** After a retraction event, highly-cited
authors continue citing the retracted paper (or its claims) for more
subsequent publications than lowly-cited authors.
τ_high / τ_low > 1.

**H_retract_2 (Influence rigidity):** Authors who are themselves highly cited
(high outgoing social mass) show slower post-retraction citation decay
than less-cited authors, controlling for field and career stage.

### Operationalization

1. Get list of retracted papers from Retraction Watch / CrossRef
2. For each retracted paper, find all authors who cited it pre-retraction
3. Track post-retraction citing behavior: do they continue citing?
4. Measure "relaxation time" = quarters until citation rate drops to baseline
5. Correlate with author citation count (outgoing mass proxy)

### Statistical Design

- **Primary test:** Cox proportional hazards model for time-to-stop-citing
- **Covariates:** h-index, field, career stage, retraction severity
- **Expected HR:** 0.80-0.90 (high-cited authors slower to stop)
- **Power:** ~500 retracted papers × ~20 citing authors each = 10,000 episodes
- **Pre-registration:** Define "stopped citing" as zero citations in 2+ consecutive years

---

## Experiment 5: ANES Panel — Political Belief Persistence

**Directory:** `experiments/anes_inertia/`
**Hypothesis tested:** H4.1 (Perseverance ∝ Precision) + H3.4 (Context-Dependent Stubbornness)

### Theory Connection

ANES panel studies re-interview the same respondents across election cycles.
The framework predicts that belief persistence (stability across waves) scales
with prior precision (confidence/certainty) not with emotional valence or
partisan identity per se.

### Dataset

- **Source:** American National Election Studies
- **URL:** `https://electionstudies.org/data-center/`
- **Key panels:**
  - 2016-2020 Panel Study
  - 2020-2024 Panel Study (if available)
  - ANES Cumulative Data File (1948-2020)
- **Format:** Stata (.dta), SPSS (.sav), CSV
- **Access:** Free registration required (but fully public data)

### Mass Proxy Mapping

| Mass Component | ANES Proxy | Variable |
|---------------|-----------|----------|
| Λ_p (prior precision) | Response certainty | "How certain are you?" (1-5 scale) |
| Λ_o (observation precision) | Political knowledge score | civics quiz items |
| Σ_k β_ik Λ̃_qk (incoming) | Discussion network size | "How many people discuss politics?" |
| Σ_j β_ji Λ_qi (outgoing) | Persuasion attempts | "Try to convince others?" |

### Primary Hypotheses

**H_anes_1 (Precision → Persistence):** Controlling for partisan strength and
emotional valence, response certainty (Λ_p proxy) predicts cross-wave
stability of issue positions better than partisan identity.

**H_anes_2 (Context-dependent stubbornness):** The same individual shows
different stability levels across issues, predicted by their issue-specific
certainty, not by a global "stubbornness" trait.

**H_anes_3 (Discussion network → rigidity):** Respondents who report trying
to persuade others (high outgoing social mass) show greater cross-wave
stability than those who don't, controlling for certainty.

### Operationalization

1. Select issue items measured identically across panel waves
2. Compute per-issue stability: |position_t2 - position_t1|
3. Regress stability on certainty, partisanship, emotional engagement
4. Within-person analysis: do certainty differences across issues predict
   differential stability across those same issues?

### Statistical Design

- **Primary test:** Multilevel model (issues nested within respondents)
- **Fixed effects:** Certainty, partisanship, emotional engagement, knowledge
- **Random effects:** Respondent intercept, issue intercept
- **Expected effect:** β_certainty > β_partisanship (certainty dominates)
- **Power:** N ≈ 3,000 respondents × 8 issues = 24,000 observations
- **Critical comparison:** Does certainty predict persistence BEYOND partisanship?

---

## Experiment 6: Financial Analyst Forecast Revisions

**Directory:** `experiments/financial_inertia/`
**Hypothesis tested:** H1.1 (Overshoot ∝ √Precision) + H2.2 (Resonance)

### Theory Connection

Financial analysts issue quarterly EPS forecasts and revise them. The framework
predicts that high-reputation analysts (high mass) should show larger overshoot
when they do revise, and their revision sequences should show oscillation.

### Dataset

- **Source:** Yahoo Finance API (free) + Federal Reserve SPF
- **Alternative:** Estimize (crowd-sourced estimates, `https://www.estimize.com/api`)
- **SPF URL:** `https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters`
- **Yahoo Finance:** `yfinance` Python package (free, no API key)
- **Access:** Fully public

### Mass Proxy Mapping (SPF version)

| Mass Component | SPF Proxy | Measurement |
|---------------|----------|-------------|
| Λ_p (prior) | Quarters in panel | experience_quarters |
| Λ_o (observation) | Historical accuracy | 1/RMSE over past 8 quarters |
| Σ β_ik (incoming) | Distance from consensus | 1/|forecast - median| |
| Σ β_ji (outgoing) | Influence on consensus | Granger-causality of their revision on subsequent consensus |

### Primary Hypotheses

**H_fin_1 (Overshoot scaling):** When forecasters revise toward consensus after
a shock, those with higher historical accuracy (high Λ_o) overshoot the
eventual settled value by more: d_overshoot ∝ √(accuracy).

**H_fin_2 (Oscillation detection):** Individual forecast revision sequences
show more sign reversals (Δforecast changes sign) than predicted by
random walk, consistent with underdamped oscillation.

**H_fin_3 (Relaxation time):** Experienced forecasters (many quarters in panel)
take more quarters to converge to post-shock consensus.

### Statistical Design

- **Primary test:** Regression of overshoot on √(accuracy) with robust SEs
- **Oscillation test:** Runs test for sign changes in revision sequences
- **Expected:** 20-30% excess sign reversals vs. random walk null
- **Power:** ~150 forecasters × ~40 shocks (recessions, policy changes) = 6,000 episodes

---

## Experiment 7: Reddit Echo Chamber Threshold

**Directory:** `experiments/reddit_echo_chambers/`
**Hypothesis tested:** H3.2 (Echo Chamber Formation Threshold)

### Theory Connection

The framework predicts stable polarization when group belief separation exceeds
‖μ_A - μ_B‖² > 2σ²κ log(N). Reddit's subreddit structure provides natural
"groups" with measurable belief separation (sentiment), uncertainty (variance),
and population size (subscriber count).

### Dataset

- **Source:** Reddit comments via Academic Torrents / Pushshift archives
- **URL:** `https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10`
- **Alternative:** Reddit API (limited but free)
- **Subreddit statistics:** `https://subredditstats.com/` + Reddit API
- **Access:** Fully public

### Mass Proxy Mapping

| Theory Variable | Reddit Proxy | Measurement |
|----------------|-------------|-------------|
| ‖μ_A - μ_B‖ | Sentiment divergence | Mean sentiment difference between paired subreddits |
| σ² | Within-subreddit variance | Variance of sentiment scores |
| κ | Cross-posting temperature | Rate of cross-subreddit commenting |
| N | Subreddit size | Subscriber count |

### Primary Hypothesis

**H_reddit:** Subreddit pairs that satisfy ‖μ_A - μ_B‖² > 2σ²κ log(N) show
stable polarization (low cross-posting, high sentiment divergence over time),
while pairs below this threshold show convergence.

### Operationalization

1. Select paired subreddits on same topic (e.g., r/politics vs r/conservative,
   r/vegan vs r/meat, r/atheism vs r/christianity)
2. Compute sentiment scores per subreddit per month
3. Measure cross-posting rates as proxy for κ
4. Test whether the threshold formula predicts which pairs polarize

---

## Experiment 8: Wikipedia Revert Wars — Oscillation Detection

**Directory:** `experiments/wikipedia_oscillation/`
**Hypothesis tested:** H2.1 (Belief Oscillation) — the flagship prediction

### Theory Connection

Edit wars on Wikipedia are direct observations of belief oscillation: an article's
content oscillates between two states as editors revert each other. The framework
predicts that oscillation frequency should scale as ω ≈ √(K/M), where K is the
"evidence strength" (number of reliable sources supporting each side) and M is
the epistemic mass of the editors.

### Dataset

- **Source:** Wikipedia API for article revision histories
- **URL:** `https://en.wikipedia.org/w/api.php`
- **Edit war articles:** `https://en.wikipedia.org/wiki/Wikipedia:Lamest_edit_wars`
- **Access:** Fully public, no authentication

### Primary Hypothesis

**H_oscillation:** Articles in edit wars show oscillation with frequency that
scales inversely with the combined reputation (edit count, admin status) of
the editors involved: ω ∝ 1/√(M_total).

### Operationalization

1. Identify articles with high revert rates (edit war detection)
2. Extract content oscillation: track specific sentences/claims over time
3. Measure oscillation period (time between reverts)
4. Correlate with editor mass proxies

---

## Immediate Execution Priority

### Tier 1 — Run This Week (no downloads needed)

| Experiment | Why First | Effort |
|-----------|----------|--------|
| **2. SPF Forecasters** | Direct CSV download, clean panel data, tests oscillation | 1 day |
| **3. Stack Overflow** | SQL query on Data Explorer, immediate results | 1 day |
| **7. Manifold Markets** | Already built, just run | Hours |

### Tier 2 — Run This Month (small downloads)

| Experiment | Why Second | Effort |
|-----------|-----------|--------|
| **4. OpenAlex Citations** | API calls, no bulk download | 2-3 days |
| **1. Wikipedia Editors** | API calls for specific articles | 2-3 days |
| **8. Wikipedia Oscillation** | API calls for edit war articles | 2 days |

### Tier 3 — Run Next Month (larger datasets)

| Experiment | Why Third | Effort |
|-----------|----------|--------|
| **5. ANES Panel** | Registration required, complex survey weights | 1 week |
| **6. Financial Analysts** | Requires careful shock identification | 1 week |
| **7. Reddit Echo Chambers** | Large data download needed | 1-2 weeks |
