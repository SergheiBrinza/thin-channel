# Findings — Experiment 6 — "Thin Channel"

*How little external, unforgeable signal is needed to keep a self-training system honest?*

Completion date: 2026-05-30

Builds on experiments 1–3 (scar mechanics and external source of truth) and continues a
theme that the Live-Evo project touched but did not exhaustively explore.

---

## 1. Goal

Experiment 3 showed that an external unforgeable source of truth (Wikipedia) cures the
system's blind spots — the ceiling of memory protection rose from 64% to 100%. But how
much of this external signal is needed? Giving it constantly is expensive. Giving it
rarely lets the system drift. Where is the boundary?

This experiment looks for the thickness of the thin channel: the minimum frequency
of contact with truth at which the system stays calibrated. Here the external signal is
truth from the future — forecasts are made now, outcomes open later at a deadline,
they cannot be faked.

Live-Evo raised this question and showed that daily contact is enough. We wanted to find
the real boundary — how far the period can be stretched before calibration collapses.

---

## 2. Design

**Data.** We took a curated benchmark of forecasting questions — `ForecastBench` by
forecastingresearch.org. We extracted 1,646 resolved atomic binary questions from
July 2024 to December 2025 (9 sources: Polymarket, yfinance, Wikipedia, FRED, Manifold,
ACLED, etc.). Each question has a date when it was frozen (`freeze_datetime`) and a
date when the outcome became known (`resolution_date`).

**Time simulation.** Since all outcomes are already known historically, we hide them and
reveal them to the system according to a simulated calendar. Each "day":
- morning: questions whose `freeze_datetime` = today appear. The system forecasts a
  probability `P(YES)`;
- evening: questions whose `resolution_date` <= today get queued;
- the queue is unsealed only on reveal days according to the frequency schedule.

**Topics.** A workflow of 4 parallel agents proposed candidate taxonomies, a synthesis step
narrowed them down to 9 topics, a coverage critic verified them on a fresh sample, the
local Qwen2.5-7B ran a classifier over all 1,646 questions (~2 min 23 s), and 3 verifiers
confirmed 83% accuracy on a 60-label spot check. Topics: Politics & Elections, Geopolitics
& Conflict, Macro & Financial Indicators, Markets & Crypto, AI & Tech, Public Health,
Climate & Weather, Sports & Games, Entertainment & Culture.

**Model and calibration.** Subject — Qwen2.5-3B-Instruct (deliberately weak so the signal
effect is clearly visible; repeats the principle from experiment 1). Weights frozen.
Outside the model runs a **per-topic Beta calibration**:
- for each topic we keep parameters `(alpha, beta)` of a Beta distribution, initialized to `(1, 1)`;
- on each revealed outcome `y` in {0, 1} for a topic: `alpha += y`, `beta += (1 - y)`;
- topic base: `base = alpha / (alpha + beta)`, "effective n" = `alpha + beta - 2`;
- mixing formula: `P_final = (1 - lam) * P_raw + lam * base_topic`, where
  `lam = n / (n + K)`, `K = 10` (after ~10 resolutions the calibration carries half the
  weight; after ~50 — almost the whole weight).

**Frequency schedules.** 11 points from daily to "never":
day / 3 days / week / 2 weeks / month / 2 months / quarter / half-year / 270 days /
year / never. The same stream of questions runs through each schedule independently.

**Metric.** Brier score on hold-out (last 20% of the stream by `freeze_datetime`,
N = 330). The lower the Brier, the more accurate the probabilistic forecasts. Reference
points: raw 3B (no calibration) ≈ 0.30; "always topic mean" (calibration ceiling) ≈ 0.19.

---

## 3. Finding about the model itself

Before measuring the thin channel, we checked what the raw 3B gives as a forecaster
in the first place.

Answer: almost nothing. On 250 random questions the baseline measurement gave:
- raw Brier = 0.354, worse than "always NO" (0.340) and much worse than "always base
  rate" (0.224);
- the model systematically overestimates YES (mean P = 0.49 vs. real YES rate of 0.34);
- discrimination (mean P on YES − mean P on NO) — weak and sometimes negative
  (on Politics & Elections the model assigns more probability to NO outcomes than to YES).

After a full calibration run, three ablations on hold-out (N=330) showed:
- (a) raw 3B only: Brier = 0.298;
- (b) topic base rate only (no model at all): Brier = 0.189;
- (c) our mix: Brier = 0.189 (equal to b).

(c) − (b) = 0.0000 (rounded). The raw 3B forecast adds nothing on top of the topic
statistics. The entire benefit of calibration comes from the system learning the base
rates across the 9 topics.

### What this means for the interpretation of the experiment

The experiment does not measure "did the system learn to predict the future". The
experiment measures "did the system manage to calibrate from the external signal" —
that is, how fast Beta for each topic converges to the real base YES rate in that topic.

If (c) were much better than (b), we would be measuring something else — the resilience
of the model's own predictive power. On a stronger model (7B+) that would be the case.
But for the main question (the thickness of the channel) it is not needed: the channel
carries statistics, and the accumulation of those statistics is measured directly.

The regression control passed cleanly. Markets & Crypto, where the model under-predicts
YES (the opposite bias), did not suffer: per-topic Beta for this topic learned its high
base rate (0.57) and shifted P up, not down. Public Health, where the model was
already well-calibrated, was not spoiled: the topic learned that almost all its questions
resolve NO, and Brier improved further. Per-topic calibration, not global, was the
correct design choice.

---

## 4. Main result — the cliff is binary

All 10 finite schedules (from daily to yearly) give Brier in the corridor of 0.205–0.220:

| Frequency | Period | Revealed | Brier (c) |
|---|---|---|---|
| day | 1 day | 1316 | 0.205 |
| 3 days | 3 days | 1315 | 0.205 |
| week | 7 days | 1313 | 0.211 |
| 2 weeks | 14 days | 1313 | 0.210 |
| month | 30 days | 1277 | 0.210 |
| 2 months | 60 days | 1173 | 0.220 |
| quarter | 90 days | 1022 | 0.220 |
| half year | 180 days | 417 | 0.209 |
| 270 days | 270 days | 220 | 0.217 |
| year | 365 days | 417 | 0.220 |
| **never** | inf | **0** | **0.298** |

The difference 0.205 vs. 0.220 is within statistical noise (standard error of Brier on
N=330 ~ +/- 0.015). Half-year is even slightly better than quarter — this is a fluctuation,
not a pattern.

The cliff sits between "year" and "never": 0.22 → 0.30.

A single reveal across the entire stream (even an annual one) is enough to keep Brier
near the lower bound of 0.19. Zero reveals and the system collapses back to the raw 3B
level.

---

## 5. Mechanics — why the cliff is binary

What matters is not frequency but batch logic. When the system unseals the queue once a
year in the evening, it gets in one shot everything accumulated over that year — about
417 resolutions in our stream. That is more than enough for the 9 topics at K=10: each
topic gets about 46 updates on average, Beta has already converged.

Daily reveal trickles in one resolution a day — the stream total is still about 1,300.
The annual reveal is one big batch of 417. Total signal volume is comparable in both
cases.

Conclusion: the thickness of the channel is measured not by the period but by the total
volume of signal that has passed through it across the lifetime of the system. What
matters is not the rhythm of reality checks but how much truth, in total, the system has
seen.

This is stronger than the naive hypothesis "constant verification is needed". In fact:
- daily verification is redundant;
- a single annual reveal with full settlement is almost equivalent;
- full absence is a catastrophe.

---

## 6. The `never` control — passed cleanly

`Brier(never) = 0.2980` matched the raw 3B Brier (0.2980) bit for bit. This confirms
that the simulator is correct: without signal, Beta stays at the starting prior
`(1, 1)`, base = 0.5 for all topics, n = 0, lam = 0, `P_final = P_raw`. Calibration is
off, and hold-out Brier equals the raw level.

The "upper bound without channel" control matters because it distinguishes a genuine
channel effect from statistical coincidence; the exact match shows the metric is clean.

---

## 7. Honest boundaries — what this experiment does not cover

1. Stream length — 17 months. A period longer than the stream is effectively "never"
   (no reveals happen before the end). The ceiling on the testable period equals the
   stream length. Behaviour with a 5-year stream and a 4-year period is untested.
2. K = 10. With smaller K calibration arrives faster, with larger K the model dominates
   longer. Whether the cliff shifts under a different K — not checked.
3. 9 coarse topics. With finer granularity (50 topics) each topic accumulates fewer
   resolutions per reveal and rare schedules might stop holding. Calibration-structure
   granularity is a material parameter.
4. Batch size is unbounded. Each reveal flushes the entire accumulated queue. If the
   batch were limited to, say, the last 50 resolutions, frequency would become the real
   bottleneck.
5. Cross-model check — 3B only. On a stronger model (7B+) the raw forecast itself
   carries signal, and the same setup would also measure the resilience of that signal.
6. Bit-reproducibility. Exact Brier values vary at the third decimal between independent
   runs because vLLM generation is not bit-reproducible on GPU. The qualitative result is
   stable across runs: the binary cliff (only `never` collapses), the ~0.21 corridor for
   all finite schedules, and the c ≈ b equality (the model adds nothing over per-topic
   base rates) all hold. The numbers in this file are from a single representative run
   reproducible via the pipeline above.

These boundaries are material for a follow-up experiment: limit the batch size, extend
the stream, increase the number of topics — and find a continuous threshold for channel
thickness rather than a binary cliff.

---

## 8. Connection to experiments 1-5

The experiments of this loop line up:

- #1 (Scar): the mechanism for correcting a memorized error works and survives reload.
- #2 (Gatekeeper 1.0): a clone of the model as verifier hits a ceiling of 64% —
  same-family blind spots.
- #3 (Gatekeeper 2.0): a real external source of truth lifts the ceiling to 100%.
- #4 + #5 (scar-survival metric validation): the metric is universal and distinguishes
  real corrections from spurious ones.
- #6 (Thin channel): of that same external source of truth, very little is needed —
  one annual revision is enough to keep calibration up.

Experiment 6 confirms and strengthens the Live-Evo project's conclusion. Live-Evo
showed that daily contact with reality is sufficient for the system to stay honest. This
experiment pushes the boundary further: even rare but full contact is enough.

Loop-level conclusion: a system can be honest under remarkably small contact with
external truth — as long as that contact exists, carries enough total signal, and
arrives through an unforgeable channel. Zero contact fails; rare-but-thorough contact
suffices.

---

## 9. Files of experiment 6

### Documentation
- `DESIGN.md` — frozen design across 8 forks
- `FINDINGS_thin_channel.md` — this report
- `README.md` — quickstart and file map

### Data (not committed; regenerate locally — see top-level `DATA.md`)
- `forecastbench_raw/` — ForecastBench dataset downloaded from HuggingFace
  (CC BY-SA 4.0; not redistributed in this repo)
- `forecastbench_clean.json` — 1,646 resolved atomic binary questions with date
  placeholders substituted into `question_text` / `background_text`
- `questions_with_topics.json` — 1,646 questions with topics (9 topics, 7B classifier)
- `raw_predictions.json` — raw forecasts from Qwen2.5-3B over all 1,646 (single batched pass)
- `pilot_baseline.json`, `wide_baseline.json`, `calibration_results.json` — full
  per-question-row outputs (embed ForecastBench outcomes)

### Scripts (committed)
- `prepare_forecastbench.py` — extraction and normalization
- `classify_topics.py` — topic classifier on Qwen2.5-7B
- `pilot_baseline.py` — pilot measurement on 50 random questions
- `wide_baseline.py` — wide measurement on 250 with per-topic breakdown
- `gen_raw_predictions.py` — raw 3B forecasts over all 1,646
- `run_calibration.py` — Beta simulation + 3 ablations + per-topic regression check
- `run_frequency_sweep.py` — tick simulator + run through 11 schedules + plot

### Aggregate results (committed)
- `frequency_sweep.json` — final numbers across 11 schedules
- `frequency_sweep.png` — main plot + learning curves
- `wide_baseline_summary.json` — aggregate-only view of `wide_baseline.json`
  (overall and per-topic metrics, no per-question rows)
- `calibration_summary.json` — aggregate-only view of `calibration_results.json`
  (`final_beta_per_topic`, `holdout_global`, `holdout_per_topic`)
