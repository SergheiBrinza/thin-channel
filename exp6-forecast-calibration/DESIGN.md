# Experiment 6 — "Thin Channel"

*How little external, unforgeable signal is needed to keep a self-training system honest?*

Start date: 2026-05-30

---

## Big goal

The external signal is truth from the future (resolutions of forecasting questions).
The forecast is made now, the outcome opens later at a deadline — it cannot be faked.
That is the "unforgeable" anchor.

We are looking for the threshold frequency of this anchor: at what reveal frequency
does the system still hold its calibration, and at what point does it start to drift.

---

## Frozen design (along 8 forks)

| # | Fork | Decision |
|---|---|---|
| 1 | Source of questions | **ForecastBench as the base** + Manifold as a supplement |
| 2 | Question type | **binary only** (clean Brier) |
| 3 | Model | **Qwen2.5-3B-Instruct** (on a stronger model the thin-channel effect is harder to see; 7B is for later verification) |
| 4 | What "learns" | **per-topic Beta calibration** (external notebook, model weights frozen) |
| 5 | Frequency grid | **6 points:** `day / 3 days / week / month / quarter / never` |
| 6 | Pilot length | **200–300 questions** in the stream |
| 7 | Hold-out | **end of the stream (~20%)** as "the future on which we measure calibration" |
| 8 | Parallel on 3 GPUs | **no**, sequential (parallelism only for the final run if needed) |

---

## Experiment mechanics

### Simulation of time

Time runs in ticks ("days"). Each tick:

1. Morning: a new question appears (from ForecastBench / Manifold).
   The system sees only the wording and the deadline.
2. Forecast: frozen Qwen2.5-3B + current calibration notebook produce `P(YES)`.
   The record `(question_id, P, t_predict)` is stored.
3. Evening: for every old question whose deadline has arrived and whose frequency mode
   allows reveal at this tick, the truth (YES/NO) is revealed.
4. Measure and update: for revealed questions:
   - compute Brier: `(P - outcome)^2` (outcome = 0 or 1);
   - update the per-topic Beta in the notebook (mechanics in the next section).
5. Next tick.

### What gets updated (per-topic Beta calibration)

The external notebook `calibration_notebook.json` stores, for each topic
(LLM-assigned label: "politics", "science", "sports", ...), a pair of parameters `(alpha, beta)`
of a Beta distribution — the aggregate statistic of how often forecasts in this topic
turned out to be YES.

On resolution of a question with forecast `P` and outcome `y` in {0, 1}:
- `alpha += y` (weight of YES truth)
- `beta += (1 - y)` (weight of NO truth)

Current "calibration base" for the topic: `base = alpha / (alpha + beta)`.

Before each new forecast:
1. The LLM emits a raw probability `P_raw`.
2. Topic correction: `P_final = mix(P_raw, base, lam)` where `lam` is the weight of
   confidence in calibration (depends on the data volume `alpha+beta`).
3. The exact form of correction is fixed during build; we start with simple linear mixing.

Model weights are never touched. The notebook is the only "trainable" component. After
a reload the notebook is read from disk again.

### What we measure

- Brier score on hold-out (last ~20% of the stream) — the main metric.
- Calibration curve — how well frequencies match predicted ones.
- Drift over time — does Brier grow towards the end of the simulation (the system
  drifts) or stay stable.

Main plot: x-axis is signal frequency (day/3days/week/month/quarter/never), y-axis is
final hold-out Brier. Hypothesis: there is a threshold frequency below which Brier rises
sharply. That threshold = "the thickness of the thin channel".

---

## Build plan, step by step

1. Data: download ForecastBench, filter binary with resolution, normalize into a common
   format. Add a Manifold supplement.
2. Topics: run every question through an LLM topic classifier (one pass), freeze the labels.
3. Pilot baseline: run the bare model without the notebook on ~50 questions, confirm
   that Brier is sensible (<< 0.5) and that the questions are non-trivial.
4. Notebook mechanics: implement per-topic Beta, the correction shape `P_raw -> P_final`,
   the update rule.
5. Tick simulator: a script for a single "day" (forecast new + reveal old by calendar +
   update notebook).
6. Run the 6 modes: the same stream of questions through 6 reveal schedules
   (day / 3 days / week / month / quarter / never).
7. Hold-out measurement: at the end of each mode compute Brier on hold-out.
8. Summary: plot "frequency -> Brier", calibration curves, drift over time.

---

## Out of scope for this experiment

- Forecast cost (length, tokens) — not measured.
- Manifold's game-like nature (meme markets) — for the pilot we just include the top-N
  binaries with real resolutions; deep noise analysis is a separate task.
- "Does the model itself understand" — no, just like in previous experiments: the entire
  honesty machinery lives in the external notebook; model weights are not touched.
- Complex multi-category outcomes — binary only.
