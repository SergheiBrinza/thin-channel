# Experiment 6 — Thin Channel

How rarely can external truth arrive before a self-calibrating system drifts?
A frozen Qwen2.5-3B forecaster plus per-topic Beta calibration is run against
ForecastBench's 1,646 resolved binary questions through 11 reveal schedules
(day, 3 days, week, 2 weeks, month, 2 months, quarter, half year, 270 days,
year, never).

The full design and findings live in `DESIGN.md` and
`FINDINGS_thin_channel.md`. This file is the quickstart and the file map.

## Quickstart

Set up the dataset and Qwen models first (see top-level `DATA.md`), then:

```bash
python3 prepare_forecastbench.py    # normalize 1,646 resolved binary Qs and
                                    # substitute date placeholders ->
                                    # forecastbench_clean.json
python3 classify_topics.py          # Qwen2.5-7B topic labels (9 topics)
python3 pilot_baseline.py           # raw 3B over 50 random Qs (sanity check)
python3 wide_baseline.py            # raw 3B over 250 with per-topic breakdown
python3 gen_raw_predictions.py      # raw 3B over all 1,646 (single batched pass)
python3 run_calibration.py          # per-topic Beta + 3 ablations (hold-out)
python3 run_frequency_sweep.py      # 11 schedules + main plot
```

GPU usage: only steps `classify_topics`, `pilot_baseline`, `wide_baseline`,
and `gen_raw_predictions` touch the GPU. `run_calibration` and
`run_frequency_sweep` are pure CPU and finish in seconds.

## Files

### Documentation
- `DESIGN.md` — frozen design across 8 forks (data, model, calibration form,
  frequency grid, hold-out split)
- `FINDINGS_thin_channel.md` — full report: design, baselines, ablations, the
  binary-cliff result, honest scope-of-claim

### Scripts (committed)
- `prepare_forecastbench.py` — extract resolved atomic binary questions from
  the raw ForecastBench dump, normalize them, and substitute date placeholders
  ({forecast_due_date}, {resolution_date}) into model-facing question_text /
  background_text fields
- `classify_topics.py` — 9-topic classifier (Qwen2.5-7B via vLLM)
- `pilot_baseline.py` — raw 3B baseline on 50 random questions
- `wide_baseline.py` — raw 3B baseline on 250 questions with per-topic breakdown
- `gen_raw_predictions.py` — raw 3B `P(YES)` over all 1,646 questions
- `run_calibration.py` — per-topic Beta calibration + 3 ablations (raw / base
  only / mixed) + per-topic regression check
- `run_frequency_sweep.py` — tick simulator; runs the stream through 11
  reveal schedules and writes the main plot

### Results (committed)
- `frequency_sweep.json` — Brier per schedule (final + base-only + raw + total
  revealed)
- `frequency_sweep.png` — main plot (Brier vs reveal period + learning curves)
- `wide_baseline_summary.json` — aggregate of `wide_baseline.json` with
  per-question rows removed; keeps overall and per-topic metrics
- `calibration_summary.json` — aggregate of `calibration_results.json` with
  per-question hold-out rows removed; keeps `final_beta_per_topic`,
  `holdout_global`, `holdout_per_topic`

### Not committed (regenerate locally; see top-level `DATA.md`)
- `forecastbench_raw/` — raw ForecastBench dump from `hf download`
  (CC BY-SA 4.0, not redistributed)
- `forecastbench_clean.json`, `questions_with_topics.json` — derived dataset
  artifacts (contain question text and resolved outcomes)
- `raw_predictions.json`, `pilot_baseline.json` — model outputs that embed
  full question text and resolved outcomes
- `wide_baseline.json`, `calibration_results.json` — full per-row dumps
  (committed summaries cover the aggregate numbers)
