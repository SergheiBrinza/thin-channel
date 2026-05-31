# Data and Models

This repo follows a strict "code, not data" policy. The processed ForecastBench
artifacts (`forecastbench_normalized.json`, `forecastbench_clean.json`,
`questions_with_topics.json`, `raw_predictions.json`, `pilot_baseline.json`,
`wide_baseline.json`, `calibration_results.json`) are derived from the
ForecastBench dataset and are not redistributed here — the repo contains only
our code and our aggregate results. Anyone reproducing the experiment
downloads the original dataset themselves and regenerates the derived files by
running the scripts.

## External dataset (not included)

* Dataset: `forecastingresearch/forecastbench-datasets` on HuggingFace
  (<https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets>).
* License: CC BY-SA 4.0. Their license terms apply to the downloaded data;
  please check the LICENSE inside that dataset on HuggingFace.
* Paper: Karger, E., Bastani, H., Yueh-Han, C., Jacobs, Z., Halawi, D.,
  Zhang, F., and Tetlock, P. E. *ForecastBench: A Dynamic Benchmark of AI
  Forecasting Capabilities.* ICLR 2025. <https://arxiv.org/abs/2409.19839>

Download:

```bash
hf download forecastingresearch/forecastbench-datasets \
    --repo-type dataset \
    --local-dir exp6-forecast-calibration/forecastbench_raw
```

This places the dataset under
`exp6-forecast-calibration/forecastbench_raw/datasets/{question_sets,
resolution_sets, forecast_sets}/`, which is the layout
`prepare_forecastbench.py` expects.

## Regenerating the derived files

Run, from `exp6-forecast-calibration/`:

| Step | Script | Writes |
|---|---|---|
| 1 | `prepare_forecastbench.py` | `forecastbench_normalized.json`, `forecastbench_clean.json` |
| 2 | `classify_topics.py` | `questions_with_topics.json` (Qwen2.5-7B topic labels) |
| 3 | `pilot_baseline.py` | `pilot_baseline.json` (raw 3B over 50 sample) |
| 4 | `wide_baseline.py` | `wide_baseline.json` (raw 3B over 250 sample, per-topic) |
| 5 | `gen_raw_predictions.py` | `raw_predictions.json` (raw 3B over all 1,646) |
| 6 | `run_calibration.py` | `calibration_results.json` |
| 7 | `run_frequency_sweep.py` | `frequency_sweep.json`, `frequency_sweep.png` |

Steps 1–6 must run in order; step 7 uses the output of step 6.
`frequency_sweep.json`, `frequency_sweep.png`, `wide_baseline_summary.json`,
and `calibration_summary.json` *are* committed in this repo — they are our
aggregate results and contain no ForecastBench question text or per-question
outcomes.

## Models (HuggingFace)

* `Qwen/Qwen2.5-3B-Instruct` — frozen forecaster; produces raw `P(YES)` for
  all 1,646 questions.
* `Qwen/Qwen2.5-7B-Instruct` — topic classifier (`classify_topics.py`).

Download:

```bash
hf download Qwen/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct
```

Both are served via vLLM at fp16. The two Qwen passes use roughly 24 GB VRAM
together; the rest of the pipeline is pure CPU.

## Why "code, not data"

ForecastBench is licensed CC BY-SA 4.0, so any redistribution would need to
carry that license forward and credit the upstream authors. To keep the
publishing chain clean (this repo is MIT) we ship only our code and our
aggregate results, and document above the exact upstream source the user
should fetch themselves.
