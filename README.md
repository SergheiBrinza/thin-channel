# Thin Channel — how rarely external truth can arrive before a self-calibrating system collapses

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21845022.svg)](https://doi.org/10.5281/zenodo.21845022)

Part 3 of 3 of the Second Loop project.

## What this repo contains

One experiment that probes the lower bound of external feedback. Repo 1
defined a metric for whether a correction to a frozen LLM survives a process
reload; repo 2 showed that a live external source (Wikipedia) heals the
system's blind spots. This repo asks the opposite question: how rarely can
that external source arrive before calibration drifts?

* `exp6-forecast-calibration/` — uses ForecastBench (1,646 resolved binary
  forecast questions, July 2024 — December 2025) as a stream of "truth from
  the future". A frozen Qwen2.5-3B-Instruct produces a raw `P(YES)` for each
  question at its freeze date. The outcome is hidden and revealed only on a
  schedule. A per-topic Beta calibration (9 topics, parameter `K=10`) sits on
  top of the model — no weight updates — and consumes each revealed outcome.
  The same stream is run independently through 11 reveal schedules: day, 3
  days, week, 2 weeks, month, 2 months, quarter, half year, 270 days, year,
  never. Hold-out Brier (last 20% by freeze date, N=330) is the metric.

## Main result

The collapse is binary. All ten finite schedules — from daily down to once a
year — sit in Brier `[0.205, 0.220]`. Only `never` collapses to `0.298`,
bit-for-bit matching the raw 3B baseline. Neither frequency nor delivered
volume is the binding constraint here: the 270-day schedule delivers 220
resolutions and reaches `0.216`, the daily schedule delivers 1,316 and reaches
`0.205` — a six-fold difference in signal moves the score by `0.011`, while the
gap to no contact at all is `0.08`–`0.09`. With 9 topics at `K=10` the
calibration saturates after a few hundred resolutions, so the experiment
separates contact from no contact and says nothing finer.

## Scope and limitations

Five points that should be read before the result is taken at face value.

(0) The subject model is frozen and decoding is greedy, so a rerun reproduces
an identical state. Across the Second Loop series the headline 100% figures are
partly a property of that determinism; what we stand behind is the shape of the
result, not the round number.

(a) The raw Qwen2.5-3B is useless as a forecaster. On 250 random questions
its Brier is `0.354`, worse than an `always NO` baseline (`0.340`) and much
worse than `always topic base rate` (`0.224`). After the calibration run, the
three ablations on hold-out (N=330) give: raw model only `0.298`; topic base
rate only `0.189`; mixed model+base `0.189`. The difference between the last
two rounds to `0.0000`. The entire benefit of calibration comes from the
system picking up per-topic base rates — not from the model learning to
forecast. This experiment therefore measures whether the system calibrates
from an external signal, not whether it learned to predict the future. On a
stronger forecaster the two questions would split.

(b) The binary cliff is partly an artifact of batch-on-reveal logic. Every
reveal flushes the entire accumulated queue at once. Delivered volume still
varies a lot by period — 1,316 resolutions for the daily schedule, 417 for the
annual one, 220 for the 270-day one — but every one of those totals is already
past the convergence point for 9 topics at `K=10`, which is why the scores stay
within `0.015` of each other. A bounded-batch variant would make frequency
itself a bottleneck and is left for follow-up work.

(c) The stream is 17 months long, so any reveal period longer than the stream
is operationally equivalent to `never`. The result also depends on `K=10` and
on the coarse 9-topic taxonomy: at finer granularity each topic accumulates
fewer updates per reveal and rare schedules may stop holding.

(d) Exact Brier values vary at the third decimal between independent runs
because vLLM generation is not bit-reproducible on GPU. The qualitative
result is stable across runs: the binary cliff (only `never` collapses), the
~0.21 corridor for all finite schedules, and the c ≈ b equality (the model
adds nothing over per-topic base rates) all hold. The numbers committed here
are from a single representative run reproducible via the pipeline above.

## The wider series

* Repo 1: `scar-survival` — the correction mechanism and metric.
* Repo 2: `external-grounding` — defending the notebook with a live external
  source of truth (Wikipedia).
* Repo 3: `thin-channel` (this one) — the lower bound on how rarely that
  external truth can arrive.

## Setup

This repo is self-contained: `exp6-forecast-calibration/` does not import
anything from `scar-survival` or `external-grounding`. Only two external
dependencies need to be set up by hand: the dataset and the two Qwen models.

```bash
# (a) install pinned deps
pip install -r requirements.txt

# (b) download the two Qwen checkpoints (~6 GB + ~14 GB)
hf download Qwen/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct

# (c) download ForecastBench into exp6-forecast-calibration/forecastbench_raw/
#     (CC BY-SA 4.0; not redistributed in this repo — see DATA.md)
hf download forecastingresearch/forecastbench-datasets \
    --repo-type dataset \
    --local-dir exp6-forecast-calibration/forecastbench_raw
```

After (c), the scripts in `exp6-forecast-calibration/` find their input under
`forecastbench_raw/datasets/{question_sets, resolution_sets, forecast_sets}/`.

## How to reproduce

```bash
cd exp6-forecast-calibration
python3 prepare_forecastbench.py    # build forecastbench_clean.json: normalize
                                    # 1,646 resolved binary questions and
                                    # substitute date placeholders into
                                    # question_text / background_text
python3 classify_topics.py          # Qwen2.5-7B labels each question with
                                    # one of 9 topics (~2 min)
python3 pilot_baseline.py           # raw 3B over 50 random questions (sanity)
python3 wide_baseline.py            # raw 3B over 250 with per-topic breakdown
python3 gen_raw_predictions.py      # raw 3B over all 1,646 (single batched pass)
python3 run_calibration.py          # per-topic Beta + 3 ablations on hold-out
python3 run_frequency_sweep.py      # 11 schedules + main plot
```

Total wall-clock on one RTX 3090 Ti (24 GB): about 30 minutes. The
calibration and frequency-sweep stages are pure CPU; only topic labelling and
the raw-prediction passes use the GPU.

## Requirements

* Python 3.10+, vLLM, matplotlib (see `requirements.txt`)
* One NVIDIA GPU with at least 24 GB VRAM for the two Qwen passes
  (`classify_topics.py`, `pilot_baseline.py`, `wide_baseline.py`,
  `gen_raw_predictions.py`). The calibration and sweep scripts are CPU only.

## Data and models

See `DATA.md` — covers the ForecastBench dataset (CC BY-SA 4.0, not bundled),
the two HuggingFace models, and the exact commands above.

## Citation

```bibtex
@misc{brinza2026secondloop,
  author       = {Brinza, Serghei},
  title        = {The Second Loop: Durable Correction of Memorized Errors in Frozen Language Models},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21845022},
  url          = {https://doi.org/10.5281/zenodo.21845022}
}
```

## License

See [LICENSE](LICENSE) — MIT. The dataset processed by these scripts
(ForecastBench) is under CC BY-SA 4.0 and is not redistributed in this repo —
the user downloads it from the original source.
