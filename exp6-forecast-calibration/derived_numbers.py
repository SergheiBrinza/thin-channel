#!/usr/bin/env python3
"""Recompute the derived numbers quoted in the paper (v3) from the run outputs.

Part A uses committed aggregates only: frequency_sweep.json, calibration_summary.json.
Part B needs raw_predictions.json, which is not redistributed (see DATA.md);
regenerate it with gen_raw_predictions.py. Without it, Part B is skipped and
marked as such in the output.
Writes derived_numbers.json (aggregates only, no question text or outcomes).
"""
import os
import json
from datetime import datetime
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_FRAC = 0.80  # same split as run_calibration.py / run_frequency_sweep.py


def parse_dt(s):
    # same day-level parsing as run_frequency_sweep.py
    if len(s) == 10:
        return datetime.strptime(s, "%Y-%m-%d").date()
    return datetime.fromisoformat(s.replace("Z", "+00:00")).date()


def part_a():
    sweep = json.load(open(os.path.join(HERE, "frequency_sweep.json")))["schedules"]
    calib = json.load(open(os.path.join(HERE, "calibration_summary.json")))["holdout_global"]
    day = next(s for s in sweep if s["name"] == "day")
    never = next(s for s in sweep if s["period_days"] is None)
    finite = [s for s in sweep if s["period_days"] is not None]
    vals = [s["holdout_brier_final"] for s in finite]
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    return {
        "offline_end_of_stream": {
            "brier_raw": calib["brier_a_raw"],
            "brier_base_only": calib["brier_b_base_only"],
            "brier_mix": calib["brier_c_mix"],
            "gap_raw_minus_mix": round(-calib["delta_c_vs_a"], 4),
            "gap_base_minus_mix": round(-calib["delta_c_vs_b"], 4),
        },
        "online_daily": {
            "brier_raw": round(day["holdout_brier_raw"], 4),
            "brier_base_only": round(day["holdout_brier_base_only"], 4),
            "brier_mix": round(day["holdout_brier_final"], 4),
            "gap_raw_minus_mix": round(day["holdout_brier_raw"] - day["holdout_brier_final"], 4),
            "gap_base_minus_mix": round(day["holdout_brier_base_only"] - day["holdout_brier_final"], 4),
        },
        "finite_schedules": {
            "n_schedules": len(finite),
            "brier_min": round(lo, 4),
            "brier_min_schedule": finite[vals.index(lo)]["name"],
            "brier_max": round(hi, 4),
            "brier_max_schedule": finite[vals.index(hi)]["name"],
            "spread": round(spread, 4),
        },
        "never_vs_finite": {
            "brier_never": round(never["holdout_brier_final"], 4),
            "gap_min": round(never["holdout_brier_final"] - hi, 4),
            "gap_max": round(never["holdout_brier_final"] - lo, 4),
            "gap_min_over_spread": round((never["holdout_brier_final"] - hi) / spread, 2),
            "gap_max_over_spread": round((never["holdout_brier_final"] - lo) / spread, 2),
        },
    }


def part_b():
    path = os.path.join(HERE, "raw_predictions.json")
    if not os.path.exists(path):
        return {"skipped": "raw_predictions.json not found; run gen_raw_predictions.py"}
    rows = json.load(open(path))["rows"]
    rows.sort(key=lambda r: parse_dt(r["freeze_datetime"]))
    n_train = int(len(rows) * TRAIN_FRAC)
    train, hold = rows[:n_train], rows[n_train:]
    start = parse_dt(rows[0]["freeze_datetime"])
    train_res = sorted(parse_dt(r["resolution_date"]) for r in train)
    hold_freeze = [parse_dt(h["freeze_datetime"]) for h in hold]
    # outcomes revealed before a hold-out forecast under the daily schedule
    # (forecast in the morning, reveal in the evening -> strictly earlier days)
    visible = [sum(1 for x in train_res if x < f) for f in hold_freeze]
    return {
        "n_total": len(rows), "n_train": n_train, "n_holdout": len(hold),
        "lookahead": {
            "train_outcomes_visible_at_holdout_forecast_min": min(visible),
            "train_outcomes_visible_at_holdout_forecast_median": median(visible),
            "train_outcomes_visible_at_holdout_forecast_max": max(visible),
            "train_outcomes_resolved_after_last_holdout_forecast":
                sum(1 for x in train_res if x > hold_freeze[-1]),
            "holdout_forecasts_using_future_outcomes_offline":
                sum(1 for f in hold_freeze if any(x >= f for x in train_res)),
        },
        "holdout_forecasts_before_day_365": sum(1 for f in hold_freeze if (f - start).days < 365),
        "first_holdout_forecast_day": (hold_freeze[0] - start).days,
    }


def main():
    out = {"committed_aggregates": part_a(), "from_raw_predictions": part_b()}
    json.dump(out, open(os.path.join(HERE, "derived_numbers.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("\nSaved derived_numbers.json")


if __name__ == "__main__":
    main()
