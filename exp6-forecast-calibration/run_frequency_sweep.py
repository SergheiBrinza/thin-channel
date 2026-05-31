#!/usr/bin/env python3
"""Step 5+6: tick simulator. Same stream of questions through 11 reveal schedules:
day / 3 days / week / 2 weeks / month / 2 months / quarter / half year / 270 days / year / never.
Computes Brier on hold-out + learning curves. Writes frequency_sweep.json + frequency_sweep.png."""
import os
import json
from datetime import datetime, date, timedelta
from copy import deepcopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10
TRAIN_FRAC = 0.80
SCHEDULES = [
    ("day", 1), ("3 days", 3), ("week", 7), ("2 weeks", 14),
    ("month", 30), ("2 months", 60), ("quarter", 90),
    ("half year", 180), ("270 days", 270), ("year", 365),
    ("never", None),
]


def parse_dt(s):
    if not s: return None
    if len(s) == 10: return datetime.strptime(s, "%Y-%m-%d").date()
    return datetime.fromisoformat(s.replace("Z", "+00:00")).date()


def brier(rows, p_key="p_final"):
    return sum((r[p_key] - r["outcome"]) ** 2 for r in rows) / len(rows) if rows else None


def simulate(records, train_ids, period_days):
    """Returns the list of forecasts in run order. period_days=None = 'never' mode."""
    pred_events = sorted([(parse_dt(r["freeze_datetime"]), r) for r in records], key=lambda x: x[0])
    res_events = sorted([(parse_dt(r["resolution_date"]), r) for r in records if r["id"] in train_ids],
                        key=lambda x: x[0])
    start_day = pred_events[0][0]
    end_day = max(pred_events[-1][0], res_events[-1][0] if res_events else pred_events[-1][0])
    topics = sorted({r["topic"] for r in records})
    beta = {t: {"a": 1.0, "b": 1.0} for t in topics}
    pred_idx = res_idx = 0
    pending = []
    predictions = []
    day = start_day
    one = timedelta(days=1)

    while day <= end_day:
        # morning: forecasts
        while pred_idx < len(pred_events) and pred_events[pred_idx][0] == day:
            r = pred_events[pred_idx][1]
            t = r["topic"]
            a, b = beta[t]["a"], beta[t]["b"]
            n = a + b - 2.0
            base = a / (a + b)
            lam = n / (n + K) if n >= 0 else 0.0
            p_raw = r["p_raw"]
            p_final = (1 - lam) * p_raw + lam * base
            predictions.append({"id": r["id"], "topic": t, "outcome": r["outcome"],
                                "p_raw": p_raw, "base": base, "n": n, "lam": lam,
                                "p_final": p_final, "is_train": r["id"] in train_ids,
                                "day": day.isoformat()})
            pred_idx += 1
        # evening: which resolutions come due today -> into the pending queue
        while res_idx < len(res_events) and res_events[res_idx][0] == day:
            pending.append(res_events[res_idx][1])
            res_idx += 1
        # reveal according to schedule
        if period_days is not None:
            if (day - start_day).days % period_days == 0 and pending:
                for r in pending:
                    t = r["topic"]
                    beta[t]["a"] += r["outcome"]
                    beta[t]["b"] += (1 - r["outcome"])
                pending = []
        day += one
    return predictions, beta


def main():
    d = json.load(open(os.path.join(HERE, "raw_predictions.json")))
    records = d["rows"]
    records.sort(key=lambda r: parse_dt(r["freeze_datetime"]))
    N = len(records)
    n_train = int(N * TRAIN_FRAC)
    train_ids = {r["id"] for r in records[:n_train]}
    hold_ids = {r["id"] for r in records[n_train:]}

    results = {}
    for name, period in SCHEDULES:
        preds, final_beta = simulate(records, train_ids, period)
        hold = [p for p in preds if p["id"] in hold_ids]
        train = [p for p in preds if p["id"] in train_ids]
        results[name] = {
            "period_days": period,
            "n_holdout": len(hold),
            "holdout_brier_final": brier(hold, "p_final"),
            "holdout_brier_raw": brier(hold, "p_raw"),
            "holdout_brier_base_only": sum((p["base"] - p["outcome"]) ** 2 for p in hold) / len(hold),
            "train_predictions": train,
            "holdout_predictions": hold,
        }
        # resolutions actually revealed (by total n across all topics)
        results[name]["total_revealed"] = sum(b["a"] + b["b"] - 2 for b in final_beta.values())

    summary = {
        "K": K, "train_frac": TRAIN_FRAC, "n_total": N, "n_train": n_train, "n_holdout": N - n_train,
        "schedules": [{"name": n, "period_days": p,
                       "holdout_brier_final": results[n]["holdout_brier_final"],
                       "holdout_brier_base_only": results[n]["holdout_brier_base_only"],
                       "holdout_brier_raw": results[n]["holdout_brier_raw"],
                       "total_revealed": results[n]["total_revealed"]}
                      for n, p in SCHEDULES],
    }
    # for the main result we save without prediction lists (large file)
    json.dump(summary, open(os.path.join(HERE, "frequency_sweep.json"), "w"), ensure_ascii=False, indent=2)

    # ---- readable output ----
    print(f"K={K}, train {n_train}/{N}, holdout {N-n_train}\n")
    print(f"{'frequency':<10}{'period':>9}{'revealed':>10}{'Brier_c':>10}{'Brier_b':>10}{'Brier_a':>10}")
    print("-" * 60)
    for s in summary["schedules"]:
        period = "inf" if s["period_days"] is None else f"{s['period_days']}d"
        print(f"{s['name']:<10}{period:>9}{s['total_revealed']:>10.0f}"
              f"{s['holdout_brier_final']:>10.4f}{s['holdout_brier_base_only']:>10.4f}{s['holdout_brier_raw']:>10.4f}")

    # ---- plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # (1) main plot: frequency -> Brier (log-x by period)
    finite = [s for s in summary["schedules"] if s["period_days"] is not None]
    never = [s for s in summary["schedules"] if s["period_days"] is None][0]
    raw_brier = finite[0]["holdout_brier_raw"]
    xs_p = [s["period_days"] for s in finite]
    yc = [s["holdout_brier_final"] for s in finite]
    yb = [s["holdout_brier_base_only"] for s in finite]

    ax1.plot(xs_p, yc, "o-", color="#1a7f37", lw=2.5, ms=10, label="(c) calibration (ours)")
    ax1.plot(xs_p, yb, "s--", color="#0969da", lw=1.5, ms=6, alpha=0.7, label="(b) base_topic only")
    # 'never' as a horizontal band on the right
    ax1.axhline(never["holdout_brier_final"], color="#cf222e", ls="-", lw=2, alpha=0.6,
                label=f"never = {never['holdout_brier_final']:.3f}  (= raw 3B)")
    ax1.axhline(raw_brier, color="#cf222e", ls=":", lw=1, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xticks(xs_p)
    ax1.set_xticklabels([s["name"] for s in finite], rotation=35, ha="right", fontsize=9)
    ax1.set_xlabel("Period between reveals (log-scale)")
    ax1.set_ylabel("Brier on hold-out (lower = better)")
    ax1.set_title("Hold-out Brier vs reveal period (11 schedules, N=330)")
    for x, v in zip(xs_p, yc):
        ax1.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontweight="bold", fontsize=8)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # (2) learning curves - subset for readability
    WINDOW = 100
    show_curves = {"day", "week", "month", "quarter", "year", "never"}
    colors = {"day": "#1a7f37", "week": "#2da44e", "month": "#bf8700",
              "quarter": "#d29922", "year": "#cf6622", "never": "#cf222e"}
    for name, period in SCHEDULES:
        if name not in show_curves: continue
        train = results[name]["train_predictions"]
        if not train: continue
        xs, ys = [], []
        for i in range(WINDOW, len(train) + 1, max(1, WINDOW // 4)):
            window = train[max(0, i - WINDOW):i]
            ys.append(brier(window, "p_final"))
            xs.append(i)
        ax2.plot(xs, ys, "-", label=name, lw=2, color=colors.get(name, "#333"))
    ax2.axhline(yb[0] if yb else 0.19, color="gray", ls=":", lw=1, alpha=0.5,
                label=f"base-only floor ~ {yb[0]:.2f}")
    ax2.set_xlabel("Position in training stream (question #)")
    ax2.set_ylabel(f"Brier in rolling window of {WINDOW}")
    ax2.set_title("Learning curves (subset of frequencies for readability)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "frequency_sweep.png"), dpi=130)
    print("\nSaved frequency_sweep.json and frequency_sweep.png")


if __name__ == "__main__":
    main()
