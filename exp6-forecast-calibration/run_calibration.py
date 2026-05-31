#!/usr/bin/env python3
"""Step 4b: per-topic Beta calibration + 3 ablations + per-topic regression check.
Formula: P_final = (1-lam)*P_raw + lam*base_topic, where lam = n/(n+K), n = alpha+beta-2, K=10.
Train = first 80% by freeze_datetime, hold-out = last 20%.
Only training resolutions update Beta. Writes calibration_results.json plus a readable table."""
import os
import json
from datetime import datetime
from statistics import pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10  # saturation parameter for lam(n) = n/(n+K)
TRAIN_FRAC = 0.80


def parse_dt(s):
    if not s:
        return datetime.max
    if len(s) == 10:  # YYYY-MM-DD
        return datetime.strptime(s, "%Y-%m-%d")
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.replace(tzinfo=None)  # drop TZ for compatibility


def brier(rows, p_key):
    if not rows: return None
    return sum((r[p_key] - r["outcome"]) ** 2 for r in rows) / len(rows)


def main():
    d = json.load(open(os.path.join(HERE, "raw_predictions.json")))
    records = d["rows"]

    # sort by freeze
    records.sort(key=lambda r: parse_dt(r["freeze_datetime"]))
    N = len(records)
    n_train = int(N * TRAIN_FRAC)
    train_ids = {r["id"] for r in records[:n_train]}
    hold_ids = {r["id"] for r in records[n_train:]}

    # unique topics
    topics = sorted({r["topic"] for r in records})
    beta = {t: {"alpha": 1.0, "beta": 1.0} for t in topics}  # uniform prior

    # events: (ts, kind, record). Stable order on equal ts: predict first, then resolve.
    events = []
    for r in records:
        events.append((parse_dt(r["freeze_datetime"]), 0, "predict", r))
        events.append((parse_dt(r["resolution_date"]), 1, "resolve", r))
    events.sort(key=lambda e: (e[0], e[1]))

    # online predictions for all questions, Beta updates only on training resolutions
    online = {}  # id -> {p_final_online, base_at_predict, n_at_predict, lam_at_predict}
    for ts, _, kind, r in events:
        t = r["topic"]
        if kind == "predict":
            a, b = beta[t]["alpha"], beta[t]["beta"]
            n = a + b - 2.0
            base = a / (a + b)
            lam = n / (n + K)
            p_raw = r["p_raw"]
            p_final = (1 - lam) * p_raw + lam * base
            online[r["id"]] = {"base": base, "n": n, "lam": lam,
                                "p_raw": p_raw, "p_final": p_final}
        else:  # resolve
            if r["id"] in train_ids:
                beta[t]["alpha"] += r["outcome"]
                beta[t]["beta"] += (1 - r["outcome"])

    # final snapshot Beta (after all training resolutions)
    final = {t: {"alpha": beta[t]["alpha"], "beta": beta[t]["beta"],
                 "base": beta[t]["alpha"] / (beta[t]["alpha"] + beta[t]["beta"]),
                 "n": beta[t]["alpha"] + beta[t]["beta"] - 2.0,
                 "lam": (beta[t]["alpha"] + beta[t]["beta"] - 2.0) / (beta[t]["alpha"] + beta[t]["beta"] - 2.0 + K)}
             for t in topics}

    # hold-out ablations: use final Beta for (b) and (c)
    hold_rows = []
    for r in records:
        if r["id"] not in hold_ids:
            continue
        t = r["topic"]
        f = final[t]
        p_a = r["p_raw"]                            # (a) raw model only
        p_b = f["base"]                             # (b) topic base rate only
        p_c = (1 - f["lam"]) * p_a + f["lam"] * p_b  # (c) mix
        hold_rows.append({"id": r["id"], "topic": t, "outcome": r["outcome"],
                           "p_raw": p_a, "p_base_only": p_b, "p_mix": p_c})

    # Per-topic Brier on hold-out for each ablation
    per_topic = {}
    for t in topics:
        ts_rows = [r for r in hold_rows if r["topic"] == t]
        if not ts_rows:
            continue
        per_topic[t] = {
            "n_holdout": len(ts_rows),
            "true_YES_rate": round(sum(r["outcome"] for r in ts_rows) / len(ts_rows), 3),
            "training_n_resolutions": round(final[t]["n"], 1),
            "training_base": round(final[t]["base"], 3),
            "lambda_at_end": round(final[t]["lam"], 3),
            "brier_a_raw": round(brier(ts_rows, "p_raw"), 4),
            "brier_b_base_only": round(brier(ts_rows, "p_base_only"), 4),
            "brier_c_mix": round(brier(ts_rows, "p_mix"), 4),
        }
        per_topic[t]["delta_c_vs_a"] = round(per_topic[t]["brier_c_mix"] - per_topic[t]["brier_a_raw"], 4)
        per_topic[t]["delta_c_vs_b"] = round(per_topic[t]["brier_c_mix"] - per_topic[t]["brier_b_base_only"], 4)

    # global
    g = {
        "n_holdout": len(hold_rows),
        "true_YES_rate": round(sum(r["outcome"] for r in hold_rows) / len(hold_rows), 3),
        "brier_a_raw": round(brier(hold_rows, "p_raw"), 4),
        "brier_b_base_only": round(brier(hold_rows, "p_base_only"), 4),
        "brier_c_mix": round(brier(hold_rows, "p_mix"), 4),
    }
    g["delta_c_vs_a"] = round(g["brier_c_mix"] - g["brier_a_raw"], 4)
    g["delta_c_vs_b"] = round(g["brier_c_mix"] - g["brier_b_base_only"], 4)

    summary = {
        "model": d["model"], "K_saturation": K, "train_fraction": TRAIN_FRAC,
        "n_total": N, "n_train": n_train, "n_holdout": N - n_train,
        "final_beta_per_topic": final,
        "holdout_global": g,
        "holdout_per_topic": per_topic,
        "hold_rows": hold_rows,
    }
    json.dump(summary, open(os.path.join(HERE, "calibration_results.json"), "w"),
              ensure_ascii=False, indent=2)

    # ---- readable output ----
    print(f"\n=== Calibration: K={K}, train {n_train}/{N} (~{int(100*TRAIN_FRAC)}%) ===\n")
    print(f"GLOBAL on hold-out (N={g['n_holdout']}, true_YES={g['true_YES_rate']}):")
    print(f"  (a) raw 3B:                Brier = {g['brier_a_raw']:.4f}")
    print(f"  (b) base_topic only:       Brier = {g['brier_b_base_only']:.4f}")
    print(f"  (c) mix (our calibration): Brier = {g['brier_c_mix']:.4f}")
    print(f"  c vs a: {g['delta_c_vs_a']:+.4f}  (negative = calibration helps)")
    print(f"  c vs b: {g['delta_c_vs_b']:+.4f}  (negative = model adds something on top of base rate)")

    print(f"\n{'topic':<32}{'N_h':>4}{'tYES':>6}{'train_n':>8}{'base':>6}{'lam':>5}"
          f"{'Brier_a':>9}{'Brier_b':>9}{'Brier_c':>9}{'D c-a':>8}{'D c-b':>8}  status")
    print("-" * 130)

    sorted_topics = sorted(per_topic.items(), key=lambda x: -x[1]["brier_a_raw"])
    for t, m in sorted_topics:
        status = ""
        if m["delta_c_vs_a"] > 0.005:
            status = "REGRESS"
        elif m["delta_c_vs_a"] < -0.005:
            status = "IMPROVE"
        else:
            status = "~ flat"
        flag = "*" if m["n_holdout"] < 8 else " "
        print(f"{t:<32}{m['n_holdout']:>3}{flag}{m['true_YES_rate']:>6.2f}"
              f"{m['training_n_resolutions']:>8.0f}{m['training_base']:>6.2f}{m['lambda_at_end']:>5.2f}"
              f"{m['brier_a_raw']:>9.4f}{m['brier_b_base_only']:>9.4f}{m['brier_c_mix']:>9.4f}"
              f"{m['delta_c_vs_a']:>+8.4f}{m['delta_c_vs_b']:>+8.4f}  {status}")
    print("* = N_hold-out < 8, noisy")
    print(f"\nSaved calibration_results.json")


if __name__ == "__main__":
    main()
