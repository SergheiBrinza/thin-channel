#!/usr/bin/env python3
"""Step 3 (wide): 250 random questions through Qwen2.5-3B without calibration.
Breakdown by the 9 topics: where the model is most wrong, where it is YES-biased,
where it discriminates. Writes wide_baseline.json plus a readable table."""
import os
import re
import json
import random
import statistics

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_SAMPLE = 250
SEED = 100

SYS = (
    "You are a calibrated probabilistic forecaster. You estimate the probability that a "
    "binary future-event question will resolve YES. Be honest about uncertainty. "
    "Reason briefly, then on the LAST line write ONLY a number between 0 and 1 - "
    "your probability that the answer is YES. Example final line: 0.37"
)


def extract_p(text):
    nums = re.findall(r"(?<![0-9.])([01](?:\.[0-9]+)?|0?\.[0-9]+|1\.0+)", text)
    if not nums:
        return None
    last = float(nums[-1])
    if 0.0 <= last <= 1.0:
        return last
    return None


def main():
    d = json.load(open(os.path.join(HERE, "questions_with_topics.json")))
    records = d["records"]
    rng = random.Random(SEED)
    sample = rng.sample(records, N_SAMPLE)

    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=200)

    convs = []
    for r in sample:
        bg = (r.get("background_text") or "")[:500]
        u = (f"Question: {r['question_text']}\n\n"
             f"Background: {bg}\n\n"
             f"What is the probability (between 0 and 1) that this question resolves YES? "
             f"Reason briefly, then on the LAST line write ONLY the number.")
        convs.append([{"role": "system", "content": SYS},
                      {"role": "user", "content": u}])

    print(f"[wide] {MODEL}, {len(convs)} questions", flush=True)
    outs = llm.chat(convs, params)

    rows = []
    parse_fail = 0
    for r, o in zip(sample, outs):
        raw = o.outputs[0].text
        p = extract_p(raw)
        if p is None:
            parse_fail += 1
            p_used = 0.5
        else:
            p_used = p
        rows.append({"id": r["id"], "topic": r.get("topic"),
                     "outcome": r["outcome"], "p": p_used})

    # global metrics
    def brier(rs): return sum((x["p"] - x["outcome"]) ** 2 for x in rs) / len(rs) if rs else None
    base_rate = sum(r["outcome"] for r in rows) / len(rows)
    overall_brier = brier(rows)
    overall_mean_p = sum(r["p"] for r in rows) / len(rows)
    brier_alwaysNO = sum((0 - r["outcome"]) ** 2 for r in rows) / len(rows)
    brier_baserate = sum((base_rate - r["outcome"]) ** 2 for r in rows) / len(rows)

    # per topic
    topics = sorted({r["topic"] for r in rows})
    per_topic = {}
    for t in topics:
        ts = [r for r in rows if r["topic"] == t]
        yes = [r for r in ts if r["outcome"] == 1]
        no = [r for r in ts if r["outcome"] == 0]
        per_topic[t] = {
            "n": len(ts), "n_YES": len(yes), "true_YES_rate": round(sum(x["outcome"] for x in ts) / len(ts), 3),
            "mean_P": round(sum(x["p"] for x in ts) / len(ts), 3),
            "brier": round(brier(ts), 4),
            "bias_meanP_minus_trueYES": round((sum(x["p"] for x in ts) / len(ts)) - sum(x["outcome"] for x in ts) / len(ts), 3),
            "mean_P_on_YES": round(sum(x["p"] for x in yes) / len(yes), 3) if yes else None,
            "mean_P_on_NO": round(sum(x["p"] for x in no) / len(no), 3) if no else None,
            "discrimination": (round(sum(x["p"] for x in yes) / len(yes) - sum(x["p"] for x in no) / len(no), 3)
                                if yes and no else None),
            "p_stdev": round(statistics.pstdev([x["p"] for x in ts]), 3),
        }

    summary = {
        "model": MODEL, "n_sample": N_SAMPLE, "seed": SEED, "parse_failures": parse_fail,
        "overall": {
            "brier": round(overall_brier, 4),
            "mean_P": round(overall_mean_p, 3),
            "true_YES_rate": round(base_rate, 3),
            "reference_brier_alwaysNO": round(brier_alwaysNO, 4),
            "reference_brier_baserate": round(brier_baserate, 4),
        },
        "per_topic": per_topic,
        "rows": rows,
    }
    json.dump(summary, open(os.path.join(HERE, "wide_baseline.json"), "w"), ensure_ascii=False, indent=2)

    # readable output
    print(f"\nGlobal: Brier={overall_brier:.3f}  mean_P={overall_mean_p:.3f}  true_YES={base_rate:.3f}")
    print(f"  reference: always-NO={brier_alwaysNO:.3f}, base-rate={brier_baserate:.3f}")
    print(f"  parse failures: {parse_fail}/{N_SAMPLE}")
    print()
    # sort by Brier desc - worst topics on top
    sorted_topics = sorted(per_topic.items(), key=lambda x: -x[1]["brier"])
    print(f"{'topic':<32} {'N':>3} {'true_YES':>9} {'mean_P':>7} {'Brier':>6} {'YES-bias':>9} {'P_YES':>6} {'P_NO':>6} {'discrim':>8} {'stdev':>6}")
    print("-" * 110)
    for t, m in sorted_topics:
        flag = "*" if m["n"] < 15 else " "
        print(f"{t:<32} {m['n']:>3}{flag}{m['true_YES_rate']:>9}  {m['mean_P']:>6}  {m['brier']:>6}  "
              f"{m['bias_meanP_minus_trueYES']:>+8}  "
              f"{(m['mean_P_on_YES'] if m['mean_P_on_YES'] is not None else '-'):>6}  "
              f"{(m['mean_P_on_NO'] if m['mean_P_on_NO'] is not None else '-'):>6}  "
              f"{(m['discrimination'] if m['discrimination'] is not None else '-'):>+8}  {m['p_stdev']:>6}")
    print("* = N<15, metric is noisy")
    print(f"\nSaved wide_baseline.json")


if __name__ == "__main__":
    main()
