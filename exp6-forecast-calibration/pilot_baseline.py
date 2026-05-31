#!/usr/bin/env python3
"""Step 3: pilot baseline. 50 random questions through Qwen2.5-3B WITHOUT calibration.
Checks whether the model produces meaningful probabilities (Brier << 0.5) and whether it
is just exploiting the class imbalance (69% NO). Writes pilot_baseline.json plus a short report."""
import os
import re
import json
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_SAMPLE = 50
SEED = 42

SYS = (
    "You are a calibrated probabilistic forecaster. You estimate the probability that a "
    "binary future-event question will resolve YES. Be honest about uncertainty. "
    "Reason briefly, then on the LAST line write ONLY a number between 0 and 1 - "
    "your probability that the answer is YES. Example final line: 0.37"
)


def extract_p(text):
    """Take the last number in [0,1] from the model output. Returns None on failure."""
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

    print(f"[pilot] {MODEL}, {len(convs)} questions", flush=True)
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
        rows.append({"id": r["id"], "source": r.get("source"), "topic": r.get("topic"),
                     "question": r["question_text"][:150],
                     "outcome": r["outcome"], "p_raw": p, "p_used": p_used,
                     "raw_answer": raw[-200:]})

    # summaries
    def brier(rows_):
        return sum((x["p_used"] - x["outcome"]) ** 2 for x in rows_) / len(rows_) if rows_ else None

    yes = [r for r in rows if r["outcome"] == 1]
    no = [r for r in rows if r["outcome"] == 0]
    brier_all = brier(rows)
    brier_yes = brier(yes)
    brier_no = brier(no)

    # reference points
    base_rate = sum(r["outcome"] for r in rows) / len(rows)
    brier_chance = 0.25            # P=0.5 everywhere
    brier_alwaysNO = sum((0 - r["outcome"]) ** 2 for r in rows) / len(rows)
    brier_baserate = sum((base_rate - r["outcome"]) ** 2 for r in rows) / len(rows)

    # distribution of P
    def bucket(p):
        if p < 0.1: return "[0.0,0.1)"
        if p < 0.3: return "[0.1,0.3)"
        if p < 0.5: return "[0.3,0.5)"
        if p == 0.5: return "0.5"
        if p < 0.7: return "(0.5,0.7]"
        if p < 0.9: return "(0.7,0.9]"
        return "(0.9,1.0]"

    from collections import Counter
    dist = Counter(bucket(r["p_used"]) for r in rows)
    mean_p_yes = sum(r["p_used"] for r in yes) / len(yes) if yes else None
    mean_p_no = sum(r["p_used"] for r in no) / len(no) if no else None

    summary = {
        "model": MODEL, "n_sample": N_SAMPLE, "seed": SEED, "parse_failures": parse_fail,
        "n_yes": len(yes), "n_no": len(no), "base_rate_yes": round(base_rate, 3),
        "brier_overall": round(brier_all, 4),
        "brier_on_YES_questions": round(brier_yes, 4) if brier_yes is not None else None,
        "brier_on_NO_questions": round(brier_no, 4) if brier_no is not None else None,
        "mean_predicted_p_on_YES": round(mean_p_yes, 3) if mean_p_yes is not None else None,
        "mean_predicted_p_on_NO": round(mean_p_no, 3) if mean_p_no is not None else None,
        "reference_brier": {
            "chance_P=0.5_everywhere": brier_chance,
            "alwaysNO_P=0": round(brier_alwaysNO, 4),
            "always_base_rate": round(brier_baserate, 4),
        },
        "p_distribution": dict(dist),
        "rows": rows,
    }
    json.dump(summary, open(os.path.join(HERE, "pilot_baseline.json"), "w"),
              ensure_ascii=False, indent=2)

    print()
    print(f"Brier overall:       {summary['brier_overall']}")
    print(f"  on YES questions:  {summary['brier_on_YES_questions']}  (n={len(yes)})")
    print(f"  on NO questions:   {summary['brier_on_NO_questions']}  (n={len(no)})")
    print()
    print(f"Mean P on YES questions: {summary['mean_predicted_p_on_YES']}  (good if closer to 1)")
    print(f"Mean P on NO questions:  {summary['mean_predicted_p_on_NO']}  (good if closer to 0)")
    print()
    print("Reference points:")
    print(f"  Brier P=0.5 everywhere:  {brier_chance}")
    print(f"  Brier always-NO:         {summary['reference_brier']['alwaysNO_P=0']}")
    print(f"  Brier by base rate:      {summary['reference_brier']['always_base_rate']}")
    print()
    print(f"Forecast distribution: {dict(dist)}")
    print(f"Parse failures: {parse_fail}/{N_SAMPLE}")
    print(f"\nSaved pilot_baseline.json")


if __name__ == "__main__":
    main()
