#!/usr/bin/env python3
"""Step 4a: raw forecasts from Qwen2.5-3B over all 1646 questions (no calibration).
Single batched pass. Writes raw_predictions.json."""
import os
import re
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-3B-Instruct"

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

    llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, enforce_eager=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=200)

    convs = []
    for r in records:
        bg = (r.get("background_text") or "")[:500]
        u = (f"Question: {r['question_text']}\n\n"
             f"Background: {bg}\n\n"
             f"What is the probability (between 0 and 1) that this question resolves YES? "
             f"Reason briefly, then on the LAST line write ONLY the number.")
        convs.append([{"role": "system", "content": SYS},
                      {"role": "user", "content": u}])

    print(f"[raw] {MODEL}, {len(convs)} questions - batched pass", flush=True)
    outs = llm.chat(convs, params)

    rows = []
    parse_fail = 0
    for r, o in zip(records, outs):
        raw = o.outputs[0].text
        p = extract_p(raw)
        if p is None:
            parse_fail += 1
        rows.append({
            "id": r["id"], "topic": r.get("topic"), "source": r.get("source"),
            "question_text": r.get("question_text", "")[:200],
            "freeze_datetime": r.get("freeze_datetime"),
            "resolution_date": r.get("resolution_date"),
            "outcome": r["outcome"],
            "p_raw": p if p is not None else 0.5,
            "p_raw_parsed": p is not None,
        })

    out = {"model": MODEL, "n": len(rows), "parse_failures": parse_fail, "rows": rows}
    json.dump(out, open(os.path.join(HERE, "raw_predictions.json"), "w"), ensure_ascii=False, indent=2)
    print(f"Parsing: ok {len(rows)-parse_fail}/{len(rows)}, failures {parse_fail}")
    print("Saved raw_predictions.json")


if __name__ == "__main__":
    main()
