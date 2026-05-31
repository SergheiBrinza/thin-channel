#!/usr/bin/env python3
"""Classify all ForecastBench questions into 9 fixed topics using Qwen2.5-7B (vLLM)."""
import os
import json
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "forecastbench_clean.json")
DST = os.path.join(HERE, "questions_with_topics.json")
MODEL = "Qwen/Qwen2.5-7B-Instruct"

TOPICS = [
    {
        "name": "Politics & Elections",
        "description": "Election outcomes, nominations, referendums, legislative or regulatory decisions, leadership transitions, and tenure of heads of state.",
    },
    {
        "name": "Geopolitics & Conflict",
        "description": "Wars, ceasefires, sanctions, diplomatic actions, mobilizations, and country-level rates of battles, riots, protests, fatalities, and civil unrest.",
    },
    {
        "name": "Macro & Financial Indicators",
        "description": "Directional changes in interest rates, treasury yields, inflation, unemployment, FX, commodities, mortgage rates, credit spreads, central-bank balance sheets, and stock indices.",
    },
    {
        "name": "Markets & Crypto",
        "description": "Speculative bets on individual asset prices, cryptocurrency thresholds, token valuations, ETF approvals, IPOs, and company-specific market-cap or valuation milestones.",
    },
    {
        "name": "AI & Tech",
        "description": "AI capability milestones, model releases, AI labs' rankings, AI-driven legal or societal incidents, consumer-tech launches, space launches, and frontier engineering breakthroughs.",
    },
    {
        "name": "Public Health",
        "description": "Disease outbreaks, pandemic declarations, human case counts, vaccine availability, WHO designations, and other biomedical or public-health milestones.",
    },
    {
        "name": "Climate & Weather",
        "description": "Temperature comparisons at weather stations, hurricane and named-storm counts, and other meteorological or climate-record outcomes.",
    },
    {
        "name": "Sports & Games",
        "description": "Outcomes of professional sports seasons, championships, coaching moves, retirements, world records, and competitive games including chess Elo and Eurovision-style contests.",
    },
    {
        "name": "Entertainment & Culture",
        "description": "Box-office grosses, film and game release timing, casting announcements, audience ratings, celebrity legal or personal events, and pop-culture milestones.",
    },
]

TOPIC_NAMES = [t["name"] for t in TOPICS]
TOPIC_NAMES_LOWER = [n.lower() for n in TOPIC_NAMES]

SYS = (
    "You are a strict classifier. Given a forecasting question and short context, "
    "pick EXACTLY ONE topic name from the provided list. "
    "Reply with just the topic name on the first line, copied verbatim from the list. "
    "Do not invent new topics. Do not add commentary."
)


def build_topic_block():
    lines = []
    for i, t in enumerate(TOPICS, 1):
        lines.append(f"{i}. {t['name']} — {t['description']}")
    return "\n".join(lines)


TOPIC_BLOCK = build_topic_block()


def build_user(q, bg):
    q = (q or "").strip()
    bg = (bg or "").strip()[:400]
    return (
        f"Question: {q}\n"
        f"Context: {bg}\n\n"
        f"Topics:\n{TOPIC_BLOCK}\n\n"
        f"Pick the single best topic for this question. Reply with just the topic name."
    )


def parse_topic(text: str) -> str:
    """Return canonical topic name, or 'Other' if not matched."""
    if not text:
        return "Other"
    head = text.strip().splitlines()[0].strip()
    # Strip leading numbering / bullets / quotes
    for prefix in ("- ", "* ", "• "):
        if head.startswith(prefix):
            head = head[len(prefix):].strip()
    # remove leading "1." style
    while head[:2].rstrip(".").isdigit() and "." in head[:4]:
        head = head.split(".", 1)[1].strip()
        break
    head = head.strip("\"'`*[]() ")
    # Some models say "Topic: X"
    if ":" in head and head.lower().startswith(("topic", "answer", "category")):
        head = head.split(":", 1)[1].strip().strip("\"'`*[]() ")
    low = head.lower()
    # Exact match
    if low in TOPIC_NAMES_LOWER:
        return TOPIC_NAMES[TOPIC_NAMES_LOWER.index(low)]
    # Unique prefix match
    prefix_matches = [n for n, nl in zip(TOPIC_NAMES, TOPIC_NAMES_LOWER) if nl.startswith(low) or low.startswith(nl)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    # Containment fallback (unique)
    contain_matches = [n for n, nl in zip(TOPIC_NAMES, TOPIC_NAMES_LOWER) if nl in low or low in nl]
    if len(contain_matches) == 1:
        return contain_matches[0]
    return "Other"


def main():
    data = json.load(open(SRC))
    records = data["records"]
    print(f"[classify] {len(records)} questions, model={MODEL}")

    llm = LLM(
        model=MODEL,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,
        seed=0,
    )
    P = SamplingParams(temperature=0.0, max_tokens=16)

    conversations = []
    for r in records:
        q = r.get("question_text") or r.get("question") or ""
        bg = r.get("background_text") or r.get("background") or ""
        conversations.append(
            [
                {"role": "system", "content": SYS},
                {"role": "user", "content": build_user(q, bg)},
            ]
        )

    print(f"  batch: {len(conversations)} prompts")
    outputs = llm.chat(conversations, P)

    counts = Counter()
    other_examples = []
    for r, o in zip(records, outputs):
        raw = o.outputs[0].text
        topic = parse_topic(raw)
        r["topic"] = topic
        counts[topic] += 1
        if topic == "Other" and len(other_examples) < 10:
            other_examples.append((r.get("id"), raw.strip().splitlines()[0] if raw else ""))

    out = dict(data)
    out["records"] = records
    out["topic_taxonomy"] = TOPICS
    json.dump(out, open(DST, "w"), ensure_ascii=False, indent=2)

    print("\n=== Per-topic counts ===")
    for name in TOPIC_NAMES:
        print(f"  {name}: {counts.get(name, 0)}")
    print(f"  Other: {counts.get('Other', 0)}")
    total_labeled = sum(counts.get(n, 0) for n in TOPIC_NAMES)
    print(f"\nLabeled: {total_labeled}/{len(records)}  |  Other/unparsed: {counts.get('Other', 0)}")
    if other_examples:
        print("Other examples:")
        for qid, txt in other_examples:
            print(f"  id={qid}  raw={txt!r}")
    print(f"\nSaved: {DST}  (records={len(records)})")


if __name__ == "__main__":
    main()
