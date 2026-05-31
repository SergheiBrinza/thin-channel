#!/usr/bin/env python3
"""Step 1: normalize resolved binary ForecastBench questions and substitute date
placeholders into question wording.
We iterate over every (question_set, resolution_set) pair, join by id, keep only atomic
(non-combo) questions where resolution is True and resolved_to is in {0.0, 1.0}.
For each kept record we also produce two model-facing fields:
  - question_text:   the question with {forecast_due_date} and {resolution_date}
                     placeholders substituted by concrete YYYY-MM-DD strings
                     (838 and 1063 occurrences in the dataset, respectively)
  - background_text: a verbatim copy of background (no placeholders occur there)
Writes forecastbench_clean.json plus a short summary."""
import os
import json
import glob
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "forecastbench_raw", "datasets")


def fill_placeholders(text, freeze_datetime, resolution_date):
    """Substitute {forecast_due_date} and {resolution_date} with YYYY-MM-DD strings.
    freeze_datetime is an ISO-8601 string (e.g. '2024-07-12T00:00:00+00:00');
    we take its date prefix. resolution_date is already YYYY-MM-DD."""
    if not text:
        return text
    due = (freeze_datetime or "")[:10]
    res = resolution_date or ""
    return text.replace("{forecast_due_date}", due).replace("{resolution_date}", res)


def main():
    qfiles = sorted(glob.glob(os.path.join(RAW, "question_sets", "*-llm.json")))
    rfiles = {os.path.basename(p): p for p in glob.glob(os.path.join(RAW, "resolution_sets", "*_resolution_set.json"))}

    by_id = {}  # id -> normalized record (dedup by id)
    skipped = Counter()
    pairs_used = 0

    for qf in qfiles:
        m = re.match(r"(\d{4}-\d{2}-\d{2})-llm\.json", os.path.basename(qf))
        if not m:
            continue
        date = m.group(1)
        rname = f"{date}_resolution_set.json"
        if rname not in rfiles:
            skipped["no_matching_resolution_set"] += 1
            continue
        pairs_used += 1

        qdata = json.load(open(qf))
        rdata = json.load(open(rfiles[rname]))
        res_by_id = {r["id"]: r for r in rdata["resolutions"] if isinstance(r.get("id"), str)}

        for q in qdata["questions"]:
            qid = q["id"]
            # skip combo questions (combination_of is not 'N/A')
            if q.get("combination_of") not in (None, "N/A", "n/a"):
                skipped["combo"] += 1
                continue
            r = res_by_id.get(qid)
            if r is None:
                skipped["no_resolution_for_question"] += 1
                continue
            if not r.get("resolved"):
                skipped["not_resolved"] += 1
                continue
            outcome = r.get("resolved_to")
            if outcome not in (0.0, 1.0, 0, 1):
                skipped["non_binary_outcome"] += 1
                continue

            if qid in by_id:
                # already seen in an earlier round - skip duplicate
                skipped["duplicate"] += 1
                continue

            freeze_dt = q.get("freeze_datetime")
            res_date = r.get("resolution_date")
            raw_question = q.get("question") or ""
            raw_background = q.get("background") or ""
            by_id[qid] = {
                "id": qid,
                "source": q.get("source"),
                "question_text": fill_placeholders(raw_question, freeze_dt, res_date),
                "background_text": fill_placeholders(raw_background, freeze_dt, res_date),
                "resolution_criteria": q.get("resolution_criteria", ""),
                "url": q.get("url", ""),
                "freeze_datetime": freeze_dt,  # when the forecast was frozen
                "market_info_open_datetime": q.get("market_info_open_datetime"),
                "market_info_close_datetime": q.get("market_info_close_datetime"),
                "resolution_date": res_date,
                "outcome": int(outcome),
            }

    records = list(by_id.values())
    records.sort(key=lambda x: (x["freeze_datetime"] or "", x["resolution_date"] or ""))
    out = {
        "dataset": "forecastbench_resolved_binary",
        "n": len(records),
        "question_set_pairs_used": pairs_used,
        "skipped_counts": dict(skipped),
        "by_source": dict(Counter(r["source"] for r in records)),
        "outcome_balance": {"YES (1)": sum(r["outcome"] for r in records),
                            "NO (0)": sum(1 for r in records if r["outcome"] == 0)},
        "freeze_date_range": [records[0]["freeze_datetime"] if records else None,
                              records[-1]["freeze_datetime"] if records else None],
        "resolution_date_range": [min((r["resolution_date"] for r in records if r["resolution_date"]), default=None),
                                  max((r["resolution_date"] for r in records if r["resolution_date"]), default=None)],
        "records": records,
    }
    json.dump(out, open(os.path.join(HERE, "forecastbench_clean.json"), "w"),
              ensure_ascii=False, indent=2)

    # console summary
    print(f"question_set <-> resolution_set pairs processed: {pairs_used}")
    print(f"Resolved atomic binary questions: {len(records)}")
    print(f"  by source: {out['by_source']}")
    print(f"  outcome balance: {out['outcome_balance']}")
    print(f"  freeze range: {out['freeze_date_range']}")
    print(f"  resolution range: {out['resolution_date_range']}")
    print(f"Skipped reasons: {dict(skipped)}")
    print()
    if records:
        ex = records[0]
        print("=== example record (first by freeze_datetime) ===")
        for k in ("id", "source", "question_text", "freeze_datetime",
                  "resolution_date", "outcome"):
            v = ex.get(k)
            v = (v[:120] + "...") if isinstance(v, str) and len(v) > 120 else v
            print(f"  {k}: {v}")
        print(f"  background_text (first 150 chars): {ex.get('background_text','')[:150]}")
    print(f"\nSaved forecastbench_clean.json ({len(records)} records)")


if __name__ == "__main__":
    main()
