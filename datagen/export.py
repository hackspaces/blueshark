"""Export a clean SFT training set from a human-reviewed datagen file.

Strips the review/verification internals (entry_point, test_program) and keeps
the chat messages + provenance, for rows the reviewer approved. Prints a
provenance breakdown so the dataset's composition (teacher vs human-edited) is
auditable — the answer to "is this all Claude output?".

    python -m datagen.export train.jsonl --out train.clean.jsonl
    python -m datagen.export train.jsonl --keep-pending     # also keep un-reviewed (teacher-verified) rows
"""
import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-pending", action="store_true",
                    help="also include rows still pending review (they are teacher-verified)")
    args = ap.parse_args()
    out = args.out or args.infile.replace(".jsonl", "") + ".clean.jsonl"

    rows = [json.loads(l) for l in open(args.infile) if l.strip()]
    keep_status = {"approved"} | ({"pending_review"} if args.keep_pending else set())

    src, dom, kept = Counter(), Counter(), 0
    with open(out, "w") as f:
        for r in rows:
            if r.get("status", "pending_review") == "rejected":
                continue
            if r.get("status", "pending_review") not in keep_status:
                continue
            f.write(json.dumps({"domain": r["domain"], "source": r.get("source", "teacher"),
                                "messages": r["messages"]}) + "\n")
            kept += 1
            src[r.get("source", "teacher")] += 1
            dom[r["domain"]] += 1

    print(f"\n  exported {kept} rows -> {out}")
    print("  provenance:")
    for s, n in src.items():
        print(f"    {s:14s} {n:4d}  ({100*n/max(kept,1):.0f}%)")
    print("  by domain: " + ", ".join(f"{d}={n}" for d, n in dom.items()))
    human = src.get("human-edited", 0)
    print(f"  human-touched: {100*human/max(kept,1):.0f}%  "
          f"(raw-teacher: {100*(kept-human)/max(kept,1):.0f}%)\n")


if __name__ == "__main__":
    main()
