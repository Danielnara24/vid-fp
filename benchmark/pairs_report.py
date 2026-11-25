#!/usr/bin/env python3
"""What one vid-fp run got right and wrong, pair by pair.

Reads a run's JSON and the labels, and prints the false positives (with the
coverage and span figures the run itself reported) and the positives it missed.
The point is to look at what separates them: if the wrong pairs differ from the
right ones in some measurable way the tool is not currently reading, that is a
filter worth writing.
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def link_rows(path):
    """(pair) -> list of the JSON's own per-direction figures."""
    with open(path) as fh:
        j = json.load(fh)
    rows = collections.defaultdict(list)
    for group in j.get("results", []):
        for f in group.get("files", []):
            for m in f.get("matches", []):
                key = S.pair_key(f["full_path"], m["full_path"])
                rows[key].append({
                    "subject": f["full_path"],
                    "duration": f.get("length_seconds"),
                    "samples": f.get("samples"),
                    "matched_seconds": m.get("matched_seconds"),
                    "from": m.get("matched_from_seconds"),
                    "to": m.get("matched_to_seconds"),
                })
    return rows


def fmt(rows):
    out = []
    for r in rows:
        out.append(f"{os.path.basename(r['subject'])[:38]:<38} "
                   f"len={r['duration']:>7} n={str(r['samples']):>4} "
                   f"matched={r['matched_seconds']:>7} "
                   f"win=[{r['from']}..{r['to']}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--labels", default=os.path.join(HERE, "labels.csv"))
    ap.add_argument("--mode", default="exact", choices=["exact", "any"])
    ap.add_argument("--show", default="fp,fn,unjudged")
    ap.add_argument("-n", type=int, default=40)
    args = ap.parse_args()

    labels = S.load_labels(args.labels)
    pairs = S.vidfp_link_pairs(args.run)
    rows = link_rows(args.run)
    s = S.Score("run", pairs, labels, args.mode)

    want = args.show.split(",")
    for name in want:
        got = getattr(s, name if name != "unjudged" else "unjudged")
        print(f"\n=== {name}: {len(got)} ===")
        for key in sorted(got)[:args.n]:
            rel = labels.get(key, {}).get("relation", "?")
            note = (labels.get(key, {}).get("row", {}).get("notes") or "").strip()
            print(f"[{rel}] {os.path.basename(key[0])} <-> {os.path.basename(key[1])}"
                  + (f"  [{note}]" if note else ""))
            for line in fmt(rows.get(key, [])):
                print("    " + line)


if __name__ == "__main__":
    sys.exit(main())
