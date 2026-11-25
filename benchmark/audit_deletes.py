#!/usr/bin/env python3
"""Judge a vid-fp JSON report's DELETE rows against the hand-labeled pairs.

`score.py` scores the pairs a tool *proposes*. This scores what vid-fp would
*do* with them, which is the question a user actually has: for every file the
report marks DELETE, look at the files that survive in the groups it appears in
and ask what the human said about that relationship. A DELETE whose kept
neighbours are all labeled DIFFERENT is a file being removed for nothing; one
labeled CLIP where the kept file is the SHORTER of the two is footage being
lost.

Pass two reports to also diff their DELETE sets:

    python3 audit_deletes.py out/default/vidfp.json out/cliques/vidfp.json

Groups can legitimately end with no survivor at all, because DELETE is resolved
across every group a file appears in -- those rows are reported separately so
the chain behind them can be traced by hand.
"""
import csv, json, sys, collections

LABELS = "/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/labels.csv"


def key(a, b):
    return (a, b) if a <= b else (b, a)


def load_labels():
    out = {}
    with open(LABELS, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[key(row["path_a"], row["path_b"])] = row["relation"]
    return out


def audit(path, labels):
    j = json.load(open(path))
    groups = [[f for f in g["files"]] for g in j["results"]]

    action = {}
    duration = {}
    for g in groups:
        for f in g:
            action[f["full_path"]] = f["action"]
            duration[f["full_path"]] = f["length_seconds"]

    deleted = {p for p, a in action.items() if a == "DELETE"}
    survives = {p for p, a in action.items() if a in ("KEEP", "REVIEW")}

    mates = collections.defaultdict(set)
    for g in groups:
        members = [f["full_path"] for f in g]
        for p in members:
            mates[p].update(m for m in members if m != p)

    emptied = sum(1 for g in groups if all(f["full_path"] in deleted for f in g))

    verdict = collections.Counter()
    detail = collections.defaultdict(list)
    for d in sorted(deleted):
        kept = sorted(mates[d] & survives)
        rels = {m: labels.get(key(d, m)) for m in kept}
        if not kept:
            v = "no survivor in any of its groups"
        elif any(r == "SAME" for r in rels.values()):
            v = "SAME as a kept file"
        elif any(r == "CLIP" and duration[m] >= duration[d] for m, r in rels.items()):
            v = "CLIP inside a longer kept file"
        elif any(r == "CLIP" for r in rels.values()):
            v = "CLIP, but the kept file is the SHORTER one"
        elif all(r == "DIFFERENT" for r in rels.values()):
            v = "kept files all labeled DIFFERENT"
        else:
            v = "unlabeled"
        verdict[v] += 1
        detail[v].append((d, kept))

    print(f"\n=== {path}")
    print(f"groups {len(groups)}  files {len(action)}  DELETE {len(deleted)}  "
          f"groups with no survivor {emptied}")
    for v, n in verdict.most_common():
        print(f"  {n:4}  {v}")
    for v in ("CLIP, but the kept file is the SHORTER one",
              "kept files all labeled DIFFERENT",
              "no survivor in any of its groups"):
        for d, kept in detail[v][:6]:
            print(f"    [{v}] {d.split('/')[-1]}  ->  "
                  + ", ".join(k.split('/')[-1] for k in kept[:3]))
    return deleted


if __name__ == "__main__":
    labels = load_labels()
    sets = [audit(p, labels) for p in sys.argv[1:]]
    if len(sets) == 2:
        old, new = sets
        print(f"\nDELETE only in {sys.argv[1]}: {len(old - new)}")
        for p in sorted(old - new)[:10]:
            print("   -", p.split("/")[-1])
        print(f"DELETE only in {sys.argv[2]}: {len(new - old)}")
        for p in sorted(new - old)[:10]:
            print("   +", p.split("/")[-1])
