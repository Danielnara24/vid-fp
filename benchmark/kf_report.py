#!/usr/bin/env python3
"""Read the kf_sweep log and report accuracy, cost and their ratio.

Cost per cell is the MINIMUM wall time over the rounds (see kf_sweep.py for
why a mean is the wrong statistic on a thermally drifting machine). Accuracy is
read from one round's JSON, and every other round's is checked against it --
the pair set is deterministic given a cell, so a mismatch means something else
moved and the run should not be trusted.

The ratio is reported two ways:

  F1/s      the literal score-per-second, averaged over the intervals. The
            intervals have different absolute costs, so this average is
            dominated by the cheapest one.
  eff       each cell expressed against its own interval's --min-keyframes 4
            cell -- (F1/F1_4) / (wall/wall_4) -- then averaged. This is the
            comparison the default actually has to win, and it weights the
            four intervals equally.
"""

import argparse
import collections
import json
import os
import statistics
import sys

import score as S

HERE = os.path.dirname(os.path.abspath(__file__))


def load(log):
    runs = collections.defaultdict(list)
    with open(log) as fh:
        for line in fh:
            r = json.loads(line)
            runs[(r["interval"], r["min_kf"])].append(r)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(HERE, "out", "kfsweep",
                                                  "runs.jsonl"))
    ap.add_argument("--labels", default=os.path.join(HERE, "labels.csv"))
    ap.add_argument("--baseline-min-kf", type=float, default=4.0)
    ap.add_argument("--csv", default=os.path.join(HERE, "out", "kfsweep",
                                                  "cells.csv"))
    ap.add_argument("--only-under", default=None,
                    help="score only pairs with BOTH files under this prefix. "
                         "A scan of a subfolder cannot propose a pair reaching "
                         "outside it, so scoring one against the whole label "
                         "set would charge it for pairs it was never shown.")
    args = ap.parse_args()

    labels = S.load_labels(args.labels)
    if args.only_under:
        pre = args.only_under
        labels = {k: v for k, v in labels.items()
                  if k[0].startswith(pre) and k[1].startswith(pre)}
        print(f"scoring against {len(labels)} pairs with both files under "
              f"{pre}")
    runs = load(args.log)

    cells = {}
    for (interval, min_kf), rs in sorted(runs.items()):
        pairsets = {r["json"]: S.vidfp_link_pairs(r["json"])
                    for r in rs if os.path.exists(r["json"])}
        sizes = {frozenset(p) for p in pairsets.values()}
        if len(sizes) > 1:
            print(f"WARNING: interval {interval} min_kf {min_kf}: rounds "
                  f"disagree on the pair set "
                  f"({[len(p) for p in pairsets.values()]})", file=sys.stderr)
        pairs = next(iter(pairsets.values()))
        exact = S.Score("x", pairs, labels, "exact")
        anym = S.Score("x", pairs, labels, "any")
        cells[(interval, min_kf)] = {
            "interval": interval, "min_kf": min_kf, "runs": len(rs),
            "wall": min(r["wall"] for r in rs),
            "wall_med": statistics.median(r["wall"] for r in rs),
            "user": min(r["user"] for r in rs),
            "rss_mb": max(r["rss_kb"] for r in rs) / 1024,
            "pairs": len(pairs),
            "precision": exact.precision, "recall": exact.recall,
            "f1": exact.f1, "f1_any": anym.f1, "recall_any": anym.recall,
        }

    intervals = sorted({k[0] for k in cells})
    min_kfs = sorted({k[1] for k in cells})

    for interval in intervals:
        print(f"\n=== --keyframe-interval {interval:g} "
              f"(-d 4 -p 20, cold cache) ===")
        hdr = (f"{'min-kf':>7}{'pairs':>7}{'prec':>8}{'recall':>8}{'F1':>8}"
               f"{'F1(any)':>9}{'wall':>8}{'med':>8}{'user':>8}"
               f"{'F1/s':>9}{'eff':>7}")
        print(hdr)
        print("-" * len(hdr))
        base = cells.get((interval, args.baseline_min_kf))
        for m in min_kfs:
            c = cells.get((interval, m))
            if not c:
                continue
            c["f1_per_s"] = c["f1"] / c["wall"]
            c["eff"] = ((c["f1"] / base["f1"]) / (c["wall"] / base["wall"])
                        if base else float("nan"))
            print(f"{m:>7g}{c['pairs']:>7}{100 * c['precision']:>7.1f}%"
                  f"{100 * c['recall']:>7.1f}%{c['f1']:>8.3f}{c['f1_any']:>9.3f}"
                  f"{c['wall']:>8.1f}{c['wall_med']:>8.1f}{c['user']:>8.1f}"
                  f"{1000 * c['f1_per_s']:>9.2f}{c['eff']:>7.3f}")
        print("  F1/s is x1000. eff is relative to this row's "
              f"--min-keyframes {args.baseline_min_kf:g}.")

    print(f"\n=== averaged over --keyframe-interval "
          f"{', '.join(f'{i:g}' for i in intervals)} ===")
    hdr = (f"{'min-kf':>7}{'F1':>8}{'F1(any)':>9}{'prec':>8}{'recall':>8}"
           f"{'wall':>8}{'F1/s':>9}{'eff':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for m in min_kfs:
        cs = [cells[(i, m)] for i in intervals if (i, m) in cells]
        if len(cs) != len(intervals):
            continue
        row = {
            "min_kf": m,
            "f1": statistics.mean(c["f1"] for c in cs),
            "f1_any": statistics.mean(c["f1_any"] for c in cs),
            "precision": statistics.mean(c["precision"] for c in cs),
            "recall": statistics.mean(c["recall"] for c in cs),
            "wall": statistics.mean(c["wall"] for c in cs),
            "f1_per_s": statistics.mean(c["f1_per_s"] for c in cs),
            "eff": statistics.mean(c["eff"] for c in cs),
        }
        rows.append(row)
        print(f"{m:>7g}{row['f1']:>8.3f}{row['f1_any']:>9.3f}"
              f"{100 * row['precision']:>7.1f}%{100 * row['recall']:>7.1f}%"
              f"{row['wall']:>8.1f}{1000 * row['f1_per_s']:>9.2f}"
              f"{row['eff']:>8.3f}")

    if rows:
        best_ratio = max(rows, key=lambda r: r["eff"])
        best_f1 = max(rows, key=lambda r: r["f1"])
        print(f"\nbest score/speed ratio: --min-keyframes "
              f"{best_ratio['min_kf']:g} (eff {best_ratio['eff']:.3f}, "
              f"F1 {best_ratio['f1']:.3f}, {best_ratio['wall']:.1f}s)")
        print(f"best F1:                --min-keyframes "
              f"{best_f1['min_kf']:g} (F1 {best_f1['f1']:.3f}, "
              f"{best_f1['wall']:.1f}s)")

    import csv as _csv
    with open(args.csv, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(next(iter(cells.values()))))
        w.writeheader()
        for k in sorted(cells):
            w.writerow(cells[k])
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
