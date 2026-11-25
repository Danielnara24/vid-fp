#!/usr/bin/env python3
"""Merge several `bench_tools.py` batches into one cost record.

Wall time on this machine is noisy for the tools that saturate all eight cores
(vid-fp swung 20% between two rounds while its peak RSS moved 1.5 MB), so the
fast tools were given extra rounds in a second batch. `bench_tools.py` rewrites
`runs.json` per batch rather than appending, so the batches are joined here
instead, with rounds renumbered in the order the batches ran.

The estimator stays the MEDIAN, deliberately. Extra samples of one tool improve
that tool's estimate without flattering it; taking the minimum would turn "more
rounds" into "a better result" and would make the tool that got extra rounds
look faster for no reason other than having been measured more.
"""

import argparse
import collections
import json
import os
import statistics
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batches", nargs="+", help="runs.json files, in run order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    merged, offset = [], collections.Counter()
    for path in args.batches:
        with open(path) as fh:
            runs = json.load(fh)
        highest = collections.Counter()
        for r in runs:
            tool = r["tool"]
            r = dict(r, round=r["round"] + offset[tool], batch=os.path.dirname(path))
            merged.append(r)
            highest[tool] = max(highest[tool], r["round"])
        offset.update({t: n for t, n in highest.items()})

    with open(args.out, "w") as fh:
        json.dump(merged, fh, indent=2)

    by_tool = collections.OrderedDict()
    for r in merged:
        by_tool.setdefault(r["tool"], []).append(r)

    print(f"{'tool':<10}{'n':>3}{'median':>9}{'min':>9}{'max':>9}{'spread':>8}"
          f"{'medRSS':>10}{'rssSpread':>11}")
    print("-" * 69)
    for tool, rs in by_tool.items():
        w = sorted(r["wall_s"] for r in rs)
        m = sorted(r["peak_rss_mb"] for r in rs)
        print(f"{tool:<10}{len(rs):>3}{statistics.median(w):>8.1f}s"
              f"{w[0]:>8.1f}s{w[-1]:>8.1f}s"
              f"{100 * (w[-1] - w[0]) / w[0]:>7.1f}%"
              f"{statistics.median(m):>9.1f}MB"
              f"{100 * (m[-1] - m[0]) / m[0]:>10.1f}%")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
