#!/usr/bin/env python3
"""Sweep --min-keyframes against --keyframe-interval: cost and accuracy.

One run per (interval, min_kf) cell, cold cache every time (--clear-cache), at
the default -d 4 -p 20. Accuracy is deterministic given the cell, so it is
scored once; wall time is not, so every cell is run in several rounds and the
per-cell MINIMUM is taken -- the machine's thermal drift only ever makes a run
slower, and the first run of a batch is systematically the fastest, so a mean
over a drifting batch measures the order the cells were visited in.

Rounds visit the cells in a different order each time (rotated, and reversed on
odd rounds) so no cell keeps the favourable slot.

Results are appended to out/kfsweep/runs.jsonl as they happen, so an
interrupted sweep is resumable and partial results are readable.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

TIME_FMT = "VIDFP_TIME %e %U %S %M"


def run_cell(binary, corpus, interval, min_kf, dest, extra=()):
    cmd = ["/usr/bin/time", "-f", TIME_FMT, binary, corpus, "-r",
           "-d", "4", "-p", "20",
           "--keyframe-interval", str(interval),
           "--min-keyframes", str(min_kf),
           "-o", dest, "--yes", "--clear-cache", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in (0, 2):
        raise SystemExit(f"vid-fp failed rc={r.returncode} on "
                         f"interval={interval} min_kf={min_kf}\n{r.stderr[-2000:]}")
    line = next(l for l in r.stderr.splitlines() if l.startswith("VIDFP_TIME"))
    _, wall, user, sysd, rss = line.split()
    return {"wall": float(wall), "user": float(user), "sys": float(sysd),
            "rss_kb": int(rss)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/daniel/Documents/AN")
    ap.add_argument("--bin", default=os.path.join(HERE, os.pardir,
                                                  "target/release/vid-fp"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "kfsweep"))
    ap.add_argument("--intervals", type=float, nargs="+",
                    default=[5, 13, 21, 29])
    ap.add_argument("--min-kf", type=float, nargs="+",
                    default=[1, 2, 3, 4, 6, 8, 12])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=15.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.out, "runs.jsonl")

    cells = [(i, m) for i in args.intervals for m in args.min_kf]
    total = len(cells) * args.rounds
    n = 0
    t0 = time.time()

    for rnd in range(args.rounds):
        # Rotate so a different cell leads each round, and reverse on odd
        # rounds, so the first-run-is-fastest effect does not land on one cell.
        order = cells[rnd * 3 % len(cells):] + cells[:rnd * 3 % len(cells)]
        if rnd % 2:
            order = list(reversed(order))
        for interval, min_kf in order:
            n += 1
            tag = f"i{interval:g}_m{min_kf:g}"
            dest = os.path.join(args.out, f"{tag}_r{rnd}.json")
            print(f"[{n}/{total}] round {rnd} {tag} ...",
                  flush=True, end=" ")
            rec = run_cell(args.bin, args.corpus, interval, min_kf, dest)
            rec.update({"interval": interval, "min_kf": min_kf,
                        "round": rnd, "json": dest, "tag": tag})
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"wall {rec['wall']:.1f}s user {rec['user']:.1f}s "
                  f"({(time.time() - t0) / 60:.0f}m elapsed)", flush=True)
            time.sleep(args.cooldown)

    print(f"done in {(time.time() - t0) / 60:.1f} min; log at {log}")


if __name__ == "__main__":
    sys.exit(main())
