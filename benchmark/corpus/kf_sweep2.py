#!/usr/bin/env python3
"""--keyframe-interval x --min-keyframes on both generated corpora (VIDS, VIDS2).

Each cell: one COLD run at the defaults (-d 6 -p 20, --clear-cache on a
per-corpus cache), timed with /usr/bin/time, then one WARM run of the same
cell at -p 0, which says how many of the pairs the cell loses are lost to the
-p gate rather than to the sampler.

The interval-0 default is run at the start, the middle and the end of each
corpus's share of the sweep, so thermal drift is visible. Cells are visited in
a seeded shuffle, alternating corpora, with a cooldown between cold runs.
Appends to out/kfsweep2/runs.jsonl and skips cells already there (resumable).
"""
import json, os, random, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.abspath(os.path.join(HERE, "..", "..", "target", "release", "vid-fp"))
OUT = os.path.join(HERE, "out", "kfsweep2")
CORPORA = {"VIDS": "/home/daniel/Documents/VIDS", "VIDS2": "/home/daniel/Documents/VIDS2"}
INTERVALS = [float(x) if '.' in x else int(x) for x in os.environ.get("KF_INTERVALS", "5,13,21,29").split(",")]
MINKF = [int(x) for x in os.environ.get("KF_MINKF", "0,4,8,12,16,20,28").split(",")]
ROUND = os.environ.get("KF_ROUND", "")          # non-empty: a timing-only repeat round, keys suffixed
BASELINES = os.environ.get("KF_BASELINES", "1") == "1"
DEADLINE = float(os.environ.get("KF_DEADLINE", "0"))   # epoch seconds; no run starts after it
COOLDOWN = 20
FMT = "VIDFP_TIME %e %U %S %M"


def vidfp(corpus, cache, i, m, p, dest, cold):
    root = CORPORA[corpus]
    excl = []
    for h in (".mezz", ".raw", ".tmp", ".tools"):
        if os.path.isdir(os.path.join(root, h)):
            excl += ["-e", os.path.join(root, h)]
    cmd = ["/usr/bin/time", "-f", FMT, BIN, root, "-r", "-d", "6", "-p", str(p),
           "--keyframe-interval", str(i), "--min-keyframes", str(m),
           "-o", dest, "-q", "--cache", cache, *excl]
    if cold:
        cmd.append("--clear-cache")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in (0, 2):
        sys.exit(f"rc={r.returncode} {corpus} i{i} m{m} p{p}\n{r.stderr[-2000:]}")
    _, wall, user, sysd, rss = next(l for l in r.stderr.splitlines() if l.startswith("VIDFP_TIME")).split()
    return dict(wall=float(wall), user=float(user), sys=float(sysd), rss_mb=int(rss) / 1024, rc=r.returncode)


def main():
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, "runs.jsonl")
    done = set()
    if os.path.exists(log):
        for line in open(log):
            r = json.loads(line)
            done.add(r["key"])
    rng = random.Random(20261121)
    plan = {}
    for c in CORPORA:
        cells = [(i, m) for i in INTERVALS for m in MINKF]
        rng.shuffle(cells)
        h = len(cells) // 2
        if BASELINES:
            plan[c] = [(0, 12, "b0")] + cells[:h] + [(0, 12, "b1")] + cells[h:] + [(0, 12, "b2")]
        else:
            plan[c] = cells
    order = [x for pair in zip(*(([(c, *t) for t in plan[c]]) for c in CORPORA)) for x in pair]
    t0 = time.time()
    for n, (c, i, m, *tag) in enumerate(order, 1):
        tag = tag[0] if tag else ""
        key = f"{c}_i{i}_m{m}{'_' + tag if tag else ''}{'_' + ROUND if ROUND else ''}"
        if key in done:
            continue
        if DEADLINE and time.time() > DEADLINE:
            print("deadline reached; stopping", flush=True)
            return
        cache = os.path.join(OUT, f"cache-{c}.redb")
        dest = os.path.join(OUT, key + ".json")
        rec = dict(key=key, corpus=c, interval=i, min_kf=m, tag=tag, round=ROUND)
        rec["cold"] = vidfp(c, cache, i, m, 20, dest, cold=True)
        rec["json"] = dest
        if not ROUND and (not tag or tag == "b0"):       # the -p 0 rerun is deterministic; once per cell
            d0 = os.path.join(OUT, key + "_p0.json")
            rec["warm_p0"] = vidfp(c, cache, i, m, 0, d0, cold=False)
            rec["json_p0"] = d0
            # warm -p 20 too: the comparison-only cost of the cell
            rec["warm_p20"] = vidfp(c, cache, i, m, 20, os.path.join(OUT, "scratch.json"), cold=False)
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{n}/{len(order)}] {key}: cold {rec['cold']['wall']:.0f}s user {rec['cold']['user']:.0f}s "
              f"rss {rec['cold']['rss_mb']:.0f}MB  ({(time.time() - t0) / 60:.0f} min)", flush=True)
        time.sleep(COOLDOWN)


if __name__ == "__main__":
    main()
