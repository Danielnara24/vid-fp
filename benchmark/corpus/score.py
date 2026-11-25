#!/usr/bin/env python3
"""Score a vid-fp run against the generated corpus. Nothing here is labelled
by hand: every pair's truth comes from pairs.csv, which make_variants.py
derived from where each file's picture came from.

    python3 score.py run -d 4 -p 20            # scan VIDS, then score
    python3 score.py report out/run.json -p 20 # score an existing JSON report

## What counts

A predicted pair is any link vid-fp measured (the `matches` of every file in
the JSON report, both directions folded into one unordered pair). Groups are
stars, so a group is NOT expanded to all its pairs: the links are the claims.

    positive   relation SAME or CLIP (one file holds >= 90% of the other's
               runtime), spatially SAME or CROP, no trap
    PARTIAL    footage shared but neither holds 90% of the other. A positive
               when max(cov_a, cov_b) >= --partial-gate (default: the run's
               -p), otherwise excluded: a gate can find it or not, both are
               defensible at a different -p.
    negative   relation DIFFERENT (same seed, no shared second), spatially
               DISJOINT, or any pair of files from different seeds that no
               compilation joins
    excluded   traps (rotate_meta, rotate90, vflip, speed, reverse) unless
               --traps, and spatially PARTIAL crops
    SCENE      the same scene at a different moment, scored on its own like
               a trap: a pair that would be a negative, but whose files hold
               pictures of one seed (disjoint stretches of it), or of two
               seeds cut from one static camera shot (SCENE_SETS). Reported
               as "scene links: k of n" and kept out of P/R.

Recall per transform is the share of (seed, derived) pairs found, which is
the question a user asks of a transform: "would it find my original?".
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import time

VIDS = os.environ.get("CORPUS_ROOT", "/home/daniel/Documents/VIDS")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "out")


def load_manifest():
    m = {}
    for r in csv.DictReader(open(os.path.join(VIDS, "manifest.csv"))):
        m[r["file"]] = r
    return m


def load_sets():
    sets = {}
    for r in csv.DictReader(open(os.path.join(VIDS, "seeds.csv"))):
        sets[r["seed"]] = r["near_miss_set"]
    return sets


SCENE_SETS = {"talkhead", "podcast"}     # cuts of one static-camera video
# Seeds that really share footage (a re-aired segment, a rebroadcast feed),
# confirmed by eye on 2026-11-18. Their intervals are not known exactly, so a
# link between the two families is neither a TP nor an FP: it is counted alone.
SHARED = [{"news_1", "news_2"}, {"starship_1", "starship_3"}]
if os.environ.get("SHARED_LYRIC"):      # the user confirmed lyric_1/lyric_2 share a real intro
    SHARED.append({"lyric_1", "lyric_2"})
if os.path.basename(VIDS.rstrip("/")) == "VIDS2":
    # real footage shared to an unknown extent; scored as the real-world tier
    SHARED = [{"heat_movie", "heat_trailer"}, {"toronto_movie", "toronto_trailer"},
              {"gta_trailer", "gta_react_1"}, {"gta_trailer", "gta_react_2"},
              {"gta_react_1", "gta_react_2"}]


def scene_keys(row, sets):
    """The scene families a manifest row's pictures come from."""
    out = set()
    for seg in row["segments"].split(";"):
        sd = seg.split("@")[0]
        out.add(sets[sd] if sets.get(sd) in SCENE_SETS else sd)
    return out


def load_pairs():
    p = {}
    for r in csv.DictReader(open(os.path.join(VIDS, "pairs.csv"))):
        p[frozenset((r["file_a"], r["file_b"]))] = r
    return p


def predicted(report):
    d = json.load(open(report))
    pred = set()
    for g in d["results"]:
        for f in g["files"]:
            a = os.path.relpath(f["full_path"], VIDS)
            for m in f.get("matches") or []:
                b = os.path.relpath(m["full_path"], VIDS)
                if a != b:
                    pred.add(frozenset((a, b)))
    return pred, d


def truth(row, gate, traps, scene=False):
    """'P', 'N', 'S' (same scene, other moment) or None (excluded)."""
    if row is None:
        return "S" if scene else "N"
    if row["trap"] and not traps:
        return None
    if row["relation"] == "DIFFERENT":
        return "S" if scene else "N"
    if row["spatial"] == "DISJOINT":
        return "N"
    if row["spatial"] == "PARTIAL":
        return None
    if row["relation"] in ("SAME", "CLIP"):
        return "P"
    return "P" if max(float(row["cov_a"]), float(row["cov_b"])) >= gate else None


def score(report, gate, traps=False, show=12, keep=None):
    man = load_manifest()
    sets = load_sets()
    pairs = load_pairs()
    pred, rep = predicted(report)
    if keep is not None:
        pred = {p for p in pred if all(keep(f) for f in p)}
        pairs = {k: v for k, v in pairs.items() if all(keep(f) for f in k)}
    files = set(man)
    stray = {f for p in pred for f in p} - files
    if stray:
        print(f"warning: {len(stray)} reported files are not in the manifest (e.g. {sorted(stray)[0]})")

    keys = {f: scene_keys(r, sets) for f, r in man.items()}

    def is_shared(p):
        a, b = tuple(p)
        sa = {k for k in keys.get(a, ())}
        sb = {k for k in keys.get(b, ())}
        return any((sa & sh) and (sb & sh) and (sa | sb) >= sh for sh in SHARED)

    def is_scene(p):
        a, b = tuple(p)
        return bool(keys.get(a, set()) & keys.get(b, set()))

    tp = fp = fn = 0
    scene_hit = 0
    shared_hit = 0
    fp_by = collections.Counter()
    fp_rows = []
    for p in pred:
        t = truth(pairs.get(p), gate, traps, is_scene(p))
        if t == "N" and p not in pairs and is_shared(p):
            shared_hit += 1
        elif t == "S":
            scene_hit += 1
        elif t == "P":
            tp += 1
        elif t == "N":
            fp += 1
            a, b = sorted(p)
            sa, sb = man.get(a, {}).get("seed"), man.get(b, {}).get("seed")
            if sa == sb:
                kind = "same seed, " + (pairs[p]["relation"] if p in pairs else "?")
            elif sets.get(sa) and sets.get(sa) == sets.get(sb):
                kind = f"near-miss set {sets[sa]}"
            else:
                kind = "unrelated seeds"
            fp_by[kind] += 1
            fp_rows.append((kind, a, b))
    per_t = collections.defaultdict(lambda: [0, 0])      # transform -> [found, total] vs its seed
    per_g = collections.defaultdict(lambda: [0, 0])
    miss_rows = []
    for k, row in pairs.items():
        t = truth(row, gate, traps)
        if t != "P":
            continue
        if k not in pred:
            fn += 1
        if "seed" in (row["transform_a"], row["transform_b"]):
            tr = row["transform_b"] if row["transform_a"] == "seed" else row["transform_a"]
            g = man[row["file_b"] if row["transform_a"] == "seed" else row["file_a"]]["group"]
            per_t[tr][1] += 1
            per_g[g][1] += 1
            if k in pred:
                per_t[tr][0] += 1
                per_g[g][0] += 1
            else:
                miss_rows.append((tr, row["file_a"], row["file_b"]))
    # every pair of files sharing a scene family that is not a positive
    fam = collections.defaultdict(list)
    for f, ks in keys.items():
        for k in ks:
            fam[k].append(f)
    scene_all = set()
    for fs in fam.values():
        fs = sorted(f for f in fs if keep is None or keep(f))
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                p = frozenset((fs[i], fs[j]))
                if truth(pairs.get(p), gate, traps, True) == "S":
                    scene_all.add(p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\n== {os.path.basename(report)}   gate for PARTIAL {gate:.2f}   traps {'in' if traps else 'out'}")
    print(f"pairs   TP {tp}  FP {fp}  FN {fn}   precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")
    print(f"groups  {len(rep['results'])}")
    print(f"scene links (same scene, other moment; not in P/R): {scene_hit} of {len(scene_all)}")
    print(f"shared-footage links (news_1/2, starship_1/3; not in P/R): {shared_hit}")
    if fp_by:
        print("false positives by kind: " + ", ".join(f"{k} {v}" for k, v in fp_by.most_common()))
    # merges: seed pairs joined by at least one false link, and groups holding one
    joined = collections.Counter()
    for kind, a, b in fp_rows:
        sa, sb = man.get(a, {}).get("seed"), man.get(b, {}).get("seed")
        if sa != sb:
            joined[tuple(sorted((sa, sb)))] += 1
    fp_set = {frozenset((a, b)) for _, a, b in fp_rows}
    bad_groups = 0
    for g in rep["results"]:
        members = [os.path.relpath(f["full_path"], VIDS) for f in g["files"]]
        rep0 = members[0]
        if any(frozenset((rep0, m)) in fp_set for m in members[1:]):
            bad_groups += 1
    print(f"merges: {len(joined)} seed pairs joined by false links; {bad_groups} groups hold a false link")
    for (sa, sb), n in joined.most_common(15):
        tag = " (near-miss set)" if sets.get(sa) and sets.get(sa) == sets.get(sb) else ""
        print(f"   {sa} + {sb}: {n} links{tag}")
    tot_f = sum(v[0] for v in per_t.values())
    tot_n = sum(v[1] for v in per_t.values())
    print(f"\nseed <-> derived recall {tot_f}/{tot_n} = {tot_f / max(1, tot_n):.4f}")
    print("by group:   " + "  ".join(f"{g} {f}/{n}" for g, (f, n) in sorted(per_g.items())))
    worst = sorted(per_t.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]))
    perfect = sum(1 for f, n in per_t.values() if f == n)
    print(f"perfect transforms {perfect} / {len(per_t)}; hardest:")
    for tr, (f, n) in worst[:show]:
        if f == n:
            break
        print(f"   {tr:20} {f:3}/{n:<3} {f / n:.3f}")
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, per_transform=dict(per_t),
                scene=scene_hit, scene_all=len(scene_all), fp_by=dict(fp_by), fp_rows=fp_rows, miss_rows=miss_rows, joined=dict(joined),
                bad_groups=bad_groups)


def run_vidfp(d, p, extra, name):
    os.makedirs(OUT, exist_ok=True)
    rep = os.path.join(OUT, name + ".json")
    binary = os.environ.get("VID_FP_BIN", os.path.join(REPO, "target", "release", "vid-fp"))
    excl = []
    for hidden in (".mezz", ".raw", ".tmp", ".tools"):      # mezzanines would be stray near-copies
        if os.path.isdir(os.path.join(VIDS, hidden)):
            excl += ["-e", os.path.join(VIDS, hidden)]
    cmd = [binary, VIDS, "-r", "-d", str(d), "-p", str(p), "-o", rep, "-q",
           "--cache", os.environ.get("SCORE_CACHE", os.path.join(OUT, "cache.redb")), *excl, *extra]
    t0 = time.time()
    r = subprocess.run(["/usr/bin/time", "-f", "%e s wall  %U s user  %M KB peak"] + cmd,
                       capture_output=True, text=True)
    print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "", f"(exit {r.returncode})")
    if not os.path.exists(rep):
        sys.exit(r.stderr[-3000:])
    return rep


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("-d", type=int, default=4)
    a.add_argument("-p", type=float, default=20)
    a.add_argument("--name")
    a.add_argument("--traps", action="store_true")
    a.add_argument("extra", nargs="*", help="passed to vid-fp after --")
    b = sub.add_parser("report")
    b.add_argument("path")
    b.add_argument("-p", type=float, default=20, help="the run's -p, used as the PARTIAL gate")
    b.add_argument("--traps", action="store_true")
    for x in (a, b):
        x.add_argument("--partial-gate", type=float, help="default: -p / 100")
        x.add_argument("--dump", help="write FP and miss lists to this file")
    args = ap.parse_args()
    gate = args.partial_gate if args.partial_gate is not None else args.p / 100
    if args.cmd == "run":
        name = args.name or f"d{args.d}-p{int(args.p)}"
        rep = run_vidfp(args.d, args.p, args.extra, name)
    else:
        rep = args.path
    res = score(rep, gate, args.traps)
    if args.dump:
        with open(args.dump, "w") as f:
            for k, a_, b_ in sorted(res["fp_rows"]):
                f.write(f"FP\t{k}\t{a_}\t{b_}\n")
            for tr, a_, b_ in sorted(res["miss_rows"]):
                f.write(f"MISS\t{tr}\t{a_}\t{b_}\n")


if __name__ == "__main__":
    main()
