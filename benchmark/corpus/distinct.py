"""Are look-alike links made of frames that recur inside their own video?

self dump: each sample's nearest look-alike in its own video at least `sep`
apart (seps: own keyframe span, 10 s, 30 s, 120 s). A matched sample is
non-distinctive at (sep, thr) when that distance is <= thr (strict or loose).
Per link: share of matched samples on each side that are distinctive.

    CORPUS_ROOT=... python3 distinct.py self.jsonl links.jsonl
"""
import collections, json, os, sys
import order as O
import order2 as O2


def load_self(path):
    f = open(path)
    tol = json.loads(f.readline())
    out = {}
    for line in f:
        r = json.loads(line)
        out[r["path"]] = {t: m for t, m in zip(r["t"], r["min"])}
    return tol, out


def shares(rec, selfd, k, thr):
    sa = {ta for ta, _, _ in rec["all"]}
    sb = {tb for _, tb, _ in rec["all"]}
    da, db = selfd[rec["a"]], selfd[rec["b"]]
    fa = sum(1 for t in sa if da.get(t, [99] * 4)[k] > thr) / max(1, len(sa))
    fb = sum(1 for t in sb if db.get(t, [99] * 4)[k] > thr) / max(1, len(sb))
    return fa, fb


def main(selfp, linkp):
    tol, selfd = load_self(selfp)
    rows = O2.load(linkp)
    for k, sep in enumerate(("span", "10s", "30s", "120s")):
        for tname in ("strict", "loose"):
            thr = tol[tname]
            print(f"\nsep {sep}, recur within {tname} ({thr} bits): share of matched samples distinctive, min of the two sides")
            for lab in ("TP", "FP_look", "FP_other", "shared"):
                r = rows.get(lab, [])
                if not r:
                    continue
                v = sorted(min(shares(rec, selfd, k, thr)) for a, b, rec in r)
                q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
                print(f"   {lab:9} {len(v):5}  p05 {q(.05):.2f}  p25 {q(.25):.2f}  med {q(.5):.2f}  p75 {q(.75):.2f}  p95 {q(.95):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
