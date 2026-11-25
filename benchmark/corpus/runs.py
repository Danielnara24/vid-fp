"""Longest single-offset run per link, in ms of a's time, from every admitted match.
A run: matches sorted by a-time whose offset stays within the pair's spacing w of the
run's start offset and whose a-times are no more than 2w apart."""
import collections, sys, json
import order as O, order2 as O2, distinct as D

def runs(rec):
    w = max(500, rec["ta"] / max(1, rec["na"]), rec["tb"] / max(1, rec["nb"]))
    by = collections.defaultdict(list)
    ps = sorted(rec["all"], key=lambda p: (p[1] - p[0], p[0]))
    # cluster offsets
    out = []
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and (ps[j + 1][1] - ps[j + 1][0]) - (ps[j][1] - ps[j][0]) <= w:
            j += 1
        ts = sorted({p[0] for p in ps[i:j + 1]})
        s = ts[0]; best = 0
        for k in range(1, len(ts)):
            if ts[k] - ts[k - 1] > 2 * w:
                best = max(best, ts[k - 1] - s); s = ts[k]
        best = max(best, ts[-1] - s)
        out.append(best)
        i = j + 1
    return max(out) if out else 0, w

def main(selfp, linkp, low=0.25):
    tol, selfd = D.load_self(selfp)
    rows = O2.load(linkp)
    for lab in ("TP", "FP_look", "FP_other"):
        r = rows.get(lab, [])
        lo = [runs(rec)[0] / 1000 for a, b, rec in r if min(D.shares(rec, selfd, 2, tol["loose"])) <= low]
        v = sorted(lo)
        if not v: continue
        q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
        print(f"   {lab:9} low-distinct links {len(v):5}/{len(r)}  longest run s: p05 {q(.05):6.1f} p25 {q(.25):6.1f} med {q(.5):6.1f} p75 {q(.75):6.1f} p95 {q(.95):6.1f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
