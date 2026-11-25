"""Ordered coverage from every admitted frame match (the dump's "all").

A match is in order when another admitted match uses a different sample on
both sides, sits at the same offset (within the pair's keyframe spacing w) and
lies within K spacings of it in a's time. Coverage on each side is scaled by
the share of that side's matched samples that have an in-order match, and the
link survives when max(scaled a, scaled b) >= -p.

    CORPUS_ROOT=... python3 order2.py links.jsonl [more.jsonl]
"""
import bisect, collections, io, contextlib, json, os, sys
import order as O


def in_order(rec, K):
    ps = rec["all"]
    w = max(500, rec["ta"] / max(1, rec["na"]), rec["tb"] / max(1, rec["nb"]))
    ps = sorted(ps)                      # by t_a
    tas = [p[0] for p in ps]
    ok = [False] * len(ps)
    for k, (ta, tb, _) in enumerate(ps):
        o = tb - ta
        lo = bisect.bisect_left(tas, ta - K * w)
        hi = bisect.bisect_right(tas, ta + K * w)
        for j in range(lo, hi):
            ta2, tb2, _ = ps[j]
            if ta2 != ta and tb2 != tb and abs((tb2 - ta2) - o) <= w:
                ok[k] = True
                break
    sa, sb = set(), set()
    oa, ob = set(), set()
    for k, (ta, tb, _) in enumerate(ps):
        sa.add(ta); sb.add(tb)
        if ok[k]:
            oa.add(ta); ob.add(tb)
    return len(oa) / max(1, len(sa)), len(ob) / max(1, len(sb)), len(sa), len(sb)


def load(path):
    rows = collections.defaultdict(list)
    V = O.S.VIDS
    for line in open(path):
        rec = json.loads(line)
        a, b = os.path.relpath(rec["a"], V), os.path.relpath(rec["b"], V)
        if a in O.man and b in O.man:
            rows[O.label(a, b)].append((a, b, rec))
    return rows


def sim(path, Ks=(1, 2, 4), gate=0.20):
    rows = load(path)
    print(path)
    for K in Ks:
        c = collections.Counter(); tot = collections.Counter(); fam = collections.Counter()
        for lab, r in rows.items():
            for a, b, rec in r:
                tot[lab] += 1
                fa, fb, na, nb = in_order(rec, K)
                if max(rec["ca"] * fa, rec["cb"] * fb) >= gate:
                    c[lab] += 1
                elif lab == "TP":
                    fam[O.man[a]["transform"] + "/" + O.man[b]["transform"]] += 0
                if lab == "FP_look" and max(rec["ca"] * fa, rec["cb"] * fb) >= gate:
                    fam[O.sets[O.man[a]["seed"]]] += 1
        print(f"  K={K}: " + "  ".join(f"{k} {c[k]}/{tot[k]}" for k in ("TP", "FP_look", "FP_other", "shared", "scene"))
              + f"   look-alikes left {dict((k, v) for k, v in fam.items() if v)}")
    return rows


if __name__ == "__main__":
    for p in sys.argv[1:]:
        sim(p)
