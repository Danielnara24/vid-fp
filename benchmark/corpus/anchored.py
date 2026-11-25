"""Coverage from matches that are distinctive (both samples recur nowhere in
their own video >= sep away within loose), or that sit in a single-offset run
holding at least one distinctive match. Gate max(scaled ca, scaled cb) >= .20."""
import collections, sys
import order2 as O2, distinct as D, order as O

def kept_share(rec, da, db, k, thr):
    w = max(500, rec["ta"] / max(1, rec["na"]), rec["tb"] / max(1, rec["nb"]))
    ps = sorted(rec["all"], key=lambda p: (p[1] - p[0], p[0]))
    good = set()
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and (ps[j + 1][1] - ps[j + 1][0]) - (ps[j][1] - ps[j][0]) <= w:
            j += 1
        grp = sorted(ps[i:j + 1])
        # split the offset group into runs by a-time gaps > 2w
        run = [grp[0]]
        def flush(run):
            if any(da.get(ta, [99]*4)[k] > thr and db.get(tb, [99]*4)[k] > thr for ta, tb, _ in run):
                good.update((ta, tb) for ta, tb, _ in run)
        for p in grp[1:]:
            if p[0] - run[-1][0] > 2 * w:
                flush(run); run = [p]
            else:
                run.append(p)
        flush(run)
        i = j + 1
    sa = {p[0] for p in rec["all"]}; sb = {p[1] for p in rec["all"]}
    ga = {ta for ta, _ in good}; gb = {tb for _, tb in good}
    return len(ga) / max(1, len(sa)), len(gb) / max(1, len(sb))

def main(selfp, linkp):
    tol, selfd = D.load_self(selfp)
    rows = O2.load(linkp)
    print(linkp)
    for k, sep in ((1, "10s"), (2, "30s"), (3, "120s")):
        c = collections.Counter(); tot = collections.Counter(); lostT = collections.Counter()
        for lab, r in rows.items():
            for a, b, rec in r:
                tot[lab] += 1
                fa, fb = kept_share(rec, selfd[rec["a"]], selfd[rec["b"]], k, tol["loose"])
                if max(rec["ca"] * fa, rec["cb"] * fb) >= 0.20:
                    c[lab] += 1
                elif lab == "TP":
                    lostT[O.sets.get(O.man[a]["seed"]) or O.man[a]["seed"]] += 1
        print(f"  sep {sep}: " + "  ".join(f"{x} {c[x]}/{tot[x]}" for x in ("TP", "FP_look", "FP_other", "shared", "scene"))
              + f"   TP lost by seed set {lostT.most_common(5)}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
