"""Do look-alike links have their matched frames in a scrambled order?

Reads a VIDFP_LINKDUMP jsonl (every reported link with its admitted frame
matches: one (t_a, t_b, dist) per matched sample of a) and labels each link
with score.py's truth. Per link:

  n        matched samples
  step     fraction of consecutive matches (in a's order) whose offset agrees
           with the previous one within the pair's keyframe spacing
  lis      longest chain increasing in both files, as a fraction of n
  chained  fraction of matches with an aligned neighbour (offset within the
           spacing, a different sample on both sides, within 10 spacings)

    CORPUS_ROOT=... python3 order.py links.jsonl
"""
import bisect, collections, json, os, sys
import score as S

man = S.load_manifest()
sets = S.load_sets()
pairs = S.load_pairs()
keys = {f: S.scene_keys(r, sets) for f, r in man.items()}


def label(a, b):
    p = frozenset((a, b))
    sa, sb = keys.get(a, set()), keys.get(b, set())
    scene = bool(sa & sb)
    t = S.truth(pairs.get(p), 0.20, False, scene)
    if t == "P":
        return "TP"
    if t == "N":
        if p not in pairs and any((sa & sh) and (sb & sh) and (sa | sb) >= sh for sh in S.SHARED):
            return "shared"
        ma, mb = man[a]["seed"], man[b]["seed"]
        if sets.get(ma) and sets.get(ma) == sets.get(mb) and ma != mb:
            return "FP_look"
        return "FP_other"
    return "excl" if t is None else "scene"


def lis(xs):
    tails = []
    for x in xs:
        i = bisect.bisect_right(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def stats(rec):
    ps = sorted(rec["pairs"])
    n = len(ps)
    w = max(500, rec["ta"] / max(1, rec["na"]), rec["tb"] / max(1, rec["nb"]))
    off = [tb - ta for ta, tb, _ in ps]
    step = sum(1 for k in range(1, n) if abs(off[k] - off[k - 1]) <= w) / max(1, n - 1)
    chained = 0
    for k in range(n):
        ok = False
        for j in range(n):
            if j != k and ps[j][0] != ps[k][0] and ps[j][1] != ps[k][1] \
                    and abs(off[j] - off[k]) <= w and abs(ps[j][0] - ps[k][0]) <= 10 * w:
                ok = True
                break
        chained += ok
    return n, step, lis([tb for _, tb, _ in ps]) / max(1, n), chained / max(1, n)


def main(path):
    V = S.VIDS
    rows = collections.defaultdict(list)
    for line in open(path):
        rec = json.loads(line)
        a, b = os.path.relpath(rec["a"], V), os.path.relpath(rec["b"], V)
        if a not in man or b not in man:
            continue
        rows[label(a, b)].append((stats(rec), a, b, rec))
    for lab in ("TP", "FP_look", "FP_other", "shared", "excl", "scene"):
        r = rows.get(lab, [])
        if not r:
            continue
        print(f"\n{lab}: {len(r)} links")
        for name, ix in (("n", 0), ("step", 1), ("lis", 2), ("chained", 3)):
            v = sorted(x[0][ix] for x in r)
            q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
            print(f"   {name:8} p01 {q(.01):7.2f}  p05 {q(.05):7.2f}  p25 {q(.25):7.2f}  med {q(.5):7.2f}  p75 {q(.75):7.2f}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1])
