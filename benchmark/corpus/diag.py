"""Why are positives missed? Per missed positive pair: best distance, matches at
<= 4/10/16 bits, coverage the rule reached, transform."""
import os, sys, collections
import numpy as np
import rules2 as R
import score as S


def positives(gate=0.20):
    pairs = S.load_pairs()
    out = {}
    for k, row in pairs.items():
        if S.truth(row, gate, False) == "P":
            out[k] = row
    return out


def main():
    D = R.Data()
    need = R.schedule(4, 10)
    ca, cb = R.apply_rule(D, need)
    nf = D.nf
    idx = {str(p): n for n, p in enumerate(D.paths)}
    k = D.gkey
    fa = (k // 2) // nf; fb = (k // 2) % nf; m = k % 2
    pa = np.minimum(ca / D.total[fa], 1); pb = np.minimum(cb / D.total[fb], 1)
    # per unordered pair: best over orientations of several stats
    st = {}
    for g in range(len(k)):
        a, b = D.gs[g], D.gs[g + 1]
        dd = D.d[a:b]
        p = (int(fa[g]), int(fb[g]))
        v = (max(pa[g], pb[g]), int(dd.min()), int((dd <= 4).sum()), int((dd <= 10).sum()), int((dd <= 16).sum()), int(m[g]))
        if p not in st or v > st[p]:
            st[p] = v
    pos = positives()
    rows = []
    for kk, row in pos.items():
        a, b = tuple(kk)
        if a not in idx or b not in idx:
            continue
        ia, ib = sorted((idx[a], idx[b]))
        v = st.get((ia, ib), (0, 99, 0, 0, 0, 0))
        tr = (row["transform_a"], row["transform_b"])
        rows.append((v, tr, row))
    found = [r for r in rows if r[0][0] >= 0.20]
    miss = [r for r in rows if r[0][0] < 0.20]
    print(f"positives {len(rows)}  found {len(found)}  missed {len(miss)}")
    b = collections.Counter()
    for v, tr, row in miss:
        cov, dmin, n4, n10, n16, _ = v
        if dmin > 16:
            c = "no match <=16"
        elif n10 == 0:
            c = "best 12-16"
        elif cov > 0:
            c = "some coverage <20%"
        else:
            c = "matches <=10 but none accepted"
        b[c] += 1
    for c, n in b.most_common():
        print(f"  {c:32} {n}")
    # by relation
    rel = collections.Counter((row["relation"], v[0] > 0) for v, tr, row in miss)
    print(rel)
    return D, rows


if __name__ == "__main__":
    main()


def buckets(rows):
    out = collections.defaultdict(collections.Counter)
    for v, tr, row in rows:
        cov, dmin, n4, n10, n16, _ = v
        if cov >= 0.2:
            continue
        c = ("none<=16" if dmin > 16 else "best12-16" if n10 == 0 else "cov<20" if cov > 0 else "unaccepted")
        for t in tr:
            out[c][t] += 1
    for c, ctr in out.items():
        print(c, ctr.most_common(14))
