"""Find footage two DIFFERENT seeds share (a re-aired segment, a rebroadcast feed)
from the pictures alone, at 5 fps, within each near-miss set. Writes VIDS/shared.csv:

    seed_a,a0,a1,seed_b,b0     seed_a[a0,a1] shows what seed_b[b0, b0+(a1-a0)] shows

A run is kept only if it is a diagonal of close matches at least MIN_RUN long that
beats the neighbouring offsets clearly, so a static set seen at two moments (the
talkhead cuts) does not read as shared footage.
"""
import csv, itertools, os, sys
import numpy as np
from dense import dense, phash

VIDS = "/home/daniel/Documents/VIDS"
FPS = 5
MIN_RUN = 2.0       # seconds
NEAR = 0.22         # fraction of bits: a match
MARGIN = 0.08       # the run's diagonal must beat offsets >= 1 s away by this much


def runs(ha, hb):
    nb = ha.shape[1]
    d = (ha[:, None, :] != hb[None, :, :]).sum(2) / nb          # [na, nb]
    na, nB = d.shape
    out = []
    for off in range(-na + 1, nB):
        diag = np.diagonal(d, off)
        m = diag <= NEAR
        i = 0
        while i < len(m):
            if not m[i]:
                i += 1
                continue
            j = i
            while j < len(m) and m[j]:
                j += 1
            if (j - i) / FPS >= MIN_RUN:
                ia = i + max(0, -off)            # row index in A
                ib = ia + off
                own = diag[i:j].mean()
                # same A frames against B shifted by >= 1 s either way
                others = []
                for s in list(range(-15, -4)) + list(range(5, 16)):
                    cols = np.arange(ib, ib + (j - i)) + s
                    ok = (cols >= 0) & (cols < nB)
                    if ok.sum() > (j - i) // 2:
                        others.append(d[np.arange(ia, ia + (j - i))[ok], cols[ok]].mean())
                # and nowhere else in B: a still that recurs is not shared footage
                rows = np.arange(ia, ia + (j - i))
                far = d[rows].copy()
                for k, r in enumerate(rows):
                    far[k, max(0, ib + k - 10):ib + k + 11] = 1.0
                elsewhere = far.min(1).mean()
                if (not others or min(others) - own >= MARGIN) and elsewhere - own >= MARGIN:
                    out.append((ia, ia + (j - i), ib, own, min(others) if others else None))
            i = j
    return out


def main():
    sets = {}
    for r in csv.DictReader(open(os.path.join(VIDS, "seeds.csv"))):
        if r["near_miss_set"]:
            sets.setdefault(r["near_miss_set"], []).append(r)
    rows = []
    for name, members in sorted(sets.items()):
        H = {}
        for r in members:
            t, g = dense(os.path.join(VIDS, r["file"]), FPS)
            H[r["seed"]] = (t, phash(g))
        for a, b in itertools.combinations(sorted(H), 2):
            (ta, ha), (tb, hb) = H[a], H[b]
            for ia, ja, ib, own, oth in runs(ha, hb):
                a0, a1, b0 = ta[ia], ta[ja - 1] + 1 / FPS, tb[ib]
                print(f"{name:12} {a} {a0:7.1f}-{a1:7.1f}  = {b} {b0:7.1f}   dist {own:.3f} vs {oth}", flush=True)
                rows.append((a, round(a0, 2), round(a1, 2), b, round(b0, 2)))
    with open(os.path.join(VIDS, "shared.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed_a", "a0", "a1", "seed_b", "b0"])
        w.writerows(rows)
    print(len(rows), "shared runs")


if __name__ == "__main__":
    main()
