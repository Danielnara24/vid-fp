"""Matching-rule lab: all frame matches <= CAP bits, then vid-fp's rule (or a variant)
over every file pair, then score.py's scoring.

    .venv/bin/python rules2.py pairs        # run the C matcher on out/lab/fps.npz
    .venv/bin/python rules2.py eval -d 4    # replicate the binary
"""
import os, sys, json, glob, subprocess, argparse, time
import numpy as np
import numba as nb

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "out", "lab")
CAP = 18


def run_pairs(fps=os.path.join(LAB, "fps.npz"), out=os.path.join(LAB, "pairs"), cap=CAP):
    z = np.load(fps)
    H, M, F = z["H"], z["M"], z["F"]
    inp = out + ".in"
    with open(inp, "wb") as f:
        np.int64(len(H)).tofile(f)
        H.astype(np.uint64).tofile(f); M.astype(np.uint64).tofile(f); F.astype(np.int32).tofile(f)
    for p in glob.glob(out + ".bin.*"):
        os.remove(p)
    t = time.time()
    subprocess.run([os.path.join(LAB, "allpairs"), inp, out + ".bin", str(cap)], check=True)
    print(f"matcher {time.time() - t:.1f} s")
    rec = np.dtype([("i", "<i4"), ("j", "<i4"), ("d", "u1"), ("m", "u1")])
    parts = [np.fromfile(p, dtype=rec) for p in sorted(glob.glob(out + ".bin.*"))]
    r = np.concatenate(parts)
    print(f"{len(r)} frame matches <= {cap}")
    np.save(out + ".npy", r)


class Data:
    def __init__(self, fps=os.path.join(LAB, "fps.npz"), pairs=os.path.join(LAB, "pairs.npy"), maxd=CAP):
        z = np.load(fps, allow_pickle=False)
        self.H, self.M, self.F = z["H"], z["M"], z["F"]
        self.T0, self.T1 = z["T0"].astype(np.int64), z["T1"].astype(np.int64)
        self.paths, self.total, self.dur = z["paths"], z["total"], z["dur"]
        self.C = z["C"] if "C" in z.files else None
        r = np.load(pairs)
        r = r[r["d"] <= maxd]
        i = r["i"].astype(np.int64); j = r["j"].astype(np.int64)
        fa = self.F[i].astype(np.int64); fb = self.F[j].astype(np.int64)
        off = self.T0[i] - self.T0[j]
        key = (fa * len(self.paths) + fb) * 2 + r["m"]
        o = np.lexsort((off, key))
        self.i, self.j, self.d, self.off, self.key = i[o], j[o], r["d"][o].astype(np.int64), off[o], key[o]
        starts = np.flatnonzero(np.r_[True, self.key[1:] != self.key[:-1]])
        self.gs = np.r_[starts, len(self.key)].astype(np.int64)
        self.gkey = self.key[starts]
        self.nf = len(self.paths)
        print(f"{len(self.i)} matches in {len(starts)} oriented file pairs")


def schedule(strict, loose, budget=2.0, anchor=5.0):
    """witness_schedule: witnesses needed per distance (huge = never)."""
    from math import comb, log10
    tail = lambda x: sum(comb(64, k) for k in range(x + 1)) / 2 ** 64
    om = lambda x: -log10(tail(x))
    anc = om(int(32 - anchor * 4))
    out = np.full(65, 10 ** 6, np.int64)
    for x in range(65):
        if x <= strict:
            out[x] = 0
        elif x <= loose:
            # cluster of 1 + w carries (1+w)*om(x) >= budget*anc
            w = 1
            while (1 + w) * om(x) < budget * anc and w < 64:
                w += 1
            out[x] = w
    return out


@nb.njit(cache=True)
def _apply(gs, I, J, D, OFF, span, need, tol, markA, markB, covA, covB, nwit_cap, HH=np.zeros(1, np.uint64), minsep=0):
    G = len(gs) - 1
    for g in range(G):
        a, b = gs[g], gs[g + 1]
        stamp = g + 1
        ca = 0
        cb = 0
        for k in range(a, b):
            if need[D[k]] == 0:
                if markA[I[k]] != stamp:
                    markA[I[k]] = stamp; ca += span[I[k]]
                if markB[J[k]] != stamp:
                    markB[J[k]] = stamp; cb += span[J[k]]
        lo = a
        hi = a
        wi = np.empty(nwit_cap, np.int64)
        wj = np.empty(nwit_cap, np.int64)
        for k in range(a, b):
            o = OFF[k]
            while o - OFF[lo] > tol:
                lo += 1
            while hi < b and OFF[hi] - o <= tol:
                hi += 1
            nd = need[D[k]]
            if nd == 0 or nd >= 1000:
                continue
            i = I[k]; j = J[k]
            if markA[i] == stamp and markB[j] == stamp:
                continue
            c = 0
            for q in range(lo, hi):
                if need[D[q]] >= 1000:
                    continue
                iq = I[q]; jq = J[q]
                if iq == i or jq == j:
                    continue
                if minsep > 0:
                    x = HH[iq] ^ HH[i]
                    y = HH[jq] ^ HH[j]
                    cx = 0
                    cy = 0
                    while x:
                        x &= x - np.uint64(1)
                        cx += 1
                    while y:
                        y &= y - np.uint64(1)
                        cy += 1
                    if cx < minsep or cy < minsep:
                        continue
                ok = True
                for w in range(c):
                    if wi[w] == iq or wj[w] == jq:
                        ok = False
                        break
                if ok:
                    wi[c] = iq; wj[c] = jq; c += 1
                    if c >= nd:
                        break
            if c >= nd:
                if markA[i] != stamp:
                    markA[i] = stamp; ca += span[i]
                if markB[j] != stamp:
                    markB[j] = stamp; cb += span[j]
        covA[g] = ca
        covB[g] = cb


def apply_rule(D_, need, tol=500, sel=None):
    """Coverage (ms) per oriented file pair."""
    n = len(D_.H)
    markA = np.zeros(n, np.int64); markB = np.zeros(n, np.int64)
    span = D_.T1 - D_.T0
    G = len(D_.gs) - 1
    covA = np.zeros(G, np.int64); covB = np.zeros(G, np.int64)
    _apply(D_.gs, D_.i, D_.j, D_.d, D_.off, span, need, tol, markA, markB, covA, covB, 128)
    return covA, covB


def decide(D_, covA, covB, gate=0.20):
    """Pick the orientation per pair (better max coverage, plain on ties), gate."""
    nf = D_.nf
    k = D_.gkey
    m = k % 2
    fa = (k // 2) // nf
    fb = (k // 2) % nf
    pa = np.minimum(covA / D_.total[fa], 1.0)
    pb = np.minimum(covB / D_.total[fb], 1.0)
    mx = np.maximum(pa, pb)
    best = {}
    for g in np.flatnonzero(mx > 0):
        p = (int(fa[g]), int(fb[g]))
        v = (mx[g], -m[g])
        if p not in best or v > best[p][0]:
            best[p] = (v, g)
    out = {}
    for p, (v, g) in best.items():
        if v[0] >= gate:
            out[p] = (pa[g], pb[g], int(m[g]))
    return out


def write_report(D_, pred, path):
    by = {}
    for (a, b) in pred:
        by.setdefault(a, []).append(b)
    res = [{"files": [{"full_path": os.path.join("/home/daniel/Documents/VIDS", str(D_.paths[a])),
                       "matches": [{"full_path": os.path.join("/home/daniel/Documents/VIDS", str(D_.paths[b]))} for b in bs]}]}
           for a, bs in by.items()]
    json.dump({"results": res}, open(path, "w"))


def score_pred(D_, pred, name="lab", gate=0.20, quiet=True, keep=None):
    import score as S, io, contextlib
    path = os.path.join(LAB, name + ".json")
    write_report(D_, pred, path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = S.score(path, gate, keep=keep)
    if not quiet:
        print(buf.getvalue())
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("-d", type=int, default=4)
    ap.add_argument("-p", type=float, default=20)
    a = ap.parse_args()
    if a.cmd == "pairs":
        run_pairs()
    else:
        D_ = Data()
        need = schedule(a.d, a.d + 6)
        t = time.time()
        ca, cb = apply_rule(D_, need)
        pred = decide(D_, ca, cb, a.p / 100)
        print(f"rule {time.time() - t:.1f} s, {len(pred)} pairs")
        score_pred(D_, pred, quiet=False)
