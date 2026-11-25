"""Rule lab with extra orientations (views). Orientation 0 plain, 1 mirror,
then per view: view(A) vs B, and mirrored view(A) vs B (A any file, B any other).
"""
import os, glob, subprocess, time
import numpy as np
import rules2 as R2

LAB = R2.LAB


def qvh(Q, FQ, V, H, FH, cap, half, name):
    inp = os.path.join(LAB, name + ".in")
    with open(inp, "wb") as f:
        np.array([len(Q), len(H)], np.int64).tofile(f)
        Q.astype(np.uint64).tofile(f); FQ.astype(np.int32).tofile(f); V.astype(np.uint8).tofile(f)
        H.astype(np.uint64).tofile(f); FH.astype(np.int32).tofile(f)
    out = os.path.join(LAB, name + ".bin")
    for p in glob.glob(out + ".*"):
        os.remove(p)
    subprocess.run([os.path.join(LAB, "qvh"), inp, out, str(cap), str(int(half))], check=True)
    rec = np.dtype([("i", "<i4"), ("j", "<i4"), ("d", "u1")])
    r = np.concatenate([np.fromfile(p, dtype=rec) for p in sorted(glob.glob(out + ".*"))])
    for p in glob.glob(out + ".*"):
        os.remove(p)
    os.remove(inp)
    return r


def view_records(view, cap=16):
    """Cached records for one view (plain and mirrored)."""
    path = os.path.join(LAB, f"rec_{view}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["p"], z["m"]
    f = np.load(os.path.join(LAB, "fps.npz"))
    v = np.load(os.path.join(LAB, "views.npz"))
    p = qvh(v["V_" + view], f["F"], v["Vv_" + view], f["H"], f["F"], cap, False, "q_" + view)
    m = qvh(v["Vm_" + view], f["F"], v["Vv_" + view], f["H"], f["F"], cap, False, "q_" + view)
    np.savez(path, p=p, m=m)
    return p, m


class Data:
    def __init__(self, views=(), maxd=16, base_mirror=True, emin=0.0):
        z = np.load(os.path.join(LAB, "fps.npz"))
        self.H, self.F = z["H"], z["F"]
        self.Hq = self.H.astype(np.uint64)
        self.T0, self.T1 = z["T0"].astype(np.int64), z["T1"].astype(np.int64)
        self.paths, self.total = z["paths"], z["total"]
        self.nf = nf = len(self.paths)
        self.onames = ["plain", "mirror"]
        parts = []
        r = np.load(os.path.join(LAB, "pairs.npy"))
        r = r[r["d"] <= maxd]
        for mm in (0, 1) if base_mirror else (0,):
            s = r[r["m"] == mm]
            parts.append((s["i"], s["j"], s["d"], mm))
        o = 2
        for v in views:
            p, m = view_records(v)
            for s, nm in ((p, v), (m, v + "~m")):
                s = s[s["d"] <= maxd]
                parts.append((s["i"], s["j"], s["d"], o))
                self.onames.append(nm)
                o += 1
        self.no = o
        if emin > 0:
            import lab2 as L
            E = L.energy(z["C"])
            low = E < emin
            parts = [(p[0][~low[p[0]] & ~low[p[1]]], p[1][~low[p[0]] & ~low[p[1]]], p[2][~low[p[0]] & ~low[p[1]]], p[3]) for p in parts]
        I = np.concatenate([p[0] for p in parts]).astype(np.int64)
        J = np.concatenate([p[1] for p in parts]).astype(np.int64)
        Dd = np.concatenate([p[2] for p in parts]).astype(np.int64)
        O = np.concatenate([np.full(len(p[0]), p[3], np.int64) for p in parts])
        self.O = O
        off = self.T0[I] - self.T0[J]
        key = (self.F[I].astype(np.int64) * nf + self.F[J]) * self.no + O
        od = np.lexsort((off, key))
        self.i, self.j, self.d, self.off, self.key = I[od], J[od], Dd[od], off[od], key[od]
        self.O = O[od]
        self.d1 = self.d.copy()
        st = np.flatnonzero(np.r_[True, self.key[1:] != self.key[:-1]])
        self.gs = np.r_[st, len(self.key)].astype(np.int64)
        self.gkey = self.key[st]
        k = self.gkey
        self.go = k % self.no
        self.gq = (k // self.no) // nf
        self.gh = (k // self.no) % nf
        print(f"{len(I)} matches, {len(st)} oriented pairs, orientations {self.onames}")


def apply_rule(D, need, tol=500, cap=128, minsep=0):
    n = len(D.H)
    markA = np.zeros(n, np.int64); markB = np.zeros(n, np.int64)
    span = D.T1 - D.T0
    G = len(D.gs) - 1
    ca = np.zeros(G, np.int64); cb = np.zeros(G, np.int64)
    R2._apply(D.gs, D.i, D.j, D.d, D.off, span, need, tol, markA, markB, ca, cb, cap, D.Hq, minsep)
    return ca, cb


def decide(D, ca, cb, gate=0.20, allowed=None, per_o_need=None):
    pq = np.minimum(ca / D.total[D.gq], 1.0)
    ph = np.minimum(cb / D.total[D.gh], 1.0)
    mx = np.maximum(pq, ph)
    lo = np.minimum(D.gq, D.gh); hi = np.maximum(D.gq, D.gh)
    ok = mx > 0
    if allowed is not None:
        ok &= np.isin(D.go, allowed)
    g = np.flatnonzero(ok)
    # best orientation per unordered pair: sort by pair then -mx then orientation
    o = np.lexsort((D.go[g], -mx[g], hi[g], lo[g]))
    g = g[o]
    first = np.r_[True, (lo[g][1:] != lo[g][:-1]) | (hi[g][1:] != hi[g][:-1])]
    g = g[first]
    g = g[mx[g] >= gate]
    return {(int(lo[x]), int(hi[x])): (float(mx[x]), int(D.go[x])) for x in g}


def halves():
    """Seeds split in two by a hash of the name; a file belongs to a half when
    every seed its pictures come from is in it (compilations across halves drop)."""
    import score as S, zlib
    man = S.load_manifest()
    def half(h):
        def keep(f):
            row = man.get(f)
            if row is None:
                return False
            seeds = {seg.split("@")[0] for seg in row["segments"].split(";")}
            return all(zlib.crc32(sd.encode()) % 2 == h for sd in seeds)
        return keep
    return half(0), half(1)


def score(D, pred, name="lab3", quiet=True, keep=None):
    return R2.score_pred(D, {k: None for k in pred}, name=name, quiet=quiet, keep=keep)


_MAN = None


def brief(r, label=""):
    global _MAN
    import score as S
    if _MAN is None:
        _MAN = S.load_manifest()
    lyr = sum(1 for _, a, b in r["fp_rows"] if {_MAN[a]["seed"], _MAN[b]["seed"]} == {"lyric_1", "lyric_2"})
    fpx = r["fp"] - lyr
    r["fpx"] = fpx
    r["px"] = r["tp"] / (r["tp"] + fpx) if r["tp"] + fpx else 0
    print(f"{label:40} TP {r['tp']:5} FP {r['fp']:3} P {r['precision']:.4f} R {r['recall']:.4f}  "
          f"[FPx {fpx:3} Px {r['px']:.4f}] seedR {sum(v[0] for v in r['per_transform'].values())}/{sum(v[1] for v in r['per_transform'].values())}  scene {r['scene']}")


def mk(s, l1, l2=0, l3=0, l4=0):
    """Custom schedule: strict to s, then 1 witness to l1, 2 to l2, 3 to l3, 4 to l4."""
    n = np.full(65, 10 ** 6, np.int64)
    n[:s + 1] = 0
    for lim, w in ((l1, 1), (l2, 2), (l3, 3), (l4, 4)):
        for d in range(65):
            if s < d <= lim and n[d] >= 10 ** 6:
                n[d] = w
    return n


def popcount64(x):
    return np.bitwise_count(x.astype(np.uint64)).astype(np.int64)


def with_hash2(D, name, mode="avg", w=0.5):
    """Replace D.d with a distance that also reads the secondary hash. Only
    plain (0) and mirror (1) orientations are given one; others keep d1."""
    z = np.load(os.path.join(LAB, "hash2.npz"))
    H2, M2 = z["H2_" + name], z["M2_" + name]
    q = np.where(D.O == 1, M2[D.i], H2[D.i])
    d2 = popcount64(q ^ H2[D.j])
    d2 = np.where(D.O <= 1, d2, D.d1)
    D.d2 = d2
    if mode == "avg":
        D.d = np.rint((1 - w) * D.d1 + w * d2).astype(np.int64)
    return D
