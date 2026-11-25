"""Data with both hashes for every orientation: plain, mirror, and view(A) vs B
for each named view (plain and mirrored). Records carry d1 and d2."""
import os
import numpy as np
import lab2 as L
import rules3 as R3

LAB = R3.LAB


def pc(x):
    return np.bitwise_count(x.astype(np.uint64)).astype(np.int64)


def view_recs(n, cap=18):
    path = os.path.join(LAB, f"rec7_{n}.npz")
    if os.path.exists(path):
        z = np.load(path); return z["p"], z["m"]
    f = np.load(os.path.join(LAB, "fps.npz"))
    v = np.load(os.path.join(LAB, f"view_{n}.npz"))
    p = R3.qvh(v["h1"], f["F"], v["valid"], f["H"], f["F"], cap, False, "q7_" + n)
    m = R3.qvh(v["m1"], f["F"], v["valid"], f["H"], f["F"], cap, False, "q7_" + n)
    np.savez(path, p=p, m=m)
    return p, m


class Data:
    def __init__(self, views=(), emin=50.0, maxd=18):
        z = np.load(os.path.join(LAB, "fps.npz"))
        self.H, self.F = z["H"], z["F"]
        self.Hq = self.H.astype(np.uint64)
        self.T0, self.T1 = z["T0"].astype(np.int64), z["T1"].astype(np.int64)
        self.paths, self.total = z["paths"], z["total"]
        self.nf = nf = len(self.paths)
        E = L.energy(z["C"])
        h2z = np.load(os.path.join(LAB, "hash2.npz"))
        H2, M2 = h2z["H2_mid32"], h2z["M2_mid32"]
        self.onames = ["plain", "mirror"]
        parts = []
        r = np.load(os.path.join(LAB, "pairs.npy"))
        r = r[r["d"] <= maxd]
        for mm in (0, 1):
            s = r[r["m"] == mm]
            i, j = s["i"], s["j"]
            ok = (E[i] >= emin) & (E[j] >= emin)
            i, j = i[ok], j[ok]
            q2 = M2[i] if mm else H2[i]
            parts.append((i, j, s["d"][ok].astype(np.int8), pc(q2 ^ H2[j]).astype(np.int8), mm))
        o = 2
        for n in views:
            v = np.load(os.path.join(LAB, f"view_{n}.npz"))
            p, m = view_recs(n)
            for s, mm, nm in ((p, 0, n), (m, 1, n + "~m")):
                s = s[s["d"] <= maxd]
                i, j = s["i"], s["j"]
                ok = (v["e"][i] >= emin) & (E[j] >= emin)
                i, j = i[ok], j[ok]
                q2 = (v["m2"] if mm else v["h2"])[i]
                parts.append((i, j, s["d"][ok].astype(np.int8), pc(q2 ^ H2[j]).astype(np.int8), o))
                self.onames.append(nm)
                o += 1
        self.no = o
        I = np.concatenate([p[0] for p in parts]); J = np.concatenate([p[1] for p in parts])
        D1 = np.concatenate([p[2] for p in parts]); D2 = np.concatenate([p[3] for p in parts])
        O = np.concatenate([np.full(len(p[0]), p[4], np.int8) for p in parts])
        del parts
        off = (self.T0[I] - self.T0[J]).astype(np.int32)
        key = (self.F[I].astype(np.int64) * nf + self.F[J]) * self.no + O
        od = np.lexsort((off, key))
        del key
        self.i, self.j = I[od], J[od]; del I, J
        self.d1, self.d2, self.off, self.O = D1[od], D2[od], off[od], O[od]
        del D1, D2, off, O
        self.key = (self.F[self.i].astype(np.int64) * nf + self.F[self.j]) * self.no + self.O
        self.d = self.d1
        st = np.flatnonzero(np.r_[True, self.key[1:] != self.key[:-1]])
        self.gs = np.r_[st, len(self.key)].astype(np.int64)
        self.gkey = self.key[st]
        k = self.gkey
        self.go = k % self.no; self.gq = (k // self.no) // nf; self.gh = (k // self.no) % nf
        print(f"{len(self.i)} matches, {len(st)} oriented pairs, {self.onames}")
