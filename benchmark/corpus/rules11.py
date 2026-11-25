"""Data like rules7, with some samples re-hashed (patch): their records are
dropped and recomputed by brute force against everything."""
import os
import numpy as np
import lab2 as L
import rules3 as R3
import rules7 as R7

LAB = R3.LAB
pc = R7.pc


def _pc64(a, b):
    return np.bitwise_count(np.bitwise_xor(a, b)).astype(np.int16)


def _recs(Qh, Qidx, Th, Tidx, FQ, FT, maxd, lower_only=False, chunk=64):
    """All (q, t) with popcount(Qh[q] ^ Th[t]) <= maxd, files differ."""
    I, J, Dd = [], [], []
    for s in range(0, len(Qidx), chunk):
        q = Qidx[s:s + chunk]
        d = _pc64(Qh[q][:, None], Th[Tidx][None, :])
        ok = (d <= maxd) & (FQ[q][:, None] != FT[Tidx][None, :])
        if lower_only:
            ok &= FQ[q][:, None] < FT[Tidx][None, :]
        a, b = np.nonzero(ok)
        I.append(q[a]); J.append(Tidx[b]); Dd.append(d[a, b])
    return np.concatenate(I), np.concatenate(J), np.concatenate(Dd)


class Data(R7.Data):
    def __init__(self, views=(), emin=50.0, maxd=18, patch=None):
        z = np.load(os.path.join(LAB, "fps.npz"))
        self.H, self.F = z["H"].copy(), z["F"]
        self.T0, self.T1 = z["T0"].astype(np.int64), z["T1"].astype(np.int64)
        self.paths, self.total = z["paths"], z["total"]
        self.nf = nf = len(self.paths)
        F = self.F
        E = L.energy(z["C"]).astype(np.float64)
        M = z["M"].copy()
        h2z = np.load(os.path.join(LAB, "hash2.npz"))
        H2, M2 = h2z["H2_mid32"].copy(), h2z["M2_mid32"].copy()
        P = np.zeros(len(F), bool)
        if patch is not None:
            k = patch["idx"]; P[k] = True
            self.H[k] = patch["H"]; M[k] = patch["M"]; H2[k] = patch["H2"]; M2[k] = patch["M2"]; E[k] = patch["E"]
        self.Hq = self.H.astype(np.uint64)
        Hu, Mu = self.H.astype(np.uint64), M.astype(np.uint64)
        pk = np.flatnonzero(P); allk = np.arange(len(F))
        self.onames = ["plain", "mirror"]
        parts = []
        r = np.load(os.path.join(LAB, "pairs.npy"))
        r = r[(r["d"] <= maxd) & ~P[r["i"]] & ~P[r["j"]]]
        for mm in (0, 1):
            s = r[r["m"] == mm]
            i, j, d = s["i"], s["j"], s["d"]
            if len(pk):
                Q = Mu if mm else Hu
                a = _recs(Q, pk, Hu, allk, F, F, maxd, lower_only=True)
                # patched sample as the higher file: lower query against it
                b = _recs(Q, allk[~P], Hu, pk, F, F, maxd, lower_only=True)
                i = np.concatenate([i, a[0], b[0]]); j = np.concatenate([j, a[1], b[1]]); d = np.concatenate([d, a[2], b[2]])
            ok = (E[i] >= emin) & (E[j] >= emin)
            i, j, d = i[ok], j[ok], d[ok]
            q2 = M2[i] if mm else H2[i]
            parts.append((i, j, d.astype(np.int8), pc(q2 ^ H2[j]).astype(np.int8), mm))
        o = 2
        for n in views:
            v = dict(np.load(os.path.join(LAB, f"view_{n}.npz")))
            if patch is not None and n in patch.get("views", {}):
                for key, val in patch["views"][n].items():
                    v[key] = v[key].copy(); v[key][patch["idx"]] = val
            p, m = R7.view_recs(n)
            for s, mm, nm in ((p, 0, n), (m, 1, n + "~m")):
                s = s[(s["d"] <= maxd) & ~P[s["i"]] & ~P[s["j"]]]
                i, j, d = s["i"], s["j"], s["d"]
                if len(pk):
                    Q = (v["m1"] if mm else v["h1"]).astype(np.uint64)
                    a = _recs(Q, pk, Hu, allk, F, F, maxd)
                    b = _recs(Q, allk[~P], Hu, pk, F, F, maxd)
                    i = np.concatenate([i, a[0], b[0]]); j = np.concatenate([j, a[1], b[1]]); d = np.concatenate([d, a[2], b[2]])
                vv = v["valid"].astype(bool)
                ok = vv[i] & (v["e"][i] >= emin) & (E[j] >= emin)
                i, j, d = i[ok], j[ok], d[ok]
                q2 = (v["m2"] if mm else v["h2"])[i]
                parts.append((i, j, d.astype(np.int8), pc(q2 ^ H2[j]).astype(np.int8), o))
                self.onames.append(nm)
                o += 1
        self._finish(parts, o)

    def _finish(self, parts, o):
        nf = self.nf
        self.no = o
        I = np.concatenate([p[0] for p in parts]); J = np.concatenate([p[1] for p in parts])
        D1 = np.concatenate([p[2] for p in parts]); D2 = np.concatenate([p[3] for p in parts])
        O = np.concatenate([np.full(len(p[0]), p[4], np.int8) for p in parts])
        off = (self.T0[I] - self.T0[J]).astype(np.int32)
        key = (self.F[I].astype(np.int64) * nf + self.F[J]) * self.no + O
        od = np.lexsort((off, key))
        self.i, self.j = I[od], J[od]
        self.d1, self.d2, self.off, self.O = D1[od], D2[od], off[od], O[od]
        self.key = key[od]
        self.d = self.d1
        st = np.flatnonzero(np.r_[True, self.key[1:] != self.key[:-1]])
        self.gs = np.r_[st, len(self.key)].astype(np.int64)
        self.gkey = self.key[st]
        k = self.gkey
        self.go = k % self.no; self.gq = (k // self.no) // nf; self.gh = (k // self.no) % nf
        print(f"{len(self.i)} matches, {len(st)} oriented pairs, {self.onames}")


def load_patch(name):
    z = np.load(os.path.join(LAB, f"patch_{name}.npz"))
    p = {k: z[k] for k in ("idx", "H", "M", "H2", "M2", "E")}
    p["views"] = {}
    for n in ("z112", "band"):
        p["views"][n] = {k: z[f"v_{n}_{k}"] for k in ("h1", "m1", "h2", "m2", "valid", "e")}
    return p
