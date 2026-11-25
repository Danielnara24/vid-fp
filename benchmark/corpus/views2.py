"""Views with both hashes: for each view, h1/m1 (primary + mirrored), h2/m2
(mid32 + mirrored), valid. Aligned with fps.npz samples."""
import os, sys, json
import numpy as np
import lab2 as L, fps as P, views as V, hash2 as H2

OUT = os.path.join(L.HERE, "out", "lab")
T2 = H2.VARIANTS["mid32"]

EXTRA = {
    "z112": V.zoom(1.12),
    "z125": V.zoom(1.25),
    "z118L": V.zoom(1.18, dx=-0.04), "z118R": V.zoom(1.18, dx=0.04),
    "z118U": V.zoom(1.18, dy=-0.04), "z118D": V.zoom(1.18, dy=0.04),
    "band": V.band(),
    "band43": V.band(4 / 3),
    "strip0": V.strip(0.0), "stripL3": V.strip(-0.3), "stripR3": V.strip(0.3),
    "stripL6": V.strip(-0.6), "stripR6": V.strip(0.6),
}


def one(fr, rect, dims, f):
    r2, ok = f(rect, dims)
    rr = np.where(ok[:, None], r2, np.array([0, 63, 0, 63]))
    c = L.coeffs(L.thumbs(fr, rr))
    h1 = L.pack(L.hash_bits(c)); m1 = L.pack(L.hash_bits(L.mirror(c)))
    valid = ok & (L.energy(c) >= L.MIN_AC_ENERGY)
    h2, m2 = H2.h2(fr, rr, *T2)
    return [h1, m1, h2, m2, valid, L.energy(c)]


PIXEL = {}   # name -> f(frames, rect) -> (rect2, ok)


def build(names):
    z = np.load(os.path.join(OUT, "fps.npz"))
    paths = [str(p) for p in z["paths"]]
    dims = json.load(open(os.path.join(OUT, "dims.json")))
    idx = {os.path.relpath(r["path"], L.VIDS): r for r in L.load_index()}
    acc = {n: [[] for _ in range(6)] for n in names}
    for fi, p in enumerate(paths):
        rec = idx[p]
        fr = L.frames_of(rec)
        h, m, flat, c, rect = P.per_file(rec, fr)
        extra = []
        for n in names:
            if n in PIXEL:
                r2, ok = PIXEL[n](fr, rect)
                extra += one(fr, rect, dims[fi], lambda r, d, r2=r2, ok=ok: (r2, ok))
            else:
                extra += one(fr, rect, dims[fi], EXTRA[n])
        _, _, _, _, ex = P.assemble(rec, h, m, flat, extra)
        for q, n in enumerate(names):
            for t in range(6):
                acc[n][t].append(ex[6 * q + t])
    for n in names:
        a = [np.concatenate(x) for x in acc[n]]
        np.savez(os.path.join(OUT, f"view_{n}.npz"), h1=a[0], m1=a[1], h2=a[2], m2=a[3], valid=a[4], e=a[5])


if __name__ == "__main__":
    build(sys.argv[1:] or list(EXTRA))
