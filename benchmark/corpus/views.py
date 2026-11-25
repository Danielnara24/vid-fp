"""Extra hashes per sample, taken from sub-rectangles of the same 64x64 keyframe.

    .venv/bin/python views.py     -> out/lab/views.npz  (V_<name>, Vm_<name>, Vv_<name>)

Each view array is aligned with fps.npz's samples (same dedup/time order, same
featureless drop on the PRIMARY hash), so spans and times are shared.
"""
import os, json
import numpy as np
import lab2 as L
import fps as P

OUT = os.path.join(L.HERE, "out", "lab")


def sub(rect, fy0, fy1, fx0, fx1):
    """Sub-rectangle of each crop rect, by fractions; integer, min 16 a side."""
    y1, y2, x1, x2 = rect[:, 0], rect[:, 1], rect[:, 2], rect[:, 3]
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    ny1 = y1 + np.round(h * fy0).astype(int)
    ny2 = y1 + np.round(h * fy1).astype(int) - 1
    nx1 = x1 + np.round(w * fx0).astype(int)
    nx2 = x1 + np.round(w * fx1).astype(int) - 1
    out = np.stack([ny1, ny2, nx1, nx2], 1)
    ok = (ny2 - ny1 + 1 >= 8) & (nx2 - nx1 + 1 >= 8)
    return out, ok


def zoom(z, dx=0.0, dy=0.0):
    m = (1 - 1 / z) / 2

    def f(rect, dims):
        return sub(rect, m + dy, 1 - m + dy, m + dx, 1 - m + dx)
    return f


def band(aspect=16 / 9):
    """Portrait frame holding a landscape picture in its middle rows."""
    def f(rect, dims):
        w, h = dims
        if not (h > w and w > 0):
            return rect, np.zeros(len(rect), bool)
        fr = (w / h) / aspect
        return sub(rect, (1 - fr) / 2, (1 + fr) / 2, 0, 1)
    return f


def strip(shift=0.0, aspect=9 / 16):
    """Landscape frame: the middle columns a vertical reframe would keep."""
    def f(rect, dims):
        w, h = dims
        if not (w > h and h > 0):
            return rect, np.zeros(len(rect), bool)
        fr = aspect / (w / h)
        slack = (1 - fr) / 2
        c = 0.5 + shift * slack
        return sub(rect, 0, 1, c - fr / 2, c + fr / 2)
    return f


VIEWS = {
    "z112": zoom(1.12),
    "z125": zoom(1.25),
    "band": band(),
    "stripC": strip(0.0),
    "stripL": strip(-0.4),
    "stripR": strip(0.4),
}


def hashes_of(fr, rect, ok):
    c = L.coeffs(L.thumbs(fr, np.where(ok[:, None], rect, np.array([0, 63, 0, 63]))))
    h = L.pack(L.hash_bits(c))
    m = L.pack(L.hash_bits(L.mirror(c)))
    valid = ok & (L.energy(c) >= L.MIN_AC_ENERGY)
    return h, m, valid


def build(names=None, out=os.path.join(OUT, "views.npz")):
    names = names or list(VIEWS)
    z = np.load(os.path.join(OUT, "fps.npz"))
    paths = [str(p) for p in z["paths"]]
    dims = json.load(open(os.path.join(OUT, "dims.json")))
    idx = {os.path.relpath(r["path"], L.VIDS): r for r in L.load_index()}
    acc = {n: ([], [], []) for n in names}
    Hcheck = []
    for fi, p in enumerate(paths):
        rec = idx[p]
        fr = L.frames_of(rec)
        rect = L.crops(fr)
        c = L.coeffs(L.thumbs(fr, rect))
        h = L.pack(L.hash_bits(c))
        m = L.pack(L.hash_bits(L.mirror(c)))
        flat = L.energy(c) < L.MIN_AC_ENERGY
        extra = []
        for n in names:
            r2, ok = VIEWS[n](rect, dims[fi])
            extra += list(hashes_of(fr, r2, ok))
        t0, t1, hh, mm, ex = P.assemble(rec, h, m, flat, extra)
        Hcheck.append(hh)
        for k, n in enumerate(names):
            acc[n][0].append(ex[3 * k]); acc[n][1].append(ex[3 * k + 1]); acc[n][2].append(ex[3 * k + 2])
        if fi % 200 == 0:
            print(fi, flush=True)
    assert (np.concatenate(Hcheck) == z["H"]).all()
    arrs = {}
    for n in names:
        arrs["V_" + n] = np.concatenate(acc[n][0])
        arrs["Vm_" + n] = np.concatenate(acc[n][1])
        arrs["Vv_" + n] = np.concatenate(acc[n][2])
    np.savez(out, **arrs)


if __name__ == "__main__":
    build()
