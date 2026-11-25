"""Build vid-fp fingerprints for the whole corpus from the frame dump.

    .venv/bin/python fps.py            -> out/lab/fps.npz
Variants pass a coefficient function; the default reproduces fingerprint.rs.
"""
import os, sys, json
import numpy as np
import lab2 as L

OUT = os.path.join(L.HERE, "out", "lab")


POLICY = {"min_area": 0.0, "mode": "whole", "min_side": 0.0}


def per_file(rec, fr=None):
    fr = L.frames_of(rec) if fr is None else fr
    rect = L.crops(fr)
    h_ = (rect[:, 1] - rect[:, 0] + 1) / 64.0
    w_ = (rect[:, 3] - rect[:, 2] + 1) / 64.0
    small = (h_ * w_ < POLICY["min_area"]) | (np.minimum(h_, w_) < POLICY["min_side"])
    if POLICY["mode"] == "whole":
        rect = np.where(small[:, None], np.array([0, 63, 0, 63]), rect)
    c = L.coeffs(L.thumbs(fr, rect))
    h = L.pack(L.hash_bits(c))
    m = L.pack(L.hash_bits(L.mirror(c)))
    flat = L.energy(c) < L.MIN_AC_ENERGY
    if POLICY["mode"] == "drop":
        flat = flat | small
    return h, m, flat, c, rect


def assemble(rec, h, m, flat, extra=()):
    """Order by time, run-length dedup on (hash, valid), drop featureless, spans."""
    times = np.array(rec["times"], dtype=np.int64)
    n = min(len(times), len(h))
    times, h, m, flat = times[:n], h[:n], m[:n], flat[:n]
    extra = [e[:n] for e in extra]
    o = np.argsort(times, kind="stable")
    times, h, m, flat = times[o], h[o], m[o], flat[o]
    extra = [e[o] for e in extra]
    keep = np.ones(n, bool)
    keep[1:] = (h[1:] != h[:-1]) | (flat[1:] != flat[:-1])
    times, h, m, flat = times[keep], h[keep], m[keep], flat[keep]
    extra = [e[keep] for e in extra]
    t1 = np.append(times[1:], rec["total_ms"])
    v = ~flat
    return times[v], t1[v], h[v], m[v], [e[v] for e in extra]


def build(out=os.path.join(OUT, "fps.npz"), keep_coeffs=True):
    os.makedirs(OUT, exist_ok=True)
    idx = L.load_index()
    idx.sort(key=lambda r: r["path"])
    H, M, F, T0, T1, C, FI = [], [], [], [], [], [], []
    paths, total, dur = [], [], []
    for fi, rec in enumerate(idx):
        h, m, flat, c, rect = per_file(rec)
        t0, t1, h, m, (c, fidx) = assemble(rec, h, m, flat, (c, np.arange(len(h))))
        FI.append(fidx)
        paths.append(os.path.relpath(rec["path"], L.VIDS))
        total.append(rec["total_ms"])
        dur.append(rec["duration"])
        H.append(h); M.append(m); T0.append(t0); T1.append(t1)
        F.append(np.full(len(h), fi, np.int32))
        if keep_coeffs:
            C.append(c.astype(np.float32))
        if fi % 100 == 0:
            print(fi, len(idx), flush=True)
    np.savez(out, H=np.concatenate(H), M=np.concatenate(M), F=np.concatenate(F),
             T0=np.concatenate(T0), T1=np.concatenate(T1),
             C=np.concatenate(C) if keep_coeffs else np.zeros(0), FI=np.concatenate(FI),
             paths=np.array(paths), total=np.array(total, np.int64), dur=np.array(dur))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        POLICY.update(json.loads(sys.argv[1]))
        build(out=sys.argv[2])
    else:
        build()
