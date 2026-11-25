"""Secondary hashes per sample, aligned with fps.npz, for verifying matches."""
import os, sys, json
import numpy as np
import lab2 as L, fps as P

OUT = os.path.join(L.HERE, "out", "lab")


def mid_mask(n=16, lo=8, step=3):
    sel = [(k, l) for k in range(n) for l in range(n) if not (k < lo and l < lo)]
    sel = sel[::step][:64]
    return sel


VARIANTS = {
    # name: (thumb, keep, selection)
    "mid32": (32, 16, mid_mask(16, 8, 3)),
    "mid24": (24, 12, [(k, l) for k in range(12) for l in range(12) if not (k < 8 and l < 8)][:64]),
    "low16b": (32, 8, [(k, l) for k in range(8) for l in range(8)]),
}


def h2(fr, rect, thumb, keep, sel):
    th = L.thumbs(fr, rect, thumb)
    b = L.basis(thumb, keep)
    p = np.einsum("ky,nyx->nkx", b, th)
    c = np.einsum("lx,nkx->nkl", b, p)
    idx = np.array([k * keep + l for k, l in sel])
    cc = c.reshape(len(th), -1)[:, idx]
    odd = np.array([l % 2 == 1 for k, l in sel])
    h = L.pack(L.hash_bits(cc))
    m = cc.copy(); m[:, odd] *= -1
    return h, L.pack(L.hash_bits(m))


def build(names=tuple(VARIANTS)):
    z = np.load(os.path.join(OUT, "fps.npz"))
    paths = [str(p) for p in z["paths"]]
    idx = {os.path.relpath(r["path"], L.VIDS): r for r in L.load_index()}
    acc = {n: ([], []) for n in names}
    for fi, p in enumerate(paths):
        rec = idx[p]
        fr = L.frames_of(rec)
        h, m, flat, c, rect = P.per_file(rec, fr)
        extra = []
        for n in names:
            t, k, sel = VARIANTS[n]
            extra += list(h2(fr, rect, t, k, sel))
        _, _, _, _, ex = P.assemble(rec, h, m, flat, extra)
        for q, n in enumerate(names):
            acc[n][0].append(ex[2 * q]); acc[n][1].append(ex[2 * q + 1])
    np.savez(os.path.join(OUT, "hash2.npz"), **{f"H2_{n}": np.concatenate(acc[n][0]) for n in names},
             **{f"M2_{n}": np.concatenate(acc[n][1]) for n in names})


if __name__ == "__main__":
    build()
