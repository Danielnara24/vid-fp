"""Lab for the VIDS corpus: dumped keyframes -> vid-fp fingerprints in numpy.

The frames come from a build with the VIDFP_DUMP hook (branch exp/match-lab):
out/frames/index.jsonl plus <key>.raw (n x 64 x 64 grey, pre-crop, decode order).
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.join(HERE, "out", "frames")
VIDS = "/home/daniel/Documents/VIDS"
THUMB, KEEP = 16, 8
BAR_MAX, BAR_MEAN = 4, 1
MIN_AC_ENERGY = 8.0


def load_index():
    recs = {}
    bad = 0
    for l in open(os.path.join(FR, "index.jsonl")):
        try:
            r = json.loads(l)
        except ValueError:
            bad += 1
            continue
        if os.path.exists(os.path.join(FR, r["key"] + ".raw")):
            recs[r["path"]] = r          # last write wins
    if bad:
        print(f"warning: {bad} unreadable index lines")
    return list(recs.values())


def frames_of(rec):
    a = np.fromfile(os.path.join(FR, rec["key"] + ".raw"), dtype=np.uint8)
    return a.reshape(-1, 64, 64)


def _depth(lines):
    """lines: (n, L, W) uint8, L lines in peel order. Count of leading flat lines."""
    first = lines[:, 0, :].astype(np.int32)
    level = first.sum(axis=1) // first.shape[1]
    d = np.abs(lines.astype(np.int32) - level[:, None, None])
    flat = (d.max(axis=2) <= BAR_MAX) & (d.sum(axis=2) <= BAR_MEAN * lines.shape[2])
    nf = ~flat
    any_nf = nf.any(axis=1)
    return np.where(any_nf, nf.argmax(axis=1), lines.shape[1])


def crops(fr):
    """frame_crop for every frame: (n, 4) of y1, y2, x1, x2 inclusive."""
    n = len(fr)
    top = _depth(fr)
    bot = _depth(fr[:, ::-1, :])
    y1 = top + (top > 0)
    y2 = 63 - (bot + (bot > 0))
    out = np.tile(np.array([0, 63, 0, 63]), (n, 1))
    okr = (y2 >= y1) & (y2 - y1 + 1 >= 16)
    for (a, b) in set(zip(y1[okr].tolist(), y2[okr].tolist())):
        sel = np.where(okr & (y1 == a) & (y2 == b))[0]
        cols = fr[sel, a:b + 1, :].transpose(0, 2, 1)      # (k, 64 cols, h)
        left = _depth(cols)
        right = _depth(cols[:, ::-1, :])
        x1 = left + (left > 0)
        x2 = 63 - (right + (right > 0))
        okc = (x2 >= x1) & (x2 - x1 + 1 >= 16)
        s2 = sel[okc]
        out[s2, 0] = a
        out[s2, 1] = b
        out[s2, 2] = x1[okc]
        out[s2, 3] = x2[okc]
    return out


_bw = {}


def box_weights(offset, length, thumb=THUMB):
    k = (offset, length, thumb)
    if k in _bw:
        return _bw[k]
    W = np.zeros((thumb, 64), dtype=np.float32)
    scale = np.float32(length) / np.float32(thumb)
    for o in range(thumb):
        start = np.float32(o) * scale
        end = start + scale
        first = int(np.floor(start))
        last = max(min(int(np.ceil(end)), length), first + 1)
        tot = np.float32(0)
        for s in range(first, last):
            ov = max(min(end, np.float32(s + 1)) - max(start, np.float32(s)), np.float32(0))
            W[o, offset + s] = ov
            tot += ov
        if tot > 0:
            W[o] /= tot
    _bw[k] = W
    return W


def basis(thumb=THUMB, keep=KEEP):
    k = np.arange(keep)[:, None]
    x = np.arange(thumb)[None, :]
    return np.cos(np.float32(np.pi) * (2 * x + 1) * k / (2 * thumb)).astype(np.float32)


B = basis()


def thumbs(fr, rect, thumb=THUMB):
    out = np.empty((len(fr), thumb, thumb), dtype=np.float32)
    keys = [tuple(r) for r in rect.tolist()]
    for key in set(keys):
        sel = np.array([i for i, k in enumerate(keys) if k == key])
        y1, y2, x1, x2 = key
        R = box_weights(y1, y2 - y1 + 1, thumb)
        C = box_weights(x1, x2 - x1 + 1, thumb)
        f = fr[sel].astype(np.float32)
        band = np.einsum("oy,nyx->nox", R, f)
        out[sel] = np.einsum("px,nox->nop", C, band)
    return out


def coeffs(th, b=B):
    p = np.einsum("ky,nyx->nkx", b, th)
    c = np.einsum("lx,nkx->nkl", b, p).reshape(len(th), -1)
    c[:, 0] = 0
    return c


def hash_bits(c):
    s = np.sort(c, axis=1)
    n = c.shape[1]
    med = (s[:, n // 2 - 1] + s[:, n // 2]) / 2
    return c > med[:, None]


W64 = (np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64))


def pack(bits):
    return (bits.astype(np.uint64) * W64).sum(axis=1, dtype=np.uint64)


ODD = np.array([(l % 2 == 1) for k in range(KEEP) for l in range(KEEP)])


def mirror(c):
    m = c.copy()
    m[:, ODD] *= -1
    return m


def energy(c):
    return np.abs(c).sum(axis=1) / c.shape[1]
