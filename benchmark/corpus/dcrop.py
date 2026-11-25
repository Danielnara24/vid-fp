"""Detail crop: peel edge lines with little pixel-to-pixel detail (a blurred
fill, a flat or gradient background), per frame, on all four sides."""
import numpy as np


def line_detail(fr):
    """Per-row and per-column mean absolute neighbour difference, (n,64) each."""
    f = fr.astype(np.float32)
    gx = np.abs(np.diff(f, axis=2))          # (n,64,63)
    gy = np.abs(np.diff(f, axis=1))          # (n,63,64)
    g = np.zeros_like(f)
    g[:, :, 1:] += gx; g[:, :, :-1] += gx
    g[:, 1:, :] += gy; g[:, :-1, :] += gy
    return g.mean(axis=2), g.mean(axis=1)    # rows, cols


def peel(d, tau):
    """Leading count of lines below tau, from each end. d: (n, 64)."""
    low = d < tau
    def lead(x):
        nf = ~x
        return np.where(nf.any(axis=1), nf.argmax(axis=1), x.shape[1])
    return lead(low), lead(low[:, ::-1])


def dcrop(fr, base, tau=2.0, rel=None, minside=16):
    """base: frame_crop rects (n,4). Returns rects inside base peeled by detail,
    and whether it differs from base (a reframing applies)."""
    rows, cols = line_detail(fr)
    n = len(fr)
    out = base.copy()
    for k in range(n):
        y1, y2, x1, x2 = base[k]
        r = rows[k, y1:y2 + 1]; c = cols[k, x1:x2 + 1]
        t = tau if rel is None else rel * np.median(np.r_[r, c])
        lr = r < t; lc = c < t
        def lead(x):
            idx = np.flatnonzero(~x)
            return (idx[0], len(x) - 1 - idx[-1]) if len(idx) else (len(x), 0)
        a, b = lead(lr); cA, cB = lead(lc)
        ny1, ny2, nx1, nx2 = y1 + a, y2 - b, x1 + cA, x2 - cB
        if ny2 - ny1 + 1 >= minside and nx2 - nx1 + 1 >= minside:
            out[k] = (ny1, ny2, nx1, nx2)
    ok = (out != base).any(axis=1)
    return out, ok
