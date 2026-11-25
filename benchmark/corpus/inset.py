"""Inset detection: the largest axis-aligned rectangle bounded on all four
sides by a sharp step (a picture pasted onto a different background)."""
import numpy as np
import numba as nb

STEP = 12      # grey levels: a step clearly above encoding noise
MINSIDE = 12   # px of 64
COVER = 0.8    # share of a side that must be a step


@nb.njit(cache=True)
def _one(f, out):
    h = np.zeros((65, 64), np.int32)   # h[y][x]: step between rows y-1 and y at column x (y=0,64 frame edge)
    v = np.zeros((64, 65), np.int32)
    for y in range(1, 64):
        for x in range(64):
            h[y, x] = 1 if abs(int(f[y, x]) - int(f[y - 1, x])) >= STEP else 0
    for y in range(64):
        for x in range(1, 64):
            v[y, x] = 1 if abs(int(f[y, x]) - int(f[y, x - 1])) >= STEP else 0
    # prefix sums along each boundary line
    hp = np.zeros((65, 65), np.int32)
    for y in range(65):
        for x in range(64):
            hp[y, x + 1] = hp[y, x] + h[y, x]
    vp = np.zeros((65, 65), np.int32)
    for x in range(65):
        for y in range(64):
            vp[y + 1, x] = vp[y, x] + v[y, x]
    best = 0
    out[:] = -1
    for y1 in range(1, 64):
        for y2 in range(y1 + MINSIDE, 64):
            for x1 in range(1, 64):
                for x2 in range(x1 + MINSIDE, 64):
                    w = x2 - x1; hh = y2 - y1
                    area = w * hh
                    if area <= best or area > 0.7 * 64 * 64:
                        continue
                    if hp[y1, x2] - hp[y1, x1] < COVER * w:
                        continue
                    if hp[y2, x2] - hp[y2, x1] < COVER * w:
                        continue
                    if vp[y2, x1] - vp[y1, x1] < COVER * hh:
                        continue
                    if vp[y2, x2] - vp[y1, x2] < COVER * hh:
                        continue
                    best = area
                    out[0] = y1; out[1] = y2 - 1; out[2] = x1; out[3] = x2 - 1


@nb.njit(cache=True, parallel=True)
def _all(fr, out):
    for k in nb.prange(fr.shape[0]):
        _one(fr[k], out[k])


def insets(fr, base):
    out = np.full((len(fr), 4), -1, np.int64)
    _all(fr, out)
    ok = out[:, 0] >= 0
    rect = np.where(ok[:, None], out, base)
    return rect, ok


@nb.njit(cache=True)
def _fast(f, out, step, minside, cover, maxarea):
    h = np.zeros((65, 64), np.int32)
    v = np.zeros((64, 65), np.int32)
    for y in range(1, 64):
        for x in range(64):
            h[y, x] = 1 if abs(int(f[y, x]) - int(f[y - 1, x])) >= step else 0
    for y in range(64):
        for x in range(1, 64):
            v[y, x] = 1 if abs(int(f[y, x]) - int(f[y, x - 1])) >= step else 0
    hp = np.zeros((65, 65), np.int32)
    for y in range(65):
        for x in range(64):
            hp[y, x + 1] = hp[y, x] + h[y, x]
    vp = np.zeros((65, 65), np.int32)
    for x in range(65):
        for y in range(64):
            vp[y + 1, x] = vp[y, x] + v[y, x]
    best = 0
    out[:] = -1
    cols = np.empty(64, np.int64)
    for y1 in range(1, 64):
        for y2 in range(y1 + minside, 64):
            hh = y2 - y1
            if hh * 63 <= best:
                continue
            nc = 0
            for x in range(1, 64):
                if vp[y2, x] - vp[y1, x] >= cover * hh:
                    cols[nc] = x; nc += 1
            for p in range(nc):
                x1 = cols[p]
                for q in range(nc - 1, p, -1):
                    x2 = cols[q]
                    w = x2 - x1
                    if w < minside:
                        break
                    area = w * hh
                    if area <= best:
                        break
                    if area > maxarea * 64 * 64:
                        continue
                    if hp[y1, x2] - hp[y1, x1] < cover * w or hp[y2, x2] - hp[y2, x1] < cover * w:
                        continue
                    best = area
                    out[0] = y1; out[1] = y2 - 1; out[2] = x1; out[3] = x2 - 1
                    break


@nb.njit(cache=True, parallel=True)
def _fast_all(fr, out, step, minside, cover, maxarea):
    for k in nb.prange(fr.shape[0]):
        _fast(fr[k], out[k], step, minside, cover, maxarea)


def insets_fast(fr, base, step=STEP, minside=MINSIDE, cover=COVER, maxarea=0.7):
    out = np.full((len(fr), 4), -1, np.int64)
    _fast_all(fr, out, step, minside, cover, maxarea)
    ok = out[:, 0] >= 0
    return np.where(ok[:, None], out, base), ok
