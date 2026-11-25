"""Hash variants, computed from the dumped 64x64 planes.

Reproduces `fingerprint.rs`'s pipeline in numpy (verified bit-for-bit against
the binary's own hashes) so alternatives can be measured without a re-decode.
"""
import json, os, sys
import numpy as np

DIR = '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/out/frames'


def load_index(dir=DIR):
    return [json.loads(l) for l in open(os.path.join(dir, 'index.jsonl'))]


def frames_of(rec, dir=DIR):
    a = np.fromfile(os.path.join(dir, f"f{rec['seq']}.raw"), dtype=np.uint8)
    return a.reshape(-1, 64, 64)


def box_weights(offset, length, thumb):
    """`fingerprint::box_weights` as a (thumb, 64) matrix of taps."""
    W = np.zeros((thumb, 64), dtype=np.float64)
    scale = length / thumb
    for o in range(thumb):
        start = o * scale
        end = start + scale
        first = int(np.floor(start))
        last = max(min(int(np.ceil(end)), length), first + 1)
        tot = 0.0
        for s in range(first, last):
            overlap = max(min(end, s + 1.0) - max(start, s), 0.0)
            W[o, offset + s] = overlap
            tot += overlap
        if tot > 0:
            W[o] /= tot
    return W


def dct_basis(thumb, keep):
    k = np.arange(keep)[:, None]
    x = np.arange(thumb)[None, :]
    return np.cos(np.pi * (2 * x + 1) * k / (2 * thumb))


def thumbs(frames, rec, thumb):
    """Box-filter the cropped region down to thumb x thumb (float32 math)."""
    R = box_weights(rec['y1'], rec['y2'] - rec['y1'] + 1, thumb).astype(np.float32)
    C = box_weights(rec['x1'], rec['x2'] - rec['x1'] + 1, thumb).astype(np.float32)
    f = frames.astype(np.float32)
    band = np.einsum('oy,nyx->nox', R, f)          # rows
    return np.einsum('px,nox->nop', C, band)       # cols


def coeffs(th, keep, drop_dc=True):
    b = dct_basis(th.shape[-1], keep).astype(np.float32)
    partial = np.einsum('ky,nyx->nkx', b, th)
    out = np.einsum('lx,nkx->nkl', b, partial)
    out = out.reshape(len(th), keep * keep)
    if drop_dc:
        out[:, 0] = 0.0
    return out


def median_bits(c):
    """One bit per coefficient, set when it is above the block median."""
    s = np.sort(c, axis=1)
    n = c.shape[1]
    med = (s[:, n // 2 - 1] + s[:, n // 2]) / 2.0
    return c > med[:, None]


def pack_u64(bits):
    """(n, 64k) bools -> (n, k) uint64, MSB first, matching `hash_of`."""
    n, w = bits.shape
    if w % 64:
        bits = np.pad(bits, ((0, 0), (0, 64 - w % 64)))
        return pack_u64(bits)
    out = np.zeros((n, w // 64), dtype=np.uint64)
    for i in range(w):
        out[:, i // 64] |= (bits[:, i].astype(np.uint64) << np.uint64(63 - i % 64))
    return out


def ac_energy(c):
    return np.abs(c).sum(axis=1) / c.shape[1]
