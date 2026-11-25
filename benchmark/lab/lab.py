"""End-to-end: dumped frames -> fingerprints -> coverage -> score.

Mirrors `fingerprint.rs` (thumbnail, DCT, median bits, featureless drop,
run-length dedup, sample clock) and `compare.rs` (per-sample nearest partner,
footage-weighted coverage) so a hash variant can be scored against labels.csv
without touching the binary.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark')
sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
import hashlab as H
import score as S

MIN_AC_ENERGY = 8.0


class Fps:
    """One fingerprint set for the whole corpus."""
    def __init__(self, paths, starts, words, t0, t1, total_ms, duration, counts):
        self.paths, self.starts, self.words = paths, starts, words
        self.t0, self.t1 = t0, t1
        self.dur_ms = t1 - t0
        self.total_ms, self.duration, self.counts = total_ms, duration, counts
        self.n = len(paths)
        self.bits = words.shape[1] * 64

    def slice(self, v):
        return slice(int(self.starts[v]), int(self.starts[v + 1]))


def build(thumb=16, keep=8, index=None, dir=H.DIR, energy_floor=MIN_AC_ENERGY,
          bit_pick=None):
    """Fingerprint every file with the given thumbnail/kept-block sizes.

    `bit_pick` optionally selects which of the keep*keep coefficients become
    bits (a mask), so a variant can keep the DCT but change the band.
    """
    index = index or H.load_index(dir)
    paths, allw, t0, t1, tot, dur, counts = [], [], [], [], [], [], []
    for rec in index:
        fr = H.frames_of(rec, dir)
        th = H.thumbs(fr, rec, thumb)
        c = H.coeffs(th, keep)
        # `MIN_AC_ENERGY` is calibrated against an unnormalised 16-wide DCT, so
        # rescale rather than re-tune when the thumbnail changes size.
        e = H.ac_energy(c) * (16.0 / thumb) ** 2
        bits = H.median_bits(c)
        if bit_pick is not None:
            bits = bits[:, bit_pick]
        w = H.pack_u64(bits)
        valid = e >= energy_floor
        times = np.array(rec['times'], dtype=np.int64)
        total = rec['total_ms']
        order = np.argsort(times, kind='stable')
        w, valid, times = w[order], valid[order], times[order]
        # run-length dedup on (hash, valid)
        same = np.ones(len(times), dtype=bool)
        if len(times) > 1:
            same[1:] = (w[1:] == w[:-1]).all(axis=1) & (valid[1:] == valid[:-1])
        k = ~same
        k[0] = True
        w, valid, times = w[k], valid[k], times[k]
        ends = np.empty_like(times)
        ends[:-1] = times[1:]
        ends[-1] = total
        sel = valid
        paths.append(rec['path'])
        allw.append(w[sel])
        t0.append(times[sel]); t1.append(ends[sel])
        tot.append(total); dur.append(rec['duration']); counts.append(int(sel.sum()))
    starts = np.concatenate([[0], np.cumsum([len(x) for x in allw])]).astype(np.int64)
    return Fps(paths, starts,
               np.concatenate(allw) if allw else np.zeros((0, 1), np.uint64),
               np.concatenate(t0), np.concatenate(t1),
               np.array(tot), np.array(dur), np.array(counts))


def pair_features(f, ds, max_d=None):
    """For every pair with any hash within max(ds): coverage/samples per d.

    One row per candidate pair, one column per tolerance in `ds`. Vectorised
    over the whole library per subject file: `rowmin` is each of A's samples
    against its nearest partner in every other file at once, `colmin` is every
    other sample in the corpus against its nearest partner in A.
    """
    max_d = max_d or max(ds)
    ds = np.asarray(ds)
    W, starts = f.words, f.starts
    nf = f.n
    keep = f.counts > 0
    dur = f.dur_ms.astype(np.float64)
    rows = []
    for a in range(nf):
        sa = f.slice(a)
        wa = W[sa]
        if wa.shape[0] == 0:
            continue
        d = np.zeros((wa.shape[0], W.shape[0]), dtype=np.int16)
        for k in range(W.shape[1]):
            d += np.bitwise_count(wa[:, k, None] ^ W[None, :, k]).astype(np.int16)
        rowmin = np.minimum.reduceat(d, starts[:-1], axis=1)     # (n_a, files)
        colmin = d.min(axis=0)                                   # (total,)
        mind = rowmin.min(axis=0)                                # (files,)
        cand = np.nonzero((mind <= max_d) & keep)[0]
        cand = cand[cand > a]
        if not len(cand):
            del d
            continue
        da = dur[sa]
        ta = max(int(f.total_ms[a]), 1)
        cov_a = np.empty((len(cand), len(ds)))
        cov_b = np.empty((len(cand), len(ds)))
        ns_a = np.empty((len(cand), len(ds)), dtype=np.int32)
        ns_b = np.empty((len(cand), len(ds)), dtype=np.int32)
        ms_a = np.empty((len(cand), len(ds)))
        ms_b = np.empty((len(cand), len(ds)))
        for j, dd in enumerate(ds):
            ma = rowmin[:, cand] <= dd                    # (n_a, cand)
            ms_a[:, j] = da @ ma
            ns_a[:, j] = ma.sum(axis=0)
            mb = colmin <= dd
            seg_ms = np.add.reduceat(dur * mb, starts[:-1])
            seg_ns = np.add.reduceat(mb.astype(np.int32), starts[:-1])
            ms_b[:, j] = seg_ms[cand]
            ns_b[:, j] = seg_ns[cand]
        cov_a = np.minimum(ms_a / ta, 1.0)
        cov_b = np.minimum(ms_b / np.maximum(f.total_ms[cand], 1)[:, None], 1.0)
        rows.append(dict(a=np.full(len(cand), a), b=cand, mind=mind[cand],
                         cov_a=cov_a, cov_b=cov_b, ns_a=ns_a, ns_b=ns_b,
                         ms_a=ms_a, ms_b=ms_b))
        del d, rowmin, colmin
    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    out['ds'] = ds
    return out
