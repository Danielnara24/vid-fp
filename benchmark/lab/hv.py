"""Flexible hash-variant builder + scale-free separation metric."""
import sys
sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
sys.path.insert(0,'/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark')
import numpy as np, hashlab as H, lab, evalfps as E, score as S

MIN_AC = 8.0

def build(thumb=16, keep=8, pre=None, bits_fn=None, energy=MIN_AC, index=None,
          post=None):
    """pre(th, rec) -> th' applied to the thumbnail; bits_fn(c) -> bool array."""
    index = index or H.load_index()
    paths, allw, t0, t1, tot, dur, counts = [], [], [], [], [], [], []
    for rec in index:
        fr = H.frames_of(rec)
        th = H.thumbs(fr, rec, thumb)
        if pre is not None:
            th = pre(th, rec)
        c = H.coeffs(th, keep)
        e = H.ac_energy(c) * (16.0/thumb)**2
        bits = (bits_fn or H.median_bits)(c)
        if post is not None:
            bits = post(bits)
        w = H.pack_u64(bits)
        times = np.array(rec['times'], dtype=np.int64)
        total = rec['total_ms']
        order = np.argsort(times, kind='stable')
        w, valid, times = w[order], (e>=energy)[order], times[order]
        same = np.ones(len(times), dtype=bool)
        if len(times) > 1:
            same[1:] = (w[1:]==w[:-1]).all(axis=1) & (valid[1:]==valid[:-1])
        k = ~same; k[0]=True
        w, valid, times = w[k], valid[k], times[k]
        ends = np.empty_like(times); ends[:-1]=times[1:]; ends[-1]=total
        paths.append(rec['path']); allw.append(w[valid])
        t0.append(times[valid]); t1.append(ends[valid])
        tot.append(total); dur.append(rec['duration']); counts.append(int(valid.sum()))
    starts = np.concatenate([[0], np.cumsum([len(x) for x in allw])]).astype(np.int64)
    return lab.Fps(paths, starts, np.concatenate(allw), np.concatenate(t0),
                   np.concatenate(t1), np.array(tot), np.array(dur), np.array(counts))

def mind_table(f, cap=None):
    """min hamming distance per labeled pair -> {key: mind}"""
    W, starts = f.words, f.starts
    pidx = {p:i for i,p in enumerate(f.paths)}
    want = {}
    for (pa,pb), rec in E.LABELS.items():
        ia, ib = pidx.get(pa), pidx.get(pb)
        if ia is None or ib is None: continue
        want.setdefault(ia, []).append((ib, (pa,pb)))
        want.setdefault(ib, []).append((ia, (pa,pb)))
    out = {}
    for a in range(f.n):
        if a not in want: continue
        sa = f.slice(a); wa = W[sa]
        if wa.shape[0]==0: continue
        d = np.zeros((wa.shape[0], W.shape[0]), dtype=np.int16)
        for k in range(W.shape[1]):
            d += np.bitwise_count(wa[:,k,None]^W[None,:,k]).astype(np.int16)
        seg = np.minimum.reduceat(d, starts[:-1], axis=1).min(axis=0)
        for ib, key in want[a]:
            v = int(seg[ib]) if f.counts[ib]>0 else 999
            out[key] = min(out.get(key, 999), v)
        del d
    return out

def separation(mind, bits=None, name=''):
    rel = {k: E.LABELS[k]['relation'] for k in mind}
    pos = np.array([mind[k] for k in mind if rel[k]=='SAME'])
    neg = np.array([mind[k] for k in mind if rel[k]=='DIFFERENT'])
    clip = np.array([mind[k] for k in mind if rel[k]=='CLIP'])
    # recall at the loosest threshold admitting <= F negatives
    res = {}
    for F in (0,1,3,10):
        ts = sorted(set(neg.tolist()))
        t = -1
        for tt in range(0, 65):
            if (neg<=tt).sum() <= F: t = tt
            else: break
        res[F] = ((pos<=t).mean() if t>=0 else 0.0, t)
    print(f"{name:<26} POS med={np.median(pos):4.0f} p75={np.percentile(pos,75):4.0f}  "
          f"NEG med={np.median(neg):4.0f} p05={np.percentile(neg,5):4.0f}  " +
          "  ".join(f"R@{F}fp={100*res[F][0]:5.1f}(t{res[F][1]})" for F in (0,1,3,10)))
    return dict(pos=pos, neg=neg, clip=clip, res=res)
