"""Per-pair alignment features: best time-offset cluster and its coverage."""
import sys; sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
sys.path.insert(0,'/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark')
import numpy as np, lab, evalfps as E, score as S

def candidates(f, max_d=16):
    feat = lab.pair_features(f, [max_d], max_d=max_d)
    return list(zip(feat['a'].tolist(), feat['b'].tolist())), feat

def pair_align(f, a, b, D, tau_ms):
    """Best-offset evidence for one pair at distance D, offset window tau.

    Returns (cov_a, cov_b, ns_a, ns_b) for the plain rule and for the
    offset-constrained rule."""
    sa, sb = f.slice(a), f.slice(b)
    wa, wb = f.words[sa], f.words[sb]
    if len(wa)==0 or len(wb)==0: return None
    d = np.zeros((len(wa), len(wb)), dtype=np.int16)
    for k in range(f.words.shape[1]):
        d += np.bitwise_count(wa[:,k,None]^wb[None,:,k]).astype(np.int16)
    ta = f.t0[sa].astype(np.int64); tb = f.t0[sb].astype(np.int64)
    da = (f.t1[sa]-f.t0[sa]).astype(np.float64)
    db = (f.t1[sb]-f.t0[sb]).astype(np.float64)
    I, J = np.nonzero(d <= D)
    if len(I)==0: return None
    off = ta[I]-tb[J]
    o = np.argsort(off, kind='stable'); I,J,off = I[o],J[o],off[o]
    # sliding window over offsets
    best = None
    lo = 0
    for hi in range(len(off)):
        while off[hi]-off[lo] > tau_ms: lo += 1
        ii, jj = I[lo:hi+1], J[lo:hi+1]
        ua = np.unique(ii); ub = np.unique(jj)
        score = da[ua].sum() + db[ub].sum()
        if best is None or score > best[0]:
            best = (score, ua, ub)
    _, ua, ub = best
    ta_tot = max(int(f.total_ms[a]),1); tb_tot = max(int(f.total_ms[b]),1)
    return (min(da[ua].sum()/ta_tot,1.0), min(db[ub].sum()/tb_tot,1.0),
            len(ua), len(ub))

def plain(f, a, b, D):
    sa, sb = f.slice(a), f.slice(b)
    wa, wb = f.words[sa], f.words[sb]
    d = np.zeros((len(wa), len(wb)), dtype=np.int16)
    for k in range(f.words.shape[1]):
        d += np.bitwise_count(wa[:,k,None]^wb[None,:,k]).astype(np.int16)
    ma = (d<=D).any(axis=1); mb = (d<=D).any(axis=0)
    da = (f.t1[sa]-f.t0[sa]).astype(np.float64); db = (f.t1[sb]-f.t0[sb]).astype(np.float64)
    return (min(da[ma].sum()/max(int(f.total_ms[a]),1),1.0),
            min(db[mb].sum()/max(int(f.total_ms[b]),1),1.0),
            int(ma.sum()), int(mb.sum()), int(d.min()))
