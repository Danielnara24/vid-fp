import numpy as np, time, lab, evalfps as E

def hash_df(f, d):
    """For every stored hash: how many distinct files hold a hash within d."""
    W, starts = f.words, f.starts
    df = np.zeros(W.shape[0], dtype=np.int32)
    for a in range(f.n):
        sa = f.slice(a)
        wa = W[sa]
        if wa.shape[0] == 0: continue
        dist = np.zeros((wa.shape[0], W.shape[0]), dtype=np.int16)
        for k in range(W.shape[1]):
            dist += np.bitwise_count(wa[:,k,None] ^ W[None,:,k]).astype(np.int16)
        df += (dist.min(axis=0) <= d).astype(np.int32)
        del dist
    return df

def detail(f, pairs, d, df):
    """Per-pair evidence features at one tolerance."""
    out = []
    W = f.words
    for (a,b) in pairs:
        sa, sb = f.slice(a), f.slice(b)
        wa, wb = W[sa], W[sb]
        dist = np.zeros((wa.shape[0], wb.shape[0]), dtype=np.int16)
        for k in range(W.shape[1]):
            dist += np.bitwise_count(wa[:,k,None] ^ wb[None,:,k]).astype(np.int16)
        ma = dist.min(axis=1) <= d; mb = dist.min(axis=0) <= d
        da, db = f.dur_ms[sa], f.dur_ms[sb]
        ta, tb = max(int(f.total_ms[a]),1), max(int(f.total_ms[b]),1)
        dfa, dfb = df[sa], df[sb]
        def runmax(m):
            best=cur=0
            for x in m:
                cur = cur+1 if x else 0
                best = max(best,cur)
            return best
        r = dict(a=a,b=b, mind=int(dist.min()),
                 cov_a=min(da[ma].sum()/ta,1.0), cov_b=min(db[mb].sum()/tb,1.0),
                 ns_a=int(ma.sum()), ns_b=int(mb.sum()),
                 run_a=runmax(ma), run_b=runmax(mb))
        for T in (2,3,5,10,20):
            ka = ma & (dfa<=T); kb = mb & (dfb<=T)
            r[f'cov{T}_a'] = min(da[ka].sum()/ta,1.0)
            r[f'cov{T}_b'] = min(db[kb].sum()/tb,1.0)
            r[f'ns{T}_a'] = int(ka.sum()); r[f'ns{T}_b'] = int(kb.sum())
        out.append(r)
    return out
