import sys; sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
sys.path.insert(0,'/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark')
import numpy as np, pickle, evalfps as E, score as S

def load(pp='pp16.pkl'):
    recs = pickle.load(open(pp,'rb'))
    keys = pickle.load(open(pp+'.keys','rb'))
    rel = np.array([E.LABELS.get(k,{}).get('relation','UNJUDGED') for k in keys], dtype=object)
    for r in recs:                       # pre-sort by offset once
        o = np.argsort(r['off'], kind='stable')
        for k in ('I','J','D','off','ta','tb'):
            if k in r: r[k]=r[k][o]
    return recs, keys, rel

def cov(r, d, Drelax=None, tau=0, K=2, seeded=True):
    ma = np.zeros(r['na'], bool); mb = np.zeros(r['nb'], bool)
    s = r['D'] <= d
    ma[r['I'][s]] = True; mb[r['J'][s]] = True
    if Drelax is not None and Drelax > d:
        s2 = r['D'] <= Drelax
        I,J,off,DD = r['I'][s2], r['J'][s2], r['off'][s2], r['D'][s2]
        if len(off):
            cut = np.nonzero(np.diff(off) > tau)[0] + 1
            for lo, hi in zip(np.r_[0,cut], np.r_[cut,len(off)]):
                ii, jj = I[lo:hi], J[lo:hi]
                ua, ub = np.unique(ii), np.unique(jj)
                if len(ua) < K or len(ub) < K: continue
                if seeded and not (DD[lo:hi] <= d).any(): continue
                ma[ua] = True; mb[ub] = True
    return (min(r['da'][ma].sum()/r['Ta'],1.0), min(r['db'][mb].sum()/r['Tb'],1.0))

def evaluate(recs, rel, d, Drelax=None, tau=0, K=2, seeded=True, ps=(0.20,)):
    C = np.array([cov(r,d,Drelax,tau,K,seeded) for r in recs])
    sc = E.scorer(rel)
    mx = C.max(axis=1)
    out=[]
    for p in ps:
        e=sc(mx>=p,'exact'); a=sc(mx>=p,'any')
        out.append((p,e,a))
    return out, C
