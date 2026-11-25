"""Precompute, per candidate pair, every sample-pair within 24 bits."""
import sys, time, pickle; sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
sys.path.insert(0,'/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark')
import numpy as np, hv, align, score as S

def prep(f, pairs, cap=24, out='pp16.pkl'):
    recs=[]
    for (a,b) in pairs:
        sa, sb = f.slice(a), f.slice(b)
        wa, wb = f.words[sa], f.words[sb]
        dist = np.zeros((len(wa), len(wb)), dtype=np.int16)
        for k in range(f.words.shape[1]):
            dist += np.bitwise_count(wa[:,k,None]^wb[None,:,k]).astype(np.int16)
        I,J = np.nonzero(dist<=cap)
        ta=f.t0[sa].astype(np.int64); tb=f.t0[sb].astype(np.int64)
        recs.append(dict(a=a,b=b,I=I.astype(np.int32),J=J.astype(np.int32),
                         D=dist[I,J].astype(np.int8),
                         off=(ta[I]-tb[J]).astype(np.int32),
                         na=len(wa), nb=len(wb),
                         da=(f.t1[sa]-f.t0[sa]).astype(np.float64),
                         db=(f.t1[sb]-f.t0[sb]).astype(np.float64),
                         Ta=max(int(f.total_ms[a]),1), Tb=max(int(f.total_ms[b]),1),
                         mind=int(dist.min())))
        del dist
    pickle.dump(recs, open(out,'wb'))
    return recs

if __name__=='__main__':
    thumb=int(sys.argv[1]) if len(sys.argv)>1 else 16
    out=sys.argv[2] if len(sys.argv)>2 else 'pp16.pkl'
    f = hv.build(thumb=thumb, keep=8)
    t=time.time(); pairs,_ = align.candidates(f,16); print('cands',len(pairs),round(time.time()-t,1))
    t=time.time(); recs=prep(f,pairs,out=out); print('prep',round(time.time()-t,1))
    pickle.dump([S.pair_key(f.paths[a],f.paths[b]) for a,b in pairs], open(out+'.keys','wb'))
