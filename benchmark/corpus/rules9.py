"""Kernel v9: v6 + aligned rescue. After the ordinary rule, accepted matches
are chained by offset (consecutive offsets within the pair's window); inside a
chain's A-time and B-time envelope, any record within the window of the chain's
offsets whose RESCUE need is satisfied (0 = take it) is marked too."""
import numpy as np
import numba as nb

NEVER = 1000


@nb.njit(cache=True)
def _run(gs, I, J, OFF, NEED, RESC, span, T0, tols, gor, fstart, gq, gh,
         markA, markB, out, accI, accJ, accO, minchain, fill=0, fs=np.zeros(1, np.int64)):
    wi = np.empty(256, np.int64)
    wj = np.empty(256, np.int64)
    for g in range(len(gs) - 1):
        a, b = gs[g], gs[g + 1]
        tol = tols[gor[g]]
        na = 0
        for k in range(a, b):
            if NEED[k] == 0:
                markA[I[k]] = 1; markB[J[k]] = 1
                accI[na] = I[k]; accJ[na] = J[k]; accO[na] = OFF[k]; na += 1
        lo = a; hi = a
        for k in range(a, b):
            o = OFF[k]
            while o - OFF[lo] > tol:
                lo += 1
            while hi < b and OFF[hi] - o <= tol:
                hi += 1
            nd = NEED[k]
            if nd == 0 or nd >= NEVER:
                continue
            i = I[k]; j = J[k]
            if markA[i] == 1 and markB[j] == 1:
                continue
            c = 0
            for q in range(lo, hi):
                if NEED[q] >= NEVER:
                    continue
                iq = I[q]; jq = J[q]
                if iq == i or jq == j:
                    continue
                ok = True
                for w in range(c):
                    if wi[w] == iq or wj[w] == jq:
                        ok = False
                        break
                if ok:
                    wi[c] = iq; wj[c] = jq; c += 1
                    if c >= nd:
                        break
            if c >= nd:
                markA[i] = 1; markB[j] = 1
                accI[na] = i; accJ[na] = j; accO[na] = o; na += 1
        if minchain > 0 and na >= minchain:
            order = np.argsort(accO[:na])
            s = 0
            while s < na:
                e = s + 1
                while e < na and accO[order[e]] - accO[order[e - 1]] <= tol:
                    e += 1
                if e - s >= minchain:
                    olo = accO[order[s]] - tol; ohi = accO[order[e - 1]] + tol
                    alo = 1 << 62; ahi = -1; blo = 1 << 62; bhi = -1
                    nda = 0
                    for q in range(s, e):
                        x = order[q]
                        ta = T0[accI[x]]; tb = T0[accJ[x]]
                        alo = min(alo, ta); ahi = max(ahi, ta); blo = min(blo, tb); bhi = max(bhi, tb)
                    if ahi > alo and fill:
                        qa = gq[g]; qb = gh[g]
                        for x in range(fstart[qa], fstart[qa + 1]):
                            if T0[x] >= alo and T0[x] <= ahi:
                                markA[x] = 1
                        for x in range(fstart[qb], fstart[qb + 1]):
                            if T0[x] >= blo and T0[x] <= bhi:
                                markB[x] = 1
                    if ahi > alo:
                        for k in range(a, b):
                            if RESC[k] != 0:
                                continue
                            if OFF[k] < olo or OFF[k] > ohi:
                                continue
                            ta = T0[I[k]]; tb = T0[J[k]]
                            if ta >= alo and ta <= ahi and tb >= blo and tb <= bhi:
                                markA[I[k]] = 1; markB[J[k]] = 1
                s = e
        ca = 0; cb = 0
        for x in range(fstart[gq[g]], fstart[gq[g] + 1]):
            if markA[x] == 1:
                ca += span[x]; markA[x] = 0
        for x in range(fstart[gh[g]], fstart[gh[g] + 1]):
            if markB[x] == 1:
                cb += span[x]; markB[x] = 0
        out[g, 0] = ca; out[g, 1] = cb


def features(D, need, resc, tols, minchain=2, fill=0):
    n = len(D.H); m = len(D.i) + 1; G = len(D.gs) - 1
    out = np.zeros((G, 6)); fstart = np.searchsorted(D.F, np.arange(D.nf + 1))
    _run(D.gs, D.i, D.j, D.off, need, resc, D.T1 - D.T0, D.T0, tols, np.arange(G), fstart, D.gq, D.gh,
         np.zeros(n, np.int64), np.zeros(n, np.int64), out,
         np.zeros(m, np.int64), np.zeros(m, np.int64), np.zeros(m, np.int64), minchain, fill)
    return out
