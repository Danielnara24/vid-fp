"""Kernel v6: a per-record witness need (any function of d1, d2, orientation),
plus optional gap fill. Coverage per oriented pair."""
import numpy as np
import numba as nb
import rules3 as R3, rules4 as R4

NEVER = 1000


@nb.njit(cache=True)
def _run(gs, I, J, OFF, NEED, span, T0, tols, gor, fill_gap, fill_tol, fstart, gq, gh,
         markA, markB, out, accI, accJ, accO):
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
        if fill_gap > 0 and na > 1:
            order = np.argsort(accO[:na])
            s = 0
            while s < na:
                e = s + 1
                while e < na and accO[order[e]] - accO[order[e - 1]] <= fill_tol:
                    e += 1
                if e - s >= 2:
                    ii = np.sort(accI[order[s:e]])
                    jj = np.sort(accJ[order[s:e]])
                    for q in range(len(ii) - 1):
                        if ii[q + 1] != ii[q] and T0[ii[q + 1]] - T0[ii[q]] <= fill_gap:
                            for x in range(ii[q], ii[q + 1] + 1):
                                markA[x] = 1
                    for q in range(len(jj) - 1):
                        if jj[q + 1] != jj[q] and T0[jj[q + 1]] - T0[jj[q]] <= fill_gap:
                            for x in range(jj[q], jj[q + 1] + 1):
                                markB[x] = 1
                s = e
        ca = 0; cb = 0; n1 = 0; n2 = 0
        for x in range(fstart[gq[g]], fstart[gq[g] + 1]):
            if markA[x] == 1:
                ca += span[x]; n1 += 1; markA[x] = 0
        for x in range(fstart[gh[g]], fstart[gh[g] + 1]):
            if markB[x] == 1:
                cb += span[x]; n2 += 1; markB[x] = 0
        out[g, 0] = ca; out[g, 1] = cb; out[g, 2] = n1; out[g, 3] = n2


def features(D, NEED, tol=2000, per_o_tol=None, fill_gap=0, fill_tol=1000):
    n = len(D.H)
    G = len(D.gs) - 1
    out = np.zeros((G, 6))
    tols = np.full(D.no, tol, np.int64)
    for o, t in (per_o_tol or {}).items():
        tols[o] = t
    fstart = np.searchsorted(D.F, np.arange(D.nf + 1))
    m = len(D.i) + 1
    _run(D.gs, D.i, D.j, D.off, NEED.astype(np.int64), D.T1 - D.T0, D.T0, tols, D.go, fill_gap, fill_tol,
         fstart, D.gq, D.gh, np.zeros(n, np.int64), np.zeros(n, np.int64), out,
         np.zeros(m, np.int64), np.zeros(m, np.int64), np.zeros(m, np.int64))
    return out


def need_of(d1, d2, rules):
    """rules: list of (witnesses, d1_max, d2_max); a record takes the smallest
    witness count among rules it satisfies, NEVER if none."""
    need = np.full(len(d1), NEVER, np.int64)
    for w, a, b in rules:
        ok = (d1 <= a) & (d2 <= b)
        need = np.where(ok, np.minimum(need, w), need)
    return need
