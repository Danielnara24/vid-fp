"""Kernel v5: v4 + self-corroboration (a match whose secondary hash agrees counts
as one witness) + gap fill along an offset-consistent chain of accepted matches."""
import numpy as np
import numba as nb
import rules3 as R3, rules4 as R4

OM = R4.OM


@nb.njit(cache=True)
def _apply(gs, I, J, D, OFF, SELF, span, T0, needs, gor, tols, fill_gap, fill_tol,
           markA, markB, out, accI, accJ, accO):
    G = len(gs) - 1
    wi = np.empty(256, np.int64)
    wj = np.empty(256, np.int64)
    for g in range(G):
        a, b = gs[g], gs[g + 1]
        stamp = g + 1
        need = needs[gor[g]]
        tol = tols[gor[g]]
        na = 0
        for k in range(a, b):
            if need[D[k]] == 0:
                markA[I[k]] = stamp; markB[J[k]] = stamp
                accI[na] = I[k]; accJ[na] = J[k]; accO[na] = OFF[k]; na += 1
        lo = a; hi = a
        for k in range(a, b):
            o = OFF[k]
            while o - OFF[lo] > tol:
                lo += 1
            while hi < b and OFF[hi] - o <= tol:
                hi += 1
            nd = need[D[k]]
            if nd == 0 or nd >= 1000:
                continue
            i = I[k]; j = J[k]
            if markA[i] == stamp and markB[j] == stamp:
                continue
            c = 0
            if SELF[k]:
                c = 1
            if c < nd:
                for q in range(lo, hi):
                    if need[D[q]] >= 1000:
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
                markA[i] = stamp; markB[j] = stamp
                accI[na] = i; accJ[na] = j; accO[na] = o; na += 1
        # gap fill: accepted matches sorted by offset; chains of offset within fill_tol
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
                                markA[x] = stamp
                    for q in range(len(jj) - 1):
                        if jj[q + 1] != jj[q] and T0[jj[q + 1]] - T0[jj[q]] <= fill_gap:
                            for x in range(jj[q], jj[q + 1] + 1):
                                markB[x] = stamp
                s = e
        out[g, 2] = na
    # coverage pass (after fills), per group
    return


@nb.njit(cache=True)
def _cover(gs, I, J, span, markA, markB, gq_lo, gq_hi, gh_lo, gh_hi, out, stamp_of):
    pass


def features(D, need, tol=500, per_o=None, self_ok=None, fill_gap=0, fill_tol=1000):
    n = len(D.H)
    G = len(D.gs) - 1
    out = np.zeros((G, 6))
    needs = np.tile(need, (D.no, 1))
    tols = np.full(D.no, tol, np.int64)
    for o, (nd, tl) in (per_o or {}).items():
        needs[o] = nd; tols[o] = tl
    SELF = np.zeros(len(D.i), np.bool_) if self_ok is None else self_ok
    # marks must survive per group for coverage: use per-group stamps, then
    # compute coverage group by group right after. Simplest: run group by group
    # in numba with coverage computed by scanning file ranges.
    fstart = np.searchsorted(D.F, np.arange(D.nf + 1))
    _apply_cov(D.gs, D.i, D.j, D.d, D.off, SELF, D.T1 - D.T0, D.T0, needs, D.go, tols,
               fill_gap, fill_tol, np.zeros(n, np.int64), np.zeros(n, np.int64), out,
               np.zeros(len(D.i) + 1, np.int64), np.zeros(len(D.i) + 1, np.int64), np.zeros(len(D.i) + 1, np.int64),
               fstart, D.gq, D.gh)
    return out


@nb.njit(cache=True)
def _apply_cov(gs, I, J, D, OFF, SELF, span, T0, needs, gor, tols, fill_gap, fill_tol,
               markA, markB, out, accI, accJ, accO, fstart, gq, gh):
    G = len(gs) - 1
    one = np.zeros(2, np.int64)
    for g in range(G):
        _apply(gs[g:g + 2], I, J, D, OFF, SELF, span, T0, needs[:, :], gor[g:g + 1], tols, fill_gap, fill_tol,
               markA, markB, out[g:g + 1], accI, accJ, accO)
        # _apply used stamp = 1 for its single group; read and clear marks over both files
        ca = 0; cb = 0; na = 0; nb_ = 0
        for x in range(fstart[gq[g]], fstart[gq[g] + 1]):
            if markA[x] == 1:
                ca += span[x]; na += 1; markA[x] = 0
        for x in range(fstart[gh[g]], fstart[gh[g] + 1]):
            if markB[x] == 1:
                cb += span[x]; nb_ += 1; markB[x] = 0
        out[g, 0] = ca; out[g, 1] = cb; out[g, 2] = na; out[g, 3] = nb_
