"""Rule kernel with per-pair features: coverage, matched sample counts, evidence.

evA/evB: sum over each side's marked samples of the best chance-improbability
(orders of magnitude, binomial tail) among the matches that marked it.
"""
import numpy as np
import numba as nb
from math import comb, log10
import rules3 as R3

OM = np.array([-log10(sum(comb(64, k) for k in range(x + 1)) / 2 ** 64) for x in range(65)])


@nb.njit(cache=True)
def _popc(x):
    c = 0
    while x:
        x &= x - np.uint64(1)
        c += 1
    return c


@nb.njit(cache=True)
def _apply(gs, I, J, D, OFF, span, needs, gor, tols, om, HH, minsep,
           markA, markB, bestA, bestB, out):
    G = len(gs) - 1
    wi = np.empty(256, np.int64)
    wj = np.empty(256, np.int64)
    for g in range(G):
        a, b = gs[g], gs[g + 1]
        stamp = g + 1
        need = needs[gor[g]]
        tol = tols[gor[g]]
        ca = 0; cb = 0; na = 0; nb_ = 0
        eva = 0.0; evb = 0.0
        # strict
        for k in range(a, b):
            if need[D[k]] == 0:
                i = I[k]; j = J[k]; e = om[D[k]]
                if markA[i] != stamp:
                    markA[i] = stamp; ca += span[i]; na += 1; bestA[i] = e; eva += e
                elif e > bestA[i]:
                    eva += e - bestA[i]; bestA[i] = e
                if markB[j] != stamp:
                    markB[j] = stamp; cb += span[j]; nb_ += 1; bestB[j] = e; evb += e
                elif e > bestB[j]:
                    evb += e - bestB[j]; bestB[j] = e
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
            for q in range(lo, hi):
                if need[D[q]] >= 1000:
                    continue
                iq = I[q]; jq = J[q]
                if iq == i or jq == j:
                    continue
                if minsep > 0:
                    if _popc(HH[iq] ^ HH[i]) < minsep or _popc(HH[jq] ^ HH[j]) < minsep:
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
                e = om[D[k]]
                if markA[i] != stamp:
                    markA[i] = stamp; ca += span[i]; na += 1; bestA[i] = e; eva += e
                if markB[j] != stamp:
                    markB[j] = stamp; cb += span[j]; nb_ += 1; bestB[j] = e; evb += e
        out[g, 0] = ca; out[g, 1] = cb; out[g, 2] = na; out[g, 3] = nb_
        out[g, 4] = eva; out[g, 5] = evb


def features(D, need, tol=500, minsep=0, per_o=None):
    """per_o: {orientation index: (need, tol)} overrides."""
    n = len(D.H)
    G = len(D.gs) - 1
    out = np.zeros((G, 6))
    needs = np.tile(need, (D.no, 1))
    tols = np.full(D.no, tol, np.int64)
    for o, (nd, tl) in (per_o or {}).items():
        needs[o] = nd; tols[o] = tl
    _apply(D.gs, D.i, D.j, D.d, D.off, D.T1 - D.T0, needs, D.go, tols, OM, D.Hq, minsep,
           np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n), np.zeros(n), out)
    return out


def counts(D):
    """Samples per file."""
    return np.bincount(D.F, minlength=D.nf)


def decide(D, out, gate=0.20, ev_min=0.0, ev_rel=None, nmin=1, allowed=None):
    """Gate on coverage, then on evidence: min(evA, evB) >= ev_min, and if ev_rel
    is set, >= log10(Na*Nb) + ev_rel (a multiple-comparison correction)."""
    pq = np.minimum(out[:, 0] / D.total[D.gq], 1.0)
    ph = np.minimum(out[:, 1] / D.total[D.gh], 1.0)
    mx = np.maximum(pq, ph)
    ev = np.minimum(out[:, 4], out[:, 5])
    nm = np.minimum(out[:, 2], out[:, 3])
    ok = (mx > 0) & (ev >= ev_min) & (nm >= nmin)
    if ev_rel is not None:
        c = counts(D)
        ok &= ev >= np.log10(c[D.gq] * c[D.gh]) + ev_rel
    if allowed is not None:
        ok &= np.isin(D.go, allowed)
    lo = np.minimum(D.gq, D.gh); hi = np.maximum(D.gq, D.gh)
    g = np.flatnonzero(ok)
    o = np.lexsort((D.go[g], -mx[g], hi[g], lo[g]))
    g = g[o]
    first = np.r_[True, (lo[g][1:] != lo[g][:-1]) | (hi[g][1:] != hi[g][:-1])]
    g = g[first]
    g = g[mx[g] >= gate]
    return {(int(lo[x]), int(hi[x])): (float(mx[x]), int(D.go[x])) for x in g}
