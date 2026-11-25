"""The existing compare.rs rule, re-derived for a W-bit hash (sigma-based):
strict = chance - (32-d)/4 sigma_W ... i.e. -d keeps its sigma meaning;
loose = strict + 1.5 sigma; witnesses from the W-bit binomial tail, anchor 5
sigma, budget 2. Distance = d1 + d2 (W=128). d1 additionally capped."""
import numpy as np
from math import comb, log10, sqrt


def tail_om(W):
    tot = 2 ** W
    acc = 0
    out = []
    for x in range(W + 1):
        acc += comb(W, x)
        out.append(-log10(acc / tot))
    return out


def rule(d, W=128, slack=1.5, anchor=5.0, budget=2.0):
    sig = sqrt(W) / 2
    sig64 = 4.0
    below = (32 - d) / sig64                     # sigmas below chance -d stands for
    strict = int(round(W / 2 - below * sig))
    loose = min(W, strict + int(round(slack * sig)))
    om = tail_om(W)
    anc = om[int(round(W / 2 - anchor * sig))]
    need = np.full(W + 1, 1000, np.int64)
    for x in range(W + 1):
        if x <= strict:
            need[x] = 0
        elif x <= loose:
            w = 1
            while (1 + w) * om[x] < budget * anc and w < 64:
                w += 1
            need[x] = w
    return strict, loose, need


def need_of(D, d, d1cap=16, **kw):
    s, l, need = rule(d, **kw)
    dd = D.d1.astype(np.int64) + D.d2.astype(np.int64)
    n = need[np.minimum(dd, 128)]
    n[D.d1 > d1cap] = 1000
    return n, (s, l)
