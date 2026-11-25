import numpy as np


def restrict(D, keep):
    """Drop records where keep[record] is False and rebuild the pair groups."""
    for a in ("i", "j", "d1", "d2", "off", "O", "key"):
        setattr(D, a, getattr(D, a)[keep])
    D.d = D.d1
    st = np.flatnonzero(np.r_[True, D.key[1:] != D.key[:-1]]) if len(D.key) else np.zeros(0, np.int64)
    D.gs = np.r_[st, len(D.key)].astype(np.int64)
    D.gkey = D.key[st]
    k = D.gkey
    D.go = k % D.no; D.gq = (k // D.no) // D.nf; D.gh = (k // D.no) % D.nf
    return D
