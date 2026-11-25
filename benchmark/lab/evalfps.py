"""Score one fingerprint variant: features -> rule search -> best F1."""
import sys, numpy as np
sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark')
sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
import lab, score as S

LABELS = S.load_labels('/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/labels.csv')
N_POS_EXACT, N_POS_ANY = 471, 605


def relations(feat, paths):
    rel = np.empty(len(feat['a']), dtype=object)
    for i, (a, b) in enumerate(zip(feat['a'], feat['b'])):
        key = S.pair_key(paths[a], paths[b])
        rel[i] = LABELS.get(key, {}).get('relation', 'UNJUDGED')
    return rel


def scorer(rel):
    same, clip = rel == 'SAME', rel == 'CLIP'
    diff, unj = rel == 'DIFFERENT', rel == 'UNJUDGED'

    def f(sel, mode='exact'):
        tp = int((sel & same).sum()); fp = int((sel & diff).sum())
        unjn = int((sel & unj).sum()); cl = int((sel & clip).sum())
        if mode == 'any':
            tp += cl; npos = N_POS_ANY
        else:
            npos = N_POS_EXACT
        P = tp / (tp + fp) if tp + fp else 0.0
        Pf = tp / (tp + fp + unjn) if tp + fp + unjn else 0.0
        R = tp / npos
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        F1f = 2 * Pf * R / (Pf + R) if Pf + R else 0.0
        return dict(n=int(sel.sum()), tp=tp, fp=fp, unj=unjn, P=P, R=R,
                    F1=F1, F1f=F1f)
    return f


def search(feat, rel, ds=None, ps=(0.01, 0.05, 0.1, 0.2), ks=(1, 2, 3, 4, 5),
           qs=(0.0, 0.2, 0.35, 0.5), top=8, quiet=False):
    sc = scorer(rel)
    DS = list(feat['ds'])
    ds = ds or DS
    rows = []
    for dd in ds:
        j = DS.index(dd)
        ca, cb = feat['cov_a'][:, j], feat['cov_b'][:, j]
        na, nb = feat['ns_a'][:, j], feat['ns_b'][:, j]
        mx, mn = np.maximum(ca, cb), np.minimum(ca, cb)
        mns = np.minimum(na, nb)
        for p in ps:
            base = mx >= p
            for k in ks:
                for q in qs:
                    strong = np.ones_like(base) if (k <= 1 and q <= 0) else \
                        ((mns >= k) | (mn >= q) if q > 0 else (mns >= k))
                    sel = base & strong
                    e = sc(sel, 'exact'); a = sc(sel, 'any')
                    rows.append(dict(d=dd, p=p, k=k, q=q, e=e, a=a,
                                     obj=e['F1f'] + a['F1f']))
    rows.sort(key=lambda r: -r['obj'])
    if not quiet:
        for r in rows[:top]:
            e, a = r['e'], r['a']
            print(f"d={r['d']:<3} p={r['p']:<5} ns>={r['k']} cov>={r['q']:<4} "
                  f"n={e['n']:<5} tp={e['tp']:<4} fp={e['fp']:<3} unj={e['unj']:<4} "
                  f"P={100*e['P']:5.1f}% R={100*e['R']:5.1f}% F1={e['F1f']:.3f}  "
                  f"any F1={a['F1f']:.3f}")
    return rows
