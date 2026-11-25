"""Graded corroboration: how many witnesses a loose frame match must have.

The shipped rule asks for exactly one witness whatever the distance. This asks
whether making the demand grow with the distance buys precision at the loose
end of `-d` without costing recall at the tight end.

The witness pool is every match out to the rule's widest distance, exactly as
`compare::corroborated` builds it -- getting that wrong (letting 24-bit matches
witness) inflates every figure, which is how the first run of this read.
"""
import sys, json
sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark')
sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
import numpy as np, rules, evalfps as E

# Rebuild with `python3 prep.py 16 <this path>` if it is missing.
PAIRS = '/tmp/vidfp-lab/pp16.pkl'

TAU = 500
NEVER = 1 << 20


def cov(r, strict, need, tau=TAU):
    """`need[x]` = witnesses a match at distance x needs (0 = unconditional)."""
    ma = np.zeros(r['na'], bool)
    mb = np.zeros(r['nb'], bool)
    D, I, J, off = r['D'], r['I'], r['J'], r['off']
    s = D <= strict
    ma[I[s]] = True
    mb[J[s]] = True
    widest = max((x for x in range(len(need)) if need[x] < NEVER), default=strict)
    if widest <= strict:
        return (min(r['da'][ma].sum() / r['Ta'], 1.0),
                min(r['db'][mb].sum() / r['Tb'], 1.0))
    m = D <= widest
    I, J, D, off = I[m], J[m], D[m], off[m]
    lo = np.searchsorted(off, off - tau, side='left')
    hi = np.searchsorted(off, off + tau, side='right')
    for k in np.nonzero(D > strict)[0]:
        a, b = lo[k], hi[k]
        if b - a < 2:
            continue
        w = int(((I[a:b] != I[k]) & (J[a:b] != J[k])).sum())
        if w >= need[D[k]]:
            ma[I[k]] = True
            mb[J[k]] = True
    return (min(r['da'][ma].sum() / r['Ta'], 1.0),
            min(r['db'][mb].sum() / r['Tb'], 1.0))


def need_shipped(d):
    """min(d,8) unconditional; out to max(d, min(d+8,12)) with one witness."""
    strict = min(d, 8)
    loose = max(d, min(d + 8, 12))
    need = np.full(33, NEVER)
    need[:strict + 1] = 0
    need[strict + 1:loose + 1] = 1
    return strict, need


def need_graded(d, cap, step, base=1):
    """Witnesses grow with the distance past `strict`, out to `max(d, cap)`."""
    strict = min(d, 8)
    loose = max(d, cap)
    need = np.full(33, NEVER)
    need[:strict + 1] = 0
    for x in range(strict + 1, min(loose, 32) + 1):
        need[x] = base + (x - strict - 1) // step
    return strict, need


def score_rule(recs, rel, mk, ds, ps=(0.20,), tau=TAU):
    sc = E.scorer(rel)
    out = []
    for d in ds:
        strict, need = mk(d)
        C = np.array([cov(r, strict, need, tau) for r in recs])
        mx = C.max(axis=1)
        for p in ps:
            sel = mx >= p
            out.append((d, p, sc(sel, 'exact'), sc(sel, 'any')))
    return out


HDR = (f"{'rule':<22}{'d':>3}{'p':>6}{'n':>6}{'tp':>5}{'fp':>4}{'unj':>5}"
       f"{'P':>8}{'R':>7}{'F1':>7}{'F1f':>7}{'any':>7}{'anyf':>7}")


def show(name, rows):
    for (d, p, e, a) in rows:
        print(f"{name:<22}{d:>3}{p:>6}{e['n']:>6}{e['tp']:>5}{e['fp']:>4}{e['unj']:>5}"
              f"{100 * e['P']:>7.1f}%{100 * e['R']:>6.1f}%{e['F1']:>7.3f}{e['F1f']:>7.3f}"
              f"{a['F1']:>7.3f}{a['F1f']:>7.3f}", flush=True)


if __name__ == '__main__':
    recs, keys, rel = rules.load(PAIRS)
    DS = [2, 4, 6, 8, 10, 12, 14, 16]
    print(HDR)
    show('shipped', score_rule(recs, rel, need_shipped, DS))
    print()
    for cap in (12, 14, 16, 20):
        for step in (1, 2, 3):
            show(f'graded c{cap} s{step}',
                 score_rule(recs, rel, lambda d, c=cap, s=step: need_graded(d, c, s), DS))
            print()
