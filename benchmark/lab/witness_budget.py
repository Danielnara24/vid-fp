"""How many witnesses, derived from how often chance produces a match that close.

A frame hash is 64 bits, so the distance between two unrelated frames is
Binomial(64, 1/2): P(<=8) is 3.7e-11, P(<=12) 2.0e-8, P(<=16) 3.5e-6, P(<=20)
1.7e-4 -- four orders of magnitude across the range `-d` covers. One witness at
12 bits and one at 20 are not the same evidence at all.

Treat -log10 P(<=x) as the evidence one frame match at distance x carries, and
require a cluster (the match plus its witnesses) to carry a fixed budget. The
budget is set so 12 bits keeps needing exactly one witness, i.e. 2*L(12).
"""
import sys, math
sys.path.insert(0, '/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
sys.path.insert(0,'/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark')
sys.path.insert(0,'/home/daniel/Documents/Vscode_repositories/deduplicator/benchmark/lab')
import numpy as np, rules, evalfps as E

# Rebuild with `python3 prep.py 16 <this path>` if it is missing.
PAIRS = '/tmp/vidfp-lab/pp16.pkl'
from witness_rules import NEVER, cov, HDR, show, need_shipped

L = [-math.log10(sum(math.comb(64,i) for i in range(x+1)) / 2**64) for x in range(33)]
print("x      : " + " ".join(f"{x:5d}" for x in range(4,25,2)))
print("L(x)   : " + " ".join(f"{L[x]:5.1f}" for x in range(4,25,2)))
for mult in (1.5, 2.0, 2.5, 3.0):
    budget = mult * L[12]
    sched = [max(1, math.ceil(budget / L[x]) - 1) for x in range(4,25,2)]
    print(f"k @{mult}L12: " + " ".join(f"{k:5d}" for k in sched))

def need_budget(d, mult):
    strict = min(d, 8)
    loose = max(d, min(d + 8, 12))
    budget = mult * L[12]
    need = np.full(33, NEVER)
    need[:strict+1] = 0
    for x in range(strict+1, min(loose,32)+1):
        need[x] = max(1, math.ceil(budget / L[x]) - 1)
    return strict, need

recs, keys, rel = rules.load(PAIRS)
sc = E.scorer(rel)
DS = [2, 4, 8, 12, 14, 16, 20]
print()
print(HDR)
def run(name, mk):
    rows = []
    for d in DS:
        strict, need = mk(d)
        C = np.array([cov(r, strict, need) for r in recs])
        sel = C.max(axis=1) >= 0.20
        rows.append((d, 0.20, sc(sel,'exact'), sc(sel,'any')))
    show(name, rows); print()
run('shipped', need_shipped)
for mult in (1.5, 2.0, 2.5, 3.0):
    run(f'budget {mult}xL12', lambda d, m=mult: need_budget(d, m))
