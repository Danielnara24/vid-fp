"""Simulate 'a link needs one frame match within strict' on a dumped ladder."""
import json, os, sys
from math import sqrt
import order as O, score as S, ladder as L
ALT_ONLY = int(os.environ.get("ALT_ONLY", "0"))
def strict(d): return round(64 - (32 - d) / 4 * sqrt(128) / 2)
def pr_rule(path, d):
    tp = fp = un = 0
    for line in open(path):
        rec = json.loads(line)
        a, b = os.path.relpath(rec["a"], S.VIDS), os.path.relpath(rec["b"], S.VIDS)
        if a not in O.man or b not in O.man: continue
        if (ALT_ONLY == 0 or rec["mir"] or rec["ref"]) and rec["pairs"] and min(p[2] for p in rec["pairs"]) > strict(d): continue
        lab = O.label(a, b)
        if lab == "TP": tp += 1
        elif lab in ("FP_look", "FP_other"): fp += 1; un += lab == "FP_other"
    p, r = tp / max(1, tp + fp), tp / L.POS
    return p, r, 2 * p * r / max(1e-9, p + r), fp, un
d = sys.argv[1]
B = {k: L.pr(f"{d}/sweep-d{k}.jsonl") for k in range(1, 13)}
curve = [(v[0], v[1]) for v in B.values()]
print(f"{'-d':>3} | {'base P':>7} {'R':>6} {'F1':>6} {'FP':>5} {'unrel':>5} | {'rule P':>7} {'R':>6} {'F1':>6} {'FP':>5} {'unrel':>5} | base R at rule P")
for k in range(1, 13):
    b, c = B[k], pr_rule(f"{d}/sweep-d{k}.jsonl", k)
    r_at = L.interp(curve, c[0]); ra = f"{r_at:.4f}" if r_at is not None else "  --"
    print(f"{k:>3} | {b[0]:.4f} {b[1]:.4f} {b[2]:.4f} {b[3]:5} {b[4]:5} | {c[0]:.4f} {c[1]:.4f} {c[2]:.4f} {c[3]:5} {c[4]:5} | {ra}")
