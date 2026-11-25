"""Run gate vs no gate over a -d sweep, and whether the gate moves the frontier.

    CORPUS_ROOT=... python3 sweep_pr.py DIR      # reads DIR/sweep-d{1..12}.jsonl
"""
import json, os, sys
import order as O, order_sim as M, score as S

POS = sum(1 for r in S.load_pairs().values() if S.truth(r, 0.20, False) == "P")


def pr(path):
    res = {"now": [0, 0], "gate": [0, 0]}
    V = S.VIDS
    for line in open(path):
        rec = json.loads(line)
        a, b = os.path.relpath(rec["a"], V), os.path.relpath(rec["b"], V)
        if a not in O.man or b not in O.man:
            continue
        lab = O.label(a, b)
        if lab not in ("TP", "FP_look", "FP_other"):
            continue
        k = 0 if lab == "TP" else 1
        res["now"][k] += 1
        f, n = M.runfrac(rec)
        if n <= 3 or max(rec["ca"], rec["cb"]) * f >= 0.20:
            res["gate"][k] += 1
    out = {}
    for name, (tp, fp) in res.items():
        p = tp / max(1, tp + fp); r = tp / POS
        out[name] = (p, r, 2 * p * r / max(1e-9, p + r), fp)
    return out


def interp(curve, x, xi, yi):
    """y on `curve` (list of points) at x, linear between the two neighbours."""
    pts = sorted(curve, key=lambda c: c[xi])
    for (a, b) in zip(pts, pts[1:]):
        if a[xi] <= x <= b[xi]:
            t = (x - a[xi]) / (b[xi] - a[xi]) if b[xi] != a[xi] else 0
            return a[yi] + t * (b[yi] - a[yi])
    return None


def main(d):
    rows = {}
    for k in range(1, 13):
        p = os.path.join(d, f"sweep-d{k}.jsonl")
        if os.path.exists(p):
            rows[k] = pr(p)
    print(f"{'-d':>3}  {'now P':>7} {'R':>6} {'F1':>6} {'FP':>5}   {'gate P':>7} {'R':>6} {'F1':>6} {'FP':>5}   "
          f"{'ungated R at gate P':>20} {'ungated P at gate R':>20}")
    now = [v["now"] for v in rows.values()]
    for k, v in rows.items():
        n, g = v["now"], v["gate"]
        r_at = interp(now, g[0], 0, 1)
        p_at = interp(now, g[1], 1, 0)
        f = lambda x, w=6: f"{x:.4f}" if x is not None else "  --  "
        print(f"{k:>3}  {n[0]:.4f} {n[1]:.4f} {n[2]:.4f} {n[3]:5}   {g[0]:.4f} {g[1]:.4f} {g[2]:.4f} {g[3]:5}   "
              f"{f(r_at):>20} {f(p_at):>20}")


if __name__ == "__main__":
    main(sys.argv[1])
