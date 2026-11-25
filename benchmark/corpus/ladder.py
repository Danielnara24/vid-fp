"""Compare two -d ladders of link dumps: baseline vs candidate.

    CORPUS_ROOT=... python3 ladder.py DIR BASE_PREFIX CAND_PREFIX
reads DIR/{prefix}-d{1..12}.jsonl. For each candidate rung: P, R, F1, false
links (all, and between unrelated families), and the baseline curve's recall at
the candidate's precision (interpolated between baseline rungs) -- the test for
"moved along the curve" versus "moved the curve".
"""
import json, os, sys
import order as O, score as S

POS = sum(1 for r in S.load_pairs().values() if S.truth(r, 0.20, False) == "P")


def pr(path):
    tp = fp = un = 0
    for line in open(path):
        rec = json.loads(line)
        a, b = os.path.relpath(rec["a"], S.VIDS), os.path.relpath(rec["b"], S.VIDS)
        if a not in O.man or b not in O.man:
            continue
        lab = O.label(a, b)
        if lab == "TP":
            tp += 1
        elif lab in ("FP_look", "FP_other"):
            fp += 1
            un += lab == "FP_other"
    p, r = tp / max(1, tp + fp), tp / POS
    return p, r, 2 * p * r / max(1e-9, p + r), fp, un


def interp(curve, x):
    """Recall of the baseline curve at precision x (the best recall any
    interpolated baseline point reaches at precision >= x)."""
    pts = sorted(curve)              # by precision
    best = None
    for a, b in zip(pts, pts[1:]):
        if a[0] <= x <= b[0]:
            t = (x - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0
            r = a[1] + t * (b[1] - a[1])
            best = r if best is None else max(best, r)
    return best


def main(d, base, cand):
    B = {k: pr(f"{d}/{base}-d{k}.jsonl") for k in range(1, 13)}
    C = {k: pr(f"{d}/{cand}-d{k}.jsonl") for k in range(1, 13) if os.path.exists(f"{d}/{cand}-d{k}.jsonl")}
    curve = [(v[0], v[1]) for v in B.values()]
    print(f"{'-d':>3} | {'base P':>7} {'R':>6} {'F1':>6} {'FP':>5} {'unrel':>5} | {'cand P':>7} {'R':>6} {'F1':>6} {'FP':>5} {'unrel':>5} | base R at cand P")
    for k in C:
        b, c = B[k], C[k]
        r_at = interp(curve, c[0])
        ra = f"{r_at:.4f}" if r_at is not None else "  --"
        print(f"{k:>3} | {b[0]:.4f} {b[1]:.4f} {b[2]:.4f} {b[3]:5} {b[4]:5} | {c[0]:.4f} {c[1]:.4f} {c[2]:.4f} {c[3]:5} {c[4]:5} | {ra}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
