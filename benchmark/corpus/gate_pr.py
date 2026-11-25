import io, contextlib, sys
import order as O, order_sim as M, score as S
pos = sum(1 for r in S.load_pairs().values() if S.truth(r, 0.20, False) == "P")
for path in sys.argv[1:]:
    with contextlib.redirect_stdout(io.StringIO()):
        rows = O.main(path)
    out = []
    for name, keep in (("now", lambda rec: True),
                       ("run gate", lambda rec: (lambda f, n: n <= 3 or max(rec["ca"], rec["cb"]) * f >= 0.20)(*M.runfrac(rec)))):
        tp = sum(1 for st, a, b, rec in rows["TP"] if keep(rec))
        fp = sum(1 for lab in ("FP_look", "FP_other") for st, a, b, rec in rows.get(lab, []) if keep(rec))
        p, r = tp / (tp + fp), tp / pos
        out.append(f"{name}: P {p:.4f} R {r:.4f} F1 {2*p*r/(p+r):.4f} (FP {fp})")
    print(f"{path.split('/')[-2]} {path.split('-')[-1][:2]}   " + "   |   ".join(out))
