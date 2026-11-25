import io, contextlib, collections, sys
import order as O
def runfrac(rec):
    ps = sorted(rec['pairs']); n = len(ps)
    w = max(500, rec['ta']/max(1,rec['na']), rec['tb']/max(1,rec['nb']))
    off = [tb-ta for ta,tb,_ in ps]
    inrun = [False]*n
    for k in range(1,n):
        if abs(off[k]-off[k-1]) <= w: inrun[k]=inrun[k-1]=True
    return sum(inrun)/max(1,n), n
def sim(path):
    with contextlib.redirect_stdout(io.StringIO()):
        rows = O.main(path)
    for exempt in (0, 3, 5):
        c = collections.Counter(); tot = collections.Counter(); fam=collections.Counter()
        for lab, r in rows.items():
            for st, a, b, rec in r:
                tot[lab] += 1
                f, n = runfrac(rec)
                if n <= exempt or max(rec['ca'], rec['cb']) * f >= 0.20:
                    c[lab] += 1
                    if lab=='FP_look': fam[O.sets[O.man[a]['seed']]]+=1
        tp,fp=c['TP'],c['FP_look']+c['FP_other']
        print(f"  exempt n<={exempt}: " + "  ".join(f"{k} {c[k]}/{tot[k]}" for k in ("TP","FP_look","FP_other","shared")) + f"   look-alikes left by set {dict(fam)}")
if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p); sim(p)
