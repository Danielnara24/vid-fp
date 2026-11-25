#!/usr/bin/env python3
"""Score kf_sweep2's runs. Usage: python3 kf_report2.py  (writes out/kfsweep2/cells.csv)

Per cell: P/R/F1 at -p 20; recall at -p 0 scored against the SAME truth
(PARTIAL gate fixed at 0.20), so the difference is pairs the -p gate removed;
cold wall/user/RSS; cost per F1 and an efficiency relative to the interval-0
default ((F1/F1_0) / (wall/wall_0)).

The m-ladder check: for every interval, pairs (true positives) that both
neighbours of a rung find and the rung does not, and how many of those the
rung still finds at -p 0.
"""
import collections, contextlib, csv, importlib, io, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "kfsweep2")
ROOTS = {"VIDS": "/home/daniel/Documents/VIDS", "VIDS2": "/home/daniel/Documents/VIDS2"}


def scorer(corpus):
    os.environ["CORPUS_ROOT"] = ROOTS[corpus]
    sys.modules.pop("score", None)
    sys.path.insert(0, HERE)
    return importlib.import_module("score")


def evaluate(S, path):
    with contextlib.redirect_stdout(io.StringIO()):
        res = S.score(path, 0.20)
    pred, _ = S.predicted(path)
    pairs = S.load_pairs()
    tps = {p for p in pred if p in pairs and S.truth(pairs[p], 0.20, False) == "P"}
    return res, tps


def main():
    os.environ["SHARED_LYRIC"] = "1"   # the user confirmed lyric_1/2 share a real intro
    runs = [json.loads(l) for l in open(os.path.join(OUT, "runs.jsonl"))]
    rows = []
    for corpus in ROOTS:
        allr = [r for r in runs if r["corpus"] == corpus]
        rs = [r for r in allr if not r.get("round")]
        reps = collections.defaultdict(list)            # (i, m, tag) -> every cold wall, all rounds
        for r in allr:
            reps[(r["interval"], r["min_kf"], r["tag"])].append(r["cold"]["wall"])
        if not rs:
            continue
        S = scorer(corpus)
        base = [r for r in allr if r["interval"] == 0]
        w0 = statistics.mean(r["cold"]["wall"] for r in base)
        u0 = statistics.mean(r["cold"]["user"] for r in base)
        print(f"\n### {corpus}: default (interval 0) cold wall {', '.join(f'{r['cold']['wall']:.0f}' for r in base)} s "
              f"(mean {w0:.0f}), user mean {u0:.0f} s")
        cells = {}
        for r in rs:
            if r["interval"] == 0 and r["tag"] != "b0":
                continue
            res, tps = evaluate(S, r["json"])
            res0, tps0 = evaluate(S, r["json_p0"]) if r.get("json_p0") else (None, set())
            cells[(r["interval"], r["min_kf"])] = (r, res, tps, res0, tps0)
        f10 = cells[(0, 12)][1]["f1"]
        print(f"{'i':>3} {'m':>3} | {'P':>6} {'R':>6} {'F1':>6} {'FP':>4} | {'R@p0':>6} {'P@p0':>6} | "
              f"{'wall':>5} {'user':>5} {'RSS':>5} {'warm':>5} | {'s/F1':>6} {'rel.cost':>8} {'rel.F1':>6} {'eff':>5}")
        for (i, m) in sorted(cells):
            r, res, tps, res0, tps0 = cells[(i, m)]
            c = r["cold"]
            eff = (res["f1"] / f10) / (c["wall"] / w0)
            row = dict(corpus=corpus, interval=i, min_kf=m, precision=res["precision"], recall=res["recall"],
                       f1=res["f1"], fp=res["fp"], tp=res["tp"],
                       recall_p0=res0["recall"] if res0 else "", precision_p0=res0["precision"] if res0 else "",
                       wall=c["wall"], user=c["user"], rss_mb=round(c["rss_mb"]),
                       warm_compare=r.get("warm_p20", {}).get("wall", ""),
                       sec_per_f1=c["wall"] / res["f1"], rel_cost=c["wall"] / w0, rel_f1=res["f1"] / f10, eff=eff)
            row["wall_runs"] = " ".join(f"{w:.0f}" for w in reps[(i, m, r["tag"])])
            rows.append(row)
            print(f"{i:>3} {m:>3} | {res['precision']:.4f} {res['recall']:.4f} {res['f1']:.4f} {res['fp']:>4} | "
                  f"{row['recall_p0'] or 0:.4f} {row['precision_p0'] or 0:.4f} | {c['wall']:>5.0f} {c['user']:>5.0f} "
                  f"{c['rss_mb']:>5.0f} {row['warm_compare'] or 0:>5.1f} | {row['sec_per_f1']:>6.0f} {row['rel_cost']:>8.2f} "
                  f"{row['rel_f1']:>6.3f} {eff:>5.2f}  [{row['wall_runs']}]")
        print(f"\n{corpus}: recall along the -m ladder at -p 20 (and -p 0); '<' marks a rung below the one before it")
        for i in sorted({k[0] for k in cells if k[0]}):
            ms = sorted(m for (ii, m) in cells if ii == i)
            out, prev = [], None
            for m in ms:
                r20, r0 = cells[(i, m)][1]["recall"], cells[(i, m)][3]["recall"]
                mark = ""
                if prev:
                    mark = ("<" if r20 < prev[0] - 0.001 else " ") + ("<" if r0 < prev[1] - 0.001 else " ")
                out.append(f"m{m} {r20:.3f}/{r0:.3f}{mark}")
                prev = (r20, r0)
            print(f"  i{i:<3} " + "  ".join(out))
    with open(os.path.join(OUT, "cells.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
