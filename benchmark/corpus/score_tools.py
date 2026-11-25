#!/usr/bin/env python3
"""Score any tool's JSON output on the generated corpus with score.py's truth.

A group is a claim that its members are duplicates of one another, so it is
scored as every pair inside it. vid-fp's reports are scored as their measured
links (score.predicted), which is what a vid-fp group asserts. Files outside
the manifest (none should appear; the hidden folders are excluded) are dropped.

    python3 score_tools.py NAME=path.json ...       # SHARED_LYRIC=1 for that scorer
"""
import importlib.util, io, contextlib, itertools, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S

spec = importlib.util.spec_from_file_location("an_score", os.path.join(HERE, "..", "score.py"))
AN = importlib.util.module_from_spec(spec); spec.loader.exec_module(AN)
man = S.load_manifest()


def group_pred(path):
    groups = AN.extract_groups(json.load(open(path)))
    pred = set()
    for g in groups:
        rel = sorted({os.path.relpath(p, S.VIDS) for p in g} & set(man))
        for a, b in itertools.combinations(rel, 2):
            pred.add(frozenset((a, b)))
    return pred, len(groups)


def real_tier(pred):
    """Seed pairs that share real footage (SHARED): is the pair of originals
    linked, and how many links join a pure copy of one to a pure copy of the
    other (files holding nothing but that seed)."""
    if not S.SHARED or not all(len(sh) == 2 for sh in S.SHARED):
        return None
    pure = collections.defaultdict(set)
    for f, r in man.items():
        ks = S.scene_keys(r, sets)
        if len(ks) == 1:
            pure[next(iter(ks))].add(f)
    out = []
    for sh in S.SHARED:
        a, b = sorted(sh)
        sa = next((f for f in pure[a] if man[f].get("transform") == "seed"), None)
        sb = next((f for f in pure[b] if man[f].get("transform") == "seed"), None)
        seed_hit = frozenset((sa, sb)) in pred
        fam = sum(1 for p in pred if any(x in pure[a] for x in p) and any(x in pure[b] for x in p))
        out.append(f"{a}~{b}:{'Y' if seed_hit else '-'}/{fam}")
    return "  ".join(out)


import collections
sets = S.load_sets()
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    d = json.load(open(path))
    is_vidfp = isinstance(d, dict) and "results" in d and "summary" in d
    if is_vidfp:
        ngroups = len(d["results"])
        pred, _ = S.predicted(path)
        with contextlib.redirect_stdout(io.StringIO()):
            r = S.score(path, 0.2, show=0)
    else:
        pred, ngroups = group_pred(path)
        orig = S.predicted
        S.predicted = lambda _p, pred=pred: (pred, {"results": []})
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = S.score(path, 0.2, show=0)
        finally:
            S.predicted = orig
    sr = sum(f for f, n in r["per_transform"].values()) / max(1, sum(n for f, n in r["per_transform"].values()))
    print(f"{name:12} groups {ngroups:5}  TP {r['tp']:6} FP {r['fp']:6} FN {r['fn']:6}  "
          f"P {r['precision']:.4f}  R {r['recall']:.4f}  F1 {r['f1']:.4f}  seed-recall {sr:.3f}")
    rt = real_tier(pred)
    if rt and os.environ.get("REAL"):
        print(f"{'':12} real: {rt}")
