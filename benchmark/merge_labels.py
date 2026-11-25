#!/usr/bin/env python3
"""Merge a review.html export back into labels.csv and re-score the runs it
was built to adjudicate.

    python3 merge_labels.py ~/Downloads/labels.csv           # dry run
    python3 merge_labels.py ~/Downloads/labels.csv --apply   # write it

Only pairs labels.csv has no verdict on are appended -- an existing label is
never overwritten, because the whole point of the file is that a human decided
once and the decision stays decided. SKIP rows are carried in with everything
else; `score.py` excludes them from both denominators rather than guessing.

A backup of labels.csv is written beside it before anything is appended.
"""
import argparse, collections, csv, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "labels.csv")
VALID = {"SAME", "CLIP", "DIFFERENT", "SKIP"}
REPORTS = "/tmp/vidfp-lab/rep"
PROFILES = ("4-20", "8-10", "10-20", "12-10")


def key(a, b):
    return (a, b) if a <= b else (b, a)


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="labels.csv exported from review.html")
    ap.add_argument("--apply", action="store_true", help="write labels.csv")
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()

    have = read(LABELS)
    cols = list(have[0].keys())
    known = {key(r["path_a"], r["path_b"]) for r in have}

    new, skipped, bad, blank = [], 0, [], 0
    for r in read(args.export):
        rel = (r.get("relation") or "").strip().upper()
        if not rel:
            blank += 1
            continue
        if rel not in VALID:
            bad.append((r.get("path_a", ""), rel))
            continue
        k = key(r["path_a"], r["path_b"])
        if k in known:
            skipped += 1
            continue
        new.append({c: r.get(c, "") for c in cols} | {"relation": rel})
        known.add(k)

    counts = collections.Counter(r["relation"] for r in new)
    print(f"{len(new)} new label(s): " +
          ", ".join(f"{c} {n}" for c, n in sorted(counts.items())))
    if blank:
        print(f"{blank} row(s) left unlabeled -- not merged")
    if skipped:
        print(f"{skipped} row(s) already in labels.csv -- left as they were")
    for path, rel in bad:
        print(f"  ! unrecognised relation {rel!r} on {os.path.basename(path)}",
              file=sys.stderr)
    if bad:
        return 1
    if not new:
        return 0

    if not args.apply:
        print("\ndry run -- pass --apply to write. Preview:")
        for r in new[:10]:
            print(f"   {r['relation']:<10}{os.path.basename(r['path_a'])} | "
                  f"{os.path.basename(r['path_b'])}")
        return 0

    backup = f"{LABELS}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
    shutil.copy2(LABELS, backup)
    with open(LABELS, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=cols).writerows(new)
    print(f"appended to {LABELS} (backup: {os.path.basename(backup)})")

    if args.no_score:
        return 0
    print()
    sys.path.insert(0, HERE)
    import importlib, score as S
    importlib.reload(S)
    L = S.load_labels(LABELS)
    pos_exact = sum(1 for v in L.values() if v["relation"] == "SAME")
    pos_any = sum(1 for v in L.values() if v["relation"] in ("SAME", "CLIP"))

    def sc(pairs, mode):
        tp = fp = unj = cl = 0
        for k in pairs:
            r = L.get(k, {}).get("relation", "UNJUDGED")
            if r == "SAME": tp += 1
            elif r == "DIFFERENT": fp += 1
            elif r == "CLIP": cl += 1
            elif r == "SKIP": pass
            else: unj += 1
        npos = pos_exact
        if mode == "any":
            tp += cl; npos = pos_any
        P = tp / (tp + fp) if tp + fp else 0.0
        Pf = tp / (tp + fp + unj) if tp + fp + unj else 0.0
        R = tp / npos if npos else 0.0
        f = lambda p: 2 * p * R / (p + R) if p + R else 0.0
        return dict(n=len(pairs), tp=tp, fp=fp, cl=cl, unj=unj, P=P, R=R,
                    F1=f(P), F1f=f(Pf))

    print(f"{'profile':<9}{'rule':<10}{'pairs':>6}{'tp':>5}{'fp':>4}{'clip':>5}"
          f"{'unj':>5}{'P':>8}{'R':>7}{'F1':>7}{'F1f':>7}")
    for prof in PROFILES:
        for b in ("base", "distinct"):
            p = os.path.join(REPORTS, f"{b}_{prof}.json")
            if not os.path.exists(p):
                continue
            e = sc(set(S.vidfp_link_pairs(p)), "exact")
            print(f"{prof:<9}{b:<10}{e['n']:>6}{e['tp']:>5}{e['fp']:>4}{e['cl']:>5}"
                  f"{e['unj']:>5}{100*e['P']:>7.1f}%{100*e['R']:>6.1f}%"
                  f"{e['F1']:>7.3f}{e['F1f']:>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
