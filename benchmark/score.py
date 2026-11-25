#!/usr/bin/env python3
"""Score duplicate-video finders against the hand-labeled pair ground truth.

`gt.py` builds a pool of candidate pairs from several independent proposers and
a human labels the union; this reads those labels and asks, of each tool run at
its own defaults, two questions:

    precision -- of the pairs it proposed, how many are real?
    recall    -- of the real pairs anyone proposed, how many did it find?

Both numbers are pair-level. Tools report *groups*, so a group of n files is
expanded to its n(n-1)/2 pairs: a tool that puts A, B and C together is
asserting A-C as much as A-B, and is scored on it. vid-fp is the exception and
is scored twice, because it reports pairwise links inside its groups and its
own safety rules distinguish them -- see `--vidfp-links` below.

## The one thing these numbers cannot say

Recall is measured against the pool, not against the corpus. A pair no proposer
ever nominated is invisible to every tool's score alike, so recall here is
*recall of the discoverable*, and it is an upper bound on true recall. This is
the standard pooling caveat and it is why `gt.py` unions several independent
proposers before anyone labels: the more independent the pool, the tighter the
bound. Precision has no such caveat -- every pair a pooled tool proposes was
labeled, so its errors are counted exactly.

A tool added to the pool *after* labeling would be scored unfairly: its unique
finds sit outside the labels and score as neither right nor wrong. Those show
up as `unjudged` below, and precision is reported as a range whose lower end
counts every one of them as a false positive.

## Two definitions of a positive

    exact   SAME is a positive. CLIP is neither -- finding one is not
            rewarded, missing one is not punished. This is the "find me
            redundant copies" question.
    any     SAME and CLIP are both positives. This is the "find me shared
            footage" question, and it is the one clip detection is for.

SKIP labels ("could not tell") are excluded from both, in both modes. A wrong
label is worse than a missing one, so they are left out of the denominator
rather than guessed at.

Usage:
    python3 score.py                    # score the default-settings runs
    python3 score.py curve              # vid-fp precision/recall across -d/-p
"""

import argparse
import collections
import csv
import glob
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Relations that count as a positive under each mode. Everything labeled and
# not listed here is a negative, except SKIP, which is dropped entirely.
MODES = {
    "exact": {"positive": {"SAME"}, "ignored": {"CLIP", "SKIP"}},
    "any": {"positive": {"SAME", "CLIP"}, "ignored": {"SKIP"}},
}


def pair_key(a, b):
    """Order-independent key for a pair.

    Paths are used verbatim -- no strip(). Leading and trailing spaces are
    legal in a filename and the labels were written by a CSV writer that emits
    them, so trimming here would silently fail to match the pair that was
    actually judged. This is the same rule `report.rs` follows for `full_path`.
    """
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------- proposals


def load_labels(path):
    """pair -> {"relation": str, "row": dict}."""
    labels = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            a, b = row["path_a"], row["path_b"]
            rel = (row.get("relation") or "").strip().upper()
            if not rel:
                continue
            labels[pair_key(a, b)] = {"relation": rel, "row": row}
    return labels


PATH_KEYS = ("Path", "path", "full_path", "FullName", "file", "File",
             "filename", "FileName", "location")
GROUP_KEYS = ("GroupId", "group_id", "group", "Group")


def _path_of(d):
    for k in PATH_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.startswith("/"):
            return v
    return None


def extract_groups(obj):
    """Recover duplicate groups from any tool's JSON by walking the tree.

    Same reader `gt.py` pools with, and deliberately so: a tool must be scored
    on exactly the pairs it was pooled on, or its own proposals would fall
    outside the labels built from them.
    """
    groups = []

    def walk(node):
        if isinstance(node, list):
            here = [d for d in node if isinstance(d, dict) and _path_of(d)]
            if len(here) >= 2:
                tagged = collections.defaultdict(list)
                for d in here:
                    gid = next((str(d[k]) for k in GROUP_KEYS if k in d), None)
                    tagged[gid].append(_path_of(d))
                for paths in tagged.values():
                    if len(paths) >= 2:
                        groups.append(paths)
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)

    walk(obj)
    return groups


def group_pairs(path):
    """Every pair implied by the groups in one tool's JSON output."""
    with open(path) as fh:
        groups = extract_groups(json.load(fh))
    pairs = set()
    for g in groups:
        for a, b in itertools.combinations(sorted(set(g)), 2):
            pairs.add(pair_key(a, b))
    return pairs


def vidfp_link_pairs(path):
    """Only the pairs vid-fp actually measured, not its groups' closure.

    vid-fp groups by maximal clique, so the two sets are currently identical:
    every pair a group asserts was measured. That was not true of the
    connected-components build, where a group could chain files never compared
    with each other, and it need not stay true -- so both are still computed and
    both reported. The two rows drifting apart is the signal that the clustering
    rule has started claiming more than the comparison stage found.
    """
    with open(path) as fh:
        j = json.load(fh)
    pairs = set()
    for group in j.get("results", []):
        for f in group.get("files", []):
            for m in f.get("matches", []):
                pairs.add(pair_key(f["full_path"], m["full_path"]))
    return pairs


def labels_source_pairs(labels, source):
    """Pairs credited to `source` in the pool's own `sources` column.

    Used for proposers with no re-runnable CLI at scoring time -- the oracle,
    whose proposals are `gt.py oracle` + `gt.py pool`'s admission thresholds
    running at their defaults. By construction these have no unjudged pairs.
    """
    out = set()
    for key, rec in labels.items():
        if source in (rec["row"].get("sources") or "").split("+"):
            out.add(key)
    return out


# ------------------------------------------------------------------ scoring


class Score:
    def __init__(self, name, proposed, labels, mode):
        spec = MODES[mode]
        self.name = name
        self.mode = mode
        self.proposed = proposed

        judged = {k: r["relation"] for k, r in labels.items()}
        positives = {k for k, rel in judged.items() if rel in spec["positive"]}
        ignored = {k for k, rel in judged.items() if rel in spec["ignored"]}

        self.tp = proposed & positives
        self.unjudged = proposed - set(judged)
        self.ignored_hits = proposed & ignored
        self.fp = proposed - positives - ignored - self.unjudged
        self.fn = positives - proposed
        self.n_positives = len(positives)

    @property
    def precision(self):
        d = len(self.tp) + len(self.fp)
        return len(self.tp) / d if d else float("nan")

    @property
    def precision_floor(self):
        """Precision if every unjudged proposal turned out to be wrong."""
        d = len(self.tp) + len(self.fp) + len(self.unjudged)
        return len(self.tp) / d if d else float("nan")

    @property
    def recall(self):
        return len(self.tp) / self.n_positives if self.n_positives else float("nan")

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def f1_floor(self):
        """F1 recomputed against `precision_floor`.

        Recall is unaffected by unjudged proposals -- they are pairs nobody
        labeled, so they cannot be positives that were missed -- but precision
        is, so a single F1 for a tool with unjudged proposals quietly reports
        its best case. Only the vid-fp group closure has any here.
        """
        p, r = self.precision_floor, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def pct(x):
    return "  n/a " if x != x else f"{100 * x:5.1f}%"


def table(scores, mode, out=sys.stdout):
    spec = MODES[mode]
    pos = "+".join(sorted(spec["positive"]))
    print(f"\n=== positives = {pos} "
          f"(ignoring {', '.join(sorted(spec['ignored']))}) ===", file=out)
    hdr = (f"{'tool':<22}{'proposed':>9}{'TP':>6}{'FP':>6}{'ign':>5}{'unjud':>7}"
           f"{'FN':>6}{'precision':>15}{'recall':>9}{'F1':>14}")
    print(hdr, file=out)
    print("-" * len(hdr), file=out)
    for s in scores:
        # Ranges, low end first, when unjudged proposals could go either way:
        # the floor counts every one of them as wrong, the ceiling ignores
        # them. Recall is not affected, so only precision and F1 get a range.
        prec = (f"{100 * s.precision_floor:.1f}–{100 * s.precision:.1f}%"
                if s.unjudged else pct(s.precision).strip())
        f1 = (f"{s.f1_floor:.3f}–{s.f1:.3f}" if s.unjudged else f"{s.f1:.3f}")
        print(f"{s.name:<22}{len(s.proposed):>9}{len(s.tp):>6}{len(s.fp):>6}"
              f"{len(s.ignored_hits):>5}{len(s.unjudged):>7}{len(s.fn):>6}"
              f"{prec:>15}{pct(s.recall):>9}{f1:>14}", file=out)
    print(f"\n{scores[0].n_positives} positive pairs in the pool. "
          f"proposed = TP + FP + ign + unjud, where `ign` is a proposal on a "
          f"pair this mode\nneither rewards nor punishes.", file=out)


def breakdown(predictors, labels, out=sys.stdout):
    """Recall split by what kind of pair was missed."""
    def bucket(key, rec):
        row = rec["row"]
        wire = "/Wire/" in key[0] or "/Wire/" in key[1]
        if wire:
            return "Wire (episodes + planted clips)"
        return ("same folder" if row.get("same_folder") == "yes"
                else "cross folder")

    for rel in ("SAME", "CLIP"):
        groups = collections.defaultdict(set)
        for key, rec in labels.items():
            if rec["relation"] == rel:
                groups[bucket(key, rec)].add(key)
        if not groups:
            continue
        names = sorted(groups)
        print(f"\n=== {rel} recall by pair kind ===", file=out)
        hdr = f"{'tool':<22}" + "".join(
            f"{n + f' ({len(groups[n])})':>36}" for n in names)
        print(hdr, file=out)
        print("-" * len(hdr), file=out)
        for name, proposed in predictors.items():
            cells = ""
            for n in names:
                hit = len(proposed & groups[n])
                cells += f"{f'{hit}/{len(groups[n])}  {pct(hit / len(groups[n]))}':>36}"
            print(f"{name:<22}{cells}", file=out)


def concentration(labels, mode, out=sys.stdout):
    """How much of the positive set is one connected blob of near-identical files.

    Pair-level recall weights a group of n files as n(n-1)/2 pairs, so a single
    large cluster of re-encodes can outvote every other finding in the corpus.
    Here one 19-file cluster is 171 pairs -- over a third of the SAME positives
    -- and every tool but one finds all of it, which lifts everyone's headline
    recall by the same ~35 points and says nothing about the hard cases. The
    per-kind tables below are the ones to read; this is just the warning label.
    """
    spec = MODES[mode]
    pos = [k for k, r in labels.items() if r["relation"] in spec["positive"]]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pos:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp = collections.Counter(find(p) for p in parent)
    sizes = sorted(comp.values(), reverse=True)
    if not sizes:
        return
    biggest = sizes[0]
    inside = sum(1 for a, b in pos if find(a) == comp.most_common(1)[0][0])
    print(f"\nconcentration: {len(comp)} connected components over "
          f"{len(pos)} positive pairs; the largest joins {biggest} files and "
          f"accounts for\n{inside} pairs ({100 * inside / len(pos):.0f}% of "
          f"them). Read the per-kind tables, not the headline recall.", file=out)


def examples(scores, labels, out=sys.stdout, n=4):
    print("\n=== a few false positives, per tool ===", file=out)
    for s in scores:
        if not s.fp:
            print(f"\n{s.name}: none", file=out)
            continue
        print(f"\n{s.name}: {len(s.fp)} total", file=out)
        for key in sorted(s.fp)[:n]:
            note = (labels[key]["row"].get("notes") or "").strip()
            print(f"  {os.path.basename(key[0])}"
                  f"  <->  {os.path.basename(key[1])}"
                  + (f"   [{note}]" if note else ""), file=out)


def write_per_pair(path, predictors, labels):
    """Every judged pair with which tools found it -- the drill-down file."""
    names = list(predictors)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "relation", "found_by_count"] + names
                   + ["path_a", "path_b", "notes"])
        for key, rec in sorted(labels.items(),
                               key=lambda kv: (kv[1]["relation"], kv[0])):
            hits = [1 if key in predictors[n] else 0 for n in names]
            w.writerow([rec["row"].get("pair_id", ""), rec["relation"], sum(hits)]
                       + hits + [key[0], key[1],
                                 (rec["row"].get("notes") or "").strip()])


# -------------------------------------------------------------- subcommands


def build_predictors(args, labels):
    """name -> set of proposed pairs, for each tool run at its defaults."""
    p = collections.OrderedDict()
    d = args.runs

    vidfp = os.path.join(d, "vidfp.json")
    if os.path.exists(vidfp):
        p["vid-fp (links)"] = vidfp_link_pairs(vidfp)
        p["vid-fp (groups)"] = group_pairs(vidfp)
    else:
        print(f"score: no {vidfp}; run vid-fp at defaults first", file=sys.stderr)

    for name, fname in (("czkawka", "czkawka.json"), ("vdf", "vdf.json")):
        path = os.path.join(d, fname)
        if os.path.exists(path):
            p[name] = group_pairs(path)
        else:
            print(f"score: no {path}, skipping {name}", file=sys.stderr)

    p["oracle"] = labels_source_pairs(labels, "oracle")

    mfdf = os.path.join(HERE, "tools", "mfdf.json")
    if args.mfdf and os.path.exists(mfdf):
        p["mfdf"] = group_pairs(mfdf)
    return p


def cmd_report(args):
    labels = load_labels(args.labels)
    counts = collections.Counter(r["relation"] for r in labels.values())
    print(f"labels: {len(labels)} judged pairs "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    predictors = build_predictors(args, labels)
    if not predictors:
        return 1

    os.makedirs(args.out, exist_ok=True)
    summary = {"labels": dict(counts), "modes": {}}

    for mode in ("exact", "any"):
        scores = [Score(n, pr, labels, mode) for n, pr in predictors.items()]
        table(scores, mode)
        summary["modes"][mode] = {
            s.name: {
                "proposed": len(s.proposed), "tp": len(s.tp), "fp": len(s.fp),
                "fn": len(s.fn), "unjudged": len(s.unjudged),
                "precision": s.precision, "precision_floor": s.precision_floor,
                "recall": s.recall, "f1": s.f1, "f1_floor": s.f1_floor,
            } for s in scores
        }
        concentration(labels, mode)
        if mode == "any":
            examples(scores, labels)

    breakdown(predictors, labels)

    per_pair = os.path.join(args.out, "per_pair.csv")
    write_per_pair(per_pair, predictors, labels)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {per_pair} and {os.path.join(args.out, 'summary.json')}")
    return 0


def cmd_curve(args):
    """vid-fp's precision/recall across its two knobs.

    Cheap to run -- every setting reads the same warm fingerprint cache -- and
    it is the thing a single default-settings row cannot show: whether the
    default sits on a knee or on a cliff.
    """
    labels = load_labels(args.labels)
    os.makedirs(args.out, exist_ok=True)
    scratch = os.path.join(args.out, "curve")
    os.makedirs(scratch, exist_ok=True)

    rows = []
    for dist in args.distances:
        for p in args.percents:
            dest = os.path.join(scratch, f"d{dist}_p{p}.json")
            if not os.path.exists(dest):
                r = subprocess.run([args.bin, args.corpus, "-r", "-d", str(dist),
                                    "-p", str(p), "-o", dest, "--yes"],
                                   capture_output=True, text=True)
                if r.returncode not in (0, 2):
                    print(f"curve: -d {dist} -p {p} failed rc={r.returncode}",
                          file=sys.stderr)
                    continue
            pairs = vidfp_link_pairs(dest)
            row = {"d": dist, "p": p}
            for mode in ("exact", "any"):
                s = Score("vid-fp", pairs, labels, mode)
                row[f"{mode}_precision"] = s.precision
                row[f"{mode}_recall"] = s.recall
                row[f"{mode}_f1"] = s.f1
            row["proposed"] = len(pairs)
            row["unjudged"] = len(Score("vid-fp", pairs, labels, "any").unjudged)
            rows.append(row)

    hdr = (f"{'-d':>3}{'-p':>5}{'pairs':>8}{'unjud':>7}"
           f"{'P(exact)':>10}{'R(exact)':>10}{'F1':>7}"
           f"{'P(any)':>9}{'R(any)':>9}{'F1':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['d']:>3}{r['p']:>5}{r['proposed']:>8}{r['unjudged']:>7}"
              f"{pct(r['exact_precision']):>10}{pct(r['exact_recall']):>10}"
              f"{r['exact_f1']:>7.3f}"
              f"{pct(r['any_precision']):>9}{pct(r['any_recall']):>9}"
              f"{r['any_f1']:>7.3f}")

    dest = os.path.join(args.out, "curve.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {dest}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=os.path.join(HERE, "labels.csv"))
    ap.add_argument("--runs", default=os.path.join(HERE, "out", "default"),
                    help="directory of each tool's default-settings JSON")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "score"))
    sub = ap.add_subparsers(dest="cmd")

    ap.set_defaults(func=cmd_report, mfdf=True)
    s = sub.add_parser("report", help="score every tool's default run")
    s.add_argument("--no-mfdf", dest="mfdf", action="store_false")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("curve", help="vid-fp precision/recall over -d and -p")
    s.add_argument("--corpus", default="/home/daniel/Documents/AN")
    s.add_argument("--bin", default=os.path.join(HERE, os.pardir,
                                                 "target/release/vid-fp"))
    s.add_argument("--distances", type=int, nargs="+",
                   default=[2, 4, 6, 8, 10, 12])
    s.add_argument("--percents", type=int, nargs="+", default=[5, 10, 20, 50, 90])
    s.set_defaults(func=cmd_curve, mfdf=True)

    args = ap.parse_args()
    if not os.path.exists(args.labels):
        print(f"score: no labels at {args.labels}", file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
