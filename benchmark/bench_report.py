#!/usr/bin/env python3
"""Join the cost measurements to the accuracy scores in one table.

`bench_tools.py` measures what each run cost, `score.py` measures whether it was
right; this reads both and prints the comparison the two were taken for. It adds
no scoring logic of its own -- `Score`, the pair expansion and the label reader
are imported from `score.py`, so a number here and a number there cannot drift.

Two things it does that `score.py --runs` alone cannot:

* it scores tool configurations that are not in `build_predictors` (VDF with
  clip detection on, VDF with parallelism raised), and
* it reports each tool's runtime and peak memory beside its precision, which is
  the trade the table exists to show.

## The caveat this table cannot hide

The ground-truth pool was built by unioning proposals from vid-fp, Czkawka, VDF
at its defaults, an independent oracle and MFDF, and a human then labeled the
union. **VDF with `--partial-clip-detection` was not in that pool.** Pairs it
proposes that nobody else ever did are therefore unlabeled: they are neither
credited nor penalised, and its precision is printed as a range whose low end
counts every one of them as wrong and whose high end ignores them. Recall is
unaffected -- an unlabeled pair cannot be a positive that something missed.

This is the pooling trade `benchmark/README.md` warns about, and the honest fix
is to label those pairs (`gt.py pool` then `gt.py ui`) rather than to pick an
end of the range. Every other row here was pooled before labeling and has an
exact precision.
"""

import argparse
import collections
import json
import os
import statistics
import sys

import score

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------- cost side


def load_runs(path):
    """tool -> list of run records, in round order."""
    with open(path) as fh:
        runs = json.load(fh)
    by_tool = collections.OrderedDict()
    for r in runs:
        by_tool.setdefault(r["tool"], []).append(r)
    return by_tool


def steady(records, field):
    """The figure to quote for `field`: the median over every round.

    `benchmark/README.md` records a first-run effect on this machine -- the
    opening run of a batch coming in 8-15% faster than the steady state,
    whichever binary it is -- and the obvious response is to discard round 1.
    That is NOT done here, because the effect did not reproduce in this batch:
    vid-fp's round 1 was its SLOWEST of five (160.7 s against a 142.2 s median),
    and Czkawka's two rounds landed 1.6% apart. Dropping round 1 on a rule that
    did not hold would be discarding a fifth of the data to no end.

    The median, not the mean or the minimum. The fast tools were given extra
    rounds because their wall times are noisy (vid-fp spans 20% across five
    runs while its peak RSS moves 0.4%), and the minimum would convert those
    extra samples into a better-looking number rather than a better-estimated
    one.
    """
    return statistics.median([r[field] for r in records])


def fmt_time(s):
    return f"{s:.0f}s" if s < 90 else f"{int(s // 60)}m{int(s % 60):02d}s"


# ------------------------------------------------------------ accuracy side


def build_predictors(runs_dir, labels):
    """name -> proposed pair set, for every run present in `runs_dir`.

    Keyed by the same tool names `bench_tools.py` uses, so the cost and accuracy
    halves join on one string.
    """
    p = collections.OrderedDict()

    vidfp = os.path.join(runs_dir, "vidfp.json")
    if os.path.exists(vidfp):
        # Both, because they come from different code paths: `links` is what the
        # comparison stage measured, `groups` the closure a user reading a group
        # as an equivalence class would act on. Cliques make them equal today.
        p["vid-fp"] = score.vidfp_link_pairs(vidfp)
        p["vid-fp (groups)"] = score.group_pairs(vidfp)

    for name, fname in (("czkawka", "czkawka.json"),
                        ("vdf", "vdf.json"),
                        ("vdf-clip", "vdf-clip.json"),
                        ("vdf-par", "vdf-par.json")):
        path = os.path.join(runs_dir, fname)
        if os.path.exists(path):
            p[name] = score.group_pairs(path)
    return p


# The cost row a scored configuration is measured by. The group closure is
# derived from vid-fp's own output, so it costs exactly what vid-fp cost.
COST_KEY = {"vid-fp (groups)": "vid-fp"}

CLIP_SUPPORT = {
    "vid-fp": "always on",
    "vid-fp (groups)": "always on",
    "czkawka": "not supported",
    "vdf": "off (default)",
    "vdf-clip": "ON (audio)",
    "vdf-par": "off (default)",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(HERE, "out", "bench"))
    ap.add_argument("--runs-json", default=None,
                    help="cost records to read instead of <runs>/runs.json, "
                         "e.g. the merged file bench_merge.py writes")
    ap.add_argument("--labels", default=os.path.join(HERE, "labels.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", "bench"))
    ap.add_argument("--mode", default="exact", choices=("exact", "any", "both"))
    args = ap.parse_args()

    labels = score.load_labels(args.labels)
    counts = collections.Counter(r["relation"] for r in labels.values())
    print(f"labels: {len(labels)} judged pairs "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    by_tool = load_runs(args.runs_json or os.path.join(args.runs, "runs.json"))
    rounds = max(len(v) for v in by_tool.values())
    per = {t: len(v) for t, v in by_tool.items()}
    print(f"cost: cold cache every run, median over all rounds; "
          f"rounds per tool = " + ", ".join(f"{t} {n}" for t, n in per.items()))

    predictors = build_predictors(args.runs, labels)
    modes = ("exact", "any") if args.mode == "both" else (args.mode,)
    summary = {"rounds": rounds, "tools": {}}

    for mode in modes:
        spec = score.MODES[mode]
        pos = "+".join(sorted(spec["positive"]))
        print(f"\n=== positives = {pos} "
              f"(ignoring {', '.join(sorted(spec['ignored']))}) ===")
        hdr = (f"{'tool':<17}{'runtime':>9}{'CPU':>7}{'peakRSS':>9}"
               f"{'prop':>6}{'TP':>5}{'FP':>4}{'unjud':>6}"
               f"{'precision':>14}{'recall':>8}{'F1':>14}{'clips':>16}")
        print(hdr)
        print("-" * len(hdr))

        for name, proposed in predictors.items():
            s = score.Score(name, proposed, labels, mode)
            rec = by_tool.get(COST_KEY.get(name, name))
            if rec:
                wall = fmt_time(steady(rec, "wall_s"))
                cpu = f"{steady(rec, 'cpu_pct'):.0f}%"
                rss = f"{steady(rec, 'peak_rss_mb'):.0f}MB"
            else:
                wall = cpu = rss = "-"

            prec = (f"{100 * s.precision_floor:.1f}–{100 * s.precision:.1f}%"
                    if s.unjudged else f"{100 * s.precision:.1f}%")
            f1 = (f"{s.f1_floor:.3f}–{s.f1:.3f}" if s.unjudged
                  else f"{s.f1:.3f}")
            print(f"{name:<17}{wall:>9}{cpu:>7}{rss:>9}"
                  f"{len(s.proposed):>6}{len(s.tp):>5}{len(s.fp):>4}"
                  f"{len(s.unjudged):>6}{prec:>14}"
                  f"{100 * s.recall:>7.1f}%{f1:>14}"
                  f"{CLIP_SUPPORT.get(name, ''):>16}")

            entry = summary["tools"].setdefault(name, {})
            entry[mode] = {
                "proposed": len(s.proposed), "tp": len(s.tp), "fp": len(s.fp),
                "fn": len(s.fn), "unjudged": len(s.unjudged),
                "precision": s.precision, "precision_floor": s.precision_floor,
                "recall": s.recall, "f1": s.f1, "f1_floor": s.f1_floor,
            }
            if rec:
                entry["cost"] = {
                    "wall_s": steady(rec, "wall_s"),
                    "cpu_pct": steady(rec, "cpu_pct"),
                    "peak_rss_mb": steady(rec, "peak_rss_mb"),
                    "maxrss_mb": steady(rec, "maxrss_mb"),
                    "settings": rec[0]["settings"],
                    "per_round_wall_s": [r["wall_s"] for r in rec],
                    "per_round_peak_rss_mb": [r["peak_rss_mb"] for r in rec],
                }
        print(f"\n{score.Score('x', set(), labels, mode).n_positives} "
              f"positive pairs in the pool.")

    # Clip recall on its own: the reason the flag was turned on, and the axis
    # the headline SAME score deliberately ignores.
    clips = {k for k, r in labels.items() if r["relation"] == "CLIP"}
    print(f"\n=== CLIP pairs found ({len(clips)} labeled) ===")
    for name, proposed in predictors.items():
        hit = len(proposed & clips)
        print(f"{name:<17}{hit:>5}/{len(clips)}  {100 * hit / len(clips):5.1f}%")
        summary["tools"][name]["clip_pairs"] = hit
    summary["clip_pairs_labeled"] = len(clips)

    dest = os.path.join(args.out, "bench_summary.json")
    with open(dest, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {dest}")

    unj = {n: len(score.Score(n, pr, labels, "any").unjudged)
           for n, pr in predictors.items()}
    if any(unj.values()):
        print("\nunjudged proposals (outside the labeled pool -- precision is a "
              "range for these):")
        for n, v in unj.items():
            if v:
                print(f"  {n}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
