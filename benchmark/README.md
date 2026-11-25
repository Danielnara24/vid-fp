# Ground truth for the accuracy corpus

`tests/local_accuracy_test.rs` asserts that vid-fp still finds what vid-fp
used to find. That catches regressions and says nothing about whether the
answers were right. This directory builds the other thing: a pair-level
ground truth the tool had no hand in choosing, so recall and precision can
be measured, and so other tools can be measured on the same footing.

## The one rule

**Do not build the ground truth out of vid-fp's own candidates.** Anything
vid-fp never proposes gets labeled "not a duplicate" by omission, and every
competing tool's unique finds then score as false positives. vid-fp comes
out at ~100% recall by construction and the number means nothing.

So the pool unions several *independent* proposers and a human adjudicates
the union. Add every tool you intend to benchmark to the pool **before**
labeling — retrofitting one later means relabeling.

## Running it

```bash
cd benchmark
python3 gt.py sweep     # vid-fp across a -d ladder        (~6 s, warm cache)
python3 gt.py oracle    # independent dense-sample pass    (slow; see below)
python3 gt.py pool      # union everything -> pool.csv     (~10 s)
python3 gt.py sheets    # contact strips                   (~1 h cold)
python3 gt.py ui        # -> out/review.html
xdg-open out/review.html
```

`--corpus` defaults to `/home/daniel/Documents/AN`, `--out` to `./out`
(gitignored). Any `tools/*.json` is picked up automatically, so adding
another tool means dropping its output in that directory — the reader sniffs
the shape rather than parsing a format per tool.

`oracle` caches its sampled hashes in `out/oracle_hashes.json` against
mtime+size, so only new or changed files are re-decoded. The first run over a
corpus containing full episodes takes hours; later ones take seconds.
`sheets` likewise skips strips it has already rendered.

## Setting the loosest useful setting

Every proposer has a knob that trades recall for noise, and on this corpus
each one has a point past which it buys almost nothing. The test is
**corroboration**: how often another independent tool proposes the same pair.

vid-fp, by the `-d` at which a pair first appears:

| first `-d` | pairs | corroborated |
|-----------:|------:|-------------:|
| 2 | 374 | 67.1% |
| 6 | 44 | 54.5% |
| 10 | 51 | 27.5% |
| 12 | 218 | 6.9% |
| 14 | 1267 | **1.4%** |

`-d 14` bought 1267 pairs and 18 corroborated ones, so the sweep stops at 12
— one rung above the loosest profile the accuracy test uses. Czkawka told the
same story: `-t 15` gave 141 groups of which 123 spanned multiple folders
(one mixed bench press, chest fly, deadlift and an Instagram post), while its
default `-t 10` gave 74 groups, kept the full 19-file needle cluster, and
dropped a bogus three-episode group. Both rejected runs are kept under
`out/unused/` and `tools/unused/` rather than deleted.

## What the oracle refuses to propose, and why

A dense-sampling gradient hash is easy to write and easy to fool, and every
way it was fooled here produced *cross-folder* pairs at 100% coverage — the
shape of a genuine discovery. Each filter below was added after looking at
what it was admitting, and the oracle-only share of the pool fell from 772
pairs to 142 across them:

- **Mirrored matching is ignored.** A centred figure on a 9x8 grey grid
  matches its own reflection trivially, so on a corpus of centred figures it
  fires constantly; it produced no true positive in two full runs. `fa`/`fb`
  are still computed and reported, just not admitted on. VDF's `--ai-matching`
  is the right tool if mirrored copies matter.
- **Letterboxing is cropped out.** A pillarboxed clip downsamples to a nearly
  uniform grid whose hash is decided by noise. `chest fly machine_27.mp4` is
  286x556 of content in a wide black frame, and it read 100% against a nature
  documentary; with `cropdetect` it reads 0%.
- **Near-static files need several distinct hashes.** `tricep pushdown_11.mp4`
  has six sampled frames and two distinct hashes, so either one landing near
  anything made it 100% covered — the same trap single-keyframe files set for
  `--match-percent`.
- **A floor on seconds matched**, because a fraction of a 5-second file is not
  evidence, and **a ceiling on how many files one hash may reach**, because a
  hash can be unique and still sit in a region so crowded its tolerance ball
  spans dozens of unrelated videos.

Roughly 130 cross-folder false positives survive all of that, mostly bright,
low-contrast, near-static footage. Tightening further starts suppressing real
finds, which is the one thing this component must not do, so they are left in
for the reviewer to reject.

## Labeling

`relation` is one of:

- **SAME** — the same footage runs through both files, end to end.
- **CLIP** — one file's footage is contained in the other.
- **DIFFERENT** — not related.
- **SKIP** — you could not tell. Leave these out of the scoring denominator
  rather than guessing; a wrong label is worse than a missing one.

Judge from the frames. The coverage numbers are behind a disclosure triangle
on purpose: they are the thing being measured, so letting them decide the
label would measure nothing. Do not define SAME as a coverage threshold.

**Don't assume transitivity.** If A–B and B–C are SAME but A–C is not in the
pool, add it. Groups are derived from the labels afterwards; the labels are
not derived from groups.

**Recurring title sequences were labeled CLIP.** Every Wire episode opens with
the same ~85 seconds of credits, so any two episodes genuinely share that
footage and the matched-window strip shows it frame for frame. All 26
different-episode pairs in `labels.csv` are CLIP, noted "Episode intro
matches" — the shared footage is real, so the tools that find it are not
wrong.

The consequence to keep in mind when reading a score: under `any`
(SAME∪CLIP) those 26 pairs are positives, so a tool is rewarded for matching
on a title sequence alone. Under `exact` (SAME) they are ignored entirely and
cannot help or hurt. If you ever want "judge by whether the *programme* is
contained, not whether any footage is", relabel those 26 DIFFERENT and rerun
`score.py` — nothing else depends on the choice.

Review order is tiered: pairs several tools agree on come first, then folder
by folder so consecutive pairs share visual context.

Two corpus-specific traps:

- The exercise clips are one house animation style, white background,
  repeated presenters, one exercise per folder. `deadlift_1` through `_5` are
  all exactly 5.005 s and ~640 KB and are pixel-similar at low resolution —
  but `1`–`2` are one exercise and `3`–`5` are another.
- A whole-file strip across a 59-minute episode cannot show a 2-minute
  insert. Where a proposer reported *where* it matched, the UI draws that
  window as a second strip underneath, outlined in blue. Judge inserts from
  that row, not the overview.

## Output

The UI writes to `localStorage` as you go and exports `labels.csv` with the
same columns as `pool.csv`, `relation` and `notes` filled in. Import accepts
that file back, resolving columns **by header name** — the same rule
`report.rs` and `local_accuracy_test.rs` follow, so a spreadsheet may reorder
the layout freely. Export often anyway; `localStorage` is per-browser-profile
and is not a backup.

## Scoring

`score.py` reads `labels.csv` and each tool's output and reports pair-level
precision and recall. Tools report *groups*, so a group of n files is expanded
to its n(n-1)/2 pairs — a tool that puts A, B and C together is asserting A–C
as much as A–B. vid-fp is scored twice because it reports pairwise links
*inside* its groups: `links` is what it claims, `groups` is the closure a user
treating the group as an equivalence class would act on.

```bash
python3 score.py            # score out/default/*.json
python3 score.py curve      # vid-fp precision/recall across -d and -p
```

Each tool at its own defaults, over the 1258 labeled pairs:

| tool | proposed | runtime | peak RSS | precision | recall (SAME) | F1 (SAME) | F1 (SAME∪CLIP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| vid-fp `-d 4 -p 20` (links) | 433 | **1m37s** | 537 MB | 99.7% | **75.4%** | **0.859** | **0.832** |
| vid-fp (group closure) | 433 | same run | same run | 99.7% | **75.4%** | **0.859** | **0.832** |
| Czkawka `-t 10` | 351 | 28m04s | 604 MB | 88.0% | 65.6% | 0.752 | 0.646 |
| VDF `--threshold 5 --percent 96` | 236 | 2m45s | **228 MB** | **100.0%** | 50.1% | 0.668 | 0.561 |
| oracle | 562 | n/a | n/a | 65.1% | 66.5% | 0.658 | 0.675 |
| MFDF (GUI, user-run) | 16 | n/a | n/a | **100.0%** | 3.4% | 0.066 | 0.052 |

The two cost columns are cold-cache medians over the whole corpus, taken by
`bench_tools.py` — see **Measuring cost** below, and do not quote them without
reading it, because the obvious way to take them is wrong. The oracle and MFDF
have no comparable figure: the oracle is `gt.py`'s own sampling pass reading a
cache stamped on mtime+size rather than a tool invocation, and MFDF was run by
hand from its GUI.

VDF's output moves by one pair between identical runs (236 or 237), so its cost
and accuracy cells come from runs that differ by that much; differences that
small in any VDF figure mean nothing. It also compares only **743 of the 756
files it enumerates**, reproducibly, with nothing in its log saying why.

Every proposal from every tool is labeled, so there are no ranges here and no
`unjudged` column: precision is exact for all six. Where a tool proposes pairs
outside the labels, `score.py` shows precision and F1 as a range whose low end
counts each unjudged proposal as wrong.

Reproduced with:

```bash
vid-fp   /home/daniel/Documents/AN -r -o out/default/vidfp.json --yes
czkawka  video -d /home/daniel/Documents/AN -C out/default/czkawka.json -N
mkdir -p out/default/vdf-db      # NOT optional -- see below
vdf-cli  scan-and-compare -i /home/daniel/Documents/AN --db out/default/vdf-db \
         -f json -o out/default/vdf.json
```

Czkawka's default `-t 10` is the same setting the pool was built at, and its
output is byte-for-byte the pooled run. VDF's defaults are *stricter* than the
pooled run, so it proposes a subset. The oracle's row is `gt.py`'s own
admission thresholds, read back out of the pool's `sources` column.

**VDF's `--db` is silently ignored when the folder does not exist.** It falls
back to a `ScannedFiles.db` beside its own binary, and if that database has seen
the corpus before, the "scan" is a database read: 0.12 s for what should be
minutes, with a normal-looking progress bar and a normal-looking result file.
The tell is the hashing progress counting against the *database's* file count
rather than the scan's — "0/756" during a 69-file scan. Hence the `mkdir`.

### Measuring cost

`bench_tools.py` takes the runtime and peak RSS columns; `bench_parity.py`
checks every tool enumerated the same files; `bench_report.py` prints cost and
accuracy in one table, importing `score.py` rather than reimplementing it so the
two halves cannot drift.

```bash
python3 bench_tools.py  --runs out/rested --rounds 2 --cooldown 150
python3 bench_parity.py --runs out/rested
python3 bench_report.py --runs out/rested --mode both
```

Two settings there are load-bearing, and both were learned by getting them
wrong:

- **Cold cache, every run.** All three tools persist hashes between runs, so a
  second run measures a database lookup rather than a scan — a warm vid-fp pass
  over this corpus is **0.12 s** against 1m37s cold. `bench_tools.py` deletes
  each tool's cache before each run (and clears VDF's fallback database, per the
  trap above). It stashes whatever the machine already had first and restores it
  afterwards, so a warm cache that cost fifteen minutes of decode survives.
- **A cooldown between runs, or the short runtimes are wrong.** The first batch
  ran the five configurations back to back and measured vid-fp at 2m22s; rested,
  it is 1m37s. The distortion is thermal and scales with how hard a tool drives
  the CPU divided by how long it runs, so it cannot be corrected after the fact
  in one row:

  | tool | CPU | back to back | rested | overhang |
  |---|---:|---:|---:|---:|
  | vid-fp | 745% | 142.2 s | 97.1 s | **+46%** |
  | VDF `--parallelism 8` | 753% | 94.2 s | 65.2 s | **+45%** |
  | VDF | 252% | 180.6 s | 165.3 s | +9% |
  | Czkawka | 749% | 1821 s | 1685 s | +8% |
  | VDF `--partial-clip-detection` | 275% | 364.6 s | 349.2 s | +4% |

  The runs that saturate all eight cores *and* finish inside two minutes take
  ~45%; a 28-minute run at the same load takes 8%, because it spends nearly all
  its time in a steady thermal state whatever it started from; the runs that use
  only 2.5 cores have headroom and barely move. Correcting it made the headline
  gap **wider** — Czkawka went from 12.8× vid-fp to 17.4×.

The cooldown is also what makes the figures reproducible, which is how you know
it is fixing something real rather than shifting a mean: rested spreads are
vid-fp 1.7%, Czkawka 1.0%, VDF 0.3%, against 20% across five back-to-back vid-fp
runs. Adding rounds without the cooldown does not help — it tightens the
precision of a biased measurement, which is what the first attempt did.

Two more rules worth keeping:

- **Peak RSS must be summed over the process tree**, not read from
  `rusage.ru_maxrss`. Czkawka and VDF both shell out to `ffmpeg`, so rusage
  reports only the largest single process: Czkawka reads 100 MB there against
  604 MB for its tree, and VDF at `--parallelism 8` reads 197 MB against 1341 MB.
  `bench_tools.py` samples `/proc` every 150 ms and reports both.
- **RSS is the trustworthy metric here and wall time is not.** vid-fp's peak RSS
  moved 0.4% across every run in this session, hot or rested, while its wall time
  moved 46%. A memory regression shows up immediately; a timing one needs the
  protocol above.

Non-default configurations measured the same way, for the two questions the
defaults table cannot answer — whether clip detection is worth turning on, and
what VDF's single-threaded default costs it:

| configuration | runtime | peak RSS | precision | recall (SAME) | CLIP pairs |
|---|---:|---:|---:|---:|---:|
| VDF `--partial-clip-detection` | 5m49s | 226 MB | 97.5–97.9% | 50.1% | 15/134 |
| VDF `--parallelism 8` | 1m05s | 1341 MB | 100.0% | 50.1% | 0/134 |

VDF ships `--parallelism 1`, so its defaults row is single-threaded hashing
against eight threads for the other two; at eight it is the fastest tool here
and takes 2.5× vid-fp's memory. Raising it changed nothing about the answers —
the two runs matched pair for pair when compared inside one round — though VDF
moves by a pair between runs whatever the setting, so that is the resolution of
the claim, not an exact equality. Clip detection
doubles its runtime and finds 15 of the 134 CLIP pairs, against 0 at its
defaults and 77 for vid-fp. Its precision is the one range in this file: the
pool was labeled before that flag existed, so one of its proposals is unjudged.

### Reading these numbers honestly

- **Recall is measured against the pool, not the corpus.** A pair no proposer
  ever nominated is invisible to every tool alike, so this is *recall of the
  discoverable* and an upper bound on the true figure. Precision carries no
  such caveat: every pair a pooled tool proposed was labeled, so its errors are
  counted exactly. This is the standard pooling trade and it is why the pool
  unions independent proposers before anyone labels.
- **One cluster is 37% of the SAME positives.** The 19-file needle cluster is
  a complete clique of 171 pairs, and every tool but MFDF finds all of it —
  which lifts every headline recall by the same ~35 points. The per-kind
  tables `score.py` prints afterwards are the ones that separate the tools.
- **Pool every proposer you intend to score, including derived ones.** The
  vid-fp group closure was scored before it was pooled — the mistake this file
  warns about two sections up, committed on the tool it was written for — and
  62 of its proposals had no label, leaving its precision as a 15-point range.
  `tools/vidfp_closure.json` (a copy of the default run) now feeds the pool.
  Labeling those 62 put the true figure at 83.5%, against a 82.8% floor and a
  98.5% ceiling: 60 of the 62 were DIFFERENT, so the pessimistic end had been
  almost exactly right.
- **A tool added to the pool after labeling cannot be scored fairly** — its
  unique finds would land in `unjudged` wholesale. Pool first, label second.

### What the numbers say

**Clip detection is where the tools separate.** Czkawka, VDF and MFDF find
**0 of the 134 CLIP pairs** at their defaults. VDF has `--partial-clip-detection`
but it is off by default and audio-based; Czkawka does not model clips at all.
vid-fp finds 79 (links and closure alike), the oracle 81. Excluding clips
entirely, the four sit within 0.25 F1 of each other; including them, the gap
to Czkawka doubles again.

**vid-fp's default `-d 4` is a long way inside its own knee.** `score.py curve`
holds precision within a point of perfect all the way out to `-d 12`:

| `-d` (`-p 20`) | strict/loose | pairs | precision | recall (SAME) | F1 | (was, flat `-d`) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 / 6 | 327 | 100.0% | 55.6% | 0.715 | 0.406 |
| 2 | 2 / 8 | 383 | 100.0% | 65.4% | 0.791 | 0.684 |
| **4** (default) | 4 / 10 | 433 | 99.7% | 75.4% | 0.859 | 0.762 |
| 6 | 6 / 12 | 479 | 99.2% | 83.9% | 0.909 | 0.827 |
| 8 | 8 / 14 | 510 | 98.3% | 88.3% | 0.931 | 0.887 |
| 10 | 10 / 16 | 548 | 96.0% | 91.9% | **0.939** | 0.918 |
| 12 | 12 / 18 | 658 | 88.5% | 94.7% | 0.915 | 0.914 |
| 14 | 14 / 20 | 1143 | 81.0% | 96.2% | 0.880 | 0.887 |

The right-hand column is what the same corpus scored when `-d` was a single
flat tolerance. Every rung improves, and both thresholds keep moving — there is
no setting at which the flag stops doing anything, which an earlier version of
this rule got wrong by clamping the corroborated side at a fitted 12 bits.

**The honest comparison is at matched precision**, because where a corpus's knee
falls says more about the corpus than the rule:

| precision | flat `-d` recall | with corroboration |
|---|---|---|
| ~99.5% | 79.8% (`-d 8`) | **86.0%** |
| ~97% | 86.4% (`-d 10`) | **91.1%** |
| ~91% | 92.1% (`-d 12`) | **93.4%** |

**It is not a default-setting effect.** The same 15-cell grid before and after,
F1 on SAME:

| | `-p 5` | `-p 20` | `-p 50` |
|---|---:|---:|---:|
| `-d 2` | 0.722 → **0.805** | 0.684 → **0.791** | 0.350 → **0.757** |
| `-d 4` | 0.788 → **0.860** | 0.762 → **0.859** | 0.573 → **0.820** |
| `-d 8` | 0.902 → **0.932** | 0.887 → **0.931** | 0.827 → **0.904** |
| `-d 12` | 0.834 → *0.835* | 0.914 → **0.915** | 0.916 → **0.924** |
| `-d 16` | 0.727 → *0.723* | 0.817 → *0.806* | 0.892 → *0.877* |

Mean F1 0.773 → 0.847. Twelve of the fifteen cells improve; the three that do
not are all at `-d 16`, where both sides of the tolerance are now 16 and 22 bits
and the rule is doing exactly what the flag asked for. Nobody should be running
`-d 16` on a corpus like this, and the point of the mapping is that they get to
find that out by walking the knob rather than by hitting a wall.

The one false positive at the default is `romanian deadlift_{7,8}` — the same
exercise, presenter and white background, filmed twice.

**The group closure and the links are now the same set**, because groups went
back to being maximal cliques: every pair a group asserts is a pair the tool
measured, so the closure claims nothing extra and scores identically (470
proposals, 99.5% precision either way). The two rows are kept separate anyway,
because they are computed by different code paths and a future change to the
clustering rule would show up here as the two rows drifting apart.

The figures the connected-components build produced are worth keeping for
comparison: 464 proposals, 83.5% precision, 68.8% SAME recall, F1 0.754 /
0.748. Reading those groups as equivalence classes cost 16.5 points of
precision to buy 7 points of recall, and the 64 pairs it added were all
chains — 57 a needle clip reached through `S03E08 MODIFIED` to the unmodified
episode or to a half of it, the rest the `1.mp4`/`2.mp4`/`3.mp4` family,
consecutive segments of `full.mkv` that are each a genuine CLIP of the host and
unrelated to each other. Under SAME∪CLIP the closure edged ahead (0.748 vs
0.745); under SAME it did not.

### Auditing the DELETE rows, not just the pairs

`score.py` measures the pairs vid-fp proposes. `audit_deletes.py` measures what
it would *do* with them: every DELETE row is checked against the labels for the
files that survive in the groups it appears in.

```bash
python3 audit_deletes.py out/default/vidfp.json out/cliques/vidfp.json
```

At `-d 4 -p 20` with the corroboration rule in: 164 DELETEs, of which 114 are
labeled SAME as a file being kept and 45 are a CLIP inside a longer kept file.
4 sit in groups where every member is deleted, and each traces to a surviving
copy through hops the human labeled SAME.

The remaining **1 is wrong**: `romanian deadlift_7` is removed against a kept
file the human called DIFFERENT. It is the same false positive the pair score
reports. Nothing keeps the shorter half of a CLIP pair.

Earlier figures for comparison: 119 DELETEs from the connected-components
build, 128 from the clique build, 181 from the clamped version of this rule
(which made two of the wrong kind).

### The numpy lab

`lab/` reproduces `fingerprint.rs` and `compare.rs` in numpy — thumbnail, DCT,
median bits, featureless drop, run-length dedup, sample clock, per-sample
coverage — bit-for-bit against the binary's own hashes, so a matching-rule or
hash change can be scored against `labels.csv` in seconds instead of a 175 s
re-decode plus a Rust build. The `-d 2…16` curve it prints matches the binary
pair for pair.

It reads `out/frames/` (69 MB: `index.jsonl` plus one `f<seq>.raw` per file,
each 64×64 plane exactly as the fingerprinter saw it). Needs numpy, which the
system python does not have — use a venv.

```bash
python3 prep.py 16 /tmp/vidfp-lab/pp16.pkl   # per-pair sample matches, ~20 s
python3 witness_rules.py                     # witness-count rules over a -d ladder
python3 witness_budget.py                    # the shipped schedule, derived
```

Three things to know before trusting a number out of it:

- `pp16.pkl` holds pairs whose minimum distance is ≤ 16 bits, so a rule that
  reaches past that reads low — the lab said 808 proposals at `-d 20` where the
  binary found 896. Re-run `prep.py` with a wider cap or check in the binary.
- Witnesses must be drawn from matches within the rule's own widest distance,
  not from everything `prep.py` stored. Getting that wrong reads 95.7%
  precision where the truth is 99.5%.
- Read `F1f`, not `F1`, past `-d 12`: the unjudged count explodes there, and
  `F1` quietly assumes every unlabeled proposal is right.

`need_shipped` in `witness_rules.py` is the *flat* one-witness rule — what the
distance-graded schedule replaced above 12 bits, kept as the baseline to
compare against.

### Outputs

`out/score/per_pair.csv` has every judged pair and which tools found it —
sort by `found_by_count` to find the pairs only one tool saw. `summary.json`
has the same table in machine-readable form, and `curve.csv` the sweep.
