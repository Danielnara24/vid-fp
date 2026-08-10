# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`vid-fp` — a Linux CLI that finds duplicate videos and clips by perceptually fingerprinting their keyframes. Binary-only crate (`[[bin]]`, no `lib.rs`); the README is the user-facing spec and the man page is generated from the clap `Args` doc comments.

## Build and test

A plain build links the system's **shared** FFmpeg. On Debian/Ubuntu: `libavcodec-dev libavformat-dev libavutil-dev libswscale-dev clang pkg-config` — only those four libav\* libraries, since `ffmpeg-next` is pinned with `default-features = false` (avdevice and avfilter are unused and were dragging libavc1394 and libncursesw into `DT_NEEDED`).

`ffmpeg-next` is pinned to **8.1**, and that is deliberately newer than any FFmpeg you are likely to have. Only the crate's *forward* compatibility is broken: 8.1 compiles and runs correctly against FFmpeg 6, 7 and 8 headers because `ffmpeg-sys-next` detects the version and gates the wrapper on cfgs, whereas the 6.x crate fails against FFmpeg 7 headers with ~30 errors inside the dependency. Pinning the newest widens the host range; do not lower it to match what the release vendors.

The **released** binary is different: it statically links a vendored FFmpeg 8 + dav1d, so it needs no FFmpeg at all at runtime.

```bash
./scripts/build-ffmpeg-static.sh   # ~10 min, once; -> ./ffmpeg-static (gitignored)

cargo build --release --features static-ffmpeg              # what ships
cargo test  --release --features static-ffmpeg -- --nocapture
```

`build.rs` defaults to `./ffmpeg-static`, so no env var is needed day to day; `FFMPEG_DIR` overrides it for CI and for anyone relinking. **Do not export `FFMPEG_DIR` from a shell profile** — `ffmpeg-sys-next` reads it whether or not the feature is on, so a plain `cargo build` would then try to link the static archives and fail at `ld`.

**Pass `--features static-ffmpeg` to `cargo test` as well as `cargo build`.** `cargo test` builds the bin target too, so a bare `cargo test --release` silently overwrites `target/release/vid-fp` with a *dynamic* binary and then measures it against the static baselines — which shows up as a `4-20` failure reading 75/186 that looks exactly like a real regression. The harness prints a warning when it detects this, but the warning is easy to scroll past.

**Accuracy numbers only mean something from the static build**, because the golden CSVs were taken with it — see the accuracy-test note below.

```bash
cargo build --release --locked     # release profile: lto + codegen-units=1; use it for any timing
cargo clippy --all-targets
cargo test                          # unit tests (in-file `mod tests`) + the accuracy integration test
cargo test --bin vid-fp share_for   # single unit test — tests live in the binary, so --bin, not --lib
cargo test --test local_accuracy_test -- --nocapture
```

`tests/local_accuracy_test.rs` is a machine-local regression harness: it shells out to `./target/release/vid-fp` against `/home/daniel/Documents/AN` with a ladder of `-d`/`-p` settings and asserts exact group/file counts against baseline CSVs stored in that folder. It **returns early and passes** when the directory is absent, so a green `cargo test` on another machine proves nothing about accuracy. Build release first — it runs the binary, not the crate.

**Build it with `--features static-ffmpeg` before trusting the result.** The baselines were re-taken against the vendored FFmpeg 8, and the `4-20` profile is genuinely FFmpeg-version-sensitive: a dynamic build against FFmpeg 6 reports 75/186 there where the static one reports 74/184, because one pair's coverage falls by a single 0.5 s hash sample and drops under the 20% gate. The other two profiles agree across both. That profile has always sat within a percent or two of its gate — it is the same one that moved for the PTS→DTS fix — so treat a lone `4-20` diff as a question about which FFmpeg produced it before treating it as a regression.

`src/fingerprint.rs` tests decode `tests/fixtures/test_video.mp4`, so unit tests need a working FFmpeg at runtime too.

## Pipeline

`main::run` drives one linear pass; every stage polls `utils::shutdown_requested()` and returns `Outcome::Interrupted` rather than unwinding.

1. **`sources::collect`** — walks folders / reads stdin or `--from-file`, stats each file exactly **once**, and deduplicates on `(device, inode)` so symlinks, hard links and overlapping scan roots collapse to one entry. Everything downstream (cache stamp, sort, thread weights, prune) reads the `ScannedFile` fields; nothing re-stats.
2. **Cache pass** — files are sorted largest-first, then *every* cache lookup happens before *any* decode. This split is deliberate: interleaving them under rayon let one slow decode block cached neighbours in the same work slice.
3. **`fingerprint::fingerprint_video`** — decodes keyframes only, box-filters each to 16×16, DCT-transforms, and keeps the 8×8 low-frequency corner as a 64-bit hash (exactly 32 bits set, hence only even `-d` values are meaningful). Featureless frames (black, fades, title cards) are dropped below `MIN_AC_ENERGY` so they can't link unrelated files. Sample times are **milliseconds of decode time anchored at the first keyframe**, not PTS — MP4 `ctts` skews PTS once non-key packets are discarded.
4. **`compare::find_all_matches`** — two phases. Phase 1 builds a `BlockIndex` over each of 4 16-bit hash blocks and proposes candidate pairs (probe radius capped at 1 by `MAX_PROBE_RADIUS`; exhaustive to `-d 7`, a filter above that). Phase 2 compares every proposed pair exactly. Coverage is **directional and measured in milliseconds of runtime**, not in frame counts — a hash stands for the span until the next sample, so two encodes sampling at different rates still agree on how much footage overlaps.
5. **`clustering::find_duplicate_groups`** — connected components via a disjoint set. A group asserts "a chain of matches runs through these", not "these are pairwise copies". Adds no thresholds of its own.
6. **`export::output_results`** — resolves KEEP/DELETE/REVIEW entirely inside one group (safe, because components partition the files), performs the disposal, and returns the paths that moved so `main` can drop their cache entries.

`utils` holds the ranking rules (`GroupMaxima`, `find_best`), formatting, and the shutdown flag. `stats::RunStats` tallies **skips** (intentional: `--min-duration`, excludes, inode aliases) separately from **problems** (the run did less than asked) — only problems set exit code 2.

**`--from-report` is a second entry point that skips all of it.** `run` branches to `main::run_from_report` before anything touches the filesystem; `report::apply` reads the CSV an earlier run wrote, takes the rows whose `action` column says `DELETE`, and disposes of exactly those. It shares four things with the grouped path and nothing else: `confirm::approve`, `export::dispose_one`, `export::disposed_line`/`trouble_lines`, and `cache_forget`. It works because the destructive step only ever consults a path and a size, and both are CSV columns — see the `dispose_one` invariant below.

## Invariants worth knowing before you change things

**Cache correctness is enforced by the table name.** `CACHE_TABLE` in `main.rs` is currently `fingerprints_dct_ct_ff8`; `SUPERSEDED_TABLES` lists the dead ones, dropped whole on first run. Appending a field to the end of `VideoFingerprint` is self-invalidating — old entries run out of bytes and bincode fails cleanly. **Changing what an existing field *means* (units, clock, hash algorithm) is not caught by anything**, so it requires renaming `CACHE_TABLE` and pushing the old name into `SUPERSEDED_TABLES`. Read the comment above those constants; each past rename records the bug that forced it.

**The vendored FFmpeg version is one of those meanings.** The `_ff8` suffix is the last rename: decoder output is not bit-identical across FFmpeg majors, and the `Stamp` records mtime, size and the sampling knobs but deliberately *not* an FFmpeg version — so entries written either side of a version change are indistinguishable to a lookup. Bumping `FFMPEG_VERSION` in `scripts/build-ffmpeg-static.sh` across a major therefore costs a `CACHE_TABLE` rename and a re-taken accuracy baseline, exactly like changing the hash algorithm would. It is not a routine dependency bump.

**What belongs in the cache `Stamp`.** mtime (with nsec), size, and the sampling knobs (`--keyframe-interval`, `--min-keyframes`) — anything that changes *which frames get hashed*. Thread count is excluded on purpose (it changes only speed). `-d`, `-p` and `--min-duration` are comparison-time and must never enter the stamp.

**Decoder threads are apportioned by bytes, not by file count** (`share_for`, `ThreadBudget`). FFmpeg fixes a decoder's thread count when it is opened, so a heavyweight must be given its share at the moment it starts or never — hence the largest-first sort, the atomic cursor instead of rayon (rayon's splitting breaks the size ordering), and the blocking `claim`. A `Grant` releases its threads on drop, including on error and Ctrl-C.

**Memory is deliberately flat across a run.** Frames live in one contiguous `FRAME_STRIDE`-sized allocation per video so glibc serves it via mmap and returns it at once; `main()` pins the mmap/trim thresholds with `mallopt` to stop dynamic growth undoing that. Don't replace the frame buffer with a `Vec<Vec<u8>>`.

**DELETE always rests on a direct measurement.** A file is only marked DELETE if it directly matched the surviving copy; one that reached the group through a chain is REVIEW unless `--trust-chains`. Bit-derived metrics (quality, size) are compared **only within a codec** — a mixed-codec group ends with one survivor per codec, all flagged REVIEW. Both rules are load-bearing safety properties described in the README's Safety section.

**Nothing is destructive without `Disposal`.** `export.rs` cannot touch a file unless `main::disposal_for` constructed one from `--delete` / `--permanent` / `--move-to`. `--from-report` is no exception: it bails with an explanation rather than reading a report it could not act on.

**`export::dispose_one` is the only copy of the destructive step**, and it takes a path and the size that file was measured at — nothing else. That narrowness is what made `--from-report` cheap, and it is worth preserving: widening it to need a `VideoFingerprint` again would mean the report mode either re-fingerprints or grows a second, weaker version of the staleness check. Both callers pass a size from the same moment the delete decision was made (the scan; the report's `size_bytes` column), and a file that no longer matches it is reported CHANGED and left alone.

**A report's DELETE rows are the user's decision, not the tool's.** The safety rules below (direct measurement, one survivor per codec, one KEEP per group) are enforced by the run that *writes* a report. `report.rs` deliberately re-imposes none of them — a group may be emptied completely — because the file has been edited by then and second-guessing it would defeat the point, which is adjudicating REVIEW rows. What it does keep is the confirmation prompt, the size check, and a hard rule that an action word it doesn't recognise is never guessed at: the file is left alone and `stats.report_unusable` fails the run. Columns are located by header name, which is what lets the CSV layout be chosen for whoever reads it: reports written by older builds, and reports a spreadsheet handed back reordered, replay identically.

**The confirmation sits between the two passes of `output_results`, not at start-up.** `confirm::approve` is asked once the DELETE set is resolved, because that is the first moment there is a count and a byte total to show. Declining does not abort: it rebinds `disposal` to `None` for the rest of the function, so the labels, the summary and the JSON all describe the run the filesystem actually saw. It is interactive-only (`stdin` *and* `stderr` must be terminals) so a pipeline can never block on it, `--yes` skips it, and everything that isn't an explicit yes — EOF, a read error, an interrupt, three unparseable answers — leaves the files alone.

## Keeping the docs honest

`README.md`'s options table, the `Args` doc comments (which become the man page and `--help`), and the tuning/safety prose describe the same behaviour three times. Any flag or default that changes has to be updated in all three. Report formats are `.txt`/`.csv`/`.json`, dispatched on the `--output` extension in `export.rs`.

**No column position is load-bearing, and it is worth keeping it that way.** The CSV order used to be frozen by two readers that indexed into it, so every new field was bolted onto the right-hand end regardless of what it meant, and `action` ended up five columns from the link data that justifies it. Both readers now resolve columns by header name — `report::column` and `local_accuracy_test::get_duplicate_files` — which is also what lets the golden `vd_results_*.csv` baselines beside the corpus stay useful: they were written by older builds with a different, shorter layout. Reorder the report freely; just never re-introduce an index.

The three formats are one layout expressed three ways, and `export.rs` writes them from a single pass, so a field added to one is added to all three in the same place. The JSON's key order is only preserved because `serde_json` carries the `preserve_order` feature — without it serde_json's `Map` is a `BTreeMap` and the tree comes out alphabetized, which interleaves each figure's raw and formatted halves and buries `summary` under the whole results array.

The JSON tree is built only when `--output` ends in `.json` (`wants_json` in `output_results`). It carries one object per measured link per file, so building it unconditionally made a report-only run pay for a structure it then discarded.

Releases are cut by pushing a `v*` tag — `.github/workflows/release.yml` runs `scripts/build-ffmpeg-static.sh`, builds the static binary against it, generates completions and the man page from the binary itself, and uploads a versioned copy, a stable-named copy, the extras tarball and the LGPL relink kit. There is no CI on pushes or PRs, so `cargo clippy` and `cargo test` are on you.

**`cargo clippy --all-targets` is clean, and three `#[allow]`s are what keep it that way.** Each carries its reasoning at the site; the short version is that all three lints are asking for a rewrite that would cost something real. `clippy::unnecessary_cast` on `AV_CODEC_FLAG2_FAST as i32` (`fingerprint.rs`) — the FFmpeg flag macros have no declared type, so bindgen infers one per `#define`, and within a single header the `AV_CODEC_FLAG_*` family lands on `c_uint` while most of `AV_CODEC_FLAG2_*` lands on `c_int`; the cast is redundant only for as long as this flag keeps its current bit. `clippy::needless_range_loop` on the variance/auto-crop scan (`fingerprint.rs`) — `y` and `x` index two flat 64×64 buffers *and* two 64-entry projections, one of which is re-walked per row. `clippy::too_many_arguments` on `output_results` — the two parameters that shared an axis are already `Policy`, and a bag for the rest would only hide which of them the destructive pass reads. Do not "fix" these by deleting the comment and taking clippy's suggestion.

Two release steps are load-bearing rather than decorative, and both guard failures that are silent in ordinary testing: one greps `DT_NEEDED` for `libav*`/`libsw*` and fails if the binary linked FFmpeg dynamically after all, the other fingerprints a generated AV1 clip. FFmpeg's built-in AV1 decoder is hwaccel-only, so a configure that quietly drops `libdav1d` yields a binary that passes every other test and cannot fingerprint a single AV1 file.

**The LGPL relink kit is a licence obligation, not an extra.** FFmpeg is LGPL 2.1+ and the link is static, so §6 requires shipping the means to relink — hence the `.a` archives, headers, build script and `RELINKING.md`. Keep `--disable-gpl --disable-nonfree` in the configure line: `vid-fp` is offered under Apache-2.0, which is incompatible with GPL-2. See `THIRD-PARTY-LICENSES.md`.
