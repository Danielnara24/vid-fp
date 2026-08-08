# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`vid-fp` — a Linux CLI that finds duplicate videos and clips by perceptually fingerprinting their keyframes. Binary-only crate (`[[bin]]`, no `lib.rs`); the README is the user-facing spec and the man page is generated from the clap `Args` doc comments.

## Build and test

Building links against FFmpeg 6.x. On Debian/Ubuntu: `libavcodec-dev libavformat-dev libavutil-dev libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev clang pkg-config`. `ffmpeg-next` is pinned to 6.x, so a host with FFmpeg 7 will fail to link.

```bash
cargo build --release --locked     # release profile: lto + codegen-units=1; use it for any timing
cargo clippy --all-targets
cargo test                          # unit tests (in-file `mod tests`) + the accuracy integration test
cargo test --bin vid-fp share_for   # single unit test — tests live in the binary, so --bin, not --lib
cargo test --test local_accuracy_test -- --nocapture
```

`tests/local_accuracy_test.rs` is a machine-local regression harness: it shells out to `./target/release/vid-fp` against `/home/daniel/Documents/AN` with a ladder of `-d`/`-p` settings and asserts exact group/file counts against baseline CSVs stored in that folder. It **returns early and passes** when the directory is absent, so a green `cargo test` on another machine proves nothing about accuracy. Build release first — it runs the binary, not the crate.

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

## Invariants worth knowing before you change things

**Cache correctness is enforced by the table name.** `CACHE_TABLE` in `main.rs` is currently `fingerprints_dct_ct`; `SUPERSEDED_TABLES` lists the dead ones, dropped whole on first run. Appending a field to the end of `VideoFingerprint` is self-invalidating — old entries run out of bytes and bincode fails cleanly. **Changing what an existing field *means* (units, clock, hash algorithm) is not caught by anything**, so it requires renaming `CACHE_TABLE` and pushing the old name into `SUPERSEDED_TABLES`. Read the comment above those constants; each past rename records the bug that forced it.

**What belongs in the cache `Stamp`.** mtime (with nsec), size, and the sampling knobs (`--keyframe-interval`, `--min-keyframes`) — anything that changes *which frames get hashed*. Thread count is excluded on purpose (it changes only speed). `-d`, `-p` and `--min-duration` are comparison-time and must never enter the stamp.

**Decoder threads are apportioned by bytes, not by file count** (`share_for`, `ThreadBudget`). FFmpeg fixes a decoder's thread count when it is opened, so a heavyweight must be given its share at the moment it starts or never — hence the largest-first sort, the atomic cursor instead of rayon (rayon's splitting breaks the size ordering), and the blocking `claim`. A `Grant` releases its threads on drop, including on error and Ctrl-C.

**Memory is deliberately flat across a run.** Frames live in one contiguous `FRAME_STRIDE`-sized allocation per video so glibc serves it via mmap and returns it at once; `main()` pins the mmap/trim thresholds with `mallopt` to stop dynamic growth undoing that. Don't replace the frame buffer with a `Vec<Vec<u8>>`.

**DELETE always rests on a direct measurement.** A file is only marked DELETE if it directly matched the surviving copy; one that reached the group through a chain is REVIEW unless `--trust-chains`. Bit-derived metrics (quality, size) are compared **only within a codec** — a mixed-codec group ends with one survivor per codec, all flagged REVIEW. Both rules are load-bearing safety properties described in the README's Safety section.

**Nothing is destructive without `Disposal`.** `export.rs` cannot touch a file unless `main::disposal_for` constructed one from `--delete` / `--permanent` / `--move-to`. Every target is re-checked against its fingerprint immediately before it's acted on; a file that changed on disk is reported as CHANGED and left alone.

## Keeping the docs honest

`README.md`'s options table, the `Args` doc comments (which become the man page and `--help`), and the tuning/safety prose describe the same behaviour three times. Any flag or default that changes has to be updated in all three. Report formats are `.txt`/`.csv`/`.json`, dispatched on the `--output` extension in `export.rs`; the CSV column order is asserted by `tests/local_accuracy_test.rs` (it reads `full_path` at index 13), and by the golden `vd_results_*.csv` baselines beside the corpus, which were written by older builds — **append new CSV columns at the end** rather than inserting them, or both sides of that comparison stop lining up.

The JSON tree is built only when `--output` ends in `.json` (`wants_json` in `output_results`). It carries one object per measured link per file, so building it unconditionally made a report-only run pay for a structure it then discarded.

Releases are cut by pushing a `v*` tag — `.github/workflows/release.yml` builds the dynamic glibc binary, generates completions and the man page from the binary itself, and uploads both a versioned and a stable-named artifact. There is no CI on pushes or PRs, so `cargo clippy` and `cargo test` are on you.
