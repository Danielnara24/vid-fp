# CLAUDE.md

Guidance for Claude Code in this repository. This file lists the rules that hold **now**. The reasons for them, the bugs that produced them, the measurements, and every alternative that was tried and refused are in `docs/design-history.md`. Search that file for the identifier named here before re-trying something: most "obvious improvements" were measured and refused there.

When a change adds or alters a rule, write the rule here in a line or two and the story in `docs/design-history.md`. Don't put fix narratives in this file.

## What this is

`vid-fp` is a Linux CLI that finds duplicate videos and clips by perceptually fingerprinting their keyframes. Everything lives in the library (`src/lib.rs` and its modules); `src/main.rs` calls `vid_fp::main`. The desktop app `vid-fp-gui` (feature `gui`, `src/gui/`, notes in `src/gui/CLAUDE.md`) is a second binary over the same library, shipped as a Flatpak (`flatpak/CLAUDE.md`). The README is the user-facing spec; the man page and `--help` come from the clap `Args` doc comments.

## Build and test

- **Dynamic build** (development): links the system's shared FFmpeg. Debian/Ubuntu needs `libavcodec-dev libavformat-dev libavutil-dev libswscale-dev clang pkg-config`. `ffmpeg-next` is pinned to 8.1 on purpose: it compiles against FFmpeg 6, 7 and 8 headers. Don't lower it.
- **Static build** (what ships): vendored FFmpeg 8 + dav1d.

```bash
./scripts/build-ffmpeg-static.sh                     # ~10 min, once -> ./ffmpeg-static
export FFMPEG_DIR="$PWD/ffmpeg-static"               # per shell, never in a profile or .cargo/config.toml
cargo build --release --features static-ffmpeg
cargo test  --release --features static-ffmpeg
```

Rules that cost hours when broken:

- `FFMPEG_DIR` is required for a static build; `build.rs` refuses without it. Setting it globally makes dynamic builds try to link the static archives.
- Setting `FFMPEG_DIR` after `ffmpeg-sys-next` already built in a target dir changes nothing. Recover with `cargo clean -p ffmpeg-sys-next --release` (the profile flag must match). `build.rs` also emits a compile-time assert that the bindings' libavcodec version matches the prefix's `.pc`.
- Pass `--features static-ffmpeg` to `cargo test` too, or the test build overwrites `target/release/vid-fp` with a dynamic binary.

Other commands:

```bash
cargo clippy --all-targets          # must stay clean; CI runs it with -D warnings
cargo test --lib <name>             # unit tests live in-file (`mod tests`)
cargo test --features gui           # window tests + tests/window_matches_cli.rs (GTK 4.10+)
```

Unit tests decode `tests/fixtures/`, so they need a working FFmpeg at runtime.

**Accuracy is not checked automatically.** `tests/local_accuracy_test.rs` (gitignored) targets the AN corpus, which was deleted on 2026-11-17, so the test returns early and passes. CI checks compilation, clippy and unit tests only. Accuracy is measured by hand on the generated corpora VIDS and VIDS2 with the harnesses in `benchmark/`, scored against labels. Use timings and accuracy numbers from the static release build only; the dynamic build can differ (different FFmpeg, different decoder behaviour).

## Pipeline

`lib::run` makes one linear pass; every stage polls `utils::shutdown_requested()` and returns `Outcome::Interrupted` rather than unwinding.

1. **`sources::collect`**: walks roots, stdin or `--from-file`. Stats each file once and deduplicates on (device, inode), preferring a real name over a link. Returns a `Library` (files + walked folders + resolved excludes). Nothing downstream re-stats.
2. **Cache pass**: largest-first; every lookup from one `cache::View` snapshot before any decode.
3. **Weighing pass** (`fingerprint::weigh_decode`): opens each miss header-only. Prices the decode in keyframe-pixels and settles files that can't decode or fall under `--min-duration`.
4. **Content pass** (`scan/identify.rs`): BLAKE3 sample (size + three 64 KB stretches) to adopt fingerprints of moved/renamed files; full digests for byte-identical sets, decoded once (`Job::followers`). `--prune-cache` runs after this.
5. **Decode** (`fingerprint::fingerprint_video`, `fingerprint/keyframes.rs`): keyframes only, scaled to a 128x128 working frame (`SIDE`) by `fingerprint/downscale.rs`, cropped per frame, DCT hashes. Threads are apportioned by decode cost (`scan/budget.rs`).
6. **Compare** (`compare::find_all_matches`): phase 1 proposes candidate pairs (block index or flat scan); phase 2 measures coverage in milliseconds, in every orientation. The two routes must hand phase 2 the same pairs (both end with `any_frame`); `index_is_cheaper` picks between them on cost alone, with `PROBE_COST` fitted to measured phase-1 timings and the bin sweep sampled from the library's real bin counts. Refit `PROBE_COST` from timings of both routes, never from reasoning.
7. **Group** (`clustering`): greedy star cover; groups overlap and each asserts only "matched the representative".
8. **Report and dispose** (`export::output_results`): labels resolved once across groups (REVIEW > DELETE > KEEP), confirmation, disposal, report streamed to its sink.

Other entry points:

- **`--from-report`** (`lib::run_from_report`, `report.rs`): acts on a reviewed CSV/JSON report.
- **`--last`** (`kept.rs`): rewrites a report from what the last scan kept beside its cache.
- **`--against LIBRARY`** (`against.rs`): compares only across the two sides; the library is pinned KEEP.
- **The window**: runs `main_with` in a child (`vid-fp-gui --worker`), speaks JSON on stdout, and gets its labels from `export::verdicts`.

`scan::Ledger` is the only place a file's ending (fingerprint, skip, failure) is counted, logged or remembered. `stats::RunStats` separates **skips** (intentional) from **problems** (exit code 2).

## Matching rules

- `-d` is two thresholds (`Tolerance::for_distance`): `strict = -d` taken outright, `loose = -d + 1.5 sigma` taken only with witnesses at the same time offset (`compare::corroborated`). Neither is clamped; every rung must move. Thresholds are stated in sigma of Binomial(width, 1/2), never as raw bit counts.
- Witnesses are a maximum bipartite matching of distinct samples on both sides (`Witnesses::enough`), with a quota from the Binomial tail (`witness_schedule`, `CORROBORATION_BUDGET` and `EVIDENCE_ANCHOR_SIGMA` are one calibration). Greedy broke monotonicity; don't go back.
- **Raising `-d` or lowering `-p` must only ever add links.** `test_no_rung_of_the_tolerance_reports_less_than_the_one_below_it` and `test_a_wider_tolerance_never_takes_a_frame_match_away` guard it. Any new admission rule must read anchors from all frame matches within reach, never from the set of matches marked so far (that set shrinks as `-d` rises).
- With detail hashes on both sides, a pair is read over 128 bits (`Tolerance::wide`); the main hash alone must be within `cap`, which phase 1 is exhaustive to. Combined distance is always even.
- Defaults: `-d 6`, `-p 20`, `--min-keyframes 64`. `-d` caps at 32 (chance).
- `-p` gates on `max(coverage_a, coverage_b)` (this is how clips are found); zero coverage is refused separately so `-p 0` doesn't link everything.
- Orientations: plain, mirrored, and the reframings `Band`, `Zoom`, `Inset`, `Strip(k)` (portrait only), `Cut43` (near-4:3 only). Reframings are one-sided and probed both ways; plain wins ties.
- Featureless frames are dropped: AC energy under `MIN_AC_ENERGY` (50), or picture under a quarter of the frame either way. A file with no usable frame fails.
- Fingerprints are compared in path order, after `thin_samples` folds samples closer than 500 ms. The alignment window is the pair's median keyframe spacing.
- Never put per-file statistics into a hash (clips sample different frames); each frame is cropped by its own edges.
- Per-run facts go in `VideoFingerprint::measured` (boxed), not new inline fields; inline growth slowed phase 1 by 30%.

## Ranking and deletion safety

- `utils::GroupMaxima::rank` must be a total order (`test_the_ranking_is_a_total_order`). An unmeasurable metric scores level with the best, never as zero.
- DELETE needs a direct link to a group member that outranks it (`export.rs`). Labels are resolved globally, so chains delete.
- Mixed codecs: bit-derived metrics are compared within a codec only, and each codec's champion is held REVIEW. Unknown runtime: held REVIEW.
- Length band is 1 s absolute (`DURATION_TOLERANCE_SECS`). Bits count the video stream when every file of that codec states it, else whole files.
- Nothing is destructive without a `Disposal` from `lib::disposal_for`. `export::dispose_one` is the only destructive step; it checks size and (from a scan) ctime, and `still_holds` checks that something the file lost to remains.
- The confirmation comes after verdicts and before disposal; only an explicit yes proceeds; `--permanent` defaults to No.
- `--from-report`: refuses non-canonical paths and unknown action words, holds each row to its `lost_to`, and otherwise trusts the user's edits.
- `--move-to` loop guard (`Library::walk_reaches`) asks about the request, including symlink targets walked; a refused run removes the directories it created (`Conjured`).
- Hard links and symlinks: removals that don't free bytes are labelled UNLINKED (`frees_nothing`, mode-dependent).
- Report, log and cache must be three different files, and never a video (`refuse_shared_outputs`, `refuse_to_overwrite_an_input`).
- A completed run always writes its report, even an empty one (`conclude_without_comparing`).

## Cache

- One redb file, one table `cache::TABLE` (`files_ff8_cpath`, the same on every architecture), one `Entry` per path: stat, identity, verdict (fingerprint or `Refusal`). `cache::apply` is the only write path. Nothing outside `cache.rs` names a table or layout.
- **When a stored field changes meaning** (units, clock, hash, scaler, FFmpeg major), rename `TABLE` and push the old name into `SUPERSEDED_TABLES` (or carry the non-fingerprint parts over, as `PREVIOUS_TABLE` is). Refusal rules: bump `REFUSAL_RULES`. Sampling rules: bump `SAMPLE_RULES`. Appending a field to the fingerprint is safe (old records read as misses).
- `Stamp` holds mtime, size and the sampling knobs only. Never `-d`, `-p`, `--min-duration` or thread count.
- Transient failures are never remembered (`is_transient`'s errno list; EINVAL is deliberately not transient).
- One run per cache (exclusive lock for the whole run). `--from-report` may run without a locked cache.
- Compaction runs only when a prune actually removed entries.
- Only `--permanent` forgets cache entries.

## Decode details that must not regress

Every rule in `fingerprint/keyframes.rs` exists so a file fails out loud instead of fingerprinting half its length silently:

- the demux loop ends on EOF or 1024 consecutive errors;
- no seeking in `AVFMT_TS_DISCONT` containers;
- the sample clock accumulates gaps from DTS;
- the scaler is rebuilt when frame shape changes;
- `EAGAIN` drains and resends;
- one undecodable picture costs one sample.

**The scaler is ours** (`fingerprint/downscale.rs`): a port of libswscale's `FAST_BILINEAR` C path to full-range `GRAY8` (8-bit and planar 9-16 bit luma), bit-identical to swscale with cpu flags 0, so fingerprints are the same on every processor. Only packed/RGB/paletted/P010 keyframes still go through libswscale. Changing anything in it is a `cache::TABLE` rename. `test_the_port_matches_libswscale_s_c_path_bit_for_bit` (ignored; set `VID_FP_SCALER_CORPUS`) checks it against swscale on a real library.

Pictures are turned by the display matrix before hashing. Matroska runtimes come from the track `DURATION` tag. Errors from `fingerprint.rs` never name the file (callers do).

## Memory and threads

- Frames live in one contiguous allocation per video; `main` pins `mallopt` thresholds. Don't use `Vec<Vec<u8>>`.
- Phase 2's aligned-match list is banded past `MAX_ALIGNED_MATCHES` with no change in verdict.
- `-t` is clamped to `MAX_THREAD_OVERSUBSCRIPTION` x cores (minimum 16) in `main_with`; memory, not CPU, is what it really controls.
- Every quadratic hash loop (`any_within`, `for_each_frame_match`) is compiled a second time with POPCNT and that copy taken at runtime; baseline x86-64 `count_ones` made phase 2 about 6x slower per comparison. A new one needs the same.
- After corroboration, `admit` reads its anchors, rescue and credit passes from one held sweep (`Near`), and sweeps again only past `MAX_ALIGNED_MATCHES`.
- Per-frame hashing speedups must keep every bit (otherwise it is a `cache::TABLE` rename): the row pass is shared by rectangles spanning the same rows (`Hasher::band_rows`), `hash_of` selects its median, and the inset search's column test is SSE2 (`edge_columns`, held to `edge_columns_scalar`).

## Output and logging

- A report row is `export/render.rs::Row`; CSV columns and JSON keys are its field order. Readers look columns up by name; never index by position.
- Reports are streamed. `ReportStream` swallows write errors and records the first, because disposal has already happened by then.
- `-o -` is stdout; it replaces the console listing.
- Failures counted in a summary are logged with target `stats::COUNTED` and kept off the console (the log file gets everything).

## Keeping docs honest

The README options table, the `Args` doc comments and the README tuning/safety prose describe the same behaviour; change all three together.

`cargo clippy --all-targets` is clean with five `#[allow]`s, each justified at its site:

- `too_many_arguments` on `export::output_results`, `kept::Kept::of`, `compare::corroborate_pair_with` and `fingerprint::assemble`;
- `unnecessary_cast` in `fingerprint/keyframes.rs`.

Don't remove them by taking clippy's suggestion.

## Release

A `v*` tag runs `release.yml`: a static build in `manylinux_2_28` (glibc 2.28 floor), tests against the shipped artifact, a `DT_NEEDED` check (no dynamic libav*), an AV1 decode check (dav1d present), and the LGPL relink kit (a licence obligation; keep `--disable-gpl --disable-nonfree`). The aarch64 binary comes from `aarch64.yml`. Both workflows hold the fixtures' report to the same checksum, because both architectures fingerprint identically; refresh it in both places together.
