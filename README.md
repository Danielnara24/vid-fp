# vid-fp

Fast video **duplicate and clip finder** for Linux. It fingerprints videos from
their keyframes and groups together files with the same content, even when they
differ in resolution, file size, or container, and even when one video is only a
**trimmed clip inside another**. It reports duplicate groups and can optionally
move the redundant copies to the trash.

> **Note:** `vid-fp` can delete files. By default it only reports. Deletion
> happens *only* when you pass `--delete`, it asks you to confirm first, and
> files go to the system trash (recoverable) unless you also pass `--permanent`.
> See [Safety](#safety).

## Requirements

Linux, x86_64, **glibc 2.35 or newer** (Ubuntu 22.04+, Debian 12+, Fedora 36+,
current Arch and Mint). The release binary bundles FFmpeg 8 and dav1d, so you do
**not** need FFmpeg installed.

## Installation

### Prebuilt binary (recommended)

```bash
curl -L -o vid-fp \
  https://github.com/Danielnara24/vid-fp/releases/latest/download/vid-fp-x86_64-linux-gnu
chmod +x vid-fp
sudo install -m 755 vid-fp /usr/local/bin/vid-fp
```

That URL always points at the newest version. Prefer not to use `sudo`? Install
into `~/.local/bin` instead and make sure that directory is on your `PATH`.

Each release also ships a `.sha256` file if you want to verify the download:

```bash
sha256sum -c vid-fp-x86_64-linux-gnu.sha256
```

### From source

Requires the Rust toolchain plus the FFmpeg **development** libraries. FFmpeg 6,
7 and 8 all work:

```bash
# Debian / Ubuntu / Mint
sudo apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
                 clang pkg-config

# Fedora
sudo dnf install ffmpeg-devel clang pkgconf-pkg-config

# Arch
sudo pacman -S ffmpeg clang pkgconf
```

```bash
cargo install --git https://github.com/Danielnara24/vid-fp
```

This links your system's shared FFmpeg, so the binary is small but tied to that
FFmpeg. Its fingerprints may differ very slightly from the released binary's,
which pins FFmpeg 8, so don't point a source build and a release build at the
same cache expecting agreement.

To reproduce the release exactly instead, self-contained and with AV1 support:

```bash
git clone https://github.com/Danielnara24/vid-fp && cd vid-fp
./scripts/build-ffmpeg-static.sh                  # ~10 min, once
cargo build --release --features static-ffmpeg    # -> target/release/vid-fp
```

Needs `git make nasm pkg-config python3 python3-venv build-essential`. The
script builds pinned FFmpeg and dav1d releases into `./ffmpeg-static` and is a
no-op on later runs.

### Shell completions and man page

Each release ships `vid-fp-<version>-extras.tar.gz` with completions for bash,
zsh, and fish plus a man page:

```bash
tar -xzf vid-fp-*-extras.tar.gz
sudo install -Dm644 completions/vid-fp.bash /usr/share/bash-completion/completions/vid-fp
sudo install -Dm644 completions/_vid-fp     /usr/share/zsh/site-functions/_vid-fp
sudo install -Dm644 completions/vid-fp.fish /usr/share/fish/vendor_completions.d/vid-fp.fish
sudo install -Dm644 man/vid-fp.1            /usr/share/man/man1/vid-fp.1
```

Without `sudo`, into your home directory instead (for zsh, use any directory
already on your `fpath`):

```bash
install -Dm644 completions/vid-fp.bash ~/.local/share/bash-completion/completions/vid-fp
install -Dm644 completions/vid-fp.fish ~/.config/fish/completions/vid-fp.fish
install -Dm644 man/vid-fp.1            ~/.local/share/man/man1/vid-fp.1
```

A source build can generate both itself:

```bash
vid-fp --completions bash | sudo tee /usr/share/bash-completion/completions/vid-fp >/dev/null
vid-fp --man | sudo tee /usr/share/man/man1/vid-fp.1 >/dev/null
```

Open a new shell (bash, fish) or run `compinit` (zsh) to pick them up. Then
`man vid-fp` is the full offline reference, and tab-completion fills in flags,
folder paths, and enum values:

```bash
vid-fp ~/Videos -e ~/Down<TAB>   # completes to a real folder
vid-fp ~/Videos -k <TAB>         # length  resolution  quality  size
```

## Updating

Use the same commands as installing; they overwrite the last version. From
source, re-run `cargo install` with `--force`. Run `vid-fp --version` to compare
against the [latest release](https://github.com/Danielnara24/vid-fp/releases/latest),
and reinstall the completions and man page too.

## Usage

Point it at one or more folders:

```bash
# Report duplicates in a folder and all its subfolders
vid-fp ~/Videos -r

# Scan several folders, excluding one
vid-fp ~/Videos ~/Downloads -e ~/Downloads/keep -r

# Write a report you can open later
vid-fp ~/Videos -r -o results.csv
```

By default the scan is **not** recursive. Add `-r` to descend into subfolders.

Files are identified by inode, not by path, so a file reachable through a
symlink, a hard link, or two overlapping scan folders is fingerprinted and
reported once. Symlinked *folders* are skipped unless you pass
`--follow-symlinks`.

### Naming files, and reading a list

Individual files can be named alongside folders, and are scanned whatever their
extension:

```bash
vid-fp ~/Videos/episode_a.mkv ~/Videos/episode_b.mp4 ~/Downloads
```

A list of paths can also be read from stdin with `-`, or from a file with
`--from-file`:

```bash
fd -e mkv --changed-within 30d ~/Media | vid-fp -
find ~/Media -type f -size +500M | vid-fp -
vid-fp --from-file suspects.txt -o results.csv
```

Blank lines are ignored and a trailing carriage return is trimmed. For filenames
containing newlines, use NUL separators on both ends:

```bash
find ~/Media -name '*.mkv' -print0 | vid-fp - -0
```

Paths read this way behave exactly as if typed as arguments. Files that aren't
videos are reported under `Problems`, so filter with `fd -e`/`find -name` rather
than piping an entire tree.

### Deleting duplicates

```bash
# Move the files marked DELETE to the trash
vid-fp ~/Videos -r --delete

# Remove them permanently (irreversible)
vid-fp ~/Videos -r --delete --permanent
```

Once the groups are resolved, an armed run stops and asks before touching
anything. Return accepts. Answering `n` doesn't abort the run, it demotes it to
a report-only one, so you still get the full table and the reclaimable figure
without re-scanning. Pass `-y`/`--yes` to skip the prompt.

The prompt only appears when there's a terminal on both stdin and stderr, so it
can never block a script or a cron job.

### Moving instead of deleting

The system trash needs a trash directory on the file's own filesystem, which
external drives, NFS mounts and headless servers frequently lack. Use
`--move-to` in those cases:

```bash
vid-fp /mnt/media -r --move-to /mnt/scratch/dupes
```

This isn't a deletion, so `--move-to` arms the run by itself. If `--delete` or
`--permanent` is passed alongside it, the files are still moved and nothing is
removed.

Each file's **absolute path is recreated inside the destination**:
`/mnt/media/show/ep01.mkv` lands at `/mnt/scratch/dupes/mnt/media/show/ep01.mkv`.
Two files with the same name never collide, you can see where each came from,
and the whole run is undone with a single copy back:

```bash
cp -a /mnt/scratch/dupes/. /
```

Nothing is ever overwritten: an occupied destination slot is reported as a
problem instead. A destination on another filesystem is copied, flushed, and
only then unlinked; if either step fails the original stays where it was.
Timestamps and permissions are preserved.

The destination must sit outside the scanned folders, or the run aborts.

### Acting on a report you have reviewed

Save a report, review it, edit it, then hand it back:

```bash
# 1. Look first
vid-fp /mnt/media -r -o dupes.csv

# 2. Edit the action column in dupes.csv, then act on exactly what it says
vid-fp --from-report dupes.csv --delete
```

`--from-report` disposes of every row whose `action` column reads `DELETE` and
touches nothing else. Change `REVIEW`/`KEEP` cells to `DELETE` and they're acted
on; change a `DELETE` to anything else and it isn't. If you mark every copy of
something `DELETE`, every copy is deleted. The confirmation prompt still shows
the count and the byte total first.

Every file is re-checked against the size the report recorded immediately before
it's touched, and left alone if it has changed, so a report can sit for a week
before you get to it. Columns are found by name, so a reordered report still
works. Only CSV reports are accepted; `.txt` and `.json` are refused.

## Options

| Flag | Description | Default |
| --- | --- | --- |
| `<PATH>...` | Folders and/or video files to scan (required). `-` reads a list of paths from stdin | |
| `--from-file <FILE>` | Read the paths to scan from a file, one per line (`-` = stdin) | |
| `-0`, `--null` | Paths in the list are NUL-separated, for `find -print0` / `fd -0` | off |
| `-r`, `--recursive` | Include subfolders | off |
| `--follow-symlinks` | Descend into symlinked folders | off |
| `-e`, `--exclude <FOLDER>` | Exclude a folder; repeat for several | |
| `-x`, `--extensions <EXT>` | Video extensions to include, comma-separated or repeated | `mp4,mkv,avi,mov,flv,webm` |
| `-d`, `--hamming-distance <N>` | Frame-match tolerance, in bits out of 64; higher = less strict matching. Raise to find more duplicates, at the cost of false positives. See [Tuning](#tuning). Values above `32` are refused | `4` |
| `-p`, `--match-percent <F>` | Min % of overlap to count as a duplicate, from `0` to `100`. Lower includes shorter matches, at the cost of false positives | `20.0` |
| `--min-duration <SECS>` | Min shared clip length in seconds for a match. Videos shorter than this are skipped entirely. `0` = off | `0.0` |
| `-k`, `--priority <P>` | Criteria for KEEPING files: `length`, `resolution`, `quality`, or `size`. The chosen one is compared first, the rest follow in the default order. See [Codecs and quality](#codecs-and-quality) | `length` |
| `--keyframe-interval <F>` | Seconds between sampled keyframes (`0` = every keyframe). Higher is faster, but makes short matches harder to find | `0.0` |
| `--min-keyframes <F>` | Min keyframes kept for short videos (only relevant when keyframe-interval > 0) | `4.0` |
| `-o`, `--output <FILE>` | Optional path to save the report: `.txt`, `.csv`, or `.json` | |
| `--delete` | Move files marked DELETE to the trash | off |
| `--permanent` | With `--delete`, permanently remove instead | off |
| `--move-to <DIR>` | Move the files marked DELETE under `DIR`, recreating their absolute paths inside it. Arms the run on its own and supersedes `--delete`/`--permanent` | |
| `--from-report <FILE>` | Act on a CSV report from an earlier run instead of scanning. Requires `--delete` or `--move-to`. See [above](#acting-on-a-report-you-have-reviewed) | |
| `-y`, `--yes` | Answer yes to the confirmation shown before any file is touched | off |
| `-t`, `--threads <N>` | Worker threads (`0` = uses all cores) | `0` |
| `-q`, `--quiet` | Only print errors | off |
| `--clear-cache` | Wipe ALL vid-fp cache before running | off |
| `--prune-cache` | Drop cached entries only for files not in this scan | off |
| `--completions <SHELL>` | Print a completion script for `bash`, `zsh`, `fish`, `elvish`, or `powershell` and exit | |
| `--man` | Print the man page (roff) and exit | |

## Tuning

The two knobs that decide what counts as a duplicate are `-d` (how different two
frames may be) and `-p` (how much of a video must match). They trade off against
each other, so change one at a time.

`-d` counts differing bits of a 64-bit frame hash. Two unrelated frames sit
about 32 bits apart, so the useful range is roughly 2 to 12: below that only a
bit-identical re-encode matches, above it unrelated footage starts to.

**Only even values of `-d` do anything.** Every hash has exactly 32 of its 64
bits set, so any two of them differ in an even number of places. `-d 5` accepts
exactly what `-d 4` accepts. Step in twos.

**A frame match is judged on its distance and on whether anything backs it up.**
Two encodes of the same footage place every frame they share at one constant
time offset, so their matches corroborate each other; two videos that merely
look alike produce matches scattered across unrelated moments. So `-d` sets two
thresholds: a match with nothing behind it must be within `-d`, while a match
that another frame match agrees with (a different frame of each video, landing
within half a second of the same offset) may reach `-d + 6`. Past 12 bits one
agreeing match stops being rare enough to mean anything, so more are required:
two at 14 bits, three at 16, four at 20.

Both thresholds move with every rung of `-d`, so the flag stays a sensitivity
control across its whole range. The default is deliberately conservative:
raising it finds more, and starts admitting false positives past about `-d 10`.

**Both knobs are monotone.** Raising `-d` or lowering `-p` only ever *adds*
files to the report; lowering `-d` or raising `-p` only ever *removes* them. The
report at any setting is a subset of the report at every looser one, so start at
the defaults and walk one knob outward until you see something you don't
recognise. The one exception is a `-d` loose enough to link nearly everything,
where a set of files can form too many overlapping groups to report and is left
out of it entirely and listed under `Problems` instead.

**Not finding duplicates you expect?** Raise `-d`, or lower `-p` if the match is
a short clip inside a long video. Two encodes of the same footage only line up
frame-for-frame when their keyframes do; when they don't, each file samples
moments the other never looked at, and no tolerance can bridge a frame that was
never sampled. Expect a group like that to report well under its full runtime as
shared even though the files are identical end to end.

**Getting false positives?** Lower `-d` first, it's the blunter of the two. Dark
scenes, fades, and letterboxed content look alike to any perceptual hash.
(Frames with no structure at all are dropped rather than hashed, so black frames
and plain title cards can't link anything on their own.)

`--min-duration` is an absolute floor, in seconds, on how much footage two files
must share. Both gates apply, so `-p 5 --min-duration 60` means "at least 5%
overlap *and* at least a minute of it". It compares against the pair's
conservative shared estimate, the lower of the two files' figures.

It also skips fingerprinting anything shorter than the floor, since such a file
can't contain a long enough shared clip. Videos whose duration the container
doesn't report are never skipped. Changing this flag doesn't invalidate the
cache.

## Codecs and quality

These rules decide which copy is kept, not which files match. Matching itself is
codec-blind, so an AV1 encode and an H.264 encode of the same footage land in
the same group.

**`quality` is bits per frame**, bitrate divided by the average frame rate.
Ranking on raw bitrate would prefer whichever copy simply had more frames in it,
since a 60 fps copy needs roughly twice the bitrate of a 30 fps one to look the
same.

**Bits are never compared across codecs.** A modern codec's job is to carry the
same picture in fewer bits, so an AV1 file half the size of an H.264 one is not
"worse". Both `quality` and `size` are only compared between files sharing a
codec; across codecs they tie and the decision falls through to something else.

**A group that spans codecs ends with one survivor per codec.** If the leading
copies match on length and resolution but use different codecs, nothing
comparable remains to rank them, so each codec keeps its own best copy, flagged
REVIEW for you to choose between. The other copies of each codec lost to a file
they *are* comparable with, so they're marked DELETE as usual: five HEVC encodes
and three H.264 encodes of one episode end up as two files to look at, not
eight.

Note that bitrate (and so quality) includes audio, so a copy with lossless 5.1
can outrank one with a better video track and stereo AAC.

Every report shows the codec, frame rate, size, bitrate and quality of each
file.

## Reading the results

**Every file in a group matched every other file in it.** A group is the tool's
evidence for deleting something, so it never asserts a comparison it did not
make.

**A file can appear in more than one group**, but its action is decided once for
the file, not once per group. A file marked DELETE in one group reads DELETE in
every group it appears in, and a file held for REVIEW anywhere is held
everywhere. So if B is redundant against A and C is redundant against B, both B
and C go in a single pass and A is what's left, rather than needing one re-run
per hop. That's a reason to read a dry run's report before passing `--delete`.

**The matched figure is footage, not frames.** It is how much of *this file's own
runtime* was found in the group member it matched most closely, in seconds, so a
pair linked only by a common title card reads as the second or two that card
lasts. Read it against the file's own length: that ratio is what separates a
re-encode from a shared clip.

Every figure on a row describes that row's file. On a genuine match both ends
agree anyway: a two-minute clip inside a twenty-two minute episode is 100% of
the clip and 9% of the episode, and both rows read two minutes. Where two rows
disagree, one file was found almost entirely inside the other while the other
was barely covered back, which is usually a sampling artifact.

**The `samples` column is how many frame hashes the file's fingerprint holds**,
after featureless frames have been dropped. A file with few samples has each one
standing for a long stretch of runtime, so its matched footage comes out coarse.
At the limit, a file with one sample has that sample standing for its entire
runtime, so any match at all covers 100% of it and no `--match-percent` can gate
it. If a row reports far more matched footage than the file on the other end of
the pair, check this column first.

Each row reports the *best* link rather than the worst, since a file only needs
one solid match to be a duplicate. In a group fused by an incidental link (three
episodes sharing an opening sequence, one of which also has a real re-encode of
itself present) the two genuine copies still read as sharing their full runtime.
The flip side is that a high figure says the file matched *something* here
closely, not everything. In a group of three or more, check the pair you care
about.

### CSV and JSON

The console and `.txt` report have one line per file; the machine-readable
formats carry more data:

| Column | Meaning |
| --- | --- |
| `matched_with` | The group member the `matched_seconds` figure on this row describes |
| `samples` | How many frame hashes this file's fingerprint holds |
| `matched_from`, `matched_to` | Where that footage sits **in this file's own runtime** |
| `matched_from_seconds`, `matched_to_seconds` | The same two as raw seconds |

The timestamps are per file, not per pair: a two-minute clip cut from the middle
of an episode reads `00:00:00` to `00:02:01` on its own row and `00:19:59` to
`00:22:40` on the episode's, which answers "where in this episode is that clip".

Read the range as an **envelope, not a continuous stretch**: it runs from the
start of the first matching moment to the end of the last, and matches in
between can be scattered. Two episodes sharing an opening and a closing theme
have an envelope covering the whole hour and a `matched_seconds` of about
thirty. When the envelope is much the wider, either the match is scattered or
the file is too coarsely sampled to tell, and `samples` separates those two.

Columns run in three blocks: what the row *is* (`group`, `action`, `full_path`),
what the file *is* (`length` through `quality_bits_per_frame`), and what it was
measured *against* (`matched_with` through `matched_to_seconds`). Anything shown
formatted is followed by the raw number it came from (`length`/`length_seconds`,
`size`/`size_bytes`, and so on), since a spreadsheet cannot sort `1.0MB` against
`900.0KB`. Sort and filter on the raw column, read the other one. Frame rate is
the exception: the reports carry only `framerate_fps`.

Column positions are not stable across versions, and nothing needs them to be.
`--from-report` finds columns by name, so an older or reordered report still
replays.

The JSON additionally gives every file a `matches` array, one entry per group
member it was directly compared against, strongest first, each with its own
`matched_seconds` and range. The top-level fields describe entry `[0]`.

### Actions

Every file in a duplicate group is labeled with an action. The label belongs to
the file, not the group: a file in several groups shows the same one in all of
them, resolved as REVIEW > DELETE > KEEP.

- **KEEP**: the best copy in the group, chosen by your `--priority`. One per
  group, except in a codec standoff (above), and except where that best copy is
  itself redundant against something in another group.
- **DELETE**: a redundant copy, in at least one of the groups it appears in.
  Nothing happens to it without `--delete`; the summary totals these into the
  reclaimable figure.
- **REVIEW**: worth a manual look before deleting, for example when the KEEP
  pick is the longest video but a different file has higher resolution, or when
  the group holds the best copy of several codecs. REVIEW files are never
  deleted.

Once armed, DELETE rows report what actually happened: **DELETED** (trashed or
removed), **MOVED** (relocated by `--move-to`), **FAILED**, or **CHANGED**,
meaning the file changed on disk after it was scanned and was left alone.

### Cache

Fingerprints are cached (under `$XDG_CACHE_HOME/vid-fp`, falling back to
`~/.cache/vid-fp`), so re-scanning the same library is near-instant. Use
`--clear-cache` or `--prune-cache` to manage it.

An entry is invalidated by the file changing (size or modification time) and by
the two flags that decide which frames get sampled, `--keyframe-interval` and,
while an interval is in force, `--min-keyframes`. The comparison flags (`-d`,
`-p`, `--min-duration`) are applied to cached fingerprints and never invalidate
them, so re-running a scan at a different tolerance is instant.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Ran clean |
| `1` | Fatal error; the run did not complete |
| `2` | Completed, but something failed (see the `Problems` summary) |
| `130` | Interrupted with Ctrl-C |

## Safety

- **Report-only by default.** Without `--delete`, nothing is ever removed.
- **You see the cost before you pay it.** A dry run prints how much the DELETE
  files would reclaim, computed from exactly the set `--delete` would act on.
- **It asks first.** An armed run stops after the groups are resolved and shows
  how many files, how many bytes, and the first ten by name. Return accepts, `n`
  turns the run into a report. The prompt is interactive-only and `--yes` skips
  it, so nothing can hang unattended.
- **Trash, not permanent.** `--delete` moves files to the system trash via the
  FreeDesktop.org spec, so they're recoverable, unless you add `--permanent`.
- **`--move-to` where the trash isn't.** Moving files under a folder of your
  choosing is recoverable in the same way and always available.
- **Do a dry run first.** Look at the output (or a saved `--output` report)
  before running with `--delete`.
- **DELETE always rests on a measurement.** Every member of a group was directly
  compared with every other, so a file marked DELETE lost the ranking to a copy
  it was actually measured against.
- **No double-counting.** Hard links and symlinks to the same file collapse into
  a single entry, so the reported space freed reflects bytes actually reclaimed.
- **Tab-complete your `-e` paths.** An exclude folder that can't be resolved
  excludes nothing.
- **Nothing is acted on twice, or blind.** Every target is re-checked against
  its recorded size immediately before it's touched, and a file that changed
  since the scan is left alone and reported.
- **Mixed codecs are never guessed at.** When the only thing separating two
  copies is which encoder made them, both are flagged REVIEW.
- **`--from-report` hands the judgement to you.** The rules above are applied by
  the run that *writes* a report. Replaying an edited one keeps the confirmation
  prompt and the size check, but nothing else: the edited file is the decision,
  and it is not checked for leaving a survivor in each group. A row it cannot
  understand is never guessed at; the file is left alone and the run exits `2`.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at
your option.

The **released binary** additionally contains FFmpeg (LGPL 2.1+) and dav1d
(BSD-2-Clause), statically linked. That does not affect the licence above, but
because the FFmpeg link is static, LGPL 2.1 §6 entitles you to relink the
program against your own build of FFmpeg. Every release therefore ships
`vid-fp-<version>-ffmpeg-static-libs.tar.gz` with the archives, headers, build
script and instructions needed to do that. The FFmpeg in it is configured
`--disable-gpl --disable-nonfree`, so it is LGPL only.

A `cargo install` build links your system's shared FFmpeg and none of this
applies to it. See [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).
