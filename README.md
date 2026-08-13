# vid-fp

Fast video **duplicate and clip finder** for Linux. It fingerprints videos from their
keyframes and groups together files that have the same content, even when they differ
in resolution, file size, or container, and even when one video is only a **trimmed
clip embedded inside another**.
It reports duplicate groups and can optionally move the redundant copies to the trash.

> **Note:** `vid-fp` can delete files. By default it does nothing destructive —
> it only reports. Deletion happens *only* when you pass `--delete`, it asks you
> to confirm before touching anything, and even then files go to the system trash
> (recoverable) unless you also pass `--permanent`. See [Safety](#safety).

## Requirements

Linux, x86_64, **glibc 2.35 or newer** (Ubuntu 22.04+, Debian 12+, Fedora 36+,
current Arch and Mint). Nothing else, the release binary bundles its own FFmpeg,
so you do **not** need FFmpeg installed.

## Installation

### Prebuilt binary (recommended)

```bash
curl -L -o vid-fp \
  https://github.com/Danielnara24/vid-fp/releases/latest/download/vid-fp-x86_64-linux-gnu
chmod +x vid-fp
sudo install -m 755 vid-fp /usr/local/bin/vid-fp
```

That URL always points at the newest version.

The binary carries FFmpeg 8 and dav1d inside it.

Now `vid-fp` runs from anywhere:

```bash
vid-fp --help
```

Prefer not to use `sudo`? Install into `~/.local/bin` instead and make sure that
directory is on your `PATH`.

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
FFmpeg — and its fingerprints may differ very slightly from the released
binary's, since the release pins FFmpeg 8. Different fingerprints mean a
different cache, and `vid-fp` cannot tell the two apart, so don't point a
source build and a release build at the same cache expecting agreement.

To reproduce the release exactly instead — self-contained, AV1 included:

```bash
git clone https://github.com/Danielnara24/vid-fp && cd vid-fp
./scripts/build-ffmpeg-static.sh                  # ~10 min, once
cargo build --release --features static-ffmpeg    # -> target/release/vid-fp
```

Needs `git make nasm pkg-config python3 python3-venv build-essential`; the
script fetches and builds pinned FFmpeg and dav1d releases into `./ffmpeg-static`
and is a no-op on later runs. Set `FFMPEG_DIR` to use a prefix somewhere else —
but pass it per command rather than exporting it, since it also changes how an
ordinary non-static build links.
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

Installed from source? The binary generates both itself:

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

Use the same commands as installing, it will overwrite the last version.

Run `vid-fp --version` to see what you have installed and compare it against the
[latest release](https://github.com/Danielnara24/vid-fp/releases/latest). Installed
from source? Re-run the `cargo install` command above with `--force`.
The completions and man page should be reinstalled too to have the latest documentation.

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

By default the scan is **not** recursive (only the folders you name and their
immediate files). Add `-r` to descend into subfolders.

Files are identified by inode, not by path. If the same file is reachable more
than once — through a symlink, a hard link, or two scan folders that overlap —
it's fingerprinted once and reported once, so nothing is ever listed as a
duplicate of itself. Symlinked *folders* are skipped unless you pass
`--follow-symlinks`.

### Naming files, and reading a list

Individual files can be named alongside folders:

```bash
vid-fp ~/Videos/episode_a.mkv ~/Videos/episode_b.mp4 ~/Downloads
```

A file you name is scanned whatever its extension.

A list of paths can also be read from stdin with `-`, or from a file with
`--from-file`, which lets the rest of the shell decide what gets scanned:

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

Paths read this way behave exactly as if they had been typed as arguments:
folders are walked, files are taken as given, `-e` still excludes, and the same
file arriving twice is still scanned once. Piping in files that aren't videos
will report them under `Problems`, so filter with `fd -e`/`find -name` rather
than piping an entire tree.

### Deleting duplicates

```bash
# Move the files marked DELETE to the trash
vid-fp ~/Videos -r --delete

# Remove them permanently (irreversible)
vid-fp ~/Videos -r --delete --permanent
```

Once the scan is done and the groups are resolved, an armed run stops and asks
before it touches anything.

Return accepts. Answering `n` doesn't abort the run, it demotes it to a
report-only one, so you still get the full table and the reclaimable figure
without re-scanning. Pass `-y`/`--yes` to skip the prompt.

The prompt only appears when there's a terminal on both stdin and stderr, so it
can never block a script, a cron job, or `fd … | vid-fp - --delete`; those runs
proceed exactly as they did before it existed. Use `--yes` in scripts anyway if
you'd rather have that in writing than inferred from the environment.

### Moving instead of deleting

The system trash needs a trash directory on the file's own filesystem, which
external drives, NFS mounts and headless servers frequently don't have.
In those cases try `--move-to` instead:

```bash
# Relocate the redundant copies instead of trashing them
vid-fp /mnt/media -r --move-to /mnt/scratch/dupes
```

This isn't a deletion, so it doesn't need `--delete`'s permission — `--move-to`
arms the run by itself. If `--delete` or `--permanent` is passed alongside it,
the files are still moved and a note says so; nothing is removed.

Each file's **absolute path is recreated inside the destination**:
`/mnt/media/show/ep01.mkv` lands at `/mnt/scratch/dupes/mnt/media/show/ep01.mkv`.
That means two files with the same name never collide, you can see where each one
came from, and the whole run is undone with a single copy back:

```bash
cp -a /mnt/scratch/dupes/. /
```

Nothing is ever overwritten: if the destination slot is already occupied (an
earlier run moved a file from that path, and the path was recreated since), the
move is refused and reported as a problem rather than silently replacing what's
there. A destination on another filesystem is copied, flushed to disk, and only
then unlinked — and if either step fails, the original is left exactly where it
was. Timestamps and permissions are preserved.

The destination must sit outside the scanned folders; the run aborts if it
doesn't, since moving files into the scan just feeds them back in next time.

### Acting on a report you have reviewed

Save a report, review it, edit it, then hand it back:

```bash
# 1. Look first
vid-fp /mnt/media -r -o dupes.csv

# 2. Edit the action column in dupes.csv, then act on exactly what it says
vid-fp --from-report dupes.csv --delete
```

`--from-report` disposes of every row whose `action` column reads `DELETE` and
touches nothing else.

Change `REVIEW`/`KEEP` cells to `DELETE` and they're acted on. Change a `DELETE` to
anything else and it isn't.

Every file is still re-checked against the size the report recorded immediately
before it's touched, and left alone if it has changed since. A report can sit for a week before you get to it.

Columns are found **by name**, so a report that has had its columns reordered still works.
Only the CSV report should be provided; `.txt` and
`.json` reports are refused.

If you mark every copy of something
`DELETE`, every copy is deleted. The confirmation prompt still shows the count
and the byte total before anything happens.

## Options

| Flag | Description | Default |
| --- | --- | --- |
| `<PATH>...` | Folders and/or video files to scan (required). `-` reads a list of paths from stdin | — |
| `--from-file <FILE>` | Read the paths to scan from a file, one per line (`-` = stdin) | — |
| `-0`, `--null` | Paths in the list are NUL-separated, for `find -print0` / `fd -0` | off |
| `-r`, `--recursive` | Include subfolders | off |
| `--follow-symlinks` | Descend into symlinked folders | off |
| `-e`, `--exclude <FOLDER>` | Exclude a folder; repeat for several | — |
| `-x`, `--extensions <EXT>` | Video extensions to include, comma-separated or repeated | `mp4,mkv,avi,mov,flv,webm` |
| `-d`, `--hamming-distance <N>` | Frame-match tolerance, in bits out of 64; higher = less strict matching. Raise to increase duplicates found (but can increase false positives). A match backed by another at the same time offset is allowed further; see [Tuning](#tuning). Values above `64` are refused | `4` |
| `-p`, `--match-percent <F>` | Min % of overlap to count as a duplicate, from `0` to `100`; Lower = Includes shorter matches (but can increase false positives). Values outside the range are refused | `20.0` |
| `--min-duration <SECS>` | Min shared clip length in seconds for a match. Videos shorter than this are skipped entirely. `0` = off; negative values are refused | `0.0` |
| `-k`, `--priority <P>` | Criteria for KEEPING files: `length`, `resolution`, `quality`, or `size`. The chosen one is compared first; the rest follow in the default order. See [Codecs and quality](#codecs-and-quality) | `length` |
| `--keyframe-interval <F>` | Seconds between sampled keyframes (`0` = every keyframe); Higher = Faster processing (Increasing this can hinder the capability of finding short matches between videos) | `0.0` |
| `--min-keyframes <F>` | Min keyframes kept for short videos (only relevant when keyframe-interval > 0) | `4.0` |
| `-o`, `--output <FILE>` | Optional path to save the report — `.txt`, `.csv`, or `.json` | — |
| `--delete` | Move files marked DELETE to the trash | off |
| `--permanent` | With `--delete`, permanently remove instead | off |
| `--move-to <DIR>` | Move the files marked DELETE under `DIR`, recreating their absolute paths inside it. Arms the run on its own and supersedes `--delete`/`--permanent` | — |
| `--from-report <FILE>` | Act on a CSV report from an earlier run instead of scanning: dispose of every row whose `action` column reads `DELETE`, and nothing else. Requires `--delete` or `--move-to`. See [Acting on a report you have read](#acting-on-a-report-you-have-read) | — |
| `-y`, `--yes` | Answer yes to the confirmation shown before any file is touched. The prompt only appears on an interactive terminal, so piped and redirected runs are never blocked either way | off |
| `-t`, `--threads <N>` | Worker threads (`0` = uses all cores) | `0` |
| `-q`, `--quiet` | Only print errors | off |
| `--clear-cache` | Wipe ALL vid-fp cache before running | off |
| `--prune-cache` | Drop cached entries only for files not in this scan | off |
| `--completions <SHELL>` | Print a completion script for `bash`, `zsh`, `fish`, `elvish`, or `powershell` and exit | — |
| `--man` | Print the man page (roff) and exit | — |

## Tuning

The two knobs that decide what counts as a duplicate are `-d` (how different two
frames may be) and `-p` (how much of a video must match). They trade off against
each other, so change one at a time.

`-d` counts differing bits of a 64-bit frame hash. Two unrelated frames sit
about 32 bits apart, so the whole useful range is roughly 2 to 12: below that
only a bit-identical re-encode matches, above it unrelated footage starts to.

**Only even values of `-d` do anything.** Every hash has exactly 32 of its 64
bits set, so any two of them differ in an even number of places — `-d 5` accepts
exactly what `-d 4` accepts, `-d 7` exactly what `-d 6` does. Step in twos.

**A frame match is judged on its distance *and* on whether anything backs it
up.** Two encodes of the same footage place every frame they share at one
constant time offset, so their matches corroborate each other; two videos that
merely look alike produce matches scattered across unrelated moments. `-d`
therefore sets two thresholds rather than one:

- a match with nothing behind it must be within `-d`, exactly as before;
- a match that another frame match agrees with — a different frame of each
  video, landing within half a second of the same offset — may reach `-d + 6`.

**Past 12 bits, one witness stops being enough.** Two unrelated frames land
within 12 bits of each other about once in fifty million; within 20 bits, once
in six thousand. A pair of coincidences that far out is not rare enough to mean
anything, so the number of agreeing frame matches required grows with the
distance: one out to 12 bits, two at 14, three at 16, four at 20.

Both thresholds move with every rung of `-d` and neither is clamped against a
constant, so the flag stays a sensitivity control across its whole range.
Measured against a hand-labeled pair set (see the accuracy notes), each rung
beats what the same setting did when `-d` was a single flat threshold:

| `-d` | matches on their own | with agreement | precision | recall |
|---:|---|---|---:|---:|
| 2 | ≤ 2 bits | ≤ 8 | 100.0% | 65.4% |
| **4** (default) | ≤ 4 bits | ≤ 10 | 99.7% | 75.4% |
| 6 | ≤ 6 bits | ≤ 12 | 99.2% | 83.9% |
| 8 | ≤ 8 bits | ≤ 14 | 98.3% | 88.3% |
| 10 | ≤ 10 bits | ≤ 16 | 96.0% | 91.9% |
| 12 | ≤ 12 bits | ≤ 18 | 88.5% | 94.7% |

Recall there is recall of what any tool found, so read it as a relative figure;
precision is exact. The knee on that corpus sits around `-d 10`, and the
default is deliberately well inside it — the default's job is to be safe on the
corpus nobody measured.

**Both knobs are monotone**, which is what makes tuning them predictable:

- Raising `-d` (or lowering `-p`) only ever *adds* files to the report. Nothing
  that matched at a tighter setting stops matching at a looser one.
- Lowering `-d` (or raising `-p`) only ever *removes* them.

So the report at any setting is a subset of the report at every looser one, and
a file that shows up when you loosen the knobs was always a weaker match than
the ones already there. Start at the defaults and walk one knob outward until
you see something you don't recognise.

**Not finding duplicates you expect?** Raise `-d`, or lower `-p` if the
match is a short clip inside a long video. Two encodes of the same footage only
line up frame-for-frame when their keyframes do; when they don't — one encoder
cutting on scene changes, another on a fixed interval — each file samples
moments the other never looked at, and no tolerance can bridge a frame that was
never sampled. Expect a group like that to report well under its full runtime as
shared even though the files are identical end to end.

**Getting false positives?** Lower `-d` first — it's the blunter of the two.
Dark scenes, fades, and letterboxed content look alike to any perceptual hash,
and a loose `-d` conflates them. (Frames with no structure at all are dropped
outright rather than hashed, so black frames and plain title cards can't link
anything on their own.)

`--min-duration` is an absolute floor, in seconds, on how much footage two files
must share. It's the tool to reach for when `-p` alone can't express what you
want. Both gates apply, so `-p 5 --min-duration 60`
means "at least 5% overlap *and* at least a minute of it".

The seconds it compares against are exactly the figure the report's **shared**
column prints — the lower of the two files' estimates, which is the conservative
one. A pair the report describes as sharing 3 seconds cannot get past
`--min-duration 5`.

It also skips fingerprinting anything shorter than the floor outright — such a
file can't contain a long enough shared clip, so there's nothing to gain by
decoding it. Videos whose duration the container doesn't report are never
skipped. Changing this flag doesn't invalidate the cache.

## Codecs and quality

These rules decide which copy is kept, not which files match. Matching itself is
codec-blind: it compares decoded pictures, so an AV1 encode and an H.264 encode
of the same footage land in the same group as they should.

**`quality` is bits per frame** — bitrate divided by the average frame rate.
Bitrate on its own double-counts the frame rate: a 60 fps copy needs roughly
twice the bitrate of a 30 fps one just to look the same, so ranking on raw
bitrate preferred whichever copy simply had more frames in it. Dividing by the
frame rate asks how much was spent on each picture instead.

**Bits are never compared across codecs.** A modern codec's whole job is to
carry the same picture in fewer bits, so an AV1 file that is half the size of an
H.264 one is doing exactly what it's supposed to — treating that as "worse"
would delete the better encode every time. Both `quality` and `size` are
therefore only compared between files sharing a codec: within a codec they rank
normally, across codecs they tie and the decision falls through to something
else.

**A group that spans codecs ends with one survivor per codec.** If the leading
copies match on length and resolution but were made with different codecs,
nothing comparable remains to rank the codecs against each other — so each
codec keeps its own best copy, flagged REVIEW for you to choose between. The
other copies of each codec lost to a file they *are* directly comparable with,
so they're marked DELETE as usual: a library holding five HEVC encodes and three
H.264 encodes of one episode ends up with two files to look at, not eight.

Note that bitrate (and so quality) includes audio, so a copy with lossless 5.1
can outrank one with a better video track and stereo AAC.

Every report shows the codec, frame rate, size, bitrate and quality of each
file.

## How it reads the results

**Every file in a group matched every other file in it.** A group is
the tool's evidence for deleting something, so it never asserts a comparison it
did not make.

**A file can appear in more than one group.** Its action is still decided once, for
the file, not once per group. A file marked DELETE in one group reads DELETE
in every group it appears in, and a file held for REVIEW anywhere is held
everywhere.

If B is redundant against A and C is redundant against B, both B and C go in a single
pass and A is what's left. It is what stops you having to re-run until a chain has collapsed one hop
at a time, and it is a reason to read a dry run's report before passing `--delete`.

**The "matched" column is footage, not frames.** It is how much of *this file's
own runtime* was found in the group member it matched most closely, in seconds —
so a pair linked only by a common title card reads as the second or two that
card lasts. Read it against the file's own length: that ratio is what separates
a re-encode from a shared clip.

Every figure on a row describes that row's file, including this one. On a
genuine match that costs nothing, because both ends agree anyway: a two-minute
clip inside a twenty-two minute episode is 100% of the clip and 9% of the
episode, and 100% × 2min and 9% × 22min are both two minutes, so both rows read
two minutes. Where the two rows *disagree*, they are supposed to — it means one
file was found almost entirely inside the other while the other was barely
covered back, and that asymmetry is a fact about the pair worth seeing. It is
most often a sampling artifact: see the `samples` column below.

Note this is not the figure `--min-duration` gates on. That one reconciles the
two ends to a single conservative number for the pair, because a gate has to
decide something about the pair; a row has to describe one file. On the honest
matches they are the same number.

**The `samples` column is how many frame hashes the file's fingerprint holds**,
after featureless frames (black, fades, title cards) have been dropped. It is
what makes the rest of the row interpretable. A file with very few samples has
each one standing for a long stretch of runtime, so its matched footage comes
out coarse; at the limit, **a file with one sample has that sample standing for
its entire runtime, so any match at all covers 100% of it** and no
`--match-percent` can gate it. If a row reports far more matched footage than
the file on the other end of the same pair, check this column first.

It reports the *best* link rather than the worst because a file only needs one
solid match to be a duplicate. In a group fused together by an incidental link —
three episodes sharing an opening sequence, one of which also has a real
re-encode of itself present — the two genuine copies still read as sharing their
full runtime with each other, instead of both being dragged down to the two
seconds of intro they share with the third file. The consequence to keep in mind
is the other direction: a high figure says this file matched *something* here
closely, not that it matched *everything* here closely. In a group of three or
more, check the pair you care about rather than assuming one number covers all
of them.

**The CSV and JSON say which file, and where.** The console and `.txt` report
have one line per file, but the machine-readable formats carry more data:

| Column | Meaning |
| --- | --- |
| `matched_with` | The group member the `matched_seconds` figure on this row describes |
| `samples` | How many frame hashes this file's fingerprint holds |
| `matched_from`, `matched_to` | Where that footage sits **in this file's own runtime** |
| `matched_from_seconds`, `matched_to_seconds` | The same two as raw seconds |

The timestamps are per file, not per pair: a two-minute
clip cut from the middle of an episode reads `00:00:00`–`00:02:01` on its own row
and `00:19:59`–`00:22:40` on the episode's. The second one is the answer to
"where in this episode is that clip", which nothing else in the report can tell
you.

Read the range as an **envelope, not a continuous stretch**: it runs from the
start of the first matching moment to the end of the last, and matches in
between can be scattered. Two episodes sharing an opening and a closing theme
have an envelope covering the whole hour and a `matched_seconds` of about
thirty. When the two figures agree, the match is one continuous run; when the
envelope is much the wider, either the match is scattered through it or the file
is too coarsely sampled to tell — `samples` is what separates those two.
`matched_seconds` never exceeds its own envelope.

**Every column runs in the same order in both formats**, in three blocks: what
the row *is* (`group`, `action`, `full_path`), what the file *is* (`length`
through `quality_bits_per_frame`), and what it was measured *against*
(`matched_with` through `matched_to_seconds`). `action` sits second because
`--from-report` exists to have it edited, and an action column you have to
scroll sideways to find is one that gets edited on the wrong row.

`samples` is the exception to that grouping: it describes the file rather than
the link, but it sits inside the last block, immediately before the figure it
qualifies.

Anything shown formatted is immediately followed by the raw number it was
formatted from — `length`/`length_seconds`, `resolution`/`width`/`height`,
`size`/`size_bytes`, and so on — because a spreadsheet cannot sort `1.0MB`
against `900.0KB`, nor `1920x1080` against `640x480`. Sort and filter on the raw
column; read the other one. The frame rate is the one exception: the reports
carry only `framerate_fps`, and the formatted `23.98fps` form appears on the
console line alone.

Column positions are not stable across versions, and nothing needs them to be:
`--from-report` finds columns by name, so a report from an older build still
replays, and so does one a spreadsheet handed back reordered.

The JSON additionally gives every file a `matches` array — one entry per group
member it was directly compared against, strongest first, each with its own
`matched_seconds` and range. The top-level fields describe entry `[0]`. That is
where a group of three or more stops needing a caveat: the whole set of
measurements is there, pair by pair.

Every file in a duplicate group is labeled with an action. The label is the
file's, not the group's: a file appearing in several groups shows the same one
in all of them, resolved as REVIEW > DELETE > KEEP.

- **KEEP** — the best copy in the group, chosen by your `--priority`. There is
  one per group, except in a codec standoff (below), where each codec's best is
  held for REVIEW instead — and except where that best copy is itself redundant
  against something in another group, in which case it reads DELETE here too.
- **DELETE** — a redundant copy, in at least one of the groups it appears in.
  Nothing happens to it without `--delete`; the summary totals these into the
  reclaimable figure so you can see the cost of the run before committing to it.
- **REVIEW** — a copy worth a manual look before deleting; for example, the KEEP
  pick is the longest video but a *different* file has higher resolution, or the
  group holds the best copy of each of several codecs and nothing comparable can
  choose between them. REVIEW files are never deleted, in any
  group.

Once armed, DELETE rows report what actually happened: **DELETED** (trashed or
removed), **MOVED** (relocated by `--move-to`), **FAILED**, or **CHANGED** — the
last meaning the file changed on disk after it was scanned, so it's no longer
the file that was judged redundant and was left alone.

Fingerprints are cached (under `$XDG_CACHE_HOME/vid-fp`, falling back to
`~/.cache/vid-fp`), so re-scanning the same library is near-instant. Use
`--clear-cache` or `--prune-cache` to manage it.

An entry is invalidated by the file changing (size or modification time) and by
the two flags that decide which frames get sampled — `--keyframe-interval` and,
while an interval is actually in force, `--min-keyframes`. With sampling off,
which is the default, `--min-keyframes` floors nothing and moving it costs you
nothing. The comparison flags (`-d`, `-p`, `--min-duration`) are applied to
cached fingerprints and never invalidate them, which is why re-running a scan at
a different tolerance is instant.

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
  what it's about to touch — how many files, how many bytes, and the first ten
  by name — before anything moves. Return accepts, `n` turns the run into a
  report. The prompt is interactive-only and `--yes` skips it, so nothing can
  hang unattended.
- **Trash, not permanent.** `--delete` moves files to the system trash via the
  FreeDesktop.org spec, so they're recoverable — unless you add `--permanent`.
- **`--move-to` where the trash isn't.** On external drives, NFS mounts and
  headless servers the trash often doesn't exist; moving the files under a
  folder of your choosing is recoverable in the same way, always available, and
  wins over `--delete`/`--permanent` whenever it's passed.
- **Do a dry run first.** Look at the output (or a saved `--output` report) before
  running with `--delete`.
- **DELETE always rests on a measurement, not on an inference.** Every member of
  a group was directly compared with every other, so a file marked DELETE lost
  the ranking to a copy it was actually measured against — never to one it
  merely shares a group with.
  - **No double-counting.** Hard links and symlinks to the same file collapse into
  a single entry, so the reported space freed reflects bytes actually reclaimed.
  - **Tab-complete your `-e` paths.** An exclude folder that can't be resolved
  excludes nothing. Letting the shell complete the path proves it exists before you
  start.
  - **Nothing is acted on twice, or blind.** Every target is re-checked against
  its fingerprint immediately before it's touched, and a file that changed since
  the scan is left alone and reported.
  - **Mixed codecs are never guessed at.** When the only thing separating two
  copies is which encoder made them, both are left alone and flagged REVIEW —
  one survivor per codec, chosen against that codec's own copies.
  - **`--from-report` hands the judgement to you, and says so.** The rules above
  are applied by the run that *writes* a report. Replaying an edited one keeps
  the confirmation prompt and the pre-disposal size check, but nothing else: the
  edited file is the decision, and it is not checked for leaving a survivor in
  each group. A row it cannot understand is never guessed at — the file is left
  alone and the run exits `2`.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your
option.

The **released binary** additionally contains FFmpeg (LGPL 2.1+) and dav1d
(BSD-2-Clause), statically linked. That does not affect the licence above — you
may use `vid-fp` under MIT or Apache-2.0 either way — but because the FFmpeg link
is static, LGPL 2.1 §6 entitles you to relink the program against your own build
of FFmpeg. Every release therefore ships
`vid-fp-<version>-ffmpeg-static-libs.tar.gz` with the exact archives, headers,
build script and instructions needed to do that. The FFmpeg in it is configured
`--disable-gpl --disable-nonfree`, so it is LGPL only.

A `cargo install` build links your system's shared FFmpeg and none of this
applies to it. See [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).