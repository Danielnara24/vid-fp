# vid-fp

Fast video **duplicate and clip finder** for Linux. It fingerprints videos from their
keyframes and groups together files that have the same content, even when they differ
in resolution, file size, or container, and even when one video is only a **trimmed
clip embedded inside another**.

Unlike tools that hash whole files or match on exact frames, `vid-fp` is built to catch
the hard cases: a 2-minute clip cut out of a full episode, the same movie at two
resolutions, or a video that was re-encoded. It reports duplicate groups and can
optionally move the redundant copies to the trash.

> **Note:** `vid-fp` can delete files. By default it does nothing destructive —
> it only reports. Deletion happens *only* when you pass `--delete`, and even then
> files go to the system trash (recoverable) unless you also pass `--permanent`.
> See [Safety](#safety).

## Requirements

`vid-fp` links against FFmpeg, so you need **FFmpeg 6.x** installed at runtime:

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# Arch
sudo pacman -S ffmpeg

# Fedora
sudo dnf install ffmpeg
```

Linux, x86_64 only.

## Installation

### Prebuilt binary (recommended)

Install FFmpeg (above), then download the latest release binary. This URL always
points at the newest version:

```bash
curl -L -o vid-fp \
  https://github.com/Danielnara24/vid-fp/releases/latest/download/vid-fp-x86_64-linux-gnu
chmod +x vid-fp
sudo install -m 755 vid-fp /usr/local/bin/vid-fp
```

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

Requires the Rust toolchain plus the FFmpeg **development** libraries
(`libavcodec-dev libavformat-dev libavutil-dev libavfilter-dev libavdevice-dev
libswscale-dev libswresample-dev`, `clang`, and `pkg-config` on Debian/Ubuntu):

```bash
cargo install --git https://github.com/Danielnara24/vid-fp
```
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
vid-fp ~/Videos -k <TAB>         # length  resolution  bitrate  size
```

## Updating

Use same command as installing, it will overwrite the last version.

Run `vid-fp --version` to see what you have installed and compare it against the
[latest release](https://github.com/Danielnara24/vid-fp/releases/latest). FFmpeg is
separate and doesn't need reinstalling. Installed from source? Re-run the
`cargo install` command above with `--force`.

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

### Deleting duplicates

```bash
# Move the files marked DELETE to the trash
vid-fp ~/Videos -r --delete

# Remove them permanently (irreversible)
vid-fp ~/Videos -r --delete --permanent
```

## Options

| Flag | Description | Default |
| --- | --- | --- |
| `<FOLDER>...` | One or more folders to scan (required) | — |
| `-r`, `--recursive` | Include subfolders | off |
| `--follow-symlinks` | Descend into symlinked folders | off |
| `-e`, `--exclude <FOLDER>` | Exclude a folder; repeat for several | — |
| `-x`, `--extensions <EXT>` | Video extensions to include, comma-separated or repeated | `mp4,mkv,avi,mov,flv,webm` |
| `-d`, `--hamming-distance <N>` | Frame-match tolerance; higher = less strict matching. Raise to increase duplicates found (but can increase false positives). Values above `7` work but see [Tuning](#tuning) | `3` |
| `-p`, `--match-percent <F>` | Min % of overlap to count as a duplicate; Lower = Finds shorter matches (but can increase false positives) | `10.0` |
| `--min-duration <SECS>` | Min shared clip length in seconds for a match. Videos shorter than this are skipped entirely. `0` = off | `0.0` |
| `-k`, `--priority <P>` | Criteria for KEEPING files: `length`, `resolution`, `bitrate`, or `size`. The chosen one is compared first; the rest follow in the default order | `length` |
| `--keyframe-interval <F>` | Seconds between sampled keyframes (`0` = every keyframe); Higher = Faster processing (Increasing this can hinder the capability of finding short matches between videos) | `0.0` |
| `--min-keyframes <F>` | Min keyframes kept for short videos (only relevant when keyframe-interval > 0) | `4.0` |
| `-o`, `--output <FILE>` | Optional path to save the report — `.txt`, `.csv`, or `.json` | — |
| `--delete` | Move files marked DELETE to the trash | off |
| `--permanent` | With `--delete`, permanently remove instead | off |
| `-t`, `--threads <N>` | Worker threads (`0` = uses all cores) | `0` |
| `-q`, `--quiet` | Only print errors | off |
| `--clear-cache` | Wipe ALL vid-fp cache before running | off |
| `--prune-cache` | Drop cached entries only for files not in this scan | off |

## Tuning

The two knobs that decide what counts as a duplicate are `-d` (how different two
frames may be) and `-p` (how much of a video must match). They trade off against
each other, so change one at a time.

**Not finding duplicates you expect?** Raise `-d` to 5–7, or lower `-p` if the
match is a short clip inside a long video.

**Getting false positives?** Lower `-d` first — it's the blunter of the two.
Dark scenes, fades, and letterboxed content look alike to any perceptual hash,
and a loose `-d` conflates them.

Above `-d 7` the index that proposes candidate pairs is no longer exhaustive: it
may fail to *propose* a pair whose frames are all near the far edge of the
tolerance. Once a pair is proposed it is always compared exactly, and genuine
duplicates share hundreds of frames, so a miss is very unlikely — but if you go
that high, raise `-p` alongside it to keep false positives in check.

`--min-duration` is an absolute floor, in seconds, on how much footage two files
must share. It's the tool to reach for when `-p` alone can't express what you
want: 5% of a two-hour film is six minutes of real content, while 5% of a
minute long clip is five seconds. Both gates apply, so `-p 5 --min-duration 60`
means "at least 5% overlap *and* at least a minute of it".

It also skips fingerprinting anything shorter than the floor outright — such a
file can't contain a long enough shared clip, so there's nothing to gain by
decoding it. Videos whose duration the container doesn't report are never
skipped. Changing this flag doesn't invalidate the cache.

## How it reads the results

Every file in a duplicate group is labeled with an action:

- **KEEP** — the best copy in the group, chosen by your `--priority`.
- **DELETE** — a redundant copy (removed only if you pass `--delete`).
- **REVIEW** — a copy worth a manual look before deleting; for example, the KEEP
  pick is the longest video but a *different* file has higher resolution. REVIEW
  files are never deleted automatically.

Fingerprints are cached (under `$XDG_CACHE_HOME/video-dedup`, falling back to
`~/.cache/video-dedup`), so re-scanning the same library is near-instant. Use
`--clear-cache` or `--prune-cache` to manage it.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Ran clean |
| `1` | Fatal error; the run did not complete |
| `2` | Completed, but something failed (see the `Problems` summary) |
| `130` | Interrupted with Ctrl-C |

## Safety

- **Report-only by default.** Without `--delete`, nothing is ever removed.
- **Trash, not permanent.** `--delete` moves files to the system trash via the
  FreeDesktop.org spec, so they're recoverable — unless you add `--permanent`.
- **Do a dry run first.** Look at the output (or a saved `--output` report) before
  running with `--delete`.
  - **No double-counting.** Hard links and symlinks to the same file collapse into
  a single entry, so the reported space freed reflects bytes actually reclaimed.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your
option.