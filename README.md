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

## Updating

Use same command as installing, it will overwrite the last version.

Run `vid-fp --version` to see what you have installed and compare it against the
[latest release](https://github.com/USERNAME/vid-fp/releases/latest). FFmpeg is
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
To change what's detected as a duplicate, try increasing/decreasing the Hamming Distance or the Min Match Percentage.

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
| `-e`, `--exclude <FOLDER>` | Exclude a folder; repeat for several | — |
| `-x`, `--extensions <EXT>` | Video extensions to include, comma-separated or repeated | `mp4,mkv,avi,mov,flv,webm` |
| `-d`, `--hamming-distance <N>` | Frame-match tolerance; higher = less strict matching. Raise to increase duplicates found (but can increase false positives) | `3` |
| `-p`, `--match-percent <F>` | Min % of overlap to count as a duplicate; Lower = Finds shorter matches (but can increase false positives) | `10.0` |
| `-k`, `--priority <P>` | Criteria for KEEPING files: `length`, `resolution`, or `size` | `length` |
| `--keyframe-interval <F>` | Seconds between sampled keyframes (`0` = every keyframe); Higher = Faster processing (Increasing this can hinder the capability of finding short matches between videos) | `0.0` |
| `--min-keyframes <F>` | Min keyframes kept for short videos (only relevant when keyframe-interval > 0) | `4.0` |
| `-o`, `--output <FILE>` | Optional path to save the report — `.txt`, `.csv`, or `.json` | — |
| `--delete` | Move files marked DELETE to the trash | off |
| `--permanent` | With `--delete`, permanently remove instead | off |
| `-t`, `--threads <N>` | Worker threads (`0` = uses all cores) | `0` |
| `-q`, `--quiet` | Only print errors | off |
| `--clear-cache` | Wipe ALL vid-fp cache before running | off |
| `--prune-cache` | Drop cached entries only for files not in this scan | off |

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

## Safety

- **Report-only by default.** Without `--delete`, nothing is ever removed.
- **Trash, not permanent.** `--delete` moves files to the system trash via the
  FreeDesktop.org spec, so they're recoverable — unless you add `--permanent`.
- **Do a dry run first.** Look at the output (or a saved `--output` report) before
  running with `--delete`.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your
option.