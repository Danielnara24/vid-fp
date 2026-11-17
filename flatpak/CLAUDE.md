# vid-fp: the Flatpak (`flatpak/`)

Read with the root `CLAUDE.md` and `src/gui/CLAUDE.md`. Built after img-fp's
(`../img-fp/flatpak`); where this one differs, it is said below.

`io.github.danielnara24.vid-fp`: both binaries in `/app/bin`, GTK 4 from the
GNOME 51 runtime (so Ubuntu 22.04 runs the window), and FFmpeg n8.1.2 + dav1d
1.5.4 in `/app/lib`. `--filesystem=host`: a duplicate finder is pointed
anywhere; Flathub's linter flags it and a submission needs the exception asked
for. Built offline from `cargo-sources.json` (flatpak-builder-tools'
`flatpak-cargo-generator.py` over `Cargo.lock`; **regenerate it whenever
Cargo.lock changes**, the release job checks). A release needs its `<release>`
at the top of the metainfo's `<releases>`, one plain sentence for users; the
release job fails without it. `release.yml`'s `flatpak` job attaches
`vid-fp-x86_64.flatpak` to each release. Local build: the command at the top of
the manifest (`org.flatpak.Builder`, user install, no sudo).

**FFmpeg is the release's, built by the release's script, and shared.** The
FFmpeg version is part of what a fingerprint means (`_ff8`, root `CLAUDE.md`),
so the Flatpak must decode with exactly what the static binaries link, not with
the runtime's. The manifest fetches the two release tarballs pinned by sha256
(the FFmpeg one is the `n8.1.2` commit the script pins, file for file, plus a
`VERSION` file; the dav1d one is the tarball the script's comment records) and
runs `scripts/build-ffmpeg-static.sh --shared --from=. /app`, so there is one
configure line for both. Shared rather than static because the LGPL is then met
by the libraries being replaceable, with no relink kit, and because a dynamic
build needs neither `FFMPEG_DIR` nor `static-ffmpeg` and so none of their traps.
**The runtime has a `libavcodec.so.62` of its own** (Freedesktop's, fewer
decoders) under the same soname: `/app/lib` comes first, and the release job
checks with `ldd` that it is ours that loads. bindgen needs libclang, which the
SDK lacks: `org.freedesktop.Sdk.Extension.llvm22`, `LIBCLANG_PATH`.

Inside the sandbox `$XDG_DATA_HOME` / `$XDG_CACHE_HOME` / `$XDG_CONFIG_HOME`
are the app's own under `~/.var/app/<id>/`: the cache, the kept scan and
`gui.json` live there, apart from a host install's. What must be the desktop's
is read through `vid_fp::desktop_dir` (`$HOST_XDG_*` when the host set them,
else the defaults under `$HOME`): the thumbnail cache the window reads, the
desktop's GTK stylesheet, and the home Trash a walk sets aside (both the
desktop's and the app's own, which a scan of the home folder walks into). The
Trash itself needs nothing: `gio::File::trash` goes through the Trash portal,
and *Open* through `FileLauncher`'s portal. The sandbox's `/tmp` is its own,
so a host-side test reads outputs written under `$HOME`. The desktop's fonts
and stylesheet are handled as in img-fp (`main::follow_desktop_fonts`,
`main::desktop_stylesheet`).

## Flathub

Not submitted. Flathub's requirements
(https://docs.flathub.org/docs/for-app-authors/requirements#generative-ai-policy)
say a Flathub manifest must not contain AI-generated or AI-assisted content,
that AI tools must not open or automate the submission PR nor write its
description, commits or replies (nor a linter-exception PR), and that
AI-generated code, documentation and packaging in the app must be disclosed,
with acceptance at the reviewers' discretion. So **Claude does not write
Flathub's manifest, `flathub.json`, the submission PR or its permission
justification**; the user does, by hand. This folder's manifest (`type: dir`,
used for local builds and the release's bundle) is not what Flathub gets.
