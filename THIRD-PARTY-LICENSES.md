# Third-party licenses

`vid-fp` itself is licensed under **MIT OR Apache-2.0** (see `LICENSE-MIT` and
`LICENSE-APACHE`). That does not change anywhere in this document. Bundling
LGPL libraries imposes obligations on *distribution of the combined binary*; it
does not relicense the source you are reading.

This file covers the native libraries compiled **into the released binary**. The
Rust dependencies are not listed individually — they are declared in
`Cargo.toml`, resolved in `Cargo.lock`, and are MIT, Apache-2.0, or both; run
`cargo license` or `cargo about` for the current enumeration.

---

## FFmpeg — LGPL 2.1 or later

<https://ffmpeg.org/>

The released binary statically links `libavcodec`, `libavformat`, `libavutil`
and `libswscale`, built by `scripts/build-ffmpeg-static.sh` at the version
pinned there.

FFmpeg is dual-natured: the core is LGPL 2.1+, but certain components make it
GPL. **This build is LGPL only.** `configure` is invoked with `--disable-gpl`,
`--disable-nonfree` and `--disable-version3`, and those must stay — `vid-fp` is
offered under Apache-2.0, which is incompatible with GPL-2, so a GPL-enabled
FFmpeg could not lawfully ship inside the same binary.

### What LGPL 2.1 §6 requires of a static link, and how it is met

Dynamic linking satisfies §6 by itself: a user replaces `libavcodec.so` and the
loader picks it up. A static link removes that possibility, so §6 requires
distributing whatever a recipient needs to **relink** the executable against a
modified FFmpeg.

Every release therefore ships `vid-fp-<version>-ffmpeg-static-libs.tar.gz`,
containing:

- the exact `.a` archives the binary was linked against,
- their headers,
- `build-ffmpeg-static.sh`, carrying the pinned upstream versions and the
  complete `configure` line,
- `RELINKING.md`, with the commands to rebuild `vid-fp` against a substituted
  FFmpeg.

FFmpeg's own source is not mirrored here; the script pins an upstream git tag
and fetches it, which is reproducible and keeps the release small. If upstream
ever became unreachable, mirroring the tarball would be the fix.

A copy of the LGPL 2.1 ships in FFmpeg's source tree as `COPYING.LGPLv2.1`, and
is at <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>.

---

## dav1d — BSD-2-Clause

<https://code.videolan.org/videolan/dav1d>

Statically linked, at the version pinned in `scripts/build-ffmpeg-static.sh`.
That is upstream's home but no longer where the build fetches from: it sits
behind a proof-of-work bot check that `git clone` cannot answer, so the source
comes from VideoLAN's GitHub mirror at a pinned commit, verified against the
official release tarball. The script records both.

dav1d is **not optional**, and the reason is worth recording: FFmpeg's built-in
AV1 decoder is hardware-accelerated only, so a software build without dav1d
fails every AV1 frame with *"Your platform doesn't support hardware accelerated
AV1 decoding"*. `vid-fp` would then report those files under "could not be
fingerprinted" and miss every AV1 duplicate.

BSD-2-Clause asks only for attribution, which this section provides. It adds no
copyleft and no relink obligation.

```
Copyright © 2018-2024, VideoLAN and dav1d authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## zlib

Enabled in the FFmpeg build for Matroska track compression. zlib's license is
permissive and attribution-only. <https://zlib.net/>

---

## Builds you make yourself

A plain `cargo build` links the system's **shared** FFmpeg instead. Nothing here
applies to that binary — the LGPL is satisfied by the dynamic link, and you are
using your distribution's FFmpeg under whatever terms it was packaged with
(which, unlike this build, is frequently GPL).
