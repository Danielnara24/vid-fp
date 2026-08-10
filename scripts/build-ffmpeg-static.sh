#!/usr/bin/env bash
#
# Builds the static FFmpeg + dav1d prefix that `--features static-ffmpeg` links
# against. CI runs this, and so should you: a vid-fp built against the distro's
# shared FFmpeg fingerprints slightly differently from the released binary (see
# "Why the versions are pinned" below), so local accuracy numbers only mean
# something if they came from the same libraries the release ships.
#
#   ./scripts/build-ffmpeg-static.sh                 # -> ./ffmpeg-static
#   ./scripts/build-ffmpeg-static.sh /opt/ffmpeg     # -> /opt/ffmpeg
#
# Then, from the repo root:
#
#   FFMPEG_DIR="$PWD/ffmpeg-static" cargo build --release --features static-ffmpeg
#
# Takes roughly ten minutes the first time. It is incremental afterwards: a
# prefix that already carries the pinned versions is left alone, so re-running
# costs a second. Pass --force to rebuild from scratch.
#
# ---------------------------------------------------------------------------
# Why the versions are pinned
#
# The FFmpeg version is part of what a fingerprint MEANS. Decoder output is not
# guaranteed bit-identical across major versions, and it measurably is not here:
# moving 6.1 -> 8.x drops one pair from the `-d 4 -p 20` accuracy profile
# because a clip's measured coverage falls by a single 0.5s hash sample. The
# cache `Stamp` deliberately does not record an FFmpeg version, so entries
# written against two different ones would mix silently. Changing FFMPEG_VERSION
# is therefore exactly as load-bearing as changing the hash algorithm: it needs
# a CACHE_TABLE rename in src/main.rs and a re-taken accuracy baseline.
#
# Why dav1d is not optional
#
# FFmpeg's built-in AV1 decoder is hardware-accelerated only -- in a software
# build it fails every frame with "Your platform doesn't support hardware
# accelerated AV1 decoding". Without libdav1d the binary silently stops finding
# AV1 duplicates (loudly, in fact: those files land in the "could not be
# fingerprinted" tally and the run exits 2, but the duplicates are gone either
# way). configure is given --enable-libdav1d unconditionally, and the build
# fails below if the resulting FFmpeg cannot report an AV1 decoder.
# ---------------------------------------------------------------------------

set -euo pipefail

FFMPEG_VERSION="n8.1.2"
DAV1D_VERSION="1.5.4"

PREFIX=""
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
        -*) echo "error: unknown option $arg" >&2; exit 1 ;;
        *)
            if [[ -n "$PREFIX" ]]; then
                echo "error: more than one prefix given ($PREFIX, $arg)" >&2
                exit 1
            fi
            PREFIX="$arg"
            ;;
    esac
done
PREFIX="${PREFIX:-$PWD/ffmpeg-static}"

mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"
BUILD="$PREFIX/.build"
STAMP="$PREFIX/.stamp"
WANT="ffmpeg=$FFMPEG_VERSION dav1d=$DAV1D_VERSION"

if [[ "$FORCE" == "1" ]]; then
    rm -rf "$PREFIX"/{.build,.stamp,lib,include,share,bin}
elif [[ -f "$STAMP" && "$(cat "$STAMP")" == "$WANT" ]]; then
    echo "==> $PREFIX is already at $WANT -- nothing to do (--force to rebuild)"
    exit 0
fi

# --- prerequisites ---------------------------------------------------------
#
# meson and ninja are dav1d's build system and are the two most likely to be
# missing, so rather than send you to your package manager they go in a venv
# under the prefix. Everything else has to come from the system.
missing=()
for tool in git make nasm pkg-config python3; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || missing+=("gcc")
if (( ${#missing[@]} )); then
    echo "error: missing required tool(s): ${missing[*]}" >&2
    echo "  Debian/Ubuntu: sudo apt install git make nasm pkg-config python3 python3-venv build-essential" >&2
    echo "  Fedora:        sudo dnf install git make nasm pkgconf-pkg-config python3 gcc" >&2
    echo "  Arch:          sudo pacman -S git make nasm pkgconf python base-devel" >&2
    exit 1
fi

mkdir -p "$BUILD"

if ! command -v meson >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
    VENV="$BUILD/venv"
    if [[ ! -x "$VENV/bin/meson" ]]; then
        echo "==> meson/ninja not found; bootstrapping them into $VENV"
        python3 -m venv "$VENV" 2>/dev/null || {
            echo "error: python3 -m venv failed. Install python3-venv, or install" >&2
            echo "       meson and ninja yourself and re-run." >&2
            exit 1
        }
        "$VENV/bin/pip" install --quiet --upgrade pip
        "$VENV/bin/pip" install --quiet meson ninja
    fi
    export PATH="$VENV/bin:$PATH"
fi

export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
JOBS="$(nproc 2>/dev/null || echo 4)"

# --- dav1d -----------------------------------------------------------------
#
# Static only, and with its own tools/tests off: we want libdav1d.a and its
# headers, nothing else. BSD-2-Clause, so it adds attribution but no copyleft.
echo "==> building dav1d $DAV1D_VERSION"
if [[ ! -d "$BUILD/dav1d" ]]; then
    git clone --depth 1 -b "$DAV1D_VERSION" \
        https://code.videolan.org/videolan/dav1d.git "$BUILD/dav1d"
fi
rm -rf "$BUILD/dav1d/build"
meson setup "$BUILD/dav1d/build" "$BUILD/dav1d" \
    --prefix="$PREFIX" \
    --libdir=lib \
    --buildtype=release \
    --default-library=static \
    -Denable_tools=false \
    -Denable_tests=false
ninja -C "$BUILD/dav1d/build" -j "$JOBS"
ninja -C "$BUILD/dav1d/build" install

# --- FFmpeg ----------------------------------------------------------------
#
# LGPL 2.1+: --enable-gpl and --enable-nonfree are deliberately absent and must
# stay that way. vid-fp is MIT OR Apache-2.0, and Apache-2.0 is incompatible
# with GPL-2, so a GPL-enabled FFmpeg could not be shipped in the same binary.
#
# The disable list is everything vid-fp cannot reach: it only ever demuxes a
# local file and decodes video from it. Encoders, muxers, filters, capture
# devices and network protocols are all dead weight and, in the case of the
# protocols, dead attack surface. Decoders and demuxers are left switched on
# WHOLESALE and on purpose -- --extensions accepts any of six containers and
# says nothing about what is inside them, so trimming the decoder list would
# turn "vid-fp doesn't support this codec" into a shipping decision rather than
# an FFmpeg one.
#
# zlib is enabled explicitly because --disable-autodetect switched it off, and
# Matroska needs it: mkv may zlib-compress track headers, and vid-fp reads mkv.
# postproc takes no flag here -- FFmpeg 8 dropped it from LIBRARY_LIST, so
# --disable-postproc is now an "Unknown option" that aborts configure.
echo "==> building FFmpeg $FFMPEG_VERSION"
if [[ ! -d "$BUILD/ffmpeg" ]]; then
    git clone --depth 1 -b "$FFMPEG_VERSION" \
        https://github.com/FFmpeg/FFmpeg "$BUILD/ffmpeg"
fi

pushd "$BUILD/ffmpeg" >/dev/null
make distclean >/dev/null 2>&1 || true
./configure \
    --prefix="$PREFIX" \
    --pkg-config-flags="--static" \
    --extra-cflags="-I$PREFIX/include" \
    --extra-ldflags="-L$PREFIX/lib" \
    --enable-static \
    --disable-shared \
    --enable-pic \
    --enable-libdav1d \
    --disable-autodetect \
    --disable-gpl \
    --disable-nonfree \
    --disable-version3 \
    --disable-debug \
    --disable-programs \
    --disable-doc \
    --disable-avdevice \
    --disable-avfilter \
    --disable-swresample \
    --enable-zlib \
    --disable-encoders \
    --disable-muxers \
    --disable-filters \
    --disable-devices \
    --disable-network \
    --disable-protocols \
    --enable-protocol=file
make -j "$JOBS"
make install
popd >/dev/null

# --- verification ----------------------------------------------------------
#
# The failure this guards against is a configure that quietly proceeded without
# dav1d, which produces a working build that cannot decode a single AV1 frame.
# Cheap to check, and catastrophic to miss.
echo "==> verifying"
for lib in libavcodec libavformat libavutil libswscale; do
    [[ -f "$PREFIX/lib/$lib.a" ]] || { echo "error: $lib.a was not installed" >&2; exit 1; }
done
[[ -f "$PREFIX/lib/libdav1d.a" ]] || { echo "error: libdav1d.a was not installed" >&2; exit 1; }

if ! grep -q '^#define CONFIG_LIBDAV1D 1' "$BUILD/ffmpeg/config.h" 2>/dev/null; then
    echo "error: FFmpeg was configured WITHOUT libdav1d -- the resulting binary" >&2
    echo "       would fail on every AV1 file. Refusing to install it." >&2
    exit 1
fi

echo "$WANT" > "$STAMP"
echo
echo "==> done: $PREFIX  ($WANT, LGPL 2.1+, AV1 via dav1d)"
echo
echo "Build vid-fp against it with:"
echo "  FFMPEG_DIR=\"$PREFIX\" cargo build --release --features static-ffmpeg"
