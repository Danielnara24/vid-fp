#!/usr/bin/env bash
#
# Builds and tests the static vid-fp inside a manylinux_2_28 container
# (AlmaLinux 8, glibc 2.28), which is what makes the released binary run on
# glibc 2.28 and newer. A binary asks for the glibc it was LINKED against, and
# so do the FFmpeg and dav1d archives inside it, so all three are built here.
#
#   docker run --rm -v "$PWD:$PWD" -w "$PWD" -e FFMPEG_DIR="$PWD/ffmpeg-static" \
#       quay.io/pypa/manylinux_2_28_x86_64 scripts/manylinux-build.sh
#
# The same works with quay.io/pypa/manylinux_2_28_aarch64 on an ARM host.
#
# Mount the checkout at the SAME path inside the container as outside it: the
# prefix's .pc files record absolute paths, and build.rs reads its link flags
# from them, so a prefix built here is then usable outside (the window's
# release job links the same archives).
#
# Runs as root only long enough to install the build tools, then as the owner
# of the checkout: everything it writes is then theirs, and the tests that make
# a folder read-only mean what they say (root writes through 0o500).
#
# FFMPEG_DIR is required. CARGO_TARGET_DIR is honoured if set.

set -euo pipefail
: "${FFMPEG_DIR:?set FFMPEG_DIR to where the static FFmpeg prefix goes}"
cd "$(dirname "$0")/.."

if [[ "$(id -u)" == 0 ]]; then
    # clang is for bindgen (ffmpeg-sys-next). nasm is x86 only: FFmpeg and
    # dav1d use it for their x86 assembly and need nothing like it on ARM.
    pkgs=(clang)
    case "$(uname -m)" in x86_64 | i?86) pkgs+=(nasm) ;; esac
    dnf install -y -q "${pkgs[@]}"

    uid="$(stat -c %u .)"
    gid="$(stat -c %g .)"
    if [[ "$uid" != 0 ]]; then
        getent group "$gid" >/dev/null || groupadd -g "$gid" builder
        getent passwd "$uid" >/dev/null || useradd -m -u "$uid" -g "$gid" builder
        home="$(getent passwd "$uid" | cut -d: -f6)"
        exec runuser -u "$(getent passwd "$uid" | cut -d: -f1)" -- \
            env HOME="$home" "$0" "$@"
    fi
fi

if ! command -v cargo >/dev/null 2>&1; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
cargo --version
ldd --version | sed -n 1p  # not head: under pipefail its early exit SIGPIPEs ldd

./scripts/build-ffmpeg-static.sh "$FFMPEG_DIR"

# The same feature flags for both, so the test build reuses the release
# artifacts and target/release/vid-fp is still the binary that ships.
cargo build --release --locked --features static-ffmpeg
cargo test --release --locked --features static-ffmpeg
