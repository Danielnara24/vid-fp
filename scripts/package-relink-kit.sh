#!/usr/bin/env bash
#
# Packages the LGPL relink kit for the static FFmpeg in ./ffmpeg-static into
# dist/KIT.tar.gz. release.yml runs it once per architecture.
#
#   scripts/package-relink-kit.sh VERSION KIT
#
# FFmpeg is LGPL 2.1+, and vid-fp links it statically. LGPL 2.1 section 6
# lets us ship that only if recipients can relink the executable against
# their own modified FFmpeg, so the archives, headers and the exact
# recipe go out with every release. This is what keeps vid-fp itself
# under MIT OR Apache-2.0; it is not a nice-to-have, and a release
# missing it is not compliant.

set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="$1"
KIT="$2"
mkdir -p dist
mkdir -p "dist/$KIT"
cp -r ffmpeg-static/lib "dist/$KIT/lib"
cp -r ffmpeg-static/include "dist/$KIT/include"
cp scripts/build-ffmpeg-static.sh "dist/$KIT/"
# The .pc files record this runner's absolute paths, which exist on no
# other machine. Rewriting them to the ${pcfiledir} form -- which
# pkg-config resolves relative to the .pc file itself -- makes the
# unpacked kit usable as an FFMPEG_DIR wherever it lands, which is the
# whole point of shipping it, since build.rs reads its link flags from
# exactly these files.
#
# Every occurrence is replaced, not just the prefix/libdir/includedir
# variables: FFmpeg bakes the --extra-ldflags we passed it into
# Libs.private as a literal -L<abs>/lib, so rewriting only the
# variables leaves dav1d resolving to a path on this runner. The
# prefix= line is rewritten last, since the first expression would
# otherwise turn it into a self-reference.
# The path to rewrite is read out of the .pc file rather than assumed
# to be $PWD/ffmpeg-static: the two are the same in CI, but a silent
# mismatch produces a kit that looks fine and resolves to a directory
# the user does not have.
FFPREFIX="$(sed -n 's|^prefix=||p' \
  "dist/$KIT/lib/pkgconfig/libavcodec.pc" | head -1)"
if [ -z "$FFPREFIX" ]; then
  echo "::error::could not read prefix= from libavcodec.pc"; exit 1
fi
sed -i \
  -e "s|$FFPREFIX|\${prefix}|g" \
  -e 's|^prefix=.*|prefix=${pcfiledir}/../..|' \
  "dist/$KIT"/lib/pkgconfig/*.pc
# Prove it before shipping it. Both halves earned their place: the
# grep for a leftover absolute path caught Libs.private, and resolving
# from the unpacked copy caught libdir.
if grep -rq -- "$FFPREFIX" "dist/$KIT"/lib/pkgconfig/; then
  echo "::error::relink kit still refers to the runner's build prefix:"
  grep -rn -- "$FFPREFIX" "dist/$KIT"/lib/pkgconfig/ || true
  exit 1
fi
KITDIR="$PWD/dist/$KIT"
KITLIBS="$(PKG_CONFIG_PATH="$KITDIR/lib/pkgconfig" \
             pkg-config --libs --static libavcodec)"
echo "relink kit link flags: $KITLIBS"
case "$KITLIBS" in
  *-ldav1d*) ;;
  *) echo "::error::relink kit does not link dav1d"; exit 1 ;;
esac
# Every -L must land inside the kit. This is the assertion that
# actually proves relocation, rather than merely that the strings
# changed shape.
for flag in $KITLIBS; do
  case "$flag" in
    -L*)
      case "${flag#-L}" in
        "$KITDIR"/*) ;;
        *) echo "::error::kit points outside itself: $flag"; exit 1 ;;
      esac ;;
  esac
done
cp THIRD-PARTY-LICENSES.md "dist/$KIT/"
cat > "dist/$KIT/RELINKING.md" <<EOF
# Relinking vid-fp against your own FFmpeg

vid-fp ${VERSION} for $(uname -m) statically links FFmpeg (LGPL 2.1+) and
dav1d (BSD-2-Clause). LGPL 2.1 section 6 entitles you to modify FFmpeg and
relink this program against your version. Everything you need is here.

\`lib/\` holds the exact static archives this release was linked
against, \`include/\` their headers, and \`lib/pkgconfig/\` the
descriptions the build reads its link flags from. The .pc files have
been made relocatable, so this directory works as an FFMPEG_DIR from
wherever you unpack it.

## Relinking against a modified FFmpeg

    git clone https://github.com/Danielnara24/vid-fp
    cd vid-fp && git checkout v${VERSION}

    # Reproduce our FFmpeg, edit it, rebuild:
    ./scripts/build-ffmpeg-static.sh "\$PWD/ffmpeg-static"
    #   ... patch \$PWD/ffmpeg-static/.build/ffmpeg, then:
    #   make -C "\$PWD/ffmpeg-static/.build/ffmpeg" install

    FFMPEG_DIR="\$PWD/ffmpeg-static" \\
      cargo build --release --features static-ffmpeg

Any prefix laid out like this one will do, so a wholly separate FFmpeg
build can be substituted instead:

    <prefix>/lib/lib{avcodec,avformat,avutil,swscale,dav1d}.a
    <prefix>/lib/pkgconfig/*.pc
    <prefix>/include/lib{avcodec,avformat,avutil,swscale}/...

## Verifying the result

    objdump -p target/release/vid-fp | grep NEEDED   # no libav*/libsw*

The build this release came from used FFmpeg and dav1d at the versions
pinned at the top of build-ffmpeg-static.sh, configured with the exact
flags in that file.
EOF
tar -C dist -czf "dist/${KIT}.tar.gz" "$KIT"
rm -rf "dist/$KIT"
