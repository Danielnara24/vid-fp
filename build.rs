//! Link the extra libraries a *static* FFmpeg needs, and nothing else.
//!
//! The dynamic build needs no help: `ffmpeg-sys-next` finds the shared FFmpeg
//! through pkg-config and the loader resolves dav1d, zlib and friends at run
//! time because libavcodec.so records them itself.
//!
//! A static build has no such record. `ffmpeg-sys-next`'s `FFMPEG_DIR` branch
//! only emits the libav* archives it was told about by its own cargo features
//! -- it never consults pkg-config on that prefix -- so libdav1d.a and libz
//! are simply left out and the link fails on a wall of undefined AV1 symbols.
//! Probing the prefix's own .pc files puts them back, and keeps working if
//! `scripts/build-ffmpeg-static.sh` ever enables another external library.
//!
//! Emitting the libav* archives a second time is deliberate rather than
//! sloppy. Static linking is order-sensitive: a dependency has to appear
//! *after* whatever needs it, and our flags land after `ffmpeg-sys-next`'s. The
//! second copy of libavcodec.a is what sits in front of libdav1d.a.
//!
//! **`FFMPEG_DIR` is required, and this script cannot supply it.** An earlier
//! version defaulted to `./ffmpeg-static` when the variable was unset, on the
//! theory that the everyday build should need no environment. That theory was
//! wrong, and wrong in the silent direction. `ffmpeg-sys-next`'s build script
//! runs *before* this one, in its own process, and `FFMPEG_DIR` is the only
//! prebuilt-prefix branch it has; with the variable unset it falls through to
//! `pkg_config::Config::new().statik(true).probe("libavutil")` against the
//! SYSTEM. By the time this script runs, the bindings and struct layouts have
//! already been generated from the system's headers, and nothing done here --
//! not the fallback path, not `set_var("PKG_CONFIG_PATH", ...)`, which reaches
//! only our own four probes -- can reach back and change that. A downstream
//! build script cannot influence an upstream one's environment.
//!
//! What that produced on this machine: bindings from libavcodec 60 (FFmpeg 6)
//! over archives resolved first out of `./ffmpeg-static` (libavcodec 62,
//! FFmpeg 8). It failed at `ld` here only because the system's link tail
//! (-lgme, -lopenmpt, -lgnutls ...) is not installed; on a host that has those
//! it links, and the result is the silently-wrong static binary this file
//! exists to prevent -- the same class of fault CLAUDE.md records as costing a
//! `CACHE_TABLE` rename, since decoder output is not bit-identical across
//! FFmpeg majors.
//!
//! It is also sticky. `ffmpeg-sys-next` emits no `rerun-if-env-changed` of any
//! kind, so once its build script has run in a target dir the choice is frozen:
//! exporting `FFMPEG_DIR` afterwards does NOT re-run it (measured -- the
//! bindings stayed at major 60), which is why the refusal below names
//! `cargo clean -p ffmpeg-sys-next` rather than just the variable -- and names it
//! with `--release`, because that clean is scoped to a single profile and a
//! bare one leaves a release target dir exactly as poisoned as it found it.
//!
//! A `.cargo/config.toml` `[env]` entry would satisfy the variable without the
//! user typing it, and is deliberately not used: `ffmpeg-sys-next` reads
//! `FFMPEG_DIR` whether or not this feature is on, so a plain dynamic
//! `cargo build` would then take the prebuilt branch too and try to link these
//! archives.
//!
//! The last line of defence is a compile-time one, because the refusal cannot
//! cover the stale-target-dir case (there `FFMPEG_DIR` *is* set and this script
//! is perfectly happy). The libavcodec version the prefix advertises is written
//! into `OUT_DIR` as a const assertion that `main.rs` includes, so a build whose
//! bindings came from anywhere else fails to compile instead of shipping.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");

    if std::env::var_os("CARGO_FEATURE_STATIC_FFMPEG").is_none() {
        return;
    }

    let manifest = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    let default = manifest.join("ffmpeg-static");

    // Not a fallback. See the module doc: with FFMPEG_DIR unset, ffmpeg-sys-next
    // has already bound itself to the system FFmpeg before this script starts,
    // and a prefix named here would only be the archive half of a mismatched
    // pair.
    let dir = match std::env::var_os("FFMPEG_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => panic!(
            "\n\
             --features static-ffmpeg requires FFMPEG_DIR to be set, and it is not.\n\
             \n\
             ffmpeg-sys-next runs before this build script and reads FFMPEG_DIR itself;\n\
             with it unset it has already generated its bindings against the SYSTEM\n\
             FFmpeg, which this script cannot undo. Linking would then resolve the\n\
             static archives against bindings from different headers.\n\
             \n\
             Build the prefix once, if you have not:\n\
             \n\
                 ./scripts/build-ffmpeg-static.sh\n\
             \n\
             then, because ffmpeg-sys-next emits no rerun-if-env-changed and will not\n\
             re-run on its own. Note that `cargo clean -p` is scoped to one profile,\n\
             so the flag has to match the build you are about to do:\n\
             \n\
                 cargo clean -p ffmpeg-sys-next --release\n\
                 FFMPEG_DIR={} cargo build --release --features static-ffmpeg\n",
            default.display()
        ),
    };

    let pkgconfig = dir.join("lib").join("pkgconfig");
    if !pkgconfig.is_dir() {
        panic!(
            "FFMPEG_DIR={} does not look like a static FFmpeg prefix: {} is missing.\n\
             Build one with: ./scripts/build-ffmpeg-static.sh",
            dir.display(),
            pkgconfig.display()
        );
    }

    write_version_check(&pkgconfig);

    // Ahead of any system path, so a host that also has a shared FFmpeg cannot
    // win the probe and link us against it by accident.
    let search = match std::env::var_os("PKG_CONFIG_PATH") {
        Some(existing) => format!("{}:{}", pkgconfig.display(), existing.to_string_lossy()),
        None => pkgconfig.display().to_string(),
    };
    std::env::set_var("PKG_CONFIG_PATH", search);

    println!(
        "cargo:rustc-link-search=native={}",
        dir.join("lib").display()
    );

    // Order matters within this list too: avcodec depends on avutil and on
    // dav1d, so the dependants are probed first.
    for lib in ["libavformat", "libavcodec", "libswscale", "libavutil"] {
        pkg_config::Config::new()
            .statik(true)
            // ffmpeg-sys-next already emitted the include paths and generated
            // the bindings; all we want out of this probe is link flags.
            .cargo_metadata(true)
            .probe(lib)
            .unwrap_or_else(|e| {
                panic!(
                    "failed to probe {lib} in {}: {e}\n\
                     The prefix may be incomplete -- try: ./scripts/build-ffmpeg-static.sh --force",
                    pkgconfig.display()
                )
            });
    }
}

/// Pin the bindings to the prefix whose archives we are about to link.
///
/// The refusal above catches an unset `FFMPEG_DIR`; it cannot catch a target
/// dir in which `ffmpeg-sys-next` already ran against the system and will not
/// run again (see the module doc). Both faults end in the same place -- one
/// FFmpeg's headers over another's archives -- so the check is made where that
/// is directly observable: `libavcodec.pc` states the version this prefix is,
/// and `ffmpeg_next::ffi::LIBAVCODEC_VERSION_*` states the version the bindings
/// were generated from. They have to be the same release.
fn write_version_check(pkgconfig: &Path) {
    let pc = pkgconfig.join("libavcodec.pc");
    let text = std::fs::read_to_string(&pc)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", pc.display()));

    let version = text
        .lines()
        .find_map(|l| l.strip_prefix("Version:"))
        .map(str::trim)
        .unwrap_or_else(|| panic!("{} has no Version: field", pc.display()));

    let parts: Vec<&str> = version.split('.').collect();
    let [major, minor, micro] = parts[..] else {
        panic!(
            "{} states an unparseable libavcodec version {version:?}",
            pc.display()
        )
    };

    let out = PathBuf::from(
        std::env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR for a build script"),
    )
    .join("static_ffmpeg_version_check.rs");

    // The message has to carry the expected version, because the compiler
    // prints this assertion without any of the surrounding context.
    let check = format!(
        r#"// Generated by build.rs. See `write_version_check` there.
const _: () = assert!(
    ffmpeg_next::ffi::LIBAVCODEC_VERSION_MAJOR == {major}
        && ffmpeg_next::ffi::LIBAVCODEC_VERSION_MINOR == {minor}
        && ffmpeg_next::ffi::LIBAVCODEC_VERSION_MICRO == {micro},
    "static-ffmpeg is about to link the archives in FFMPEG_DIR (libavcodec {version}), \
but ffmpeg-sys-next generated its bindings from a different libavcodec. That is a \
silently-wrong binary, so it is refused here. ffmpeg-sys-next caches which FFmpeg it \
bound to per target dir and emits no rerun-if-env-changed, so setting FFMPEG_DIR is not \
enough on its own: run `cargo clean -p ffmpeg-sys-next --release` (that clean is scoped to \
one profile, so drop --release when cleaning a debug build), then build again with \
FFMPEG_DIR set to the same prefix."
);
"#
    );
    std::fs::write(&out, check).unwrap_or_else(|e| panic!("failed to write {}: {e}", out.display()));
}
