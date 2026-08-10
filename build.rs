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

fn main() {
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");

    if std::env::var_os("CARGO_FEATURE_STATIC_FFMPEG").is_none() {
        return;
    }

    // FFMPEG_DIR is honoured first, for CI and for anyone relinking against a
    // substituted FFmpeg. Falling back to the script's own default location
    // keeps the env var out of the everyday build, which matters more than it
    // looks: FFMPEG_DIR cannot simply be exported from a shell profile, because
    // ffmpeg-sys-next reads it whether or not this feature is on and a plain
    // dynamic build would then try to link these static archives and fail.
    // Making the common case need no variable removes the temptation.
    let manifest = std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    let dir = match std::env::var_os("FFMPEG_DIR") {
        Some(dir) => std::path::PathBuf::from(dir),
        None => {
            let default = manifest.join("ffmpeg-static");
            if !default.is_dir() {
                // Falling through would let ffmpeg-sys-next link the system's
                // shared FFmpeg, producing a binary that claims to be static
                // and still refuses to start on a host with a different
                // libavcodec soname.
                panic!(
                    "--features static-ffmpeg needs a static FFmpeg prefix, and neither \
                     FFMPEG_DIR nor {} is set/present.\n\
                     Build one with: ./scripts/build-ffmpeg-static.sh",
                    default.display()
                );
            }
            default
        }
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
