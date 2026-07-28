mod clustering;
mod compare;
mod export;
mod fingerprint;
mod utils;

use anyhow::{Context, Result};
use clap::Parser;
use compare::find_all_matches;
use fingerprint::{fingerprint_video, VideoFingerprint};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;
use std::collections::HashSet;
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use utils::{shutdown_requested, Priority};
use walkdir::WalkDir;

// Cache schema versioning - bump this if VideoFingerprint struct ever changes!
const CACHE_VERSION: &str = "v1";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Folders to scan (one or more)
    #[arg(required = true, num_args = 1.., value_name = "FOLDER")]
    include: Vec<String>,

    /// Folder to exclude from the scan. Repeat the flag to exclude several
    /// (e.g. -e ~/a -e ~/b).
    #[arg(short = 'e', long = "exclude", value_name = "FOLDER")]
    exclude: Vec<String>,

    /// Scan folders recursively. Off by default (only the given folders and
    /// their immediate files are scanned).
    #[arg(short = 'r', long = "recursive")]
    recursive: bool,

    /// Follow symbolic links while scanning. Off by default, which means a
    /// symlinked directory is not descended into. Safe to enable: files are
    /// deduplicated by (device, inode), so following a link never fingerprints
    /// the same bytes twice.
    #[arg(long = "follow-symlinks")]
    follow_symlinks: bool,

    /// Video file extensions to search for (case-insensitive; a leading dot is
    /// optional). Repeat the flag or comma-separate, e.g. `-x mp4,mkv` or
    /// `-x mp4 -x mkv`. Defaults to the common video containers.
    #[arg(
        short = 'x',
        long = "extensions",
        value_delimiter = ',',
        value_name = "EXT",
        default_values_t = ["mp4", "mkv", "avi", "mov", "flv", "webm"].map(String::from)
    )]
    extensions: Vec<String>,

    /// Maximum Hamming distance.
    /// Higher = looser matching, lower = stricter matching. Default is 3.
    #[arg(short = 'd', long = "hamming-distance", default_value_t = 3)]
    hamming_distance: u32,

    /// Minimum match percentage required to be considered a duplicate. Default is 10.0 (10%).
    #[arg(short = 'p', long = "match-percent", default_value_t = 10.0)]
    match_percent: f32,

    /// Base keyframe sampling interval in seconds (0 = decode every keyframe).
    /// Long videos sample at this interval; short videos use a finer interval
    /// automatically so they keep at least --min-keyframes frames.
    #[arg(long = "keyframe-interval", default_value_t = 0.0)]
    kf_interval: f64,

    /// Minimum keyframes to keep for short videos. They use a finer interval
    /// automatically so subsampling never drops them below this count.
    /// Default is 4.0.
    /// This is only used when --keyframe-interval is > 0.0.
    #[arg(long = "min-keyframes", default_value_t = 4.0)]
    min_kf_samples: f64,

    /// Priority for determining the best file to KEEP
    #[arg(short = 'k', long = "priority", default_value = "length")]
    priority: Priority,

    /// Output file for the results (supports .txt, .csv, .json)
    #[arg(short = 'o', long = "output")]
    output: Option<String>,

    /// Delete the files marked DELETE. By default they are moved to the system
    /// trash (recoverable). Files marked KEEP or REVIEW are never touched.
    #[arg(long = "delete")]
    delete: bool,

    /// With --delete, remove files permanently instead of moving them to the
    /// trash. Irreversible — use with care. Has no effect without --delete.
    #[arg(long = "permanent")]
    permanent: bool,

    /// Delete ALL cache before running
    #[arg(long = "clear-cache")]
    clear_cache: bool,

    /// Delete the cache of files not included in the current folder to scan
    #[arg(long = "prune-cache")]
    prune_cache: bool,

    /// Suppress all terminal output except errors
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,

    /// Maximum number of threads to use. 0 uses all available CPU cores (default).
    #[arg(short = 't', long = "threads", default_value_t = 0)]
    threads: usize,
}

/// How the run ended. `Interrupted` carries the phase name purely so the final
/// message can tell the user where it stopped.
enum Outcome {
    Completed,
    Interrupted,
}

/// Arrange for Ctrl-C (and, thanks to the `termination` feature, SIGTERM and
/// SIGHUP) to unwind the run instead of killing it.
///
/// The handler does the absolute minimum: flip one atomic and print. It never
/// touches the database, never takes a lock, and never waits on a worker, so it
/// cannot deadlock against whatever the program happened to be doing.
///
/// The second signal is the escape hatch. It bypasses the clean shutdown, which
/// is tolerable precisely because fingerprints are written as they are produced
/// -- at worst it costs whatever sled's flusher has not yet fsynced.
fn install_signal_handler() -> Result<()> {
    let hits = AtomicUsize::new(0);

    ctrlc::set_handler(move || {
        if hits.fetch_add(1, Ordering::SeqCst) == 0 {
            utils::request_shutdown();
            // Straight to stderr, not through the logger, so it still shows
            // under --quiet. The user pressed a key; they get an answer now.
            eprintln!(
                "\nInterrupt received — saving finished fingerprints to cache."
            );
        } else {
            eprintln!("\nSecond interrupt — quitting now.");
            std::process::exit(130);
        }
    })
    .context("Failed to install signal handler")
}

fn main() -> Result<()> {
    let start_time = Instant::now();
    let args = Args::parse();

    // 1. Initialize custom CLI Logger
    let log_level = if args.quiet {
        log::LevelFilter::Error
    } else {
        log::LevelFilter::Info
    };

    env_logger::Builder::new()
        .filter_level(log_level)
        .format(|buf, record| {
            if record.level() == log::Level::Error {
                writeln!(buf, "Error: {}", record.args())
            } else {
                writeln!(buf, "{}", record.args()) // Clean output for CLI tools
            }
        })
        .init();

    // Installed before any real work so even the directory walk is cancellable.
    install_signal_handler()?;

    // --- Allocator tuning (Linux / glibc) ------------------------------------
    // Each video's frame data is now a single large buffer. We pin glibc's mmap
    // and trim thresholds low and FIXED so those buffers are always served by
    // mmap and handed straight back to the OS the moment a video finishes,
    // instead of being parked on the main heap where RSS ratchets up across the
    // run. Pinning them also disables glibc's *dynamic* mmap-threshold growth,
    // which would otherwise start routing big buffers back onto the heap after a
    // few large frees. (This is why MALLOC_ARENA_MAX had no effect: the issue was
    // heap retention of freed memory, not the number of arenas.)
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::mallopt(libc::M_MMAP_THRESHOLD, 1024 * 1024); // 1 MiB
        libc::mallopt(libc::M_TRIM_THRESHOLD, 1024 * 1024);
    }
    // Configure Rayon thread pool if a specific limit is requested

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .context("Failed to configure global thread pool")?;
    }

    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let active_threads = if args.threads > 0 { args.threads } else { default_threads };

    ffmpeg_next::init().context("Failed to initialize FFmpeg bindings.")?;
    ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

    // Follow XDG Base Directory Specification for caching
    let cache_dir = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home).join(".cache")
            } else {
                PathBuf::from("/tmp")
            }
        })
        .join("vid-fp");

    std::fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;
    let db_path = cache_dir.join("video_hashes.db");

    // A background flush every second means the cache survives even the paths
    // that skip our own shutdown: a second Ctrl-C, a SIGKILL, a power cut.
    // Nothing already fingerprinted is ever more than a second from durable.
    let db = sled::Config::new()
        .path(&db_path)
        .flush_every_ms(Some(1_000))
        .open()
        .context("Failed to open or lock cache database")?;

    let outcome = run(&args, &db, start_time, active_threads);

    // --- The only exit path ---------------------------------------------------
    // Every route out of `run` lands here -- success, failure, or interrupt --
    // and the database is flushed and DROPPED before the process is allowed to
    // end. The drop is the point: std::process::exit skips destructors, so
    // exiting with a live Db means sled never runs its own shutdown and its
    // flusher thread dies mid-write. That is how "saved 20 videos" becomes 6 on
    // the next run. Flush, drop, then leave.
    if let Err(e) = db.flush() {
        log::error!("Failed to flush the fingerprint cache: {}", e);
    }
    drop(db);

    match outcome? {
        Outcome::Completed => Ok(()),
        Outcome::Interrupted => {
            // 130 is the shell convention for "terminated by SIGINT".
            std::process::exit(130);
        }
    }
}

fn run(args: &Args, db: &sled::Db, start_time: Instant, active_threads: usize) -> Result<Outcome> {
    if args.clear_cache {
        info!("Clearing all cache...");
        db.clear().context("Failed to clear cache database")?;
    }

    let max_hamming = args.hamming_distance;
    let min_match_pct = args.match_percent / 100.0;

    // Normalize the requested extensions: strip an optional leading dot and
    // lowercase, so `-x .MP4`, `-x MP4`, and `-x mp4` all behave identically.
    // A HashSet gives O(1) lookups during the walk and dedups automatically.
    let extensions: HashSet<String> = args
        .extensions
        .iter()
        .map(|e| e.trim().trim_start_matches('.').to_lowercase())
        .filter(|e| !e.is_empty())
        .collect();

    if extensions.is_empty() {
        anyhow::bail!("No valid video extensions to search for (--extensions was empty).");
    }

    let mut video_files = Vec::new();

    info!("Scanning folders: {:?}", args.include);
    if !args.exclude.is_empty() {
        info!("Excluding folders: {:?}", args.exclude);
    }

    // HashSet iteration order is unspecified; sort for a stable, readable log.
    let mut ext_display: Vec<&str> = extensions.iter().map(|s| s.as_str()).collect();
    ext_display.sort_unstable();
    info!("Searching extensions: {:?}", ext_display);

    info!(
        "Settings -> Max Hamming: {}, Min Match: {}%, Priority: {:?}, Threads: {}, Recursive: {}, Follow symlinks: {}",
        max_hamming, args.match_percent, args.priority, active_threads, args.recursive, args.follow_symlinks
    );

    // Canonicalize exclude paths so prefix matching is safe and reliable
    let exclude_paths: Vec<PathBuf> = args
        .exclude
        .iter()
        .filter_map(|p| std::fs::canonicalize(p).ok())
        .collect();

    // Identity-based deduplication. A symlink, a hard link, a second scan root
    // that overlaps the first, and a bind-mount alias all resolve to the same
    // (device, inode) pair. Keying on that identity means each set of bytes is
    // fingerprinted exactly once, under the first path we reach it by -- so the
    // report never lists a file as a duplicate of itself, and the "space freed"
    // figure never counts bytes that deleting a link would not actually return.
    let mut seen_inodes: HashSet<(u64, u64)> = HashSet::new();
    let mut alias_skipped = 0usize;

    for include_dir in &args.include {
        let base_path = match std::fs::canonicalize(include_dir) {
            Ok(p) => p,
            Err(e) => {
                log::error!("Could not resolve include path '{}': {}", include_dir, e);
                continue;
            }
        };

        let mut walker = WalkDir::new(&base_path).follow_links(args.follow_symlinks);

        if !args.recursive {
            // Non-recursive by default: limit depth so only the directory itself
            // and its immediate files are scanned.
            // 0 = The directory itself, 1 = Immediate files inside the directory
            walker = walker.max_depth(1);
        }

        // Filter out any paths that begin with an excluded directory path
        let it = walker.into_iter().filter_entry(|e| {
            let p = e.path();
            !exclude_paths.iter().any(|ex| p.starts_with(ex))
        });

        for entry in it.filter_map(|e| e.ok()) {
            if shutdown_requested() {
                return Ok(Outcome::Interrupted);
            }

            let path = entry.path();

            // Extension first: it is free, and it keeps us from stat()ing every
            // non-video file in the tree.
            let ext_matches = path
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| extensions.contains(ext.to_lowercase().as_str()))
                .unwrap_or(false);

            if !ext_matches {
                continue;
            }

            // One stat does three jobs: it follows symlinks (so a link to a
            // video is treated as that video), confirms this is a regular file,
            // and yields the identity used for deduplication.
            let meta = match std::fs::metadata(path) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("Cannot stat {}: {}", path.display(), e);
                    continue;
                }
            };

            if !meta.is_file() {
                continue;
            }

            if !seen_inodes.insert((meta.dev(), meta.ino())) {
                alias_skipped += 1;
                log::debug!(
                    "Skipping {}: same inode as a path already queued",
                    path.display()
                );
                continue;
            }

            video_files.push(path.to_string_lossy().to_string());
        }
    }

    if alias_skipped > 0 {
        info!(
            "Skipped {} path(s) resolving to files already queued (symlinks, hard links, or overlapping folders).",
            alias_skipped
        );
    }

    if args.prune_cache && !args.clear_cache {
        info!("Pruning cache for files not in the current scan...");
        let valid_files: HashSet<&str> = video_files.iter().map(|s| s.as_str()).collect();
        let mut batch = sled::Batch::default();
        let mut pruned_count = 0;

        for kv in db.iter() {
            if shutdown_requested() {
                return Ok(Outcome::Interrupted);
            }

            if let Ok((key_bytes, _)) = kv {
                let mut should_remove = true;
                if let Ok(key_str) = std::str::from_utf8(&key_bytes) {
                    // Extract filepath from cache key (format: filepath_mtime_size_version)
                    let mut parts = key_str.rsplitn(6, '_');
                    let _version = parts.next();
                    let _size = parts.next();
                    let _mtime = parts.next();
                    let _kf_interval = parts.next();
                    let _min_kf_samples = parts.next();
                    if let Some(filepath) = parts.next() {
                        if valid_files.contains(filepath) {
                            should_remove = false; // It's still valid, keep it
                        }
                    }
                }

                if should_remove {
                    batch.remove(key_bytes);
                    pruned_count += 1;
                }
            }
        }

        if pruned_count > 0 {
            db.apply_batch(batch).context("Failed to apply cache pruning")?;
            info!("Pruned {} stale entries from cache.", pruned_count);
        } else {
            info!("No stale entries found to prune.");
        }
    }

    if video_files.is_empty() {
        info!("No videos found.");
        return Ok(Outcome::Completed);
    }

    video_files.sort_by_cached_key(|vf| {
        std::cmp::Reverse(std::fs::metadata(vf).map(|m| m.len()).unwrap_or(0))
    });

    let total_videos = video_files.len();

    let kf_interval = args.kf_interval;
    let min_kf_samples = args.min_kf_samples;
    if kf_interval > 0.0 {
        info!("Using keyframe interval: {}s, minimum keyframes: {}", kf_interval, min_kf_samples);
    }

    // --- Pass 1: resolve the entire cache before decoding anything ------------
    // This used to be interleaved with fingerprinting in a single par_iter, and
    // rayon gives each worker a contiguous slice of the input: a cached file
    // sitting behind an uncached one in the same slice could not be read until
    // that decode finished, no matter how many threads were idle. Hence a bar
    // stuck at 7/22 while 21 files were already known and exactly one was being
    // worked on.
    //
    // A lookup is a stat, a tree read and a bincode decode -- microseconds -- so
    // this pass finishes effectively instantly even on a large library, and by
    // the time the bar appears we know exactly how much real work there is.
    enum Lookup {
        Hit(VideoFingerprint),
        Miss { path: String, cache_key: String },
        Unreadable,
    }

    let lookups: Vec<Lookup> = video_files
        .par_iter()
        .map(|vf| {
            let metadata = match std::fs::metadata(vf) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("Cannot access metadata for {}: {}", vf, e);
                    return Lookup::Unreadable;
                }
            };

            let mtime = metadata
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let size = metadata.len();

            // Versioned cache key ensures schema changes don't cause deserialization crashes
            let cache_key = format!("{}_{}_{}_{}_{}_{}", vf, mtime, size, CACHE_VERSION, kf_interval, min_kf_samples);

            if let Ok(Some(data)) = db.get(&cache_key) {
                match bincode::deserialize::<VideoFingerprint>(&data) {
                    Ok(fp) => return Lookup::Hit(fp),
                    Err(e) => {
                        // Corrupted or outdated schema; ignore and re-process.
                        log::debug!("Cache deserialization failed for {}: {}. Re-processing.", vf, e);
                    }
                }
            }

            Lookup::Miss { path: vf.clone(), cache_key }
        })
        .collect();

    // collect() preserves input order, so `todo` inherits the largest-first sort
    // and the heaviest decodes still start first.
    let mut fingerprints: Vec<VideoFingerprint> = Vec::with_capacity(total_videos);
    let mut todo: Vec<(String, String)> = Vec::new();
    for lookup in lookups {
        match lookup {
            Lookup::Hit(fp) => fingerprints.push(fp),
            Lookup::Miss { path, cache_key } => todo.push((path, cache_key)),
            Lookup::Unreadable => {}
        }
    }

    let cached_count = fingerprints.len();
    let todo_count = todo.len();

    if cached_count > 0 {
        info!(
            "Found {} video files; {} already cached, {} to fingerprint.",
            total_videos, cached_count, todo_count
        );
    } else {
        info!("Found {} video files. Fingerprinting...", total_videos);
    }

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    // --- Pass 2: the work that actually costs something -----------------------
    // Declared out here so the counters survive the block and can be reported
    // even when every file was cached and the block never ran.
    let newly_cached = AtomicUsize::new(0);
    let abandoned = AtomicUsize::new(0);

    if todo_count > 0 {
        // The bar now counts decodes and nothing else, so its denominator is the
        // work remaining rather than the size of the library.
        let pb = if args.quiet {
            ProgressBar::hidden()
        } else {
            let pb = ProgressBar::new(todo_count as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - {msg}",
                )
                .unwrap()
                .progress_chars("=>-"),
            );
            pb
        };

        let fresh: Vec<VideoFingerprint> = todo
            .par_iter()
            .filter_map(|(vf, cache_key)| {
                // Cheapest possible bail-out. Every video still queued costs one
                // relaxed atomic load, so the tail of a 50k-file scan drains in
                // microseconds; the videos actually being decoded right now stop
                // via the identical check inside fingerprint_video's demux loop.
                if shutdown_requested() {
                    return None;
                }

                let file_name = Path::new(vf).file_name().unwrap_or_default().to_string_lossy().into_owned();
                pb.set_message(file_name);

                let fp = match fingerprint_video(vf, kf_interval, min_kf_samples) {
                    Ok(f) => f,
                    Err(e) => {
                        // A video that abandoned its decode because of Ctrl-C is
                        // not a failure, and printing one line per in-flight
                        // worker would just bury the interrupt message.
                        if shutdown_requested() {
                            abandoned.fetch_add(1, Ordering::Relaxed);
                        } else {
                            log::error!("Failed to process {}: {:#}", vf, e);
                        }
                        pb.inc(1); // Keep the bar consistent with work attempted
                        return None;
                    }
                };

                // Written to the tree the moment it exists, not staged in a
                // shared batch. The batch this replaces held up to 31 finished
                // fingerprints in RAM -- precisely the work an interrupt was
                // throwing away -- and on a small library the 32-item threshold
                // was never reached at all, so NOTHING was written until the run
                // ended. One sled insert costs tens of microseconds against a
                // decode measured in seconds.
                match bincode::serialize(&fp) {
                    Ok(encoded) => match db.insert(cache_key.as_bytes(), encoded) {
                        Ok(_) => {
                            newly_cached.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => log::error!("Failed to cache fingerprint for {}: {}", vf, e),
                    },
                    Err(e) => log::error!("Failed to serialize fingerprint for {}: {}", vf, e),
                }

                pb.inc(1);
                Some(fp)
            })
            .collect();

        pb.finish_and_clear();
        fingerprints.extend(fresh);
    }

    let newly_cached = newly_cached.into_inner();

    if shutdown_requested() {
        info!(
            "Cached {} video(s); {} of {} are now cached in total.",
            newly_cached,
            fingerprints.len(),
            total_videos
        );
        return Ok(Outcome::Interrupted);
    }

    let n = fingerprints.len();
    if n < 2 {
        info!("Not enough valid videos to compare.");
        return Ok(Outcome::Completed);
    }

    info!("\nFingerprinting complete. Cross-analyzing {} videos...", n);

    let edges = find_all_matches(&fingerprints, max_hamming, min_match_pct);

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    info!("Grouping duplicate clusters...");

    let final_groups = clustering::find_duplicate_groups(n, edges, &fingerprints);

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    // Announce destructive intent up front so it's visible above the per-file log.
    if args.delete {
        if args.permanent {
            info!("\n--permanent enabled: files marked DELETE will be removed permanently.");
        } else {
            info!("\n--delete enabled: files marked DELETE will be moved to the trash.");
        }
    }

    export::output_results(
        &final_groups,
        &fingerprints,
        args.output.as_ref(),
        start_time.elapsed().as_secs(),
        args.priority,
        args.delete,
        args.permanent,
    )?;

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    Ok(Outcome::Completed)
}