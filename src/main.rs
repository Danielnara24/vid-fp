mod clustering;
mod compare;
mod export;
mod fingerprint;
mod stats;
mod utils;
mod sources;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use clap_complete::Shell;
use compare::{find_all_matches, MatchIndex};
use fingerprint::{fingerprint_video, VideoFingerprint, MAX_DECODE_THREADS};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use stats::RunStats;
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use utils::{shutdown_requested, Priority};

/// The one table in the cache: absolute file path -> bincode'd `CacheEntry`.
///
/// One entry per path, overwritten in place. That is the entire invalidation
/// story, and it is why the cache is bounded by the number of files ever
/// scanned rather than by the number of times they have changed: a file that is
/// re-encoded, re-muxed, or scanned with different sampling settings replaces
/// its own entry instead of growing a second one beside it.
const CACHE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("fingerprints_by_path");

/// The table this replaced, keyed by path AND mtime AND size AND the sampling
/// knobs.
///
/// That key was the leak. Touching a file wrote a NEW key and left the old one
/// behind for the life of the cache, and `--prune-cache` could not tell the two
/// apart because it only ever compared the path portion of the key -- so the
/// one tool for the job saw a live path and kept every dead entry under it.
/// Nothing reads this table now, so it is dropped whole on the first run.
const SUPERSEDED_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("fingerprints");

/// Hard ceiling on the cache's page cache.
///
/// A cached fingerprint is a few kilobytes, so even a fifty-thousand-file
/// library is a few hundred megabytes on disk and is read exactly once per run,
/// in key order. There is nothing here for a large cache to help with, and this
/// binary works hard elsewhere (see the `mallopt` calls below) to keep resident
/// memory flat across a long scan -- letting the database quietly grow a
/// gigabyte-scale cache would undo that.
const CACHE_SIZE_BYTES: usize = 64 * 1024 * 1024;

/// Exit code for a run that finished, reported, and did everything it could --
/// but hit at least one problem on the way (an unreadable file, a video that
/// would not decode, a deletion that failed). 0 means clean, 1 is anyhow's
/// fatal path, 130 is the shell convention for SIGINT. Scripts that don't care
/// can ignore it; scripts that do finally have something to test.
const EXIT_WITH_PROBLEMS: i32 = 2;

/// Everything other than the file's contents that decides whether a cached
/// fingerprint is still the right answer for the file at a given path.
///
/// This used to be spelled into the cache key. Moving it into the value is what
/// makes a path map to exactly one entry: on a mismatch the file is
/// re-fingerprinted and the insert OVERWRITES, rather than filing the new
/// fingerprint under a new key and leaving the old one to accumulate.
///
/// Fixed-size fields only, and serialized BEFORE the fingerprint, so the
/// fingerprint always begins at the same byte offset. That is what keeps the
/// discipline `VideoFingerprint` already documents working: a field appended to
/// the end of it makes older payloads run out of bytes, bincode fails, and the
/// file is fingerprinted again. Changing THIS struct moves the offset and has
/// no such guarantee, so it needs the same treatment `SUPERSEDED_TABLE` got.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
struct Stamp {
    mtime: u64,
    size: u64,
    kf_interval: f64,
    min_kf_samples: f64,
}

impl Stamp {
    /// Whether an entry stamped `self` may be reused for a file stamped
    /// `other`.
    ///
    /// Deliberately not `PartialEq`: `NaN != NaN`, and clap parses
    /// `--keyframe-interval nan` without complaint. Derived equality would make
    /// every lookup in such a run miss, re-decode the whole library, and miss
    /// again on the next run. Nothing would leak any more -- the entry is
    /// overwritten either way -- but there is no reason to decode twice for it.
    fn matches(&self, other: &Stamp) -> bool {
        self.mtime == other.mtime
            && self.size == other.size
            && same_setting(self.kf_interval, other.kf_interval)
            && same_setting(self.min_kf_samples, other.min_kf_samples)
    }
}

fn same_setting(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// What one cache value holds.
///
/// A tuple rather than a pair of structs so the borrowed write path
/// (`&(stamp, fp)`) and the owned read path cannot drift apart, and so storing
/// a fingerprint never has to clone one.
type CacheEntry = (Stamp, VideoFingerprint);

#[derive(Parser, Debug)]
#[command(
    author, version,
    about = "Fast video duplicate and clip finder",
    long_about = "Fingerprints videos from their keyframes and groups files with the \
                  same content, even at different resolutions or containers, and even \
                  when one video is only a trimmed clip inside another.\n\n\
                  Report-only by default: nothing is deleted unless --delete is given."
)]

struct Args {
        /// Folders and video files to scan (one or more). A folder is searched for
    /// videos; a file you name is scanned whatever its extension, since
    /// --extensions is only a guess about what a FOLDER contains. Use `-` to
    /// read a list of paths from stdin, e.g. `fd -e mkv | vid-fp -`.
    #[arg(
        required_unless_present_any = ["completions", "man", "from_file"],
        num_args = 1..,
        value_name = "PATH",
        value_hint = clap::ValueHint::AnyPath
    )]
    include: Vec<String>,

    /// Folder to exclude from the scan. Repeat the flag to exclude several
    /// (e.g. -e ~/a -e ~/b). Applies to piped and explicitly named paths too.
    #[arg(short = 'e', long = "exclude", value_name = "FOLDER",
          value_hint = clap::ValueHint::DirPath)]
    exclude: Vec<String>,

    /// Read the paths to scan from a file, one per line. `-` means stdin.
    /// Entries may be folders or files, exactly as if given as arguments, and
    /// combine with any paths already passed on the command line.
    #[arg(long = "from-file", value_name = "FILE",
          value_hint = clap::ValueHint::FilePath)]
    from_file: Option<String>,

    /// Paths in the list are separated by NUL bytes rather than newlines, for
    /// `find -print0` or `fd -0`. The only way to pass a filename containing a
    /// newline.
    #[arg(short = '0', long = "null")]
    null: bool,

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

    /// Minimum shared clip length, in seconds, for two videos to count as a
    /// match. Also skips fingerprinting any video shorter than this. 0 = off.
    /// Independent of --match-percent; both must be satisfied.
    #[arg(long = "min-duration", default_value_t = 0.0)]
    min_duration: f64,

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
    #[arg(short = 'o', long = "output", value_hint = clap::ValueHint::FilePath)]
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

    /// Print a shell completion script to stdout and exit.
    #[arg(long = "completions", value_name = "SHELL", exclusive = true)]
    completions: Option<Shell>,

    /// Print the roff man page to stdout and exit.
    #[arg(long = "man", exclusive = true)]
    man: bool,
}

/// How the run ended. `Interrupted` carries the phase name purely so the final
/// message can tell the user where it stopped.
enum Outcome {
    Completed,
    Interrupted,
}

fn decode_threads_for(budget: usize, jobs: usize) -> usize {
    let budget = budget.max(1);
    let concurrent = jobs.clamp(1, budget);
    (budget / concurrent).clamp(1, MAX_DECODE_THREADS)
}

struct JobSlot<'a> {
    running: &'a AtomicUsize,
    threads: usize,
}

impl<'a> JobSlot<'a> {
    fn claim(budget: usize, running: &'a AtomicUsize, queued: &AtomicUsize) -> Self {
        // fetch_* return the value from BEFORE the update, hence the +1 / -1:
        // this job is part of the work still outstanding, and is no longer
        // queued now that it has started.
        let now_running = running.fetch_add(1, Ordering::SeqCst) + 1;
        let still_queued = queued.fetch_sub(1, Ordering::SeqCst).saturating_sub(1);

        JobSlot {
            running,
            threads: decode_threads_for(budget, now_running.saturating_add(still_queued)),
        }
    }
}

impl Drop for JobSlot<'_> {
    fn drop(&mut self) {
        self.running.fetch_sub(1, Ordering::SeqCst);
    }
}

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

/// Create the cache table if this is a fresh database.
///
/// Opening a table inside a write transaction creates it, so this is also the
/// cheapest way to guarantee that every later *read* transaction finds
/// something to open -- reads cannot create tables, and a missing one would
/// otherwise turn every lookup on a new machine into an error path.
fn ensure_cache_table(db: &Database) -> Result<()> {
    let txn = db.begin_write().context("Failed to start a cache transaction")?;
    txn.open_table(CACHE_TABLE).context("Failed to create the cache table")?;
    txn.commit().context("Failed to commit the cache table")?;
    Ok(())
}

/// Drop the table this build's cache replaced, if it is still there.
///
/// Returns whether there was one, so the caller can decide about reclaiming the
/// space: deleting a table frees its pages inside the file but does not shrink
/// the file, and only a compaction hands that back to the filesystem.
fn retire_superseded_table(db: &Database) -> Result<bool> {
    let txn = db.begin_write().context("Failed to start a cache transaction")?;
    let existed = txn
        .delete_table(SUPERSEDED_TABLE)
        .context("Failed to remove the superseded cache table")?;
    txn.commit().context("Failed to commit the cache table removal")?;
    Ok(existed)
}

/// Read one fingerprint out of the cache, or `None` for anything that isn't a
/// clean hit.
///
/// A missing entry, an unreadable table, a payload that no longer deserializes,
/// and a payload whose stamp no longer describes the file on disk all mean
/// exactly the same thing to the caller -- fingerprint the file again -- so
/// they collapse into one answer here rather than making every call site handle
/// four failure shapes that need identical treatment. A stale entry is left
/// where it is on the way past, because the re-fingerprint is about to
/// overwrite it: one path, one entry, always.
///
/// The read transaction is per-call on purpose. It is a snapshot handle rather
/// than a lock, readers never block each other or the writer, and taking it
/// here keeps the whole thing usable from a rayon worker without threading a
/// borrow through the closure.
fn cache_lookup(db: &Database, path: &str, stamp: &Stamp) -> Option<VideoFingerprint> {
    let read = db.begin_read().ok()?;
    let table = read.open_table(CACHE_TABLE).ok()?;
    let stored = table.get(path).ok()??;

    match bincode::deserialize::<CacheEntry>(stored.value()) {
        Ok((cached, fp)) if cached.matches(stamp) => Some(fp),
        // The file was edited, or the sampling knobs moved. Either way the
        // fingerprint on record describes something that is no longer there.
        Ok(_) => None,
        Err(e) => {
            // Corrupt, or written by a build whose struct no longer matches.
            // Either way it is about to be overwritten by a fresh decode.
            log::debug!("Cache entry for {} did not deserialize ({}); re-processing.", path, e);
            None
        }
    }
}

/// Write one fingerprint to the cache, durably, replacing whatever was filed
/// under this path before.
///
/// One transaction per video, committed immediately -- deliberately not
/// batched. The batch this pattern replaces held finished fingerprints in
/// memory, which is precisely the work an interrupt threw away. A commit costs
/// an fsync against a decode measured in seconds, and when it returns the entry
/// is on disk: no background flusher, no window, nothing for a
/// `std::process::exit` to cut short.
///
/// Only one write transaction exists at a time, so concurrent callers queue
/// here. The serialization happens outside the transaction so that queue is
/// only ever holding a b-tree insert.
fn cache_store(db: &Database, path: &str, stamp: Stamp, fp: &VideoFingerprint) -> Result<()> {
    // `&(stamp, fp)` rather than an owned pair: serde follows the reference, so
    // this writes the exact bytes `CacheEntry` reads back without copying the
    // hash vectors to get there.
    let encoded = bincode::serialize(&(stamp, fp)).context("Failed to serialize fingerprint")?;

    let txn = db.begin_write().context("Failed to start a cache transaction")?;
    {
        let mut table = txn.open_table(CACHE_TABLE).context("Failed to open the cache table")?;
        table
            .insert(path, encoded.as_slice())
            .context("Failed to insert the fingerprint")?;
    }
    txn.commit().context("Failed to commit the fingerprint")?;

    Ok(())
}

/// Forget the fingerprints of files that are no longer where they were.
///
/// Called with what `--delete` actually removed. Those entries describe bytes
/// that are in the trash or gone entirely, so nothing will ever match them
/// again: keeping them means every cleanup run grows the cache a little and it
/// only shrinks if the user happens to remember `--prune-cache`. Deleting
/// duplicates is the one thing this tool does that makes an entry obsolete
/// WITHOUT writing a replacement over it, so it is the one place that has to
/// say so out loud.
///
/// One transaction for the whole list, unlike `cache_store`. There is no
/// partial state worth protecting here -- the files are already gone, and a run
/// killed midway simply leaves entries the next `--prune-cache` will collect --
/// so this pays one fsync rather than one per file.
///
/// Returns how many entries actually existed, which is normally all of them and
/// is only interesting when it isn't.
fn cache_forget(db: &Database, paths: &[String]) -> Result<usize> {
    if paths.is_empty() {
        return Ok(0);
    }

    let mut forgotten = 0usize;

    let txn = db.begin_write().context("Failed to start a cache transaction")?;
    {
        let mut table = txn.open_table(CACHE_TABLE).context("Failed to open the cache table")?;
        for path in paths {
            let existed = table
                .remove(path.as_str())
                .context("Failed to remove a cache entry")?
                .is_some();
            if existed {
                forgotten += 1;
            }
        }
    }
    txn.commit().context("Failed to commit the cache removals")?;

    Ok(forgotten)
}

/// Delete the cache this tool used to keep, if it is still lying around.
///
/// sled stored a directory; redb stores a single file, so the two cannot even
/// share a name and an old cache is bytes nothing will ever read again. Failure
/// is uninteresting -- the worst case is some dead weight in the cache
/// directory -- so it is logged at debug and otherwise ignored.
fn retire_legacy_cache(cache_dir: &Path) {
    let legacy = cache_dir.join("video_hashes.db");
    if !legacy.is_dir() {
        return;
    }

    match std::fs::remove_dir_all(&legacy) {
        Ok(()) => info!("Removed the superseded cache at {}.", legacy.display()),
        Err(e) => log::debug!(
            "Could not remove the superseded cache at {}: {}",
            legacy.display(),
            e
        ),
    }
}

fn main() -> Result<()> {
    let start_time = Instant::now();
    let args = Args::parse();

    if let Some(shell) = args.completions {
        let mut cmd = Args::command();
        let name = cmd.get_name().to_string();
        clap_complete::generate(shell, &mut cmd, name, &mut std::io::stdout());
        return Ok(());
    }

    if args.man {
        let mut buf: Vec<u8> = Vec::new();
        clap_mangen::Man::new(Args::command()).render(&mut buf)?;
        if !buf.ends_with(b"\n") {
            buf.push(b'\n');
        }
        buf.extend_from_slice(
b".SH EXIT STATUS
.TP
.B 0
Ran clean.
.TP
.B 1
Fatal error; the run did not complete.
.TP
.B 2
Completed, but something failed. See the Problems summary.
.TP
.B 130
Interrupted with Ctrl-C.
");
        std::io::stdout().write_all(&buf)?;
        return Ok(());
    }

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
    retire_legacy_cache(&cache_dir);

    let db_path = cache_dir.join("fingerprints.redb");

    // A single file, an explicitly bounded page cache, and no background
    // threads: every write is durable when its transaction commits, so there is
    // no buffered work to lose and nothing that has to be told to stop. Opening
    // takes an exclusive lock on the file, which is what keeps two concurrent
    // runs from fighting over the same cache.
    let mut db = Database::builder()
        .set_cache_size(CACHE_SIZE_BYTES)
        .create(&db_path)
        .context("Failed to open or lock cache database")?;

    ensure_cache_table(&db).context("Failed to initialize the fingerprint cache")?;

    // Once, on the first run of this build. The old table is keyed by a string
    // that no lookup will ever construct again, so every byte of it is dead --
    // and by construction it is the bigger of the two, since it held an entry
    // per version of every file rather than an entry per file. Compacting right
    // afterwards is the only thing that returns those pages to the filesystem.
    match retire_superseded_table(&db) {
        Ok(true) => {
            info!("Removed the superseded fingerprint cache; fingerprints will be rebuilt once.");
            if let Err(e) = db.compact() {
                log::error!("Could not compact the fingerprint cache: {}", e);
            }
        }
        Ok(false) => {}
        Err(e) => log::error!("{:#}", e),
    }

    // Every non-fatal problem and every deliberate skip lands here, from any
    // thread and any phase, so the run can account for itself at the end
    // instead of leaving the user to reconstruct it from scrollback.
    let stats = RunStats::default();

    let outcome = run(&args, &db, start_time, active_threads, &stats);

    // --- The only exit path ---------------------------------------------------
    // Every route out of `run` lands here -- success, failure, or interrupt.
    // There is nothing to flush: a fingerprint is on disk the moment its
    // transaction commits, so the cache does not depend on this code running at
    // all. The database is still dropped explicitly, before the process is
    // allowed to end, so the file lock is released rather than left for the
    // kernel to reap.
    //
    // Emptying a table frees its pages inside the file but does not shrink the
    // file, so a cleared or pruned cache would otherwise keep occupying whatever
    // it peaked at. Compaction is the only thing that hands that space back, and
    // it is skipped after an interrupt: the user asked for this to stop, and
    // rewriting the database is the opposite of stopping.
    if !shutdown_requested() && (args.clear_cache || args.prune_cache) {
        match db.compact() {
            Ok(true) => info!("Compacted the fingerprint cache."),
            Ok(false) => {}
            Err(e) => log::error!("Could not compact the fingerprint cache: {}", e),
        }
    }
    drop(db);

    // Printed last, deliberately: it is the part you act on, and after a long
    // scan it is the only part still on screen.
    stats.print_summary();

    match outcome? {
        Outcome::Completed => {
            if stats.had_problems() {
                std::process::exit(EXIT_WITH_PROBLEMS);
            }
            Ok(())
        }
        Outcome::Interrupted => {
            // 130 is the shell convention for "terminated by SIGINT", and it
            // takes precedence over EXIT_WITH_PROBLEMS: "you stopped it"
            // explains the unfinished work better than "something failed".
            std::process::exit(130);
        }
    }
}

fn run(
    args: &Args,
    db: &Database,
    start_time: Instant,
    active_threads: usize,
    stats: &RunStats,
) -> Result<Outcome> {
    if args.clear_cache {
        info!("Clearing all cache...");
        let txn = db.begin_write().context("Failed to start a cache transaction")?;
        txn.delete_table(CACHE_TABLE).context("Failed to clear cache database")?;
        txn.commit().context("Failed to commit the cache clear")?;

        // Dropping the table dropped its definition too; every later lookup
        // needs something to open.
        ensure_cache_table(db).context("Failed to recreate the cache table")?;
    }

    let max_hamming = args.hamming_distance;
    let min_match_pct = args.match_percent / 100.0;
    let min_duration = args.min_duration;
    if min_duration < 0.0 {
        anyhow::bail!("--min-duration cannot be negative.");
    }

    info!(
        "Settings -> Max Hamming: {}, Min Match: {}%, Min Duration: {}s, Priority: {:?}, Threads: {}, Recursive: {}",
        max_hamming, args.match_percent, min_duration, args.priority, active_threads, args.recursive
    );

    let mut video_files = match sources::collect(
        &sources::Sources {
            include: &args.include,
            exclude: &args.exclude,
            from_file: args.from_file.as_deref(),
            null_separated: args.null,
            extensions: &args.extensions,
            recursive: args.recursive,
            follow_symlinks: args.follow_symlinks,
        },
        stats,
    )? {
        sources::Scan::Complete(files) => files,
        sources::Scan::Interrupted => return Ok(Outcome::Interrupted),
    };

    if args.prune_cache && !args.clear_cache {
        info!("Pruning cache for files not in the current scan...");
        let valid_files: HashSet<&str> = video_files.iter().map(|s| s.as_str()).collect();

        // A key IS a path now, so this is the whole of pruning: an entry
        // survives exactly when the file it describes is in front of us. It
        // used to have to reconstruct a path out of a compound key and could
        // not see the difference between the current entry for a file and four
        // superseded ones -- all five matched a live path, so all five stayed.
        //
        // Collected under a read transaction and removed under a write one,
        // rather than filtered in place. Only this shape can abandon the scan
        // partway through on Ctrl-C without leaving a half-applied prune behind
        // -- the removals are one atomic commit or nothing at all.
        let mut stale: Vec<String> = Vec::new();
        {
            let read = db.begin_read().context("Failed to open the cache for reading")?;
            let table = read
                .open_table(CACHE_TABLE)
                .context("Failed to open the cache table")?;

            for entry in table.iter().context("Failed to iterate the cache")? {
                if shutdown_requested() {
                    return Ok(Outcome::Interrupted);
                }

                let (key, _) = entry.context("Failed to read a cache entry")?;
                let path = key.value();

                if !valid_files.contains(path) {
                    stale.push(path.to_string());
                }
            }
        }

        if !stale.is_empty() {
            let txn = db.begin_write().context("Failed to start a cache transaction")?;
            {
                let mut table = txn
                    .open_table(CACHE_TABLE)
                    .context("Failed to open the cache table")?;
                for key in &stale {
                    table
                        .remove(key.as_str())
                        .context("Failed to remove a stale cache entry")?;
                }
            }
            txn.commit().context("Failed to apply cache pruning")?;
            info!("Pruned {} stale entries from cache.", stale.len());
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
    // A lookup is a stat, a b-tree descent and a bincode decode -- microseconds
    // -- so this pass finishes effectively instantly even on a large library,
    // and by the time the bar appears we know exactly how much real work there
    // is. That count is now load-bearing twice over: it also sets the per-video
    // decoder thread budget, which is only knowable once the cache has spoken.
    enum Lookup {
        Hit(VideoFingerprint),
        Miss { path: String, stamp: Stamp },
        Unreadable,
    }

    let lookups: Vec<Lookup> = video_files
        .par_iter()
        .map(|vf| {
            let metadata = match std::fs::metadata(vf) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("Cannot access metadata for {}: {}", vf, e);
                    stats.unreadable.record(format!("{}: {}", vf, e));
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

            // Thread count is deliberately NOT part of the stamp. Threading
            // changes only how fast frames arrive, never which ones, so a
            // fingerprint made with 8 threads is byte-identical to one made with
            // 1 and stays valid across runs with different -t.
            let stamp = Stamp {
                mtime,
                size,
                kf_interval,
                min_kf_samples,
            };

            if let Some(fp) = cache_lookup(db, vf, &stamp) {
                return Lookup::Hit(fp);
            }

            Lookup::Miss { path: vf.clone(), stamp }
        })
        .collect();

    // collect() preserves input order, so `todo` inherits the largest-first sort
    // and the heaviest decodes still start first -- which also means the biggest
    // files are the ones that claim the widest thread allocations.
    let mut fingerprints: Vec<VideoFingerprint> = Vec::with_capacity(total_videos);
    let mut todo: Vec<(String, Stamp)> = Vec::new();
    for lookup in lookups {
        match lookup {
            Lookup::Hit(fp) => {
                if min_duration > 0.0 && fp.duration > 0.0 && fp.duration < min_duration {
                    // Already cached, so we know without decoding that it is too
                    // short to matter. Counted in the same bucket as the ones
                    // discovered by reading a header: from the user's side it is
                    // the same skip for the same reason.
                    stats.skipped_short.bump();
                } else {
                    fingerprints.push(fp);
                }
            }
            Lookup::Miss { path, stamp } => todo.push((path, stamp)),
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
    // Declared out here so the counter survives the block and can be reported
    // even when every file was cached and the block never ran.
    let newly_cached = AtomicUsize::new(0);

    if todo_count > 0 {
        // --- Thread budget ---------------------------------------------------
        let queued = AtomicUsize::new(todo_count);
        let running = AtomicUsize::new(0);

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
            .filter_map(|(vf, stamp)| {
                // Cheapest possible bail-out. Every video still queued costs one
                // relaxed atomic load, so the tail of a 50k-file scan drains in
                // microseconds; the videos actually being decoded right now stop
                // via the identical check inside fingerprint_video's demux loop.
                if shutdown_requested() {
                    return None;
                }

                // Claimed AFTER the bail-out, so a shutdown-abandoned job never
                // takes a share of the budget it will not use. Held for exactly
                // as long as the decode runs; the guard releases it on drop,
                // including on the error path below.
                let slot = JobSlot::claim(active_threads, &running, &queued);

                let file_name = Path::new(vf).file_name().unwrap_or_default().to_string_lossy().into_owned();
                pb.set_message(file_name);

                let fp = match fingerprint_video(vf, kf_interval, min_kf_samples, slot.threads, min_duration) {
                    Ok(Some(f)) => f,
                    Ok(None) => {
                        // Shorter than --min-duration. Not a failure, and nothing
                        // to cache: the header read that decided this is cheap
                        // enough to repeat next run.
                        stats.skipped_short.bump();
                        pb.inc(1);
                        return None;
                    }
                    Err(e) => {
                        // An interrupt unwinds through here as our own
                        // "Interrupted while fingerprinting ..." error. That is
                        // the user's doing rather than the file's, so it is
                        // neither logged as a failure nor counted anywhere:
                        // exit code 130 already says what happened.
                        if !shutdown_requested() {
                            log::error!("Failed to process {}: {:#}", vf, e);
                            stats
                                .fingerprint_failed
                                .record(format!("{}: {:#}", vf, e));
                        }
                        pb.inc(1); // Keep the bar consistent with work attempted
                        return None;
                    }
                };

                // Committed the moment it exists, over whatever this path held
                // before. See cache_store: this is one transaction per video on
                // purpose, so an interrupt (or a kill, or a crash) can only ever
                // cost the decode that was still in flight.
                match cache_store(db, vf, *stamp, &fp) {
                    Ok(()) => {
                        newly_cached.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        log::error!("Failed to cache fingerprint for {}: {:#}", vf, e);
                        stats.cache_write_failed.record(format!("{}: {:#}", vf, e));
                    }
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
        // Print time elapsed anyway in same format.
        let total_hours = start_time.elapsed().as_secs() / 3600;
        let total_minutes = (start_time.elapsed().as_secs() % 3600) / 60;
        let total_seconds = start_time.elapsed().as_secs() % 60;
        info!("Total time elapsed: {:02}:{:02}:{:02}", total_hours, total_minutes, total_seconds);
        return Ok(Outcome::Completed);
    }

    info!("\nFingerprinting complete. Cross-analyzing {} videos...", n);

    let matches = find_all_matches(&fingerprints, max_hamming, min_match_pct, min_duration);

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    info!("Grouping duplicate clusters...");

    // Clustering only needs to know which pairs matched -- it decides membership
    // and nothing else. The coverage figures travel separately to the report,
    // where they are the only thing that tells a genuine re-encode apart from a
    // clip that happened to clear --match-percent.
    let edges: Vec<(usize, usize)> = matches.iter().map(|m| (m.a, m.b)).collect();
    let final_groups = clustering::find_duplicate_groups(n, edges, &fingerprints);

    // Consumes the Vec, so the pair list is not kept alive alongside the index.
    let matches = MatchIndex::new(matches);

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

    let deleted_paths = export::output_results(
        &final_groups,
        &fingerprints,
        &matches,
        args.output.as_ref(),
        start_time.elapsed().as_secs(),
        args.priority,
        args.delete,
        args.permanent,
        stats,
    )?;

    // Those files are gone, so the fingerprints filed under their paths describe
    // nothing at all. Deliberately BEFORE the interrupt check below: the
    // deletions have already happened, and stopping is no reason to leave the
    // cache claiming otherwise. It is one small transaction either way.
    if !deleted_paths.is_empty() {
        match cache_forget(db, &deleted_paths) {
            Ok(forgotten) => {
                log::debug!("Dropped {} cache entry(ies) for deleted file(s).", forgotten)
            }
            Err(e) => {
                log::error!("Failed to drop cache entries for deleted files: {:#}", e);
                stats.cache_purge_failed.record(format!("{:#}", e));
            }
        }
    }

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    Ok(Outcome::Completed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use redb::ReadableTableMetadata;

    #[test]
    fn test_budget_is_split_evenly_when_videos_are_scarce() {
        // The cases from the spec: 8 threads, N videos.
        assert_eq!(decode_threads_for(8, 4), 2);
        assert_eq!(decode_threads_for(8, 2), 4);
        assert_eq!(decode_threads_for(8, 1), 8);
    }

    #[test]
    fn test_one_thread_each_once_videos_outnumber_threads() {
        // Rayon runs at most `budget` decodes at once, so one thread each IS the
        // whole budget here. Dividing further would leave cores idle.
        assert_eq!(decode_threads_for(8, 8), 1);
        assert_eq!(decode_threads_for(8, 9), 1);
        assert_eq!(decode_threads_for(8, 5_000), 1);
    }

    #[test]
    fn test_never_exceeds_the_budget() {
        // The invariant that matters for -t: threads-per-video times videos
        // running concurrently must never exceed what the user allowed.
        for budget in 1..=64usize {
            for jobs in 1..=200usize {
                let per_video = decode_threads_for(budget, jobs);
                let concurrent = jobs.min(budget);
                assert!(per_video >= 1, "a decode must never get zero threads");
                assert!(
                    per_video * concurrent <= budget.max(1),
                    "budget {} with {} jobs handed out {} x {}",
                    budget, jobs, per_video, concurrent
                );
            }
        }
    }

    #[test]
    fn test_uneven_split_rounds_down_rather_than_overcommitting() {
        // 8 threads over 3 videos is 2 each and two threads left on the table.
        // Rounding up would put 9 decoder threads on 8 cores.
        assert_eq!(decode_threads_for(8, 3), 2);
        assert_eq!(decode_threads_for(4, 3), 1);
    }

    #[test]
    fn test_degenerate_budgets() {
        assert_eq!(decode_threads_for(0, 4), 1, "a zero budget still runs, single-threaded");
        assert_eq!(decode_threads_for(4, 0), 4, "no jobs left is not a division by zero");
    }

    #[test]
    fn test_huge_budget_is_capped_at_what_a_decoder_can_use() {
        assert_eq!(decode_threads_for(256, 1), MAX_DECODE_THREADS);
    }

    #[test]
    fn test_slot_widens_as_the_queue_drains() {
        let queued = AtomicUsize::new(4);
        let running = AtomicUsize::new(0);

        // Four outstanding on a budget of 8: two threads each.
        let a = JobSlot::claim(8, &running, &queued);
        assert_eq!(a.threads, 2);
        let b = JobSlot::claim(8, &running, &queued);
        assert_eq!(b.threads, 2);

        // Two finish. Only two jobs remain outstanding, so the next to start
        // claims half the budget instead of a quarter.
        drop(a);
        drop(b);
        let c = JobSlot::claim(8, &running, &queued);
        assert_eq!(c.threads, 4, "the tail of a scan must widen out");

        // And the last one on its own gets everything.
        drop(c);
        let d = JobSlot::claim(8, &running, &queued);
        assert_eq!(d.threads, 8);
    }

    #[test]
    fn test_slot_releases_its_claim_on_drop() {
        let queued = AtomicUsize::new(1);
        let running = AtomicUsize::new(0);

        {
            let _slot = JobSlot::claim(8, &running, &queued);
            assert_eq!(running.load(Ordering::SeqCst), 1);
        }

        assert_eq!(running.load(Ordering::SeqCst), 0, "a finished job must free its threads");
    }

    fn stamp(mtime: u64, size: u64) -> Stamp {
        Stamp {
            mtime,
            size,
            kf_interval: 0.0,
            min_kf_samples: 4.0,
        }
    }

    #[test]
    fn test_an_edited_file_invalidates_the_entry_it_is_about_to_overwrite() {
        // The bug this whole change exists to fix, stated as a property: a
        // re-encode does not produce a SECOND valid entry, it produces a
        // mismatch against the only entry this path has -- which is then
        // replaced. Nothing accumulates, so nothing needs pruning.
        let before = stamp(1_700_000_000, 12_345);
        let after = stamp(1_700_009_999, 999);

        assert!(before.matches(&before), "an untouched file must still hit");
        assert!(!before.matches(&after), "a rewritten file must miss");
        assert!(
            !stamp(1_700_000_000, 999).matches(&before),
            "a same-second edit is still caught by the size"
        );
    }

    #[test]
    fn test_the_sampling_knobs_are_part_of_what_makes_an_entry_valid() {
        // These change which keyframes are decoded, so a fingerprint made under
        // one setting is not an answer for a run using another.
        let base = stamp(1_700_000_000, 12_345);

        let coarser = Stamp { kf_interval: 5.0, ..base };
        let fewer = Stamp { min_kf_samples: 2.0, ..base };

        assert!(!base.matches(&coarser));
        assert!(!base.matches(&fewer));
        assert!(coarser.matches(&coarser));
    }

    #[test]
    fn test_an_unparseable_interval_does_not_miss_forever() {
        // clap accepts `--keyframe-interval nan`, and NaN != NaN. Derived
        // equality would re-decode the entire library on every single run.
        let nonsense = Stamp {
            kf_interval: f64::NAN,
            ..stamp(1_700_000_000, 12_345)
        };

        assert!(nonsense.matches(&nonsense), "an entry must be able to match itself");
    }

    fn mock_fp(path: &str) -> VideoFingerprint {
        VideoFingerprint {
            path: path.to_string(),
            valid_hashes: vec![1, 2, 3],
            valid_t_start: vec![0, 1, 2],
            valid_t_end: vec![1, 2, 3],
            total_frames: 3,
            width: 1920,
            height: 1080,
            duration: 60.0,
            file_size: 12_345,
            codec: "h264".to_string(),
            frame_rate: 30.0,
        }
    }

    /// A cache of its own, inside a directory that cleans itself up.
    fn temp_db(dir: &tempfile::TempDir) -> Database {
        let db = Database::create(dir.path().join("fingerprints.redb")).unwrap();
        ensure_cache_table(&db).unwrap();
        db
    }

    #[test]
    fn test_a_fingerprint_round_trips_through_the_cache() {
        // The stamp is written ahead of the fingerprint, so this also pins the
        // offset everything after it is read from: the borrowed write path and
        // the owned read path have to agree byte for byte.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let written = stamp(1_700_000_000, 12_345);
        let fp = mock_fp("/videos/some_show_s01e01.mkv");

        cache_store(&db, &fp.path, written, &fp).unwrap();

        let hit = cache_lookup(&db, &fp.path, &written).expect("an untouched file must hit");
        assert_eq!(hit.valid_hashes, fp.valid_hashes);
        assert_eq!(hit.codec, fp.codec);
    }

    #[test]
    fn test_a_rewritten_file_replaces_its_entry_rather_than_adding_one() {
        // What the key change actually buys, at the level the bug lived at: a
        // path holds ONE entry no matter how many times its file is rewritten.
        // The superseded fingerprint is not merely unreachable, it is gone.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let path = "/videos/some_show_s01e01.mkv";

        let old = stamp(1_700_000_000, 12_345);
        let new = stamp(1_700_009_999, 6_000);

        cache_store(&db, path, old, &mock_fp(path)).unwrap();
        cache_store(&db, path, new, &mock_fp(path)).unwrap();

        assert!(cache_lookup(&db, path, &new).is_some(), "the file as it stands now hits");
        assert!(cache_lookup(&db, path, &old).is_none(), "the superseded stamp cannot");

        let read = db.begin_read().unwrap();
        let table = read.open_table(CACHE_TABLE).unwrap();
        assert_eq!(table.len().unwrap(), 1, "one file, one entry, however often it changes");
    }

    #[test]
    fn test_a_deleted_file_is_forgotten() {
        // The other half of keeping the cache bounded. A cleanup run takes the
        // duplicate off disk and nothing will ever ask about that path again,
        // so leaving the fingerprint would make every --delete run grow the
        // cache by exactly what it just cleaned up.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let kept = "/videos/keep.mkv";
        let gone = "/videos/duplicate.mkv";
        let s = stamp(1_700_000_000, 12_345);

        cache_store(&db, kept, s, &mock_fp(kept)).unwrap();
        cache_store(&db, gone, s, &mock_fp(gone)).unwrap();

        let forgotten = cache_forget(&db, &[gone.to_string()]).unwrap();

        assert_eq!(forgotten, 1);
        assert!(cache_lookup(&db, gone, &s).is_none(), "a deleted file keeps no fingerprint");
        assert!(cache_lookup(&db, kept, &s).is_some(), "and its neighbours are untouched");
    }

    #[test]
    fn test_forgetting_something_uncached_is_not_an_error() {
        // A file whose cache write failed earlier in the run is still a
        // perfectly good deletion target, and its absence here means nothing.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        assert_eq!(cache_forget(&db, &[]).unwrap(), 0);
        assert_eq!(cache_forget(&db, &["/videos/never-cached.mkv".to_string()]).unwrap(), 0);
    }
}