mod clustering;
mod compare;
mod confirm;
mod export;
mod fingerprint;
mod report;
mod stats;
mod utils;
mod sources;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use clap_complete::Shell;
use compare::{find_all_matches, max_hamming_distance, MatchIndex, HASH_BITS};
use export::Disposal;
use fingerprint::{fingerprint_video, VideoFingerprint, MAX_DECODE_THREADS};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use log::info;
use rayon::prelude::*;
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use sources::ScannedFile;
use stats::RunStats;
use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};
use utils::{shutdown_requested, Priority};

/// The one table in the cache: absolute file path -> bincode'd `CacheEntry`.
///
/// One entry per path, overwritten in place. That is the entire invalidation
/// story, and it is why the cache is bounded by the number of files ever
/// scanned rather than by the number of times they have changed: a file that is
/// re-encoded, re-muxed, or scanned with different sampling settings replaces
/// its own entry instead of growing a second one beside it.
const CACHE_TABLE: TableDefinition<&str, &[u8]> =
    TableDefinition::new("fingerprints_dct_ct_ff8_tail");

/// Tables earlier builds wrote, all dead. They are dropped whole on the first
/// run of this one.
///
/// Renaming the table is how a fingerprint that no longer MEANS the same thing
/// is retired, and it is the only mechanism that works here. `VideoFingerprint`
/// relies on a changed layout failing to deserialize, which catches a field
/// being added but not a field changing units underneath the same `u32` -- an
/// entry written when those numbers counted keyframes decodes perfectly as one
/// that counts milliseconds, and quietly reports nonsense. A name no lookup
/// will ever ask for cannot be misread.
///
/// - `fingerprints` was keyed by path AND mtime AND size AND the sampling
///   knobs. That key was a leak: touching a file wrote a NEW key and left the
///   old one behind for the life of the cache, and `--prune-cache` could not
///   tell the two apart because it only compared the path portion.
/// - `fingerprints_by_path` fixed the key, but holds difference hashes over an
///   8x9 thumbnail and sample times counted in keyframes. This build hashes
///   DCT coefficients and counts milliseconds; the two are not comparable.
/// - `fingerprints_dct` holds the right hashes against the wrong clock. Its
///   sample times came from the presentation timestamps libavformat reported,
///   and on MP4 those are skewed by up to a reorder delay because discarding
///   the non-key samples desynchronises the demuxer's `ctts` cursor. This build
///   measures decode time from the first keyframe instead. Same field, same
///   units, different milliseconds -- exactly the case a layout check cannot
///   catch.
/// - `fingerprints_dct_ct` holds the right hashes off a different decoder. The
///   released binary now links a vendored FFmpeg 8 (see
///   scripts/build-ffmpeg-static.sh) where it used to link whatever the host
///   shipped, in practice FFmpeg 6. Decoder output is not bit-identical across
///   major versions and measurably is not here: on the `-d 4 -p 20` accuracy
///   profile one pair's coverage falls by a single 0.5s sample and drops under
///   the gate. That is a smaller change than the ones above, which is exactly
///   why it needed the rename -- the `Stamp` records mtime, size and the
///   sampling knobs but deliberately not an FFmpeg version, so entries written
///   either side of the switch are indistinguishable to a lookup and would have
///   mixed silently, forever.
/// - `fingerprints_dct_ct_ff8` holds the right hashes against a `total_ms` that
///   is a few tenths of a percent short. `sample_times` extends the last sample
///   by one average gap when the samples outrun the runtime the container
///   reported, and it divided the span by the number of SAMPLES rather than by
///   the number of gaps between them. The field keeps its name and its units and
///   only its value moves, which is precisely the case a layout check cannot
///   catch -- and it moves for real files, three of the 727 in the local corpus.
const SUPERSEDED_TABLES: [TableDefinition<&str, &[u8]>; 5] = [
    TableDefinition::new("fingerprints"),
    TableDefinition::new("fingerprints_by_path"),
    TableDefinition::new("fingerprints_dct"),
    TableDefinition::new("fingerprints_dct_ct"),
    TableDefinition::new("fingerprints_dct_ct_ff8"),
];

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

/// How far the demuxer has to travel through a file before the progress bar is
/// told about it.
///
/// The hook behind this fires once per demuxed packet, which on a container we
/// scan linearly is thousands of times a second per decode, from every worker at
/// once -- and each report that gets through takes indicatif's state lock. The
/// bar redraws at most 20 times a second no matter how often it is nudged, so
/// anything finer than this is contention bought for pixels that are never
/// drawn.
///
/// 4 MiB is the coarsest step that still looks continuous on the file sizes that
/// need it: it puts ~2000 updates across an 8 GB decode (several a second over
/// minutes of work) while a file small enough to yield fewer than a handful is,
/// by definition, one that finishes before anybody looks at the bar.
const PROGRESS_STEP_BYTES: u64 = 4 * 1024 * 1024;

/// Everything other than the file's contents that decides whether a cached
/// fingerprint is still the right answer for the file at a given path.
///
/// This used to be spelled into the cache key. Moving it into the value is what
/// makes a path map to exactly one entry: on a mismatch the file is
/// re-fingerprinted and the insert OVERWRITES, rather than filing the new
/// fingerprint under a new key and leaving the old one to accumulate.
///
/// `mtime`, `mtime_nsec` and `size` are whatever the scan's single stat()
/// reported -- see `sources::ScannedFile`. Nothing here re-derives them, so the
/// figure the stamp is written with is the same figure the file was selected
/// and sorted on, and a file changing mid-run cannot make those three disagree.
///
/// Fixed-size fields only, and serialized BEFORE the fingerprint, so the
/// fingerprint always begins at the same byte offset. That is what keeps the
/// discipline `VideoFingerprint` already documents working: a field appended to
/// the end of it makes older payloads run out of bytes, bincode fails, and the
/// file is fingerprinted again. Changing THIS struct moves the offset and has
/// no such guarantee, so it needs the same treatment `SUPERSEDED_TABLES` got.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
struct Stamp {
    mtime: i64,
    mtime_nsec: i64,
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
            && self.mtime_nsec == other.mtime_nsec
            && self.size == other.size
            && same_setting(self.kf_interval, other.kf_interval)
            // `--min-keyframes` is a FLOOR on the sampling interval, and with no
            // interval in force there is nothing for it to floor -- see
            // `effective_interval`, which requires a real interval before it
            // reads this at all. Comparing it regardless threw away a whole
            // library's fingerprints for a flag that provably changed not one
            // frame. The two intervals are already known equal here, so testing
            // either side's is the same test.
            && (!sampling_is_on(other.kf_interval)
                || same_setting(self.min_kf_samples, other.min_kf_samples))
    }
}

fn same_setting(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// Whether keyframe subsampling actually happens at this interval.
///
/// Mirrors `fingerprint::fingerprint_video`'s own `effective_interval` guard
/// exactly, NaN and all: a nonsense interval subsamples nothing, so nothing is
/// floored by `--min-keyframes` either. Spelled as a named predicate rather
/// than inlined so the negation in `Stamp::matches` reads as "sampling is off"
/// instead of as a comparison against a float that might not be comparable.
fn sampling_is_on(kf_interval: f64) -> bool {
    kf_interval > 0.0
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
                  Report-only by default: it tells you what is redundant and what that \
                  is costing you. Files are touched only when --delete or --move-to is \
                  given."
)]
struct Args {
    /// Folders and video files to scan (one or more). A folder is searched for
    /// videos; a file you name is scanned whatever its extension, since
    /// --extensions is only a guess about what a FOLDER contains. Use `-` to
    /// read a list of paths from stdin, e.g. `fd -e mkv | vid-fp -`.
    #[arg(
        required_unless_present_any = ["completions", "man", "from_file", "from_report"],
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

    /// Maximum Hamming distance between two frame hashes, out of 64 bits.
    /// Higher = looser matching, lower = stricter matching. Default is 4.
    /// Two unrelated frames sit about 32 bits apart, so the useful range is
    /// roughly 2 (only near-identical frames) to 12 (visibly the same shot);
    /// past that unrelated footage starts linking whole groups together.
    /// A frame match standing on its own must be within this; one that another
    /// frame match agrees with about the time offset between the two videos may
    /// reach 6 bits further. Past 12 bits one agreeing match stops being enough
    /// and the number required grows with the distance -- two at 14, three at
    /// 16, four at 20 -- so raising this trades precision for recall smoothly
    /// rather than falling off a cliff.
    /// Values above 32 are rejected before the scan starts: that is how far
    /// apart two unrelated frames sit on average, so a higher tolerance matches
    /// everything against everything and has nothing left to control.
    #[arg(short = 'd', long = "hamming-distance", default_value_t = 4)]
    hamming_distance: u32,

    /// Minimum match percentage required to be considered a duplicate, from 0
    /// to 100. Default is 20.0 (20%). Values outside that range are rejected
    /// before the scan starts: coverage is capped at 100%, so a higher floor is
    /// one no pair could ever clear.
    #[arg(short = 'p', long = "match-percent", default_value_t = 20.0)]
    match_percent: f32,

    /// Minimum shared clip length, in seconds, for two videos to count as a
    /// match. Also skips fingerprinting any video shorter than this. 0 = off,
    /// and negative values are rejected before the scan starts. Independent of
    /// --match-percent; both must be satisfied.
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
    /// trash (recoverable); add --permanent to remove them for good. Files
    /// marked KEEP or REVIEW are never touched.
    #[arg(long = "delete")]
    delete: bool,

    /// With --delete, remove files permanently instead of moving them to the
    /// trash. Irreversible — use with care. Has no effect on its own.
    #[arg(long = "permanent")]
    permanent: bool,

    /// Move the files marked DELETE under this folder, recreating each file's
    /// absolute path inside it (/mnt/media/ep.mkv -> DIR/mnt/media/ep.mkv).
    /// Nothing is overwritten and nothing is renamed, so the whole run can be
    /// undone with a single copy back from DIR. Use this wherever the system
    /// trash isn't available — external disks, NFS mounts, headless servers.
    /// Acts on its own: it is not a deletion, so it needs no --delete, and it
    /// supersedes --delete and --permanent if they are given.
    #[arg(long = "move-to", value_name = "DIR",
          value_hint = clap::ValueHint::DirPath)]
    move_to: Option<String>,

    /// Act on a CSV report from an earlier run instead of scanning anything.
    /// Every row whose action column reads DELETE is disposed of, and nothing
    /// else is touched — so editing that column is how you act on the rows the
    /// tool would not decide for you (REVIEW), or spare one it would. Nothing
    /// is re-fingerprinted and no groups are recomputed: the report is the
    /// decision. Each file is still re-checked against the size the report
    /// recorded and left alone if it changed since. Requires --delete or
    /// --move-to, and cannot be combined with scanning or matching options.
    #[arg(
        long = "from-report",
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath,
        conflicts_with_all = [
            "include", "exclude", "from_file", "null", "recursive", "follow_symlinks",
            "extensions", "hamming_distance", "match_percent", "min_duration",
            "kf_interval", "min_kf_samples", "priority", "output",
            "prune_cache", "threads",
        ]
    )]
    from_report: Option<String>,

    /// Answer yes to the confirmation shown before any file is touched. That
    /// confirmation only appears on an interactive terminal -- a run whose
    /// input or output is piped or redirected is never prompted and never
    /// blocks -- so this flag is for saying so out loud, and for the
    /// interactive runs that would rather not be asked.
    #[arg(short = 'y', long = "yes")]
    yes: bool,

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
#[derive(Debug)]
enum Outcome {
    Completed,
    Interrupted,
}

/// What, if anything, the run is armed to do with the files marked DELETE.
///
/// `--move-to` outranks both of the others, and outranks them SILENTLY as far
/// as the file's fate is concerned (the caller prints a note): a move is the
/// most conservative outcome of the three by every measure -- the file still
/// exists, at a path the user named, on a filesystem the user chose -- and
/// there is no reading of `--move-to --permanent` where doing the permanent
/// thing is what was meant. Precedence therefore runs toward the recoverable
/// option, which is the direction a mistake in this function should fail.
fn disposal_for(move_to: Option<PathBuf>, delete: bool, permanent: bool) -> Option<Disposal> {
    match (move_to, delete, permanent) {
        (Some(dir), _, _) => Some(Disposal::MoveTo(dir)),
        (None, true, true) => Some(Disposal::Permanent),
        (None, true, false) => Some(Disposal::Trash),
        // --permanent alone is not an arming flag and never has been. The whole
        // point of report-only-by-default is that exactly one of two named
        // flags can take a file off its path.
        (None, false, _) => None,
    }
}

/// Turn `--move-to DIR` into the absolute folder the files will land under.
///
/// Resolved BEFORE any work on purpose: a destination that cannot be created is
/// a typo, and finding a typo out after fingerprinting a library is finding it
/// out several hours too late. Canonical, so the containment check and the
/// mirrored landing paths are both working from the same absolute form the
/// scanner produces.
fn resolve_move_to(dir: Option<&String>) -> Result<Option<PathBuf>> {
    let Some(dir) = dir else { return Ok(None) };

    std::fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create the --move-to folder {}", dir))?;
    let resolved = std::fs::canonicalize(dir)
        .with_context(|| format!("Failed to resolve the --move-to folder {}", dir))?;
    if !resolved.is_dir() {
        anyhow::bail!("--move-to {} is not a folder.", dir);
    }
    Ok(Some(resolved))
}

/// Refuse a `--output` path that plainly cannot be written, before any work
/// starts.
///
/// The same reasoning `resolve_move_to` is built on, applied to the other
/// argument naming a path: a destination that cannot exist is a typo, and
/// finding a typo out after fingerprinting a library is finding it out several
/// hours too late. The report used to be the one path checked only at the point
/// of writing it, which is the last statement of the run -- so `-o
/// /nonexistant/report.csv` did the entire scan and then threw the results
/// away, and with `--delete` armed it did so *after* the files were gone.
///
/// Deliberately does not create anything. `--move-to` names a folder the user
/// wants files put in, so creating it is the request; `-o` names a file, and
/// conjuring parent directories for it is not. Nor does it touch the target
/// itself -- an existing report stays intact until the new one replaces it, and
/// an interrupted run leaves no empty file behind.
///
/// What it cannot see is permissions, which still surface at the end. That is
/// now the only late failure, and it is the rare one: a mistyped path is a
/// typo, an unwritable directory you own is a decision.
fn check_output_path(output: Option<&String>) -> Result<()> {
    let Some(out) = output else { return Ok(()) };
    let path = Path::new(out);

    if path.is_dir() {
        anyhow::bail!("--output {} is a folder, not a file to write.", out);
    }

    let parent = match path.parent() {
        // A bare "report.csv" has an empty parent, which is the working
        // directory rather than nowhere.
        Some(p) if p.as_os_str().is_empty() => Path::new("."),
        Some(p) => p,
        None => Path::new("/"),
    };

    if !parent.is_dir() {
        anyhow::bail!(
            "--output {} cannot be written: {} is not an existing folder.",
            out,
            parent.display()
        );
    }

    Ok(())
}

/// Say up front what is going to happen to the files marked DELETE, so it is
/// visible above the per-file log rather than only in the summary underneath it.
fn announce(disposal: Option<&Disposal>) {
    match disposal {
        Some(Disposal::MoveTo(dir)) => info!(
            "\n--move-to enabled: files marked DELETE will be moved under {}.",
            dir.display()
        ),
        Some(Disposal::Permanent) => {
            info!("\n--permanent enabled: files marked DELETE will be removed permanently.")
        }
        Some(Disposal::Trash) => {
            info!("\n--delete enabled: files marked DELETE will be moved to the trash.")
        }
        None => {}
    }
}

/// `--from-report`: dispose of what a previous run's CSV marks DELETE, and
/// nothing else.
///
/// A separate entry point rather than a branch threaded through `run`, because
/// it shares almost nothing with it. There is no scan, no cache pass, no
/// fingerprinting, no comparison and no clustering -- the report already holds
/// every figure this needs, and the whole cost of the mode is reading it. What
/// it does share is the part that matters: the same confirmation prompt, the
/// same per-file staleness check, the same disposal code, and the same cache
/// bookkeeping afterwards.
///
/// Refusing to run without an arming flag is the one piece of validation clap
/// cannot express. It is not merely useless without one -- a report-only
/// `--from-report` would read a file, decide nothing, and print nothing that
/// the report does not already say.
fn run_from_report(
    args: &Args,
    db: &Database,
    report_path: &str,
    stats: &RunStats,
) -> Result<Outcome> {
    let move_to = resolve_move_to(args.move_to.as_ref())?;

    let Some(disposal) = disposal_for(move_to, args.delete, args.permanent) else {
        anyhow::bail!(
            "--from-report needs to be told what to do with the rows marked DELETE. Add --delete \
             (to the trash), --delete --permanent, or --move-to <DIR>."
        );
    };

    announce(Some(&disposal));

    // No `scan_encloses` counterpart here: that check protects the NEXT run from
    // re-scanning what this one moved, and it needs a scan to compare against.
    // This mode has no scan roots, so there is nothing it could be checked
    // against and nothing it could conclude.
    let deleted_paths = report::apply(report_path, &disposal, args.yes, stats)?;

    if !deleted_paths.is_empty() {
        match cache_forget(db, &deleted_paths) {
            Ok(forgotten) => {
                log::debug!("Dropped {} cache entry(ies) for removed file(s).", forgotten)
            }
            Err(e) => {
                log::error!("Failed to drop cache entries for removed files: {:#}", e);
                stats.cache_purge_failed.record(format!("{:#}", e));
            }
        }
    }

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    Ok(Outcome::Completed)
}

/// The scanned folder that contains `dest`, if any.
///
/// The question is deliberately this way round. A landing path is `dest` plus
/// the source's own absolute path, so `dest` being inside a scanned folder makes
/// EVERY landing path inside it too -- that is the loop worth refusing. The
/// reverse arrangement is not: `--move-to ~/Documents` while scanning
/// `~/Documents/AN` sends `~/Documents/AN/ep.mkv` to
/// `~/Documents/home/you/Documents/AN/ep.mkv`, which no scan of `~/Documents/AN`
/// will ever reach. Testing containment the other way round -- "is a scanned
/// file under dest" -- is what made every parent folder look like a loop.
///
/// Folders are approximated by the parents of the files that were found, which
/// is what the scan actually reached: an exclude, a non-recursive walk, or an
/// extension filter can all mean a named folder contributed nothing, and a
/// folder nothing came out of is not one a moved file can be re-ingested from.
fn scan_encloses(dest: &Path, scanned: &[ScannedFile]) -> Option<String> {
    scanned.iter().find_map(|f| {
        let parent = Path::new(&f.path).parent()?;
        dest.starts_with(parent)
            .then(|| parent.display().to_string())
    })
}

/// One video that has to be decoded, and everything needed to decide how much
/// of the machine to give it.
///
/// `weight` is what the decode is going to cost, in the keyframe-pixels
/// `fingerprint::weigh_decode` measures -- not the file's size. Both numbers are
/// here because they answer different questions and neither substitutes for the
/// other: `weight` is how long this file will take, `size` is how many bytes the
/// demuxer will walk through to do it, which is the only thing the in-flight
/// progress reports are denominated in.
struct Job {
    path: String,
    stamp: Stamp,
    weight: u64,
    size: u64,
}

/// The progress bar's speedometer: work units a second, as pixels of keyframe.
///
/// The bar's position counts the keyframe-pixels `fingerprint::weigh_decode`
/// measures, so its rate is pixels of keyframe decoded per second -- the same
/// figure the per-codec cost table is calibrated in, and one with a real scale
/// behind it: a core of this development machine does about 75 megapixels of
/// H.264 a second, and an eight-thread scan on it reads somewhere in 320-460
/// Mpx/s depending on how hard the laptop is throttling.
///
/// It is an H.264-EQUIVALENT rate, because the weight is scaled by
/// `codec_cost`. On an all-H.264 scan it is exactly the pixel throughput; on an
/// HEVC one the true pixel count is lower and the number says what that work
/// would have cost in H.264. That is the right choice for a speedometer -- it
/// holds still when the codec changes, which is the whole reason it reads more
/// steadily than the megabytes a second it replaces.
///
/// The reading carries no padding: the bar rules its fields apart with a single
/// space either side of a box-drawing bar, and a field that reserved room for its
/// widest value read as a second, wider gap in front of the number. What keeps
/// the field from swinging is the unit ladder rather than a column width -- the
/// number keeps three significant figures and the unit absorbs the magnitude, so
/// the whole range from a stalled thread to a workstation spans one character.
/// The file name to the right does shift by that character, which is the price of
/// the tighter spacing and the reason the digits are held to three.
fn work_rate(per_sec: f64) -> String {
    // The dash is "no reading", and it is also where anything unusable lands:
    // the rate is a division by an elapsed time that starts at zero, so the
    // first tick of a run can hand this an absurdity, and a bar is the wrong
    // place to find out. Nothing real reaches the top of the ladder below.
    let unmeasurable = || "-".to_string();
    if !per_sec.is_finite() || per_sec <= 0.0 {
        return unmeasurable();
    }

    let (scaled, unit) = match per_sec {
        r if r >= 1e12 => (r / 1e12, "Tpx"),
        r if r >= 1e9 => (r / 1e9, "Gpx"),
        _ => (per_sec / 1e6, "Mpx"),
    };

    // Three significant figures whatever the magnitude, which is three or four
    // characters: 374, 56.2, 2.50. Above a thousand the ladder has already
    // changed unit, except at the top of it, where there is nothing bigger to
    // change to -- so a number that still needs five columns is out of range.
    let number = if scaled >= 100.0 {
        format!("{:.0}", scaled)
    } else if scaled >= 10.0 {
        format!("{:.1}", scaled)
    } else {
        format!("{:.2}", scaled)
    };
    if number.len() > 4 {
        return unmeasurable();
    }
    format!("{} {}/s", number, unit)
}

/// How many of `free` threads a video of `weight` units of work should decode
/// with, given that `queued` units have not started yet.
///
/// This is the whole scheduling rule, and it comes out of asking what actually
/// ends the run. A video decoded on `k` of the `A` free threads finishes in
/// roughly `weight / k`; everything still queued gets the other `A - k` threads
/// and finishes in roughly `queued / (A - k)`. The run is over when the later
/// of the two is, so the best `k` is the one that makes them equal:
///
/// ```text
///     weight / k = queued / (A - k)   =>   k = A * weight / (weight + queued)
/// ```
///
/// A proportional share, in other words -- and note that it needs no special
/// case for "big" files, because the same formula produces both behaviours that
/// matter:
///
///   * Hundreds of similar videos: every share rounds to 1, which is correct.
///     Decoder threading scales sublinearly, so spreading one video per thread
///     extracts more total throughput than widening any single decode -- and
///     with that much work queued, no one file's own decode can outlast it.
///   * A handful of heavyweights, or one giant among small change: the giant's
///     share is most of the budget, taken UP FRONT. That is the point. Threads
///     cannot be added to a decode that has already started, so a file that
///     will still be running when the queue empties has to be given its width
///     at the moment it starts or never.
///
/// Rounding is to nearest rather than up: rounding up steals a thread from the
/// queued work permanently, which just moves the straggler rather than removing
/// it. A file owed 6.4 threads of 8 is better served by 6 than by 7, because the
/// 2 threads left cover the rest of the queue and the 1 thread left does not.
///
/// The single exception is the step from 2 down to 1, which is not a rounding
/// error like the others but a different kind of decision: 1 is the only width a
/// decode cannot recover from. Every share below 1 is already rounded UP to 1 --
/// a decode needs the thread it runs on -- so in a queue of many small files the
/// budget is systematically over-promised to the small change, and the whole of
/// that deficit lands on the one file too big to absorb it. Note that nearest
/// already rounds 1.5 up, so this only decides the band 1.0 < share < 1.5: a
/// file whose own decode runs up to half again as long as the whole run should.
///
/// Measured, on a 41-file 54 GB scan across 8 threads, 6 alternating cold-cache
/// runs of each rule: a 9.8 GB 2160p HEVC file was owed 1.43 threads and given
/// 1, which made it the last thing on the machine for the final minute of every
/// run. Rounding it to 2 finishes the scan in 258.4 s against 294.0 s (-12.1%,
/// sd 2.2 and 2.9, ranges nowhere near overlapping) and lifts average occupancy
/// from 6.3 of 8 cores to 7.5. The second thread is only worth 1.36x on that
/// file, so the run does about 3% more total work -- and buys back far more than
/// that from the threads the straggler used to leave idle behind it.
fn share_for(weight: u64, queued: u128, free: usize) -> usize {
    if free == 0 {
        return 0;
    }

    // Nothing is waiting behind this video, so there is nobody to spread the
    // budget for. Take everything a decoder can actually use.
    if queued == 0 {
        return free.min(MAX_DECODE_THREADS);
    }

    let mine = weight as f64;
    let rest = queued as f64;

    // `mine + rest` is strictly positive here: `rest` is, since `queued != 0`.
    let exact = free as f64 * mine / (mine + rest);
    let share = if exact > 1.0 {
        (exact.round() as usize).max(2)
    } else {
        exact.round() as usize
    };

    // Never zero (a decode always needs the thread it is running on), never
    // more than is free, and never more than a decoder can use.
    share.clamp(1, free.min(MAX_DECODE_THREADS))
}

/// The process-wide decoder thread budget.
///
/// A ledger rather than a calculation, because `share_for` needs two numbers
/// that only a shared, serialized view can supply: how many threads are
/// genuinely unclaimed right now, and how much work is still waiting behind
/// this one. Holding them under one lock is also what makes the invariant
/// enforceable rather than merely intended -- a decode RESERVES its threads, so
/// the sum handed out can never exceed `--threads`, whatever order the claims
/// arrive in.
///
/// A worker that finds the budget fully claimed blocks here instead of
/// overcommitting. That is not lost time: every thread is already promised to a
/// decode that is running, and one of them will hand its share back. It cannot
/// deadlock, because `free == 0` means somebody is holding threads, and every
/// holder is a live decode whose `Grant` releases them on the way out --
/// including when it unwinds on an error or a Ctrl-C.
struct ThreadBudget {
    state: Mutex<BudgetState>,
    released: Condvar,
}

struct BudgetState {
    /// Threads not currently promised to a running decode.
    free: usize,
    /// Total weight of the videos that have not started decoding yet. Shrinks
    /// as jobs are claimed, never as they finish -- a video that is already
    /// running is accounted for by the threads it holds, not by its weight.
    queued: u128,
}

/// A reservation against the budget, returned to it on drop.
struct Grant<'a> {
    budget: &'a ThreadBudget,
    threads: usize,
}

impl Drop for Grant<'_> {
    fn drop(&mut self) {
        {
            let mut state = self.budget.state.lock().unwrap_or_else(|e| e.into_inner());
            state.free += self.threads;
        }
        // notify_all rather than notify_one: several threads may come back at
        // once, and a single waiter waking to find one thread free when four
        // were released is how a wide decode gets starved by a narrow one.
        self.budget.released.notify_all();
    }
}

impl ThreadBudget {
    fn new(total: usize, queued: u128) -> Self {
        ThreadBudget {
            state: Mutex::new(BudgetState {
                free: total.max(1),
                queued,
            }),
            released: Condvar::new(),
        }
    }

    /// Reserve this video's share of the budget, waiting if nothing is free.
    ///
    /// `None` means the run is shutting down and the caller should stop rather
    /// than wait for threads it is not going to use.
    fn claim(&self, weight: u64) -> Option<Grant<'_>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        while state.free == 0 {
            if shutdown_requested() {
                return None;
            }
            // A timeout rather than a bare wait purely so the shutdown check
            // above is reached even if the notification is missed.
            let (next, _) = self
                .released
                .wait_timeout(state, Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner());
            state = next;
        }

        // No longer queued: from here on this video is represented by the
        // threads it is about to take, not by its weight. Doing this BEFORE
        // computing the share is what keeps `queued` meaning "the work I am
        // competing with" rather than "the work including me".
        state.queued = state.queued.saturating_sub(weight as u128);

        let threads = share_for(weight, state.queued, state.free);
        state.free -= threads;

        Some(Grant {
            budget: self,
            threads,
        })
    }
}

#[cfg(test)]
impl ThreadBudget {
    fn free(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).free
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

/// Empty the cache, and hand the space back to the filesystem while doing so is
/// still free.
///
/// Emptying a table frees its pages inside the file but does not shrink the
/// file, so the space only comes back via a compaction -- and a compaction is a
/// copy of every live page into a fresh, dense file. Right here there are no
/// live pages, so it costs microseconds and returns the whole of whatever the
/// cache had grown to.
///
/// This used to run at the END of the run instead, which is the one moment it
/// is worth nothing: by then the clear has been followed by a full re-scan, so
/// every page is live, the file is already dense (the inserts reused the very
/// pages the clear freed), and the compaction rewrites the entire library's
/// fingerprints to reclaim nothing.
fn clear_cache(db: &mut Database) -> Result<()> {
    let txn = db.begin_write().context("Failed to start a cache transaction")?;
    txn.delete_table(CACHE_TABLE).context("Failed to clear cache database")?;
    txn.commit().context("Failed to commit the cache clear")?;

    // Dropping the table dropped its definition too; every later lookup needs
    // something to open.
    ensure_cache_table(db).context("Failed to recreate the cache table")?;

    // Not fatal: a cache that could not be shrunk is still a correct, empty
    // cache, and the run that follows works either way.
    if let Err(e) = db.compact() {
        log::error!("Could not compact the fingerprint cache: {}", e);
    }

    Ok(())
}

/// Drop every table this build's cache replaced, if any are still there.
///
/// Returns whether there were any, so the caller can decide about reclaiming
/// the space: deleting a table frees its pages inside the file but does not
/// shrink the file, and only a compaction hands that back to the filesystem.
fn retire_superseded_tables(db: &Database) -> Result<bool> {
    let txn = db.begin_write().context("Failed to start a cache transaction")?;
    let mut existed = false;
    for table in SUPERSEDED_TABLES {
        existed |= txn
            .delete_table(table)
            .context("Failed to remove a superseded cache table")?;
    }
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
/// here keeps the whole thing usable from a worker without threading a borrow
/// through the closure.
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
/// only ever holding a b-tree insert -- and the caller hands its decoder
/// threads back BEFORE calling this, so nothing waits on an fsync while
/// holding a share of the budget.
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
/// Called with what the run actually acted on -- trashed, unlinked, or moved
/// under `--move-to`. Those entries describe bytes that are somewhere else or
/// gone entirely, so nothing will ever match them at that path again: keeping
/// them means every cleanup run grows the cache a little and it only shrinks if
/// the user happens to remember `--prune-cache`. Acting on duplicates is the
/// one thing this tool does that makes an entry obsolete WITHOUT writing a
/// replacement over it, so it is the one place that has to say so out loud.
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

    // Once, on the first run of this build. Nothing will ever look in those
    // tables again, so every byte of them is dead. Compacting right afterwards
    // is the only thing that returns those pages to the filesystem.
    match retire_superseded_tables(&db) {
        Ok(true) => {
            info!("Removed the superseded fingerprint cache; fingerprints will be rebuilt once.");
            if let Err(e) = db.compact() {
                log::error!("Could not compact the fingerprint cache: {}", e);
            }
        }
        Ok(false) => {}
        Err(e) => log::error!("{:#}", e),
    }

    // Handled here rather than inside `run` because compaction needs an
    // exclusive handle on the database, and because this is the only point in
    // the run where compacting is free. See `clear_cache`.
    if args.clear_cache {
        info!("Clearing all cache...");
        clear_cache(&mut db)?;
    }

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
    //
    // Only pruning can leave dead space behind at this point: it removes
    // entries the run will not refill, by design. A cleared cache was already
    // compacted while it was empty, and everything in the file now is live.
    //
    // Hence `!args.clear_cache`, which this condition used to be missing even
    // though the sentence above already stated the rule. `--prune-cache` and
    // `--clear-cache` together skip the prune itself (see `run`: after a clear
    // there is nothing stale, because there is nothing at all), so the run
    // removed no entries -- and then compacted anyway, copying every live page
    // of a cache the scan had just rewritten from scratch to reclaim exactly
    // nothing. That is the same waste compaction was moved out of the tail of
    // the run to avoid; it is invisible on a small cache and is a full rewrite
    // plus an fsync on a large one.
    if !shutdown_requested() && args.prune_cache && !args.clear_cache {
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
    // Before anything that reads the filesystem: this mode's whole premise is
    // that the decisions were made by an earlier run and are not being retaken.
    if let Some(report_path) = &args.from_report {
        return run_from_report(args, db, report_path, stats);
    }

    let max_hamming = args.hamming_distance;
    let min_match_pct = args.match_percent / 100.0;
    let min_duration = args.min_duration;

    // Both of these are range tests rather than the `< 0.0` this used to be,
    // and the reason is NaN: every comparison against it is false whichever way
    // it is written, and clap parses `--min-duration=nan` without complaint. A
    // NaN floor would disable the gate silently; a NaN percentage is worse,
    // because `pct_a.max(pct_b) < min_match_percent` is then false for every
    // pair and the run accepts everything the index proposed -- which with
    // --delete armed can only ever ADD files to the DELETE set.
    // `RangeInclusive::contains` is false for NaN, so one form catches the lot.
    if !(0.0..).contains(&min_duration) {
        anyhow::bail!("--min-duration must be zero or more seconds (0 turns it off).");
    }
    // The upper bound is not a matter of taste. `match_overlap` clamps every
    // coverage figure to 1.0, so a floor above 100% is one no pair can ever
    // clear: the run would fingerprint the whole library and be structurally
    // incapable of reporting a single group. 0 stays legal and means "report
    // every pair the index proposes".
    if !(0.0..=100.0).contains(&args.match_percent) {
        anyhow::bail!(
            "--match-percent must be between 0 and 100 (coverage is capped at 100%, so a \
             higher floor could never be met)."
        );
    }
    // The same reasoning as --match-percent, arriving from the other end, and it
    // stops at CHANCE rather than at the width of the hash. Two unrelated frames
    // sit 32 bits apart on average, so a tolerance of 32 already accepts half of
    // all unrelated frame pairs and one above it accepts most of them: the
    // library stops being a library and becomes one enormous group, which with
    // --delete armed can only ever ADD files to the DELETE set. Left unchecked,
    // `-d 100` collapsed a 727-file library into one group with 322 files marked
    // DELETE, and said nothing.
    //
    // The ceiling used to be 64 -- the widest two hashes can possibly be apart --
    // which refused only the arithmetic that could not mean anything and let
    // through a whole range that could not mean anything either. Past chance the
    // graph also stops being sparse enough to enumerate: `-d 36` on the local
    // corpus is refused by the clustering ceiling in 9 seconds, but `-d 40` is a
    // nearly COMPLETE graph, which has few enough groups to slip under that
    // ceiling while costing minutes of quadratic pivot work to prove it. Chance
    // is the honest edge of the range, and it is comfortably past the edge of the
    // useful one (`--help` says 2 to 14).
    //
    // MAX_HAMMING_DISTANCE itself stays legal, the mirror of `-p 0`: it is a
    // sensitivity control, and refusing its last rung would be arbitrary where
    // refusing everything past chance is not.
    let widest_useful = max_hamming_distance();
    if max_hamming > widest_useful {
        anyhow::bail!(
            "--hamming-distance must be between 0 and {} (a frame hash is {} bits, and two \
             unrelated frames already differ in about {} of them -- a higher tolerance \
             matches everything against everything).",
            widest_useful,
            HASH_BITS,
            widest_useful
        );
    }

    let move_to = resolve_move_to(args.move_to.as_ref())?;
    check_output_path(args.output.as_ref())?;

    // Say out loud which flags are not doing anything. A user who passed
    // --permanent and watched a report scroll by has to be told the files were
    // moved rather than removed -- silence there is how someone comes away
    // believing an irreversible run happened when it did not, and goes looking
    // for the files in the wrong place.
    if move_to.is_some() {
        if args.delete {
            info!(
                "Note: --move-to supersedes --delete; the files marked DELETE will be moved, \
                 not trashed."
            );
        }
        if args.permanent {
            info!(
                "Note: --permanent has no effect alongside --move-to; the files will be moved, \
                 not removed."
            );
        }
    } else if args.permanent && !args.delete {
        info!("Note: --permanent has no effect without --delete; running in report-only mode.");
    }

    // Same rule, for the flag that answers a question this run will not ask.
    if move_to.is_none() && !args.delete && args.yes {
        info!("Note: --yes has nothing to confirm without --delete or --move-to.");
    }

    info!(
        "Settings -> Max Hamming: {}, Min Match: {}%, Min Duration: {}s, Priority: {:?}, Threads: {}, Recursive: {}",
        max_hamming, args.match_percent, min_duration, args.priority, active_threads, args.recursive
    );

    // Every file here has been stat'ed exactly once, and carries the size and
    // mtime that stat returned. Nothing below asks the filesystem about these
    // files again unless it is going to decode one: the sort, the cache stamp,
    // the thread budget's weights and the prune all read from this list. On a
    // network mount that is the difference between three round trips per file
    // and one.
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

// Moving a duplicate into a folder that is itself being scanned puts the
    // file straight back into the next run's input, where it is still a
    // duplicate of the copy that was kept -- and the run after that would move
    // it again, one directory deeper each time. Nothing is lost, but the one
    // thing this mode is for (getting files OUT of the library, reversibly) is
    // quietly not happening, so it is worth stopping over rather than warning
    // about.
    if let Some(dest) = &move_to {
        if let Some(scanned) = scan_encloses(dest, &video_files) {
            anyhow::bail!(
                "The --move-to folder {} is inside {}, which this run scans, so the moved \
                 files would be picked up again next time. Exclude it with -e {}, or choose \
                 a destination outside the scanned folders.",
                dest.display(),
                scanned,
                dest.display()
            );
        }

        // The opposite arrangement, which is fine and common: the destination is
        // an ANCESTOR of the scan. Landing paths mirror the source's absolute
        // path, so they sit in a sibling subtree the current scan never reaches
        // -- but a later recursive scan of the destination itself would find
        // both copies, so it is worth a word.
        if video_files.iter().any(|f| Path::new(&f.path).starts_with(dest)) {
            info!(
                "Note: {} is above the folder(s) being scanned. The moved files land in a \
                 separate subtree under it, so this run is unaffected -- but exclude it with \
                 -e if you ever scan {} itself recursively.",
                dest.display(),
                dest.display()
            );
        }
    }

    if args.prune_cache && !args.clear_cache {
        info!("Pruning cache for files not in the current scan...");
        let valid_files: HashSet<&str> = video_files.iter().map(|f| f.path.as_str()).collect();

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

    // Largest first, so the cache pass below hands `todo` back in roughly the
    // order the decode wants it. Roughly is all this can be: the ordering the
    // schedule actually rests on is by decode COST, which nothing here has
    // measured yet -- see the weighing pass, which re-sorts on the real figure
    // once it has one. Sorting on size anyway costs nothing (the scan already
    // read it) and puts the list within a rotation of where it needs to be.
    video_files.sort_by_key(|f| std::cmp::Reverse(f.size));

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
    // A lookup is now a b-tree descent and a bincode decode and nothing else --
    // the stat it used to open with came out of the scan -- so this pass costs
    // no I/O against the library at all and finishes effectively instantly even
    // on a large one. By the time the bar appears we know exactly how much real
    // work there is, and that work is what the decoder thread budget is
    // apportioned against.
    enum Lookup {
        Hit(VideoFingerprint),
        Miss(Job),
    }

    let lookups: Vec<Lookup> = video_files
        .par_iter()
        .map(|file| {
            // Thread count is deliberately NOT part of the stamp. Threading
            // changes only how fast frames arrive, never which ones, so a
            // fingerprint made with 8 threads is byte-identical to one made with
            // 1 and stays valid across runs with different -t.
            let stamp = Stamp {
                mtime: file.mtime,
                mtime_nsec: file.mtime_nsec,
                size: file.size,
                kf_interval,
                min_kf_samples,
            };

            match cache_lookup(db, &file.path, &stamp) {
                Some(fp) => Lookup::Hit(fp),
                None => Lookup::Miss(Job {
                    path: file.path.clone(),
                    stamp,
                    // Weighed by the pass below, once it is known which files
                    // are actually going to be decoded. Until then the size is
                    // a placeholder, and the only thing that reads it before
                    // then is the `sort_by_key` that put this list in order.
                    weight: file.size,
                    size: file.size,
                }),
            }
        })
        .collect();

    // collect() preserves input order, so `todo` inherits the largest-first sort
    // and the heaviest decodes are still claimed first -- which is exactly when
    // the budget has the most to give them.
    let mut fingerprints: Vec<VideoFingerprint> = Vec::with_capacity(total_videos);
    let mut todo: Vec<Job> = Vec::new();
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
            Lookup::Miss(job) => todo.push(job),
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

    // --- Pass 1b: weigh what is left ------------------------------------------
    // Everything from here to the end of the decode is apportioned by how much
    // work each file represents, and until this pass that figure was the file's
    // SIZE -- which is a statement about its bitrate at least as much as about
    // its decode cost. The two come apart badly and in both directions. Measured
    // here, single-threaded: a 10.5 GB 2160p HEVC feature costs 14.7x a 1.8 GB
    // 1080p H.264 one while its size claims 5.8x, and two encodes of the SAME
    // two-minute clip with the SAME 61 keyframes -- one H.264, one AV1 -- differ
    // 3.6x in size and 1.4x in cost. `--keyframe-interval`, which can remove 90%
    // of the decode, moves the size not at all.
    //
    // So each file is opened once, header only, and asked how many keyframes it
    // holds and how big they are -- see `weigh_decode`. That is the actual shape
    // of the work: ~93% of fingerprinting time is intra decode of exactly those
    // frames. It costs one open per file and no decoding, roughly 0.8 ms against
    // decodes measured in seconds, and it is paid ONLY on cache misses, so a
    // re-run over a warm cache does not pay it at all -- which is why this sits
    // after the cache pass rather than beside the scan that stat'ed the sizes.
    //
    // The pass is separate and complete before any decoding starts, for the same
    // reason the cache pass is: the thread budget and the bar are both
    // apportioned against the TOTAL, so both need every weight in hand before
    // the first file starts. Parallel because it is pure I/O latency -- open,
    // read a header, close -- and nothing here competes with a decode yet.
    if !todo.is_empty() {
        let weighed: Vec<u64> = todo
            .par_iter()
            .map(|job| {
                if shutdown_requested() {
                    return job.size;
                }
                fingerprint::weigh_decode(&job.path, kf_interval, min_kf_samples, job.size)
            })
            .collect();
        for (job, weight) in todo.iter_mut().zip(weighed) {
            job.weight = weight;
        }

        // NOW the largest-first order the schedule depends on is real. It was
        // approximated by size up to here; a folder mixing codecs is exactly
        // where that approximation puts the wrong file first, since an AV1 copy
        // and an HEVC copy of the same footage differ threefold in cost and
        // barely at all in the direction size predicts.
        todo.sort_by_key(|job| std::cmp::Reverse(job.weight));
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
        // The threads are apportioned by WORK, not by file count. Counting files
        // is what left a 4 GB file decoding on a single thread: with hundreds of
        // videos queued the per-file share rounds to 1, the biggest file
        // (correctly) starts first, and by the time the queue has drained enough
        // for the share to widen it is far too late -- FFmpeg's thread count is
        // fixed when the decoder is opened, so a decode's width is decided once,
        // at the moment it starts, and never again. See `share_for` for the
        // rule that replaces it.
        //
        // Ownership of the loop moves off rayon for this reason too. Rayon's
        // adaptive splitting hands each worker a region of the input, so claims
        // do NOT arrive in weight order -- and the whole schedule rests on the
        // heaviest video claiming while the budget is still wide. An atomic
        // cursor over the weight-sorted list makes that ordering exact, and a
        // worker per budgeted thread is the most decodes that can run at once
        // anyway, since every decode reserves at least one.
        let total_weight: u128 = todo.iter().map(|job| job.weight as u128).sum();
        let budget = ThreadBudget::new(active_threads, total_weight);

        let cursor = AtomicUsize::new(0);
        let collected: Mutex<Vec<(usize, VideoFingerprint)>> =
            Mutex::new(Vec::with_capacity(todo_count));

        // Files finished. The bar's own position is decode work, so it cannot
        // answer "how many videos is that?" -- and that is the number a user
        // quotes back. It rides in `{prefix}` rather than sharing `{msg}`
        // with the file name because the two change on different events: the
        // name when a decode STARTS, the count when one ENDS. Formatting both at
        // claim time pinned the count at 0/6 for an entire six-file run -- every
        // worker had claimed before anything finished, so the only reads of the
        // counter all happened while it was still zero.
        let files_done = AtomicUsize::new(0);

        // The decodes running right now, keyed by their index in `todo`. The bar
        // has one line for a file name and up to `--threads` files in flight, so
        // it has to choose one, and the newest claim -- which is what it used to
        // print -- is the worst of the candidates: it is by construction the
        // file that has been running for the shortest time, and near the end of
        // a run it is a scrap of filler that finished seconds ago while the
        // heavyweight everybody is actually waiting on goes unnamed. A 54 GB
        // scan spent its last minute reporting a 30 MB clip.
        //
        // `todo` is sorted heaviest-first, so the lowest key in this map is both
        // the earliest claim and the heaviest file still open -- the honest
        // answer to "what is this waiting on". A BTreeMap for that ordering; the
        // lock is taken twice per video, against decodes measured in seconds.
        let in_flight: Mutex<BTreeMap<usize, String>> = Mutex::new(BTreeMap::new());

        // The bar measures WORK, not files and no longer bytes. Its denominator
        // is the same `total_weight` the thread budget is apportioned against, so
        // the bar and the scheduler agree about what half-done means; counting
        // files did not, and with inputs spanning three orders of magnitude a
        // count is close to meaningless -- a run can sit at 90% for most of its
        // wall time. Counting bytes agreed with the scheduler but was wrong in
        // the same place the scheduler was, and being consistently wrong is what
        // let a bar cross 50% a quarter of the way through a mixed-codec folder.
        //
        // What is on screen changed with it. The position itself is a count of
        // decoded keyframe-pixels, which nobody wants the raw value of, so it
        // shows as a percentage; the byte pair that used to stand next to it is
        // gone, because "3.2GB of 54GB" describes something the bar is no longer
        // measuring and the two would disagree in front of the user.
        //
        // The RATE survives, and is the one field here worth the plumbing. It is
        // the same speedometer the old {bytes_per_sec} was -- indicatif's own
        // windowed estimate, so it reads current speed rather than the run's
        // average -- but denominated in the work, which is what makes it steady:
        // megabytes a second swung by a factor of five between a high-bitrate
        // remux and a well-compressed encode of identical footage, while the
        // pixel rate is a property of the machine and barely moves.
        //
        // Still no ETA. The weight is a much better predictor than the size was,
        // but it predicts DECODE cost, and the tail of a run is threads going
        // idle rather than work being done -- a rate extrapolated across that
        // reads high for most of a scan and then stalls.
        let pb = if args.quiet {
            ProgressBar::hidden()
        } else {
            let pb = ProgressBar::new(total_weight.min(u64::MAX as u128) as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "{elapsed_precise} \u{2502} [{bar:28.cyan/blue}] \u{2502} {percent}% \u{2502} {work_rate} \u{2502} {prefix} \u{2502} {msg}",
                )
                .unwrap()
                .with_key("work_rate", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                    let _ = w.write_str(&work_rate(state.per_sec()));
                })
                .progress_chars("=>-"),
            );
            // Five fields of digits with only spaces between them read as one
            // long number at a glance, so they are ruled apart. A box-drawing
            // bar rather than an ASCII pipe: it cannot be confused with the `|`
            // that shows up in file names, and the run already prints text that
            // assumes a UTF-8 terminal.
            //
            // The bar is narrower than the 40 it used to be, because the size
            // pair, the rate and the rules all have to fit on the same line as
            // the file name -- a line that wraps redraws as two and flickers.
            // `{wide_msg}` looked like the tidier answer, since it truncates the
            // name to the space left instead of wrapping, but it renders the
            // message empty once the bar has any colour in it and the name
            // simply vanished a second into every run.
            pb.set_prefix(format!("0/{}", todo_count));
            pb
        };

        let workers = active_threads.min(todo_count).max(1);
        let todo = &todo;
        let budget = &budget;
        let cursor = &cursor;
        let collected = &collected;
        let pb = &pb;
        let files_done = &files_done;
        let in_flight = &in_flight;
        let newly_cached = &newly_cached;

        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(move || {
                    // Accumulated locally and merged once, so the shared lock is
                    // taken a handful of times per run rather than once per
                    // video. The index travels with the fingerprint purely so
                    // the finished list can be put back into input order --
                    // completion order is whatever the disk decided that day,
                    // and a reproducible run should not depend on it.
                    let mut local: Vec<(usize, VideoFingerprint)> = Vec::new();

                    // One video is off the queue, however it ended. `charged` is
                    // what the demuxer already moved the bar by while the file
                    // was open, so only the remainder is owed here -- the two
                    // together are always exactly the file's weight, which is
                    // what makes the bar land on 100% and not near it.
                    //
                    // Called for a file that was skipped or that failed too:
                    // neither will be attempted again, so their work has left
                    // the queue and a bar that ended short of full would be the
                    // lie the old per-file `inc` was already avoiding.
                    let finish = |idx: usize, job: &Job, charged: u64| {
                        let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
                        pb.set_prefix(format!("{}/{}", done, todo_count));
                        pb.inc(job.weight.saturating_sub(charged));

                        let mut open = in_flight.lock().unwrap_or_else(|e| e.into_inner());
                        open.remove(&idx);
                        if let Some((_, name)) = open.iter().next() {
                            pb.set_message(name.clone());
                        }
                    };

                    loop {
                        // Cheapest possible bail-out. Every video still queued
                        // costs one relaxed atomic load, so the tail of a
                        // 50k-file scan drains in microseconds; the videos
                        // actually being decoded right now stop via the
                        // identical check inside fingerprint_video's demux loop.
                        if shutdown_requested() {
                            break;
                        }

                        let idx = cursor.fetch_add(1, Ordering::SeqCst);
                        let Some(job) = todo.get(idx) else { break };

                        // Blocks if every thread is already promised to a decode
                        // that is running -- which is precisely when there is
                        // nothing useful for this worker to be doing. `None`
                        // means the run is shutting down.
                        let Some(grant) = budget.claim(job.weight) else { break };

                        let file_name = Path::new(&job.path)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned();
                        {
                            let mut open =
                                in_flight.lock().unwrap_or_else(|e| e.into_inner());
                            open.insert(idx, file_name);
                            if let Some((_, name)) = open.iter().next() {
                                pb.set_message(name.clone());
                            }
                        }

                        // How far into this file the demuxer has got, and how much
                        // of the file's weight that has already been credited to
                        // the bar. `fingerprint_video` reports a byte offset per
                        // packet, which on a linearly scanned container is
                        // thousands of times a second and far more often than the
                        // bar can redraw; PROGRESS_STEP_BYTES is what turns that
                        // firehose into a bounded number of increments.
                        //
                        // The two are in different units, which is why there are
                        // two of them: the demuxer can only say where it is in
                        // BYTES, and the bar is measured in decode work. The
                        // conversion is the file's own byte fraction, i.e. it
                        // assumes this file's keyframes are spread evenly through
                        // it -- true enough to move a bar with, and it does not
                        // accumulate error, because the offsets are absolute and
                        // `finish` pays whatever rounding left owing.
                        //
                        // `Cell`s because the hook has to stay `Fn` and the
                        // totals have to outlive it -- this closure is built
                        // fresh per video and never leaves this thread.
                        let charged_bytes = Cell::new(0u64);
                        let charged_work = Cell::new(0u64);
                        let advance = |pos: u64| {
                            // Clamped, never a delta from the raw offset: a seek
                            // that lands short walks the offset BACKWARDS, and an
                            // offset past the size we stat'ed (a file being
                            // appended to as we read it) would overrun the bar.
                            // Only forward movement inside the file counts.
                            let reached = pos.min(job.size);
                            if reached.saturating_sub(charged_bytes.get()) < PROGRESS_STEP_BYTES {
                                return;
                            }
                            charged_bytes.set(reached);

                            if job.size == 0 {
                                return;
                            }
                            let work =
                                (job.weight as u128 * reached as u128 / job.size as u128) as u64;
                            let delta = work.saturating_sub(charged_work.get());
                            charged_work.set(work);
                            pb.inc(delta);
                        };

                        let decoded = fingerprint_video(
                            &job.path,
                            kf_interval,
                            min_kf_samples,
                            grant.threads,
                            min_duration,
                            // The scan's own stat, the same figure this job's
                            // cache stamp was built from -- see the note on
                            // `fingerprint_video`.
                            job.size,
                            &advance,
                        );

                        // Handed back before the cache write below: an fsync is
                        // not decoding, and holding a wide share of the budget
                        // across one is how the machine goes quiet at exactly
                        // the wrong moment.
                        drop(grant);

                        let fp = match decoded {
                            Ok(Some(f)) => f,
                            Ok(None) => {
                                // Shorter than --min-duration. Not a failure, and
                                // nothing to cache: the header read that decided
                                // this is cheap enough to repeat next run.
                                stats.skipped_short.bump();
                                finish(idx, job, charged_work.get());
                                continue;
                            }
                            Err(e) => {
                                // An interrupt unwinds through here as our own
                                // "Interrupted while fingerprinting ..." error.
                                // That is the user's doing rather than the
                                // file's, so it is neither logged as a failure
                                // nor counted anywhere: exit code 130 already
                                // says what happened.
                                //
                                // A file that vanished between the scan and now
                                // also lands here rather than in a separate
                                // "unreadable" bucket, which is the honest
                                // description: the run was asked to fingerprint
                                // it and could not.
                                if !shutdown_requested() {
                                    log::error!("Failed to process {}: {:#}", job.path, e);
                                    stats
                                        .fingerprint_failed
                                        .record(format!("{}: {:#}", job.path, e));
                                }
                                finish(idx, job, charged_work.get()); // Work attempted
                                continue;
                            }
                        };

                        // Committed the moment it exists, over whatever this path
                        // held before. See cache_store: this is one transaction
                        // per video on purpose, so an interrupt (or a kill, or a
                        // crash) can only ever cost the decode still in flight.
                        match cache_store(db, &job.path, job.stamp, &fp) {
                            Ok(()) => {
                                newly_cached.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) => {
                                log::error!(
                                    "Failed to cache fingerprint for {}: {:#}",
                                    job.path,
                                    e
                                );
                                stats
                                    .cache_write_failed
                                    .record(format!("{}: {:#}", job.path, e));
                            }
                        }

                        local.push((idx, fp));
                        finish(idx, job, charged_work.get());
                    }

                    collected
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .extend(local);
                });
            }
        });

        pb.finish_and_clear();

        let mut fresh = collected
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .split_off(0);
        fresh.sort_unstable_by_key(|(idx, _)| *idx);
        fingerprints.extend(fresh.into_iter().map(|(_, fp)| fp));
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
    let final_groups = clustering::find_duplicate_groups(n, edges, &fingerprints, stats);

    // Consumes the Vec, so the pair list is not kept alive alongside the index.
    let matches = MatchIndex::new(matches);

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    // The single place that turns flags into intent. Report-only runs produce
    // None, and export.rs cannot touch a file without one of these.
    let disposal = disposal_for(move_to, args.delete, args.permanent);
    announce(disposal.as_ref());

    let deleted_paths = export::output_results(
        &final_groups,
        &fingerprints,
        &matches,
        args.output.as_ref(),
        start_time.elapsed().as_secs(),
        args.priority,
        disposal.as_ref(),
        args.yes,
        stats,
    )?;

    // Those files are no longer where they were, so the fingerprints filed
    // under their paths describe nothing at all. Deliberately BEFORE the
    // interrupt check below: the removals have already happened, and stopping is
    // no reason to leave the cache claiming otherwise. It is one small
    // transaction either way.
    if !deleted_paths.is_empty() {
        match cache_forget(db, &deleted_paths) {
            Ok(forgotten) => {
                log::debug!("Dropped {} cache entry(ies) for removed file(s).", forgotten)
            }
            Err(e) => {
                log::error!("Failed to drop cache entries for removed files: {:#}", e);
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

    fn dest() -> PathBuf {
        PathBuf::from("/mnt/scratch/dupes")
    }

    #[test]
    fn test_move_to_arms_the_run_on_its_own() {
        // It is not a deletion, so it does not need --delete's permission. A
        // user who has to type --delete to say "don't delete these" has been
        // handed a flag whose name is a lie.
        let d = disposal_for(Some(dest()), false, false);
        assert!(matches!(d, Some(Disposal::MoveTo(p)) if p == dest()));
    }

    #[test]
    fn test_move_to_outranks_both_deletion_flags() {
        // Precedence runs toward the recoverable option: whatever else was on
        // the command line, a run that mentions --move-to ends with the files
        // sitting in that folder.
        for (delete, permanent) in [(true, false), (false, true), (true, true)] {
            let d = disposal_for(Some(dest()), delete, permanent);
            assert!(
                matches!(d, Some(Disposal::MoveTo(ref p)) if *p == dest()),
                "--move-to with delete={} permanent={} must still move",
                delete,
                permanent
            );
        }
    }

    #[test]
    fn test_delete_still_means_trash_unless_told_otherwise() {
        assert!(matches!(disposal_for(None, true, false), Some(Disposal::Trash)));
        assert!(matches!(disposal_for(None, true, true), Some(Disposal::Permanent)));
    }

    #[test]
    fn test_nothing_is_armed_without_a_flag_that_says_so() {
        // The promise the README makes, as a property: exactly two flags can
        // take a file off its path, and --permanent is not one of them.
        assert!(disposal_for(None, false, false).is_none());
        assert!(
            disposal_for(None, false, true).is_none(),
            "--permanent alone must never act"
        );
    }

    fn scanned_at(path: &str) -> ScannedFile {
        ScannedFile {
            path: path.to_string(),
            size: 12_345,
            mtime: 1_700_000_000,
            mtime_nsec: 0,
        }
    }

    #[test]
    fn test_a_destination_inside_the_scan_is_recognised() {
        // Every landing path is dest + the source's absolute path, so a dest
        // under a scanned folder puts all of them back in next run's input.
        let scanned = vec![scanned_at("/mnt/media/show/ep01.mkv")];

        assert_eq!(
            scan_encloses(Path::new("/mnt/media/show/dupes"), &scanned).as_deref(),
            Some("/mnt/media/show")
        );
    }

    #[test]
    fn test_a_destination_above_the_scan_is_not_a_loop() {
        // The arrangement that used to trip the check: the moved file lands in
        // a sibling subtree the scan never reaches.
        let scanned = vec![scanned_at("/home/you/Documents/AN/ep01.mkv")];

        assert!(scan_encloses(Path::new("/home/you/Documents"), &scanned).is_none());
    }

    /// The bar spaces its fields itself -- one space either side of each rule --
    /// so a reading that carried its own padding would show up as a wider gap in
    /// front of one number than in front of all the others. Nothing this returns
    /// pads, including the "nothing measured yet" reading, and the unit ladder
    /// holds the rest of the range inside a single character of width.
    #[test]
    fn test_the_speedometer_pads_nothing_and_stays_narrow() {
        let readings = [
            work_rate(0.0),
            work_rate(f64::NAN),
            work_rate(1.0),
            work_rate(463e6),
            work_rate(2.5e9),
            work_rate(9.9e11),
            work_rate(4.2e13),
            work_rate(f64::MAX),
        ];
        for r in &readings {
            assert_eq!(r.trim(), r, "{:?}", readings);
        }

        // The dash is the one narrow reading, and it is only on screen before
        // there is anything to measure. Every real one is 9 or 10 characters.
        let measured: Vec<usize> =
            readings.iter().filter(|r| r.as_str() != "-").map(|r| r.chars().count()).collect();
        let lo = measured.iter().min().copied().unwrap();
        let hi = measured.iter().max().copied().unwrap();
        assert!(hi - lo <= 1, "{:?}", readings);
    }

    #[test]
    fn test_the_speedometer_changes_unit_rather_than_growing_digits() {
        assert_eq!(work_rate(463e6).trim(), "463 Mpx/s");
        assert_eq!(work_rate(2.5e9).trim(), "2.50 Gpx/s");
        assert_eq!(work_rate(9.9e11).trim(), "990 Gpx/s");
        // A rate too slow to have a megapixel in it still reads as a number and
        // not as the dash that means "no measurement yet".
        assert!(work_rate(1.0).trim().starts_with('0'));
    }

    // --- The decoder thread schedule -----------------------------------------

    // One "ordinary file" of decode work, so the shapes below read as
    // multiples of each other. The rule is scale-free -- only the ratio of
    // weight to queued matters -- so the size of the unit is arbitrary, and
    // these numbers meant bytes back when the weight did.
    const LOAD: u64 = 1 << 30;

    #[test]
    fn test_plenty_of_similar_work_means_one_thread_each() {
        // The common case, and the one the old count-based rule got right: with
        // seven times this much work queued behind it, widening this decode
        // would only take threads away from work that will outlast it anyway.
        assert_eq!(share_for(LOAD, 7 * LOAD as u128, 8), 1);
        assert_eq!(share_for(LOAD, 99 * LOAD as u128, 8), 1);
    }

    #[test]
    fn test_a_heavyweight_claims_its_share_of_the_run_up_front() {
        // The case the old rule got wrong. Eight tenths of the work left, on
        // eight threads: this file IS most of the run, so most of the machine
        // goes to it -- at the moment it starts, which is the only moment
        // FFmpeg will accept a thread count.
        assert_eq!(share_for(8 * LOAD, 2 * LOAD as u128, 8), 6);

        // Half the work gets half the threads.
        assert_eq!(share_for(4 * LOAD, 4 * LOAD as u128, 8), 4);
    }

    #[test]
    fn test_a_video_owed_more_than_one_thread_never_decodes_on_one() {
        // 1.43 threads of 8, which nearest-rounding gave 1. This is the shape a
        // library actually has -- one outsized file among a pile of ordinary
        // ones -- and 1 is the width the run cannot walk back later, so the
        // rounding goes the other way here and only here.
        assert_eq!(share_for(10 * LOAD, 48 * LOAD as u128, 8), 2);

        // Not a licence to round everything up: a share that is exactly one
        // thread is one thread, and a share below it still floors at one.
        assert_eq!(share_for(LOAD, 7 * LOAD as u128, 8), 1);
        assert_eq!(share_for(LOAD, 15 * LOAD as u128, 8), 1);

        // And above 2 the ordinary rule resumes -- 6.4 stays 6, because the
        // threads it would take are the ones the rest of the queue runs on.
        assert_eq!(share_for(8 * LOAD, 2 * LOAD as u128, 8), 6);
    }

    #[test]
    fn test_the_last_video_standing_takes_everything_free() {
        // Nothing queued behind it, so there is nobody to spread the budget
        // for. This is the straggler the whole change exists to prevent, and
        // the one place where the widest possible decode is unambiguously
        // right.
        assert_eq!(share_for(LOAD, 0, 8), 8);
        assert_eq!(share_for(LOAD, 0, 3), 3, "only what is actually free");
    }

    #[test]
    fn test_a_share_is_capped_at_what_a_decoder_can_use() {
        // Past MAX_DECODE_THREADS the extra threads buy nothing and cost a
        // full-resolution frame buffer each, so they are better left for the
        // next video.
        assert_eq!(share_for(LOAD, 0, 64), MAX_DECODE_THREADS);
    }

    #[test]
    fn test_a_small_video_never_gets_zero_threads() {
        // A decode always needs the thread it runs on, however little the file
        // weighs against the rest of the queue.
        assert_eq!(share_for(1, 100 * LOAD as u128, 8), 1);
        assert_eq!(share_for(0, 100 * LOAD as u128, 8), 1, "and a file we could not weigh still runs");
    }

    #[test]
    fn test_the_budget_is_never_overcommitted() {
        // The invariant `-t` is a promise about. Every claim here is made while
        // every earlier one is still running, which is the worst case for it.
        let shapes: [&[u64]; 4] = [
            &[8 * LOAD, LOAD, LOAD, LOAD, LOAD],
            &[LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD],
            &[64 * LOAD, 1, 1, 1, 1, 1, 1, 1],
            &[3 * LOAD, 2 * LOAD, LOAD],
        ];

        for total in 1..=16usize {
            for weights in shapes {
                let queued: u128 = weights.iter().map(|&w| w as u128).sum();
                let budget = ThreadBudget::new(total, queued);

                let mut held = 0usize;
                let mut grants = Vec::new();

                for &w in weights {
                    // Claiming with nothing free would (correctly) block, so
                    // this is where a single-threaded test has to stop.
                    if budget.free() == 0 {
                        break;
                    }
                    let grant = budget.claim(w).expect("not shutting down");
                    held += grant.threads;
                    grants.push(grant);

                    assert!(
                        held <= total,
                        "budget {} handed out {} threads across {} concurrent decodes",
                        total,
                        held,
                        grants.len()
                    );
                }
            }
        }
    }

    #[test]
    fn test_shares_widen_as_the_queue_drains() {
        // Four equal videos on eight threads. Two threads each while the queue
        // is full, four when half of it is gone, and the whole budget for the
        // last one -- which is exactly the ramp that was missing.
        let budget = ThreadBudget::new(8, 4 * LOAD as u128);

        let a = budget.claim(LOAD).unwrap();
        let b = budget.claim(LOAD).unwrap();
        assert_eq!((a.threads, b.threads), (2, 2));

        drop(a);
        drop(b);

        let c = budget.claim(LOAD).unwrap();
        assert_eq!(c.threads, 4);

        drop(c);

        let d = budget.claim(LOAD).unwrap();
        assert_eq!(d.threads, 8, "the tail of a scan must not decode single-threaded");
    }

    #[test]
    fn test_the_heaviest_video_is_provisioned_before_the_filler() {
        // The exact shape that motivated this: one enormous file and a pile of
        // small ones. The big one is claimed first (the todo list is sorted
        // largest-first) and takes most of the machine; the small ones share
        // what is left, one thread apiece, and finish alongside it instead of
        // finishing an hour early and leaving seven cores idle.
        let small = LOAD / 10;
        let queued = 8 * LOAD as u128 + 20 * small as u128;
        let budget = ThreadBudget::new(8, queued);

        let heavy = budget.claim(8 * LOAD).unwrap();
        assert_eq!(heavy.threads, 6);

        let a = budget.claim(small).unwrap();
        let b = budget.claim(small).unwrap();
        assert_eq!((a.threads, b.threads), (1, 1));
        assert_eq!(budget.free(), 0, "and the machine is fully committed");
    }

    #[test]
    fn test_a_grant_is_returned_on_drop() {
        // Including when a decode unwinds on an error: the release is the
        // destructor, not a line at the end of the happy path.
        let budget = ThreadBudget::new(8, LOAD as u128);

        {
            let grant = budget.claim(LOAD).unwrap();
            assert_eq!(grant.threads, 8);
            assert_eq!(budget.free(), 0);
        }

        assert_eq!(budget.free(), 8, "a finished decode must free its threads");
    }

    #[test]
    fn test_a_single_thread_budget_still_runs() {
        let budget = ThreadBudget::new(1, 4 * LOAD as u128);
        let grant = budget.claim(LOAD).unwrap();
        assert_eq!(grant.threads, 1);
        assert_eq!(budget.free(), 0);
    }

    fn stamp(mtime: i64, size: u64) -> Stamp {
        Stamp {
            mtime,
            mtime_nsec: 0,
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
    fn test_a_same_second_edit_that_preserves_the_size_is_still_caught() {
        // The hole whole seconds leave open: a re-mux or a scripted re-encode
        // finishing inside the same second the last one did, landing on the
        // same byte count. Seconds and size both agree, and the fingerprint on
        // record describes footage that is no longer in the file.
        let before = stamp(1_700_000_000, 12_345);
        let after = Stamp { mtime_nsec: 500_000_000, ..before };

        assert!(!before.matches(&after), "the nanoseconds are the only thing that differs");
        assert!(after.matches(&after));
    }

    #[test]
    fn test_the_sampling_knobs_are_part_of_what_makes_an_entry_valid() {
        // These change which keyframes are decoded, so a fingerprint made under
        // one setting is not an answer for a run using another.
        let base = stamp(1_700_000_000, 12_345);

        let sampled = Stamp { kf_interval: 5.0, ..base };
        assert!(!base.matches(&sampled));
        assert!(sampled.matches(&sampled));
    }

    #[test]
    fn test_min_keyframes_only_invalidates_while_it_is_flooring_something() {
        // `--min-keyframes` is a floor on the sampling interval. With sampling
        // off -- which is the DEFAULT -- there is no interval for it to floor
        // and it changes not one decoded frame, so an entry cannot stop being
        // valid because it moved. Comparing it regardless re-fingerprinted whole
        // libraries for a flag that did nothing.
        let base = stamp(1_700_000_000, 12_345);
        let fewer = Stamp { min_kf_samples: 2.0, ..base };

        assert!(
            base.matches(&fewer),
            "with no interval in force this flag decides nothing and must not invalidate"
        );

        // Turn sampling on and it is load-bearing again, so it counts.
        let sampled = Stamp { kf_interval: 5.0, ..base };
        let sampled_fewer = Stamp { kf_interval: 5.0, ..fewer };

        assert!(!sampled.matches(&sampled_fewer), "now it changes which frames are kept");
        assert!(sampled_fewer.matches(&sampled_fewer));
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
            total_ms: 3,
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
        // duplicate off disk (or moves it under --move-to) and nothing will ever
        // ask about that path again, so leaving the fingerprint would make every
        // armed run grow the cache by exactly what it just cleaned up.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let kept = "/videos/keep.mkv";
        let gone = "/videos/duplicate.mkv";
        let s = stamp(1_700_000_000, 12_345);

        cache_store(&db, kept, s, &mock_fp(kept)).unwrap();
        cache_store(&db, gone, s, &mock_fp(gone)).unwrap();

        let forgotten = cache_forget(&db, &[gone.to_string()]).unwrap();

        assert_eq!(forgotten, 1);
        assert!(cache_lookup(&db, gone, &s).is_none(), "a removed file keeps no fingerprint");
        assert!(cache_lookup(&db, kept, &s).is_some(), "and its neighbours are untouched");
    }

    /// A report and a real file to go with it, sized to agree with `mock_fp`.
    fn armed_report(dir: &tempfile::TempDir) -> (String, String) {
        let doomed = dir.path().join("dupe.mkv");
        std::fs::write(&doomed, vec![b'x'; 12_345]).unwrap();
        let doomed = doomed.to_string_lossy().to_string();

        let report = dir.path().join("report.csv");
        std::fs::write(
            &report,
            format!("full_path;size_bytes;action\n{};12345;DELETE\n", doomed),
        )
        .unwrap();

        (doomed, report.to_string_lossy().to_string())
    }

    #[test]
    fn test_a_report_run_forgets_what_it_removed() {
        // --from-report reaches the cache by a different route from the grouped
        // run, and the reason to keep it is the same: nothing will ever ask
        // about that path again, so an entry left behind is one the cache grows
        // by every time a cleanup succeeds.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let (doomed, report) = armed_report(&dir);

        let s = stamp(1_700_000_000, 12_345);
        cache_store(&db, &doomed, s, &mock_fp(&doomed)).unwrap();
        assert!(cache_lookup(&db, &doomed, &s).is_some(), "precondition");

        let args = Args::parse_from([
            "vid-fp",
            "--from-report",
            &report,
            "--delete",
            "--permanent",
            "--yes",
        ]);
        let stats = RunStats::default();

        assert!(matches!(
            run(&args, &db, Instant::now(), 1, &stats),
            Ok(Outcome::Completed)
        ));
        assert!(!Path::new(&doomed).exists(), "the file the report named is gone");
        assert!(
            cache_lookup(&db, &doomed, &s).is_none(),
            "and so is the fingerprint filed under its path"
        );
        assert!(!stats.had_problems());
    }

    #[test]
    fn test_a_report_run_refuses_to_start_unarmed() {
        // The one piece of validation clap cannot express. Without it the mode
        // would read a file, decide nothing, and print nothing the report does
        // not already say -- while leaving the user believing it had acted.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let (doomed, report) = armed_report(&dir);

        let args = Args::parse_from(["vid-fp", "--from-report", &report]);
        let stats = RunStats::default();

        let err = run(&args, &db, Instant::now(), 1, &stats)
            .expect_err("no disposal was ever constructed")
            .to_string();

        assert!(err.contains("--delete"), "{}", err);
        assert!(err.contains("--move-to"), "{}", err);
        assert!(Path::new(&doomed).exists(), "and nothing was touched");
    }

    /// Everything the run needs to reach the argument checks: a real scan root
    /// so nothing earlier can fail for an unrelated reason.
    fn args_over(dir: &tempfile::TempDir, extra: &[&str]) -> Args {
        let root = dir.path().to_string_lossy().to_string();
        let mut argv = vec!["vid-fp".to_string(), root];
        argv.extend(extra.iter().map(|s| s.to_string()));
        Args::parse_from(argv)
    }

    fn run_error(dir: &tempfile::TempDir, db: &Database, extra: &[&str]) -> String {
        let stats = RunStats::default();
        run(&args_over(dir, extra), db, Instant::now(), 1, &stats)
            .expect_err("this run should have been refused")
            .to_string()
    }

    #[test]
    fn test_a_percentage_outside_zero_to_a_hundred_is_refused_before_any_work() {
        // Above 100 is the case worth catching: `match_overlap` clamps coverage
        // to 1.0, so such a floor is one no pair can ever clear -- the run would
        // fingerprint the whole library and be structurally incapable of
        // reporting a group. Below 0 is the same typo in the other direction,
        // and it only ever LOOSENS the run, which with --delete armed can add
        // files to the DELETE set.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        for bad in ["--match-percent=500", "--match-percent=-50", "--match-percent=nan"] {
            let err = run_error(&dir, &db, &[bad]);
            assert!(err.contains("--match-percent"), "{}: {}", bad, err);
        }

        // The ends of the range are legal: 0 means "report every pair the index
        // proposes", which is a supported way to run this.
        for ok in ["--match-percent=0", "--match-percent=100"] {
            let stats = RunStats::default();
            assert!(
                run(&args_over(&dir, &[ok]), &db, Instant::now(), 1, &stats).is_ok(),
                "{} is inside the range and must be accepted",
                ok
            );
        }
    }

    #[test]
    fn test_a_tolerance_past_chance_is_refused_before_any_work() {
        // 32 is the mean distance between two unrelated frame hashes, so past it
        // every file matches every other and the flag has nothing left to
        // control. It used to be refused only above 64 -- the widest two hashes
        // can be apart -- which let through a whole range where the answer was
        // one group holding the library, and where the clustering ceilings stop
        // helping: a NEARLY complete graph has few enough maximal cliques to slip
        // under them while costing minutes to prove it.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        for bad in ["-d=33", "-d=40", "-d=64", "-d=100"] {
            let err = run_error(&dir, &db, &[bad]);
            assert!(err.contains("--hamming-distance"), "{}: {}", bad, err);
        }

        // Both ends stay legal: 0 is "only identical frames", and chance itself
        // is the last rung of a sensitivity control rather than a special case.
        for ok in ["-d=0", "-d=32"] {
            let stats = RunStats::default();
            assert!(
                run(&args_over(&dir, &[ok]), &db, Instant::now(), 1, &stats).is_ok(),
                "{} is inside the range and must be accepted",
                ok
            );
        }
    }

    #[test]
    fn test_a_negative_or_unparseable_duration_floor_is_refused() {
        // NaN is the reason this is a range test rather than `< 0.0`: every
        // comparison against it is false, so a NaN floor slipped through and
        // silently disabled the gate it was meant to tighten.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        for bad in ["--min-duration=-5", "--min-duration=nan"] {
            let err = run_error(&dir, &db, &[bad]);
            assert!(err.contains("--min-duration"), "{}: {}", bad, err);
        }
    }

    #[test]
    fn test_an_unwritable_report_path_is_refused_before_the_scan() {
        // The whole point is WHEN this fires. Checked only at the point of
        // writing, a mistyped -o did the entire scan and then threw the results
        // away -- and with --delete armed, it did that after the files were
        // already gone.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        let err = run_error(&dir, &db, &["-o", "/nonexistent/vid-fp/report.csv"]);
        assert!(err.contains("--output"), "{}", err);
        assert!(err.contains("/nonexistent/vid-fp"), "{}", err);

        // A folder is not a file to write either.
        let as_dir = dir.path().to_string_lossy().to_string();
        let err = run_error(&dir, &db, &["-o", &as_dir]);
        assert!(err.contains("folder"), "{}", err);
    }

    #[test]
    fn test_an_ordinary_report_path_is_accepted() {
        // Including the bare relative name, whose parent is the empty string
        // rather than a directory -- the one shape this check could plausibly
        // get wrong in the direction that refuses a perfectly good run.
        assert!(check_output_path(None).is_ok());
        assert!(check_output_path(Some(&"report.csv".to_string())).is_ok());

        let dir = tempfile::tempdir().unwrap();
        let inside = dir.path().join("report.csv").to_string_lossy().to_string();
        assert!(check_output_path(Some(&inside)).is_ok());
    }

    #[test]
    fn test_forgetting_something_uncached_is_not_an_error() {
        // A file whose cache write failed earlier in the run is still a
        // perfectly good target, and its absence here means nothing.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        assert_eq!(cache_forget(&db, &[]).unwrap(), 0);
        assert_eq!(cache_forget(&db, &["/videos/never-cached.mkv".to_string()]).unwrap(), 0);
    }

    #[test]
    fn test_clearing_the_cache_leaves_a_usable_empty_table() {
        // The clear now compacts on its way past, which drops and recreates
        // pages under the table definition. The run that follows immediately
        // starts writing fingerprints into it, so "empty" must not mean "gone".
        let dir = tempfile::tempdir().unwrap();
        let mut db = temp_db(&dir);
        let path = "/videos/some_show_s01e01.mkv";
        let s = stamp(1_700_000_000, 12_345);

        cache_store(&db, path, s, &mock_fp(path)).unwrap();
        clear_cache(&mut db).unwrap();

        assert!(cache_lookup(&db, path, &s).is_none(), "nothing survives a clear");

        cache_store(&db, path, s, &mock_fp(path)).unwrap();
        assert!(
            cache_lookup(&db, path, &s).is_some(),
            "and the table is ready for the scan that follows"
        );
    }
}