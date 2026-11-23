mod against;
mod cache;
mod clustering;
mod compare;
mod confirm;
mod content;
mod export;
mod fingerprint;
pub mod found;
pub mod progress;
mod report;
mod scan;
mod stats;
pub mod thumbnail;
pub mod utils;
mod sources;
pub use sources::desktop_dir;
pub use cache::default_cache_file;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use clap_complete::Shell;
use compare::{find_all_matches, max_hamming_distance, MatchIndex, HASH_BITS};
use export::{Disposal, Format, ReportTarget, Sink};
use fingerprint::{Floor, VideoFingerprint};
use log::info;
use cache::Cache;
use stats::RunStats;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;
use utils::{shutdown_requested, Priority};

// Refuse a static build whose bindings and archives came from different FFmpeg
// releases. `build.rs` writes this file from the version `libavcodec.pc` states
// for the prefix in FFMPEG_DIR; ffmpeg-sys-next runs first and can bind itself
// to the system FFmpeg without anything downstream being able to tell, so the
// assertion is made against what it really generated. See build.rs's module doc.
#[cfg(feature = "static-ffmpeg")]
include!(concat!(env!("OUT_DIR"), "/static_ffmpeg_version_check.rs"));


/// Exit code for a run that finished, reported, and did everything it could --
/// but hit at least one problem on the way (an unreadable file, a video that
/// would not decode, a deletion that failed). 0 means clean, 1 is anyhow's
/// fatal path, 130 is the shell convention for SIGINT. Scripts that don't care
/// can ignore it; scripts that do finally have something to test.
const EXIT_WITH_PROBLEMS: i32 = 2;

/// How far past the machine's core count `--threads` is allowed to reach.
///
/// **What actually bounds this flag is MEMORY, not CPU time**, and that is worth
/// stating first because it is not what the flag looks like it controls. Peak RSS
/// is close to linear in `--threads` while throughput is flat, measured here on
/// two libraries with a fresh cache each run (so every file is decoded):
///
/// ```text
///     -t     100 clips, 720p          16 clips, 1080p
///      8     5.2-5.4 s   112 MB       5.3 s    422 MB
///     16     5.2-5.5 s   205 MB       5.2 s    647 MB
///     32     5.5-6.0 s   370 MB       5.3 s  1,137 MB
///     64     5.4-6.4 s   620 MB       5.5 s  2,022 MB
/// ```
///
/// Wall time does not improve once at any rung -- CPU utilisation is already
/// 71-73% of eight cores at `-t 8` -- and the slope of the memory is a property
/// of the LIBRARY rather than of the machine: ~9 MB per extra thread at 720p and
/// ~28 MB at 1080p, so 4K footage would cost around four times the latter again.
/// Note the 1080p column keeps climbing past 32 even though 16 files cap the
/// worker count at 16: the budget is what widens, so each decode is handed more
/// decoder threads and each of those holds its own full-resolution frame buffer
/// (see `MAX_DECODE_THREADS`). Both mechanisms scale with this number. This is
/// also the one figure the rest of the binary works hardest to keep flat -- the
/// `mallopt` calls in `main` and the single contiguous frame buffer exist for it
/// -- so an unbounded `-t` is the one input that can undo all of it.
///
/// Thread creation is the far cruder limit and only bites much later. Timed on a
/// warm two-file cache, so the pool is the entire cost: 0.02 s at 8 through 256,
/// 0.10 s at 512, 0.58 s at 1,024, 2.0 s at 2,048, 5.3 s at 4,096, 14.0 s at
/// 8,192, and at 16,384 it burns 63 s of wall and 459 s of user time before
/// failing. That is the reported hang -- `-t 1000` spending 19.7 s on a two-file
/// library, `-t 20000` failing inside rayon's `build_global` with EAGAIN against
/// this machine's `RLIMIT_NPROC` of 22,609 and then surviving a first SIGTERM.
///
/// So the useful floor and the harmful ceiling are far apart, and 4 sits between
/// them with room on both sides rather than being tuned to either. It is
/// generous against the measured throughput, which says one worker per core is
/// already enough on local storage; it is restrained against the measured memory,
/// since 4x on this machine is 32 workers and about 3.5 GB on a 4K library. The
/// gap is left deliberately wide because the case that justifies ANY
/// oversubscription is the one that cannot be measured from here: decoding off a
/// network mount or a spinning disk, where workers block on I/O and cores sit
/// idle. That case is real, it is not reproducible on this machine, and a ceiling
/// tuned to local decode would refuse it.
///
/// Stated per core rather than as a flat number so it travels: a flat 32 would be
/// a hard cap on a 128-core machine and 16x oversubscription on a dual-core one.
/// Per-core also tracks the memory, since the machines with many cores are the
/// ones with the RAM to spend on many frame buffers.
const MAX_THREAD_OVERSUBSCRIPTION: usize = 4;

/// The smallest ceiling `active_thread_count` will impose, whatever the core
/// count.
///
/// Purely a portability floor, and it does nothing on any machine with 4 cores or
/// more. The per-core rule above is about cores a decode can keep BUSY, but the
/// one case that justifies oversubscription at all is workers blocked on I/O,
/// where the useful number tracks the storage's latency and not the CPU at all --
/// so on a one- or two-core VM with its videos on a network mount, 4x lands at 4
/// or 8 and would be refusing a request that makes sense. 16 costs nothing here
/// (8 cores put the ceiling at 32) and is still three orders of magnitude below
/// where thread creation starts to hurt.
const MIN_THREAD_CEILING: usize = 16;


/// What a folder walk assumes a video is called.
///
/// This list is a guess and the cost of it being wrong is asymmetric: an
/// extension too many means one file the decoder rejects, named in the run's
/// problem list; an extension too few means a folder that reports "No videos
/// found" and a user who concludes the tool is broken. It was six entries long
/// -- mp4, mkv, avi, mov, flv, webm -- and a camcorder's `.mts`, a TV capture's
/// `.ts`, a DVD rip's `.vob` and an iTunes `.m4v` all fell straight through it.
///
/// So the rule for what belongs here is "a container FFmpeg can demux that
/// something in the wild writes video into", not "a container this author
/// uses". `.ts` is the one entry with a real cost -- a TypeScript source tree
/// scanned recursively now hands every file to the decoder and gets a problem
/// row per file -- and it stays, because it is also how every DVB capture and
/// half the camcorders on earth name their footage, and the failure it causes is
/// loud and localised while the failure it prevents is silent. `-x` narrows it
/// for anyone that bites.
///
/// Not a substitute for `-x '*'`: no list can name a file that has no extension.
///
/// The second ten came in together, each something a camera or an archive
/// writes and FFmpeg demuxes: `mxf` (broadcast and cinema cameras), `m2t`
/// (HDV camcorders), `mod`/`tod` (JVC and Panasonic SD camcorders), `dv`,
/// `m2v` (elementary MPEG-2, a DVD's video track), `f4v` (Flash-era MP4),
/// `rm`/`rmvb` (RealMedia) and `ogm`. `mod` is also a tracker-music suffix,
/// the same loud-and-localised cost `.ts` carries.
///
/// The third ten likewise: `trp`/`tp` (DVB and set-top-box captures, MPEG-TS),
/// `wtv` (Windows Media Center recordings), `3g2` and `qt` (MOV family), `mk3d`
/// (Matroska 3D), `vro` (DVD-VR camcorder and recorder discs, MPEG-PS), `lrv`
/// (the low-resolution MP4 proxy a GoPro writes beside every clip, a duplicate
/// by construction) and `mpe`/`m1v` (MPEG-1 elementary and program streams).
const DEFAULT_EXTENSIONS: [&str; 38] = [
    "mp4", "m4v", "mkv", "webm", "avi", "mov", "flv", "wmv", "asf", "mpg", "mpeg", "m2ts", "mts",
    "ts", "vob", "ogv", "3gp", "divx", "mxf", "m2t", "mod", "tod", "dv", "m2v", "f4v", "rm",
    "rmvb", "ogm", "trp", "tp", "wtv", "3g2", "qt", "mk3d", "vro", "lrv", "mpe", "m1v",
];

#[derive(Parser, Debug)]
#[command(
    author, version,
    about = "Fast video duplicate and clip finder",
    long_about = "Fingerprints videos from their keyframes and groups files with the \
                  same content, whatever their resolution or container, including one \
                  video that is only a trimmed clip inside another.\n\n\
                  Report-only by default. Files are touched only with --delete or \
                  --move-to."
)]
struct Args {
    /// Folders and video files to scan. `-` reads a list of paths from stdin
    ///
    /// A folder is searched for videos; a file you name is scanned whatever its
    /// extension, since --extensions is only a guess about what a FOLDER
    /// contains. Example: `fd -e mkv | vid-fp -`.
    #[arg(
        required_unless_present_any = ["completions", "man", "from_file", "from_report"],
        num_args = 1..,
        value_name = "PATH",
        value_hint = clap::ValueHint::AnyPath
    )]
    include: Vec<String>,

    /// Exclude a folder or a single file; repeat for several
    ///
    /// Matched one whole path component at a time, so -e ~/keep covers
    /// everything under it while -e ~/clips/take does NOT cover
    /// ~/clips/take.mkv. Applies to piped and named paths too, and resolves
    /// through symlinks, so excluding either a link or its target covers both.
    #[arg(short = 'e', long = "exclude", value_name = "PATH",
          value_hint = clap::ValueHint::AnyPath)]
    exclude: Vec<String>,

    /// Read the paths to scan from a file, one per line (`-` = stdin)
    ///
    /// Entries may be folders or files, and combine with any paths given on the
    /// command line.
    #[arg(long = "from-file", value_name = "FILE",
          value_hint = clap::ValueHint::FilePath)]
    from_file: Option<String>,

    /// Paths in the list are NUL-separated, for `find -print0` or `fd -0`
    ///
    /// The only way to pass a filename containing a newline. Also separates
    /// what --list prints.
    #[arg(short = '0', long = "null")]
    null: bool,

    /// Scan folders recursively (off by default)
    ///
    /// A Trash or recycle bin met on the way is not walked; name it to scan it.
    #[arg(short = 'r', long = "recursive")]
    recursive: bool,

    /// Follow symbolic links while scanning
    ///
    /// Files are deduplicated by (device, inode), so the same bytes are never
    /// fingerprinted twice however many links reach them. Deleting or moving a
    /// file found through a link acts on the file the link leads to, not on the
    /// link; protect a folder with --exclude, which follows links too.
    #[arg(long = "follow-symlinks")]
    follow_symlinks: bool,

    /// Extensions a FOLDER walk treats as video, comma-separated or repeated
    ///
    /// Case-insensitive; a leading dot or `*.` is optional. `-x '*'` takes every
    /// file whatever it is called, the only way to reach files with no
    /// extension; expect failures for the non-videos it hands to the decoder.
    /// An entry prefixed with `!` is an exception: `-x '!flac'` is every file
    /// but those, `-x 'mp4,mkv,!mkv'` is the list with one removed. Quote both
    /// forms, since a shell eats a bare `*` or `!`.
    #[arg(
        short = 'x',
        long = "extensions",
        value_delimiter = ',',
        value_name = "EXT",
        default_values_t = DEFAULT_EXTENSIONS.map(String::from)
    )]
    extensions: Vec<String>,

    /// Frame-match tolerance, in bits out of 64; higher is looser
    ///
    /// Two unrelated frames sit about 32 bits apart, so the useful range is
    /// roughly 2 to 8. Each frame carries a second, finer hash, and two frames
    /// are compared on both together: -d sets how far below chance that
    /// combined distance must be, the same distance below chance as -d bits are
    /// on one hash. A lone frame match must be within it; one that another
    /// match agrees with about the time offset may reach 1.5 standard
    /// deviations further, and further still needs more agreeing matches.
    /// Values above 32 are refused.
    #[arg(short = 'd', long = "hamming-distance", value_name = "N", default_value_t = 6)]
    hamming_distance: u32,

    /// Minimum percentage of a video that must match, from 0 to 100
    ///
    /// 0 turns the gate off and reports every pair sharing any footage at all;
    /// a pair sharing none is never reported. Values outside the range are
    /// refused.
    #[arg(short = 'p', long = "match-percent", value_name = "F", default_value_t = 20.0)]
    match_percent: f32,

    /// Minimum shared clip length in seconds; also skips shorter videos
    ///
    /// 0 = off. Independent of --match-percent; both must be satisfied.
    /// Negative and non-finite values are refused.
    #[arg(long = "min-duration", value_name = "SECS", default_value_t = 0.0)]
    min_duration: f64,

    /// Seconds between sampled keyframes (0 = decode every keyframe)
    ///
    /// Long videos sample at this interval; short ones use a finer spacing so
    /// they keep at least --min-keyframes samples, so the interval only reaches
    /// videos longer than interval x --min-keyframes seconds. Higher is faster
    /// on those, but makes short matches in them harder to find. Negative and
    /// non-finite values are refused.
    #[arg(long = "keyframe-interval", value_name = "F", default_value_t = 0.0)]
    kf_interval: f64,

    /// Minimum keyframes to keep for short videos
    ///
    /// Only used when --keyframe-interval is above 0; 0 removes the floor.
    /// Higher finds more matches and costs decode time, mostly on libraries of
    /// short videos. It is not strictly monotonic: denser sampling makes each
    /// hash stand for a shorter span, so frames that match nothing can dilute a
    /// pair's coverage under --match-percent. Negative and non-finite values
    /// are refused.
    #[arg(long = "min-keyframes", value_name = "F", default_value_t = 64.0)]
    min_kf_samples: f64,

    /// Property that decides which file to KEEP
    #[arg(short = 'k', long = "priority", value_name = "P", default_value = "length")]
    priority: Priority,

    /// Write the report here; `-` writes it to stdout
    ///
    /// The format follows the extension (.txt, .csv, .json; anything else is
    /// written as text) unless --format overrides it. On stdout the report
    /// replaces the terminal listing rather than repeating it, and progress,
    /// warnings and prompts stay on stderr. A file actually named - is
    /// reachable as ./-. A file the scan reads, or an existing file named like
    /// a video, is refused rather than written over.
    #[arg(short = 'o', long = "output", value_name = "FILE",
          value_hint = clap::ValueHint::FilePath)]
    output: Option<String>,

    /// Write the report in this format whatever --output is called
    ///
    /// Needed for stdout, which has no extension to read, and for a report kept
    /// under a name that carries no format. Without it the extension decides.
    #[arg(long = "format", value_enum, value_name = "FORMAT")]
    format: Option<Format>,

    /// Move the files marked DELETE to the system trash
    ///
    /// Add --permanent to remove them for good. Files marked KEEP or REVIEW are
    /// never touched.
    #[arg(long = "delete")]
    delete: bool,

    /// With --delete, remove the files permanently instead of trashing them
    ///
    /// Irreversible. No effect on its own.
    #[arg(long = "permanent")]
    permanent: bool,

    /// Move the files marked DELETE under this folder
    ///
    /// Each file's absolute path is recreated inside it (/mnt/media/ep.mkv ->
    /// DIR/mnt/media/ep.mkv), so the whole run can be undone with one copy back
    /// from DIR. For where the system trash is not available: external disks,
    /// NFS mounts, headless servers. Arms the run on its own and supersedes
    /// --delete and --permanent.
    #[arg(long = "move-to", value_name = "DIR",
          value_hint = clap::ValueHint::DirPath)]
    move_to: Option<String>,

    /// Act on a CSV or JSON report from an earlier run instead of scanning
    ///
    /// Every row whose action reads DELETE is disposed of and nothing else is
    /// touched, so editing that field is how you act on REVIEW rows or spare a
    /// file. Nothing is re-fingerprinted and no groups are recomputed: the
    /// report is the decision. Each file is re-checked against the size the
    /// report recorded and left alone if it changed. The format is read from
    /// the file, not its name; a .txt report cannot be replayed because it
    /// records no size. Requires --delete or --move-to, and cannot be combined
    /// with scanning or matching options.
    #[arg(
        long = "from-report",
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath,
        conflicts_with_all = [
            "include", "exclude", "from_file", "null", "recursive", "follow_symlinks",
            "extensions", "hamming_distance", "match_percent", "min_duration",
            "kf_interval", "min_kf_samples", "priority", "output", "format",
            "prune_cache", "threads", "against",
        ]
    )]
    from_report: Option<String>,

    /// Answer each scanned file against this library instead of grouping them
    ///
    /// Repeat for several. The library is scanned with the same options and
    /// cached the same way, but never compared with itself and never touched:
    /// each scanned file is a repeat (the library holds it at least as well:
    /// marked DELETE, so --delete and --move-to act on repeats only), better
    /// (it outranks every library file it matched), review, or new. A file
    /// reached from both sides belongs to the side whose folder holds it more
    /// closely, the scan's on a tie.
    #[arg(long = "against", value_name = "LIBRARY",
          value_hint = clap::ValueHint::AnyPath)]
    against: Vec<String>,

    /// With --against, print the paths given these answers to stdout
    ///
    /// Comma-separated or repeated: repeat, better, review, new. One path a
    /// line, or NUL-separated with -0; without -0 a path holding a newline
    /// stops the run, since it would read as two. Nothing else is printed to
    /// stdout, so the list can be piped straight on.
    #[arg(long = "list", value_name = "ANSWER", value_enum, value_delimiter = ',',
          requires = "against")]
    list: Vec<against::Answer>,

    /// Answer yes to the confirmation shown before any file is touched
    ///
    /// The confirmation only appears on an interactive terminal, so a piped or
    /// redirected run is never prompted and never blocks.
    #[arg(short = 'y', long = "yes")]
    yes: bool,

    /// Use this cache file instead of the default one
    ///
    /// The default is under $XDG_CACHE_HOME/vid-fp (or ~/.cache/vid-fp). A run
    /// locks its cache for the whole scan, so scanning libraries in parallel
    /// needs a cache each. An existing directory, or a path with a trailing
    /// slash, gets the default filename inside it; missing parents are created.
    /// --clear-cache and --prune-cache act on whichever cache this names.
    #[arg(long = "cache", value_name = "PATH",
          value_hint = clap::ValueHint::FilePath)]
    cache: Option<String>,

    /// Delete ALL cache before running
    ///
    /// A cache file that cannot be read as one is moved aside to
    /// <name>.damaged and a new cache started.
    #[arg(long = "clear-cache")]
    clear_cache: bool,

    /// Write every log line to this file, including the ones the terminal hides
    ///
    /// A failure the end-of-run summary accounts for is not also printed while
    /// the run works, so this is where the unabridged list goes when you want
    /// to grep it. --quiet narrows the terminal and not this file, so the two
    /// together give a silent run and a full account on disk. Truncated at the
    /// start of each run; a file the scan reads, or an existing file named like
    /// a video, is refused.
    #[arg(long = "log-file", value_name = "PATH",
          value_hint = clap::ValueHint::FilePath)]
    log_file: Option<String>,

    /// Drop the cached fingerprint of every file this scan did not find
    ///
    /// Keeps a cache from growing with libraries that have moved on. Skipped,
    /// loudly, when the scan was not complete enough to measure against: a path
    /// that would not resolve, a folder that could not be read, or a scan that
    /// found no videos. Pruning against a partial scan would throw away good
    /// fingerprints, so the run keeps them and exits 2. A file moved since the
    /// last run is recognised by its content before anything is dropped, so it
    /// keeps its fingerprint.
    #[arg(long = "prune-cache")]
    prune_cache: bool,

    /// Suppress all terminal output except errors
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,

    /// Maximum number of threads to use (0 = all CPU cores)
    ///
    /// All cores is fastest for videos on a local disk: raising it further does
    /// not measurably speed a scan up, and every concurrent decode holds a
    /// full-resolution frame buffer, so peak memory grows in step with this
    /// number. More workers than cores is worth trying when the videos are on a
    /// network mount or a slow disk. Anything above four per core, or 16 if
    /// that is larger, is capped at that with a note saying so.
    #[arg(short = 't', long = "threads", value_name = "N", default_value_t = 0)]
    threads: usize,

    /// Print a shell completion script to stdout and exit
    #[arg(long = "completions", value_name = "SHELL", exclusive = true)]
    completions: Option<Shell>,

    /// Print the roff man page to stdout and exit
    #[arg(long = "man", exclusive = true)]
    man: bool,

    /// Where the desktop app's worker saves the groups (`worker_main`). Not a
    /// flag: nothing on the command line can set it.
    #[arg(skip)]
    window_result: Option<PathBuf>,
}

/// How the run ended. `Interrupted` carries the phase name purely so the final
/// message can tell the user where it stopped.
#[derive(Debug)]
enum Outcome {
    Completed,
    Interrupted,
}

/// Whether the files this disposal removes take their cache entries with them.
///
/// Only a permanent delete. A file sent to the Trash or moved under `--move-to`
/// still exists, keeps its modification time and size, and is one restore away
/// from its old path -- where the entry left in place makes it a cache hit, and
/// its identity lets the content pass recognise it anywhere else. Forgetting
/// both meant a restored library was decoded all over again, which the copy
/// fallback's own care to carry the mtime across (`copy_durably`) was there to
/// prevent. An entry for a file that never comes back is what `--prune-cache`
/// collects.
fn forgets(disposal: &Disposal) -> bool {
    matches!(disposal, Disposal::Permanent)
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

/// The folders this run had to make for `--move-to`, and its promise to take
/// them back if it then refuses to run.
///
/// Creating the destination IS the request, but only for a run that goes on to
/// use it. The destination has to exist before anything can be decided about it
/// -- `canonicalize` needs a real inode, and an `-e` naming the same folder is
/// canonicalized too, so a destination that does not exist yet cannot be
/// excluded -- and the checks that can refuse the run all come after that.
/// `vid-fp lib -r --move-to lib/dupes` is refused by the loop guard in `run`
/// having already created `lib/dupes` INSIDE the scan root: correct refusal,
/// workable advice, and an empty directory left behind by a run that did
/// nothing, which the user now has to clean up before the folder stops being
/// part of their library. Every start-up refusal between the two left the same
/// thing -- an `--output` that cannot be written, a scan path that will not
/// resolve.
///
/// Taking it back can never destroy anything, which is what makes it safe to
/// fire on every early return: `remove_dir` refuses a directory that is not
/// empty, so anything that appeared inside one meanwhile -- including files
/// this run moved -- keeps it. `keep` is called once the run has passed
/// everything that could turn it away.
#[derive(Default)]
struct Conjured {
    /// Deepest first, which is the order they have to be removed in.
    dirs: Vec<PathBuf>,
}

impl Conjured {
    /// The run is committed: the folder is the user's now.
    fn keep(&mut self) {
        self.dirs.clear();
    }
}

impl Drop for Conjured {
    fn drop(&mut self) {
        for dir in &self.dirs {
            // Best effort by design. A destination that is no longer empty is
            // one this run must not touch, and a run that is already on its way
            // out with an error to report has nothing to say about a directory
            // it could not tidy up.
            let _ = std::fs::remove_dir(dir);
        }
    }
}

/// The directories `create_dir_all(dir)` will have to make, deepest first.
///
/// Asked BEFORE the creation, because afterwards there is nothing left to tell
/// what was already there from what this run conjured -- and only the latter is
/// this run's to remove.
fn missing_ancestors(dir: &Path) -> Vec<PathBuf> {
    let mut missing = Vec::new();
    let mut cursor = dir;

    // `exists` follows symlinks, which is the right question here: a link to an
    // existing folder is not something this run made, and a dangling one is
    // something `create_dir_all` will refuse rather than replace.
    while !cursor.as_os_str().is_empty() && !cursor.exists() {
        missing.push(cursor.to_path_buf());
        match cursor.parent() {
            Some(parent) => cursor = parent,
            None => break,
        }
    }

    missing
}

/// Turn `--move-to DIR` into the absolute folder the files will land under.
///
/// Resolved BEFORE any work on purpose: a destination that cannot be created is
/// a typo, and finding a typo out after fingerprinting a library is finding it
/// out several hours too late. Canonical, so the containment check and the
/// mirrored landing paths are both working from the same absolute form the
/// scanner produces.
///
/// The `Conjured` it returns beside the path is what undoes the creation if the
/// run is refused after all; see there for why that is not simply deferred
/// until the checks have run. Its own failures need no `keep`: dropping it on
/// the way out is exactly what a half-made destination deserves.
fn resolve_move_to(dir: Option<&String>) -> Result<(Option<PathBuf>, Conjured)> {
    let Some(dir) = dir else { return Ok((None, Conjured::default())) };

    let conjured = Conjured { dirs: missing_ancestors(Path::new(dir)) };

    std::fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create the --move-to folder {}", dir))?;
    let resolved = std::fs::canonicalize(dir)
        .with_context(|| format!("Failed to resolve the --move-to folder {}", dir))?;
    if !resolved.is_dir() {
        anyhow::bail!("--move-to {} is not a folder.", dir);
    }
    Ok((Some(resolved), conjured))
}

/// The format an `--output` path implies when `--format` is silent.
///
/// Anything unrecognised is text, which is what the writer's fallthrough arm
/// did before this was written down: `-o report.xml` gets the console listing,
/// not a refusal, because the extension is a hint about a file the user names
/// and not a declaration this tool gets to reject.
fn format_from_extension(path: &Path) -> Format {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "csv" => Format::Csv,
        "json" => Format::Json,
        _ => Format::Txt,
    }
}

/// Turn `--output` and `--format` into the single value the writer reads, and
/// refuse a destination that plainly cannot be written before any work starts.
///
/// The one place either flag is interpreted. `--format` alone is refused rather
/// than assumed to mean stdout: with no `-o` a run writes no report at all, and
/// quietly turning it into one that does would be deciding on the user's behalf
/// where the fix is to name the destination.
///
/// The path check is the same reasoning `resolve_move_to` is built on, applied to the other
/// argument naming a path: a destination that cannot exist is a typo, and
/// finding a typo out after fingerprinting a library is finding it out several
/// hours too late. The report used to be the one path checked only at the point
/// of writing it, which is the last statement of the run -- so `-o
/// /nonexistant/report.csv` did the entire scan and then threw the results
/// away, and with `--delete` armed it did so *after* the files were gone.
///
/// Deliberately does not create anything that outlives the check. `--move-to`
/// names a folder the user wants files put in, so creating it is the request;
/// `-o` names a file, and conjuring parent directories for it is not. Nor does
/// it truncate the target -- an existing report stays intact until the new one
/// replaces it, and an interrupted run leaves no empty file behind.
///
/// Permissions are part of what it proves, because they were the one failure
/// left that a destructive run could not survive: the report is written as the
/// last statement of `output_results`, *after* the disposal pass, so `-o
/// ro/rep.csv --delete --permanent` into a directory it cannot write reported
/// "Permanently deleted 1 file(s)" and then lost the only record of which file
/// that was. Exit 2 and a line in the problem summary do not bring it back.
/// `ensure_writable` is what closes it, and it is worth the two syscalls at
/// start-up for the same reason the rest of this function is: an hour of
/// decoding is a bad time to find out where the report was going to go. None
/// of it applies to stdout, which is already open.
fn report_target_for(output: Option<&str>, format: Option<Format>) -> Result<Option<ReportTarget>> {
    let Some(out) = output else {
        if format.is_some() {
            anyhow::bail!(
                "--format says how to write the report, not where to put it. Add --output <FILE>, \
                 or -o - to write it to stdout."
            );
        }
        return Ok(None);
    };

    if out == "-" {
        return Ok(Some(ReportTarget {
            sink: Sink::Stdout,
            // Text is what the console prints, so a bare `-o -` is the listing
            // the user is already reading, on a stream they can pipe.
            format: format.unwrap_or(Format::Txt),
        }));
    }

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

    ensure_writable(path)?;

    Ok(Some(ReportTarget {
        sink: Sink::File(path.to_path_buf()),
        format: format.unwrap_or_else(|| format_from_extension(path)),
    }))
}

/// Refuse a report or log destination that is one of the videos this run reads.
///
/// Both are written over whatever is there -- the log is truncated at start-up,
/// the report renamed over its destination at the end -- and neither flag is a
/// disposal flag, so a report-only run promised to touch no video and
/// `vid-fp a.mp4 b.mkv -o b.mkv` replaced `b.mkv` with a 570-byte listing.
/// Asked before anything is opened, of the two things knowable then: a file the
/// command line names, and an existing file named like a video, by the default
/// list or by `-x`. A file named like nothing in particular that turns out to be
/// a video (only reachable under `-x '*'`) is caught for the report by
/// `report_replaces_an_input` once the run knows what it fingerprinted; the log
/// file is already open by then, which is the one gap left, and it needs a
/// video with no video extension to be named as the log.
fn refuse_to_overwrite_an_input(flag: &str, target: &str, args: &Args) -> Result<()> {
    if target == "-" {
        return Ok(());
    }
    let path = Path::new(target);
    if !std::fs::metadata(path).is_ok_and(|m| m.is_file()) {
        return Ok(());
    }
    let canonical = std::fs::canonicalize(path).ok();
    let named = args
        .include
        .iter()
        .filter(|p| p.as_str() != "-")
        .any(|p| canonical.is_some() && std::fs::canonicalize(p).ok() == canonical);
    let extension = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    let video_named = extension.is_some_and(|ext| {
        DEFAULT_EXTENSIONS.contains(&ext.as_str())
            || args.extensions.iter().any(|x| {
                let x = x.trim().to_ascii_lowercase();
                let x = x.trim_start_matches("*.").trim_start_matches('.');
                !x.starts_with('!') && x != "*" && x == ext
            })
    });
    if named || video_named {
        anyhow::bail!(
            "{} {} is {}, and writing there would replace it. Choose another file name.",
            flag,
            target,
            if named { "one of the files this run scans" } else { "an existing video file" }
        );
    }
    Ok(())
}

/// Whether the report would be written over a file this run fingerprinted,
/// which is then certainly a video. See `refuse_to_overwrite_an_input`.
fn report_replaces_an_input(target: Option<&ReportTarget>, fingerprints: &[VideoFingerprint]) -> Result<()> {
    let Some(ReportTarget { sink: Sink::File(path), .. }) = target else { return Ok(()) };
    let Ok(canonical) = std::fs::canonicalize(path) else { return Ok(()) };
    if let Some(fp) = fingerprints.iter().find(|f| Path::new(&f.path) == canonical) {
        anyhow::bail!(
            "--output {} is {}, a video this run fingerprinted, and writing the report there \
             would replace it. Nothing was touched; choose another file name.",
            path.display(),
            fp.path
        );
    }
    Ok(())
}

/// Prove the report can be written where it is going, without writing anything
/// that is still there afterwards.
///
/// One `open` answers it, and which of the two cases it lands in is the file
/// system's to decide rather than something to race on with a prior `exists()`:
///
/// * The file is not there yet, so `create_new` makes it -- which is exactly
///   the permission the real write needs, on exactly the name it will use --
///   and it is removed again immediately. Creating and keeping it would break
///   the promise above (an interrupted run leaves nothing behind) and would
///   also stand an empty file where a previous run's report used to be.
/// * The file is already there, so the create loses to `EEXIST` (POSIX checks
///   O_EXCL before the directory's write bit, so this arm is reached even in a
///   folder that would refuse a new file) and it is re-opened for writing
///   instead. Deliberately WITHOUT `truncate`: the old report is a document the
///   user may still be working from, and emptying it at start-up to prove the
///   run could have filled it is the opposite of the point. Note this is the
///   case where a read-only *folder* is fine and must stay allowed -- writing
///   over an existing file needs no permission on the directory holding it.
///
/// What no check can rule out is the permissions changing under a run that is
/// already going, or the disk filling up. Those are still late failures, and
/// `output_results` still records them as problems rather than propagating
/// them; the difference is that they are now accidents of timing rather than
/// the state the run started in.
fn ensure_writable(path: &Path) -> Result<()> {
    use std::fs::OpenOptions;

    let refuse = |e: std::io::Error| -> anyhow::Error {
        anyhow::anyhow!("--output {} cannot be written: {}.", path.display(), e)
    };

    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(file) => {
            drop(file);
            // Best effort: the run is going to write this path in a moment
            // whatever happens here, so a probe that cannot clean up after
            // itself is not a reason to refuse the run.
            let _ = std::fs::remove_file(path);
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // Only a regular file is opened to be asked. Opening a FIFO for
            // writing BLOCKS until a reader arrives, and then closing the probe
            // hands that reader an EOF -- so `mkfifo p; vid-fp lib -o p & cat p`
            // would have the check consume the reader the report was for. A
            // device or a socket is the same kind of thing: whether it can be
            // written is not a question that can be asked without doing it.
            // Those keep the behaviour they had before this check existed.
            match std::fs::metadata(path) {
                Ok(m) if !m.is_file() => Ok(()),
                _ => OpenOptions::new().write(true).open(path).map(drop).map_err(refuse),
            }
        }
        Err(e) => Err(refuse(e)),
    }
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

/// End a run that never reached the comparison, having written the report it
/// was asked for.
///
/// A scan that finds fewer than two videos has nothing to group, and both
/// routes to that used to return before `output_results` -- so `--output` was
/// silently not written and the run exited 0. The file it names is then an
/// EARLIER run's report, and the documented workflow is to hand exactly that
/// file back with `--from-report --delete`. A second scan that comes up short
/// (an unmounted share, a `--min-duration` that swallowed the lot, an
/// `--extensions` list that matches nothing any more) therefore replayed a plan
/// for a library that is no longer there: `dispose_one`'s size check spares
/// only the files whose bytes moved, and a file that is unchanged but no longer
/// has a duplicate beside it is removed against a measurement nothing in this
/// run made. It could take the last copy -- run 1 marks `b` DELETE against `a`,
/// `a` is moved away, run 2 sees one file and says nothing, and the replay
/// removes `b`.
///
/// So the rule is that a run which COMPLETES writes its report, whether or not
/// it found anything, exactly as a scan of two unrelated videos already does:
/// zero groups, and a report that says so. Nothing is special-cased about the
/// content -- `output_results` is asked for a run with no groups, which is what
/// keeps the three formats' empty bodies the same shape as their full ones and
/// what makes the overwrite visible ("Results saved to ...") rather than
/// silent. An interrupted run still writes nothing, and says so with exit 130.
///
/// The disposal is passed through rather than dropped so the summary and the
/// JSON `mode` describe the run that was asked for; with no groups there is
/// nothing for it to act on, and no confirmation is asked.
///
/// A scan that fell short -- an unresolved root, an unreadable folder -- writes
/// its empty report too, which is the OPPOSITE of what `prune_obstacle` decides
/// about the same run, and deliberately. The two are not the same trade: a
/// prune against an incomplete scan destroys hours of decode that only a
/// re-decode brings back, while an emptied report costs a second scan against a
/// warm cache. What the report has that the cache does not is the other
/// failure mode -- leaving it alone is what stands a deletion plan up in front
/// of a user as though this run had endorsed it. `-o` says where this run
/// writes, a successful scan would overwrite the file just the same, and the
/// overwrite is announced ("Results saved to ...") on a run that is already
/// exiting 2 and saying why.
fn conclude_without_comparing(
    report_target: Option<&ReportTarget>,
    disposal: Option<&Disposal>,
    args: &Args,
    (found, compared): (usize, usize),
    start_time: Instant,
    stats: &RunStats,
) -> Result<Outcome> {
    // The window's copy too: a scan that found nothing to compare is a result
    // to show, not a scan that left none.
    if let Some(path) = &args.window_result {
        found::Found::empty(found, compared).save(path)?;
    }
    export::output_results(
        &[],
        &[],
        &MatchIndex::new(Vec::new()),
        report_target,
        start_time.elapsed().as_secs(),
        args.priority,
        disposal,
        args.yes,
        stats,
    )?;

    Ok(Outcome::Completed)
}

/// `--from-report`: dispose of what a previous run's report marks DELETE, and
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
/// That bookkeeping is the only thing in this mode that touches the cache at
/// all, which is why `db` is optional: another run holding the cache is not a
/// reason to refuse a deletion this user has already reviewed. `main` makes
/// that call and explains it; this end of it records the entries it could not
/// drop and carries on.
///
/// Refusing to run without an arming flag is the one piece of validation clap
/// cannot express. It is not merely useless without one -- a report-only
/// `--from-report` would read a file, decide nothing, and print nothing that
/// the report does not already say.
fn run_from_report(
    args: &Args,
    db: Option<&Cache>,
    report_path: &str,
    stats: &RunStats,
) -> Result<Outcome> {
    let (move_to, mut conjured) = resolve_move_to(args.move_to.as_ref())?;

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
    // Past every refusal this mode has: a report it could not read is one of
    // them, and it leaves no more behind than a scan refused by the loop guard.
    conjured.keep();

    if !deleted_paths.is_empty() && forgets(&disposal) {
        match db {
            Some(db) => match db.forget(&deleted_paths) {
                Ok(forgotten) => {
                    log::debug!("Dropped {} cache entry(ies) for removed file(s).", forgotten)
                }
                Err(e) => {
                    log::error!(target: stats::COUNTED, "Failed to drop cache entries for removed files: {:#}", e);
                    stats.cache_purge_failed.record(format!("{:#}", e));
                }
            },
            // The cache was busy when this run started, and `main` chose the
            // deletions over the bookkeeping -- the reasoning is there. One
            // record for the whole batch, exactly as the failure arm above
            // does, because `Cache::forget` is one transaction either way, and
            // because a count with a single sample would render as "... and N
            // more (see the errors above)" pointing at errors nothing printed.
            None => stats.cache_purge_failed.record(format!(
                "{} entry(ies) left behind: the cache was in use by another vid-fp run",
                deleted_paths.len()
            )),
        }
    }

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    Ok(Outcome::Completed)
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
pub(crate) fn work_rate(per_sec: f64) -> String {
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

/// How wide the run actually gets to be, given what `--threads` asked for and
/// how many cores the machine has.
///
/// `0` means "one worker per core", which is what the flag has always
/// documented. Everything else is taken at face value up to the ceiling
/// `MAX_THREAD_OVERSUBSCRIPTION` and `MIN_THREAD_CEILING` set between them, and
/// clamped above it. Both of those carry the measurements that chose them.
///
/// This is the only numeric flag that is CLAMPED rather than refused, and the
/// difference is what a bad value can do. `-d`, `-p` and `--min-duration` change
/// what the run REPORTS, so a value that cannot mean anything has to stop the
/// run before it wastes an hour producing an answer nobody wants. A thread count
/// changes only how fast the same answer arrives -- that is not a guess, it is
/// why the count is deliberately kept out of the cache `Stamp`, and
/// `test_threaded_decode_is_bit_identical` holds the claim up over the whole
/// corpus. Aborting a scan over a performance knob would cost the user more than
/// running at a sane width and saying so out loud.
fn active_thread_count(requested: usize, cores: usize) -> usize {
    // `available_parallelism` cannot return 0 and the fallback is 1, but this
    // function is also the one place the invariant is cheap to state: a run with
    // no workers decodes nothing and finishes instantly, reporting no groups.
    let cores = cores.max(1);
    if requested == 0 {
        return cores;
    }
    let ceiling = cores
        .saturating_mul(MAX_THREAD_OVERSUBSCRIPTION)
        .max(MIN_THREAD_CEILING);
    requested.min(ceiling)
}

fn install_signal_handler() -> Result<()> {
    let hits = AtomicUsize::new(0);

    ctrlc::set_handler(move || {
        if hits.fetch_add(1, Ordering::SeqCst) == 0 {
            utils::request_shutdown();
            // Straight to stderr, not through the logger, so it still shows
            // under --quiet. The user pressed a key; they get an answer now.
            // Above the bar, which goes on drawing while the run winds down.
            progress::println("\nInterrupt received — saving finished fingerprints to cache.");
        } else {
            progress::clear_for_exit();
            eprintln!("\nSecond interrupt — quitting now.");
            std::process::exit(130);
        }
    })
    .context("Failed to install signal handler")
}

/// Why `--prune-cache` must not run this time, if it must not.
///
/// A prune keeps exactly the entries whose file is in front of the run and
/// removes every other one, so it is only ever as correct as the scan is
/// complete. A run that could not resolve a scan root, or could not read a
/// folder underneath one, is looking at a subset of the library and reads
/// everything it never reached as stale -- so a mistyped path with nothing
/// behind it used to empty the whole cache, and the run KNEW the path had
/// failed (it counts it, and exits 2) while pruning anyway. That is hours of
/// decode discarded by one keystroke. This is the rule `resolve_excludes`
/// already applies in the other direction: a path that will not resolve must
/// not silently change what the run acts on.
///
/// An empty file list is refused for the same reason and needs no failure to
/// get there, which is what makes it worth checking separately: a nested
/// library scanned without `-r`, or one whose containers are not in
/// `--extensions`, resolves cleanly, finds nothing, and would take the entire
/// cache with it.
///
/// `unresolved_excludes` is deliberately not here. An `--exclude` that will not
/// resolve excludes nothing, so the scan comes out WIDER than asked rather than
/// narrower -- the prune then removes less than it should, which costs a
/// re-decode of nothing.
fn prune_obstacle(stats: &RunStats, found: usize) -> Option<String> {
    let incomplete = [
        (stats.unresolved_includes.count(), "scan path(s) could not be resolved"),
        (stats.unwalkable.count(), "folder(s) could not be read"),
        (stats.unreadable.count(), "file(s) could not be read"),
    ];

    if let Some((n, what)) = incomplete.into_iter().find(|(n, _)| *n > 0) {
        return Some(format!("{} {}", n, what));
    }

    if found == 0 {
        return Some("the scan found no videos".to_string());
    }

    None
}

/// Whether `a` and `b` name the same file: the same inode when both exist,
/// otherwise the same absolute path once the folder holding each is resolved.
fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    if let (Ok(x), Ok(y)) = (std::fs::metadata(a), std::fs::metadata(b)) {
        return x.dev() == y.dev() && x.ino() == y.ino();
    }
    let resolved = |p: &Path| -> Option<PathBuf> {
        let abs = std::path::absolute(p).ok()?;
        let name = abs.file_name()?.to_owned();
        let parent = abs.parent()?;
        Some(std::fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf()).join(name))
    };
    matches!((resolved(a), resolved(b)), (Some(x), Some(y)) if x == y)
}

/// Refuse a run whose report, log and cache are not three different files.
///
/// Each of them replaces what is at its path: the report is renamed over it,
/// the log is truncated at start-up, and the cache is a database the other two
/// would write straight through. `-o cache.redb` turned the whole fingerprint
/// cache into a text report at the end of the run; `--log-file cache.redb`
/// emptied it before it was opened, so every file was decoded again; and
/// `-o x.txt --log-file x.txt` left the report where the log had been.
fn refuse_shared_outputs(args: &Args) -> Result<()> {
    let cache = cache::cache_file_for(args.cache.as_deref());
    let output = args.output.as_deref().filter(|o| *o != "-").map(Path::new);
    let log = args.log_file.as_deref().map(Path::new);
    let mut named: Vec<(&str, &Path)> = vec![("--cache", cache.as_path())];
    named.extend(output.map(|p| ("--output", p)));
    named.extend(log.map(|p| ("--log-file", p)));
    for i in 0..named.len() {
        for j in i + 1..named.len() {
            let ((fa, a), (fb, b)) = (named[i], named[j]);
            if same_file(a, b) {
                let what = if fa == "--cache" { "the fingerprint cache".to_string() } else { format!("{} {}", fa, a.display()) };
                anyhow::bail!(
                    "{} {} is {}, and writing both to one file would destroy one of them. \
                     Choose another file name.",
                    fb,
                    b.display(),
                    what
                );
            }
        }
    }
    Ok(())
}

/// How loud each destination is, which is two answers rather than one.
///
/// `filter` is what `env_logger` is given, and it stands in front of BOTH
/// destinations; `console` is applied a second time inside the format closure,
/// after the file has been written. They differ only under `-q --log-file`.
///
/// One filter for both is what made `-q --log-file` lose almost everything it
/// was asked for. `-q` is documented as suppressing TERMINAL output and
/// `--log-file` as recording every line "including the ones the terminal
/// hides", so the two flags pull the same knob in opposite directions and the
/// front filter is the wrong place to settle it: `info!` never reached the
/// closure at all, and the file the closure writes first got only whatever
/// `error!` had left. Measured on a four-file scan, the log went from 0 lines
/// to the same 24 a verbose run records. Failures survived either way, which is
/// what made it look harmless -- but the flag exists to be grepped, and a run
/// that finds nothing wrong wrote nothing at all.
///
/// So the filter admits whatever EITHER destination will read, and the console
/// narrows again downstream. Note the fallback is `Off` and not `Error`: with
/// no log file there is nothing to widen for, and the console's own level is
/// then the whole answer, so a run without `--log-file` is filtered exactly as
/// it was before.
#[derive(Debug, PartialEq, Eq)]
struct Verbosity {
    console: log::LevelFilter,
    filter: log::LevelFilter,
}

fn verbosity(quiet: bool, has_log_file: bool) -> Verbosity {
    let console = if quiet { log::LevelFilter::Error } else { log::LevelFilter::Info };
    let file = if has_log_file { log::LevelFilter::Info } else { log::LevelFilter::Off };
    Verbosity { console, filter: console.max(file) }
}

/// The command line, from `std::env::args`: what `vid-fp` runs.
pub fn main() -> Result<()> {
    main_with(Args::parse())
}

/// The scan behind the desktop app's Scan button, in a process of its own
/// (`vid-fp-gui --worker RESULT ARGV...`): exactly the run `argv` would be on
/// the command line, except for three things that only the window reads.
///
/// - Every log line goes to stdout as one JSON object (`{"l": line}`), and
///   where the progress bars stand as another (`{"p": fraction, "m": what}`);
///   see `progress`. stderr carries what it always does: the problem summary,
///   and the error of a run that failed.
/// - The per-group listing is not printed: the window shows the groups.
/// - The groups, with each file's KEEP/DELETE/REVIEW, are saved to `result`
///   (`found::Found`) before anything else is written, so a run that fails at
///   the very end still leaves them to show.
///
/// Nothing about what is compared or decided moves: the verdicts come from
/// `export::verdicts`, the same function the report's labels do.
pub fn worker_main(result: &Path, argv: impl IntoIterator<Item = std::ffi::OsString>) -> Result<()> {
    let mut args = Args::try_parse_from(argv).map_err(|e| anyhow::anyhow!("{}", e.to_string().trim_end()))?;
    args.window_result = Some(result.to_path_buf());
    main_with(args)
}

/// The command line `vid-fp` reads, for the window to build its own from
/// and to check what it built.
pub fn command() -> clap::Command {
    Args::command()
}

/// Whether `argv` (program name first) is a command line `vid-fp` accepts:
/// clap's own rules and the range checks `run` makes before any work. `Err`
/// is what the command line would say.
pub fn check_args(argv: impl IntoIterator<Item = std::ffi::OsString>) -> std::result::Result<(), String> {
    let args = Args::try_parse_from(argv).map_err(|e| e.to_string())?;
    check_ranges(&args).map_err(|e| e.to_string())
}

fn main_with(args: Args) -> Result<()> {
    let start_time = Instant::now();

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
    let Verbosity { console: console_level, filter: log_level } =
        verbosity(args.quiet, args.log_file.is_some());

    // Progress bars are drawn on stderr rather than logged, so the filter above
    // says nothing about them. See `utils::console_is_verbose`.
    if args.quiet {
        utils::silence_console();
    }

    // Before the log file is opened, which truncates it: see the function.
    if let Some(path) = &args.log_file {
        refuse_to_overwrite_an_input("--log-file", path, &args)?;
    }
    if let Some(path) = &args.output {
        refuse_to_overwrite_an_input("--output", path, &args)?;
    }
    refuse_shared_outputs(&args)?;

    // Opened before the logger exists, so a bad path is reported by the ordinary
    // error path rather than by a logger that is not installed yet. Truncating
    // rather than appending: the file describes THIS run, and the alternative is
    // a file that silently grows by a quarter of a million lines a scan.
    let log_file = match &args.log_file {
        Some(path) => Some(Mutex::new(
            std::fs::File::create(path)
                .with_context(|| format!("Could not open the log file {}", path))?,
        )),
        None => None,
    };

    // Under the desktop app the log and the bars are the window's, said on
    // stdout as JSON (`progress`), and the listing is its results page.
    let window = args.window_result.is_some();
    if window {
        progress::speak();
        utils::for_window();
    }

    env_logger::Builder::new()
        .filter_level(log_level)
        .format(move |buf, record| {
            let line = if record.level() == log::Level::Error {
                format!("Error: {}", record.args())
            } else {
                format!("{}", record.args()) // Clean output for CLI tools
            };

            // The file gets everything the filter admits, uncapped and before
            // any decision about the console -- that is what it is for, and it
            // is the only place the unabridged list of failures exists. The
            // filter is widened to Info whenever this file exists precisely so
            // that "everything" survives `--quiet`.
            //
            // Deliberately unbuffered: one write(2) per line. A `BufWriter`
            // would be free to add and there is nowhere honest to flush it --
            // the logger is a `'static` global that is never dropped, so the
            // tail of the file would survive only a return through `main`, and
            // not a SIGKILL, a panic or the process being killed while it
            // hangs. That last case is the one this flag is for: the 80-minute
            // spin `MAX_CONSECUTIVE_DEMUX_ERRORS` now prevents was diagnosed
            // from the last line written before the kill, which a buffer would
            // have swallowed. The cost is a debugging flag's alone and it is
            // small: measured at ~3 us a line (4,000 refusals add 0.01-0.02 s
            // to a 0.04 s warm run), so the widest case there is -- a `-x '*'`
            // scan of a 248k-file home directory -- pays well under a second.
            if let Some(file) = &log_file {
                let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
                let _ = writeln!(file, "{}", line);
            }

            // Everything below is the console's alone, and the first question
            // is whether the console is listening at all. `-q` is answered
            // here rather than by the filter above so that the file keeps the
            // line either way; see the two levels at the call site.
            if record.level() > console_level {
                return Ok(());
            }

            // A failure the run is going to account for is not also announced
            // as it happens: `stats::print_summary` reports it at the end, with
            // a count and up to `MAX_SAMPLES` examples, and saying it twice is
            // how a `-x '*'` scan buried its own results under 226,863 lines.
            // See `stats::COUNTED` for what is and is not tagged this way.
            if record.target() == stats::COUNTED {
                return Ok(());
            }

            // The desktop app's worker: the line goes to the window, on
            // stdout, beside where the progress bar stands. Through the same
            // lock `progress` writes under, so the two never interleave.
            if window {
                progress::log_line(&line);
                return Ok(());
            }

            // Above the progress bar rather than through it.
            let _ = buf;
            progress::println(&line);
            Ok(())
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
    // --- How wide the run is --------------------------------------------------
    // `-t` is the one numeric flag whose bound is enforced HERE rather than in
    // `run`, and the reason is that it is spent before `run` is reached: rayon's
    // `build_global` allocates every thread in the pool eagerly, so an absurd
    // value has already cost its 19.7 seconds -- or wedged the process -- by the
    // time any validation `run` could do would look at it.
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let active_threads = active_thread_count(args.threads, cores);
    if args.threads > active_threads {
        // Named in terms of memory rather than of cores, because that is what
        // the ceiling is actually protecting: peak RSS is close to linear in
        // this number and throughput is flat above one worker per core. A user
        // who reads "more than this machine has cores" concludes the tool
        // miscounted their cores; one who reads this can tell whether the
        // setting was worth asking for.
        log::warn!(
            "--threads {} capped at {} on this machine ({} cores); memory grows with this \
             setting, throughput does not.",
            args.threads,
            active_threads,
            cores
        );
    }

    // Built from the clamped figure, never the requested one, so the rayon pool
    // and the decoder thread budget can never disagree about how wide the run
    // is. Left at rayon's own default when `-t` is absent, which is already one
    // thread per core.
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(active_threads)
            .build_global()
            .context("Failed to configure global thread pool")?;
    }

    ffmpeg_next::init().context("Failed to initialize FFmpeg bindings.")?;
    ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

    let db_path = cache::resolve_cache_path(args.cache.as_deref())?;

    // `--from-report` is the one mode that can run without a cache, and a busy
    // cache is the one failure it can survive. That mode reads nothing out of
    // the cache: it only drops the entries of the files it removes, afterwards.
    // Refusing to replay a report the user has already reviewed -- for a piece
    // of bookkeeping -- would block a deletion for as long as some unrelated
    // scan takes, and the advice the locked message gives would be actively
    // wrong here, since a cache handed over with `--cache` is not the one
    // holding those entries: the run would delete the files and purge nothing.
    // So the purge is given up instead, and loudly: a warning before the
    // confirmation prompt, `cache_purge_failed` in the problem summary, exit 2,
    // and `--prune-cache` collects the entries whenever it next runs.
    //
    // `--clear-cache` is exempt. Emptying a cache is not bookkeeping -- it is
    // the thing that was asked for, and it cannot be given up quietly.
    let mut db = match Cache::open(&db_path) {
        Ok(db) => Some(db),
        Err(e) if e.locked && args.from_report.is_some() && !args.clear_cache => {
            log::warn!(
                "The fingerprint cache at {} is in use by another vid-fp run.\n\
                 Continuing without it, since a report replay does not read it -- but the \
                 files this run removes will keep their cached fingerprints until a later \
                 run with --prune-cache.",
                db_path.display()
            );
            None
        }
        Err(e) if e.damaged && args.clear_cache => {
            let aside = cache::set_aside_path(&db_path);
            std::fs::rename(&db_path, &aside).with_context(|| {
                format!("Could not move the damaged cache {} aside", db_path.display())
            })?;
            log::warn!(
                "The fingerprint cache at {} could not be read; it was moved to {} and a new \
                 one started.",
                db_path.display(),
                aside.display()
            );
            Some(Cache::open(&db_path).map_err(|e| e.reason)?)
        }
        Err(e) => return Err(e.reason),
    };

    if let Some(db) = db.as_mut() {
        // Handled here rather than inside `run` because compaction needs an
        // exclusive handle on the database, and because this is the only point
        // in the run where compacting is free. A clear drops every table, the
        // superseded ones included, so it needs no preparing first.
        if args.clear_cache {
            info!("Clearing all cache...");
            db.clear()?;
        } else {
            // Once, on the first run of this build: an older layout is carried
            // over and every dead table dropped. Nothing will ever look in those
            // again, and compacting right afterwards is the only thing that
            // returns their pages to the filesystem.
            let prepared = db.prepare().context("Failed to initialize the fingerprint cache")?;
            if prepared.rebuilt {
                info!("Removed the superseded fingerprint cache; fingerprints will be rebuilt once.");
            }
            if prepared.converted > 0 {
                info!("Converted {} cache entries to this build's layout.", prepared.converted);
            }
            if prepared.reclaimable {
                if let Err(e) = db.compact() {
                    log::error!("{:#}", e);
                }
            }
        }
    }

    let stats = RunStats::default();

    let outcome = run(&args, db.as_ref(), start_time, active_threads, &stats);

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
    // So the condition is what the prune REMOVED, not what the flags asked for.
    // The flags cannot answer it: `--prune-cache` is a request, and there are
    // three ways a run can decline it after taking one. With `--clear-cache`
    // beside it the prune is skipped outright (see `run`: after a clear there
    // is nothing stale, because there is nothing at all). A scan that fell
    // short refuses to prune and says so -- `prune_obstacle` -- and that run
    // then printed "Not pruning the cache: ..." and "Compacted the fingerprint
    // cache." one after the other, a full rewrite of every live page to reclaim
    // exactly nothing. And a prune with nothing stale in front of it removes
    // nothing either. `run` failing before it reaches the prune is a fourth,
    // since this line is above the `outcome?` that returns the error.
    //
    // All four are the same waste compaction was moved out of the tail of the
    // run to avoid: invisible on a small cache, a full rewrite plus an fsync on
    // a large one. A counted removal is the only thing that says there is dead
    // space, and it is the only thing that can be wrong in the direction that
    // costs anything -- a compaction skipped is space handed back by the next
    // prune, where a compaction taken is a rewrite the user waits for now.
    //
    // `db` is None only under `--from-report`, which cannot be given
    // `--prune-cache` at all, so this never silently skips a compaction the
    // user asked for.
    if !shutdown_requested() && stats.entries_pruned() > 0 {
        if let Some(db) = db.as_mut() {
            match db.compact() {
                Ok(true) => info!("Compacted the fingerprint cache."),
                Ok(false) => {}
                Err(e) => log::error!("{:#}", e),
            }
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


/// The numeric flags refused out of range, before any work: the checks clap
/// cannot express. Shared with `check_args`, which the window asks before it
/// starts a scan, so a value the window could send is never one only `run`
/// would have refused.
fn check_ranges(args: &Args) -> Result<()> {
    // Both of these are range tests rather than the `< 0.0` this used to be,
    // and the reason is NaN: every comparison against it is false whichever way
    // it is written, and clap parses `--min-duration=nan` without complaint. A
    // NaN floor would disable the gate silently; a NaN percentage is worse,
    // because `pct_a.max(pct_b) < min_match_percent` is then false for every
    // pair and the run accepts everything the index proposed -- which with
    // --delete armed can only ever ADD files to the DELETE set.
    //
    // A range test catches NaN, and only `--match-percent`'s catches infinity,
    // because only that one has an upper end. `(0.0..)` is a `RangeFrom` with no
    // upper bound at all, so `--min-duration=inf` -- and `--min-duration=1e400`,
    // which is the way a typo reaches it -- passed straight through: every file
    // with a measurable runtime is then shorter than the floor, the whole
    // library is skipped, and the run exits 0 having compared nothing. This was
    // the one numeric knob whose nonsense value produced a clean exit rather
    // than a message, which is the exact failure the sampling knobs below were
    // given `is_finite` to prevent. It is stated the same way as those two, so
    // all four flags now refuse the same set of values.
    //
    // Nothing in the cache has to be retired for it. A skipped file writes
    // `Refusal::TooShort(duration)` -- the measured runtime, not the verdict, by
    // the invariant on that variant -- so the entries an `inf` run left behind
    // are exactly the ones a sane run wants, and the next run re-decides from
    // them without opening anything.
    if !args.min_duration.is_finite() || args.min_duration < 0.0 {
        anyhow::bail!("--min-duration must be zero or more seconds (0 turns it off).");
    }
    // The upper bound is not a matter of taste. `match_overlap` clamps every
    // coverage figure to 1.0, so a floor above 100% is one no pair can ever
    // clear: the run would fingerprint the whole library and be structurally
    // incapable of reporting a single group. 0 stays legal and means "report
    // every pair with any measured overlap at all" -- it turns the gate off, it
    // does not turn the measurement off, and `measure_pair` refuses a pair that
    // shares nothing whatever this says.
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
    // through a whole range that could not mean anything either. Chance is the
    // honest edge of the range, and it is comfortably past the edge of the
    // useful one (`--help` says 2 to 12).
    //
    // MAX_HAMMING_DISTANCE itself stays legal, the mirror of `-p 0`: it is a
    // sensitivity control, and refusing its last rung would be arbitrary where
    // refusing everything past chance is not.
    let widest_useful = max_hamming_distance();
    if args.hamming_distance > widest_useful {
        anyhow::bail!(
            "--hamming-distance must be between 0 and {} (a frame hash is {} bits, and two \
             unrelated frames already differ in about {} of them -- a higher tolerance \
             matches everything against everything).",
            widest_useful,
            HASH_BITS,
            widest_useful
        );
    }

    // The sampling knobs, refused on the same terms and for a reason the three
    // above do not have. `-d`, `-p` and `--min-duration` are read by
    // comparisons, so an out-of-range value shows up as a report nobody can
    // explain; these two are read by `effective_interval`, which asks
    // `kf_interval > 0.0` and `min_kf_samples > 0.0` -- so a negative, an
    // infinity or a NaN is not out of range at all, it is silently the DEFAULT.
    // `--keyframe-interval=-5` decodes every keyframe of the library and says
    // "Using keyframe interval" nowhere, because the line that would have said
    // it is gated on the same test; the run then costs what a full-rate scan
    // costs and reports what a full-rate scan reports, with nothing anywhere to
    // connect either to the flag. 0 stays legal in both and is documented: it
    // turns the interval off, and it removes the floor.
    //
    // `Stamp::same_sampling` and `same_floor` still treat every one of those
    // values as "off", and must keep doing so: caches written by builds that
    // accepted them are still out there, and an entry written with a negative
    // interval holds exactly the fingerprint a default run wants.
    if !args.kf_interval.is_finite() || args.kf_interval < 0.0 {
        anyhow::bail!(
            "--keyframe-interval must be zero or more seconds (0 decodes every keyframe)."
        );
    }
    if !args.min_kf_samples.is_finite() || args.min_kf_samples < 0.0 {
        anyhow::bail!(
            "--min-keyframes must be zero or more frames (0 removes the floor and leaves \
             --keyframe-interval to stand on its own)."
        );
    }
    Ok(())
}

fn run(
    args: &Args,
    db: Option<&Cache>,
    start_time: Instant,
    active_threads: usize,
    stats: &RunStats,
) -> Result<Outcome> {
    // Before anything that reads the filesystem: this mode's whole premise is
    // that the decisions were made by an earlier run and are not being retaken.
    if let Some(report_path) = &args.from_report {
        return run_from_report(args, db, report_path, stats);
    }

    // `main` only ever hands over a `None` for `--from-report`, which has
    // already returned by here. A scan reads the cache before every decode and
    // writes it after, so there is nothing for it to degrade to.
    let db = db.context("A scan cannot run without a fingerprint cache")?;

    let max_hamming = args.hamming_distance;
    let min_match_pct = args.match_percent / 100.0;
    let min_duration = args.min_duration;

    check_ranges(args)?;

    let (move_to, mut conjured) = resolve_move_to(args.move_to.as_ref())?;
    let report_target = report_target_for(args.output.as_deref(), args.format)?;

    // `--list` is the one thing an `--against` run prints to stdout, which is
    // what lets it be piped; a report there as well would be two documents
    // run together.
    if !args.list.is_empty() && report_target.as_ref().is_some_and(|t| matches!(t.sink, Sink::Stdout)) {
        anyhow::bail!("--list prints to stdout, so the report cannot go there too; give -o a file.");
    }
    // The window is not offered this mode, and its results would carry a
    // scan's labels rather than the answers.
    if !args.against.is_empty() && args.window_result.is_some() {
        anyhow::bail!("--against is not available in the desktop app.");
    }

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
    // One bar for the whole run (`progress`), up until the report's listing.
    let _bar = progress::start(active_threads);
    progress::learn(|k| k.window = args.window_result.is_some());
    let mut library = match sources::collect(
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
        sources::Scan::Complete(library) => library,
        sources::Scan::Interrupted => return Ok(Outcome::Interrupted),
    };

    // `--against`: the library is walked with the scan's own options, so the
    // two sides are found by the same rules, and its files join the run as
    // files that are answered against rather than answered.
    let held = if args.against.is_empty() {
        None
    } else {
        match sources::collect(
            &sources::Sources {
                include: &args.against,
                exclude: &args.exclude,
                from_file: None,
                null_separated: args.null,
                extensions: &args.extensions,
                recursive: args.recursive,
                follow_symlinks: args.follow_symlinks,
            },
            stats,
        )? {
            sources::Scan::Complete(held) => Some(held),
            sources::Scan::Interrupted => return Ok(Outcome::Interrupted),
        }
    };

    // Moving a duplicate into a folder that is itself being scanned puts the
    // file straight back into the next run's input, where it is still a
    // duplicate of the copy that was kept -- and the run after that would move
    // it again, one directory deeper each time. Nothing is lost, but the one
    // thing this mode is for (getting files OUT of the library, reversibly) is
    // quietly not happening, so it is worth stopping over rather than warning
    // about.
    if let (Some(dest), Some(held)) = (&move_to, &held) {
        // A repeat moved into the library would be a second copy of what the
        // library already holds, and the next run would find it there.
        if let Some(reached) = held.walk_reaches(dest) {
            anyhow::bail!(
                "The --move-to folder {} is in the library: it is {}. Choose a destination \
                 outside both the scanned folders and the library.",
                dest.display(),
                reached
            );
        }
    }

    if let Some(dest) = &move_to {
        if let Some(scanned) = library.walk_reaches(dest) {
            anyhow::bail!(
                "The --move-to folder {} would be scanned again by this run: it is {}, so the \
                 moved files would be picked up next time. Exclude it with -e {}, or choose a \
                 destination outside the scanned folders.",
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
        if library.files.iter().any(|f| Path::new(&f.path).starts_with(dest)) {
            info!(
                "Note: {} is above the folder(s) being scanned. The moved files land in a \
                 separate subtree under it, so this run is unaffected -- but exclude it with \
                 -e if you ever scan {} itself recursively.",
                dest.display(),
                dest.display()
            );
        }
    }

    // Nothing left that can turn this run away over where it was told to put
    // things, so a destination it had to create is the user's from here.
    conjured.keep();

    // What to call the things that came back. Under `-x '*'` the walk applied no
    // extension filter, so calling them video files is a claim the run has not
    // checked and, on the folder that flag exists for, usually a false one --
    // "Found 229112 video files" over a home directory, of which a handful were
    // video. Every other setting has already turned away anything not named like
    // a video, so there the old wording was the accurate one and it is kept.
    let found_noun = if library.any_extension { "files" } else { "video files" };

    // Every question about the REQUEST has been answered; from here on the run
    // works from what it found.
    let mut video_files = std::mem::take(&mut library.files);

    // Which files are the library's. A file both walks reached goes to the
    // side whose folder holds it more closely, the scan's on a tie: scanning
    // an import folder inside the library answers the import folder, and
    // scanning a folder that holds the library answers everything but it.
    let held_paths: HashSet<String> = match held {
        None => HashSet::new(),
        Some(held) => {
            let closeness = |side: &sources::Library, path: &str| {
                side.closeness(Path::new(path)).unwrap_or(usize::MAX)
            };
            let scanned: HashSet<&str> = video_files.iter().map(|f| f.path.as_str()).collect();
            let held_paths: HashSet<String> = held
                .files
                .iter()
                .filter(|f| {
                    !scanned.contains(f.path.as_str())
                        || closeness(&held, &f.path) > closeness(&library, &f.path)
                })
                .map(|f| f.path.clone())
                .collect();
            // The scan's copy of a file the library took is dropped, and the
            // library's added, so every path is in the run once.
            video_files.retain(|f| !held_paths.contains(&f.path));
            let answering = video_files.len();
            video_files.extend(held.files.into_iter().filter(|f| held_paths.contains(&f.path)));
            info!(
                "Answering {} scanned file(s) against a library of {}.",
                answering,
                held_paths.len()
            );
            held_paths
        }
    };
    let answering = !args.against.is_empty();

    // A prune is measured against what the scan FOUND, so it is only ever as
    // correct as the scan was complete -- and a run that fell short knows it.
    let pruning = args.prune_cache && !args.clear_cache;
    let obstacle = if pruning { prune_obstacle(stats, video_files.len()) } else { None };
    // The prune itself runs inside the scan, after the content pass: see
    // `scan::fingerprint_all`.
    let prune_now = pruning && obstacle.is_none();

    if let Some(reason) = obstacle {
        log::warn!("Not pruning the cache: {}, so this run cannot tell what is stale.", reason);
        stats.cache_prune_skipped.record(reason);
    }

    if video_files.is_empty() {
        info!("No {} found.", found_noun);
        // Still this run's report: see `conclude_without_comparing` for why an
        // earlier run's is not allowed to stand in for it.
        let disposal = disposal_for(move_to, args.delete, args.permanent);
        return conclude_without_comparing(
            report_target.as_ref(),
            disposal.as_ref(),
            args,
            (0, 0),
            start_time,
            stats,
        );
    }

    let total_videos = video_files.len();
    let settings = scan::Settings {
        sampling: cache::Sampling { kf_interval: args.kf_interval, min_kf_samples: args.min_kf_samples },
        floor: Floor(min_duration),
        threads: active_threads,
        prune: prune_now,
        noun: found_noun,
    };
    let Some(scan::Scanned { mut fingerprints, identical: identical_paths }) =
        scan::fingerprint_all(db, video_files, &settings, stats)?
    else {
        return Ok(Outcome::Interrupted);
    };

    // Before anything is written or disposed of, on either ending below.
    report_replaces_an_input(report_target.as_ref(), &fingerprints)?;

    let n = fingerprints.len();
    if n < 2 && !answering {
        info!("Not enough valid videos to compare.");
        // The elapsed time used to be formatted again here so this ending would
        // say it too; the report's summary carries it, in the same format, for
        // every other ending.
        let disposal = disposal_for(move_to, args.delete, args.permanent);
        return conclude_without_comparing(
            report_target.as_ref(),
            disposal.as_ref(),
            args,
            (total_videos, n),
            start_time,
            stats,
        );
    }

    info!("\nFingerprinting complete. Cross-analyzing {} videos...", n);

    // Samples closer together than the comparison can tell apart are folded
    // into one before anything is measured (see `compare::thin_samples`).
    fingerprints.iter_mut().for_each(compare::thin_samples);

    // Which fingerprints are the library's, by their place in the list.
    let library_side: Vec<bool> = fingerprints.iter().map(|f| held_paths.contains(&f.path)).collect();

    // What the comparison is in hash pairs: every pair of the run's hashes,
    // or under `--against` the product of the two sides'.
    progress::learn(|k| {
        let (mut held, mut scanned) = (0.0, 0.0);
        for (fp, &lib) in fingerprints.iter().zip(&library_side) {
            *(if lib { &mut held } else { &mut scanned }) += fp.valid_hashes.len() as f64;
        }
        k.hash_pairs = Some(if answering { held * scanned } else { scanned * scanned / 2.0 });
    });

    let matches = if answering {
        compare::find_matches_across(&fingerprints, &library_side, max_hamming, min_match_pct, min_duration)
    } else {
        find_all_matches(&fingerprints, max_hamming, min_match_pct, min_duration)
    };

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    // The video stream's bytes, for the ranking to read in place of the file's
    // (`fingerprint::Measured::video_bytes`): one header read of each file that
    // matched something, since only those are ever ranked.
    {
        use rayon::prelude::*;
        let began = Instant::now();
        let mut ranked: Vec<usize> = matches.iter().flat_map(|m| [m.a, m.b]).collect();
        ranked.sort_unstable();
        ranked.dedup();
        let measured: Vec<(usize, u64)> = ranked
            .par_iter()
            .map(|&i| (i, fingerprint::video_stream_bytes(&fingerprints[i].path)))
            .collect();
        for (i, bytes) in measured {
            fingerprints[i].measured_mut().video_bytes = bytes;
        }
        log::debug!("Read the video stream size of {} matched file(s) in {:.2?}.", ranked.len(), began.elapsed());
    }

    if answering {
        info!("Answering each scanned file against the library...");
    } else {
        info!("Grouping duplicate clusters...");
    }
    progress::learn(|k| k.matches = Some(matches.len()));
    progress::begin(progress::Stage::Grouping);

    // Clustering only needs to know which pairs matched -- it decides membership
    // and nothing else. The coverage figures travel separately to the report,
    // where they are the only thing that tells a genuine re-encode apart from a
    // clip that happened to clear --match-percent.
    // `--against` is not grouped: each scanned file is answered against the
    // library files it matched, and its labels come with the answer.
    let (final_groups, answered) = if answering {
        let mut answered = against::answer(&fingerprints, &matches, &library_side, args.priority);
        (std::mem::take(&mut answered.groups), Some(answered))
    } else {
        let edges: Vec<(usize, usize)> = matches.iter().map(|m| (m.a, m.b)).collect();
        (clustering::find_duplicate_groups(n, edges, &fingerprints), None)
    };

    // Consumes the Vec, so the pair list is not kept alive alongside the index.
    // Which compared files are the same bytes, by their place in the list the
    // comparison read. A set can lose members on the way here -- a copy too
    // short for `--min-duration`, one whose decode failed -- and a set of one
    // says nothing.
    let identical: Vec<Vec<usize>> = {
        let at: HashMap<&str, usize> =
            fingerprints.iter().enumerate().map(|(i, f)| (f.path.as_str(), i)).collect();
        identical_paths
            .iter()
            .map(|set| set.iter().filter_map(|p| at.get(p.as_str()).copied()).collect::<Vec<_>>())
            .filter(|set| set.len() > 1)
            .collect()
    };
    let matches = MatchIndex::new(matches).with_identical(n, &identical);

    if shutdown_requested() {
        return Ok(Outcome::Interrupted);
    }

    // The single place that turns flags into intent. Report-only runs produce
    // None, and export.rs cannot touch a file without one of these.
    let disposal = disposal_for(move_to, args.delete, args.permanent);
    announce(disposal.as_ref());

    // The desktop app's copy, before anything else is written: a report that
    // then fails to save still leaves the window its groups to show.
    // `--list` writes paths for a script to act on, so a path that cannot be
    // written that way stops the run here, before anything is disposed of.
    let listed: Vec<&str> = match &answered {
        Some(a) => a
            .answers
            .iter()
            .filter(|(_, answer)| args.list.contains(answer))
            .map(|&(i, _)| fingerprints[i].path.as_str())
            .collect(),
        None => Vec::new(),
    };
    if !args.null {
        if let Some(bad) = listed.iter().find(|p| p.contains('\n')) {
            anyhow::bail!(
                "{:?} holds a line break, so --list would print it as two paths. Add -0 to \
                 separate the list with NUL bytes instead.",
                bad
            );
        }
    }

    if let Some(a) = &answered {
        let count = |want| a.answers.iter().filter(|(_, got)| *got == want).count();
        info!(
            "Against the library: {} repeat(s), {} better, {} to review, {} new.",
            count(against::Answer::Repeat),
            count(against::Answer::Better),
            count(against::Answer::Review),
            count(against::Answer::New)
        );
    }

    if let Some(path) = &args.window_result {
        found::Found::of(&final_groups, &fingerprints, &matches, total_videos, max_hamming).save(path)?;
    }

    let deleted_paths = export::output_results(
        &final_groups,
        &fingerprints,
        &matches,
        report_target.as_ref(),
        start_time.elapsed().as_secs(),
        match answered {
            Some(a) => export::Ranking::Pinned(a.verdicts),
            None => export::Ranking::By(args.priority),
        },
        disposal.as_ref(),
        args.yes,
        stats,
    )?;

    if !listed.is_empty() {
        let end = if args.null { '\0' } else { '\n' };
        let mut out = std::io::stdout().lock();
        for path in &listed {
            // A reader closing early is how a pipeline ends (`| head`).
            if write!(out, "{}{}", path, end).is_err() {
                break;
            }
        }
        let _ = out.flush();
    }

    // Those files are gone for good, so the fingerprints filed under their
    // paths describe nothing at all (see `forgets` for why only for good).
    // Deliberately BEFORE the interrupt check below: the removals have already
    // happened, and stopping is no reason to leave the cache claiming
    // otherwise. It is one small transaction either way.
    if !deleted_paths.is_empty() && disposal.as_ref().is_some_and(forgets) {
        match db.forget(&deleted_paths) {
            Ok(forgotten) => {
                log::debug!("Dropped {} cache entry(ies) for removed file(s).", forgotten)
            }
            Err(e) => {
                log::error!(target: stats::COUNTED, "Failed to drop cache entries for removed files: {:#}", e);
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
    use cache::{Known, Refusal, Stamp, Write as CacheWrite};

    fn store_refusals(db: &Cache, refused: &[(String, Stamp, Refusal)]) -> Result<()> {
        let writes: Vec<CacheWrite> =
            refused.iter().map(|(p, s, r)| CacheWrite::Refusal(p, *s, r.clone())).collect();
        db.write(&writes)
    }

    fn dest() -> PathBuf {
        PathBuf::from("/mnt/scratch/dupes")
    }

    /// `-q` is about the terminal and `--log-file` is about the file, and one
    /// filter in front of both let the first one silence the second.
    ///
    /// The case that matters is the last one: with a log file open, an `info!`
    /// has to survive the filter to reach the closure that writes it, so
    /// `filter` must be Info even though the console is only taking errors.
    /// That is the whole of the bug -- `-q --log-file` wrote 0 lines of the 24
    /// a verbose run recorded -- and it is the one assertion here that fails on
    /// the old code.
    #[test]
    fn test_a_log_file_is_written_at_full_volume_however_quiet_the_terminal_is() {
        use log::LevelFilter::{Error, Info};

        // Without a log file nothing is widened, so both flags read exactly as
        // they did: the console's level IS the filter.
        assert_eq!(verbosity(false, false), Verbosity { console: Info, filter: Info });
        assert_eq!(verbosity(true, false), Verbosity { console: Error, filter: Error });

        // A log file on a loud run changes nothing either -- the filter was
        // already admitting everything the file wants.
        assert_eq!(verbosity(false, true), Verbosity { console: Info, filter: Info });

        // And the pair someone actually reaches for.
        assert_eq!(verbosity(true, true), Verbosity { console: Error, filter: Info });

        // The console half is what keeps `-q` a promise about the terminal:
        // widening the filter must not widen what is printed.
        for has_file in [false, true] {
            assert_eq!(verbosity(true, has_file).console, Error);
            assert_eq!(verbosity(false, has_file).console, Info);
        }

        // The file's own level is Off rather than Error when there is no
        // file, so `console.max(file)` can never raise the filter on its own.
        assert!(verbosity(true, false).filter < verbosity(true, true).filter);
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

    #[test]
    fn test_zero_threads_means_one_per_core() {
        for cores in [1, 2, 8, 128] {
            assert_eq!(active_thread_count(0, cores), cores);
        }
        // The `unwrap_or(1)` in `main` should be the only fallback, but a run
        // with no workers at all decodes nothing and reports nothing, so the
        // floor is stated here too.
        assert_eq!(active_thread_count(0, 0), 1);
    }

    #[test]
    fn test_a_reasonable_thread_request_is_taken_at_face_value() {
        // Including the oversubscription a network mount or a slow disk makes
        // worth asking for: more workers than cores is not a mistake, and up to
        // the ceiling it is honoured exactly.
        for requested in [1, 4, 8, 16, 32] {
            assert_eq!(
                active_thread_count(requested, 8),
                requested,
                "-t {} is inside 8 x {} and must be honoured",
                requested,
                MAX_THREAD_OVERSUBSCRIPTION
            );
        }
    }

    #[test]
    fn test_an_absurd_thread_request_is_clamped_rather_than_obeyed() {
        // The failure this exists for: rayon's `build_global` allocates the
        // whole pool eagerly, so `-t 1000` on a two-file library used to spend
        // 19.7 s building threads it would use two of, and `-t 20000` failed
        // with EAGAIN and then hung. Neither number can reach the pool now.
        assert_eq!(active_thread_count(1000, 8), 32);
        assert_eq!(active_thread_count(20_000, 8), 32);
        assert_eq!(active_thread_count(usize::MAX, 8), 32);
    }

    #[test]
    fn test_the_ceiling_scales_with_the_machine_rather_than_being_a_flat_number() {
        // A flat ceiling is wrong in both directions -- a hard cap on a large
        // machine, and wild oversubscription on a small one -- and the memory
        // this bounds is spent per worker, so it has to track the machine.
        for cores in [1, 2, 4, 8, 16, 64, 128] {
            let ceiling = active_thread_count(usize::MAX, cores);
            assert!(
                ceiling >= cores,
                "{} cores must be reachable, not itself clamped",
                cores
            );
            assert_eq!(
                ceiling,
                (cores * MAX_THREAD_OVERSUBSCRIPTION).max(MIN_THREAD_CEILING)
            );
        }

        // The floor is a portability allowance, not a rung of the per-core rule:
        // a one-core VM reading its videos off a network mount has a useful
        // worker count set by the storage's latency, which has nothing to do
        // with its cores. It must not bind on any ordinary machine.
        assert_eq!(active_thread_count(usize::MAX, 1), MIN_THREAD_CEILING);
        assert_eq!(active_thread_count(usize::MAX, 8), 32, "the floor is slack at 8 cores");
    }

    #[test]
    fn test_the_ceiling_never_overflows_on_a_machine_it_cannot_imagine() {
        // `cores * 4` is a multiplication on a number the OS supplies, so the
        // saturating form is load-bearing rather than decorative.
        assert!(active_thread_count(usize::MAX, usize::MAX) > 0);
    }

    pub(crate) fn stamp(mtime: i64, size: u64) -> Stamp {
        Stamp {
            mtime,
            mtime_nsec: 0,
            size,
            kf_interval: 0.0,
            min_kf_samples: 4.0,
        }
    }


    pub(crate) fn mock_fp(path: &str) -> VideoFingerprint {
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
            mirrored_hashes: Vec::new(),
            detail_hashes: Vec::new(),
            mirrored_detail_hashes: Vec::new(),
            reframed: Vec::new(),
            picture: 0.0,
            measured: None,
        }
    }

    /// A cache of its own, inside a directory that cleans itself up.
    pub(crate) fn temp_db(dir: &tempfile::TempDir) -> Cache {
        Cache::create(&dir.path().join("fingerprints.redb"))
    }

    pub(crate) fn cache_store_one(db: &Cache, path: &str, stamp: Stamp, fp: &VideoFingerprint) -> Result<()> {
        db.write(&[CacheWrite::Fingerprint(path, stamp, fp)])
    }

    pub(crate) fn cache_lookup(db: &Cache, path: &str, now: &Stamp) -> Option<VideoFingerprint> {
        match db.view().lookup(path, now) {
            Known::Fingerprint(fp) => Some(*fp),
            _ => None,
        }
    }

    fn refusal_lookup(db: &Cache, path: &str, now: &Stamp) -> Option<Refusal> {
        match db.view().lookup(path, now) {
            Known::Refused(r) => Some(r),
            _ => None,
        }
    }

    /// Every record's verdict, read whatever stamp it was made under, so a test
    /// does not have to rebuild the `Stamp` of a file `run` stat'ed.
    pub(crate) fn verdicts(db: &Cache) -> Vec<(String, Known)> {
        db.view().verdicts()
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
        cache_store_one(&db, &doomed, s, &mock_fp(&doomed)).unwrap();
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
            run(&args, Some(&db), Instant::now(), 1, &stats),
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
    fn test_a_report_run_without_a_cache_still_deletes_and_reports_the_lost_purge() {
        // The deletions were reviewed by a human before this mode was ever
        // invoked, so a cache some unrelated scan happens to be holding must
        // not block them -- see `main`, which is where that is decided. What it
        // costs is the purge, and the cost has to be audible: the entries stay
        // until --prune-cache collects them, and a run that skipped them
        // silently would exit 0 with nothing said.
        let dir = tempfile::tempdir().unwrap();
        let (doomed, report) = armed_report(&dir);

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
            run(&args, None, Instant::now(), 1, &stats),
            Ok(Outcome::Completed)
        ));
        assert!(!Path::new(&doomed).exists(), "the deletion happens either way");

        assert_eq!(stats.cache_purge_failed.count(), 1, "one record for the batch");
        assert!(stats.had_problems(), "so the run exits 2 rather than reading as a clean one");
        let sample = &stats.cache_purge_failed.samples()[0];
        assert!(sample.contains("1 entry"), "and says how much was left behind: {sample}");
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

        let err = run(&args, Some(&db), Instant::now(), 1, &stats)
            .expect_err("no disposal was ever constructed")
            .to_string();

        assert!(err.contains("--delete"), "{}", err);
        assert!(err.contains("--move-to"), "{}", err);
        assert!(Path::new(&doomed).exists(), "and nothing was touched");
    }

    /// A cache holding one entry for a file that is not on disk anywhere, which
    /// is exactly what `--prune-cache` exists to collect -- and exactly what a
    /// prune measured against a scan that failed would collect by mistake.
    const ORPHAN: &str = "/videos/long_gone.mkv";

    fn cache_with_an_orphan(dir: &tempfile::TempDir) -> Cache {
        let db = temp_db(dir);
        cache_store_one(&db, ORPHAN, stamp(1_700_000_000, 12_345), &mock_fp(ORPHAN)).unwrap();
        db
    }

    pub(crate) fn cached_paths(db: &Cache) -> Vec<String> {
        verdicts(db).into_iter().filter(|(_, k)| matches!(k, Known::Fingerprint(_))).map(|(p, _)| p).collect()
    }

    fn refused_paths(db: &Cache) -> Vec<String> {
        refusals(db).into_iter().map(|(p, _)| p).collect()
    }

    /// Every refusal in the cache, with the verdict it holds.
    ///
    fn refusals(db: &Cache) -> Vec<(String, Refusal)> {
        verdicts(db)
            .into_iter()
            .filter_map(|(p, k)| match k {
                Known::Refused(r) => Some((p, r)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_a_video_that_would_not_decode_is_not_called_not_video() {
        // Both halves of the same table, and only one of them is about a file
        // that is not video. The reported case was a 1080p HEVC episode: it
        // opened, libavcodec would not build a picture out of it, the run said
        // so in its own words -- and then the warm run's headline counted it
        // beside the object files and told the user it was not video.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();

        // Nothing in it a demuxer recognises: the probe gate's own verdict.
        let junk = library.join("notes.rlib");
        std::fs::write(&junk, "not a video at all\n".repeat(200)).unwrap();

        // A real MP4 as far as the header goes, with the picture data cut off:
        // it opens, it has a video stream, and no frame ever decodes.
        let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture.push("tests/fixtures/test_video.mp4");
        let whole = std::fs::read(&fixture).expect("the fixture video is part of the tree");
        let broken = library.join("broken.mp4");
        std::fs::write(&broken, &whole[..2000]).unwrap();

        let args = Args::parse_from(["vid-fp", &library.to_string_lossy(), "-x", "*"]);
        let stats = RunStats::default();
        run(&args, Some(&db), Instant::now(), 1, &stats).unwrap();

        let mut remembered = refusals(&db);
        remembered.sort_by(|a, b| a.0.cmp(&b.0));
        let kinds: Vec<(String, bool)> = remembered
            .iter()
            .map(|(path, verdict)| {
                (
                    PathBuf::from(path).file_name().unwrap().to_string_lossy().to_string(),
                    verdict.is_not_media(),
                )
            })
            .collect();
        assert_eq!(
            kinds,
            vec![("broken.mp4".to_string(), false), ("notes.rlib".to_string(), true)],
            "the truncated video opened and failed; the object file never opened at all"
        );
    }

    /// A file that is not video is reported by whichever pass discovers it, and
    /// the run's account of itself does not depend on which one that was.
    ///
    /// This is the whole outside-visible surface of moving the discovery into
    /// the weighing pass: the same count, the same bucket, the same exit code,
    /// one open instead of two. The saving itself cannot be asserted from here
    /// -- nothing counts `avformat_open_input` calls -- so what this pins is
    /// that the file is still reported at all, which is the thing dropping it
    /// from the decode queue could plausibly have lost.
    #[test]
    fn test_a_file_that_is_not_video_is_still_reported_by_the_run_that_skips_it() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        // A folder of its own: `-x '*'` takes everything, and the cache this
        // test just created lives in the temp directory too.
        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        std::fs::write(library.join("notes.mkv"), b"this file is not a video at all").unwrap();

        let args =
            Args::parse_from(["vid-fp", &library.to_string_lossy(), "-x", "*"]);
        let stats = RunStats::default();
        assert!(matches!(
            run(&args, Some(&db), Instant::now(), 1, &stats),
            Ok(Outcome::Completed)
        ));

        assert_eq!(stats.fingerprint_failed.count(), 1, "the file is named as a problem");
        assert!(stats.had_problems(), "so the run still exits 2");
        assert!(cached_paths(&db).is_empty(), "and nothing about it is cached");
    }

    /// The same property through `run`, because the argument for it is about a
    /// route rather than about two cache calls: the file has to really reach the
    /// weighing pass on the run that decodes it, which is the step the reasoning
    /// this fixes denied could happen at all.
    #[test]
    fn test_lowering_the_duration_floor_leaves_one_entry_rather_than_two() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture.push("tests/fixtures/test_video.mp4");
        let scanned = library.join("clip.mp4");
        std::fs::copy(&fixture, &scanned).expect("the fixture video is part of the tree");
        let scanned = scanned.to_string_lossy().to_string();
        let root = library.to_string_lossy().to_string();

        let run_with = |floor: &str| {
            let args = Args::parse_from(["vid-fp", &root, "--min-duration", floor]);
            let stats = RunStats::default();
            run(&args, Some(&db), Instant::now(), 1, &stats).expect("the scan itself is fine");
        };

        // Far longer than the fixture, so the weighing pass measures it and
        // remembers the runtime.
        run_with("100");
        assert!(cached_paths(&db).is_empty(), "nothing was decoded");
        assert_eq!(refused_paths(&db), vec![scanned.clone()], "the runtime is on record");

        // Off, so the same runtime now clears the floor and the file decodes.
        run_with("0");
        assert_eq!(cached_paths(&db), vec![scanned.clone()], "and is fingerprinted");
        assert!(
            refused_paths(&db).is_empty(),
            "with the refusal it superseded taken away rather than left beside it"
        );
    }

    /// The same invariant from the other side: a REFUSAL landing on a path a
    /// fingerprint is already filed under. An edited file whose new runtime
    /// falls under `--min-duration` is refused and never decoded, so nothing
    /// else would ever write over its old fingerprint. When the two lived in
    /// separate tables this left one path in both; one record per path makes
    /// it impossible, and this drives the route that used to reach it.
    #[test]
    fn test_a_refusal_supersedes_the_fingerprint_it_lands_on() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture.push("tests/fixtures/test_video.mp4");
        let scanned = library.join("clip.mp4");
        std::fs::copy(&fixture, &scanned).expect("the fixture video is part of the tree");
        let root = library.to_string_lossy().to_string();
        let path = scanned.to_string_lossy().to_string();

        let run_with = |floor: &str| {
            let args = Args::parse_from(["vid-fp", &root, "--min-duration", floor]);
            let stats = RunStats::default();
            run(&args, Some(&db), Instant::now(), 1, &stats).expect("the scan itself is fine");
        };

        // Run one decodes it, so the path means a fingerprint.
        run_with("0");
        assert_eq!(cached_paths(&db), vec![path.clone()], "the fingerprint is on record");

        // The file is edited, which is the whole of what it takes: the stamp no
        // longer matches, so run two is a cache MISS and the file reaches the
        // weighing pass again.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().append(true).open(&scanned).unwrap();
            f.write_all(&[0u8; 64]).unwrap();
        }

        // Run two measures the header and files the runtime, and never decodes.
        run_with("100");
        assert_eq!(refused_paths(&db), vec![path.clone()], "the runtime is on record");
        assert!(
            cached_paths(&db).is_empty(),
            "and the fingerprint it superseded went with it, rather than both standing at once"
        );

        // The same removal for the other shape of refusal, which the weighing
        // pass reaches by the same route -- a file edited into something that is
        // not video -- and which no fixture can be driven through `run` to
        // produce. Nothing here is verdict-specific, and this is what says so.
        let elsewhere = "/videos/was-a-video.mkv";
        let written = stamp(1_700_000_000, 12_345);
        cache_store_one(&db, elsewhere, written, &mock_fp(elsewhere)).unwrap();
        store_refusals(
            &db,
            &[(
                elsewhere.to_string(),
                stamp(1_700_000_001, 999),
                Refusal::NotMedia(fingerprint::NotMedia { bytes: 2048, score: 0 }),
            )],
        )
        .unwrap();
        assert!(
            refused_paths(&db).iter().any(|p| p == elsewhere),
            "the verdict is on record"
        );
        assert!(
            !cached_paths(&db).iter().any(|p| p == elsewhere),
            "and the fingerprint it replaced is not"
        );
    }

    /// A prune collects refusals as well as fingerprints. Refusals are what
    /// grows without bound -- one per file that is not video -- so a prune that
    /// only swept fingerprints would leave the larger half behind.
    #[test]
    fn test_pruning_collects_refusals_as_well_as_fingerprints() {
        let dir = tempfile::tempdir().unwrap();
        let db = cache_with_an_orphan(&dir);
        store_refusals(
            &db,
            &[(
                ORPHAN.to_string(),
                stamp(1_700_000_000, 12_345),
                Refusal::NotMedia(fingerprint::NotMedia { bytes: 2048, score: 0 }),
            )],
        )
        .unwrap();

        let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture.push("tests/fixtures/test_video.mp4");
        let scanned = dir.path().join("clip.mp4");
        std::fs::copy(&fixture, &scanned).expect("the fixture video is part of the tree");

        let stats = RunStats::default();
        assert!(matches!(
            run(&args_over(&dir, &["--prune-cache"]), Some(&db), Instant::now(), 1, &stats),
            Ok(Outcome::Completed)
        ));

        assert!(!cached_paths(&db).iter().any(|p| p == ORPHAN), "the fingerprint is gone");
        assert!(
            refusal_lookup(&db, ORPHAN, &stamp(1_700_000_000, 12_345)).is_none(),
            "and so is the refusal, which nothing else would ever overwrite"
        );
    }

    /// A remembered refusal has to read exactly like the one that discovered it.
    ///
    /// The two travel by completely different routes -- one is an `anyhow` chain
    /// built as the file is examined, the other is regenerated from two numbers
    /// in the cache -- so nothing but a test keeps them the same sentence. They
    /// were not: the discovering run wrapped the verdict in "Failed to open
    /// video file", which the replay had no way to add and which describes an
    /// open that never happened, so the same file read one way on Monday and
    /// another on Tuesday. See `fingerprint::open_video`.
    #[test]
    fn test_a_remembered_refusal_reads_exactly_like_a_fresh_one() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let path = library.join("notes.rlib");
        std::fs::write(&path, "not a video at all\n".repeat(200)).unwrap();

        let args = Args::parse_from(["vid-fp", &library.to_string_lossy(), "-x", "*"]);

        let discovered = RunStats::default();
        run(&args, Some(&db), Instant::now(), 1, &discovered).unwrap();
        let remembered = RunStats::default();
        run(&args, Some(&db), Instant::now(), 1, &remembered).unwrap();

        assert_eq!(discovered.fingerprint_failed.count(), 1, "found the hard way");
        assert_eq!(remembered.fingerprint_failed.count(), 1, "and then the cheap way");
        assert_eq!(
            discovered.fingerprint_failed.samples(),
            remembered.fingerprint_failed.samples(),
            "the same finding about the same file has to be the same sentence"
        );
    }

    #[test]
    fn test_a_scan_path_that_would_not_resolve_prunes_nothing() {
        // The keystroke this guard is about: `vid-fp vidz --prune-cache` with
        // `vids` meant. The scan resolves nothing, finds nothing, and every
        // entry in the cache reads as stale -- so the run used to answer a
        // typo by discarding hours of decode, having already recorded that the
        // path failed and that it would be exiting 2 for it.
        let dir = tempfile::tempdir().unwrap();
        let db = cache_with_an_orphan(&dir);

        let missing = dir.path().join("vidz").to_string_lossy().to_string();
        let args = Args::parse_from(["vid-fp", &missing, "--prune-cache"]);
        let stats = RunStats::default();

        assert!(matches!(
            run(&args, Some(&db), Instant::now(), 1, &stats),
            Ok(Outcome::Completed)
        ));

        assert_eq!(cached_paths(&db), vec![ORPHAN.to_string()], "the cache is untouched");
        assert_eq!(stats.cache_prune_skipped.count(), 1, "and the refusal is said out loud");
        assert!(stats.had_problems(), "so the run exits 2 rather than reading as a clean one");
    }

    #[test]
    fn test_a_scan_that_found_nothing_prunes_nothing() {
        // The same damage with nothing whatever going wrong: an existing folder
        // holding no videos this run can see -- a nested library scanned without
        // -r, or one whose containers are outside --extensions. It resolves, it
        // walks, it reports no problem, and pruning against it would still take
        // the whole cache.
        let dir = tempfile::tempdir().unwrap();
        let db = cache_with_an_orphan(&dir);

        let stats = RunStats::default();
        assert!(matches!(
            run(&args_over(&dir, &["--prune-cache"]), Some(&db), Instant::now(), 1, &stats),
            Ok(Outcome::Completed)
        ));

        assert_eq!(cached_paths(&db), vec![ORPHAN.to_string()], "the cache is untouched");
        assert_eq!(stats.cache_prune_skipped.count(), 1);
    }

    #[test]
    fn test_a_scan_that_saw_the_whole_library_still_prunes() {
        // The other half of the guard, and the reason it is written as a list of
        // obstacles rather than "prune only when everything is perfect": a
        // complete scan that finds files must still collect the entries those
        // files do not account for, which is the whole point of the flag.
        let dir = tempfile::tempdir().unwrap();
        let db = cache_with_an_orphan(&dir);

        let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture.push("tests/fixtures/test_video.mp4");
        let scanned = dir.path().join("clip.mp4");
        std::fs::copy(&fixture, &scanned).expect("the fixture video is part of the tree");

        let stats = RunStats::default();
        assert!(matches!(
            run(&args_over(&dir, &["--prune-cache"]), Some(&db), Instant::now(), 1, &stats),
            Ok(Outcome::Completed)
        ));

        let left = cached_paths(&db);
        assert!(!left.iter().any(|p| p == ORPHAN), "the orphan is gone: {:?}", left);
        assert_eq!(stats.cache_prune_skipped.count(), 0, "nothing stood in the way");
    }

    /// The compaction `main` runs afterwards is decided by what the prune
    /// removed, never by the flag that asked for one.
    ///
    /// A compaction copies every live page into a fresh file and fsyncs it, so
    /// on a large cache it is the most expensive thing a warm run does. It used
    /// to be gated on `--prune-cache` alone, which is a request rather than an
    /// outcome: a run that fell short printed "Not pruning the cache: 1 scan
    /// path(s) could not be resolved" and "Compacted the fingerprint cache."
    /// one after the other, having removed nothing and rewritten everything.
    /// A prune with nothing stale in front of it is the same trade.
    #[test]
    fn test_a_prune_that_removed_nothing_leaves_nothing_to_compact() {
        let dir = tempfile::tempdir().unwrap();
        let db = cache_with_an_orphan(&dir);

        let missing = dir.path().join("vidz").to_string_lossy().to_string();
        let refused = RunStats::default();
        run(
            &Args::parse_from(["vid-fp", &missing, "--prune-cache"]),
            Some(&db),
            Instant::now(),
            1,
            &refused,
        )
        .unwrap();

        assert_eq!(refused.cache_prune_skipped.count(), 1, "the prune was declined");
        assert_eq!(refused.entries_pruned(), 0, "so there is no dead space to reclaim");

        // The same cache, scanned completely: now the orphan really goes, and
        // the pages it held are what the compaction is for.
        let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture.push("tests/fixtures/test_video.mp4");
        std::fs::copy(&fixture, dir.path().join("clip.mp4"))
            .expect("the fixture video is part of the tree");

        let pruned = RunStats::default();
        run(&args_over(&dir, &["--prune-cache"]), Some(&db), Instant::now(), 1, &pruned).unwrap();

        assert_eq!(pruned.entries_pruned(), 1, "one orphan removed, so one thing to reclaim");

        // And a second pass over the same complete scan: the prune runs, finds
        // the cache already accounted for, and removes nothing.
        let again = RunStats::default();
        run(&args_over(&dir, &["--prune-cache"]), Some(&db), Instant::now(), 1, &again).unwrap();

        assert_eq!(again.cache_prune_skipped.count(), 0, "nothing stood in the way this time");
        assert_eq!(again.entries_pruned(), 0, "there was simply nothing stale left");
    }

    /// A library of two identical clips and the report that plans their merge.
    pub(crate) fn library_with_a_pair(dir: &tempfile::TempDir) -> (PathBuf, PathBuf) {
        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();

        let mut fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture.push("tests/fixtures/test_video.mp4");
        for name in ["a.mp4", "b.mp4"] {
            std::fs::copy(&fixture, library.join(name))
                .expect("the fixture video is part of the tree");
        }

        (library, dir.path().join("plan.csv"))
    }

    /// A report-only run touches no video, so neither the report nor the log
    /// may be written over one. `vid-fp a.mp4 b.mkv -o b.mkv` replaced `b.mkv`
    /// with the listing, and `--log-file` truncated its target at start-up.
    #[test]
    fn test_a_report_or_a_log_is_never_written_over_a_video() {
        let dir = tempfile::tempdir().unwrap();
        let (library, _) = library_with_a_pair(&dir);
        let (a, b) = (library.join("a.mp4"), library.join("b.mp4"));
        let (a, b) = (a.to_string_lossy().to_string(), b.to_string_lossy().to_string());
        let size = std::fs::metadata(&b).unwrap().len();

        // Named on the command line, by either flag.
        let args = Args::parse_from(["vid-fp", &a, &b, "-o", &b]);
        assert!(refuse_to_overwrite_an_input("--output", &b, &args).is_err());
        assert!(refuse_to_overwrite_an_input("--log-file", &b, &args).is_err());

        // Not scanned, but an existing file named like a video, by the default
        // list or by -x.
        let elsewhere = dir.path().join("keep.mkv");
        std::fs::copy(&a, &elsewhere).unwrap();
        let elsewhere = elsewhere.to_string_lossy().to_string();
        assert!(refuse_to_overwrite_an_input("--output", &elsewhere, &args).is_err());
        let odd = dir.path().join("clip.dat");
        std::fs::copy(&a, &odd).unwrap();
        let odd = odd.to_string_lossy().to_string();
        let with_dat = Args::parse_from(["vid-fp", &a, "-x", "*.DAT,!mkv"]);
        assert!(refuse_to_overwrite_an_input("--output", &odd, &with_dat).is_err());
        assert!(refuse_to_overwrite_an_input("--output", &odd, &args).is_ok(), "not a video name without -x");

        // The ordinary cases stay allowed: an earlier report, a new file, stdout.
        let report = dir.path().join("report.csv");
        std::fs::write(&report, "old").unwrap();
        assert!(refuse_to_overwrite_an_input("--output", &report.to_string_lossy(), &args).is_ok());
        let new = dir.path().join("new.mp4").to_string_lossy().to_string();
        assert!(refuse_to_overwrite_an_input("--output", &new, &args).is_ok(), "nothing there to lose");
        assert!(refuse_to_overwrite_an_input("--output", "-", &args).is_ok());

        // A video with no video name, reached under -x '*': caught once the
        // run knows what it fingerprinted, before anything is written.
        let hidden = library.join("c.bin");
        std::fs::copy(&a, &hidden).unwrap();
        let db = temp_db(&dir);
        let err = run_error(&dir, &db, &["-r", "-x", "*", "-o", &hidden.to_string_lossy()]);
        assert!(err.contains("a video this run fingerprinted"), "{err}");
        assert_eq!(std::fs::metadata(&hidden).unwrap().len(), size, "and it is intact");
        assert_eq!(std::fs::metadata(&b).unwrap().len(), size);
    }

    /// A file sent to the Trash or moved aside keeps its cache entry, so
    /// putting it back is a cache hit rather than a re-decode; only a
    /// permanent delete forgets.
    #[test]
    fn test_a_file_moved_aside_keeps_its_fingerprint_and_a_deleted_one_does_not() {
        for (flags, kept) in [(&["--move-to"][..], true), (&["--delete", "--permanent"][..], false)] {
            let dir = tempfile::tempdir().unwrap();
            let (library, _) = library_with_a_pair(&dir);
            let db = temp_db(&dir);
            let dest = dir.path().join("aside");
            let mut extra: Vec<String> = flags.iter().map(|s| s.to_string()).collect();
            if kept {
                extra.push(dest.to_string_lossy().to_string());
            }
            let root = library.to_string_lossy().to_string();
            let mut argv = vec!["vid-fp".to_string(), root];
            argv.extend(extra);
            let stats = RunStats::default();
            assert!(matches!(
                run(&Args::parse_from(argv), Some(&db), Instant::now(), 1, &stats),
                Ok(Outcome::Completed)
            ));
            let gone: Vec<PathBuf> = ["a.mp4", "b.mp4"].iter().map(|n| library.join(n)).filter(|p| !p.exists()).collect();
            assert_eq!(gone.len(), 1, "one copy of the pair went ({flags:?})");
            let path = std::fs::canonicalize(&library).unwrap().join(gone[0].file_name().unwrap());
            let remembered = cached_paths(&db).contains(&path.to_string_lossy().to_string());
            assert_eq!(remembered, kept, "{flags:?}");
        }
    }

    #[test]
    fn test_a_scan_with_nothing_left_to_compare_replaces_the_plan_it_was_asked_for() {
        // The hazard is the documented two-step workflow: scan with -o, review
        // the report, hand it back with --from-report --delete. A second scan
        // that comes up short used to write no report at all and exit 0, so the
        // file the user then replayed was the FIRST run's plan for a library
        // that no longer exists. `dispose_one`'s size check spares only files
        // whose bytes moved; a file that is unchanged but no longer has a
        // duplicate beside it is removed against a measurement nothing in the
        // second run made, and here that is the last copy of it.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let (library, report) = library_with_a_pair(&dir);

        let args = Args::parse_from([
            "vid-fp",
            &library.to_string_lossy(),
            "-o",
            &report.to_string_lossy(),
        ]);

        run(&args, Some(&db), Instant::now(), 1, &RunStats::default()).unwrap();
        let plan = std::fs::read_to_string(&report).unwrap();
        assert!(plan.contains("DELETE"), "the first run leaves something to replay:\n{}", plan);

        // The file that won is taken out from under the plan, which leaves the
        // one it condemned as the only copy there is.
        std::fs::remove_file(library.join("a.mp4")).unwrap();

        let stats = RunStats::default();
        assert!(matches!(
            run(&args, Some(&db), Instant::now(), 1, &stats),
            Ok(Outcome::Completed)
        ));
        assert!(!stats.had_problems(), "a library of one file is not a failed run");

        let refreshed = std::fs::read_to_string(&report).unwrap();
        assert!(
            !refreshed.contains("DELETE"),
            "the second run found no duplicate, so its report may not name one:\n{}",
            refreshed
        );
        assert!(
            refreshed.contains("full_path"),
            "and it is still a report, readable by --from-report:\n{}",
            refreshed
        );
    }

    #[test]
    fn test_a_scan_that_finds_no_videos_at_all_replaces_the_plan_too() {
        // The other route to the same silence, and the one an unmounted share
        // takes: the run ends before it has fingerprinted anything rather than
        // before it has compared anything. Both used to return ahead of
        // `output_results`.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let (library, report) = library_with_a_pair(&dir);

        let args = Args::parse_from([
            "vid-fp",
            &library.to_string_lossy(),
            "-o",
            &report.to_string_lossy(),
        ]);
        run(&args, Some(&db), Instant::now(), 1, &RunStats::default()).unwrap();
        assert!(std::fs::read_to_string(&report).unwrap().contains("DELETE"));

        for name in ["a.mp4", "b.mp4"] {
            std::fs::remove_file(library.join(name)).unwrap();
        }

        run(&args, Some(&db), Instant::now(), 1, &RunStats::default()).unwrap();

        let refreshed = std::fs::read_to_string(&report).unwrap();
        assert!(
            !refreshed.contains("DELETE"),
            "an empty library is not a reason to leave yesterday's deletions standing:\n{}",
            refreshed
        );
    }

    #[test]
    fn test_only_the_failures_that_narrow_a_scan_stop_a_prune() {
        // An --exclude that will not resolve excludes nothing, so the scan comes
        // out WIDER than asked and the prune removes less than it could. That
        // costs nothing and must not block the flag; every failure that hides
        // files must.
        let wider = RunStats::default();
        wider.unresolved_excludes.record("/typo: No such file or directory");
        assert_eq!(prune_obstacle(&wider, 3), None);

        let narrower = RunStats::default();
        narrower.unwalkable.record("/library/private: Permission denied");
        assert!(prune_obstacle(&narrower, 3).is_some(), "a folder that could not be read");

        let unreadable = RunStats::default();
        unreadable.unreadable.record("/library/odd\u{fffd}name.mp4: filename is not valid UTF-8");
        assert!(prune_obstacle(&unreadable, 3).is_some(), "a file that could not be read");

        assert!(prune_obstacle(&RunStats::default(), 0).is_some(), "and an empty scan");
        assert_eq!(prune_obstacle(&RunStats::default(), 1), None, "but a clean one prunes");
    }

    /// Everything the run needs to reach the argument checks: a real scan root
    /// so nothing earlier can fail for an unrelated reason.
    fn args_over(dir: &tempfile::TempDir, extra: &[&str]) -> Args {
        let root = dir.path().to_string_lossy().to_string();
        let mut argv = vec!["vid-fp".to_string(), root];
        argv.extend(extra.iter().map(|s| s.to_string()));
        Args::parse_from(argv)
    }

    fn run_error(dir: &tempfile::TempDir, db: &Cache, extra: &[&str]) -> String {
        let stats = RunStats::default();
        run(&args_over(dir, extra), Some(db), Instant::now(), 1, &stats)
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
                run(&args_over(&dir, &[ok]), Some(&db), Instant::now(), 1, &stats).is_ok(),
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
        // one group holding the library.
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
                run(&args_over(&dir, &[ok]), Some(&db), Instant::now(), 1, &stats).is_ok(),
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
        //
        // Infinity is the other half, and it is the half a `RangeFrom` cannot
        // see: `(0.0..)` has no upper end, so `inf` passed and every file in the
        // library was then shorter than the floor -- a run that skipped
        // everything, compared nothing and exited 0. `1e400` is in the list
        // because that is how a typo reaches it; it parses to `inf` rather than
        // being refused as unreadable, so nothing before this check objects.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        for bad in [
            "--min-duration=-5",
            "--min-duration=nan",
            "--min-duration=inf",
            "--min-duration=-inf",
            "--min-duration=1e400",
        ] {
            let err = run_error(&dir, &db, &[bad]);
            assert!(err.contains("--min-duration"), "{}: {}", bad, err);
        }

        // 0 is documented as the setting that turns the gate off, and a real
        // floor is the whole point of the flag. Neither may be caught by the
        // widened test.
        for ok in ["--min-duration=0", "--min-duration=30"] {
            let stats = RunStats::default();
            assert!(
                run(&args_over(&dir, &[ok]), Some(&db), Instant::now(), 1, &stats).is_ok(),
                "{} is a setting, not a typo",
                ok
            );
        }
    }

    #[test]
    fn test_a_negative_or_unparseable_sampling_knob_is_refused() {
        // The two flags where "out of range" and "the default" are the same
        // thing. `effective_interval` asks `kf_interval > 0.0` and
        // `min_kf_samples > 0.0`, so a negative or a NaN does not misbehave --
        // it turns the flag off, silently, and a mistyped --keyframe-interval
        // then decodes every keyframe in the library while the line that would
        // have announced the interval is gated on the same test and says
        // nothing.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        for bad in ["--keyframe-interval=-5", "--keyframe-interval=nan", "--keyframe-interval=inf"]
        {
            let err = run_error(&dir, &db, &[bad]);
            assert!(err.contains("--keyframe-interval"), "{}: {}", bad, err);
        }
        for bad in ["--min-keyframes=-1", "--min-keyframes=nan", "--min-keyframes=inf"] {
            let err = run_error(&dir, &db, &[bad]);
            assert!(err.contains("--min-keyframes"), "{}: {}", bad, err);
        }

        // 0 is a documented setting in both and has to stay one: it turns the
        // interval off, and it removes the floor so the interval stands alone.
        for ok in ["--keyframe-interval=0", "--min-keyframes=0", "--keyframe-interval=5"] {
            let stats = RunStats::default();
            assert!(
                run(&args_over(&dir, &[ok]), Some(&db), Instant::now(), 1, &stats).is_ok(),
                "{} is a setting, not a typo",
                ok
            );
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
        // get wrong in the direction that refuses a perfectly good run. Given a
        // name of its own because the check now touches the file system, and a
        // fixed one would collide with whatever else the suite is doing in the
        // working directory.
        assert!(report_target_for(None, None).unwrap().is_none());
        let bare = format!("vid-fp-test-report-{}.csv", std::process::id());
        assert!(report_target_for(Some(&bare), None).unwrap().is_some());
        // And the probe took nothing with it and left nothing behind: an
        // interrupted run must not leave an empty report standing where the
        // real one was going to go.
        assert!(!Path::new(&bare).exists(), "the write check left {} behind", bare);

        let dir = tempfile::tempdir().unwrap();
        let inside = dir.path().join("report.csv").to_string_lossy().to_string();
        assert!(report_target_for(Some(&inside), None).unwrap().is_some());
        assert!(!Path::new(&inside).exists());
    }

    #[test]
    fn test_a_report_that_cannot_be_written_is_refused_before_the_run_not_after_it() {
        // The failure this closes: the report is written after the disposal
        // pass, so a destination that refuses it costs the user the record of
        // an irreversible run rather than costing them the run.
        let dir = tempfile::tempdir().unwrap();
        let ro = dir.path().join("ro");
        std::fs::create_dir(&ro).unwrap();
        std::fs::set_permissions(&ro, std::os::unix::fs::PermissionsExt::from_mode(0o500))
            .unwrap();

        let target = ro.join("rep.csv").to_string_lossy().to_string();
        let err = format!("{:#}", report_target_for(Some(&target), None).unwrap_err());
        assert!(err.contains(&target), "{}", err);
        assert!(err.contains("cannot be written"), "{}", err);

        // Same folder, but the report already exists and is writable: writing
        // over a file needs no permission on the directory holding it, so this
        // run is fine and refusing it would be a regression of its own.
        std::fs::set_permissions(&ro, std::os::unix::fs::PermissionsExt::from_mode(0o700))
            .unwrap();
        let existing = ro.join("old.csv");
        std::fs::write(&existing, b"an earlier run\n").unwrap();
        std::fs::set_permissions(&ro, std::os::unix::fs::PermissionsExt::from_mode(0o500))
            .unwrap();

        let as_str = existing.to_string_lossy().to_string();
        assert!(report_target_for(Some(&as_str), None).unwrap().is_some());
        // And the check did not empty it on the way past. The new report
        // replaces the old one when it is written, not before.
        assert_eq!(std::fs::read(&existing).unwrap(), b"an earlier run\n");

        // A file nothing can write is refused whatever the folder says.
        std::fs::set_permissions(&ro, std::os::unix::fs::PermissionsExt::from_mode(0o700))
            .unwrap();
        std::fs::set_permissions(&existing, std::os::unix::fs::PermissionsExt::from_mode(0o400))
            .unwrap();
        let err = format!("{:#}", report_target_for(Some(&as_str), None).unwrap_err());
        assert!(err.contains("cannot be written"), "{}", err);
    }

    #[test]
    fn test_a_move_to_folder_a_refused_run_made_is_not_left_behind() {
        // `vid-fp lib -r --move-to lib/dupes`: the loop guard is right to refuse
        // it, and the folder was already there by the time it did. What that
        // costs is the advice the refusal gives -- the destination is now INSIDE
        // the library, so the second attempt at the same command has an empty
        // directory of this tool's making to clean up first.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        let dupes = dir.path().join("dupes");
        let err = run_error(&dir, &db, &["-r", "--move-to", &dupes.to_string_lossy()]);
        assert!(err.contains("would be scanned again"), "{}", err);
        assert!(!dupes.exists(), "the refused run left {} behind", dupes.display());

        // Every level it had to conjure goes, not just the last one.
        let nested = dir.path().join("a/b/c");
        let err = run_error(&dir, &db, &["-r", "--move-to", &nested.to_string_lossy()]);
        assert!(err.contains("would be scanned again"), "{}", err);
        assert!(!dir.path().join("a").exists(), "the whole chain is this run's to take back");

        // And a destination that was already there is never this run's to
        // remove, however it feels about being pointed at it.
        let existing = dir.path().join("existing");
        std::fs::create_dir(&existing).unwrap();
        let err = run_error(&dir, &db, &["-r", "--move-to", &existing.to_string_lossy()]);
        assert!(err.contains("would be scanned again"), "{}", err);
        assert!(existing.is_dir(), "a folder this run did not make is not its to delete");
    }

    #[test]
    fn test_a_move_to_folder_a_run_accepts_stays_even_when_nothing_moves() {
        // The other half: creating the folder IS the request, and a run that
        // gets past every check keeps it whether or not it found anything to
        // put in there. An empty scan reaches the same place, which is what
        // this uses -- there is nothing to move and the folder is still made.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let elsewhere = tempfile::tempdir().unwrap();

        let dest = elsewhere.path().join("dupes");
        let stats = RunStats::default();
        assert!(matches!(
            run(
                &args_over(&dir, &["--move-to", &dest.to_string_lossy()]),
                Some(&db),
                Instant::now(),
                1,
                &stats
            ),
            Ok(Outcome::Completed)
        ));
        assert!(dest.is_dir(), "the run was accepted, so the folder is the user's now");
    }

    #[test]
    fn test_only_the_folders_a_run_had_to_make_are_taken_back() {
        // The list has to be read off the filesystem BEFORE anything is created,
        // and it stops at the first thing that is already there. The empty
        // parent of a bare relative name is the shape that could hang it.
        let dir = tempfile::tempdir().unwrap();
        assert!(missing_ancestors(dir.path()).is_empty(), "nothing to make");

        let leaf = dir.path().join("x/y");
        assert_eq!(
            missing_ancestors(&leaf),
            vec![leaf.clone(), dir.path().join("x")],
            "deepest first, stopping at the folder that already exists"
        );

        let bare = Path::new("vid-fp-test-no-such-folder");
        assert_eq!(missing_ancestors(bare), vec![bare.to_path_buf()]);
    }

    #[test]
    fn test_a_destination_that_cannot_be_asked_is_not_asked() {
        // A FIFO opened for writing blocks until someone reads it, and the
        // close that ends the probe would then be an EOF for that reader --
        // the check would consume the pipe the report was meant to go down.
        // So a target that is not a regular file is accepted unexamined, which
        // is exactly the behaviour it had before this check existed. The test
        // is that it RETURNS: under the plain open it never would.
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("pipe");
        let c_path = std::ffi::CString::new(fifo.to_string_lossy().as_bytes()).unwrap();
        // SAFETY: a path this process just made up, inside a temp dir it owns.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);

        let as_str = fifo.to_string_lossy().to_string();
        assert!(report_target_for(Some(&as_str), None).unwrap().is_some());
        // And it is still a pipe afterwards, not a regular file the probe
        // replaced.
        assert!(!std::fs::metadata(&fifo).unwrap().is_file());
    }

    #[test]
    fn test_an_irreversible_run_whose_report_has_nowhere_to_go_never_starts() {
        // The same guard from the outside, which is the only place the cost of
        // losing it is visible: `run` has to refuse before the disposal pass,
        // because the report is written after it. Reproduced with the exact
        // shape reported -- two copies of one video, a folder the report cannot
        // be written into, and --delete --permanent -y.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        let fixture = std::fs::read("tests/fixtures/test_video.mp4").unwrap();
        let copies = ["a.mp4", "b.mp4"].map(|name| dir.path().join(name));
        for copy in &copies {
            std::fs::write(copy, &fixture).unwrap();
        }

        // Not scanned itself: the walk is non-recursive here, and a subfolder
        // of the library is where a user would naturally put the report.
        let ro = dir.path().join("ro");
        std::fs::create_dir(&ro).unwrap();
        std::fs::set_permissions(&ro, std::os::unix::fs::PermissionsExt::from_mode(0o500))
            .unwrap();
        let target = ro.join("rep.csv").to_string_lossy().to_string();

        let err = run_error(&dir, &db, &["-o", &target, "--delete", "--permanent", "-y"]);
        assert!(err.contains("cannot be written"), "{}", err);

        // The refusal is worth nothing if it arrives after the deletion. Both
        // copies are still here, and the run that would have removed one of
        // them never reached the point of deciding which.
        for copy in &copies {
            assert!(copy.exists(), "{} was removed by a run that could not report it", copy.display());
        }
    }

    /// What a resolved target came out as, in the two words the test cares
    /// about.
    fn resolved(output: &str, format: Option<Format>) -> (String, Format) {
        let target = report_target_for(Some(output), format).unwrap().unwrap();
        let where_to = match &target.sink {
            Sink::Stdout => "-".to_string(),
            Sink::File(p) => p.to_string_lossy().to_string(),
        };
        (where_to, target.format)
    }

    #[test]
    fn test_the_extension_still_decides_the_format_when_nothing_overrides_it() {
        // The behaviour --format was added beside, not instead of: every -o
        // that worked before it existed resolves the way it always did, and an
        // extension nobody recognises is still text rather than a refusal.
        assert_eq!(resolved("report.csv", None).1, Format::Csv);
        assert_eq!(resolved("report.JSON", None).1, Format::Json);
        assert_eq!(resolved("report.txt", None).1, Format::Txt);
        assert_eq!(resolved("report.xml", None).1, Format::Txt);
        assert_eq!(resolved("report", None).1, Format::Txt);
    }

    #[test]
    fn test_format_overrides_whatever_the_file_is_called() {
        // The point of the flag: the name and the layout are separate
        // decisions, so a report can be kept under any name and still be
        // written -- and read back -- as itself.
        assert_eq!(resolved("dupes.bak", Some(Format::Csv)).1, Format::Csv);
        assert_eq!(resolved("dupes.bak", Some(Format::Json)).1, Format::Json);

        // Including when the extension says something else outright. An
        // explicit flag is not a hint to be weighed against a filename.
        assert_eq!(resolved("dupes.json", Some(Format::Csv)).1, Format::Csv);
    }

    #[test]
    fn test_a_lone_dash_is_stdout_rather_than_a_file_of_that_name() {
        // It used to be a path like any other: the parent of "-" is the working
        // directory, so the check passed and the run wrote a file literally
        // called - into the cwd.
        assert!(matches!(
            report_target_for(Some("-"), None).unwrap().unwrap().sink,
            Sink::Stdout
        ));

        // Text by default, because that is what the console prints and a bare
        // `-o -` is that listing on a stream you can pipe.
        assert_eq!(resolved("-", None).1, Format::Txt);
        assert_eq!(resolved("-", Some(Format::Json)).1, Format::Json);

        // And the file of that name is still reachable, just not by accident.
        assert!(matches!(
            report_target_for(Some("./-"), None).unwrap().unwrap().sink,
            Sink::File(_)
        ));
    }

    #[test]
    fn test_a_format_with_nowhere_to_go_is_refused() {
        // --format alone could have been read as "and put it on stdout", which
        // would turn a run that writes no report into one that does. It says
        // how, not where.
        let err = format!(
            "{:#}",
            report_target_for(None, Some(Format::Json)).unwrap_err()
        );
        assert!(err.contains("--output"), "{}", err);
        assert!(err.contains("-o -"), "{}", err);
    }

}
