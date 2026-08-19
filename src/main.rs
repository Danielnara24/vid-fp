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
use export::{Disposal, Format, ReportTarget, Sink};
use fingerprint::{fingerprint_video, VideoFingerprint, MAX_DECODE_THREADS};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use log::info;
use rayon::prelude::*;
use redb::{Database, DatabaseError, ReadableTable, StorageError, TableDefinition};
use serde::{Deserialize, Serialize};
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

/// Files this build looked at and refused before opening: see `NotMedia`.
///
/// A SEPARATE table, which is the whole reason this was affordable. The
/// fingerprint table's layout is untouched, so adding this invalidates nothing
/// and costs no re-decode -- and the two halves can be retired on completely
/// different schedules, which they need to be.
///
/// The name carries the rules, and it is meant to be thrown away. A positive
/// entry is expensive to rebuild (it is a decode) so `Stamp` guards it as
/// tightly as it can and the table is renamed only when a field changes meaning.
/// A refusal is worth half a millisecond, so it can be discarded on the merest
/// suspicion -- and it has to be, because it depends on things `Stamp`
/// deliberately does not record: the vendored FFmpeg's demuxer set (`ff8`) and
/// this tool's own gate (`probe1` = `FIRST_PROBE_BYTES`/`SECOND_PROBE_BYTES`/
/// `NO_EVIDENCE`). Change any of those and the entries are wrong in the
/// direction that matters -- a file that would now be read as video, remembered
/// as junk -- so bump the suffix and push the old name into `SUPERSEDED_TABLES`.
/// That costs one cheap re-probe of the library and nothing else.
const REFUSED_TABLE: TableDefinition<&str, &[u8]> =
    TableDefinition::new("refused_ff8_probe2");

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

/// Refusal tables earlier builds wrote. Kept apart from the list above because
/// what it costs to drop one is not comparable: retiring a fingerprint table is
/// a re-decode of the whole library, retiring this is a re-probe measured in
/// seconds, and the run says so in different words.
///
/// `refused_ff8_probe1` was retired for a reason no fingerprint table would
/// ever be retired for: `Refusal::Said` stores the sentence the run printed, so
/// the WORDING is part of what is in there. Trimming the duplicated path out of
/// those messages would otherwise have left a cache printing two formats in one
/// run, half of them naming the file twice.
const SUPERSEDED_REFUSAL_TABLES: [TableDefinition<&str, &[u8]>; 1] =
    [TableDefinition::new("refused_ff8_probe1")];

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
            && same_sampling(self.kf_interval, other.kf_interval)
            // `--min-keyframes` is a FLOOR on the sampling interval, and with no
            // interval in force there is nothing for it to floor -- see
            // `effective_interval`, which requires a real interval before it
            // reads this at all. Comparing it regardless threw away a whole
            // library's fingerprints for a flag that provably changed not one
            // frame. Either both intervals are known equal by here or both are
            // off, so testing either side's flag is the same test.
            && (!sampling_is_on(other.kf_interval)
                || same_setting(self.min_kf_samples, other.min_kf_samples))
    }
}

fn same_setting(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// Whether two runs sample keyframes the same way.
///
/// The interval is not a number the decode reads, it is a number the decode
/// asks a question of: `effective_interval` is `kf_interval > 0.0`, so EVERY
/// value that fails that test -- 0.0, any negative, NaN -- decodes every
/// keyframe and produces bit-identical fingerprints. Comparing the figures
/// instead of the answers made `--keyframe-interval=-5` a different setting
/// from the default, and the miss overwrites the entry it missed, so the next
/// default run missed too: alternating the two re-decoded the whole library
/// every time, twice over, for a flag that provably changed not one frame.
/// Same class of bug as the one `--min-keyframes` used to cause below, and the
/// same fix -- ask the predicate the decode asks.
fn same_sampling(a: f64, b: f64) -> bool {
    if !sampling_is_on(a) && !sampling_is_on(b) {
        // Both mean "decode every keyframe". Nothing downstream can tell them
        // apart, so neither may invalidate the other's entry.
        return true;
    }
    same_setting(a, b)
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
const DEFAULT_EXTENSIONS: [&str; 18] = [
    "mp4", "m4v", "mkv", "webm", "avi", "mov", "flv", "wmv", "asf", "mpg", "mpeg", "m2ts", "mts",
    "ts", "vob", "ogv", "3gp", "divx",
];

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

    /// Path to exclude from the scan: a folder, or a single file. Repeat the
    /// flag to exclude several (e.g. -e ~/a -e ~/b). Matched one whole path
    /// component at a time, so -e ~/keep covers everything under it, while
    /// -e ~/clips/take does NOT cover ~/clips/take.mkv -- a file has to be
    /// named exactly. Applies to piped and explicitly named paths too, and
    /// protects a file whichever route the scan reached it by, so excluding
    /// either a symlink or what it points at covers both.
    #[arg(short = 'e', long = "exclude", value_name = "PATH",
          value_hint = clap::ValueHint::AnyPath)]
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
    /// symlinked directory is not descended into. The same bytes are never
    /// fingerprinted twice however many links reach them, because files are
    /// deduplicated by (device, inode). Worth knowing before arming a
    /// destructive run: deleting or moving a file found through a link acts on
    /// the file the link leads to, not on the link, so a folder you want left
    /// alone is worth naming with --exclude (which follows links too, and
    /// protects such a file whichever path the walk reached it by).
    #[arg(long = "follow-symlinks")]
    follow_symlinks: bool,

    /// Video file extensions to search for in a FOLDER (case-insensitive; a
    /// leading dot or `*.` is optional). Repeat the flag or comma-separate,
    /// e.g. `-x mp4,mkv` or `-x mp4 -x mkv`. Defaults to the common video
    /// containers. Use `-x '*'` — quoted — to fingerprint every file whatever
    /// it is called, which is the only way to reach files with no extension at
    /// all; expect failures for the non-videos it then hands to the decoder.
    /// A file named on the command line is scanned whatever its extension, so
    /// this never has to be widened just to reach one file.
    #[arg(
        short = 'x',
        long = "extensions",
        value_delimiter = ',',
        value_name = "EXT",
        default_values_t = DEFAULT_EXTENSIONS.map(String::from)
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
    /// one no pair could ever clear. 0 turns the gate off and reports every
    /// pair that shares any footage at all; pairs sharing nothing are never
    /// reported at any setting.
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

    /// Minimum keyframes to keep for short videos. When duration divided by
    /// this count is finer than --keyframe-interval, that finer spacing is used
    /// instead, so every video keeps at least this many samples. The two rules
    /// meet at a runtime of --keyframe-interval times this count: above it the
    /// interval alone already yields enough samples and this never applies,
    /// below it this takes over. Raising it is NOT monotonic, because a denser
    /// sample makes each hash stand for a shorter span, so extra frames that
    /// match nothing dilute a pair's coverage and can push it under
    /// --match-percent. 4, 12, 20 and 28 all measure well; 8 and 16 lose pairs
    /// that way. Only used when --keyframe-interval is > 0.0.
    #[arg(long = "min-keyframes", default_value_t = 12.0)]
    min_kf_samples: f64,

    /// Priority for determining the best file to KEEP
    #[arg(short = 'k', long = "priority", default_value = "length")]
    priority: Priority,

    /// Output file for the results. The format follows the extension (.txt,
    /// .csv, .json; anything else is written as text) unless --format overrides
    /// it. Use - to write the report to stdout, where it replaces the terminal
    /// listing rather than repeating it; progress, warnings and prompts stay on
    /// stderr, so `vid-fp DIR -o - --format csv | grep DELETE` pipes the report
    /// alone. A file actually named - is reachable as ./-
    #[arg(short = 'o', long = "output", value_hint = clap::ValueHint::FilePath)]
    output: Option<String>,

    /// Write the report in this format regardless of what --output is called.
    /// Needed for stdout, which has no extension to read, and for a report kept
    /// under a name that carries no format (dupes.bak). Without it the
    /// extension decides.
    #[arg(long = "format", value_enum, value_name = "FORMAT")]
    format: Option<Format>,

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

    /// Act on a CSV or JSON report from an earlier run instead of scanning
    /// anything (the format is read from the file itself, so its name does not
    /// matter). Every row whose
    /// action reads DELETE is disposed of, and nothing else is touched — so
    /// editing that field is how you act on the rows the tool would not decide
    /// for you (REVIEW), or spare one it would. Nothing is re-fingerprinted and
    /// no groups are recomputed: the report is the decision. Each file is still
    /// re-checked against the size the report recorded and left alone if it
    /// changed since. A .txt report cannot be replayed: it records no size to
    /// check against. Requires --delete or --move-to, and cannot be combined
    /// with scanning or matching options.
    #[arg(
        long = "from-report",
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath,
        conflicts_with_all = [
            "include", "exclude", "from_file", "null", "recursive", "follow_symlinks",
            "extensions", "hamming_distance", "match_percent", "min_duration",
            "kf_interval", "min_kf_samples", "priority", "output", "format",
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

    /// Use this cache file instead of the default one under
    /// $XDG_CACHE_HOME/vid-fp (or ~/.cache/vid-fp). A run locks its cache for
    /// the whole scan, so two runs at once need two caches — this is what
    /// makes scanning separate libraries in parallel possible. A path that is
    /// an existing directory, or is written with a trailing slash, gets the
    /// default filename inside it; missing parent directories are created.
    /// --clear-cache and --prune-cache act on whichever cache this names.
    #[arg(long = "cache", value_name = "PATH",
          value_hint = clap::ValueHint::FilePath)]
    cache: Option<String>,

    /// Delete ALL cache before running
    #[arg(long = "clear-cache")]
    clear_cache: bool,

    /// Write every log line to this file, including the failures the terminal
    /// never shows. A failure that the end-of-run summary is going to account
    /// for is not printed while the run works — under -x '*' every file that is
    /// not a video is one, and a scan of a home directory produces hundreds of
    /// thousands — so the summary's count and its examples are what you see.
    /// This is where the unabridged list goes when you want to grep it. The
    /// file is truncated at the start of each run.
    #[arg(long = "log-file", value_name = "PATH",
          value_hint = clap::ValueHint::FilePath)]
    log_file: Option<String>,

    /// Drop the cached fingerprint of every file this scan did not find, so a
    /// cache stops growing with libraries that have moved on. It is skipped,
    /// loudly, when the scan was not complete enough to measure against: a scan
    /// path that would not resolve, a folder that could not be read, or a scan
    /// that found no videos at all. Pruning against a partial scan would throw
    /// away fingerprints that are still good -- a mistyped folder empties the
    /// cache -- so the run says so, keeps the entries, and exits 2.
    #[arg(long = "prune-cache")]
    prune_cache: bool,

    /// Suppress all terminal output except errors
    #[arg(short = 'q', long = "quiet")]
    quiet: bool,

    /// Maximum number of threads to use. 0 uses all available CPU cores
    /// (default), which is the fastest setting for videos on a local disk:
    /// raising it further does not measurably speed a scan up, and every
    /// concurrent decode holds a full-resolution frame buffer, so peak memory
    /// grows roughly in step with this number. More workers than cores is still
    /// worth trying when the videos are on a network mount or a slow disk,
    /// where workers spend their time waiting on I/O rather than on the CPU.
    /// Anything above four per core (or 16, whichever is larger) is capped at
    /// that, with a note saying so.
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
    start_time: Instant,
    stats: &RunStats,
) -> Result<Outcome> {
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
    db: Option<&Database>,
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
        match db {
            Some(db) => match cache_forget(db, &deleted_paths) {
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
            // does, because `cache_forget` is one transaction either way, and
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

/// The bar for the weighing pass, which counts FILES.
///
/// The decode bar measures work, and cannot be imitated here for the reason this
/// pass exists at all: nothing has measured any work yet. A count is honest
/// about that, and the two bars are visibly different jobs rather than one bar
/// that appears to restart.
///
/// It exists because this pass is silent and, on the run it was written for, was
/// most of the wall clock: 229k files under `-x '*'`, an `avformat_open_input`
/// each, and the only thing on screen was a "Found 229112 files. Fingerprinting"
/// printed minutes before any fingerprinting started. On an ordinary library it
/// is over in well under a second and the bar is a flicker, which is the right
/// trade in the direction that matters -- the case it explains is the slow one.
fn weighing_bar(files: usize, quiet: bool) -> ProgressBar {
    if quiet {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(files as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} \u{2502} [{bar:28.cyan/blue}] \u{2502} {percent}% \u{2502} {pos}/{len} \u{2502} measuring decode cost",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb
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
    txn.open_table(REFUSED_TABLE).context("Failed to create the refusals table")?;
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
    txn.delete_table(REFUSED_TABLE).context("Failed to clear the refusals table")?;
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
fn retire_superseded_tables(db: &Database) -> Result<Retired> {
    let txn = db.begin_write().context("Failed to start a cache transaction")?;
    // Tracked apart, because what the user is told it costs depends on which
    // one went: a retired fingerprint table is a re-decode of the whole
    // library, a retired refusals table is a re-probe measured in seconds.
    let mut retired = Retired::default();
    for table in SUPERSEDED_TABLES {
        retired.fingerprints |= txn
            .delete_table(table)
            .context("Failed to remove a superseded cache table")?;
    }
    for table in SUPERSEDED_REFUSAL_TABLES {
        retired.refusals |= txn
            .delete_table(table)
            .context("Failed to remove a superseded refusals table")?;
    }
    txn.commit().context("Failed to commit the cache table removal")?;
    Ok(retired)
}

/// Which kinds of superseded table this run found and dropped.
#[derive(Default)]
struct Retired {
    fingerprints: bool,
    refusals: bool,
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

/// The refusal on record for this path, if this exact file was refused before.
///
/// Guarded by the same `Stamp` a fingerprint is, so an edited file is re-asked;
/// the sampling knobs are in there too, which is stricter than this verdict
/// needs but costs nothing and cannot be wrong.
fn refusal_lookup(db: &Database, path: &str, stamp: &Stamp) -> Option<Refusal> {
    let read = db.begin_read().ok()?;
    let table = read.open_table(REFUSED_TABLE).ok()?;
    let stored = table.get(path).ok()??;

    match bincode::deserialize::<(Stamp, Refusal)>(stored.value()) {
        Ok((cached, verdict)) if cached.matches(stamp) => Some(verdict),
        Ok(_) => None,
        Err(_) => None,
    }
}

/// Why a file will not be fingerprinted, in the form the cache keeps it.
///
/// Two shapes because the common one deserves to be small. The probe gate
/// refuses hundreds of thousands of files in a single `-x '*'` run and its
/// sentence is two numbers in a fixed template, so it is stored as the two
/// numbers; everything else is a handful of files a run, and storing what they
/// actually said is both cheaper to write and impossible to get wrong.
#[derive(Serialize, Deserialize, Clone)]
enum Refusal {
    /// Nothing in the file looked like media. Regenerated for printing, so the
    /// wording always belongs to the build doing the printing.
    NotMedia(fingerprint::NotMedia),
    /// It opened, and then something else was wrong with it -- no video stream,
    /// streams that would not parse, no frame that decoded. The message is the
    /// one the run printed when it found out.
    Said(String),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::NotMedia(verdict) => write!(f, "{}", verdict),
            Refusal::Said(said) => write!(f, "{}", said),
        }
    }
}

/// What to remember about a failure, if anything.
///
/// `None` for a failure that is about the moment rather than the file -- see
/// `fingerprint::is_transient`. This is the only gate between an error and the
/// cache, so both callers (the weighing pass and the decode) ask it rather than
/// classifying for themselves.
fn worth_remembering(error: &anyhow::Error) -> Option<Refusal> {
    if let Some(verdict) = fingerprint::not_media(error) {
        return Some(Refusal::NotMedia(verdict));
    }
    if fingerprint::is_transient(error) {
        return None;
    }
    Some(Refusal::Said(format!("{:#}", error)))
}

/// Write a whole pass worth of refusals in ONE transaction.
///
/// Deliberately not shaped like `cache_store`, which commits per video because a
/// decode costs seconds and an interrupt must not throw one away. A refusal
/// costs half a millisecond, and there can be a quarter of a million of them in
/// a single `-x '*'` run: one fsync each would cost far more than re-probing
/// every file next time. Losing the lot to an interrupt is fine -- the next run
/// simply pays the same half-millisecond again.
fn refusals_store(db: &Database, refused: &[(String, Stamp, Refusal)]) -> Result<()> {
    if refused.is_empty() {
        return Ok(());
    }

    let txn = db.begin_write().context("Failed to start a cache transaction")?;
    {
        let mut table =
            txn.open_table(REFUSED_TABLE).context("Failed to open the refusals table")?;
        for (path, stamp, verdict) in refused {
            let encoded = bincode::serialize(&(stamp, verdict))
                .context("Failed to serialize a refusal")?;
            table
                .insert(path.as_str(), encoded.as_slice())
                .context("Failed to record a refusal")?;
        }
    }
    txn.commit().context("Failed to commit the refusals")?;

    Ok(())
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
        // Both tables: a file that was trashed or moved is no longer at this
        // path whichever of the two had something to say about it, and a
        // refusal left behind would outlive the bytes it was about exactly the
        // way a fingerprint would.
        let mut refused =
            txn.open_table(REFUSED_TABLE).context("Failed to open the refusals table")?;
        for path in paths {
            let existed = table
                .remove(path.as_str())
                .context("Failed to remove a cache entry")?
                .is_some();
            refused.remove(path.as_str()).context("Failed to remove a refusal")?;
            if existed {
                forgotten += 1;
            }
        }
    }
    txn.commit().context("Failed to commit the cache removals")?;

    Ok(forgotten)
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

/// Where the cache lives when `--cache` is not given.
///
/// Only the default location gets the legacy sweep and the XDG lookup; a cache
/// the user named is theirs, and this build has never written anything else
/// beside it.
const CACHE_FILE_NAME: &str = "fingerprints.redb";

/// Decide which file this run's fingerprints are cached in.
///
/// A cache holds an exclusive lock for the whole run -- not just the cache pass
/// -- so two concurrent scans cannot share one, and `--cache` is the only way
/// to run them at all. See `open_cache`.
///
/// `--cache` names the database FILE, because that is what it is and because a
/// user pointing at a scratch disk wants to know exactly what appears there.
/// A path that is already a directory, or that is written with a trailing
/// slash, is treated as one instead and gets the default filename inside it:
/// `--cache /mnt/scratch` obviously means "a cache in here", and the
/// alternative is a redb file named `scratch`.
fn resolve_cache_path(explicit: Option<&str>) -> Result<PathBuf> {
    let Some(raw) = explicit else {
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

        return Ok(cache_dir.join(CACHE_FILE_NAME));
    };

    let given = Path::new(raw);
    let path = if raw.ends_with('/') || given.is_dir() {
        given.join(CACHE_FILE_NAME)
    } else {
        given.to_path_buf()
    };

    // The default location is created for the user, so a named one is too --
    // otherwise `--cache /mnt/scratch/vid-fp/cache.redb` fails on a directory
    // the user would have created without thinking about it.
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("Failed to create the cache directory {}", parent.display())
        })?;
    }

    Ok(path)
}

/// Why the cache could not be opened, and whether the run has to stop for it.
///
/// `locked` is the one failure a mode can survive: it says another run holds
/// the file, not that anything is wrong with the file. Everything else -- an
/// unreadable format, no permission, a full disk -- is a reason to stop, since
/// nothing about it improves by carrying on. Only `--from-report` uses the
/// distinction; see `main`.
struct CacheUnavailable {
    locked: bool,
    reason: anyhow::Error,
}

/// Open the fingerprint cache, and say something useful when that fails.
///
/// A single file, an explicitly bounded page cache, and no background threads:
/// every write is durable when its transaction commits, so there is no buffered
/// work to lose and nothing that has to be told to stop.
///
/// Opening takes an exclusive `flock` on the file, which is what keeps two
/// concurrent runs from fighting over the same cache -- and is also the most
/// likely way this call fails, because scanning two libraries at once is a
/// perfectly ordinary thing to do. redb reports that case as its own error
/// variant, so every branch here can name both the file and what to do about
/// it; a single `.context()` over the lot used to print one sentence with no
/// path and no cause for four unrelated failures. The lock is released by the
/// kernel when a process ends, however it ends, so there is no such thing as a
/// stale one to advise clearing.
fn open_cache(db_path: &Path) -> std::result::Result<Database, CacheUnavailable> {
    let fatal = |reason| CacheUnavailable { locked: false, reason };

    match Database::builder().set_cache_size(CACHE_SIZE_BYTES).create(db_path) {
        Ok(db) => Ok(db),
        Err(DatabaseError::DatabaseAlreadyOpen) => Err(CacheUnavailable {
            locked: true,
            reason: anyhow::anyhow!(
                "Another vid-fp run is using the fingerprint cache at {}.\n\
                 A run holds that cache for its whole scan, not just while it reads it, \
                 so only one at a time can have it.\n\
                 Wait for the other run to finish, or give this one a cache of its own:\n    \
                 vid-fp --cache /path/to/other-cache.redb ...",
                db_path.display()
            ),
        }),
        Err(DatabaseError::UpgradeRequired(version)) => Err(fatal(anyhow::anyhow!(
            "The fingerprint cache at {} is in an older on-disk format (version {}) \
             that this build cannot open.\n\
             Delete the file and it will be rebuilt on the next scan.",
            db_path.display(),
            version
        ))),
        Err(DatabaseError::Storage(StorageError::Io(e)))
            if e.kind() == std::io::ErrorKind::PermissionDenied =>
        {
            Err(fatal(anyhow::anyhow!(
                "No permission to open the fingerprint cache at {}: {}.\n\
                 Fix the permissions, or point this run somewhere writable with \
                 --cache /path/to/cache.redb.",
                db_path.display(),
                e
            )))
        }
        Err(e) => Err(fatal(anyhow::Error::new(e).context(format!(
            "Failed to open the fingerprint cache at {}",
            db_path.display()
        )))),
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

    env_logger::Builder::new()
        .filter_level(log_level)
        .format(move |buf, record| {
            let line = if record.level() == log::Level::Error {
                format!("Error: {}", record.args())
            } else {
                format!("{}", record.args()) // Clean output for CLI tools
            };

            // The file gets everything, uncapped and unconditionally -- that is
            // what it is for, and it is the only place the unabridged list of
            // failures exists.
            if let Some(file) = &log_file {
                let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
                let _ = writeln!(file, "{}", line);
            }

            // A failure the run is going to account for is not also announced
            // as it happens: `stats::print_summary` reports it at the end, with
            // a count and up to `MAX_SAMPLES` examples, and saying it twice is
            // how a `-x '*'` scan buried its own results under 226,863 lines.
            // See `stats::COUNTED` for what is and is not tagged this way.
            if record.target() == stats::COUNTED {
                return Ok(());
            }

            writeln!(buf, "{}", line)
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

    let db_path = resolve_cache_path(args.cache.as_deref())?;

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
    let mut db = match open_cache(&db_path) {
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
        Err(e) => return Err(e.reason),
    };

    if let Some(db) = db.as_mut() {
        ensure_cache_table(db).context("Failed to initialize the fingerprint cache")?;

        // Once, on the first run of this build. Nothing will ever look in those
        // tables again, so every byte of them is dead. Compacting right
        // afterwards is the only thing that returns those pages to the
        // filesystem.
        match retire_superseded_tables(db) {
            Ok(retired) if retired.fingerprints || retired.refusals => {
                if retired.fingerprints {
                    info!(
                        "Removed the superseded fingerprint cache; fingerprints will be rebuilt \
                         once."
                    );
                } else {
                    // Worth a different sentence rather than the one above:
                    // nothing is being re-decoded, and a user who reads
                    // "fingerprints will be rebuilt" before a scan of a large
                    // library is being told to expect an hour that is not
                    // coming.
                    info!(
                        "Removed the superseded record of files that are not video; they will be \
                         re-checked once."
                    );
                }
                if let Err(e) = db.compact() {
                    log::error!("Could not compact the fingerprint cache: {}", e);
                }
            }
            Ok(_) => {}
            Err(e) => log::error!("{:#}", e),
        }

        // Handled here rather than inside `run` because compaction needs an
        // exclusive handle on the database, and because this is the only point
        // in the run where compacting is free. See `clear_cache`.
        if args.clear_cache {
            info!("Clearing all cache...");
            clear_cache(db)?;
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
    // Hence `!args.clear_cache`, which this condition used to be missing even
    // though the sentence above already stated the rule. `--prune-cache` and
    // `--clear-cache` together skip the prune itself (see `run`: after a clear
    // there is nothing stale, because there is nothing at all), so the run
    // removed no entries -- and then compacted anyway, copying every live page
    // of a cache the scan had just rewritten from scratch to reclaim exactly
    // nothing. That is the same waste compaction was moved out of the tail of
    // the run to avoid; it is invisible on a small cache and is a full rewrite
    // plus an fsync on a large one.
    //
    // `db` is None only under `--from-report`, which cannot be given
    // `--prune-cache` at all, so this never silently skips a compaction the
    // user asked for.
    if !shutdown_requested() && args.prune_cache && !args.clear_cache {
        if let Some(db) = db.as_mut() {
            match db.compact() {
                Ok(true) => info!("Compacted the fingerprint cache."),
                Ok(false) => {}
                Err(e) => log::error!("Could not compact the fingerprint cache: {}", e),
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

fn run(
    args: &Args,
    db: Option<&Database>,
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
    let report_target = report_target_for(args.output.as_deref(), args.format)?;

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
    let library = match sources::collect(
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

    // Moving a duplicate into a folder that is itself being scanned puts the
    // file straight back into the next run's input, where it is still a
    // duplicate of the copy that was kept -- and the run after that would move
    // it again, one directory deeper each time. Nothing is lost, but the one
    // thing this mode is for (getting files OUT of the library, reversibly) is
    // quietly not happening, so it is worth stopping over rather than warning
    // about.
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

    // What to call the things that came back. Under `-x '*'` the walk applied no
    // extension filter, so calling them video files is a claim the run has not
    // checked and, on the folder that flag exists for, usually a false one --
    // "Found 229112 video files" over a home directory, of which a handful were
    // video. Every other setting has already turned away anything not named like
    // a video, so there the old wording was the accurate one and it is kept.
    let found_noun = if library.any_extension { "files" } else { "video files" };

    // Every question about the REQUEST has been answered; from here on the run
    // works from what it found.
    let mut video_files = library.files;

    // A prune is measured against what the scan FOUND, so it is only ever as
    // correct as the scan was complete -- and a run that fell short knows it.
    let pruning = args.prune_cache && !args.clear_cache;
    let obstacle = if pruning { prune_obstacle(stats, video_files.len()) } else { None };

    if let Some(reason) = obstacle {
        log::warn!("Not pruning the cache: {}, so this run cannot tell what is stale.", reason);
        stats.cache_prune_skipped.record(reason);
    } else if pruning {
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
        // Both tables, for one reason: an entry the scan cannot see is an entry
        // nothing will ever overwrite, and that is as true of a refusal as of a
        // fingerprint. The refusals table is also the one that GROWS without
        // bound -- a `-x '*'` run files one entry per file that is not video,
        // so a library that moves around leaves far more of them behind than it
        // ever leaves fingerprints.
        let mut stale: Vec<String> = Vec::new();
        let mut stale_refusals: Vec<String> = Vec::new();
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

            let refused = read
                .open_table(REFUSED_TABLE)
                .context("Failed to open the refusals table")?;

            for entry in refused.iter().context("Failed to iterate the refusals")? {
                if shutdown_requested() {
                    return Ok(Outcome::Interrupted);
                }

                let (key, _) = entry.context("Failed to read a refusal")?;
                let path = key.value();

                if !valid_files.contains(path) {
                    stale_refusals.push(path.to_string());
                }
            }
        }

        if !stale.is_empty() || !stale_refusals.is_empty() {
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

                let mut refused = txn
                    .open_table(REFUSED_TABLE)
                    .context("Failed to open the refusals table")?;
                for key in &stale_refusals {
                    refused
                        .remove(key.as_str())
                        .context("Failed to remove a stale refusal")?;
                }
            }
            txn.commit().context("Failed to apply cache pruning")?;
            // Counted apart: they are different things, and a user watching the
            // fingerprint count is watching the one that cost something.
            if stale_refusals.is_empty() {
                info!("Pruned {} stale entries from cache.", stale.len());
            } else {
                info!(
                    "Pruned {} stale entries and {} stale refusal(s) from cache.",
                    stale.len(),
                    stale_refusals.len()
                );
            }
        } else {
            info!("No stale entries found to prune.");
        }
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
            start_time,
            stats,
        );
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
        /// Refused before, by this build's rules, against these exact bytes.
        /// The file never reaches the weighing pass -- which is the entire
        /// saving, since on a `-x '*'` run that pass IS the run.
        Refused((String, String)),
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

            if let Some(verdict) = refusal_lookup(db, &file.path, &stamp) {
                // Regenerated rather than stored: the cache keeps the two
                // numbers, so the sentence a re-run prints is this build's
                // sentence and cannot drift from the one a first run prints.
                return Lookup::Refused((format!("{}", verdict), file.path.clone()));
            }

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
    let mut remembered_refusals = 0usize;
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
            // Counted and worded as if it had just been discovered, because it
            // is the same finding about the same bytes -- the run simply did
            // not have to read them again. Still a problem, still exit 2: the
            // user asked for a fingerprint of this file and there is none.
            Lookup::Refused(reason) => {
                remembered_refusals += 1;
                log::error!(target: stats::COUNTED, "Failed to process {}: {}", reason.1, reason.0);
                stats.fingerprint_failed.record(format!("{}: {}", reason.1, reason.0));
            }
        }
    }
    let cached_count = fingerprints.len();
    let todo_count = todo.len();

    // Said out loud because it is most of what a re-run does. A `-x '*'` scan of
    // a home directory is a quarter of a million files that are not video, and
    // "already known not to be video" is the difference between the four
    // minutes the first run spent finding that out and the seconds this one did.
    let remembered = if remembered_refusals > 0 {
        format!(", {} already known not to be video", remembered_refusals)
    } else {
        String::new()
    };

    if cached_count > 0 || remembered_refusals > 0 {
        info!(
            "Found {} {}; {} already cached{}, {} to fingerprint.",
            total_videos, found_noun, cached_count, remembered, todo_count
        );
    } else {
        info!("Found {} {}. Fingerprinting...", total_videos, found_noun);
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
    //
    // It is also where a file that will not decode at all is now reported and
    // dropped, rather than being given a weight and opened a second time to find
    // out what this pass already found out -- see `Weighed`. Two things follow
    // from that, and both are the point:
    //
    // - The open is paid once per file instead of twice. On the run that
    //   prompted this (229k files under `-x '*'`, almost none of them video)
    //   that is half of everything before the comparison.
    // - What is left in `todo` is work that is really going to happen, so the
    //   weights below add up to the run's actual decode. They did not before: an
    //   unopenable file fell to the bottom rung of the ladder, `size *
    //   WORK_PER_BYTE`, and then finished the instant it was looked at -- so on
    //   a folder that is mostly not video the bar was denominated in the bytes
    //   of files that cost nothing, sat at 100% while the few real videos
    //   decoded, and reported five 20 MB junk files as 71% of the work.
    //
    // A bar of its own, because this pass is now where the time goes on such a
    // run and it used to be minutes of silence between "Found ..." and the
    // decode bar. It counts files rather than work: nothing here has measured
    // anything yet, which is the whole reason the pass exists.
    if !todo.is_empty() {
        let wb = weighing_bar(todo.len(), args.quiet);

        let weighed: Vec<fingerprint::Weighed> = todo
            .par_iter()
            .map(|job| {
                if shutdown_requested() {
                    return fingerprint::Weighed::Work(job.size);
                }
                let weighed =
                    fingerprint::weigh_decode(&job.path, kf_interval, min_kf_samples, job.size);
                wb.inc(1);
                weighed
            })
            .collect();
        wb.finish_and_clear();

        let mut kept: Vec<Job> = Vec::with_capacity(todo.len());
        let mut refused: Vec<(String, Stamp, Refusal)> = Vec::new();
        for (mut job, weighed) in std::mem::take(&mut todo).into_iter().zip(weighed) {
            match weighed {
                fingerprint::Weighed::Work(weight) => {
                    job.weight = weight;
                    kept.push(job);
                }
                // Worded, counted and exit-coded exactly as the decode would
                // have done it, because it IS the decode's error: the file is
                // still one the run was asked to fingerprint and could not.
                fingerprint::Weighed::Undecodable(e) => {
                    log::error!(target: stats::COUNTED, "Failed to process {}: {:#}", job.path, e);
                    stats.fingerprint_failed.record(format!("{}: {:#}", job.path, e));

                    // Remembered unless it was about the moment rather than
                    // the file -- see `worth_remembering` and `REFUSED_TABLE`.
                    if let Some(verdict) = worth_remembering(&e) {
                        refused.push((job.path.clone(), job.stamp, verdict));
                    }
                }
            }
        }
        todo = kept;

        // One transaction for the whole pass -- see `refusals_store`. A failure
        // to write is not a failure of the run: everything these entries would
        // have saved is time, and the next run simply spends it again.
        if let Err(e) = refusals_store(db, &refused) {
            log::debug!("Could not record {} refusal(s): {:#}", refused.len(), e);
        }

        // NOW the largest-first order the schedule depends on is real. It was
        // approximated by size up to here; a folder mixing codecs is exactly
        // where that approximation puts the wrong file first, since an AV1 copy
        // and an HEVC copy of the same footage differ threefold in cost and
        // barely at all in the direction size predicts.
        todo.sort_by_key(|job| std::cmp::Reverse(job.weight));
    }

    // What the decode is actually facing, which is not what was queued for it:
    // everything the pass above refused has been reported already. Shadowed
    // rather than assigned so the count in the message above stays the count
    // that was true when it was printed.
    let todo_count = todo.len();

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
        // Failures worth remembering, gathered from every worker and written
        // once when they have all joined -- see `refusals_store`.
        let refused_here: Mutex<Vec<(String, Stamp, Refusal)>> = Mutex::new(Vec::new());

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
        let refused_here = &refused_here;

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
                                    log::error!(target: stats::COUNTED, "Failed to process {}: {:#}", job.path, e);
                                    stats
                                        .fingerprint_failed
                                        .record(format!("{}: {:#}", job.path, e));

                                    // The half of the memory that saves a
                                    // DECODE rather than a probe. These are the
                                    // files that open and then turn out to hold
                                    // nothing to fingerprint -- an image
                                    // container with no frame, a stream that
                                    // will not parse -- and re-reading them
                                    // every run was the whole of a warm scan.
                                    if let Some(verdict) = worth_remembering(&e) {
                                        refused_here
                                            .lock()
                                            .unwrap_or_else(|p| p.into_inner())
                                            .push((job.path.clone(), job.stamp, verdict));
                                    }
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
                                    target: stats::COUNTED,
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

        let remember = refused_here.lock().unwrap_or_else(|e| e.into_inner()).split_off(0);
        if let Err(e) = refusals_store(db, &remember) {
            log::debug!("Could not record {} refusal(s): {:#}", remember.len(), e);
        }

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
        // The elapsed time used to be formatted again here so this ending would
        // say it too; the report's summary carries it, in the same format, for
        // every other ending.
        let disposal = disposal_for(move_to, args.delete, args.permanent);
        return conclude_without_comparing(
            report_target.as_ref(),
            disposal.as_ref(),
            args,
            start_time,
            stats,
        );
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
        report_target.as_ref(),
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

    #[test]
    fn test_every_interval_that_samples_nothing_is_the_same_setting() {
        // `effective_interval` is gated on `kf_interval > 0.0`, so 0.0, a
        // negative and NaN all decode every keyframe and cannot be told apart
        // from the fingerprint they produce. Comparing the figures made them
        // three settings: `--keyframe-interval=-5` missed against a default
        // entry, overwrote it, and the next default run missed in turn, so
        // alternating them never hit the cache at all.
        let base = stamp(1_700_000_000, 12_345); // kf_interval 0.0, the default
        let negative = Stamp { kf_interval: -5.0, ..base };
        let other_negative = Stamp { kf_interval: -0.5, ..base };
        let nonsense = Stamp { kf_interval: f64::NAN, ..base };

        for off in [negative, other_negative, nonsense] {
            assert!(base.matches(&off), "sampling is off either way; the entry stands");
            assert!(off.matches(&base), "and it stands in the other direction too");
        }
        assert!(negative.matches(&nonsense), "two ways of spelling the same non-setting");

        // A real interval is still a real change, in both directions.
        let sampled = Stamp { kf_interval: 5.0, ..base };
        assert!(!negative.matches(&sampled));
        assert!(!sampled.matches(&negative));
        assert!(!nonsense.matches(&sampled), "NaN is off, 5.0 is not");
        assert!(!sampled.matches(&nonsense));
    }

    #[test]
    fn test_min_keyframes_still_decides_nothing_beside_an_interval_that_is_off() {
        // The two guards have to agree about what "off" means, or the wrong
        // one of them starts comparing a flag the decode never reads.
        let negative = Stamp { kf_interval: -5.0, ..stamp(1_700_000_000, 12_345) };
        let fewer = Stamp { min_kf_samples: 2.0, ..negative };

        assert!(negative.matches(&fewer), "no interval to floor, so this floors nothing");
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

    fn cache_with_an_orphan(dir: &tempfile::TempDir) -> Database {
        let db = temp_db(dir);
        cache_store(&db, ORPHAN, stamp(1_700_000_000, 12_345), &mock_fp(ORPHAN)).unwrap();
        db
    }

    fn cached_paths(db: &Database) -> Vec<String> {
        let read = db.begin_read().unwrap();
        let table = read.open_table(CACHE_TABLE).unwrap();
        table
            .iter()
            .unwrap()
            .map(|e| e.unwrap().0.value().to_string())
            .collect()
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

    /// A refusal survives the run that made it, and is guarded by the same stamp
    /// a fingerprint is -- so editing the file asks again rather than trusting a
    /// verdict about bytes that are no longer there.
    #[test]
    fn test_a_refusal_is_remembered_until_the_file_changes() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let path = "/videos/not-really.mkv";
        let verdict = Refusal::NotMedia(fingerprint::NotMedia { bytes: 2048, score: 0 });
        let written = stamp(1_700_000_000, 12_345);

        refusals_store(&db, &[(path.to_string(), written, verdict)]).unwrap();

        let remembered = refusal_lookup(&db, path, &written).expect("the same file is remembered");
        assert!(format!("{}", remembered).contains("2048"), "and says what it saw");

        assert!(
            refusal_lookup(&db, path, &stamp(1_700_000_001, 12_345)).is_none(),
            "an edit re-opens the question"
        );
        assert!(
            refusal_lookup(&db, path, &stamp(1_700_000_000, 12_346)).is_none(),
            "and so does a change of size"
        );
    }

    /// The two tables are pruned together. The refusals table is the one that
    /// grows without bound -- one entry per file that is not video -- so a prune
    /// that only swept fingerprints would leave the larger half behind.
    #[test]
    fn test_pruning_collects_refusals_as_well_as_fingerprints() {
        let dir = tempfile::tempdir().unwrap();
        let db = cache_with_an_orphan(&dir);
        refusals_store(
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

    /// Disposing of a file forgets both of the things the cache might know about
    /// it. A refusal left behind outlives its bytes exactly the way a
    /// fingerprint would, and the next file to land on that path would inherit
    /// a verdict about a different one -- caught by the stamp, but only by luck.
    #[test]
    fn test_forgetting_a_file_forgets_its_refusal_too() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let path = "/videos/gone.mkv";
        let written = stamp(1_700_000_000, 12_345);
        refusals_store(
            &db,
            &[(
                path.to_string(),
                written,
                Refusal::NotMedia(fingerprint::NotMedia { bytes: 16384, score: 1 }),
            )],
        )
        .unwrap();

        cache_forget(&db, &[path.to_string()]).unwrap();

        assert!(refusal_lookup(&db, path, &written).is_none());
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

    /// A library of two identical clips and the report that plans their merge.
    fn library_with_a_pair(dir: &tempfile::TempDir) -> (PathBuf, PathBuf) {
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

    fn run_error(dir: &tempfile::TempDir, db: &Database, extra: &[&str]) -> String {
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

    #[test]
    fn test_a_locked_cache_names_the_file_and_the_way_out() {
        // Scanning two libraries at once is ordinary, and this is what the
        // second run hits. flock conflicts between two open file descriptions
        // whether or not they belong to the same process, so holding `_first`
        // here reproduces the concurrent run exactly.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fingerprints.redb");
        let _first = Database::create(&path).unwrap();

        let err = open_cache(&path).expect_err("the lock is held");
        let message = format!("{:#}", err.reason);

        assert!(
            err.locked,
            "the one failure --from-report is allowed to carry on through has to be \
             distinguishable from the ones it is not"
        );

        assert!(
            message.contains(&path.display().to_string()),
            "the message has to say WHICH cache: {message}"
        );
        assert!(
            message.contains("--cache"),
            "and how to get a run of your own going: {message}"
        );
    }

    #[test]
    fn test_the_named_cache_path_is_used_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        let wanted = dir.path().join("libraries").join("anime.redb");

        let resolved = resolve_cache_path(Some(wanted.to_str().unwrap())).unwrap();

        assert_eq!(resolved, wanted, "a named file is the cache, whatever it is called");
        assert!(
            wanted.parent().unwrap().is_dir(),
            "and its directory is created, as the default location's is"
        );
    }

    #[test]
    fn test_a_directory_gets_the_default_filename_inside_it() {
        // `--cache /mnt/scratch` means "a cache in here". Both the existing
        // directory and the trailing slash say so; the second is the one a
        // shell's tab completion writes for a directory that isn't there yet.
        let dir = tempfile::tempdir().unwrap();

        let existing = resolve_cache_path(Some(dir.path().to_str().unwrap())).unwrap();
        assert_eq!(existing, dir.path().join(CACHE_FILE_NAME));

        let not_yet = dir.path().join("scratch");
        let trailing = format!("{}/", not_yet.display());
        let resolved = resolve_cache_path(Some(&trailing)).unwrap();

        assert_eq!(resolved, not_yet.join(CACHE_FILE_NAME));
        assert!(not_yet.is_dir(), "the directory it named is created, not a file called `scratch`");
    }
}