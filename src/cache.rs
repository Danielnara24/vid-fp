//! The fingerprint cache: one redb table, one record per path.
//!
//! A record is everything this tool knows about the bytes at one path, as of
//! one stat of them:
//!
//! - `stat` -- mtime (with nsec) and size, the one copy of the file's stat;
//! - `identity` -- what the bytes hold (`content::sample`, and a digest when a
//!   run needed one), which is how a moved file finds its old record;
//! - `verdict` -- either a fingerprint, or a refusal (why this build will not
//!   make one).
//!
//! **Everything in a record is about its `stat`, and that is the whole design.**
//! Writing anything at a different stat starts the record afresh, so a verdict
//! and an identity can never describe two different versions of a file, and
//! writing a verdict replaces the one before it, so a path can never be both
//! fingerprinted and refused. Both used to be rules every writer had to keep by
//! hand across three tables (a fingerprint write removed the refusal, a refusal
//! write removed the fingerprint, an identity vouched for a fingerprint only
//! after comparing two copies of the stat); each was the fix for a bug. Here
//! they are the shape of the data.
//!
//! **How a stored value is retired, by what it costs to rebuild.**
//!
//! - A fingerprint is a decode. When a field changes MEANING (units, clock,
//!   hash algorithm, the vendored FFmpeg major), rename `TABLE` and push the
//!   old name into `SUPERSEDED_TABLES`; the run then says the library will be
//!   fingerprinted again. Appending a field to the end of `VideoFingerprint` is
//!   self-invalidating: it is the last thing serialized, so an older record
//!   runs out of bytes, fails to deserialize, and reads as a miss.
//! - A refusal is half a millisecond. It depends on things the stat does not
//!   record -- the vendored FFmpeg's demuxers, this tool's probe gate
//!   (`FIRST_PROBE_BYTES`, `SECOND_PROBE_BYTES`, `NO_EVIDENCE`, what a
//!   `NotMedia` counts) and, for `Refusal::Said`, which decoder answers fail a
//!   file and in what words. Change any of them and bump `REFUSAL_RULES`: every
//!   refusal written under another number reads as a miss and is re-probed.
//! - An identity is a 192 KB read. Change what `content::sample` reads and
//!   bump `SAMPLE_RULES`.
//!
//! The record is bincode: `stat`, then `identity`, then `verdict`, with the
//! fingerprint last of all. `Head` reads the first two alone, which is what
//! keeps a walk of the whole table (the moved-file index) from deserializing
//! every fingerprint in it. Only the fingerprint can grow by appending a field;
//! a change to `Stat`, `Held`, `Sampling` or `Refusal`'s existing shape moves
//! everything behind it, and is a `TABLE` rename.
//!
//! Per architecture: the keyframes are shrunk by libswscale's fast bilinear
//! scaler, whose x86 SIMD rounds differently from the C code ARM runs, so a
//! cache two architectures share keeps a table each (`VID_FP_ARCH_TABLE_SUFFIX`
//! from `build.rs`, empty on x86_64). Every x86 level from SSE2 to AVX2 gives
//! the same pictures. Making every processor agree (the scaler built from the C
//! code everywhere) was measured and declined: it cost every x86 user a
//! re-decode and 2-6% of a cold scan for agreement no x86 user needs.

use anyhow::{Context, Result};
use log::info;
use redb::{Database, DatabaseError, ReadableTable, StorageError, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::content::{self, Identity};
use crate::fingerprint::{self, VideoFingerprint};
use crate::utils::shutdown_requested;

/// The cache: absolute path -> bincode'd `Entry`.
const TABLE: TableDefinition<&str, &[u8]> =
    TableDefinition::new(concat!("files_ff8_picture", env!("VID_FP_ARCH_TABLE_SUFFIX")));

/// Bump when anything a refusal depends on changes; see the module comment.
/// Continues the numbering the refusal table's name used to carry (`probe6`).
const REFUSAL_RULES: u32 = 6;

/// Bump when what `content::sample` reads changes.
const SAMPLE_RULES: u32 = 1;

/// The three tables the cache was split across before `TABLE`. Read once, by
/// `migrate`, into `TABLE`, and then dropped -- so upgrading cost nobody a
/// re-decode. The conversion can be deleted a few releases on, when these
/// names can join `SUPERSEDED_TABLES`.
const LEGACY_FINGERPRINTS: TableDefinition<&str, &[u8]> = TableDefinition::new(concat!(
    "fingerprints_dct_ct_ff8_picture",
    env!("VID_FP_ARCH_TABLE_SUFFIX")
));
const LEGACY_REFUSALS: TableDefinition<&str, &[u8]> =
    TableDefinition::new(concat!("refused_ff8_probe6", env!("VID_FP_ARCH_TABLE_SUFFIX")));
const LEGACY_IDENTITIES: TableDefinition<&str, &[u8]> = TableDefinition::new("identity_s1");

/// Fingerprint tables earlier builds wrote, all dead: each was renamed because
/// a stored value changed meaning in a way no layout check could see. Dropped
/// whole on the first run of this build, which then says the library will be
/// fingerprinted again.
///
/// `_picture` (in `LEGACY_FINGERPRINTS`): entries without
/// `VideoFingerprint::picture`. `_turned`: a phone video's pictures hashed as
/// stored rather than as its display matrix turns them, and Matroska runtimes
/// read from the segment. `_s128`: a 64x64 working frame. `_inset`: no `Inset`
/// reframing. `_detail`: no detail hashes, and the old featureless floor.
/// `_framecrop`: one crop rectangle per file. `_pts`: HEVC samples a FFmpeg 6.x
/// build filed under another keyframe's time. `_kffloor`: an interval stamped
/// but not applied under `--min-keyframes 0`. `_rescale`: fingerprints
/// truncated at a mid-file resolution change. `_splice`: truncated at a clock
/// restart. `_tail`: `total_ms` a few tenths of a percent short. `_ff8`: the
/// vendored FFmpeg major changed, and decoder output is not bit-identical
/// across majors.
const SUPERSEDED_TABLES: [TableDefinition<&str, &[u8]>; 15] = [
    TableDefinition::new("fingerprints"),
    TableDefinition::new("fingerprints_by_path"),
    TableDefinition::new("fingerprints_dct"),
    TableDefinition::new("fingerprints_dct_ct"),
    TableDefinition::new("fingerprints_dct_ct_ff8"),
    TableDefinition::new("fingerprints_dct_ct_ff8_tail"),
    TableDefinition::new("fingerprints_dct_ct_ff8_splice"),
    TableDefinition::new("fingerprints_dct_ct_ff8_rescale"),
    TableDefinition::new("fingerprints_dct_ct_ff8_kffloor"),
    TableDefinition::new(concat!("fingerprints_dct_ct_ff8_pts", env!("VID_FP_ARCH_TABLE_SUFFIX"))),
    TableDefinition::new(concat!("fingerprints_dct_ct_ff8_framecrop", env!("VID_FP_ARCH_TABLE_SUFFIX"))),
    TableDefinition::new(concat!("fingerprints_dct_ct_ff8_detail", env!("VID_FP_ARCH_TABLE_SUFFIX"))),
    TableDefinition::new(concat!("fingerprints_dct_ct_ff8_inset", env!("VID_FP_ARCH_TABLE_SUFFIX"))),
    TableDefinition::new(concat!("fingerprints_dct_ct_ff8_s128", env!("VID_FP_ARCH_TABLE_SUFFIX"))),
    TableDefinition::new(concat!("fingerprints_dct_ct_ff8_turned", env!("VID_FP_ARCH_TABLE_SUFFIX"))),
];

/// Refusal tables of builds before `REFUSAL_RULES`. Dropped quietly: what they
/// held is a re-probe measured in seconds, not worth a line. A closed list --
/// refusals are retired by number now, not by table.
const SUPERSEDED_REFUSAL_TABLES: [TableDefinition<&str, &[u8]>; 5] = [
    TableDefinition::new("refused_ff8_probe1"),
    TableDefinition::new("refused_ff8_probe2"),
    TableDefinition::new("refused_ff8_probe3"),
    TableDefinition::new("refused_ff8_probe4"),
    TableDefinition::new(concat!("refused_ff8_probe5", env!("VID_FP_ARCH_TABLE_SUFFIX"))),
];

/// redb's page cache. Explicitly bounded so the cache's memory stays flat
/// whatever the library's size.
const CACHE_SIZE_BYTES: usize = 64 * 1024 * 1024;

/// The file name a cache gets when only its folder is given.
pub(crate) const CACHE_FILE_NAME: &str = "fingerprints.redb";

// --- What a record is checked against ------------------------------------------

/// The stat a record is about. One copy per record: the identity and the
/// verdict beside it are both about exactly this.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
struct Stat {
    mtime: i64,
    mtime_nsec: i64,
    size: u64,
}

impl Stat {
    fn of(identity: &Identity) -> Self {
        Self { mtime: identity.mtime, mtime_nsec: identity.mtime_nsec, size: identity.size }
    }
}

/// The settings that decide WHICH frames are hashed: `--keyframe-interval` and
/// `--min-keyframes`. Thread count is deliberately not here (it changes only
/// how fast frames arrive; `test_threaded_decode_is_bit_identical`), and
/// neither are `-d`, `-p` or `--min-duration`, which are comparison-time.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub(crate) struct Sampling {
    pub kf_interval: f64,
    pub min_kf_samples: f64,
}

impl Sampling {
    /// Whether a fingerprint made under `self` may be reused by a run sampling
    /// as `now` does. Asks the questions the decode asks rather than comparing
    /// the figures: `effective_interval` samples only when `kf_interval > 0.0`
    /// and floors only when `min_kf_samples > 0.0`, so 0, every negative and NaN
    /// are all "off" and none may invalidate another's entry (alternating
    /// `--keyframe-interval=-5` with the default used to re-decode the library
    /// every run). `--min-keyframes` is compared only while an interval is in
    /// force, since with none there is nothing for it to floor.
    fn matches(&self, now: &Sampling) -> bool {
        same_when_on(self.kf_interval, now.kf_interval)
            && (!on(now.kf_interval) || same_when_on(self.min_kf_samples, now.min_kf_samples))
    }
}

/// Whether a sampling setting does anything: the `> 0.0` test the decode makes,
/// NaN and all.
fn on(setting: f64) -> bool {
    setting > 0.0
}

/// Two settings that act only when positive are the same setting when both are
/// off, and otherwise when equal -- NaN included, since clap parses `nan` and
/// `NaN != NaN` would make such a run miss for ever.
fn same_when_on(a: f64, b: f64) -> bool {
    (!on(a) && !on(b)) || a == b || (a.is_nan() && b.is_nan())
}

/// What a lookup is checked against: the file's stat, as the scan took it, and
/// this run's sampling. Kept flat because that is how a run builds one.
///
/// Also the layout the pre-`TABLE` caches stored ahead of every value, which
/// `migrate` still reads: do not reorder the fields.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub(crate) struct Stamp {
    pub mtime: i64,
    pub mtime_nsec: i64,
    pub size: u64,
    pub kf_interval: f64,
    pub min_kf_samples: f64,
}

impl Stamp {
    fn stat(&self) -> Stat {
        Stat { mtime: self.mtime, mtime_nsec: self.mtime_nsec, size: self.size }
    }

    fn sampling(&self) -> Sampling {
        Sampling { kf_interval: self.kf_interval, min_kf_samples: self.min_kf_samples }
    }

    /// Whether something recorded at `self` answers for a file stamped `now`.
    /// Deliberately not `PartialEq`; see `Sampling::matches`. `View::lookup`
    /// asks the two halves apart; this is the same rule in one call, for tests.
    #[cfg(test)]
    pub fn matches(&self, now: &Stamp) -> bool {
        self.stat() == now.stat() && self.sampling().matches(&now.sampling())
    }
}

// --- What a record holds -------------------------------------------------------

/// Why a file will not be fingerprinted, in the form the cache keeps it.
///
/// Two shapes for a verdict because the common one deserves to be small. The
/// probe gate refuses hundreds of thousands of files in a single `-x '*'` run
/// and its sentence is two numbers in a fixed template, so it is stored as the
/// two numbers; everything else is a handful of files a run, and storing what
/// they actually said is both cheaper to write and impossible to get wrong.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) enum Refusal {
    /// Nothing in the file looked like media. Regenerated for printing, so the
    /// wording always belongs to the build doing the printing.
    NotMedia(fingerprint::NotMedia),
    /// It opened, and then something else was wrong with it -- no video stream,
    /// streams that would not parse, no frame that decoded. The message is the
    /// one the run printed when it found out.
    Said(String),
    /// The runtime the header reported, for a file the weighing pass found
    /// shorter than `--min-duration`.
    ///
    /// A MEASUREMENT rather than a verdict, because "too short" is not a fact
    /// about the file: `--min-duration` is comparison-time and kept out of the
    /// `Stamp`, so a remembered verdict would hand a skip to a later run that
    /// asked for a lower floor. Every run decides from the number, without
    /// opening the file; one that finds it long enough decodes it like any
    /// miss, and the fingerprint then replaces this record's verdict. Never a
    /// problem and never counted as one.
    ///
    /// bincode tags a variant by its position: append new ones, never reorder.
    TooShort(f64),
}

impl Refusal {
    /// Whether this is "no demuxer recognised it" rather than "it opened and
    /// then something went wrong" -- two findings with opposite responses, which
    /// the warm run's headline counts apart (`scan::Answered::headline`).
    pub fn is_not_media(&self) -> bool {
        matches!(self, Refusal::NotMedia(_))
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::NotMedia(verdict) => write!(f, "{}", verdict),
            Refusal::Said(said) => write!(f, "{}", said),
            // Never printed as a failure -- the cache pass takes this one apart
            // before anything asks it for a sentence.
            Refusal::TooShort(duration) => {
                write!(f, "shorter than --min-duration ({:.2}s)", duration)
            }
        }
    }
}

/// What to remember about a failure, if anything.
///
/// `None` for a failure that is about the moment rather than the file -- see
/// `fingerprint::is_transient`; a transient failure remembered is a file
/// skipped until it changes. This is the only gate between an error and the
/// cache, so both callers (the weighing pass and the decode) ask it.
pub(crate) fn worth_remembering(error: &anyhow::Error) -> Option<Refusal> {
    if let Some(verdict) = fingerprint::not_media(error) {
        return Some(Refusal::NotMedia(verdict));
    }
    if fingerprint::is_transient(error) {
        return None;
    }
    Some(Refusal::Said(format!("{:#}", error)))
}

/// An identity without its stat, which is the record's.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
struct Held {
    rules: u32,
    sample: content::Sample,
    full: Option<content::Digest>,
}

#[derive(Serialize, Deserialize)]
enum Verdict<F> {
    Refused { rules: u32, sampling: Sampling, refusal: Refusal },
    /// `fp` last of everything in the record; see the module comment.
    Fingerprint { sampling: Sampling, fp: F },
}

/// One record. Generic over the fingerprint so a write can serialize a
/// borrowed one; `&T` and `T` serialize to the same bytes.
#[derive(Serialize, Deserialize)]
struct Entry<F> {
    stat: Stat,
    identity: Option<Held>,
    verdict: Option<Verdict<F>>,
}

/// The front of an `Entry`, read without the verdict behind it. bincode's
/// top-level `deserialize` allows trailing bytes, which is what makes this
/// work (`test_a_record_s_head_reads_without_its_verdict`).
#[derive(Deserialize)]
struct Head {
    stat: Stat,
    identity: Option<Held>,
}

impl Head {
    fn identity(&self) -> Option<Identity> {
        let held = self.identity.filter(|h| h.rules == SAMPLE_RULES)?;
        Some(Identity {
            mtime: self.stat.mtime,
            mtime_nsec: self.stat.mtime_nsec,
            size: self.stat.size,
            sample: held.sample,
            full: held.full,
        })
    }
}

fn held(identity: &Identity) -> Held {
    Held { rules: SAMPLE_RULES, sample: identity.sample, full: identity.full }
}

// --- Reading -------------------------------------------------------------------

/// What the cache says about one file.
pub(crate) enum Known {
    Fingerprint(Box<VideoFingerprint>),
    Refused(Refusal),
    Nothing,
}

type ReadTable = redb::ReadOnlyTable<&'static str, &'static [u8]>;

/// One read snapshot of the cache.
///
/// Holding one pins every page it can see, so it is dropped the moment a pass
/// is done with it: held across the decode, it would stop redb reclaiming
/// anything the decode's writes supersede. The read transaction itself is not
/// kept -- `open_table` clones its guard into the table it returns.
pub(crate) struct View {
    table: Option<ReadTable>,
}

impl View {
    fn bytes<T>(&self, path: &str, read: impl FnOnce(&[u8]) -> Option<T>) -> Option<T> {
        let stored = self.table.as_ref()?.get(path).ok()??;
        read(stored.value())
    }

    /// The verdict recorded for this file, if it still answers for it.
    pub fn lookup(&self, path: &str, now: &Stamp) -> Known {
        let entry = self.bytes(path, |b| match bincode::deserialize::<Entry<VideoFingerprint>>(b) {
            Ok(entry) => Some(entry),
            Err(e) => {
                log::debug!("Cache entry for {} did not deserialize ({}); re-processing.", path, e);
                None
            }
        });
        let Some(Entry { stat, verdict: Some(verdict), .. }) = entry else { return Known::Nothing };
        if stat != now.stat() {
            return Known::Nothing;
        }
        match verdict {
            Verdict::Fingerprint { sampling, fp } if sampling.matches(&now.sampling()) => {
                Known::Fingerprint(Box::new(fp))
            }
            Verdict::Refused { rules, sampling, refusal }
                if rules == REFUSAL_RULES && sampling.matches(&now.sampling()) =>
            {
                Known::Refused(refusal)
            }
            _ => Known::Nothing,
        }
    }

    /// What this path held, as of the stat it was recorded at.
    pub fn identity(&self, path: &str) -> Option<Identity> {
        self.bytes(path, |b| bincode::deserialize::<Head>(b).ok()?.identity())
    }

    /// Every recorded identity. Reads each record's head only.
    pub fn identities(&self) -> Vec<(String, Identity)> {
        let Some(table) = self.table.as_ref() else { return Vec::new() };
        let Ok(iter) = table.iter() else { return Vec::new() };
        iter.flatten()
            .filter_map(|(key, value)| {
                let identity = bincode::deserialize::<Head>(value.value()).ok()?.identity()?;
                Some((key.value().to_string(), identity))
            })
            .collect()
    }

    /// The fingerprint recorded beside an identity `identities` returned for
    /// `path`, if it was made with this run's sampling. The two are one record,
    /// so the identity vouches for exactly this fingerprint and no other.
    pub fn fingerprint_held_with(&self, path: &str, now: &Sampling) -> Option<VideoFingerprint> {
        let entry = self.bytes(path, |b| bincode::deserialize::<Entry<VideoFingerprint>>(b).ok())?;
        entry.identity?;
        match entry.verdict? {
            Verdict::Fingerprint { sampling, fp } if sampling.matches(now) => Some(fp),
            _ => None,
        }
    }

    /// Every path the cache holds a record for, sorted. For tests.
    #[cfg(test)]
    pub fn paths(&self) -> Vec<String> {
        let table = self.table.as_ref().expect("the table exists");
        table.iter().unwrap().map(|e| e.unwrap().0.value().to_string()).collect()
    }

    /// Every record's verdict, whatever it was made under. For tests.
    #[cfg(test)]
    pub fn verdicts(&self) -> Vec<(String, Known)> {
        let table = self.table.as_ref().expect("the table exists");
        table
            .iter()
            .unwrap()
            .map(|e| {
                let (key, value) = e.unwrap();
                let entry: Entry<VideoFingerprint> = bincode::deserialize(value.value()).unwrap();
                let known = match entry.verdict {
                    Some(Verdict::Fingerprint { fp, .. }) => Known::Fingerprint(Box::new(fp)),
                    Some(Verdict::Refused { refusal, .. }) => Known::Refused(refusal),
                    None => Known::Nothing,
                };
                (key.value().to_string(), known)
            })
            .collect()
    }
}

// --- Writing -------------------------------------------------------------------

/// One thing learned about one path.
pub(crate) enum Write<'a> {
    Fingerprint(&'a str, Stamp, &'a VideoFingerprint),
    Refusal(&'a str, Stamp, Refusal),
    Identity(&'a str, Identity),
}

type WriteTable<'t> = redb::Table<'t, &'static str, &'static [u8]>;

fn put<F: Serialize>(table: &mut WriteTable<'_>, path: &str, entry: &Entry<F>) -> Result<()> {
    let encoded = bincode::serialize(entry).context("Failed to serialize a cache record")?;
    table.insert(path, encoded.as_slice()).context("Failed to write a cache record")?;
    Ok(())
}

/// The record at `path` if it is about `stat`; anything about another stat is
/// about other bytes and is dropped by the write that asked.
fn current(table: &WriteTable<'_>, path: &str, stat: Stat) -> Option<Entry<VideoFingerprint>> {
    let stored = table.get(path).ok()??;
    let entry = bincode::deserialize::<Entry<VideoFingerprint>>(stored.value()).ok()?;
    (entry.stat == stat).then_some(entry)
}

/// The identity recorded at `path` if it is about `stat`. Reads the head only.
fn identity_at(table: &WriteTable<'_>, path: &str, stat: Stat) -> Option<Held> {
    let stored = table.get(path).ok()??;
    let head = bincode::deserialize::<Head>(stored.value()).ok()?;
    if head.stat == stat { head.identity } else { None }
}

fn apply(table: &mut WriteTable<'_>, write: &Write<'_>) -> Result<()> {
    match write {
        Write::Fingerprint(path, stamp, fp) => {
            let stat = stamp.stat();
            let identity = identity_at(table, path, stat);
            let verdict = Verdict::Fingerprint { sampling: stamp.sampling(), fp: *fp };
            put(table, path, &Entry { stat, identity, verdict: Some(verdict) })
        }
        Write::Refusal(path, stamp, refusal) => {
            let stat = stamp.stat();
            let identity = identity_at(table, path, stat);
            let verdict: Verdict<&VideoFingerprint> = Verdict::Refused {
                rules: REFUSAL_RULES,
                sampling: stamp.sampling(),
                refusal: refusal.clone(),
            };
            put(table, path, &Entry { stat, identity, verdict: Some(verdict) })
        }
        Write::Identity(path, identity) => {
            let stat = Stat::of(identity);
            let verdict = current(table, path, stat).and_then(|e| e.verdict);
            put(table, path, &Entry { stat, identity: Some(held(identity)), verdict })
        }
    }
}

// --- The cache -----------------------------------------------------------------

/// What `Cache::prepare` found to do.
#[derive(Default)]
pub(crate) struct Prepared {
    /// A superseded fingerprint table went: the library is decoded again.
    pub rebuilt: bool,
    /// Records carried over from the pre-`TABLE` layout.
    pub converted: usize,
    /// Something was dropped, so a compaction has pages to hand back.
    pub reclaimable: bool,
}

/// Why the cache could not be opened, and whether the run has to stop for it.
///
/// `locked` is the one failure a mode can survive: it says another run holds
/// the file, not that anything is wrong with the file. Only `--from-report`
/// uses the distinction; see `main_with`.
pub(crate) struct CacheUnavailable {
    pub locked: bool,
    /// The file is there but is not a cache this build can read: damaged, or
    /// not a redb file at all. `--clear-cache` sets it aside and starts afresh.
    pub damaged: bool,
    pub reason: anyhow::Error,
}

pub(crate) struct Cache {
    db: Database,
    /// `--no-cache`: an empty database in memory that nothing is written to,
    /// so every lookup misses and the run keeps nothing. See `Cache::none`.
    discards: bool,
}

impl Cache {
    /// Open the cache, and say something useful when that fails.
    ///
    /// A single file, an explicitly bounded page cache, and no background
    /// threads: every write is durable when its transaction commits, so there
    /// is no buffered work to lose.
    ///
    /// Opening takes an exclusive `flock` for as long as the handle lives,
    /// which is the whole run -- so two concurrent scans cannot share one, and
    /// `--cache` is the only way to run them at all. redb reports that case as
    /// its own variant, so each branch here names the file and what to do.
    pub fn open(db_path: &Path) -> std::result::Result<Self, CacheUnavailable> {
        let fatal = |reason| CacheUnavailable { locked: false, damaged: false, reason };

        match Database::builder().set_cache_size(CACHE_SIZE_BYTES).create(db_path) {
            Ok(db) => Ok(Self { db, discards: false }),
            Err(DatabaseError::DatabaseAlreadyOpen) => Err(CacheUnavailable {
                locked: true,
                damaged: false,
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
            // Not a cache: a file that is not redb at all, or one a crash or a
            // full disk left unreadable. Every later run would fail the same
            // way, and `--clear-cache` could not help, since it needs the
            // database open -- so it is the one flag told to set the file aside.
            Err(DatabaseError::Storage(StorageError::Corrupted(_))) if is_regular(db_path) => {
                Err(damaged(db_path))
            }
            Err(DatabaseError::Storage(StorageError::Io(e)))
                if matches!(e.kind(), std::io::ErrorKind::InvalidData | std::io::ErrorKind::UnexpectedEof)
                    && is_regular(db_path) =>
            {
                Err(damaged(db_path))
            }
            Err(e) => Err(fatal(anyhow::Error::new(e).context(format!(
                "Failed to open the fingerprint cache at {}",
                db_path.display()
            )))),
        }
    }

    /// The cache of a run under `--no-cache`: it holds nothing and keeps
    /// nothing. Every lookup misses, every write is dropped, so nothing the
    /// run learns accumulates in memory either -- the fingerprints the run
    /// needs are held by the run itself. One empty in-memory database rather
    /// than an `Option` at every call site, so the scan reads the same way
    /// with or without a cache.
    pub fn none() -> Result<Self> {
        let db = Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .context("Failed to set up a run without a cache")?;
        let cache = Self { db, discards: false };
        cache.prepare()?;
        Ok(Self { discards: true, ..cache })
    }

    /// A cache in a file nothing else holds, ready to use. For tests.
    #[cfg(test)]
    pub fn create(db_path: &Path) -> Self {
        let cache = Self { db: Database::create(db_path).unwrap(), discards: false };
        cache.prepare().unwrap();
        cache
    }

    #[cfg(test)]
    pub fn database(&mut self) -> &mut Database {
        &mut self.db
    }

    /// Make the cache this build's: create `TABLE`, carry the pre-`TABLE`
    /// tables over into it, and drop every table a build before this one left.
    ///
    /// `TABLE` is created here because a read transaction cannot create a
    /// table, and a missing one would turn every lookup on a new machine into
    /// an error path.
    pub fn prepare(&self) -> Result<Prepared> {
        let txn = self.db.begin_write().context("Failed to start a cache transaction")?;
        let present: HashSet<String> = txn
            .list_tables()
            .context("Failed to list the cache's tables")?
            .map(|t| redb::TableHandle::name(&t).to_string())
            .collect();
        let has = |t: &TableDefinition<&str, &[u8]>| present.contains(redb::TableHandle::name(t));

        let mut prepared = Prepared::default();
        {
            let mut table = txn.open_table(TABLE).context("Failed to create the cache table")?;
            if has(&LEGACY_FINGERPRINTS) {
                let old = txn.open_table(LEGACY_FINGERPRINTS)?;
                prepared.converted += migrate_fingerprints(&old, &mut table)?;
            }
            if has(&LEGACY_REFUSALS) {
                let old = txn.open_table(LEGACY_REFUSALS)?;
                prepared.converted += migrate_refusals(&old, &mut table)?;
            }
            if has(&LEGACY_IDENTITIES) {
                let old = txn.open_table(LEGACY_IDENTITIES)?;
                migrate_identities(&old, &mut table)?;
            }
        }

        let mut drop_table = |t: TableDefinition<&str, &[u8]>| -> Result<bool> {
            let dropped = txn.delete_table(t).context("Failed to remove a superseded cache table")?;
            prepared.reclaimable |= dropped;
            Ok(dropped)
        };
        for t in [LEGACY_FINGERPRINTS, LEGACY_REFUSALS, LEGACY_IDENTITIES] {
            drop_table(t)?;
        }
        let mut rebuilt = false;
        for t in SUPERSEDED_TABLES {
            rebuilt |= drop_table(t)?;
        }
        for t in SUPERSEDED_REFUSAL_TABLES {
            drop_table(t)?;
        }
        prepared.rebuilt = rebuilt;
        txn.commit().context("Failed to commit the cache's preparation")?;
        Ok(prepared)
    }

    /// Empty the cache -- every table in the file, a cache shared with another
    /// architecture's tables included -- and compact it while doing so is free:
    /// with no live pages, the rewrite costs microseconds and returns the whole
    /// file. Not fatal if the compaction fails: an uncompacted empty cache is
    /// still a correct one.
    pub fn clear(&mut self) -> Result<()> {
        let txn = self.db.begin_write().context("Failed to start a cache transaction")?;
        let tables: Vec<_> = txn.list_tables().context("Failed to list the cache's tables")?.collect();
        for table in tables {
            txn.delete_table(table).context("Failed to clear the cache")?;
        }
        txn.commit().context("Failed to commit the cache clear")?;
        self.prepare().context("Failed to recreate the cache table")?;
        if let Err(e) = self.db.compact() {
            log::error!("Could not compact the fingerprint cache: {}", e);
        }
        Ok(())
    }

    /// Hand the pages emptied tables left behind back to the filesystem, by
    /// copying every live page into a fresh file. The most expensive thing a
    /// warm run can do, so only asked for when something was removed.
    pub fn compact(&mut self) -> Result<bool> {
        self.db.compact().context("Could not compact the fingerprint cache")
    }

    pub fn view(&self) -> View {
        let table = self.db.begin_read().ok().and_then(|read| read.open_table(TABLE).ok());
        View { table }
    }

    /// Record what was learned, in one transaction.
    ///
    /// The one rule every write obeys: a record is about one stat, so writing
    /// at another starts it afresh, and it holds one verdict, so a verdict
    /// replaces the one before it. An identity written at the record's own
    /// stat keeps the verdict beside it; a verdict keeps the identity.
    ///
    /// The caller decides the batch. A decode commits each video on its own,
    /// so an interrupt can only cost the decode in flight; refusals are
    /// batched per pass, because there can be a quarter of a million of them
    /// and one fsync each would cost more than re-probing next time.
    pub fn write(&self, writes: &[Write<'_>]) -> Result<()> {
        if writes.is_empty() || self.discards {
            return Ok(());
        }
        let txn = self.db.begin_write().context("Failed to start a cache transaction")?;
        {
            let mut table = txn.open_table(TABLE).context("Failed to open the cache table")?;
            for write in writes {
                apply(&mut table, write)?;
            }
        }
        txn.commit().context("Failed to commit to the cache")?;
        Ok(())
    }

    /// Drop the records of files that are gone, and say how many there were.
    pub fn forget(&self, paths: &[String]) -> Result<usize> {
        if paths.is_empty() {
            return Ok(0);
        }
        let mut forgotten = 0;
        let txn = self.db.begin_write().context("Failed to start a cache transaction")?;
        {
            let mut table = txn.open_table(TABLE).context("Failed to open the cache table")?;
            for path in paths {
                if table.remove(path.as_str()).context("Failed to remove a cache record")?.is_some() {
                    forgotten += 1;
                }
            }
        }
        txn.commit().context("Failed to commit the cache removals")?;
        Ok(forgotten)
    }

    /// Drop every record whose path is not in `keep`. `None` if interrupted,
    /// with nothing removed.
    pub fn prune(&self, keep: &HashSet<&str>) -> Result<Option<usize>> {
        let mut stale: Vec<String> = Vec::new();
        {
            let read = self.db.begin_read().context("Failed to open the cache for reading")?;
            let table = read.open_table(TABLE).context("Failed to open the cache table")?;
            for entry in table.iter().context("Failed to iterate the cache")? {
                if shutdown_requested() {
                    return Ok(None);
                }
                let (key, _) = entry.context("Failed to read a cache record")?;
                if !keep.contains(key.value()) {
                    stale.push(key.value().to_string());
                }
            }
        }
        self.forget(&stale).map(Some)
    }
}

// --- The pre-`TABLE` layout ----------------------------------------------------

/// `(Stamp, VideoFingerprint)` per path. Entries with no hashes are not
/// carried over: only an older build wrote them (an all-featureless file used
/// to fingerprint to nothing instead of failing), so the file is decoded once
/// more and fails out loud.
fn migrate_fingerprints(old: &WriteTable<'_>, table: &mut WriteTable<'_>) -> Result<usize> {
    let mut carried = 0;
    for entry in old.iter()? {
        let (key, value) = entry?;
        let Ok((stamp, fp)) = bincode::deserialize::<(Stamp, VideoFingerprint)>(value.value()) else {
            continue;
        };
        if fp.valid_hashes.is_empty() || table.get(key.value())?.is_some() {
            continue;
        }
        let verdict = Verdict::Fingerprint { sampling: stamp.sampling(), fp };
        put(table, key.value(), &Entry { stat: stamp.stat(), identity: None, verdict: Some(verdict) })?;
        carried += 1;
    }
    Ok(carried)
}

/// `(Stamp, Refusal)` per path, written under the rules this build has
/// (`refused_ff8_probe6` is `REFUSAL_RULES` 6). A path that already has a
/// record keeps it: the two tables were kept exclusive, so a clash is a
/// record this build wrote, which is the newer.
fn migrate_refusals(old: &WriteTable<'_>, table: &mut WriteTable<'_>) -> Result<usize> {
    let mut carried = 0;
    for entry in old.iter()? {
        let (key, value) = entry?;
        let Ok((stamp, refusal)) = bincode::deserialize::<(Stamp, Refusal)>(value.value()) else {
            continue;
        };
        if table.get(key.value())?.is_some() {
            continue;
        }
        let verdict: Verdict<VideoFingerprint> =
            Verdict::Refused { rules: REFUSAL_RULES, sampling: stamp.sampling(), refusal };
        put(table, key.value(), &Entry { stat: stamp.stat(), identity: None, verdict: Some(verdict) })?;
        carried += 1;
    }
    Ok(carried)
}

/// `content::Identity` per path, which carried a stat of its own. It joins the
/// record only when the two stats agree -- exactly the condition under which
/// the old layout let it vouch for that fingerprint -- or makes a record of its
/// own where there was none.
fn migrate_identities(old: &WriteTable<'_>, table: &mut WriteTable<'_>) -> Result<()> {
    for entry in old.iter()? {
        let (key, value) = entry?;
        let Ok(identity) = bincode::deserialize::<Identity>(value.value()) else { continue };
        let path = key.value();
        let stat = Stat::of(&identity);
        let exists = table.get(path)?.is_some();
        match current(table, path, stat) {
            Some(entry) if entry.identity.is_none() => {
                put(table, path, &Entry { identity: Some(held(&identity)), ..entry })?;
            }
            None if !exists => {
                put(table, path, &Entry::<VideoFingerprint> {
                    stat,
                    identity: Some(held(&identity)),
                    verdict: None,
                })?;
            }
            _ => {}
        }
    }
    Ok(())
}

// --- Where the cache lives -----------------------------------------------------

/// Delete the sled cache this tool used to keep, if it is still lying around.
/// sled stored a directory; redb stores a single file, so an old cache is
/// bytes nothing will ever read again.
fn retire_legacy_cache(cache_dir: &Path) {
    let legacy = cache_dir.join("video_hashes.db");
    if !legacy.is_dir() {
        return;
    }
    match std::fs::remove_dir_all(&legacy) {
        Ok(()) => info!("Removed the superseded cache at {}.", legacy.display()),
        Err(e) => log::debug!("Could not remove the superseded cache at {}: {}", legacy.display(), e),
    }
}

/// Where the cache lives when `--cache` does not say: the XDG Base Directory
/// Specification's cache folder, `vid-fp` in it.
fn default_cache_dir() -> PathBuf {
    std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home).join(".cache")
            } else {
                PathBuf::from("/tmp")
            }
        })
        .join("vid-fp")
}

/// The cache file a run without `--cache` uses, for the desktop app to show
/// where it is. Nothing is created.
pub fn default_cache_file() -> PathBuf {
    default_cache_dir().join(CACHE_FILE_NAME)
}

/// Decide which file this run's fingerprints are cached in, creating its
/// folder. `--cache` names the database FILE; a path that is already a
/// directory, or written with a trailing slash, gets the default filename
/// inside it. Only the default location gets the legacy sweep: a cache the
/// user named is theirs.
pub(crate) fn resolve_cache_path(explicit: Option<&str>) -> Result<PathBuf> {
    let path = cache_file_for(explicit);
    let Some(_) = explicit else {
        let cache_dir = default_cache_dir();
        std::fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;
        retire_legacy_cache(&cache_dir);
        return Ok(path);
    };
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("Failed to create the cache directory {}", parent.display())
        })?;
    }
    Ok(path)
}

/// The cache file `resolve_cache_path` will open, worked out without creating
/// anything, so it can be compared with the report and log paths first.
pub(crate) fn cache_file_for(explicit: Option<&str>) -> PathBuf {
    let Some(raw) = explicit else {
        return default_cache_dir().join(CACHE_FILE_NAME);
    };
    let given = Path::new(raw);
    if raw.ends_with('/') || given.is_dir() {
        given.join(CACHE_FILE_NAME)
    } else {
        given.to_path_buf()
    }
}

/// The refusal for a cache file that is there and cannot be read as one.
fn damaged(db_path: &Path) -> CacheUnavailable {
    CacheUnavailable {
        locked: false,
        damaged: true,
        reason: anyhow::anyhow!(
            "The fingerprint cache at {} is damaged or is not a vid-fp cache.\n\
             Run again with --clear-cache to set it aside as {} and start a new one \
             (every video will be fingerprinted again), or delete it.",
            db_path.display(),
            set_aside_path(db_path).display()
        ),
    }
}

/// Only a regular file is ever called damaged: anything else at the cache
/// path is a mistake of another kind, and is not moved.
fn is_regular(db_path: &Path) -> bool {
    std::fs::symlink_metadata(db_path).is_ok_and(|m| m.is_file())
}

/// Where `--clear-cache` moves a damaged cache: beside it, under a name that
/// is not taken.
pub(crate) fn set_aside_path(db_path: &Path) -> PathBuf {
    let mut name = db_path.file_name().unwrap_or_default().to_os_string();
    name.push(".damaged");
    let first = db_path.with_file_name(&name);
    if std::fs::symlink_metadata(&first).is_err() {
        return first;
    }
    (1..)
        .map(|n| {
            let mut numbered = name.clone();
            numbered.push(format!(".{}", n));
            db_path.with_file_name(numbered)
        })
        .find(|p| std::fs::symlink_metadata(p).is_err())
        .expect("some numbered name is free")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{mock_fp, stamp};

    /// A cache of its own, inside a directory that cleans itself up.
    fn temp_db(dir: &tempfile::TempDir) -> Cache {
        Cache::create(&dir.path().join(CACHE_FILE_NAME))
    }

    fn store_fp(db: &Cache, path: &str, stamp: Stamp, fp: &VideoFingerprint) -> Result<()> {
        db.write(&[Write::Fingerprint(path, stamp, fp)])
    }

    fn store_refusals(db: &Cache, refused: &[(String, Stamp, Refusal)]) -> Result<()> {
        let writes: Vec<Write> =
            refused.iter().map(|(p, s, r)| Write::Refusal(p, *s, r.clone())).collect();
        db.write(&writes)
    }

    fn hit(db: &Cache, path: &str, now: &Stamp) -> Option<VideoFingerprint> {
        match db.view().lookup(path, now) {
            Known::Fingerprint(fp) => Some(*fp),
            _ => None,
        }
    }

    fn refusal(db: &Cache, path: &str, now: &Stamp) -> Option<Refusal> {
        match db.view().lookup(path, now) {
            Known::Refused(r) => Some(r),
            _ => None,
        }
    }

    fn identity_of(stamp: Stamp, byte: u8) -> Identity {
        Identity {
            mtime: stamp.mtime,
            mtime_nsec: stamp.mtime_nsec,
            size: stamp.size,
            sample: content::Sample([byte; 16]),
            full: None,
        }
    }

    fn not_media() -> Refusal {
        Refusal::NotMedia(fingerprint::NotMedia { bytes: 2048, score: 0 })
    }

    /// The moved-file index walks every record's head and never its verdict,
    /// which rests on bincode's top-level `deserialize` accepting the bytes
    /// left over behind a prefix.
    #[test]
    fn test_a_record_s_head_reads_without_its_verdict() {
        let s = stamp(1_700_000_000, 12_345);
        let fp = mock_fp("/videos/a.mkv");
        let entry = Entry {
            stat: s.stat(),
            identity: Some(held(&identity_of(s, 7))),
            verdict: Some(Verdict::Fingerprint { sampling: s.sampling(), fp: &fp }),
        };
        let bytes = bincode::serialize(&entry).unwrap();

        let head: Head = bincode::deserialize(&bytes).expect("trailing bytes are allowed");
        assert_eq!(head.stat, s.stat());
        assert_eq!(head.identity().unwrap(), identity_of(s, 7));

        // And a borrowed fingerprint writes the bytes an owned one reads.
        let owned: Entry<VideoFingerprint> = bincode::deserialize(&bytes).unwrap();
        assert!(matches!(owned.verdict, Some(Verdict::Fingerprint { fp, .. }) if fp.valid_hashes == vec![1, 2, 3]));
    }

    /// The one write rule, against a model of it: a record is about one stat,
    /// so a write at another starts it afresh; it holds one verdict, so a
    /// verdict replaces the last; and an identity and a verdict at the same
    /// stat keep each other. Driven through random sequences, so it says what
    /// every interleaving does rather than the few a hand-written test thinks of.
    #[test]
    fn test_every_sequence_of_writes_leaves_one_record_about_one_stat() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        enum Kind {
            Fingerprint,
            Refused,
        }
        #[derive(Clone, Copy)]
        struct Model {
            stat: usize,
            identity: Option<u8>,
            verdict: Option<Kind>,
        }

        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let paths = ["/v/a.mkv", "/v/b.mkv", "/v/c.mkv"];
        let stats = [stamp(1_700_000_000, 100), stamp(1_700_000_000, 200), stamp(1_700_000_001, 100)];
        let fp = mock_fp("/v/x.mkv");
        let mut model: [Option<Model>; 3] = [None; 3];

        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n) as usize
        };

        for step in 0..3_000 {
            let (p, s, op, byte) = (next(3), next(3), next(4), next(250) as u8);
            let fresh = |m: Option<Model>| m.filter(|m| m.stat == s).unwrap_or(Model { stat: s, identity: None, verdict: None });
            match op {
                0 => {
                    db.write(&[Write::Fingerprint(paths[p], stats[s], &fp)]).unwrap();
                    model[p] = Some(Model { verdict: Some(Kind::Fingerprint), ..fresh(model[p]) });
                }
                1 => {
                    db.write(&[Write::Refusal(paths[p], stats[s], not_media())]).unwrap();
                    model[p] = Some(Model { verdict: Some(Kind::Refused), ..fresh(model[p]) });
                }
                2 => {
                    db.write(&[Write::Identity(paths[p], identity_of(stats[s], byte))]).unwrap();
                    model[p] = Some(Model { identity: Some(byte), ..fresh(model[p]) });
                }
                _ => {
                    db.forget(&[paths[p].to_string()]).unwrap();
                    model[p] = None;
                }
            }

            let view = db.view();
            for (p, path) in paths.iter().enumerate() {
                for (s, at) in stats.iter().enumerate() {
                    let found = match view.lookup(path, at) {
                        Known::Fingerprint(_) => Some(Kind::Fingerprint),
                        Known::Refused(_) => Some(Kind::Refused),
                        Known::Nothing => None,
                    };
                    let due = model[p].filter(|m| m.stat == s).and_then(|m| m.verdict);
                    assert_eq!(found, due, "step {step}: {path} at stat {s}");
                }
                let identity = view.identity(path);
                let due = model[p].and_then(|m| Some(identity_of(stats[m.stat], m.identity?)));
                assert_eq!(identity, due, "step {step}: {path}'s identity");
            }
        }
    }

    /// A refusal is retired by the rules it was made under, not by a table.
    #[test]
    fn test_a_refusal_made_under_other_rules_is_asked_again() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = temp_db(&dir);
        let s = stamp(1_700_000_000, 12_345);
        let entry: Entry<VideoFingerprint> = Entry {
            stat: s.stat(),
            identity: None,
            verdict: Some(Verdict::Refused { rules: REFUSAL_RULES - 1, sampling: s.sampling(), refusal: not_media() }),
        };
        let txn = db.database().begin_write().unwrap();
        put(&mut txn.open_table(TABLE).unwrap(), "/v/a.o", &entry).unwrap();
        txn.commit().unwrap();

        assert!(refusal(&db, "/v/a.o", &s).is_none());
    }

    /// An identity vouches only for the fingerprint in its own record, and only
    /// for a run that samples the way that fingerprint was made.
    #[test]
    fn test_only_a_fingerprint_held_beside_an_identity_can_be_taken_over() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let s = stamp(1_700_000_000, 12_345);
        let fp = mock_fp("/v/a.mkv");
        db.write(&[Write::Fingerprint("/v/a.mkv", s, &fp), Write::Fingerprint("/v/b.mkv", s, &fp)]).unwrap();
        db.write(&[Write::Identity("/v/a.mkv", identity_of(s, 1))]).unwrap();

        let view = db.view();
        assert!(view.fingerprint_held_with("/v/a.mkv", &s.sampling()).is_some());
        assert!(view.fingerprint_held_with("/v/b.mkv", &s.sampling()).is_none(), "no identity, nothing to vouch");
        let sampled = Sampling { kf_interval: 5.0, ..s.sampling() };
        assert!(view.fingerprint_held_with("/v/a.mkv", &sampled).is_none(), "other frames");
        assert_eq!(view.identities(), vec![("/v/a.mkv".to_string(), identity_of(s, 1))]);
    }

    /// The pre-`TABLE` layout, carried over once and dropped. Each of the three
    /// old tables, and the cases the split allowed that one record cannot hold.
    #[test]
    fn test_the_three_old_tables_are_carried_over_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CACHE_FILE_NAME);
        let s = stamp(1_700_000_000, 12_345);
        let later = stamp(1_700_000_009, 12_345);
        let mut empty = mock_fp("/v/black.mkv");
        empty.valid_hashes.clear();
        {
            let db = redb::Database::create(&path).unwrap();
            let txn = db.begin_write().unwrap();
            {
                let mut fps = txn.open_table(LEGACY_FINGERPRINTS).unwrap();
                for (p, st, fp) in [
                    ("/v/a.mkv", s, mock_fp("/v/a.mkv")),
                    ("/v/black.mkv", s, empty.clone()),
                    ("/v/stale_id.mkv", s, mock_fp("/v/stale_id.mkv")),
                    ("/v/both.mkv", s, mock_fp("/v/both.mkv")),
                ] {
                    fps.insert(p, bincode::serialize(&(st, fp)).unwrap().as_slice()).unwrap();
                }
                let mut refused = txn.open_table(LEGACY_REFUSALS).unwrap();
                for p in ["/v/a.o", "/v/both.mkv"] {
                    refused.insert(p, bincode::serialize(&(s, not_media())).unwrap().as_slice()).unwrap();
                }
                let mut ids = txn.open_table(LEGACY_IDENTITIES).unwrap();
                for (p, id) in [
                    ("/v/a.mkv", identity_of(s, 1)),
                    ("/v/stale_id.mkv", identity_of(later, 2)),
                    ("/v/only_id.mkv", identity_of(later, 3)),
                ] {
                    ids.insert(p, bincode::serialize(&id).unwrap().as_slice()).unwrap();
                }
                txn.open_table(SUPERSEDED_REFUSAL_TABLES[0]).unwrap().insert("/v/x", &b"x"[..]).unwrap();
            }
            txn.commit().unwrap();
        }

        let db = Cache { db: redb::Database::create(&path).unwrap(), discards: false };
        let prepared = db.prepare().unwrap();
        assert_eq!(prepared.converted, 4, "three fingerprints and one refusal");
        assert!(!prepared.rebuilt, "nothing is decoded again for it");
        assert!(prepared.reclaimable);

        assert!(hit(&db, "/v/a.mkv", &s).is_some());
        assert!(hit(&db, "/v/black.mkv", &s).is_none(), "an empty fingerprint is decoded again, and fails out loud");
        assert!(refusal(&db, "/v/a.o", &s).is_some());
        assert!(hit(&db, "/v/both.mkv", &s).is_some(), "one verdict, and the fingerprint is the newer");

        let view = db.view();
        assert_eq!(view.identity("/v/a.mkv"), Some(identity_of(s, 1)), "joins the record it vouched for");
        assert_eq!(view.identity("/v/stale_id.mkv"), None, "about other bytes than the fingerprint");
        assert!(hit(&db, "/v/stale_id.mkv", &s).is_some(), "which still answers by path");
        assert_eq!(view.identity("/v/only_id.mkv"), Some(identity_of(later, 3)), "a record of its own");
        drop(view);

        let names: Vec<String> = db
            .db
            .begin_read()
            .unwrap()
            .list_tables()
            .unwrap()
            .map(|t| redb::TableHandle::name(&t).to_string())
            .collect();
        assert_eq!(names, vec![redb::TableHandle::name(&TABLE).to_string()]);

        let again = db.prepare().unwrap();
        assert_eq!((again.converted, again.rebuilt, again.reclaimable), (0, false, false));
    }

    /// A superseded fingerprint table is the one thing worth telling the user
    /// about: it is a re-decode of the library.
    #[test]
    fn test_only_a_superseded_fingerprint_table_says_the_library_is_rebuilt() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = temp_db(&dir);
        let txn = db.database().begin_write().unwrap();
        txn.open_table(SUPERSEDED_TABLES[14]).unwrap().insert("/v/a", &b"x"[..]).unwrap();
        txn.commit().unwrap();

        let prepared = db.prepare().unwrap();
        assert!(prepared.rebuilt && prepared.reclaimable);
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

    #[test]
    fn test_every_floor_that_floors_nothing_is_the_same_setting() {
        // One rung down from the interval test above, and the same argument:
        // `effective_interval` computes the floor only when
        // `min_kf_samples > 0.0`, so 0.0, a negative and NaN all leave the
        // interval exactly as typed and produce identical fingerprints. Told
        // apart, `-i 5 -m 0` and `-i 5 -m nan` would each overwrite the other's
        // entry and neither would ever hit.
        let sampled = Stamp { kf_interval: 5.0, ..stamp(1_700_000_000, 12_345) };
        let unfloored = [
            Stamp { min_kf_samples: 0.0, ..sampled },
            Stamp { min_kf_samples: -5.0, ..sampled },
            Stamp { min_kf_samples: f64::NAN, ..sampled },
        ];

        for a in unfloored {
            for b in unfloored {
                assert!(a.matches(&b), "both leave the 5 s interval standing alone");
            }
            // A real floor is a real change, in both directions: it makes the
            // interval finer for anything shorter than 5 s x 12.
            assert!(!a.matches(&sampled), "12 samples is a floor, none is not");
            assert!(!sampled.matches(&a));
        }
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

        store_fp(&db, &fp.path, written, &fp).unwrap();

        let hit = hit(&db, &fp.path, &written).expect("an untouched file must hit");
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

        store_fp(&db, path, old, &mock_fp(path)).unwrap();
        store_fp(&db, path, new, &mock_fp(path)).unwrap();

        assert!(hit(&db, path, &new).is_some(), "the file as it stands now hits");
        assert!(hit(&db, path, &old).is_none(), "the superseded stamp cannot");

        assert_eq!(db.view().paths(), vec![path], "one file, one entry, however often it changes");
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

        store_fp(&db, kept, s, &mock_fp(kept)).unwrap();
        store_fp(&db, gone, s, &mock_fp(gone)).unwrap();

        let forgotten = db.forget(&[gone.to_string()]).unwrap();

        assert_eq!(forgotten, 1);
        assert!(hit(&db, gone, &s).is_none(), "a removed file keeps no fingerprint");
        assert!(hit(&db, kept, &s).is_some(), "and its neighbours are untouched");
    }

    /// A refusal is a record like a fingerprint is, and forgetting one counts.
    #[test]
    fn test_a_forgotten_path_is_counted_whatever_its_record_held() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let refused_path = "/videos/not_really_a_video.mkv";
        let s = stamp(1_700_000_000, 12_345);

        store_refusals(
            &db,
            &[(refused_path.to_string(), s, Refusal::Said("nope".to_string()))],
        )
        .unwrap();

        let forgotten = db.forget(&[refused_path.to_string()]).unwrap();

        assert_eq!(forgotten, 1, "the cache had something to say about this path");
        assert!(
            refusal(&db, refused_path, &s).is_none(),
            "and it no longer says it"
        );

        assert_eq!(
            db.forget(&["/videos/never_seen.mkv".to_string()]).unwrap(),
            0,
            "a path with no record is still not an entry"
        );
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

        store_refusals(&db, &[(path.to_string(), written, verdict)]).unwrap();

        let remembered = refusal(&db, path, &written).expect("the same file is remembered");
        assert!(format!("{}", remembered).contains("2048"), "and says what it saw");

        assert!(
            refusal(&db, path, &stamp(1_700_000_001, 12_345)).is_none(),
            "an edit re-opens the question"
        );
        assert!(
            refusal(&db, path, &stamp(1_700_000_000, 12_346)).is_none(),
            "and so does a change of size"
        );
    }

    /// What the cache keeps about a file under `--min-duration` is its RUNTIME,
    /// not the skip -- because the threshold belongs to the run and is kept out
    /// of the `Stamp` along with every other comparison-time flag. Remembering
    /// the verdict would hand a run that lowered the flag a skip it never asked
    /// for, and there is no bit in the stamp that would catch it.
    #[test]
    fn test_a_remembered_runtime_is_re_decided_against_each_run_s_floor() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let path = "/videos/clip.mp4";
        let written = stamp(1_700_000_000, 12_345);

        store_refusals(&db, &[(path.to_string(), written, Refusal::TooShort(8.84))]).unwrap();

        let Some(Refusal::TooShort(duration)) = refusal(&db, path, &written) else {
            panic!("the runtime is what comes back");
        };
        assert_eq!(duration, 8.84, "verbatim, so every run can decide for itself");

        // Which is the whole point: one threshold skips this file and the other
        // sends it to the decode, off one entry neither run had to open a file
        // to read.
        assert!(duration < 60.0, "--min-duration 60 skips it");
        assert!(duration >= 5.0, "--min-duration 5 does not");

        assert!(
            refusal(&db, path, &stamp(1_700_000_001, 12_345)).is_none(),
            "and an edit re-opens the question, exactly as for a refusal"
        );
    }

    /// One path, one table. A `TooShort` entry is the only one a fingerprint can
    /// ever land on top of -- the other two verdicts never fall through to a
    /// decode -- and the run that lowers `--min-duration` is exactly the run
    /// that does it: the file is decoded and filed under the same path the
    /// refusal is under.
    ///
    /// Both halves are asserted because the failure is silent from either side
    /// on its own. A leftover refusal is not stale (its `Stamp` still matches
    /// the file, so nothing collects it), and the cache pass read a remembered
    /// runtime BEFORE `cache_lookup`, so the header runtime it holds would outrank the
    /// runtime the decode went on to measure for as long as the entry lived.
    #[test]
    fn test_a_fingerprint_supersedes_the_refusal_it_fell_through() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let path = "/videos/clip.mp4";
        let written = stamp(1_700_000_000, 12_345);

        // Run one, `--min-duration 60`: the weighing pass measures 8.84s and
        // files the number rather than the verdict.
        store_refusals(&db, &[(path.to_string(), written, Refusal::TooShort(8.84))]).unwrap();
        assert!(refusal(&db, path, &written).is_some(), "which run two will read");

        // Run two, `--min-duration 0`: the number clears this run's floor, so
        // the file falls through, decodes, and is stored.
        let mut fp = mock_fp(path);
        fp.duration = 8.84;
        store_fp(&db, path, written, &fp).unwrap();

        assert!(
            hit(&db, path, &written).is_some(),
            "the fingerprint is what this path means now"
        );
        assert!(
            refusal(&db, path, &written).is_none(),
            "and the measurement it superseded went with it"
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
        store_refusals(
            &db,
            &[(
                path.to_string(),
                written,
                Refusal::NotMedia(fingerprint::NotMedia { bytes: 16384, score: 1 }),
            )],
        )
        .unwrap();

        db.forget(&[path.to_string()]).unwrap();

        assert!(refusal(&db, path, &written).is_none());
    }

    #[test]
    fn test_forgetting_something_uncached_is_not_an_error() {
        // A file whose cache write failed earlier in the run is still a
        // perfectly good target, and its absence here means nothing.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);

        assert_eq!(db.forget(&[]).unwrap(), 0);
        assert_eq!(db.forget(&["/videos/never-cached.mkv".to_string()]).unwrap(), 0);
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

        store_fp(&db, path, s, &mock_fp(path)).unwrap();
        db.clear().unwrap();

        assert!(hit(&db, path, &s).is_none(), "nothing survives a clear");

        store_fp(&db, path, s, &mock_fp(path)).unwrap();
        assert!(
            hit(&db, path, &s).is_some(),
            "and the table is ready for the scan that follows"
        );
    }

    #[test]
    fn test_clearing_the_cache_clears_another_architecture_s_tables_too() {
        // A cache on a share an ARM and an x86 machine both scan holds a
        // fingerprint table for each; --clear-cache from either empties both.
        let dir = tempfile::tempdir().unwrap();
        let mut db = temp_db(&dir);
        let other: TableDefinition<&str, &[u8]> = TableDefinition::new("fingerprints_dct_ct_ff8_s128_riscv64");
        let txn = db.database().begin_write().unwrap();
        txn.open_table(other).unwrap().insert("/videos/a.mkv", &b"x"[..]).unwrap();
        txn.commit().unwrap();

        db.clear().unwrap();

        let names: Vec<String> =
            db.database().begin_read().unwrap().list_tables().unwrap().map(|t| redb::TableHandle::name(&t).to_string()).collect();
        assert!(!names.iter().any(|n| n.ends_with("_riscv64")), "{names:?}");
    }

    #[test]
    fn test_a_locked_cache_names_the_file_and_the_way_out() {
        // Scanning two libraries at once is ordinary, and this is what the
        // second run hits. flock conflicts between two open file descriptions
        // whether or not they belong to the same process, so holding `_first`
        // here reproduces the concurrent run exactly.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fingerprints.redb");
        let _first = redb::Database::create(&path).unwrap();

        let Err(err) = Cache::open(&path) else { panic!("the lock is held") };
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
