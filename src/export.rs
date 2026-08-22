use anyhow::{anyhow, Context, Result};
use log::info;
use crate::compare::MatchIndex;
use crate::confirm::{self, Target};
use crate::fingerprint::VideoFingerprint;
use crate::stats::RunStats;
use crate::utils::{
    find_best, format_bitrate, format_codec, format_duration, format_frame_rate, format_quality,
    format_shared, format_size, measurable, shutdown_requested, GroupMaxima, Priority,
};
use std::collections::{HashMap, HashSet};
use std::fs::{FileTimes, OpenOptions};
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

/// Which of the three layouts a report is written in.
///
/// The extension used to be the only way to say this, which made two things
/// unsayable: a report under a name that carries no format (`dupes.bak`, and
/// whatever a browser calls a download), and any format at all on stdout, which
/// has no name to read. `--format` is that decision on its own, and the
/// extension stays the default when it isn't given.
///
/// `Txt` is also what an unrecognised extension resolves to. That was a
/// fallthrough arm in the writer before; it is a stated default now, decided in
/// one place (`main::report_target_for`) rather than at the point of writing.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Txt,
    Csv,
    Json,
}

/// Where a report goes.
///
/// `Stdout` is `-o -`. It works because nothing else in a scan writes there:
/// every result line, the progress bars and the confirmation prompt are all on
/// stderr, so the report is the only thing on stdout and `vid-fp DIR -o -
/// --format csv | grep DELETE` reads exactly the report. A file genuinely named
/// `-` is still reachable as `./-`.
#[derive(Clone, Debug)]
pub enum Sink {
    Stdout,
    File(PathBuf),
}

/// A resolved `--output`/`--format` pair: where the report goes and how it is
/// written.
///
/// One value rather than two arguments because they are one decision -- with
/// the extension no longer answering for both, splitting them would have let a
/// destination reach the writer without a format and have it guess again.
#[derive(Clone, Debug)]
pub struct ReportTarget {
    pub sink: Sink,
    pub format: Format,
}

/// Hand the finished report to its destination.
///
/// The one thing here that is not `fs::write` is what happens when the reader
/// of a pipe goes away first: `vid-fp DIR -o - | head` closes stdout under us,
/// and that is a normal end to a pipeline rather than a run that did less than
/// it was asked. Every other write failure is still a problem the caller
/// records. (Rust ignores SIGPIPE, so this arrives as an ordinary error rather
/// than killing the process.)
fn write_report(target: &ReportTarget, bytes: &[u8]) -> Result<()> {
    match &target.sink {
        Sink::File(path) => std::fs::write(path, bytes)
            .with_context(|| format!("Failed to write the report to {}", path.display())),
        Sink::Stdout => {
            let mut out = std::io::stdout().lock();
            match out.write_all(bytes).and_then(|()| out.flush()) {
                Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                other => other.context("Failed to write the report to stdout"),
            }
        }
    }
}

/// Which parts of the run stderr still says out loud once the report has a
/// destination.
struct Console {
    /// The per-group table of files.
    listing: bool,
    /// The counts, the reclaimable total and what the disposal actually did.
    summary: bool,
}

/// A report on stdout REPLACES the console listing rather than joining it.
///
/// The two are the same run said twice: `-o -` printed the whole listing on
/// stderr and the whole report on stdout, so a terminal showed everything twice
/// and `vid-fp DIR -o - | grep DELETE` still scrolled the entire listing past
/// the user on the way to the one line they asked for. A report written to a
/// FILE is a different case and keeps both: the terminal is then the only place
/// the run is visible while it happens.
///
/// The summary is the one part that does not simply follow the listing, because
/// two of the three formats already end with it -- a `.txt` report is exactly
/// this listing plus that summary, and the JSON carries the same figures under
/// `summary`. The CSV carries neither, so there stderr keeps the receipt: what
/// was reclaimable, what was removed, what went wrong. That is the part of a
/// destructive run nobody should have to read out of a pipe.
fn console_for(target: Option<&ReportTarget>) -> Console {
    match target {
        Some(ReportTarget { sink: Sink::Stdout, format }) => Console {
            listing: false,
            summary: *format == Format::Csv,
        },
        _ => Console { listing: true, summary: true },
    }
}

/// What `--delete` does with a file marked DELETE.
///
/// Report-only runs pass `None` and never construct one of these, so the type
/// carries the promise the README makes: there is no value of any argument to
/// this module that removes a file without the caller having decided to.
#[derive(Clone, Debug)]
pub enum Disposal {
    /// The system trash, via the FreeDesktop.org spec. Recoverable, and the
    /// default -- but it needs a trash directory on the file's own filesystem,
    /// which external drives, NFS mounts and headless servers frequently do not
    /// have. `MoveTo` is the answer when it doesn't.
    Trash,
    /// unlink(2). Irreversible.
    Permanent,
    /// Relocate under a folder, mirroring the file's absolute path.
    MoveTo(PathBuf),
}

/// Width of the leading action column in both listings, comma included.
///
/// It has to hold the longest word either listing can print, and that word is
/// UNLINKED -- nine characters with the comma. It was 8 for as long as DELETED,
/// CHANGED and SKIPPED were the longest, so adding UNLINKED shifted every field
/// of exactly those rows one place right: the rows a reader is most likely to
/// be squinting at, since they are the ones whose byte total does not add up.
/// A named constant rather than a literal in two format strings, because the
/// two listings are the same table and drifted apart silently the last time a
/// word was added.
pub const ACTION_COLUMN: usize = 9;

impl Disposal {
    /// Label for the results table once the file has been dealt with.
    ///
    /// A file whose bytes did not go where the mode says gets its own word
    /// rather than the mode's, because that row is the whole explanation for a
    /// byte total smaller than the sum of the rows above it. UNLINKED is
    /// literally what happened to it: `remove_file`, the trash's rename and
    /// `--move-to`'s rename all take the name off this path, and for these rows
    /// that is all they take. `frees_nothing` decides which rows those are, and
    /// it is not simply "every alias" -- a hard link relocated by a rename
    /// arrives at the destination with the whole video in it and is MOVED like
    /// any other row.
    pub fn done_label(&self, aliased: bool) -> &'static str {
        if aliased {
            return "UNLINKED";
        }
        match self {
            Disposal::Trash | Disposal::Permanent => "DELETED",
            Disposal::MoveTo(_) => "MOVED",
        }
    }

    /// Short name for the machine-readable summary.
    fn mode(&self) -> &'static str {
        match self {
            Disposal::Trash => "trash",
            Disposal::Permanent => "permanent",
            Disposal::MoveTo(_) => "move",
        }
    }
}

/// A duration in seconds for the machine-readable outputs, rounded the same way
/// the JSON rounds it, or an empty field when it was never measured.
///
/// Empty is the CSV's spelling of the JSON's `null`. It has to stay
/// distinguishable from `0.00`: one means the overlap is unknown, the other
/// means it was measured and there was none.
fn csv_seconds(value: Option<f64>) -> String {
    value.map(|s| format!("{:.2}", s)).unwrap_or_default()
}

/// What one look at the disk says about a file, taken immediately before the
/// irreversible step.
///
/// Two questions, one `symlink_metadata`: is this still the file that was
/// judged, and would removing this path actually give its bytes back.
struct OnDisk {
    /// The file's current length, or `None` when it could not be read. See
    /// `changed_since` for why that is not treated as a change.
    size: Option<u64>,
    /// Whether this path is one of several names for the same data, and which
    /// kind -- which is not the same question in every disposal mode. See
    /// `Alias`.
    alias: Alias,
}

/// What a path is a name for, as far as the removal about to happen is
/// concerned.
///
/// Two kinds rather than one boolean, because they behave in OPPOSITE
/// directions once the mode is not `Permanent`. A rename carries an inode with
/// it, so a hard link moved to trash or under `--move-to` arrives with every
/// byte of the video in it -- the destination holds the file, the sibling name
/// still holds the file, and nothing anywhere was unlinked. A symlink renamed
/// the same way arrives as a pointer: the footage never moved, it is still at a
/// path this run did not print, and calling that row MOVED claims a copy exists
/// at the destination that does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Alias {
    /// The only name for these bytes, as far as this filesystem knows.
    None,
    /// A hard link with at least one sibling. Unlinking it frees nothing;
    /// renaming it moves the data with it.
    HardLink,
    /// A symbolic link. Nothing that happens to it happens to the video.
    Symlink,
}

/// Read both of them.
///
/// A symlink needs the second stat: the scan measured the file the link leads
/// to (its own stat followed the link, which is how the link entered the
/// library carrying the target's size and codec), so the staleness check has to
/// follow it as well or every symlinked target would read as CHANGED. Nothing
/// else pays for that, and links are rare.
///
/// `nlink` is the hard-link half and costs nothing at all -- it is in the stat
/// that was already being taken. It cannot fire on a file whose other names are
/// in this same run: `sources::collect` deduplicates on (device, inode), so at
/// most one name for any set of bytes ever reaches a DELETE decision. A file
/// that still reads `nlink > 1` here has a name OUTSIDE the run, which is
/// exactly the case where removing this one frees nothing.
fn on_disk(path: &str) -> OnDisk {
    let Ok(md) = std::fs::symlink_metadata(path) else {
        return OnDisk { size: None, alias: Alias::None };
    };

    if md.file_type().is_symlink() {
        // A broken link has no size to check and is still an alias: what goes
        // when it goes is a pointer.
        OnDisk {
            size: std::fs::metadata(path).ok().map(|m| m.len()),
            alias: Alias::Symlink,
        }
    } else {
        let alias = if md.nlink() > 1 { Alias::HardLink } else { Alias::None };
        OnDisk { size: Some(md.len()), alias }
    }
}

/// Whether this disposal took the name and left the data behind.
///
/// This is the question `Fate::Done { aliased }` answers, and it is a question
/// about the MODE as much as about the file. `--permanent` promises the bytes
/// are gone, so either kind of alias falsifies it. Trash and `--move-to`
/// promise the file is somewhere else instead, and a rename keeps that promise
/// for a hard link -- the destination holds the same inode, every byte of it,
/// restorable or undoable exactly like any other row. It is only the symlink
/// that arrives at the destination without its video.
///
/// Getting this wrong the old way (any alias, any mode) cost `--move-to` twice
/// over: the row read UNLINKED, a word that says the data stayed put, and its
/// bytes were struck out of the "(N total)" figure -- so a run that relocated
/// nothing but hard links reported "Moved 3 file(s) (0B total) under /dupes"
/// over a destination holding three complete videos.
fn frees_nothing(alias: Alias, disposal: &Disposal) -> bool {
    match alias {
        Alias::None => false,
        Alias::Symlink => true,
        // The one that depends on the mode.
        Alias::HardLink => matches!(disposal, Disposal::Permanent),
    }
}

/// Whether the file at `path` is still the length it was when it was measured,
/// and a description of the discrepancy if not.
///
/// Every DELETE decision rests on measurements taken at the start of the scan,
/// and a scan of a large library runs for hours -- longer still when the
/// decision is being replayed from a report written yesterday. In that window a
/// download finishes, a re-encode lands over the top, a copy is truncated --
/// and the file about to be removed is no longer the file that was judged
/// redundant. A stat immediately before an irreversible syscall is a cheap way
/// to notice, and `on_disk` has already taken it -- `current` is what it read.
///
/// Size is the only property compared, because it is the only one the
/// fingerprint records. It catches everything that changes a file's length,
/// which is essentially every way a video file is rewritten in practice, but it
/// cannot see an in-place edit that happens to preserve the byte count.
/// Recording an mtime or an inode alongside the fingerprint would close that
/// gap at the cost of a field on every cache entry; nothing has needed it yet.
///
/// A length that could not be read is deliberately NOT reported as a change. It
/// proves nothing about the file's contents, and the removal that follows will fail with the
/// real reason -- so a file that vanished mid-scan reads as a deletion that
/// could not happen rather than as a file quietly left alone.
fn changed_since(path: &str, measured_size: u64, current: Option<u64>) -> Option<String> {
    let size = current?;

    (size != measured_size).then(|| {
        format!(
            "{}: {} bytes when scanned, {} bytes now",
            path, measured_size, size
        )
    })
}

/// What became of one file the disposal pass reached.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fate {
    /// Gone from the path it was at -- trashed, unlinked or relocated.
    ///
    /// `aliased` says whether the bytes stayed where they were: the name went
    /// and the data did not follow it anywhere. A run that counted those bytes
    /// against its own total reported space that is still occupied, or a
    /// destination holding footage it does not hold -- and the caller cannot
    /// work it out for itself, because the only moment the question can be
    /// asked is the stat immediately before the removal. Which files it is true
    /// of depends on the mode as well as on the file; see `frees_nothing`.
    Done { aliased: bool },
    /// It no longer matches the size it was measured at, so it was left alone.
    Changed,
    /// The disposal itself failed. The file is still there.
    Failed,
}

/// Take one file off its path, with the staleness check that has to precede it.
///
/// This is the whole of the destructive step, and the only copy of it: the
/// grouped run reaches it with a size it fingerprinted, `--from-report` with a
/// size a report recorded, and neither gets a version of the check that the
/// other does not. `measured_size` is the file's length at the moment the
/// decision to remove it was made, whenever that was.
///
/// Errors are logged and tallied here rather than returned, because both callers
/// want the same thing from a failure -- a line in the log, a line in the
/// summary, a non-zero exit -- and none of them wants the loop to stop.
pub fn dispose_one(
    path: &str,
    measured_size: u64,
    disposal: &Disposal,
    stats: &RunStats,
) -> Fate {
    // The last thing before the irreversible step, and the only look here at
    // the disk rather than at the measurement. See `changed_since`: the
    // decision being acted on may be hours or days old.
    let disk = on_disk(path);

    if let Some(detail) = changed_since(path, measured_size, disk.size) {
        log::error!(
            target: crate::stats::COUNTED,
            "Not removing {}: it changed on disk after it was scanned",
            path
        );
        stats.delete_stale.record(detail);
        return Fate::Changed;
    }

    match dispose_of(path, disposal) {
        Ok(()) => Fate::Done { aliased: frees_nothing(disk.alias, disposal) },
        Err(e) => {
            log::error!(target: crate::stats::COUNTED, "{:#}", e);
            stats.delete_failed.record(path.to_string());
            Fate::Failed
        }
    }
}

/// Get a single file out of the way, however the user asked for that to happen.
fn dispose_of(path: &str, disposal: &Disposal) -> Result<()> {
    match disposal {
        Disposal::Permanent => std::fs::remove_file(path)
            .with_context(|| format!("Failed to permanently delete {}", path)),
        Disposal::Trash => {
            trash::delete(path).map_err(|e| anyhow!("Failed to move {} to trash: {}", path, e))
        }
        Disposal::MoveTo(dest_root) => move_under(path, dest_root),
    }
}

/// Relocate `path` under `dest_root`, recreating its absolute path inside it.
///
/// `/mnt/media/show/ep.mkv` with `--move-to /backup/dupes` lands at
/// `/backup/dupes/mnt/media/show/ep.mkv`. Mirroring rather than flattening is
/// what makes the destination collision-free without inventing names: every
/// scanned path is canonical and unique, so no two files can ever want the same
/// slot, and a whole run can be undone with a single `cp -a` from the
/// destination root. Flattening would need a disambiguation scheme, and a
/// disambiguation scheme is a thing that can be got wrong while holding the
/// only remaining copy of a file.
fn move_under(path: &str, dest_root: &Path) -> Result<()> {
    let src = Path::new(path);

    // Scanned paths are always absolute (sources.rs canonicalizes every one), so
    // this strip is what turns the mirror into a join rather than an overwrite
    // of dest_root itself.
    let relative = src.strip_prefix("/").unwrap_or(src);
    let dest = dest_root.join(relative);

    let parent = dest
        .parent()
        .ok_or_else(|| anyhow!("Failed to move {}: {} has no parent", path, dest.display()))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create {}", parent.display()))?;

    // rename(2) replaces an existing file without a word, which would destroy
    // whatever an earlier run put there -- reachable whenever a path is moved
    // away, recreated, and moved away again. symlink_metadata rather than
    // exists() because a broken symlink occupying the slot is still something
    // rename would silently consume.
    if std::fs::symlink_metadata(&dest).is_ok() {
        return Err(anyhow!(
            "Failed to move {}: {} already exists (an earlier run moved a file from this path)",
            path,
            dest.display()
        ));
    }

    match std::fs::rename(src, &dest) {
        Ok(()) => Ok(()),
        // The whole reason this mode is worth having on a NAS: the destination
        // is routinely on a different filesystem from the library.
        Err(e) if e.raw_os_error() == Some(libc::EXDEV) => copy_then_unlink(src, &dest),
        Err(e) => Err(anyhow!(
            "Failed to move {} to {}: {}",
            path,
            dest.display(),
            e
        )),
    }
}

/// The cross-filesystem fallback: copy, make it durable, then unlink.
///
/// Both failure paths put the filesystem back the way they found it. A copy
/// that fails leaves a partial file that would make the next run's
/// already-exists check fire for no reason; an unlink that fails leaves the
/// original in place, which makes the copy the redundant one. Either way the
/// caller is told the move failed, and that is exactly what happened.
fn copy_then_unlink(src: &Path, dest: &Path) -> Result<()> {
    let meta = std::fs::metadata(src)
        .with_context(|| format!("Failed to stat {}", src.display()))?;

    if let Err(e) = copy_durably(src, dest, &meta) {
        let _ = std::fs::remove_file(dest);
        return Err(e);
    }

    std::fs::remove_file(src).map_err(|e| {
        let _ = std::fs::remove_file(dest);
        anyhow!(
            "Failed to remove {} after copying it to {}: {}",
            src.display(),
            dest.display(),
            e
        )
    })
}

fn copy_durably(src: &Path, dest: &Path, meta: &std::fs::Metadata) -> Result<()> {
    std::fs::copy(src, dest)
        .with_context(|| format!("Failed to copy {} to {}", src.display(), dest.display()))?;

    let file = OpenOptions::new()
        .write(true)
        .open(dest)
        .with_context(|| format!("Failed to reopen {}", dest.display()))?;

    // Best effort, deliberately. A destination that will not carry timestamps
    // is not a reason to fail a move that has otherwise worked -- but carrying
    // them matters, because a restored file with its original mtime is a cache
    // hit rather than a re-decode of the whole library.
    let mut times = FileTimes::new();
    if let Ok(accessed) = meta.accessed() {
        times = times.set_accessed(accessed);
    }
    if let Ok(modified) = meta.modified() {
        times = times.set_modified(modified);
    }
    let _ = file.set_times(times);

    // The unlink that follows is the point of no return, so the bytes have to
    // be on the far disk before it happens -- not merely in the page cache of a
    // machine that is about to be power-cycled.
    file.sync_all()
        .with_context(|| format!("Failed to flush {} to disk", dest.display()))?;

    Ok(())
}

/// Report every duplicate group, act on the deletion flags, and hand back the
/// paths that are no longer where they were as a result.
///
/// That list is the caller's cue to forget those files' cached fingerprints. It
/// is deliberately a return value rather than a database handle passed in the
/// other direction: this module decides which files die, and knowing nothing
/// about the cache is what keeps that the only thing it decides.
///
/// It carries only what was successfully disposed of. A file that failed, or
/// that was skipped because it changed under us, is still sitting there -- and
/// in the second case its cached fingerprint is already invalid by size, so the
/// next run re-fingerprints it without being told anything here. Trashed and
/// moved files both count as gone: they are somewhere else now, and nothing
/// will ever find this fingerprint under the path it was cached against.
///
/// Nine arguments, and clippy is right to count them -- but they are nine
/// distinct facts with no grouping that isn't arbitrary. Wrapping them in a
/// parameter bag would move the same nine values one line up at the call site
/// and hide which of them the destructive pass actually reads, so the count
/// stands.
#[allow(clippy::too_many_arguments)]
pub fn output_results(
    final_groups: &[Vec<usize>],
    fingerprints: &[VideoFingerprint],
    matches: &MatchIndex,
    report_target: Option<&ReportTarget>,
    total_elapsed_secs: u64,
    priority: Priority,
    disposal: Option<&Disposal>,
    assume_yes: bool,
    stats: &RunStats,
) -> Result<Vec<String>> {

    // --- Pass 1: resolve each file's fate across ALL groups ------------------
    // Cliques overlap, so a file can appear in several groups with different
    // per-group roles. Precedence is REVIEW > DELETE > KEEP:
    //   * REVIEW anywhere -> always kept for manual inspection (never deleted).
    //   * DELETE anywhere -> deleted, even if it is the KEEP pick of another
    //     group. This is what lets a single run remove every redundant copy: a
    //     file that is best in one group but redundant in an overlapping one is
    //     still removed, so you don't have to re-run until the chain collapses.
    //   * otherwise -> kept (it was the best in every group it appears in).
    // Using sets also guarantees each file is considered exactly once, so a
    // file shared by several groups is never queued for deletion twice.
    //
    // Every DELETE here still rests on a direct measurement, which is the point
    // of insisting on cliques upstream: the file lost the ranking inside a group
    // whose members were all compared with each other, so it was compared with
    // the file that beat it. What the global rule does add is a step of
    // indirection at the far end -- the winner of that group may itself lose
    // another one and be removed -- so a run can leave one survivor for a chain
    // of files that were never all measured together. Every hop of such a chain
    // is a direct comparison and the survivor won the last of them; the
    // end-to-end conclusion is their sum. That is deliberate: the alternative is
    // holding the tail of every chain back and asking the user to re-run until
    // it has collapsed a hop at a time.
    let mut review_set: HashSet<usize> = HashSet::new();
    let mut delete_candidates: HashSet<usize> = HashSet::new();

    for group in final_groups {
        let maxima = GroupMaxima::of(group, fingerprints);

        let keep_idx = find_best(group, fingerprints, priority, &maxima);
        let keep_fp = &fingerprints[keep_idx];

        // Everything this group wants held back from deletion. A group can now
        // raise more than one, so it is a set rather than an Option.
        let mut group_review: HashSet<usize> = HashSet::new();

        // --- A runtime nobody could measure ---------------------------------
        // Length is the first thing the ranking compares and the property that
        // keeps a long file safe from the clip cut out of it. A file whose
        // container never reported a runtime, and whose packets carried no
        // clock to measure one from, has no length to compare with anything:
        // it ties on that metric rather than losing it (`GroupMaxima::tier`),
        // which stops it being condemned for a measurement nobody took, and it
        // is held back here, which stops it being condemned on the metrics that
        // are left. Size is the one that would decide, and size is no
        // substitute -- a five-minute clip at a high bitrate is comfortably
        // larger than the two-hour capture it was cut from, so the fall-through
        // marks the host for deletion just as surely as the zero did.
        //
        // That is the same shape as the codec standoff below and it is answered
        // the same way: an unmeasurable runtime makes a file INCOMPARABLE with
        // the rest, not worse than it, so the group ends with one survivor per
        // class rather than one survivor. Every unmeasured file lives, and the
        // measured ones elect a champion of their own to live beside them --
        // deferring to the group's pick when the pick is one of them, for the
        // same reason the standoff defers to it. The also-rans of the measured
        // side are deleted exactly as they always were, because they lost to a
        // file they really were measured against.
        let unmeasured: Vec<usize> = group
            .iter()
            .copied()
            .filter(|&idx| !measurable(&fingerprints[idx], Priority::Length))
            .collect();

        if !unmeasured.is_empty() {
            group_review.extend(unmeasured.iter().copied());

            let measured: Vec<usize> = group
                .iter()
                .copied()
                .filter(|&idx| measurable(&fingerprints[idx], Priority::Length))
                .collect();

            if !measured.is_empty() {
                let champion = if measurable(keep_fp, Priority::Length) {
                    keep_idx
                } else {
                    let measured_maxima = GroupMaxima::of(&measured, fingerprints);
                    find_best(&measured, fingerprints, priority, &measured_maxima)
                };
                group_review.insert(champion);
            }
        }

        // If the KEEP pick isn't top-tier on some quality metric, surface the
        // file that IS as worth a manual look. Metrics are checked in default
        // precedence order, skipping the one the user prioritised (KEEP wins
        // that by construction). Laziness matters here: find_best only runs for
        // the first metric KEEP actually falls short on.
        if let Some(r) = REVIEW_METRICS
            .iter()
            .copied()
            .filter(|&m| m != priority && maxima.tier(keep_fp, m) == 0)
            .map(|m| find_best(group, fingerprints, m, &maxima))
            .find(|&candidate| candidate != keep_idx)
        {
            group_review.insert(r);
        }

        // --- The codec standoff ----------------------------------------------
        // Contenders are the files that are top-tier on every codec-independent
        // metric: same footage length, same resolution, nothing yet separating
        // them. Below that tier something already has separated them and the
        // ranking is safe.
        //
        // Among contenders the only metrics left are quality and size, and both
        // are bit counts, which say nothing across codecs -- an AV1 copy is
        // SUPPOSED to carry fewer bits than an H.264 one of the same footage.
        // So nothing can choose BETWEEN the codecs, and the group has to end
        // with one survivor per codec rather than one survivor.
        //
        // Within a codec, though, everything still works: those files are
        // directly comparable and one of them is plainly the best. So each
        // codec elects a champion and the also-rans are deleted exactly as they
        // would be in a single-codec group -- a library holding five HEVC
        // encodes of one episode does not need five of them held for review.
        //
        // The election runs against maxima built from the codec's own
        // contenders, not the group's, so a file that already lost a
        // codec-blind metric cannot set the bar its codec's contenders are
        // tiered against -- its quality or size is no longer the standard any
        // of them has to reach.
        let contenders: Vec<usize> = group
            .iter()
            .copied()
            .filter(|&idx| {
                REVIEW_METRICS
                    .iter()
                    .all(|&m| maxima.tier(&fingerprints[idx], m) == 1)
            })
            .collect();

        let mut codecs: Vec<&str> = contenders
            .iter()
            .map(|&idx| fingerprints[idx].codec.as_str())
            .collect();
        codecs.sort_unstable();
        codecs.dedup();

        let standoff = codecs.len() > 1;

        // Whether the standoff has anything to say about the pick at all: it
        // ranks contenders, and a pick that is not one of them never stood in
        // its codec's election.
        let pick_contends = contenders.contains(&keep_idx);

        if standoff {
            for codec in codecs {
                let same_codec: Vec<usize> = contenders
                    .iter()
                    .copied()
                    .filter(|&idx| fingerprints[idx].codec == codec)
                    .collect();

                // The KEEP pick IS its own codec's champion, by decree rather
                // than by re-election. Re-tiering against the codec's own
                // contenders can only widen the top band (dropping files lowers
                // every maximum, and a lower maximum admits more of them), so
                // the smaller election sees ties the group-wide one had already
                // broken -- and breaks them again, on a metric that comes
                // EARLIER in the order than the one which settled the group.
                // Two full-length h264 copies half a second apart, with a dense
                // clip owning the group's h264 quality bar: group-wide both are
                // tier 0 on quality and length decides, among contenders alone
                // the denser one takes the quality tier and wins. The champion
                // then replaced the pick below and the pick -- the file this
                // group's own ranking chose -- was deleted in favour of one it
                // had just outranked. Armed, that is the KEEP copy destroyed.
                //
                // Deferring to the pick keeps both rules whole: still exactly
                // one survivor per codec, and the survivor of the pick's codec
                // is the same file a group with no foreign codec in it would
                // have kept. Every other codec is elected as before, because
                // nothing group-wide ever ranked those files against each other
                // -- that is what the standoff means.
                let champion = if pick_contends && fingerprints[keep_idx].codec == codec {
                    keep_idx
                } else {
                    let codec_maxima = GroupMaxima::of(&same_codec, fingerprints);
                    find_best(&same_codec, fingerprints, priority, &codec_maxima)
                };

                group_review.insert(champion);
            }
        }

        // Everything in the group that isn't KEEP or REVIEW is a delete
        // candidate. DELETE wins over KEEP globally, so we don't care whether
        // the file is a KEEP pick in some other group.
        //
        // The KEEP pick is protected unconditionally, and a standoff does not
        // change that. It used to: the champions REPLACED the pick rather than
        // joining it, so that a group could not end with two survivors of one
        // codec -- one because it won its codec, one because it won the group.
        // That reasoning holds, but the way to have it is to make the pick its
        // own codec's champion (above), not to leave it unprotected on the
        // strength of an election it did not stand in. The insert is therefore
        // redundant whenever the pick contends, and load-bearing when it does
        // not (a pick that leads on the prioritised metric but lost a
        // codec-blind one is no contender, and the standoff says nothing about
        // it) -- so it is stated once, for every group, and the group's own
        // ranking can never be overturned by a smaller one.
        let mut protected: HashSet<usize> = group_review.clone();
        protected.insert(keep_idx);

        for &idx in group {
            if !protected.contains(&idx) {
                delete_candidates.insert(idx);
            }
        }

        review_set.extend(group_review);
    }

    // REVIEW protection overrides DELETE. Sorted for deterministic ordering.
    //
    // The subtraction is what makes the precedence global rather than
    // per-group: a file held back in one group is held back everywhere, even
    // where another group ranked it bottom. A group can therefore end up with
    // no DELETE at all, which is the conservative direction.
    let mut delete_indices: Vec<usize> =
        delete_candidates.difference(&review_set).copied().collect();
    delete_indices.sort_unstable();

    // --- The confirmation ----------------------------------------------------
    // Here rather than at start-up because only now is there a question worth
    // answering: how many files, and how many bytes. A prompt raised from the
    // flags alone could ask nothing more useful than "you typed --delete, did
    // you mean it?".
    //
    // Declining does not abort the run. It demotes it to exactly what a run
    // without --delete would have been -- the same report, the same reclaimable
    // figure -- because the user who says no is the user who wants to see the
    // list before committing, and making them re-fingerprint the library to get
    // it would teach them to answer yes.
    let declined = match disposal {
        Some(d) => {
            let targets: Vec<Target> = delete_indices
                .iter()
                .map(|&idx| Target {
                    path: &fingerprints[idx].path,
                    size: fingerprints[idx].file_size,
                })
                .collect();
            !confirm::approve(d, &targets, assume_yes, confirm::Decline::ShowsReport)
        }
        None => false,
    };

    // From here down `disposal` is what the run is going to DO, not what its
    // flags asked for, so the labels, the summary and the JSON all describe the
    // same run the filesystem saw.
    let disposal = if declined { None } else { disposal };
    let acting = disposal.is_some();

    // --- Pass 2: act on each unique target exactly once ----------------------
    let mut removed_count = 0usize;
    let mut failed_count = 0usize;
    let mut changed_count = 0usize;
    let mut aliased_count = 0usize;
    let mut aliased_bytes = 0u64;
    let mut removed_bytes = 0u64;
    let delete_candidate_count = delete_indices.len();

    // What a run WOULD reclaim, computed from the same set the loop below
    // walks, so the dry run's figure and the real run's figure cannot disagree
    // about which files they describe. Sizes come from the fingerprints, which
    // the deletion pass re-checks against disk before touching anything, and
    // the set is index-based -- so a file reached through a hard link
    // contributes its bytes once.
    //
    // It is still an upper bound, and deliberately: a file whose data has
    // another name OUTSIDE the run reclaims nothing when it goes, and the only
    // way to know that is to stat every candidate. The real run does exactly
    // that at the moment it acts (`on_disk`) and reports the difference; a dry
    // run stats nothing and promises nothing it has measured.
    let reclaimable_bytes: u64 = delete_indices
        .iter()
        .map(|&idx| fingerprints[idx].file_size)
        .sum();

    // What this function returns. Only paths that are genuinely no longer there
    // go in here, so it is built next to the counter rather than reconstructed
    // from `delete_indices` afterwards -- the two would differ by every
    // failure, every file that changed under us, and everything an interrupt
    // left untouched.
    let mut deleted_paths: Vec<String> = Vec::new();

    // Maps a file index to the outcome label to print in the results table.
    let mut delete_outcome: HashMap<usize, &'static str> = HashMap::new();

    if let Some(disposal) = disposal {
        for &idx in &delete_indices {
            if shutdown_requested() {
                info!(
                    "Interrupted: stopped after {} file(s); {} left untouched.",
                    removed_count,
                    delete_candidate_count - removed_count - failed_count - changed_count
                );
                break;
            }
            let fp = &fingerprints[idx];

            match dispose_one(&fp.path, fp.file_size, disposal, stats) {
                Fate::Done { aliased } => {
                    removed_count += 1;
                    // Only bytes that actually went away. See `OnDisk`.
                    if aliased {
                        aliased_count += 1;
                        aliased_bytes += fp.file_size;
                    } else {
                        removed_bytes += fp.file_size;
                    }
                    deleted_paths.push(fp.path.clone());
                    delete_outcome.insert(idx, disposal.done_label(aliased));
                }
                Fate::Changed => {
                    changed_count += 1;
                    delete_outcome.insert(idx, "CHANGED");
                }
                Fate::Failed => {
                    failed_count += 1;
                    delete_outcome.insert(idx, "FAILED");
                }
            }
        }
    }

    // --- Reporting -----------------------------------------------------------
    let Console { listing: show_listing, summary: show_summary } = console_for(report_target);

    if show_listing {
        info!("\n========================================");
        info!("             RESULTS");
        info!("========================================\n");
    }

    // Groups overlap, so summing their sizes counts a shared file once per
    // group it appears in -- a figure that can exceed the number of videos
    // scanned and means nothing to anyone. What is actually being reported is
    // how many distinct files are implicated in a duplicate relationship at
    // all, which is the same set the KEEP/DELETE/REVIEW labels below describe.
    let matched_file_count = final_groups
        .iter()
        .flatten()
        .copied()
        .collect::<HashSet<usize>>()
        .len();

    // Which of the three report bodies this run is actually going to write.
    //
    // At most one of them is, and the other two used to be built in full and
    // thrown away. The JSON tree was gated for exactly that reason and the
    // other two were left ungated, which is the more expensive half: the text
    // body is the whole listing accumulated in a `String` and the CSV is the
    // whole listing again through a serializer, so a run writing JSON built
    // both, and a run writing NO report at all still built both. Measured at
    // `-d 18` on the local corpus (9,003 groups, a 31 MB CSV and a 17 MB txt),
    // peak RSS: report-only 66 -> 18 MB, `-o x.csv` 67 -> 50 MB. Every format
    // comes out byte-identical either way -- what is gone is the two copies of
    // the report nobody asked for.
    //
    // The console listing is a separate question, answered by `console_for`: a
    // run with no `--output` prints the listing to the terminal and builds no
    // body at all, which is why the text line below is built when EITHER wants
    // it and pushed only when the report does.
    //
    // The per-link JSON list is the one that scales worst -- one object per
    // pair, so a group of `g` members contributes `g * (g - 1)` of them.
    let format = report_target.map(|t| t.format);
    let wants_txt = format == Some(Format::Txt);
    let wants_csv = format == Some(Format::Csv);
    let wants_json = format == Some(Format::Json);

    let mut txt_out = String::new();
    let mut json_out_groups = Vec::new();

    // Use csv crate for robust and RFC-compliant CSV generation
    let mut csv_wtr = csv::WriterBuilder::new()
        .delimiter(b';')
        .from_writer(Vec::new());

    // The CSV carries exactly what the JSON carries, field for field, in the
    // same order. Anything shown in a formatted column is immediately followed
    // by the raw number it was formatted from, because the formatted one is for
    // reading and the raw one is for sorting and filtering -- a spreadsheet
    // cannot sort "1.0MB" against "900.0KB", nor "1920x1080" against "640x480",
    // and no consumer of a CSV should have to parse a unit suffix or split a
    // label to get a figure this tool already has.
    //
    // The columns run in three blocks, each answering one question:
    //
    //   who        group, action, full_path -- the row's identity and its fate.
    //                Action sits beside the path because --from-report exists to
    //                have it edited, and an action column you have to scroll to
    //                is one that gets edited on the wrong row.
    //   what       length .. quality_bits_per_frame -- the file's own properties,
    //                ordered the way the ranking reads them: footage first
    //                (length, resolution, frame rate), then the codec, then the
    //                three bit-derived figures the codec governs. Codec leads
    //                that trio deliberately: it is the reason two rows' size,
    //                bitrate and quality may not be compared with each other.
    //   against what   matched_with .. matched_to_seconds -- the strongest
    //                measured link: which file, how much footage, and where in
    //                THIS file it sits.
    //
    // Every column in that last block answers for the row's OWN file, which is
    // what the `matched_` prefix is there to promise. It replaced a `shared_`
    // block that broke the promise: `shared_seconds` was the pair's reconciled
    // figure while the range beside it was this file's, so a row whose matched
    // footage ran 0.00-8.84 reported 1.88 seconds of it and read as a
    // malfunction. Nothing in the row said one of the numbers had changed
    // subject, and nothing could -- the fix was to stop mixing them.
    //
    // `samples` is the number of hashes this file's fingerprint holds. It is a
    // property of the file rather than of the link, but it sits immediately
    // before `matched_seconds` because that is the figure it qualifies: a range
    // much wider than the footage means either a scattered match or a file too
    // coarsely sampled to know, and only the sample count tells those apart --
    // at the limit, a file with ONE sample has that sample standing for its
    // whole runtime, so its matched footage can only ever come out as all of it
    // or none of it, and no --match-percent can gate it.
    //
    // `framerate` is the one figure with no formatted twin, by request: the
    // formatted form is still on the console line, where a human reads it, and
    // the reports keep only `framerate_fps` to sort on.
    if wants_csv {
        csv_wtr
            .write_record([
                "group",
                "action",
                "full_path",
                "length",
                "length_seconds",
                "resolution",
                "width",
                "height",
                "framerate_fps",
                "codec",
                "size",
                "size_bytes",
                "bitrate",
                "bitrate_bps",
                "quality",
                "quality_bits_per_frame",
                "matched_with",
                "samples",
                "matched_seconds",
                "matched_from",
                "matched_to",
                "matched_from_seconds",
                "matched_to_seconds",
            ])
            .context("Failed to write CSV header")?;
    }

    for (i, group) in final_groups.iter().enumerate() {
        let group_name = format!("group_{}", i + 1);

        if show_listing {
            info!("{}:", group_name);
        }
        if wants_txt {
            txt_out.push_str(&format!("{}:\n", group_name));
        }

        let mut json_files = Vec::new();

        // How much footage each file has in common with the rest of its group,
        // in seconds rather than as a percentage. Every member was measured
        // against every other, so this says how strong the strongest of those
        // measurements was: a row reporting the whole of its runtime is a copy
        // of something here, and one reporting eight seconds is a file that
        // shares eight seconds with something here.
        //
        // A percentage was the obvious choice and the wrong one. It invites
        // comparison against --match-percent, which measures the opposite end of
        // the pair and so routinely sits above what is shown here -- making a
        // correct report look broken. Worse, on short videos it quantizes
        // brutally: a file with four keyframes can only ever score 0, 25, 50, 75
        // or 100%, so a single incidental frame reads as an authoritative-looking
        // 25%. In seconds that same match reads "0.8s", next to a length of
        // "00:00:03", and needs no explaining.
        for &idx in group {
            let fp = &fingerprints[idx];

            // Computed once and read three ways: the strongest link fills the
            // single-figure columns the console and the CSV show, and the whole
            // list goes to the JSON so a group of three or more can be read pair
            // by pair. Scoped to THIS group, because a file appearing in two of
            // them matched files in both and a row must only name files the
            // reader can see beside it. Groups are small and this is the
            // reporting pass, not the comparison one -- there is no measurement
            // here, only lookups against figures phase 2 already took.
            let links = matches.links_of(idx, group, fingerprints);
            let best = links.first();
            // Every figure in this block is stated in the SUBJECT's own
            // seconds, which is `coverage x duration` -- so a file whose
            // runtime nobody could measure has no seconds to state, and "0s
            // matched" would read as evidence that nothing matched when what is
            // missing is the runtime to scale the coverage by. Blank, like the
            // length it is derived from. The link itself is kept: which file it
            // matched, and where in that file, are both known.
            let seconds_known = fp.duration > 0.0;
            let matched = best.map(|l| l.matched_seconds).filter(|_| seconds_known);

            let size_str = format_size(fp.file_size);
            let bitrate_str = format_bitrate(fp.bitrate());
            // "-" rather than 00:00:00 when no runtime could be measured, the
            // same sentinel the frame rate and the quality columns use for the
            // same reason: a zero-length video and a video of unknown length
            // are different findings, and the row that reads 00:00:00 beside 20
            // samples looks like a malfunction. See `GroupMaxima::tier`, which
            // is where the distinction decides something.
            let duration_str = if fp.duration > 0.0 {
                format_duration(fp.duration)
            } else {
                "-".to_string()
            };
            let res_str = format!("{}x{}", fp.width, fp.height);

            // The codec is the reason two rows' bit figures may not be compared
            // with each other, so it is shown next to them rather than tucked
            // away in the JSON: a reader who sees "h264" and "av1" in one group
            // needs no explanation for why nothing was deleted.
            let codec_str = format_codec(&fp.codec);
            let frame_rate_str = format_frame_rate(fp.frame_rate);
            // Bits per frame -- the figure that actually ranks these files.
            // Bitrate stays alongside it because it is the number people know
            // and the one their other tools print, but it never ranks anything.
            let quality_str = format_quality(fp.quality());
            // The console and text report show only the formatted figure,
            // because a human is reading it at a glance.
            let matched_str = format_shared(matched);
            // Hashes held, not keyframes decoded: featureless frames are dropped
            // below MIN_AC_ENERGY, so this is what the comparison actually had
            // to work with, which is the number that explains the row.
            let samples_raw = fp.valid_hashes.len().to_string();

            // The same values as numbers, for the CSV and JSON. An unknown one
            // is an empty field / a null rather than a zero: a container that
            // never reported a frame rate is not a container that reported no
            // frames, and a quality figure derived from one is unknowable
            // rather than worst-in-group. The runtime joins them for the same
            // reason and with more at stake -- it is the metric the ranking
            // compares first, and a consumer sorting on this column must not
            // see an unmeasured file as a zero-length one.
            let length_num = (fp.duration > 0.0).then(|| (fp.duration * 100.0).round() / 100.0);
            let frame_rate_num = (fp.frame_rate > 0.0)
                .then(|| (fp.frame_rate * 1000.0).round() / 1000.0);
            let quality_num = (fp.quality() > 0).then(|| fp.quality());

            let frame_rate_raw = frame_rate_num.map(|f| f.to_string()).unwrap_or_default();
            let quality_raw = quality_num.map(|q| q.to_string()).unwrap_or_default();
            let size_bytes_raw = fp.file_size.to_string();
            let bitrate_bps_raw = fp.bitrate().to_string();
            let matched_seconds_raw = csv_seconds(matched);
            // Runtime and frame geometry as plain numbers, so every figure the
            // ranking uses can be sorted on. Resolution's raw form is the two
            // sides rather than their product: the product is one multiplication
            // away in any spreadsheet, and the sides are what was measured.
            let length_seconds_raw = if fp.duration > 0.0 {
                format!("{:.2}", fp.duration)
            } else {
                String::new()
            };
            let width_raw = fp.width.to_string();
            let height_raw = fp.height.to_string();

            // The file the figures above describe. Empty rather than "-" when
            // there is none, for the same reason every other unknown is empty:
            // a CSV consumer should see a blank cell, not a sentinel it has to
            // know about.
            let matched_with_raw = best
                .map(|l| fingerprints[l.other].path.clone())
                .unwrap_or_default();

            // The envelope, in this file's own timeline -- the same timeline
            // `matched_seconds` is stated in, so the two can be read against
            // each other. Both the clock form and the raw seconds, like every
            // other figure here.
            let best_span = best.and_then(|l| l.span);
            let matched_from_str =
                best_span.map(|s| format_duration(s.start_seconds())).unwrap_or_default();
            let matched_to_str =
                best_span.map(|s| format_duration(s.end_seconds())).unwrap_or_default();
            let matched_from_raw = csv_seconds(best_span.map(|s| s.start_seconds()));
            let matched_to_raw = csv_seconds(best_span.map(|s| s.end_seconds()));

            // Label by the file's GLOBAL fate (precedence REVIEW > DELETE > KEEP).
            // A file that is redundant in an overlapping group is shown DELETE/
            // DELETED in every group, including one where it was the local best.
            // In a dry run the delete targets stay as the recommendation DELETE;
            // once armed they become DELETED or MOVED, FAILED if that errored, or
            // CHANGED if the file stopped matching its fingerprint before we got
            // to it.
            let action_str = if review_set.contains(&idx) {
                "REVIEW"
            } else if delete_candidates.contains(&idx) {
                if acting {
                    delete_outcome.get(&idx).copied().unwrap_or("SKIPPED")
                } else {
                    "DELETE"
                }
            } else if acting {
                "KEPT"
            } else {
                "KEEP"
            };

            // 1. Console / Text Output
            //
            // The same three blocks as the CSV, minus the raw duplicates and the
            // path of the matched file -- both are noise at a glance and neither
            // fits on a terminal line.
            //
            // The action leads and is padded to a fixed width so it forms a
            // column the eye can run down; the path trails because it is the one
            // field with no bounded length, and anything after it would be
            // ragged. "matched" is spelled out on every row because the console
            // has no header, and two time values side by side ("00:00:09, 0.8s")
            // would otherwise be ambiguous about which is which. It reads
            // "matched" rather than the older "shared" because the figure is now
            // this file's own footage rather than the pair's -- "8.8s matched"
            // against a length of 00:00:09 says copy, and says it without the
            // reader having to know which end of the pair it was measured from.
            // The frame rate, the sample count and the bits-per-frame figure
            // carry their units for the same reason.
            //
            // Built when either the terminal or a .txt report is going to read
            // it, and not otherwise: on a CSV or JSON run this line is the one
            // piece of the text body that is neither shown nor saved.
            if show_listing || wants_txt {
                let line = format!(
                    "\t{:<width$} {}, {}, {}, {}, {}, {}, {}, {} samples, {} matched, {}",
                    format!("{},", action_str),
                    duration_str,
                    res_str,
                    frame_rate_str,
                    codec_str,
                    size_str,
                    bitrate_str,
                    quality_str,
                    samples_raw,
                    matched_str,
                    fp.path,
                    width = ACTION_COLUMN
                );
                if show_listing {
                    info!("{}", line);
                }
                if wants_txt {
                    txt_out.push_str(&line);
                    txt_out.push('\n');
                }
            }

            // 2. CSV Output
            if wants_csv {
                csv_wtr.write_record([
                    &group_name,
                    action_str,
                    &fp.path,
                    &duration_str,
                    &length_seconds_raw,
                    &res_str,
                    &width_raw,
                    &height_raw,
                    &frame_rate_raw,
                    &codec_str,
                    &size_str,
                    &size_bytes_raw,
                    &bitrate_str,
                    &bitrate_bps_raw,
                    &quality_str,
                    &quality_raw,
                    &matched_with_raw,
                    &samples_raw,
                    &matched_seconds_raw,
                    &matched_from_str,
                    &matched_to_str,
                    &matched_from_raw,
                    &matched_to_raw,
                ]).context("Failed to write CSV record")?;
            }

            // 3. JSON File Output
            //
            // The JSON carries every link rather than only the strongest,
            // because it is the format read by something that can hold a list.
            // A group of three is where the single-figure columns stop being
            // the whole story, and this is where the rest of it lives.
            if !wants_json {
                continue; // Nothing below this point but the JSON tree.
            }

            let json_matches: Vec<serde_json::Value> = links
                .iter()
                .map(|l| {
                    serde_json::json!({
                        "full_path": fingerprints[l.other].path,
                        "matched_seconds": seconds_known
                            .then(|| (l.matched_seconds * 100.0).round() / 100.0),
                        "matched_from": l.span.map(|s| format_duration(s.start_seconds())),
                        "matched_to": l.span.map(|s| format_duration(s.end_seconds())),
                        "matched_from_seconds": l.span
                            .map(|s| (s.start_seconds() * 100.0).round() / 100.0),
                        "matched_to_seconds": l.span
                            .map(|s| (s.end_seconds() * 100.0).round() / 100.0),
                    })
                })
                .collect();

            // Key for key and block for block, the CSV row above. Written in
            // that order and kept in it by serde_json's `preserve_order`.
            json_files.push(serde_json::json!({
                "action": action_str,
                "full_path": fp.path,
                "length": duration_str,
                "length_seconds": length_num,
                "resolution": res_str,
                "width": fp.width,
                "height": fp.height,
                "framerate_fps": frame_rate_num,
                "codec": codec_str,
                "size": size_str,
                "size_bytes": fp.file_size,
                "bitrate": bitrate_str,
                "bitrate_bps": fp.bitrate(),
                "quality": quality_str,
                "quality_bits_per_frame": quality_num,
                // The strongest link, spelled out so the JSON says the same
                // thing the CSV does without a consumer having to re-derive it
                // from the list below.
                "matched_with": best.map(|l| &fingerprints[l.other].path),
                "samples": fp.valid_hashes.len(),
                "matched_seconds": matched.map(|s| (s * 100.0).round() / 100.0),
                "matched_from": best_span.map(|s| format_duration(s.start_seconds())),
                "matched_to": best_span.map(|s| format_duration(s.end_seconds())),
                "matched_from_seconds": best_span
                    .map(|s| (s.start_seconds() * 100.0).round() / 100.0),
                "matched_to_seconds": best_span.map(|s| (s.end_seconds() * 100.0).round() / 100.0),
                // Every measured link this file has in the group, strongest
                // first. The entry at index 0 is the one described above.
                "matches": json_matches,
            }));
        }

        if show_listing {
            info!(""); // Empty line for spacing
        }
        if wants_txt {
            txt_out.push('\n');
        }

        if wants_json {
            json_out_groups.push(serde_json::json!({
                "group": group_name,
                "files": json_files
            }));
        }
    }

    let total_hours = total_elapsed_secs / 3600;
    let total_mins = (total_elapsed_secs % 3600) / 60;
    let total_secs = total_elapsed_secs % 60;

    let mut summary = format!(
        "Total groups found: {}\nTotal files matched: {}\nTotal time elapsed: {:02}:{:02}:{:02}",
        final_groups.len(),
        matched_file_count,
        total_hours,
        total_mins,
        total_secs
    );

    match disposal {
        // The figure a dry run exists to produce. Printing it only after the
        // files are gone is printing it to the one person who no longer needs
        // it -- the decision is made here, on the strength of this number.
        None => {
            if delete_candidate_count > 0 {
                summary.push_str(&format!(
                    "\nReclaimable: {} across {} file(s) marked DELETE (nothing was removed).",
                    format_size(reclaimable_bytes),
                    delete_candidate_count
                ));
            }
            // Said plainly, because "nothing was removed" alone reads as a
            // report-only run and the user who declined needs to know the
            // difference between "I wasn't asked to" and "you told me not to".
            if declined {
                summary.push_str("\nCancelled at the confirmation prompt.");
            }
        }
        Some(d) => summary.push_str(&format!("\n{}", disposed_line(d, removed_count, removed_bytes))),
    }

    if acting {
        summary.push_str(&aliased_line(aliased_count, aliased_bytes));
        summary.push_str(&trouble_lines(failed_count, changed_count));
    }

    if show_summary {
        info!("{}", summary);
    }

    // --- Writing the report ---------------------------------------------------
    // Logged and tallied rather than returned, which is the same treatment
    // `dispose_one` gives a failed deletion and for a stronger reason. By this
    // line the destructive pass has already run: propagating an error here threw
    // away `deleted_paths` on the way out, so the caller never got to drop those
    // files' cache entries and the cache went on claiming fingerprints for files
    // this run had just removed. A report that could not be saved is a problem
    // (exit 2, and a line in the Problems summary) -- it is not a reason to
    // forget what the run did to the filesystem.
    //
    // `report_target_for` has already rejected the mistyped paths AND proved
    // this destination writable before any work started, so what reaches here
    // is a full disk, or permissions that changed while the run was going.
    if let Some(target) = report_target {
        let written = (move || -> Result<()> {
            let bytes = match target.format {
                Format::Csv => {
                    csv_wtr.into_inner().context("Failed to finalize CSV buffer")?
                }
                Format::Json => {
                    // Summary first, then the groups. It is the part a human opens
                    // the file to read, and burying it under an array with a row per
                    // duplicate makes it something you have to go looking for.
                    //
                    // Its own three blocks run what was found -> what was decided ->
                    // what was done, so a dry run's keys stop after the second one
                    // has said everything it can.
                    let json_final = serde_json::json!({
                        "summary": {
                            "total_groups": final_groups.len(),
                            "total_files_matched": matched_file_count,
                            "time_elapsed_seconds": total_elapsed_secs,
                            // What the run would reclaim, present whether or not it
                            // did anything: a dry run's whole output is a plan, and
                            // a plan with no cost attached is not one.
                            "files_marked_delete": delete_candidate_count,
                            "reclaimable_bytes": reclaimable_bytes,
                            "deletion_enabled": acting,
                            "mode": disposal.map(|d| d.mode()).unwrap_or("report"),
                            "move_to": match disposal {
                                Some(Disposal::MoveTo(dir)) => {
                                    Some(dir.to_string_lossy().to_string())
                                }
                                _ => None,
                            },
                            "files_removed": removed_count,
                            "bytes_removed": removed_bytes,
                            // Of those removed, the ones that were a second
                            // name for data still on disk. `bytes_removed`
                            // excludes them, so this is what a reader needs to
                            // reconcile it against the rows.
                            "files_unlinked": aliased_count,
                            "bytes_unlinked": aliased_bytes,
                            "files_failed": failed_count,
                            "files_changed": changed_count,
                        },
                        "results": json_out_groups
                    });
                    serde_json::to_string_pretty(&json_final).unwrap().into_bytes()
                }
                Format::Txt => {
                    let mut full_txt = String::new();
                    full_txt.push_str(&txt_out);
                    full_txt.push_str(&summary);
                    full_txt.push('\n');

                    full_txt.into_bytes()
                }
            };

            write_report(target, &bytes)
        })();

        match written {
            // Nothing to announce for stdout: the report is already there, and
            // saying so on stderr would be describing the pipe to whoever is
            // reading the other end of it.
            Ok(()) => {
                if let Sink::File(path) = &target.sink {
                    info!("\nResults saved to {}", path.display());
                }
            }
            Err(e) => {
                log::error!(target: crate::stats::COUNTED, "{:#}", e);
                stats.report_write_failed.record(format!("{:#}", e));
            }
        }
    }

    // Helpful nudge when there's something to clean up but nothing was touched.
    // Not for someone who declined: they typed the flag, saw the question, and
    // said no. Telling them to type it again would be answering back.
    //
    // Printed whatever the report does, because no format carries it: it is
    // advice about the next command to run, not a figure this one measured.
    // Last, therefore, and after the report has been written rather than before
    // -- on `-o -` it was landing on stderr while the report it refers to was
    // still being built, so the terminal read "run with --delete" and only then
    // showed the run it was talking about.
    if !acting && !declined && delete_candidate_count > 0 {
        info!(
            "\nRun with --delete to move the file(s) marked DELETE to the trash or with --move-to <DIR> to relocate them instead."
        );
    }

    Ok(deleted_paths)
}

/// What a disposal pass did, in the words of the mode that did it.
///
/// Shared with `--from-report`, which performs the same three operations on a
/// list it read rather than one it computed, and should not describe them
/// differently for it.
pub fn disposed_line(disposal: &Disposal, count: usize, bytes: u64) -> String {
    match disposal {
        Disposal::Permanent => format!(
            "Permanently deleted {} file(s), {} freed.",
            count,
            format_size(bytes)
        ),
        Disposal::Trash => format!(
            "Moved {} file(s) to trash ({} total).",
            count,
            format_size(bytes)
        ),
        // Deliberately not "freed": if the destination is on the same
        // filesystem the bytes have not gone anywhere, and claiming otherwise
        // in the one mode where it might not be true is how a summary stops
        // being trusted.
        Disposal::MoveTo(dir) => format!(
            "Moved {} file(s) ({} total) under {}.",
            count,
            format_size(bytes),
            dir.display()
        ),
    }
}

/// What the byte figure above it leaves out, when there is anything to leave
/// out.
///
/// Removing one of several names for a file is a perfectly good outcome -- the
/// user asked for that path to go, and it went -- so this is not a problem, not
/// a skip, and does not touch the exit code. It exists because the figure
/// beside it would otherwise be unaccountable: rows totalling 40GB above a line
/// reading "12GB freed", with nothing to say which rows were which. The rows
/// are labelled UNLINKED; this says how many, how much, and why. It carries the
/// bytes because a `--move-to` run of nothing else reads "Moved 1 file(s) (0B
/// total)", which looks like a malfunction rather than a caveat.
///
/// Shared with `--from-report` for the same reason `disposed_line` is: it
/// performs the same removals and can reach the same files.
pub fn aliased_line(count: usize, bytes: u64) -> String {
    if count == 0 {
        return String::new();
    }
    format!(
        "\n{} file(s) were another name for data that is still on disk (a hard link or a \
         symlink), so their {} is not counted above; they are marked UNLINKED.",
        count,
        format_size(bytes)
    )
}

/// The lines that follow it when the pass did not get everything, each omitted
/// when its count is zero.
pub fn trouble_lines(failed: usize, changed: usize) -> String {
    let mut out = String::new();
    if failed > 0 {
        out.push_str(&format!(
            "\n{} file(s) could not be removed (see errors above).",
            failed
        ));
    }
    if changed > 0 {
        out.push_str(&format!(
            "\n{} file(s) changed on disk after they were scanned and were left alone \
             (re-run to judge them as they now stand).",
            changed
        ));
    }
    out
}

/// Metrics that can justify a REVIEW flag, in default precedence order.
///
/// These are also exactly the metrics that mean the same thing regardless of
/// codec, which is why the same list does double duty above as the definition
/// of a group's genuine contenders.
///
/// Quality and size are absent, for different reasons. Size carries no quality
/// information that quality and length don't already express (size IS quality x
/// frame rate x length), so flagging on it would only ever repeat a flag one of
/// those two already raised. Quality is absent because it is only meaningful
/// within a single codec: across codecs it deliberately ties, so a REVIEW
/// derived from it would be reporting a comparison this tool has just declared
/// impossible. The cross-codec case gets its own rule instead.
const REVIEW_METRICS: [Priority; 2] = [Priority::Length, Priority::Resolution];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::Match;
    use crate::utils::Priority;
    use tempfile::NamedTempFile;
    use std::fs;

    const CSV_HEADER: &str = "group;action;full_path;length;length_seconds;resolution;width;\
height;framerate_fps;codec;size;size_bytes;bitrate;bitrate_bps;quality;\
quality_bits_per_frame;matched_with;samples;matched_seconds;matched_from;matched_to;\
matched_from_seconds;matched_to_seconds";

    fn mock_fp() -> VideoFingerprint {
        VideoFingerprint {
            path: "/fake/path/vid.mp4".to_string(),
            valid_hashes: vec![], valid_t_start: vec![], valid_t_end: vec![],
            total_ms: 100_000, width: 1920, height: 1080, duration: 60.0, file_size: 1048576,
            codec: "h264".to_string(), frame_rate: 30.0,
        }
    }

    /// A mock whose size scales with duration, holding bitrate at exactly
    /// 1 Mbps and the frame rate at 30. This matters now that bitrate is
    /// derived: give a 10s file the same byte count as a 90s one and it
    /// genuinely IS a 9x higher-quality copy, which the REVIEW rule will
    /// correctly flag and protect from deletion. These tests are about the
    /// KEEP/DELETE precedence logic, so they hold every other variable still.
    fn mock_fp_at(path: &str, duration: f64) -> VideoFingerprint {
        let mut fp = mock_fp();
        fp.path = path.to_string();
        fp.duration = duration;
        fp.file_size = (duration * 131_072.0) as u64; // 1 Mbps
        fp
    }

    /// The same, plus a codec. Everything else stays identical, so the codec is
    /// the ONLY thing these files differ by.
    fn mock_fp_coded(path: &str, duration: f64, codec: &str) -> VideoFingerprint {
        let mut fp = mock_fp_at(path, duration);
        fp.codec = codec.to_string();
        fp
    }

    /// The same again with an explicit byte count, for the tests that need two
    /// files of one codec to differ by a controlled amount.
    fn mock_fp_sized(path: &str, duration: f64, codec: &str, file_size: u64) -> VideoFingerprint {
        let mut fp = mock_fp_coded(path, duration, codec);
        fp.file_size = file_size;
        fp
    }

    /// A path inside a directory that cleans itself up, with a name WE choose.
    /// NamedTempFile picks random ones, which is useless when the point of a
    /// test is that alphabetical order must not decide the outcome.
    fn at(dir: &tempfile::TempDir, name: &str) -> String {
        dir.path().join(name).to_string_lossy().to_string()
    }

    /// Where `--move-to <root>` will put the file currently at `path`.
    fn landing_spot(root: &Path, path: &str) -> PathBuf {
        root.join(Path::new(path).strip_prefix("/").unwrap())
    }

    /// Put a real file on disk at `fp.path`, exactly as long as the fingerprint
    /// claims it is.
    ///
    /// Mandatory for anything running armed: the deletion path re-stats every
    /// target and refuses to touch one whose length no longer matches, so a
    /// mock claiming 7.8 MB against an empty temp file is skipped as CHANGED
    /// rather than removed. `set_len` leaves the file sparse, so a 24 MB mock
    /// costs one syscall and no blocks.
    fn materialize(fp: &VideoFingerprint) {
        fs::File::create(&fp.path)
            .unwrap()
            .set_len(fp.file_size)
            .unwrap();
    }

    fn materialize_all(fps: &[VideoFingerprint]) {
        for fp in fps {
            materialize(fp);
        }
    }

    /// An index in which every file was compared with every other and found to
    /// contain it whole.
    ///
    /// Tests about deletion precedence are asking which copy wins, not how the
    /// group got linked, so this states the simplest thing that makes the
    /// question well-posed: everything here was measured against everything
    /// else -- which is also what a real group is, since clustering only ever
    /// hands this module complete subgraphs.
    ///
    /// Degenerates to an empty index for a single file, which is what the
    /// report-formatting tests want: no pair, so every overlap reads "-".
    fn all_compared(n: usize) -> MatchIndex {
        let mut matches = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                matches.push(Match::new(a, b, 1.0, 1.0));
            }
        }
        MatchIndex::new(matches)
    }

    /// Run a report into a temp file of the given extension and hand back the
    /// contents. The NamedTempFile guard is dropped, so the caller is
    /// responsible for removing the extension-suffixed path it created.
    fn report_to(extension: &str) -> String {
        NamedTempFile::new()
            .unwrap()
            .path()
            .with_extension(extension)
            .to_string_lossy()
            .to_string()
    }

    /// The target a path implies on its own, which is what these tests were
    /// written against and what a run with no `--format` still resolves to.
    fn to_file(path: &str) -> ReportTarget {
        ReportTarget {
            sink: Sink::File(PathBuf::from(path)),
            format: crate::format_from_extension(Path::new(path)),
        }
    }

    fn read_json(path: &str) -> serde_json::Value {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn test_a_report_on_stdout_replaces_the_console_listing_instead_of_joining_it() {
        let stdout = |format| ReportTarget { sink: Sink::Stdout, format };

        // Both halves of a .txt report are on stdout, so stderr says neither.
        // This is the case that printed the whole run twice.
        assert!(!console_for(Some(&stdout(Format::Txt))).listing);
        assert!(!console_for(Some(&stdout(Format::Txt))).summary);

        // The JSON carries the same figures under its own `summary` key.
        assert!(!console_for(Some(&stdout(Format::Json))).summary);

        // The CSV carries no summary at all, so stderr keeps the receipt --
        // what was reclaimable, what was removed, what went wrong.
        assert!(!console_for(Some(&stdout(Format::Csv))).listing);
        assert!(console_for(Some(&stdout(Format::Csv))).summary);

        // A report going to a file changes nothing: the terminal is still the
        // only place the run is visible while it happens.
        let file = to_file("report.csv");
        assert!(console_for(Some(&file)).listing);
        assert!(console_for(Some(&file)).summary);
        assert!(console_for(None).listing);
        assert!(console_for(None).summary);
    }

    #[test]
    fn test_csv_output() {
        let fps = vec![mock_fp()];
        let groups = vec![vec![0]];

        let path_str = report_to("csv");

        // Report-only run: single item defaults to KEEP.
        let deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 120, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        assert!(deleted.is_empty(), "a report-only run removes nothing, so it forgets nothing");

        let contents = fs::read_to_string(&path_str).unwrap();

        // Assert headers exist (separated by semicolons)
        assert!(contents.contains(CSV_HEADER), "{}", contents);
        // 1 MiB over 60s = ~140kbps, which at 30fps is ~4.7kbit in each frame.
        // Assert data exists and defaults to KEEP. The overlap was never
        // measured, so both seconds fields are empty rather than 0.
        // A group of one has nobody to be compared against, so every column
        // that describes a link is empty too.
        assert!(contents.contains(
            "group_1;KEEP;/fake/path/vid.mp4;00:01:00;60.00;1920x1080;1920;1080;30;h264;\
1.0MB;1048576;140kbps;139810;4.7kb/f;4660;;0;;;;;"
        ), "{}", contents);

        // Clean up
        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_dry_run_says_what_it_would_reclaim() {
        // The number the decision is actually made on. It has to be available
        // BEFORE anything is removed, which is the one case where the old
        // "freed" figure was never printed at all.
        let fps = vec![
            mock_fp_at("/fake/keep.mkv", 100.0),
            mock_fp_at("/fake/dupe.mkv", 10.0),
        ];
        let groups = vec![vec![0, 1]];
        let path_str = report_to("json");

        output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let summary = read_json(&path_str)["summary"].clone();

        assert_eq!(summary["mode"], "report");
        assert_eq!(summary["files_marked_delete"], 1);
        assert_eq!(
            summary["reclaimable_bytes"], 1_310_720u64,
            "the DELETE target's bytes, and only those"
        );
        assert_eq!(summary["files_removed"], 0, "and nothing actually happened");
        assert_eq!(summary["bytes_removed"], 0u64);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_reclaimable_totals_the_delete_targets_of_every_group() {
        // Each group contributes its own redundant copies and nothing else. The
        // figure is what the whole run would free, not what its largest group
        // would.
        let fps = vec![
            mock_fp_at("/fake/best_a.mkv", 60.0),
            mock_fp_at("/fake/dupe_a.mkv", 10.0),
            mock_fp_at("/fake/best_b.mkv", 60.0),
            mock_fp_at("/fake/dupe_b.mkv", 10.0),
        ];
        let groups = vec![vec![0, 1], vec![2, 3]];
        let path_str = report_to("json");

        output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let summary = read_json(&path_str)["summary"].clone();
        assert_eq!(summary["files_marked_delete"], 2);
        assert_eq!(summary["reclaimable_bytes"], 2_621_440u64);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_csv_carries_every_figure_the_json_does() {
        // The two machine-readable outputs are the same data in two shapes.
        // Anything the JSON knows, a CSV consumer can sort and filter on too,
        // including the raw counterpart of every formatted column.
        let fps = vec![
            mock_fp_at("/fake/a.mp4", 100.0),
            mock_fp_at("/fake/b.mp4", 100.0),
        ];
        let groups = vec![vec![0, 1]];
        let matches = MatchIndex::new(vec![Match::new(0, 1, 0.80, 0.80)]);

        let csv_path = report_to("csv");
        let json_path = report_to("json");

        output_results(
            &groups, &fps, &matches, Some(&to_file(&csv_path)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();
        output_results(
            &groups, &fps, &matches, Some(&to_file(&json_path)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let csv = fs::read_to_string(&csv_path).unwrap();
        let report = read_json(&json_path);
        let file = &report["results"][0]["files"][0];

        assert!(
            csv.contains(
                "group_1;KEEP;/fake/a.mp4;00:01:40;100.00;1920x1080;1920;1080;30;h264;\
12.5MB;13107200;1.0Mbps;1048576;35.0kb/f;34952;/fake/b.mp4;0;80.00;"
            ),
            "{}",
            csv
        );

        // Every raw figure in the row above is the one the JSON reports.
        assert_eq!(file["length_seconds"], 100.0);
        assert_eq!(file["width"], 1920);
        assert_eq!(file["height"], 1080);
        assert_eq!(file["framerate_fps"], 30.0);
        assert_eq!(file["size_bytes"], 13_107_200u64);
        assert_eq!(file["bitrate_bps"], 1_048_576u64);
        assert_eq!(file["quality_bits_per_frame"], 34_952u64);
        assert_eq!(file["matched_seconds"], 80.0);

        let _ = fs::remove_file(csv_path);
        let _ = fs::remove_file(json_path);
    }

    #[test]
    fn test_a_lopsided_pair_reports_each_row_against_its_own_envelope() {
        // The bug this column layout exists to fix, from the case that found it:
        // `leg raises_19.mp4` holds a single keyframe, so its one hash stands
        // for its whole 8.84s runtime and any match at all covers 100% of it,
        // while only 1.88s of the 6.01s `_18` matched back.
        //
        // Reconciled to one figure both rows read 1.88s -- and on _19's row that
        // sat beside an envelope running 0.00-8.84, a range 4.7x wider than the
        // duration printed next to it, with nothing to say the two numbers were
        // measured from different ends. Each row now answers for its own file,
        // so each row's duration fits inside its own envelope.
        let mut host = mock_fp_at("/fake/_18.mp4", 6.01);
        let mut lone = mock_fp_at("/fake/_19.mp4", 8.84);
        host.valid_hashes = vec![0; 10];
        lone.valid_hashes = vec![0; 1];
        let fps = vec![host, lone];
        let groups = vec![vec![0, 1]];

        // 1.877 / 6.01 = 31.2% of _18; all of _19.
        let matches = MatchIndex::new(vec![
            Match::new(0, 1, 0.3123, 1.0).with_spans((3753, 5630), (0, 8842)),
        ]);

        let path_str = report_to("csv");
        output_results(
            &groups, &fps, &matches, Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let contents = fs::read_to_string(&path_str).unwrap();
        for line in contents.lines().skip(1) {
            let cells: Vec<&str> = line.split(';').collect();
            let matched: f64 = cells[18].parse().unwrap();
            let from: f64 = cells[21].parse().unwrap();
            let to: f64 = cells[22].parse().unwrap();
            assert!(
                matched <= to - from + 1e-9,
                "matched footage must fit inside its own envelope: {}",
                line
            );
        }

        // And the two rows now differ, which is the signal the minimum hid.
        assert!(contents.contains(";/fake/_19.mp4;10;1.88;"), "_18 row: {}", contents);
        assert!(contents.contains(";/fake/_18.mp4;1;8.84;"), "_19 row: {}", contents);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_matched_duration_reads_the_same_on_a_clip_and_its_host() {
        // The reason this column is a duration. As coverage these two rows read
        // 10% and 100% and look like a malfunction next to any --match-percent;
        // as seconds they both read a minute, which is the truth.
        //
        // It is also why going directional cost nothing here. The figure is now
        // each row's OWN footage rather than the pair's reconciled minimum, and
        // on an honest match that is the same number from both ends: the host's
        // `10% x 600s` and the clip's `100% x 60s` are both a minute. The
        // minimum only ever did work when the two sides disagreed, which is
        // exactly the case worth showing rather than hiding.
        let fps = vec![
            mock_fp_at("/fake/host.mp4", 600.0),
            mock_fp_at("/fake/clip.mp4", 60.0),
        ];
        let groups = vec![vec![0, 1]];

        let matches = MatchIndex::new(vec![Match::new(0, 1, 0.10, 1.0)]);

        let path_str = report_to("csv");

        output_results(
            &groups, &fps, &matches, Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        // Read as `matched_with;matched_seconds`: each row names the other file
        // and reports the same minute of footage.
        let contents = fs::read_to_string(&path_str).unwrap();
        assert!(contents.contains(";/fake/clip.mp4;0;60.00;"), "host row: {}", contents);
        assert!(contents.contains(";/fake/host.mp4;0;60.00;"), "clip row: {}", contents);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_the_csv_names_the_file_each_row_is_talking_about() {
        // The single-figure columns speak for ONE pair, so the CSV has to say
        // which. Here the host's minute of shared footage sits 14 minutes into
        // it and at the very start of the clip -- two different right answers to
        // "where", which is why the column is per-row rather than per-group.
        let fps = vec![
            mock_fp_at("/fake/host.mp4", 600.0),
            mock_fp_at("/fake/clip.mp4", 60.0),
        ];
        let groups = vec![vec![0, 1]];

        let matches = MatchIndex::new(vec![
            Match::new(0, 1, 0.10, 1.0).with_spans((840_000, 900_000), (0, 60_000)),
        ]);

        let path_str = report_to("csv");
        output_results(
            &groups, &fps, &matches, Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let contents = fs::read_to_string(&path_str).unwrap();

        assert!(
            contents.contains("group_1;KEEP;/fake/host.mp4;")
                && contents.contains("/fake/clip.mp4;0;60.00;00:14:00;00:15:00;840.00;900.00"),
            "the host row should point into the host's own timeline: {}",
            contents
        );
        assert!(
            contents.contains("group_1;DELETE;/fake/clip.mp4;")
                && contents.contains("/fake/host.mp4;0;60.00;00:00:00;00:01:00;0.00;60.00"),
            "the clip row should point into the clip's own timeline: {}",
            contents
        );

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_the_json_carries_every_link_not_only_the_strongest() {
        // A three-file group is where one figure stops being the whole story: 0
        // is a full copy of 1 and merely brushes 2. The top-level fields speak
        // for the strongest link, and `matches` holds both so the weak one is
        // still visible.
        let fps = vec![
            mock_fp_at("/fake/a.mp4", 600.0),
            mock_fp_at("/fake/b.mp4", 600.0),
            mock_fp_at("/fake/c.mp4", 600.0),
        ];
        let groups = vec![vec![0, 1, 2]];

        let matches = MatchIndex::new(vec![
            Match::new(0, 1, 1.0, 1.0).with_spans((0, 600_000), (0, 600_000)),
            Match::new(0, 2, 0.01, 0.01).with_spans((0, 6_000), (594_000, 600_000)),
            Match::new(1, 2, 0.01, 0.01).with_spans((0, 6_000), (594_000, 600_000)),
        ]);

        let path_str = report_to("json");
        output_results(
            &groups, &fps, &matches, Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let json = read_json(&path_str);
        let file = &json["results"][0]["files"][0];
        assert_eq!(file["full_path"], "/fake/a.mp4");

        // The headline figures describe the strongest link, and name it.
        assert_eq!(file["matched_with"], "/fake/b.mp4");
        assert_eq!(file["matched_seconds"], 600.0);
        assert_eq!(file["matched_from"], "00:00:00");
        assert_eq!(file["matched_to"], "00:10:00");

        // Both links are present, strongest first, and the weak one is not
        // rounded away or merged into the strong one.
        let links = file["matches"].as_array().expect("matches should be an array");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0]["full_path"], "/fake/b.mp4");
        assert_eq!(links[0]["matched_seconds"], 600.0);
        assert_eq!(links[1]["full_path"], "/fake/c.mp4");
        assert_eq!(links[1]["matched_seconds"], 6.0);
        assert_eq!(links[1]["matched_from"], "00:00:00");
        assert_eq!(links[1]["matched_to"], "00:00:06");

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_chained_member_reports_only_the_links_it_actually_has() {
        // 0-1 and 1-2 were measured; 0 and 2 never were. Clustering would report
        // that as two groups rather than the one this test passes in, so the
        // case is defensive: the reporting pass reads the links it has rather
        // than assuming the group is complete. 0's list must hold one entry, not
        // two with a fabricated zero -- the absent pair is unmeasured, which is
        // a different claim from "these two share nothing".
        let fps = vec![
            mock_fp_at("/fake/a.mp4", 600.0),
            mock_fp_at("/fake/b.mp4", 600.0),
            mock_fp_at("/fake/c.mp4", 600.0),
        ];
        let groups = vec![vec![0, 1, 2]];

        let matches = MatchIndex::new(vec![
            Match::new(0, 1, 1.0, 1.0).with_spans((0, 600_000), (0, 600_000)),
            Match::new(1, 2, 0.5, 0.5).with_spans((0, 300_000), (300_000, 600_000)),
        ]);

        let path_str = report_to("json");
        output_results(
            &groups, &fps, &matches, Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let json = read_json(&path_str);
        let files = json["results"][0]["files"].as_array().unwrap();

        let a = files.iter().find(|f| f["full_path"] == "/fake/a.mp4").unwrap();
        assert_eq!(a["matches"].as_array().unwrap().len(), 1);
        assert_eq!(a["matched_with"], "/fake/b.mp4");

        // b sits in the middle of the chain and has both links.
        let b = files.iter().find(|f| f["full_path"] == "/fake/b.mp4").unwrap();
        assert_eq!(b["matches"].as_array().unwrap().len(), 2);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_single_shared_keyframe_reports_as_a_fraction_of_a_second() {
        // Four short clips from the same camera sharing one incidental frame.
        // Every coverage figure here is a headline-looking 10-25%; every shared
        // duration is under a second, which is what the reader needs.
        let fps = vec![
            mock_fp_at("/fake/bench_29.mp4", 9.0),
            mock_fp_at("/fake/bench_38.mp4", 3.0),
        ];
        let groups = vec![vec![0, 1]];

        let matches = MatchIndex::new(vec![Match::new(0, 1, 0.0714, 0.25)]);

        let path_str = report_to("csv");

        output_results(
            &groups, &fps, &matches, Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let contents = fs::read_to_string(&path_str).unwrap();
        assert!(contents.contains(";0.64;"), "sub-second overlap must stay legible: {}", contents);
        assert!(
            !contents.contains(";00:00:00;"),
            "a real overlap must never render as a zeroed clock: {}",
            contents
        );

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_total_files_matched_counts_every_grouped_file() {
        // Groups partition the matched files, so the total is the sum of their
        // sizes -- and it can be compared directly against the number of videos
        // scanned without any caveat about shared members.
        let fps = vec![
            mock_fp_at("/fake/a.mp4", 100.0),
            mock_fp_at("/fake/b.mp4", 90.0),
            mock_fp_at("/fake/c.mp4", 100.0),
            mock_fp_at("/fake/d.mp4", 10.0),
        ];
        let groups = vec![vec![0, 1], vec![2, 3]];

        let path_str = report_to("json");

        output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let report = read_json(&path_str);

        assert_eq!(report["summary"]["total_groups"], 2);
        assert_eq!(report["summary"]["total_files_matched"], 4);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_json_carries_raw_figures_alongside_the_formatted_ones() {
        let fps = vec![
            mock_fp_at("/fake/a.mp4", 100.0),
            mock_fp_at("/fake/b.mp4", 100.0),
        ];
        let groups = vec![vec![0, 1]];
        let matches = MatchIndex::new(vec![Match::new(0, 1, 0.80, 0.80)]);

        let path_str = report_to("json");

        output_results(
            &groups, &fps, &matches, Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let report = read_json(&path_str);
        let group = &report["results"][0];

        assert_eq!(group["files"][0]["matched_seconds"], 80.0);

        // 1 Mbps at 30fps. The frame rate is the one figure carried raw only --
        // its formatted twin lives on the console line and nowhere else.
        assert_eq!(group["files"][0]["codec"], "h264");
        assert!(group["files"][0]["framerate"].is_null(), "framerate should be raw-only");
        assert_eq!(group["files"][0]["framerate_fps"], 30.0);
        assert_eq!(group["files"][0]["bitrate_bps"], 1_048_576);
        assert_eq!(group["files"][0]["quality_bits_per_frame"], 34_952);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_an_unreported_frame_rate_is_empty_rather_than_zero() {
        let mut lonely = mock_fp_at("/fake/a.mp4", 60.0);
        lonely.frame_rate = 0.0;

        let fps = vec![lonely];
        let groups = vec![vec![0]];
        let json_path = report_to("json");
        let csv_path = report_to("csv");

        output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&json_path)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();
        output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&csv_path)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let report = read_json(&json_path);
        let file = &report["results"][0]["files"][0];

        assert!(file["framerate_fps"].is_null(), "unknown is not 0 fps");
        assert!(file["quality_bits_per_frame"].is_null(), "and it makes quality unknowable");
        assert_eq!(file["quality"], "-");
        assert_eq!(file["bitrate_bps"], 1_048_576, "the bitrate is still perfectly knowable");

        // The CSV says the same thing with empty fields rather than nulls. The
        // frame rate is a single empty cell because it is carried raw only --
        // quality below it still shows the pair, a dash in the formatted column
        // and nothing at all in the raw one.
        let csv = fs::read_to_string(&csv_path).unwrap();
        assert!(
            csv.contains(";1920;1080;;h264;7.5MB;7864320;1.0Mbps;1048576;-;;"),
            "{}",
            csv
        );

        let _ = fs::remove_file(json_path);
        let _ = fs::remove_file(csv_path);
    }

    #[test]
    fn test_permanent_delete_removes_only_delete_targets() {
        // Two real files so we can verify actual filesystem effects. We use
        // permanent deletion here specifically so the test never pollutes the
        // user's trash.
        let dir = tempfile::tempdir().unwrap();
        let keep_path = at(&dir, "keep.mkv");
        let del_path = at(&dir, "duplicate.mkv");

        // Longer duration => higher "tier" => KEEP under Priority::Length.
        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&del_path, 10.0),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();

        let deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();

        assert!(Path::new(&keep_path).exists(), "KEEP file must remain");
        assert!(!Path::new(&del_path).exists(), "DELETE file must be removed");
        assert!(!stats.had_problems(), "a clean deletion must not fail the run");
        assert_eq!(
            deleted,
            vec![del_path],
            "the caller is told exactly which fingerprints to forget"
        );
    }

    #[test]
    fn test_move_to_relocates_a_duplicate_under_a_mirrored_path() {
        // The mode that exists because trash::delete fails on exactly the
        // storage this tool is most useful on: an external disk with no
        // .Trash-1000, an NFS export, a headless box with no XDG trash dir.
        let dir = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();
        let keep_path = at(&dir, "keep.mkv");
        let dup_path = at(&dir, "duplicate.mkv");

        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&dup_path, 10.0),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();
        let path_str = report_to("json");

        let moved = output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length,
            Some(&Disposal::MoveTo(dest.path().to_path_buf())), true, &stats,
        ).unwrap();

        let landed = landing_spot(dest.path(), &dup_path);

        assert!(Path::new(&keep_path).exists(), "the KEEP pick is untouched");
        assert!(!Path::new(&dup_path).exists(), "the duplicate left its original path");
        assert!(landed.exists(), "and arrived at {}", landed.display());
        assert_eq!(
            fs::metadata(&landed).unwrap().len(),
            fps[1].file_size,
            "a move must not change the file"
        );
        assert!(!stats.had_problems());
        assert_eq!(
            moved,
            vec![dup_path],
            "a moved file is no longer at the path its fingerprint was cached under"
        );

        let report = read_json(&path_str);
        assert_eq!(report["results"][0]["files"][1]["action"], "MOVED");
        assert_eq!(report["summary"]["mode"], "move");
        assert_eq!(report["summary"]["files_removed"], 1);
        assert_eq!(report["summary"]["bytes_removed"], 1_310_720u64);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_removing_one_of_several_names_frees_nothing_and_says_so() {
        // "N MB freed" is a claim about the filesystem, and unlinking a hard
        // link with a sibling makes it false: the name goes, every byte stays.
        // `sources::collect` deduplicates on (device, inode), so this is only
        // reachable when the other name is OUTSIDE the scan -- which is the
        // ordinary shape of it, a library that hard-links into a store folder
        // nobody scans.
        //
        // Two DELETE targets rather than one, because the figure has to come
        // out as a partial sum: zeroing it whenever anything was aliased would
        // be a different bug in the same line.
        let dir = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();

        let keep_path = at(&dir, "keep.mkv");
        let linked_path = at(&dir, "linked.mkv");
        let plain_path = at(&dir, "plain.mkv");

        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&linked_path, 10.0),
            mock_fp_at(&plain_path, 11.0),
        ];
        materialize(&fps[0]);
        materialize(&fps[2]);

        // The name that is not in the library, and the one that is.
        let elsewhere = store.path().join("original.mkv");
        fs::File::create(&elsewhere).unwrap().set_len(fps[1].file_size).unwrap();
        fs::hard_link(&elsewhere, &linked_path).unwrap();

        let groups = vec![vec![0, 1, 2]];
        let stats = RunStats::default();
        let path_str = report_to("json");

        let mut gone = output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();
        gone.sort();

        assert!(!Path::new(&linked_path).exists(), "the name it was asked to remove is gone");
        assert!(elsewhere.exists(), "and the data is not");
        assert_eq!(
            fs::metadata(&elsewhere).unwrap().len(),
            fps[1].file_size,
            "untouched, not truncated"
        );
        assert!(!stats.had_problems(), "removing a redundant name is not a failure");

        let mut expected = vec![linked_path, plain_path];
        expected.sort();
        assert_eq!(gone, expected, "both paths are empty, so both fingerprints are stale");

        let report = read_json(&path_str);
        assert_eq!(report["results"][0]["files"][1]["action"], "UNLINKED");
        assert_eq!(report["results"][0]["files"][2]["action"], "DELETED");
        assert_eq!(report["summary"]["files_removed"], 2);
        assert_eq!(report["summary"]["files_unlinked"], 1);
        assert_eq!(
            report["summary"]["bytes_removed"], 1_441_792u64,
            "the plain file's bytes, and only those"
        );
        assert_eq!(
            report["summary"]["bytes_unlinked"], 1_310_720u64,
            "and the ones that stayed are still accounted for"
        );

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_symlink_is_checked_against_its_target_and_frees_none_of_it() {
        // The other half, and the one that also has to keep working the other
        // way round: the scan stats THROUGH a symlink, so the link enters the
        // library carrying the target's size. The staleness check has to follow
        // it too, or a fix that reads only `symlink_metadata` reports every
        // symlinked file as CHANGED and quietly stops deleting any of them.
        //
        // What `remove_file` then takes is the link. The video survives at a
        // path this run never mentioned, so counting its bytes as freed is the
        // same false claim with a worse aftertaste: a re-run finds a clean
        // library, because the path that led to the duplicate is gone.
        let dir = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();

        let keep_path = at(&dir, "keep.mkv");
        let link_path = at(&dir, "link.mkv");

        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&link_path, 10.0),
        ];
        materialize(&fps[0]);

        let target = store.path().join("original.mkv");
        fs::File::create(&target).unwrap().set_len(fps[1].file_size).unwrap();
        std::os::unix::fs::symlink(&target, &link_path).unwrap();

        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();
        let path_str = report_to("json");

        let gone = output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();

        assert!(
            fs::symlink_metadata(&link_path).is_err(),
            "the link was removed, so it was not read as CHANGED"
        );
        assert!(target.exists(), "and its target was not");
        assert_eq!(gone, vec![link_path], "the path is empty, so its fingerprint is stale");
        assert_eq!(stats.delete_stale.count(), 0);

        let report = read_json(&path_str);
        assert_eq!(report["results"][0]["files"][1]["action"], "UNLINKED");
        assert_eq!(report["summary"]["files_removed"], 1);
        assert_eq!(report["summary"]["files_unlinked"], 1);
        assert_eq!(report["summary"]["bytes_unlinked"], 1_310_720u64);
        assert_eq!(report["summary"]["bytes_removed"], 0u64);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_move_to_mirrors_the_source_tree_so_two_files_cannot_collide() {
        // The reason the destination mirrors absolute paths instead of
        // flattening basenames: `Season 1/ep01.mkv` and `Season 2/ep01.mkv` are
        // routine, and a flat destination would need a naming scheme -- a thing
        // that can be got wrong while holding the only remaining copy.
        let dir = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let keep_path = at(&dir, "keep.mkv");
        let a_path = dir.path().join("season_1").join("ep01.mkv");
        let b_path = dir.path().join("season_2").join("ep01.mkv");
        fs::create_dir_all(a_path.parent().unwrap()).unwrap();
        fs::create_dir_all(b_path.parent().unwrap()).unwrap();

        let a = a_path.to_string_lossy().to_string();
        let b = b_path.to_string_lossy().to_string();

        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&a, 10.0),
            mock_fp_at(&b, 10.0),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2]];
        let stats = RunStats::default();

        let mut moved = output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::MoveTo(dest.path().to_path_buf())), true, &stats,
        ).unwrap();
        moved.sort();

        assert!(landing_spot(dest.path(), &a).exists());
        assert!(landing_spot(dest.path(), &b).exists());
        assert_eq!(stats.delete_failed.count(), 0, "same basename, different slots");

        let mut expected = vec![a, b];
        expected.sort();
        assert_eq!(moved, expected);
    }

    #[test]
    fn test_a_rename_carries_a_hard_link_with_it_and_leaves_a_symlink_behind() {
        // The alias rule is not the same question in every mode, and this is the
        // pair that shows it. `--move-to` renames, and rename(2) moves an INODE:
        // a hard link arrives at the destination holding every byte of the
        // video, so that row is MOVED like any other and its bytes belong in the
        // "(N total)" figure. A symlink renamed the same way arrives as a
        // pointer -- the footage never left the store -- so that row is the one
        // the total must leave out.
        //
        // Both were called UNLINKED and both were struck out of the total, which
        // made a run that relocated nothing but hard links report "Moved 1
        // file(s) (0B total)" over a destination holding a complete video.
        let dir = tempfile::tempdir().unwrap();
        let store = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();

        let keep_path = at(&dir, "keep.mkv");
        let linked_path = at(&dir, "linked.mkv");
        let symlinked_path = at(&dir, "symlinked.mkv");

        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&linked_path, 10.0),
            mock_fp_at(&symlinked_path, 11.0),
        ];
        materialize(&fps[0]);

        // Both duplicates have a second name outside the scan, which is the only
        // way either case is reachable: `sources::collect` deduplicates on
        // (device, inode), so no two names for one file ever both reach a
        // DELETE decision.
        let hard_target = store.path().join("hard.mkv");
        fs::File::create(&hard_target).unwrap().set_len(fps[1].file_size).unwrap();
        fs::hard_link(&hard_target, &linked_path).unwrap();

        let sym_target = store.path().join("sym.mkv");
        fs::File::create(&sym_target).unwrap().set_len(fps[2].file_size).unwrap();
        std::os::unix::fs::symlink(&sym_target, &symlinked_path).unwrap();

        let groups = vec![vec![0, 1, 2]];
        let stats = RunStats::default();
        let path_str = report_to("json");

        output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0,
            Priority::Length, Some(&Disposal::MoveTo(dest.path().to_path_buf())), true, &stats,
        ).unwrap();

        // What is really where, before asking what the report called it.
        let landed_hard = landing_spot(dest.path(), &linked_path);
        assert_eq!(
            fs::metadata(&landed_hard).unwrap().len(),
            fps[1].file_size,
            "the whole video is at the destination"
        );
        assert!(hard_target.exists(), "and its sibling name still reaches it too");

        let landed_sym = landing_spot(dest.path(), &symlinked_path);
        assert!(
            fs::symlink_metadata(&landed_sym).unwrap().file_type().is_symlink(),
            "what arrived is the pointer, not the video"
        );
        assert_eq!(
            fs::metadata(&sym_target).unwrap().len(),
            fps[2].file_size,
            "the footage never moved"
        );

        let report = read_json(&path_str);
        let action_for = |path: &str| -> String {
            report["results"][0]["files"]
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["full_path"] == path)
                .unwrap()["action"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert_eq!(action_for(&linked_path), "MOVED", "a rename took the data with it");
        assert_eq!(action_for(&symlinked_path), "UNLINKED", "and this one did not");

        assert_eq!(report["summary"]["files_removed"], 2);
        assert_eq!(report["summary"]["files_unlinked"], 1);
        assert_eq!(
            report["summary"]["bytes_removed"], fps[1].file_size,
            "the hard link's bytes are at the destination, so they count"
        );
        assert_eq!(
            report["summary"]["bytes_unlinked"], fps[2].file_size,
            "and the symlink's are still in the store, so they do not"
        );
        assert!(!stats.had_problems());

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_move_to_never_overwrites_what_an_earlier_run_put_there() {
        // rename(2) replaces an existing file silently. Reachable whenever a
        // path is moved away, recreated, and moved away again -- and the file
        // it would destroy is one the user deliberately kept out of the trash.
        let dir = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();
        let keep_path = at(&dir, "keep.mkv");
        let dup_path = at(&dir, "duplicate.mkv");

        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&dup_path, 10.0),
        ];
        materialize_all(&fps);

        let landed = landing_spot(dest.path(), &dup_path);
        fs::create_dir_all(landed.parent().unwrap()).unwrap();
        fs::write(&landed, b"an earlier run's copy").unwrap();

        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();

        let moved = output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::MoveTo(dest.path().to_path_buf())), true, &stats,
        ).unwrap();

        assert!(Path::new(&dup_path).exists(), "the source must survive a refused move");
        assert_eq!(
            fs::read(&landed).unwrap(),
            b"an earlier run's copy",
            "and the file already there must be exactly as it was"
        );
        assert_eq!(stats.delete_failed.count(), 1);
        assert!(stats.had_problems(), "a move the run was told to make and did not make");
        assert!(moved.is_empty(), "nothing moved, so no fingerprint may be forgotten");
    }

    #[test]
    fn test_a_target_that_changed_since_the_scan_is_not_moved_either() {
        // The staleness guard is about the decision, not about the disposal, so
        // it has to hold in every mode. A file that grew is no longer the file
        // that was judged redundant, whether it was about to be deleted or
        // quarantined.
        let dir = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();
        let keep_path = at(&dir, "keep.mkv");
        let grew_path = at(&dir, "still_downloading.mkv");

        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&grew_path, 10.0),
        ];
        materialize_all(&fps);
        fs::File::create(&grew_path)
            .unwrap()
            .set_len(fps[1].file_size + 4096)
            .unwrap();

        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();

        let moved = output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::MoveTo(dest.path().to_path_buf())), true, &stats,
        ).unwrap();

        assert!(Path::new(&grew_path).exists());
        assert!(!landing_spot(dest.path(), &grew_path).exists());
        assert_eq!(stats.delete_stale.count(), 1);
        assert!(moved.is_empty());
    }

    #[test]
    fn test_a_target_that_changed_since_the_scan_is_not_deleted() {
        // The window this guard exists for. The DELETE decision was made from a
        // measurement taken at the start of a scan that may have run for hours;
        // by the time we reach the file it has finished downloading, and it is
        // no longer the file that was judged redundant.
        let dir = tempfile::tempdir().unwrap();
        let keep_path = at(&dir, "keep.mkv");
        let grew_path = at(&dir, "still_downloading.mkv");

        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&grew_path, 10.0),
        ];
        materialize_all(&fps);

        // ...and then it grew.
        fs::File::create(&grew_path)
            .unwrap()
            .set_len(fps[1].file_size + 4096)
            .unwrap();

        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();
        let path_str = report_to("json");

        let deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();

        assert!(Path::new(&grew_path).exists(), "a file that changed under us must survive");
        assert!(Path::new(&keep_path).exists(), "and so must the KEEP pick");
        assert!(
            deleted.is_empty(),
            "nothing was removed, so no fingerprint may be forgotten"
        );
        assert_eq!(stats.delete_stale.count(), 1);
        assert!(
            stats.had_problems(),
            "a deletion the run was told to make and did not make must fail the run"
        );

        let report = read_json(&path_str);
        assert_eq!(report["results"][0]["files"][1]["action"], "CHANGED");
        assert_eq!(report["summary"]["files_changed"], 1);
        assert_eq!(report["summary"]["files_removed"], 0);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_truncated_target_is_caught_just_as_a_grown_one_is() {
        // The other direction, and the more dangerous one: a copy that was
        // truncated is now demonstrably not the video that was compared, so
        // deleting it on the strength of that comparison is deleting something
        // nobody looked at.
        let dir = tempfile::tempdir().unwrap();
        let keep_path = at(&dir, "keep.mkv");
        let shrunk_path = at(&dir, "truncated.mkv");

        let fps = vec![
            mock_fp_at(&keep_path, 60.0),
            mock_fp_at(&shrunk_path, 10.0),
        ];
        materialize_all(&fps);

        fs::File::create(&shrunk_path).unwrap().set_len(1024).unwrap();

        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();

        output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();

        assert!(Path::new(&shrunk_path).exists());
        assert_eq!(stats.delete_stale.count(), 1);
        assert_eq!(
            stats.delete_failed.count(), 0,
            "refusing to act on stale information is not a failed removal"
        );
    }

    /// A chain: 0-1 and 1-2 matched, 0 and 2 never were. Clustering turns that
    /// into two overlapping groups -- [0,1] and [1,2] -- rather than one group
    /// of three, so file 1 is reported twice with a different role each time.
    fn chain_of_three() -> MatchIndex {
        MatchIndex::new(vec![
            Match::new(0, 1, 1.0, 1.0),
            Match::new(1, 2, 1.0, 1.0),
        ])
    }

    #[test]
    fn test_a_chain_collapses_in_a_single_pass() {
        // Each group settles its own ranking -- 1 loses to 0, then 2 loses to 1
        // -- and DELETE is resolved across all of them, so both losers go in one
        // pass and the best copy of the chain is what is left. Nothing survives
        // to be caught by a later run, which is the point: a per-group rule that
        // spared 2 because its own group's winner is being removed would need
        // the user to re-run until the chain had collapsed a hop at a time.
        //
        // The step worth naming: 2 was never measured against 0. It was measured
        // against 1, and 1 was measured against 0. Every hop rests on a direct
        // comparison; the chain of hops does not.
        let dir = tempfile::tempdir().unwrap();
        let p0 = at(&dir, "best.mkv");
        let p1 = at(&dir, "bridge.mkv");
        let p2 = at(&dir, "tail.mkv");

        let fps = vec![
            mock_fp_at(&p0, 100.0), // best of the chain
            mock_fp_at(&p1, 90.0),  // bridge
            mock_fp_at(&p2, 10.0),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1], vec![1, 2]];

        let mut deleted = output_results(
            &groups, &fps, &chain_of_three(), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();
        deleted.sort();

        assert!(Path::new(&p0).exists(), "the chain's best must remain");
        assert!(!Path::new(&p1).exists(), "bridge duplicate must be deleted in one pass");
        assert!(!Path::new(&p2).exists(), "tail duplicate must be deleted");

        let mut expected = vec![p1, p2];
        expected.sort();
        assert_eq!(deleted, expected);
    }

    #[test]
    fn test_a_file_marked_delete_anywhere_reads_delete_everywhere() {
        // File 1 is the best copy of its second group and redundant in its
        // first. It is removed, and BOTH of its rows say so -- a report whose
        // group_2 called it KEEP while the file was being deleted underneath
        // would be describing a run that never happened.
        let dir = tempfile::tempdir().unwrap();
        let p0 = at(&dir, "best.mkv");
        let p1 = at(&dir, "bridge.mkv");
        let p2 = at(&dir, "tail.mkv");

        let fps = vec![
            mock_fp_at(&p0, 100.0),
            mock_fp_at(&p1, 90.0),
            mock_fp_at(&p2, 10.0),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1], vec![1, 2]];
        let path_str = report_to("json");

        let deleted = output_results(
            &groups, &fps, &chain_of_three(), Some(&to_file(&path_str)), 0,
            Priority::Length, None, true, &RunStats::default(),
        ).unwrap();

        assert!(deleted.is_empty(), "report-only, so nothing moved");

        let report = read_json(&path_str);
        assert_eq!(report["results"][0]["files"][1]["action"], "DELETE");
        assert_eq!(
            report["results"][1]["files"][0]["action"], "DELETE",
            "the same file, in the group where it won the ranking"
        );
        assert_eq!(
            report["summary"]["files_marked_delete"], 2,
            "and it is counted once, not once per group"
        );
        assert_eq!(
            report["summary"]["total_files_matched"], 3,
            "three distinct files, though the groups list four rows between them"
        );

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_review_in_one_group_protects_a_file_in_every_other() {
        // File 1 is held for review by a codec standoff in its first group and
        // ranked bottom in its second. REVIEW outranks DELETE globally, so it
        // survives: the flag says a human has to look, and a run that removed
        // the file on another group's say-so would have answered the question
        // for them.
        let dir = tempfile::tempdir().unwrap();
        let p0 = at(&dir, "h264.mkv");
        let p1 = at(&dir, "av1.mkv");
        let p2 = at(&dir, "longer.mkv");

        let fps = vec![
            mock_fp_coded(&p0, 100.0, "h264"),
            mock_fp_coded(&p1, 100.0, "av1"),
            mock_fp_coded(&p2, 200.0, "h264"),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1], vec![1, 2]];
        let path_str = report_to("json");

        let deleted = output_results(
            &groups, &fps, &chain_of_three(), Some(&to_file(&path_str)), 0,
            Priority::Length, Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p1).exists(), "a file held for review is never removed");
        assert!(deleted.is_empty());

        let report = read_json(&path_str);
        assert_eq!(report["results"][1]["files"][0]["action"], "REVIEW");
        assert_eq!(report["summary"]["files_marked_delete"], 0);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_codec_standoff_anchors_the_guard_on_every_champion() {
        // In a standoff there is no KEEP pick to match against -- each codec
        // keeps its own champion. A file directly matched against ANY survivor
        // has its evidence, so the 720p copy below is still deleted even though
        // it never matched the av1 champion.
        let dir = tempfile::tempdir().unwrap();
        let p_h264 = at(&dir, "full_h264.mkv");
        let p_av1 = at(&dir, "full_av1.mkv");
        let p_small = at(&dir, "small_h264.mkv");

        let fp_h264 = mock_fp_coded(&p_h264, 60.0, "h264");
        let fp_av1 = mock_fp_coded(&p_av1, 60.0, "av1");
        let mut fp_small = mock_fp_coded(&p_small, 60.0, "h264");
        fp_small.width = 1280;
        fp_small.height = 720;

        let fps = vec![fp_h264, fp_av1, fp_small];
        materialize_all(&fps);

        // 2 matched only the h264 champion, never the av1 one.
        let matches = MatchIndex::new(vec![
            Match::new(0, 1, 1.0, 1.0),
            Match::new(0, 2, 1.0, 1.0),
        ]);

        let groups = vec![vec![0, 1, 2]];

        let deleted = output_results(
            &groups, &fps, &matches, None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_h264).exists(), "1080p h264 is a champion");
        assert!(Path::new(&p_av1).exists(), "1080p av1 is a champion");
        assert!(
            !Path::new(&p_small).exists(),
            "matching one surviving champion is evidence enough"
        );
        assert_eq!(deleted, vec![p_small]);
    }

    #[test]
    fn test_a_group_keeps_exactly_one_file() {
        // The rule the partition buys: one survivor per group. Files 0 and 1 are
        // equally good and would each have won a group of their own under the
        // old overlapping cliques; here they are related, so they are ranked
        // against each other and only one is kept.
        let dir = tempfile::tempdir().unwrap();
        let p0 = at(&dir, "a_best.mkv");
        let p1 = at(&dir, "b_equal.mkv");
        let p2 = at(&dir, "c_short.mkv");

        let fps = vec![
            mock_fp_at(&p0, 60.0),
            mock_fp_at(&p1, 60.0),
            mock_fp_at(&p2, 10.0),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2]];
        let stats = RunStats::default();
        let path_str = report_to("json");

        let mut deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();
        deleted.sort();

        assert!(Path::new(&p0).exists(), "the tiebreak keeps the first path");
        assert!(!Path::new(&p1).exists(), "an equal copy is still a redundant copy");
        assert!(!Path::new(&p2).exists());
        assert_eq!(stats.delete_failed.count(), 0);

        let mut expected = vec![p1, p2];
        expected.sort();
        assert_eq!(deleted, expected, "each target is handed back exactly once");

        let actions: Vec<String> = read_json(&path_str)["results"][0]["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["action"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            actions.iter().filter(|a| *a == "KEPT").count(),
            1,
            "exactly one survivor: {:?}",
            actions
        );

        let _ = fs::remove_file(path_str);
    }

    /// Each report body is now built only when it is the one being written, and
    /// this is what keeps that from quietly writing an empty file.
    ///
    /// The saving itself is not observable from out here -- a run writes the
    /// same bytes either way, and what changed is the two bodies it no longer
    /// builds on the way -- so what is testable is the half that could break:
    /// every format still says everything it said. The text one had no test at
    /// all before, which is exactly the body a gating mistake would empty.
    #[test]
    fn test_each_format_still_carries_its_whole_body_now_that_only_one_is_built() {
        let fps = vec![mock_fp_at("/fake/keep.mkv", 60.0), mock_fp_at("/fake/dupe.mkv", 10.0)];
        let groups = vec![vec![0, 1]];

        let write = |extension: &str| {
            let path_str = report_to(extension);
            output_results(
                &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 3661,
                Priority::Length, None, true, &RunStats::default(),
            ).unwrap();
            let contents = fs::read_to_string(&path_str).unwrap();
            let _ = fs::remove_file(&path_str);
            contents
        };

        // The text body is the listing plus the summary, and both halves of it
        // are accumulated inside the per-group loop the gate now sits in.
        let txt = write("txt");
        assert!(txt.contains("group_1:"), "{}", txt);
        assert!(txt.contains("/fake/keep.mkv"), "{}", txt);
        assert!(txt.contains("/fake/dupe.mkv"), "{}", txt);
        assert!(txt.contains("KEEP"), "{}", txt);
        assert!(txt.contains("DELETE"), "{}", txt);
        assert!(txt.contains("Total groups found: 1"), "{}", txt);
        assert!(txt.contains("Total time elapsed: 01:01:01"), "{}", txt);

        // The CSV is the header and a row per file, both behind the same gate.
        let csv = write("csv");
        assert!(csv.contains(CSV_HEADER), "{}", csv);
        assert_eq!(csv.lines().count(), 3, "header and one row per file: {}", csv);
        assert!(csv.contains(";KEEP;/fake/keep.mkv;"), "{}", csv);
        assert!(csv.contains(";DELETE;/fake/dupe.mkv;"), "{}", csv);

        // And the JSON, whose gate is the one that was already here.
        let json: serde_json::Value = serde_json::from_str(&write("json")).unwrap();
        assert_eq!(json["summary"]["total_groups"], 1);
        assert_eq!(json["results"][0]["files"].as_array().unwrap().len(), 2);
        assert_eq!(json["results"][0]["files"][0]["full_path"], "/fake/keep.mkv");
    }

    #[test]
    fn test_every_report_format_that_records_a_size_can_be_handed_straight_back() {
        // The writer and the readers, tied together. `report.rs` locates three
        // fields in each format and this is what proves it is locating the ones
        // THIS module wrote -- in the same shape, with the same spelling of
        // DELETE, and with a size the staleness check will agree with. A report
        // this tool wrote and cannot replay is the failure worth a test of its
        // own, because everything else about it looks right.
        // Each format under the extension that implies it, and then under one
        // that implies nothing. The second pair is what `--format` made
        // writable, and it is the round trip that broke when only the writer
        // was decoupled: a JSON report called .bak went to the CSV reader and
        // was refused for want of columns it was never going to have.
        let cases: [(&str, Option<Format>); 4] = [
            ("csv", None),
            ("json", None),
            ("bak", Some(Format::Csv)),
            ("bak", Some(Format::Json)),
        ];

        for (extension, format) in cases {
            let case = format!(".{} as {:?}", extension, format);

            let dir = tempfile::tempdir().unwrap();
            let keep = at(&dir, "long.mkv");
            let doomed = at(&dir, "short.mkv");

            let fps = vec![mock_fp_at(&keep, 60.0), mock_fp_at(&doomed, 10.0)];
            materialize_all(&fps);

            let groups = vec![vec![0, 1]];
            let path_str = report_to(extension);
            let target = match format {
                Some(f) => ReportTarget { sink: Sink::File(PathBuf::from(&path_str)), format: f },
                None => to_file(&path_str),
            };

            // Report-only: nothing is touched, and the report is the whole
            // output -- which is the run --from-report exists to follow.
            output_results(
                &groups, &fps, &all_compared(fps.len()), Some(&target), 0, Priority::Length,
                None, true, &RunStats::default(),
            ).unwrap();

            assert!(Path::new(&doomed).exists(), "{}: nothing is removed yet", case);

            let stats = RunStats::default();
            let gone = crate::report::apply(&path_str, &Disposal::Permanent, true, &stats).unwrap();

            assert_eq!(gone, vec![doomed.clone()], "{}", case);
            assert!(!Path::new(&doomed).exists(), "{}: the DELETE row was acted on", case);
            assert!(Path::new(&keep).exists(), "{}: and nothing else was", case);
            assert!(!stats.had_problems(), "{}: every row was understood", case);

            let _ = fs::remove_file(path_str);
        }
    }

    #[test]
    fn test_quality_settles_groups_that_tie_on_length_and_resolution() {
        // Same length, same resolution, same codec: under the default order the
        // decision reaches quality, and the denser copy is kept. Nothing is
        // flagged REVIEW, because the KEEP pick is top-tier on every metric.
        let dir = tempfile::tempdir().unwrap();
        let p_hi = at(&dir, "high.mkv");
        let p_lo = at(&dir, "low.mkv");

        let mut fp_hi = mock_fp_at(&p_hi, 60.0);
        fp_hi.file_size *= 2; // 2 Mbps
        let fp_lo = mock_fp_at(&p_lo, 60.0); // 1 Mbps

        let fps = vec![fp_hi, fp_lo];
        materialize_all(&fps);

        let groups = vec![vec![0, 1]];

        output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_hi).exists(), "higher-quality copy must be kept");
        assert!(!Path::new(&p_lo).exists(), "lower-quality copy must be deleted");
    }

    #[test]
    fn test_a_codec_standoff_is_left_for_a_human() {
        // Same footage, same length, same resolution, different codecs, one
        // copy each. Every metric that could break the tie is a bit count, and
        // bit counts say nothing across codecs -- the av1 copy being half the
        // size is what av1 is FOR. Each file is the whole of its codec's field,
        // so each is elected champion and neither may be deleted, even with
        // --delete --permanent armed.
        let dir = tempfile::tempdir().unwrap();
        let p_h264 = at(&dir, "copy_h264.mkv");
        let p_av1 = at(&dir, "copy_av1.mkv");

        let fp_h264 = mock_fp_coded(&p_h264, 60.0, "h264");
        let mut fp_av1 = mock_fp_coded(&p_av1, 60.0, "av1");
        fp_av1.file_size /= 2; // half the bytes, same picture

        let fps = vec![fp_h264, fp_av1];
        materialize_all(&fps);

        let groups = vec![vec![0, 1]];
        let path_str = report_to("json");

        let deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_h264).exists(), "the h264 copy must survive");
        assert!(Path::new(&p_av1).exists(), "the av1 copy must survive");
        assert!(deleted.is_empty(), "a standoff deletes nothing, so it forgets nothing");

        let report = read_json(&path_str);
        let files = &report["results"][0]["files"];
        assert_eq!(files[0]["action"], "REVIEW");
        assert_eq!(files[1]["action"], "REVIEW");

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_file_whose_length_nobody_measured_is_never_the_one_deleted() {
        // The reported case: a 20 second raw H.264 elementary stream holding a
        // 10 second MP4 of the same footage. No container runtime, no clock on
        // the packets, so the ranking has no length for it at all -- and the
        // metrics left over are no substitute, since size prefers whichever
        // file spent more bits and says nothing about which contains which.
        // Neither file may be deleted, exactly as in a codec standoff.
        let dir = tempfile::tempdir().unwrap();
        let p_raw = at(&dir, "long.h264");
        let p_clip = at(&dir, "clip.mp4");

        let mut fp_raw = mock_fp_at(&p_raw, 0.0);
        fp_raw.file_size = 141_000; // twice the clip, being twice the footage
        let mut fp_clip = mock_fp_at(&p_clip, 10.0);
        fp_clip.file_size = 73_000;

        let fps = vec![fp_raw, fp_clip];
        materialize_all(&fps);

        let groups = vec![vec![0, 1]];
        let path_str = report_to("json");

        let deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_raw).exists(), "the file of unknown length must survive");
        assert!(Path::new(&p_clip).exists(), "and so must the one it was not ranked against");
        assert!(deleted.is_empty());

        let report = read_json(&path_str);
        let files = &report["results"][0]["files"];
        assert_eq!(files[0]["action"], "REVIEW");
        assert_eq!(files[1]["action"], "REVIEW");
        // The row says unknown rather than 00:00:00, because a zero-length
        // video and a video of unknown length are different findings.
        let raw_row = if files[0]["full_path"] == serde_json::json!(p_raw) { &files[0] } else { &files[1] };
        assert_eq!(raw_row["length"], "-");
        assert!(raw_row["length_seconds"].is_null(), "and nothing to sort as a zero");
        // Its matched footage is stated in its own seconds, so it is unknown
        // for the same reason rather than zero. The link is still named.
        assert!(raw_row["matched_seconds"].is_null(), "no runtime, no seconds to state");
        assert_eq!(raw_row["matched_with"], serde_json::json!(p_clip));
        assert!(raw_row["matches"][0]["matched_seconds"].is_null());

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_group_of_nothing_but_unmeasured_lengths_deletes_nothing() {
        // A library of raw streams: no runtime anywhere, so there is no length
        // comparison to be had between any two of them and the rule DELETE
        // rests on cannot be applied at all. Every file lives, which is the
        // honest answer rather than a conservative one -- and the empty
        // measured side must not be asked to elect a champion.
        let dir = tempfile::tempdir().unwrap();
        let paths: Vec<String> =
            ["a.h264", "b.h264", "c.h264"].iter().map(|n| at(&dir, n)).collect();

        let fps: Vec<VideoFingerprint> = paths
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut fp = mock_fp_at(p, 0.0);
                fp.file_size = 1_000_000 * (i as u64 + 1); // and size cannot stand in
                fp
            })
            .collect();
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2]];
        let deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(deleted.is_empty());
        for p in &paths {
            assert!(Path::new(p).exists(), "{} must survive", p);
        }
    }

    #[test]
    fn test_an_unmeasured_length_does_not_protect_the_rest_of_the_group() {
        // One file of unknown length beside two that were measured. The
        // unknown one is incomparable, not the group's winner, so the two that
        // ARE comparable still rank against each other: the 720p copy lost to
        // the 1080p one on a metric both of them were measured on, and it goes.
        // The 1080p copy is held beside the unknown file for the same reason a
        // codec champion is -- nothing ranked the two of them against each
        // other.
        let dir = tempfile::tempdir().unwrap();
        let p_raw = at(&dir, "capture.h264");
        let p_full = at(&dir, "full.mkv");
        let p_small = at(&dir, "small.mkv");

        let mut fp_raw = mock_fp_at(&p_raw, 0.0);
        fp_raw.file_size = 20_000_000; // the biggest file, and unrankable
        let fp_full = mock_fp_at(&p_full, 60.0);
        let mut fp_small = mock_fp_at(&p_small, 60.0);
        fp_small.width = 1280;
        fp_small.height = 720;

        let fps = vec![fp_raw, fp_full, fp_small];
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2]];
        let path_str = report_to("json");

        output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_raw).exists(), "unknown length is never a reason to delete");
        assert!(Path::new(&p_full).exists(), "the measured side's champion lives beside it");
        assert!(!Path::new(&p_small).exists(), "720p lost to 1080p, both of them measured");

        let report = read_json(&path_str);
        let files = &report["results"][0]["files"];
        let action_of = |path: &str| -> String {
            files
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["full_path"] == serde_json::json!(path))
                .unwrap()["action"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert_eq!(action_of(&p_raw), "REVIEW");
        assert_eq!(action_of(&p_full), "REVIEW");
        assert_eq!(action_of(&p_small), "DELETED", "armed, so the row is past tense");

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_codec_standoff_does_not_protect_the_rest_of_the_group() {
        // Two 1080p copies in different codecs deadlock, and a 720p copy of the
        // same footage sits under both of them. Resolution is codec-independent,
        // so nothing about the deadlock makes the 720p file worth keeping.
        let dir = tempfile::tempdir().unwrap();
        let p_h264 = at(&dir, "full_h264.mkv");
        let p_av1 = at(&dir, "full_av1.mkv");
        let p_small = at(&dir, "small_h264.mkv");

        let fp_h264 = mock_fp_coded(&p_h264, 60.0, "h264");
        let fp_av1 = mock_fp_coded(&p_av1, 60.0, "av1");
        let mut fp_small = mock_fp_coded(&p_small, 60.0, "h264");
        fp_small.width = 1280;
        fp_small.height = 720;

        let fps = vec![fp_h264, fp_av1, fp_small];
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2]];

        output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_h264).exists(), "1080p h264 is a contender");
        assert!(Path::new(&p_av1).exists(), "1080p av1 is a contender");
        assert!(!Path::new(&p_small).exists(), "720p loses on a codec-blind metric");
    }

    #[test]
    fn test_a_standoff_keeps_the_best_copy_of_each_codec_and_no_more() {
        // The shape of a real transcode pile: several encodes per codec, all of
        // the same footage at the same length and resolution. Each codec's best
        // copy is held for review because nothing can rank it against the other
        // codecs' bests -- but the also-rans lost to a file they ARE comparable
        // with, so they go, and the group does not need a human to look at
        // fifteen rows.
        let dir = tempfile::tempdir().unwrap();
        let h264_best = at(&dir, "h264_best.mkv");
        let h264_mid = at(&dir, "h264_mid.mkv");
        let hevc_best = at(&dir, "hevc_best.mkv");
        let hevc_worst = at(&dir, "hevc_worst.mkv");
        let av1_only = at(&dir, "av1_only.mkv");

        let fps = vec![
            mock_fp_sized(&h264_best, 60.0, "h264", 24_000_000),
            mock_fp_sized(&h264_mid, 60.0, "h264", 9_000_000),
            mock_fp_sized(&hevc_best, 60.0, "hevc", 12_000_000),
            mock_fp_sized(&hevc_worst, 60.0, "hevc", 3_000_000),
            mock_fp_sized(&av1_only, 60.0, "av1", 6_000_000),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2, 3, 4]];

        output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&h264_best).exists(), "best h264 is its codec's champion");
        assert!(Path::new(&hevc_best).exists(), "best hevc is its codec's champion");
        assert!(Path::new(&av1_only).exists(), "the lone av1 is its codec's champion");
        assert!(!Path::new(&h264_mid).exists(), "beaten by a comparable h264 copy");
        assert!(!Path::new(&hevc_worst).exists(), "beaten by a comparable hevc copy");
    }

    #[test]
    fn test_a_champion_is_elected_on_quality_not_on_filename() {
        // Two h264 encodes four percent apart: inside both tolerance bands, so
        // every tier ties and the quality figure is the only thing left. The
        // election runs against the codec's own contenders, so the av1 file
        // neither sets the bar the two h264 copies are tiered against nor gets
        // to be separated from them on bits -- and the WORSE h264 copy does not
        // win the codec by sorting first.
        let dir = tempfile::tempdir().unwrap();
        let a_worse = at(&dir, "a_worse.mkv");
        let z_best = at(&dir, "z_best.mkv");
        let other_codec = at(&dir, "m_av1.mkv");

        let fps = vec![
            mock_fp_sized(&a_worse, 60.0, "h264", 9_600_000),
            mock_fp_sized(&z_best, 60.0, "h264", 10_000_000),
            mock_fp_sized(&other_codec, 60.0, "av1", 4_000_000),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2]];

        output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&z_best).exists(), "the denser h264 copy must win its codec");
        assert!(
            !Path::new(&a_worse).exists(),
            "sorting first is not a reason to survive a codec you lost"
        );
        assert!(Path::new(&other_codec).exists(), "the av1 copy stands on its own");
    }

    #[test]
    fn test_a_foreign_codec_that_is_not_even_a_contender_decides_nothing() {
        // The same two h264 encodes, but the av1 file is 720p and so loses the
        // group on resolution before bits are ever consulted. That makes it a
        // bystander rather than a contender, no standoff fires, and the whole
        // decision rests on the group-wide ranking -- which used to suppress
        // quality and size for every file the moment ANY foreign codec was
        // present, hand the group to alphabetical order, and delete the better
        // copy with nothing flagged for review.
        let dir = tempfile::tempdir().unwrap();
        let a_worse = at(&dir, "a_worse.mkv");
        let z_best = at(&dir, "z_best.mkv");
        let bystander = at(&dir, "m_av1_720p.mkv");

        let mut small = mock_fp_sized(&bystander, 60.0, "av1", 4_000_000);
        small.width = 1280;
        small.height = 720;

        let fps = vec![
            mock_fp_sized(&a_worse, 60.0, "h264", 9_600_000),
            mock_fp_sized(&z_best, 60.0, "h264", 10_000_000),
            small,
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2]];

        output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&z_best).exists(), "the denser h264 copy is the one to keep");
        assert!(
            !Path::new(&a_worse).exists(),
            "sorting first is not a reason to survive a copy that beat you on bits"
        );
    }

    #[test]
    fn test_a_standoff_cannot_delete_the_copy_the_group_ranked_first() {
        // The two elections disagree, and the disagreement used to condemn the
        // file the group itself chose to keep.
        //
        // `a_pick` and `c_denser` are both full-length, full-resolution h264
        // copies, half a second apart -- inside DURATION_TOLERANCE_SECS, so
        // both are contenders. The two short clips are not, and they are what
        // sets the trap: each is dense enough to own its codec's quality and
        // size maxima outright, so group-wide every long copy is tier 0 on both
        // and the ranking falls through to raw length, where `a_pick` wins by
        // its extra half second. Re-tiered against the h264 contenders ALONE
        // the clip is gone, `c_denser` owns the bar, and the election crowns it
        // instead -- so the champion replaced the pick, and `a_pick` was left
        // unprotected and deleted in favour of a copy it had just outranked.
        let dir = tempfile::tempdir().unwrap();
        let a_pick = at(&dir, "a_pick.mkv");
        let c_denser = at(&dir, "c_denser.mkv");
        let b_av1 = at(&dir, "b_av1.mkv");
        let h264_clip = at(&dir, "h264_clip.mkv");
        let av1_clip = at(&dir, "av1_clip.mkv");

        let fps = vec![
            mock_fp_sized(&a_pick, 60.0, "h264", 27_000_000),
            mock_fp_sized(&c_denser, 59.5, "h264", 30_000_000),
            mock_fp_sized(&b_av1, 60.0, "av1", 12_000_000),
            mock_fp_sized(&h264_clip, 5.0, "h264", 40_000_000),
            mock_fp_sized(&av1_clip, 5.0, "av1", 40_000_000),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2, 3, 4]];

        output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(
            Path::new(&a_pick).exists(),
            "the file the group's own ranking chose must never be the one deleted"
        );
        assert!(Path::new(&b_av1).exists(), "the av1 contender is its codec's champion");
        assert!(
            !Path::new(&c_denser).exists(),
            "losing the group ranking is still a loss: one survivor per codec"
        );
        assert!(!Path::new(&h264_clip).exists(), "a clip is not a contender");
        assert!(!Path::new(&av1_clip).exists(), "nor is the other one");
    }

    #[test]
    fn test_a_shorter_copy_still_loses_to_a_different_codec() {
        // The standoff rule keys on the contenders, not on the group. These two
        // are not tied: one is a minute longer, which is true regardless of what
        // encoded it, so the shorter one is deleted exactly as before.
        let dir = tempfile::tempdir().unwrap();
        let p_long = at(&dir, "long_av1.mkv");
        let p_short = at(&dir, "short_h264.mkv");

        let fps = vec![
            mock_fp_coded(&p_long, 120.0, "av1"),
            mock_fp_coded(&p_short, 60.0, "h264"),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1]];

        output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_long).exists(), "the longer copy is the KEEP pick");
        assert!(!Path::new(&p_short).exists(), "a genuine loss is still a loss");
    }

    #[test]
    fn test_failed_deletion_is_recorded_for_the_summary_and_exit_code() {
        // A path that cannot be stat'd proves nothing about what the file
        // contained, so the re-check waves it through and the removal itself
        // reports the truth: this is a deletion that could not happen, not a
        // file that changed underneath us.
        let dir = tempfile::tempdir().unwrap();
        let keep_path = at(&dir, "keep.mkv");
        let missing = "/nonexistent/vid-fp/definitely-not-here.mp4".to_string();

        let fps = vec![mock_fp_at(&keep_path, 60.0), mock_fp_at(&missing, 10.0)];
        materialize(&fps[0]);

        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();

        let deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();

        assert_eq!(stats.delete_failed.count(), 1, "the failure must be tallied");
        assert_eq!(
            stats.delete_stale.count(), 0,
            "an unreadable path is not evidence that the file changed"
        );
        assert!(stats.had_problems(), "a failed deletion must fail the run");
        assert!(Path::new(&keep_path).exists(), "the KEEP pick is untouched either way");
        assert!(
            deleted.is_empty(),
            "a file that is still on disk must keep the fingerprint that describes it"
        );
    }
}