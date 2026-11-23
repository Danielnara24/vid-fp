mod render;
mod stream;

use anyhow::{anyhow, Context, Result};
use log::info;
use crate::compare::MatchIndex;
use crate::confirm::{self, Target};
use crate::fingerprint::VideoFingerprint;
use crate::stats::RunStats;
use crate::utils::{find_best, format_size, measurable, outranks, shutdown_requested, GroupMaxima, Priority};
use render::{console_for, open_report, Body, Entry, Listing, Summary};
use std::collections::{HashMap, HashSet};
use std::fs::{FileTimes, OpenOptions};
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


/// What one look at the disk says about a file, taken immediately before the
/// irreversible step.
///
/// Two questions, one `symlink_metadata`: is this still the file that was
/// judged, and would removing this path actually give its bytes back.
struct OnDisk {
    /// The file's current length, or `None` when it could not be read. See
    /// `changed_since` for why that is not treated as a change.
    size: Option<u64>,
    /// Its change time, read from the same stat; `None` with `size`.
    ctime: Option<(i64, i64)>,
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
        return OnDisk { size: None, ctime: None, alias: Alias::None };
    };

    if md.file_type().is_symlink() {
        // A broken link has no size to check and is still an alias: what goes
        // when it goes is a pointer. The change time is the target's, which is
        // what the walk read through the link.
        let target = std::fs::metadata(path).ok();
        OnDisk {
            size: target.as_ref().map(|m| m.len()),
            ctime: target.as_ref().map(|m| (m.ctime(), m.ctime_nsec())),
            alias: Alias::Symlink,
        }
    } else {
        let alias = if md.nlink() > 1 { Alias::HardLink } else { Alias::None };
        OnDisk { size: Some(md.len()), ctime: Some((md.ctime(), md.ctime_nsec())), alias }
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

/// Whether the file at `path` is still the file that was measured, and a
/// description of the discrepancy if not.
///
/// Every DELETE decision rests on measurements taken at the start of the scan,
/// and a scan of a large library runs for hours -- longer still when the
/// decision is being replayed from a report written yesterday. In that window a
/// download finishes, a re-encode lands over the top, a copy is truncated --
/// and the file about to be removed is no longer the file that was judged
/// redundant. A stat immediately before an irreversible syscall is a cheap way
/// to notice, and `on_disk` has already taken it -- `current` is what it read.
///
/// Two properties, the second only where the measurement recorded one. The
/// length catches essentially every way a video file is rewritten. The change
/// time catches the rest: a file replaced by another of the same length, which
/// `cp -p` and `rsync -a` do while putting the old mtime back, and which the
/// window has always refused (`last::changed_since`) while this, its
/// command-line counterpart, compared the length alone. A scan records the
/// walk's ctime (`fingerprint::Measured::ctime`) and compares it exactly, with no
/// clock involved; a report records none, so a replay is held to the length.
/// Anything that touches the inode moves a ctime -- a chmod, a new hard link --
/// and that errs the safe way: the file is left alone and the run says so.
///
/// A file that could not be read is deliberately NOT reported as a change. It
/// proves nothing about the file's contents, and the removal that follows will
/// fail with the real reason -- so a file that vanished mid-scan reads as a
/// deletion that could not happen rather than as a file quietly left alone.
fn changed_since(
    path: &str,
    measured: (u64, Option<(i64, i64)>),
    current: (Option<u64>, Option<(i64, i64)>),
) -> Option<String> {
    let (measured_size, measured_ctime) = measured;
    let size = current.0?;

    if size != measured_size {
        return Some(format!("{}: {} bytes when scanned, {} bytes now", path, measured_size, size));
    }
    match (measured_ctime, current.1) {
        (Some(then), Some(now)) if then != now => {
            Some(format!("{}: replaced or changed in place since it was scanned (same length)", path))
        }
        _ => None,
    }
}

/// What became of one file the disposal pass reached.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fate {
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

/// Whether a file a DELETE rests on is still at `path` as it was measured:
/// there, a regular file, the same length, and -- where the measurement
/// recorded one -- the same change time (see `changed_since`).
///
/// The window asks the same of a deletion's holders before it trashes it
/// (`Move::stays`); this is the command line's copy of that check.
pub fn still_holds(path: &str, measured_size: u64, measured_ctime: Option<(i64, i64)>) -> bool {
    std::fs::metadata(path).is_ok_and(|m| {
        m.is_file()
            && m.len() == measured_size
            && measured_ctime.is_none_or(|then| then == (m.ctime(), m.ctime_nsec()))
    })
}

/// Log and tally a DELETE left alone because nothing it lost to remains.
pub fn refuse_unheld(path: &str, stats: &RunStats) {
    log::error!(
        target: crate::stats::COUNTED,
        "Not removing {}: every copy it lost to is gone or changed since the scan, so it may be \
         the last one",
        path
    );
    stats.delete_unheld.record(path.to_string());
}

/// Take one file off its path, with the staleness check that has to precede it.
///
/// This is the whole of the destructive step, and the only copy of it: the
/// grouped run reaches it with a size it fingerprinted and the change time its
/// walk read, `--from-report` with a size a report recorded and no change time,
/// and the check is the same function for both. `measured` is the file's length
/// (and ctime, where known) at the moment the decision to remove it was made,
/// whenever that was.
///
/// Errors are logged and tallied here rather than returned, because both callers
/// want the same thing from a failure -- a line in the log, a line in the
/// summary, a non-zero exit -- and none of them wants the loop to stop.
fn dispose_one(
    path: &str,
    measured: (u64, Option<(i64, i64)>),
    disposal: &Disposal,
    stats: &RunStats,
) -> Fate {
    // The last thing before the irreversible step, and the only look here at
    // the disk rather than at the measurement. See `changed_since`: the
    // decision being acted on may be hours or days old.
    let disk = on_disk(path);

    if let Some(detail) = changed_since(path, measured, (disk.size, disk.ctime)) {
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

    // And the folder's entry for it: fsync on a new file makes its bytes
    // durable, not its name, so a power cut after the unlink could leave the
    // bytes on the far disk under no name at all. Not every filesystem lets a
    // folder be opened to be synced (some network ones refuse), and there it
    // is the best that can be had; nor does every one support syncing a folder
    // (EINVAL, ENOTSUP), which is the same answer. Any other failure is real.
    if let Some(parent) = dest.parent() {
        if let Ok(dir) = std::fs::File::open(parent) {
            if let Err(e) = dir.sync_all() {
                let unsupported = matches!(e.raw_os_error(), Some(libc::EINVAL) | Some(libc::ENOTSUP) | Some(libc::EBADF));
                if !unsupported {
                    return Err(e).with_context(|| format!("Failed to flush {} to disk", parent.display()));
                }
            }
        }
    }

    Ok(())
}

/// One ranking of `group`: the file `--priority` keeps, and every file the
/// group wants held for REVIEW beside it.
///
/// It decides who SURVIVES and nothing else. Which of the rest are deleted is
/// the caller's question, because it depends on what each of them was measured
/// against -- see the labeling pass in `output_results`.
pub(crate) fn judge(
    group: &[usize],
    fingerprints: &[VideoFingerprint],
    priority: Priority,
) -> (usize, HashSet<usize>) {
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

    (keep_idx, group_review)
}

/// Each file's fate, from the groups: what the report labels REVIEW and
/// DELETE, and everything else is KEEP.
/// How a run's labels are decided: elected over its groups by one ranking,
/// the ordinary scan, or handed over already decided, which is `--against`
/// (see `against`). A `Priority` converts, so a caller with nothing special to
/// say passes one.
pub enum Ranking {
    By(Priority),
    Pinned(Verdicts),
}

impl From<Priority> for Ranking {
    fn from(priority: Priority) -> Self {
        Ranking::By(priority)
    }
}

pub struct Verdicts {
    /// Held for a person to look at, in every group it is in.
    pub review: HashSet<usize>,
    /// Marked DELETE, ascending. Never one of `review`.
    pub delete: Vec<usize>,
    /// For each file of `delete`, every member of a group it shares that it
    /// was measured against and lost the ranking to, ascending: what its
    /// DELETE rests on. The report needs only that there is one; the desktop
    /// app asks, before it suggests a deletion, whether any of them is still
    /// there and unchanged.
    pub lost_to: HashMap<usize, Vec<usize>>,
}

/// Pass 1 of `output_results`: resolve each file's fate across ALL groups.
/// Its own function so that the desktop app's result (`found`) is labelled by
/// the very code the report is, rather than by a copy of it.
pub fn verdicts(
    final_groups: &[Vec<usize>],
    fingerprints: &[VideoFingerprint],
    matches: &MatchIndex,
    priority: Priority,
) -> Verdicts {
    let mut review_set: HashSet<usize> = HashSet::new();
    let mut delete_candidates: HashSet<usize> = HashSet::new();
    let mut lost_to: HashMap<usize, Vec<usize>> = HashMap::new();

    for group in final_groups {
        let (keep_idx, group_review) = judge(group, fingerprints, priority);

        // The KEEP pick is protected unconditionally, and a standoff does not
        // change that. A pick that leads on the prioritised metric but lost a
        // codec-blind one is no contender, and the standoff says nothing about
        // it -- so the insert is stated once, for every group, and the group's
        // own ranking can never be overturned by a smaller election. DELETE
        // still wins over KEEP globally, so the pick may yet be deleted as a
        // loser in another group.
        let mut protected: HashSet<usize> = group_review.clone();
        protected.insert(keep_idx);

        // A group is a star (see `clustering`): every member was measured
        // against the representative, and nothing promises two other members
        // were ever measured against each other. So a file is marked DELETE
        // only if it lost the ranking to a member it has a measurement with.
        // A member that outranks everything it matched survives beside the KEEP
        // pick: it matched the representative, the representative lost to a
        // file this one never matched, and nothing in the group it could be a
        // duplicate of ranks above it. Where every member matched every other
        // this is exactly the old rule -- everything unprotected lost to the
        // pick and was measured against it.
        //
        // The chain is the case that decides between this and "DELETE only
        // against a survivor". 0-1-2 with 0 the best and 0-2 unmatched is one
        // star around 1; 1 lost to 0 and 2 lost to 1, and both are removed
        // exactly as they are when the same edges arrive as two groups. The
        // survivor-only rule would spare 2 here and not there, so the same
        // library would dispose of different files depending on how the
        // grouping happened to cut it.
        let maxima = GroupMaxima::of(group, fingerprints);
        for &idx in group {
            if protected.contains(&idx) {
                continue;
            }
            // Every member it lost to, not just the first: the window asks
            // whether any of them is still there before it suggests deleting
            // this one (`found::File::lost_to`). Whether there is one is the
            // whole of what the report decides on.
            let winners: Vec<usize> = group
                .iter()
                .copied()
                .filter(|&w| {
                    w != idx
                        && matches.coverage(idx, w).is_some()
                        && outranks(&fingerprints[w], &fingerprints[idx], priority, &maxima)
                })
                .collect();
            if !winners.is_empty() {
                delete_candidates.insert(idx);
                lost_to.entry(idx).or_default().extend(winners);
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

    lost_to.retain(|idx, _| delete_candidates.contains(idx) && !review_set.contains(idx));
    for winners in lost_to.values_mut() {
        winners.sort_unstable();
        winners.dedup();
    }
    Verdicts { review: review_set, delete: delete_indices, lost_to }
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
    ranking: impl Into<Ranking>,
    disposal: Option<&Disposal>,
    assume_yes: bool,
    stats: &RunStats,
) -> Result<Vec<String>> {
    // The run's bar comes down before the listing and the confirmation prompt:
    // a prompt under a redrawing bar is a prompt nobody can read.
    crate::progress::finish();

    // Each file's fate across ALL groups, REVIEW > DELETE > KEEP; see
    // `verdicts`, and `judge` for why every DELETE rests on a direct
    // measurement.
    let Verdicts { review, delete, lost_to } = match ranking.into() {
        Ranking::By(priority) => verdicts(final_groups, fingerprints, matches, priority),
        Ranking::Pinned(decided) => decided,
    };
    let targets: Vec<(usize, Target)> = delete
        .iter()
        .map(|&idx| (idx, Target {
            path: &fingerprints[idx].path,
            size: fingerprints[idx].file_size,
            ctime: fingerprints[idx].ctime(),
        }))
        .collect();

    // Asked here rather than at start-up because only now is there a question
    // worth answering: how many files, and how many bytes. Declining does not
    // abort: it demotes the run to exactly what a run without the flag would
    // have been, so the user who wants to see the list before committing gets
    // it without re-fingerprinting the library. From here on `disposal` is what
    // the run DOES, so every label and figure below describes what the
    // filesystem saw.
    let declined = disposal.is_some_and(|d| {
        let plain: Vec<Target> = targets.iter().map(|(_, t)| *t).collect();
        !confirm::approve(d, &plain, assume_yes, confirm::Decline::ShowsReport)
    });
    let disposal = if declined { None } else { disposal };

    // What it lost to has to still be there as it was measured, or have gone
    // in this pass on the same terms: the scan may be hours old, and the prompt
    // waits as long as it is left. Chains still work, because a winner removed
    // in this pass was removed on the same condition.
    let mut outcome: HashMap<usize, &'static str> = HashMap::new();
    let (tally, deleted_paths) = match disposal {
        None => (Tally::default(), Vec::new()),
        Some(d) => dispose_all(
            &targets,
            |idx, removed| {
                lost_to.get(&idx).is_none_or(|winners| {
                    winners.is_empty()
                        || winners.iter().any(|&w| {
                            removed.contains(&w)
                                || still_holds(&fingerprints[w].path, fingerprints[w].file_size, fingerprints[w].ctime())
                        })
                })
            },
            d,
            stats,
            |idx, label| {
                outcome.insert(idx, label);
            },
        ),
    };

    // Labelled by the file's GLOBAL fate. A dry run recommends; an acting run
    // says what happened, and SKIPPED is a target an interrupt or a vanished
    // winner left alone.
    let marked: HashSet<usize> = delete.iter().copied().collect();
    let rests_on: HashMap<usize, usize> = lost_to
        .iter()
        .filter_map(|(&idx, winners)| rests_on(winners, &marked).map(|w| (idx, w)))
        .collect();
    let action = |idx: usize| -> &'static str {
        match (review.contains(&idx), marked.contains(&idx), disposal.is_some()) {
            (true, _, _) => "REVIEW",
            (_, true, false) => "DELETE",
            (_, true, true) => outcome.get(&idx).copied().unwrap_or("SKIPPED"),
            (_, false, false) => "KEEP",
            (_, false, true) => "KEPT",
        }
    };

    let summary = Summary {
        groups: final_groups.len(),
        // A group of one implicates nothing: a scan never makes one, and an
        // `--against` file the library did not match is not "matched".
        matched_files: final_groups.iter().filter(|g| g.len() > 1).flatten().collect::<HashSet<_>>().len(),
        elapsed_secs: total_elapsed_secs,
        marked: delete.len(),
        reclaimable_bytes: targets.iter().map(|(_, t)| t.size).sum(),
        disposal,
        declined,
        tally: &tally,
    };

    // Where the groups go. Everything the listing and the summary amount to
    // leaves through `info!`, so under `-q` nobody is at the other end of them
    // and they are not built; nor under the desktop app, whose results page is
    // this listing. With no listing and no report there is nothing to format,
    // and the loop below walks nothing -- `-q` with no `--output` is how a
    // script asks for only the disposal and the exit code.
    let console = console_for(report_target);
    let speaking = log::log_enabled!(log::Level::Info);
    let mut listing = (console.listing && speaking && !crate::utils::is_for_window()).then(Listing::open);
    let mut report = report_target.map(|t| open_report(t, &summary));

    for (i, group) in final_groups.iter().enumerate() {
        let mut bodies: Vec<&mut dyn Body> = Vec::with_capacity(2);
        if let Some(l) = &mut listing {
            bodies.push(l);
        }
        if let Some(r) = &mut report {
            bodies.push(r.as_mut());
        }
        if bodies.is_empty() {
            break;
        }

        let name = format!("group_{}", i + 1);
        let files: Vec<Entry> = group
            .iter()
            .map(|&idx| {
                let lost = rests_on.get(&idx).map(|&w| fingerprints[w].path.as_str());
                Entry::of(idx, group, fingerprints, matches, action(idx), lost)
            })
            .collect();
        for body in bodies {
            body.group(&name, &files);
        }
    }

    if console.summary && speaking {
        info!("{}", summary);
    }

    // A report that could not be saved is a problem (exit 2, a line in the
    // summary) and not an error this function returns: the disposal pass has
    // already run, and returning would throw away `deleted_paths`, leaving the
    // cache claiming fingerprints for files this run has just removed.
    // `report_target_for` proved the destination writable before any work
    // started, so what fails here is a full disk or permissions that changed.
    if let (Some(report), Some(target)) = (report, report_target) {
        match report.finish(&summary) {
            // Nothing to announce for stdout: saying so on stderr would be
            // describing the pipe to whoever is reading the other end of it.
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

    // Advice about the next command, which no format carries, so it comes
    // last, after the report it refers to. Not for someone who declined: they
    // saw the question and said no.
    if disposal.is_none() && !declined && !delete.is_empty() && !crate::utils::is_for_window() {
        info!(
            "\nRun with --delete to move the file(s) marked DELETE to the trash or with --move-to <DIR> to relocate them instead."
        );
    }

    Ok(deleted_paths)
}

/// The one file a DELETE's report row names as what it rests on (`lost_to`),
/// out of every file it lost the ranking to.
///
/// A report row has room for one, and `--from-report` acts on the row only
/// while that file is still there, so the choice is the one most likely to
/// stay: a winner nothing marks DELETE, the lowest-numbered (the comparison
/// reads in path order, so a run names the same file every time). Only where
/// every winner is itself marked -- a chain, see `verdicts` -- is it one of
/// those, and a replay then holds it to having gone earlier in the same pass,
/// exactly as the scan holds its own disposal (`output_results`).
pub(crate) fn rests_on(winners: &[usize], marked: &HashSet<usize>) -> Option<usize> {
    let mut sorted = winners.to_vec();
    sorted.sort_unstable();
    sorted.iter().copied().find(|w| !marked.contains(w)).or_else(|| sorted.first().copied())
}

/// What a disposal pass did: the counts and bytes both of its callers report.
#[derive(Default, Debug)]
pub struct Tally {
    /// Every target taken off its path, aliases included.
    pub removed: usize,
    /// Only the bytes that went away with them. See `OnDisk`.
    pub removed_bytes: u64,
    /// Of `removed`, the second names for data still on disk (UNLINKED).
    pub aliased: usize,
    pub aliased_bytes: u64,
    pub failed: usize,
    pub changed: usize,
    /// Left alone because every copy it lost to is gone or changed.
    pub unheld: usize,
}

impl Tally {
    fn settled(&self) -> usize {
        self.removed + self.failed + self.changed + self.unheld
    }

    /// What the pass did, in the words of the mode that did it, followed by
    /// what the byte figure leaves out and what was not done, each omitted when
    /// there is nothing to say.
    pub fn lines(&self, disposal: &Disposal) -> String {
        let size = format_size(self.removed_bytes);
        let mut out = match disposal {
            Disposal::Permanent => format!("Permanently deleted {} file(s), {} freed.", self.removed, size),
            Disposal::Trash => format!("Moved {} file(s) to trash ({} total).", self.removed, size),
            // Deliberately not "freed": if the destination is on the same
            // filesystem the bytes have not gone anywhere.
            Disposal::MoveTo(dir) => {
                format!("Moved {} file(s) ({} total) under {}.", self.removed, size, dir.display())
            }
        };

        // Removing one of several names for a file is a good outcome -- not a
        // problem, not a skip -- but it leaves the byte total above
        // unaccountable without this. It carries the bytes because a
        // `--move-to` run of nothing else reads "Moved 1 file(s) (0B total)".
        if self.aliased > 0 {
            out.push_str(&format!(
                "\n{} file(s) were another name for data that is still on disk (a hard link or a \
                 symlink), so their {} is not counted above; they are marked UNLINKED.",
                self.aliased,
                format_size(self.aliased_bytes)
            ));
        }
        if self.unheld > 0 {
            out.push_str(&format!(
                "\n{} file(s) were left alone because every copy they lost to is gone or changed.",
                self.unheld
            ));
        }
        if self.failed > 0 {
            out.push_str(&format!("\n{} file(s) could not be removed (see errors above).", self.failed));
        }
        if self.changed > 0 {
            out.push_str(&format!(
                "\n{} file(s) changed on disk after they were scanned and were left alone \
                 (re-run to judge them as they now stand).",
                self.changed
            ));
        }
        out
    }
}

/// The disposal pass: each target once, in order, each told to `each` with
/// the label its row will carry. Shared by the grouped run and
/// `--from-report`, so the two cannot count or describe it differently.
///
/// `held(key, removed)` asks whether a target may still go: whether something
/// it lost to remains, given the keys this pass has already removed. A target
/// it refuses is SKIPPED and counted as unheld. Returns the tally and the paths
/// that are genuinely no longer where they were -- every failure, every file
/// that changed under us and everything an interrupt left untouched is missing
/// from it.
pub fn dispose_all<K: Copy + Eq + std::hash::Hash>(
    targets: &[(K, Target)],
    held: impl Fn(K, &HashSet<K>) -> bool,
    disposal: &Disposal,
    stats: &RunStats,
    mut each: impl FnMut(K, &'static str),
) -> (Tally, Vec<String>) {
    let mut tally = Tally::default();
    let mut removed: HashSet<K> = HashSet::new();
    let mut deleted_paths = Vec::new();

    for &(key, target) in targets {
        if shutdown_requested() {
            info!(
                "Interrupted: stopped after {} file(s); {} left untouched.",
                tally.removed,
                targets.len() - tally.settled()
            );
            break;
        }

        if !held(key, &removed) {
            refuse_unheld(target.path, stats);
            tally.unheld += 1;
            each(key, "SKIPPED");
            continue;
        }

        let label = match dispose_one(target.path, (target.size, target.ctime), disposal, stats) {
            Fate::Done { aliased } => {
                tally.removed += 1;
                if aliased {
                    tally.aliased += 1;
                    tally.aliased_bytes += target.size;
                } else {
                    tally.removed_bytes += target.size;
                }
                deleted_paths.push(target.path.to_string());
                removed.insert(key);
                disposal.done_label(aliased)
            }
            Fate::Changed => {
                tally.changed += 1;
                "CHANGED"
            }
            Fate::Failed => {
                tally.failed += 1;
                "FAILED"
            }
        };
        each(key, label);
    }

    (tally, deleted_paths)
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
    use super::render::CsvBody;
    use super::stream::{replaceable_by_rename, scratch_beside, write_file_atomically, Scratch};
    use std::os::unix::fs::PermissionsExt;
    use crate::compare::Match;
    use crate::utils::Priority;
    use tempfile::NamedTempFile;
    use std::fs;

    const CSV_HEADER: &str = "group;action;full_path;lost_to;length;length_seconds;resolution;width;\
height;framerate_fps;codec;size;size_bytes;bitrate;bitrate_bps;quality;\
quality_bits_per_frame;matched_with;identical;samples;matched_seconds;matched_from;matched_to;\
matched_from_seconds;matched_to_seconds";

    /// What a DELETE row names as its rest: a winner that stays, before one
    /// that is itself marked, and the lowest-numbered of either kind.
    #[test]
    fn test_a_delete_rests_on_a_winner_that_stays_when_there_is_one() {
        let marked: HashSet<usize> = [1, 2].into_iter().collect();
        assert_eq!(rests_on(&[2, 1, 5, 3], &marked), Some(3));
        assert_eq!(rests_on(&[2, 1], &marked), Some(1), "a chain names a marked winner");
        assert_eq!(rests_on(&[], &marked), None);
    }

    fn mock_fp() -> VideoFingerprint {
        VideoFingerprint {
            path: "/fake/path/vid.mp4".to_string(),
            valid_hashes: vec![], valid_t_start: vec![], valid_t_end: vec![],
            total_ms: 100_000, width: 1920, height: 1080, duration: 60.0, file_size: 1048576,
            codec: "h264".to_string(), frame_rate: 30.0,
            mirrored_hashes: Vec::new(),
            detail_hashes: Vec::new(),
            mirrored_detail_hashes: Vec::new(),
            reframed: Vec::new(),
            picture: 0.0,
            measured: None,
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
            "group_1;KEEP;/fake/path/vid.mp4;;00:01:00;60.0;1920x1080;1920;1080;30.0;h264;\
1.0MB;1048576;140kbps;139810;4.7kb/f;4660;;;0;;;;;"
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
                "group_1;KEEP;/fake/a.mp4;;00:01:40;100.0;1920x1080;1920;1080;30.0;h264;\
12.5MB;13107200;1.0Mbps;1048576;35.0kb/f;34952;/fake/b.mp4;false;0;80.0;"
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
    fn test_the_csv_columns_are_the_json_keys_in_the_same_order() {
        // One schema, read two ways: the CSV's columns after `group` are a JSON
        // file object's keys before `matches`, name for name and in order, and
        // each JSON link carries the row's link block under the same names.
        let fps = vec![mock_fp_at("/fake/a.mp4", 100.0), mock_fp_at("/fake/b.mp4", 100.0)];
        let groups = vec![vec![0, 1]];
        let matches = MatchIndex::new(vec![Match::new(0, 1, 0.80, 0.80)]);
        let csv_path = report_to("csv");
        let json_path = report_to("json");
        for path in [&csv_path, &json_path] {
            output_results(
                &groups, &fps, &matches, Some(&to_file(path)), 0, Priority::Length, None,
                true, &RunStats::default(),
            ).unwrap();
        }

        let csv = fs::read_to_string(&csv_path).unwrap();
        let columns: Vec<&str> = csv.lines().next().unwrap().split(';').collect();
        assert_eq!(columns.join(";"), CSV_HEADER);

        let report = read_json(&json_path);
        let file = report["results"][0]["files"][0].as_object().unwrap();
        let keys: Vec<&str> = file.keys().map(String::as_str).collect();
        assert_eq!(keys.last(), Some(&"matches"));
        assert_eq!(columns[1..], keys[..keys.len() - 1]);

        let link: Vec<&str> =
            file["matches"][0].as_object().unwrap().keys().map(String::as_str).collect();
        let block: Vec<&str> =
            columns.iter().copied().filter(|c| c.starts_with("matched_") && *c != "matched_with").collect();
        assert_eq!(link[..2], ["full_path", "identical"]);
        assert_eq!(link[2..], block[..]);

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
            let matched: f64 = cells[20].parse().unwrap();
            let from: f64 = cells[23].parse().unwrap();
            let to: f64 = cells[24].parse().unwrap();
            assert!(
                matched <= to - from + 1e-9,
                "matched footage must fit inside its own envelope: {}",
                line
            );
        }

        // And the two rows now differ, which is the signal the minimum hid.
        assert!(contents.contains(";/fake/_19.mp4;false;10;1.88;"), "_18 row: {}", contents);
        assert!(contents.contains(";/fake/_18.mp4;false;1;8.84;"), "_19 row: {}", contents);

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
        assert!(contents.contains(";/fake/clip.mp4;false;0;60.0;"), "host row: {}", contents);
        assert!(contents.contains(";/fake/host.mp4;false;0;60.0;"), "clip row: {}", contents);

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
                && contents.contains("/fake/clip.mp4;false;0;60.0;00:14:00;00:15:00;840.0;900.0"),
            "the host row should point into the host's own timeline: {}",
            contents
        );
        assert!(
            contents.contains("group_1;DELETE;/fake/clip.mp4;")
                && contents.contains("/fake/host.mp4;false;0;60.0;00:00:00;00:01:00;0.0;60.0"),
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
        // 0-1 and 1-2 matched; 0 and 2 did not -- a star around 1, which is
        // what clustering hands over for a chain. The reporting pass reads the
        // links it has rather than assuming the group is complete. 0's list must hold one entry, not
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
    fn test_a_run_with_nowhere_to_say_anything_still_does_everything_else() {
        // `-q` with no `--output` is a run whose per-file rows have no reader:
        // no listing, no report body. The reporting loop is skipped outright
        // for it, so what this pins is that nothing else went with it -- the
        // disposal pass runs before that loop and its answer is what the caller
        // needs to drop the cache entries of files that are no longer there.
        //
        // The condition it turns on is the log level, which a unit test has no
        // way to set without leaking into every test running beside it -- so it
        // is read the other way around: no logger is installed here, `info!` is
        // going nowhere, and this run is already the one being described.
        assert!(
            !log::log_enabled!(log::Level::Info),
            "a test binary installs no logger, which is what makes this run the quiet one"
        );

        let dir = tempfile::tempdir().unwrap();
        let keep_path = at(&dir, "keep.mkv");
        let del_path = at(&dir, "duplicate.mkv");

        let fps = vec![mock_fp_at(&keep_path, 60.0), mock_fp_at(&del_path, 10.0)];
        materialize_all(&fps);

        let stats = RunStats::default();

        let deleted = output_results(
            &[vec![0, 1]], &fps, &all_compared(fps.len()), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();

        assert!(Path::new(&keep_path).exists(), "the KEEP pick is untouched");
        assert!(!Path::new(&del_path).exists(), "the duplicate is still disposed of");
        assert_eq!(deleted, vec![del_path], "and the caller is still told which one");
        assert!(!stats.had_problems());
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
    fn test_a_chain_inside_one_star_collapses_exactly_as_across_two_groups() {
        // The same chain as above, but grouped the way `clustering` groups it:
        // one star around the bridge. 1 lost to 0 and 2 lost to 1, each against
        // a file it was measured with, so both go -- the same files the two
        // group split removes. How the grouping cut the edges must not decide
        // what is deleted.
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

        let groups = vec![vec![1, 0, 2]];

        let mut deleted = output_results(
            &groups, &fps, &chain_of_three(), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();
        deleted.sort();

        let mut expected = vec![p1, p2];
        expected.sort();
        assert_eq!(deleted, expected);
        assert!(Path::new(&p0).exists());
    }

    #[test]
    fn test_a_deletion_whose_every_winner_is_gone_is_left_alone() {
        // The chain above, with the head removed between the scan and the
        // disposal (the prompt waits as long as it is left). 1 lost only to 0,
        // which is gone, so 1 would be the last copy and stays. 2 lost to 1,
        // which is still there, so 2 goes as before.
        let dir = tempfile::tempdir().unwrap();
        let p0 = at(&dir, "best.mkv");
        let p1 = at(&dir, "bridge.mkv");
        let p2 = at(&dir, "tail.mkv");
        let fps = vec![mock_fp_at(&p0, 100.0), mock_fp_at(&p1, 90.0), mock_fp_at(&p2, 10.0)];
        materialize_all(&fps);
        std::fs::remove_file(&p0).unwrap();

        let stats = RunStats::default();
        let deleted = output_results(
            &[vec![1, 0, 2]], &fps, &chain_of_three(), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();

        assert_eq!(deleted, vec![p2]);
        assert!(Path::new(&p1).exists(), "the last copy is kept");
        assert_eq!(stats.delete_unheld.count(), 1);
    }

    /// The walk's change time, as `scan::fingerprint_all` sets it.
    fn walked(mut fp: VideoFingerprint) -> VideoFingerprint {
        let md = fs::metadata(&fp.path).unwrap();
        fp.measured_mut().ctime = Some((md.ctime(), md.ctime_nsec()));
        fp
    }

    /// Put another file of the same length at `path`, as `rsync -a` would: a
    /// new inode renamed over the old one, the old modification time put back.
    /// Nothing about its length can tell.
    fn replace_keeping_length(path: &str) {
        let len = fs::metadata(path).unwrap().len();
        let mtime = fs::metadata(path).unwrap().modified().unwrap();
        // Past the filesystem's timestamp granularity, which can be coarser
        // than the nanoseconds it reports.
        std::thread::sleep(std::time::Duration::from_millis(30));
        let staged = format!("{path}.staged");
        fs::write(&staged, vec![0x5a; len as usize]).unwrap();
        fs::File::options().write(true).open(&staged).unwrap().set_modified(mtime).unwrap();
        fs::rename(&staged, path).unwrap();
        assert_eq!(fs::metadata(path).unwrap().len(), len);
    }

    /// The window refuses a file replaced by one of the same length between
    /// the scan and the disposal (`last::changed_since`, by ctime), and the
    /// command line compared lengths alone: the prompt waits as long as it is
    /// left, and `rsync -a` puts back the length and the mtime. Both the
    /// target and the file it lost to are held to their walk's ctime now.
    #[test]
    fn test_a_file_replaced_by_one_of_the_same_length_is_not_the_file_that_was_judged() {
        // The target replaced: left alone, CHANGED.
        let dir = tempfile::tempdir().unwrap();
        let keep = at(&dir, "keep.mkv");
        let target = at(&dir, "target.mkv");
        let fps = vec![mock_fp_at(&keep, 60.0), mock_fp_at(&target, 10.0)];
        materialize_all(&fps);
        let fps: Vec<VideoFingerprint> = fps.into_iter().map(walked).collect();
        replace_keeping_length(&target);

        let stats = RunStats::default();
        let deleted = output_results(
            &[vec![0, 1]], &fps, &all_compared(2), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();
        assert!(deleted.is_empty());
        assert!(Path::new(&target).exists(), "a replaced file is not the one that was judged");
        assert_eq!(stats.delete_stale.count(), 1);

        // The file it lost to replaced: it holds nothing, so the target stays.
        let dir = tempfile::tempdir().unwrap();
        let keep = at(&dir, "keep.mkv");
        let target = at(&dir, "target.mkv");
        let fps = vec![mock_fp_at(&keep, 60.0), mock_fp_at(&target, 10.0)];
        materialize_all(&fps);
        let fps: Vec<VideoFingerprint> = fps.into_iter().map(walked).collect();
        replace_keeping_length(&keep);

        let stats = RunStats::default();
        let deleted = output_results(
            &[vec![0, 1]], &fps, &all_compared(2), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();
        assert!(deleted.is_empty());
        assert!(Path::new(&target).exists(), "what it lost to is not there as measured");
        assert_eq!(stats.delete_unheld.count(), 1);

        // Untouched since the walk: removed as before.
        let dir = tempfile::tempdir().unwrap();
        let keep = at(&dir, "keep.mkv");
        let target = at(&dir, "target.mkv");
        let fps = vec![mock_fp_at(&keep, 60.0), mock_fp_at(&target, 10.0)];
        materialize_all(&fps);
        let fps: Vec<VideoFingerprint> = fps.into_iter().map(walked).collect();
        let stats = RunStats::default();
        let deleted = output_results(
            &[vec![0, 1]], &fps, &all_compared(2), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();
        assert_eq!(deleted, vec![target]);
        assert!(!stats.had_problems());
    }

    #[test]
    fn test_a_winner_that_changed_size_holds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let p0 = at(&dir, "best.mkv");
        let p1 = at(&dir, "bridge.mkv");
        let p2 = at(&dir, "tail.mkv");
        let fps = vec![mock_fp_at(&p0, 100.0), mock_fp_at(&p1, 90.0), mock_fp_at(&p2, 10.0)];
        materialize_all(&fps);
        std::fs::write(&p0, b"truncated").unwrap();

        let stats = RunStats::default();
        output_results(
            &[vec![1, 0, 2]], &fps, &chain_of_three(), None, 0, Priority::Length,
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();
        assert!(Path::new(&p1).exists());
        assert_eq!(stats.delete_unheld.count(), 1);
    }

    #[test]
    fn test_a_star_member_that_outranks_everything_it_matched_survives() {
        // A star around 1, the worst copy: 0 and 2 each matched it and did not
        // match each other. 0 is the group's pick and 1 lost to it. 2 lost the
        // group's ranking to 0 as well -- but never met 0, and the only file it
        // was measured against is the one being removed. Nothing it is a
        // duplicate of ranks above it, so it is kept beside 0 rather than
        // deleted on the strength of a comparison nobody made.
        let dir = tempfile::tempdir().unwrap();
        let p0 = at(&dir, "best.mkv");
        let p1 = at(&dir, "hub.mkv");
        let p2 = at(&dir, "second.mkv");

        let fps = vec![
            mock_fp_at(&p0, 100.0),
            mock_fp_at(&p1, 10.0),
            mock_fp_at(&p2, 90.0),
        ];
        materialize_all(&fps);

        let matches = MatchIndex::new(vec![
            Match::new(1, 0, 1.0, 1.0),
            Match::new(1, 2, 1.0, 1.0),
        ]);
        let groups = vec![vec![1, 0, 2]];
        let path_str = report_to("json");

        let deleted = output_results(
            &groups, &fps, &matches, Some(&to_file(&path_str)), 0, Priority::Length,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert_eq!(deleted, vec![p1.clone()], "only the hub lost to a file it matched");
        assert!(Path::new(&p0).exists());
        assert!(Path::new(&p2).exists());

        let report = read_json(&path_str);
        let action_of = |path: &str| -> String {
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
        assert_eq!(action_of(&p0), "KEPT");
        assert_eq!(action_of(&p2), "KEPT");
        assert_eq!(report["summary"]["files_marked_delete"], 1);

        let _ = fs::remove_file(path_str);
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
        // Files 0 and 1 are equally good and were measured against each other,
        // so they are ranked against each other and only one is kept.
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
    fn test_a_report_replaces_the_last_one_in_one_step_and_leaves_nothing_beside_it() {
        // The report is the last thing `output_results` does, after the
        // disposal pass, and `--from-report` hands exactly this file back. A
        // truncated one is a deletion plan missing its last rows, so the file
        // at this name has to be one whole report or the other.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dupes.csv");

        write_file_atomically(&path, b"the first run\n").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"the first run\n");

        write_file_atomically(&path, b"the second run, longer\n").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"the second run, longer\n");

        // No scratch copy survives a write that worked.
        let left: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(left, vec!["dupes.csv".to_string()]);

        // And a write that does NOT reach its rename takes the scratch copy
        // with it and leaves the report it was replacing exactly as it was.
        let scratch_path = scratch_beside(&path).unwrap();
        std::fs::write(&scratch_path, b"half a report").unwrap();
        drop(Scratch { path: Some(scratch_path.clone()) });
        assert!(!scratch_path.exists());
        assert_eq!(std::fs::read(&path).unwrap(), b"the second run, longer\n");

        // A rename carries the scratch file's mode, so the report a user had
        // narrowed would quietly widen back to the default without this.
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        write_file_atomically(&path, b"the third run\n").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "the report's own permissions have to survive the replace");
    }

    #[test]
    fn test_the_streamed_json_is_laid_out_exactly_as_one_rendered_whole() {
        // The document is written a group at a time, so nothing anywhere holds
        // it whole and nothing can diff the streamed layout against a rendered
        // one. What CAN be compared is the file against itself: parse it back
        // -- `preserve_order` hands the keys back in the order they went in --
        // and render that the way `to_string_pretty` would have. Every place
        // the streaming has to reproduce by hand is in that comparison: the two
        // spaces per level, the depth each piece is written at, the comma
        // between groups, and the brackets around them.
        //
        // Three groups, because one exercises no separator and two exercise
        // only the first; and a three-member group, because that is where a
        // file carries more than one entry in its `matches` array.
        let fps = vec![
            mock_fp_at("/fake/a.mp4", 100.0),
            mock_fp_at("/fake/b.mp4", 90.0),
            mock_fp_at("/fake/c.mp4", 80.0),
            mock_fp_at("/fake/d.mp4", 70.0),
            mock_fp_at("/fake/e.mp4", 60.0),
        ];
        let groups = vec![vec![0, 1, 2], vec![1, 3], vec![3, 4]];
        let path_str = report_to("json");

        output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&to_file(&path_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let text = fs::read_to_string(&path_str).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            text,
            serde_json::to_string_pretty(&parsed).unwrap(),
            "the streamed document has to be laid out exactly as a rendered one"
        );
        assert_eq!(parsed["results"].as_array().unwrap().len(), 3);
        assert_eq!(parsed["results"][0]["files"][0]["matches"].as_array().unwrap().len(), 2);

        // And the empty document, which is the one shape the streaming cannot
        // reach by writing groups: `results` is written closed rather than
        // opened and closed. It is what `conclude_without_comparing` produces,
        // so it is a report a real run writes rather than a degenerate case.
        let empty_str = report_to("json");
        output_results(
            &[], &fps, &all_compared(fps.len()), Some(&to_file(&empty_str)), 0, Priority::Length, None,
            true, &RunStats::default(),
        ).unwrap();

        let empty = fs::read_to_string(&empty_str).unwrap();
        let parsed_empty: serde_json::Value = serde_json::from_str(&empty).unwrap();
        assert_eq!(empty, serde_json::to_string_pretty(&parsed_empty).unwrap());
        assert!(empty.contains("\"results\": []"), "{}", empty);

        let _ = fs::remove_file(path_str);
        let _ = fs::remove_file(empty_str);
    }

    #[test]
    fn test_a_csv_record_that_will_not_write_is_reported_rather_than_returned() {
        // The one guarantee `ReportStream` could not make on its own. It makes
        // the SINK infallible, and a `csv::Writer` fails for a reason the sink
        // never sees -- a record that is not the width of the header. That
        // failure used to leave `output_results` through a `?`, taking
        // `deleted_paths` with it, so the caller never dropped the cache
        // entries of the files the disposal pass had just removed.
        //
        // It is unreachable from `output_results` today, because every record
        // there is the same fixed-width array. That is the point: `record`
        // hands nothing back, so the pass cannot start propagating one by
        // growing a column. Driven here directly, which is the only way to make
        // the writer fail at all.
        let dir = tempfile::tempdir().unwrap();
        let path = at(&dir, "dupes.csv");
        let target = to_file(&path);

        let mut body = CsvBody::open(&target);
        body.record(["group", "action", "full_path"]);
        // Two fields where the header set three. `csv::Writer` refuses it, and
        // the compiler will not let this line be written with a `?` on it.
        body.record(["group_1", "DELETE"]);
        // And the pass carries on rather than unwinding -- everything after the
        // first failure is skipped, exactly as the sink skips.
        body.record(["group_1", "KEEP", "/fake/a.mp4"]);

        let failure = body.close().expect_err("a refused record is a failed report");
        assert!(
            format!("{:#}", failure).contains("Failed to write the CSV report"),
            "answered for in the same words a failed whole-document write uses: {:#}",
            failure
        );
        assert!(
            !Path::new(&path).exists(),
            "and a report that could not be written is not renamed into place"
        );
    }

    #[test]
    fn test_a_report_that_cannot_be_written_still_hands_back_what_was_disposed_of() {
        // The other half of the same rule, and the half with the consequence:
        // the report is the LAST thing `output_results` does, so a failure
        // there must not take the disposal pass's answer out with it. Without
        // those paths `main` never calls `Cache::forget`, and the cache goes on
        // offering fingerprints for files that are gone.
        let dir = tempfile::tempdir().unwrap();
        let fps = vec![
            mock_fp_at(&at(&dir, "a.mp4"), 100.0),
            mock_fp_at(&at(&dir, "b.mp4"), 10.0),
        ];
        materialize_all(&fps);

        // A destination that will not take a scratch file and will not take a
        // write either: the report is a directory.
        let blocked = at(&dir, "dupes.csv");
        fs::create_dir(&blocked).unwrap();

        let stats = RunStats::default();
        let disposed = output_results(
            &[vec![0, 1]], &fps, &all_compared(fps.len()), Some(&to_file(&blocked)), 0,
            Priority::Length, Some(&Disposal::Permanent), true, &stats,
        )
        .expect("a report that cannot be saved is not an error this function returns");

        assert_eq!(disposed, vec![fps[1].path.clone()], "the loser was removed and named");
        assert!(!Path::new(&fps[1].path).exists(), "and it really went");
        assert_eq!(stats.report_write_failed.count(), 1, "the failure is tallied instead");
        assert!(stats.had_problems(), "so the run still exits 2");
    }

    #[test]
    fn test_a_streamed_report_replaces_the_last_one_and_leaves_no_litter() {
        // No format reaches `write_file_atomically` any more -- each holds a
        // scratch copy open across the whole reporting pass instead of handing
        // over a finished document -- so every promise that function makes has
        // to be made again by the sink that replaced it. The report is replaced
        // in one step, its permissions survive, nothing is left beside it, and
        // the destinations a rename must not replace still get the write they
        // had. Asked of all three, because one sink now serves all three.
        let fps = vec![
            mock_fp_at("/fake/a.mp4", 100.0),
            mock_fp_at("/fake/b.mp4", 10.0),
        ];
        let groups = vec![vec![0, 1]];

        for extension in ["csv", "txt", "json"] {
            let dir = tempfile::tempdir().unwrap();
            let run_into = |path: &str| {
                output_results(
                    &groups, &fps, &all_compared(fps.len()), Some(&to_file(path)), 0,
                    Priority::Length, None, true, &RunStats::default(),
                ).unwrap();
            };

            let name = format!("dupes.{}", extension);
            let path = at(&dir, &name);
            fs::write(&path, "an earlier, longer report that must not survive in part").unwrap();
            run_into(&path);

            let text = fs::read_to_string(&path).unwrap();
            assert!(text.contains("/fake/a.mp4"), "{}: {}", extension, text);
            assert!(!text.contains("earlier"), "{}: replaced, not appended to", extension);

            let left: Vec<String> = fs::read_dir(dir.path())
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
                .collect();
            assert_eq!(left, vec![name.clone()], "{}: no scratch copy left behind", extension);

            // A report the user had narrowed keeps its mode across the replace.
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
            run_into(&path);
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600,
                "{}: the report's own permissions have to survive the replace",
                extension
            );

            // A symlinked report is one kept somewhere else under a local name:
            // written through, so the link survives and the bytes land at its
            // target. `Scratch::beside` refuses it and the sink falls back to
            // holding the document, which is the one case that still does.
            let elsewhere = dir.path().join(format!("elsewhere.{}", extension));
            fs::write(&elsewhere, "").unwrap();
            let link = at(&dir, &format!("linked.{}", extension));
            std::os::unix::fs::symlink(&elsewhere, Path::new(&link)).unwrap();

            run_into(&link);
            assert!(
                fs::symlink_metadata(&link).unwrap().file_type().is_symlink(),
                "{}: the link itself must survive",
                extension
            );
            assert_eq!(
                fs::read_to_string(&elsewhere).unwrap(),
                text,
                "{}: and the report has to land at what it points at",
                extension
            );
        }
    }

    #[test]
    fn test_a_destination_that_is_not_a_plain_file_is_written_through_rather_than_replaced() {
        // `ensure_writable` accepts a FIFO unexamined (`mkfifo p; vid-fp lib -o
        // p & cat p` is a shape people use), and a symlinked report is one kept
        // somewhere else under a local name. Replacing either by rename would
        // destroy the thing that was asked for -- the reader waits for ever,
        // the link is gone -- so both are written in place.
        let dir = tempfile::tempdir().unwrap();

        let fifo = dir.path().join("pipe");
        let c_path = std::ffi::CString::new(fifo.to_string_lossy().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        assert!(!replaceable_by_rename(&fifo));

        let elsewhere = dir.path().join("elsewhere.csv");
        std::fs::write(&elsewhere, b"").unwrap();
        let link = dir.path().join("report.csv");
        std::os::unix::fs::symlink(&elsewhere, &link).unwrap();
        assert!(!replaceable_by_rename(&link));

        write_file_atomically(&link, b"through the link\n").unwrap();
        assert!(std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
        assert_eq!(std::fs::read(&elsewhere).unwrap(), b"through the link\n");

        // The ordinary two: a plain file, and a name with nothing at it.
        assert!(replaceable_by_rename(&elsewhere));
        assert!(replaceable_by_rename(&dir.path().join("not there yet.csv")));
    }

    #[test]
    fn test_a_read_only_folder_still_takes_a_report_over_the_one_already_in_it() {
        // `ensure_writable` allows this on purpose -- writing over an existing
        // file needs no permission on the directory holding it -- and a rename
        // cannot create the scratch copy there. Falling back to the write this
        // replaced is no worse than what was there before.
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("locked");
        std::fs::create_dir(&folder).unwrap();
        let path = folder.join("dupes.csv");
        std::fs::write(&path, b"an earlier run\n").unwrap();
        std::fs::set_permissions(&folder, std::fs::Permissions::from_mode(0o500)).unwrap();

        let wrote = write_file_atomically(&path, b"this run\n");

        // Restore before any assertion, so a failure does not leave the temp
        // directory undeletable.
        std::fs::set_permissions(&folder, std::fs::Permissions::from_mode(0o700)).unwrap();
        wrote.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"this run\n");
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

    /// Which copy a group loses must not depend on the order its files arrive
    /// in, and it did whenever one of them could not be measured on a metric.
    /// `find_best` hands `max_by` a relation that was not transitive then, so
    /// the fold landed wherever the input order took it -- see
    /// `utils::tests::test_a_metric_nobody_could_measure_does_not_make_the_ranking_cyclic`
    /// for the relation itself. This is what it cost with `--delete` armed.
    ///
    /// Three copies of one 10 second clip. `a` beats `b` on quality, which is
    /// where their comparison stopped. `c` reported no frame rate, so quality
    /// said nothing about it either way and both of ITS comparisons fell
    /// through to size, where it is the biggest file of its own codec and beat
    /// them both: a > b > c > a. Four of the six orderings then deleted `b` and
    /// two deleted `a` -- a different video destroyed, on nothing the user
    /// could see, since renaming a file is enough to reorder a group.
    #[test]
    fn test_which_copy_is_deleted_does_not_depend_on_the_order_the_group_arrives_in() {
        let dir = tempfile::tempdir().unwrap();

        // Same footage, same length, same frame: only the encode differs.
        let clip = |name: &str, size: u64, codec: &str, frame_rate: f64| {
            let mut fp = mock_fp_at(&at(&dir, name), 10.0);
            fp.width = 640;
            fp.height = 480;
            fp.file_size = size;
            fp.codec = codec.to_string();
            fp.frame_rate = frame_rate;
            fp
        };

        let fps = vec![
            clip("a.mkv", 42_807, "av1", 33.0),
            clip("b.mkv", 43_451, "av1", 34.0),
            clip("c.mkv", 87_955, "hevc", 0.0), // the container never said
        ];

        for order in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
            materialize_all(&fps);

            let deleted = output_results(
                &[order.to_vec()], &fps, &all_compared(fps.len()), None, 0, Priority::Length,
                Some(&Disposal::Permanent), true, &RunStats::default(),
            ).unwrap();

            assert_eq!(
                deleted,
                vec![fps[1].path.clone()],
                "group order {:?} lost a different copy",
                order
            );
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