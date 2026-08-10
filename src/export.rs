use anyhow::{anyhow, Context, Result};
use log::info;
use crate::compare::MatchIndex;
use crate::confirm::{self, Target};
use crate::fingerprint::VideoFingerprint;
use crate::stats::RunStats;
use crate::utils::{
    find_best, format_bitrate, format_codec, format_duration, format_frame_rate, format_quality,
    format_shared, format_size, shutdown_requested, GroupMaxima, Policy, Priority,
};
use std::collections::{HashMap, HashSet};
use std::fs::{FileTimes, OpenOptions};
use std::path::{Path, PathBuf};

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

impl Disposal {
    /// Label for the results table once the file has been dealt with.
    pub fn done_label(&self) -> &'static str {
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

/// Whether the file at `path` is still the length it was when it was measured,
/// and a description of the discrepancy if not.
///
/// Every DELETE decision rests on measurements taken at the start of the scan,
/// and a scan of a large library runs for hours -- longer still when the
/// decision is being replayed from a report written yesterday. In that window a
/// download finishes, a re-encode lands over the top, a copy is truncated --
/// and the file about to be removed is no longer the file that was judged
/// redundant. Two syscalls immediately before an irreversible one is a cheap way
/// to notice.
///
/// Size is the only property compared, because it is the only one the
/// fingerprint records. It catches everything that changes a file's length,
/// which is essentially every way a video file is rewritten in practice, but it
/// cannot see an in-place edit that happens to preserve the byte count.
/// Recording an mtime or an inode alongside the fingerprint would close that
/// gap at the cost of a field on every cache entry; nothing has needed it yet.
///
/// A metadata error is deliberately NOT reported as a change. It proves nothing
/// about the file's contents, and the removal that follows will fail with the
/// real reason -- so a file that vanished mid-scan reads as a deletion that
/// could not happen rather than as a file quietly left alone.
fn changed_since(path: &str, measured_size: u64) -> Option<String> {
    let size = std::fs::metadata(path).ok()?.len();

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
    Done,
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
    // The last thing before the irreversible step, and the only check here that
    // consults the disk rather than the measurement. See `changed_since`: the
    // decision being acted on may be hours or days old.
    if let Some(detail) = changed_since(path, measured_size) {
        log::error!(
            "Not removing {}: it changed on disk after it was scanned",
            path
        );
        stats.delete_stale.record(detail);
        return Fate::Changed;
    }

    match dispose_of(path, disposal) {
        Ok(()) => Fate::Done,
        Err(e) => {
            log::error!("{:#}", e);
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
pub fn output_results(
    final_groups: &[Vec<usize>],
    fingerprints: &[VideoFingerprint],
    matches: &MatchIndex,
    output_file: Option<&String>,
    total_elapsed_secs: u64,
    policy: Policy,
    disposal: Option<&Disposal>,
    assume_yes: bool,
    stats: &RunStats,
) -> Result<Vec<String>> {

    // --- Pass 1: resolve each group ------------------------------------------
    // Groups are connected components, so every file belongs to exactly one and
    // its fate is settled entirely within that group: one KEEP, whatever the
    // group flags REVIEW, and the rest DELETE. There is nothing to reconcile
    // afterwards -- no file can be the KEEP pick of one group and a duplicate in
    // another, and none can be queued for deletion twice.
    let mut review_set: HashSet<usize> = HashSet::new();
    let mut delete_candidates: HashSet<usize> = HashSet::new();

    // Files this group would delete on the strength of a chain rather than a
    // measurement. Reused across groups, so it is cleared at the end of each.
    let mut indirect: HashSet<usize> = HashSet::new();

    for group in final_groups {
        let maxima = GroupMaxima::of(group, fingerprints);

        let keep_idx = find_best(group, fingerprints, policy.priority, &maxima);
        let keep_fp = &fingerprints[keep_idx];

        // Everything this group wants held back from deletion. A group can now
        // raise more than one, so it is a set rather than an Option.
        let mut group_review: HashSet<usize> = HashSet::new();

        // If the KEEP pick isn't top-tier on some quality metric, surface the
        // file that IS as worth a manual look. Metrics are checked in default
        // precedence order, skipping the one the user prioritised (KEEP wins
        // that by construction). Laziness matters here: find_best only runs for
        // the first metric KEEP actually falls short on.
        if let Some(r) = REVIEW_METRICS
            .iter()
            .copied()
            .filter(|&m| m != policy.priority && maxima.tier(keep_fp, m) == 0)
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
        // contenders, not the group's. That is what re-enables the raw-value
        // tiebreak: within one codec `mixed_codecs` is false, so two files
        // sitting inside the same tolerance band are separated by their actual
        // quality instead of falling through to alphabetical order.
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

        if standoff {
            for codec in codecs {
                let same_codec: Vec<usize> = contenders
                    .iter()
                    .copied()
                    .filter(|&idx| fingerprints[idx].codec == codec)
                    .collect();

                let codec_maxima = GroupMaxima::of(&same_codec, fingerprints);
                group_review.insert(find_best(&same_codec, fingerprints, policy.priority, &codec_maxima));
            }
        }

        // Everything in the group that isn't KEEP or REVIEW is a delete
        // candidate.
        //
        // In a standoff the champions REPLACE the group's KEEP pick rather than
        // joining it: the pick is one of the contenders, and holding it as well
        // would leave two survivors of the same codec -- one because it won its
        // codec, one because it happened to sort first. The exception is a pick
        // that is not a contender at all (it leads on the prioritised metric but
        // lost a codec-blind one); the standoff says nothing about that file, so
        // it keeps its usual protection.
        let mut protected: HashSet<usize> = group_review.clone();
        if !standoff || !contenders.contains(&keep_idx) {
            protected.insert(keep_idx);
        }

        for &idx in group {
            if protected.contains(&idx) {
                continue;
            }

            // A file may only be deleted on the strength of a comparison that
            // actually happened against a file that is staying on disk. A group
            // is a chain of matches, so losing the group's ranking does not by
            // itself mean the loser was ever measured against its survivor --
            // and deleting on that basis is deleting something nobody looked at
            // next to the thing replacing it.
            //
            // The anchor is the whole protected set rather than the KEEP pick
            // alone, so a codec standoff still works: each champion survives,
            // and a file directly matched against any of them has its evidence.
            //
            // --trust-chains says to accept the chain instead. It marks these
            // DELETE rather than REVIEW; it does not delete anything, which
            // still takes --delete or --move-to.
            if !policy.trust_chains
                && !protected.iter().any(|&s| matches.coverage(idx, s).is_some())
            {
                indirect.insert(idx);
                continue;
            }

            delete_candidates.insert(idx);
        }

        review_set.extend(group_review);
        review_set.extend(indirect.iter().copied());
        indirect.clear();
    }

    // Sorted for deterministic ordering. No subtraction of the REVIEW set is
    // needed: a group's protected files are excluded before its remainder is
    // added, and groups are disjoint, so the two sets cannot intersect.
    let mut delete_indices: Vec<usize> = delete_candidates.iter().copied().collect();
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
    let mut removed_bytes = 0u64;
    let delete_candidate_count = delete_indices.len();

    // What a run WOULD reclaim, computed from the same set the loop below
    // walks, so the dry run's figure and the real run's figure cannot disagree
    // about which files they describe. Sizes come from the fingerprints, which
    // the deletion pass re-checks against disk before touching anything, and
    // the set is index-based -- so a file reached through a hard link
    // contributes its bytes once.
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
                Fate::Done => {
                    removed_count += 1;
                    removed_bytes += fp.file_size;
                    deleted_paths.push(fp.path.clone());
                    delete_outcome.insert(idx, disposal.done_label());
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
    info!("\n========================================");
    info!("             RESULTS");
    info!("========================================\n");

    // Every file is in exactly one group, so this is simply how many distinct
    // files are implicated in a duplicate relationship at all -- the same set
    // the KEEP/DELETE/REVIEW labels below describe.
    let matched_file_count: usize = final_groups.iter().map(|g| g.len()).sum();

    let mut txt_out = String::new();
    // Whether the JSON tree is worth building at all.
    //
    // It was always built and then thrown away on a run that asked for text or
    // CSV, which was a small waste while every file contributed a fixed handful
    // of fields. The per-link list below is not fixed -- it is one object per
    // pair, so a group of `g` members contributes `g * (g - 1)` of them, and on
    // a library with one big group that is the largest allocation in the report.
    // Nobody should pay it to write a .txt.
    let wants_json = output_file
        .map(|p| {
            Path::new(p)
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("json"))
        })
        .unwrap_or(false);

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
    //   against what   shared_with .. shared_to_seconds -- the strongest measured
    //                link: which file, how much footage, and where in THIS file
    //                it sits.
    csv_wtr
        .write_record(&[
            "group",
            "action",
            "full_path",
            "length",
            "length_seconds",
            "resolution",
            "width",
            "height",
            "framerate",
            "framerate_fps",
            "codec",
            "size",
            "size_bytes",
            "bitrate",
            "bitrate_bps",
            "quality",
            "quality_bits_per_frame",
            "shared_with",
            "shared_seconds",
            "shared_from",
            "shared_to",
            "shared_from_seconds",
            "shared_to_seconds",
        ])
        .context("Failed to write CSV header")?;

    for (i, group) in final_groups.iter().enumerate() {
        let group_name = format!("group_{}", i + 1);

        info!("{}:", group_name);
        txt_out.push_str(&format!("{}:\n", group_name));

        let mut json_files = Vec::new();

        // How much footage each file has in common with the group members it
        // was actually matched against, in seconds rather than as a percentage.
        // A group is a chain of matches, not a set of mutual duplicates, so this
        // column is what tells the two apart: a row reporting the whole of its
        // runtime is a copy of something here, and one reporting eight seconds
        // is a file that shares eight seconds with something here.
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
            // by pair. Groups are small and this is the reporting pass, not the
            // comparison one -- there is no measurement here, only lookups
            // against figures phase 2 already took.
            let links = matches.links_of(idx, fingerprints);
            let best = links.first();
            let shared = best.map(|l| l.shared_seconds);

            let size_str = format_size(fp.file_size);
            let bitrate_str = format_bitrate(fp.bitrate());
            let duration_str = format_duration(fp.duration);
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
            let shared_str = format_shared(shared);

            // The same values as numbers, for the CSV and JSON. An unknown one
            // is an empty field / a null rather than a zero: a container that
            // never reported a frame rate is not a container that reported no
            // frames, and a quality figure derived from one is unknowable
            // rather than worst-in-group.
            let frame_rate_num = (fp.frame_rate > 0.0)
                .then(|| (fp.frame_rate * 1000.0).round() / 1000.0);
            let quality_num = (fp.quality() > 0).then(|| fp.quality());

            let frame_rate_raw = frame_rate_num.map(|f| f.to_string()).unwrap_or_default();
            let quality_raw = quality_num.map(|q| q.to_string()).unwrap_or_default();
            let size_bytes_raw = fp.file_size.to_string();
            let bitrate_bps_raw = fp.bitrate().to_string();
            let shared_seconds_raw = csv_seconds(shared);
            // Runtime and frame geometry as plain numbers, so every figure the
            // ranking uses can be sorted on. Resolution's raw form is the two
            // sides rather than their product: the product is one multiplication
            // away in any spreadsheet, and the sides are what was measured.
            let length_seconds_raw = format!("{:.2}", fp.duration);
            let width_raw = fp.width.to_string();
            let height_raw = fp.height.to_string();

            // The file the figures above describe. Empty rather than "-" when
            // there is none, for the same reason every other unknown is empty:
            // a CSV consumer should see a blank cell, not a sentinel it has to
            // know about.
            let shared_with_raw = best
                .map(|l| fingerprints[l.other].path.clone())
                .unwrap_or_default();

            // The envelope, in this file's own timeline. Both the clock form
            // and the raw seconds, like every other figure here.
            let best_span = best.and_then(|l| l.span);
            let shared_from_str =
                best_span.map(|s| format_duration(s.start_seconds())).unwrap_or_default();
            let shared_to_str =
                best_span.map(|s| format_duration(s.end_seconds())).unwrap_or_default();
            let shared_from_raw = csv_seconds(best_span.map(|s| s.start_seconds()));
            let shared_to_raw = csv_seconds(best_span.map(|s| s.end_seconds()));

            // Label by the fate the file's own group gave it -- the only one it
            // has, since it appears nowhere else.
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
            // ragged. "shared" is spelled out on every row because the console
            // has no header, and two time values side by side ("00:00:09, 0.8s")
            // would otherwise be ambiguous about which is which. The frame rate
            // and the bits-per-frame figure carry their units for the same
            // reason.
            let line = format!(
                "\t{:<8} {}, {}, {}, {}, {}, {}, {}, {} shared, {}",
                format!("{},", action_str),
                duration_str,
                res_str,
                frame_rate_str,
                codec_str,
                size_str,
                bitrate_str,
                quality_str,
                shared_str,
                fp.path
            );
            info!("{}", line);
            txt_out.push_str(&line);
            txt_out.push('\n');

            // 2. CSV Output
            csv_wtr.write_record(&[
                &group_name,
                action_str,
                &fp.path,
                &duration_str,
                &length_seconds_raw,
                &res_str,
                &width_raw,
                &height_raw,
                &frame_rate_str,
                &frame_rate_raw,
                &codec_str,
                &size_str,
                &size_bytes_raw,
                &bitrate_str,
                &bitrate_bps_raw,
                &quality_str,
                &quality_raw,
                &shared_with_raw,
                &shared_seconds_raw,
                &shared_from_str,
                &shared_to_str,
                &shared_from_raw,
                &shared_to_raw,
            ]).context("Failed to write CSV record")?;

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
                        "shared_seconds": (l.shared_seconds * 100.0).round() / 100.0,
                        "shared_from": l.span.map(|s| format_duration(s.start_seconds())),
                        "shared_to": l.span.map(|s| format_duration(s.end_seconds())),
                        "shared_from_seconds": l.span
                            .map(|s| (s.start_seconds() * 100.0).round() / 100.0),
                        "shared_to_seconds": l.span
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
                "length_seconds": (fp.duration * 100.0).round() / 100.0,
                "resolution": res_str,
                "width": fp.width,
                "height": fp.height,
                "framerate": frame_rate_str,
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
                "shared_with": best.map(|l| &fingerprints[l.other].path),
                "shared_seconds": shared.map(|s| (s * 100.0).round() / 100.0),
                "shared_from": best_span.map(|s| format_duration(s.start_seconds())),
                "shared_to": best_span.map(|s| format_duration(s.end_seconds())),
                "shared_from_seconds": best_span
                    .map(|s| (s.start_seconds() * 100.0).round() / 100.0),
                "shared_to_seconds": best_span.map(|s| (s.end_seconds() * 100.0).round() / 100.0),
                // Every measured link this file has in the group, strongest
                // first. The entry at index 0 is the one described above.
                "matches": json_matches,
            }));
        }

        info!(""); // Empty line for spacing
        txt_out.push_str("\n");

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
        summary.push_str(&trouble_lines(failed_count, changed_count));
    }

    info!("{}", summary);

    // Helpful nudge when there's something to clean up but nothing was touched.
    // Not for someone who declined: they typed the flag, saw the question, and
    // said no. Telling them to type it again would be answering back.
    if !acting && !declined && delete_candidate_count > 0 {
        info!(
            "\nRun with --delete to move the file(s) marked DELETE to the trash or with --move-to <DIR> to relocate them instead."
        );
    }

    // Save outputs cleanly returning Result<()>
    if let Some(out_path) = output_file {
        let path = Path::new(&out_path);
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "csv" => {
                let csv_bytes = csv_wtr.into_inner().context("Failed to finalize CSV buffer")?;
                std::fs::write(path, csv_bytes)
                    .context(format!("Failed to write CSV to {}", out_path))?;
            }
            "json" => {
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
                        "files_failed": failed_count,
                        "files_changed": changed_count,
                    },
                    "results": json_out_groups
                });
                std::fs::write(path, serde_json::to_string_pretty(&json_final).unwrap())
                    .context(format!("Failed to write JSON to {}", out_path))?;
            }
            _ => {
                let mut full_txt = String::new();
                full_txt.push_str(&txt_out);
                full_txt.push_str(&summary);
                full_txt.push_str("\n");

                std::fs::write(path, full_txt)
                    .context(format!("Failed to write Text to {}", out_path))?;
            }
        };

        info!("\nResults saved to {}", out_path);
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
    use crate::utils::{Policy, Priority};
    use tempfile::NamedTempFile;
    use std::fs;

    const CSV_HEADER: &str = "group;action;full_path;length;length_seconds;resolution;width;\
height;framerate;framerate_fps;codec;size;size_bytes;bitrate;bitrate_bps;quality;\
quality_bits_per_frame;shared_with;shared_seconds;shared_from;shared_to;shared_from_seconds;\
shared_to_seconds";

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

    /// The default posture, which is what almost every test wants: rank by
    /// `priority`, and hold a file that only reached its group through a chain
    /// for review.
    fn policy(priority: Priority) -> Policy {
        Policy { priority, trust_chains: false }
    }

    /// An index in which every file was compared with every other and found to
    /// contain it whole.
    ///
    /// Tests about deletion precedence are asking which copy wins, not how the
    /// group got linked, so this states the simplest thing that makes the
    /// question well-posed: everything here was measured against everything
    /// else. Without it a DELETE would be resting on a chain rather than a
    /// comparison, and the default policy holds those for REVIEW -- correctly,
    /// but it is not what these tests are about.
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

    fn read_json(path: &str) -> serde_json::Value {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn test_csv_output() {
        let fps = vec![mock_fp()];
        let groups = vec![vec![0]];

        let path_str = report_to("csv");

        // Report-only run: single item defaults to KEEP.
        let deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&path_str), 120, policy(Priority::Length), None,
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
            "group_1;KEEP;/fake/path/vid.mp4;00:01:00;60.00;1920x1080;1920;1080;30fps;30;h264;\
1.0MB;1048576;140kbps;139810;4.7kb/f;4660;;;;;;"
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
            &groups, &fps, &all_compared(fps.len()), Some(&path_str), 0, policy(Priority::Length), None,
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
            &groups, &fps, &all_compared(fps.len()), Some(&path_str), 0, policy(Priority::Length), None,
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
            &groups, &fps, &matches, Some(&csv_path), 0, policy(Priority::Length), None,
            true, &RunStats::default(),
        ).unwrap();
        output_results(
            &groups, &fps, &matches, Some(&json_path), 0, policy(Priority::Length), None,
            true, &RunStats::default(),
        ).unwrap();

        let csv = fs::read_to_string(&csv_path).unwrap();
        let report = read_json(&json_path);
        let file = &report["results"][0]["files"][0];

        assert!(
            csv.contains(
                "group_1;KEEP;/fake/a.mp4;00:01:40;100.00;1920x1080;1920;1080;30fps;30;h264;\
12.5MB;13107200;1.0Mbps;1048576;35.0kb/f;34952;/fake/b.mp4;80.00;"
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
        assert_eq!(file["shared_seconds"], 80.0);

        let _ = fs::remove_file(csv_path);
        let _ = fs::remove_file(json_path);
    }

    #[test]
    fn test_shared_duration_reads_the_same_on_a_clip_and_its_host() {
        // The reason this column is a duration. As coverage these two rows read
        // 10% and 100% and look like a malfunction next to any --match-percent;
        // as seconds they both read a minute, which is the truth.
        let fps = vec![
            mock_fp_at("/fake/host.mp4", 600.0),
            mock_fp_at("/fake/clip.mp4", 60.0),
        ];
        let groups = vec![vec![0, 1]];

        let matches = MatchIndex::new(vec![Match::new(0, 1, 0.10, 1.0)]);

        let path_str = report_to("csv");

        output_results(
            &groups, &fps, &matches, Some(&path_str), 0, policy(Priority::Length), None,
            true, &RunStats::default(),
        ).unwrap();

        // Read as `shared_with;shared_seconds`: each row names the other file
        // and reports the same minute of footage.
        let contents = fs::read_to_string(&path_str).unwrap();
        assert!(contents.contains(";/fake/clip.mp4;60.00;"), "host row: {}", contents);
        assert!(contents.contains(";/fake/host.mp4;60.00;"), "clip row: {}", contents);

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
            &groups, &fps, &matches, Some(&path_str), 0, policy(Priority::Length), None,
            true, &RunStats::default(),
        ).unwrap();

        let contents = fs::read_to_string(&path_str).unwrap();

        assert!(
            contents.contains("group_1;KEEP;/fake/host.mp4;")
                && contents.contains("/fake/clip.mp4;60.00;00:14:00;00:15:00;840.00;900.00"),
            "the host row should point into the host's own timeline: {}",
            contents
        );
        assert!(
            contents.contains("group_1;DELETE;/fake/clip.mp4;")
                && contents.contains("/fake/host.mp4;60.00;00:00:00;00:01:00;0.00;60.00"),
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
            &groups, &fps, &matches, Some(&path_str), 0, policy(Priority::Length), None,
            true, &RunStats::default(),
        ).unwrap();

        let json = read_json(&path_str);
        let file = &json["results"][0]["files"][0];
        assert_eq!(file["full_path"], "/fake/a.mp4");

        // The headline figures describe the strongest link, and name it.
        assert_eq!(file["shared_with"], "/fake/b.mp4");
        assert_eq!(file["shared_seconds"], 600.0);
        assert_eq!(file["shared_from"], "00:00:00");
        assert_eq!(file["shared_to"], "00:10:00");

        // Both links are present, strongest first, and the weak one is not
        // rounded away or merged into the strong one.
        let links = file["matches"].as_array().expect("matches should be an array");
        assert_eq!(links.len(), 2);
        assert_eq!(links[0]["full_path"], "/fake/b.mp4");
        assert_eq!(links[0]["shared_seconds"], 600.0);
        assert_eq!(links[1]["full_path"], "/fake/c.mp4");
        assert_eq!(links[1]["shared_seconds"], 6.0);
        assert_eq!(links[1]["shared_from"], "00:00:00");
        assert_eq!(links[1]["shared_to"], "00:00:06");

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_a_chained_member_reports_only_the_links_it_actually_has() {
        // 0-1 and 1-2 were measured; 0 and 2 never were. 0's list must hold one
        // entry, not two with a fabricated zero -- the absent pair is unmeasured,
        // which is a different claim from "these two share nothing".
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
            &groups, &fps, &matches, Some(&path_str), 0, policy(Priority::Length), None,
            true, &RunStats::default(),
        ).unwrap();

        let json = read_json(&path_str);
        let files = json["results"][0]["files"].as_array().unwrap();

        let a = files.iter().find(|f| f["full_path"] == "/fake/a.mp4").unwrap();
        assert_eq!(a["matches"].as_array().unwrap().len(), 1);
        assert_eq!(a["shared_with"], "/fake/b.mp4");

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
            &groups, &fps, &matches, Some(&path_str), 0, policy(Priority::Length), None,
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
            &groups, &fps, &all_compared(fps.len()), Some(&path_str), 0, policy(Priority::Length), None,
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
            &groups, &fps, &matches, Some(&path_str), 0, policy(Priority::Length), None,
            true, &RunStats::default(),
        ).unwrap();

        let report = read_json(&path_str);
        let group = &report["results"][0];

        assert_eq!(group["files"][0]["shared_seconds"], 80.0);

        // 1 Mbps at 30fps.
        assert_eq!(group["files"][0]["codec"], "h264");
        assert_eq!(group["files"][0]["framerate"], "30fps");
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
            &groups, &fps, &all_compared(fps.len()), Some(&json_path), 0, policy(Priority::Length), None,
            true, &RunStats::default(),
        ).unwrap();
        output_results(
            &groups, &fps, &all_compared(fps.len()), Some(&csv_path), 0, policy(Priority::Length), None,
            true, &RunStats::default(),
        ).unwrap();

        let report = read_json(&json_path);
        let file = &report["results"][0]["files"][0];

        assert!(file["framerate_fps"].is_null(), "unknown is not 0 fps");
        assert!(file["quality_bits_per_frame"].is_null(), "and it makes quality unknowable");
        assert_eq!(file["quality"], "-");
        assert_eq!(file["bitrate_bps"], 1_048_576, "the bitrate is still perfectly knowable");

        // The CSV says the same thing with empty fields rather than nulls: a
        // dash in the formatted column, nothing at all in the raw one.
        let csv = fs::read_to_string(&csv_path).unwrap();
        assert!(
            csv.contains(";1920;1080;-;;h264;7.5MB;7864320;1.0Mbps;1048576;-;;"),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), Some(&path_str), 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), Some(&path_str), 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
            Some(&Disposal::Permanent), true, &stats,
        ).unwrap();

        assert!(Path::new(&shrunk_path).exists());
        assert_eq!(stats.delete_stale.count(), 1);
        assert_eq!(
            stats.delete_failed.count(), 0,
            "refusing to act on stale information is not a failed removal"
        );
    }

    #[test]
    fn test_a_chain_collapses_in_a_single_pass() {
        // A chain of matches arrives as one group, so the whole chain is settled
        // at once: file 1 was a bridge between 0 and 2 and is redundant against
        // the best copy of the lot. Nothing survives to be caught by a later
        // run.
        let dir = tempfile::tempdir().unwrap();
        let p0 = at(&dir, "best.mkv");
        let p1 = at(&dir, "bridge.mkv");
        let p2 = at(&dir, "tail.mkv");

        let fps = vec![
            mock_fp_at(&p0, 100.0), // best of the component
            mock_fp_at(&p1, 90.0),  // bridge
            mock_fp_at(&p2, 10.0),
        ];
        materialize_all(&fps);

        let groups = vec![vec![0, 1, 2]];

        let mut deleted = output_results(
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();
        deleted.sort();

        assert!(Path::new(&p0).exists(), "the component's best must remain");
        assert!(!Path::new(&p1).exists(), "bridge duplicate must be deleted in one pass");
        assert!(!Path::new(&p2).exists(), "tail duplicate must be deleted");

        let mut expected = vec![p1, p2];
        expected.sort();
        assert_eq!(deleted, expected);
    }

    /// The chain the guard exists for: 0-1 and 1-2 matched, 0 and 2 never did.
    /// One group of three, and 2 has no measurement against 0.
    fn chain_of_three() -> MatchIndex {
        MatchIndex::new(vec![
            Match::new(0, 1, 1.0, 1.0),
            Match::new(1, 2, 1.0, 1.0),
        ])
    }

    #[test]
    fn test_a_file_that_never_matched_the_kept_copy_is_held_for_review() {
        // File 2 loses the group's ranking, but the only file it was ever
        // compared with is file 1, which is itself being deleted. Deleting 2 on
        // that basis removes a file nobody ever measured against the copy
        // replacing it -- so it is flagged for a human instead.
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

        let groups = vec![vec![0, 1, 2]];
        let path_str = report_to("json");

        let deleted = output_results(
            &groups, &fps, &chain_of_three(), Some(&path_str), 0,
            policy(Priority::Length), Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p0).exists(), "the KEEP pick stands");
        assert!(!Path::new(&p1).exists(), "1 matched 0 directly, so its loss is measured");
        assert!(Path::new(&p2).exists(), "2 never matched 0 and must survive");
        assert_eq!(deleted, vec![p1], "and only the measured loss is reported gone");

        let files = &read_json(&path_str)["results"][0]["files"];
        assert_eq!(files[2]["action"], "REVIEW");

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_trust_chains_deletes_the_indirect_file_instead() {
        // The same group with the guard off. The chain is accepted as evidence,
        // so 2 goes with 1 -- which is the whole point of the flag, and why it
        // is not the default.
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

        let groups = vec![vec![0, 1, 2]];
        let policy = Policy {
            priority: Priority::Length,
            trust_chains: true,
        };

        let mut deleted = output_results(
            &groups, &fps, &chain_of_three(), None, 0, policy,
            Some(&Disposal::Permanent), true, &RunStats::default(),
        ).unwrap();
        deleted.sort();

        assert!(Path::new(&p0).exists());
        assert!(!Path::new(&p1).exists());
        assert!(!Path::new(&p2).exists(), "--trust-chains accepts the chain");

        let mut expected = vec![p1, p2];
        expected.sort();
        assert_eq!(deleted, expected);
    }

    #[test]
    fn test_trust_chains_does_not_arm_deletion_by_itself() {
        // The naming hazard the flag was named around. It changes labels; it is
        // not a deletion switch, and a report-only run stays report-only.
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

        let groups = vec![vec![0, 1, 2]];
        let policy = Policy {
            priority: Priority::Length,
            trust_chains: true,
        };
        let path_str = report_to("json");

        let deleted = output_results(
            &groups, &fps, &chain_of_three(), Some(&path_str), 0, policy,
            None, true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p0).exists());
        assert!(Path::new(&p1).exists(), "no disposal was passed, so nothing may go");
        assert!(Path::new(&p2).exists());
        assert!(deleted.is_empty());

        let report = read_json(&path_str);
        assert_eq!(report["summary"]["deletion_enabled"], false);
        assert_eq!(report["summary"]["files_marked_delete"], 2, "labels still change");
        assert_eq!(report["results"][0]["files"][2]["action"], "DELETE");

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
            &groups, &fps, &matches, None, 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), Some(&path_str), 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), Some(&path_str), 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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
        // every tier ties and the raw quality figure is the only thing left.
        // Group-wide that figure is suppressed (the group spans codecs), which
        // would hand the election to alphabetical order and elect the WORSE
        // file -- the champion is therefore elected against its own codec's
        // maxima, where the comparison is legitimate.
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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
            &groups, &fps, &all_compared(fps.len()), None, 0, policy(Priority::Length),
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