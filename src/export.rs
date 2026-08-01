use anyhow::{anyhow, Context, Result};
use log::info;
use crate::compare::MatchIndex;
use crate::fingerprint::VideoFingerprint;
use crate::stats::RunStats;
use crate::utils::{
    find_best, format_bitrate, format_codec, format_duration, format_frame_rate, format_quality,
    format_shared, format_size, shutdown_requested, GroupMaxima, Priority,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Metrics that can justify a REVIEW flag, in default precedence order.
///
/// These are also exactly the metrics that mean the same thing regardless of
/// codec, which is why the same list does double duty below as the definition
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

/// Remove a single file, either by moving it to the system trash (default,
/// recoverable) or by deleting it permanently. Trash semantics are handled by
/// the `trash` crate, which implements the FreeDesktop.org spec on Linux.
fn remove_path(path: &str, permanent: bool) -> Result<()> {
    if permanent {
        std::fs::remove_file(path)
            .with_context(|| format!("Failed to permanently delete {}", path))?;
    } else {
        trash::delete(path)
            .map_err(|e| anyhow!("Failed to move {} to trash: {}", path, e))?;
    }
    Ok(())
}

pub fn output_results(
    final_groups: &[Vec<usize>],
    fingerprints: &[VideoFingerprint],
    matches: &MatchIndex,
    output_file: Option<&String>,
    total_elapsed_secs: u64,
    priority: Priority,
    delete: bool,
    permanent: bool,
    stats: &RunStats,
) -> Result<()> {

    // `--permanent` only has meaning alongside `--delete`; on its own it must
    // never trigger any destructive action.
    if permanent && !delete {
        info!("Note: --permanent has no effect without --delete; running in report-only mode.");
    }
    let permanent_delete = permanent && delete;

    // --- Pass 1: resolve each file's fate across ALL groups ------------------
    // Clusters overlap, so a file can appear in several groups with different
    // per-group roles. Precedence is REVIEW > DELETE > KEEP:
    //   * REVIEW anywhere -> always kept for manual inspection (never deleted).
    //   * DELETE anywhere -> deleted, even if it is the KEEP pick of another
    //     group. This is what lets a single run remove every redundant copy: a
    //     file that is best in one group but redundant in an overlapping one is
    //     still removed, so you don't have to re-run until the chain collapses.
    //   * otherwise -> kept (it was the best in every group it appears in).
    // Using sets also guarantees each file is considered exactly once, so a
    // file shared by several groups is never queued for deletion twice.
    let mut review_set: HashSet<usize> = HashSet::new();
    let mut delete_candidates: HashSet<usize> = HashSet::new();

    for group in final_groups {
        let maxima = GroupMaxima::of(group, fingerprints);

        let keep_idx = find_best(group, fingerprints, priority, &maxima);
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
                group_review.insert(find_best(&same_codec, fingerprints, priority, &codec_maxima));
            }
        }

        // Everything in the group that isn't KEEP or REVIEW is a delete
        // candidate. DELETE wins over KEEP globally, so we don't care whether
        // the file is a KEEP pick in some other group.
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
            if !protected.contains(&idx) {
                delete_candidates.insert(idx);
            }
        }

        review_set.extend(group_review);
    }

    // REVIEW protection overrides DELETE. Sorted for deterministic ordering.
    let mut delete_indices: Vec<usize> =
        delete_candidates.difference(&review_set).copied().collect();
    delete_indices.sort_unstable();

    // --- Pass 2: delete each unique target exactly once ----------------------
    let mut deleted_count = 0usize;
    let mut failed_count = 0usize;
    let mut freed_bytes = 0u64;
    let delete_candidate_count = delete_indices.len();

    // Maps a file index to the outcome label to print in the results table.
    let mut delete_outcome: HashMap<usize, &'static str> = HashMap::new();

    if delete {
        for &idx in &delete_indices {
            if shutdown_requested() {
                info!(
                    "Interrupted: stopped after {} deletion(s); {} file(s) left untouched.",
                    deleted_count,
                    delete_candidate_count - deleted_count - failed_count
                );
                break;
            }
            let fp = &fingerprints[idx];
            match remove_path(&fp.path, permanent_delete) {
                Ok(()) => {
                    deleted_count += 1;
                    freed_bytes += fp.file_size;
                    delete_outcome.insert(idx, "DELETED");
                }
                Err(e) => {
                    log::error!("{:#}", e);
                    failed_count += 1;
                    stats.delete_failed.record(fp.path.clone());
                    delete_outcome.insert(idx, "FAILED");
                }
            }
        }
    }

    // --- Reporting -----------------------------------------------------------
    info!("\n========================================");
    info!("             RESULTS");
    info!("========================================\n");

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

    let mut txt_out = String::new();
    let mut json_out_groups = Vec::new();

    // Use csv crate for robust and RFC-compliant CSV generation
    let mut csv_wtr = csv::WriterBuilder::new()
        .delimiter(b';')
        .from_writer(Vec::new());

    csv_wtr
        .write_record(&[
            "group",
            "resolution",
            "codec",
            "framerate",
            "size",
            "bitrate",
            "quality",
            "length",
            "shared",
            "shared_seconds",
            "match_percent",
            "full_path",
            "action",
        ])
        .context("Failed to write CSV header")?;

    for (i, group) in final_groups.iter().enumerate() {
        let group_name = format!("group_{}", i + 1);

        info!("{}:", group_name);
        txt_out.push_str(&format!("{}:\n", group_name));

        let mut json_files = Vec::new();
        let mut group_low: Option<f64> = None;

        for &idx in group {
            let fp = &fingerprints[idx];
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

            // How much footage this file has in common with the rest of its
            // group, in seconds rather than as a percentage.
            //
            // A percentage was the obvious choice and the wrong one. It invites
            // comparison against --match-percent, which measures the opposite
            // end of the pair and so routinely sits above what is shown here --
            // making a correct report look broken. Worse, on short videos it
            // quantizes brutally: a file with four keyframes can only ever
            // score 0, 25, 50, 75 or 100%, so a single incidental frame reads
            // as an authoritative-looking 25%. In seconds that same match reads
            // "0.8s", next to a length of "00:00:03", and needs no explaining.
            let shared = matches.weakest_shared_in_group(idx, group, fingerprints);
            let shared_str = format_shared(shared);

            if let Some(s) = shared {
                group_low = Some(group_low.map_or(s, |g: f64| g.min(s)));
            }

            // The percentage still goes to the machine-readable outputs, where
            // nothing is being read at a glance and a filter like "at least 90%
            // redundant" is a reasonable thing to want.
            let coverage = matches.weakest_in_group(idx, group);
            let coverage_pct = coverage
                .map(|c| ((c * 100.0).clamp(0.0, 100.0) as f64 * 10.0).round() / 10.0);

            // Label by the file's GLOBAL fate (precedence REVIEW > DELETE > KEEP).
            // A file that is redundant in an overlapping group is shown DELETE/
            // DELETED in every group, including one where it was the local best.
            // In a dry run the delete targets stay as the recommendation DELETE;
            // with --delete they become DELETED (or FAILED on error).
            let action_str = if review_set.contains(&idx) {
                "REVIEW"
            } else if delete_candidates.contains(&idx) {
                if delete {
                    delete_outcome.get(&idx).copied().unwrap_or("SKIPPED")
                } else {
                    "DELETE"
                }
            } else if delete {
                "KEPT"
            } else {
                "KEEP"
            };

            // 1. Console / Text Output
            //
            // "shared" is spelled out on every row because the console has no
            // header, and two time values side by side ("00:00:09, 0.8s") would
            // otherwise be ambiguous about which is which. The frame rate and
            // the bits-per-frame figure carry their units for the same reason.
            info!(
                "\t{}, {}, {}, {}, {}, {}, {}, {} shared, {}, {}",
                res_str,
                codec_str,
                frame_rate_str,
                size_str,
                bitrate_str,
                quality_str,
                duration_str,
                shared_str,
                fp.path,
                action_str
            );
            txt_out.push_str(&format!(
                "\t{}, {}, {}, {}, {}, {}, {}, {} shared, {}, {}\n",
                res_str,
                codec_str,
                frame_rate_str,
                size_str,
                bitrate_str,
                quality_str,
                duration_str,
                shared_str,
                fp.path,
                action_str
            ));

            // 2. CSV Output
            csv_wtr.write_record(&[
                &group_name,
                &res_str,
                &codec_str,
                &frame_rate_str,
                &size_str,
                &bitrate_str,
                &quality_str,
                &duration_str,
                &shared_str,
                &shared.map(|s| format!("{:.2}", s)).unwrap_or_default(),
                &coverage_pct.map(|p| format!("{:.1}", p)).unwrap_or_default(),
                &fp.path,
                action_str,
            ]).context("Failed to write CSV record")?;

            // 3. JSON File Output
            json_files.push(serde_json::json!({
                "resolution": res_str,
                "codec": codec_str,
                // null, not 0: a container that never reported a frame rate is
                // not a container that reported no frames.
                "frame_rate": (fp.frame_rate > 0.0)
                    .then(|| (fp.frame_rate * 1000.0).round() / 1000.0),
                "size": size_str,
                "size_bytes": fp.file_size,
                "bitrate": bitrate_str,
                "bitrate_bps": fp.bitrate(),
                "quality": quality_str,
                "quality_bits_per_frame": (fp.quality() > 0).then(|| fp.quality()),
                "length": duration_str,
                "shared": shared_str,
                "shared_seconds": shared.map(|s| (s * 100.0).round() / 100.0),
                "match_percent": coverage_pct,
                "full_path": fp.path,
                "action": action_str,
            }));
        }

        info!(""); // Empty line for spacing
        txt_out.push_str("\n");

        json_out_groups.push(serde_json::json!({
            "group": group_name,
            // The group's thinnest link, for filtering. Not printed to the
            // console: every row already carries its own worst-case figure, so
            // a header repeating one of them would be noise.
            "lowest_shared_seconds": group_low.map(|s| (s * 100.0).round() / 100.0),
            "files": json_files
        }));
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

    if delete {
        if permanent_delete {
            summary.push_str(&format!(
                "\nPermanently deleted {} file(s), {} freed.",
                deleted_count,
                format_size(freed_bytes)
            ));
        } else {
            summary.push_str(&format!(
                "\nMoved {} file(s) to trash ({} total).",
                deleted_count,
                format_size(freed_bytes)
            ));
        }
        if failed_count > 0 {
            summary.push_str(&format!(
                "\n{} file(s) could not be removed (see errors above).",
                failed_count
            ));
        }
    }

    info!("{}", summary);

    // Helpful nudge when there's something to clean up but nothing was touched.
    if !delete && delete_candidate_count > 0 {
        info!(
            "\nRun with --delete to move the {} file(s) marked DELETE to the trash \
             (add --permanent to remove them for good).",
            delete_candidate_count
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
                let json_final = serde_json::json!({
                    "summary": {
                        "total_groups": final_groups.len(),
                        "total_files_matched": matched_file_count,
                        "time_elapsed_seconds": total_elapsed_secs,
                        "deletion_enabled": delete,
                        "permanent": permanent_delete,
                        "files_deleted": deleted_count,
                        "files_failed": failed_count,
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::Match;
    use crate::utils::Priority;
    use tempfile::NamedTempFile;
    use std::fs;

    fn mock_fp() -> VideoFingerprint {
        VideoFingerprint {
            path: "/fake/path/vid.mp4".to_string(),
            valid_hashes: vec![], valid_t_start: vec![], valid_t_end: vec![],
            total_frames: 100, width: 1920, height: 1080, duration: 60.0, file_size: 1048576,
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

    /// Real files with names WE choose, inside a directory that cleans itself
    /// up. NamedTempFile picks random names, which is useless when the point of
    /// a test is that alphabetical order must not decide the outcome.
    fn touch(dir: &tempfile::TempDir, name: &str) -> String {
        let path = dir.path().join(name);
        fs::write(&path, b"video").unwrap();
        path.to_string_lossy().to_string()
    }

    /// Tests about deletion precedence don't care about overlap; an empty index
    /// reports every figure as unknown, which the report renders as "-".
    fn no_matches() -> MatchIndex {
        MatchIndex::new(vec![])
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

    #[test]
    fn test_csv_output() {
        let fps = vec![mock_fp()];
        let groups = vec![vec![0]];

        let path_str = report_to("csv");

        // Report-only run: single item defaults to KEEP.
        output_results(
            &groups, &fps, &no_matches(), Some(&path_str), 120, Priority::Length, false, false,
            &RunStats::default(),
        ).unwrap();

        let contents = fs::read_to_string(&path_str).unwrap();

        // Assert headers exist (separated by semicolons)
        assert!(contents.contains(
            "group;resolution;codec;framerate;size;bitrate;quality;length;shared;shared_seconds;match_percent;full_path;action"
        ));
        // 1 MiB over 60s = ~140kbps, which at 30fps is ~4.7kbit in each frame.
        // Assert data exists and defaults to KEEP.
        assert!(contents.contains(
            "group_1;1920x1080;h264;30fps;1.0MB;140kbps;4.7kb/f;00:01:00;-;;;/fake/path/vid.mp4;KEEP"
        ), "{}", contents);

        // Clean up
        let _ = fs::remove_file(path_str);
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

        let matches = MatchIndex::new(vec![Match {
            a: 0,
            b: 1,
            coverage_a: 0.10,
            coverage_b: 1.0,
        }]);

        let path_str = report_to("csv");

        output_results(
            &groups, &fps, &matches, Some(&path_str), 0, Priority::Length, false, false,
            &RunStats::default(),
        ).unwrap();

        let contents = fs::read_to_string(&path_str).unwrap();
        assert!(contents.contains(";1m00s;60.00;10.0;/fake/host.mp4;"), "host row: {}", contents);
        assert!(contents.contains(";1m00s;60.00;100.0;/fake/clip.mp4;"), "clip row: {}", contents);

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

        let matches = MatchIndex::new(vec![Match {
            a: 0,
            b: 1,
            coverage_a: 0.0714, // 1 of 14 keyframes
            coverage_b: 0.25,   // 1 of 4
        }]);

        let path_str = report_to("csv");

        output_results(
            &groups, &fps, &matches, Some(&path_str), 0, Priority::Length, false, false,
            &RunStats::default(),
        ).unwrap();

        let contents = fs::read_to_string(&path_str).unwrap();
        assert!(contents.contains(";0.6s;"), "sub-second overlap must stay legible: {}", contents);
        assert!(
            !contents.contains(";00:00:00;"),
            "a real overlap must never render as a zeroed clock: {}",
            contents
        );

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_files_in_several_groups_are_counted_once() {
        // File 1 belongs to both groups. Summing group sizes gives 4; there are
        // only 3 files, and 4 would be a nonsense figure the moment a user
        // compared it against the number of videos scanned.
        let fps = vec![
            mock_fp_at("/fake/a.mp4", 100.0),
            mock_fp_at("/fake/b.mp4", 90.0),
            mock_fp_at("/fake/c.mp4", 10.0),
        ];
        let groups = vec![vec![0, 1], vec![1, 2]];

        let path_str = report_to("json");

        output_results(
            &groups, &fps, &no_matches(), Some(&path_str), 0, Priority::Length, false, false,
            &RunStats::default(),
        ).unwrap();

        let report: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path_str).unwrap()).unwrap();

        assert_eq!(report["summary"]["total_groups"], 2);
        assert_eq!(
            report["summary"]["total_files_matched"], 3,
            "a file in two groups is still one file"
        );

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_json_carries_raw_figures_alongside_the_formatted_ones() {
        let fps = vec![
            mock_fp_at("/fake/a.mp4", 100.0),
            mock_fp_at("/fake/b.mp4", 100.0),
        ];
        let groups = vec![vec![0, 1]];
        let matches = MatchIndex::new(vec![Match {
            a: 0,
            b: 1,
            coverage_a: 0.80,
            coverage_b: 0.80,
        }]);

        let path_str = report_to("json");

        output_results(
            &groups, &fps, &matches, Some(&path_str), 0, Priority::Length, false, false,
            &RunStats::default(),
        ).unwrap();

        let report: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path_str).unwrap()).unwrap();
        let group = &report["results"][0];

        assert_eq!(group["lowest_shared_seconds"], 80.0);
        assert_eq!(group["files"][0]["shared"], "1m20s");
        assert_eq!(group["files"][0]["shared_seconds"], 80.0);
        assert_eq!(group["files"][0]["match_percent"], 80.0);

        // 1 Mbps at 30fps.
        assert_eq!(group["files"][0]["codec"], "h264");
        assert_eq!(group["files"][0]["frame_rate"], 30.0);
        assert_eq!(group["files"][0]["bitrate_bps"], 1_048_576);
        assert_eq!(group["files"][0]["quality_bits_per_frame"], 34_952);

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_an_unreported_frame_rate_is_null_rather_than_zero() {
        let mut lonely = mock_fp_at("/fake/a.mp4", 60.0);
        lonely.frame_rate = 0.0;

        let fps = vec![lonely];
        let groups = vec![vec![0]];
        let path_str = report_to("json");

        output_results(
            &groups, &fps, &no_matches(), Some(&path_str), 0, Priority::Length, false, false,
            &RunStats::default(),
        ).unwrap();

        let report: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path_str).unwrap()).unwrap();
        let file = &report["results"][0]["files"][0];

        assert!(file["frame_rate"].is_null(), "unknown is not 0 fps");
        assert!(file["quality_bits_per_frame"].is_null(), "and it makes quality unknowable");
        assert_eq!(file["quality"], "-");
        assert_eq!(file["bitrate_bps"], 1_048_576, "the bitrate is still perfectly knowable");

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_permanent_delete_removes_only_delete_targets() {
        // Two real temp files so we can verify actual filesystem effects. We use
        // permanent deletion here specifically so the test never pollutes the
        // user's trash.
        let keep_file = NamedTempFile::new().unwrap();
        let del_file = NamedTempFile::new().unwrap();
        let keep_path = keep_file.path().to_string_lossy().to_string();
        let del_path = del_file.path().to_string_lossy().to_string();

        // Longer duration => higher "tier" => KEEP under Priority::Length.
        let fp_keep = mock_fp_at(&keep_path, 60.0);
        let fp_del = mock_fp_at(&del_path, 10.0);

        let fps = vec![fp_keep, fp_del];
        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();

        output_results(
            &groups, &fps, &no_matches(), None, 0, Priority::Length, true, true, &stats,
        ).unwrap();

        assert!(Path::new(&keep_path).exists(), "KEEP file must remain");
        assert!(!Path::new(&del_path).exists(), "DELETE file must be removed");
        assert!(!stats.had_problems(), "a clean deletion must not fail the run");
    }

    #[test]
    fn test_delete_wins_across_overlapping_groups() {
        // A duplicate chain spread across overlapping groups must collapse in a
        // SINGLE pass. File 1 is the bridge: it is the KEEP pick of group B but
        // redundant in group A. Under DELETE-wins it is removed on the first run
        // rather than surviving until a later run.
        let f0 = NamedTempFile::new().unwrap();
        let f1 = NamedTempFile::new().unwrap();
        let f2 = NamedTempFile::new().unwrap();
        let p0 = f0.path().to_string_lossy().to_string();
        let p1 = f1.path().to_string_lossy().to_string();
        let p2 = f2.path().to_string_lossy().to_string();

        let fp0 = mock_fp_at(&p0, 100.0); // global best
        let fp1 = mock_fp_at(&p1, 90.0);  // bridge
        let fp2 = mock_fp_at(&p2, 10.0);

        let fps = vec![fp0, fp1, fp2];
        let groups = vec![vec![0, 1], vec![1, 2]];

        output_results(
            &groups, &fps, &no_matches(), None, 0, Priority::Length, true, true,
            &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p0).exists(), "global best must remain");
        assert!(!Path::new(&p1).exists(), "bridge duplicate must be deleted in one pass");
        assert!(!Path::new(&p2).exists(), "tail duplicate must be deleted");
    }

    #[test]
    fn test_shared_target_deleted_once_without_over_collapsing() {
        // The shared duplicate (index 2) is a DELETE target in BOTH groups. It
        // must be removed exactly once (the original double-delete bug), while
        // the two independent "best" files are both retained -- DELETE-wins is
        // not a blunt component-collapse.
        let f0 = NamedTempFile::new().unwrap();
        let f1 = NamedTempFile::new().unwrap();
        let f2 = NamedTempFile::new().unwrap();
        let p0 = f0.path().to_string_lossy().to_string();
        let p1 = f1.path().to_string_lossy().to_string();
        let p2 = f2.path().to_string_lossy().to_string();

        let fp0 = mock_fp_at(&p0, 60.0);
        let fp1 = mock_fp_at(&p1, 60.0);
        let fp2 = mock_fp_at(&p2, 10.0);

        let fps = vec![fp0, fp1, fp2];
        let groups = vec![vec![0, 2], vec![1, 2]];
        let stats = RunStats::default();

        // Previously errored on the second deletion attempt; must succeed now.
        output_results(
            &groups, &fps, &no_matches(), None, 0, Priority::Length, true, true, &stats,
        ).unwrap();

        assert!(Path::new(&p0).exists(), "independent best A must remain");
        assert!(Path::new(&p1).exists(), "independent best B must remain");
        assert!(!Path::new(&p2).exists(), "shared duplicate must be removed exactly once");
        assert_eq!(
            stats.delete_failed.count(), 0,
            "a target queued by two groups must not report a second, failing attempt"
        );
    }

    #[test]
    fn test_quality_settles_groups_that_tie_on_length_and_resolution() {
        // Same length, same resolution, same codec: under the default order the
        // decision reaches quality, and the denser copy is kept. Nothing is
        // flagged REVIEW, because the KEEP pick is top-tier on every metric.
        let f_hi = NamedTempFile::new().unwrap();
        let f_lo = NamedTempFile::new().unwrap();
        let p_hi = f_hi.path().to_string_lossy().to_string();
        let p_lo = f_lo.path().to_string_lossy().to_string();

        let mut fp_hi = mock_fp_at(&p_hi, 60.0);
        fp_hi.file_size *= 2; // 2 Mbps
        let fp_lo = mock_fp_at(&p_lo, 60.0); // 1 Mbps

        let fps = vec![fp_hi, fp_lo];
        let groups = vec![vec![0, 1]];

        output_results(
            &groups, &fps, &no_matches(), None, 0, Priority::Length, true, true,
            &RunStats::default(),
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
        let f_h264 = NamedTempFile::new().unwrap();
        let f_av1 = NamedTempFile::new().unwrap();
        let p_h264 = f_h264.path().to_string_lossy().to_string();
        let p_av1 = f_av1.path().to_string_lossy().to_string();

        let fp_h264 = mock_fp_coded(&p_h264, 60.0, "h264");
        let mut fp_av1 = mock_fp_coded(&p_av1, 60.0, "av1");
        fp_av1.file_size /= 2; // half the bytes, same picture

        let fps = vec![fp_h264, fp_av1];
        let groups = vec![vec![0, 1]];
        let path_str = report_to("json");

        output_results(
            &groups, &fps, &no_matches(), Some(&path_str), 0, Priority::Length, true, true,
            &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_h264).exists(), "the h264 copy must survive");
        assert!(Path::new(&p_av1).exists(), "the av1 copy must survive");

        let report: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path_str).unwrap()).unwrap();
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
        let f_h264 = NamedTempFile::new().unwrap();
        let f_av1 = NamedTempFile::new().unwrap();
        let f_small = NamedTempFile::new().unwrap();
        let p_h264 = f_h264.path().to_string_lossy().to_string();
        let p_av1 = f_av1.path().to_string_lossy().to_string();
        let p_small = f_small.path().to_string_lossy().to_string();

        let fp_h264 = mock_fp_coded(&p_h264, 60.0, "h264");
        let fp_av1 = mock_fp_coded(&p_av1, 60.0, "av1");
        let mut fp_small = mock_fp_coded(&p_small, 60.0, "h264");
        fp_small.width = 1280;
        fp_small.height = 720;

        let fps = vec![fp_h264, fp_av1, fp_small];
        let groups = vec![vec![0, 1, 2]];

        output_results(
            &groups, &fps, &no_matches(), None, 0, Priority::Length, true, true,
            &RunStats::default(),
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
        let h264_best = touch(&dir, "h264_best.mkv");
        let h264_mid = touch(&dir, "h264_mid.mkv");
        let hevc_best = touch(&dir, "hevc_best.mkv");
        let hevc_worst = touch(&dir, "hevc_worst.mkv");
        let av1_only = touch(&dir, "av1_only.mkv");

        let fps = vec![
            mock_fp_sized(&h264_best, 60.0, "h264", 24_000_000),
            mock_fp_sized(&h264_mid, 60.0, "h264", 9_000_000),
            mock_fp_sized(&hevc_best, 60.0, "hevc", 12_000_000),
            mock_fp_sized(&hevc_worst, 60.0, "hevc", 3_000_000),
            mock_fp_sized(&av1_only, 60.0, "av1", 6_000_000),
        ];
        let groups = vec![vec![0, 1, 2, 3, 4]];

        output_results(
            &groups, &fps, &no_matches(), None, 0, Priority::Length, true, true,
            &RunStats::default(),
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
        let a_worse = touch(&dir, "a_worse.mkv");
        let z_best = touch(&dir, "z_best.mkv");
        let other_codec = touch(&dir, "m_av1.mkv");

        let fps = vec![
            mock_fp_sized(&a_worse, 60.0, "h264", 9_600_000),
            mock_fp_sized(&z_best, 60.0, "h264", 10_000_000),
            mock_fp_sized(&other_codec, 60.0, "av1", 4_000_000),
        ];
        let groups = vec![vec![0, 1, 2]];

        output_results(
            &groups, &fps, &no_matches(), None, 0, Priority::Length, true, true,
            &RunStats::default(),
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
        let f_long = NamedTempFile::new().unwrap();
        let f_short = NamedTempFile::new().unwrap();
        let p_long = f_long.path().to_string_lossy().to_string();
        let p_short = f_short.path().to_string_lossy().to_string();

        let fp_long = mock_fp_coded(&p_long, 120.0, "av1");
        let fp_short = mock_fp_coded(&p_short, 60.0, "h264");

        let fps = vec![fp_long, fp_short];
        let groups = vec![vec![0, 1]];

        output_results(
            &groups, &fps, &no_matches(), None, 0, Priority::Length, true, true,
            &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_long).exists(), "the longer copy is the KEEP pick");
        assert!(!Path::new(&p_short).exists(), "a genuine loss is still a loss");
    }

    #[test]
    fn test_failed_deletion_is_recorded_for_the_summary_and_exit_code() {
        let keep_file = NamedTempFile::new().unwrap();
        let keep_path = keep_file.path().to_string_lossy().to_string();
        let missing = "/nonexistent/vid-fp/definitely-not-here.mp4".to_string();

        let fps = vec![mock_fp_at(&keep_path, 60.0), mock_fp_at(&missing, 10.0)];
        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();

        output_results(
            &groups, &fps, &no_matches(), None, 0, Priority::Length, true, true, &stats,
        ).unwrap();

        assert_eq!(stats.delete_failed.count(), 1, "the failure must be tallied");
        assert!(stats.had_problems(), "a failed deletion must fail the run");
        assert!(Path::new(&keep_path).exists(), "the KEEP pick is untouched either way");
    }
}