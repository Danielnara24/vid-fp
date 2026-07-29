use anyhow::{anyhow, Context, Result};
use log::info;
use crate::fingerprint::VideoFingerprint;
use crate::stats::RunStats;
use crate::utils::{
    find_best, format_bitrate, format_duration, format_size, shutdown_requested, GroupMaxima, Priority,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Metrics that can justify a REVIEW flag, in default precedence order. Size is
/// absent on purpose: it carries no quality information that bitrate and length
/// don't already express (size IS bitrate x length), so flagging on it would
/// only ever repeat a flag one of those two already raised.
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

        // If the KEEP pick isn't top-tier on some quality metric, surface the
        // file that IS as worth a manual look. Metrics are checked in default
        // precedence order, skipping the one the user prioritised (KEEP wins
        // that by construction). Laziness matters here: find_best only runs for
        // the first metric KEEP actually falls short on.
        let review_idx = REVIEW_METRICS
            .iter()
            .copied()
            .filter(|&m| m != priority && maxima.tier(keep_fp, m) == 0)
            .map(|m| find_best(group, fingerprints, m, &maxima))
            .find(|&candidate| candidate != keep_idx);

        if let Some(r) = review_idx {
            review_set.insert(r);
        }
        // Everything in the group that isn't this group's KEEP or REVIEW is a
        // delete candidate. DELETE wins over KEEP globally, so we don't care
        // whether the file is a KEEP pick in some other group.
        for &idx in group {
            if idx != keep_idx && Some(idx) != review_idx {
                delete_candidates.insert(idx);
            }
        }
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

    let mut total_files_linked = 0;

    let mut txt_out = String::new();
    let mut json_out_groups = Vec::new();

    // Use csv crate for robust and RFC-compliant CSV generation
    let mut csv_wtr = csv::WriterBuilder::new()
        .delimiter(b';')
        .from_writer(Vec::new());

    csv_wtr
        .write_record(&["group", "resolution", "size", "bitrate", "length", "full_path", "action"])
        .context("Failed to write CSV header")?;

    for (i, group) in final_groups.iter().enumerate() {
        let group_name = format!("group_{}", i + 1);

        info!("{}:", group_name);
        txt_out.push_str(&format!("{}:\n", group_name));

        total_files_linked += group.len();
        let mut json_files = Vec::new();

        for &idx in group {
            let fp = &fingerprints[idx];
            let size_str = format_size(fp.file_size);
            let bitrate_str = format_bitrate(fp.bitrate());
            let duration_str = format_duration(fp.duration);
            let res_str = format!("{}x{}", fp.width, fp.height);

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
            info!(
                "\t{}, {}, {}, {}, {}, {}",
                res_str, size_str, bitrate_str, duration_str, fp.path, action_str
            );
            txt_out.push_str(&format!(
                "\t{}, {}, {}, {}, {}, {}\n",
                res_str, size_str, bitrate_str, duration_str, fp.path, action_str
            ));

            // 2. CSV Output
            csv_wtr.write_record(&[
                &group_name,
                &res_str,
                &size_str,
                &bitrate_str,
                &duration_str,
                &fp.path,
                action_str,
            ]).context("Failed to write CSV record")?;

            // 3. JSON File Output
            json_files.push(serde_json::json!({
                "resolution": res_str,
                "size": size_str,
                "bitrate": bitrate_str,
                "bitrate_bps": fp.bitrate(),
                "length": duration_str,
                "full_path": fp.path,
                "action": action_str,
            }));
        }

        info!(""); // Empty line for spacing
        txt_out.push_str("\n");

        json_out_groups.push(serde_json::json!({
            "group": group_name,
            "files": json_files
        }));
    }

    let total_hours = total_elapsed_secs / 3600;
    let total_mins = (total_elapsed_secs % 3600) / 60;
    let total_secs = total_elapsed_secs % 60;

    let mut summary = format!(
        "Total groups found: {}\nTotal files linked: {}\nTotal time elapsed: {:02}:{:02}:{:02}",
        final_groups.len(),
        total_files_linked,
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
                        "total_files_linked": total_files_linked,
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
    use crate::utils::Priority;
    use tempfile::NamedTempFile;
    use std::fs;

    fn mock_fp() -> VideoFingerprint {
        VideoFingerprint {
            path: "/fake/path/vid.mp4".to_string(),
            valid_hashes: vec![], valid_t_start: vec![], valid_t_end: vec![],
            total_frames: 100, width: 1920, height: 1080, duration: 60.0, file_size: 1048576,
        }
    }

    /// A mock whose size scales with duration, holding bitrate at exactly
    /// 1 Mbps. This matters now that bitrate is derived: give a 10s file the
    /// same byte count as a 90s one and it genuinely IS a 9x higher-bitrate
    /// copy, which the REVIEW rule will correctly flag and protect from
    /// deletion. These tests are about the KEEP/DELETE precedence logic, so
    /// they hold every other variable still.
    fn mock_fp_at(path: &str, duration: f64) -> VideoFingerprint {
        let mut fp = mock_fp();
        fp.path = path.to_string();
        fp.duration = duration;
        fp.file_size = (duration * 131_072.0) as u64; // 1 Mbps
        fp
    }

    #[test]
    fn test_csv_output() {
        let fps = vec![mock_fp()];
        let groups = vec![vec![0]];

        let temp_file = NamedTempFile::new().unwrap();
        let path_str = temp_file.path().with_extension("csv").to_string_lossy().to_string();

        // Report-only run: single item defaults to KEEP.
        output_results(
            &groups, &fps, Some(&path_str), 120, Priority::Length, false, false,
            &RunStats::default(),
        ).unwrap();

        let contents = fs::read_to_string(&path_str).unwrap();

        // Assert headers exist (separated by semicolons)
        assert!(contents.contains("group;resolution;size;bitrate;length;full_path;action"));
        // 1 MiB over 60s = ~140kbps. Assert data exists and defaults to KEEP.
        assert!(contents.contains(
            "group_1;1920x1080;1.0MB;140kbps;00:01:00;/fake/path/vid.mp4;KEEP"
        ));

        // Clean up
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

        output_results(&groups, &fps, None, 0, Priority::Length, true, true, &stats).unwrap();

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
            &groups, &fps, None, 0, Priority::Length, true, true, &RunStats::default(),
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
        output_results(&groups, &fps, None, 0, Priority::Length, true, true, &stats).unwrap();

        assert!(Path::new(&p0).exists(), "independent best A must remain");
        assert!(Path::new(&p1).exists(), "independent best B must remain");
        assert!(!Path::new(&p2).exists(), "shared duplicate must be removed exactly once");
        assert_eq!(
            stats.delete_failed.count(), 0,
            "a target queued by two groups must not report a second, failing attempt"
        );
    }

    #[test]
    fn test_bitrate_settles_groups_that_tie_on_length_and_resolution() {
        // Same length, same resolution: under the default order the decision
        // reaches bitrate, and the denser copy is kept. Nothing is flagged
        // REVIEW, because the KEEP pick is top-tier on every metric.
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
            &groups, &fps, None, 0, Priority::Length, true, true, &RunStats::default(),
        ).unwrap();

        assert!(Path::new(&p_hi).exists(), "higher-bitrate copy must be kept");
        assert!(!Path::new(&p_lo).exists(), "lower-bitrate copy must be deleted");
    }

    #[test]
    fn test_failed_deletion_is_recorded_for_the_summary_and_exit_code() {
        let keep_file = NamedTempFile::new().unwrap();
        let keep_path = keep_file.path().to_string_lossy().to_string();
        let missing = "/nonexistent/vid-fp/definitely-not-here.mp4".to_string();

        let fps = vec![mock_fp_at(&keep_path, 60.0), mock_fp_at(&missing, 10.0)];
        let groups = vec![vec![0, 1]];
        let stats = RunStats::default();

        output_results(&groups, &fps, None, 0, Priority::Length, true, true, &stats).unwrap();

        assert_eq!(stats.delete_failed.count(), 1, "the failure must be tallied");
        assert!(stats.had_problems(), "a failed deletion must fail the run");
        assert!(Path::new(&keep_path).exists(), "the KEEP pick is untouched either way");
    }
}