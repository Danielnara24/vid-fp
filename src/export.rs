use anyhow::{Context, Result};
use log::info;
use crate::fingerprint::VideoFingerprint;
use crate::utils::{format_duration, format_size, find_best, Priority};
use std::path::Path;

pub fn output_results(
    final_groups: &[Vec<usize>],
    fingerprints: &[VideoFingerprint],
    output_file: Option<&String>,
    total_elapsed_secs: u64,
    priority: Priority,
) -> Result<()> {
    
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
        
    csv_wtr.write_record(&["group", "resolution", "size", "length", "full_path", "action"])
        .context("Failed to write CSV header")?;

    for (i, group) in final_groups.iter().enumerate() {
        let group_name = format!("group_{}", i + 1);

        // 1. Determine KEEP and REVIEW logic
        let max_dur = group.iter().map(|&idx| fingerprints[idx].duration).fold(0.0, f64::max);
        let max_res = group.iter().map(|&idx| fingerprints[idx].width * fingerprints[idx].height).max().unwrap_or(0);

        let keep_idx = find_best(group, fingerprints, priority, max_dur);
        let keep_fp = &fingerprints[keep_idx];
        let keep_res = keep_fp.width * keep_fp.height;

        let mut review_idx = None;

        match priority {
            Priority::Length => {
                // If KEEP isn't the absolute max resolution, find the best Res file for REVIEW
                if keep_res < max_res {
                    let candidate = find_best(group, fingerprints, Priority::Resolution, max_dur);
                    if candidate != keep_idx { review_idx = Some(candidate); }
                }
            },
            Priority::Resolution => {
                // If KEEP isn't close to absolute max length, find the best Length file for REVIEW
                if keep_fp.duration < max_dur - 0.5 {
                    let candidate = find_best(group, fingerprints, Priority::Length, max_dur);
                    if candidate != keep_idx { review_idx = Some(candidate); }
                }
            },
            Priority::Size => {
                // Recommend length if lacking, else recommend resolution if lacking
                if keep_fp.duration < max_dur - 0.5 {
                    let candidate = find_best(group, fingerprints, Priority::Length, max_dur);
                    if candidate != keep_idx { review_idx = Some(candidate); }
                } else if keep_res < max_res {
                    let candidate = find_best(group, fingerprints, Priority::Resolution, max_dur);
                    if candidate != keep_idx { review_idx = Some(candidate); }
                }
            }
        }

        info!("{}:", group_name);
        txt_out.push_str(&format!("{}:\n", group_name));

        total_files_linked += group.len();
        let mut json_files = Vec::new();

        for &idx in group {
            let fp = &fingerprints[idx];
            let size_str = format_size(fp.file_size);
            let duration_str = format_duration(fp.duration);
            let res_str = format!("{}x{}", fp.width, fp.height);
            
            let action_str = if idx == keep_idx {
                "KEEP"
            } else if Some(idx) == review_idx {
                "REVIEW"
            } else {
                "DELETE"
            };

            // 1. Console / Text Output
            info!("\t{}, {}, {}, {}, {}", res_str, size_str, duration_str, fp.path, action_str);
            txt_out.push_str(&format!(
                "\t{}, {}, {}, {}, {}\n",
                res_str, size_str, duration_str, fp.path, action_str
            ));

            // 2. CSV Output
            csv_wtr.write_record(&[
                &group_name,
                &res_str,
                &size_str,
                &duration_str,
                &fp.path,
                action_str,
            ]).context("Failed to write CSV record")?;

            // 3. JSON File Output
            json_files.push(serde_json::json!({
                "resolution": res_str,
                "size": size_str,
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

    let summary = format!(
        "Total groups found: {}\nTotal files linked: {}\nTotal time elapsed: {:02}:{:02}:{:02}",
        final_groups.len(),
        total_files_linked,
        total_hours,
        total_mins,
        total_secs
    );

    info!("{}", summary);

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

    #[test]
    fn test_csv_output() {
        let fps = vec![mock_fp()];
        let groups = vec![vec![0]];
        
        let temp_file = NamedTempFile::new().unwrap();
        let path_str = temp_file.path().with_extension("csv").to_string_lossy().to_string();

        output_results(&groups, &fps, Some(&path_str), 120, Priority::Length).unwrap();

        let contents = fs::read_to_string(&path_str).unwrap();
        
        // Assert headers exist (separated by semicolons)
        assert!(contents.contains("group;resolution;size;length;full_path;action"));
        // Assert data exists and defaults single item to KEEP
        assert!(contents.contains("group_1;1920x1080;1.0MB;00:01:00;/fake/path/vid.mp4;KEEP"));
        
        // Clean up
        let _ = fs::remove_file(path_str);
    }
}