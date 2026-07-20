use anyhow::{Context, Result};
use log::info;
use crate::fingerprint::VideoFingerprint;
use crate::utils::{format_duration, format_size};
use std::path::Path;

pub fn output_results(
    final_groups: &[Vec<usize>],
    fingerprints: &[VideoFingerprint],
    output_file: Option<&String>,
    total_elapsed_secs: u64,
) -> Result<()> {
    
    info!("\n========================================");
    info!("             RESULTS");
    info!("========================================\n");

    let mut total_files_linked = 0;

    let mut txt_out = String::new();
    let mut csv_out = String::from("group;resolution;size;length;full_path\n");
    let mut json_out_groups = Vec::new();

    for (i, group) in final_groups.iter().enumerate() {
        let group_name = format!("group_{}", i + 1);

        info!("{}:", group_name);
        txt_out.push_str(&format!("{}:\n", group_name));

        total_files_linked += group.len();
        let mut json_files = Vec::new();

        for &idx in group {
            let fp = &fingerprints[idx];
            let size_str = format_size(fp.file_size);
            let duration_str = format_duration(fp.duration);
            let res_str = format!("{}x{}", fp.width, fp.height);

            // 1. Console / Text Output
            info!("\t{}, {}, {}, {}", res_str, size_str, duration_str, fp.path);
            txt_out.push_str(&format!(
                "\t{}, {}, {}, {}\n",
                res_str, size_str, duration_str, fp.path
            ));

            // 2. CSV Output
            let escaped_path = fp.path.replace('"', "\"\"");
            csv_out.push_str(&format!(
                "{};{};{};{};\"{}\"\n",
                group_name, res_str, size_str, duration_str, escaped_path
            ));

            // 3. JSON File Output
            json_files.push(serde_json::json!({
                "resolution": res_str,
                "size": size_str,
                "length": duration_str,
                "full_path": fp.path,
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
                std::fs::write(path, csv_out)
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