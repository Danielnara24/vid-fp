mod compare;
mod fingerprint;

use clap::Parser;
use compare::find_all_matches;
use fingerprint::{fingerprint_video, VideoFingerprint};
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::io::Write;
use std::time::Instant;
use walkdir::WalkDir;

// Define the CLI arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Folder path to scan for videos
    folder_path: String,

    /// Maximum Hamming distance
    #[arg(short = 'd', long = "hamming-distance", default_value_t = 6)]
    hamming_distance: u32,

    /// Minimum match percentage (e.g., 15 for 15%)
    #[arg(short = 'p', long = "match-percent", default_value_t = 10.0)]
    match_percent: f32,

    /// Output file for the results (supports .txt, .csv, .json)
    #[arg(short = 'o', long = "output")]
    output: Option<String>,
}

// Graph Traversal with Pivoting logic to handle aggressive densifying cliques flawlessly
fn bron_kerbosch(
    r: HashSet<usize>,
    mut p: HashSet<usize>,
    mut x: HashSet<usize>,
    adjacency: &[HashSet<usize>],
    base_cliques: &mut Vec<HashSet<usize>>,
) {
    if p.is_empty() && x.is_empty() {
        if r.len() > 1 {
            base_cliques.push(r);
        }
        return;
    }

    // Heuristic: Use Pivot methodology choosing largest neighbor intersection maximizing node exclusion
    let pivot = p.union(&x).max_by_key(|&&v| adjacency[v].intersection(&p).count()).cloned();
    
    let p_explore: Vec<usize> = if let Some(u) = pivot {
        p.difference(&adjacency[u]).cloned().collect()
    } else {
        p.iter().cloned().collect()
    };

    for v in p_explore {
        let mut new_r = r.clone();
        new_r.insert(v);

        let neighbors = &adjacency[v];
        let new_p: HashSet<usize> = neighbors.intersection(&p).cloned().collect();
        let new_x: HashSet<usize> = neighbors.intersection(&x).cloned().collect();

        bron_kerbosch(new_r, new_p, new_x, adjacency, base_cliques);

        p.remove(&v);
        x.insert(v);
    }
}

// Helpers for formatted printing
fn format_size(bytes: u64) -> String {
    let b = bytes as f64;
    // Rust's formatting inherently uses a dot (.) for decimals
    if b >= 1_073_741_824.0 {
        format!("{:.1}GB", b / 1_073_741_824.0)
    } else if b >= 1_048_576.0 {
        format!("{:.1}MB", b / 1_048_576.0)
    } else if b >= 1024.0 {
        format!("{:.1}KB", b / 1024.0)
    } else {
        format!("{}B", bytes)
    }
}

fn format_duration(seconds: f64) -> String {
    let total_secs = seconds.round() as u64;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}

fn main() {
    // Start tracking time immediately upon execution
    let start_time = Instant::now();

    // Parse command line arguments
    let args = Args::parse();

    ffmpeg_next::init().expect("Failed to initialize FFmpeg bindings.");
    ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

    let folder_path = args.folder_path;
    
    // Initialize embedded Sled Database Cache 
    let db = sled::open("video_hashes.db").expect("Failed to open cache database");
    
    // Extract parameters and convert percentage to float (15.0 -> 0.15)
    let max_hamming = args.hamming_distance;
    let min_match_pct = args.match_percent / 100.0;

    let extensions = ["mp4", "mkv", "avi", "mov", "flv", "webm"];
    let mut video_files = Vec::new();

    println!("Scanning folder recursively: {}", folder_path);
    println!("Settings -> Max Hamming: {}, Min Match: {}%", max_hamming, args.match_percent);

    for entry in WalkDir::new(&folder_path).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if extensions.contains(&ext.to_lowercase().as_str()) {
                    video_files.push(path.to_string_lossy().to_string());
                }
            }
        }
    }

    if video_files.is_empty() { 
        println!("No videos found.");
        return; 
    }
    
    let total_videos = video_files.len();
    println!("Found {} video files. Starting parallel fingerprinting...", total_videos);

    let processed_count = AtomicUsize::new(0);

    // Helper closure to format and cleanly overwrite the single line progress bar output
    let print_progress = |status: &str, done: usize, total: usize, start: Instant, vf: &str| {
        let pct = (done as f64 / total as f64) * 100.0;
        let elapsed = start.elapsed().as_secs();
        let hours = elapsed / 3600;
        let mins = (elapsed % 3600) / 60;
        let secs = elapsed % 60;
        let filename = Path::new(vf).file_name().unwrap_or_default().to_string_lossy();

        let mut stdout = std::io::stdout().lock();
        let _ = write!(
            stdout,
            "\x1B[2K\r[{}] {}/{} [{:.1}%] - Time elapsed: {:02}:{:02}:{:02} - {}",
            status, done, total, pct, hours, mins, secs, filename
        );
        let _ = stdout.flush();
    };
    
    // Process with File Size + Modification Time cache logic
    let fingerprints: Vec<VideoFingerprint> = video_files
        .par_iter()
        .filter_map(|vf| {
            let metadata = std::fs::metadata(vf).ok()?;
            let mtime = metadata.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_secs();
            let size = metadata.len();
            
            let cache_key = format!("{}_{}_{}", vf, mtime, size);

            // Fetch from Cache
            if let Ok(Some(data)) = db.get(&cache_key) {
                if let Ok(fp) = bincode::deserialize::<VideoFingerprint>(&data) {
                    let done = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                    print_progress("Cached", done, total_videos, start_time, vf);
                    return Some(fp);
                }
            }

            // Fingerprint natively and save to Sled Database Cache
            let fp = fingerprint_video(vf)?;
            
            if let Ok(encoded) = bincode::serialize(&fp) {
                let _ = db.insert(&cache_key, encoded);
            }

            let done = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
            print_progress("Processing", done, total_videos, start_time, vf);
            Some(fp)
        })
        .collect();

    println!(); // Move to a new line once the single-line progress iteration is complete
    let _ = db.flush();

    let n = fingerprints.len();
    if n < 2 {
        println!("Not enough valid videos to compare.");
        return;
    }

    println!("\nFingerprinting complete. Cross-analyzing {} videos...", n);

    // 2. Global Multi-Index Pair Analysis mapped in Parallel completely bypassing nested loop O(N*N) traps
    let mut adjacency = vec![HashSet::new(); n];
    let edges = find_all_matches(&fingerprints, max_hamming, min_match_pct);

    for (i, j) in edges {
        adjacency[i].insert(j);
        adjacency[j].insert(i);
    }

    println!("Grouping duplicate clusters...");
    let mut base_cliques = Vec::new();
    let all_nodes: HashSet<usize> = (0..n).collect();
    
    bron_kerbosch(HashSet::new(), all_nodes.clone(), HashSet::new(), &adjacency, &mut base_cliques);

    let mut expanded_groups = Vec::new();
    for clique in base_cliques {
        let mut group = clique.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for v in 0..n {
                if !group.contains(&v) {
                    if adjacency[v].intersection(&group).count() >= 2 {
                        group.insert(v);
                        changed = true;
                    }
                }
            }
        }
        expanded_groups.push(group);
    }

    expanded_groups.sort_by(|a, b| b.len().cmp(&a.len()));

    let mut final_groups_sets: Vec<HashSet<usize>> = Vec::new();
    for g in expanded_groups {
        let mut is_subset = false;
        for fg in &final_groups_sets {
            if g.is_subset(fg) {
                is_subset = true;
                break;
            }
        }
        if !is_subset {
            final_groups_sets.push(g);
        }
    }

    // Keep indices mapped directly to the original struct to print properties naturally
    let mut final_groups: Vec<Vec<usize>> = Vec::new();
    for g in final_groups_sets {
        let mut group_indices: Vec<usize> = g.into_iter().collect();
        group_indices.sort_by(|&a, &b| fingerprints[a].path.cmp(&fingerprints[b].path));
        final_groups.push(group_indices);
    }

    final_groups.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| fingerprints[a[0]].path.cmp(&fingerprints[b[0]].path)));

    println!("\n========================================");
    println!("             RESULTS");
    println!("========================================\n");

    let mut total_files_linked = 0;

    // Output builders to support file saving options (.txt, .csv, .json)
    let mut txt_out = String::new();
    // Switched to semicolon strictly to bypass any commas from paths or sizing locational defaults
    let mut csv_out = String::from("group;resolution;size;length;full_path\n"); 
    let mut json_out_groups = Vec::new();

    for (i, group) in final_groups.iter().enumerate() {
        let group_name = format!("group_{}", i + 1);
        
        println!("{}:", group_name);
        txt_out.push_str(&format!("{}:\n", group_name));
        
        total_files_linked += group.len();
        
        let mut json_files = Vec::new();

        for &idx in group {
            let fp = &fingerprints[idx];
            let size_str = format_size(fp.file_size);
            let duration_str = format_duration(fp.duration);
            let res_str = format!("{}x{}", fp.width, fp.height);
            
            // 1. Console / Text Output
            println!("\t{}, {}, {}, {}", res_str, size_str, duration_str, fp.path);
            txt_out.push_str(&format!("\t{}, {}, {}, {}\n", res_str, size_str, duration_str, fp.path));
            
            // 2. CSV Output (Semicolon Delimiter)
            let escaped_path = fp.path.replace('"', "\"\"");
            csv_out.push_str(&format!("{};{};{};{};\"{}\"\n", group_name, res_str, size_str, duration_str, escaped_path));
            
            // 3. JSON File Output
            json_files.push(serde_json::json!({
                "resolution": res_str,
                "size": size_str,
                "length": duration_str,
                "full_path": fp.path,
            }));
        }
        
        println!();
        txt_out.push_str("\n");
        
        json_out_groups.push(serde_json::json!({
            "group": group_name,
            "files": json_files
        }));
    }

    let total_elapsed = start_time.elapsed().as_secs();
    let total_hours = total_elapsed / 3600;
    let total_mins = (total_elapsed % 3600) / 60;
    let total_secs = total_elapsed % 60;

    let summary = format!("Total groups found: {}\nTotal files linked: {}\nTotal time elapsed: {:02}:{:02}:{:02}", 
        final_groups.len(), total_files_linked, total_hours, total_mins, total_secs);
    
    println!("{}", summary);

    // Save to the specified output file if requested by the user
    if let Some(out_path) = args.output {
        let path = Path::new(&out_path);
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        
        let write_result = match ext.as_str() {
            "csv" => std::fs::write(path, csv_out),
            "json" => {
                let json_final = serde_json::json!({
                    "summary": {
                        "total_groups": final_groups.len(),
                        "total_files_linked": total_files_linked,
                        "time_elapsed_seconds": total_elapsed,
                    },
                    "results": json_out_groups
                });
                std::fs::write(path, serde_json::to_string_pretty(&json_final).unwrap())
            },
            _ => {
                // Default to .txt format if nothing specific (.txt or random extension)
                let mut full_txt = String::new();
                full_txt.push_str(&txt_out);
                full_txt.push_str(&summary);
                full_txt.push_str("\n");
                
                std::fs::write(path, full_txt)
            }
        };

        match write_result {
            Ok(_) => println!("\nResults successfully saved to {}", out_path),
            Err(e) => eprintln!("\nError saving results to {}: {}", out_path, e),
        }
    }
}