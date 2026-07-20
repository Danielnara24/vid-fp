mod compare;
mod fingerprint;

use compare::find_all_matches;
use fingerprint::{fingerprint_video, VideoFingerprint};
use rayon::prelude::*;
use std::collections::HashSet;
use std::env;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::io::Write;
use std::time::Instant;
use walkdir::WalkDir;

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

fn main() {
    // Start tracking time immediately upon execution
    let start_time = Instant::now();

    ffmpeg_next::init().expect("Failed to initialize FFmpeg bindings.");
    ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

    let folder_path = env::args().nth(1).expect("Please provide a folder path!");
    
    // Initialize embedded Sled Database Cache 
    let db = sled::open("video_hashes.db").expect("Failed to open cache database");
    
    let max_hamming = 6;
    let min_match_pct = 0.10;

    let extensions = ["mp4", "mkv", "avi", "mov", "flv", "webm"];
    let mut video_files = Vec::new();

    println!("Scanning folder recursively: {}", folder_path);

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

    let mut final_groups: Vec<Vec<String>> = Vec::new();
    for g in final_groups_sets {
        let mut group_paths: Vec<String> = g.into_iter().map(|idx| fingerprints[idx].path.clone()).collect();
        group_paths.sort();
        final_groups.push(group_paths);
    }

    final_groups.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0])));

    println!("\n========================================");
    println!("             RESULTS");
    println!("========================================\n");

    let mut total_files_linked = 0;

    for (i, group) in final_groups.iter().enumerate() {
        println!("group{}:", i + 1);
        total_files_linked += group.len();
        
        for path_str in group {
            println!("{}", Path::new(path_str).file_name().unwrap_or_default().to_string_lossy());
        }
        println!();
    }

    let total_elapsed = start_time.elapsed().as_secs();
    let total_hours = total_elapsed / 3600;
    let total_mins = (total_elapsed % 3600) / 60;
    let total_secs = total_elapsed % 60;

    println!("Total groups found: {}", final_groups.len());
    println!("Total files linked: {}", total_files_linked);
    println!("Total time elapsed: {:02}:{:02}:{:02}", total_hours, total_mins, total_secs);
}