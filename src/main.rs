mod compare;
mod fingerprint;

use compare::compare_videos;
use fingerprint::{fingerprint_video, VideoFingerprint};
use std::collections::HashSet;
use std::env;
use std::path::Path;
use walkdir::WalkDir;

// Exact translation of the Python bron_kerbosch function
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

    let p_clone = p.clone();
    for v in p_clone {
        let mut new_r = r.clone();
        new_r.insert(v);

        let neighbors = &adjacency[v];
        let new_p: HashSet<usize> = p.intersection(neighbors).cloned().collect();
        let new_x: HashSet<usize> = x.intersection(neighbors).cloned().collect();

        bron_kerbosch(new_r, new_p, new_x, adjacency, base_cliques);

        p.remove(&v);
        x.insert(v);
    }
}

fn main() {
    // Get folder from command line argument, or use a default if not provided
    let folder_path = env::args().nth(1).expect("Please provide a folder path! (e.g., cargo run --release -- /path/to/folder)");
    
    let max_hamming = 5;
    let min_match_pct = 0.15;

    let extensions = ["mp4", "mkv", "avi", "mov", "flv", "webm"];
    let mut video_files = Vec::new();

    println!("Scanning folder recursively: {}", folder_path);

    // 1. Find all videos recursively
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
        println!("No video files found.");
        return;
    }
    println!("Found {} video files. Starting fingerprinting...", video_files.len());

    // 2. Fingerprint all videos
    let mut fingerprints: Vec<VideoFingerprint> = Vec::new();
    for vf in &video_files {
        if let Some(fp) = fingerprint_video(vf) {
            fingerprints.push(fp);
        }
    }

    let n = fingerprints.len();
    if n < 2 {
        println!("Not enough valid videos to compare.");
        return;
    }

    println!("\nFingerprinting complete. Cross-analyzing {} videos...", n);

    // 3. Compare all pairs (N * (N-1) / 2) and build adjacency list
    let mut adjacency = vec![HashSet::new(); n];
    
    for i in 0..n {
        for j in (i + 1)..n {
            if compare_videos(&fingerprints[i], &fingerprints[j], max_hamming, min_match_pct).is_some() {
                adjacency[i].insert(j);
                adjacency[j].insert(i);
            }
        }
    }

    // 4. Find Duplicate Clusters (Bron-Kerbosch)
    println!("Grouping duplicate clusters...");
    let mut base_cliques = Vec::new();
    let all_nodes: HashSet<usize> = (0..n).collect();
    
    bron_kerbosch(HashSet::new(), all_nodes.clone(), HashSet::new(), &adjacency, &mut base_cliques);

    // 5. Expand Groups (same as python while changed loop)
    let mut expanded_groups = Vec::new();
    for clique in base_cliques {
        let mut group = clique.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for v in 0..n {
                if !group.contains(&v) {
                    let intersect_count = adjacency[v].intersection(&group).count();
                    if intersect_count >= 2 {
                        group.insert(v);
                        changed = true;
                    }
                }
            }
        }
        expanded_groups.push(group);
    }

    // Sort by length descending to make subset filtering work like Python
    expanded_groups.sort_by(|a, b| b.len().cmp(&a.len()));

    // 6. Filter Subsets
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

    // Convert sets to sorted vectors of strings (full paths)
    let mut final_groups: Vec<Vec<String>> = Vec::new();
    for g in final_groups_sets {
        let mut group_paths: Vec<String> = g.into_iter().map(|idx| fingerprints[idx].path.clone()).collect();
        group_paths.sort(); // Sort alphabetically within the group
        final_groups.push(group_paths);
    }

    // Sort all groups: highest length first, then alphabetically by the first item
    final_groups.sort_by(|a, b| {
        b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0]))
    });

    // 7. Print Results in Requested Format
    println!("\n========================================");
    println!("             RESULTS");
    println!("========================================\n");

    let mut total_files_linked = 0;

    for (i, group) in final_groups.iter().enumerate() {
        println!("group{}:", i + 1);
        total_files_linked += group.len();
        
        for path_str in group {
            // Extract just the filename to match your requested print layout
            let filename = Path::new(path_str)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            println!("{}", filename);
        }
        println!(); // Blank line between groups
    }

    println!("Total groups found: {}", final_groups.len());
    println!("Total files linked: {}", total_files_linked);
}