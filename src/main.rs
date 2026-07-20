mod clustering;
mod compare;
mod export;
mod fingerprint;
mod utils;

use clap::Parser;
use compare::find_all_matches;
use fingerprint::{fingerprint_video, VideoFingerprint};
use rayon::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
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

    /// Suppress all terminal output except errors
    #[arg(short = 's', long = "silent")]
    silent: bool,
}

fn main() {
    let start_time = Instant::now();
    let args = Args::parse();
    let silent = args.silent;

    ffmpeg_next::init().expect("Failed to initialize FFmpeg bindings.");
    ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

    // Follow XDG Base Directory Specification for caching on Linux
    let cache_dir = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home).join(".cache")
            } else {
                PathBuf::from("/tmp")
            }
        })
        .join("video-dedup");

    std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");
    let db_path = cache_dir.join("video_hashes.db");

    let db = sled::open(&db_path).expect("Failed to open cache database");

    let max_hamming = args.hamming_distance;
    let min_match_pct = args.match_percent / 100.0;

    let extensions = ["mp4", "mkv", "avi", "mov", "flv", "webm"];
    let mut video_files = Vec::new();

    if !silent {
        println!("Scanning folder recursively: {}", args.folder_path);
        println!(
            "Settings -> Max Hamming: {}, Min Match: {}%",
            max_hamming, args.match_percent
        );
        println!("Using Cache Directory: {}", cache_dir.display());
    }

    for entry in WalkDir::new(&args.folder_path).into_iter().filter_map(|e| e.ok()) {
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
        if !silent {
            println!("No videos found.");
        }
        return;
    }

    let total_videos = video_files.len();
    if !silent {
        println!("Found {} video files. Fingerprinting...", total_videos);
    }

    let processed_count = AtomicUsize::new(0);

    let print_progress = |status: &str, done: usize, total: usize, start: Instant, vf: &str| {
        if silent {
            return;
        }
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

    let fingerprints: Vec<VideoFingerprint> = video_files
        .par_iter()
        .filter_map(|vf| {
            let metadata = std::fs::metadata(vf).ok()?;
            let mtime = metadata
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let size = metadata.len();

            let cache_key = format!("{}_{}_{}", vf, mtime, size);

            if let Ok(Some(data)) = db.get(&cache_key) {
                if let Ok(fp) = bincode::deserialize::<VideoFingerprint>(&data) {
                    let done = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                    print_progress("Cached", done, total_videos, start_time, vf);
                    return Some(fp);
                }
            }

            let fp = fingerprint_video(vf)?;

            if let Ok(encoded) = bincode::serialize(&fp) {
                let _ = db.insert(&cache_key, encoded);
            }

            let done = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
            print_progress("Processing", done, total_videos, start_time, vf);
            Some(fp)
        })
        .collect();

    if !silent {
        println!();
    }
    let _ = db.flush();

    let n = fingerprints.len();
    if n < 2 {
        if !silent {
            println!("Not enough valid videos to compare.");
        }
        return;
    }

    if !silent {
        println!(
            "\nFingerprinting complete. Cross-analyzing {} videos...",
            n
        );
    }

    // Process logic is now cleanly offloaded
    let edges = find_all_matches(&fingerprints, max_hamming, min_match_pct);
    
    if !silent {
        println!("Grouping duplicate clusters...");
    }
    
    let final_groups = clustering::find_duplicate_groups(n, edges, &fingerprints);

    export::output_results(
        &final_groups,
        &fingerprints,
        silent,
        args.output.as_ref(),
        start_time.elapsed().as_secs(),
    );
}