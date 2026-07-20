mod clustering;
mod compare;
mod export;
mod fingerprint;
mod utils;

use anyhow::{Context, Result};
use clap::Parser;
use compare::find_all_matches;
use fingerprint::{fingerprint_video, VideoFingerprint};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

// Cache schema versioning - bump this if VideoFingerprint struct ever changes!
const CACHE_VERSION: &str = "v1";

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

    /// Delete ALL cache before running
    #[arg(short = 'C', long = "clear-cache")]
    clear_cache: bool,

    /// Delete the cache of files not included in the current folder to scan
    #[arg(short = 'P', long = "prune-cache")]
    prune_cache: bool,

    /// Suppress all terminal output except errors
    #[arg(short = 's', long = "silent")]
    silent: bool,
}

fn main() -> Result<()> {
    let start_time = Instant::now();
    let args = Args::parse();

    // 1. Initialize custom CLI Logger
    // This removes the need for `if !silent` everywhere. If silent, only Errors print.
    let log_level = if args.silent {
        log::LevelFilter::Error
    } else {
        log::LevelFilter::Info
    };

    env_logger::Builder::new()
        .filter_level(log_level)
        .format(|buf, record| {
            if record.level() == log::Level::Error {
                writeln!(buf, "Error: {}", record.args())
            } else {
                writeln!(buf, "{}", record.args()) // Clean output for CLI tools
            }
        })
        .init();

    ffmpeg_next::init().context("Failed to initialize FFmpeg bindings.")?;
    ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);

    // Follow XDG Base Directory Specification for caching
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

    std::fs::create_dir_all(&cache_dir).context("Failed to create cache directory")?;
    let db_path = cache_dir.join("video_hashes.db");

    let db = sled::open(&db_path).context("Failed to open or lock cache database")?;

    if args.clear_cache {
        info!("Clearing all cache...");
        db.clear().context("Failed to clear cache database")?;
    }

    let max_hamming = args.hamming_distance;
    let min_match_pct = args.match_percent / 100.0;

    let extensions = ["mp4", "mkv", "avi", "mov", "flv", "webm"];
    let mut video_files = Vec::new();

    info!("Scanning folder recursively: {}", args.folder_path);
    info!(
        "Settings -> Max Hamming: {}, Min Match: {}%",
        max_hamming, args.match_percent
    );
    info!("Using Cache Directory: {}", cache_dir.display());

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

    if args.prune_cache && !args.clear_cache {
        info!("Pruning cache for files not in the current scan...");
        let valid_files: HashSet<&str> = video_files.iter().map(|s| s.as_str()).collect();
        let mut batch = sled::Batch::default();
        let mut pruned_count = 0;

        for kv in db.iter() {
            if let Ok((key_bytes, _)) = kv {
                let mut should_remove = true;
                if let Ok(key_str) = std::str::from_utf8(&key_bytes) {
                    // Extract filepath from cache key (format: filepath_mtime_size_version)
                    let mut parts = key_str.rsplitn(4, '_');
                    let _version = parts.next();
                    let _size = parts.next();
                    let _mtime = parts.next();
                    if let Some(filepath) = parts.next() {
                        if valid_files.contains(filepath) {
                            should_remove = false; // It's still valid, keep it
                        }
                    }
                }
                
                if should_remove {
                    batch.remove(key_bytes);
                    pruned_count += 1;
                }
            }
        }

        if pruned_count > 0 {
            db.apply_batch(batch).context("Failed to apply cache pruning")?;
            info!("Pruned {} stale entries from cache.", pruned_count);
        } else {
            info!("No stale entries found to prune.");
        }
    }

    if video_files.is_empty() {
        info!("No videos found.");
        return Ok(());
    }

    let total_videos = video_files.len();
    info!("Found {} video files. Fingerprinting...", total_videos);

    // 2. Setup robust Thread-Safe Progress Bar
    let pb = if args.silent {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(total_videos as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - {msg}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb
    };

    // Thread-safe Sled Batching 
    let batch_lock = std::sync::Mutex::new((sled::Batch::default(), 0));
    const BATCH_SIZE: usize = 32; // Flush to disk after every 32 writes

    let fingerprints: Vec<VideoFingerprint> = video_files
        .par_iter()
        .filter_map(|vf| {
            let file_name = Path::new(vf).file_name().unwrap_or_default().to_string_lossy().into_owned();
            pb.set_message(file_name);

            let metadata = match std::fs::metadata(vf) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("Cannot access metadata for {}: {}", vf, e);
                    pb.inc(1);
                    return None;
                }
            };

            let mtime = metadata
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let size = metadata.len();

            // Versioned cache key ensures schema changes don't cause deserialization crashes
            let cache_key = format!("{}_{}_{}_{}", vf, mtime, size, CACHE_VERSION);

            if let Ok(Some(data)) = db.get(&cache_key) {
                match bincode::deserialize::<VideoFingerprint>(&data) {
                    Ok(fp) => {
                        pb.inc(1);
                        return Some(fp);
                    }
                    Err(e) => {
                        // Corrupted or outdated schema; ignore and overwrite below
                        log::debug!("Cache deserialization failed for {}: {}. Re-processing.", vf, e);
                    }
                }
            }

            let fp = match fingerprint_video(vf) {
                Ok(f) => f,
                Err(e) => {
                    log::error!("Failed to process {}: {:#}", vf, e);
                    pb.inc(1); // Increment progress bar so it completes properly
                    return None;
                }
            };

            // Batch database insertions to heavily reduce Disk I/O bottlenecks
            if let Ok(encoded) = bincode::serialize(&fp) {
                let mut b = batch_lock.lock().unwrap();
                b.0.insert(cache_key.as_bytes(), encoded);
                b.1 += 1;
                
                if b.1 >= BATCH_SIZE {
                    let current_batch = std::mem::take(&mut b.0);
                    b.1 = 0;
                    
                    // Release the lock BEFORE writing so other threads can keep appending to the new empty batch
                    drop(b);
                    let _ = db.apply_batch(current_batch);
                }
            }

            pb.inc(1);
            Some(fp)
        })
        .collect();

    pb.finish_and_clear();
    
    // Apply any remaining queued database items that haven't hit the size trigger
    if let Ok(b) = batch_lock.into_inner() {
        if b.1 > 0 {
            let _ = db.apply_batch(b.0);
        }
    }
    
    let _ = db.flush();

    let n = fingerprints.len();
    if n < 2 {
        info!("Not enough valid videos to compare.");
        return Ok(());
    }

    info!("\nFingerprinting complete. Cross-analyzing {} videos...", n);

    let edges = find_all_matches(&fingerprints, max_hamming, min_match_pct);
    
    info!("Grouping duplicate clusters...");
    
    let final_groups = clustering::find_duplicate_groups(n, edges, &fingerprints);

    export::output_results(
        &final_groups,
        &fingerprints,
        args.output.as_ref(),
        start_time.elapsed().as_secs(),
    )?;

    Ok(())
}