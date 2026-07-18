use std::process::Command;
use std::str;
use std::time::Instant;

pub struct VideoFingerprint {
    pub path: String,
    pub duration: f32,
    pub valid_hashes: Vec<u64>,
    pub valid_t_start: Vec<f32>,
    pub valid_t_end: Vec<f32>,
}

fn get_exact_duration(filepath: &str) -> f32 {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath])
        .output();

    if let Ok(out) = output {
        if let Ok(s) = str::from_utf8(&out.stdout) {
            if let Ok(duration) = s.trim().parse::<f32>() {
                return duration;
            }
        }
    }
    0.0
}

pub fn fingerprint_video(filepath: &str) -> Option<VideoFingerprint> {
    let start_time = Instant::now();
    let exact_duration = get_exact_duration(filepath);
    if exact_duration <= 0.0 {
        println!("Could not get duration for {}", filepath);
        return None;
    }

    // 1. FFmpeg Subprocess Extraction
    let output = Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "error",
            "-skip_frame", "nokey",
            "-threads", "1",
            "-i", filepath,
            "-map", "0:v:0",
            "-an", "-sn",
            "-vf", "scale=64:64:flags=fast_bilinear",
            "-avoid_negative_ts", "make_zero",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"
        ])
        .output()
        .ok()?;

    let raw_bytes = output.stdout;
    let frame_size = 64 * 64;
    let total_frames = raw_bytes.len() / frame_size;
    
    if total_frames == 0 { return None; }
    println!("[{}] Extracted {} keyframes.", filepath, total_frames);

    // 2. Filter duplicate adjacent frames (np.where(change_mask))
    let mut u_frames = Vec::new();
    let mut unique_frame_indices = Vec::new();
    
    u_frames.push(raw_bytes[0..frame_size].to_vec());
    unique_frame_indices.push(0);

    for i in 1..total_frames {
        let prev = &raw_bytes[(i - 1) * frame_size .. i * frame_size];
        let curr = &raw_bytes[i * frame_size .. (i + 1) * frame_size];
        if curr != prev {
            u_frames.push(curr.to_vec());
            unique_frame_indices.push(i);
        }
    }
    let n_unique = u_frames.len();
    println!("[{}] Filtered to {} unique dynamic frames.", filepath, n_unique);

    // 3. Calculate Variance for Auto-Cropping
    let mut row_max_var = vec![0.0f32; 64];
    let mut col_max_var = vec![0.0f32; 64];

    for y in 0..64 {
        for x in 0..64 {
            let mut sum = 0.0;
            for f in 0..n_unique {
                sum += u_frames[f][y * 64 + x] as f32;
            }
            let mean = sum / n_unique as f32;
            
            let mut var_sum = 0.0;
            for f in 0..n_unique {
                let diff = u_frames[f][y * 64 + x] as f32 - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / n_unique as f32;

            if variance > row_max_var[y] { row_max_var[y] = variance; }
            if variance > col_max_var[x] { col_max_var[x] = variance; }
        }
    }

    let var_threshold = 2.0;
    let valid_rows: Vec<usize> = (0..64).filter(|&i| row_max_var[i] > var_threshold).collect();
    let valid_cols: Vec<usize> = (0..64).filter(|&i| col_max_var[i] > var_threshold).collect();

    let mut y1 = 0; let mut y2 = 63;
    let mut x1 = 0; let mut x2 = 63;

    if !valid_rows.is_empty() && !valid_cols.is_empty() {
        let temp_y1 = *valid_rows.first().unwrap();
        let temp_y2 = *valid_rows.last().unwrap();
        let temp_x1 = *valid_cols.first().unwrap();
        let temp_x2 = *valid_cols.last().unwrap();

        if (temp_y2 - temp_y1 + 1) >= 16 && (temp_x2 - temp_x1 + 1) >= 16 {
            y1 = if temp_y1 > 0 { temp_y1 + 1 } else { temp_y1 };
            y2 = if temp_y2 < 63 { temp_y2 - 1 } else { temp_y2 };
            x1 = if temp_x1 > 0 { temp_x1 + 1 } else { temp_x1 };
            x2 = if temp_x2 < 63 { temp_x2 - 1 } else { temp_x2 };
        }
    }
    
    let crop_h = y2 - y1 + 1;
    let crop_w = x2 - x1 + 1;
    println!("[{}] Autocropped to {}x{} (y:{}-{}, x:{}-{})", filepath, crop_w, crop_h, y1, y2, x1, x2);

    // 4. Generate 8x9 Grids via Bilinear Interpolation & Pack into Hashes
    let mut hashes = Vec::new();
    
    for f in 0..n_unique {
        let frame = &u_frames[f];
        let mut frame_8x9 = vec![vec![0u8; 9]; 8];

        if crop_h != 8 || crop_w != 9 {
            for out_y in 0..8 {
                for out_x in 0..9 {
                    let y_f = (out_y as f32) * (crop_h as f32 - 1.0) / 7.0;
                    let x_f = (out_x as f32) * (crop_w as f32 - 1.0) / 8.0;

                    let y_idx = y_f.floor() as usize;
                    let x_idx = x_f.floor() as usize;
                    
                    let yw = y_f - y_idx as f32;
                    let xw = x_f - x_idx as f32;

                    let y_idx_next = (y_idx + 1).min(crop_h - 1);
                    let x_idx_next = (x_idx + 1).min(crop_w - 1);

                    let p11 = frame[(y1 + y_idx) * 64 + (x1 + x_idx)] as f32;
                    let p12 = frame[(y1 + y_idx) * 64 + (x1 + x_idx_next)] as f32;
                    let p21 = frame[(y1 + y_idx_next) * 64 + (x1 + x_idx)] as f32;
                    let p22 = frame[(y1 + y_idx_next) * 64 + (x1 + x_idx_next)] as f32;

                    let val = p11 * (1.0 - yw) * (1.0 - xw)
                            + p12 * (1.0 - yw) * xw
                            + p21 * yw * (1.0 - xw)
                            + p22 * yw * xw;
                            
                    frame_8x9[out_y][out_x] = val as u8;
                }
            }
        } else {
            // Already 8x9 (Rare case but handled)
            for out_y in 0..8 {
                for out_x in 0..9 {
                    frame_8x9[out_y][out_x] = frame[(y1 + out_y) * 64 + (x1 + out_x)];
                }
            }
        }

        // Horizontal Difference to 64-bit Hash
        let mut hash: u64 = 0;
        for r in 0..8 {
            for c in 0..8 {
                if frame_8x9[r][c + 1] > frame_8x9[r][c] {
                    // Python np.packbits MSB first mapping
                    let bit_idx = r * 8 + c;
                    hash |= 1 << (63 - bit_idx);
                }
            }
        }
        hashes.push(hash);
    }

    // 5. Final Filter (Bit count bounds and contiguous duplicates)
    let timestamps: Vec<f32> = (0..total_frames).map(|i| (i as f32 / total_frames as f32) * exact_duration).collect();
    
    let mut final_hashes = Vec::new();
    let mut final_t_start = Vec::new();
    let mut final_idx_tracker = Vec::new();

    for i in 0..hashes.len() {
        let h = hashes[i];
        let bits = h.count_ones(); // HARDWARE POPCNT! SO FAST!
        let valid = bits > 2 && bits < 62;

        let should_push = if i == 0 {
            true
        } else {
            let prev_h = hashes[i-1];
            let prev_bits = prev_h.count_ones();
            let prev_valid = prev_bits > 2 && prev_bits < 62;
            h != prev_h || valid != prev_valid
        };

        if should_push && valid {
            final_hashes.push(h);
            final_t_start.push(timestamps[unique_frame_indices[i]]);
            final_idx_tracker.push(unique_frame_indices[i]);
        }
    }

    // Calculate end times
    let mut final_t_end = Vec::with_capacity(final_t_start.len());
    for i in 0..final_t_start.len() {
        if i + 1 < final_t_start.len() {
            // Use the timestamp of the NEXT valid change as the end time
            // Wait, Python script uses `timestamps[idx[final_idx[1:]]]`
            // Actually Python pulls from `timestamps[idx[final_idx[i+1]]]`. We tracked this!
            final_t_end.push(timestamps[final_idx_tracker[i+1]]);
        } else {
            final_t_end.push(exact_duration);
        }
    }

    println!("[{}] Final valid hashes to compare: {} (Completed in {:.2}s)", filepath, final_hashes.len(), start_time.elapsed().as_secs_f32());

    Some(VideoFingerprint {
        path: filepath.to_string(),
        duration: exact_duration,
        valid_hashes: final_hashes,
        valid_t_start: final_t_start,
        valid_t_end: final_t_end,
    })
}