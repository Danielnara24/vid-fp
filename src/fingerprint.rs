use std::process::Command;

pub struct VideoFingerprint {
    pub path: String,
    pub valid_hashes: Vec<u64>,
    pub valid_t_start: Vec<u32>,
    pub valid_t_end: Vec<u32>,
    pub total_frames: u32,
}

pub fn fingerprint_video(filepath: &str) -> Option<VideoFingerprint> {
    // 1. FFmpeg Subprocess Extraction (Subprocess halved: We completely removed the redundant ffprobe)
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

    // 2. Filter duplicate adjacent frames 
    let mut u_frames = Vec::new();
    let mut unique_frame_indices = Vec::new();
    
    let chunks: Vec<&[u8]> = raw_bytes.chunks_exact(frame_size).collect();
    u_frames.push(chunks[0]);
    unique_frame_indices.push(0);

    for i in 1..total_frames {
        if chunks[i] != chunks[i - 1] {
            u_frames.push(chunks[i]);
            unique_frame_indices.push(i as u32);
        }
    }
    
    let n_unique = u_frames.len();

    // 3. Calculate Variance for Auto-Cropping algebraically in a single fast pass
    let mut sum = vec![0u64; 64 * 64];
    let mut sum_sq = vec![0u64; 64 * 64];

    for &frame in &u_frames {
        for i in 0..(64 * 64) {
            let val = frame[i] as u64;
            sum[i] += val;
            sum_sq[i] += val * val;
        }
    }

    let mut row_max_var = [0.0f32; 64];
    let mut col_max_var = [0.0f32; 64];
    let n_f32 = n_unique as f32;

    for y in 0..64 {
        for x in 0..64 {
            let i = y * 64 + x;
            let mean = sum[i] as f32 / n_f32;
            let mean_sq = sum_sq[i] as f32 / n_f32;
            let variance = mean_sq - (mean * mean); 

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

    struct InterpWeight {
        y_idx: usize, x_idx: usize,
        y_idx_next: usize, x_idx_next: usize,
        w11: f32, w12: f32,
        w21: f32, w22: f32,
    }

    let mut weights = Vec::with_capacity(8 * 9);
    if crop_h != 8 || crop_w != 9 {
        for out_y in 0..8 {
            for out_x in 0..9 {
                let y_f = (out_y as f32) * (crop_h as f32 - 1.0) / 7.0;
                let x_f = (out_x as f32) * (crop_w as f32 - 1.0) / 8.0;

                let y_idx = y_f.floor() as usize;
                let x_idx = x_f.floor() as usize;
                
                let yw = y_f - y_idx as f32;
                let xw = x_f - x_idx as f32;

                weights.push(InterpWeight {
                    y_idx: y1 + y_idx,
                    x_idx: x1 + x_idx,
                    y_idx_next: y1 + (y_idx + 1).min(crop_h - 1),
                    x_idx_next: x1 + (x_idx + 1).min(crop_w - 1),
                    w11: (1.0 - yw) * (1.0 - xw),
                    w12: (1.0 - yw) * xw,
                    w21: yw * (1.0 - xw),
                    w22: yw * xw,
                });
            }
        }
    }

    // 4. Generate Grids & Pack into Hashes immediately
    let mut hashes = Vec::with_capacity(n_unique);
    
    for &frame in &u_frames {
        let mut frame_8x9 = [0u8; 72]; 
        
        if crop_h != 8 || crop_w != 9 {
            for (i, w) in weights.iter().enumerate() {
                let p11 = frame[w.y_idx * 64 + w.x_idx] as f32;
                let p12 = frame[w.y_idx * 64 + w.x_idx_next] as f32;
                let p21 = frame[w.y_idx_next * 64 + w.x_idx] as f32;
                let p22 = frame[w.y_idx_next * 64 + w.x_idx_next] as f32;

                frame_8x9[i] = (p11 * w.w11 + p12 * w.w12 + p21 * w.w21 + p22 * w.w22) as u8;
            }
        } else {
            let mut i = 0;
            for out_y in 0..8 {
                for out_x in 0..9 {
                    frame_8x9[i] = frame[(y1 + out_y) * 64 + (x1 + out_x)];
                    i += 1;
                }
            }
        }

        let mut hash: u64 = 0;
        let mut bit_idx = 0;
        for r in 0..8 {
            let row_offset = r * 9;
            for c in 0..8 {
                if frame_8x9[row_offset + c + 1] > frame_8x9[row_offset + c] {
                    hash |= 1 << (63 - bit_idx);
                }
                bit_idx += 1;
            }
        }
        hashes.push(hash);
    }

    // 5. Integer-based Timestamp Mapping
    let mut changes_hashes = Vec::new();
    let mut changes_t_start = Vec::new();
    let mut changes_valid = Vec::new();

    for i in 0..hashes.len() {
        let h = hashes[i];
        let bits = h.count_ones(); 
        let valid = bits > 2 && bits < 62;

        let should_push = if i == 0 {
            true
        } else {
            let prev_h = hashes[i-1];
            let prev_bits = prev_h.count_ones();
            let prev_valid = prev_bits > 2 && prev_bits < 62;
            h != prev_h || valid != prev_valid
        };

        if should_push {
            changes_hashes.push(h);
            changes_t_start.push(unique_frame_indices[i]);
            changes_valid.push(valid);
        }
    }

    let mut final_hashes = Vec::new();
    let mut final_t_start = Vec::new();
    let mut final_t_end = Vec::new();

    for i in 0..changes_hashes.len() {
        if changes_valid[i] {
            final_hashes.push(changes_hashes[i]);
            final_t_start.push(changes_t_start[i]);

            if i + 1 < changes_hashes.len() {
                final_t_end.push(changes_t_start[i + 1]);
            } else {
                final_t_end.push(total_frames as u32);
            }
        }
    }

    Some(VideoFingerprint {
        path: filepath.to_string(),
        valid_hashes: final_hashes,
        valid_t_start: final_t_start,
        valid_t_end: final_t_end,
        total_frames: total_frames as u32,
    })
}