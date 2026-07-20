use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct VideoFingerprint {
    pub path: String,
    pub valid_hashes: Vec<u64>,
    pub valid_t_start: Vec<u32>,
    pub valid_t_end: Vec<u32>,
    pub total_frames: u32,
}

pub fn fingerprint_video(filepath: &str) -> Option<VideoFingerprint> {
    // 1. Native Zero-Copy Extraction (No Subprocess Overhead)
    let mut ictx = ffmpeg_next::format::input(&filepath).ok()?;
    let input = ictx.streams().best(ffmpeg_next::media::Type::Video)?;
    let video_stream_index = input.index();

    let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(input.parameters()).ok()?;
    let mut decoder = context_decoder.decoder().video().ok()?;

    let mut scaler = ffmpeg_next::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg_next::format::Pixel::GRAY8,
        64,
        64,
        ffmpeg_next::software::scaling::flag::Flags::FAST_BILINEAR,
    ).ok()?;

    let mut u_frames = Vec::new();
    let mut unique_frame_indices = Vec::new();

    let mut sum = vec![0u64; 64 * 64];
    let mut sum_sq = vec![0u64; 64 * 64];

    let mut frame_idx = 0;
    let mut prev_frame = vec![0u8; 4096];
    let mut is_first = true;

    // By hoisting `decoded`, `scaled`, and `current_frame` outside the loop, 
    // we prevent extremely slow and fragmenting continuous allocation of AVFrame structures and memory buffers.
    // FFmpeg's zero-copy buffer pool is now properly utilized.
    let mut decoded = ffmpeg_next::frame::Video::empty();
    let mut scaled = ffmpeg_next::frame::Video::empty();
    let mut current_frame = vec![0u8; 4096];

    let mut process_frame = |dec: &ffmpeg_next::frame::Video| -> Result<(), ffmpeg_next::Error> {
        scaler.run(dec, &mut scaled)?;

        let data = scaled.data(0);
        let stride = scaled.stride(0);

        for y in 0..64 {
            let src_idx = y * stride;
            let dst_idx = y * 64;
            current_frame[dst_idx..dst_idx + 64].copy_from_slice(&data[src_idx..src_idx + 64]);
        }

        if is_first || current_frame != prev_frame {
            u_frames.push(current_frame.clone());
            unique_frame_indices.push(frame_idx);

            for (i, &val) in current_frame.iter().enumerate() {
                let v = val as u64;
                sum[i] += v;
                sum_sq[i] += v * v;
            }

            prev_frame.copy_from_slice(&current_frame);
            is_first = false;
        }
        frame_idx += 1;
        Ok(())
    };

    // Rapid Demuxing: Only push Key-frames (I-Frames) into decoder 
    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index && packet.is_key() {
            if decoder.send_packet(&packet).is_ok() {
                while decoder.receive_frame(&mut decoded).is_ok() {
                    let _ = process_frame(&decoded);
                }
            }
        }
    }

    let _ = decoder.send_eof();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let _ = process_frame(&decoded);
    }

    let total_frames = frame_idx;
    if total_frames == 0 { return None; }

    let n_unique = u_frames.len();
    let n_f32 = n_unique as f32;

    // 2. Variance & Auto-Crop Algebra (Untouched - accurate)
    let mut row_max_var = [0.0f32; 64];
    let mut col_max_var = [0.0f32; 64];

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

    let mut hashes = Vec::with_capacity(n_unique);
    for frame in &u_frames {
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

    let mut changes_hashes = Vec::new();
    let mut changes_t_start = Vec::new();
    let mut changes_valid = Vec::new();

    for i in 0..hashes.len() {
        let h = hashes[i];
        let valid = h.count_ones() > 2 && h.count_ones() < 62;

        let should_push = if i == 0 {
            true
        } else {
            let prev_h = hashes[i - 1];
            let prev_valid = prev_h.count_ones() > 2 && prev_h.count_ones() < 62;
            h != prev_h || valid != prev_valid
        };

        if should_push {
            changes_hashes.push(h);
            changes_t_start.push(unique_frame_indices[i]);
            changes_valid.push(valid);
        }
    }

    let mut final_hashes = Vec::with_capacity(changes_hashes.len());
    let mut final_t_start = Vec::with_capacity(changes_hashes.len());
    let mut final_t_end = Vec::with_capacity(changes_hashes.len());

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