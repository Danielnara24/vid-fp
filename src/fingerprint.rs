use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

// Every stored frame is a 64x64 GRAY8 buffer. Crucially we keep them all in ONE
// contiguous allocation (`u_frames` below) instead of a Vec<Vec<u8>>. A single
// large buffer is served by mmap on Linux and returned to the OS (munmap) the
// moment this video is done, so RSS falls back between videos instead of
// ratcheting up across a multi-threaded run.
const FRAME_STRIDE: usize = 64 * 64;

#[derive(Serialize, Deserialize, Clone)]
pub struct VideoFingerprint {
    pub path: String,
    pub valid_hashes: Vec<u64>,
    pub valid_t_start: Vec<u32>,
    pub valid_t_end: Vec<u32>,
    pub total_frames: u32,
    pub width: u32,
    pub height: u32,
    pub duration: f64,
    pub file_size: u64,
}

impl VideoFingerprint {
    /// Overall bitrate in bits per second.
    ///
    /// Derived from bytes and seconds rather than read from the container, on
    /// purpose. Per-stream `bit_rate` is set for MP4/AVI but almost never for
    /// MKV/WebM, so reading it would compare a video-only number against a
    /// container-total one whenever a group spans containers -- a bias the
    /// size of an audio track, in the exact case this tool exists to catch.
    /// This definition is identical for every file and costs nothing: both
    /// operands are already stored, so it adds no decode work and no cache
    /// churn.
    ///
    /// Caveat worth knowing: it counts audio. A copy with lossless 5.1 can
    /// outrank one with a better video track and stereo AAC.
    pub fn bitrate(&self) -> u64 {
        if self.duration > 0.0 {
            ((self.file_size as f64 * 8.0) / self.duration) as u64
        } else {
            0
        }
    }
}

pub fn fingerprint_video(filepath: &str, kf_interval: f64, min_kf_samples: f64) -> Result<VideoFingerprint> {
    // 1. Native Zero-Copy Extraction (No Subprocess Overhead)
    let mut ictx = ffmpeg_next::format::input(&filepath)
        .with_context(|| format!("Failed to open video file: {}", filepath))?;

    let input = ictx.streams().best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| anyhow!("No video stream found in {}", filepath))?;
    let video_stream_index = input.index();

    // Extract duration natively from stream or format context without delay
    let mut duration_sec = 0.0;
    let stream_duration = input.duration();
    if stream_duration >= 0 {
        let tb = input.time_base();
        if tb.denominator() > 0 {
            duration_sec = stream_duration as f64 * (tb.numerator() as f64 / tb.denominator() as f64);
        }
    }
    if duration_sec <= 0.0 {
        duration_sec = ictx.duration() as f64 / 1_000_000.0; // Fallback to FFmpeg's AV_TIME_BASE format duration
    }

    let mut context_decoder = ffmpeg_next::codec::context::Context::from_parameters(input.parameters())
        .context("Failed to create codec context from parameters")?;

    unsafe {
        let ctx = context_decoder.as_mut_ptr();
        (*ctx).thread_count = 1;
        (*ctx).skip_loop_filter = ffmpeg_next::ffi::AVDiscard::AVDISCARD_ALL;
        (*ctx).flags2 |= ffmpeg_next::ffi::AV_CODEC_FLAG2_FAST as i32;
        // Emit decoded frames immediately with no reordering delay. We only decode
        // independent keyframes, so this is safe and keeps the decoder from parking
        // several full-resolution frames in its internal pool (a big per-thread cost
        // on hi-res, hour-long inputs).
        (*ctx).flags |= ffmpeg_next::ffi::AV_CODEC_FLAG_LOW_DELAY as i32;
        (*ctx).skip_frame = ffmpeg_next::ffi::AVDiscard::AVDISCARD_NONKEY;
    }

    let mut decoder = context_decoder.decoder().video()
        .context("Failed to initialize video decoder")?;

    let width = decoder.width();
    let height = decoder.height();
    let file_size = std::fs::metadata(filepath).map(|m| m.len()).unwrap_or(0);

    let mut scaler = ffmpeg_next::software::scaling::context::Context::get(
        decoder.format(),
        width,
        height,
        ffmpeg_next::format::Pixel::GRAY8,
        64,
        64,
        ffmpeg_next::software::scaling::flag::Flags::FAST_BILINEAR,
    ).context("Failed to initialize video scaler")?;

    // All unique frames packed back-to-back, FRAME_STRIDE bytes each. One growable
    // allocation instead of N tiny ones -> no heap fragmentation/retention, and the
    // whole thing is released to the OS when this function returns.
    let mut u_frames: Vec<u8> = Vec::with_capacity(FRAME_STRIDE * 64);
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
            // Append into the single flat buffer instead of pushing a new Vec.
            u_frames.extend_from_slice(&current_frame);
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

    unsafe {
        let fmt_ctx = ictx.as_mut_ptr();
        let stream_ptr = *(*fmt_ctx).streams.add(video_stream_index);
        (*stream_ptr).discard = ffmpeg_next::ffi::AVDiscard::AVDISCARD_NONKEY;
    }

    // --- Length-aware keyframe subsampling -----------------------------------
    // A fixed interval decimates short videos long before long ones. Scaling the
    // interval WITH duration would fix that but break clip detection: a long host
    // sampled sparsely no longer has sampled frames inside a short clip's time
    // window, so the clip's hashes find nothing to match. So we bound the interval
    // in absolute time (protecting clip detection in long hosts) and FLOOR it for
    // short videos so they always keep at least min_kf_samples frames.
    // min_kf_samples guards short videos: they get a finer interval so they always
    // keep at least this many frames. Guard against <= 0 to avoid div-by-zero / NaN.
    let effective_interval = if kf_interval > 0.0 && duration_sec > 0.0 && min_kf_samples > 0.0 {
        kf_interval.min(duration_sec / min_kf_samples)
    } else {
        0.0
    };
    let mut last_kept_t: Option<f64> = None;

    // Rapid Demuxing: Only push Key-frames (I-Frames) into decoder
    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index && packet.is_key() {
            if effective_interval > 0.0 {
                let tb = stream.time_base();
                let t = packet.pts().or_else(|| packet.dts()).map(|ts| {
                    ts as f64 * tb.numerator() as f64 / tb.denominator() as f64
                });
                if let Some(t) = t {
                    if let Some(last) = last_kept_t {
                        if t - last < effective_interval {
                            continue; // too close to the last kept keyframe; skip decode
                        }
                    }
                    last_kept_t = Some(t);
                }
                // If PTS/DTS is missing we fall through and keep the frame (safe default).
            }

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
    if total_frames == 0 {
        return Err(anyhow!("No valid frames found or successfully decoded in {}", filepath));
    }

    // u_frames holds n_unique * FRAME_STRIDE bytes; the frame count is the index list.
    let n_unique = unique_frame_indices.len();
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
    // Iterate the flat buffer in FRAME_STRIDE-sized windows; each `frame` is a
    // &[u8] of length 4096, so all the indexing below is unchanged.
    for frame in u_frames.chunks_exact(FRAME_STRIDE) {
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

    // Frame pixels are no longer needed; release the large buffer (munmap) now
    // rather than at end of scope, trimming the peak during the cheap tail work.
    drop(u_frames);

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

    Ok(VideoFingerprint {
        path: filepath.to_string(),
        valid_hashes: final_hashes,
        valid_t_start: final_t_start,
        valid_t_end: final_t_end,
        total_frames: total_frames as u32,
        width,
        height,
        duration: duration_sec,
        file_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Once;

    // Rust runs tests in parallel. FFmpeg's global initialization must only happen once,
    // so we use `std::sync::Once` to prevent thread-safety crashes during testing.
    static INIT: Once = Once::new();

    fn init_ffmpeg_for_tests() {
        INIT.call_once(|| {
            ffmpeg_next::init().expect("Failed to initialize FFmpeg for tests");
            ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);
        });
    }

    #[test]
    fn test_fingerprint_real_video() {
        init_ffmpeg_for_tests();

        let mut fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        fixture_path.push("tests");
        fixture_path.push("fixtures");
        fixture_path.push("test_video.mp4"); // <--- CHANGE THIS TO YOUR EXACT FILE NAME

        let filepath = fixture_path.to_string_lossy().to_string();
        assert!(
            fixture_path.exists(),
            "Fixture video not found at: {}.",
            filepath
        );

        // Run the fingerprinting function
        let result = fingerprint_video(&filepath, 0.0, 4.0);
        assert!(result.is_ok(), "Failed to fingerprint video: {:?}", result.err());
        let fp = result.unwrap();

        // Assert the properties of the generated fingerprint
        assert!(fp.total_frames > 0, "Video should have parsed at least 1 frame");
        assert!(fp.width > 0, "Video width should be parsed correctly");
        assert!(fp.height > 0, "Video height should be parsed correctly");
        assert!(fp.file_size > 0, "File size should be captured");
        assert!(
            fp.duration > 0.0 && fp.duration < 5.0,
            "Duration should be roughly 1 second, got: {}", fp.duration
        );

        // Ensure the hashing logic successfully triggered and populated arrays
        assert!(
            !fp.valid_hashes.is_empty(),
            "Video should have generated at least one hash"
        );
        assert_eq!(
            fp.valid_hashes.len(),
            fp.valid_t_start.len(),
            "Hash lists and timing lists must stay synchronized"
        );
    }
}