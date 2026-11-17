//! One still frame of a video, in colour and at the size it is shown: the
//! desktop app's picture of each file of a group.
//!
//! The fingerprint keeps nothing a person could look at (each sample is a 64-bit
//! hash of a 16x16 grey square), so the window decodes a frame again. It is
//! opened the way the scan opens it (`fingerprint::open_video`, with its probe
//! gate), and only keyframes are decoded: a thumbnail needs one picture, and a
//! keyframe is the only one that can be had without decoding what leads up to
//! it.
//!
//! **Where in the video.** The first frame is the worst choice there is: a
//! fade from black, a studio logo, a title card, the same in every episode of
//! a series, which makes two different files look like copies. So the frame is
//! taken a tenth of the way in, and if that one is nearly flat (a black frame,
//! a fade), a third and then half way, keeping the busiest of those looked at.
//! A file whose runtime is unknown is read from its start.

use crate::fingerprint::{clock_may_restart, duration_seconds, header_is_complete, open_video, probe_streams};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use anyhow::{anyhow, Context, Result};
use ffmpeg_next::format::Pixel;
use ffmpeg_next::software::scaling;

/// How far into the video each try looks, as a share of its runtime.
const TRIES: [f64; 3] = [0.1, 0.33, 0.5];

/// The spread of brightness (standard deviation, 0-255) below which a frame
/// is taken for flat: black, a fade, a plain title card. A frame of picture
/// sits well above it.
const FLAT: f64 = 12.0;

/// Packets read after a seek before giving up on it: a seek that lands in a
/// stretch with no keyframe of this stream is not read to the end of a
/// two-hour file.
const MOST_PACKETS: usize = 4000;

/// Start FFmpeg, quietly, once, before the first `still`.
pub fn init() -> Result<()> {
    ffmpeg_next::init().context("Failed to initialize FFmpeg")?;
    ffmpeg_next::log::set_level(ffmpeg_next::log::Level::Quiet);
    Ok(())
}

/// A frame of `path` as RGBA, as large as fits inside `max_w` x `max_h` with
/// its shape kept (its pixel aspect applied), or, with `cover`, as small as
/// fills it, the rest cut off. Width, height and the pixels, rows packed.
pub fn still(path: &str, max_w: u32, max_h: u32, cover: bool) -> Result<(u32, u32, Vec<u8>)> {
    let mut ictx = open_video(path)?;
    if !header_is_complete(&ictx) {
        probe_streams(&mut ictx)?;
    }
    let (index, duration, parameters) = {
        let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).ok_or_else(|| anyhow!("No video stream found"))?;
        (stream.index(), duration_seconds(&stream, &ictx), stream.parameters())
    };
    let mut context = ffmpeg_next::codec::context::Context::from_parameters(parameters).context("Failed to create codec context")?;
    unsafe {
        let ctx = context.as_mut_ptr();
        (*ctx).thread_count = 2;
        (*ctx).skip_frame = ffmpeg_next::ffi::AVDiscard::AVDISCARD_NONKEY;
    }
    let mut decoder = context.decoder().video().context("Failed to open a video decoder")?;

    // The busiest frame looked at so far, and how busy.
    let mut best: Option<(f64, ffmpeg_next::frame::Video)> = None;
    let tries: &[f64] = if duration > 0.0 { &TRIES } else { &[0.0] };
    for &share in tries {
        if share > 0.0 {
            let at = (duration * share * 1_000_000.0) as i64;
            if ictx.seek(at, ..at).is_err() {
                continue;
            }
            decoder.flush();
        }
        let Some(frame) = first_frame(&mut ictx, &mut decoder, index) else { continue };
        let spread = spread(&frame);
        if best.as_ref().is_none_or(|(s, _)| spread > *s) {
            best = Some((spread, frame));
        }
        if best.as_ref().is_some_and(|(s, _)| *s >= FLAT) {
            break;
        }
    }
    // Nothing found where it was looked for: the start, which any file that
    // decodes at all has.
    if best.is_none() && duration > 0.0 && ictx.seek(0, ..0).is_ok() {
        decoder.flush();
        best = first_frame(&mut ictx, &mut decoder, index).map(|f| (0.0, f));
    }
    let (_, frame) = best.ok_or_else(|| anyhow!("No frame could be decoded"))?;
    to_rgba(&frame, max_w, max_h, cover)
}

/// The keyframe of `path` that the scan sampled at `at_ms`, as `still` gives
/// one: RGBA, fitted inside `max_w` x `max_h`.
///
/// A sample's time is milliseconds of decode time from the file's first
/// keyframe (`fingerprint::fingerprint_video`), so the frame is found the
/// same way round: the first keyframe's timestamp is read as the anchor, the
/// demuxer is sent to the keyframe at or before `anchor + at_ms`, and of the
/// keyframes from there the one nearest that time is decoded. Every sample is
/// a keyframe, so this gives back the very frame that was hashed.
///
/// Not by timestamp in a container whose clock may restart (MPEG-TS and -PS,
/// which `cat` joins): there one timestamp can name a frame in each segment,
/// and a seek found the second when the scan meant the first. The scan never
/// seeks in those either; it reads them in order, and so does `restart_index`,
/// once per file, replaying the scan's clock to find each keyframe's place in
/// the file's bytes.
pub fn frame_at(path: &str, at_ms: u32, max_w: u32, max_h: u32) -> Result<(u32, u32, Vec<u8>)> {
    let mut ictx = open_video(path)?;
    if !header_is_complete(&ictx) {
        probe_streams(&mut ictx)?;
    }
    let (index, time_base, parameters) = {
        let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).ok_or_else(|| anyhow!("No video stream found"))?;
        (stream.index(), stream.time_base(), stream.parameters())
    };
    let tb = if time_base.numerator() > 0 && time_base.denominator() > 0 { f64::from(time_base) } else { 1.0 / 90_000.0 };
    let stamp = |p: &ffmpeg_next::Packet| p.dts().or_else(|| p.pts());
    if clock_may_restart(&ictx) {
        let packet = by_restart_index(path, &mut ictx, index, tb, at_ms)?;
        return decode_one(parameters, &packet, at_ms, max_w, max_h);
    }

    // The anchor: the first keyframe, read as the scan read it.
    let anchor = ictx
        .packets()
        .take(MOST_PACKETS)
        .filter(|(s, p)| s.index() == index && p.is_key())
        .find_map(|(_, p)| stamp(&p))
        .ok_or_else(|| anyhow!("No keyframe with a timestamp"))?;
    let target = anchor + (at_ms as f64 / 1000.0 / tb).round() as i64;

    // The keyframe nearest the target, of those from where a seek lands to a
    // second past the target. A seek is asked for the keyframe at or before
    // the target, and MPEG-TS, which seeks by a binary search over its bytes,
    // can land after it: then nothing it reads is early enough, so it backs
    // off further each time, down to reading from the start.
    let past = (1.0 / tb) as i64;
    let mut nearest: Option<(i64, ffmpeg_next::Packet)> = None;
    for back_s in [0.0, 2.0, 10.0, 60.0, f64::INFINITY] {
        let from = if back_s.is_finite() { target - (back_s / tb) as i64 } else { i64::MIN };
        let us = if back_s.is_finite() { (from as f64 * tb * 1_000_000.0) as i64 } else { 0 };
        if ictx.seek(us, ..us + 1).is_err() {
            continue;
        }
        nearest = None;
        let mut first: Option<i64> = None;
        for (stream, packet) in ictx.packets().take(MOST_PACKETS * 4) {
            if stream.index() != index || !packet.is_key() {
                continue;
            }
            let Some(ts) = stamp(&packet) else { continue };
            first.get_or_insert(ts);
            let off = (ts - target).abs();
            if nearest.as_ref().is_none_or(|(o, _)| off < *o) {
                nearest = Some((off, packet));
            }
            if ts > target + past {
                break;
            }
        }
        // Landed at or before the target: the nearest is among what was read.
        if first.is_some_and(|f| f <= target) || !back_s.is_finite() {
            break;
        }
    }
    let (_, packet) = nearest.ok_or_else(|| anyhow!("No keyframe near {at_ms} ms"))?;
    decode_one(parameters, &packet, at_ms, max_w, max_h)
}

/// Decode `packet`, a keyframe, alone, and fit it inside `max_w` x `max_h`.
fn decode_one(parameters: ffmpeg_next::codec::Parameters, packet: &ffmpeg_next::Packet, at_ms: u32, max_w: u32, max_h: u32) -> Result<(u32, u32, Vec<u8>)> {
    let mut context = ffmpeg_next::codec::context::Context::from_parameters(parameters).context("Failed to create codec context")?;
    unsafe {
        // One thread: a single keyframe, wanted back at once rather than
        // after a thread pipeline fills.
        (*context.as_mut_ptr()).thread_count = 1;
    }
    let mut decoder = context.decoder().video().context("Failed to open a video decoder")?;
    let mut frame = ffmpeg_next::frame::Video::empty();
    let _ = decoder.send_packet(packet);
    if decoder.receive_frame(&mut frame).is_err() {
        let _ = decoder.send_eof();
        decoder.receive_frame(&mut frame).map_err(|e| anyhow!("The keyframe near {at_ms} ms would not decode: {e}"))?;
    }
    to_rgba(&frame, max_w, max_h, false)
}

/// Each keyframe of a file, as `(the scan's ms, its byte position)`, by path,
/// length and modification time: read once and kept, since a comparison asks
/// for many frames of one file.
type RestartIndex = Arc<Vec<(u32, i64)>>;
/// A file as it stood when its index was read: path, length, modification
/// time. A file that changed is read again.
type FileStamp = (String, u64, std::time::SystemTime);
static RESTART_INDEXES: Mutex<Option<HashMap<FileStamp, RestartIndex>>> = Mutex::new(None);

/// The keyframe the scan sampled at `at_ms` in a file whose clock may restart,
/// found through `restart_index` and read from its byte position.
fn by_restart_index(path: &str, ictx: &mut ffmpeg_next::format::context::Input, index: usize, tb: f64, at_ms: u32) -> Result<ffmpeg_next::Packet> {
    let md = std::fs::metadata(path).context("The video could not be looked at")?;
    let key = (path.to_string(), md.len(), md.modified().unwrap_or(std::time::UNIX_EPOCH));
    let kept = RESTART_INDEXES.lock().unwrap().get_or_insert_with(HashMap::new).get(&key).cloned();
    let keyframes = match kept {
        Some(k) => k,
        None => {
            let built = Arc::new(restart_index(ictx, index, tb));
            RESTART_INDEXES.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, built.clone());
            built
        }
    };
    let &(_, pos) = keyframes
        .iter()
        .min_by_key(|(ms, _)| ms.abs_diff(at_ms))
        .ok_or_else(|| anyhow!("No keyframe near {at_ms} ms"))?;
    // SAFETY: the context is open, and a byte seek takes no stream.
    let r = unsafe { ffmpeg_next::ffi::av_seek_frame(ictx.as_mut_ptr(), -1, pos, ffmpeg_next::ffi::AVSEEK_FLAG_BYTE) };
    if r < 0 {
        return Err(anyhow!("The video could not be read at byte {pos}"));
    }
    ictx.packets()
        .take(MOST_PACKETS * 4)
        .find(|(s, p)| s.index() == index && p.is_key() && p.position() as i64 >= pos)
        .map(|(_, p)| p)
        .ok_or_else(|| anyhow!("No keyframe at byte {pos}"))
}

/// Every keyframe of the stream from the start, timed by the scan's clock:
/// gaps between timestamps accumulated, and a step backwards (a restart)
/// charged the last gap seen, exactly as `fingerprint_video` times its
/// samples, so each lands on the millisecond the scan gave it.
fn restart_index(ictx: &mut ffmpeg_next::format::context::Input, index: usize, tb: f64) -> Vec<(u32, i64)> {
    if ictx.seek(0, ..1).is_err() {
        // SAFETY: as in `by_restart_index`.
        unsafe { ffmpeg_next::ffi::av_seek_frame(ictx.as_mut_ptr(), -1, 0, ffmpeg_next::ffi::AVSEEK_FLAG_BYTE) };
    }
    let (mut out, mut prev, mut elapsed, mut last_gap) = (Vec::new(), None::<i64>, 0i64, 0i64);
    for (stream, packet) in ictx.packets() {
        if stream.index() != index || !packet.is_key() {
            continue;
        }
        let Some(ts) = packet.dts().or_else(|| packet.pts()) else { continue };
        match prev {
            None => elapsed = 0,
            Some(p) if ts >= p => {
                if ts > p {
                    last_gap = ts - p;
                }
                elapsed = elapsed.saturating_add(ts - p);
            }
            Some(_) => elapsed = elapsed.saturating_add(last_gap.max(1)),
        }
        prev = Some(ts);
        out.push(((elapsed as f64 * tb * 1000.0) as u32, packet.position() as i64));
    }
    out
}

/// The first frame the decoder gives after where `ictx` stands.
fn first_frame(
    ictx: &mut ffmpeg_next::format::context::Input,
    decoder: &mut ffmpeg_next::decoder::Video,
    index: usize,
) -> Option<ffmpeg_next::frame::Video> {
    let mut frame = ffmpeg_next::frame::Video::empty();
    let mut read = 0;
    for (stream, packet) in ictx.packets() {
        if stream.index() != index || !packet.is_key() {
            continue;
        }
        // A packet the decoder will not take is skipped, as the scan skips
        // one: the next keyframe will do as well.
        let _ = decoder.send_packet(&packet);
        if decoder.receive_frame(&mut frame).is_ok() {
            return Some(frame);
        }
        read += 1;
        if read >= MOST_PACKETS {
            return None;
        }
    }
    // The end of the file: what the decoder still holds.
    let _ = decoder.send_eof();
    decoder.receive_frame(&mut frame).ok().map(|_| frame)
}

/// How much the brightness of `frame` varies: the standard deviation of a
/// sparse grid of its luma, or for a frame not in a YUV format, of nothing
/// (0, so any other frame is preferred to it).
fn spread(frame: &ffmpeg_next::frame::Video) -> f64 {
    let (w, h) = (frame.width() as usize, frame.height() as usize);
    let planar_yuv = matches!(
        frame.format(),
        Pixel::YUV420P | Pixel::YUVJ420P | Pixel::YUV422P | Pixel::YUVJ422P | Pixel::YUV444P | Pixel::YUVJ444P | Pixel::NV12 | Pixel::NV21
    );
    if !planar_yuv || w == 0 || h == 0 {
        return 0.0;
    }
    let (data, stride) = (frame.data(0), frame.stride(0));
    let (mut n, mut sum, mut sq) = (0.0, 0.0, 0.0);
    for y in (0..h).step_by((h / 32).max(1)) {
        for x in (0..w).step_by((w / 32).max(1)) {
            let v = data[y * stride + x] as f64;
            n += 1.0;
            sum += v;
            sq += v * v;
        }
    }
    let mean = sum / n;
    (sq / n - mean * mean).max(0.0).sqrt()
}

/// The size a `w` x `h` picture is shown at in a `max_w` x `max_h` place:
/// inside it, or with `cover` filling it, its shape kept, never enlarged.
pub fn fitted((w, h): (f64, f64), max_w: u32, max_h: u32, cover: bool) -> (u32, u32) {
    if w <= 0.0 || h <= 0.0 {
        return (max_w.max(1), max_h.max(1));
    }
    let (sw, sh) = (max_w as f64 / w, max_h as f64 / h);
    let s = if cover { sw.max(sh) } else { sw.min(sh) }.min(1.0);
    (((w * s).round() as u32).max(1), ((h * s).round() as u32).max(1))
}

/// `frame` as packed RGBA at the size it is shown (`fitted`), cut to the place
/// with `cover`.
fn to_rgba(frame: &ffmpeg_next::frame::Video, max_w: u32, max_h: u32, cover: bool) -> Result<(u32, u32, Vec<u8>)> {
    // The shape it is meant to be seen at: anamorphic DVD and broadcast video
    // stores 720 x 576 pixels for a 16:9 picture.
    let sar = frame.aspect_ratio();
    let par = if sar.numerator() > 0 && sar.denominator() > 0 { sar.numerator() as f64 / sar.denominator() as f64 } else { 1.0 };
    let shape = (frame.width() as f64 * par, frame.height() as f64);
    let (w, h) = fitted(shape, max_w, max_h, cover);
    let mut scaler = scaling::Context::get(frame.format(), frame.width(), frame.height(), Pixel::RGBA, w, h, scaling::Flags::AREA)
        .context("Failed to set up the scaler")?;
    let mut rgba = ffmpeg_next::frame::Video::empty();
    scaler.run(frame, &mut rgba).context("Failed to scale the frame")?;
    // Cut to the place, from the middle.
    let (cw, ch) = if cover { (w.min(max_w), h.min(max_h)) } else { (w, h) };
    let (x0, y0) = (((w - cw) / 2) as usize, ((h - ch) / 2) as usize);
    let (data, stride) = (rgba.data(0), rgba.stride(0));
    let mut out = Vec::with_capacity(cw as usize * ch as usize * 4);
    for y in y0..y0 + ch as usize {
        let row = y * stride + x0 * 4;
        out.extend_from_slice(&data[row..row + cw as usize * 4]);
    }
    Ok((cw, ch, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame of a real video, at the size asked for and the video's shape,
    /// with picture in it rather than black.
    #[test]
    fn a_still_is_a_frame_of_the_video_at_its_shape() {
        init().unwrap();
        let path = "tests/fixtures/test_video.mp4";
        let (w, h, px) = still(path, 320, 320, false).expect("the fixture decodes");
        assert_eq!(px.len(), (w * h * 4) as usize);
        assert!(w == 320 || h == 320, "fitted to the place: {w}x{h}");
        assert!(w <= 320 && h <= 320);
        let lumas: Vec<f64> = px.chunks(4).map(|p| p[0] as f64 * 0.3 + p[1] as f64 * 0.59 + p[2] as f64 * 0.11).collect();
        let mean = lumas.iter().sum::<f64>() / lumas.len() as f64;
        assert!(mean > 5.0, "a picture, not black");
        // Filling a place exactly instead.
        let (w, h, px) = still(path, 160, 90, true).unwrap();
        assert_eq!((w, h, px.len()), (160, 90, 160 * 90 * 4));
    }

    /// Every keyframe of `path` read in order from its start, each with its
    /// time in ms from the first keyframe and its picture as `frame_at` sizes
    /// it: an oracle that does no seeking at all.
    fn keyframes_in_order(path: &str, max: u32) -> Vec<(f64, Vec<u8>)> {
        let mut ictx = open_video(path).unwrap();
        let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).unwrap();
        let (index, tb, params) = (stream.index(), f64::from(stream.time_base()), stream.parameters());
        let mut out = Vec::new();
        let packets: Vec<ffmpeg_next::Packet> = ictx.packets().filter(|(s, p)| s.index() == index && p.is_key()).map(|(_, p)| p).collect();
        // The scan's clock, which goes on through a restart.
        let (mut prev, mut elapsed, mut last_gap) = (None::<i64>, 0i64, 0i64);
        for packet in packets {
            let ts = packet.dts().or_else(|| packet.pts()).unwrap();
            match prev {
                None => {}
                Some(p) if ts >= p => {
                    if ts > p {
                        last_gap = ts - p;
                    }
                    elapsed += ts - p;
                }
                Some(_) => elapsed += last_gap.max(1),
            }
            prev = Some(ts);
            let mut ctx = ffmpeg_next::codec::context::Context::from_parameters(params.clone()).unwrap();
            unsafe { (*ctx.as_mut_ptr()).thread_count = 1 };
            let mut dec = ctx.decoder().video().unwrap();
            let mut frame = ffmpeg_next::frame::Video::empty();
            let _ = dec.send_packet(&packet);
            if dec.receive_frame(&mut frame).is_err() {
                let _ = dec.send_eof();
                dec.receive_frame(&mut frame).unwrap();
            }
            out.push((elapsed as f64 * tb * 1000.0, to_rgba(&frame, max, max, false).unwrap().2));
        }
        out
    }

    /// The frame shown for a sample is the frame that was hashed: for every
    /// sample of a real fingerprint, `frame_at` gives back exactly the
    /// keyframe that sits at that time when the file is read from the start.
    #[test]
    fn the_frame_at_a_sample_time_is_the_frame_the_scan_hashed() {
        init().unwrap();
        for path in ["tests/fixtures/test_video_capture.ts", "tests/fixtures/test_video_segment.ts", "tests/fixtures/test_video_rescaled.ts", "tests/fixtures/test_video.mp4"] {
            let size = std::fs::metadata(path).unwrap().len();
            let fp = crate::fingerprint::fingerprint_video(path, 0.0, 0.0, 1, 0.0, size, &|_| {}).unwrap().expect("the fixture fingerprints");
            let oracle = keyframes_in_order(path, 64);
            assert!(fp.valid_t_start.len() > 1 || path.ends_with(".mp4"), "{path}: more than one sample to tell apart");
            for &t in &fp.valid_t_start {
                let (_, _, px) = frame_at(path, t, 64, 64).unwrap_or_else(|e| panic!("{path} at {t} ms: {e:#}"));
                let (at, want) = oracle.iter().min_by(|a, b| (a.0 - t as f64).abs().total_cmp(&(b.0 - t as f64).abs())).unwrap();
                assert!((at - t as f64).abs() < 1.0, "{path}: no keyframe at {t} ms (nearest {at})");
                assert!(px == *want, "{path}: the frame at {t} ms is not the keyframe there");
            }
        }
    }

    /// Something that is not a video says so rather than drawing nothing.
    #[test]
    fn a_file_that_is_not_video_is_an_error() {
        init().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("notes.mp4");
        std::fs::write(&p, b"not a video at all").unwrap();
        assert!(still(p.to_str().unwrap(), 100, 100, false).is_err());
    }

    #[test]
    fn a_picture_is_fitted_without_being_enlarged() {
        assert_eq!(fitted((1920.0, 1080.0), 320, 320, false), (320, 180));
        assert_eq!(fitted((1920.0, 1080.0), 160, 160, true), (284, 160));
        assert_eq!(fitted((100.0, 50.0), 320, 320, false), (100, 50), "never enlarged");
        assert_eq!(fitted((0.0, 0.0), 32, 18, false), (32, 18));
    }
}
