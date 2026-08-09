use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
// `Packet::as_mut_ptr` lives on this trait; the demux loop below needs the raw
// pointer to unref the packet it reuses.
use ffmpeg_next::codec::packet::Mut as _;
use crate::utils::shutdown_requested;

// Every stored frame is a 64x64 GRAY8 buffer. Crucially we keep them all in ONE
// contiguous allocation (`u_frames` below) instead of a Vec<Vec<u8>>. A single
// large buffer is served by mmap on Linux and returned to the OS (munmap) the
// moment this video is done, so RSS falls back between videos instead of
// ratcheting up across a multi-threaded run.
const FRAME_STRIDE: usize = 64 * 64;

/// `FF_THREAD_FRAME` and `FF_THREAD_SLICE` from libavcodec. Spelled out rather
/// than taken from the bindings because they are plain `#define`s, and the name
/// bindgen gives them has moved between ffmpeg-sys-next releases.
const FF_THREAD_FRAME: i32 = 1;
const FF_THREAD_SLICE: i32 = 2;

/// Ceiling on decoder threads for a single video. Past roughly this point extra
/// threads buy almost nothing on keyframe-only decoding, and each one costs a
/// full-resolution frame buffer in the decoder's pool. Exported so the
/// scheduler in `main` never plans an allocation this function would clamp away.
pub const MAX_DECODE_THREADS: usize = 16;

/// Anything above this is not a frame rate. Containers occasionally park a
/// timebase (90000/1) or a placeholder in the frame-rate fields, and a bogus
/// denominator there would silently divide the quality figure into nothing.
/// Above the ceiling we record "unknown", which every consumer already handles.
const MAX_PLAUSIBLE_FRAME_RATE: f64 = 1000.0;

/// Side of the thumbnail the frame hash is computed from.
///
/// The auto-cropped region of the 64x64 buffer is box-filtered down to
/// THUMB x THUMB and transformed; the 8x8 lowest-frequency corner of that
/// transform is what becomes the 64-bit hash. 16 is deliberately small: the
/// coefficients above it describe detail that a re-encode is free to throw
/// away, and including them made two encodes of the same frame *less* alike
/// without making two different frames any less alike.
const THUMB: usize = 16;

/// Side of the low-frequency block kept from the transform. 8x8 = 64
/// coefficients = the 64 bits of the hash.
const KEEP: usize = 8;

/// Mean absolute AC coefficient below which a frame carries no usable
/// structure -- a black frame, a fade, a plain title card.
///
/// The bits of such a frame are thresholded noise: two unrelated blank frames
/// land a coin-flip apart and can drift inside any tolerance worth using. Real
/// content sits two orders of magnitude above this floor (the 1st percentile of
/// a full episode's keyframes is around 60), so nothing with a picture in it is
/// at risk of being dropped.
const MIN_AC_ENERGY: f32 = 8.0;

#[derive(Serialize, Deserialize, Clone)]
pub struct VideoFingerprint {
    pub path: String,
    pub valid_hashes: Vec<u64>,
    /// Milliseconds from the start of the video at which each stored hash was
    /// sampled, ascending.
    ///
    /// Milliseconds rather than a sample ordinal, because what a match is worth
    /// is how much *footage* it accounts for and samples are not evenly spread.
    /// Encoders put keyframes where the picture changes, so one sample can stand
    /// for a ten-second static shot and the next for half a second of cuts;
    /// counting them equally reads a re-encode that happens to sample the busy
    /// half as barely overlapping. Two files of the same footage disagree wildly
    /// about how many samples it is worth and not at all about how long it is.
    pub valid_t_start: Vec<u32>,
    /// Milliseconds at which each stored hash stops standing in for the
    /// picture: the next sample's time, or the end of the video for the last.
    pub valid_t_end: Vec<u32>,
    /// The video's runtime in milliseconds -- the denominator every coverage
    /// figure is taken over.
    pub total_ms: u32,
    pub width: u32,
    pub height: u32,
    pub duration: f64,
    pub file_size: u64,
    /// FFmpeg's short name for the codec of the video stream ("h264", "hevc",
    /// "av1", ...). Empty only when FFmpeg cannot name the id it read.
    ///
    /// This is the one field that decides whether two copies' bitrate-derived
    /// numbers mean the same thing, so it is stored rather than re-derived:
    /// ranking must not depend on whether a file happened to be re-opened.
    ///
    /// Appended at the END of the struct on purpose. Cache entries written
    /// before this field existed simply run out of bytes here, so bincode
    /// fails cleanly and the file is fingerprinted again -- no version bump,
    /// no chance of an old payload being misread as a new one.
    pub codec: String,
    /// Average frames per second, or 0.0 when the container never said.
    pub frame_rate: f64,
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
    ///
    /// Reported, but never used to rank: see `quality`.
    pub fn bitrate(&self) -> u64 {
        if self.duration > 0.0 {
            ((self.file_size as f64 * 8.0) / self.duration) as u64
        } else {
            0
        }
    }

    /// Bits spent on an average frame: bitrate divided by frame rate.
    ///
    /// This is what ranks two copies, not raw bitrate. Bitrate answers "how
    /// many bits per second", and a 60 fps copy spends those bits across twice
    /// as many frames as a 30 fps one -- so the same bitrate is half the
    /// picture. Dividing by the frame rate asks how much was spent on each
    /// frame instead, which is much closer to what a viewer actually sees.
    ///
    /// It is still NOT comparable across codecs: 20 kbit/frame of AV1 looks
    /// considerably better than 20 kbit/frame of MPEG-4, and ranking the two
    /// against each other would punish exactly the file that was encoded well.
    /// Only compare this figure between files that share a codec --
    /// `GroupMaxima` is what enforces that, and nothing else should be using
    /// this number to choose what to delete.
    ///
    /// 0 when the frame rate (or duration) is unknown. Read that as "no claim
    /// either way", not as "the worst copy here".
    pub fn quality(&self) -> u64 {
        if self.frame_rate > 0.0 {
            (self.bitrate() as f64 / self.frame_rate) as u64
        } else {
            0
        }
    }
}

/// When each sample was taken, in milliseconds, and the runtime they are
/// measured against.
///
/// The container's own timestamps are used whenever it gave a complete set. If
/// even one is missing the whole video falls back to spacing its samples evenly
/// across the runtime: a half-timed video would otherwise weight the timed part
/// of itself against a guess about the rest, and an even spread is at least the
/// same assumption everywhere. That is also exactly the old accounting, so a
/// container with no timestamps behaves as it always did.
///
/// The runtime comes from the header when it is known. When it is not, the last
/// sample is extended by one average gap -- the last sample stands for footage
/// too, and giving it a zero-length span would quietly delete it from every
/// coverage figure.
fn sample_times(times: &[Option<u32>], duration_sec: f64) -> (Vec<u32>, u32) {
    let n = times.len();
    let known: Option<Vec<u32>> = times.iter().copied().collect();

    let mut total_ms = if duration_sec > 0.0 {
        (duration_sec * 1000.0) as u32
    } else {
        0
    };

    let times = match known {
        Some(t) => t,
        None => {
            // No usable clock: spread the samples evenly over whatever runtime
            // we have, or over a nominal millisecond each if we have none.
            if total_ms == 0 {
                total_ms = n as u32;
            }
            (0..n)
                .map(|i| ((i as u64 * total_ms as u64) / n.max(1) as u64) as u32)
                .collect()
        }
    };

    let last = times.iter().copied().max().unwrap_or(0);
    if total_ms <= last {
        let average_gap = last / n.max(1) as u32;
        total_ms = last.saturating_add(average_gap.max(1));
    }

    (times, total_ms)
}

/// Box-filter weights taking `len` source pixels starting at `offset` down to
/// `THUMB` output samples.
///
/// Every output sample is the *average of the source pixels it covers*, edges
/// included fractionally, rather than a reading of the one or two pixels
/// nearest its centre. That distinction is the whole reason this exists: point
/// sampling a 1080p frame through a 64x64 buffer lands on a handful of
/// individual pixels, and which value those pixels hold is exactly what an
/// encoder is licensed to change. An average over the same area is a statistic
/// of hundreds of source pixels and barely moves.
///
/// `len` can be a little under `THUMB` -- the auto-crop will not act on a span
/// narrower than 16 and then insets it by a pixel each side -- in which case
/// this is an upscale and each output sample reads one source pixel. The tap
/// list is never empty either way.
fn box_weights(offset: usize, len: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = len as f32 / THUMB as f32;
    (0..THUMB)
        .map(|o| {
            let start = o as f32 * scale;
            let end = start + scale;
            let first = start.floor() as usize;
            let last = ((end.ceil() as usize).min(len)).max(first + 1);

            let mut taps: Vec<(usize, f32)> = (first..last)
                .map(|s| {
                    // How much of source pixel `s` falls inside [start, end).
                    let overlap = (end.min(s as f32 + 1.0) - start.max(s as f32)).max(0.0);
                    (offset + s, overlap)
                })
                .collect();

            let total: f32 = taps.iter().map(|(_, w)| w).sum();
            if total > 0.0 {
                for (_, w) in taps.iter_mut() {
                    *w /= total;
                }
            }
            taps
        })
        .collect()
}

/// The DCT-II basis restricted to the `KEEP` lowest frequencies of a `THUMB`
/// wide signal, unnormalised (`cos(pi (2x+1) k / 2N)`).
///
/// Normalisation is left out on purpose: every coefficient is only ever
/// compared against other coefficients of the same frame, so a constant scale
/// factor cancels -- except in `MIN_AC_ENERGY`, which is calibrated against
/// this scale.
fn dct_basis() -> [[f32; THUMB]; KEEP] {
    let mut basis = [[0.0f32; THUMB]; KEEP];
    for (k, row) in basis.iter_mut().enumerate() {
        for (x, cell) in row.iter_mut().enumerate() {
            *cell = (std::f32::consts::PI * (2.0 * x as f32 + 1.0) * k as f32
                / (2.0 * THUMB as f32))
                .cos();
        }
    }
    basis
}

/// Box-filter the cropped region of a 64x64 frame down to `THUMB` x `THUMB`.
///
/// Separable: rows first into `band` (THUMB rows still 64 wide), then columns.
/// Both scratch buffers belong to the caller so a whole video's worth of frames
/// runs without allocating.
fn resample(
    frame: &[u8],
    rows: &[Vec<(usize, f32)>],
    cols: &[Vec<(usize, f32)>],
    band: &mut [f32],
    out: &mut [f32],
) {
    for (oy, taps) in rows.iter().enumerate() {
        let dst = &mut band[oy * 64..oy * 64 + 64];
        dst.fill(0.0);
        for &(sy, w) in taps {
            let src = &frame[sy * 64..sy * 64 + 64];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += s as f32 * w;
            }
        }
    }

    for oy in 0..THUMB {
        let src = &band[oy * 64..oy * 64 + 64];
        for (ox, taps) in cols.iter().enumerate() {
            let mut acc = 0.0;
            for &(sx, w) in taps {
                acc += src[sx] * w;
            }
            out[oy * THUMB + ox] = acc;
        }
    }
}

/// The `KEEP` x `KEEP` lowest-frequency corner of the thumbnail's 2-D DCT, with
/// the DC term zeroed.
///
/// DC is the frame's average brightness, which says nothing about what is in
/// the picture and everything about how the encoder handled levels; dropping it
/// is what makes the hash indifferent to a copy that is a shade darker.
fn low_frequency_block(thumb: &[f32], basis: &[[f32; THUMB]; KEEP]) -> [f32; KEEP * KEEP] {
    // rows: basis * thumb -> KEEP x THUMB
    let mut partial = [[0.0f32; THUMB]; KEEP];
    for (k, prow) in partial.iter_mut().enumerate() {
        for (y, &b) in basis[k].iter().enumerate() {
            let src = &thumb[y * THUMB..y * THUMB + THUMB];
            for (p, &s) in prow.iter_mut().zip(src) {
                *p += b * s;
            }
        }
    }

    // columns: partial * basis^T -> KEEP x KEEP
    let mut out = [0.0f32; KEEP * KEEP];
    for k in 0..KEEP {
        for l in 0..KEEP {
            let mut acc = 0.0;
            for x in 0..THUMB {
                acc += partial[k][x] * basis[l][x];
            }
            out[k * KEEP + l] = acc;
        }
    }
    out[0] = 0.0;
    out
}

/// Whether a frame has too little structure for its hash to mean anything.
fn is_featureless(coefficients: &[f32; KEEP * KEEP]) -> bool {
    let energy: f32 = coefficients.iter().map(|c| c.abs()).sum();
    energy / ((KEEP * KEEP) as f32) < MIN_AC_ENERGY
}

/// One bit per coefficient: is it above the median of the block?
///
/// A median split rather than a sign test or a comparison against a neighbour.
/// It fixes the popcount at 32 whatever the frame is, so two unrelated frames
/// sit ~32 bits apart by construction and the tolerance has the same meaning
/// everywhere -- there is no bright frame whose hash is mostly ones and
/// therefore close to every other bright frame's.
///
/// The fixed popcount has a consequence worth knowing when reading a tolerance:
/// two hashes with the same number of set bits always differ in an EVEN number
/// of places, so `--hamming-distance 5` accepts exactly what 4 does. Odd values
/// are not wrong, just indistinguishable from the even one below them.
fn hash_of(coefficients: &[f32; KEEP * KEEP]) -> u64 {
    let mut sorted = *coefficients;
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = KEEP * KEEP;
    let median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;

    let mut hash = 0u64;
    for (i, &c) in coefficients.iter().enumerate() {
        if c > median {
            hash |= 1 << (63 - i);
        }
    }
    hash
}

/// A `Rational` as a plain float, with FFmpeg's "unknown" (a zero denominator)
/// mapped to 0.0 instead of an infinity or a NaN.
fn rational_f64(r: ffmpeg_next::Rational) -> f64 {
    if r.denominator() == 0 {
        return 0.0;
    }
    r.numerator() as f64 / r.denominator() as f64
}

/// The runtime the container claims, in seconds, or <= 0 when it never said.
///
/// Lifted out of `fingerprint_video` unchanged so the header-completeness check
/// below can ask the same question and get the same answer.
fn duration_seconds(
    stream: &ffmpeg_next::format::stream::Stream<'_>,
    ictx: &ffmpeg_next::format::context::Input,
) -> f64 {
    let mut duration_sec = 0.0;

    let stream_duration = stream.duration();
    if stream_duration >= 0 {
        let tb = stream.time_base();
        if tb.denominator() > 0 {
            duration_sec =
                stream_duration as f64 * (tb.numerator() as f64 / tb.denominator() as f64);
        }
    }

    if duration_sec <= 0.0 {
        // FFmpeg's AV_TIME_BASE format duration.
        duration_sec = ictx.duration() as f64 / 1_000_000.0;
    }

    duration_sec
}

/// Average frames per second, or 0.0 when the container never said.
///
/// The sample-count fallback is what makes the header sufficient for MP4:
/// stts gives the frame count and mdhd the runtime, so the rate is knowable
/// from the index alone without reading a single packet. It only fires where
/// both rate fields are empty -- which today yields "unknown" -- so it can
/// only add information, never change a figure that already existed.
fn frame_rate_of(stream: &ffmpeg_next::format::stream::Stream<'_>, duration_sec: f64) -> f64 {
    let mut frame_rate = rational_f64(stream.avg_frame_rate());
    if frame_rate <= 0.0 {
        frame_rate = rational_f64(stream.rate());
    }
    if frame_rate <= 0.0 && duration_sec > 0.0 {
        let frames = stream.frames();
        if frames > 0 {
            frame_rate = frames as f64 / duration_sec;
        }
    }
    if !frame_rate.is_finite() || frame_rate <= 0.0 || frame_rate > MAX_PLAUSIBLE_FRAME_RATE {
        frame_rate = 0.0;
    }
    frame_rate
}

/// Open the container WITHOUT probing it.
///
/// `ffmpeg_next::format::input` is `avformat_open_input` followed by
/// `avformat_find_stream_info`. The first reads the header; the second reads up
/// to five seconds of EVERY stream, opens a decoder for each, decodes frames
/// through them to discover what it is looking at, and closes them again -- and
/// then we throw all of that away and open our own decoder.
///
/// For a container whose header already answers everything (see
/// `header_is_complete`) that is an entire extra keyframe decode, an entire
/// audio decoder, and a re-read of the head of the file, per file, per run.
fn open_input(filepath: &str) -> Result<ffmpeg_next::format::context::Input> {
    let path = std::ffi::CString::new(filepath)
        .map_err(|_| anyhow!("Path contains an interior NUL byte: {}", filepath))?;

    unsafe {
        let mut ps: *mut ffmpeg_next::ffi::AVFormatContext = std::ptr::null_mut();
        match ffmpeg_next::ffi::avformat_open_input(
            &mut ps,
            path.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ) {
            0 => Ok(ffmpeg_next::format::context::Input::wrap(ps)),
            e => Err(anyhow!(ffmpeg_next::Error::from(e))),
        }
    }
}

/// Whether the header alone answered every question this file will be asked.
///
/// Deliberately conservative: any gap at all falls back to the full probe, so
/// the worst case is exactly today's behaviour. The extradata requirement is
/// the subtle one -- for streams that need parsing (raw H.264 in TS, say) the
/// probe is where the SPS gets extracted into the codec parameters, so a stream
/// that has not already got extradata from its header must not skip it.
fn header_is_complete(ictx: &ffmpeg_next::format::context::Input) -> bool {
    let Some(stream) = ictx.streams().best(ffmpeg_next::media::Type::Video) else {
        return false;
    };

    let (codec_id, width, height, extradata_size) = unsafe {
        let par = (*stream.as_ptr()).codecpar;
        if par.is_null() {
            return false;
        }
        (
            (*par).codec_id,
            (*par).width,
            (*par).height,
            (*par).extradata_size,
        )
    };

    if codec_id == ffmpeg_next::ffi::AVCodecID::AV_CODEC_ID_NONE
        || width <= 0
        || height <= 0
        || extradata_size <= 0
        || stream.time_base().denominator() <= 0
    {
        return false;
    }

    let duration = duration_seconds(&stream, ictx);
    duration > 0.0 && frame_rate_of(&stream, duration) > 0.0
}

/// Fingerprint one video.
///
/// `decode_threads` is this video's share of the process-wide thread budget,
/// decided by the caller from how many decodes are still outstanding. It is
/// clamped to `1..=MAX_DECODE_THREADS`, so passing 0 is harmless.
///
/// `min_duration` (seconds, 0 = off) skips the video entirely if it is shorter
/// than the shortest match we are willing to report — such a file cannot
/// possibly contain a long enough shared clip, so decoding it is pure waste.
/// Returns `Ok(None)` in that case: a skip is not an error, and the caller must
/// not log it as one. A video whose duration cannot be determined is NOT
/// skipped; unknown is not the same as short.
///
/// `progress` is handed the byte offset of each packet of the VIDEO STREAM
/// BEING FINGERPRINTED as the demuxer advances through it, so a caller drawing
/// a bar can move it DURING a decode rather than only when one ends. A single
/// 8 GB file is minutes of work, and a bar that cannot move until it is over is
/// indistinguishable from a hung one. Other streams are deliberately not
/// reported -- see the call site for the cover-art packet that made that
/// distinction matter. Offsets are absolute and reported raw: they are not
/// necessarily monotonic (a seek can land short) and a container that does not
/// track them reports -1, which is filtered here, leaving that file to move the
/// bar once at the end. Rate-limiting is the caller's job -- this fires per
/// packet, and on a linear-scanning container that is thousands of times a
/// second.
pub fn fingerprint_video(
    filepath: &str,
    kf_interval: f64,
    min_kf_samples: f64,
    decode_threads: usize,
    min_duration: f64,
    progress: &dyn Fn(u64),
) -> Result<Option<VideoFingerprint>> {
    // 1. Native Zero-Copy Extraction (No Subprocess Overhead)
    let mut ictx = open_input(filepath)
        .with_context(|| format!("Failed to open video file: {}", filepath))?;

    // The probe is the expensive half of opening a file, and for an ordinary
    // MP4/MKV it is answering questions the header already answered. Run it
    // only when something is genuinely missing.
    if !header_is_complete(&ictx) {
        log::debug!("{}: header incomplete, probing streams.", filepath);
        unsafe {
            let e = ffmpeg_next::ffi::avformat_find_stream_info(
                ictx.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if e < 0 {
                return Err(anyhow!(ffmpeg_next::Error::from(e)))
                    .with_context(|| format!("Failed to read the streams of {}", filepath));
            }
        }
    }

    let input = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| anyhow!("No video stream found in {}", filepath))?;
    let video_stream_index = input.index();

    let duration_sec = duration_seconds(&input, &ictx);

    // Bail before the decoder, the scaler, and the 4 MiB frame buffer exist.
    // Only a positive, known duration can disqualify a file.
    if min_duration > 0.0 && duration_sec > 0.0 && duration_sec < min_duration {
        return Ok(None);
    }

    // --- Codec and frame rate ------------------------------------------------
    // Both come straight out of the stream parameters the demuxer parsed while
    // reading the header, so they cost nothing beyond the open that just
    // happened -- no probe, no decode, no second pass.
    //
    // The codec is stored by NAME rather than as a numeric id: the id is an
    // enum whose values are FFmpeg's business and could be renumbered, while
    // "h264" is stable, is what the user sees in the report, and is what the
    // comparison rules key on.
    //
    // `Id::name` is avcodec_get_name, which answers "what codec is this" from
    // the id alone. It deliberately does NOT go through avcodec_find_decoder:
    // that returns whichever DECODER the local FFmpeg prefers for the id, and
    // reports its name -- "libdav1d" for AV1 on a build that has it, "av1" on
    // one that doesn't, "libvpx-vp9" or "vp9" for the same file depending on
    // how FFmpeg was compiled.
    //
    // That made the recorded codec a property of the machine rather than of the
    // video, which is wrong three times over: the report named a decoder where
    // the user expected a codec, the codec-standoff rule compared those names
    // for equality (so the same AV1 file scanned on two builds looked like two
    // different codecs and deadlocked a group that should have resolved), and
    // the cache stores the string -- so entries written by one build disagreed
    // with entries written by another.
    let codec = input.parameters().id().name().to_string();

    let frame_rate = frame_rate_of(&input, duration_sec);

    let mut context_decoder = ffmpeg_next::codec::context::Context::from_parameters(input.parameters())
        .context("Failed to create codec context from parameters")?;

    let decode_threads = decode_threads.clamp(1, MAX_DECODE_THREADS);

    unsafe {
        let ctx = context_decoder.as_mut_ptr();
        (*ctx).thread_count = decode_threads as i32;
        (*ctx).skip_loop_filter = ffmpeg_next::ffi::AVDiscard::AVDISCARD_ALL;
        (*ctx).flags2 |= ffmpeg_next::ffi::AV_CODEC_FLAG2_FAST as i32;
        if decode_threads > 1 {
            // Frame threading preferred; slice threading is the fallback for
            // codecs that don't advertise AV_CODEC_CAP_FRAME_THREADS. FFmpeg
            // picks between them itself from the codec's capabilities.
            (*ctx).thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
        } else {
            (*ctx).flags |= ffmpeg_next::ffi::AV_CODEC_FLAG_LOW_DELAY as i32;
        }
        (*ctx).skip_frame = ffmpeg_next::ffi::AVDiscard::AVDISCARD_NONKEY;
    }

    let mut decoder = context_decoder.decoder().video()
        .context("Failed to initialize video decoder")?;

    let file_size = std::fs::metadata(filepath).map(|m| m.len()).unwrap_or(0);

    // Built from the first frame the decoder produces rather than from the
    // header. Without the probe the container never reports a pixel format --
    // and this was always the more honest source anyway, since the decoder, not
    // the header, is the authority on what it is emitting.
    let mut scaler: Option<ffmpeg_next::software::scaling::context::Context> = None;
    let mut frame_dims: Option<(u32, u32)> = None;

    // All unique frames packed back-to-back, FRAME_STRIDE bytes each. One growable
    // allocation instead of N tiny ones -> no heap fragmentation/retention, and the
    // whole thing is released to the OS when this function returns.
    let mut u_frames: Vec<u8> = Vec::with_capacity(FRAME_STRIDE * 64);
    // When each kept frame is shown, in milliseconds. `None` for a frame the
    // container gave no timestamp for; a video with any of those falls back to
    // spacing its samples evenly (see `sample_times`).
    let mut unique_frame_times: Vec<Option<u32>> = Vec::new();

    let mut sum = vec![0u64; 64 * 64];
    let mut sum_sq = vec![0u64; 64 * 64];

    // Seconds per unit of the video stream's timestamps. Zero when the header
    // never said, which makes every frame time unknown.
    let stream_time_base = {
        let tb = ictx.stream(video_stream_index).unwrap().time_base();
        if tb.denominator() > 0 {
            tb.numerator() as f64 / tb.denominator() as f64
        } else {
            0.0
        }
    };

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
        if scaler.is_none() {
            scaler = Some(ffmpeg_next::software::scaling::context::Context::get(
                dec.format(),
                dec.width(),
                dec.height(),
                ffmpeg_next::format::Pixel::GRAY8,
                64,
                64,
                ffmpeg_next::software::scaling::flag::Flags::FAST_BILINEAR,
            )?);
            frame_dims = Some((dec.width(), dec.height()));
        }
        let scaler = scaler.as_mut().unwrap();

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

            // Read off the FRAME rather than the packet, because with frame
            // threading a frame surfaces several packets after the one that
            // carried it and the packet in hand when it arrives is somebody
            // else's. What the frame carries is whatever the demux loop stamped
            // onto its packet -- see `sample_ts` there -- so this is the
            // container's own clock making the round trip, not the decoder's
            // opinion of it.
            unique_frame_times.push(match dec.timestamp() {
                Some(pts) if stream_time_base > 0.0 && pts >= 0 => {
                    Some((pts as f64 * stream_time_base * 1000.0) as u32)
                }
                _ => None,
            });

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

    // Non-key video packets are skipped, and the audio and subtitle streams are
    // dropped outright: their payloads used to be read off disk, allocated and
    // copied into a packet on every iteration of the demux loop below, purely
    // so the `stream.index()` test could throw them away again.
    unsafe {
        let fmt_ctx = ictx.as_mut_ptr();
        for i in 0..(*fmt_ctx).nb_streams as usize {
            let stream_ptr = *(*fmt_ctx).streams.add(i);
            (*stream_ptr).discard = if i == video_stream_index {
                ffmpeg_next::ffi::AVDiscard::AVDISCARD_NONKEY
            } else {
                ffmpeg_next::ffi::AVDiscard::AVDISCARD_ALL
            };
        }
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

    // --- Rapid demuxing: only key packets ever reach the decoder --------------
    //
    // `AVDISCARD_NONKEY` on the stream asks the demuxer to drop everything else
    // before it is ever assembled into a packet, and MP4/MOV honours that by
    // walking its sample table and reading only the sync samples -- roughly a
    // tenth of the file. Matroska has no such shortcut: it hands back every
    // block in the file, payload and all, and we throw ~90% of the bytes away.
    // On a 714 MB episode that is 615 MB read, allocated and copied for nothing,
    // and it lands on the one thread that also has to feed the decoder.
    //
    // So when a demuxer proves it is going to do that -- by handing us a non-key
    // packet we already asked it not to -- we stop reading forward and start
    // SEEKING from one keyframe to the next instead. The seek costs an index
    // lookup and lands on exactly the keyframe a linear scan would have reached,
    // having read none of the bytes in between.
    //
    // Two properties keep this from changing what gets fingerprinted:
    //
    //   * A seek that lands short (or on a keyframe already decoded) is caught
    //     by the `ts <= last` guard below, so no frame is ever hashed twice.
    //   * Anything that goes wrong -- a container with no index, a stream with
    //     no timestamps, a seek that simply fails -- sets `seeking_broken` and
    //     the loop finishes as a plain linear scan, which is exactly the old
    //     behaviour.
    //
    // The decoder is deliberately NOT flushed across a seek. Every packet it is
    // fed is a keyframe and therefore self-contained, so there is no state to
    // invalidate, and flushing would throw away frames still in flight.
    let mut packet = ffmpeg_next::codec::packet::Packet::empty();
    let mut last_key_ts: Option<i64> = None;
    // The first keyframe's timestamp, which every later one is measured from.
    let mut first_key_ts: Option<i64> = None;
    let mut demuxer_ignores_discard = false;
    let mut seeking_broken = false;

    loop {
        if shutdown_requested() {
            return Err(anyhow!("Interrupted while fingerprinting {}", filepath));
        }

        // `av_read_frame` fills the packet it is handed WITHOUT releasing what
        // that packet already holds -- it moves a reference in over the top of
        // the old one, which is then unreachable and never freed. The packet
        // iterator this loop replaced hid that by allocating a fresh packet per
        // read and dropping it (ffmpeg-next's `Drop` is the unref); reusing one
        // packet, as this loop does to avoid that churn, has to say it out loud.
        //
        // Every key packet's payload leaked otherwise -- ~100 KB a frame, held
        // for the life of the process -- which is a gigabyte across a few
        // thousand keyframes and is why RSS climbed for the whole of a run
        // instead of levelling off.
        //
        // Unreffing at the TOP rather than after `send_packet` is what makes it
        // exhaustive: every `continue` below (wrong stream, non-key, a seek
        // landing short, a demux error) leaves a packet in hand, and all of them
        // come back through here.
        unsafe {
            ffmpeg_next::ffi::av_packet_unref(packet.as_mut_ptr());
        }

        match packet.read(&mut ictx) {
            Ok(()) => {}
            Err(ffmpeg_next::Error::Eof) => break,
            // Mirrors the packet iterator this loop replaces: a recoverable
            // demux error skips the packet rather than ending the video.
            Err(_) => continue,
        }

        if packet.stream() != video_stream_index {
            continue;
        }

        // Reported after the stream filter and before the keyframe one, so it
        // follows the stream we are actually walking. On a container we seek
        // through the offset leaps a keyframe at a time; on one we scan
        // linearly it creeps, and either way it is the truthful answer to "how
        // far into this file are we". Two pointer derefs and an indirect call
        // -- the caller decides how much of that is worth drawing.
        //
        // Reporting EVERY packet's offset, which is what this used to do, is
        // not a wider version of the same measurement: an attached picture
        // (cover art, a poster frame) is a stream of its own whose single
        // packet libavformat hands back FIRST, ahead of any demuxing, and it is
        // stored at the END of the file. One 5 GB MP4 with cover art therefore
        // opened by reporting offset 5321112043 of 5321218376 before a frame
        // had been decoded, which credited the whole file to the bar in one
        // step and then left it motionless for the 100 seconds the decode
        // actually took. Any stream laid out away from the video's bytes does
        // the same thing to a smaller degree.
        let pos = packet.position();
        if pos >= 0 {
            progress(pos as u64);
        }

        if !packet.is_key() {
            // The discard hint was ignored. Note it: from the next keyframe on
            // we jump rather than read.
            demuxer_ignores_discard = true;
            continue;
        }

        // The timestamp the container's index is keyed on, which is what a seek
        // target is measured against.
        let index_ts = packet.dts().or_else(|| packet.pts());

        // A seek may land on or before a keyframe already decoded. Skip those
        // rather than hashing the same frame twice.
        if let (Some(ts), Some(last)) = (index_ts, last_key_ts) {
            if ts <= last {
                continue;
            }
        }

        // --- When this keyframe happens -------------------------------------
        //
        // Measured from the first keyframe rather than from zero, and taken
        // from `index_ts` -- decode order -- rather than from the packet's
        // presentation timestamp. Both of those are deliberate, and the reason
        // is that `AVDISCARD_NONKEY` is not free.
        //
        // MP4 stores presentation time as a per-sample offset from decode time
        // (the `ctts` table) and the demuxer walks that table with a cursor
        // that only advances over samples it RETURNS. Discarding the non-key
        // samples desynchronises it, so the pts we are handed is a real offset
        // belonging to the wrong sample: on one 2-second clip the container
        // says its keyframes are at 0 / 29029 / 58058 and libavformat reports
        // 0 / 32032 / 57057 -- 100 ms late and then 34 ms early. Matroska has
        // no such table and its pts is exact; AVI has no pts at all.
        //
        // Decode time is immune. It comes out of a plain cumulative sum
        // (`stts`) with no cursor to lose, and it was correct in every
        // container tested. It differs from presentation time by the codec's
        // reorder delay, which is constant across a file's keyframes -- so
        // subtracting the first sample's value cancels it exactly, and what is
        // left reproduces the container's own presentation times. On the clip
        // above it gives 0 / 29029 / 58058, to the tick.
        //
        // Anchoring also settles two things the raw clock got wrong: a first
        // keyframe with a negative dts (an MP4 with B-frames opens at -1001)
        // used to fail the `pts >= 0` test and drop the whole video to evenly
        // spaced samples, and a stream whose clock does not start at zero used
        // to measure its samples against a runtime that does.
        let sample_ts = match (index_ts, first_key_ts) {
            (Some(ts), Some(first)) => Some(ts.saturating_sub(first).max(0)),
            (Some(ts), None) => {
                first_key_ts = Some(ts);
                Some(0)
            }
            (None, _) => None,
        };

        // Hand the corrected clock to the decoder, which copies a packet's pts
        // onto the frame it produces. That is what carries it back out through
        // frame threading, where nothing else can pair the two up.
        if let Some(ts) = sample_ts {
            packet.set_pts(Some(ts));
        }

        // Seconds, for the interval rule.
        let show_t = if stream_time_base > 0.0 {
            sample_ts.map(|ts| ts as f64 * stream_time_base)
        } else {
            None
        };

        let mut kept = true;
        if effective_interval > 0.0 {
            if let Some(t) = show_t {
                if let Some(last) = last_kept_t {
                    if t - last < effective_interval {
                        kept = false; // too close to the last kept keyframe
                    }
                }
                if kept {
                    last_kept_t = Some(t);
                }
            }
            // If PTS/DTS is missing we fall through and keep the frame (safe default).
        }

        if kept && decoder.send_packet(&packet).is_ok() {
            while decoder.receive_frame(&mut decoded).is_ok() {
                let _ = process_frame(&decoded);
            }
        }

        if let Some(ts) = index_ts {
            last_key_ts = Some(ts);
        }

        // --- Jump to the next keyframe we actually want ----------------------
        if demuxer_ignores_discard && !seeking_broken {
            match index_ts {
                // No clock to seek by. Reading forward is the only option left.
                None => seeking_broken = true,
                Some(ts) => {
                    // One tick past the current keyframe is the least that can
                    // land on a new one. When an interval is in force, skip the
                    // whole rest of it in the same hop -- the keyframes in
                    // between were going to be read and discarded.
                    let mut step = 1i64;
                    if effective_interval > 0.0 && stream_time_base > 0.0 {
                        if let (Some(t), Some(last)) = (show_t, last_kept_t) {
                            let remaining = last + effective_interval - t;
                            if remaining > 0.0 {
                                step = step.max((remaining / stream_time_base).ceil() as i64);
                            }
                        }
                    }

                    let ok = unsafe {
                        ffmpeg_next::ffi::av_seek_frame(
                            ictx.as_mut_ptr(),
                            video_stream_index as i32,
                            ts.saturating_add(step),
                            0,
                        ) >= 0
                    };
                    if !ok {
                        // Past the last keyframe a forward seek fails, which is
                        // indistinguishable from a container that cannot seek at
                        // all. Either way, finish by reading.
                        seeking_broken = true;
                    }
                }
            }
        }
    }

    // With frame threading the decoder holds up to thread_count frames back, so
    // this drain is not a formality -- on a short video it can be where MOST of
    // the frames arrive.
    let _ = decoder.send_eof();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let _ = process_frame(&decoded);
    }

    if frame_idx == 0 {
        return Err(anyhow!("No valid frames found or successfully decoded in {}", filepath));
    }

    // From the frames the decoder actually produced. Identical to what the
    // probe used to leave in the stream parameters -- it copies the decoder's
    // view back into them -- and correct even where a header disagrees with its
    // own bitstream.
    let (width, height) = frame_dims.unwrap_or((decoder.width(), decoder.height()));

    // u_frames holds n_unique * FRAME_STRIDE bytes; the frame count is the time list.
    let n_unique = unique_frame_times.len();
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

    // The crop rectangle is fixed for the whole video, so the resampler and the
    // transform basis are built once here and reused for every frame.
    let rows = box_weights(y1, crop_h);
    let cols = box_weights(x1, crop_w);
    let basis = dct_basis();

    let mut hashes = Vec::with_capacity(n_unique);
    let mut flat = Vec::with_capacity(n_unique);
    let mut thumb = [0.0f32; THUMB * THUMB];
    let mut band = [0.0f32; 64 * THUMB];

    // Iterate the flat buffer in FRAME_STRIDE-sized windows; each `frame` is a
    // &[u8] of length 4096.
    for frame in u_frames.chunks_exact(FRAME_STRIDE) {
        resample(frame, &rows, &cols, &mut band, &mut thumb);
        let coefficients = low_frequency_block(&thumb, &basis);
        hashes.push(hash_of(&coefficients));
        flat.push(is_featureless(&coefficients));
    }

    // Frame pixels are no longer needed; release the large buffer (munmap) now
    // rather than at end of scope, trimming the peak during the cheap tail work.
    drop(u_frames);

    let (times, total_ms) = sample_times(&unique_frame_times, duration_sec);

    // Keyframes do not always leave the decoder in presentation order -- an open
    // GOP or a B-pyramid can put a later one out first -- and every span below
    // is "until the next sample", so the samples are put in time order before
    // anything is measured from them.
    let mut order: Vec<usize> = (0..hashes.len()).collect();
    order.sort_by_key(|&i| times[i]);

    let mut changes_hashes = Vec::new();
    let mut changes_t_start = Vec::new();
    let mut changes_valid = Vec::new();

    for (k, &i) in order.iter().enumerate() {
        let h = hashes[i];
        let valid = !flat[i];

        let should_push = k == 0 || {
            let prev = order[k - 1];
            h != hashes[prev] || valid != !flat[prev]
        };

        if should_push {
            changes_hashes.push(h);
            changes_t_start.push(times[i]);
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
                final_t_end.push(total_ms);
            }
        }
    }

    Ok(Some(VideoFingerprint {
        path: filepath.to_string(),
        valid_hashes: final_hashes,
        valid_t_start: final_t_start,
        valid_t_end: final_t_end,
        total_ms,
        width,
        height,
        duration: duration_sec,
        file_size,
        codec,
        frame_rate,
    }))
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

    fn fixture_path() -> PathBuf {
        fixture_named("test_video.mp4")
    }

    fn fixture_named(name: &str) -> PathBuf {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests");
        p.push("fixtures");
        p.push(name);
        p
    }

    fn fingerprint_fixture(threads: usize) -> VideoFingerprint {
        init_ffmpeg_for_tests();

        let fixture_path = fixture_path();
        let filepath = fixture_path.to_string_lossy().to_string();
        assert!(fixture_path.exists(), "Fixture video not found at: {}.", filepath);

        let result = fingerprint_video(&filepath, 0.0, 4.0, threads, 0.0, &|_| {});
        assert!(result.is_ok(), "Failed to fingerprint video: {:?}", result.err());
        result.unwrap().expect("min_duration is off, nothing should be skipped")
    }

    /// A fingerprint with nothing in it but the fields the quality figure needs.
    fn timing_only(size: u64, duration: f64, frame_rate: f64) -> VideoFingerprint {
        VideoFingerprint {
            path: "mock.mp4".to_string(),
            valid_hashes: vec![],
            valid_t_start: vec![],
            valid_t_end: vec![],
            total_ms: 1,
            width: 1920,
            height: 1080,
            duration,
            file_size: size,
            codec: "h264".to_string(),
            frame_rate,
        }
    }

    #[test]
    fn test_video_shorter_than_min_duration_is_skipped() {
        init_ffmpeg_for_tests();
        let filepath = fixture_path().to_string_lossy().to_string();

        // The fixture is ~1s; an hour-long floor must reject it without error.
        let skipped = fingerprint_video(&filepath, 0.0, 4.0, 1, 3600.0, &|_| {}).unwrap();
        assert!(skipped.is_none(), "a short video must be skipped, not fingerprinted");
    }

    /// The hook exists so a bar can move during a decode rather than only when
    /// one ends, and the way that silently regresses is the hook never firing --
    /// which looks identical from outside except that a long file appears hung.
    ///
    /// What this fixture can prove is that it fires from inside the demux loop
    /// with real offsets into the file. It cannot prove the offsets ADVANCE: it
    /// is a one-second clip with a single keyframe, so there is exactly one
    /// packet to report. Movement across a multi-keyframe file is a property of
    /// the demuxer's own position, not of this plumbing.
    #[test]
    fn test_progress_reports_real_file_offsets_during_the_decode() {
        init_ffmpeg_for_tests();
        let path = fixture_path();
        let size = std::fs::metadata(&path).unwrap().len();
        let filepath = path.to_string_lossy().to_string();

        let offsets = std::cell::RefCell::new(Vec::new());
        fingerprint_video(&filepath, 0.0, 4.0, 1, 0.0, &|pos| {
            offsets.borrow_mut().push(pos)
        })
        .unwrap()
        .expect("the fixture fingerprints");

        let offsets = offsets.into_inner();
        assert!(!offsets.is_empty(), "the demuxer walked the file and reported nothing");
        // A caller sizes its bar by the file's length on disk, so an offset past
        // that would push the bar beyond full.
        assert!(
            offsets.iter().all(|&pos| pos < size),
            "offsets must fall inside the {} byte file: {:?}",
            size,
            offsets
        );
    }

    /// An attached picture -- cover art, a poster frame -- is a stream of its
    /// own, libavformat hands its single packet back before it demuxes
    /// anything, and MP4 writes it after the video's samples. Reporting every
    /// packet's offset therefore opened such a file by announcing an offset
    /// near its end, which credited the whole file to the caller's progress bar
    /// before a frame had been decoded and left it motionless for the rest of
    /// the decode. On one 5 GB MP4 that was offset 5321112043 of 5321218376,
    /// reported 0.2 s into a 100 s decode.
    ///
    /// The fixture is `test_video.mp4` with a JPEG attached (`-map 1 -c copy
    /// -disposition:v:1 attached_pic`); the cover sits at byte 5760 of 6434,
    /// past every packet of the video stream.
    #[test]
    fn test_progress_ignores_streams_other_than_the_video() {
        init_ffmpeg_for_tests();
        let path = fixture_named("test_video_cover_art.mp4");
        let size = std::fs::metadata(&path).unwrap().len();
        let filepath = path.to_string_lossy().to_string();

        let offsets = std::cell::RefCell::new(Vec::new());
        fingerprint_video(&filepath, 0.0, 4.0, 1, 0.0, &|pos| {
            offsets.borrow_mut().push(pos)
        })
        .unwrap()
        .expect("the fixture fingerprints");

        let offsets = offsets.into_inner();
        assert!(!offsets.is_empty(), "the demuxer walked the file and reported nothing");
        assert!(
            offsets[0] < size / 2,
            "the first offset reported for a {} byte file was {}, which is the \
             cover art at the end of it rather than the video at the start",
            size,
            offsets[0]
        );
        assert!(
            offsets.iter().all(|&pos| pos < size),
            "offsets must fall inside the {} byte file: {:?}",
            size,
            offsets
        );
    }

    #[test]
    fn test_fingerprint_real_video() {
        let fp = fingerprint_fixture(1);

        // Assert the properties of the generated fingerprint
        assert!(fp.total_ms > 0, "Video should have a measured runtime");
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

        // The two fields the ranking rules key on. A file whose codec could not
        // be named would silently land in its own comparison bucket, so this is
        // worth asserting on a real container rather than only on mocks.
        assert!(!fp.codec.is_empty(), "the stream's codec must be recorded");
        assert!(
            fp.frame_rate >= 0.0 && fp.frame_rate <= MAX_PLAUSIBLE_FRAME_RATE,
            "an implausible frame rate must be recorded as unknown, got {}",
            fp.frame_rate
        );
    }

    #[test]
    fn test_a_codec_is_named_by_the_codec_not_by_the_local_decoder() {
        // The regression this exists to prevent. Reading the name off
        // avcodec_find_decoder reported "libdav1d" for AV1 on any build that
        // ships dav1d, and "av1" on one that doesn't -- so the string recorded
        // in the cache and compared by the standoff rule depended on how the
        // machine's FFmpeg was compiled rather than on the file.
        //
        // Asserted over the ids most likely to have a differently-named
        // preferred decoder, because those are exactly the ones that broke: AV1
        // has libdav1d and libaom-av1, VP8/VP9 have the libvpx pair.
        use ffmpeg_next::codec::Id;

        for (id, expected) in [
            (Id::AV1, "av1"),
            (Id::VP9, "vp9"),
            (Id::VP8, "vp8"),
            (Id::H264, "h264"),
            (Id::HEVC, "hevc"),
        ] {
            assert_eq!(
                id.name(),
                expected,
                "a codec must be named after itself, whatever decodes it here"
            );
        }
    }

    #[test]
    fn test_quality_is_bitrate_spent_per_frame() {
        // 7,864,320 bytes over 60s = 1,048,576 bits/s. At 32 fps each frame
        // gets 32,768 of them.
        let fp = timing_only(7_864_320, 60.0, 32.0);
        assert_eq!(fp.bitrate(), 1_048_576);
        assert_eq!(fp.quality(), 32_768);
    }

    #[test]
    fn test_quality_is_unknown_rather_than_zero_when_the_frame_rate_is() {
        // A container that never reported a frame rate makes no claim about how
        // its bits are spread. 0 is the caller's cue to skip the metric, which
        // is why nothing here divides by a guessed rate.
        let fp = timing_only(7_864_320, 60.0, 0.0);
        assert!(fp.bitrate() > 0, "the bitrate is still perfectly knowable");
        assert_eq!(fp.quality(), 0);
    }

    #[test]
    fn test_threaded_decode_is_bit_identical() {
        // The whole point of the thread budget is that it is a pure speed knob.
        // Frame threading changes WHEN frames come out of the decoder, never
        // which frames or in what order, so every field of the fingerprint --
        // and therefore every cache key's payload -- must be unchanged.
        let single = fingerprint_fixture(1);
        let threaded = fingerprint_fixture(4);

        assert_eq!(single.total_ms, threaded.total_ms, "runtime must not depend on thread count");
        assert_eq!(single.valid_hashes, threaded.valid_hashes, "hashes must not depend on thread count");
        assert_eq!(single.valid_t_start, threaded.valid_t_start, "hash start times must not depend on thread count");
        assert_eq!(single.valid_t_end, threaded.valid_t_end, "hash end times must not depend on thread count");
        assert_eq!(single.codec, threaded.codec, "codec must not depend on thread count");
        assert_eq!(single.frame_rate, threaded.frame_rate, "frame rate must not depend on thread count");
    }

    #[test]
    fn test_zero_threads_is_treated_as_one() {
        // A caller that computes a share of 0 must not hand FFmpeg
        // thread_count = 0, which means "autodetect" and would quietly blow past
        // the user's -t budget.
        let fp = fingerprint_fixture(0);
        assert!(!fp.valid_hashes.is_empty());
    }

    #[test]
    fn test_an_ordinary_container_needs_no_probe() {
        // The whole point of the change: a normal MP4 answers everything from
        // its header, so avformat_find_stream_info -- an extra keyframe decode
        // and a whole audio decoder, per file -- is never called. If this ever
        // fails, the fast path has silently stopped applying.
        init_ffmpeg_for_tests();
        let ictx = open_input(&fixture_path().to_string_lossy()).unwrap();
        assert!(header_is_complete(&ictx), "an ordinary MP4 must not need probing");
    }
}