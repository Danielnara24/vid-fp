use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

mod keyframes;
pub(crate) use keyframes::{clock_may_restart, decode_stamp, ticks_to_ms, SampleClock};
use keyframes::{seconds_per_tick, Captured, FrameCapture, KeyframeDecoder, Keyframes, Thinning};
#[cfg(test)]
use keyframes::{feed_keyframe, is_decoding_error, take_ready_frames};

/// Side of the working frame every keyframe is scaled to before it is hashed.
///
/// 128 rather than the 64 it was. The thumbnails the hashes are read from stay
/// 16x16 and 32x32, so on a whole frame this changes little; what it changes
/// is every hash taken inside PART of a frame. A reframing reads a sub-rectangle
/// -- the 16:9 band of a vertical re-upload, a 9:16 strip, an inset -- and at 64
/// that was ~20 lines, fewer than the 32 the detail thumbnail needs, so the
/// detail hash of a reframed picture was mostly interpolation. The detail hash
/// of a whole frame sharpens too, and that is what separates look-alikes.
/// Generated corpus, `-d 6 -p 20`: R 0.839 -> 0.864 with false links 18 -> 3.
/// 256 bought little more for a 450 s cold scan and 1.5 GB of peak memory.
pub(crate) const SIDE: usize = 128;
const FRAME_STRIDE: usize = SIDE * SIDE;

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
/// The auto-cropped region of the working frame (`SIDE` x `SIDE`) is box-filtered down to
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
const MIN_AC_ENERGY: f32 = 50.0;

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
    /// The hash each sample would have had if the picture were mirrored left to
    /// right, index for index with `valid_hashes`. A mirrored copy is the most
    /// common re-upload trick there is, and a DCT hash is not mirror-invariant:
    /// mirroring negates every coefficient with an odd horizontal frequency, so
    /// this is computed from the same coefficients, with no second decode.
    /// Comparison measures each pair both ways round (see `compare`).
    pub mirrored_hashes: Vec<u64>,
    /// A second hash of each sample, from the frequencies just above the ones
    /// `valid_hashes` reads (a 32x32 thumbnail, see `detail_coefficients`),
    /// plain and mirrored. Two different frames that share a coarse layout --
    /// two lines of white text on black, two road scenes -- agree in the low
    /// frequencies and not in these, while two encodes of one frame agree in
    /// both, so comparison reads it to confirm a match. Empty in a fixture,
    /// which then confirms nothing on it.
    pub detail_hashes: Vec<u64>,
    pub mirrored_detail_hashes: Vec<u64>,
    /// Each sample hashed again inside a smaller part of the picture, one entry
    /// per `REFRAMINGS`, so a copy cut down to that part can be found.
    pub reframed: Vec<Reframed>,
    /// The share of the frame inside its bars, the largest any usable sample
    /// shows (`frame_crop`, `largest_share`), in (0, 1]; 0 when not measured
    /// (a fixture).
    ///
    /// What the Resolution ranking reads (`utils::resolution`), rather than the
    /// stored width x height: a letterboxed 1920x1080 copy and the same film
    /// cropped to 1920x800 hold the same picture, and on stored pixels the bars
    /// alone won the first by 26%. Each copy's own crop finds the same picture,
    /// so the products agree to within the line `frame_crop` peels beside each
    /// bar (under 1% a side at `SIDE` = 128). The reported width and height
    /// stay the stored ones.
    pub picture: f32,
    /// What this run measured about the file on disk beside its fingerprint
    /// (`Measured`), or `None` where nothing did (a fixture). Never stored.
    ///
    /// One pointer rather than the fields themselves, and that is measured:
    /// growing this struct by the 32 bytes `Measured` holds slowed phase 1 of
    /// the comparison by ~30% on VIDS (7.7 s -> 10.1 s warm, three interleaved
    /// runs each, an unrelated change to the same build moving nothing), almost
    /// certainly because the fields every pair check reads then straddle one
    /// more cache line across ~50 million checks. A boxed `Option` is 8 bytes
    /// and measured level with the struct without it.
    #[serde(skip)]
    pub measured: Option<Box<Measured>>,
}

/// What a run measures about a file on disk, beside the fingerprint it read
/// from the cache or decoded: facts about this moment rather than about the
/// footage, so they are never stored.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Measured {
    /// The bytes of the video stream alone, when the container states them
    /// (`video_stream_bytes`); 0 when it does not. What `quality` and the Size
    /// ranking read in place of `file_size` whenever every file they are
    /// compared with has one too (`utils::GroupMaxima`), so a cover picture, a
    /// second audio track or subtitles are not ranked as picture. Measured
    /// from the header of each file a comparison matched, just before ranking
    /// (`lib::run`); storing it would cost every cache a full re-decode for a
    /// figure that takes one header read of the few files that are ranked.
    pub video_bytes: u64,
    /// The file's change time (seconds, nanoseconds) as this run's walk saw
    /// it. The disposal pass refuses a file, or a holder, whose change time
    /// has moved since: replaced by another of the same length, even with
    /// `cp -p` or `rsync -a`, which put back an mtime but cannot put back a
    /// ctime (`export::changed_since`).
    pub ctime: Option<(i64, i64)>,
}

/// The samples of a fingerprint that a reframing applies to, and their hashes
/// under it: main, mirrored, detail, mirrored detail.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Reframed {
    /// Indices into `valid_hashes`, ascending.
    pub samples: Vec<u32>,
    pub hashes: Vec<[u64; 4]>,
}

impl VideoFingerprint {
    /// `Measured::video_bytes`, or 0 when nothing measured it.
    pub fn video_bytes(&self) -> u64 {
        self.measured.as_ref().map_or(0, |m| m.video_bytes)
    }

    /// `Measured::ctime`, or `None` when nothing measured it.
    pub fn ctime(&self) -> Option<(i64, i64)> {
        self.measured.as_ref().and_then(|m| m.ctime)
    }

    /// What this run has measured about the file, to be added to.
    pub fn measured_mut(&mut self) -> &mut Measured {
        self.measured.get_or_insert_with(Box::default)
    }

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

    /// Bits spent on an average frame of the picture: the video stream's
    /// bits per second, divided by the frame rate.
    ///
    /// This is what ranks two copies, not raw bitrate. Bitrate answers "how
    /// many bits per second", and a 60 fps copy spends those bits across twice
    /// as many frames as a 30 fps one -- so the same bitrate is half the
    /// picture. Dividing by the frame rate asks how much was spent on each
    /// frame instead, which is much closer to what a viewer actually sees.
    ///
    /// Counted from the video stream's own bytes where the container states
    /// them (`video_bytes`), and from the whole file where it does not. The
    /// whole file counts audio, subtitles and a cover picture as picture, so a
    /// copy with a second audio track, or an embedded JPEG, outranked the same
    /// video stream without one -- a byte-identical stream marked DELETE beside
    /// it. The ranking never mixes the two readings within a codec
    /// (`utils::GroupMaxima`); this one, for display, takes the better reading
    /// each file has.
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
        self.quality_from(match self.video_bytes() {
            0 => self.file_size,
            stream => stream,
        })
    }

    /// Bits per frame, counting `bytes` as the file's picture.
    pub fn quality_from(&self, bytes: u64) -> u64 {
        if self.frame_rate > 0.0 && self.duration > 0.0 {
            // Truncated to whole bits a second first, as `bitrate` is, so a
            // file measured on its whole size reads exactly as it always has.
            let per_second = ((bytes as f64 * 8.0) / self.duration) as u64;
            (per_second as f64 / self.frame_rate) as u64
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
        // `n` samples spanning 0..last have `n - 1` gaps between them, not `n`.
        // Dividing by `n` understates the spacing by a factor of (n-1)/n -- 17%
        // at six samples -- and the figure is used to give the LAST sample a
        // span, so the tail of such a video was consistently credited with less
        // footage than it stands for.
        //
        // Reached more often than the doc comment above suggests: not only by a
        // container that reported no runtime, but by any file whose last sample
        // lands past the runtime it did report. An MP4 that opens on a negative
        // dts is the common way in, since anchoring at the first keyframe shifts
        // every sample later by that amount. Three of the 727 files in the local
        // corpus arrive here with a perfectly good duration.
        let gaps = (n as u32).saturating_sub(1).max(1);
        let average_gap = last / gaps;
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
/// sampling a 1080p frame through a small buffer lands on a handful of
/// individual pixels, and which value those pixels hold is exactly what an
/// encoder is licensed to change. An average over the same area is a statistic
/// of hundreds of source pixels and barely moves.
///
/// `len` can be a little under `THUMB` -- the auto-crop will not act on a span
/// narrower than 16 and then insets it by a pixel each side -- in which case
/// this is an upscale and each output sample reads one source pixel. The tap
/// list is never empty either way.
fn box_weights(offset: usize, len: usize) -> Vec<Vec<(usize, f32)>> {
    box_weights_to(offset, len, THUMB)
}

/// `box_weights` down to `out` samples rather than `THUMB`.
fn box_weights_to(offset: usize, len: usize, out: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = len as f32 / out as f32;
    (0..out)
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

/// Box-filter the cropped region of a working frame down to `THUMB` x `THUMB`.
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
        let dst = &mut band[oy * SIDE..oy * SIDE + SIDE];
        dst.fill(0.0);
        for &(sy, w) in taps {
            let src = &frame[sy * SIDE..sy * SIDE + SIDE];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += s as f32 * w;
            }
        }
    }

    for oy in 0..THUMB {
        let src = &band[oy * SIDE..oy * SIDE + SIDE];
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

/// A Matroska track's own runtime, from the `DURATION` tag mkvmerge and FFmpeg
/// write per track (`HH:MM:SS.nnnnnnnnn`), or `None` when there is none.
///
/// Matroska has no per-stream duration field, so libavformat leaves
/// `stream.duration` unset and the only runtime it reports is the SEGMENT's --
/// the longest track, which is very often the audio. A 60 s picture with 61.5 s
/// of sound read as 61.5 s, a full second and a half past the MP4 copy of the
/// same picture, and that was enough to win the length ranking for a 360p
/// copy over a 1080p one. The tag is the video track's own figure.
fn tagged_track_duration(stream: &ffmpeg_next::format::stream::Stream<'_>) -> Option<f64> {
    let metadata = stream.metadata();
    let value = metadata
        .iter()
        .find(|(k, _)| {
            let k = k.to_ascii_uppercase();
            k == "DURATION" || k.starts_with("DURATION-")
        })
        .map(|(_, v)| v.to_string())?;
    parse_clock(&value)
}

/// `HH:MM:SS(.fraction)` as seconds, for `tagged_track_duration`.
fn parse_clock(value: &str) -> Option<f64> {
    let mut parts = value.trim().split(':');
    let (h, m, s) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let h: f64 = h.parse().ok()?;
    let m: f64 = m.parse().ok()?;
    let s: f64 = s.parse().ok()?;
    let total = h * 3600.0 + m * 60.0 + s;
    (total.is_finite() && total > 0.0 && h >= 0.0 && m >= 0.0 && s >= 0.0).then_some(total)
}

/// The runtime the container claims, in seconds, or <= 0 when it never said.
///
/// Lifted out of `fingerprint_video` unchanged so the header-completeness check
/// below can ask the same question and get the same answer.
///
/// The video stream's own figure first, then a Matroska track's `DURATION`
/// tag (`tagged_track_duration`), and only then the container's, which spans
/// every track and so includes an audio track that runs past the picture.
pub(crate) fn duration_seconds(
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
        if let Some(tagged) = tagged_track_duration(stream) {
            duration_sec = tagged;
        }
    }

    if duration_sec <= 0.0 {
        // FFmpeg's AV_TIME_BASE format duration. "Unknown" is AV_NOPTS_VALUE,
        // i.e. i64::MIN, and dividing that out yields -9223372036854.78
        // seconds -- a number that every consumer downstream happens to reject,
        // but only by failing a `> 0.0` guard, and which reached the report's
        // sortable `length_seconds` column and the JSON verbatim before it got
        // there. Anything not positive says the same thing the fallback exists
        // to detect: the container never reported a runtime.
        let reported = ictx.duration();
        duration_sec = if reported > 0 {
            reported as f64 / 1_000_000.0
        } else {
            0.0
        };
    }

    duration_sec
}

/// How many quarter turns CLOCKWISE the stream's pictures are to be turned to
/// be seen as intended: 0 to 3, from the display matrix a phone writes
/// beside its frames instead of turning the pixels.
///
/// A phone films portrait into a landscape-shaped buffer and says "turn this
/// 90 degrees" in the container; every player obeys, and every re-encode,
/// upload or messenger copy bakes the turn into the pixels. Read as stored, the
/// original and every copy of it were two different pictures and never
/// matched at any `-d`. Rounded to the nearest quarter turn, the only kind a
/// camera writes; a mirror in the matrix is not read.
pub(crate) fn display_turns(stream: &ffmpeg_next::format::stream::Stream<'_>) -> u8 {
    let Some(data) = display_matrix(stream) else { return 0 };
    if data.len() < 9 * 4 {
        return 0;
    }
    // Copied out first because the side data is a byte buffer with no promise
    // of alignment.
    let mut matrix = [0i32; 9];
    for (i, m) in matrix.iter_mut().enumerate() {
        *m = i32::from_ne_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
    }
    // SAFETY: nine i32s, which is all the function reads.
    let ccw = unsafe { ffmpeg_next::ffi::av_display_rotation_get(matrix.as_ptr()) };
    if !ccw.is_finite() {
        return 0;
    }
    // Counter-clockwise from the matrix, so clockwise is its negation.
    ((-ccw / 90.0).round() as i64).rem_euclid(4) as u8
}

/// The stream's display matrix, as raw bytes, wherever this FFmpeg keeps it:
/// in the codec parameters from 6.1 on, which the demuxer fills while reading
/// the header, and in the stream's own list before that. See build.rs.
fn display_matrix<'a>(stream: &'a ffmpeg_next::format::stream::Stream<'_>) -> Option<&'a [u8]> {
    #[cfg(ffmpeg_coded_side_data)]
    {
        // SAFETY: the parameters belong to the open stream and outlive this
        // borrow; the function only reads the array it is given.
        unsafe {
            let par = stream.parameters().as_ptr();
            let side = ffmpeg_next::ffi::av_packet_side_data_get(
                (*par).coded_side_data,
                (*par).nb_coded_side_data,
                ffmpeg_next::ffi::AVPacketSideDataType::AV_PKT_DATA_DISPLAYMATRIX,
            );
            if side.is_null() || (*side).data.is_null() {
                return None;
            }
            Some(std::slice::from_raw_parts((*side).data, (*side).size))
        }
    }
    #[cfg(not(ffmpeg_coded_side_data))]
    {
        stream
            .side_data()
            .find(|side| side.kind() == ffmpeg_next::codec::packet::side_data::Type::DisplayMatrix)
            .map(|side| side.data())
    }
}

/// `frame`, a `SIDE` x `SIDE` square, turned `turns` quarter turns clockwise.
fn turn_square(frame: &mut [u8], turns: u8, scratch: &mut Vec<u8>) {
    if turns % 4 == 0 {
        return;
    }
    scratch.clear();
    scratch.extend_from_slice(frame);
    let src = |y: usize, x: usize| scratch[y * SIDE + x];
    let last = SIDE - 1;
    for y in 0..SIDE {
        for x in 0..SIDE {
            frame[y * SIDE + x] = match turns % 4 {
                1 => src(last - x, y),
                2 => src(last - y, last - x),
                _ => src(x, last - y),
            };
        }
    }
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


/// The probe gate's verdict: nothing in this file looks like any format
/// libavformat knows.
///
/// A type rather than a string because two callers need to RECOGNISE it, not
/// just print it. `main` remembers this verdict in the cache and no other, since
/// it is the only failure that is purely a function of bytes the run has already
/// read -- a permission error or a vanished file is about the moment, not the
/// file, and must be re-asked every run. Both fields are kept so the sentence
/// can be regenerated from the cache instead of stored in it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NotMedia {
    /// How much of the file was actually LOOKED AT before giving up, which is
    /// `FIRST_PROBE_BYTES` (or `SECOND_PROBE_BYTES` when the name earned a
    /// second look) capped by what the file holds. The cap is the whole reason
    /// this is not simply the constant: an empty file used to be refused with
    /// "no demuxer recognised the first 16384 bytes of this file", which is a
    /// sentence about 16 KB that do not exist, and it was reached by exactly the
    /// files most likely to prompt the question -- a truncated download, an
    /// interrupted copy, a 0-byte placeholder. Nothing about the VERDICT moves;
    /// a file with nothing in it is not media either way.
    pub bytes: usize,
    /// What the best guess scored. At most `NO_EVIDENCE`, or this would not exist.
    pub score: i32,
}

impl std::fmt::Display for NotMedia {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // An empty file has no bytes to have recognised and no score worth
        // quoting: "no demuxer recognised the first 0 bytes (the best guess
        // scored 0 of 100)" is three clauses saying nothing about the one thing
        // the user needs to know, which is that there is nothing in the file.
        // It arrives here rather than anywhere earlier because emptiness is not
        // a special case to the probe, only to the sentence.
        if self.bytes == 0 {
            return write!(f, "this file is empty");
        }
        // Reached by a file holding exactly one byte, which is the same class of
        // near-empty file the zero case above is about -- a truncated download,
        // an interrupted copy -- so it is the one length that is actually likely
        // rather than a pedantic possibility.
        let unit = if self.bytes == 1 { "byte" } else { "bytes" };
        write!(
            f,
            "no demuxer recognised the first {} {} of this file (the best guess scored {} of \
             {}, which libavformat treats as no evidence)",
            self.bytes,
            unit,
            self.score,
            ffmpeg_next::ffi::AVPROBE_SCORE_MAX
        )
    }
}

impl std::error::Error for NotMedia {}

/// Errnos that describe the RUN rather than the file: the machine, the mount,
/// the permissions, this process's file descriptors. Every one of them can be
/// different next run with the bytes completely unchanged, which is exactly what
/// makes a refusal built on one unsafe to remember.
///
/// A list rather than "anything with an errno", which is what this started as
/// and which was wrong in a way only the corpus showed: `AVERROR(EINVAL)` is
/// how libavformat reports plenty of ordinary content problems, and 746 files in
/// a home directory fail that way. Treating them as environmental meant
/// re-opening and re-decoding all 746 on every single run -- the exact cost the
/// cache exists to remove. EINVAL is deliberately absent, and so is every other
/// errno not written here: the default is that a failure is about the file.
const ENVIRONMENTAL: &[i32] = &[
    libc::EPERM,
    libc::ENOENT,
    libc::EINTR,
    libc::EIO,
    libc::ENXIO,
    libc::EBADF,
    libc::EAGAIN,
    libc::ENOMEM,
    libc::EACCES,
    libc::EBUSY,
    libc::ENODEV,
    libc::ENFILE,
    libc::EMFILE,
    libc::ENOSPC,
    libc::EROFS,
    libc::ELOOP,
    libc::ETIMEDOUT,
    libc::ECONNREFUSED,
    libc::EHOSTUNREACH,
    libc::ENETUNREACH,
    libc::ESTALE,
];

/// Whether a failure is about the MOMENT rather than about the file.
///
/// A permission that will be granted tomorrow, a file that vanished mid-scan, a
/// network mount that dropped: asking again next run can legitimately give a
/// different answer, so nothing about these may be remembered. Everything else
/// here -- no demuxer recognised it, the streams would not parse, there is no
/// video stream, no frame decoded, no decoder for this codec -- is a statement
/// about bytes the `Stamp` already guards, and asking again costs the same work
/// for the same answer.
///
/// The test is the error's TYPE and errno, never its text. `ffmpeg_next::Error`
/// maps every AVERROR it knows to a named variant (`InvalidData`, `Eof`,
/// `DecoderNotFound`, ...) and everything else -- the `AVERROR(errno)` family --
/// to `Other`, so an OS error arriving through libavformat is as visible as one
/// arriving through `std::io`. Only the errnos in `ENVIRONMENTAL` count.
pub fn is_transient(error: &anyhow::Error) -> bool {
    error.chain().any(|link| {
        if let Some(ffmpeg_next::Error::Other { errno }) = link.downcast_ref::<ffmpeg_next::Error>()
        {
            return ENVIRONMENTAL.contains(errno);
        }
        // An io::Error with no errno behind it (an EOF, a timeout the runtime
        // synthesised) is treated as environmental: unlike the FFmpeg side,
        // nothing in this pipeline raises one to describe a file's contents.
        if let Some(io) = link.downcast_ref::<std::io::Error>() {
            return io.raw_os_error().is_none_or(|errno| ENVIRONMENTAL.contains(&errno));
        }
        false
    })
}

/// The probe gate's verdict, if that is what this error is.
///
/// Walks the chain rather than testing the outermost error, because every route
/// here wraps it in context first ("Failed to open video file: ...").
pub fn not_media(error: &anyhow::Error) -> Option<NotMedia> {
    error.chain().find_map(|link| link.downcast_ref::<NotMedia>().copied())
}

/// How much of a file libavformat is given to say what format it is, before this
/// tool concludes there is nothing there.
///
/// `PROBE_BUF_MIN`: the size of the FIRST pass `av_probe_input_buffer2` runs, so
/// asking at this size adds no question of our own. It reads the answer
/// libavformat is about to compute anyway, and only decides whether to let it
/// carry on computing.
///
/// What carrying on costs is the escalation: that function doubles its buffer --
/// 2 KB, 4 KB, ... to 1 MB -- for as long as nothing has scored above
/// `AVPROBE_SCORE_RETRY`, re-running all ~300 demuxer probes at every step. For
/// a file that is not media that loop always runs to the end. Measured over the
/// 248k files of a home directory: 0.5 ms for a 4 KB file, 24 ms at 64 KB,
/// 130 ms at 512 KB, flat at ~140 ms above 1 MB. It was 2,042 s of the 2,099 s
/// such a scan spent weighing -- 97% of the pass, spent proving a negative about
/// object files.
const FIRST_PROBE_BYTES: usize = 2048;

/// How much is read before turning away a file that showed a whisper of a
/// signal.
///
/// A score of exactly 1 is not a weak opinion, it is libavformat's way of
/// spelling "no opinion": `av_probe_input_format3` floors at 1 the score of a
/// demuxer that HAS a `read_probe` and whose extension list matches the NAME,
/// whatever the bytes said, and `mp3_probe` returns 1 for a single frame sync
/// anywhere in the buffer.
/// Both are usually noise -- 16,652 files in that home directory score exactly 1
/// at 2 KB, nearly all of them `.o` files with one accidental sync, and a bigger
/// buffer takes the score back to 0 (mp3 wants `max_frames >= buf_size/10000`,
/// which one stray sync stops satisfying as the buffer grows). Escalating them
/// to 1 MB costs 20 ms each to disprove a signal that was never there.
///
/// But it is not ALWAYS noise, and that is what this second size is for. A real
/// 11.8 MB MP3 in the same corpus scores 1 at 2 KB, 25 at 4 KB and 51 from 8 KB
/// on, because its first frames sit behind an ID3 tag; turning it away on the
/// 2 KB reading alone was wrong, even though nothing downstream would have
/// fingerprinted it. 16 KB is chosen with room to spare: across those 248k
/// files, every file a full 1 MB probe ever identified with a real score is
/// already identified by 16 KB except eight, and all eight are `.o` or
/// `query-cache.bin` that a big buffer briefly reads as `mpegvideo` -- a score
/// that oscillates 51, then 0, as the buffer grows again.
const SECOND_PROBE_BYTES: usize = 16384;

/// The score libavformat assigns when it has nothing: see `SECOND_PROBE_BYTES`.
/// A file has to beat this, at one size or the other, to be worth opening.
///
/// The safety property that makes this hard to get wrong: a match on the file's
/// EXTENSION alone scores at LEAST 1, so any file named like media -- `.mp4`,
/// `.mkv`, `.mp3`, anything at all that some demuxer claims -- is guaranteed to
/// reach the second look rather than being turned away at 2 KB. It is only ever
/// exactly 1 for a demuxer that brought a `read_probe` of its own and got
/// nothing out of the bytes; one with an extension list and no `read_probe`
/// scores `AVPROBE_SCORE_EXTENSION` (50) on the name alone and is opened
/// outright, and an extension match over an ID3 tag too long for the buffer
/// floors at 24 or 50 rather than 1. Every one of those clears this line, which
/// is why the rule can be stated as the floor. Only a file whose name tells
/// nothing AND whose first bytes match nothing is refused outright, which
/// on that corpus is 205,014 of 229,144 files. Of the 3,404 that a full-budget
/// probe went on to identify with a real score, 40 fall below this line: 39 are
/// object files and build caches, and the fortieth is that MP3, which the second
/// look recovers. Every one of the 735 real videos in the two test corpora
/// scores 98 or 100 at 2 KB -- the margin is not a few points, it is the scale.
const NO_EVIDENCE: i32 = 1;

/// The zeroed tail every `read_probe` implementation is allowed to read into.
const AVPROBE_PADDING_SIZE: usize = 32;

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
        .map_err(|_| anyhow!("the path contains an interior NUL byte"))?;

    // Ask libavformat's own first probe pass what this file looks like, and stop
    // here if the answer is "nothing". See `first_probe_score` for why that is
    // both safe and the single biggest cost in a `-x '*'` run.
    if let Some((bytes, score)) = unrecognised(filepath) {
        return Err(anyhow!(NotMedia { bytes, score }));
    }

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

thread_local! {
    /// One probe buffer per decode worker, grown once and reused for every file
    /// that worker looks at.
    ///
    /// A `-x '*'` scan asks this question a quarter of a million times, and the
    /// allocation it replaces was `SECOND_PROBE_BYTES + AVPROBE_PADDING_SIZE`
    /// of freshly zeroed heap per file -- for a buffer that is read, probed and
    /// dropped before the next file is opened. Thread-local rather than
    /// threaded through the call chain because `open_input`'s two callers are
    /// the whole of the decode path and neither has anything else to hand down.
    static PROBE_BUF: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// How much of a file was read before concluding it is not media, and what the
/// best guess scored -- or `None` if it IS worth opening.
///
/// Two sizes rather than one, because the cheap answer is right about 90% of a
/// home directory and wrong about a real MP3. See `FIRST_PROBE_BYTES` and
/// `SECOND_PROBE_BYTES`.
///
/// The second size is read only when the first one comes back with a whisper,
/// which is the ~10% of files that score exactly `NO_EVIDENCE`. The bytes are
/// read from the handle the first stage left open, so a file that needs the
/// second look pays one more `read` rather than a second `open`.
///
/// A file that cannot be read here is never a verdict: it is handed to
/// `avformat_open_input` anyway, so the user gets the real error ("No such file
/// or directory", "Permission denied") rather than this function's opinion of a
/// file it never saw.
fn unrecognised(filepath: &str) -> Option<(usize, i32)> {
    // The name is part of the question: `av_probe_input_format3` scores a
    // demuxer whose extension list matches it on the name alone -- see
    // `NO_EVIDENCE` for what that is worth and why it is a safety property.
    let cname = std::ffi::CString::new(filepath).ok()?;
    let mut file = std::fs::File::open(filepath).ok()?;

    PROBE_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();

        // `held` is what the file really gave up, which is what the refusal
        // reports so it names bytes that exist -- see `NotMedia::bytes`. A
        // short file comes back short and a missing tail is not evidence of
        // anything, so the reading is unchanged; only the sentence is.
        let held = fill_to(&mut file, &mut buf, 0, FIRST_PROBE_BYTES)?;
        let first = probe_head(&mut buf, held, &cname);
        if first > NO_EVIDENCE {
            return None;
        }
        if first == 0 {
            return Some((held, first));
        }

        let held = fill_to(&mut file, &mut buf, held, SECOND_PROBE_BYTES)?;
        let second = probe_head(&mut buf, held, &cname);
        if second > NO_EVIDENCE {
            return None;
        }
        Some((held, second))
    })
}

/// Read `buf` up to `want` real bytes, resuming from the `filled` it already
/// holds, and return how many it holds now. Short files come back short.
///
/// The buffer is left with libavformat's probe padding zeroed on the end -- the
/// tail every `read_probe` is allowed to read into, which
/// `av_probe_input_buffer2` memsets before every one of its own probes. It has
/// to be re-zeroed here rather than once, because the padding of the first
/// reading sits where the second reading's real bytes land. Getting this wrong
/// is what the old single-read version did: it probed 2 KB out of a buffer
/// holding 16 KB, so the 32 bytes past the probe window were the file's own
/// data instead of zeros, and a demuxer that reads its padding was answering a
/// question libavformat never asks.
///
/// The buffer itself is never shrunk: it belongs to the worker, not to the file.
fn fill_to(
    file: &mut std::fs::File,
    buf: &mut Vec<u8>,
    filled: usize,
    want: usize,
) -> Option<usize> {
    use std::io::Read;

    let mut filled = filled;
    if filled < want {
        if buf.len() < want + AVPROBE_PADDING_SIZE {
            buf.resize(want + AVPROBE_PADDING_SIZE, 0);
        }
        while filled < want {
            match file.read(&mut buf[filled..want]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(_) => return None,
            }
        }
    }
    buf[filled..filled + AVPROBE_PADDING_SIZE].fill(0);
    Some(filled)
}

/// What libavformat makes of the first `len` bytes of `buf`.
///
/// `is_opened = 1` is the value `av_probe_input_buffer2` passes, so this
/// considers exactly the demuxers the real open will consider. `len` is the
/// real byte count `fill_to` reported, and the padding it zeroed sits directly
/// after it -- which is the whole of what makes this the same question.
fn probe_head(buf: &mut [u8], len: usize, filename: &std::ffi::CStr) -> i32 {
    debug_assert!(buf.len() >= len + AVPROBE_PADDING_SIZE, "the padding must be there");
    let mut score: i32 = 0;
    let probe = ffmpeg_next::ffi::AVProbeData {
        filename: filename.as_ptr(),
        buf: buf.as_mut_ptr(),
        buf_size: len as i32,
        mime_type: std::ptr::null(),
    };
    unsafe { ffmpeg_next::ffi::av_probe_input_format3(&probe, 1, &mut score) };
    score
}

/// `open_input`, wrapped the way both callers want it -- except for the one
/// error it would be wrong to wrap.
///
/// `NotMedia` is not a failure to open the file, it is a decision not to, so
/// "Failed to open video file" describes something that never happened. It is
/// also the one verdict `main` replays out of the cache, where there is no open
/// to blame and nothing to add the prefix -- so wrapping it here made the same
/// file read one way on the run that discovered it and another on the run that
/// remembered it. Everything else keeps the context: an `ENOENT` or an `EACCES`
/// is exactly a failure to open a file.
pub(crate) fn open_video(filepath: &str) -> Result<ffmpeg_next::format::context::Input> {
    open_input(filepath).map_err(|e| {
        if not_media(&e).is_some() {
            e
        } else {
            e.context("Failed to open video file")
        }
    })
}

/// `avformat_find_stream_info`: the probe `open_input` deliberately skips.
///
/// One function so the two callers cannot drift. The weighing pass reports a
/// file the probe rejects on the decode's behalf, and a complaint the user has
/// to match against a decode that never ran had better be the decode's own
/// sentence rather than a paraphrase of it.
///
/// It does not take the path, and nothing here names one: every caller of every
/// route out of this module already prints the file it was asked about, so a
/// name in the message is the same name twice on one line.
pub(crate) fn probe_streams(ictx: &mut ffmpeg_next::format::context::Input) -> Result<()> {
    unsafe {
        let e =
            ffmpeg_next::ffi::avformat_find_stream_info(ictx.as_mut_ptr(), std::ptr::null_mut());
        if e < 0 {
            return Err(anyhow!(ffmpeg_next::Error::from(e)))
                .context("Failed to read the streams");
        }
    }
    Ok(())
}

/// Whether the header alone answered every question this file will be asked.
///
/// Deliberately conservative: any gap at all falls back to the full probe, so
/// the worst case is exactly today's behaviour. The extradata requirement is
/// the subtle one -- for streams that need parsing (raw H.264 in TS, say) the
/// probe is where the SPS gets extracted into the codec parameters, so a stream
/// that has not already got extradata from its header must not skip it.
pub(crate) fn header_is_complete(ictx: &ffmpeg_next::format::context::Input) -> bool {
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

/// How much a pixel of this codec costs to decode, relative to H.264 = 1.0.
///
/// Intra decode is the whole of the cost this weight is trying to predict, and
/// it is not the same price per pixel in every codec. Measured here across 133
/// files that take longer than 0.3 s each -- 90% of all the decode time in the
/// three corpora -- as median megapixels of keyframe per second on one core:
/// H.264 74, HEVC 43, AV1 (dav1d) 95. Hence the two entries below. Ignoring the
/// difference is how a folder holding an HEVC and an AV1 copy of the same
/// footage gets weighed as if the two were interchangeable.
///
/// Deliberately short, and deliberately only what was measured. A codec not
/// listed is charged H.264's price -- not because that is right for VP9 or
/// MPEG-2, but because a fabricated ratio is worse than the middle of the range,
/// and even a badly-priced keyframe count beats the file size this replaced.
///
/// The numbers are ratios between decoder families rather than a benchmark of
/// any one machine, so they travel: what moves with CPU and build is the
/// absolute rate, which nothing here uses. What they cannot see is bit depth --
/// 10-bit HEVC decodes at roughly half the rate of 8-bit, and neither the
/// container header nor the codec name says which one this is, so both are
/// charged the average of the two. Finding out costs an
/// `avformat_find_stream_info`, which is a keyframe decode per file: more than
/// the whole weighing pass, to sharpen one term of it.
fn codec_cost(codec: &str) -> f64 {
    match codec {
        "hevc" => 1.7,
        "av1" => 0.8,
        _ => 1.0,
    }
}

/// Keyframe spacing assumed for a container that will not say how many keyframes
/// it holds.
///
/// There is no single right answer here, because the two things people scan sit
/// either side of it: across 753 measured files, short clips out of a phone
/// average 0.9 s between keyframes and releases over ten minutes average 4.4 s.
/// This is the geometric middle of those, which is the value that is wrong by
/// the same factor in both directions rather than badly wrong in one.
const ASSUMED_KEYFRAME_SECONDS: f64 = 2.0;

/// Work units per byte, for a file that could not be opened or measured at all.
///
/// Only reached when the ladder in `weigh_decode` runs out -- an unreadable
/// file, a container with no width, no duration and no index. It is the old byte
/// proxy, rescaled so a file nothing is known about lands among the ones that
/// were measured instead of swamping them or vanishing beside them: the same 753
/// files average 3.8 keyframe-pixels per byte.
pub(crate) const WORK_PER_BYTE: f64 = 3.8;

/// `--min-duration`: the shortest runtime, in seconds, a video may have and
/// still be compared. 0 is off.
///
/// The one statement of the rule. Every place a runtime meets the flag -- the
/// weighing pass on a header, the decode on the same header, the cache pass and
/// the content pass on a fingerprint they already hold -- asks `rejects`, so
/// the passes cannot drift apart on what "too short" means. Only a positive,
/// known runtime can be rejected: 0 is how `duration_seconds` says the
/// container never told us, and an unmeasured file is not a short one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Floor(pub f64);

impl Floor {
    #[cfg(test)]
    pub const OFF: Floor = Floor(0.0);

    pub fn is_on(self) -> bool {
        self.0 > 0.0
    }

    pub fn rejects(self, duration: f64) -> bool {
        self.is_on() && duration > 0.0 && duration < self.0
    }
}

/// How much decoding this file is going to cost, in keyframe-pixels.
///
/// The unit is "one pixel of one decoded keyframe", scaled by `codec_cost`.
/// That is not an arbitrary index: ~93% of fingerprinting time is intra decode
/// of exactly those frames, at a rate that is close to constant per pixel within
/// a codec, so a number twice as large really does mean about twice the work.
/// The file's SIZE, which this replaces, only correlates with that through
/// bitrate -- so it charged a well-compressed 4K file less than a bloated SD one
/// and mis-sized the thread budget accordingly.
///
/// Costs one `avformat_open_input` per file and no decoding: the keyframe count
/// comes out of the container's own index, which MP4 and AVI build while reading
/// the header. Matroska builds its Cues lazily instead, so a single seek is
/// issued to force them -- 80 to 250 us warm, against a decode measured in
/// seconds. The count that comes back is not an approximation: across the 753
/// files measured it equalled the keyframes actually decoded on every one.
///
/// What it is still an estimate OF is time, and the residual is real: on the
/// files big enough to time, half land within 20% of their share of the run and
/// nine in ten within 55%, against 46% and 167% for the file size this replaced.
/// The spread between the 10th and 90th percentile falls from 5.5x to 2.2x --
/// which is the figure that matters, since `share_for` reads nothing but ratios.
///
/// The answer is an estimate and is always usable where it is a `Work` at all --
/// every rung of the ladder falls through to a cruder one rather than failing,
/// because a file this cannot MEASURE still has to be scheduled.
///
/// The two things it does not fall through are the file that will not decode
/// and the file that is not going to be decoded. `Undecodable` is not an
/// estimate: see `weigh_from_container` for why every route to it is one
/// `fingerprint_video` provably takes too. `TooShort` is not an estimate either,
/// and rests on the same kind of argument -- see the same function.
///
/// `floor` is this run's `--min-duration`, and it is here for the same reason
/// the keyframe interval is: it decides how much decoding this file implies,
/// and for a file under it the answer is none.
pub fn weigh_decode(
    filepath: &str,
    kf_interval: f64,
    min_kf_samples: f64,
    floor: Floor,
    size: u64,
) -> Weighed {
    let from_bytes = || ((size as f64 * WORK_PER_BYTE) as u64).max(1);

    match weigh_from_container(filepath, kf_interval, min_kf_samples, floor) {
        Ok(Some(weighed)) => weighed,
        Ok(None) => Weighed::Work(from_bytes(), 0),
        Err(e) => Weighed::Undecodable(e),
    }
}

/// What the weighing pass learned about one file.
///
/// It used to be a bare `u64`, and a file nothing could be learned about at all
/// got the bottom rung of the ladder -- its size times `WORK_PER_BYTE` -- on the
/// grounds that a file that cannot be weighed still has to be scheduled and the
/// decode is where a broken one gets reported. That is right for a file whose
/// container merely would not answer a question, and wrong for one no decoder
/// will open, which was charged for a decode that could not happen and then
/// probed a second time to discover what this pass already knew. Under
/// `-x '*'` over a folder that is mostly not video, that is the whole run: every
/// file opened twice, and a progress bar denominated in the bytes of files that
/// finish the instant they are looked at.
pub enum Weighed {
    /// Estimated decode cost, in keyframe-pixels, and the keyframes that
    /// counted (0 when the cost came from the file's bytes). The thread budget
    /// reads the cost; the progress bar also charges each keyframe the fixed
    /// cost of hashing it (`progress::KEYFRAME_PIXELS`).
    Work(u64, u64),
    /// No decode is going to happen, and this is the error it would have raised.
    /// The caller reports it now and never opens the file again.
    Undecodable(anyhow::Error),
    /// Shorter than `--min-duration`, carrying the runtime the header reported.
    ///
    /// The same finding `fingerprint_video` makes when it returns `Ok(None)`,
    /// made here so the file is opened once rather than twice: the decode's
    /// first act on such a file is to read the header this pass has just read,
    /// see the runtime this pass has just seen, and give up. It is a skip, not a
    /// problem, and the caller counts it as such.
    ///
    /// The runtime comes back with it because it is the only thing that makes
    /// this verdict worth remembering. `--min-duration` is a comparison-time
    /// flag and is deliberately not in the cache `Stamp`, so "too short" is not
    /// a fact about the file and cannot be cached; the number it was measured
    /// against is, and a later run with a different threshold can re-decide from
    /// it without opening anything.
    TooShort(f64),
}

/// The measurement half of `weigh_decode`: `Ok(None)` means the container opened
/// and holds video, but could not say enough to be weighed.
///
/// `Ok(Some(Weighed::TooShort))` is the second verdict this pass is allowed to
/// reach, and it rests on the same kind of correspondence as the `Err` below:
/// it is issued
/// only when `header_is_complete` says this header answers every question the
/// decode asks of it, which is exactly the condition under which
/// `fingerprint_video` will NOT probe -- so it reads its `floor` against
/// the same `duration_seconds` of the same stream of the same header, and
/// reaches the same conclusion. Where the header is not complete the question is
/// left alone entirely, because a probe can move the runtime and the direction
/// that matters is the one where this pass would skip a file the decode would
/// have kept.
///
/// It is asked before the index is counted and before the seek that forces
/// Matroska's Cues, because a file that is not going to be decoded does not need
/// a weight.
///
/// An `Err` is not a failure to measure -- it is the verdict that this file has
/// no fingerprint to give, carrying the error `fingerprint_video` would have
/// raised. Only three things produce one, and each is a point that function
/// provably reaches on the same file:
///
/// - `open_input` fails, which is its first call and its first failure.
/// - the probe fails. It is run here only when the header did not answer (no
///   video stream at all, or a stream with no picture size), and both of those
///   make `header_is_complete` false -- so `fingerprint_video` runs the same
///   probe on the same file and gets the same error.
/// - the probe succeeds and there is still no video stream, which is exactly the
///   `No video stream found in` that follows it there.
///
/// That correspondence is the whole safety argument for skipping the decode of
/// such a file, so a fourth route must not be added casually: the cost of being
/// wrong is a video reported as broken without anything having tried to read it.
/// `test_the_weigher_and_the_decoder_agree_about_a_file_that_is_not_video` pins
/// it against a real file.
///
/// The keyframe count is taken FIRST, before anything that might read packets,
/// and this order is load-bearing. An index is either complete or absent for the
/// containers that have one at all -- MP4's sample table and Matroska's Cues
/// both describe the whole file -- but for the ones that do not, libavformat
/// grows an index as packets go past. `avformat_find_stream_info` reads five
/// seconds of them, so asking after it would find an index holding the first two
/// or three keyframes of an hour-long stream and read it as the total, which is
/// an underestimate no rung of the ladder below would catch: it is not zero, so
/// nothing falls back.
///
/// The seek one rung down had exactly that hole and the same shape of
/// underestimate: it left MPEG-TS holding the one entry it landed on, which is
/// not zero either. See the note at that seek.
fn weigh_from_container(
    filepath: &str,
    kf_interval: f64,
    min_kf_samples: f64,
    floor: Floor,
) -> Result<Option<Weighed>> {
    let mut ictx = open_video(filepath)?;

    let stream_index = match ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .map(|s| s.index())
    {
        Some(index) => index,
        None => {
            // Nothing in the header claims to be video, which is not yet a
            // verdict: a few containers only publish their streams once packets
            // have been read (raw H.264 in TS is the usual one). So ask the
            // probe -- the same call `fingerprint_video` is about to make on
            // this file, since a header with no video stream in it cannot be
            // `header_is_complete`. Nothing here is extra work; it is that
            // function's work, done once instead of twice.
            probe_streams(&mut ictx)?;

            if ictx.streams().best(ffmpeg_next::media::Type::Video).is_none() {
                return Err(anyhow!("No video stream found"));
            }

            // A stream appeared, so this file is going to be decoded after all
            // -- but the probe just read packets through it, and for a container
            // with no index of its own libavformat has been building one out of
            // them. Counting that now would read the first few seconds of an
            // hour-long stream as the total, which is the underestimate the
            // ordering at the top of this function exists to avoid. The bytes
            // are the honest answer once the index cannot be trusted.
            return Ok(None);
        }
    };

    // Shorter than the shortest match this run is willing to report, decided
    // here rather than by the decode that would otherwise open this file a
    // second time to decide it. Only a header the decode would trust unprobed
    // may answer -- see the note above.
    if floor.is_on() && header_is_complete(&ictx) {
        if let Some(stream) = ictx.stream(stream_index) {
            let duration = duration_seconds(&stream, &ictx);
            if floor.rejects(duration) {
                return Ok(Some(Weighed::TooShort(duration)));
            }
        }
    }

    // MP4 and AVI have their index the moment the header is read; Matroska
    // parses its Cues on the first seek and reports nothing until then, so the
    // seek is what makes this question answerable for the container the tool
    // sees most. Seeking to the START specifically: it is the one target that
    // cannot be wrong, and for a format with no index at all it is the cheapest
    // possible request -- the generic fallback finds it without reading forward.
    let mut keyframes = index_keyframes(&ictx, stream_index);
    if keyframes == 0 {
        unsafe {
            let _ = ffmpeg_next::ffi::av_seek_frame(
                ictx.as_mut_ptr(),
                stream_index as i32,
                0,
                ffmpeg_next::ffi::AVSEEK_FLAG_BACKWARD,
            );
        }
        keyframes = index_keyframes(&ictx, stream_index);

        // One entry is the seek's own residue rather than an index. A container
        // that answers a seek by publishing its index publishes all of it --
        // Matroska hands over the whole Cue table -- while one with no index to
        // publish is seeked by libavformat's generic fallback, which reads
        // packets forward until it has landed and adds the single entry it
        // landed on. MPEG-TS is that case and it is in `DEFAULT_EXTENSIONS` for
        // the DVB captures and camcorder splits the splice invariant is about:
        // a 90-keyframe 1080p capture came back weighed at one keyframe's
        // pixels, 1% of its decode, so it sorted LAST, was handed one decoder
        // thread at the one moment threads can still be handed out, and left
        // ~99% of its work off the progress bar's denominator.
        //
        // This is asked only of a count the seek produced, so a file whose
        // header really did carry an index of one keyframe is untouched: that
        // one is non-zero before the seek and never reaches here. What it costs
        // is a container with no index and a genuine single keyframe, which
        // then takes the duration rung below and is charged as though it were
        // keyframed on the usual clock -- an overcharge in the safe direction,
        // since a decode of one enormous GOP still demuxes the whole file.
        if keyframes <= 1 {
            keyframes = 0;
        }
    }

    let mut facts = weighable_facts(&ictx, stream_index);

    // The header did not carry the picture's size. Probe -- the same probe
    // `fingerprint_video` is about to run on this file for the same reason (a
    // stream with no width is not `header_is_complete` either), and the
    // alternative is weighing it by its bytes.
    if facts.is_none() {
        probe_streams(&mut ictx)?;
        facts = weighable_facts(&ictx, stream_index);
    }

    let Some((pixels, duration, cost)) = facts else {
        return Ok(None);
    };

    // No index anywhere. The runtime is still worth something: keyframes are
    // placed on a rough clock, not on a rough number of bytes.
    if keyframes == 0 {
        if duration <= 0.0 {
            return Ok(None);
        }
        keyframes = (duration / ASSUMED_KEYFRAME_SECONDS).ceil().max(1.0) as i64;
    }

    // --keyframe-interval throws keyframes away before they are ever decoded,
    // and the file's bytes have no way to express that -- this is the one input
    // under which the old proxy was not merely imprecise but pointed the wrong
    // way, since the interval cuts a sparsely-keyframed file's work not at all
    // and a densely-keyframed one's by 90%. The rule mirrors `effective_interval`
    // in `fingerprint_video`, including its floor for short videos.
    if kf_interval > 0.0 && duration > 0.0 {
        let floor = if min_kf_samples > 0.0 { duration / min_kf_samples } else { f64::INFINITY };
        let interval = kf_interval.min(floor);
        if interval > 0.0 {
            keyframes = keyframes.min((duration / interval).ceil().max(1.0) as i64);
        }
    }

    Ok(Some(Weighed::Work(
        ((keyframes as f64) * (pixels as f64) * cost).max(1.0) as u64,
        keyframes.max(0) as u64,
    )))
}

/// Pixels per frame, runtime, and the codec's price, or `None` when the picture
/// has no size yet and the weight would therefore be nothing at all.
///
/// Only the size is required. A missing duration is survivable -- it is needed
/// solely by the rungs that guess a keyframe count -- which is why this asks
/// less of the header than `header_is_complete` does, and so probes less often.
fn weighable_facts(
    ictx: &ffmpeg_next::format::context::Input,
    stream_index: usize,
) -> Option<(i64, f64, f64)> {
    let stream = ictx.stream(stream_index)?;
    let pixels = unsafe {
        let par = (*stream.as_ptr()).codecpar;
        if par.is_null() {
            return None;
        }
        ((*par).width as i64) * ((*par).height as i64)
    };
    if pixels <= 0 {
        return None;
    }
    Some((
        pixels,
        duration_seconds(&stream, ictx),
        codec_cost(stream.parameters().id().name()),
    ))
}

/// The bytes of one stream, as the container states them, or 0 when it does
/// not state them completely.
///
/// Two sources, both read from the header with nothing decoded. An index that
/// holds every packet of the stream with its size, which is what MP4, MOV and
/// AVI build while reading the header: trusted only when its entry count is the
/// stream's own frame count, since a fragmented MP4 or a Matroska file's Cues
/// index part of a stream, and a sum over part of it is not the stream's size.
/// Then a Matroska track's `NUMBER_OF_BYTES` statistics tag, which mkvmerge
/// writes and FFmpeg's muxer does not. Anything else is unknown, and an
/// unknown never stands beside a measured figure in the ranking
/// (`utils::GroupMaxima`).
fn stream_bytes(ictx: &ffmpeg_next::format::context::Input, stream_index: usize) -> u64 {
    let Some(stream) = ictx.stream(stream_index) else {
        return 0;
    };
    let frames = stream.frames();
    let indexed = unsafe {
        let sp = stream.as_ptr();
        let count = ffmpeg_next::ffi::avformat_index_get_entries_count(sp);
        let mut sum = 0u64;
        let mut whole = count > 0 && i64::from(count) == frames;
        for i in 0..count {
            if !whole {
                break;
            }
            let entry = ffmpeg_next::ffi::avformat_index_get_entry(sp as *mut ffmpeg_next::ffi::AVStream, i);
            if entry.is_null() || (*entry).size() <= 0 {
                whole = false;
            } else {
                sum += (*entry).size() as u64;
            }
        }
        whole.then_some(sum)
    };
    indexed
        .or_else(|| {
            stream
                .metadata()
                .iter()
                .find(|(k, _)| {
                    let k = k.to_ascii_uppercase();
                    k == "NUMBER_OF_BYTES" || k.starts_with("NUMBER_OF_BYTES-")
                })
                .and_then(|(_, v)| v.trim().parse::<u64>().ok())
        })
        .unwrap_or(0)
}

/// The bytes of a file's video stream, from its header (`stream_bytes`), or 0
/// when it cannot be opened or does not say. About a millisecond: the run asks
/// it only of the files a comparison matched, just before they are ranked
/// (`lib::run`), so it costs a library nothing it is not about to report on.
pub fn video_stream_bytes(path: &str) -> u64 {
    let Ok(ictx) = open_video(path) else {
        return 0;
    };
    let Some(index) = ictx.streams().best(ffmpeg_next::media::Type::Video).map(|s| s.index()) else {
        return 0;
    };
    stream_bytes(&ictx, index)
}

/// How many entries of this stream's index are keyframes.
///
/// Counted rather than taken from the entry total because MP4 indexes every
/// sample and only some of them are sync samples, while Matroska's Cues are
/// keyframes only -- the flag is what makes one number out of two layouts.
///
/// `AVINDEX_KEYFRAME` is used without a cast, unlike the codec flags elsewhere
/// in this file: those are compared against a field bindgen types independently,
/// so the two can drift apart silently, whereas a mismatch here is a type error
/// on a bitand and cannot compile.
fn index_keyframes(ictx: &ffmpeg_next::format::context::Input, stream_index: usize) -> i64 {
    let Some(stream) = ictx.stream(stream_index) else {
        return 0;
    };
    unsafe {
        let sp = stream.as_ptr();
        let count = ffmpeg_next::ffi::avformat_index_get_entries_count(sp);
        let mut keyframes = 0i64;
        for i in 0..count {
            let entry =
                ffmpeg_next::ffi::avformat_index_get_entry(sp as *mut ffmpeg_next::ffi::AVStream, i);
            if !entry.is_null() && ((*entry).flags() & ffmpeg_next::ffi::AVINDEX_KEYFRAME) != 0 {
                keyframes += 1;
            }
        }
        keyframes
    }
}

/// within compression noise; an edge of the picture itself almost never is.
const BAR_MAX_DEVIATION: i32 = 4;
const BAR_MEAN_DEVIATION: i32 = 1;

/// The rows or columns at the start of `lines` that sit at the level of the
/// first one for at least `INSET_COVER` of their length: a bar that may carry a
/// caption or a channel's logo.
fn captioned_bar_depth<'a>(lines: impl Iterator<Item = &'a [u8]> + Clone) -> usize {
    let mut it = lines.clone();
    let Some(first) = it.next() else { return 0 };
    let mut sorted = first.to_vec();
    sorted.sort_unstable();
    let level = sorted[sorted.len() / 2] as i32;
    let flat = |line: &[u8]| {
        let at = line.iter().filter(|&&p| (p as i32 - level).abs() <= BAR_MAX_DEVIATION).count();
        at as f64 >= INSET_COVER * line.len() as f64
    };
    lines.take_while(|l| flat(l)).count()
}

/// Two bars on opposite edges: each peeled strictly, and a side whose bar
/// carries something peeled leniently, up to the depth of the clean bar
/// opposite it -- a letterbox is symmetric, so the clean side vouches for it.
///
/// What a bar carries is a channel's logo or a subtitle, put there precisely
/// because the bar is empty. Peeled strictly, such a bar stopped the crop at
/// once and the hash was taken over a black band with text in it, which
/// matched nothing (tv_capture 3/7 -> 5/7 on the generated corpus). Lenient on
/// its own the rule peels a dark scene's mostly-dark rows as well, so it is
/// only ever applied where the opposite edge has already shown a clean bar.
fn paired_bars<'a>(
    near: impl Iterator<Item = &'a [u8]> + Clone,
    far: impl Iterator<Item = &'a [u8]> + Clone,
) -> (usize, usize) {
    let (mut a, mut b) = (bar_depth(near.clone()), bar_depth(far.clone()));
    if a > b {
        b = b.max(captioned_bar_depth(far).min(a));
    } else if b > a {
        a = a.max(captioned_bar_depth(near).min(b));
    }
    (a, b)
}

/// The rows or columns at the start of `lines` that form one flat bar: each
/// within the bar limits of the level of the first one. 0 when the first line
/// is not flat itself.
fn bar_depth<'a>(lines: impl Iterator<Item = &'a [u8]> + Clone) -> usize {
    let mut it = lines.clone();
    let Some(first) = it.next() else { return 0 };
    let level = first.iter().map(|&p| p as i32).sum::<i32>() / first.len() as i32;
    let flat = |line: &[u8]| {
        let (mut max, mut sum) = (0, 0);
        for &p in line {
            let d = (p as i32 - level).abs();
            max = max.max(d);
            sum += d;
        }
        max <= BAR_MAX_DEVIATION && sum <= BAR_MEAN_DEVIATION * line.len() as i32
    };
    lines.take_while(|l| flat(l)).count()
}

/// The part of one working frame inside any flat bars along its edges, as
/// (top, bottom, left, right) inclusive. Each bar is peeled along with the row
/// or column next to it, which the scaler blends half into the bar.
///
/// A function of the frame alone, and that is the point of it. The crop used to
/// be found from the variance of every sampled frame of the video together, so
/// a clip, which samples a different set of frames from its source, could get a
/// different rectangle -- and a different rectangle reframes every hash: two
/// pixel-identical frames came out 22 bits apart that way. Asked of one frame,
/// the same picture gets the same crop in every copy of it, letterboxed or not:
/// a bar the copy added is peeled, and a flat edge the picture itself has is
/// peeled in both.
#[cfg(test)]
fn frame_crop(frame: &[u8]) -> (usize, usize, usize, usize) {
    frame_crop_or_tiny(frame).0
}

/// `frame_crop`, and whether the picture inside the bars was smaller than a
/// quarter of the frame either way -- the whole frame is then returned, and
/// the frame counts as featureless.
///
/// Such a frame is a line of text, a logo or a small bright patch on black, and
/// its hash says little more than where on the black that patch is: it matched
/// every other frame of that shape across unrelated videos (blobs on black, a
/// lyric line against a lecture slide). The AC-energy floor cannot see it,
/// because a bright patch on black has plenty of contrast. A video made of
/// nothing else then has no usable frames and is reported as such, like an
/// all-black one.
fn frame_crop_or_tiny(frame: &[u8]) -> ((usize, usize, usize, usize), bool) {
    const WHOLE: (usize, usize, usize, usize) = (0, SIDE - 1, 0, SIDE - 1);
    const MIN_SPAN: usize = SIDE / 4;
    let row = |y: usize| &frame[y * SIDE..y * SIDE + SIDE];
    let (top, bottom) = paired_bars((0..SIDE).map(row), (0..SIDE).rev().map(row));
    let (y1, y2) = (top + (top > 0) as usize, (SIDE - 1).saturating_sub(bottom + (bottom > 0) as usize));
    if y2 < y1 || y2 - y1 + 1 < MIN_SPAN {
        return (WHOLE, true);
    }
    // Columns are read inside the rows that survived, so a letterbox does not
    // hide a pillarbox and vice versa.
    let mut cols = vec![[0u8; SIDE]; SIDE];
    let h = y2 - y1 + 1;
    for (x, col) in cols.iter_mut().enumerate() {
        for y in 0..h {
            col[y] = frame[(y1 + y) * SIDE + x];
        }
    }
    let (left, right) = paired_bars(cols.iter().map(|c| &c[..h]), cols.iter().rev().map(|c| &c[..h]));
    let (x1, x2) = (left + (left > 0) as usize, (SIDE - 1).saturating_sub(right + (right > 0) as usize));
    if x2 < x1 || x2 - x1 + 1 < MIN_SPAN {
        return (WHOLE, true);
    }
    ((y1, y2, x1, x2), false)
}

/// The hash of each working GRAY8 frame in `u_frames` (packed back to back),
/// and whether it is featureless, each taken inside that frame's own crop
/// (`frame_crop`): a letterbox or a pillarbox is a property of the encode, not
/// of the picture, and would otherwise weigh on every hash.
#[cfg(test)]
fn hash_frames(u_frames: &[u8]) -> (Vec<u64>, Vec<bool>) {
    let (hashes, _, flat) = hash_frames_both_ways(u_frames);
    (hashes, flat)
}

/// Everything hashed out of one video's frames, index for index with them.
pub(crate) struct Hashed {
    pub hashes: Vec<u64>,
    pub mirrored: Vec<u64>,
    pub flat: Vec<bool>,
    pub detail: Vec<u64>,
    pub mirrored_detail: Vec<u64>,
    /// Share of each frame inside its crop.
    pub picture: Vec<f32>,
    /// One per `Reframing`, in `REFRAMINGS` order: each frame's hashes inside
    /// that sub-rectangle, or `None` where the reframing does not apply.
    pub reframed: Vec<Vec<Option<[u64; 4]>>>,
}

/// A sub-rectangle of each frame's own crop that a re-upload is often cut down
/// to, hashed alongside the whole frame so that copy can still be found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Reframing {
    /// A portrait frame holding a 16:9 picture in its middle rows: the blurred
    /// or coloured "fill" a vertical re-upload pads a landscape video with.
    Band,
    /// The middle 1/1.12 of the frame: a light zoom, crop or tilt-to-fill.
    Zoom,
    /// The largest rectangle with a sharp edge along all four sides: a picture
    /// pasted onto a background of its own -- picture-in-picture, a reaction
    /// layout, a screen recording of a player (`inset_rect`).
    Inset,
    /// A 9:16 strip of a landscape frame, the `k`th of `STRIPS` spread evenly
    /// across its width: a vertical re-upload cut out of a landscape video.
    /// Only ever compared against a portrait file.
    Strip(u8),
    /// The middle 4:3 of a wider frame, full height: a widescreen picture
    /// cut down for a 4:3 screen (pan and scan). Only ever compared against a
    /// narrower file.
    Cut43,
}

/// How many strip positions a landscape frame is hashed at.
pub const STRIPS: u8 = 13;

/// The reframings every sample is also hashed under, in storage order.
pub const REFRAMINGS: [Reframing; 4 + STRIPS as usize] = [
    Reframing::Band,
    Reframing::Zoom,
    Reframing::Inset,
    Reframing::Cut43,
    Reframing::Strip(0),
    Reframing::Strip(1),
    Reframing::Strip(2),
    Reframing::Strip(3),
    Reframing::Strip(4),
    Reframing::Strip(5),
    Reframing::Strip(6),
    Reframing::Strip(7),
    Reframing::Strip(8),
    Reframing::Strip(9),
    Reframing::Strip(10),
    Reframing::Strip(11),
    Reframing::Strip(12),
];

/// A step between neighbouring pixels this large (grey levels) is an edge, for
/// `inset_rect`: three times what `frame_crop` still calls flat.
const INSET_STEP: i32 = 3 * BAR_MAX_DEVIATION;
/// Share of each side of an inset that must be such an edge; the rest may be
/// blended into a background of the same shade.
const INSET_COVER: f64 = 0.8;
/// An inset larger than this share of the frame is the frame itself, cropped,
/// which `frame_crop` and `Zoom` already read.
const INSET_MAX_AREA: f64 = 0.7;

/// How many working-frame pixels one pixel of the 64x64 reading `inset_rect`
/// works on spans.
const F: usize = SIDE / 64;

/// The frame averaged down to 64x64, the resolution `inset_rect`'s edge step
/// was measured at and the one that keeps its O(n^3) search cheap.
fn coarse(full: &[u8]) -> [u8; 4096] {
    let mut frame = [0u8; 4096];
    for y in 0..64 {
        for x in 0..64 {
            let mut acc = 0u32;
            for dy in 0..F {
                for dx in 0..F {
                    acc += full[(y * F + dy) * SIDE + x * F + dx] as u32;
                }
            }
            frame[y * 64 + x] = ((acc + (F * F / 2) as u32) / (F * F) as u32) as u8;
        }
    }
    frame
}

/// A rectangle of the 64x64 reading, in working-frame pixels.
fn widen((y1, y2, x1, x2): (usize, usize, usize, usize)) -> (usize, usize, usize, usize) {
    (y1 * F, y2 * F + F - 1, x1 * F, x2 * F + F - 1)
}

/// The largest rectangle of `frame` bounded by edges on all four sides, as
/// (top, bottom, left, right) inclusive, at least `THUMB` a side -- anything
/// smaller would be hashed from fewer pixels than the thumbnail has.
///
/// For each pair of rows, the candidate sides are the columns whose edge runs
/// the height between them, which is a short list, so this is O(64^3) at worst
/// and ~0.2 ms a frame in practice. It is searched on the 64x64 average of
/// the working frame (`coarse`) and widened back.
fn inset_rect(full: &[u8]) -> Option<(usize, usize, usize, usize)> {
    inset_rect_64(&coarse(full)).map(widen)
}

fn inset_rect_64(frame: &[u8]) -> Option<(usize, usize, usize, usize)> {
    let px = |y: usize, x: usize| frame[y * 64 + x] as i32;
    // hp[y][x]: edges along the boundary above row y, over columns 0..x.
    // vp[x][y]: edges along the boundary left of column x, over rows 0..y.
    let mut hp = [[0u16; 65]; 64];
    let mut vp = [[0u16; 65]; 64];
    for (y, line) in hp.iter_mut().enumerate().skip(1) {
        let mut edges = 0;
        for (x, cell) in line.iter_mut().skip(1).enumerate() {
            edges += ((px(y, x) - px(y - 1, x)).abs() >= INSET_STEP) as u16;
            *cell = edges;
        }
    }
    for (x, line) in vp.iter_mut().enumerate().skip(1) {
        let mut edges = 0;
        for (y, cell) in line.iter_mut().skip(1).enumerate() {
            edges += ((px(y, x) - px(y, x - 1)).abs() >= INSET_STEP) as u16;
            *cell = edges;
        }
    }
    let covers = |n: u16, len: usize| n as f64 >= INSET_COVER * len as f64;
    let mut best = 0usize;
    let mut found = None;
    let mut cols = [0usize; 64];
    for y1 in 1..64 {
        for y2 in (y1 + THUMB)..64 {
            let h = y2 - y1;
            if h * 63 <= best {
                continue;
            }
            let mut n = 0;
            for (x, line) in vp.iter().enumerate().skip(1) {
                if covers(line[y2] - line[y1], h) {
                    cols[n] = x;
                    n += 1;
                }
            }
            for p in 0..n {
                let x1 = cols[p];
                for q in (p + 1..n).rev() {
                    let x2 = cols[q];
                    let w = x2 - x1;
                    if w < THUMB || w * h <= best {
                        break;
                    }
                    if (w * h) as f64 > INSET_MAX_AREA * 4096.0 {
                        continue;
                    }
                    if !covers(hp[y1][x2] - hp[y1][x1], w) || !covers(hp[y2][x2] - hp[y2][x1], w) {
                        continue;
                    }
                    best = w * h;
                    found = Some((y1, y2 - 1, x1, x2 - 1));
                    break;
                }
            }
        }
    }
    found
}

const ZOOM: f64 = 1.12;
const BAND_ASPECT: f64 = 16.0 / 9.0;

/// The thumbnail the detail hash is taken from, and the side of the transform
/// block its coefficients are chosen out of.
const DETAIL_THUMB: usize = 32;
const DETAIL_KEEP: usize = 16;

/// The 64 coefficients of the detail hash: every third of the 16x16 block
/// with the 8x8 corner the main hash reads taken out, so the two hashes share
/// no coefficient.
fn detail_coefficients() -> [(usize, usize); 64] {
    let mut out = [(0, 0); 64];
    let mut n = 0;
    let mut seen = 0;
    for k in 0..DETAIL_KEEP {
        for l in 0..DETAIL_KEEP {
            if k < KEEP && l < KEEP {
                continue;
            }
            if seen % 3 == 0 && n < 64 {
                out[n] = (k, l);
                n += 1;
            }
            seen += 1;
        }
    }
    out
}

/// Fractions of a crop rectangle to a smaller integer rectangle, or `None` when
/// it would be under 8 pixels a side.
fn sub_rect(
    (y1, y2, x1, x2): (usize, usize, usize, usize),
    (fy0, fy1, fx0, fx1): (f64, f64, f64, f64),
) -> Option<(usize, usize, usize, usize)> {
    let h = (y2 - y1 + 1) as f64;
    let w = (x2 - x1 + 1) as f64;
    let ny1 = y1 as i64 + (h * fy0).round_ties_even() as i64;
    let ny2 = y1 as i64 + (h * fy1).round_ties_even() as i64 - 1;
    let nx1 = x1 as i64 + (w * fx0).round_ties_even() as i64;
    let nx2 = x1 as i64 + (w * fx1).round_ties_even() as i64 - 1;
    let min = (SIDE / 8) as i64;
    let last = (SIDE - 1) as i64;
    if ny2 - ny1 + 1 < min || nx2 - nx1 + 1 < min || ny1 < 0 || nx1 < 0 || ny2 > last || nx2 > last {
        return None;
    }
    Some((ny1 as usize, ny2 as usize, nx1 as usize, nx2 as usize))
}

impl Reframing {
    /// The part of a frame's crop this reframing keeps, given the decoded
    /// picture's width and height.
    fn rect(
        self,
        frame: &[u8],
        crop: (usize, usize, usize, usize),
        (width, height): (u32, u32),
    ) -> Option<(usize, usize, usize, usize)> {
        match self {
            Reframing::Inset => inset_rect(frame),
            Reframing::Band => {
                if !(height > width && width > 0) {
                    return None;
                }
                let f = (width as f64 / height as f64) / BAND_ASPECT;
                sub_rect(crop, ((1.0 - f) / 2.0, (1.0 + f) / 2.0, 0.0, 1.0))
            }
            Reframing::Zoom => {
                let m = (1.0 - 1.0 / ZOOM) / 2.0;
                sub_rect(crop, (m, 1.0 - m, m, 1.0 - m))
            }
            Reframing::Cut43 => {
                if !(height > 0 && width as f64 > height as f64 * 4.0 / 3.0 * 1.05) {
                    return None;
                }
                let f = (4.0 / 3.0) / (width as f64 / height as f64);
                sub_rect(crop, (0.0, 1.0, (1.0 - f) / 2.0, (1.0 + f) / 2.0))
            }
            Reframing::Strip(k) => {
                if !(width > height && height > 0) {
                    return None;
                }
                let f = (height as f64 / width as f64) / BAND_ASPECT;
                let c = f / 2.0 + k as f64 * (1.0 - f) / (STRIPS - 1) as f64;
                sub_rect(crop, (0.0, 1.0, c - f / 2.0, c + f / 2.0))
            }
        }
    }
}

/// Weights per rectangle, built once each: a video has a handful of them.
type Taps = Vec<Vec<(usize, f32)>>;

struct Hasher {
    basis: [[f32; THUMB]; KEEP],
    detail_basis: Vec<[f32; DETAIL_THUMB]>,
    detail_pick: [(usize, usize); 64],
    weights: std::collections::HashMap<(usize, usize, usize, usize), (Taps, Taps)>,
    detail_weights: std::collections::HashMap<(usize, usize, usize, usize), (Taps, Taps)>,
    thumb: [f32; THUMB * THUMB],
    band: [f32; SIDE * THUMB],
    dthumb: Vec<f32>,
    dband: Vec<f32>,
}

impl Hasher {
    fn new() -> Self {
        let detail_basis = (0..DETAIL_KEEP)
            .map(|k| {
                let mut row = [0.0f32; DETAIL_THUMB];
                for (x, cell) in row.iter_mut().enumerate() {
                    *cell = (std::f32::consts::PI * (2.0 * x as f32 + 1.0) * k as f32
                        / (2.0 * DETAIL_THUMB as f32))
                        .cos();
                }
                row
            })
            .collect();
        Hasher {
            basis: dct_basis(),
            detail_basis,
            detail_pick: detail_coefficients(),
            weights: Default::default(),
            detail_weights: Default::default(),
            thumb: [0.0; THUMB * THUMB],
            band: [0.0; SIDE * THUMB],
            dthumb: vec![0.0; DETAIL_THUMB * DETAIL_THUMB],
            dband: vec![0.0; SIDE * DETAIL_THUMB],
        }
    }

    /// The low-frequency block of `frame` inside `rect`.
    fn coefficients(&mut self, frame: &[u8], rect: (usize, usize, usize, usize)) -> [f32; KEEP * KEEP] {
        let (y1, y2, x1, x2) = rect;
        let (rows, cols) = self
            .weights
            .entry(rect)
            .or_insert_with(|| (box_weights(y1, y2 - y1 + 1), box_weights(x1, x2 - x1 + 1)));
        resample(frame, rows, cols, &mut self.band, &mut self.thumb);
        low_frequency_block(&self.thumb, &self.basis)
    }

    /// The detail hash of `frame` inside `rect`, plain and mirrored.
    fn detail(&mut self, frame: &[u8], rect: (usize, usize, usize, usize)) -> (u64, u64) {
        let (y1, y2, x1, x2) = rect;
        let (rows, cols) = self.detail_weights.entry(rect).or_insert_with(|| {
            (box_weights_to(y1, y2 - y1 + 1, DETAIL_THUMB), box_weights_to(x1, x2 - x1 + 1, DETAIL_THUMB))
        });
        let n = DETAIL_THUMB;
        for (oy, taps) in rows.iter().enumerate() {
            let dst = &mut self.dband[oy * SIDE..oy * SIDE + SIDE];
            dst.fill(0.0);
            for &(sy, w) in taps {
                for (d, &p) in dst.iter_mut().zip(&frame[sy * SIDE..sy * SIDE + SIDE]) {
                    *d += p as f32 * w;
                }
            }
        }
        for oy in 0..n {
            let src = &self.dband[oy * SIDE..oy * SIDE + SIDE];
            for (ox, taps) in cols.iter().enumerate() {
                let mut acc = 0.0;
                for &(sx, w) in taps {
                    acc += src[sx] * w;
                }
                self.dthumb[oy * n + ox] = acc;
            }
        }
        // rows of the transform first, for every k any picked coefficient needs
        let mut partial = [[0.0f32; DETAIL_THUMB]; DETAIL_KEEP];
        for (k, prow) in partial.iter_mut().enumerate() {
            for (y, &b) in self.detail_basis[k].iter().enumerate() {
                let src = &self.dthumb[y * n..y * n + n];
                for (p, &s) in prow.iter_mut().zip(src) {
                    *p += b * s;
                }
            }
        }
        let mut c = [0.0f32; 64];
        for (q, &(k, l)) in self.detail_pick.iter().enumerate() {
            c[q] = partial[k].iter().zip(&self.detail_basis[l]).map(|(p, b)| p * b).sum();
        }
        let mut m = c;
        for (q, &(_, l)) in self.detail_pick.iter().enumerate() {
            if l % 2 == 1 {
                m[q] = -m[q];
            }
        }
        (hash_of(&c), hash_of(&m))
    }
}

/// `hash_frames`, plus each frame mirrored left to right, its detail hash, and
/// its hashes under every reframing (`dims` is the decoded picture's size,
/// which says whether a reframing applies).
#[cfg(test)]
fn hash_frames_both_ways(u_frames: &[u8]) -> (Vec<u64>, Vec<u64>, Vec<bool>) {
    let h = hash_frames_fully(u_frames, (0, 0), &|_| {});
    (h.hashes, h.mirrored, h.flat)
}

/// What `fingerprint_video` tells its caller as it goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// The demuxer has reached this byte offset of the video stream.
    Read(u64),
    /// This many of the sampled frames are hashed, of so many.
    Hashed(usize, usize),
}

/// `hash_frames_fully` on up to `threads` threads, each hashing one contiguous
/// run of the frames, the runs joined back in order.
///
/// The decode pass reserves a video's share of the thread budget for the
/// whole of `fingerprint_video`, and the hashing comes after the last keyframe
/// is decoded: run on the calling thread alone, it left the rest of that share
/// idle while every other worker waited on the budget for it. On a 1h43m
/// 540p file at `-t 8` that was the last 3.5 s of a 14.5 s decode at one core.
/// Each frame is hashed on its own and a `Hasher`'s caches only save work, so
/// the result is bit-identical to the serial one whatever `threads` is
/// (`test_hashing_on_several_threads_is_bit_identical`).
///
/// `on_frame` is called on this thread only, with how many frames all the
/// threads have hashed so far, so a caller's progress hook need not be `Sync`.
pub(crate) fn hash_frames_on(
    u_frames: &[u8],
    dims: (u32, u32),
    threads: usize,
    on_frame: &dyn Fn(usize),
) -> Hashed {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let n = u_frames.len() / FRAME_STRIDE;
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        return hash_frames_fully(u_frames, dims, on_frame);
    }
    let per = n.div_ceil(threads);
    let runs: Vec<&[u8]> = u_frames[..n * FRAME_STRIDE].chunks(per * FRAME_STRIDE).collect();
    let done = AtomicUsize::new(0);
    let parts: Vec<Hashed> = std::thread::scope(|scope| {
        let others: Vec<_> = runs[1..]
            .iter()
            .map(|run| {
                let done = &done;
                scope.spawn(move || {
                    hash_frames_fully(run, dims, &|i| {
                        if i > 0 {
                            done.fetch_add(1, Ordering::Relaxed);
                        }
                    })
                })
            })
            .collect();
        let first = hash_frames_fully(runs[0], dims, &|i| {
            if i > 0 {
                on_frame(done.fetch_add(1, Ordering::Relaxed) + 1);
            }
        });
        std::iter::once(first)
            .chain(others.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))))
            .collect()
    });

    let mut parts = parts.into_iter();
    let mut out = parts.next().expect("at least one run");
    for part in parts {
        out.hashes.extend(part.hashes);
        out.mirrored.extend(part.mirrored);
        out.flat.extend(part.flat);
        out.detail.extend(part.detail);
        out.mirrored_detail.extend(part.mirrored_detail);
        out.picture.extend(part.picture);
        for (into, from) in out.reframed.iter_mut().zip(part.reframed) {
            into.extend(from);
        }
    }
    out
}

/// `on_frame` is told how many frames are hashed so far, as each next one starts.
pub(crate) fn hash_frames_fully(u_frames: &[u8], dims: (u32, u32), on_frame: &dyn Fn(usize)) -> Hashed {
    let n_unique = u_frames.len() / FRAME_STRIDE;
    let mut hasher = Hasher::new();
    let mut out = Hashed {
        hashes: Vec::with_capacity(n_unique),
        mirrored: Vec::with_capacity(n_unique),
        flat: Vec::with_capacity(n_unique),
        detail: Vec::with_capacity(n_unique),
        mirrored_detail: Vec::with_capacity(n_unique),
        reframed: vec![Vec::with_capacity(n_unique); REFRAMINGS.len()],
        picture: Vec::with_capacity(n_unique),
    };

    for (i, frame) in u_frames.chunks_exact(FRAME_STRIDE).enumerate() {
        if i > 0 {
            on_frame(i);
        }
        let (rect, tiny) = frame_crop_or_tiny(frame);
        let (y1, y2, x1, x2) = rect;
        out.picture.push(((y2 - y1 + 1) * (x2 - x1 + 1)) as f32 / FRAME_STRIDE as f32);
        let coefficients = hasher.coefficients(frame, rect);
        out.hashes.push(hash_of(&coefficients));
        out.mirrored.push(hash_of(&mirror_coefficients(&coefficients)));
        out.flat.push(tiny || is_featureless(&coefficients));
        let (d, md) = hasher.detail(frame, rect);
        out.detail.push(d);
        out.mirrored_detail.push(md);
        for (v, reframing) in REFRAMINGS.iter().enumerate() {
            let hashed = reframing.rect(frame, rect, dims).and_then(|r| {
                let c = hasher.coefficients(frame, r);
                if is_featureless(&c) {
                    return None;
                }
                let (d, md) = hasher.detail(frame, r);
                Some([hash_of(&c), hash_of(&mirror_coefficients(&c)), d, md])
            });
            out.reframed[v].push(hashed);
        }
    }

    out
}

/// The largest of the per-sample picture shares, or 0 for none.
///
/// The largest rather than a median because the two things `frame_crop` peels
/// differ in how often they appear: a letterbox or pillarbox is on every frame,
/// while a flat edge belonging to the picture (a dark scene, a sky) is on some.
/// The median read those edges as bars, so copies of one stream that sampled
/// different scenes (a clip, a stream copy) measured up to 22% apart on the
/// generated corpus where their stored pixels were identical.
fn largest_share(shares: Vec<f32>) -> f32 {
    shares.into_iter().fold(0.0, f32::max)
}

/// The coefficients of the same picture mirrored left to right. The basis
/// function of horizontal frequency `l` is symmetric about the centre for even
/// `l` and antisymmetric for odd `l`, so mirroring negates exactly the odd
/// columns. `frame_crop` peels each side on its own, so the mirrored frame
/// gets the mirrored rectangle and this is exact, not an approximation.
fn mirror_coefficients(c: &[f32; KEEP * KEEP]) -> [f32; KEEP * KEEP] {
    let mut m = *c;
    for k in 0..KEEP {
        for l in (1..KEEP).step_by(2) {
            m[k * KEEP + l] = -m[k * KEEP + l];
        }
    }
    m
}

/// Fingerprint one video.
///
/// `decode_threads` is this video's share of the process-wide thread budget,
/// decided by the caller from how many decodes are still outstanding. It is
/// clamped to `1..=MAX_DECODE_THREADS`, so passing 0 is harmless.
///
/// `floor` (`--min-duration`) skips the video entirely if it is shorter
/// than the shortest match we are willing to report — such a file cannot
/// possibly contain a long enough shared clip, so decoding it is pure waste.
/// Returns `Ok(None)` in that case: a skip is not an error, and the caller must
/// not log it as one. A video whose duration cannot be determined is NOT
/// skipped; unknown is not the same as short.
///
/// `file_size` is the length the caller measured, and is recorded on the
/// fingerprint as given rather than re-derived here. `sources::collect` stats
/// every file exactly once and everything downstream reads that figure; this
/// function used to be the one exception, calling `std::fs::metadata` again
/// purely to fill this field. Taking it as an argument removes a syscall per
/// decode and, more usefully, makes the fingerprint's `file_size` and the cache
/// `Stamp`'s `size` the same measurement from the same moment -- so a file that
/// grew between the scan and its decode now reads as CHANGED at disposal time
/// instead of being deleted against a size nothing else in the run agreed with.
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
///
/// Then, once the file is read, it is handed `Step::Hashed` for each sampled
/// frame hashed: the frames are kept as they are decoded and hashed after the
/// last one, and that is a stretch of its own. On a ProRes file (every frame a
/// keyframe) the read ended at 3.8 s of an 8 s decode, and a bar that moved by
/// bytes alone then sat still for the other four.
pub fn fingerprint_video(
    filepath: &str,
    kf_interval: f64,
    min_kf_samples: f64,
    decode_threads: usize,
    floor: Floor,
    file_size: u64,
    progress: &dyn Fn(Step),
) -> Result<Option<VideoFingerprint>> {
    let mut ictx = open_video(filepath)?;

    // The probe is the expensive half of opening a file, and for an ordinary
    // MP4/MKV it is answering questions the header already answered. Run it
    // only when something is genuinely missing.
    if !header_is_complete(&ictx) {
        log::debug!("{}: header incomplete, probing streams.", filepath);
        probe_streams(&mut ictx)?;
    }

    let input = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| anyhow!("No video stream found"))?;
    let stream = input.index();
    let duration_sec = duration_seconds(&input, &ictx);

    // Bail before the decoder and the frame buffer exist. Only a positive,
    // known duration can disqualify a file.
    if floor.rejects(duration_sec) {
        return Ok(None);
    }

    // The codec by NAME (`avcodec_get_name`), never through
    // `avcodec_find_decoder`: that names whichever decoder this FFmpeg prefers
    // ("libdav1d" or "av1" for one AV1 file), which made the recorded codec a
    // property of the machine -- in the report, in the codec-standoff rule's
    // equality test, and in the cache.
    let codec = input.parameters().id().name().to_string();
    let frame_rate = frame_rate_of(&input, duration_sec);
    let turns = display_turns(&input);
    let tb = seconds_per_tick(&input);

    let mut decoder = KeyframeDecoder::open(input.parameters(), decode_threads)?;
    let mut capture = FrameCapture::new(turns, tb);
    let mut thinning = Thinning::new(kf_interval, min_kf_samples, duration_sec);
    let mut clock = SampleClock::default();
    let mut keyframes = Keyframes::new(ictx, stream, progress);

    while let Some(packet) = keyframes.next()? {
        let at = decode_stamp(packet).map(|ts| clock.tick(ts));
        // The sample clock rides through the decoder on the packet's pts,
        // which libavcodec copies onto the frame it produces; under frame
        // threading nothing else pairs a frame with its packet.
        if let Some(ticks) = at {
            packet.set_pts(Some(ticks));
        }
        let t = at.filter(|_| tb > 0.0).map(|ticks| ticks as f64 * tb);
        if thinning.keep(t) {
            decoder.feed(packet, &mut |frame| capture.take(frame));
        }
        if clock.restarted() {
            keyframes.stop_seeking();
        }
        // Past the keyframes the interval would discard anyway, in one hop.
        let step = thinning.wait_after(t).map_or(1, |s| ((s / tb).ceil() as i64).max(1));
        keyframes.seek_past(step);
    }
    decoder.finish(&mut |frame| capture.take(frame));

    // Each of these is a sample missing from a fingerprint that would still
    // claim the whole runtime, and asked in this order so the file says which
    // happened: a picture decoded and not scaled, then a keyframe the decoder
    // refused, then nothing decoded at all.
    if let Some(e) = capture.unscalable() {
        return Err(anyhow!(e).context("Failed to convert a decoded frame"));
    }
    if let Some(e) = decoder.refused.take() {
        return Err(anyhow!(e).context("The decoder would not take a keyframe"));
    }
    let captured = capture
        .finish()
        .ok_or_else(|| anyhow!("No valid frames found or successfully decoded"))?;

    // Not a problem and not a skip: the file is still worth comparing, it is
    // covered by fewer samples than its keyframes promised. Both numbers,
    // because 8 of 835 is a remux libavcodec quibbles with and 800 of 835 is a
    // file to go and look at.
    if decoder.lost > 0 {
        log::warn!(
            "{}: the decoder could not build {} of {} keyframes; those samples are missing",
            filepath,
            decoder.lost,
            decoder.lost as usize + captured.count
        );
    }

    // The decoder and the demuxer are done with: let their threads and buffers
    // go before the hashing, which takes over this video's share of the budget.
    drop(decoder);
    drop(keyframes);
    let hashed = hash_frames_on(&captured.frames, captured.dims, decode_threads.clamp(1, MAX_DECODE_THREADS), &|done| {
        progress(Step::Hashed(done, captured.times.len()))
    });
    assemble(filepath, captured, hashed, duration_sec, clock.restarted(), codec, frame_rate, file_size)
}

/// The fingerprint of the frames a decode kept: put in time order, a sample
/// that repeats the one before it folded in, and featureless ones dropped, each
/// survivor standing for the footage until the next sample.
#[allow(clippy::too_many_arguments)]
fn assemble(
    filepath: &str,
    captured: Captured,
    hashed: Hashed,
    duration_sec: f64,
    clock_restarted: bool,
    codec: String,
    frame_rate: f64,
    file_size: u64,
) -> Result<Option<VideoFingerprint>> {
    let Captured { frames, times, dims: (width, height), .. } = captured;
    // Release the frame buffer (munmap) before the cheap tail work.
    drop(frames);
    let sampled = times.len();
    let (times, total_ms) = sample_times(&times, duration_sec);

    // A container whose clock restarted mis-states its runtime by the same
    // arithmetic -- two five-second recordings joined report five seconds --
    // and the samples, taken by walking the whole file, are the better
    // witness. Length is the first ranking metric, so left at the header's
    // figure a capture CONTAINING another video loses to it on quality and is
    // marked DELETE. Only such a file is touched, so `total_ms >= duration *
    // 1000` still holds everywhere.
    let duration_sec = if clock_restarted { duration_sec.max(total_ms as f64 / 1000.0) } else { duration_sec };

    // Keyframes do not always leave the decoder in presentation order, and a
    // span runs "until the next sample", so order by time first. A sample
    // that repeats the one before it (hash and featurelessness both) adds
    // nothing and is folded into it. A featureless sample still ends the span
    // before it, but holds no hash of its own.
    let mut order: Vec<usize> = (0..hashed.hashes.len()).collect();
    order.sort_by_key(|&i| times[i]);
    order.dedup_by(|cur, prev| hashed.hashes[*cur] == hashed.hashes[*prev] && hashed.flat[*cur] == hashed.flat[*prev]);

    let n = order.len();
    let mut valid_hashes = Vec::with_capacity(n);
    let mut valid_t_start = Vec::with_capacity(n);
    let mut valid_t_end = Vec::with_capacity(n);
    let mut mirrored_hashes = Vec::with_capacity(n);
    let mut detail_hashes = Vec::with_capacity(n);
    let mut mirrored_detail_hashes = Vec::with_capacity(n);
    let mut reframed: Vec<Reframed> = vec![Reframed::default(); REFRAMINGS.len()];
    let mut pictures: Vec<f32> = Vec::with_capacity(n);

    for (k, &i) in order.iter().enumerate() {
        if hashed.flat[i] {
            continue;
        }
        for (v, r) in reframed.iter_mut().enumerate() {
            if let Some(h) = hashed.reframed[v][i] {
                r.samples.push(valid_hashes.len() as u32);
                r.hashes.push(h);
            }
        }
        valid_hashes.push(hashed.hashes[i]);
        mirrored_hashes.push(hashed.mirrored[i]);
        detail_hashes.push(hashed.detail[i]);
        mirrored_detail_hashes.push(hashed.mirrored_detail[i]);
        pictures.push(hashed.picture[i]);
        valid_t_start.push(times[i]);
        valid_t_end.push(order.get(k + 1).map_or(total_ms, |&next| times[next]));
    }

    // Every frame was featureless -- black, a fade, a title card. A failure
    // rather than an empty fingerprint, which would be cached like any other
    // and match nothing at any `-d`: two identical all-black recordings
    // reported no duplicates, exit 0. A different sentence from "decoded
    // nothing", and `Refusal::Said` remembers it.
    if valid_hashes.is_empty() {
        return Err(anyhow!("No usable frames: all {} sampled frame(s) were featureless", sampled));
    }

    Ok(Some(VideoFingerprint {
        path: filepath.to_string(),
        valid_hashes,
        valid_t_start,
        valid_t_end,
        total_ms,
        width,
        height,
        duration: duration_sec,
        file_size,
        codec,
        frame_rate,
        mirrored_hashes,
        detail_hashes,
        mirrored_detail_hashes,
        reframed,
        picture: largest_share(pictures),
        measured: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A working frame with structure everywhere and no flat edge: no two
    /// neighbouring pixels agree, so nothing about it reads as a bar.
    fn busy_frame(seed: u32) -> Vec<u8> {
        let mut x = seed.wrapping_mul(2654435761).max(1);
        (0..FRAME_STRIDE)
            .map(|i| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                let (r, c) = (i / SIDE, i % SIDE);
                ((r * 3 + c * 5) as u32 % 128 + (x % 96)) as u8
            })
            .collect()
    }

    /// Splitting the frames across threads is a schedule, not a different
    /// computation: every hash, flag, share and reframing comes back the same
    /// and in the same order, whatever the thread count and however unevenly
    /// the frames divide. Frames with bars are in it so the crop and the
    /// reframings are exercised, at both a landscape and a portrait shape.
    /// The video stream's bytes come out of an MP4's index exactly, and the
    /// cover picture beside the stream is not among them: the two fixtures
    /// hold one byte-identical H.264 stream, 4,818 bytes by `ffprobe
    /// -show_packets`. A Matroska file FFmpeg wrote states no figure, and
    /// says so with 0 rather than a sum over the keyframes its Cues index.
    #[test]
    fn test_the_video_stream_is_measured_without_what_sits_beside_it() {
        let at = |name: &str| format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
        assert_eq!(video_stream_bytes(&at("test_video.mp4")), 4818);
        assert_eq!(video_stream_bytes(&at("test_video_cover_art.mp4")), 4818);
        assert_eq!(video_stream_bytes(&at("test_video_open_gop.mkv")), 0);
        assert_eq!(video_stream_bytes(&at("no_such_file.mp4")), 0);
    }

    #[test]
    fn test_hashing_on_several_threads_is_bit_identical() {
        let mut frames = Vec::new();
        for seed in 1..=23u32 {
            let mut f = busy_frame(seed);
            if seed % 3 == 0 {
                // A letterbox: flat rows top and bottom.
                for r in (0..SIDE / 8).chain(SIDE - SIDE / 8..SIDE) {
                    f[r * SIDE..(r + 1) * SIDE].fill(16);
                }
            }
            frames.extend(f);
        }
        for dims in [(1920, 1080), (720, 1280)] {
            let serial = hash_frames_fully(&frames, dims, &|_| {});
            for threads in [1, 2, 3, 4, 7, 23, 64] {
                let seen = std::sync::Mutex::new(Vec::new());
                let h = hash_frames_on(&frames, dims, threads, &|d| seen.lock().unwrap().push(d));
                assert_eq!(h.hashes, serial.hashes, "{threads} threads");
                assert_eq!(h.mirrored, serial.mirrored);
                assert_eq!(h.flat, serial.flat);
                assert_eq!(h.detail, serial.detail);
                assert_eq!(h.mirrored_detail, serial.mirrored_detail);
                assert_eq!(h.picture, serial.picture);
                assert_eq!(h.reframed, serial.reframed);
                let seen = seen.into_inner().unwrap();
                assert!(seen.windows(2).all(|w| w[0] < w[1]), "progress only moves forward: {seen:?}");
                assert!(seen.iter().all(|&d| d < 23), "and never past the frames there are");
            }
        }
        assert_eq!(hash_frames_on(&[], (0, 0), 8, &|_| {}).hashes, Vec::<u64>::new());
    }

    #[test]
    fn test_a_mirrored_frame_hashes_to_the_mirrored_hash() {
        // The mirrored hash is derived from the coefficients rather than from a
        // second pass over flipped pixels; it has to be the same answer.
        for seed in 1..20 {
            let frame = busy_frame(seed);
            let flipped: Vec<u8> = frame.chunks_exact(SIDE).flat_map(|row| row.iter().rev().copied()).collect();
            let mut both = frame.clone();
            both.extend_from_slice(&flipped);
            let (hashes, mirrored, _) = hash_frames_both_ways(&both);
            assert!((mirrored[0] ^ hashes[1]).count_ones() <= 1, "seed {seed}: {:016x} vs {:016x}", mirrored[0], hashes[1]);
            assert!((mirrored[1] ^ hashes[0]).count_ones() <= 1, "seed {seed}");
        }
    }

    #[test]
    fn test_a_picture_pasted_onto_a_background_is_found_and_a_plain_frame_has_none() {
        // A diagonal gradient, the kind of animated backdrop a reaction or
        // picture-in-picture layout sits on, with a textured picture at
        // (rows 8..32, columns 30..50) of a 64-unit frame, `F` pixels a unit.
        // The gradient's own steps are a grey level or two, far under an edge.
        let mut frame = vec![0u8; FRAME_STRIDE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                frame[y * SIDE + x] = (60 + (y + x) / F) as u8;
            }
        }
        assert_eq!(inset_rect(&frame), None, "a gradient has no inset");
        for y in 8 * F..32 * F {
            for x in 30 * F..50 * F {
                frame[y * SIDE + x] = if (x / (3 * F) + y / (4 * F)) % 2 == 0 { 230 } else { 20 };
            }
        }
        assert_eq!(inset_rect(&frame), Some((8 * F, 32 * F - 1, 30 * F, 50 * F - 1)));
        // Too small to hash from: under THUMB units on a side.
        let mut small = vec![100u8; FRAME_STRIDE];
        for y in 10 * F..20 * F {
            for x in 10 * F..40 * F {
                small[y * SIDE + x] = 250;
            }
        }
        assert_eq!(inset_rect(&small), None);
    }

    #[test]
    fn test_a_small_patch_on_black_is_featureless() {
        // A line of text on black: plenty of contrast, but the picture inside
        // the bars is a sliver, and its hash would only say where it is.
        let mut text = vec![0u8; FRAME_STRIDE];
        for y in SIDE / 2..SIDE / 2 + SIDE / 16 {
            for x in SIDE / 4..3 * SIDE / 4 {
                text[y * SIDE + x] = if x % 5 < 3 { 235 } else { 0 };
            }
        }
        assert!(frame_crop_or_tiny(&text).1);
        let (_, flat) = hash_frames(&text);
        assert_eq!(flat, vec![true]);
        // A real picture is not.
        let (_, flat) = hash_frames(&busy_frame(3));
        assert_eq!(flat, vec![false]);
    }

    #[test]
    fn test_each_frame_is_cropped_by_its_own_flat_edges() {
        let frame = busy_frame(7);
        assert_eq!(frame_crop(&frame), (0, SIDE - 1, 0, SIDE - 1), "a picture with no flat edge is not cropped");

        // A letterbox (8 units of `F` rows) and a pillarbox (5 units of
        // columns) of different levels.
        let (bar, pillar) = (8 * F, 5 * F);
        let mut boxed = frame.clone();
        for y in 0..SIDE {
            for x in 0..SIDE {
                if !(bar..SIDE - bar).contains(&y) {
                    boxed[y * SIDE + x] = 16;
                } else if !(pillar..SIDE - pillar).contains(&x) {
                    boxed[y * SIDE + x] = 40;
                }
            }
        }
        // Each bar is peeled with the line beside it, which a real scaler blends.
        let expected = (bar + 1, SIDE - 1 - (bar + 1), pillar + 1, SIDE - 1 - (pillar + 1));
        assert_eq!(frame_crop(&boxed), expected);

        // A caption in the bottom bar: the top bar vouches for it, so it is
        // peeled all the same.
        let mut captioned = boxed.clone();
        for y in SIDE - bar + 2..SIDE - 2 {
            for x in SIDE / 2..SIDE / 2 + SIDE / 10 {
                captioned[y * SIDE + x] = 235;
            }
        }
        assert_eq!(frame_crop(&captioned), expected, "a logo in a letterbox bar does not stop the crop");
        // With no clean bar opposite, a mostly flat edge is the picture's own.
        let mut lone = frame.clone();
        for y in SIDE - bar..SIDE {
            for x in 0..SIDE {
                lone[y * SIDE + x] = if (SIDE / 2..SIDE / 2 + SIDE / 10).contains(&x) { 235 } else { 16 };
            }
        }
        assert_eq!(frame_crop(&lone), (0, SIDE - 1, 0, SIDE - 1));
    }
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

    /// One file, sampled at every keyframe it has.
    fn fingerprint_path(path: &std::path::Path) -> VideoFingerprint {
        init_ffmpeg_for_tests();

        let filepath = path.to_string_lossy().to_string();
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        fingerprint_video(&filepath, 0.0, 4.0, 1, Floor::OFF, size, &|_| {})
            .unwrap_or_else(|e| panic!("failed to fingerprint {}: {:?}", filepath, e))
            .expect("min_duration is off, nothing should be skipped")
    }

    #[test]
    fn test_a_letterboxed_copy_measures_the_same_picture_as_the_bare_one() {
        // The same 160x80 footage, stored bare and padded to 160x120 with black
        // bars: 50% more stored pixels and the same picture. Resolution ranks
        // on the picture (`utils::resolution`), so the two must agree within
        // the band that calls two resolutions level (`RESOLUTION_TOLERANCE`).
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let bare = fingerprint_path(&dir.join("test_video_cropped.mp4"));
        let boxed = fingerprint_path(&dir.join("test_video_letterboxed.mp4"));
        assert_eq!((bare.width * bare.height, boxed.width * boxed.height), (12_800, 19_200));
        assert!(boxed.picture > 0.0 && boxed.picture < 0.75, "the bars are measured: {}", boxed.picture);
        let (a, b) = (crate::utils::resolution(&bare) as f64, crate::utils::resolution(&boxed) as f64);
        assert!((a - b).abs() / a.max(b) < crate::utils::RESOLUTION_TOLERANCE, "bare {a} against letterboxed {b}");
    }

    fn fingerprint_fixture(threads: usize) -> VideoFingerprint {
        init_ffmpeg_for_tests();

        let fixture_path = fixture_path();
        let filepath = fixture_path.to_string_lossy().to_string();
        assert!(fixture_path.exists(), "Fixture video not found at: {}.", filepath);

        let size = std::fs::metadata(&fixture_path).map(|m| m.len()).unwrap_or(0);
        let result = fingerprint_video(&filepath, 0.0, 4.0, threads, Floor::OFF, size, &|_| {});
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
            mirrored_hashes: Vec::new(),
            detail_hashes: Vec::new(),
            mirrored_detail_hashes: Vec::new(),
            reframed: Vec::new(),
            picture: 0.0,
            measured: None,
        }
    }

    #[test]
    fn test_the_last_sample_is_extended_by_a_real_average_gap() {
        // Six samples a second apart, and a container that under-reports its
        // runtime -- which is how this branch is actually reached: anchoring at
        // the first keyframe shifts every sample later, so an MP4 opening on a
        // negative dts can put the last one past the duration in the header.
        //
        // The spacing is 1000 ms and the last sample has to stand for about that
        // much footage. Dividing the span by the sample COUNT rather than by the
        // number of gaps between them gave 5000/6 = 833.
        let times: Vec<Option<u32>> = (0..6).map(|i| Some(i * 1000)).collect();
        let (out, total_ms) = sample_times(&times, 4.5);

        assert_eq!(out.len(), 6);
        assert_eq!(total_ms, 6000, "5000 + one 1000 ms gap, not 5000 + 833");
    }

    #[test]
    fn test_a_single_sample_still_gets_a_span() {
        // One sample has no gap to average, and a zero-length span would delete
        // it from every coverage figure. The `.max(1)` is what keeps it.
        let (_, total_ms) = sample_times(&[Some(0)], 0.0);
        assert!(total_ms >= 1, "a lone sample must stand for something, got {}", total_ms);

        // And nothing at all must not divide by zero.
        let (out, total_ms) = sample_times(&[], 0.0);
        assert!(out.is_empty());
        assert!(total_ms >= 1);
    }

    #[test]
    fn test_a_known_runtime_is_left_alone() {
        // The branch above must not fire when the header's duration already
        // covers the samples: that figure is the container's own and is better
        // than anything derived from the spacing.
        let times: Vec<Option<u32>> = (0..6).map(|i| Some(i * 1000)).collect();
        let (_, total_ms) = sample_times(&times, 10.0);
        assert_eq!(total_ms, 10_000);
    }

    #[test]
    fn test_video_shorter_than_min_duration_is_skipped() {
        init_ffmpeg_for_tests();
        let filepath = fixture_path().to_string_lossy().to_string();

        // The fixture is ~1s; an hour-long floor must reject it without error.
        let size = std::fs::metadata(fixture_path()).map(|m| m.len()).unwrap_or(0);
        let skipped = fingerprint_video(&filepath, 0.0, 4.0, 1, Floor(3600.0), size, &|_| {}).unwrap();
        assert!(skipped.is_none(), "a short video must be skipped, not fingerprinted");
    }

    /// A file under `--min-duration` is not going to be decoded, so the weighing
    /// pass says so and the decode never opens it -- one open instead of two,
    /// on every run, for every file the flag skips (such a file writes no
    /// fingerprint, so it is a cache miss for ever).
    ///
    /// That is only allowed because both passes read the same runtime out of the
    /// same header. Pinned as a bracket rather than as a float comparison
    /// between the two: a threshold either side of the runtime the weigher
    /// reports has to move BOTH passes, and the same way.
    #[test]
    fn test_the_weigher_and_the_decoder_agree_about_a_file_that_is_too_short() {
        init_ffmpeg_for_tests();
        let path = fixture_path();
        let size = std::fs::metadata(&path).unwrap().len();
        let filepath = path.to_string_lossy().to_string();

        let Weighed::TooShort(duration) = weigh_decode(&filepath, 0.0, 4.0, Floor(3600.0), size) else {
            panic!("the fixture is about a second long and an hour-long floor rejects it");
        };
        assert!(duration > 0.0, "the verdict carries the runtime it was measured against");

        let under = duration - 0.01;
        assert!(
            matches!(weigh_decode(&filepath, 0.0, 4.0, Floor(under), size), Weighed::Work(..)),
            "a floor under the runtime is not one this file falls under"
        );
        assert!(
            fingerprint_video(&filepath, 0.0, 4.0, 1, Floor(under), size, &|_| {})
                .unwrap()
                .is_some(),
            "and the decode reads it the same way, which is what lets the weigher answer at all"
        );

        let over = duration + 0.01;
        assert!(
            matches!(weigh_decode(&filepath, 0.0, 4.0, Floor(over), size), Weighed::TooShort(_)),
            "a floor over the runtime is"
        );
        assert!(
            fingerprint_video(&filepath, 0.0, 4.0, 1, Floor(over), size, &|_| {}).unwrap().is_none(),
            "and again the decode agrees, on the same side of the same number"
        );

        assert!(
            matches!(weigh_decode(&filepath, 0.0, 4.0, Floor::OFF, size), Weighed::Work(..)),
            "0 turns the flag off and nothing is short"
        );
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
        fingerprint_video(&filepath, 0.0, 4.0, 1, Floor::OFF, size, &|step| {
            if let Step::Read(pos) = step {
                offsets.borrow_mut().push(pos)
            }
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
        fingerprint_video(&filepath, 0.0, 4.0, 1, Floor::OFF, size, &|step| {
            if let Step::Read(pos) = step {
                offsets.borrow_mut().push(pos)
            }
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

    // --- Refusing files that are not media, cheaply and without spinning ------

    /// The hang this guard exists for, reproduced exactly.
    ///
    /// `ncdec.c` ends `while (state != NC_VIDEO_FLAG) { if (avio_feof(pb))
    /// return AVERROR_INVALIDDATA; ... }` -- at the end of the file it reports
    /// invalid data rather than EOF, and the demux loop's "a recoverable error
    /// skips the packet" arm then asks it again forever. Four bytes of
    /// `00 00 01 A5` and a declared packet size larger than the file are enough
    /// to be scored 25 by `nc_probe`, which is what carries this past the probe
    /// gate and into the loop; that is not a contrivance, it is what a 132-byte
    /// linker version script does when it is called `libavcodec.v`.
    ///
    /// Run on its own thread with a deadline, so that losing the guard fails
    /// this test in ten seconds instead of hanging the suite the way it hung the
    /// scan: 80 minutes of one core per file, with eight workers eventually
    /// stuck on eight of them and no output at all.
    #[test]
    fn test_a_demuxer_that_cries_invalid_data_at_eof_does_not_spin_forever() {
        init_ffmpeg_for_tests();

        let dir = tempfile::tempdir().unwrap();
        // `.v` because `nc` claims that extension, which is how the real files
        // got here; the bytes are what make it score.
        let path = dir.path().join("libavcodec.v");
        let mut bytes = vec![0u8; 20];
        bytes[0..4].copy_from_slice(&[0x00, 0x00, 0x01, 0xA5]);
        // Declared packet size, little-endian at offset 5. Larger than the file,
        // so `nc_probe` takes its "cannot check the next header" branch and
        // returns AVPROBE_SCORE_MAX / 4.
        bytes[5] = 0x60;
        bytes[6] = 0xEA;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        bytes.extend((0..4000).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u8
        }));
        std::fs::write(&path, &bytes).unwrap();

        let filepath = path.to_string_lossy().to_string();
        let size = bytes.len() as u64;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let outcome = fingerprint_video(&filepath, 0.0, 4.0, 1, Floor::OFF, size, &|_| {});
            let _ = tx.send(outcome.is_err());
        });

        match rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(errored) => assert!(errored, "there are no frames in this file to find"),
            Err(_) => panic!(
                "fingerprint_video did not return: the demux loop is asking a demuxer that \
                 will never say EOF for another packet"
            ),
        }
    }

    /// A sample is the picture of the keyframe it is timed at, whatever order
    /// the decoder hands keyframes back in.
    ///
    /// HEVC's open-GOP keyframes (CRA pictures) carry their picture order
    /// count modulo 2^k, and a decoder that is only ever shown keyframes cannot
    /// tell when that count wrapped: two keyframes more than half the range
    /// apart (128 frames at x265's default, 5.3 s at 24 fps) and it reads the
    /// later as the earlier, and outputs the two the wrong way round. The scan
    /// used to time each picture by `best_effort_timestamp`, which FFmpeg
    /// switches to the dts of whatever packet is being SENT once it has seen
    /// presentation times go backwards -- so from the first reordering on,
    /// every picture was filed under a later keyframe's time. FFmpeg 6.1 does
    /// this to 17 of the 19 HEVC files in the local corpus and the vendored
    /// FFmpeg 8 to none, so on a static build this test cannot fail; on a
    /// dynamic one against 6.1 it is what tells the two answers apart.
    ///
    /// The fixture is x265 at a 150-frame keyframe interval, open GOP. The
    /// oracle decodes each keyframe alone with a fresh decoder, in demux
    /// order, where nothing can be reordered, times it by the scan's clock and
    /// hashes it the way the scan does.
    #[test]
    fn test_a_sample_is_the_picture_of_the_keyframe_it_is_timed_at() {
        init_ffmpeg_for_tests();
        let path = fixture_named("test_video_open_gop.mkv").to_str().unwrap().to_string();
        let size = std::fs::metadata(&path).unwrap().len();
        let fp = fingerprint_video(&path, 0.0, 0.0, 1, Floor::OFF, size, &|_| {}).unwrap().expect("the fixture fingerprints");

        let mut ictx = open_video(&path).unwrap();
        let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).unwrap();
        let (index, tb, params) = (stream.index(), f64::from(stream.time_base()), stream.parameters());
        let packets: Vec<ffmpeg_next::Packet> =
            ictx.packets().filter(|(s, p)| s.index() == index && p.is_key()).map(|(_, p)| p).collect();
        assert!(packets.len() >= 4, "the fixture has keyframes enough to reorder");
        let (mut frames, mut times, mut first) = (Vec::new(), Vec::new(), None);
        for packet in &packets {
            let ts = packet.dts().or_else(|| packet.pts()).unwrap();
            let anchor = *first.get_or_insert(ts);
            let mut ctx = ffmpeg_next::codec::context::Context::from_parameters(params.clone()).unwrap();
            unsafe { (*ctx.as_mut_ptr()).thread_count = 1 };
            let mut dec = ctx.decoder().video().unwrap();
            let mut frame = ffmpeg_next::frame::Video::empty();
            let _ = dec.send_packet(packet);
            if dec.receive_frame(&mut frame).is_err() {
                dec.send_eof().unwrap();
                dec.receive_frame(&mut frame).unwrap();
            }
            let mut scaler = ffmpeg_next::software::scaling::context::Context::get(
                frame.format(),
                frame.width(),
                frame.height(),
                ffmpeg_next::format::Pixel::GRAY8,
                SIDE as u32,
                SIDE as u32,
                ffmpeg_next::software::scaling::flag::Flags::FAST_BILINEAR,
            )
            .unwrap();
            let mut gray = ffmpeg_next::frame::Video::empty();
            scaler.run(&frame, &mut gray).unwrap();
            for y in 0..SIDE {
                frames.extend_from_slice(&gray.data(0)[y * gray.stride(0)..][..SIDE]);
            }
            times.push(((ts - anchor) as f64 * tb * 1000.0) as u32);
        }
        let (hashes, _) = hash_frames(&frames);

        assert!(!fp.valid_t_start.is_empty());
        for (&t, &h) in fp.valid_t_start.iter().zip(&fp.valid_hashes) {
            let k = times.iter().position(|&kt| kt == t).unwrap_or_else(|| panic!("no keyframe at {t} ms; they are at {times:?}"));
            // Decoded alone rather than after the keyframes before it, a
            // keyframe can come out a bit or two apart; another keyframe of
            // this fixture is never that close.
            let d = |k: usize| (h ^ hashes[k]).count_ones();
            assert!(d(k) <= 4, "the sample at {t} ms is {} bits from that keyframe: {:?}", d(k), fp.valid_t_start);
            for other in (0..hashes.len()).filter(|&o| times[o] != t) {
                assert!(d(other) > d(k), "the sample at {t} ms looks more like the keyframe at {} ms", times[other]);
            }
        }
    }

    /// A file whose clock restarts part way through keeps everything after the
    /// restart.
    ///
    /// `cat` of two recordings is how a DVB capture, a camcorder split and half
    /// the `.ts` files on a disk are laid out, and the second recording carries
    /// the same timestamps the first one already used. Everything after the
    /// join used to be discarded -- by the dedup guard, which read the restart
    /// as a seek landing short, and by the seek path, which navigates by the
    /// very clock that just lied. The file then fingerprinted as its own first
    /// segment: same hashes and same runtime as a file it CONTAINS, ranked
    /// below it on encoder quality, and marked DELETE against it.
    ///
    /// The fixture is doubled rather than paired with a second one so the
    /// expected answer is exact: the whole of the first half's fingerprint,
    /// twice, on a clock that only goes forwards.
    #[test]
    fn test_a_clock_that_restarts_mid_file_does_not_truncate_the_fingerprint() {
        init_ffmpeg_for_tests();

        let segment = fixture_named("test_video_segment.ts");
        let bytes = std::fs::read(&segment)
            .unwrap_or_else(|e| panic!("fixture {} unreadable: {}", segment.display(), e));

        let dir = tempfile::tempdir().unwrap();
        let spliced = dir.path().join("capture.ts");
        std::fs::write(&spliced, [bytes.as_slice(), bytes.as_slice()].concat()).unwrap();

        let one = fingerprint_path(&segment);
        let two = fingerprint_path(&spliced);

        assert_eq!(
            two.valid_hashes,
            [one.valid_hashes.clone(), one.valid_hashes.clone()].concat(),
            "a spliced capture holds both segments: {} hashes against {} for one segment",
            two.valid_hashes.len(),
            one.valid_hashes.len()
        );

        let ascending = two.valid_t_start.windows(2).all(|w| w[0] < w[1]);
        assert!(
            ascending,
            "the sample clock only ever goes forwards, splice or no splice: {:?}",
            two.valid_t_start
        );

        // Length is the first thing the ranking compares and the reason a long
        // file survives the clip cut out of it. Left at what the container
        // says, this file is the same length as the one inside it.
        assert!(
            two.duration > one.duration * 1.5,
            "a file holding two recordings is longer than one of them: {} against {}",
            two.duration,
            one.duration
        );
    }

    /// A stream that changes resolution part way through keeps the frames on
    /// both sides of the change.
    ///
    /// The scaler is built from the first frame the decoder produces, and
    /// `Context::run` refuses a frame whose format or geometry is not the one
    /// it was built for -- so a rescale mid-stream turns every later frame into
    /// `Error::InputChanged`. Both call sites used to discard that, so the file
    /// simply stopped being sampled at the change: no frame counted, nothing
    /// logged, exit code 0, and the truncated fingerprint cached for every run
    /// after. It is the same silent half-a-video as the splice bug and it needs
    /// no clock restart to happen -- a broadcaster switching resolution, a
    /// camcorder that drops to a lower mode, or an encode of two sources
    /// concatenated is enough.
    ///
    /// The fixture is the `.ts` segment above followed by the same footage at
    /// 320x240, so the answer is exact: three keyframes on each side of a join
    /// that changes both the clock and the geometry.
    #[test]
    fn test_a_stream_that_changes_resolution_mid_file_keeps_sampling() {
        init_ffmpeg_for_tests();

        let one = fingerprint_path(&fixture_named("test_video_segment.ts"));
        let two = fingerprint_path(&fixture_named("test_video_rescaled.ts"));

        assert_eq!(
            two.valid_hashes.len(),
            one.valid_hashes.len() * 2,
            "both halves are sampled: {} hashes against {} for the first half alone",
            two.valid_hashes.len(),
            one.valid_hashes.len()
        );

        // A count on its own would also pass on a rebuilt scaler writing
        // garbage, so the two halves are held against each other: they are the
        // same footage at twice the size, keyframe for keyframe, and after the
        // 64x64 box filter they have to land close. Stated in sigma of the
        // hash's own width -- the same arithmetic as `compare::sigma()`, which
        // is private -- so the bound travels if the hash ever widens.
        let sigma = (crate::compare::HASH_BITS as f64).sqrt() / 2.0;
        for (i, (a, b)) in two.valid_hashes[..one.valid_hashes.len()]
            .iter()
            .zip(&two.valid_hashes[one.valid_hashes.len()..])
            .enumerate()
        {
            let distance = (a ^ b).count_ones() as f64;
            assert!(
                distance <= 3.0 * sigma,
                "keyframe {} of the rescaled half is the same footage as the first half: \
                 {} bits apart",
                i,
                distance
            );
        }

        let ascending = two.valid_t_start.windows(2).all(|w| w[0] < w[1]);
        assert!(
            ascending,
            "the sample clock only ever goes forwards: {:?}",
            two.valid_t_start
        );
    }

    /// `--min-keyframes 0` removes the FLOOR, not the interval.
    ///
    /// The flag is a minimum sample count for short videos, imposed by making
    /// the interval finer; a minimum of none is a minimum that never binds, so
    /// what is left is `--keyframe-interval` on its own. It used to turn that
    /// off as well and decode every keyframe, which is the one reading of the
    /// flag under which raising `--keyframe-interval` and lowering
    /// `--min-keyframes` -- each of them a request for FEWER samples -- combine
    /// into a request for all of them.
    ///
    /// The cache is the half that made it more than a curiosity: those
    /// every-keyframe fingerprints were stamped with the interval the user
    /// asked for, so `-i 2 -m 0` and a default run produced byte-identical
    /// fingerprints under stamps that disagree, and alternating the two
    /// re-decoded the whole library each way.
    ///
    /// The fixture is 3 seconds with a keyframe every second, so the answer is
    /// exact at every rung below.
    #[test]
    fn test_a_floor_of_nothing_leaves_the_keyframe_interval_standing() {
        init_ffmpeg_for_tests();

        let path = fixture_named("test_video_segment.ts");
        let filepath = path.to_string_lossy().to_string();
        let size = std::fs::metadata(&path).unwrap().len();
        let samples = |kf_interval: f64, min_kf_samples: f64| {
            fingerprint_video(&filepath, kf_interval, min_kf_samples, 1, Floor::OFF, size, &|_| {})
                .expect("the fixture decodes")
                .expect("min_duration is off")
                .valid_hashes
                .len()
        };

        let every = samples(0.0, 12.0);
        assert_eq!(every, 3, "the fixture is 3 keyframes, one a second");

        // 3 s / 12 is finer than 2 s, so the floor wins and nothing is dropped.
        assert_eq!(samples(2.0, 12.0), every, "a floor this fine keeps every keyframe");

        // With no floor the interval decides alone: keep at 0 s, drop at 1 s,
        // keep at 2 s.
        for off in [0.0, -5.0, f64::NAN] {
            assert_eq!(
                samples(2.0, off),
                2,
                "--min-keyframes {} removes the floor, so a 2 s interval still applies",
                off
            );
        }

        // And it is the floor that is gone, not the flag: an interval of
        // nothing is still every keyframe, whatever this says.
        assert_eq!(samples(0.0, 0.0), every, "no interval to floor, so nothing is dropped");
    }

    /// The seek path is answered from the container's clock, so it is kept away
    /// from the containers whose clock is allowed to restart -- which is a
    /// property libavformat states about the format, not a guess about the
    /// file.
    #[test]
    fn test_only_the_containers_that_can_be_spliced_are_kept_out_of_the_seek_path() {
        init_ffmpeg_for_tests();

        for (name, expected) in [
            ("test_video_segment.ts", true),
            ("test_video.mp4", false),
        ] {
            let path = fixture_named(name);
            let ictx = open_video(&path.to_string_lossy()).expect("fixture opens");
            assert_eq!(
                clock_may_restart(&ictx),
                expected,
                "{} is {}a container that may restart its clock",
                name,
                if expected { "" } else { "not " }
            );
        }
    }

    /// The cheap half of the probe gate: nothing in the first 2 KB, and nothing
    /// in the name either, so the file is refused without libavformat ever
    /// opening it.
    #[test]
    fn test_a_file_no_demuxer_recognises_is_refused_at_the_first_probe() {
        init_ffmpeg_for_tests();

        let dir = tempfile::tempdir().unwrap();
        // Not `.txt`: FFmpeg's `tty` demuxer claims that one, which is why a
        // scan of a source tree decodes text files as ANSI art. `.rlib` is
        // claimed by nothing, so the name contributes no score and the bytes
        // have to earn one on their own.
        let path = dir.path().join("notes.rlib");
        std::fs::write(&path, "the quick brown fox\n".repeat(400)).unwrap();

        let Weighed::Undecodable(e) = weigh_decode(&path.to_string_lossy(), 0.0, 4.0, Floor::OFF, 8000) else {
            panic!("a text file is not a video");
        };
        let said = format!("{:#}", e);
        assert!(said.contains("2048"), "refused at the first probe, and says so: {}", said);
    }

    /// The other half, and the reason there are two: a name some demuxer claims
    /// scores 1 whatever the bytes say, so it can never be turned away at 2 KB.
    ///
    /// This is what stands between the gate and a real MP3 -- whose first frames
    /// sit behind an ID3 tag, which scores 1 at 2 KB and 51 by 8 KB. The bytes
    /// here never earn a score, so the file is still refused; what the test pins
    /// is WHERE, because that is the difference between reading 2 KB and reading
    /// enough.
    #[test]
    fn test_a_name_a_demuxer_claims_earns_a_second_look() {
        init_ffmpeg_for_tests();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("song.mp3");
        // Comfortably past SECOND_PROBE_BYTES, so the size the refusal reports
        // is the probe's own and not the file's -- see the test below, which
        // pins the other side of that.
        let text = "the quick brown fox\n".repeat(1_500);
        std::fs::write(&path, &text).unwrap();

        let Weighed::Undecodable(e) =
            weigh_decode(&path.to_string_lossy(), 0.0, 4.0, Floor::OFF, text.len() as u64)
        else {
            panic!("this is text, whatever it is called");
        };
        let said = format!("{:#}", e);
        assert!(
            said.contains("16384"),
            "a claimed extension has to reach the second probe: {}",
            said
        );
    }

    /// The refusal counts bytes that exist. A file shorter than the probe size
    /// is read short, judged on what it holds, and must say so: quoting the
    /// constant instead described 16 KB of a file that has none of them, and it
    /// is exactly the truncated downloads and interrupted copies that land here.
    #[test]
    fn test_a_file_shorter_than_the_probe_is_refused_over_the_bytes_it_has() {
        init_ffmpeg_for_tests();

        let dir = tempfile::tempdir().unwrap();
        // A claimed extension, so the second probe is reached and the constant
        // that would be quoted is the larger of the two.
        let path = dir.path().join("stub.mp3");
        std::fs::write(&path, "the quick brown fox\n").unwrap();

        let Weighed::Undecodable(e) = weigh_decode(&path.to_string_lossy(), 0.0, 4.0, Floor::OFF, 20)
        else {
            panic!("twenty bytes of text is not a video");
        };
        let said = format!("{:#}", e);
        assert!(said.contains("first 20 bytes"), "the bytes it really read: {}", said);
        assert!(!said.contains("16384"), "and not the ones it did not: {}", said);
    }

    /// The end of that same line. An empty file has no bytes to have recognised
    /// and no score worth quoting, so it gets a sentence about the one thing
    /// that is wrong with it rather than three clauses about a probe.
    #[test]
    fn test_an_empty_file_is_refused_for_being_empty() {
        init_ffmpeg_for_tests();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("interrupted.mp4");
        std::fs::write(&path, b"").unwrap();

        let Weighed::Undecodable(e) = weigh_decode(&path.to_string_lossy(), 0.0, 4.0, Floor::OFF, 0) else {
            panic!("an empty file is not a video");
        };
        let said = format!("{:#}", e);
        assert!(said.contains("empty"), "say what is wrong with it: {}", said);
        assert!(!said.contains("16384"), "not what a probe would have read: {}", said);
    }

    /// The rung between them. A one-byte file is the same class of near-empty
    /// thing the case above is about -- a truncated download, an interrupted
    /// copy -- and it is the one length at which the sentence has to say "byte".
    #[test]
    fn test_a_file_of_one_byte_is_refused_in_the_singular() {
        init_ffmpeg_for_tests();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.mp4");
        std::fs::write(&path, b"\0").unwrap();

        let Weighed::Undecodable(e) = weigh_decode(&path.to_string_lossy(), 0.0, 4.0, Floor::OFF, 1) else {
            panic!("one byte is not a video");
        };
        let said = format!("{:#}", e);
        assert!(said.contains("first 1 byte of"), "one byte, not one bytes: {}", said);
        assert!(!said.contains("1 bytes"), "the plural has no business here: {}", said);
    }

    /// And the plural is still the plural everywhere else, which is the half a
    /// naive fix breaks.
    #[test]
    fn test_more_than_one_byte_keeps_the_plural() {
        let said = format!("{}", NotMedia { bytes: 2, score: 0 });
        assert!(said.contains("first 2 bytes of"), "{}", said);
    }

    /// And the gate lets the real thing through untouched, which every other
    /// test in this file would also catch -- stated here because it is the half
    /// of the bargain the measurements cannot prove on their own.
    #[test]
    fn test_the_probe_gate_does_not_stand_in_front_of_a_real_video() {
        init_ffmpeg_for_tests();
        let path = fixture_path();
        let size = std::fs::metadata(&path).unwrap().len();
        assert!(
            matches!(weigh_decode(&path.to_string_lossy(), 0.0, 4.0, Floor::OFF, size), Weighed::Work(..)),
            "the fixture is a real MP4 and scores 100 on its first 2 KB"
        );
    }

    // --- Weighing the work before doing it -----------------------------------

    /// The weight of a file that has one. Every test below this line is
    /// asserting about a number, so the verdict is unwrapped in one place.
    fn work_of(weighed: Weighed) -> u64 {
        match weighed {
            Weighed::Work(weight, _) => weight,
            Weighed::Undecodable(e) => panic!("expected a weight, got a verdict: {:#}", e),
            Weighed::TooShort(d) => panic!("expected a weight, got a {:.2}s skip", d),
        }
    }

    /// The claim the scheduler rests on: the weight is the decode, counted
    /// before it happens. Cross-checked against the decode itself rather than
    /// against a literal, so it stays true if the fixture is ever replaced.
    #[test]
    fn test_a_weight_is_the_keyframes_that_will_be_decoded_times_their_pixels() {
        init_ffmpeg_for_tests();
        let path = fixture_path();
        let size = std::fs::metadata(&path).unwrap().len();
        let filepath = path.to_string_lossy().to_string();

        let fp = fingerprint_fixture(1);
        let decoded_pixels = fp.valid_hashes.len() as u64 * (fp.width * fp.height) as u64;

        let weight = work_of(weigh_decode(&filepath, 0.0, 4.0, Floor::OFF, size));
        let expected = (decoded_pixels as f64 * codec_cost(&fp.codec)) as u64;

        // Not an equality: the decode drops frames that hash the same as their
        // predecessor and frames with no structure in them, so the count it
        // ends up with is a floor on what the index promised. What must hold is
        // that the weight is that scale and not the file's 5.7 kB.
        assert!(
            weight >= expected && weight <= expected * 2,
            "weighed {} keyframe-pixels for a decode of {} ({}x{}, codec {})",
            weight,
            expected,
            fp.width,
            fp.height,
            fp.codec
        );
    }

    /// The seek that makes Matroska publish its Cues leaves MPEG-TS holding a
    /// single entry -- the one it landed on -- and one is not zero, so the
    /// ladder below never ran and every `.ts` in a library was weighed at one
    /// keyframe's pixels whatever its length. That is a constant, so the error
    /// grows with the file: this 30 s fixture was charged 3% of its decode, and
    /// the 1080p DVB captures the extension list is there for are charged 1%.
    ///
    /// All three of the things the weighing pass exists to get right go with
    /// it. The capture sorts LAST instead of first; `share_for` hands it one
    /// decoder thread at the only moment threads can still be handed out, since
    /// FFmpeg fixes them when the decoder opens; and the progress bar's
    /// denominator omits work that then has to be finished after the bar reads
    /// 100%.
    ///
    /// Checked against the decode rather than against a literal, and with room
    /// either side, because what it lands on now is the duration rung -- an
    /// estimate, and only ever meant to be the right scale. What must not come
    /// back is the constant.
    #[test]
    fn test_the_one_entry_a_seek_leaves_behind_is_not_a_keyframe_count() {
        init_ffmpeg_for_tests();

        let path = fixture_named("test_video_capture.ts");
        let filepath = path.to_string_lossy().to_string();
        let size = std::fs::metadata(&path).unwrap().len();

        let fp = fingerprint_path(&path);
        let pixels = (fp.width * fp.height) as f64 * codec_cost(&fp.codec);
        let truth = fp.valid_hashes.len() as f64 * pixels;

        let weight = work_of(weigh_decode(&filepath, 0.0, 12.0, Floor::OFF, size)) as f64;

        assert!(
            weight > pixels,
            "weighed {} keyframe-pixels, which is the one index entry the seek left behind, \
             for a decode of {} keyframes",
            weight,
            fp.valid_hashes.len()
        );
        assert!(
            weight >= truth / 4.0 && weight <= truth * 4.0,
            "a {:.0}s capture weighed {} against a decode of {} -- the weight has to be the \
             file's scale, not one keyframe's",
            fp.duration,
            weight,
            truth
        );
    }

    /// A file no decoder will open is a verdict, not a rung of the ladder. It
    /// used to be weighed by its bytes and queued, which bought a second open
    /// for the decode to fail at and a share of the progress bar for a decode
    /// that could not happen.
    #[test]
    fn test_a_file_that_cannot_be_opened_is_a_verdict_rather_than_a_weight() {
        init_ffmpeg_for_tests();

        let Weighed::Undecodable(e) =
            weigh_decode("/nonexistent/not-a-video.mkv", 0.0, 4.0, Floor::OFF, 1_000_000)
        else {
            panic!("a path with nothing behind it cannot be decoded, and this pass knows it");
        };

        // It says what went wrong and NOT which file: every caller prints the
        // path it was asked about, and a message that repeated it produced
        // "Failed to process X: Failed to open video file: X: ...", which is
        // the same path twice on a line there can be a quarter of a million of.
        let said = format!("{:#}", e);
        assert!(said.contains("Failed to open video file"), "{}", said);
        assert!(
            !said.contains("/nonexistent/not-a-video.mkv"),
            "the caller owns the name, not the error: {}",
            said
        );
    }

    /// The property the skip rests on: where this pass says a file has no
    /// fingerprint to give, the decode says the same thing in the same words.
    ///
    /// Written as an agreement rather than as an expected string on purpose --
    /// which of the three routes a given FFmpeg build takes on a given pile of
    /// bytes is not this tool's business (the linked library here opens random
    /// data as a low-score format and fails at the probe, where ffprobe rejects
    /// it outright), and any of them is fine. What is not fine is the two
    /// disagreeing, because then a file is reported broken on the strength of a
    /// question the decoder was never asked.
    #[test]
    fn test_the_weigher_and_the_decoder_agree_about_a_file_that_is_not_video() {
        init_ffmpeg_for_tests();

        // Deterministic bytes that are not any container: an LCG rather than
        // anything random, so a failure here is reproducible.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-video.mkv");
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let bytes: Vec<u8> = (0..300_000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let filepath = path.to_string_lossy().to_string();
        let size = bytes.len() as u64;

        let Weighed::Undecodable(weighed) = weigh_decode(&filepath, 0.0, 4.0, Floor::OFF, size) else {
            panic!("300 kB of noise is not a video, and opening it twice will not make it one");
        };

        let Err(decoded) = fingerprint_video(&filepath, 0.0, 4.0, 1, Floor::OFF, size, &|_| {}) else {
            panic!("the decode has to fail on the same file");
        };

        assert_eq!(
            format!("{:#}", weighed),
            format!("{:#}", decoded),
            "the weighing pass reports this file on the decode's behalf, so it has to be the \
             decode's own complaint"
        );
    }

    /// `--keyframe-interval` removes decode work that the file's size records no
    /// trace of, which is the case the byte proxy this replaced could not see at
    /// all: it charged an interval-subsampled run exactly what it charged a full
    /// one.
    #[test]
    fn test_subsampling_is_charged_at_what_it_will_actually_decode() {
        init_ffmpeg_for_tests();
        let filepath = fixture_path().to_string_lossy().to_string();
        let size = std::fs::metadata(fixture_path()).unwrap().len();

        let full = work_of(weigh_decode(&filepath, 0.0, 4.0, Floor::OFF, size));
        // One sample over the whole clip: the floor for short videos is what
        // decides this, exactly as it does inside the decode.
        let thinned = work_of(weigh_decode(&filepath, 3600.0, 1.0, Floor::OFF, size));

        assert!(
            thinned <= full,
            "subsampling cannot cost more than decoding everything ({} vs {})",
            thinned,
            full
        );
        assert!(thinned >= 1, "and never nothing");
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

    /// A container that never says how long it is must report that as 0, not as
    /// FFmpeg's sentinel divided by a million.
    ///
    /// `ictx.duration()` is AV_NOPTS_VALUE (i64::MIN) for a raw elementary
    /// stream, and the old fallback divided it straight through into
    /// -9223372036854.78 seconds. Every consumer downstream happened to reject
    /// that -- but only by failing a `> 0.0` guard, which is luck rather than
    /// design -- and it reached the report's sortable `length_seconds` column
    /// and the JSON verbatim, where a spreadsheet sorting on runtime got it at
    /// the top.
    ///
    /// The fixture is `test_video.mp4` remuxed to Annex-B H.264, which is the
    /// cheapest container with no duration field at all.
    #[test]
    fn test_a_container_with_no_duration_reports_zero_rather_than_a_sentinel() {
        init_ffmpeg_for_tests();
        let path = fixture_named("test_video_no_duration.h264");
        let filepath = path.to_string_lossy().to_string();
        let size = std::fs::metadata(&path).unwrap().len();

        let fp = fingerprint_video(&filepath, 0.0, 4.0, 1, Floor::OFF, size, &|_| {})
            .unwrap()
            .expect("min_duration is off, nothing should be skipped");

        assert_eq!(fp.duration, 0.0, "unknown runtime must read as unknown, not as -9.2e12");
        // And the figures derived from it stay honest rather than inheriting a
        // sign from the sentinel.
        assert_eq!(fp.bitrate(), 0);
        assert_eq!(fp.quality(), 0);
        assert!(!fp.valid_hashes.is_empty(), "the stream still fingerprints");
    }

    /// A phone's portrait video, stored landscape with a turn in its display
    /// matrix, fingerprints as the picture it shows: the same as a copy with
    /// the turn baked into its pixels, which is what every re-encode makes.
    /// Read as stored, the two were unrelated at every `-d`.
    #[test]
    fn test_a_turn_the_container_asks_for_is_applied_before_hashing() {
        init_ffmpeg_for_tests();
        let flagged = fingerprint_path(&fixture_named("test_video_rotated.mp4"));
        let baked = fingerprint_path(&fixture_named("test_video_turned.mp4"));

        assert_eq!((flagged.width, flagged.height), (180, 320), "reported as shown, not as stored");
        assert_eq!((baked.width, baked.height), (180, 320));
        assert_eq!(flagged.valid_hashes.len(), baked.valid_hashes.len());
        for (i, (a, b)) in flagged.valid_hashes.iter().zip(&baked.valid_hashes).enumerate() {
            let d = (a ^ b).count_ones();
            assert!(d <= 6, "sample {i} is {d} bits from its re-encode; unrelated frames sit near 32");
        }
    }

    /// Each quarter turn moves a corner to the next corner clockwise, and four
    /// of them are no turn at all.
    #[test]
    fn test_a_square_is_turned_clockwise() {
        let mut frame = vec![0u8; FRAME_STRIDE];
        frame[0] = 1; // top left
        let mut scratch = Vec::new();
        let corner = |f: &[u8]| f.iter().position(|&v| v == 1).unwrap();
        turn_square(&mut frame, 1, &mut scratch);
        assert_eq!(corner(&frame), SIDE - 1, "top right");
        turn_square(&mut frame, 1, &mut scratch);
        assert_eq!(corner(&frame), FRAME_STRIDE - 1, "bottom right");
        turn_square(&mut frame, 1, &mut scratch);
        assert_eq!(corner(&frame), (SIDE - 1) * SIDE, "bottom left");
        turn_square(&mut frame, 3, &mut scratch);
        assert_eq!(corner(&frame), FRAME_STRIDE - 1, "three turns are one turn back");
        turn_square(&mut frame, 2, &mut scratch);
        assert_eq!(corner(&frame), 0);
    }

    /// A Matroska file whose audio runs past its picture reports the video
    /// track's own runtime, not the segment's. Read off the segment, a 360p MKV
    /// with a long audio tail out-lasted the 1080p MP4 of the same footage and
    /// won the length ranking.
    #[test]
    fn test_a_matroska_track_is_as_long_as_its_own_duration_tag() {
        init_ffmpeg_for_tests();
        let fp = fingerprint_path(&fixture_named("test_video_audio_tail.mkv"));
        assert!((fp.duration - 2.0).abs() < 0.01, "the video track is 2 s, the audio 4.1 s: read {}", fp.duration);

        assert_eq!(parse_clock("00:00:02.000000000"), Some(2.0));
        assert_eq!(parse_clock("01:02:03.5"), Some(3723.5));
        assert_eq!(parse_clock("00:00:00.000000000"), None, "no runtime is not a runtime");
        assert_eq!(parse_clock("2.0"), None);
        assert_eq!(parse_clock("a:b:c"), None);
    }

    /// A video with nothing in the picture fails rather than fingerprinting to
    /// nothing.
    ///
    /// `is_featureless` drops a frame below `MIN_AC_ENERGY` so a fade or a title
    /// card cannot link two unrelated files, and the only guard on the result
    /// was `frame_idx == 0` -- which counts frames the DECODER produced, not
    /// hashes that survived. A file whose every frame is blank therefore
    /// returned a fingerprint with an empty hash list: it was cached, it matched
    /// nothing at any tolerance, and two identical all-black recordings reported
    /// zero groups at `-d 32 -p 0` with exit code 0 and an empty problem list.
    ///
    /// The failure has to be an error rather than a smaller answer, because the
    /// empty entry is indistinguishable from a good one on every run after --
    /// same `Stamp`, same file, no re-decode.
    #[test]
    fn test_a_video_with_no_features_in_it_fails_rather_than_fingerprinting_to_nothing() {
        init_ffmpeg_for_tests();
        let path = fixture_named("test_video_featureless.mp4");
        let filepath = path.to_string_lossy().to_string();
        let size = std::fs::metadata(&path).unwrap().len();

        let err = match fingerprint_video(&filepath, 0.0, 4.0, 1, Floor::OFF, size, &|_| {}) {
            Err(e) => e,
            Ok(_) => panic!("an all-black video has nothing to fingerprint"),
        };

        let said = format!("{:#}", err);
        assert!(
            said.contains("featureless"),
            "the sentence has to say what was wrong with the file, said: {}",
            said
        );
        // Distinct from the "decoded nothing" line: the frames arrived, they
        // just held no picture, and the two want different things done about it.
        assert!(
            !said.contains("No valid frames found"),
            "a file that decoded should not be reported as one that did not: {}",
            said
        );
        // And it is a verdict about the file, so a re-run is spared the decode.
        assert!(!is_transient(&err), "a blank video is blank again next run");
    }

    #[test]
    fn test_the_recorded_size_is_the_one_the_caller_measured() {
        // The field is no longer re-stat'ed here: it is the scan's own figure,
        // which is also what the cache stamp is built from. A caller that
        // measured 12345 bytes gets 12345 back, so the staleness check at
        // disposal time compares against the size the decision was made on.
        init_ffmpeg_for_tests();
        let filepath = fixture_path().to_string_lossy().to_string();

        let fp = fingerprint_video(&filepath, 0.0, 4.0, 1, Floor::OFF, 12_345, &|_| {})
            .unwrap()
            .unwrap();

        assert_eq!(fp.file_size, 12_345);
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

    /// The probe buffer belongs to the worker and is reused, so every reading it
    /// hands libavformat has to look exactly like a freshly allocated one: the
    /// real bytes, then `AVPROBE_PADDING_SIZE` of zeros, and nothing of the file
    /// before it. `av_probe_input_buffer2` memsets that tail before every one of
    /// its own probes and demuxers are allowed to read into it, so a stale byte
    /// there is this tool asking a question libavformat never asks.
    ///
    /// Both ways of getting it wrong are checked, because they fail on opposite
    /// files: the tail of the FIRST reading is where the second reading's real
    /// bytes land, and the tail of the SECOND is where the previous file's bytes
    /// are still sitting.
    #[test]
    fn test_every_probe_reading_is_padded_with_zeros_and_not_with_leftovers() {
        let dir = tempfile::tempdir().expect("temp dir");

        let long = dir.path().join("long.bin");
        std::fs::write(&long, vec![0xABu8; SECOND_PROBE_BYTES * 2]).expect("write");
        let mut file = std::fs::File::open(&long).expect("open");
        let mut buf = Vec::new();

        let first = fill_to(&mut file, &mut buf, 0, FIRST_PROBE_BYTES).expect("first read");
        assert_eq!(first, FIRST_PROBE_BYTES);
        assert!(
            buf[first..first + AVPROBE_PADDING_SIZE].iter().all(|&b| b == 0),
            "the first reading's padding is where the second reading's bytes land"
        );

        let second = fill_to(&mut file, &mut buf, first, SECOND_PROBE_BYTES).expect("second read");
        assert_eq!(second, SECOND_PROBE_BYTES);
        assert!(
            buf[..second].iter().all(|&b| b == 0xAB),
            "the second reading must not have lost the first reading's bytes"
        );
        assert!(
            buf[second..second + AVPROBE_PADDING_SIZE].iter().all(|&b| b == 0),
            "the second reading is padded too"
        );

        // Now a SHORT file through the same worker's buffer, which is still
        // holding 16 KB of 0xAB. Its padding must be zeros, not those.
        let short = dir.path().join("short.bin");
        std::fs::write(&short, [1u8, 2, 3, 4]).expect("write");
        let mut file = std::fs::File::open(&short).expect("open");
        let held = fill_to(&mut file, &mut buf, 0, FIRST_PROBE_BYTES).expect("short read");
        assert_eq!(held, 4, "a short file comes back short");
        assert!(
            buf[held..held + AVPROBE_PADDING_SIZE].iter().all(|&b| b == 0),
            "the previous file's bytes must not survive into this one's padding"
        );
    }

    /// The second reading is paid for only by the files that earned it. A file
    /// whose first 2 KB say nothing at all is settled there and the 14 KB behind
    /// it is never read, which is the whole saving; a file that showed a whisper
    /// goes on to the second size, which is the whole safety net. Both are
    /// visible in the byte count the refusal reports, which is the only place
    /// the difference surfaces.
    #[test]
    fn test_only_a_file_that_showed_a_whisper_is_read_past_the_first_probe() {
        let dir = tempfile::tempdir().expect("temp dir");

        let plain = dir.path().join("nothing.bin");
        std::fs::write(&plain, "hello world\n".repeat(3000)).expect("write");
        let (bytes, score) = unrecognised(&plain.to_string_lossy()).expect("text is not media");
        assert_eq!(score, 0, "nothing in it scored at all");
        assert_eq!(
            bytes, FIRST_PROBE_BYTES,
            "settled on the first reading, so that is all it may claim to have read"
        );

        // Pseudo-random bytes carry an accidental frame sync, which is exactly
        // the whisper `SECOND_PROBE_BYTES` exists to look past -- 16,652 files
        // of one home directory do this. It has to reach the second size.
        let mut byte = 0x5Au8;
        let junk: Vec<u8> = (0..SECOND_PROBE_BYTES * 2)
            .map(|_| {
                byte = byte.wrapping_mul(37).wrapping_add(11);
                byte
            })
            .collect();
        let noisy = dir.path().join("whisper.bin");
        std::fs::write(&noisy, &junk).expect("write");
        let (bytes, _) = unrecognised(&noisy.to_string_lossy()).expect("noise is not media");
        assert_eq!(
            bytes, SECOND_PROBE_BYTES,
            "a whisper at 2 KB must be disproved at 16 KB, not taken as a refusal"
        );
    }

    /// The bytes a refusal reports are the bytes the file really held, at both
    /// sizes -- the property `NotMedia::bytes` rests on, and the one the lazy
    /// second read could most easily have broken.
    #[test]
    fn test_a_refusal_never_claims_more_bytes_than_the_file_holds() {
        let dir = tempfile::tempdir().expect("temp dir");
        for size in [0usize, 1, 100, FIRST_PROBE_BYTES - 1, FIRST_PROBE_BYTES] {
            let path = dir.path().join(format!("junk_{size}.bin"));
            std::fs::write(&path, vec![0u8; size]).expect("write");
            let name = path.to_string_lossy().to_string();
            if let Some((bytes, _)) = unrecognised(&name) {
                assert!(
                    bytes <= size,
                    "a {size}-byte file was refused on {bytes} bytes it does not have"
                );
            }
        }
    }
    /// Every keyframe packet of a fixture, and a decoder ready to take them.
    ///
    /// Keyframes only, because that is all the demux loop ever sends: the
    /// streams are set to `AVDISCARD_NONKEY` before it starts.
    fn keyframes_and_decoder(
        name: &str,
    ) -> (Vec<ffmpeg_next::Packet>, impl Fn() -> ffmpeg_next::decoder::Video) {
        let path = fixture_named(name);
        let mut ictx = ffmpeg_next::format::input(&path)
            .unwrap_or_else(|e| panic!("fixture {} unreadable: {}", path.display(), e));
        let stream = ictx.streams().best(ffmpeg_next::media::Type::Video).unwrap();
        let index = stream.index();
        let parameters = stream.parameters();

        let mut packets = Vec::new();
        for (stream, packet) in ictx.packets() {
            if stream.index() == index && packet.is_key() {
                packets.push(packet.clone());
            }
        }

        let new_decoder = move || {
            ffmpeg_next::codec::context::Context::from_parameters(parameters.clone())
                .unwrap()
                .decoder()
                .video()
                .unwrap()
        };
        (packets, new_decoder)
    }

    /// Which keyframes came back out, identified by their PICTURE.
    ///
    /// A set rather than a count so that priming the decoder below -- which
    /// sends one packet more than once -- cannot inflate the answer: the
    /// question is which keyframes of the file were sampled, and a keyframe
    /// decoded twice is still one keyframe. By content rather than by
    /// timestamp for the same reason: a decoder pairs the timestamps it was
    /// given with the frames it produces in order, so an extra submission
    /// shifts every later pairing, and the picture is what this pass actually
    /// hashes anyway.
    fn keyframes_taken(
        decoder: &mut ffmpeg_next::decoder::Video,
        packets: &[ffmpeg_next::Packet],
        lost: &mut u32,
    ) -> std::collections::BTreeSet<u64> {
        let mut seen = std::collections::BTreeSet::new();
        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut take = |frame: &ffmpeg_next::frame::Video| {
            let mut digest = 0xcbf2_9ce4_8422_2325u64;
            for &byte in frame.data(0) {
                digest = (digest ^ byte as u64).wrapping_mul(0x100_0000_01b3);
            }
            seen.insert(digest);
        };
        for packet in packets {
            feed_keyframe(decoder, packet, &mut decoded, &mut take, lost)
                .expect("the fixture decodes");
        }
        let _ = decoder.send_eof();
        take_ready_frames(decoder, &mut decoded, &mut take, lost).expect("the fixture drains");
        seen
    }

    /// A decoder asking for its output to be read is answered, not abandoned.
    ///
    /// `EAGAIN` from `send_packet` is the decoder saying its queue is full and
    /// the packet must be resent once that queue has been drained. The demux
    /// loop used to read it as a failure -- `if decoder.send_packet(&packet)
    /// .is_ok() { while receive_frame ... }` -- which is the one reading it
    /// cannot recover from, because the drain that would clear the queue sits
    /// behind the same `&&`. So the queue stayed full, every later keyframe met
    /// the same full queue, and the file simply stopped being sampled there:
    /// no counter, no message, exit 0, and the short fingerprint cached, still
    /// claiming the header's runtime, so its last surviving hash stood for all
    /// the footage nothing had looked at.
    ///
    /// One skipped drain is all it takes to get there, and this test starts
    /// from exactly that state: packets sent with nothing read back, until the
    /// decoder says `EAGAIN`. Two of them are enough on this fixture.
    #[test]
    fn test_a_decoder_asking_for_its_output_to_be_read_keeps_every_keyframe() {
        init_ffmpeg_for_tests();

        let (packets, new_decoder) = keyframes_and_decoder("test_video_capture.ts");
        assert!(packets.len() >= 3, "need several keyframes to lose");

        let expected = keyframes_taken(&mut new_decoder(), &packets, &mut 0);
        assert_eq!(expected.len(), packets.len(), "a clean run samples every keyframe");

        // The wedge: send without reading until the decoder refuses to take any
        // more. Nothing here is contrived about the decoder -- this is what the
        // demux loop's own state looks like the moment after a drain is missed.
        let mut decoder = new_decoder();
        let mut wedged = false;
        for _ in 0..8 {
            match decoder.send_packet(&packets[0]) {
                Ok(()) => {}
                Err(ffmpeg_next::Error::Other { errno }) if errno == libc::EAGAIN => {
                    wedged = true;
                    break;
                }
                Err(e) => panic!("unexpected error priming the decoder: {e:?}"),
            }
        }
        assert!(wedged, "the decoder never filled up, so this proves nothing");

        assert_eq!(
            keyframes_taken(&mut decoder, &packets, &mut 0),
            expected,
            "a full decoder queue must cost no keyframe: it is drained and the packet resent"
        );
    }

    /// An answer this code has no model of is reported, not swallowed.
    ///
    /// A keyframe the decoder will not take is a sample missing from the middle
    /// of a fingerprint that still claims the whole file's runtime, which is the
    /// same wrong answer as a truncated one -- so it fails the file rather than
    /// returning it short, exactly as an unscalable frame does. `EAGAIN` is
    /// deliberately not one of these (see the test above); what is left has to
    /// come back as an error.
    ///
    /// Sending after the input has been declared over is the one such error
    /// that can be asked for on demand. The demux loop cannot reach it -- it
    /// sends `EOF` once, at the end -- and that is the point: this pins the
    /// wiring, not a route.
    #[test]
    fn test_a_keyframe_the_decoder_refuses_is_reported() {
        init_ffmpeg_for_tests();

        let (packets, new_decoder) = keyframes_and_decoder("test_video_capture.ts");
        let mut decoder = new_decoder();
        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut taken = 0usize;
        let mut take = |_: &ffmpeg_next::frame::Video| taken += 1;

        let _ = decoder.send_eof();
        take_ready_frames(&mut decoder, &mut decoded, &mut take, &mut 0)
            .expect("draining is not an error");

        let refused = feed_keyframe(&mut decoder, &packets[0], &mut decoded, &mut take, &mut 0);
        assert!(
            matches!(refused, Err(ffmpeg_next::Error::Eof)),
            "a decoder that will take nothing more must say so to the caller, got {refused:?}"
        );
    }

    /// A picture the decoder cannot build costs one sample, not the file.
    ///
    /// The other half of the test above, and the half that cost nine files of
    /// one folder. `send_packet` consumes the packet before it decodes, so a
    /// picture it cannot build is reported out of the SEND -- and reading that
    /// as "the decoder would not take a keyframe" threw away a 1080p episode
    /// over 8 frames of 835, then remembered the refusal so no later run would
    /// look at the file again. What reaches this in the wild is a 10-bit HEVC
    /// remux sampled key-frames-only, where hevcdec meets a POC already in its
    /// DPB; what reaches it here is one keyframe overwritten with nonsense,
    /// which is the same finding by a route a fixture can carry.
    ///
    /// Both halves are asserted, because either alone passes on a mistake: that
    /// the run does not fail, and that everything else in the file is still
    /// sampled and still sampled to the same pictures.
    #[test]
    fn test_a_keyframe_the_decoder_cannot_build_costs_one_sample_and_not_the_file() {
        init_ffmpeg_for_tests();

        let (packets, new_decoder) = keyframes_and_decoder("test_video_capture.ts");
        assert!(packets.len() >= 3, "need a keyframe in the middle to spoil");
        let clean = keyframes_taken(&mut new_decoder(), &packets, &mut 0);
        assert_eq!(clean.len(), packets.len(), "a clean run samples every keyframe");

        let spoiled_at = packets.len() / 2;
        let mut packets = packets;
        for byte in packets[spoiled_at]
            .data_mut()
            .expect("a keyframe packet carries data")
            .iter_mut()
        {
            *byte = 0xA5;
        }

        let mut lost = 0u32;
        let taken = keyframes_taken(&mut new_decoder(), &packets, &mut lost);

        assert_eq!(lost, 1, "exactly one picture was unbuildable, got {lost}");
        assert_eq!(
            taken.len(),
            clean.len() - 1,
            "one spoiled keyframe must cost one sample and no other"
        );
        assert!(
            taken.is_subset(&clean),
            "the keyframes that still decoded must decode to the same pictures"
        );
    }

    /// The line between the two, drawn where the API draws it.
    ///
    /// `avcodec_send_packet` documents four answers about the decoder's state
    /// and calls everything else a legitimate decoding error, and the whole of
    /// this file's handling hangs off which side an answer falls on: a
    /// transaction answer is a packet not taken, which the caller must act on;
    /// a decoding error is a packet already spent, which costs one sample. Get
    /// `EINVAL` or `ENOMEM` wrong and a broken decoder is silently counted
    /// frame by frame instead of failing; get `InvalidData` wrong and a whole
    /// video is refused over one picture.
    #[test]
    fn test_only_the_decoders_four_answers_about_itself_are_not_about_the_picture() {
        for transaction in [
            ffmpeg_next::Error::Eof,
            ffmpeg_next::Error::Other { errno: libc::EAGAIN },
            ffmpeg_next::Error::Other { errno: libc::EINVAL },
            ffmpeg_next::Error::Other { errno: libc::ENOMEM },
        ] {
            assert!(
                !is_decoding_error(&transaction),
                "{transaction:?} is an answer about the decoder, not about a picture"
            );
        }

        for decoding in [
            ffmpeg_next::Error::InvalidData,
            ffmpeg_next::Error::Bug,
            ffmpeg_next::Error::Unknown,
            ffmpeg_next::Error::PatchWelcome,
            ffmpeg_next::Error::Other { errno: libc::EIO },
        ] {
            assert!(
                is_decoding_error(&decoding),
                "{decoding:?} is not one of the four, so it is about a picture"
            );
        }
    }
}
