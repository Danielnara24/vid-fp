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

/// How many times in a row the demuxer may fail before the file is treated as
/// finished.
///
/// The backstop for a demuxer that errors forever without ever reaching EOF --
/// `input_is_spent` covers the far commoner case where it has, and mis-reports
/// it. Generous on purpose: a real file that stumbles this many times CONSECUTIVELY,
/// with not one packet in between, is damaged past the point where the tail of
/// its keyframes would be worth having. Each iteration is one failed read, so
/// even the full count costs milliseconds.
const MAX_CONSECUTIVE_DEMUX_ERRORS: u32 = 1024;


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
        write!(
            f,
            "no demuxer recognised the first {} bytes of this file (the best guess scored {} of \
             {}, which libavformat treats as no evidence)",
            self.bytes,
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
/// spelling "no opinion": `av_probe_input_format3` floors the score of any
/// demuxer whose extension list matches the NAME at 1 whatever the bytes said,
/// and `mp3_probe` returns 1 for a single frame sync anywhere in the buffer.
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
/// EXTENSION alone scores 1, so any file named like media -- `.mp4`, `.mkv`,
/// `.mp3`, anything at all that some demuxer claims -- is guaranteed to reach
/// the second look rather than being turned away at 2 KB. Only a file whose name
/// tells nothing AND whose first bytes match nothing is refused outright, which
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

/// How much of a file was read before concluding it is not media, and what the
/// best guess scored -- or `None` if it IS worth opening.
///
/// Two sizes rather than one, because the cheap answer is right about 90% of a
/// home directory and wrong about a real MP3. See `FIRST_PROBE_BYTES` and
/// `SECOND_PROBE_BYTES`.
///
/// A file that cannot be read here is never a verdict: it is handed to
/// `avformat_open_input` anyway, so the user gets the real error ("No such file
/// or directory", "Permission denied") rather than this function's opinion of a
/// file it never saw.
fn unrecognised(filepath: &str) -> Option<(usize, i32)> {
    let mut buf = read_head(filepath, SECOND_PROBE_BYTES)?;

    // What the file actually holds, which is what the two sizes below are capped
    // against so the refusal reports bytes that exist -- see `NotMedia::bytes`.
    // A short file comes back short and a missing tail is not evidence of
    // anything, so the reading is unchanged; only the sentence is.
    let held = buf.len().saturating_sub(AVPROBE_PADDING_SIZE);

    // The name is part of the question: `av_probe_input_format3` floors the score
    // of any demuxer whose extension list matches it at 1, whatever the bytes
    // say. That is the whole reason the second look exists -- and also why no
    // file named like media can ever be turned away by the first one.
    let cname = std::ffi::CString::new(filepath).ok()?;

    let first = probe_head(&mut buf, FIRST_PROBE_BYTES, &cname);
    if first > NO_EVIDENCE {
        return None;
    }
    if first == 0 {
        return Some((FIRST_PROBE_BYTES.min(held), first));
    }

    let second = probe_head(&mut buf, SECOND_PROBE_BYTES, &cname);
    if second > NO_EVIDENCE {
        return None;
    }
    Some((SECOND_PROBE_BYTES.min(held), second))
}

/// The first `want` bytes of a file, in a buffer with libavformat's probe
/// padding on the end -- the zeroed tail every `read_probe` is allowed to read
/// into. Short files come back short; the padding is still there.
fn read_head(filepath: &str, want: usize) -> Option<Vec<u8>> {
    use std::io::Read;

    let mut file = std::fs::File::open(filepath).ok()?;
    let mut buf = vec![0u8; want + AVPROBE_PADDING_SIZE];
    let mut filled = 0usize;
    while filled < want {
        match file.read(&mut buf[filled..want]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(_) => return None,
        }
    }
    buf.truncate(filled + AVPROBE_PADDING_SIZE);
    Some(buf)
}

/// What libavformat makes of the first `len` bytes of `buf`.
///
/// `is_opened = 1` is the value `av_probe_input_buffer2` passes, so this
/// considers exactly the demuxers the real open will consider.
fn probe_head(buf: &mut [u8], len: usize, filename: &std::ffi::CStr) -> i32 {
    let len = len.min(buf.len().saturating_sub(AVPROBE_PADDING_SIZE));
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
fn open_video(filepath: &str) -> Result<ffmpeg_next::format::context::Input> {
    open_input(filepath).map_err(|e| {
        if not_media(&e).is_some() {
            e
        } else {
            e.context("Failed to open video file")
        }
    })
}

/// Whether the demuxer has read everything there is, whatever error it chose to
/// report instead of saying so.
///
/// `avio_feof` is the byte-level truth: the packet layer's `AVERROR_EOF` is a
/// convention demuxers are free to ignore, and several do. Null-checked because
/// `pb` is only guaranteed for a context that owns its I/O.
fn input_is_spent(ictx: &ffmpeg_next::format::context::Input) -> bool {
    unsafe {
        let pb = (*ictx.as_ptr()).pb;
        !pb.is_null() && ffmpeg_next::ffi::avio_feof(pb) != 0
    }
}

/// Whether libavformat says this container is allowed to restart its clock
/// part way through.
///
/// `AVFMT_TS_DISCONT` is the demuxer's own declaration that the timestamps it
/// produces need not run forwards, and it is set on exactly the formats that
/// can be built by joining two recordings end to end: MPEG-TS (`.ts`, `.mts`,
/// `.m2ts` -- DVB captures, camcorder splits) and MPEG-PS (`.mpg`, `.vob`).
/// Every container that has to be remuxed rather than concatenated -- MP4,
/// Matroska, AVI, ASF -- leaves it clear, so the seek path this gates keeps the
/// file it was written for.
fn clock_may_restart(ictx: &ffmpeg_next::format::context::Input) -> bool {
    unsafe {
        let iformat = (*ictx.as_ptr()).iformat;
        !iformat.is_null() && ((*iformat).flags & ffmpeg_next::ffi::AVFMT_TS_DISCONT) != 0
    }
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
fn probe_streams(ictx: &mut ffmpeg_next::format::context::Input) -> Result<()> {
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
const WORK_PER_BYTE: f64 = 3.8;

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
/// `min_duration` is the `--min-duration` this run was given (seconds, 0 = off),
/// and it is here for the same reason the keyframe interval is: it decides how
/// much decoding this file implies, and for a file under it the answer is none.
pub fn weigh_decode(
    filepath: &str,
    kf_interval: f64,
    min_kf_samples: f64,
    min_duration: f64,
    size: u64,
) -> Weighed {
    let from_bytes = || ((size as f64 * WORK_PER_BYTE) as u64).max(1);

    match weigh_from_container(filepath, kf_interval, min_kf_samples, min_duration) {
        Ok(Some(weighed)) => weighed,
        Ok(None) => Weighed::Work(from_bytes()),
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
    /// Estimated decode cost, in keyframe-pixels.
    Work(u64),
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
/// `fingerprint_video` will NOT probe -- so it reads its `min_duration` against
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
fn weigh_from_container(
    filepath: &str,
    kf_interval: f64,
    min_kf_samples: f64,
    min_duration: f64,
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
    if min_duration > 0.0 && header_is_complete(&ictx) {
        if let Some(stream) = ictx.stream(stream_index) {
            let duration = duration_seconds(&stream, &ictx);
            if duration > 0.0 && duration < min_duration {
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
pub fn fingerprint_video(
    filepath: &str,
    kf_interval: f64,
    min_kf_samples: f64,
    decode_threads: usize,
    min_duration: f64,
    file_size: u64,
    progress: &dyn Fn(u64),
) -> Result<Option<VideoFingerprint>> {
    // 1. Native Zero-Copy Extraction (No Subprocess Overhead)
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
        // The cast is redundant against the headers we build on today and is
        // kept anyway: these constants have no declared type in FFmpeg, so
        // bindgen infers one per #define from the literal's width and sign, and
        // the results are not uniform. In the very same header AV_CODEC_FLAG2_*
        // comes out c_int except ICC_PROFILES (bit 31) which is u32, and the
        // whole AV_CODEC_FLAG_* family is c_uint -- which is why LOW_DELAY
        // below needs a cast that really does convert. Both fields are c_int.
        #[allow(clippy::unnecessary_cast)]
        {
            (*ctx).flags2 |= ffmpeg_next::ffi::AV_CODEC_FLAG2_FAST as i32;
        }
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

    // Built from the first frame the decoder produces rather than from the
    // header. Without the probe the container never reports a pixel format --
    // and this was always the more honest source anyway, since the decoder, not
    // the header, is the authority on what it is emitting.
    let mut scaler: Option<ffmpeg_next::software::scaling::context::Context> = None;
    // The geometry of the FIRST frame, which is what the report and the
    // resolution ranking read. A file that rescales part way through has no one
    // true answer and this is the one the header would have given, so no file's
    // reported resolution moves because of the rebuild below.
    let mut frame_dims: Option<(u32, u32)> = None;
    // Frames the decoder produced that could not be turned into a sample. Not a
    // count to report: any at all and the fingerprint covers less footage than
    // it claims to, which is the whole failure this guards against, so the file
    // is failed at the end and re-read next run rather than cached short.
    let mut unscalable: Option<ffmpeg_next::Error> = None;

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

    // Rebuilt whenever the decoder's output changes shape, not merely built once.
    // A stream is allowed to change resolution or pixel format mid-file -- a
    // broadcaster switching feeds, a camcorder dropping to a lower mode, two
    // recordings concatenated -- and `Context::run` refuses outright any frame
    // that is not the shape its context was built for. Built once and never
    // rebuilt, every frame after such a change returned `Error::InputChanged`,
    // both call sites discarded that, and the file simply stopped being sampled
    // at the change: nothing counted, nothing logged, exit code 0, and the
    // half-length fingerprint cached for every run after. That is the same
    // silent half-a-video as the splice bug and it needs no clock restart to
    // reach -- one `cat` of two encodes at different sizes is enough.
    //
    // The check is asked of the context's own record of what it was built for
    // rather than of a second copy kept beside it, so there is nothing to drift.
    let mut process_frame = |dec: &ffmpeg_next::frame::Video| -> Result<(), ffmpeg_next::Error> {
        let rebuild = match &scaler {
            None => true,
            Some(built) => {
                let input = built.input();
                input.format != dec.format()
                    || input.width != dec.width()
                    || input.height != dec.height()
            }
        };
        if rebuild {
            scaler = Some(ffmpeg_next::software::scaling::context::Context::get(
                dec.format(),
                dec.width(),
                dec.height(),
                ffmpeg_next::format::Pixel::GRAY8,
                64,
                64,
                ffmpeg_next::software::scaling::flag::Flags::FAST_BILINEAR,
            )?);
            frame_dims.get_or_insert((dec.width(), dec.height()));
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
    //
    // The interval is the setting; the floor only ever makes it FINER. So a
    // floor that cannot be computed -- 0, a negative, NaN -- means "no floor",
    // and the interval stands on its own. It used to mean "no interval either":
    // `-i 3 -m 0` decoded every keyframe, which is the one reading of a
    // MINIMUM that turns the maximum spacing off as well, and it stamped those
    // every-keyframe fingerprints with `kf_interval = 3.0` -- so alternating
    // `-i 3 -m 0` with a default run re-decoded the whole library each way for
    // a setting that provably changed not one frame. Same class of bug as the
    // one `main::same_sampling` fixed for the interval itself.
    //
    // A duration of zero is deliberately still "sample everything", and that is
    // not the same question: with no runtime there is nothing for the interval
    // to be measured against here either, and `sample_times` is what recovers
    // such a file's clock afterwards.
    let floor = if min_kf_samples > 0.0 { duration_sec / min_kf_samples } else { f64::INFINITY };
    let effective_interval = if kf_interval > 0.0 && duration_sec > 0.0 {
        kf_interval.min(floor)
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
    // A container whose clock is allowed to restart never enters the seek path
    // at all, because a timestamp there is no guide to what is further into the
    // file. `cat s1.ts s2.ts > capture.ts` gives the second segment the same
    // timestamps as the first, and every way of navigating by them lands
    // somewhere wrong: a forward seek out of segment one binary-searched its
    // way into the middle of segment two (three of six keyframes read, two of
    // them never looked at), and on the mpeg-ps build of the same file it
    // reached the end of segment one and then reported EOF. What the seek does
    // wrong varies with the demuxer; that it cannot be trusted does not.
    //
    // The decoder is deliberately NOT flushed across a seek. Every packet it is
    // fed is a keyframe and therefore self-contained, so there is no state to
    // invalidate, and flushing would throw away frames still in flight.
    let mut packet = ffmpeg_next::codec::packet::Packet::empty();
    let mut last_key_ts: Option<i64> = None;
    // Whether a seek has been issued that no keyframe has been accepted after
    // yet. The dedup guard is asked only while this is set: a seek is the only
    // thing that can hand the loop a frame it has already had, and asking it of
    // a plain linear read is what used to truncate a spliced file at the join.
    let mut after_seek = false;
    let mut demuxer_ignores_discard = false;
    let mut seeking_broken = clock_may_restart(&ictx);
    // Consecutive failed reads. Reset by any successful one, so a file that
    // stumbles and recovers is unaffected however often it stumbles.
    let mut demux_errors: u32 = 0;

    // --- The sample clock ----------------------------------------------------
    //
    // Kept as a running total of the gaps between keyframes rather than as an
    // offset from the first one, because a container's clock is not guaranteed
    // to run forwards for the whole file. Concatenating two recordings -- how a
    // DVB capture, a camcorder split and half the `.ts` files on a disk are
    // made -- restarts it at the join, and measuring from the first keyframe
    // then puts the whole second half back on top of the first half's times.
    // Accumulating gaps instead means the clock only ever moves forward, and a
    // restart costs one estimated gap rather than every sample after it.
    //
    // For a file whose clock does run forwards -- every file with one segment,
    // which is nearly all of them -- this is `ts - first_key_ts` exactly, term
    // by term, and no sample time moves.
    let mut prev_raw_ts: Option<i64> = None;
    let mut elapsed: i64 = 0;
    // The last forward gap seen, which is what a restart is charged. There is
    // no way to know how long the join really was: the second segment's own
    // clock says only where it starts, not when that was.
    let mut last_gap: i64 = 0;
    // Set the first time the clock goes backwards. Two things read it: seeking,
    // which is answered from that clock and is therefore no longer trustworthy,
    // and the runtime this file reports (see below), which was measured from
    // the same broken arithmetic.
    let mut clock_restarted = false;

    loop {
        if shutdown_requested() {
            return Err(anyhow!("Interrupted while fingerprinting"));
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
            Ok(()) => demux_errors = 0,
            Err(ffmpeg_next::Error::Eof) => break,
            // A recoverable demux error skips the packet rather than ending the
            // video -- but only while there is something left to skip TO.
            //
            // `AVERROR_EOF` is a convention, not a guarantee, and a demuxer that
            // does not follow it turns this arm into an infinite loop: it is
            // asked for a packet at the end of the file, says "invalid data"
            // instead of "end of file", and is asked again forever. That is not
            // hypothetical and it is not obscure -- `ncdec.c` ends
            // `while (state != NC_VIDEO_FLAG) { if (avio_feof(pb)) return
            // AVERROR_INVALIDDATA; ... }`, and the `nc` demuxer claims any file
            // named `*.v` on an extension match alone. A 132-byte linker version
            // script (`libavcodec.v`, one per FFmpeg build directory) probes as
            // an mpeg4 video of unspecified size, and fingerprinting it span one
            // core until the process was killed: 80 minutes, 2.9 MB RSS, no
            // output. Under `-x '*'` a handful of those is the whole run, since
            // every decode worker eventually lands on one.
            //
            // So the loop ends when the input is spent, whatever the demuxer
            // calls that, and `MAX_CONSECUTIVE_DEMUX_ERRORS` catches the same
            // shape of failure mid-file, where `feof` is not yet true. Both are
            // ordinary ends of stream rather than errors: the file keeps the
            // frames it did give up, and if that is none, the caller's existing
            // "no frames" path reports it.
            Err(_) => {
                demux_errors += 1;
                if demux_errors >= MAX_CONSECUTIVE_DEMUX_ERRORS || input_is_spent(&ictx) {
                    break;
                }
                continue;
            }
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
        //
        // Asked only after a seek, because a seek is the only thing that can
        // produce one. Asked of every keyframe -- which is what it used to be
        // -- it reads a container whose clock restarts mid-file as a seek that
        // keeps landing short, and throws away everything after the restart:
        // `cat s1.ts s2.ts > capture.ts` kept the five keyframes of its first
        // segment and silently dropped all five of its second. A file that
        // CONTAINED another was then fingerprinted as its equal, ranked below
        // it on encoder quality, and marked DELETE against it -- exit code 0,
        // nothing in the problem summary, and the truncated fingerprint cached
        // for every run after.
        if after_seek {
            if matches!((index_ts, last_key_ts), (Some(ts), Some(last)) if ts <= last) {
                continue;
            }
            after_seek = false;
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
        //
        // The gap is accumulated rather than subtracted so that a clock which
        // restarts mid-file still produces times that only go forwards -- see
        // `prev_raw_ts` above. A backwards step is a splice, never a late
        // packet: only keyframes reach this point, and they are handed over in
        // decode order.
        let sample_ts = match index_ts {
            Some(ts) => {
                match prev_raw_ts {
                    None => elapsed = 0,
                    Some(prev) if ts >= prev => {
                        let gap = ts - prev;
                        if gap > 0 {
                            last_gap = gap;
                        }
                        elapsed = elapsed.saturating_add(gap);
                    }
                    // The clock restarted. Charge the join the last gap this
                    // file showed -- an estimate, and the only one available --
                    // and stop navigating by timestamps, which have just been
                    // shown to be no guide to what is further into the file.
                    Some(_) => {
                        clock_restarted = true;
                        seeking_broken = true;
                        elapsed = elapsed.saturating_add(last_gap.max(1));
                    }
                }
                prev_raw_ts = Some(ts);
                Some(elapsed)
            }
            None => None,
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
                if let Err(e) = process_frame(&decoded) {
                    unscalable.get_or_insert(e);
                }
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
                    if ok {
                        after_seek = true;
                    } else {
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
        if let Err(e) = process_frame(&decoded) {
            unscalable.get_or_insert(e);
        }
    }

    // A frame the decoder produced and this pass could not use. Failed rather
    // than returned short, because a fingerprint quietly missing its second
    // half is not a smaller answer but a wrong one: it reads as the same length
    // as the file it was cut from, wins the ranking on encoder quality, and
    // marks that file DELETE -- and cached, it does so on every run after.
    //
    // Asked before the "no valid frames" line below rather than after it, so a
    // file that decoded and could not be scaled says which of the two happened.
    // The rebuild above answers the one cause of this that is a normal property
    // of a stream, so what is left is a pixel format libswscale will not take
    // or a frame with no size -- neither of which a single frame can have on
    // its own, which is why in practice this fires for a whole file or not at
    // all. Nothing in the 756-file local corpus reaches it.
    if let Some(e) = unscalable {
        return Err(anyhow!(e).context("Failed to convert a decoded frame"));
    }

    if frame_idx == 0 {
        return Err(anyhow!("No valid frames found or successfully decoded"));
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

    // Indexed rather than iterated on purpose, and clippy's needless_range_loop
    // is wrong about it: each cell reads two flat 64x64 buffers and updates two
    // 64-entry projections, so the iterator form is a zip of chunks_exact(64)
    // over sum and sum_sq nested inside a zip against row_max_var -- and
    // col_max_var is walked afresh on every row rather than consumed once.
    // y and x are coordinates here, not stand-ins for a cursor.
    #[allow(clippy::needless_range_loop)]
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

    // A container whose clock restarted mis-stated its runtime by the same
    // arithmetic: what a demuxer reports is its last timestamp minus its first,
    // and across a splice both of those belong to segments, not to the file --
    // two five-second recordings cat'ed together report five seconds. The
    // samples are the better witness here, because they were taken by walking
    // the whole file, so where they outrun the header the header is what is
    // wrong.
    //
    // This is a ranking metric, not a cosmetic one. Length is the first thing
    // `utils::find_best` compares and the property that normally keeps a long
    // file safe from the clip cut out of it; left at the header's figure, a
    // capture that CONTAINS another video reads as no longer than it, loses the
    // comparison on encoder quality or size, and is the one marked DELETE.
    //
    // Only a file that was seen to restart its clock is touched, so no runtime
    // that was merely rounded moves, and `total_ms >= duration * 1000` -- which
    // `MatchIndex::matched_seconds` rests on -- still holds, with equality.
    let duration_sec = if clock_restarted {
        duration_sec.max(total_ms as f64 / 1000.0)
    } else {
        duration_sec
    };

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

    /// One file, sampled at every keyframe it has.
    fn fingerprint_path(path: &std::path::Path) -> VideoFingerprint {
        init_ffmpeg_for_tests();

        let filepath = path.to_string_lossy().to_string();
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        fingerprint_video(&filepath, 0.0, 4.0, 1, 0.0, size, &|_| {})
            .unwrap_or_else(|e| panic!("failed to fingerprint {}: {:?}", filepath, e))
            .expect("min_duration is off, nothing should be skipped")
    }

    fn fingerprint_fixture(threads: usize) -> VideoFingerprint {
        init_ffmpeg_for_tests();

        let fixture_path = fixture_path();
        let filepath = fixture_path.to_string_lossy().to_string();
        assert!(fixture_path.exists(), "Fixture video not found at: {}.", filepath);

        let size = std::fs::metadata(&fixture_path).map(|m| m.len()).unwrap_or(0);
        let result = fingerprint_video(&filepath, 0.0, 4.0, threads, 0.0, size, &|_| {});
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
        let skipped = fingerprint_video(&filepath, 0.0, 4.0, 1, 3600.0, size, &|_| {}).unwrap();
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

        let Weighed::TooShort(duration) = weigh_decode(&filepath, 0.0, 4.0, 3600.0, size) else {
            panic!("the fixture is about a second long and an hour-long floor rejects it");
        };
        assert!(duration > 0.0, "the verdict carries the runtime it was measured against");

        let under = duration - 0.01;
        assert!(
            matches!(weigh_decode(&filepath, 0.0, 4.0, under, size), Weighed::Work(_)),
            "a floor under the runtime is not one this file falls under"
        );
        assert!(
            fingerprint_video(&filepath, 0.0, 4.0, 1, under, size, &|_| {})
                .unwrap()
                .is_some(),
            "and the decode reads it the same way, which is what lets the weigher answer at all"
        );

        let over = duration + 0.01;
        assert!(
            matches!(weigh_decode(&filepath, 0.0, 4.0, over, size), Weighed::TooShort(_)),
            "a floor over the runtime is"
        );
        assert!(
            fingerprint_video(&filepath, 0.0, 4.0, 1, over, size, &|_| {}).unwrap().is_none(),
            "and again the decode agrees, on the same side of the same number"
        );

        assert!(
            matches!(weigh_decode(&filepath, 0.0, 4.0, 0.0, size), Weighed::Work(_)),
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
        fingerprint_video(&filepath, 0.0, 4.0, 1, 0.0, size, &|pos| {
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
        fingerprint_video(&filepath, 0.0, 4.0, 1, 0.0, size, &|pos| {
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
            let outcome = fingerprint_video(&filepath, 0.0, 4.0, 1, 0.0, size, &|_| {});
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
            fingerprint_video(&filepath, kf_interval, min_kf_samples, 1, 0.0, size, &|_| {})
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

        let Weighed::Undecodable(e) = weigh_decode(&path.to_string_lossy(), 0.0, 4.0, 0.0, 8000) else {
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
            weigh_decode(&path.to_string_lossy(), 0.0, 4.0, 0.0, text.len() as u64)
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

        let Weighed::Undecodable(e) = weigh_decode(&path.to_string_lossy(), 0.0, 4.0, 0.0, 20)
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

        let Weighed::Undecodable(e) = weigh_decode(&path.to_string_lossy(), 0.0, 4.0, 0.0, 0) else {
            panic!("an empty file is not a video");
        };
        let said = format!("{:#}", e);
        assert!(said.contains("empty"), "say what is wrong with it: {}", said);
        assert!(!said.contains("16384"), "not what a probe would have read: {}", said);
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
            matches!(weigh_decode(&path.to_string_lossy(), 0.0, 4.0, 0.0, size), Weighed::Work(_)),
            "the fixture is a real MP4 and scores 100 on its first 2 KB"
        );
    }

    // --- Weighing the work before doing it -----------------------------------

    /// The weight of a file that has one. Every test below this line is
    /// asserting about a number, so the verdict is unwrapped in one place.
    fn work_of(weighed: Weighed) -> u64 {
        match weighed {
            Weighed::Work(weight) => weight,
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

        let weight = work_of(weigh_decode(&filepath, 0.0, 4.0, 0.0, size));
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

    /// A file no decoder will open is a verdict, not a rung of the ladder. It
    /// used to be weighed by its bytes and queued, which bought a second open
    /// for the decode to fail at and a share of the progress bar for a decode
    /// that could not happen.
    #[test]
    fn test_a_file_that_cannot_be_opened_is_a_verdict_rather_than_a_weight() {
        init_ffmpeg_for_tests();

        let Weighed::Undecodable(e) =
            weigh_decode("/nonexistent/not-a-video.mkv", 0.0, 4.0, 0.0, 1_000_000)
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

        let Weighed::Undecodable(weighed) = weigh_decode(&filepath, 0.0, 4.0, 0.0, size) else {
            panic!("300 kB of noise is not a video, and opening it twice will not make it one");
        };

        let Err(decoded) = fingerprint_video(&filepath, 0.0, 4.0, 1, 0.0, size, &|_| {}) else {
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

        let full = work_of(weigh_decode(&filepath, 0.0, 4.0, 0.0, size));
        // One sample over the whole clip: the floor for short videos is what
        // decides this, exactly as it does inside the decode.
        let thinned = work_of(weigh_decode(&filepath, 3600.0, 1.0, 0.0, size));

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

        let fp = fingerprint_video(&filepath, 0.0, 4.0, 1, 0.0, size, &|_| {})
            .unwrap()
            .expect("min_duration is off, nothing should be skipped");

        assert_eq!(fp.duration, 0.0, "unknown runtime must read as unknown, not as -9.2e12");
        // And the figures derived from it stay honest rather than inheriting a
        // sign from the sentinel.
        assert_eq!(fp.bitrate(), 0);
        assert_eq!(fp.quality(), 0);
        assert!(!fp.valid_hashes.is_empty(), "the stream still fingerprints");
    }

    #[test]
    fn test_the_recorded_size_is_the_one_the_caller_measured() {
        // The field is no longer re-stat'ed here: it is the scan's own figure,
        // which is also what the cache stamp is built from. A caller that
        // measured 12345 bytes gets 12345 back, so the staleness check at
        // disposal time compares against the size the decision was made on.
        init_ffmpeg_for_tests();
        let filepath = fixture_path().to_string_lossy().to_string();

        let fp = fingerprint_video(&filepath, 0.0, 4.0, 1, 0.0, 12_345, &|_| {})
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
}