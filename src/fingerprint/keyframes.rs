//! Reading a video's keyframes: which ones, when, and what they show.
//!
//! `fingerprint_video` is a pipeline of five parts, each of which owns one
//! question and every rule about it:
//!
//! * [`Keyframes`] -- which packets are the video stream's keyframes, in file
//!   order, and when the input is over. It may SEEK between them to avoid
//!   reading the bytes in between; that is its own business and nothing it
//!   does changes which keyframes come out.
//! * [`SampleClock`] -- when each keyframe happens, on a clock that only runs
//!   forwards even where the container's restarts.
//! * [`Thinning`] -- which of them `--keyframe-interval` keeps.
//! * [`KeyframeDecoder`] -- the decoder transaction: feed, drain, resend, and
//!   which of its answers cost a sample and which fail the file.
//! * [`FrameCapture`] -- each decoded picture scaled to the working frame,
//!   turned as it is shown, and kept with its time.
//!
//! Every one of these used to be inline in one demux loop sharing twenty
//! locals, and each bug that loop had (a clock restart truncating a capture, a
//! resolution change dropping the rest of a file, a full decoder queue losing
//! every later keyframe, a demuxer that never says EOF) was fixed with one
//! more flag in it. Here each lives in the part it is about.

use anyhow::{anyhow, Context, Result};
// `Packet::as_mut_ptr` lives on this trait; the reader unrefs the packet it
// reuses through the raw pointer.
use ffmpeg_next::codec::packet::Mut as _;
use ffmpeg_next::format::context::Input;
use ffmpeg_next::frame::Video as Frame;

use super::{turn_square, Step, FRAME_STRIDE, MAX_DECODE_THREADS, SIDE};
use crate::utils::shutdown_requested;

/// `FF_THREAD_FRAME` and `FF_THREAD_SLICE` from libavcodec. Spelled out rather
/// than taken from the bindings because they are plain `#define`s, and the name
/// bindgen gives them has moved between ffmpeg-sys-next releases.
const FF_THREAD_FRAME: i32 = 1;
const FF_THREAD_SLICE: i32 = 2;

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

/// How many pictures in a row the decoder may fail to produce before the file
/// is failed rather than sampled short.
///
/// The backstop for the other half of `is_decoding_error`: a single picture the
/// decoder could not build is one lost sample, which is what a featureless
/// frame already costs and is not worth failing a file over, but a decoder that
/// produces NOTHING while claiming to consume everything is a fingerprint made
/// of nothing at all. Counted consecutively -- one good frame resets it -- and
/// generous for the same reason the demux figure is. A file whose every
/// keyframe fails still lands on the "no valid frames" path, so this is only
/// about the middle of a file.
const MAX_CONSECUTIVE_DECODE_ERRORS: u32 = 1024;

/// The timestamp a keyframe is timed and indexed by: decode time, falling back
/// to presentation time.
///
/// Decode time rather than presentation time because `AVDISCARD_NONKEY` is not
/// free. MP4 stores presentation time as a per-sample offset from decode time
/// (the `ctts` table) and the demuxer walks that table with a cursor that only
/// advances over samples it RETURNS, so with the non-key samples discarded the
/// pts handed over is a real offset belonging to the wrong sample: on one clip
/// the container says 0 / 29029 / 58058 and libavformat reports 0 / 32032 /
/// 57057. Decode time comes out of a plain cumulative sum (`stts`) with no
/// cursor to lose, and differs from presentation time by the codec's reorder
/// delay, which is constant across a file's keyframes -- so measured from the
/// first keyframe (`SampleClock`) it reproduces the container's presentation
/// times exactly. It is also the timestamp the container's index is keyed on,
/// which is what a seek target is measured against.
pub(crate) fn decode_stamp(packet: &ffmpeg_next::Packet) -> Option<i64> {
    packet.dts().or_else(|| packet.pts())
}

/// Seconds per tick of a stream's timestamps, or 0 when the header never said.
pub(crate) fn seconds_per_tick(stream: &ffmpeg_next::format::stream::Stream<'_>) -> f64 {
    let tb = stream.time_base();
    if tb.denominator() > 0 {
        tb.numerator() as f64 / tb.denominator() as f64
    } else {
        0.0
    }
}

/// Milliseconds for `ticks` of a clock running at `tb` seconds a tick, as the
/// scan stores a sample's time.
pub(crate) fn ticks_to_ms(ticks: i64, tb: f64) -> u32 {
    (ticks as f64 * tb * 1000.0) as u32
}

/// When each keyframe happens: ticks since the first keyframe, accumulated gap
/// by gap.
///
/// A running total of the gaps rather than an offset from the first keyframe,
/// because a container's clock is not guaranteed to run forwards for the whole
/// file. Concatenating two recordings -- how a DVB capture, a camcorder split
/// and half the `.ts` files on a disk are made -- restarts it at the join, and
/// measuring from the first keyframe then put the whole second half back on top
/// of the first half's times. A backwards step is always a restart, never a
/// late packet: only keyframes reach the clock, in decode order. It is charged
/// the last forward gap seen, which is an estimate and the only one available,
/// since the second segment's own clock says only where it starts.
///
/// For a file whose clock does run forwards -- nearly all of them -- this is
/// `ts - first_ts` exactly, term by term. Anchoring at the first keyframe also
/// cancels a first keyframe with a negative dts (an MP4 with B-frames opens at
/// -1001) and a stream whose clock does not start at zero.
///
/// One definition for the scan and for `thumbnail::frame_at`, which has to find
/// a keyframe by the very time the scan gave it.
#[derive(Default)]
pub(crate) struct SampleClock {
    prev: Option<i64>,
    elapsed: i64,
    last_gap: i64,
    restarted: bool,
}

impl SampleClock {
    /// The time of the keyframe stamped `ts`, in ticks since the first one.
    pub(crate) fn tick(&mut self, ts: i64) -> i64 {
        match self.prev {
            None => self.elapsed = 0,
            Some(prev) if ts >= prev => {
                let gap = ts - prev;
                if gap > 0 {
                    self.last_gap = gap;
                }
                self.elapsed = self.elapsed.saturating_add(gap);
            }
            Some(_) => {
                self.restarted = true;
                self.elapsed = self.elapsed.saturating_add(self.last_gap.max(1));
            }
        }
        self.prev = Some(ts);
        self.elapsed
    }

    /// Whether the container's clock has been seen to go backwards. From then
    /// on its timestamps are no guide to what is further into the file, and
    /// the runtime the header reports was measured from the same broken clock.
    pub(crate) fn restarted(&self) -> bool {
        self.restarted
    }
}

/// Which keyframes `--keyframe-interval` keeps: no two closer than the interval.
///
/// The interval is absolute, not scaled with duration, because clip detection
/// works only while a clip and its host are sampled on the same grid; a long
/// host sampled sparsely would have no samples inside a short clip's window.
/// `--min-keyframes` FLOORS it for short videos so they keep at least that many
/// samples. The floor only ever makes the interval finer, so a floor that
/// cannot be computed (0, a negative, NaN) means "no floor" and the interval
/// stands on its own -- it once meant "no interval either", which decoded every
/// keyframe while stamping the entry with the interval. With no runtime there
/// is nothing to measure the interval against, and everything is kept.
/// A keyframe with no time is always kept.
pub(super) struct Thinning {
    /// Seconds; 0 keeps everything.
    interval: f64,
    last_kept: Option<f64>,
}

impl Thinning {
    pub(super) fn new(kf_interval: f64, min_kf_samples: f64, duration_sec: f64) -> Self {
        let floor = if min_kf_samples > 0.0 { duration_sec / min_kf_samples } else { f64::INFINITY };
        let interval = if kf_interval > 0.0 && duration_sec > 0.0 { kf_interval.min(floor) } else { 0.0 };
        Thinning { interval, last_kept: None }
    }

    /// Whether the keyframe at `t` seconds is kept.
    pub(super) fn keep(&mut self, t: Option<f64>) -> bool {
        if self.interval <= 0.0 {
            return true;
        }
        let Some(t) = t else { return true };
        let kept = self.last_kept.is_none_or(|last| t - last >= self.interval);
        if kept {
            self.last_kept = Some(t);
        }
        kept
    }

    /// Seconds after `t` before another keyframe could be kept, if any: the
    /// stretch a reader may skip without looking.
    pub(super) fn wait_after(&self, t: Option<f64>) -> Option<f64> {
        if self.interval <= 0.0 {
            return None;
        }
        let remaining = self.last_kept? + self.interval - t?;
        (remaining > 0.0).then_some(remaining)
    }
}

/// The video stream's keyframes, in file order.
///
/// Non-key packets are asked away (`AVDISCARD_NONKEY`) and every other stream
/// dropped outright, so their payloads are never read or copied. MP4 honours
/// that by reading only its sync samples; Matroska does not, and hands back
/// every block in the file -- 615 MB read of a 714 MB episode, for nothing.
/// So once a demuxer proves it ignores the request, by handing over a non-key
/// packet, the reader SEEKS from each keyframe to the next instead
/// (`seek_past`). Two properties keep that from changing what comes out:
///
/// * a seek that lands on or before a keyframe already returned is caught by
///   the guard in `next`, which is asked only after a seek -- asked of a
///   linear read, it took a spliced capture's restarted clock for a seek
///   landing short and dropped the whole second segment;
/// * anything that goes wrong -- no index, no timestamps, a failed seek, a
///   clock seen to restart -- turns seeking off for good and the file is
///   finished as a plain linear read.
///
/// A container whose clock may restart (`clock_may_restart`) never seeks at
/// all: a forward seek out of a spliced `.ts` binary-searched into the middle
/// of its second segment, and on the MPEG-PS build of the same file reached
/// the end of the first and reported EOF. For MPEG-TS this costs nothing, its
/// seek being a binary search over the same bytes a read covers.
///
/// The input is over when the demuxer says EOF, when `avio_feof` says so
/// whatever the demuxer says (`ncdec.c` answers "invalid data" at EOF forever,
/// and a 132-byte linker script named `*.v` spun a core for 80 minutes), or
/// after `MAX_CONSECUTIVE_DEMUX_ERRORS` failed reads in a row. Any other failed
/// read is a damaged packet, skipped.
pub(super) struct Keyframes<'p> {
    ictx: Input,
    stream: usize,
    packet: ffmpeg_next::Packet,
    progress: &'p dyn Fn(Step),
    /// The stamp of the keyframe last returned, read before its caller
    /// restamps its pts with the sample clock.
    current_ts: Option<i64>,
    /// The stamp of the last keyframe returned that had one.
    last_ts: Option<i64>,
    /// A seek has been issued and no keyframe accepted since.
    after_seek: bool,
    /// The demuxer has handed over a packet it was asked to discard.
    ignores_discard: bool,
    seeking: bool,
}

impl<'p> Keyframes<'p> {
    pub(super) fn new(mut ictx: Input, stream: usize, progress: &'p dyn Fn(Step)) -> Self {
        unsafe {
            let fmt_ctx = ictx.as_mut_ptr();
            for i in 0..(*fmt_ctx).nb_streams as usize {
                let stream_ptr = *(*fmt_ctx).streams.add(i);
                (*stream_ptr).discard = if i == stream {
                    ffmpeg_next::ffi::AVDiscard::AVDISCARD_NONKEY
                } else {
                    ffmpeg_next::ffi::AVDiscard::AVDISCARD_ALL
                };
            }
        }
        let seeking = !clock_may_restart(&ictx);
        Keyframes {
            ictx,
            stream,
            packet: ffmpeg_next::Packet::empty(),
            progress,
            current_ts: None,
            last_ts: None,
            after_seek: false,
            ignores_discard: false,
            seeking,
        }
    }

    /// The next keyframe, or `None` when the input is over.
    pub(super) fn next(&mut self) -> Result<Option<&mut ffmpeg_next::Packet>> {
        let mut demux_errors = 0u32;
        loop {
            if shutdown_requested() {
                return Err(anyhow!("Interrupted while fingerprinting"));
            }

            // `av_read_frame` moves a reference in over whatever the packet
            // holds without releasing it, so a reused packet has to be unreffed
            // or every keyframe's payload leaks -- a gigabyte over a few
            // thousand keyframes. At the top, so every `continue` passes here.
            unsafe {
                ffmpeg_next::ffi::av_packet_unref(self.packet.as_mut_ptr());
            }

            match self.packet.read(&mut self.ictx) {
                Ok(()) => demux_errors = 0,
                Err(ffmpeg_next::Error::Eof) => return Ok(None),
                Err(_) => {
                    demux_errors += 1;
                    if demux_errors >= MAX_CONSECUTIVE_DEMUX_ERRORS || input_is_spent(&self.ictx) {
                        return Ok(None);
                    }
                    continue;
                }
            }

            if self.packet.stream() != self.stream {
                continue;
            }

            // This stream's offset only: an attached picture is a stream of its
            // own, handed over FIRST and stored at the END of the file, and
            // reporting its offset credited a whole 5 GB file to the bar before
            // a frame was decoded. A container that does not track offsets
            // reports -1.
            let pos = self.packet.position();
            if pos >= 0 {
                (self.progress)(Step::Read(pos as u64));
            }

            if !self.packet.is_key() {
                self.ignores_discard = true;
                continue;
            }

            let ts = decode_stamp(&self.packet);
            if self.after_seek {
                if matches!((ts, self.last_ts), (Some(ts), Some(last)) if ts <= last) {
                    continue;
                }
                self.after_seek = false;
            }
            self.current_ts = ts;
            if ts.is_some() {
                self.last_ts = ts;
            }
            return Ok(Some(&mut self.packet));
        }
    }

    /// Stop navigating by timestamps: the clock has been seen to restart.
    pub(super) fn stop_seeking(&mut self) {
        self.seeking = false;
    }

    /// Move on from the keyframe last returned, at least `step` ticks past it.
    ///
    /// Reads on unless the demuxer has shown it reads the bytes between
    /// keyframes anyway, in which case it seeks there instead. The decoder is
    /// not flushed across a seek: every packet it is fed is a keyframe, so
    /// there is no state to invalidate.
    pub(super) fn seek_past(&mut self, step: i64) {
        if !self.ignores_discard || !self.seeking {
            return;
        }
        let Some(ts) = self.current_ts else {
            self.seeking = false;
            return;
        };
        let ok = unsafe {
            ffmpeg_next::ffi::av_seek_frame(self.ictx.as_mut_ptr(), self.stream as i32, ts.saturating_add(step), 0) >= 0
        };
        // Past the last keyframe a forward seek fails, which is
        // indistinguishable from a container that cannot seek at all. Either
        // way, finish by reading.
        if ok {
            self.after_seek = true;
        } else {
            self.seeking = false;
        }
    }
}

/// Whether the demuxer has read everything there is, whatever error it chose to
/// report instead of saying so.
///
/// `avio_feof` is the byte-level truth: the packet layer's `AVERROR_EOF` is a
/// convention demuxers are free to ignore, and several do. Null-checked because
/// `pb` is only guaranteed for a context that owns its I/O.
fn input_is_spent(ictx: &Input) -> bool {
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
pub(crate) fn clock_may_restart(ictx: &Input) -> bool {
    unsafe {
        let iformat = (*ictx.as_ptr()).iformat;
        !iformat.is_null() && ((*iformat).flags & ffmpeg_next::ffi::AVFMT_TS_DISCONT) != 0
    }
}

/// A decoder set up for keyframes only, and the transaction with it.
///
/// What it says about a keyframe falls in three kinds (`is_decoding_error`):
/// a picture it could not build costs that one sample and is counted in
/// `lost`; a full output queue is drained and the packet resent; anything
/// else is the decoder refusing the transaction, kept in `refused` and fatal
/// to the file, because a sample missing from the middle of a fingerprint that
/// still claims the whole runtime is a wrong answer, not a smaller one.
pub(super) struct KeyframeDecoder {
    decoder: ffmpeg_next::decoder::Video,
    decoded: Frame,
    pub(super) lost: u32,
    pub(super) refused: Option<ffmpeg_next::Error>,
}

impl KeyframeDecoder {
    /// `threads` is clamped to `1..=MAX_DECODE_THREADS`, so 0 is harmless.
    pub(super) fn open(parameters: ffmpeg_next::codec::Parameters, threads: usize) -> Result<Self> {
        let mut context = ffmpeg_next::codec::context::Context::from_parameters(parameters)
            .context("Failed to create codec context from parameters")?;
        let threads = threads.clamp(1, MAX_DECODE_THREADS);
        unsafe {
            let ctx = context.as_mut_ptr();
            (*ctx).thread_count = threads as i32;
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
            if threads > 1 {
                // Frame threading preferred; slice threading is the fallback for
                // codecs without AV_CODEC_CAP_FRAME_THREADS. FFmpeg picks.
                (*ctx).thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
            } else {
                (*ctx).flags |= ffmpeg_next::ffi::AV_CODEC_FLAG_LOW_DELAY as i32;
            }
            (*ctx).skip_frame = ffmpeg_next::ffi::AVDiscard::AVDISCARD_NONKEY;
        }
        let decoder = context.decoder().video().context("Failed to initialize video decoder")?;
        Ok(KeyframeDecoder { decoder, decoded: Frame::empty(), lost: 0, refused: None })
    }

    /// Decode one keyframe, handing every picture that is ready to `take`.
    pub(super) fn feed(&mut self, packet: &ffmpeg_next::Packet, take: &mut impl FnMut(&Frame)) {
        if let Err(e) = feed_keyframe(&mut self.decoder, packet, &mut self.decoded, take, &mut self.lost) {
            self.refused.get_or_insert(e);
        }
    }

    /// Tell the decoder the input is over and take what it was holding back.
    ///
    /// With frame threading it holds up to its thread count of frames, so on a
    /// short video this can be where most of them arrive. `Eof` from the send
    /// says it was already draining, which is what is being asked for.
    pub(super) fn finish(&mut self, take: &mut impl FnMut(&Frame)) {
        match self.decoder.send_eof() {
            Ok(()) | Err(ffmpeg_next::Error::Eof) => {}
            Err(e) => {
                self.refused.get_or_insert(e);
            }
        }
        if let Err(e) = take_ready_frames(&mut self.decoder, &mut self.decoded, take, &mut self.lost) {
            self.refused.get_or_insert(e);
        }
    }
}

/// Whether an answer from the decoder is about the PICTURE or about the
/// TRANSACTION, which are two different findings and must not be treated alike.
///
/// `avcodec_send_packet` and `avcodec_receive_frame` both document four answers
/// about the state of the decoder -- `EAGAIN`, `EOF`, `EINVAL`, `ENOMEM` -- and
/// then say of everything else, in as many words, "other errors: legitimate
/// decoding errors". Those two halves want opposite handling, and the reason is
/// what the call did with the packet before it answered. A transaction answer
/// is a packet NOT taken: the caller has to drain, or stop, or give up. A
/// decoding error is a packet already spent -- `avcodec_send_packet` hands it
/// to the bitstream filter and only THEN decodes eagerly, so by the time a
/// decode can fail the packet is gone -- and the only thing lost is the picture
/// it would have produced. Resending such a packet decodes it twice.
///
/// Reading the second half as the first is what this exists to stop, and it
/// cost whole files. A 10-bit HEVC remux sampled with `AVDISCARD_NONKEY` hands
/// the decoder keyframes with the frames between them removed, and hevcdec
/// eventually meets an IRAP whose POC is already in its DPB: "Duplicate POC in
/// a sequence", `AVERROR_INVALIDDATA`, out of `send_packet`. Eight of them in a
/// 1080p episode -- against 827 keyframes that decoded perfectly -- and the
/// whole file was refused, cached as refused, and never looked at again. The
/// vendored FFmpeg 8 hands that frame back with a log line instead of an error,
/// which is the only reason the shipped binary was not affected.
pub(super) fn is_decoding_error(e: &ffmpeg_next::Error) -> bool {
    match e {
        ffmpeg_next::Error::Eof => false,
        ffmpeg_next::Error::Other { errno } => {
            !matches!(*errno, libc::EAGAIN | libc::EINVAL | libc::ENOMEM)
        }
        _ => true,
    }
}

/// Hand every frame the decoder has ready to `process`, until it has none.
///
/// `EAGAIN` (needs input) and `Eof` (fully drained) both mean "nothing more
/// now", which is the end of a drain and not an error. A picture the decoder
/// could not build is counted in `lost` and the drain carries on: the frames
/// queued behind it are good, and stopping is what leaves the queue full for
/// the next send. `MAX_CONSECUTIVE_DECODE_ERRORS` in a row is a decoder
/// producing nothing, and anything else is returned.
pub(super) fn take_ready_frames(
    decoder: &mut ffmpeg_next::decoder::Video,
    decoded: &mut Frame,
    process: &mut impl FnMut(&Frame),
    lost: &mut u32,
) -> Result<(), ffmpeg_next::Error> {
    let mut consecutive = 0u32;
    loop {
        match decoder.receive_frame(decoded) {
            Ok(()) => {
                consecutive = 0;
                process(decoded);
            }
            Err(ffmpeg_next::Error::Eof) => return Ok(()),
            Err(ffmpeg_next::Error::Other { errno }) if errno == libc::EAGAIN => return Ok(()),
            Err(e) if is_decoding_error(&e) => {
                *lost += 1;
                consecutive += 1;
                log::debug!("decoder produced no picture for a keyframe ({e}); sample lost");
                if consecutive >= MAX_CONSECUTIVE_DECODE_ERRORS {
                    return Err(e);
                }
            }
            Err(e) => return Err(e),
        }
    }
}

/// Send one keyframe and take every frame that is then ready.
///
/// `EAGAIN` from the send is the decoder asking for its output to be read
/// first: the packet was NOT taken and is to be resent once the queue has been
/// drained, a resend the API promises "will not fail with EAGAIN". Reading it
/// as a failure skipped the drain that would have cleared the queue, so every
/// later keyframe met the same full queue and the rest of the file was lost.
pub(super) fn feed_keyframe(
    decoder: &mut ffmpeg_next::decoder::Video,
    packet: &ffmpeg_next::Packet,
    decoded: &mut Frame,
    process: &mut impl FnMut(&Frame),
    lost: &mut u32,
) -> Result<(), ffmpeg_next::Error> {
    match send_and_take(decoder, packet, decoded, process, lost) {
        Err(ffmpeg_next::Error::Other { errno }) if errno == libc::EAGAIN => {
            log::debug!("decoder queue full; draining and resending the keyframe");
            take_ready_frames(decoder, decoded, process, lost)?;
            send_and_take(decoder, packet, decoded, process, lost)
        }
        other => other,
    }
}

/// One send and the drain after it, as one transaction, so a resend is held to
/// the same rules as the first attempt: a picture that cannot be built costs
/// one sample whichever try it fails on.
fn send_and_take(
    decoder: &mut ffmpeg_next::decoder::Video,
    packet: &ffmpeg_next::Packet,
    decoded: &mut Frame,
    process: &mut impl FnMut(&Frame),
    lost: &mut u32,
) -> Result<(), ffmpeg_next::Error> {
    match decoder.send_packet(packet) {
        Ok(()) => take_ready_frames(decoder, decoded, process, lost),
        Err(e) if is_decoding_error(&e) => {
            *lost += 1;
            log::debug!("decoder could not build a keyframe's picture ({e}); sample lost");
            take_ready_frames(decoder, decoded, process, lost)
        }
        Err(e) => Err(e),
    }
}

/// Every decoded picture, scaled to the `SIDE` x `SIDE` working frame, turned
/// as it is shown, and kept with its time unless it repeats the one before.
///
/// All kept frames are packed back to back in ONE allocation, which glibc
/// serves by mmap and returns to the OS when the video is done; do not make it
/// a `Vec<Vec<u8>>` (see `main`'s `mallopt`).
pub(super) struct FrameCapture {
    turns: u8,
    /// Seconds per tick of the stream, 0 when unknown.
    tb: f64,
    scaler: Option<ffmpeg_next::software::scaling::context::Context>,
    scaled: Frame,
    current: Vec<u8>,
    turn_scratch: Vec<u8>,
    frames: Vec<u8>,
    times: Vec<Option<u32>>,
    /// The shown geometry of the FIRST frame.
    dims: Option<(u32, u32)>,
    /// Frames decoded and scaled, repeats included.
    count: usize,
    unscalable: Option<ffmpeg_next::Error>,
}

/// What a capture kept.
pub(super) struct Captured {
    /// `times.len()` frames of `FRAME_STRIDE` bytes.
    pub(super) frames: Vec<u8>,
    /// Each frame's time in ms, `None` where the frame carried none.
    pub(super) times: Vec<Option<u32>>,
    pub(super) dims: (u32, u32),
    pub(super) count: usize,
}

impl FrameCapture {
    pub(super) fn new(turns: u8, tb: f64) -> Self {
        FrameCapture {
            turns,
            tb,
            scaler: None,
            scaled: Frame::empty(),
            current: vec![0u8; FRAME_STRIDE],
            turn_scratch: Vec::with_capacity(if turns == 0 { 0 } else { FRAME_STRIDE }),
            frames: Vec::with_capacity(FRAME_STRIDE * 64),
            times: Vec::new(),
            dims: None,
            count: 0,
            unscalable: None,
        }
    }

    /// Take one decoded picture. A picture that cannot be scaled is remembered
    /// and fails the file in `finish`.
    pub(super) fn take(&mut self, frame: &Frame) {
        if let Err(e) = self.scale(frame) {
            self.unscalable.get_or_insert(e);
        }
    }

    fn scale(&mut self, dec: &Frame) -> Result<(), ffmpeg_next::Error> {
        // Rebuilt whenever the decoder's output changes shape: a stream may
        // change resolution or pixel format mid-file (a feed switch, two
        // encodes joined), and a context refuses any frame not of the shape it
        // was built for. Built once, it silently dropped the rest of the file.
        // Asked of the context's own record, so there is no second copy to drift.
        let rebuild = self.scaler.as_ref().is_none_or(|built| {
            let input = built.input();
            input.format != dec.format() || input.width != dec.width() || input.height != dec.height()
        });
        if rebuild {
            self.scaler = Some(ffmpeg_next::software::scaling::context::Context::get(
                dec.format(),
                dec.width(),
                dec.height(),
                ffmpeg_next::format::Pixel::GRAY8,
                SIDE as u32,
                SIDE as u32,
                ffmpeg_next::software::scaling::flag::Flags::FAST_BILINEAR,
            )?);
            // The first frame's geometry is what the report and the ranking
            // read: a file that rescales has no one true answer, and this is
            // the one the header would have given.
            let shown = if self.turns % 2 == 1 { (dec.height(), dec.width()) } else { (dec.width(), dec.height()) };
            self.dims.get_or_insert(shown);
        }
        self.scaler.as_mut().expect("built above").run(dec, &mut self.scaled)?;

        let data = self.scaled.data(0);
        let stride = self.scaled.stride(0);
        for y in 0..SIDE {
            self.current[y * SIDE..(y + 1) * SIDE].copy_from_slice(&data[y * stride..y * stride + SIDE]);
        }
        turn_square(&mut self.current, self.turns, &mut self.turn_scratch);

        let repeat = !self.times.is_empty() && self.frames[self.frames.len() - FRAME_STRIDE..] == self.current[..];
        if !repeat {
            self.frames.extend_from_slice(&self.current);
            // The frame's own `pts`, which is the sample clock the reader's
            // caller stamped on its packet (`fingerprint_video`) making the
            // round trip through the decoder -- the only thing that pairs a
            // frame with its packet under frame threading. NOT
            // `best_effort_timestamp` (`Frame::timestamp`): once presentation
            // times go backwards, as HEVC open-GOP keyframes do when the
            // decoder cannot see the picture order count wrap, that answers
            // with the dts of the packet being SENT, a later keyframe's, and
            // filed pictures under other keyframes' times.
            self.times.push(match dec.pts() {
                Some(pts) if self.tb > 0.0 && pts >= 0 => Some(ticks_to_ms(pts, self.tb)),
                _ => None,
            });
        }
        self.count += 1;
        Ok(())
    }

    /// A picture that could not be scaled, if there was one. Fatal: a
    /// fingerprint quietly missing part of a file reads as the same length as
    /// the file, wins the ranking on encoder quality, and marks it DELETE.
    pub(super) fn unscalable(&mut self) -> Option<ffmpeg_next::Error> {
        self.unscalable.take()
    }

    /// The frames kept, or `None` if none were.
    pub(super) fn finish(self) -> Option<Captured> {
        let dims = self.dims?;
        (self.count > 0).then_some(Captured { frames: self.frames, times: self.times, dims, count: self.count })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ticks(clock: &mut SampleClock, stamps: &[i64]) -> Vec<i64> {
        stamps.iter().map(|&ts| clock.tick(ts)).collect()
    }

    #[test]
    fn test_a_forward_clock_is_measured_from_its_first_keyframe() {
        let mut clock = SampleClock::default();
        assert_eq!(ticks(&mut clock, &[-1001, 2002, 2002, 5005]), [0, 3003, 3003, 6006]);
        assert!(!clock.restarted());
    }

    #[test]
    fn test_a_restart_is_charged_the_last_forward_gap_and_runs_on() {
        let mut clock = SampleClock::default();
        assert_eq!(ticks(&mut clock, &[100, 200, 300, 100, 200]), [0, 100, 200, 300, 400]);
        assert!(clock.restarted());

        // Restarting before any forward gap is seen still moves forwards.
        let mut clock = SampleClock::default();
        assert_eq!(ticks(&mut clock, &[50, 50, 10]), [0, 0, 1]);
    }

    #[test]
    fn test_thinning_keeps_keyframes_an_interval_apart_floored_by_runtime() {
        // 10 s interval over a 600 s file, floor 600/12 = 50 s: the interval stands.
        let mut thin = Thinning::new(10.0, 12.0, 600.0);
        let kept: Vec<f64> = [0.0, 4.0, 9.9, 10.0, 15.0, 21.0]
            .into_iter()
            .filter(|&t| thin.keep(Some(t)))
            .collect();
        assert_eq!(kept, [0.0, 10.0, 21.0]);
        assert_eq!(thin.wait_after(Some(25.0)), Some(6.0));
        assert_eq!(thin.wait_after(Some(31.0)), None);
        assert!(thin.keep(None), "a keyframe with no time is kept");

        // A 30 s file with a floor of 12 samples: 2.5 s, finer than 10.
        let mut thin = Thinning::new(10.0, 12.0, 30.0);
        assert!(thin.keep(Some(0.0)) && !thin.keep(Some(2.0)) && thin.keep(Some(2.5)));

        // No floor (0, negative, NaN) leaves the interval alone.
        for floor in [0.0, -3.0, f64::NAN] {
            let mut thin = Thinning::new(10.0, floor, 30.0);
            assert!(thin.keep(Some(0.0)) && !thin.keep(Some(9.0)) && thin.keep(Some(10.0)));
        }

        // No interval, or no runtime to measure it against, keeps everything.
        for (interval, duration) in [(0.0, 30.0), (10.0, 0.0)] {
            let mut thin = Thinning::new(interval, 12.0, duration);
            assert!((0..5).all(|t| thin.keep(Some(t as f64 * 0.1))));
            assert_eq!(thin.wait_after(Some(0.0)), None);
        }
    }
}
