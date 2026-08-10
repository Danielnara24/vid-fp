//! Formatting helpers and the rules for ranking two copies of the same video.
//!
//! The ranking half of this file rests on one idea: some metrics mean the same
//! thing no matter how a file was encoded, and some do not. Length and
//! resolution are the first kind -- ninety minutes is ninety minutes, 1080p is
//! 1080p. Bits are the second kind. A bitrate figure only means something next
//! to the codec that produced it, so comparing an AV1 copy's bits against an
//! H.264 copy's bits and keeping the bigger number is a rule that deletes the
//! better-encoded file for the crime of being efficient.
//!
//! So the bit-based metrics -- `Quality` and `Size` -- are compared only
//! between files that share a codec, and tie across codecs. Ties fall through
//! to the next metric, and when nothing is left to fall through to, `export.rs`
//! flags the standoff for a human instead of picking one by name.

use crate::fingerprint::VideoFingerprint;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

/// Set once, never cleared. Every long-running loop polls it and unwinds.
///
/// Relaxed ordering is deliberate: this is a hint, not a data handoff. The
/// fingerprints themselves travel through the batch mutex and rayon's collect,
/// which carry their own synchronization.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Called from the signal handler. Doing only this — no locks, no I/O — is what
/// makes the handler impossible to deadlock.
pub fn request_shutdown() {
    SHUTDOWN.store(true, std::sync::atomic::Ordering::SeqCst);
}

#[inline(always)]
pub fn shutdown_requested() -> bool {
    SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(clap::ValueEnum, Clone, Debug, Copy, PartialEq, Eq)]
pub enum Priority {
    Length,
    Resolution,
    /// Bits spent on an average frame: bitrate divided by frame rate. Replaces
    /// the old raw-bitrate metric, which double-counted frame rate (a 60 fps
    /// copy needs roughly twice the bitrate to look the same as a 30 fps one)
    /// and so preferred whichever copy simply had more frames.
    Quality,
    Size,
}

/// The two decisions a group's labels depend on, travelling together because
/// neither is meaningful without the other: which metric ranks the copies, and
/// how much evidence a DELETE has to rest on.
#[derive(Clone, Copy, Debug)]
pub struct Policy {
    pub priority: Priority,
    /// Whether a chain of matches counts as evidence on its own.
    ///
    /// Groups are connected components, so a member can be ranked below the
    /// copy being kept without the two ever having been compared -- it reached
    /// the group through some third file. By default such a file is flagged
    /// REVIEW rather than DELETE: the ranking that condemned it is sound, but
    /// nothing ever measured it against the file that is staying on disk.
    ///
    /// Setting this marks it DELETE instead, which is to say: treat the match
    /// relation as transitive. It is not (three episodes can share an opening
    /// without any two being copies), which is why this is off by default.
    pub trust_chains: bool,
}

/// Default precedence when ranking two copies of the same video:
/// length > resolution > quality > size.
///
/// `--priority X` does not reshuffle the rest of the list, it only moves X to
/// the front: `--priority quality` ranks quality > length > resolution > size.
const DEFAULT_ORDER: [Priority; 4] = [
    Priority::Length,
    Priority::Resolution,
    Priority::Quality,
    Priority::Size,
];

/// Tolerance bands. Files inside the same band are treated as equal on that
/// metric, so the decision falls through to the next one instead of being
/// settled by noise. See the README for the reasoning behind each width.
pub const DURATION_TOLERANCE_SECS: f64 = 1.0;
pub const RESOLUTION_TOLERANCE: f64 = 0.05;
pub const QUALITY_TOLERANCE: f64 = 0.05;
pub const SIZE_TOLERANCE: f64 = 0.05;

/// Whether a metric's value is only meaningful alongside the codec that
/// produced it.
///
/// Quality is bits per frame and size is bits per file; both are the encoder's
/// output, and a good encoder's job is to produce fewer of them. Neither can be
/// read across codecs. Length and resolution are properties of the footage
/// rather than of the encode, so they compare freely.
fn is_codec_relative(metric: Priority) -> bool {
    matches!(metric, Priority::Quality | Priority::Size)
}

pub fn format_size(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1_073_741_824.0 {
        format!("{:.1}GB", b / 1_073_741_824.0)
    } else if b >= 1_048_576.0 {
        format!("{:.1}MB", b / 1_048_576.0)
    } else if b >= 1024.0 {
        format!("{:.1}KB", b / 1024.0)
    } else {
        format!("{}B", bytes)
    }
}

pub fn format_duration(seconds: f64) -> String {
    let total_secs = seconds.round() as u64;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, secs)
}

/// Bitrates are conventionally decimal (1 Mbps = 1_000_000 bits/s), unlike
/// file sizes, so this deliberately does not share format_size's binary steps.
pub fn format_bitrate(bits_per_sec: u64) -> String {
    let b = bits_per_sec as f64;
    if b >= 1_000_000.0 {
        format!("{:.1}Mbps", b / 1_000_000.0)
    } else if b >= 1_000.0 {
        format!("{:.0}kbps", b / 1_000.0)
    } else {
        format!("{}bps", bits_per_sec)
    }
}

/// Bits spent on an average frame, in the same decimal steps as a bitrate.
///
/// The `/f` suffix is doing real work in a table that also carries a bitrate:
/// the two figures differ by a factor of the frame rate and would otherwise be
/// mistaken for each other. `0` means the frame rate was never reported, which
/// prints as a dash for the same reason an unmeasured overlap does -- unknown
/// is not the same as none.
pub fn format_quality(bits_per_frame: u64) -> String {
    if bits_per_frame == 0 {
        return "-".to_string();
    }

    let b = bits_per_frame as f64;
    if b >= 1_000_000.0 {
        format!("{:.1}Mb/f", b / 1_000_000.0)
    } else if b >= 1_000.0 {
        format!("{:.1}kb/f", b / 1_000.0)
    } else {
        format!("{}b/f", bits_per_frame)
    }
}

/// Frames per second, or a dash when the container never said.
///
/// Whole rates print whole (`30fps`, not `30.00fps`) so the common case stays
/// short, while 29.97 and friends keep the two decimals that distinguish them.
///
/// The guard tests finiteness explicitly rather than leaning on `!(rate > 0.0)`
/// to catch NaN as a side effect of the comparison being false either way. That
/// form covered NaN but let infinity straight through to the `as u64` cast
/// below, which saturates -- so an infinite rate printed as a twenty-digit fps.
/// `fingerprint::frame_rate_of` flattens anything non-finite to 0.0 before it
/// gets here, but this is the last stop before a number is printed and neither
/// value has a rendering that beats "the container never said".
pub fn format_frame_rate(frame_rate: f64) -> String {
    if !frame_rate.is_finite() || frame_rate <= 0.0 {
        return "-".to_string();
    }

    if (frame_rate - frame_rate.round()).abs() < 0.01 {
        format!("{}fps", frame_rate.round() as u64)
    } else {
        format!("{:.2}fps", frame_rate)
    }
}

/// The codec's name, or a dash if FFmpeg could not name it.
pub fn format_codec(codec: &str) -> String {
    if codec.is_empty() {
        "-".to_string()
    } else {
        codec.to_string()
    }
}

/// How much footage two files have in common, in a form a person can act on.
///
/// Deliberately NOT `format_duration`'s `HH:MM:SS`. Overlaps are frequently
/// under a second -- one shared keyframe between two short clips is a real and
/// common result -- and `00:00:00` is indistinguishable from a bug, while
/// `0.6s` says exactly what happened. The differing shape also keeps this
/// column from being mistaken for the file's own length sitting next to it.
///
/// `None` prints as a dash: the overlap was never measured, which is a
/// different statement from "they share nothing".
pub fn format_shared(seconds: Option<f64>) -> String {
    let Some(s) = seconds else {
        return "-".to_string();
    };

    if s <= 0.0 {
        return "0s".to_string();
    }
    if s < 0.1 {
        // A genuine but tiny overlap: one keyframe of a very long video. "0.0s"
        // would read as nothing at all, which is the wrong conclusion.
        return "<0.1s".to_string();
    }
    if s < 10.0 {
        return format!("{:.1}s", s);
    }

    // From here on the figure is whole seconds, and the unit is chosen from the
    // ROUNDED value rather than the raw one. Picking the unit first and
    // rounding inside it is how 119.7s renders as "1m60s" and 3599.7s as
    // "60m00s" -- both arithmetically defensible, both obviously wrong on the
    // page, and both reachable only from values a hair under a boundary.
    let total = s.round() as u64;

    if total < 60 {
        format!("{}s", total)
    } else if total < 3600 {
        format!("{}m{:02}s", total / 60, total % 60)
    } else {
        let total_mins = (total as f64 / 60.0).round() as u64;
        format!("{}h{:02}m", total_mins / 60, total_mins % 60)
    }
}

pub fn resolution(fp: &VideoFingerprint) -> u64 {
    fp.width as u64 * fp.height as u64
}

/// The best bit-based values seen for ONE codec within a duplicate group.
#[derive(Clone, Copy, Debug, Default)]
struct CodecMaxima {
    quality: u64,
    file_size: u64,
}

/// The best value seen for each metric within one duplicate group. Tolerance
/// is always measured against these, never between two files directly:
/// "within 5% of each other" is not a transitive relation, and feeding a
/// non-transitive comparator to `max_by` yields an arbitrary winner.
///
/// Codec-relative metrics get one maximum PER CODEC rather than one for the
/// group. That is what makes "do not compare bits across codecs" expressible as
/// a per-file property instead of a per-pair one: the best H.264 copy and the
/// best AV1 copy are each top-tier against their own kind, so they tie at the
/// top and the decision falls through -- while a genuinely worse H.264 copy
/// still loses to the better H.264 copy it is actually comparable with.
#[derive(Clone, Debug)]
pub struct GroupMaxima {
    duration: f64,
    resolution: u64,
    per_codec: HashMap<String, CodecMaxima>,
    mixed_codecs: bool,
}

impl GroupMaxima {
    pub fn of(group: &[usize], fps: &[VideoFingerprint]) -> Self {
        let mut duration = 0.0f64;
        let mut resolution_max = 0u64;
        let mut per_codec: HashMap<String, CodecMaxima> = HashMap::new();

        for &idx in group {
            let fp = &fps[idx];
            duration = duration.max(fp.duration);
            resolution_max = resolution_max.max(resolution(fp));

            let entry = per_codec.entry(fp.codec.clone()).or_default();
            entry.quality = entry.quality.max(fp.quality());
            entry.file_size = entry.file_size.max(fp.file_size);
        }

        let mixed_codecs = per_codec.len() > 1;

        GroupMaxima {
            duration,
            resolution: resolution_max,
            per_codec,
            mixed_codecs,
        }
    }

    fn codec_maxima(&self, fp: &VideoFingerprint) -> CodecMaxima {
        self.per_codec.get(&fp.codec).copied().unwrap_or_default()
    }

    /// 1 when `fp` is within tolerance of the best value `metric` can be
    /// measured against for it, 0 otherwise. For length and resolution that is
    /// the group's best; for quality and size it is the best among files
    /// sharing `fp`'s codec.
    pub fn tier(&self, fp: &VideoFingerprint, metric: Priority) -> u8 {
        match metric {
            Priority::Length => u8::from(fp.duration >= self.duration - DURATION_TOLERANCE_SECS),
            Priority::Resolution => within(resolution(fp), self.resolution, RESOLUTION_TOLERANCE),
            Priority::Quality => {
                // An unreported frame rate is missing information, not evidence
                // of a worse copy, so it costs the file nothing here and the
                // decision moves on to a metric that is actually known.
                if fp.frame_rate <= 0.0 {
                    return 1;
                }
                within(fp.quality(), self.codec_maxima(fp).quality, QUALITY_TOLERANCE)
            }
            Priority::Size => within(fp.file_size, self.codec_maxima(fp).file_size, SIZE_TOLERANCE),
        }
    }

    /// Raw comparable value for a metric, or 0 where the metric cannot legally
    /// separate this group's members.
    ///
    /// Duration is milliseconds so everything is an integer and comparison is
    /// total (floats are not Ord). Bit-based metrics collapse to 0 for every
    /// file the moment the group spans codecs: tiers have already ranked each
    /// file against its own kind, and going on to compare the raw numbers would
    /// be doing across codecs precisely what the tiers were built to prevent.
    fn value(&self, fp: &VideoFingerprint, metric: Priority) -> u64 {
        if self.mixed_codecs && is_codec_relative(metric) {
            return 0;
        }

        match metric {
            Priority::Length => (fp.duration * 1000.0) as u64,
            Priority::Resolution => resolution(fp),
            Priority::Quality => fp.quality(),
            Priority::Size => fp.file_size,
        }
    }
}

/// Relative tolerance check. A zero maximum means the metric is unknown for the
/// whole comparison set (e.g. quality when no frame rate could be read); everyone
/// ties so the ranking falls through to a metric that is actually known.
fn within(value: u64, max: u64, tolerance: f64) -> u8 {
    if max == 0 {
        return 1;
    }
    u8::from(value as f64 >= max as f64 * (1.0 - tolerance))
}

/// The chosen metric first, then the remaining three in default order.
fn ordered_metrics(priority: Priority) -> [Priority; 4] {
    let mut out = [priority; 4];
    let mut i = 1;
    for m in DEFAULT_ORDER {
        if m != priority {
            out[i] = m;
            i += 1;
        }
    }
    out
}

/// Deterministically pick the "best" file in a group.
///
/// Two passes over the same metric order. The first compares tolerance tiers,
/// so anything inside a band is a draw and defers to the next metric. The
/// second compares raw values, separating files that tied on every band. Path
/// order settles the rest, so the result never depends on input ordering.
///
/// In a mixed-codec group both passes can legitimately run out of ways to
/// separate two files, and the answer will then be the alphabetically first
/// path -- a tiebreak that exists for reproducibility, not because it means
/// anything. `export.rs` watches for exactly that situation and flags it REVIEW
/// rather than deleting a file on the strength of its name.
pub fn find_best(
    group: &[usize],
    fps: &[VideoFingerprint],
    priority: Priority,
    maxima: &GroupMaxima,
) -> usize {
    let order = ordered_metrics(priority);

    *group
        .iter()
        .max_by(|&&a, &&b| {
            let fp_a = &fps[a];
            let fp_b = &fps[b];

            let mut ord = Ordering::Equal;

            for m in order {
                ord = ord.then(maxima.tier(fp_a, m).cmp(&maxima.tier(fp_b, m)));
            }
            for m in order {
                ord = ord.then(maxima.value(fp_a, m).cmp(&maxima.value(fp_b, m)));
            }

            // Reversed so the alphabetically FIRST path wins, since max_by
            // keeps the greater element.
            ord.then_with(|| fp_b.path.cmp(&fp_a.path))
        })
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full mock. Most tests only vary one thing, so `fp` below fixes the
    /// codec and frame rate and this exists for the ones that don't.
    fn fp_full(
        path: &str,
        dur: f64,
        w: u32,
        h: u32,
        size: u64,
        codec: &str,
        frame_rate: f64,
    ) -> VideoFingerprint {
        VideoFingerprint {
            path: path.to_string(),
            valid_hashes: vec![],
            valid_t_start: vec![],
            valid_t_end: vec![],
            total_ms: 100_000,
            width: w,
            height: h,
            duration: dur,
            file_size: size,
            codec: codec.to_string(),
            frame_rate,
        }
    }

    /// h264 at 30 fps: one codec, one frame rate, so bits compare freely and
    /// quality behaves exactly like a bitrate.
    fn fp(path: &str, dur: f64, w: u32, h: u32, size: u64) -> VideoFingerprint {
        fp_full(path, dur, w, h, size, "h264", 30.0)
    }

    fn best(fps: &[VideoFingerprint], priority: Priority) -> usize {
        let group: Vec<usize> = (0..fps.len()).collect();
        let maxima = GroupMaxima::of(&group, fps);
        find_best(&group, fps, priority, &maxima)
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(500), "500B");
        assert_eq!(format_size(1023), "1023B");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1_024), "1.0KB");
        assert_eq!(format_size(1_536), "1.5KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1_048_576), "1.0MB");
        assert_eq!(format_size(5_242_880), "5.0MB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1_073_741_824), "1.0GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.0), "00:00:00");
        assert_eq!(format_duration(59.4), "00:00:59");
        assert_eq!(format_duration(59.6), "00:01:00");
        assert_eq!(format_duration(3661.0), "01:01:01");
        assert_eq!(format_duration(36000.0), "10:00:00");
    }

    #[test]
    fn test_format_bitrate() {
        assert_eq!(format_bitrate(0), "0bps");
        assert_eq!(format_bitrate(950), "950bps");
        assert_eq!(format_bitrate(128_000), "128kbps");
        assert_eq!(format_bitrate(4_500_000), "4.5Mbps");
    }

    #[test]
    fn test_format_quality() {
        assert_eq!(format_quality(0), "-", "no frame rate means no figure");
        assert_eq!(format_quality(750), "750b/f");
        assert_eq!(format_quality(4_660), "4.7kb/f");
        assert_eq!(format_quality(1_500_000), "1.5Mb/f");
    }

    #[test]
    fn test_format_frame_rate() {
        assert_eq!(format_frame_rate(0.0), "-");
        assert_eq!(format_frame_rate(30.0), "30fps");
        assert_eq!(format_frame_rate(29.97), "29.97fps");
        assert_eq!(format_frame_rate(23.976), "23.98fps");

        // A rate that isn't a number isn't a rate. NaN was always caught;
        // infinity was not, and reached the saturating `as u64` cast, which
        // rendered it as `18446744073709551615fps`.
        assert_eq!(format_frame_rate(f64::NAN), "-");
        assert_eq!(format_frame_rate(f64::INFINITY), "-");
        assert_eq!(format_frame_rate(-30.0), "-");
    }

    #[test]
    fn test_format_shared_keeps_sub_second_overlaps_legible() {
        // The case that motivated the whole column. HH:MM:SS would render every
        // one of these as 00:00:00 and look broken.
        assert_eq!(format_shared(Some(0.64)), "0.6s");
        assert_eq!(format_shared(Some(0.05)), "<0.1s");
        assert_eq!(format_shared(Some(3.2)), "3.2s");
    }

    #[test]
    fn test_format_shared_scales_with_magnitude() {
        assert_eq!(format_shared(Some(12.0)), "12s");
        assert_eq!(format_shared(Some(59.0)), "59s");
        assert_eq!(format_shared(Some(118.0)), "1m58s");
        assert_eq!(format_shared(Some(3600.0)), "1h00m");
        assert_eq!(format_shared(Some(7_500.0)), "2h05m");
    }

    #[test]
    fn test_format_shared_never_renders_an_impossible_clock() {
        // Each of these sits a hair below a unit boundary, and each rounds up
        // across it. The unit has to be chosen after that rounding, or they
        // render as "1m60s", "60m00s" and "60s" respectively.
        assert_eq!(format_shared(Some(119.7)), "2m00s");
        assert_eq!(format_shared(Some(3_599.7)), "1h00m");
        assert_eq!(format_shared(Some(59.6)), "1m00s");
    }

    #[test]
    fn test_format_shared_distinguishes_unknown_from_none() {
        assert_eq!(format_shared(None), "-", "never measured");
        assert_eq!(format_shared(Some(0.0)), "0s", "measured, and it was nothing");
    }

    #[test]
    fn test_bitrate_derived_from_size_and_duration() {
        // 7,864,320 bytes over 60s = 1,048,576 bits/s.
        let f = fp("a.mp4", 60.0, 1920, 1080, 7_864_320);
        assert_eq!(f.bitrate(), 1_048_576);

        // Unknown duration must not divide by zero.
        assert_eq!(fp("b.mp4", 0.0, 1920, 1080, 1024).bitrate(), 0);
    }

    #[test]
    fn test_quality_outranks_size_by_default() {
        // Same length, same resolution, same codec: the decision reaches
        // quality, and the denser file wins even though it is the smaller one.
        let fps = vec![
            fp("low.mp4", 120.0, 1920, 1080, 30_000_000),
            fp("high.mp4", 60.0 * 2.0, 1920, 1080, 60_000_000),
        ];
        assert_eq!(best(&fps, Priority::Length), 1);
    }

    #[test]
    fn test_frame_rate_is_what_separates_quality_from_bitrate() {
        // Identical bitrate, identical everything else, different frame rates.
        // Under the old raw-bitrate rule these tied and the winner came down to
        // filename; spending the same bits on 24 frames instead of 60 puts two
        // and a half times as much into each one, and quality says so.
        let fps = vec![
            fp_full("sixty.mp4", 60.0, 1920, 1080, 9_000_000, "h264", 60.0),
            fp_full("twentyfour.mp4", 60.0, 1920, 1080, 9_000_000, "h264", 24.0),
        ];
        assert_eq!(best(&fps, Priority::Length), 1, "fewer frames, more bits in each");
    }

    #[test]
    fn test_quality_is_not_compared_across_codecs() {
        // The bug this whole change exists to fix. The h264 copy carries three
        // times the bits; that is what a less efficient codec needs to look the
        // same, not evidence that it looks better. Neither file may win on
        // quality, so the tiebreak falls all the way to path order -- and
        // export.rs never deletes on the strength of that alone.
        let fps = vec![
            fp_full("a_av1.mp4", 60.0, 1920, 1080, 3_000_000, "av1", 30.0),
            fp_full("b_h264.mp4", 60.0, 1920, 1080, 9_000_000, "h264", 30.0),
        ];

        assert_eq!(
            best(&fps, Priority::Quality),
            0,
            "3x the bits from a hungrier codec must not win the group"
        );
    }

    #[test]
    fn test_quality_still_decides_within_one_codec() {
        // The other half of the rule: two files that ARE comparable are still
        // compared, and the thin one loses.
        let fps = vec![
            fp_full("a_low.mp4", 60.0, 1920, 1080, 3_000_000, "h264", 30.0),
            fp_full("b_high.mp4", 60.0, 1920, 1080, 9_000_000, "h264", 30.0),
        ];
        assert_eq!(best(&fps, Priority::Quality), 1);
    }

    #[test]
    fn test_size_is_not_compared_across_codecs_either() {
        // Size is bits too -- it is quality times frame rate times length -- so
        // ranking on it across codecs reintroduces the same bias by the back
        // door, one metric later in the order.
        let fps = vec![
            fp_full("a_av1.mp4", 60.0, 1920, 1080, 4_000_000, "av1", 30.0),
            fp_full("b_h264.mp4", 60.0, 1920, 1080, 8_000_000, "h264", 30.0),
        ];

        assert_eq!(best(&fps, Priority::Size), 0, "twice the bytes is not twice the video");
    }

    #[test]
    fn test_length_and_resolution_still_decide_across_codecs() {
        // Codec neutrality applies to bits, not to footage. A longer file is
        // longer whatever encoded it, and that must still settle the group.
        let fps = vec![
            fp_full("short_av1.mp4", 30.0, 1920, 1080, 3_000_000, "av1", 30.0),
            fp_full("long_h264.mp4", 60.0, 1920, 1080, 9_000_000, "h264", 30.0),
        ];
        assert_eq!(best(&fps, Priority::Length), 1);

        let fps = vec![
            fp_full("sd_h264.mp4", 60.0, 1280, 720, 9_000_000, "h264", 30.0),
            fp_full("hd_av1.mp4", 60.0, 1920, 1080, 3_000_000, "av1", 30.0),
        ];
        assert_eq!(best(&fps, Priority::Resolution), 1);
    }

    #[test]
    fn test_a_worse_copy_of_the_same_codec_still_loses_in_a_mixed_group() {
        // Codec-relative maxima are per codec, not per group, so the presence
        // of an av1 copy does not blind the two h264 copies to each other.
        let fps = vec![
            fp_full("a_h264_thin.mp4", 60.0, 1920, 1080, 2_000_000, "h264", 30.0),
            fp_full("b_h264_fat.mp4", 60.0, 1920, 1080, 9_000_000, "h264", 30.0),
            fp_full("c_av1.mp4", 60.0, 1920, 1080, 3_000_000, "av1", 30.0),
        ];

        let group: Vec<usize> = (0..3).collect();
        let maxima = GroupMaxima::of(&group, &fps);

        assert!(maxima.mixed_codecs, "the group spans two codecs");
        assert_eq!(maxima.tier(&fps[0], Priority::Quality), 0, "thin h264 loses to fat h264");
        assert_eq!(maxima.tier(&fps[1], Priority::Quality), 1);
        assert_eq!(maxima.tier(&fps[2], Priority::Quality), 1, "best of its own kind");
    }

    #[test]
    fn test_unknown_frame_rate_is_not_treated_as_a_worse_copy() {
        // Quality is unknowable for the second file. If that counted against it
        // the first file would win on quality; instead the tie falls through to
        // size, where the second file is plainly the bigger copy.
        let fps = vec![
            fp_full("a_known.mp4", 60.0, 1920, 1080, 3_000_000, "h264", 30.0),
            fp_full("b_unknown_fps.mp4", 60.0, 1920, 1080, 9_000_000, "h264", 0.0),
        ];

        assert_eq!(
            best(&fps, Priority::Length),
            1,
            "a missing frame rate is missing information, not a demerit"
        );
    }

    #[test]
    fn test_resolution_tolerance_absorbs_crop() {
        // 1920x1040 is 96.3% of 1920x1080's pixel count -- a crop, not a
        // downscale -- so both sit in the top tier and quality decides.
        let fps = vec![
            fp("uncropped.mp4", 60.0, 1920, 1080, 6_000_000),
            fp("cropped.mp4", 60.0, 1920, 1040, 9_000_000),
        ];
        assert_eq!(best(&fps, Priority::Resolution), 1);
    }

    #[test]
    fn test_resolution_tiers_are_not_absorbed() {
        // 720p vs 1080p is far outside the band, so resolution settles it
        // regardless of the 720p file's higher quality.
        let fps = vec![
            fp("720p.mp4", 60.0, 1280, 720, 20_000_000),
            fp("1080p.mp4", 60.0, 1920, 1080, 9_000_000),
        ];
        assert_eq!(best(&fps, Priority::Resolution), 1);
    }

    #[test]
    fn test_priority_moves_its_metric_to_the_front() {
        let fps = vec![
            fp("long_thin.mp4", 100.0, 1920, 1080, 12_500_000), // 1 Mbps
            fp("short_fat.mp4", 60.0, 1920, 1080, 15_000_000),  // 2 Mbps
        ];
        assert_eq!(best(&fps, Priority::Length), 0, "length first by default");
        assert_eq!(best(&fps, Priority::Quality), 1, "quality first when asked");
    }

    #[test]
    fn test_identical_files_break_ties_on_path() {
        let fps = vec![
            fp("z.mp4", 60.0, 1920, 1080, 7_864_320),
            fp("a.mp4", 60.0, 1920, 1080, 7_864_320),
        ];
        assert_eq!(best(&fps, Priority::Length), 1, "alphabetically first wins");
    }
}