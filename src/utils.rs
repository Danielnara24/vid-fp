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

/// Set once, at start-up, when `--quiet` was asked for.
///
/// `log_enabled!(Info)` used to be a good enough stand-in for "the terminal is
/// listening", because one filter stood in front of the only destination there
/// was. It stopped being one when `--log-file` started raising that filter on
/// its own behalf: under `-q --log-file` an Info line is now enabled, and
/// rightly so -- the file wants it -- but the terminal is still meant to be
/// silent. Anything drawing on stderr at Info level has to ask this as well.
///
/// A flag rather than a parameter because the one caller is inside `compare`,
/// which would otherwise thread a bool through `find_all_matches` and every one
/// of its twenty-odd test call sites to reach a progress bar. Relaxed for the
/// same reason `SHUTDOWN` is: it is a hint about pixels, not a data handoff.
static CONSOLE_QUIET: AtomicBool = AtomicBool::new(false);

pub fn silence_console() {
    CONSOLE_QUIET.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Whether an `info!` line would actually reach the terminal.
///
/// False under `--quiet` whatever the log filter says, and false when no logger
/// is installed at all -- which is what keeps progress bars out of unit tests
/// without any of them having to be told they are one.
pub fn console_is_verbose() -> bool {
    !CONSOLE_QUIET.load(std::sync::atomic::Ordering::Relaxed)
        && log::log_enabled!(log::Level::Info)
}

#[derive(clap::ValueEnum, Clone, Debug, Copy, PartialEq, Eq)]
pub enum Priority {
    Length,
    Resolution,
    // Not raw bitrate, which double-counts frame rate (a 60 fps copy needs
    // roughly twice the bitrate to look the same as a 30 fps one) and so
    // preferred whichever copy simply had more frames.
    /// Bits spent on an average frame: bitrate divided by frame rate
    Quality,
    Size,
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

        GroupMaxima {
            duration,
            resolution: resolution_max,
            per_codec,
        }
    }

    fn codec_maxima(&self, fp: &VideoFingerprint) -> CodecMaxima {
        self.per_codec.get(&fp.codec).copied().unwrap_or_default()
    }

    /// The best value `metric` can be measured against for `fp`, in the units
    /// `value` returns. Exactly the frame of reference `tier` uses, which is
    /// what lets the value pass refine the tiers rather than contradict them.
    ///
    /// The two arms are the whole of "some metrics mean the same thing however
    /// a file was encoded, and some do not". Quality is bits per frame and size
    /// is bits per file; both are the encoder's output, and a good encoder's
    /// job is to produce fewer of them, so neither can be read across codecs
    /// and each is measured against the best its own codec managed. Length and
    /// resolution are properties of the footage rather than of the encode, so
    /// they are measured against the whole group's best.
    fn metric_max(&self, fp: &VideoFingerprint, metric: Priority) -> u64 {
        match metric {
            Priority::Length => (self.duration * 1000.0) as u64,
            Priority::Resolution => self.resolution,
            Priority::Quality => self.codec_maxima(fp).quality,
            Priority::Size => self.codec_maxima(fp).file_size,
        }
    }

    /// 1 when `fp` is within tolerance of the best value `metric` can be
    /// measured against for it, 0 otherwise. For length and resolution that is
    /// the group's best; for quality and size it is the best among files
    /// sharing `fp`'s codec.
    pub fn tier(&self, fp: &VideoFingerprint, metric: Priority) -> u8 {
        match metric {
            Priority::Length => {
                // The same reading as the frame rate below, and it matters more
                // here: a runtime nothing could measure is missing information,
                // not a short file. `duration >= max - 1s` is false for 0
                // whatever the group holds, so an unguarded zero does not merely
                // fail to rank a file -- it condemns it, on the metric that
                // normally keeps a long file safe from the clip cut out of it.
                // A 20 second raw H.264 stream that CONTAINS a 10 second MP4 of
                // the same footage was the file marked DELETE.
                //
                // Reached by anything libavformat cannot get a runtime out of
                // whose packets carry no clock to measure one from either: a
                // raw elementary stream above all, where FFmpeg's own tools
                // report N/A as well. `fingerprint_video` does measure a
                // runtime off the samples where the header is wrong, but that
                // needs a clock to read, and these files have none: their
                // samples fall back to one nominal millisecond each.
                //
                // Ties here, rather than winning: unknown is not evidence of a
                // LONGER file either. What stops the metrics that are left from
                // condemning it anyway -- size favours a dense clip over the
                // sparse capture it came from perfectly happily -- is that
                // `export.rs` holds a file of unmeasurable length for REVIEW.
                if fp.duration <= 0.0 {
                    return 1;
                }
                u8::from(fp.duration >= self.duration - DURATION_TOLERANCE_SECS)
            }
            Priority::Resolution => within(resolution(fp), self.resolution, RESOLUTION_TOLERANCE),
            Priority::Quality => {
                // An unreported frame rate is missing information, not evidence
                // of a worse copy, so it costs the file nothing here and the
                // decision moves on to a metric that is actually known. An
                // unreported runtime says the same thing by a second route:
                // `bitrate` divides by it, so quality reads 0 for an unknown
                // duration exactly as it does for an unknown frame rate, and a
                // file already spared the length tier would have lost this one
                // for the very same missing number.
                if fp.frame_rate <= 0.0 || fp.duration <= 0.0 {
                    return 1;
                }
                within(fp.quality(), self.codec_maxima(fp).quality, QUALITY_TOLERANCE)
            }
            Priority::Size => within(fp.file_size, self.codec_maxima(fp).file_size, SIZE_TOLERANCE),
        }
    }

    /// Raw value for a metric. Duration is milliseconds so everything is an
    /// integer and comparison is total (floats are not Ord).
    fn value(&self, fp: &VideoFingerprint, metric: Priority) -> u64 {
        match metric {
            Priority::Length => (fp.duration * 1000.0) as u64,
            Priority::Resolution => resolution(fp),
            Priority::Quality => fp.quality(),
            Priority::Size => fp.file_size,
        }
    }

    /// One file's standing on one metric, as the fraction `num / den` of the
    /// best value that metric can be measured against for it. Never divided --
    /// `compare` cross-multiplies, so the ordering is exact and nothing new
    /// ties through rounding.
    ///
    /// A fraction rather than a raw value because the bit-based metrics have to
    /// be read against their OWN codec's best: a bit count is never held
    /// against a bit count some other codec produced, and two files that are
    /// each the best of their own kind tie at 1 exactly as they tie on tiers.
    /// Two files sharing a codec share a denominator, which reduces their
    /// comparison to their raw values -- that is what stops a third file of
    /// some foreign codec blinding them to each other. For length and
    /// resolution every file in the group shares the denominator, so the
    /// fraction is the raw comparison written the same way.
    ///
    /// **A metric nobody could measure for this file scores 1 -- level with the
    /// best, the same "no claim either way" the tier pass makes one step
    /// earlier.** That is the load-bearing line, and it is what makes the whole
    /// ranking a total order. The unknown has to land SOMEWHERE definite,
    /// because a score is a property of one file and a comparison of scores is
    /// transitive by construction; what it must not do is land at the bottom,
    /// which is where reading the unmeasured value as a zero puts it, and which
    /// is the inversion `tier` documents at length. See
    /// `test_a_metric_nobody_could_measure_does_not_make_the_ranking_cyclic`
    /// for what answering "equal to whatever it is being compared with"
    /// instead -- a tie that says nothing about either file and so does not
    /// compose -- cost: a > b > c > a over three copies of one video, and a
    /// KEEP pick that moved with the order the group arrived in.
    ///
    /// A maximum of 0 is the same statement about the whole comparison set
    /// (quality when not one file of a codec reported a frame rate), and
    /// everyone in it scores 1 for the same reason -- exactly how `within`
    /// treats it one pass earlier.
    fn score(&self, fp: &VideoFingerprint, metric: Priority) -> (u128, u128) {
        let max = self.metric_max(fp, metric);

        if max == 0 || !measurable(fp, metric) {
            return (1, 1);
        }

        (self.value(fp, metric) as u128, max as u128)
    }

    /// Order two of the group's files on one metric, greater meaning better.
    ///
    /// Both scores are fractions of a maximum (see `score`), so this is the
    /// comparison of two rationals with non-zero denominators -- a total order,
    /// which `find_best` requires and `max_by` silently gives an arbitrary
    /// answer without. u128 because the product of two u64 metrics does not fit
    /// in one.
    fn compare(&self, a: &VideoFingerprint, b: &VideoFingerprint, metric: Priority) -> Ordering {
        let ((num_a, den_a), (num_b, den_b)) = (self.score(a, metric), self.score(b, metric));

        (num_a * den_b).cmp(&(num_b * den_a))
    }

    /// Order two of the group's files outright: every tier, then every value,
    /// then the path. `Greater` means the better copy, which is the one
    /// `find_best` keeps.
    ///
    /// This is the relation `max_by` is handed, so it has to be a total order
    /// -- see `test_the_ranking_is_a_total_order`. Paths are unique within a
    /// library, so nothing below the last line can ever tie.
    pub fn rank(
        &self,
        a: &VideoFingerprint,
        b: &VideoFingerprint,
        order: [Priority; 4],
    ) -> Ordering {
        let mut ord = Ordering::Equal;

        for m in order {
            ord = ord.then(self.tier(a, m).cmp(&self.tier(b, m)));
        }
        for m in order {
            ord = ord.then(self.compare(a, b, m));
        }

        // Reversed so the alphabetically FIRST path wins, since max_by keeps
        // the greater element.
        ord.then_with(|| b.path.cmp(&a.path))
    }
}

/// Whether `metric` was measured for `fp` at all.
///
/// Resolution and size come off the frame and the filesystem and are always
/// known. The other two rest on a runtime the container may never have reported
/// and the samples may have had no clock to measure -- length directly, quality
/// through the bitrate that divides by it -- and a file that could not be
/// measured must not be ranked as though it had scored zero. See the tiers in
/// `GroupMaxima::tier`, which this keeps whole.
pub fn measurable(fp: &VideoFingerprint, metric: Priority) -> bool {
    match metric {
        Priority::Length => fp.duration > 0.0,
        Priority::Quality => fp.frame_rate > 0.0 && fp.duration > 0.0,
        Priority::Resolution | Priority::Size => true,
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
/// second compares values (see `GroupMaxima::compare`, which keeps the
/// codec-relative ones codec-relative), separating files that tied on every
/// band. Path order settles the rest, so the result never depends on input
/// ordering.
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
        .max_by(|&&a, &&b| maxima.rank(&fps[a], &fps[b], order))
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

    /// A runtime nothing could measure is unknown, and the ranking must not
    /// read unknown as "the shortest file here".
    ///
    /// The case that forced it: a 20 second raw H.264 elementary stream holding
    /// a 10 second MP4 of the same footage. FFmpeg's own tools report N/A for
    /// such a file -- there is no duration in the container and no clock on the
    /// packets to measure one from -- so `duration` is 0, and `0 >= 10 - 1` is
    /// false. The host therefore lost the length tier to the clip cut out of
    /// it, lost the quality tier as well (bitrate divides by the same missing
    /// number), and was the file marked DELETE: the exact inversion of the
    /// property the whole ranking rests on.
    #[test]
    fn test_an_unmeasured_runtime_is_not_ranked_as_the_shortest_file() {
        let host = fp("/host.h264", 0.0, 320, 240, 141_000);
        let clip = fp("/clip.mp4", 10.0, 320, 240, 73_000);

        let group = vec![0, 1];
        let fps = vec![host, clip];
        let maxima = GroupMaxima::of(&group, &fps);

        assert_eq!(maxima.tier(&fps[0], Priority::Length), 1, "unknown is not short");
        assert_eq!(
            maxima.tier(&fps[0], Priority::Quality),
            1,
            "quality divides by the same missing runtime, so it is unknown too"
        );
        assert_eq!(
            maxima.compare(&fps[0], &fps[1], Priority::Length),
            Ordering::Equal,
            "the value pass must not undo the tier one file later"
        );
        assert_eq!(
            maxima.compare(&fps[0], &fps[1], Priority::Quality),
            Ordering::Equal
        );

        // Which leaves size, the one metric both files were measured on, and
        // the host is the bigger file.
        assert_eq!(best(&fps, Priority::Length), 0, "the host must not lose to its own clip");
    }

    /// The same file against a group that really is longer: an unknown runtime
    /// ties, it does not win. Nothing here says the file is long, only that
    /// nobody measured it, so it must not take the tier off a file that WAS
    /// measured and is plainly the longest thing in the group.
    #[test]
    fn test_an_unmeasured_runtime_does_not_win_the_length_ranking_either() {
        let unknown = fp("/unknown.h264", 0.0, 320, 240, 10_000);
        let long = fp("/long.mp4", 600.0, 320, 240, 10_000);
        let short = fp("/short.mp4", 10.0, 320, 240, 10_000);

        let group = vec![0, 1, 2];
        let fps = vec![unknown, long, short];
        let maxima = GroupMaxima::of(&group, &fps);

        assert_eq!(maxima.tier(&fps[1], Priority::Length), 1, "600s is the group's longest");
        assert_eq!(maxima.tier(&fps[2], Priority::Length), 0, "10s is not, and still is not");
        assert_eq!(
            maxima.tier(&fps[0], Priority::Length),
            1,
            "the unknown file joins the top tier rather than taking it"
        );
        // And it is a tie in both directions: the file that WAS measured is not
        // ranked below one that was not.
        assert_eq!(maxima.compare(&fps[0], &fps[1], Priority::Length), Ordering::Equal);
        assert_eq!(maxima.compare(&fps[1], &fps[0], Priority::Length), Ordering::Equal);
        // The measured pair still ranks against each other exactly as before,
        // which is the half of the metric this must not touch.
        assert_eq!(maxima.compare(&fps[1], &fps[2], Priority::Length), Ordering::Greater);
    }

    /// The ranking `find_best` hands to `max_by` has to be a TOTAL order, and
    /// treating "nobody measured this metric for one of these two files" as a
    /// pairwise tie is what stopped it being one.
    ///
    /// `Equal` from unmeasurability is not an equivalence relation: it says
    /// nothing about the two files, so it does not compose. A metric skipped
    /// for one pair is still decided for the next, and the lexicographic walk
    /// then reaches a LATER metric for some pairs and not for others -- which
    /// is all a cycle needs.
    ///
    /// Three copies of the same 10 second 640x480 footage do it. `a` beats `b`
    /// on quality, which is where their comparison stops. `c` has no frame
    /// rate, so quality says nothing about it either way and both of its
    /// comparisons fall through to size, where it is the biggest file of its
    /// own codec and beats them both. a > b > c > a.
    ///
    /// `max_by` folds left over whatever order the group arrives in, so a cycle
    /// makes the survivor of a duplicate group a function of the input order --
    /// the KEEP pick can be a file that lost a direct comparison to a file the
    /// same run marks DELETE, which is the one property `export.rs`'s delete
    /// rule rests on.
    #[test]
    fn test_a_metric_nobody_could_measure_does_not_make_the_ranking_cyclic() {
        let fps = vec![
            fp_full("/a.mkv", 10.0, 640, 480, 42_807, "av1", 33.0),
            fp_full("/b.mkv", 10.0, 640, 480, 43_451, "av1", 34.0),
            fp_full("/c.mkv", 10.0, 640, 480, 87_955, "hevc", 0.0),
        ];
        let group: Vec<usize> = (0..3).collect();
        let maxima = GroupMaxima::of(&group, &fps);
        let order = ordered_metrics(Priority::Length);

        // Not a claim about WHICH file wins -- only that the three answers can
        // be laid on one line. The old comparator answered a > b, b > c, c > a.
        let mut ranked = group.clone();
        ranked.sort_by(|&x, &y| maxima.rank(&fps[x], &fps[y], order));
        for pair in ranked.windows(2) {
            assert_ne!(
                maxima.rank(&fps[pair[0]], &fps[pair[1]], order),
                Ordering::Greater,
                "{} outranks {}, yet sorts below it",
                fps[pair[0]].path,
                fps[pair[1]].path
            );
        }

        // And the winner is the winner however the group is handed over.
        assert_permutation_invariant(&fps, Priority::Length);
    }

    /// `find_best` must not depend on the order the group arrives in. Every
    /// permutation of the group is the same set of files, so it is the same
    /// answer or the ranking is not a ranking.
    fn assert_permutation_invariant(fps: &[VideoFingerprint], priority: Priority) {
        let group: Vec<usize> = (0..fps.len()).collect();
        let maxima = GroupMaxima::of(&group, fps);
        let expected = find_best(&group, fps, priority, &maxima);

        let mut perm = group.clone();
        // Heap's algorithm, iterative: every ordering of the group, in place.
        let mut c = vec![0usize; perm.len()];
        let mut i = 0;
        while i < perm.len() {
            if c[i] < i {
                perm.swap(if i % 2 == 0 { 0 } else { c[i] }, i);
                assert_eq!(
                    find_best(&perm, fps, priority, &maxima),
                    expected,
                    "the winner moved when the group was reordered: {:?}",
                    perm.iter().map(|&j| &fps[j].path).collect::<Vec<_>>()
                );
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
    }

    /// The property the case above is one instance of, asserted directly over
    /// pseudo-random groups: the ranking is a total order, so `Greater` and
    /// `Equal` are both transitive and the relation is antisymmetric.
    ///
    /// The generator makes unmeasurable files on purpose -- a runtime no
    /// container reported and a frame rate none did -- because those are the
    /// only inputs that can break it, and it keeps the groups small and their
    /// values coarse so files really do tie and the later metrics are reached.
    #[test]
    fn test_the_ranking_is_a_total_order() {
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };

        let codecs = ["h264", "hevc", "av1"];

        for round in 0..3_000 {
            let n = 3 + (next() % 4) as usize;
            let fps: Vec<VideoFingerprint> = (0..n)
                .map(|i| {
                    // Zero means the container never said, for both of them.
                    let dur = [0.0, 10.0, 10.4, 11.0, 600.0][(next() % 5) as usize];
                    let rate = [0.0, 25.0, 30.0][(next() % 3) as usize];
                    let side = [320u32, 640, 1920][(next() % 3) as usize];
                    fp_full(
                        &format!("/r{}/f{}.mkv", round, i),
                        dur,
                        side,
                        side * 3 / 4,
                        1 + next() % 100_000,
                        codecs[(next() % 3) as usize],
                        rate,
                    )
                })
                .collect();

            let group: Vec<usize> = (0..n).collect();
            let maxima = GroupMaxima::of(&group, &fps);
            let order = ordered_metrics(Priority::Length);
            let rank = |x: usize, y: usize| maxima.rank(&fps[x], &fps[y], order);

            for &x in &group {
                for &y in &group {
                    assert_eq!(
                        rank(x, y),
                        rank(y, x).reverse(),
                        "not antisymmetric: {} against {}",
                        fps[x].path,
                        fps[y].path
                    );
                    for &z in &group {
                        if rank(x, y) == Ordering::Greater && rank(y, z) == Ordering::Greater {
                            assert_eq!(
                                rank(x, z),
                                Ordering::Greater,
                                "{} > {} > {}, yet not {} > {}",
                                fps[x].path,
                                fps[y].path,
                                fps[z].path,
                                fps[x].path,
                                fps[z].path
                            );
                        }
                        if rank(x, y) == Ordering::Equal && rank(y, z) == Ordering::Equal {
                            assert_eq!(
                                rank(x, z),
                                Ordering::Equal,
                                "{} == {} == {}, yet not {} == {}",
                                fps[x].path,
                                fps[y].path,
                                fps[z].path,
                                fps[x].path,
                                fps[z].path
                            );
                        }
                    }
                }
            }
        }
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

        assert_eq!(maxima.per_codec.len(), 2, "the group spans two codecs");
        assert_eq!(maxima.tier(&fps[0], Priority::Quality), 0, "thin h264 loses to fat h264");
        assert_eq!(maxima.tier(&fps[1], Priority::Quality), 1);
        assert_eq!(maxima.tier(&fps[2], Priority::Quality), 1, "best of its own kind");
    }

    #[test]
    fn test_a_foreign_codec_bystander_does_not_blind_the_raw_value_tiebreak() {
        // The tier pass cannot separate two h264 copies 4% apart -- that is
        // inside the 5% band -- so the raw-value pass is the only thing left
        // that can, and it must still run for files that share a codec even
        // when some third file in the group does not.
        let fps = vec![
            fp_full("a_worse.mp4", 60.0, 1920, 1080, 8_640_000, "h264", 30.0),
            fp_full("b_better.mp4", 60.0, 1920, 1080, 9_000_000, "h264", 30.0),
            fp_full("c_av1.mp4", 60.0, 1280, 720, 3_000_000, "av1", 30.0),
        ];

        assert_eq!(
            best(&fps, Priority::Quality),
            1,
            "the better h264 copy must win whether or not an av1 file is watching"
        );
    }

    #[test]
    fn test_the_best_of_each_codec_still_ties_when_one_codec_has_several() {
        // The other half of the same rule. Restoring the tiebreak must not
        // restore it ACROSS codecs: each file is measured as a fraction of the
        // best its own codec managed, so both leaders read as their codec's
        // best and tie, however many bits apart they are. The av1 file sorts
        // first, so it can only win by that tie -- if the h264 leader's three
        // times the bits counted, it would win outright and this would fail.
        let fps = vec![
            fp_full("a_av1.mp4", 60.0, 1920, 1080, 3_000_000, "av1", 30.0),
            fp_full("b_h264_best.mp4", 60.0, 1920, 1080, 9_000_000, "h264", 30.0),
            fp_full("c_h264_worse.mp4", 60.0, 1920, 1080, 8_600_000, "h264", 30.0),
        ];

        assert_eq!(
            best(&fps, Priority::Quality),
            0,
            "a leader of one codec must not be outranked by a leader of another"
        );
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