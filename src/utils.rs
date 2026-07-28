use crate::fingerprint::VideoFingerprint;
use std::cmp::Ordering;

#[derive(clap::ValueEnum, Clone, Debug, Copy, PartialEq, Eq)]
pub enum Priority {
    Length,
    Resolution,
    Bitrate,
    Size,
}

/// Default precedence when ranking two copies of the same video:
/// length > resolution > bitrate > size.
///
/// `--priority X` does not reshuffle the rest of the list, it only moves X to
/// the front: `--priority bitrate` ranks bitrate > length > resolution > size.
const DEFAULT_ORDER: [Priority; 4] = [
    Priority::Length,
    Priority::Resolution,
    Priority::Bitrate,
    Priority::Size,
];

/// Tolerance bands. Files inside the same band are treated as equal on that
/// metric, so the decision falls through to the next one instead of being
/// settled by noise. See the README for the reasoning behind each width.
pub const DURATION_TOLERANCE_SECS: f64 = 1.0;
pub const RESOLUTION_TOLERANCE: f64 = 0.05;
pub const BITRATE_TOLERANCE: f64 = 0.10;
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

pub fn resolution(fp: &VideoFingerprint) -> u64 {
    fp.width as u64 * fp.height as u64
}

/// Raw comparable value for a metric. Duration is milliseconds so everything
/// is an integer and comparison is total (floats are not Ord).
fn metric_value(fp: &VideoFingerprint, metric: Priority) -> u64 {
    match metric {
        Priority::Length => (fp.duration * 1000.0) as u64,
        Priority::Resolution => resolution(fp),
        Priority::Bitrate => fp.bitrate(),
        Priority::Size => fp.file_size,
    }
}

/// The best value seen for each metric within one duplicate group. Tolerance
/// is always measured against these, never between two files directly:
/// "within 5% of each other" is not a transitive relation, and feeding a
/// non-transitive comparator to `max_by` yields an arbitrary winner.
#[derive(Clone, Copy, Debug)]
pub struct GroupMaxima {
    pub duration: f64,
    pub resolution: u64,
    pub bitrate: u64,
    pub file_size: u64,
}

impl GroupMaxima {
    pub fn of(group: &[usize], fps: &[VideoFingerprint]) -> Self {
        let mut m = GroupMaxima {
            duration: 0.0,
            resolution: 0,
            bitrate: 0,
            file_size: 0,
        };
        for &idx in group {
            let fp = &fps[idx];
            m.duration = m.duration.max(fp.duration);
            m.resolution = m.resolution.max(resolution(fp));
            m.bitrate = m.bitrate.max(fp.bitrate());
            m.file_size = m.file_size.max(fp.file_size);
        }
        m
    }

    /// 1 when `fp` is within tolerance of the group's best value for `metric`,
    /// 0 otherwise.
    pub fn tier(&self, fp: &VideoFingerprint, metric: Priority) -> u8 {
        match metric {
            Priority::Length => u8::from(fp.duration >= self.duration - DURATION_TOLERANCE_SECS),
            Priority::Resolution => {
                within(resolution(fp), self.resolution, RESOLUTION_TOLERANCE)
            }
            Priority::Bitrate => within(fp.bitrate(), self.bitrate, BITRATE_TOLERANCE),
            Priority::Size => within(fp.file_size, self.file_size, SIZE_TOLERANCE),
        }
    }
}

/// Relative tolerance check. A zero maximum means the metric is unknown for the
/// whole group (e.g. bitrate when no duration could be read); everyone ties so
/// the ranking falls through to a metric that is actually known.
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
                ord = ord.then(metric_value(fp_a, m).cmp(&metric_value(fp_b, m)));
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

    fn fp(path: &str, dur: f64, w: u32, h: u32, size: u64) -> VideoFingerprint {
        VideoFingerprint {
            path: path.to_string(),
            valid_hashes: vec![],
            valid_t_start: vec![],
            valid_t_end: vec![],
            total_frames: 100,
            width: w,
            height: h,
            duration: dur,
            file_size: size,
        }
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
    fn test_bitrate_derived_from_size_and_duration() {
        // 7,864,320 bytes over 60s = 1,048,576 bits/s.
        let f = fp("a.mp4", 60.0, 1920, 1080, 7_864_320);
        assert_eq!(f.bitrate(), 1_048_576);

        // Unknown duration must not divide by zero.
        assert_eq!(fp("b.mp4", 0.0, 1920, 1080, 1024).bitrate(), 0);
    }

    #[test]
    fn test_bitrate_outranks_size_by_default() {
        // Same length, same resolution: the decision reaches bitrate, and the
        // denser file wins even though it is the smaller one.
        let fps = vec![
            fp("low.mp4", 120.0, 1920, 1080, 30_000_000),
            fp("high.mp4", 60.0 * 2.0, 1920, 1080, 60_000_000),
        ];
        assert_eq!(best(&fps, Priority::Length), 1);
    }

    #[test]
    fn test_resolution_tolerance_absorbs_crop() {
        // 1920x1040 is 96.3% of 1920x1080's pixel count -- a crop, not a
        // downscale -- so both sit in the top tier and bitrate decides.
        let fps = vec![
            fp("uncropped.mp4", 60.0, 1920, 1080, 6_000_000),
            fp("cropped.mp4", 60.0, 1920, 1040, 9_000_000),
        ];
        assert_eq!(best(&fps, Priority::Resolution), 1);
    }

    #[test]
    fn test_resolution_tiers_are_not_absorbed() {
        // 720p vs 1080p is far outside the band, so resolution settles it
        // regardless of the 720p file's higher bitrate.
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
        assert_eq!(best(&fps, Priority::Bitrate), 1, "bitrate first when asked");
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