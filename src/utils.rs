use crate::fingerprint::VideoFingerprint;

#[derive(clap::ValueEnum, Clone, Debug, Copy, PartialEq)]
pub enum Priority {
    Length,
    Resolution,
    Size,
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

// Helper function to deterministically find the "best" file in a group based on priority logic.
pub fn find_best(
    group: &[usize],
    fps: &[VideoFingerprint],
    priority: Priority,
    max_dur: f64,
) -> usize {
    *group
        .iter()
        .max_by(|&&a, &&b| {
            let fp_a = &fps[a];
            let fp_b = &fps[b];

            let dur_a = (fp_a.duration * 1000.0) as u64;
            let dur_b = (fp_b.duration * 1000.0) as u64;

            // Tier 1 if within 0.5s of the absolute max duration, 0 otherwise (Duration Tolerance)
            let tier_a = if fp_a.duration >= max_dur - 0.5 { 1 } else { 0 };
            let tier_b = if fp_b.duration >= max_dur - 0.5 { 1 } else { 0 };

            let res_a = fp_a.width * fp_a.height;
            let res_b = fp_b.width * fp_b.height;

            let size_a = fp_a.file_size;
            let size_b = fp_b.file_size;

            // Tie breaker: Alphabetically first path (b.cmp(a) reverses normal string sort so A wins)
            let path_ord = fp_b.path.cmp(&fp_a.path);

            match priority {
                Priority::Length => tier_a
                    .cmp(&tier_b)
                    .then(res_a.cmp(&res_b))
                    .then(size_a.cmp(&size_b))
                    .then(dur_a.cmp(&dur_b))
                    .then(path_ord),
                Priority::Resolution => res_a
                    .cmp(&res_b)
                    .then(tier_a.cmp(&tier_b))
                    .then(dur_a.cmp(&dur_b))
                    .then(size_a.cmp(&size_b))
                    .then(path_ord),
                Priority::Size => size_a
                    .cmp(&size_b)
                    .then(tier_a.cmp(&tier_b))
                    .then(dur_a.cmp(&dur_b))
                    .then(res_a.cmp(&res_b))
                    .then(path_ord),
            }
        })
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

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
}