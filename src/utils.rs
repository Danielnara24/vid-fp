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