use crate::fingerprint::VideoFingerprint;

pub struct CompareResult {
    pub match_length_seconds: f32,
    pub match_percent: f32,
    pub interval_a: (f32, f32),
    pub interval_b: (f32, f32),
}

pub fn compare_videos(
    v1: &VideoFingerprint,
    v2: &VideoFingerprint,
    max_hamming_dist: u32,
    min_match_percent: f32,
) -> Option<CompareResult> {
    if v1.valid_hashes.is_empty() || v2.valid_hashes.is_empty() {
        return None;
    }

    // Track which indices matched at least once
    let mut matched_a = vec![false; v1.valid_hashes.len()];
    let mut matched_b = vec![false; v2.valid_hashes.len()];
    
    let mut has_any_match = false;

    // Cross-compare all hashes
    for (i, h_a) in v1.valid_hashes.iter().enumerate() {
        for (j, h_b) in v2.valid_hashes.iter().enumerate() {
            // Native Hardware Popcount (The main reason Rust will obliterate Python's speed here)
            let dist = (h_a ^ h_b).count_ones(); 
            
            if dist <= max_hamming_dist {
                matched_a[i] = true;
                matched_b[j] = true;
                has_any_match = true;
            }
        }
    }

    if !has_any_match {
        return None;
    }

    // Calculate match durations
    let mut match_dur_a = 0.0;
    let mut first_match_a = -1isize;
    let mut last_match_a = -1isize;
    for i in 0..v1.valid_hashes.len() {
        if matched_a[i] {
            match_dur_a += v1.valid_t_end[i] - v1.valid_t_start[i];
            if first_match_a == -1 { first_match_a = i as isize; }
            last_match_a = i as isize;
        }
    }

    let mut match_dur_b = 0.0;
    let mut first_match_b = -1isize;
    let mut last_match_b = -1isize;
    for j in 0..v2.valid_hashes.len() {
        if matched_b[j] {
            match_dur_b += v2.valid_t_end[j] - v2.valid_t_start[j];
            if first_match_b == -1 { first_match_b = j as isize; }
            last_match_b = j as isize;
        }
    }

    let pct_a = if v1.duration > 0.0 { match_dur_a / v1.duration } else { 0.0 };
    let pct_b = if v2.duration > 0.0 { match_dur_b / v2.duration } else { 0.0 };
    
    let match_percent = pct_a.max(pct_b);

    if match_percent >= min_match_percent {
        let match_length_seconds = if pct_a >= pct_b { match_dur_a } else { match_dur_b };
        
        let interval_a = (
            v1.valid_t_start[first_match_a as usize],
            v1.valid_t_end[last_match_a as usize],
        );
        let interval_b = (
            v2.valid_t_start[first_match_b as usize],
            v2.valid_t_end[last_match_b as usize],
        );

        Some(CompareResult {
            match_length_seconds,
            match_percent,
            interval_a,
            interval_b,
        })
    } else {
        None
    }
}