use crate::fingerprint::VideoFingerprint;
use log::info;
use rayon::prelude::*;
use crate::utils::shutdown_requested;

/// One stored hash, tagged with the video it came from.
///
/// The per-video hash index is deliberately absent: phase 1 only needs to know
/// *which videos* could overlap. All per-frame detail is recomputed exactly in
/// phase 2, so carrying it through the index is dead weight.
#[derive(Clone, Copy)]
struct IndexEntry {
    video_idx: u32,
    hash: u64,
}

// Flat Index mapping utilizing Counting Sort methodology for cache alignment optimizations
struct FlatIndex {
    entries: Vec<IndexEntry>,
    offsets: Vec<u32>,
}

impl FlatIndex {
    fn build(k: usize, fingerprints: &[VideoFingerprint]) -> Self {
        let mut counts = vec![0u32; 65537];

        for fp in fingerprints.iter() {
            for &h in &fp.valid_hashes {
                let bin = ((h >> (48 - k * 16)) & 0xFFFF) as usize;
                counts[bin + 1] += 1;
            }
        }

        for i in 1..=65536 {
            counts[i] += counts[i - 1];
        }

        let mut entries = vec![IndexEntry { video_idx: 0, hash: 0 }; counts[65536] as usize];
        let mut offsets = counts.clone();

        for (v_idx, fp) in fingerprints.iter().enumerate() {
            for &h in fp.valid_hashes.iter() {
                let bin = ((h >> (48 - k * 16)) & 0xFFFF) as usize;
                let pos = offsets[bin] as usize;
                entries[pos] = IndexEntry { video_idx: v_idx as u32, hash: h };
                offsets[bin] += 1;
            }
        }

        Self { entries, offsets: counts }
    }

    #[inline(always)]
    fn get_bin(&self, bin: usize) -> &[IndexEntry] {
        let bin = bin & 0xFFFF; // Bounds hint allowing LLVM to elide panic branches safely
        let start = self.offsets[bin] as usize;
        let end = self.offsets[bin + 1] as usize;
        &self.entries[start..end]
    }
}

/// --- Phase 1: candidate generation ---------------------------------------
///
/// Probe the 4x16-bit multi-index at radius 1 and emit every ordered pair
/// `(a, b)` with `a < b` that shares at least one frame within
/// `max_hamming_dist`. This is a *filter*, not the answer: it decides which
/// video pairs are worth a full comparison, and nothing else.
///
/// Completeness note: a pair is found only if some shared frame has a 16-bit
/// block differing by <= 1 bit, which by pigeonhole is guaranteed for frame
/// distances <= 7. Above that the probe can miss an individual frame pair, but
/// real duplicates share hundreds of frames, so missing *every* one of them is
/// vanishingly unlikely. Phase 2 then recovers the frames the probe skipped.
fn candidate_pairs(
    fingerprints: &[VideoFingerprint],
    max_hamming_dist: u32,
) -> Vec<(usize, usize)> {
    let n = fingerprints.len();

    let indices: Vec<FlatIndex> = (0..4)
        .into_par_iter()
        .map(|k| FlatIndex::build(k, fingerprints))
        .collect();

    (0..n)
        .into_par_iter()
        .flat_map(|v_a| {
            if shutdown_requested() {
                return Vec::new();
            }
            let fp_a = &fingerprints[v_a];

            // Once a video is a known candidate, further hits against it are
            // pure waste -- `seen` lets us skip the popcount entirely. This is
            // also why the old first-block deduplication is gone: it existed to
            // stop the same frame pair being counted twice, and set membership
            // makes double counting impossible by construction.
            let mut seen = vec![false; n];
            let mut candidates: Vec<(usize, usize)> = Vec::new();

            for &h_a in fp_a.valid_hashes.iter() {
                if shutdown_requested() {
                    return Vec::new();
                }
                let b = [
                    ((h_a >> 48) & 0xFFFF) as usize,
                    ((h_a >> 32) & 0xFFFF) as usize,
                    ((h_a >> 16) & 0xFFFF) as usize,
                    (h_a & 0xFFFF) as usize,
                ];

                for k in 0..4 {
                    for bit in 0..17 {
                        let key = if bit == 16 { b[k] } else { b[k] ^ (1 << bit) };
                        let bin_entries = indices[k].get_bin(key);

                        // Entries are built in video order, so a binary search
                        // skips every already-processed video in one step.
                        let start_idx =
                            bin_entries.partition_point(|e| (e.video_idx as usize) <= v_a);

                        for entry in &bin_entries[start_idx..] {
                            let v_b = entry.video_idx as usize;
                            if seen[v_b] {
                                continue;
                            }
                            if (h_a ^ entry.hash).count_ones() <= max_hamming_dist {
                                seen[v_b] = true;
                                candidates.push((v_a, v_b));
                            }
                        }
                    }
                }
            }

            candidates
        })
        .collect()
}

/// --- Phase 2: exact verification ------------------------------------------
///
/// Brute-force every frame of A against every frame of B. Two ~400-hash videos
/// is ~160k XOR+popcount operations -- microseconds -- and since genuine
/// candidates are rare relative to the library size, this is close to free.
///
/// Returns the fraction of each video's runtime that is matched by the other.
/// Unlike the index-driven count it replaces, this sees *all* matching frames,
/// including ones whose blocks all differ by >= 2 bits and are therefore
/// invisible to the radius-1 probe.
fn match_overlap(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    max_hamming_dist: u32,
) -> (f32, f32) {
    let mut matched_a = vec![false; fp_a.valid_hashes.len()];
    let mut matched_b = vec![false; fp_b.valid_hashes.len()];

    for (i, &h_a) in fp_a.valid_hashes.iter().enumerate() {
        for (j, &h_b) in fp_b.valid_hashes.iter().enumerate() {
            if (h_a ^ h_b).count_ones() <= max_hamming_dist {
                matched_a[i] = true;
                matched_b[j] = true;
            }
        }
    }

    // Each stored hash covers the frame span [t_start, t_end), so matched
    // frames are the sum of the spans of the hashes that matched.
    let span_sum = |matched: &[bool], fp: &VideoFingerprint| -> u32 {
        matched
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| fp.valid_t_end[i] - fp.valid_t_start[i])
            .sum()
    };

    let frames_a = span_sum(&matched_a, fp_a);
    let frames_b = span_sum(&matched_b, fp_b);

    let pct_a = if fp_a.total_frames > 0 {
        frames_a as f32 / fp_a.total_frames as f32
    } else {
        0.0
    };
    let pct_b = if fp_b.total_frames > 0 {
        frames_b as f32 / fp_b.total_frames as f32
    } else {
        0.0
    };

    (pct_a, pct_b)
}

pub fn find_all_matches(
    fingerprints: &[VideoFingerprint],
    max_hamming_dist: u32,
    min_match_percent: f32,
    min_duration: f64,
) -> Vec<(usize, usize)> {
    let candidates = candidate_pairs(fingerprints, max_hamming_dist);
    if shutdown_requested() {
        return Vec::new();
    }
    info!("Index scan produced {} candidate pair(s); verifying...", candidates.len());

    candidates
        .into_par_iter()
        .filter(|&(v_a, v_b)| {
            if shutdown_requested() {
                return false;
            }
            let fp_a = &fingerprints[v_a];
            let fp_b = &fingerprints[v_b];

            let (pct_a, pct_b) = match_overlap(fp_a, fp_b, max_hamming_dist);

            if pct_a.max(pct_b) < min_match_percent {
                return false;
            }

            if min_duration > 0.0 {
                let matched_secs =
                    (pct_a as f64 * fp_a.duration).max(pct_b as f64 * fp_b.duration);
                if matched_secs < min_duration {
                    return false;
                }
            }

            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_fp_with_hashes(hashes: Vec<u64>, frames: u32) -> VideoFingerprint {
        let len = hashes.len();
        VideoFingerprint {
            path: "mock.mp4".to_string(),
            valid_hashes: hashes,
            // Mock time intervals directly correlating to the index
            valid_t_start: (0..len as u32).collect(),
            valid_t_end: (1..=len as u32).collect(),
            total_frames: frames,
            width: 1920,
            height: 1080,
            duration: 10.0,
            file_size: 1024,
        }
    }

    #[test]
    fn test_find_all_matches_exact() {
        let hash = 0xFFFF_0000_FFFF_0000;
        let fps = vec![
            mock_fp_with_hashes(vec![hash, hash], 2), // Video A
            mock_fp_with_hashes(vec![hash, hash], 2), // Video B (Exact match)
        ];

        let matches = find_all_matches(&fps, 0, 1.0, 0.0);

        assert!(!matches.is_empty(), "Exact duplicates should match");
        assert!(matches.contains(&(0, 1)) || matches.contains(&(1, 0)));
    }

    #[test]
    fn test_find_all_matches_hamming_limit() {
        let base_hash = 0x0000_0000_0000_0000;
        let diff_hash = 0x0000_0000_0000_0007; // 3 bits different

        let fps = vec![
            mock_fp_with_hashes(vec![base_hash, base_hash], 2),
            mock_fp_with_hashes(vec![diff_hash, diff_hash], 2),
        ];

        // Should NOT match if max_hamming is 2
        let no_matches = find_all_matches(&fps, 2, 1.0, 0.0);
        assert!(no_matches.is_empty(), "Should be filtered by hamming distance");

        // SHOULD match if max_hamming is 3
        let valid_matches = find_all_matches(&fps, 3, 1.0, 0.0);
        assert!(!valid_matches.is_empty(), "Should pass hamming distance check");
    }

    #[test]
    fn test_pairs_are_emitted_once_in_ascending_order() {
        let hash = 0xABCD_1234_ABCD_1234;
        let fps = vec![
            mock_fp_with_hashes(vec![hash], 1),
            mock_fp_with_hashes(vec![hash], 1),
            mock_fp_with_hashes(vec![hash], 1),
        ];

        let mut matches = find_all_matches(&fps, 0, 1.0, 0.0);
        matches.sort_unstable();

        // Each unordered pair exactly once, always with the lower index first.
        assert_eq!(matches, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn test_two_phase_recovers_frames_the_index_misses() {
        // Frame 2 differs by exactly 2 bits in EVERY 16-bit block (total
        // distance 8), so the radius-1 probe can never see it. Frame 1 is
        // identical, so the pair still becomes a candidate -- and phase 2's
        // brute force then counts frame 2 as well.
        let shared = 0x0000_0000_0000_0000u64;
        let invisible_a = 0xFFFF_FFFF_FFFF_FFFFu64;
        let invisible_b = invisible_a ^ 0x0003_0003_0003_0003;

        assert_eq!((invisible_a ^ invisible_b).count_ones(), 8);

        let fps = vec![
            mock_fp_with_hashes(vec![shared, invisible_a], 2),
            mock_fp_with_hashes(vec![shared, invisible_b], 2),
        ];

        // Demanding 100% overlap: only reachable if BOTH frames are counted.
        // The old index-only accounting would have scored this pair at 50%.
        let matches = find_all_matches(&fps, 8, 1.0, 0.0);
        assert_eq!(matches, vec![(0, 1)], "phase 2 must recover the index-invisible frame");
    }

    #[test]
    fn test_min_duration_gates_independently_of_match_percent() {
        let hash = 0xABCD_1234_ABCD_1234u64;
        let fps = vec![
            mock_fp_with_hashes(vec![hash, hash], 2),
            mock_fp_with_hashes(vec![hash, hash], 2),
        ];

        // 100% overlap of a 10s mock = 10 matched seconds.
        assert_eq!(find_all_matches(&fps, 0, 1.0, 5.0), vec![(0, 1)], "10s clears a 5s floor");
        assert!(
            find_all_matches(&fps, 0, 1.0, 30.0).is_empty(),
            "a full-coverage match must still fail a 30s floor"
        );
    }
}