use crate::fingerprint::VideoFingerprint;
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct IndexEntry {
    video_idx: u32,
    hash_idx: u32,
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
        
        let mut entries = vec![IndexEntry { video_idx: 0, hash_idx: 0, hash: 0 }; counts[65536] as usize];
        let mut offsets = counts.clone();
        
        for (v_idx, fp) in fingerprints.iter().enumerate() {
            for (h_idx, &h) in fp.valid_hashes.iter().enumerate() {
                let bin = ((h >> (48 - k * 16)) & 0xFFFF) as usize;
                let pos = offsets[bin] as usize;
                entries[pos] = IndexEntry {
                    video_idx: v_idx as u32,
                    hash_idx: h_idx as u32,
                    hash: h,
                };
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

pub fn find_all_matches(
    fingerprints: &[VideoFingerprint],
    max_hamming_dist: u32,
    min_match_percent: f32,
) -> Vec<(usize, usize)> {
    let indices: Vec<FlatIndex> = (0..4).into_par_iter().map(|k| FlatIndex::build(k, fingerprints)).collect();

    (0..fingerprints.len())
        .into_par_iter()
        .flat_map(|v_a| {
            let fp_a = &fingerprints[v_a];
            
            // Replaced O(N^2) Vector nested matrix with Flat List
            let mut matches_flat: Vec<(usize, u32, u32)> = Vec::new();
            
            for (idx_a, &h_a) in fp_a.valid_hashes.iter().enumerate() {
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
                        
                        // Binary search to skip past already processed videos to bypass O(N^2) linear scanning waste
                        let start_idx = bin_entries.partition_point(|e| (e.video_idx as usize) <= v_a);
                        
                        for entry in &bin_entries[start_idx..] {
                            let xor = h_a ^ entry.hash;
                            if xor.count_ones() <= max_hamming_dist {
                                // First-block deduplication: Only count the match if `k` is the very first block that would have successfully triggered it
                                let mut is_first = true;
                                for j in 0..k {
                                    if ((xor >> (48 - j * 16)) & 0xFFFF).count_ones() <= 1 {
                                        is_first = false;
                                        break;
                                    }
                                }
                                
                                if is_first {
                                    matches_flat.push((entry.video_idx as usize, idx_a as u32, entry.hash_idx));
                                }
                            }
                        }
                    }
                }
            }
            
            let mut local_edges = Vec::new();
            if matches_flat.is_empty() { return local_edges; }

            // Quicksort completely naturally groups by Target Video -> Frame A -> Frame B
            matches_flat.sort_unstable();
            
            let mut i = 0;
            while i < matches_flat.len() {
                let v_b = matches_flat[i].0;
                
                let mut j = i + 1;
                while j < matches_flat.len() && matches_flat[j].0 == v_b { j += 1; }
                
                let group = &matches_flat[i..j];
                i = j; 
                
                let fp_b = &fingerprints[v_b];
                
                let mut match_frames_a = 0;
                let mut last_a = u32::MAX;
                for &(_, idx_a, _) in group {
                    if idx_a != last_a {
                        match_frames_a += fp_a.valid_t_end[idx_a as usize] - fp_a.valid_t_start[idx_a as usize];
                        last_a = idx_a;
                    }
                }

                let mut unique_b: Vec<u32> = group.iter().map(|&(_, _, b)| b).collect();
                unique_b.sort_unstable();
                unique_b.dedup();
                
                let mut match_frames_b = 0;
                for &idx_b in &unique_b {
                    match_frames_b += fp_b.valid_t_end[idx_b as usize] - fp_b.valid_t_start[idx_b as usize];
                }

                let pct_a = if fp_a.total_frames > 0 { match_frames_a as f32 / fp_a.total_frames as f32 } else { 0.0 };
                let pct_b = if fp_b.total_frames > 0 { match_frames_b as f32 / fp_b.total_frames as f32 } else { 0.0 };
                
                if pct_a.max(pct_b) >= min_match_percent {
                    local_edges.push((v_a, v_b));
                }
            }
            local_edges
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

        // max_hamming: 0, min_match: 1.0 (100%)
        let matches = find_all_matches(&fps, 0, 1.0);
        
        // Remember that find_all_matches tests pairs. Depending on thread order, 
        // it usually yields pairs like (0, 1) or (1, 0)
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
        let no_matches = find_all_matches(&fps, 2, 1.0);
        assert!(no_matches.is_empty(), "Should be filtered by hamming distance");

        // SHOULD match if max_hamming is 3
        let valid_matches = find_all_matches(&fps, 3, 1.0);
        assert!(!valid_matches.is_empty(), "Should pass hamming distance check");
    }
}