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
                        for entry in indices[k].get_bin(key) {
                            if entry.video_idx as usize > v_a {
                                if (h_a ^ entry.hash).count_ones() <= max_hamming_dist {
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