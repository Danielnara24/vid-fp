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
        
        Self { entries, offsets: counts } // Retain unmodified initial prefix sums for O(1) array slicing later
    }
    
    #[inline(always)]
    fn get_bin(&self, bin: usize) -> &[IndexEntry] {
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
    
    // 1. Construct Multi-Index Hashing tables mapping chunks to frames simultaneously.
    let indices: Vec<FlatIndex> = (0..4).into_par_iter().map(|k| FlatIndex::build(k, fingerprints)).collect();

    // 2. Query in parallel mapping direct index limits via LSH
    (0..fingerprints.len())
        .into_par_iter()
        .flat_map(|v_a| {
            let fp_a = &fingerprints[v_a];
            
            // Replaces HashMap lookup penalties globally. Pre-sizes zeroed vecs cleanly caching on the stack sizes.
            let mut pair_matches: Vec<Vec<(u32, u32)>> = vec![Vec::new(); fingerprints.len()];
            
            for (idx_a, &h_a) in fp_a.valid_hashes.iter().enumerate() {
                let b = [
                    ((h_a >> 48) & 0xFFFF) as usize,
                    ((h_a >> 32) & 0xFFFF) as usize,
                    ((h_a >> 16) & 0xFFFF) as usize,
                    (h_a & 0xFFFF) as usize,
                ];
                
                // Pigeonhole principle: If Error <= 5, split across 4 blocks ensures AT LEAST one block has <= 1 error 
                for k in 0..4 {
                    for bit in 0..17 {
                        let key = if bit == 16 { b[k] } else { b[k] ^ (1 << bit) };
                        for entry in indices[k].get_bin(key) {
                            if entry.video_idx as usize > v_a {
                                if (h_a ^ entry.hash).count_ones() <= max_hamming_dist {
                                    pair_matches[entry.video_idx as usize].push((idx_a as u32, entry.hash_idx));
                                }
                            }
                        }
                    }
                }
            }
            
            let mut local_edges = Vec::new();
            for (v_b, mut matches) in pair_matches.into_iter().enumerate() {
                if matches.is_empty() { continue; }
                
                matches.sort_unstable();
                matches.dedup();
                
                let fp_b = &fingerprints[v_b];
                
                let mut matched_a = vec![false; fp_a.valid_hashes.len()];
                let mut matched_b = vec![false; fp_b.valid_hashes.len()];
                
                for (idx_a, idx_b) in matches {
                    matched_a[idx_a as usize] = true;
                    matched_b[idx_b as usize] = true;
                }
                
                let mut match_frames_a = 0;
                for i in 0..fp_a.valid_hashes.len() {
                    if matched_a[i] { match_frames_a += fp_a.valid_t_end[i] - fp_a.valid_t_start[i]; }
                }

                let mut match_frames_b = 0;
                for j in 0..fp_b.valid_hashes.len() {
                    if matched_b[j] { match_frames_b += fp_b.valid_t_end[j] - fp_b.valid_t_start[j]; }
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