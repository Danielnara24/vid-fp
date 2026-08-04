use crate::fingerprint::VideoFingerprint;
use crate::utils::shutdown_requested;
use log::info;
use rayon::prelude::*;
use std::collections::HashMap;

/// The hash is split into this many equal blocks for indexing.
///
/// This is the number the whole probe strategy is derived from: two hashes
/// within total distance `d` must, by pigeonhole, agree to within `d / BLOCKS`
/// bits in at least ONE block, because if every block were further apart than
/// that the total would exceed `d`.
const BLOCKS: usize = 4;
const BLOCK_BITS: usize = 64 / BLOCKS;
const BINS: usize = 1 << BLOCK_BITS;

/// Ceiling on how far the probe will widen, whatever `--hamming-distance` says.
///
/// The number of keys probed per block is the number of 16-bit patterns with at
/// most `radius` bits set: 1, 17, 137, 697. The first three steps are a
/// reasonable price for exhaustiveness; the fourth is a 5x jump on top of an
/// already 137x one, at tolerances where the candidate list is exploding anyway.
/// Past the cap the index goes back to being a very good filter rather than a
/// complete one -- which is exactly what it was at EVERY setting before, so
/// nothing regresses, the honest-answer threshold just moves from 7 to 11.
const MAX_PROBE_RADIUS: u32 = 2;

/// The `k`th 16-bit block of a hash, most significant first.
#[inline(always)]
fn block_of(hash: u64, k: usize) -> usize {
    ((hash >> (64 - BLOCK_BITS * (k + 1))) & (BINS as u64 - 1)) as usize
}

/// How far around each block key the index must look to be exhaustive at
/// `max_hamming_dist`.
///
/// Integer division is the whole rule. At the default tolerance of 3 this is 0:
/// three differing bits cannot be spread across four blocks without leaving one
/// of them untouched, so probing the exact bins alone finds every pair, and the
/// 68 lookups per hash this code used to do were 64 lookups of pure ceremony.
fn probe_radius(max_hamming_dist: u32) -> u32 {
    (max_hamming_dist / BLOCKS as u32).min(MAX_PROBE_RADIUS)
}

/// Every bit pattern to XOR a block key with, i.e. every 16-bit value with at
/// most `radius` bits set.
///
/// Enumerated by filtering all 65,536 patterns rather than by generating
/// combinations. It runs once per scan, it is obviously complete by
/// construction, and it does not have to be rewritten if the cap ever moves.
fn probe_masks(radius: u32) -> Vec<u16> {
    (0..=u16::MAX).filter(|m| m.count_ones() <= radius).collect()
}

/// A counting-sorted index over ONE block position: every stored hash, bucketed
/// by the 16 bits at that position, tagged with the video it came from.
///
/// Hashes and video ids live in two parallel arrays rather than one array of
/// `{ u32, u64 }` structs. That struct aligns to 8 and pads to 16 bytes to hold
/// 12 bytes of payload, and this is the largest allocation in the program by a
/// wide margin -- one entry per keyframe in the entire library.
///
/// The per-video hash index is deliberately absent: this phase only needs to
/// know *which videos* could overlap. All per-frame detail is recomputed
/// exactly in phase 2, so carrying it through the index is dead weight.
struct BlockIndex {
    hashes: Vec<u64>,
    videos: Vec<u32>,
    offsets: Vec<u32>,
}

impl BlockIndex {
    fn build(k: usize, fingerprints: &[VideoFingerprint]) -> Self {
        let mut offsets = vec![0u32; BINS + 1];

        for fp in fingerprints {
            for &h in &fp.valid_hashes {
                offsets[block_of(h, k) + 1] += 1;
            }
        }
        for i in 1..=BINS {
            offsets[i] += offsets[i - 1];
        }

        let total = offsets[BINS] as usize;
        let mut hashes = vec![0u64; total];
        let mut videos = vec![0u32; total];

        // A separate write cursor so `offsets` keeps its bin boundaries; the
        // cursor is 256 KiB of scratch and dies with this function.
        let mut cursor = offsets.clone();

        for (v_idx, fp) in fingerprints.iter().enumerate() {
            for &h in &fp.valid_hashes {
                let bin = block_of(h, k);
                let pos = cursor[bin] as usize;
                hashes[pos] = h;
                videos[pos] = v_idx as u32;
                cursor[bin] += 1;
            }
        }

        Self { hashes, videos, offsets }
    }

    /// The hashes in one bin and the videos they came from, index for index.
    #[inline(always)]
    fn bin(&self, key: usize) -> (&[u64], &[u32]) {
        let key = key & (BINS - 1); // Bounds hint allowing LLVM to elide panic branches safely
        let start = self.offsets[key] as usize;
        let end = self.offsets[key + 1] as usize;
        (&self.hashes[start..end], &self.videos[start..end])
    }
}

/// Two videos that share enough content to be reported together, and by how
/// much.
///
/// Coverage is *directional*, and that asymmetry is real rather than an
/// artefact: a two-minute clip cut from a twenty-two minute episode covers
/// ~100% of the clip and ~9% of the episode. Both numbers are correct, they
/// describe completely different situations, and no single figure expresses
/// either. They are kept apart here and reconciled only at the point of
/// reporting -- see `shared_seconds`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Match {
    pub a: usize,
    pub b: usize,
    /// Fraction of `a`'s runtime that is also present in `b` (0.0 ..= 1.0).
    pub coverage_a: f32,
    /// Fraction of `b`'s runtime that is also present in `a` (0.0 ..= 1.0).
    pub coverage_b: f32,
}

/// Coverage figures made lookup-friendly, keyed by (subject, other).
///
/// Both directions of every pair are stored, so `coverage(i, j)` always answers
/// "how much of i does j contain" without the caller having to know which way
/// round the pair was originally emitted.
pub struct MatchIndex {
    coverage: HashMap<(usize, usize), f32>,
}

impl MatchIndex {
    pub fn new(matches: Vec<Match>) -> Self {
        let mut coverage = HashMap::with_capacity(matches.len() * 2);
        for m in matches {
            coverage.insert((m.a, m.b), m.coverage_a);
            coverage.insert((m.b, m.a), m.coverage_b);
        }
        Self { coverage }
    }

    /// How much of `subject` is contained in `other`, if the two were compared.
    pub fn coverage(&self, subject: usize, other: usize) -> Option<f32> {
        self.coverage.get(&(subject, other)).copied()
    }

    /// Seconds of content two files have in common.
    ///
    /// Each file gives its own estimate -- its coverage times its runtime --
    /// and for genuinely shared footage the two agree, because the shared
    /// segment has one real duration no matter which file you measure it in.
    /// The clip's `100% x 2min` and the host's `9% x 22min` are both two
    /// minutes.
    ///
    /// That agreement is the whole point. Coverage is asymmetric, and its
    /// asymmetry is what makes a report confusing; duration is symmetric and
    /// reads the same from either end. Where the two estimates disagree --
    /// different keyframe densities, tolerance landing differently on each side
    /// -- the lower is taken, the conservative reading for a tool that deletes
    /// things.
    ///
    /// A file whose runtime the container never reported contributes no
    /// estimate. If neither file has a known runtime the answer is `None`: the
    /// overlap is unknown, which is not the same as zero.
    pub fn shared_seconds(&self, a: usize, b: usize, fps: &[VideoFingerprint]) -> Option<f64> {
        let mut estimate: Option<f64> = None;
        for (subject, other) in [(a, b), (b, a)] {
            let runtime = fps[subject].duration;
            if runtime <= 0.0 {
                continue;
            }
            let seconds = self.coverage(subject, other)? as f64 * runtime;
            estimate = Some(estimate.map_or(seconds, |e: f64| e.min(seconds)));
        }
        estimate
    }

    /// The least content `subject` shares with any other member of `group`.
    ///
    /// Groups are maximal cliques, so every pair here was genuinely compared
    /// and this is a measured floor rather than an average over unrelated
    /// things: it reads as "this file shares at least this much with everything
    /// it is grouped with". Printed beside the file's own length it makes a
    /// group interpretable at a glance -- a nine-second clip sharing eight
    /// seconds is a duplicate, and one sharing half a second is not, however
    /// impressive that half second looks as a percentage.
    pub fn weakest_shared_in_group(
        &self,
        subject: usize,
        group: &[usize],
        fps: &[VideoFingerprint],
    ) -> Option<f64> {
        let mut weakest: Option<f64> = None;
        for &other in group {
            if other == subject {
                continue;
            }
            let shared = self.shared_seconds(subject, other, fps)?;
            weakest = Some(weakest.map_or(shared, |w: f64| w.min(shared)));
        }
        weakest
    }
}

/// --- Phase 1: candidate generation ---------------------------------------
///
/// Probe the 4x16-bit multi-index and emit every ordered pair `(a, b)` with
/// `a < b` that shares at least one frame within `max_hamming_dist`. This is a
/// *filter*, not the answer: it decides which video pairs are worth a full
/// comparison, and nothing else.
///
/// The probe radius is derived from the tolerance rather than fixed (see
/// `probe_radius`), which makes the index exhaustive for every `max_hamming_dist`
/// up to `BLOCKS * MAX_PROBE_RADIUS + BLOCKS - 1` = 11: every pair within the
/// tolerance is proposed. Above that the radius stops widening and an individual
/// frame pair can be missed, but real duplicates share hundreds of frames, so
/// missing *every* one of them is vanishingly unlikely -- and phase 2 then
/// recovers the frames the probe skipped for any pair that was proposed at all.
///
/// The blocks are indexed ONE AT A TIME, and each index is dropped before the
/// next is built. Four live indices is four entries per stored hash; one is one.
/// The cost is that a pair found in several blocks is emitted several times,
/// which a sort and a dedup at the end of each pass settles -- candidate pairs
/// are orders of magnitude scarcer than the hashes they were found from.
fn candidate_pairs(
    fingerprints: &[VideoFingerprint],
    max_hamming_dist: u32,
) -> Vec<(usize, usize)> {
    let n = fingerprints.len();
    let radius = probe_radius(max_hamming_dist);
    let masks = probe_masks(radius);

    log::debug!(
        "Probing each block at radius {} ({} key(s) per block) for -d {}.",
        radius,
        masks.len(),
        max_hamming_dist
    );

    let mut candidates: Vec<(usize, usize)> = Vec::new();

    for k in 0..BLOCKS {
        if shutdown_requested() {
            return Vec::new();
        }

        let index = BlockIndex::build(k, fingerprints);

        let found: Vec<(usize, usize)> = (0..n)
            .into_par_iter()
            .flat_map(|v_a| {
                if shutdown_requested() {
                    return Vec::new();
                }
                let fp_a = &fingerprints[v_a];

                // Once a video is a known candidate in this pass, further hits
                // against it are pure waste -- `seen` lets us skip the popcount
                // entirely.
                let mut seen = vec![false; n];
                let mut local: Vec<(usize, usize)> = Vec::new();

                for &h_a in fp_a.valid_hashes.iter() {
                    if shutdown_requested() {
                        return Vec::new();
                    }
                    let key = block_of(h_a, k);

                    for &mask in masks.iter() {
                        let (hashes, videos) = index.bin(key ^ mask as usize);

                        // Entries are built in video order, so a binary search
                        // skips every already-processed video in one step.
                        let start = videos.partition_point(|&v| (v as usize) <= v_a);

                        for (&v_b, &h_b) in videos[start..].iter().zip(&hashes[start..]) {
                            let v_b = v_b as usize;
                            if seen[v_b] {
                                continue;
                            }
                            if (h_a ^ h_b).count_ones() <= max_hamming_dist {
                                seen[v_b] = true;
                                local.push((v_a, v_b));
                            }
                        }
                    }
                }

                local
            })
            .collect();

        candidates.extend(found);

        // Deduped per pass rather than once at the end, so the list never holds
        // more than one pass worth of repeats -- the same reason only one index
        // is alive at a time.
        candidates.sort_unstable();
        candidates.dedup();

        // `index` dies here, before the next block's is allocated.
    }

    candidates
}

/// --- Phase 2: exact verification ------------------------------------------
///
/// Brute-force every frame of A against every frame of B. Two ~400-hash videos
/// is ~160k XOR+popcount operations -- microseconds -- and since genuine
/// candidates are rare relative to the library size, this is close to free.
///
/// Returns the fraction of each video's runtime that is matched by the other.
/// Unlike the index-driven count it replaces, this sees *all* matching frames,
/// including ones the probe cannot reach at a tolerance above the radius cap.
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

/// Every pair of videos that clears both gates, with the coverage that got them
/// there.
///
/// Note what the `--match-percent` gate actually asks: whether EITHER file is
/// sufficiently covered by the other. That is deliberate, and it is what makes
/// clip detection possible -- a short cut from a long video only ever clears a
/// threshold from the clip's side. It also means neither individual coverage
/// figure can be read as "this pair scored above the threshold", which is why
/// the report speaks in seconds rather than percentages.
pub fn find_all_matches(
    fingerprints: &[VideoFingerprint],
    max_hamming_dist: u32,
    min_match_percent: f32,
    min_duration: f64,
) -> Vec<Match> {
    let candidates = candidate_pairs(fingerprints, max_hamming_dist);
    if shutdown_requested() {
        return Vec::new();
    }
    info!("Index scan produced {} candidate pair(s); verifying...", candidates.len());

    candidates
        .into_par_iter()
        .filter_map(|(v_a, v_b)| {
            if shutdown_requested() {
                return None;
            }
            let fp_a = &fingerprints[v_a];
            let fp_b = &fingerprints[v_b];

            let (pct_a, pct_b) = match_overlap(fp_a, fp_b, max_hamming_dist);

            if pct_a.max(pct_b) < min_match_percent {
                return None;
            }

            if min_duration > 0.0 {
                let matched_secs =
                    (pct_a as f64 * fp_a.duration).max(pct_b as f64 * fp_b.duration);
                if matched_secs < min_duration {
                    return None;
                }
            }

            Some(Match {
                a: v_a,
                b: v_b,
                coverage_a: pct_a,
                coverage_b: pct_b,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Matching is codec-blind by design -- a perceptual hash of a decoded frame
    // says nothing about what encoded it -- so the codec and frame rate here are
    // plausible filler and never affect a result in this module.
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
            codec: "h264".to_string(),
            frame_rate: 30.0,
        }
    }

    /// Only the runtime matters for the shared-duration arithmetic.
    fn mock_fp_lasting(seconds: f64) -> VideoFingerprint {
        let mut fp = mock_fp_with_hashes(vec![], 1);
        fp.duration = seconds;
        fp
    }

    /// Coverage is stored as f32, so a shared duration is a float that has been
    /// through a narrowing conversion: `0.40f32` is really 0.4000000059604645,
    /// and 40% of 100 seconds lands at 40.00000059604645. Asserting exact
    /// equality on that is asserting something about IEEE 754 rather than about
    /// this code. A millisecond is far tighter than any real precision concern
    /// and far looser than the round-trip error.
    fn assert_near(actual: Option<f64>, expected: f64) {
        let got = actual.expect("expected a measured overlap, got none");
        assert!(
            (got - expected).abs() < 1e-3,
            "expected about {} seconds, got {}",
            expected,
            got
        );
    }

    /// Most tests only care about which videos were linked, not by how much.
    fn pairs(matches: &[Match]) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = matches.iter().map(|m| (m.a, m.b)).collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn test_blocks_partition_the_hash_without_gaps_or_overlap() {
        let h = 0x0123_4567_89AB_CDEFu64;
        assert_eq!(block_of(h, 0), 0x0123);
        assert_eq!(block_of(h, 1), 0x4567);
        assert_eq!(block_of(h, 2), 0x89AB);
        assert_eq!(block_of(h, 3), 0xCDEF);
    }

    #[test]
    fn test_probe_radius_is_derived_from_the_tolerance() {
        // The pigeonhole rule, and the reason the default setting needs no
        // neighbour lookups at all: three differing bits cannot cover four
        // blocks, so one block always matches exactly.
        assert_eq!(probe_radius(0), 0);
        assert_eq!(probe_radius(3), 0, "the default tolerance probes exact bins only");
        assert_eq!(probe_radius(4), 1);
        assert_eq!(probe_radius(7), 1);
        assert_eq!(probe_radius(8), 2);
        assert_eq!(probe_radius(11), 2);
        assert_eq!(probe_radius(64), MAX_PROBE_RADIUS, "and it stops widening at the cap");
    }

    #[test]
    fn test_probe_masks_are_exactly_the_patterns_within_the_radius() {
        // 1, 17, 137: sum of C(16, i) up to the radius. These are the multiplier
        // on every single bin lookup in the scan, which is why the cap exists.
        assert_eq!(probe_masks(0), vec![0u16]);
        assert_eq!(probe_masks(1).len(), 17);
        assert_eq!(probe_masks(2).len(), 137);
        assert!(probe_masks(2).iter().all(|m| m.count_ones() <= 2));
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
        assert_eq!(pairs(&matches), vec![(0, 1)]);
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
        // Identical hashes hit in all four block passes, so this is also the
        // test that the per-pass dedup actually collapses them.
        let hash = 0xABCD_1234_ABCD_1234;
        let fps = vec![
            mock_fp_with_hashes(vec![hash], 1),
            mock_fp_with_hashes(vec![hash], 1),
            mock_fp_with_hashes(vec![hash], 1),
        ];

        let matches = find_all_matches(&fps, 0, 1.0, 0.0);

        // Each unordered pair exactly once, always with the lower index first.
        assert_eq!(pairs(&matches), vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn test_the_default_tolerance_is_exhaustive_with_no_bit_flips() {
        // Three differing bits spread across three blocks. Block 3 is untouched
        // -- it has to be -- so the exact-bin probe finds the pair even though
        // three quarters of the hash differs somewhere.
        let a = 0x0000_0000_0000_0000u64;
        let b = 0x0001_0001_0001_0000u64;
        assert_eq!((a ^ b).count_ones(), 3);
        assert_eq!(probe_radius(3), 0);

        let fps = vec![mock_fp_with_hashes(vec![a], 1), mock_fp_with_hashes(vec![b], 1)];

        assert_eq!(pairs(&find_all_matches(&fps, 3, 1.0, 0.0)), vec![(0, 1)]);
    }

    #[test]
    fn test_a_pair_with_no_identical_block_is_found_once_the_radius_widens() {
        // One differing bit in EVERY block: total distance 4, and nothing to
        // find in an exact bin. floor(4 / 4) = 1, so the probe widens to radius
        // 1 and the pair is proposed -- the pigeonhole guarantee, from the other
        // side.
        let a = 0x0000_0000_0000_0000u64;
        let b = 0x0001_0001_0001_0001u64;
        assert_eq!((a ^ b).count_ones(), 4);

        let fps = vec![mock_fp_with_hashes(vec![a], 1), mock_fp_with_hashes(vec![b], 1)];

        assert_eq!(pairs(&find_all_matches(&fps, 4, 1.0, 0.0)), vec![(0, 1)]);
        assert!(
            find_all_matches(&fps, 3, 1.0, 0.0).is_empty(),
            "and four bits apart is genuinely outside a three-bit tolerance"
        );
    }

    #[test]
    fn test_two_phase_recovers_frames_the_index_cannot_propose() {
        // Only reachable above the radius cap, which is the entire remaining
        // gap in the index. Frame 2 differs by 3 bits in EVERY block (total 12)
        // and the radius is capped at 2, so no probe can see it. Frame 1 is
        // identical, so the PAIR still becomes a candidate -- and phase 2's
        // brute force then counts frame 2 as well.
        let shared = 0x0000_0000_0000_0000u64;
        let unprobeable_a = 0xFFFF_FFFF_FFFF_FFFFu64;
        let unprobeable_b = unprobeable_a ^ 0x0007_0007_0007_0007;

        assert_eq!((unprobeable_a ^ unprobeable_b).count_ones(), 12);
        assert!(probe_radius(12) < 3, "no block is within reach of the probe");

        let fps = vec![
            mock_fp_with_hashes(vec![shared, unprobeable_a], 2),
            mock_fp_with_hashes(vec![shared, unprobeable_b], 2),
        ];

        // Demanding 100% overlap: only reachable if BOTH frames are counted.
        // Index-only accounting would have scored this pair at 50%.
        let matches = find_all_matches(&fps, 12, 1.0, 0.0);
        assert_eq!(
            pairs(&matches),
            vec![(0, 1)],
            "phase 2 must recover the frame the probe could not reach"
        );
    }

    #[test]
    fn test_min_duration_gates_independently_of_match_percent() {
        let hash = 0xABCD_1234_ABCD_1234u64;
        let fps = vec![
            mock_fp_with_hashes(vec![hash, hash], 2),
            mock_fp_with_hashes(vec![hash, hash], 2),
        ];

        // 100% overlap of a 10s mock = 10 matched seconds.
        assert_eq!(pairs(&find_all_matches(&fps, 0, 1.0, 5.0)), vec![(0, 1)], "10s clears a 5s floor");
        assert!(
            find_all_matches(&fps, 0, 1.0, 30.0).is_empty(),
            "a full-coverage match must still fail a 30s floor"
        );
    }

    #[test]
    fn test_coverage_is_directional_for_a_clip_inside_a_host() {
        // Four frames of host, one of which is the whole of the clip. The clip
        // is fully contained; the host is only a quarter accounted for. Both
        // numbers are correct and they must not be conflated.
        let shared = 0x0F0F_0F0F_0F0F_0F0Fu64;
        let host = mock_fp_with_hashes(
            vec![shared, 0x1111_2222_3333_4444, 0x5555_6666_7777_8888, 0x9999_AAAA_BBBB_CCCC],
            4,
        );
        let clip = mock_fp_with_hashes(vec![shared], 1);

        let matches = find_all_matches(&[host, clip], 0, 0.10, 0.0);

        assert_eq!(matches.len(), 1);
        let m = matches[0];
        assert_eq!((m.a, m.b), (0, 1));
        assert!((m.coverage_a - 0.25).abs() < 1e-6, "host is a quarter covered, got {}", m.coverage_a);
        assert!((m.coverage_b - 1.0).abs() < 1e-6, "clip is fully covered, got {}", m.coverage_b);
    }

    #[test]
    fn test_index_answers_both_directions_of_a_pair() {
        let idx = MatchIndex::new(vec![Match { a: 0, b: 1, coverage_a: 0.25, coverage_b: 1.0 }]);

        assert_eq!(idx.coverage(0, 1), Some(0.25));
        assert_eq!(idx.coverage(1, 0), Some(1.0));
        assert_eq!(idx.coverage(0, 2), None, "a pair that never matched has no figure");
    }

    #[test]
    fn test_shared_seconds_reads_the_same_from_either_end() {
        // The property the report now rests on. A 2 minute clip inside a 22
        // minute host: 100% of 120s and 9.09% of 1320s are both 120 seconds.
        // Coverage disagrees wildly between the two files; duration does not.
        let fps = vec![mock_fp_lasting(1320.0), mock_fp_lasting(120.0)];
        let idx = MatchIndex::new(vec![Match {
            a: 0,
            b: 1,
            coverage_a: 120.0 / 1320.0,
            coverage_b: 1.0,
        }]);

        let from_host = idx.shared_seconds(0, 1, &fps).unwrap();
        let from_clip = idx.shared_seconds(1, 0, &fps).unwrap();

        assert!((from_host - 120.0).abs() < 0.5, "got {}", from_host);
        assert!((from_clip - from_host).abs() < 1e-9, "argument order must not matter");
    }

    #[test]
    fn test_shared_seconds_takes_the_lower_estimate_when_the_sides_disagree() {
        // Keyframe density differs between the files, so the two estimates do
        // not land on the same value. The smaller is reported.
        let fps = vec![mock_fp_lasting(100.0), mock_fp_lasting(100.0)];
        let idx = MatchIndex::new(vec![Match {
            a: 0,
            b: 1,
            coverage_a: 0.50, // 50s
            coverage_b: 0.40, // 40s
        }]);

        assert_near(idx.shared_seconds(0, 1, &fps), 40.0);
    }

    #[test]
    fn test_one_shared_keyframe_is_a_fraction_of_a_second_not_a_big_percentage() {
        // The report that prompted this design: short clips sharing a single
        // keyframe each. As a percentage a lone keyframe in a 4-keyframe video
        // is a headline 25%; in time it is under a second, which is what a
        // reader actually needs to know before deleting anything.
        let fps = vec![mock_fp_lasting(3.0), mock_fp_lasting(9.0)];
        let idx = MatchIndex::new(vec![Match {
            a: 0,
            b: 1,
            coverage_a: 0.25,   // 1 of 4 keyframes
            coverage_b: 0.0714, // 1 of 14
        }]);

        let shared = idx.shared_seconds(0, 1, &fps).unwrap();
        assert!(shared < 1.0, "one keyframe is well under a second, got {}", shared);
    }

    #[test]
    fn test_an_unknown_runtime_falls_back_to_the_other_file() {
        // A container that never reported a duration cannot estimate, but the
        // other side still can, and one estimate beats reporting nothing.
        let fps = vec![mock_fp_lasting(0.0), mock_fp_lasting(60.0)];
        let idx = MatchIndex::new(vec![Match { a: 0, b: 1, coverage_a: 1.0, coverage_b: 0.5 }]);

        assert_eq!(idx.shared_seconds(0, 1, &fps), Some(30.0));
    }

    #[test]
    fn test_two_unknown_runtimes_report_unknown_rather_than_zero() {
        let fps = vec![mock_fp_lasting(0.0), mock_fp_lasting(0.0)];
        let idx = MatchIndex::new(vec![Match { a: 0, b: 1, coverage_a: 1.0, coverage_b: 1.0 }]);

        assert_eq!(idx.shared_seconds(0, 1, &fps), None);
    }

    #[test]
    fn test_weakest_shared_in_group_finds_the_thinnest_link() {
        // 0 and 1 are proper duplicates; 2 only brushes against both. Every
        // member's figure reflects its worst link, so nothing in the group can
        // look better supported than it is.
        let fps = vec![
            mock_fp_lasting(600.0),
            mock_fp_lasting(600.0),
            mock_fp_lasting(10.0),
        ];
        let idx = MatchIndex::new(vec![
            Match { a: 0, b: 1, coverage_a: 1.0, coverage_b: 1.0 },   // 600s
            Match { a: 0, b: 2, coverage_a: 0.005, coverage_b: 0.3 }, // 3s
            Match { a: 1, b: 2, coverage_a: 0.005, coverage_b: 0.3 }, // 3s
        ]);
        let group = [0, 1, 2];

        assert_near(idx.weakest_shared_in_group(0, &group, &fps), 3.0);
        assert_near(idx.weakest_shared_in_group(1, &group, &fps), 3.0);
        assert_near(idx.weakest_shared_in_group(2, &group, &fps), 3.0);
    }
}