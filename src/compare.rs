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
/// most `radius` bits set: 1, 17, 137, 697. Radius 1 is exhaustive up to a
/// tolerance of 7 and a very good filter above it; radius 2 costs 8x more
/// lookups on every hash in the library to chase pairs that share exactly one
/// marginal frame and nothing else.
///
/// The default tolerance of 4 sits inside that exhaustive range, so the cap
/// costs a default run nothing; it starts binding at `-d 8`, which the accuracy
/// ladder and any deliberately loose scan reach. It is safe there because phase
/// 1 only has to *propose* a pair: two encodes of the same footage agree closely
/// on many frames, not one, and phase 2 then measures all of them exactly.
/// Measured over a 1,000-file library the wider probe changed not a single
/// group.
const MAX_PROBE_RADIUS: u32 = 1;

/// The `k`th 16-bit block of a hash, most significant first.
#[inline(always)]
fn block_of(hash: u64, k: usize) -> usize {
    ((hash >> (64 - BLOCK_BITS * (k + 1))) & (BINS as u64 - 1)) as usize
}

/// How far around each block key the index must look to be exhaustive at
/// `max_hamming_dist`.
///
/// Integer division is the whole rule, and it is what makes a tight tolerance
/// cheap: below `-d 4` this is 0, because three differing bits cannot be spread
/// across four blocks without leaving one of them untouched, so probing the
/// exact bins alone finds every pair. The default tolerance of 4 is the first
/// value that needs a neighbour lookup, which costs 17 bins per block instead
/// of 1; the cap keeps it there however high `-d` goes.
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

/// Where in one file's own runtime its matched footage lies: the start of the
/// first sample that matched and the end of the last, in milliseconds.
///
/// This is an ENVELOPE, not a contiguous run. Matched samples can be scattered
/// -- two episodes sharing an opening and a closing theme match at both ends and
/// nowhere in between, and the envelope then spans the whole episode while only
/// a few seconds inside it actually matched. The envelope is therefore only
/// meaningful read next to the shared duration: when the two agree the match is
/// one continuous stretch, and when the envelope is much the wider of the two
/// the match is scattered through it.
///
/// Kept per side, because the same shared footage sits at different times in
/// each file -- that is the entire point of reporting it. A two-minute clip cut
/// from the middle of an episode spans 0..2min of itself and 14min..16min of the
/// episode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub start_ms: u32,
    pub end_ms: u32,
}

impl Span {
    /// The envelope of the samples flagged in `matched`, or `None` when none
    /// were. Relies on the sample times being ascending, which `fingerprint`
    /// guarantees.
    fn of(matched: &[bool], fp: &VideoFingerprint) -> Option<Span> {
        let first = matched.iter().position(|&m| m)?;
        let last = matched.iter().rposition(|&m| m)?;
        Some(Span {
            start_ms: fp.valid_t_start[first],
            end_ms: fp.valid_t_end[last],
        })
    }

    pub fn start_seconds(&self) -> f64 {
        self.start_ms as f64 / 1000.0
    }

    pub fn end_seconds(&self) -> f64 {
        self.end_ms as f64 / 1000.0
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
///
/// The spans are directional for the same reason and are never reconciled at
/// all: "where the shared footage sits" has two different right answers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Match {
    pub a: usize,
    pub b: usize,
    /// Fraction of `a`'s runtime that is also present in `b` (0.0 ..= 1.0).
    pub coverage_a: f32,
    /// Fraction of `b`'s runtime that is also present in `a` (0.0 ..= 1.0).
    pub coverage_b: f32,
    /// Where the matched footage lies in `a`'s own runtime.
    pub span_a: Option<Span>,
    /// Where the matched footage lies in `b`'s own runtime.
    pub span_b: Option<Span>,
}

impl Match {
    /// A pair whose overlap was measured but whose position in either file was
    /// not. Only tests build these; every real match carries its spans, because
    /// the same pass that measures the overlap already knows where it is.
    #[cfg(test)]
    pub fn new(a: usize, b: usize, coverage_a: f32, coverage_b: f32) -> Self {
        Match { a, b, coverage_a, coverage_b, span_a: None, span_b: None }
    }

    /// The same, positioned. Milliseconds, `(start, end)` per side.
    #[cfg(test)]
    pub fn with_spans(mut self, a: (u32, u32), b: (u32, u32)) -> Self {
        self.span_a = Some(Span { start_ms: a.0, end_ms: a.1 });
        self.span_b = Some(Span { start_ms: b.0, end_ms: b.1 });
        self
    }
}

/// One direction of one measured pair: how much of the subject the other file
/// contains, and where in the subject that footage sits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    pub coverage: f32,
    pub span: Option<Span>,
}

/// Match figures made lookup-friendly, keyed by (subject, other).
///
/// Both directions of every pair are stored, so `coverage(i, j)` always answers
/// "how much of i does j contain" without the caller having to know which way
/// round the pair was originally emitted.
///
/// `neighbours` is the same information as an adjacency list, and it exists
/// because the report asks a question the pair map answers badly: "everything
/// this file matched". Answering that by probing the map for every other member
/// of the group is quadratic in the group size, and a loose scan can produce one
/// component with hundreds of members -- while the number of edges inside it
/// stays proportional to the pairs actually measured, which is far smaller.
pub struct MatchIndex {
    links: HashMap<(usize, usize), Link>,
    neighbours: HashMap<usize, Vec<usize>>,
}

impl MatchIndex {
    pub fn new(matches: Vec<Match>) -> Self {
        let mut links = HashMap::with_capacity(matches.len() * 2);
        let mut neighbours: HashMap<usize, Vec<usize>> = HashMap::new();

        for m in matches {
            links.insert((m.a, m.b), Link { coverage: m.coverage_a, span: m.span_a });
            links.insert((m.b, m.a), Link { coverage: m.coverage_b, span: m.span_b });
            neighbours.entry(m.a).or_default().push(m.b);
            neighbours.entry(m.b).or_default().push(m.a);
        }

        Self { links, neighbours }
    }

    /// How much of `subject` is contained in `other`, if the two were compared.
    pub fn coverage(&self, subject: usize, other: usize) -> Option<f32> {
        self.links.get(&(subject, other)).map(|l| l.coverage)
    }

    /// Where the footage `subject` shares with `other` sits in `subject`'s own
    /// runtime, if the two were compared and anything matched.
    pub fn span(&self, subject: usize, other: usize) -> Option<Span> {
        self.links.get(&(subject, other))?.span
    }

    /// Seconds of content two files have in common -- the figure every report
    /// prints, and the one `--min-duration` gates on.
    ///
    /// The arithmetic and the reasoning behind it live in `overlap_seconds`,
    /// which the gate calls directly because it runs before any `MatchIndex`
    /// exists. This is that function reached by index: it looks the pair's two
    /// directional coverages up and hands them over. `None` when the pair was
    /// never compared, which is not the same as compared and sharing nothing.
    pub fn shared_seconds(&self, a: usize, b: usize, fps: &[VideoFingerprint]) -> Option<f64> {
        let cov_a = self.coverage(a, b)?;
        let cov_b = self.coverage(b, a)?;
        overlap_seconds(cov_a, fps[a].duration, cov_b, fps[b].duration)
    }

    /// Every measured link `subject` has, strongest first.
    ///
    /// No group has to be supplied, because every file `subject` matched is in
    /// `subject`'s group by construction: clustering unions each matched pair,
    /// so a neighbour and its subject always land in the same component. The
    /// group is the transitive closure of this list, never smaller than it.
    ///
    /// Pairs that were never measured are absent rather than present with a
    /// zero, because "never compared" and "compared and shares nothing" are
    /// different statements and only the second one is evidence about the files.
    /// That is why a chained group -- A-B and B-C measured, A-C never -- gives A
    /// one link rather than two.
    ///
    /// Ordered by shared duration descending, ties broken on path, so the
    /// report is reproducible run to run: the strongest link is `first()`, which
    /// is what the single-figure columns show, and the whole list is what the
    /// JSON carries so a three-file group can be read pair by pair.
    pub fn links_of(&self, subject: usize, fps: &[VideoFingerprint]) -> Vec<GroupLink> {
        let Some(others) = self.neighbours.get(&subject) else {
            return Vec::new();
        };

        let mut links: Vec<GroupLink> = others
            .iter()
            .filter_map(|&other| {
                Some(GroupLink {
                    other,
                    shared_seconds: self.shared_seconds(subject, other, fps)?,
                    span: self.span(subject, other),
                })
            })
            .collect();

        links.sort_by(|x, y| {
            y.shared_seconds
                .partial_cmp(&x.shared_seconds)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| fps[x.other].path.cmp(&fps[y.other].path))
        });

        links
    }

    /// The single link that best answers "is this file a copy of something
    /// here": the most content `subject` shares with anything it was actually
    /// matched against. `links_of(..).first()`, named.
    ///
    /// The strongest link rather than the weakest, because one solid match makes
    /// a file a duplicate however many incidental ones sit beside it. Taking the
    /// minimum instead let a single thin edge -- a shared title card, an episode
    /// that merely brushes the group -- rewrite the figure for files that are
    /// perfect duplicates of each other, so a 22-minute re-encode pulled into a
    /// group by a common intro reported the intro's two seconds and read as
    /// though nothing in the group was a duplicate of anything. The cost is that
    /// it speaks for one pair rather than all of them, which is exactly why the
    /// report names the file it is talking about.
    ///
    /// Printed beside the file's own length it makes a group interpretable at a
    /// glance -- a nine-second clip sharing eight seconds is a duplicate, and
    /// one sharing half a second is not, however impressive that half second
    /// looks as a percentage. `None` means nothing about this file's links could
    /// be measured at all.
    ///
    /// Nothing is deleted on the strength of this figure. It is reported, and
    /// the DELETE rule reads `coverage` directly -- see `export.rs`.
    #[cfg(test)]
    pub fn best_link_in_group(
        &self,
        subject: usize,
        _group: &[usize],
        fps: &[VideoFingerprint],
    ) -> Option<GroupLink> {
        self.links_of(subject, fps).into_iter().next()
    }
}

/// Seconds of footage one measured pair has in common.
///
/// Each side estimates it as its own coverage times its own runtime, and for
/// genuinely shared footage the two agree, because the shared segment has one
/// real duration no matter which file you measure it in: a clip's `100% x 2min`
/// and its host's `9% x 22min` are both two minutes.
///
/// That agreement is the whole point. Coverage is asymmetric, and its asymmetry
/// is what makes a report confusing; duration is symmetric and reads the same
/// from either end. Where the two estimates disagree -- different keyframe
/// densities, tolerance landing differently on each side -- the LOWER is taken,
/// the conservative reading for a tool that deletes things.
///
/// A file whose runtime the container never reported contributes no estimate.
/// If neither file has a known runtime the answer is `None`: the overlap is
/// unknown, which is not the same as zero.
///
/// A free function rather than a method because `--min-duration` gates on this
/// figure in `find_all_matches`, long before a `MatchIndex` exists, and the
/// report prints it afterwards. Those two used to compute it separately and
/// disagreed: the gate took the HIGHER of the two estimates while the report
/// took the lower, so `--min-duration 5` admitted -- and marked DELETE -- pairs
/// whose own reported overlap read 2.9s. One definition, one answer.
fn overlap_seconds(cov_a: f32, duration_a: f64, cov_b: f32, duration_b: f64) -> Option<f64> {
    let mut estimate: Option<f64> = None;
    for (coverage, runtime) in [(cov_a, duration_a), (cov_b, duration_b)] {
        if runtime <= 0.0 {
            continue;
        }
        let seconds = coverage as f64 * runtime;
        estimate = Some(estimate.map_or(seconds, |e: f64| e.min(seconds)));
    }
    estimate
}

/// One file's measured relationship with one other member of its group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupLink {
    /// Index into the fingerprint list of the file on the other end.
    pub other: usize,
    /// Seconds of footage the two have in common -- symmetric, so this reads
    /// the same from either end of the pair.
    pub shared_seconds: f64,
    /// Where that footage sits in the SUBJECT's runtime, not the other file's.
    pub span: Option<Span>,
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
/// up to `BLOCKS * MAX_PROBE_RADIUS + BLOCKS - 1` = 7: every pair within the
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

        // Once a video is a known candidate for THIS subject, further hits
        // against it are pure waste -- `seen` lets us skip the popcount
        // entirely. It is a scratch buffer belonging to the rayon worker rather
        // than to the subject, and it is stamped with the subject's own id
        // instead of being cleared: a fresh `vec![false; n]` per subject cost an
        // allocation and `n` bytes of memset every time round, so a pass was
        // O(n^2) in bookkeeping before it compared a single hash. That is
        // invisible at a thousand files and roughly 1.6e11 byte-writes at two
        // hundred thousand.
        //
        // 0 is "never touched by anybody", so the stamp is `v_a + 1` and cannot
        // collide with it.
        let found: Vec<(usize, usize)> = (0..n)
            .into_par_iter()
            .map_init(
                || vec![0u32; n],
                |seen, v_a| {
                    if shutdown_requested() {
                        return Vec::new();
                    }
                    let fp_a = &fingerprints[v_a];
                    let mark = v_a as u32 + 1;
                    let mut local: Vec<(usize, usize)> = Vec::new();

                    for &h_a in fp_a.valid_hashes.iter() {
                        if shutdown_requested() {
                            return Vec::new();
                        }
                        let key = block_of(h_a, k);

                        for &mask in masks.iter() {
                            let (hashes, videos) = index.bin(key ^ mask as usize);

                            // Entries are built in video order, so a binary
                            // search skips every already-processed video in one
                            // step.
                            let start = videos.partition_point(|&v| (v as usize) <= v_a);

                            for (&v_b, &h_b) in videos[start..].iter().zip(&hashes[start..]) {
                                let v_b = v_b as usize;
                                if seen[v_b] == mark {
                                    continue;
                                }
                                if (h_a ^ h_b).count_ones() <= max_hamming_dist {
                                    seen[v_b] = mark;
                                    local.push((v_a, v_b));
                                }
                            }
                        }
                    }

                    local
                },
            )
            .flatten()
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
/// Returns the fraction of each video's runtime that is matched by the other,
/// and where in each of them the matched footage lies. Unlike the index-driven
/// count it replaces, this sees *all* matching frames, including ones the probe
/// cannot reach at a tolerance above the radius cap.
///
/// The spans cost two linear scans of an array of bools this function has
/// already built and is still holding in cache, against the quadratic hash
/// comparison above them. They are free in the only sense that matters here:
/// nothing about which frames are examined, or which pairs survive, changes.
fn match_overlap(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    max_hamming_dist: u32,
) -> (f32, f32, Option<Span>, Option<Span>) {
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

    // Each stored hash stands in for the picture over [t_start, t_end), so the
    // matched footage is the sum of the spans of the hashes that matched --
    // milliseconds, not a count of samples. A sample that covers a ten-second
    // static shot is worth ten seconds of agreement; one covering half a second
    // of cuts is worth half a second. Two encodes of the same footage sample it
    // at completely different rates and still measure the same overlap.
    let covered_ms = |matched: &[bool], fp: &VideoFingerprint| -> u64 {
        matched
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| (fp.valid_t_end[i].saturating_sub(fp.valid_t_start[i])) as u64)
            .sum()
    };

    let pct = |ms: u64, total: u32| -> f32 {
        if total > 0 {
            (ms as f32 / total as f32).min(1.0)
        } else {
            0.0
        }
    };

    (
        pct(covered_ms(&matched_a, fp_a), fp_a.total_ms),
        pct(covered_ms(&matched_b, fp_b), fp_b.total_ms),
        Span::of(&matched_a, fp_a),
        Span::of(&matched_b, fp_b),
    )
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

            let (pct_a, pct_b, span_a, span_b) = match_overlap(fp_a, fp_b, max_hamming_dist);

            if pct_a.max(pct_b) < min_match_percent {
                return None;
            }

            if min_duration > 0.0 {
                // Measured exactly the way the report measures it -- see
                // `overlap_seconds`, which both sides now share. A pair whose
                // overlap cannot be measured at all (neither file reported a
                // runtime) cannot clear a floor stated in seconds, so `None`
                // fails the gate rather than passing it.
                let cleared = overlap_seconds(pct_a, fp_a.duration, pct_b, fp_b.duration)
                    .is_some_and(|secs| secs >= min_duration);
                if !cleared {
                    return None;
                }
            }

            Some(Match {
                a: v_a,
                b: v_b,
                coverage_a: pct_a,
                coverage_b: pct_b,
                span_a,
                span_b,
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
    /// `slots` is the runtime in milliseconds; each hash occupies one of them,
    /// so a video of `n` hashes and `n` slots is fully accounted for and each
    /// hash is worth exactly `1 / slots` of it.
    fn mock_fp_with_hashes(hashes: Vec<u64>, slots: u32) -> VideoFingerprint {
        let len = hashes.len();
        VideoFingerprint {
            path: "mock.mp4".to_string(),
            valid_hashes: hashes,
            // Mock time intervals directly correlating to the index
            valid_t_start: (0..len as u32).collect(),
            valid_t_end: (1..=len as u32).collect(),
            total_ms: slots,
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

    /// A video sampled at a fixed interval: hash `i` stands for the picture over
    /// `[i * step_ms, (i + 1) * step_ms)`. Real keyframes are not evenly spaced,
    /// but a span test needs to know exactly which millisecond each sample
    /// claims, and nothing here depends on the spacing being irregular.
    fn mock_fp_sampled(hashes: Vec<u64>, step_ms: u32) -> VideoFingerprint {
        let len = hashes.len() as u32;
        let mut fp = mock_fp_with_hashes(hashes, len * step_ms);
        fp.valid_t_start = (0..len).map(|i| i * step_ms).collect();
        fp.valid_t_end = (1..=len).map(|i| i * step_ms).collect();
        fp.duration = (len * step_ms) as f64 / 1000.0;
        fp
    }

    /// Distinct hashes far enough apart that only an exact pairing matches at a
    /// tolerance of 0, so a test can say precisely which samples are shared.
    fn distinct_hash(i: u64) -> u64 {
        i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x0F0F_0F0F_0F0F_0F0F
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
        // The pigeonhole rule: three differing bits cannot cover four blocks, so
        // one block always matches exactly and no neighbour lookup is needed.
        assert_eq!(probe_radius(0), 0);
        assert_eq!(probe_radius(3), 0, "a tight tolerance probes exact bins only");
        assert_eq!(probe_radius(4), 1);
        assert_eq!(probe_radius(7), 1);
        // Past 7 the rule would ask for radius 2, and the cap declines: the
        // index becomes a filter rather than an enumerator, which is what phase
        // 2 exists to make safe.
        assert_eq!(probe_radius(8), MAX_PROBE_RADIUS);
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
        // and the radius is capped at 1, so no probe can see it. Frame 1 is
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
    fn test_min_duration_gates_on_the_figure_the_report_will_print() {
        // The gate used to take the HIGHER of the two directional estimates
        // while `shared_seconds` printed the lower, so `--min-duration 5`
        // admitted -- and marked DELETE -- pairs whose own reported overlap read
        // well under five seconds. On the 727-file corpus that was seven rows,
        // the thinnest reporting 0.50s against a 5s floor.
        //
        // Two files of the same runtime whose coverages disagree: 60% of 10s
        // from one side, 20% from the other. The honest overlap is 2s, and a 3s
        // floor has to reject it however impressive the other side looks.
        let fps = vec![mock_fp_lasting(10.0), mock_fp_lasting(10.0)];
        let idx = MatchIndex::new(vec![Match::new(0, 1, 0.6, 0.2)]);

        assert_near(idx.shared_seconds(0, 1, &fps), 2.0);

        // And the gate agrees, because both now read `overlap_seconds`.
        assert_near(overlap_seconds(0.6, 10.0, 0.2, 10.0), 2.0);
        assert!(
            overlap_seconds(0.6, 10.0, 0.2, 10.0).is_some_and(|s| s < 3.0),
            "a 3s floor must reject a 2s overlap, whatever the louder side claims"
        );
    }

    #[test]
    fn test_a_pair_with_no_measurable_runtime_cannot_clear_a_seconds_floor() {
        // `--min-duration` is stated in seconds, so a pair whose overlap cannot
        // be expressed in seconds at all has not cleared it. `None` fails the
        // gate rather than passing it.
        assert_eq!(overlap_seconds(1.0, 0.0, 1.0, 0.0), None);

        let hash = 0xABCD_1234_ABCD_1234u64;
        let mut a = mock_fp_with_hashes(vec![hash, hash], 2);
        let mut b = mock_fp_with_hashes(vec![hash, hash], 2);
        a.duration = 0.0;
        b.duration = 0.0;

        assert!(
            find_all_matches(&[a, b], 0, 1.0, 5.0).is_empty(),
            "an unmeasurable overlap must not satisfy a floor stated in seconds"
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
        let idx = MatchIndex::new(vec![Match::new(0, 1, 0.25, 1.0)]);

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
        let idx = MatchIndex::new(vec![Match::new(0, 1, 120.0 / 1320.0, 1.0)]);

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
        let idx = MatchIndex::new(vec![Match::new(0, 1, 0.50, 0.40)]);

        assert_near(idx.shared_seconds(0, 1, &fps), 40.0);
    }

    #[test]
    fn test_one_shared_keyframe_is_a_fraction_of_a_second_not_a_big_percentage() {
        // The report that prompted this design: short clips sharing a single
        // keyframe each. As a percentage a lone keyframe in a 4-keyframe video
        // is a headline 25%; in time it is under a second, which is what a
        // reader actually needs to know before deleting anything.
        let fps = vec![mock_fp_lasting(3.0), mock_fp_lasting(9.0)];
        let idx = MatchIndex::new(vec![Match::new(0, 1, 0.25, 0.0714)]);

        let shared = idx.shared_seconds(0, 1, &fps).unwrap();
        assert!(shared < 1.0, "one keyframe is well under a second, got {}", shared);
    }

    #[test]
    fn test_an_unknown_runtime_falls_back_to_the_other_file() {
        // A container that never reported a duration cannot estimate, but the
        // other side still can, and one estimate beats reporting nothing.
        let fps = vec![mock_fp_lasting(0.0), mock_fp_lasting(60.0)];
        let idx = MatchIndex::new(vec![Match::new(0, 1, 1.0, 0.5)]);

        assert_eq!(idx.shared_seconds(0, 1, &fps), Some(30.0));
    }

    #[test]
    fn test_two_unknown_runtimes_report_unknown_rather_than_zero() {
        let fps = vec![mock_fp_lasting(0.0), mock_fp_lasting(0.0)];
        let idx = MatchIndex::new(vec![Match::new(0, 1, 1.0, 1.0)]);

        assert_eq!(idx.shared_seconds(0, 1, &fps), None);
    }

    #[test]
    fn test_strongest_shared_in_group_finds_the_best_link() {
        // 0 and 1 are proper duplicates; 2 only brushes against both. 0 and 1
        // must report the ten minutes they share with each other, not the three
        // seconds the interloper contributes -- a thin edge into the group says
        // nothing about how well two other members match.
        let fps = vec![
            mock_fp_lasting(600.0),
            mock_fp_lasting(600.0),
            mock_fp_lasting(10.0),
        ];
        let idx = MatchIndex::new(vec![
            Match::new(0, 1, 1.0, 1.0),   // 600s
            Match::new(0, 2, 0.005, 0.3), // 3s
            Match::new(1, 2, 0.005, 0.3), // 3s
        ]);
        let group = [0, 1, 2];

        assert_near(idx.best_link_in_group(0, &group, &fps).map(|l| l.shared_seconds), 600.0);
        assert_near(idx.best_link_in_group(1, &group, &fps).map(|l| l.shared_seconds), 600.0);
        // 2 has nothing better than its three seconds, and still reports them.
        assert_near(idx.best_link_in_group(2, &group, &fps).map(|l| l.shared_seconds), 3.0);
    }

    #[test]
    fn test_a_pair_that_was_never_compared_is_skipped_rather_than_erasing_the_figure() {
        // A chain: 0-1 and 1-2 matched, 0 and 2 never did. They share a group
        // because groups are connected components, so 0's row must report what
        // it shares with 1 -- the link that put it there. Treating the absent
        // 0-2 pair as unknown would blank the column for most of a chained
        // group, and treating it as zero would claim a comparison nobody made.
        let fps = vec![
            mock_fp_lasting(600.0),
            mock_fp_lasting(600.0),
            mock_fp_lasting(600.0),
        ];
        let idx = MatchIndex::new(vec![
            Match::new(0, 1, 1.0, 1.0), // 600s
            Match::new(1, 2, 0.5, 0.5), // 300s
        ]);
        let group = [0, 1, 2];

        assert_near(idx.best_link_in_group(0, &group, &fps).map(|l| l.shared_seconds), 600.0);
        assert_near(idx.best_link_in_group(1, &group, &fps).map(|l| l.shared_seconds), 600.0);
        assert_near(idx.best_link_in_group(2, &group, &fps).map(|l| l.shared_seconds), 300.0);
    }

    #[test]
    fn test_a_file_with_no_measurable_link_reports_unknown() {
        // Not the same as sharing nothing: the figure was never obtained.
        let fps = vec![mock_fp_lasting(600.0), mock_fp_lasting(600.0)];
        let idx = MatchIndex::new(vec![]);

        assert_eq!(idx.best_link_in_group(0, &[0, 1], &fps).map(|l| l.shared_seconds), None);
    }

    #[test]
    fn test_the_span_locates_a_clip_inside_its_host() {
        // The headline case for the feature: a 3s clip cut from the middle of a
        // 10s host. The clip is all of itself, so its own span is its whole
        // runtime; the host's span is where the clip sits INSIDE it, which is
        // the number that could not be read off any other column.
        let host = mock_fp_sampled((0..10).map(distinct_hash).collect(), 1000);
        let clip = mock_fp_sampled((4..7).map(distinct_hash).collect(), 1000);

        let (_, _, span_clip, span_host) = match_overlap(&clip, &host, 0);

        assert_eq!(span_clip, Some(Span { start_ms: 0, end_ms: 3000 }));
        assert_eq!(
            span_host,
            Some(Span { start_ms: 4000, end_ms: 7000 }),
            "the host's span must point at the clip's position in the host, not at the clip"
        );
    }

    #[test]
    fn test_the_span_is_an_envelope_and_says_so_next_to_the_shared_duration() {
        // Two episodes sharing an opening and a closing theme and nothing in
        // between. The envelope covers the whole runtime while only two of the
        // ten seconds actually matched -- which is precisely why the report
        // carries both figures and why the envelope alone must not be read as
        // "this much footage is shared".
        let a = mock_fp_sampled((0..10).map(distinct_hash).collect(), 1000);
        let b = mock_fp_sampled(vec![distinct_hash(0), distinct_hash(9)], 1000);

        let (coverage_a, _, span_a, _) = match_overlap(&a, &b, 0);

        assert_eq!(span_a, Some(Span { start_ms: 0, end_ms: 10_000 }));
        // Two samples of ten, i.e. two seconds of the ten the envelope spans.
        assert!(
            (coverage_a - 0.2).abs() < 1e-6,
            "expected 20% shared inside a 100% envelope, got {}",
            coverage_a
        );
    }

    #[test]
    fn test_spans_survive_the_trip_through_find_all_matches() {
        // The unit above tests the measurement; this tests that it reaches the
        // struct the report reads, through the phase-1 index and both gates.
        let host = mock_fp_sampled((0..10).map(distinct_hash).collect(), 1000);
        let clip = mock_fp_sampled((4..7).map(distinct_hash).collect(), 1000);
        let fps = vec![host, clip];

        let matches = find_all_matches(&fps, 0, 0.2, 0.0);
        assert_eq!(matches.len(), 1, "the clip should match its host");

        let m = matches[0];
        assert_eq!((m.a, m.b), (0, 1));
        assert_eq!(m.span_a, Some(Span { start_ms: 4000, end_ms: 7000 }), "position in the host");
        assert_eq!(m.span_b, Some(Span { start_ms: 0, end_ms: 3000 }), "position in the clip");

        // And that the index hands each side back its OWN span rather than the
        // pair's, which is the one way this could be wired up backwards.
        let idx = MatchIndex::new(matches);
        assert_eq!(idx.span(0, 1), Some(Span { start_ms: 4000, end_ms: 7000 }));
        assert_eq!(idx.span(1, 0), Some(Span { start_ms: 0, end_ms: 3000 }));
    }

    #[test]
    fn test_links_in_group_are_ordered_strongest_first_and_name_the_other_file() {
        // 0 is a full copy of 1 and brushes 2. Its row speaks for the link with
        // 1, and the list behind it still records the weaker link with 2 -- the
        // whole point of carrying every link into the JSON.
        let mut fps = vec![
            mock_fp_lasting(600.0),
            mock_fp_lasting(600.0),
            mock_fp_lasting(600.0),
        ];
        fps[0].path = "/a.mp4".into();
        fps[1].path = "/b.mp4".into();
        fps[2].path = "/c.mp4".into();

        let idx = MatchIndex::new(vec![
            Match::new(0, 1, 1.0, 1.0),   // 600s
            Match::new(0, 2, 0.01, 0.01), // 6s
        ]);
        let links = idx.links_of(0, &fps);

        assert_eq!(links.len(), 2);
        assert_eq!(links[0].other, 1);
        assert_near(Some(links[0].shared_seconds), 600.0);
        assert_eq!(links[1].other, 2);
        assert_near(Some(links[1].shared_seconds), 6.0);
    }

    #[test]
    fn test_equally_strong_links_are_ordered_by_path_so_the_report_is_reproducible() {
        // Two identical-strength links. Without a tiebreak the winner would be
        // whatever order the group happened to arrive in, and the "shared_with"
        // column would change between runs over the same library.
        let mut fps = vec![
            mock_fp_lasting(600.0),
            mock_fp_lasting(600.0),
            mock_fp_lasting(600.0),
        ];
        fps[0].path = "/subject.mp4".into();
        fps[1].path = "/zebra.mp4".into();
        fps[2].path = "/aardvark.mp4".into();

        // The neighbour list is built in the order the matches arrive, and that
        // order is phase 2's, not anything a reader controls. Both arrival
        // orders of the same tie must produce the same report.
        for matches in [
            vec![Match::new(0, 1, 1.0, 1.0), Match::new(0, 2, 1.0, 1.0)],
            vec![Match::new(0, 2, 1.0, 1.0), Match::new(0, 1, 1.0, 1.0)],
        ] {
            let links = MatchIndex::new(matches).links_of(0, &fps);
            assert_eq!(
                fps[links[0].other].path, "/aardvark.mp4",
                "an exact tie should fall to the alphabetically first path"
            );
        }
    }

    #[test]
    fn test_a_link_with_no_matched_sample_reports_no_span() {
        // `-p 0` accepts a pair that shares nothing at all. There is no position
        // to report, and a zero-length span at 0 would look like a real match at
        // the start of the file.
        let a = mock_fp_sampled(vec![distinct_hash(1)], 1000);
        let b = mock_fp_sampled(vec![distinct_hash(2)], 1000);

        let (_, _, span_a, span_b) = match_overlap(&a, &b, 0);

        assert_eq!(span_a, None);
        assert_eq!(span_b, None);
    }
}