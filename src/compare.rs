use crate::fingerprint::{Reframing, VideoFingerprint, REFRAMINGS};
use crate::utils::shutdown_requested;
use crate::progress::{self, Batch, Counter};
use log::info;
use rayon::prelude::*;
use std::collections::HashMap;

/// Width of a frame hash, and therefore the largest Hamming distance any two of
/// them can be apart.
pub const HASH_BITS: u32 = 64;

/// The largest `--hamming-distance` `run` accepts: the mean distance between two
/// UNRELATED frame hashes, i.e. chance itself.
///
/// Stated as zero sigma below chance rather than as `HASH_BITS / 2`, because
/// that is what it means -- see `sigma()`. At this tolerance half of all
/// unrelated frame pairs already match, so it is far past the useful range and
/// exactly where "tolerant" turns into "indiscriminate". Above it there is
/// nothing left to control: every file matches every other, the report is one
/// group holding the library, and with `--delete` armed that is the whole
/// library minus one file marked DELETE.
pub fn max_hamming_distance() -> u32 {
    sigma_below_chance(0.0)
}

/// The hash is split into this many equal blocks for indexing.
///
/// This is the number the whole probe strategy is derived from: two hashes
/// within total distance `d` must, by pigeonhole, agree to within `d / BLOCKS`
/// bits in at least ONE block, because if every block were further apart than
/// that the total would exceed `d`.
const BLOCKS: usize = 4;
const BLOCK_BITS: usize = HASH_BITS as usize / BLOCKS;
const BINS: usize = 1 << BLOCK_BITS;

/// What one bin probe costs, in units of "one hash comparison".
///
/// A probe loads two offsets, binary-searches past the videos already handled
/// and builds a slice before it looks at a single hash; a comparison is an XOR,
/// a popcount and a branch. Four is the ratio the measurements in
/// `index_is_cheaper` imply, and nothing is sensitive to it being exactly right:
/// the two routes it chooses between sit within a factor of two of each other on
/// either side of the crossover.
const PROBE_COST: usize = 4;

/// What handing one pair to phase 2 costs before it compares anything: two
/// `vec![false; _]` allocations and the loop setup, in the same units.
///
/// Only ever decisive for a library of many very short files, where there are
/// too few hashes for the comparison itself to outweigh the bookkeeping.
const PAIR_COST: usize = 8;

/// One standard deviation of the distance between two UNRELATED frame hashes.
///
/// That distance is Binomial(`HASH_BITS`, 1/2): mean `HASH_BITS / 2`, standard
/// deviation `sqrt(HASH_BITS) / 2` -- 32 +/- 4 bits for the 64-bit hash. Every
/// constant below is expressed as a multiple of it rather than in bits, because
/// a bit count is only meaningful against the width of the hash it came from:
/// "6 bits" says nothing on its own, "1.5 sigma below chance" says the same
/// thing about a 64-bit hash and a 256-bit one. Widen the hash and these
/// rescale themselves; the tuning that would otherwise be silently wrong is the
/// tuning that never has to be redone.
fn sigma() -> f64 {
    sigma_of(HASH_BITS)
}

/// `sigma` for a distance taken over `width` bits.
fn sigma_of(width: u32) -> f64 {
    (width as f64).sqrt() / 2.0
}

/// The width of the combined distance: the main hash and the detail hash
/// (`VideoFingerprint::detail_hashes`) read together, 64 bits each.
const WIDE_BITS: u32 = 2 * HASH_BITS;

/// How much looser than `-d` a corroborated frame match may be, in sigma.
///
/// 1.5 sigma is 6 bits on a 64-bit hash. **This is added to `-d`, never capped
/// against a constant**, which is the whole point: the flag has to stay a
/// sensitivity control across its range. An earlier version clamped the
/// corroborated side at 12 bits, and the effect was that `-d 4` through `-d 12`
/// all produced within twenty pairs of each other on the local corpus -- five
/// rungs of a knob doing nothing, with the flat range positioned by whichever
/// corpus it was fitted to.
///
/// Measured against the alternatives over the hand-labeled pairs, scoring the
/// whole ladder rather than one setting. A gap of 6 beats the flat single
/// threshold at every rung (F1 0.762 -> 0.859 at the default `-d 4`, 0.827 ->
/// 0.909 at `-d 6`, 0.887 -> 0.931 at `-d 8`). Pulling the strict side down to
/// `-d - 2` and narrowing the gap to 4 scores WORSE than a flat threshold below
/// `-d 10` -- the lone matches it gives up cost more than the narrow
/// corroborated window wins back. Widening to 8 buys recall at the default and
/// gives it back from `-d 10` up, where the strict side is what starts letting
/// false positives through.
const CORROBORATION_SLACK_SIGMA: f64 = 1.5;

/// The distance at which one witness is exactly enough, in sigma below chance.
///
/// 5 sigma is 12 bits on a 64-bit hash. It anchors the witness schedule below
/// and **caps nothing**: a match further out than this is not refused, it is
/// asked for more agreement.
const EVIDENCE_ANCHOR_SIGMA: f64 = 5.0;

/// How much evidence a corroborated cluster of frame matches has to carry,
/// counted in multiples of what ONE match at `EVIDENCE_ANCHOR_SIGMA` carries.
///
/// Two is the value that makes the anchor come out at exactly one witness, so
/// this and `EVIDENCE_ANCHOR_SIGMA` are one calibration between them, not two.
/// What they buy is the loose end: against a flat "one witness will do", `-d 10`
/// goes from 88.1% precision to 93.2% and `-d 12` from 78.4% to 85.9%, F1 rising
/// at both.
const CORROBORATION_BUDGET: f64 = 2.0;

/// A distance expressed as sigma below the mean of the unrelated-pair
/// distribution, rounded to a whole number of bits.
fn sigma_below_chance(sigmas: f64) -> u32 {
    sigma_below_chance_of(HASH_BITS, sigmas)
}

fn sigma_below_chance_of(width: u32, sigmas: f64) -> u32 {
    let mean = width as f64 / 2.0;
    (mean - sigmas * sigma_of(width)).round().max(0.0) as u32
}

/// How improbable a frame match at `distance` is by chance, as a base-10
/// order of magnitude.
///
/// Two unrelated 64-bit frame hashes differ in Binomial(64, 1/2) places, so the
/// chance of landing within `distance` of each other spans four orders of
/// magnitude across the range `-d` covers: 3.7e-11 at 8 bits, 2.0e-8 at 12,
/// 3.5e-6 at 16, 1.7e-4 at 20. One witness is not the same evidence at 20 bits
/// that it is at 12, and this is the function that says so.
///
/// Computed rather than tabulated because `HASH_BITS` is the only input and a
/// table would have to be re-derived by hand if the hash ever widened. It runs
/// once per scan.
#[cfg(test)]
fn chance_orders_of_magnitude(distance: u32) -> f64 {
    chance_orders_of_magnitude_of(HASH_BITS, distance)
}

fn chance_orders_of_magnitude_of(width: u32, distance: u32) -> f64 {
    let mut term = 1.0f64; // C(width, 0)
    let mut sum = 1.0f64;
    for i in 1..=distance.min(width) {
        term *= (width - i + 1) as f64 / i as f64;
        sum += term;
    }
    width as f64 * std::f64::consts::LOG10_2 - sum.log10()
}

/// How many witnesses a frame match at each distance needs, indexed by that
/// distance.
///
/// A cluster of `m` matches that agree on one time offset is roughly `m`
/// independent coincidences, so its evidence is `m` times one match's. Solving
/// for the smallest `m` that carries `CORROBORATION_BUDGET` gives the schedule:
/// one witness out to 12 bits, two at 14, three at 16, four at 20. The
/// independence is an approximation -- near-static footage produces matches that
/// are anything but independent, which is exactly why a witness must be a
/// different sample on both sides -- so the budget is calibrated rather than
/// derived, and only the SHAPE comes from the arithmetic.
///
/// `u32::MAX` where no attainable cluster could carry the budget, which is every
/// distance past ~30 bits: unrelated frames sit around 32 bits apart, so a match
/// out there is not evidence of anything at any cluster size.
fn witness_schedule() -> [u32; HASH_BITS as usize + 1] {
    let mut schedule = [u32::MAX; HASH_BITS as usize + 1];
    schedule.copy_from_slice(&witness_schedule_of(HASH_BITS));
    schedule
}

/// `witness_schedule` for a distance taken over `width` bits: the same anchor
/// and budget, in sigma, so a witness count stands for the same improbability
/// at either width.
fn witness_schedule_of(width: u32) -> Vec<u32> {
    let anchor = sigma_below_chance_of(width, EVIDENCE_ANCHOR_SIGMA);
    let budget = CORROBORATION_BUDGET * chance_orders_of_magnitude_of(width, anchor);
    let mut schedule = vec![u32::MAX; width as usize + 1];
    for (distance, needed) in schedule.iter_mut().enumerate() {
        let each = chance_orders_of_magnitude_of(width, distance as u32);
        // Multiply rather than divide: at exactly the anchor distance the two
        // sides are equal, and a division would land on 2.0000000001 as often
        // as on 2.0 and quietly demand a second witness there.
        *needed = (2..=HASH_BITS)
            .find(|m| *m as f64 * each >= budget)
            .map(|m| m - 1)
            .unwrap_or(u32::MAX);
    }
    schedule
}

/// The most witnesses any distance can be asked for.
///
/// `witness_schedule` searches cluster sizes `2..=HASH_BITS`, so the largest
/// number of witnesses it can return is `HASH_BITS - 1`; every distance past
/// that is `u32::MAX` and refused outright. In practice the schedule runs out
/// of attainable cluster sizes long before that -- 51 witnesses at 32 bits is
/// the largest finite entry -- and this is only ever read to size the search
/// `Witnesses::enough` is allowed to do.
const MAX_WITNESSES: usize = HASH_BITS as usize;

/// How far apart two frame matches may place the videos and still count as the
/// same alignment, in milliseconds.
///
/// Well under a keyframe interval on any real encode, so two matches landing in
/// the same window are describing the same instant rather than two instants that
/// happen to be nearby. Measured against 200, 800 and 1500: 200 and 800 score
/// identically to this, and 1500 starts admitting scattered coincidences.
const ALIGNMENT_TOLERANCE_MS: i64 = 500;

/// The most frame matches the corroboration rule will hold for one pair at once.
///
/// The list of loose matches is the only allocation phase 2 makes that is not
/// bounded by the number of SAMPLES, and it has no natural ceiling: it holds one
/// 24-byte entry per matching frame PAIR, so it is `|A| x |B|` for two videos
/// whose every sample matches every sample of the other. That is not a
/// contrived input -- it is what a static camera produces. Two 2-hour
/// recordings of one lecture, sampled at their keyframes, are 4,000 samples
/// each and 16 million matches: **372 MB for one pair**, on a stage that runs
/// one pair per rayon worker, so eight cores of it is 3 GB. The whole rest of
/// this program is written to keep memory flat (see the frame buffer and the
/// `mallopt` calls in `main`), and this was the one place a long, repetitive
/// library could undo that.
///
/// A ceiling alone would have to throw matches away, which is a wrong answer
/// rather than a refused one. Instead it is a WORKING SET: past this many, the
/// pair is corroborated in bands of the time-offset axis, and only the entries
/// in one band at a time are held. Corroboration is local in that axis -- a
/// witness has to sit within `ALIGNMENT_TOLERANCE_MS` of the match it supports
/// -- so a band that carries a skirt of that width on each side sees every
/// witness the whole-pair list would have offered, and the result is identical.
/// What it costs is one more sweep of the quadratic compare per band, on
/// exactly the pairs that would otherwise have been the most expensive thing in
/// the program.
///
/// A million entries is ~25 MB, and it is far above anything an ordinary
/// library reaches: this list cost the whole local corpus +0.6 MB of peak RSS
/// at the default `-d 4` and +2.4 MB at `-d 24` -- i.e. its widest pair holds
/// perhaps a hundred thousand matches. Nothing measured here bands, and the
/// corpus reports are byte-identical either side of the change at `-d 4` and
/// `-d 18`. Measured on the one pair that does: 4,000 static samples a side is
/// 16,000,000 entries held before and 1,051,365 after (the cap plus one band's
/// skirt), for 2.5% more time in that pair's comparison and the same verdict on
/// every sample.
const MAX_ALIGNED_MATCHES: usize = 1 << 20;

/// A frame match kept for the corroboration pass: the time offset it implies,
/// the sample it pairs on each side, and how far apart the two hashes were.
type Aligned = (i64, u32, u32, u32);

/// The band of time offsets `corroborated` may mark matches in, when it is
/// being handed one band of a pair at a time. Half-open, and this is the whole
/// axis.
const EVERY_OFFSET: (i64, i64) = (i64::MIN, i64::MAX);

/// The two Hamming distances one `-d` implies.
///
/// `strict` is what a frame match needs on its own; `loose` is what it needs
/// when another match agrees with it about the time offset between the two
/// videos. `strict <= loose` always, and both move monotonically with `-d`, so
/// raising the tolerance can only ever admit more -- a property the rule that
/// reads them has to preserve as well, which is what `Witnesses` is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tolerance {
    strict: u32,
    loose: u32,
    /// `Some` for a tolerance over `WIDE_BITS`: the distance is the main hash's
    /// plus the detail hash's, and the main hash alone has to be within this
    /// many bits -- what phase 1's index, which holds main hashes only, can be
    /// exhaustive to.
    cap: Option<u32>,
}

impl Tolerance {
    fn for_distance(max_hamming_dist: u32) -> Self {
        // `-d` IS the strict tolerance -- a lone frame match within it counts,
        // exactly as it did before corroboration existed -- and the corroborated
        // side sits a fixed distance above it. Nothing clamps either against a
        // constant, so every rung of the flag moves both, which is what keeps it
        // a sensitivity control rather than a switch between two fitted regimes.
        let slack = (CORROBORATION_SLACK_SIGMA * sigma()).round() as u32;
        Tolerance {
            strict: max_hamming_dist,
            // Saturating, then clamped to the hash width: nothing here should
            // depend on `run` having already refused everything above it, and a
            // tolerance wider than the hash accepts every pair anyway.
            loose: max_hamming_dist.saturating_add(slack).min(HASH_BITS),
            cap: None,
        }
    }

    /// The tolerance `-d` stands for, over the main hash and the detail hash
    /// read together (`WIDE_BITS`).
    ///
    /// `-d` keeps its meaning in sigma: `-d 4` is 7 sigma below chance on 64
    /// bits, so the strict side here is 7 sigma below chance on 128, and the
    /// corroborated side the same 1.5 sigma above it. Read that way, the pair of
    /// hashes is a sharper test of the same claim: two frames whose coarse
    /// layout agrees by accident -- two lines of text on black, two road scenes
    /// -- disagree in the detail half and stay far apart, while two encodes of
    /// one frame agree in both.
    ///
    /// The main hash is held to half the corroborated distance on its own,
    /// which is what makes the index exhaustive for this rule.
    fn wide(max_hamming_dist: u32) -> Self {
        let below = (HASH_BITS as f64 / 2.0 - max_hamming_dist as f64) / sigma();
        let strict = sigma_below_chance_of(WIDE_BITS, below);
        let slack = (CORROBORATION_SLACK_SIGMA * sigma_of(WIDE_BITS)).round() as u32;
        let loose = strict.saturating_add(slack).min(WIDE_BITS);
        Tolerance { strict, loose, cap: Some((loose / 2).min(HASH_BITS)) }
    }

    /// The widest distance any frame match can be admitted at -- what phase 1
    /// has to be exhaustive to, and what the probe radius is derived from.
    fn widest(&self) -> u32 {
        self.cap.unwrap_or(self.loose)
    }
}

/// What one pair is judged by: the two tolerances `-d` implies and how many
/// witnesses each distance has to carry.
///
/// One argument rather than two because they are one calibration -- both are
/// derived from the hash width in sigma, and a frame match is read against both
/// of them in the same breath. Copied freely; the schedule is a borrow of the
/// single table `find_all_matches` builds for the whole run.
#[derive(Clone, Copy)]
struct Rule<'a> {
    tol: Tolerance,
    schedule: &'a [u32],
}

/// The two rules a run judges pairs by: over the main hash alone, and over the
/// main and detail hashes together. A pair is read the wide way whenever both
/// fingerprints carry detail hashes, which every decoded fingerprint does.
#[derive(Clone, Copy)]
struct Rules<'a> {
    narrow: Rule<'a>,
    wide: Rule<'a>,
}

impl<'a> Rules<'a> {
    fn for_pair(&self, fp_a: &VideoFingerprint, fp_b: &VideoFingerprint) -> Rule<'a> {
        if has_detail(fp_a) && has_detail(fp_b) { self.wide } else { self.narrow }
    }

    /// The widest main-hash distance either rule admits.
    fn widest(&self) -> u32 {
        self.narrow.tol.widest().max(self.wide.tol.widest())
    }
}

fn has_detail(fp: &VideoFingerprint) -> bool {
    !fp.valid_hashes.is_empty() && fp.detail_hashes.len() == fp.valid_hashes.len()
}

/// Thin `fp`'s samples to at most one per `ALIGNMENT_TOLERANCE_MS`, in place.
///
/// A sample that starts sooner than that after the last one kept is folded into
/// it: dropped, its span added to the kept one's, so coverage still counts the
/// same footage. The comparison cannot place a match more finely than that
/// window (`alignment_window` never goes under it), so samples denser than it
/// carry no timing. What they do carry is more draws of each picture's hash
/// noise, and every draw is one more chance to land near some frame of an
/// unrelated file. A file sampled at every frame (an all-intra encode, which a
/// keyframe-only decode reads frame by frame) had twenty-five times the chances
/// of the same footage sampled at ordinary keyframes, and on the second
/// generated corpus one such file stood in 130 of the 227 false links between
/// unrelated videos at `-d 10`.
///
/// A file whose samples are already at least that far apart is untouched,
/// which is nearly every ordinary encode. Applied to every fingerprint just
/// before comparison, so the report's sample count and the window's evidence
/// read the samples the comparison measured. Decoding and the cache are not
/// involved: an entry keeps every sample, and changing this costs no rename.
pub(crate) fn thin_samples(fp: &mut VideoFingerprint) {
    let n = fp.valid_hashes.len();
    let step = ALIGNMENT_TOLERANCE_MS as u32;
    if fp.valid_t_start.windows(2).all(|w| w[1] >= w[0] + step) {
        return;
    }
    // The new index of each sample kept, `u32::MAX` for one folded away.
    let mut new_of = vec![u32::MAX; n];
    let mut kept = 0usize;
    let mut last_start = 0u32;
    for i in 0..n {
        let start = fp.valid_t_start[i];
        if kept > 0 && start < last_start + step {
            let end = fp.valid_t_end[i];
            let k = kept - 1;
            fp.valid_t_end[k] = fp.valid_t_end[k].max(end);
            continue;
        }
        new_of[i] = kept as u32;
        fp.valid_hashes[kept] = fp.valid_hashes[i];
        fp.valid_t_start[kept] = start;
        fp.valid_t_end[kept] = fp.valid_t_end[i];
        for v in [&mut fp.mirrored_hashes, &mut fp.detail_hashes, &mut fp.mirrored_detail_hashes] {
            if v.len() == n {
                v[kept] = v[i];
            }
        }
        last_start = start;
        kept += 1;
    }
    fp.valid_hashes.truncate(kept);
    fp.valid_t_start.truncate(kept);
    fp.valid_t_end.truncate(kept);
    for v in [&mut fp.mirrored_hashes, &mut fp.detail_hashes, &mut fp.mirrored_detail_hashes] {
        if v.len() == n {
            v.truncate(kept);
        }
    }
    for r in &mut fp.reframed {
        let mut w = 0;
        for j in 0..r.samples.len() {
            let ni = new_of.get(r.samples[j] as usize).copied().unwrap_or(u32::MAX);
            if ni != u32::MAX {
                r.samples[w] = ni;
                r.hashes[w] = r.hashes[j];
                w += 1;
            }
        }
        r.samples.truncate(w);
        r.hashes.truncate(w);
    }
}

/// How far apart two matches' time offsets may be and still stand witness for
/// each other.
///
/// Each sample is a keyframe, and two encodes place their keyframes where
/// their own encoders chose, so the offset between a matched pair of samples
/// is the true offset plus up to one sample gap of either file. The window is
/// the larger of the two files' typical (median) gap, and never under
/// `ALIGNMENT_TOLERANCE_MS`.
fn alignment_window(fp_a: &VideoFingerprint, fp_b: &VideoFingerprint) -> i64 {
    fn median_gap(fp: &VideoFingerprint) -> i64 {
        let mut gaps: Vec<u32> =
            fp.valid_t_start.iter().zip(&fp.valid_t_end).map(|(s, e)| e.saturating_sub(*s)).collect();
        if gaps.is_empty() {
            return 0;
        }
        let mid = gaps.len() / 2;
        *gaps.select_nth_unstable(mid).1 as i64
    }
    ALIGNMENT_TOLERANCE_MS.max(median_gap(fp_a)).max(median_gap(fp_b))
}

/// The `k`th 16-bit block of a hash, most significant first.
#[inline(always)]
fn block_of(hash: u64, k: usize) -> usize {
    ((hash >> (HASH_BITS as usize - BLOCK_BITS * (k + 1))) & (BINS as u64 - 1)) as usize
}

/// How far around each block key the index must look to be exhaustive at
/// `max_hamming_dist`.
///
/// Integer division is the whole rule, and it is what makes a tight tolerance
/// cheap: below `-d 4` this is 0, because three differing bits cannot be spread
/// across four blocks without leaving one of them untouched, so probing the
/// exact bins alone finds every pair. The default tolerance of 4 is the first
/// value that needs a neighbour lookup, which costs 17 bins per block instead
/// of 1.
///
/// Nothing caps this, so phase 1 proposes every pair within the tolerance at
/// EVERY tolerance -- the pigeonhole guarantee above holds unconditionally. It
/// used to be capped at radius 1, which made the index exhaustive only to `-d 7`
/// and a filter above that, and what the filter dropped was substantial enough
/// to matter: on the 727-file local corpus at `-p 10` it reported 434 files at
/// `-d 14` against 517 exhaustive, and 559 against 691 at `-d 16`. The cap paid
/// for itself in nothing but time, and `index_is_cheaper` buys that back at the
/// loose end far more effectively than a cap did.
fn probe_radius(max_hamming_dist: u32) -> u32 {
    max_hamming_dist / BLOCKS as u32
}

/// Whether to propose candidates through the index or skip it and compare every
/// pair of videos outright.
///
/// The two routes return the same pairs -- the index is exhaustive at every
/// tolerance and phase 2 measures whatever it is handed -- so this is purely a
/// question of which is faster, and the answer flips as `-d` rises. Per stored
/// hash the index examines `BLOCKS * masks` bins holding `total_hashes / BINS`
/// hashes apiece, where the direct route examines every other hash in the
/// library. The mask count is what moves: 1, 17, 137, 697, 2517 keys per block
/// for radius 0 through 4, so the index's side of this grows by roughly 5x per
/// rung of `-d 4` while the direct side does not move at all.
///
/// Measured, compare stage only, warm cache, 8 threads -- the local 727-file
/// corpus (9k hashes) and a 42-file library of full episodes (33k hashes):
///
/// ```text
///     -d            8      12      16      20      24      32
///     radius        2       3       4       5       6       8
///     9k  index   11ms    33ms   106ms   303ms   681ms      --
///     9k  direct  36ms    35ms    37ms    36ms    45ms      --
///     33k index  316ms   675ms  1511ms  3375ms  6906ms 17932ms
///     33k direct 379ms   394ms   416ms   405ms   497ms  1064ms
/// ```
///
/// The direct route is flat because it compares everything whatever the
/// tolerance says; the index is not, and by `-d 32` it spends eighteen seconds
/// proposing the same pairs the direct route proposes in one. That is the whole
/// reason this function exists: an uncapped radius is exhaustive but degenerates
/// at the loose end, where the probe set has grown wide enough to touch most of
/// the library and the index has stopped being an index.
///
/// The estimate picks the faster route everywhere except the two near-ties at
/// radius 3, where it keeps the index for the 33k library that would have run
/// 1.7x quicker directly, and drops it for the 9k one that was already 1.06x
/// ahead. Being within a factor of two at the crossover is all the accuracy this
/// needs: what it exists to avoid is the order of magnitude at the right-hand
/// end of the table, not a coin-flip in the middle.
fn index_is_cheaper(masks: usize, videos: usize, total_hashes: usize) -> bool {
    let bin_len = total_hashes / BINS;
    let index = total_hashes * BLOCKS * masks * (PROBE_COST + bin_len);
    let direct = total_hashes * total_hashes / 2 + videos * videos / 2 * PAIR_COST;

    index <= direct
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
/// either. They are kept apart here and stay apart through the report, which
/// states each one against its own file's runtime -- see `matched_seconds`.
/// Only `--min-duration` reconciles them, and it does so for itself -- see
/// `overlap_seconds`.
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
    /// The pair matched with `a` mirrored left to right: one is a mirrored copy
    /// of the other. Every figure above was measured that way round.
    pub mirrored: bool,
    /// The pair matched with one file's pictures read inside a smaller part of
    /// them (`Reframing`): one is a reframed copy of the other.
    pub reframed: Option<Reframing>,
}

impl Match {
    /// A pair whose overlap was measured but whose position in either file was
    /// not. Only tests build these; every real match carries its spans, because
    /// the same pass that measures the overlap already knows where it is.
    #[cfg(test)]
    pub fn new(a: usize, b: usize, coverage_a: f32, coverage_b: f32) -> Self {
        Match { a, b, coverage_a, coverage_b, span_a: None, span_b: None, mirrored: false, reframed: None }
    }

    /// The same match with its two files the other way round.
    fn swapped(self) -> Self {
        Match {
            a: self.b,
            b: self.a,
            coverage_a: self.coverage_b,
            coverage_b: self.coverage_a,
            span_a: self.span_b,
            span_b: self.span_a,
            ..self
        }
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
/// The pair map is the whole index. It used to carry an adjacency list beside
/// it, for the report's question -- "everything this file matched" -- on the
/// grounds that answering it from the map alone means probing every other file
/// in the library. That was never the question actually asked: every link the
/// report prints is scoped to the group being printed, so the walk is over the
/// group's handful of members and each one is a single lookup. The adjacency
/// list only ever added a `deg(subject)`-sized scan in front of it, plus a
/// `Vec` per file for a loose scan to grow.
pub struct MatchIndex {
    links: HashMap<(usize, usize), Link>,
    /// Which set of byte-identical files each file belongs to, as the set's
    /// number plus one; 0, or past the end, for a file in none. Empty unless
    /// `with_identical` filled it.
    same_bytes: Vec<u32>,
}

impl MatchIndex {
    pub fn new(matches: Vec<Match>) -> Self {
        let mut links = HashMap::with_capacity(matches.len() * 2);

        for m in matches {
            links.insert((m.a, m.b), Link { coverage: m.coverage_a, span: m.span_a });
            links.insert((m.b, m.a), Link { coverage: m.coverage_b, span: m.span_b });
        }

        Self { links, same_bytes: Vec::new() }
    }

    /// Record which files are byte-identical copies of each other: `sets` are
    /// indices into the fingerprint list, each set two or more files whose
    /// every byte agrees (see `content`). It changes no coverage and no
    /// verdict -- identical files match each other in full already -- only what
    /// the report can say about the link.
    pub fn with_identical(mut self, files: usize, sets: &[Vec<usize>]) -> Self {
        self.same_bytes = vec![0; files];
        for (n, set) in sets.iter().enumerate() {
            for &i in set {
                if let Some(slot) = self.same_bytes.get_mut(i) {
                    *slot = n as u32 + 1;
                }
            }
        }
        self
    }

    /// Whether `a` and `b` are the same bytes.
    pub fn identical(&self, a: usize, b: usize) -> bool {
        let of = |i: usize| self.same_bytes.get(i).copied().unwrap_or(0);
        a != b && of(a) != 0 && of(a) == of(b)
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

    /// Seconds of `subject`'s OWN runtime that `other` was found to contain --
    /// the figure the report prints on `subject`'s row.
    ///
    /// Directional, and deliberately so. Every other figure on a report row
    /// describes the row's own file, including the envelope from `span`, and a
    /// symmetric figure sitting among them cannot be read: on the low-coverage
    /// side of a lopsided pair it contradicts the envelope printed beside it,
    /// with nothing in the row to say that one of the numbers changed subject.
    /// The report used to print `overlap_seconds` here and a file whose matched
    /// footage envelope ran 0.00-8.84 reported 1.88 seconds of it.
    ///
    /// Taken over `duration` rather than `total_ms`, which matters on the files
    /// where those two clocks differ. `sample_times` only ever extends the
    /// runtime -- `total_ms` is `max(duration * 1000, last_sample + gap)` -- so
    /// `total_ms >= duration * 1000` always, and taking the coverage over the
    /// smaller of the two keeps this figure under BOTH of the things a reader
    /// puts it next to: the file's own `length` column, which is `duration`,
    /// and the envelope from `span`, which is stated on the `total_ms` clock.
    /// Over `total_ms` it would be the exact matched milliseconds back, but a
    /// fully covered file whose samples outran its container runtime would then
    /// report more matched footage than the length printed beside it.
    ///
    /// This does NOT make the pair figure redundant in general -- it is what
    /// `--min-duration` gates on, and `overlap_seconds` still owns it -- but on
    /// an honest match the two agree: the clip's `100% x 2min` and the host's
    /// `9% x 22min` are both two minutes, so each row prints two minutes and
    /// the symmetry the pair figure was reaching for survives. Where they
    /// disagree the pair is lopsided, and that is worth seeing rather than
    /// flattening to a minimum.
    ///
    /// `None` when the pair was never compared, which is not the same as
    /// compared and sharing nothing.
    pub fn matched_seconds(&self, subject: usize, other: usize, fps: &[VideoFingerprint])
        -> Option<f64>
    {
        let coverage = self.coverage(subject, other)?;
        Some(coverage as f64 * fps[subject].duration)
    }

    /// Seconds of content two files have in common, reconciled to one figure --
    /// what `--min-duration` gates on.
    ///
    /// The arithmetic and the reasoning behind it live in `overlap_seconds`,
    /// which the gate calls directly because it runs before any `MatchIndex`
    /// exists. This is that function reached by index: it looks the pair's two
    /// directional coverages up and hands them over. `None` when the pair was
    /// never compared, which is not the same as compared and sharing nothing.
    ///
    /// Test-only since the report went directional -- see `matched_seconds`.
    /// The gate itself never had a `MatchIndex` to reach it through, so nothing
    /// in the run path lost a caller; what these tests keep pinned is that the
    /// gate's definition still reconciles the way it always did.
    #[cfg(test)]
    pub fn shared_seconds(&self, a: usize, b: usize, fps: &[VideoFingerprint]) -> Option<f64> {
        let cov_a = self.coverage(a, b)?;
        let cov_b = self.coverage(b, a)?;
        overlap_seconds(cov_a, fps[a].duration, cov_b, fps[b].duration)
    }

    /// Every measured link `subject` has *within `group`*, strongest first.
    ///
    /// The group has to be supplied because groups overlap: a file that is a
    /// duplicate in two groups matched files in both, and a row reported under
    /// one of them must not name a file the reader cannot see beside it. So the
    /// neighbour list is filtered to the group being printed.
    ///
    /// A group is a star (see `clustering`), so only the representative is
    /// guaranteed a link to every member; any other member has the
    /// representative and whichever other members it also matched. A pair with
    /// no link reports nothing rather than a zero, because it is a comparison
    /// that did not clear the gate, not a figure.
    ///
    /// It walks the GROUP and probes the pair map, rather than walking the
    /// subject's neighbours and testing each against the group. The two answer
    /// identically -- a neighbour is exactly a pair the map holds -- but the
    /// second is `deg(subject) x group.len()`, and `deg` is a property of the
    /// whole library at the chosen `-d` rather than of the group being printed.
    /// This way the reporting pass costs one hash lookup per pair it prints,
    /// whatever a loose scan did to the graph around it.
    ///
    /// Ordered by matched duration descending, ties broken on path, so the
    /// report is reproducible run to run: the strongest link is `first()`, which
    /// is what the single-figure columns show, and the whole list is what the
    /// JSON carries so a three-file group can be read pair by pair.
    ///
    /// "Strongest" is measured in `subject`'s own footage, which is the only
    /// reading that makes the ordering agree with the row it sorts: the link
    /// printed is the one accounting for the most of THIS file, so a file
    /// reporting most of its runtime is a copy of something here.
    pub fn links_of(
        &self,
        subject: usize,
        group: &[usize],
        fps: &[VideoFingerprint],
    ) -> Vec<GroupLink> {
        let mut links: Vec<GroupLink> = group
            .iter()
            .filter(|&&other| other != subject)
            .filter_map(|&other| {
                Some(GroupLink {
                    other,
                    matched_seconds: self.matched_seconds(subject, other, fps)?,
                    span: self.span(subject, other),
                    identical: self.identical(subject, other),
                })
            })
            .collect();

        links.sort_by(|x, y| {
            y.matched_seconds
                .partial_cmp(&x.matched_seconds)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Of two links accounting for the same footage, the copy is
                // the stronger statement, so it is the one the row names.
                .then_with(|| y.identical.cmp(&x.identical))
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
        group: &[usize],
        fps: &[VideoFingerprint],
    ) -> Option<GroupLink> {
        self.links_of(subject, group, fps).into_iter().next()
    }
}

/// Seconds of footage one measured pair has in common, as a single figure.
///
/// Each side estimates it as its own coverage times its own runtime, and for
/// genuinely shared footage the two agree, because the shared segment has one
/// real duration no matter which file you measure it in: a clip's `100% x 2min`
/// and its host's `9% x 22min` are both two minutes.
///
/// Where the two estimates disagree -- different keyframe densities, tolerance
/// landing differently on each side -- the LOWER is taken, the conservative
/// reading for a tool that deletes things.
///
/// A file whose runtime the container never reported contributes no estimate.
/// If neither file has a known runtime the answer is `None`: the overlap is
/// unknown, which is not the same as zero.
///
/// **`--min-duration` is the only caller.** The report deliberately does not use
/// this: reconciling to one number is right for a gate, which has to decide
/// something about the pair, and wrong for a row, which has to describe one
/// file -- see `MatchIndex::matched_seconds`. The two are not in danger of
/// drifting the way the gate and the report once did (the gate took the HIGHER
/// estimate while the report took the lower, so `--min-duration 5` admitted --
/// and marked DELETE -- pairs whose own reported overlap read 2.9s), because
/// they no longer answer the same question: this one is the pair's floor and
/// the row's is that row's own footage, and the row's is never the smaller of
/// the two.
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
///
/// Every field is stated from the SUBJECT's end. That uniformity is the point:
/// these become one report row, and a row whose figures answer for two
/// different files cannot be read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupLink {
    /// Index into the fingerprint list of the file on the other end.
    pub other: usize,
    /// Seconds of the SUBJECT's own runtime that `other` was found to contain.
    /// Directional -- see `MatchIndex::matched_seconds`. On an honest match it
    /// reads the same from both ends anyway; where it does not, the pair is
    /// lopsided and the two rows are supposed to say so.
    pub matched_seconds: f64,
    /// Where that footage sits in the SUBJECT's runtime, not the other file's.
    /// Always an envelope of `matched_seconds`, never narrower than it.
    pub span: Option<Span>,
    /// Whether `other` is a byte-identical copy of the subject.
    pub identical: bool,
}

/// --- Phase 1: candidate generation ---------------------------------------
///
/// Probe the 4x16-bit multi-index and emit every ordered pair `(a, b)` with
/// `a < b` that shares at least one frame within `max_hamming_dist`. This is a
/// *filter*, not the answer: it decides which video pairs are worth a full
/// comparison, and nothing else.
///
/// The probe radius is derived from the tolerance rather than fixed (see
/// `probe_radius`), and nothing bounds it, so this is exhaustive at every
/// `max_hamming_dist`: any pair within the tolerance is proposed. What phase 2
/// adds is not recall of missed pairs but the measurement itself -- the index
/// says which videos could overlap and never how much, which is why it carries
/// no per-frame detail.
///
/// `masks` is passed in rather than derived here because its length is also what
/// decides whether this function is worth calling at all -- see
/// `index_is_cheaper`.
///
/// The blocks are indexed ONE AT A TIME, and each index is dropped before the
/// next is built. Four live indices is four entries per stored hash; one is one.
/// The cost is that a pair found in several blocks is emitted several times,
/// which a sort and a dedup at the end of each pass settles -- candidate pairs
/// are orders of magnitude scarcer than the hashes they were found from.
#[cfg(test)]
fn candidate_pairs(
    fingerprints: &[VideoFingerprint],
    masks: &[u16],
    max_hamming_dist: u32,
) -> Vec<(usize, usize)> {
    candidate_pairs_probing(fingerprints, fingerprints, masks, max_hamming_dist, false, &Counter::default())
}

/// `candidate_pairs`, with each lower-numbered file `a` probing the index with
/// the hashes of `probes[a]` instead of its own -- the mirrored fingerprints,
/// to propose the pairs that match only with `a` mirrored.
/// Phase 1 without an index: every probe file's hashes against every other
/// file's, stopping at the first frame the pair's rule could admit at all.
/// Proposes a subset of what `candidate_pairs_probing` does, and drops only
/// pairs that measure to nothing; which route runs is purely a question of
/// speed (`index_is_cheaper`).
///
/// Each probe file's row is reported to `done` when it is finished, weighing
/// `row_weights[q]` (`scan_row_weights`).
fn candidate_pairs_scanning(
    fingerprints: &[VideoFingerprint],
    probes: &[VideoFingerprint],
    rules: Rules,
    both_ways: bool,
    wanted: &(impl Fn(usize, usize) -> bool + Sync),
    (row_weights, done): (&[u64], &Counter),
) -> Vec<(usize, usize)> {
    let n = fingerprints.len();
    // A frame can only be admitted inside the rule's own widest reach, which
    // for a pair read the wide way is the combined distance as well as the
    // main hash's cap -- so that is what is asked here, of whichever rule the
    // pair will be measured by.
    let any_frame = |a: &VideoFingerprint, b: &VideoFingerprint| -> bool {
        let tol = rules.for_pair(a, b).tol;
        match tol.cap {
            None => any_within(&a.valid_hashes, None, &b.valid_hashes, None, tol.loose, tol.loose),
            Some(cap) => any_within(
                &a.valid_hashes,
                Some(&a.detail_hashes),
                &b.valid_hashes,
                Some(&b.detail_hashes),
                cap,
                tol.loose,
            ),
        }
    };
    (0..n)
        .into_par_iter()
        .flat_map_iter(|q| {
            let probe = &probes[q].valid_hashes;
            let first = if both_ways { 0 } else { q + 1 };
            let row: Vec<(usize, usize)> = (first..n)
                .filter(move |&t| {
                    t != q
                        && !probe.is_empty()
                        && !shutdown_requested()
                        && wanted(q, t)
                        && any_frame(&probes[q], &fingerprints[t])
                })
                .map(move |t| (q, t))
                .collect();
            done.add(1, row_weights[q]);
            row
        })
        .collect()
}

/// What each probe file's row of `candidate_pairs_scanning` costs, in hash
/// pairs: its hashes times the hashes of every file it is compared with. The
/// same filter the scan applies, so a reframing measured only against portrait
/// files weighs only what it compares.
fn scan_row_weights(
    fingerprints: &[VideoFingerprint],
    probes: &[VideoFingerprint],
    both_ways: bool,
    wanted: &(impl Fn(usize, usize) -> bool + Sync),
) -> Vec<u64> {
    let n = fingerprints.len();
    (0..n)
        .into_par_iter()
        .map(|q| {
            let hq = probes[q].valid_hashes.len() as u64;
            if hq == 0 {
                return 0;
            }
            let first = if both_ways { 0 } else { q + 1 };
            let others: u64 = (first..n)
                .filter(|&t| t != q && wanted(q, t))
                .map(|t| fingerprints[t].valid_hashes.len() as u64)
                .sum();
            hq * others
        })
        .collect()
}

/// Whether any frame of `a` is within `cap` of a frame of `b` on the main hash
/// and within `loose` of it on main plus detail (main alone when either side
/// has no detail hashes).
///
/// The innermost loop of phase 1's scan, so it is compiled a second time with
/// the POPCNT instruction and that copy taken when the processor has it: the
/// release binary targets baseline x86-64, where `count_ones` is a dozen
/// instructions of bit arithmetic. Both copies are the same source and give
/// the same answer.
fn any_within(a: &[u64], a_detail: Option<&[u64]>, b: &[u64], b_detail: Option<&[u64]>, cap: u32, loose: u32) -> bool {
    #[inline(always)]
    fn body(a: &[u64], a_detail: Option<&[u64]>, b: &[u64], b_detail: Option<&[u64]>, cap: u32, loose: u32) -> bool {
        match (a_detail, b_detail) {
            (Some(da), Some(db)) => a.iter().zip(da).any(|(&h_a, &d_a)| {
                b.iter().zip(db).any(|(&h_b, &d_b)| {
                    let main = (h_a ^ h_b).count_ones();
                    main <= cap && main + (d_a ^ d_b).count_ones() <= loose
                })
            }),
            _ => a.iter().any(|&h_a| b.iter().any(|&h_b| (h_a ^ h_b).count_ones() <= cap.min(loose))),
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        // `unsafe` rather than a safe `#[target_feature]` fn, which needs Rust
        // 1.86 and the declared MSRV is 1.85.
        #[target_feature(enable = "popcnt")]
        unsafe fn with_popcnt(a: &[u64], da: Option<&[u64]>, b: &[u64], db: Option<&[u64]>, cap: u32, loose: u32) -> bool {
            body(a, da, b, db, cap, loose)
        }
        if std::arch::is_x86_feature_detected!("popcnt") {
            // SAFETY: the feature was detected on this processor just above.
            return unsafe { with_popcnt(a, a_detail, b, b_detail, cap, loose) };
        }
    }
    body(a, a_detail, b, b_detail, cap, loose)
}

///
/// Each probe file's pass over one block is reported to `done`, weighing its
/// hash count.
fn candidate_pairs_probing(
    fingerprints: &[VideoFingerprint],
    probes: &[VideoFingerprint],
    masks: &[u16],
    max_hamming_dist: u32,
    both_ways: bool,
    done: &Counter,
) -> Vec<(usize, usize)> {
    let n = fingerprints.len();

    log::debug!(
        "Probing each block at radius {} ({} key(s) per block) for -d {}.",
        probe_radius(max_hamming_dist),
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
                    let fp_a = &probes[v_a];
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
                            // Unless the orientation is one-sided, when
                            // every other video is a candidate either way.
                            let start =
                                if both_ways { 0 } else { videos.partition_point(|&v| (v as usize) <= v_a) };

                            for (&v_b, &h_b) in videos[start..].iter().zip(&hashes[start..]) {
                                let v_b = v_b as usize;
                                if seen[v_b] == mark || v_b == v_a {
                                    continue;
                                }
                                if (h_a ^ h_b).count_ones() <= max_hamming_dist {
                                    seen[v_b] = mark;
                                    local.push((v_a, v_b));
                                }
                            }
                        }
                    }

                    done.add(1, fp_a.valid_hashes.len() as u64);
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
///
/// ## Two tolerances, and why the looser one needs a witness
///
/// A frame match inside `tol.strict` is taken at face value. One between
/// `strict` and `loose` is only taken if a *different* frame of each video also
/// matches at the same time offset -- see `ALIGNMENT_TOLERANCE_MS`. Two encodes
/// of the same footage place their shared frames at one constant offset, so real
/// matches corroborate each other for free; two videos that merely look alike
/// scatter, and a scattered near-match is exactly the one worth refusing.
///
/// This cuts both ways, which is the point. Below `-d 8` it *admits* matches the
/// tolerance alone would have refused, and above it, it *refuses* uncorroborated
/// ones the tolerance alone would have taken.
fn match_overlap(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    rule: Rule,
    witnesses: &mut Witnesses,
) -> (f32, f32, Option<Span>, Option<Span>) {
    // Per pair, and deliberately left that way. `Witnesses` beside them is the
    // worker's, and moving these two (and `corroborate_pair`'s match list) in
    // with it was tried and reverted: it is correct, it is byte-identical, and
    // it buys NOTHING measurable -- see the note in CLAUDE.md for the numbers.
    let mut matched_a = vec![false; fp_a.valid_hashes.len()];
    let mut matched_b = vec![false; fp_b.valid_hashes.len()];

    admit(fp_a, fp_b, rule, &mut matched_a, &mut matched_b, witnesses, &mut |_, _, _| {});

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

/// Every frame match the rule admits for one pair, flagged in `matched_a` and
/// `matched_b` and told to `on_mark` as `(i, j, distance)`: the strict ones,
/// the corroborated ones, and then the ones an established alignment rescues.
///
/// ## Rescue along an alignment
///
/// Two accepted matches at one offset (within the pair's `alignment_window` of
/// each other) say where `b` sits in `a`. A frame between them no longer has to
/// stand against every frame of the other file, only against the few at that
/// offset, so it is held to one more `CORROBORATION_SLACK_SIGMA` step than the
/// corroborated tolerance -- the same step corroboration itself takes beyond
/// strict -- and only inside the stretch both ends of the chain bound, on both
/// sides. This is what recovers the middle of a clip whose keyframes the
/// encoder placed between the source's: the ends anchor the offset, and the
/// frames in between are near misses rather than chance.
fn admit<M: FnMut(u32, u32, u32)>(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    rule: Rule,
    matched_a: &mut [bool],
    matched_b: &mut [bool],
    witnesses: &mut Witnesses,
    on_mark: &mut M,
) {
    let tol = rule.tol;
    if tol.loose > tol.strict {
        corroborate_pair_with(fp_a, fp_b, rule, matched_a, matched_b, MAX_ALIGNED_MATCHES, witnesses, on_mark);
    } else {
        // Nothing to corroborate: the strict pass marks everything the rule
        // itself admits. Only `-d` at the width of the hash and tests reach it.
        for_each_frame_match(fp_a, fp_b, tol, matched_a, matched_b, |_| {});
    }
    if !matched_a.iter().any(|&m| m) {
        return;
    }
    // The anchors of an alignment: every frame match within the rule's reach
    // between two samples that are both matched. Not "the matches corroboration
    // admitted", which is what it first was: corroboration skips a row whose
    // two samples are already settled, so that list could SHRINK as `-d` rose
    // and take a rescue with it. Both conditions here only ever grow with `-d`,
    // so the anchors do, and so does everything rescued from them. The same
    // sweep names the strict matches, which `for_each_frame_match` flagged
    // without saying which pair they were.
    let mut anchors: Vec<Aligned> = Vec::new();
    let (mut scratch_a, mut scratch_b) = (matched_a.to_vec(), matched_b.to_vec());
    for_each_frame_match(fp_a, fp_b, tol, &mut scratch_a, &mut scratch_b, |entry| {
        if entry.3 <= tol.strict {
            on_mark(entry.1, entry.2, entry.3);
        }
        if matched_a[entry.1 as usize] && matched_b[entry.2 as usize] {
            anchors.push(entry);
        }
    });
    rescue_along_alignments(fp_a, fp_b, tol, anchors, matched_a, matched_b, on_mark);
    // What is credited is read the way the anchors are, after the rescue: every
    // frame match within the rule's reach between two samples both matched. A
    // set that only grows with `-d`, where the marks reported along the way
    // need not (corroboration skips a row already settled).
    let mut settled: Vec<(u32, u32)> = Vec::new();
    let (mut scratch_a, mut scratch_b) = (matched_a.to_vec(), matched_b.to_vec());
    for_each_frame_match(fp_a, fp_b, tol, &mut scratch_a, &mut scratch_b, |(_, i, j, _)| {
        if matched_a[i as usize] && matched_b[j as usize] {
            settled.push((i, j));
        }
    });
    credit_matched_spans(fp_a, fp_b, &mut settled, matched_a, matched_b);
}

/// Whether two of `group` differ on both sides. Asked of every pair rather
/// than against one of them, so that adding a match can never make it false.
/// If none differs from the first on both sides, every match shares its `i` or
/// its `j`, and two apart exist exactly when one of each kind does.
fn two_apart(group: &[(u32, u32)]) -> bool {
    let Some(&(i0, j0)) = group.first() else { return false };
    if group.iter().any(|&(i, j)| i != i0 && j != j0) {
        return true;
    }
    group.iter().any(|&(i, j)| i == i0 && j != j0) && group.iter().any(|&(i, j)| j == j0 && i != i0)
}

/// The footage a matched hash stands for, credited on both sides.
///
/// Each hash stands for its picture until its file's next sample, which is how
/// coverage has always been counted -- but only on the side the hash belongs
/// to. Two files sampled at different densities then disagree about how much
/// they share: a long source sampled every few seconds is credited a few
/// seconds for one match, while the short clip it matched, sampled every half
/// second, is credited half a second for the same footage. A compilation of
/// four such clips read under a 20% gate on its own side. So a match also
/// credits the other side's samples that fall inside the matched sample's own
/// span, read from the matched sample on: the same assumption, made of the
/// same match, from both ends.
///
/// Only for matches that agree on an offset with at least one other match of
/// different samples (the chains `rescue_along_alignments` builds on): a lone
/// coincidence is not allowed to stand for anything beyond itself, which is
/// what kept the near-miss sets out (credited from every match, `-d 7` took 31
/// false links where this takes none).
fn credit_matched_spans(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    marked: &mut Vec<(u32, u32)>,
    matched_a: &mut [bool],
    matched_b: &mut [bool],
) {
    if marked.len() < 2 {
        return;
    }
    let window = alignment_window(fp_a, fp_b);
    let off = |&(i, j): &(u32, u32)| fp_b.valid_t_start[j as usize] as i64 - fp_a.valid_t_start[i as usize] as i64;
    marked.sort_unstable_by_key(|m| (off(m), *m));
    marked.dedup();
    let credit = |from: &VideoFingerprint, k: usize, onto: &VideoFingerprint, at: usize, m: &mut [bool]| {
        let lo = onto.valid_t_start[at];
        let hi = lo.saturating_add(from.valid_t_end[k].saturating_sub(from.valid_t_start[k]));
        let first = onto.valid_t_start.partition_point(|&t| t < lo);
        for (x, &t) in onto.valid_t_start.iter().enumerate().skip(first) {
            if t >= hi {
                break;
            }
            m[x] = true;
        }
    };
    let mut s = 0;
    while s < marked.len() {
        let mut e = s + 1;
        while e < marked.len() && off(&marked[e]) - off(&marked[e - 1]) <= window {
            e += 1;
        }
        if two_apart(&marked[s..e]) {
            for &(i, j) in &marked[s..e] {
                credit(fp_b, j as usize, fp_a, i as usize, matched_a);
                credit(fp_a, i as usize, fp_b, j as usize, matched_b);
            }
        }
        s = e;
    }
}

/// The rescue step of `admit`. `accepted` is its anchors.
fn rescue_along_alignments<M: FnMut(u32, u32, u32)>(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    tol: Tolerance,
    mut accepted: Vec<Aligned>,
    matched_a: &mut [bool],
    matched_b: &mut [bool],
    on_mark: &mut M,
) {
    if accepted.len() < 2 {
        return;
    }
    let window = alignment_window(fp_a, fp_b);
    accepted.sort_unstable();
    // Each chain: its offset range with the window either side, then the
    // stretch of a and the stretch of b it bounds.
    type Chain = ((i64, i64), (u32, u32), (u32, u32));
    let mut chains: Vec<Chain> = Vec::new();
    let mut s = 0;
    while s < accepted.len() {
        let mut e = s + 1;
        while e < accepted.len() && accepted[e].0 - accepted[e - 1].0 <= window {
            e += 1;
        }
        if e - s >= 2 {
            let chain = &accepted[s..e];
            let ta = |&(_, i, _, _): &Aligned| fp_a.valid_t_start[i as usize];
            let tb = |&(_, _, j, _): &Aligned| fp_b.valid_t_start[j as usize];
            let a = (chain.iter().map(ta).min().unwrap_or(0), chain.iter().map(ta).max().unwrap_or(0));
            let b = (chain.iter().map(tb).min().unwrap_or(0), chain.iter().map(tb).max().unwrap_or(0));
            if a.1 > a.0 {
                chains.push(((chain[0].0 - window, chain[e - s - 1].0 + window), a, b));
            }
        }
        s = e;
    }
    if chains.is_empty() {
        return;
    }
    let width = if tol.cap.is_some() { WIDE_BITS } else { HASH_BITS };
    let slack = (CORROBORATION_SLACK_SIGMA * sigma_of(width)).round() as u32;
    let reach = tol.loose.saturating_add(slack).min(width);
    let wider = Tolerance { strict: reach, loose: reach, cap: tol.cap.map(|_| (reach / 2).min(HASH_BITS)) };
    let (mut scratch_a, mut scratch_b) = (vec![false; matched_a.len()], vec![false; matched_b.len()]);
    for_each_frame_match(fp_a, fp_b, wider, &mut scratch_a, &mut scratch_b, |(off, i, j, d)| {
        let (ta, tb) = (fp_a.valid_t_start[i as usize], fp_b.valid_t_start[j as usize]);
        let inside = chains.iter().any(|&((olo, ohi), (alo, ahi), (blo, bhi))| {
            off >= olo && off <= ohi && ta >= alo && ta <= ahi && tb >= blo && tb <= bhi
        });
        if inside && !(matched_a[i as usize] && matched_b[j as usize]) {
            matched_a[i as usize] = true;
            matched_b[j as usize] = true;
            on_mark(i, j, d);
        }
    });
}

/// Every frame pair of A against every frame pair of B, marking the strict
/// matches and handing the loose ones to `keep`.
///
/// The whole quadratic comparison lives here, in one place, because the
/// corroboration pass may have to walk a pair more than once -- see
/// `MAX_ALIGNED_MATCHES` -- and a second copy of this loop is a second place for
/// the two tolerances to be applied differently. Marking a strict match is
/// idempotent, so a repeated sweep costs time and changes nothing.
///
/// `keep` is called for every match out to `loose`, strict ones included: they
/// are the strongest witnesses there are, and leaving them out would make a
/// loose match's fate depend on whether its corroborator happened to be close
/// enough to stand alone.
#[inline]
fn for_each_frame_match<F: FnMut(Aligned)>(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    tol: Tolerance,
    matched_a: &mut [bool],
    matched_b: &mut [bool],
    mut keep: F,
) {
    for (i, &h_a) in fp_a.valid_hashes.iter().enumerate() {
        for (j, &h_b) in fp_b.valid_hashes.iter().enumerate() {
            let main = (h_a ^ h_b).count_ones();
            let distance = match tol.cap {
                None => main,
                Some(cap) => {
                    if main > cap {
                        continue;
                    }
                    main + (fp_a.detail_hashes[i] ^ fp_b.detail_hashes[j]).count_ones()
                }
            };
            if distance <= tol.strict {
                matched_a[i] = true;
                matched_b[j] = true;
            }
            if distance <= tol.loose {
                let offset = fp_a.valid_t_start[i] as i64 - fp_b.valid_t_start[j] as i64;
                keep((offset, i as u32, j as u32, distance));
            }
        }
    }
}

/// Run the corroboration rule over one pair, holding at most `cap` matches at a
/// time.
///
/// The straightforward version of this collects every loose match and sorts it,
/// which is what happens whenever a pair has `cap` matches or fewer -- i.e.
/// always, on any library anyone has measured this against. See
/// `MAX_ALIGNED_MATCHES` for the pairs that do not fit and why holding them all
/// is not an option.
///
/// Those are walked in bands of the time-offset axis instead. Each band is
/// collected with a skirt of `ALIGNMENT_TOLERANCE_MS` on either side, so every
/// witness of every match inside the band is present, and `corroborated` is told
/// to mark only the matches whose own offset is inside it -- the skirt entries
/// are witnesses here and targets in their own band. The bands partition the
/// offset axis, so every match is judged exactly once, against exactly the
/// witnesses the whole-pair list would have offered it.
///
/// Returns the largest number of entries it held at once, which is what the
/// tests measure the working set by.
#[cfg(test)]
fn corroborate_pair(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    rule: Rule,
    matched_a: &mut [bool],
    matched_b: &mut [bool],
    cap: usize,
    witnesses: &mut Witnesses,
) -> usize {
    corroborate_pair_with(fp_a, fp_b, rule, matched_a, matched_b, cap, witnesses, &mut |_, _, _| {})
}

/// `corroborate_pair`, telling `on_mark` of every loose frame match it admits
/// (`corroborated_with`). The comparison passes a closure that does nothing,
/// which compiles away; only `evidence` listens.
#[allow(clippy::too_many_arguments)]
fn corroborate_pair_with<M: FnMut(u32, u32, u32)>(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    rule: Rule,
    matched_a: &mut [bool],
    matched_b: &mut [bool],
    cap: usize,
    witnesses: &mut Witnesses,
    on_mark: &mut M,
) -> usize {
    let tol = rule.tol;
    let window = alignment_window(fp_a, fp_b);
    let mut aligned: Vec<Aligned> = Vec::new();
    let mut total = 0usize;
    for_each_frame_match(fp_a, fp_b, tol, matched_a, matched_b, |entry| {
        total += 1;
        if aligned.len() < cap {
            aligned.push(entry);
        }
    });

    if total <= cap {
        let held = aligned.len();
        corroborated_with(&mut aligned, (rule.schedule, window), matched_a, matched_b, EVERY_OFFSET, witnesses, on_mark);
        return held;
    }

    // The whole pair does not fit. What was collected is a prefix of it and is
    // no use for anything, since a band needs its own witnesses in full.
    aligned = Vec::new();

    let bands = plan_bands(fp_a, fp_b, tol, window, matched_a, matched_b, cap);
    log::debug!(
        "{} loose frame match(es) between {} and {}: corroborating in {} band(s)",
        total,
        fp_a.path,
        fp_b.path,
        bands.len()
    );

    let mut held = 0usize;
    for &(lo, hi) in &bands {
        aligned.clear();
        for_each_frame_match(fp_a, fp_b, tol, matched_a, matched_b, |entry| {
            // Saturating so the widest band there is (`EVERY_OFFSET`, which
            // `plan_bands` falls back to for a pair with no samples at all)
            // cannot overflow the skirt off either end.
            if entry.0 >= lo.saturating_sub(window) && entry.0 <= hi.saturating_add(window)
            {
                aligned.push(entry);
            }
        });
        held = held.max(aligned.len());
        corroborated_with(&mut aligned, (rule.schedule, window), matched_a, matched_b, (lo, hi), witnesses, on_mark);
    }

    held
}

/// Cut the time-offset axis into bands of at most `cap` matches each.
///
/// One more sweep of the pair, counting matches into fixed-width buckets, and
/// then a greedy walk that closes a band whenever the next bucket would overrun
/// the cap. Buckets are `ALIGNMENT_TOLERANCE_MS` wide so the skirt a band
/// carries is one bucket on each side; a single bucket holding more than `cap`
/// is left as its own band and simply exceeds it, because everything in it is
/// within witnessing distance of everything else and splitting it would not
/// reduce what has to be held.
fn plan_bands(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    tol: Tolerance,
    window: i64,
    matched_a: &mut [bool],
    matched_b: &mut [bool],
    cap: usize,
) -> Vec<(i64, i64)> {
    // Measured rather than assumed from the ends of the two lists: an offset
    // that fell outside the histogram would index it out of bounds, and sample
    // times being ascending is a property of the decoder rather than of this
    // module.
    let (Some(&a_lo), Some(&a_hi)) = (fp_a.valid_t_start.iter().min(), fp_a.valid_t_start.iter().max())
    else {
        return vec![EVERY_OFFSET];
    };
    let (Some(&b_lo), Some(&b_hi)) = (fp_b.valid_t_start.iter().min(), fp_b.valid_t_start.iter().max())
    else {
        return vec![EVERY_OFFSET];
    };

    let first = a_lo as i64 - b_hi as i64;
    let last = a_hi as i64 - b_lo as i64;
    let span = last - first + 1;

    // Wider buckets than the alignment window only if the offsets span so much
    // that a bucket each would be a bigger allocation than the matches were.
    // A day of runtime either side of zero is 350k buckets at 500 ms, so this
    // is a guard rather than a working part.
    const MAX_BUCKETS: i64 = 1 << 20;
    let width = window.max(span / MAX_BUCKETS + 1);
    let buckets = (span / width + 1) as usize;

    let mut counts = vec![0usize; buckets];
    for_each_frame_match(fp_a, fp_b, tol, matched_a, matched_b, |(offset, _, _, _)| {
        counts[((offset - first) / width) as usize] += 1;
    });

    let mut bands = Vec::new();
    let mut start = 0usize;
    let mut running = 0usize;
    for (b, &count) in counts.iter().enumerate() {
        if running > 0 && running + count > cap {
            bands.push((first + start as i64 * width, first + b as i64 * width));
            start = b;
            running = 0;
        }
        running += count;
    }
    // The last band runs past the last bucket, so nothing can fall off the end.
    bands.push((first + start as i64 * width, last + 1));
    bands
}

/// No vertex: an unused slot in `Witnesses`, and an unmatched vertex in the
/// matching it builds.
const NO_VERTEX: u32 = u32::MAX;

/// The frame matches in one alignment window that could witness a candidate:
/// the ones that are a different sample from it on both sides.
///
/// `window` is the window already -- `corroborated` tracks both of its edges as
/// it goes -- so all this drops is the candidate's own row and the matches that
/// share a sample with it. One definition for both passes of
/// `Witnesses::enough`, so the cheap pass and the exact one cannot disagree
/// about which matches were on offer.
#[inline]
fn eligible(window: &[Aligned], i: u32, j: u32) -> impl Iterator<Item = (u32, u32)> + '_ {
    window
        .iter()
        .filter(move |&&(_, wi, wj, _)| wi != i && wj != j)
        .map(|&(_, wi, wj, _)| (wi, wj))
}

/// Whether a frame match has enough *distinct* other frame matches agreeing
/// with it about the time offset, where "distinct" is a pairing rather than a
/// count.
///
/// One alignment window is a bipartite graph: the samples of A it touches on
/// one side, the samples of B on the other, and one edge per frame match. A
/// witness must be a different sample from the candidate on both sides and from
/// every other witness on both sides -- which is exactly to say the witnesses
/// are a MATCHING in that graph, and the quota is met when the largest matching
/// reaches `needed`.
///
/// This used to be answered greedily, in offset order, and greedy is not the
/// same question. It is a lower bound on the matching, and one that a wider
/// `-d` can make WORSE: an entry admitted by a looser tolerance sorts earlier,
/// is taken first, and consumes both sides of two matches that would otherwise
/// have paid the quota between them. Constructed at 14 bits with a quota of two
/// and one 16-bit blocker a bucket earlier, `-d 8` marks the candidate and
/// `-d 10` does not -- the flag documented to only ever admit more, taking a
/// match away. The exact matching cannot do that: adding an edge to a bipartite
/// graph can only raise the largest matching, so every rung of `-d` sees at
/// least what the rung below it saw, by construction rather than by luck.
///
/// Measured on the local corpus, that is also nearly all it changes. Reports are
/// byte-identical at `-d 4`, `6`, `8` and `10` -- the quota is one witness out
/// to 12 bits and a matching of one is a witness of one -- and it adds 1, 7 and
/// 18 links at `-d 12`, `14` and `16` against 631, 981 and 3,282, taking none
/// away at any rung. Those are rungs where precision is 84% and falling, and the
/// one of them the labels reach is a false positive, so this is not bought for
/// accuracy: it is bought so that the sentence above about `-d` is true.
///
/// # Four questions before the search, in ascending order of cost
///
/// Deciding the matching outright for every candidate costs 13% of phase 2 at
/// `-d 24`, where a loose tolerance throws up 58.7 million candidates. Nothing
/// is gained by that: the answer is settled without building anything almost
/// every time, and each of these is exact.
///
/// **Is the window even big enough?** A witness is a row of the window, so a
/// window shorter than the quota cannot pay it. `corroborated` tracks both edges
/// of the window as it walks -- they both only move forward -- so this costs a
/// comparison and no walk at all. At a loose `-d` it is where most candidates
/// end: two unrelated videos throw up a handful of far-apart frame matches and
/// are asked for eight witnesses.
///
/// **Did greedy reach the quota?** A greedy pairing IS a matching, so reaching
/// the quota settles it. This is the whole of the rule for every distance out to
/// 12 bits, where the quota is one.
///
/// **Did greedy fall below half of it?** What greedy leaves is a MAXIMAL
/// matching -- it considered every eligible match in the window and took the
/// ones whose two samples were both still free -- so every edge of the largest
/// matching touches one of its endpoints and the largest is at most twice it.
///
/// **Does the window hold `needed` distinct samples on each side?** A matching
/// pairs one sample with one sample, so it cannot outrun the smaller side. One
/// light walk, and 5.54 million of the 5.65 million candidates that got this far
/// at `-d 24` ended here; 107 thousand went on to build anything.
///
/// # What bounds the search itself
///
/// A window that keeps `needed * (needed - 1) + 1` edges is over the quota
/// whatever they look like: Koenig's theorem puts a vertex cover the size of the
/// largest matching on the graph, and no vertex is allowed more than `needed`
/// edges here, so a graph short of the quota cannot hold that many. Both the
/// collecting walk and the matching stop there, which is why a static camera
/// cannot make this walk a window of a million matches looking for a witness it
/// already has.
///
/// Dropping an edge at a vertex already carrying `needed` of them is safe for
/// the same counting reason: a matching of at most `needed` edges covers at
/// most `needed - 1` other vertices, so a rejected edge always has a kept
/// neighbour free to stand in for it. It is what stops one static shot's worth
/// of matches filling the graph.
///
/// # What it costs
///
/// Less than the greedy rule it replaced, because the window-size question is
/// one the greedy rule never asked. The local corpus, warm, whole run, three
/// runs of each: 0.10-0.11 s against 0.11-0.12 s at `-d 4`, 0.19 s either way at
/// `-d 12`, 0.88-0.92 s against 0.89-0.92 s at `-d 18`, and **5.11-5.28 s
/// against 5.41-5.49 s** at `-d 24`. Peak RSS is unmoved at the first three and
/// within the run-to-run spread at the last.
///
/// Take any one of the four questions out and that turns into a loss. The
/// version that searched every candidate ran phase 2 at `-d 24` in 5.40-5.56 s
/// against 4.73-4.86 s for greedy; the window-size question alone is worth more
/// than the whole of the difference.
struct Witnesses {
    /// How many samples each side has, so the search can size its slots the
    /// first time one is actually run. Most pairs never reach it.
    samples_a: usize,
    samples_b: usize,
    /// The greedy pass: the witnesses accepted so far, at most `needed` of them.
    taken: Vec<(u32, u32)>,
    /// Sample index on each side to the vertex it became in this window, or
    /// `NO_VERTEX`. Cleared through `side_a`/`side_b`, so a window costs only
    /// the vertices it touched.
    slot_a: Vec<u32>,
    slot_b: Vec<u32>,
    side_a: Vec<Vertex>,
    side_b: Vec<Vertex>,
    edges: Vec<(u32, u32)>,
    seen: Vec<bool>,
}

/// One sample of one video, as the matching sees it: which sample it is, how
/// many edges it has been given, and what it is currently paired with.
///
/// One array rather than three, because every step of the search reads all
/// three of a vertex's fields together.
#[derive(Clone, Copy)]
struct Vertex {
    sample: u32,
    degree: u32,
    mate: u32,
}

impl Witnesses {
    /// One of these per phase-2 worker, reused for every pair it is handed:
    /// the buffers below are cleared per candidate and grown at most once per
    /// worker, so the ordinary path allocates nothing at all.
    fn new() -> Self {
        Witnesses {
            samples_a: 0,
            samples_b: 0,
            taken: Vec::new(),
            slot_a: Vec::new(),
            slot_b: Vec::new(),
            side_a: Vec::new(),
            side_b: Vec::new(),
            edges: Vec::new(),
            seen: Vec::new(),
        }
    }

    /// Take up a new pair. Only the slot arrays care, and they are grown lazily
    /// by `reset`, since a pair whose every candidate is settled by the greedy
    /// pass never indexes them.
    fn begin(&mut self, samples_a: usize, samples_b: usize) {
        self.samples_a = samples_a;
        self.samples_b = samples_b;
    }

    /// Whether the largest matching in `window`, excluding the candidate's own
    /// sample on each side, reaches `needed`.
    ///
    /// `window` must CONTAIN the candidate's row -- `corroborated` hands over
    /// `aligned[lo..hi]` with `lo <= k < hi`, so it always does -- because the
    /// window-size exit below counts on `eligible` dropping exactly that one.
    /// It must also hold at most one row per sample pair, which
    /// `for_each_frame_match` guarantees by visiting each pair exactly once;
    /// the degree cap in `exact` is where that becomes load-bearing.
    fn enough(&mut self, window: &[Aligned], i: u32, j: u32, needed: usize) -> bool {
        if needed == 0 {
            return true;
        }
        debug_assert!(needed <= MAX_WITNESSES);
        // A witness is a row of this window, so a window holding fewer rows than
        // the quota cannot pay it whatever they are. Free -- `corroborated`
        // already knows where the window ends -- and at a loose `-d` it is the
        // first thing most candidates die on, before anything is walked at all.
        //
        // `<=`, not `<`: the candidate's own row is always one of these (the
        // window is `aligned[lo..hi]` and `lo <= k < hi`), and `eligible` drops
        // it, so a window can only ever offer `len() - 1` witnesses. `<` was
        // conservative rather than wrong -- it let a window of exactly `needed`
        // rows walk the greedy pass to find the one row it was always short of.
        if window.len() <= needed {
            debug_assert!(
                window.iter().any(|&(_, wi, wj, _)| wi == i && wj == j),
                "the candidate's own row has to be in its window for this bound to hold"
            );
            return false;
        }

        // Greedy, in offset order, exactly as the rule was before it was exact.
        // `taken` never grows past `needed`, so the scan of it is the same
        // handful of comparisons the stack array it replaced held.
        self.taken.clear();
        for (wi, wj) in eligible(window, i, j) {
            if self.taken.iter().any(|&(ui, uj)| ui == wi || uj == wj) {
                continue;
            }
            self.taken.push((wi, wj));
            if self.taken.len() >= needed {
                return true;
            }
        }
        // Maximal, so the largest matching is at most twice it -- see the type
        // doc. This is where nearly every candidate at a loose `-d` ends.
        if self.taken.len() * 2 < needed {
            return false;
        }
        self.exact(window, i, j, needed)
    }

    /// The largest matching in the window, decided rather than approximated.
    ///
    /// Only reached from a greedy pairing that landed between half the quota and
    /// the quota, which is the one range where the two can disagree.
    fn exact(&mut self, window: &[Aligned], i: u32, j: u32, needed: usize) -> bool {
        if !self.side_is_wide_enough(window, i, j, needed) {
            return false;
        }
        self.reset();
        let conclusive = needed * (needed - 1) + 1;
        let mut paired = 0usize;
        for (wi, wj) in eligible(window, i, j) {
            // The degree cap is asked of the slots rather than of fresh
            // vertices, so an edge it drops leaves nothing behind. Creating the
            // vertices first put ISOLATED ones in `side_a`: a sample whose only
            // eligible rows all died on the other side's degree still got a
            // vertex, and `untried` below is `side_a.len() - paired`, so every
            // one of them made the "what is left to gain" exit fire later than
            // it had to. Never a wrong answer -- an isolated vertex simply
            // fails to augment -- just work the bound existed to avoid.
            //
            // Capping the degree at `needed` keeps a matching of that size if
            // one existed, but only because each edge is a DISTINCT pair of
            // samples: two copies of one edge would spend a vertex's budget on
            // a single neighbour and could then turn away the edge that paid
            // the quota. See the precondition on `enough`.
            if self.at_degree_cap(self.slot_a[wi as usize], &self.side_a, needed)
                || self.at_degree_cap(self.slot_b[wj as usize], &self.side_b, needed)
            {
                continue;
            }
            let a = self.vertex_a(wi) as usize;
            let b = self.vertex_b(wj) as usize;
            self.side_a[a].degree += 1;
            self.side_b[b].degree += 1;
            self.edges.push((a as u32, b as u32));
            if self.side_a[a].mate == NO_VERTEX && self.side_b[b].mate == NO_VERTEX {
                self.side_a[a].mate = b as u32;
                self.side_b[b].mate = a as u32;
                paired += 1;
                if paired >= needed {
                    return true;
                }
            }
            if self.edges.len() >= conclusive {
                return true;
            }
        }

        // Every vertex here carries at least one edge, so this is the same
        // question `side_is_wide_enough` asks, over what the collecting loop
        // really kept. It is not redundant with it: the degree cap can drop
        // every edge a sample had, so a sample that side_is_wide_enough counted
        // need not have a vertex here. A matching pairs one sample with one
        // sample, so it cannot outrun the smaller side either way.
        //
        // It was a `debug_assert` until the cap started dropping edges before
        // creating vertices, and it was wrong: `side_is_wide_enough` used to
        // give up and say yes past 64 samples, so the property it asserted was
        // simply never established for a long video and every debug build
        // aborted on the first such pair. It is exact on both counts now.
        if self.side_a.len() < needed || self.side_b.len() < needed {
            return false;
        }

        self.seen.clear();
        self.seen.resize(self.side_b.len(), false);
        // Only an unmatched A-side vertex can start an augmenting path, and each
        // path raises the pairing by exactly one, so what is left to try is what
        // is left to gain. Two failures in a row usually settle it.
        let mut untried = self.side_a.len() - paired;
        for a in 0..self.side_a.len() {
            if paired + untried < needed {
                return false;
            }
            if self.side_a[a].mate != NO_VERTEX {
                continue;
            }
            untried -= 1;
            self.seen.iter_mut().for_each(|s| *s = false);
            if augment(a as u32, &self.edges, &mut self.side_a, &mut self.side_b, &mut self.seen) {
                paired += 1;
                if paired >= needed {
                    return true;
                }
            }
        }
        false
    }

    /// Whether the window has `needed` distinct samples on each side, which a
    /// matching cannot outrun -- it pairs one sample with one sample.
    ///
    /// This is the test 98% of the candidates that reach the search die on: a
    /// window of a dozen frame matches asked for a dozen witnesses. Answering it
    /// first, out of one light walk, is what keeps the exact rule as cheap as
    /// the greedy one -- it was measured at 5.65 million searches over the local
    /// corpus at `-d 24`, of which 5.54 million ended here and 107 thousand went
    /// on to build anything.
    ///
    /// A bit per sample answers it in two instructions a row, and is what the
    /// measurement above was taken over -- but only while the samples fit in
    /// the word. The version this replaced had nothing else, so past
    /// `u64::BITS` samples it gave up and returned `true`, excused on the
    /// grounds that a file with more is a long video whose pairs are the
    /// expensive ones anyway. That is the argument FOR the cheap rejection
    /// rather than against it: it left the 3.3% of files in the local corpus
    /// carrying more than 64 samples building a graph for every candidate that
    /// reached here, and it meant the property this function's name states was
    /// never established for them at all -- which is what the `debug_assert` it
    /// used to stand behind found out the hard way (see `exact`).
    ///
    /// So a wider file gets `wide_sides_are_wide_enough` instead, which is the
    /// same question decided by collecting distinct samples rather than by
    /// marking them. Two routes to one predicate is the thing this file
    /// otherwise avoids, and it is bought here because the narrow one is the
    /// hot one: making every pair take the wide route costs ~1.5% of wall time
    /// at `-d 24` (5.37 s against 5.29 s, interleaved, means of five) for a
    /// verdict that is identical by construction. The differential test walks
    /// every one of its 4,000 windows past BOTH, so the collecting route cannot
    /// refuse a window the bitmask would have allowed. What no verdict test can
    /// catch is the opposite -- a route that gives up and says `true` reaches
    /// the same answer through the graph, which is how the old one survived
    /// every test there was -- so the refusal itself is pinned directly, in
    /// `test_a_file_too_wide_for_the_bitmask_still_needs_witnesses_on_both_sides`.
    fn side_is_wide_enough(&self, window: &[Aligned], i: u32, j: u32, needed: usize) -> bool {
        debug_assert!(needed <= MAX_WITNESSES);
        if self.samples_a > u64::BITS as usize || self.samples_b > u64::BITS as usize {
            return Self::wide_sides_are_wide_enough(window, i, j, needed);
        }
        let (mut seen_a, mut seen_b) = (0u64, 0u64);
        for (wi, wj) in eligible(window, i, j) {
            seen_a |= 1 << wi;
            seen_b |= 1 << wj;
        }
        seen_a.count_ones() as usize >= needed && seen_b.count_ones() as usize >= needed
    }

    /// `side_is_wide_enough` for a file with more samples than a word holds.
    ///
    /// Nothing here grows with the sample count, which is the whole reason it
    /// can be asked of a long video: `needed` is at most `MAX_WITNESSES`,
    /// neither array is filled past it -- a side that has already reached the
    /// quota cannot get wider in any way that matters -- and the walk stops the
    /// moment both sides have. So a row costs at most `2 * needed` comparisons
    /// against arrays that fit in a few cache lines, against a graph build and
    /// an augmenting search if it is skipped.
    fn wide_sides_are_wide_enough(window: &[Aligned], i: u32, j: u32, needed: usize) -> bool {
        let (mut seen_a, mut seen_b) = ([0u32; MAX_WITNESSES], [0u32; MAX_WITNESSES]);
        let (mut wide_a, mut wide_b) = (0usize, 0usize);
        for (wi, wj) in eligible(window, i, j) {
            if wide_a < needed && !seen_a[..wide_a].contains(&wi) {
                seen_a[wide_a] = wi;
                wide_a += 1;
            }
            if wide_b < needed && !seen_b[..wide_b].contains(&wj) {
                seen_b[wide_b] = wj;
                wide_b += 1;
            }
            if wide_a >= needed && wide_b >= needed {
                return true;
            }
        }
        false
    }

    /// Whether an already-created vertex has taken all the edges it is allowed.
    /// `NO_VERTEX` is a sample this window has not reached yet, which has no
    /// edges at all and so is never at the cap.
    fn at_degree_cap(&self, slot: u32, side: &[Vertex], needed: usize) -> bool {
        slot != NO_VERTEX && side[slot as usize].degree as usize >= needed
    }

    fn reset(&mut self) {
        // Sized on first use, and grown rather than rebuilt: `reset` restores
        // every slot it touched, so what is already there is already clear.
        if self.slot_a.len() < self.samples_a {
            self.slot_a.resize(self.samples_a, NO_VERTEX);
        }
        if self.slot_b.len() < self.samples_b {
            self.slot_b.resize(self.samples_b, NO_VERTEX);
        }
        for v in &self.side_a {
            self.slot_a[v.sample as usize] = NO_VERTEX;
        }
        for v in &self.side_b {
            self.slot_b[v.sample as usize] = NO_VERTEX;
        }
        self.side_a.clear();
        self.side_b.clear();
        self.edges.clear();
    }

    fn vertex_a(&mut self, sample: u32) -> u32 {
        let slot = &mut self.slot_a[sample as usize];
        if *slot == NO_VERTEX {
            *slot = self.side_a.len() as u32;
            self.side_a.push(Vertex { sample, degree: 0, mate: NO_VERTEX });
        }
        *slot
    }

    fn vertex_b(&mut self, sample: u32) -> u32 {
        let slot = &mut self.slot_b[sample as usize];
        if *slot == NO_VERTEX {
            *slot = self.side_b.len() as u32;
            self.side_b.push(Vertex { sample, degree: 0, mate: NO_VERTEX });
        }
        *slot
    }
}

/// One augmenting step of the matching: find an alternating path from an
/// unmatched A-side vertex to an unmatched B-side one, and flip it.
///
/// Recursion depth is bounded by the number of edges, which `Witnesses::exact`
/// caps at `needed * (needed - 1) + 1` before it ever gets here.
fn augment(
    a: u32,
    edges: &[(u32, u32)],
    side_a: &mut [Vertex],
    side_b: &mut [Vertex],
    seen: &mut [bool],
) -> bool {
    for &(from, to) in edges {
        if from != a || seen[to as usize] {
            continue;
        }
        seen[to as usize] = true;
        let held = side_b[to as usize].mate;
        if held == NO_VERTEX || augment(held, edges, side_a, side_b, seen) {
            side_b[to as usize].mate = a;
            side_a[a as usize].mate = to;
            return true;
        }
    }
    false
}

/// Flag every frame match that enough *other* frame matches agree with about the
/// time offset between the two videos.
///
/// `aligned` is every match out to the loose tolerance. Sorting it by offset
/// puts every match that could witness another one next to it, so one pass with
/// a sliding window settles the whole pair. A witness must differ on BOTH sides
/// from the candidate AND from every other witness: one frame of A matching two
/// neighbouring frames of B says only that B holds a static shot, and a static
/// shot is what admits unrelated footage. Distinctness from each other is what
/// makes the quota mean what the schedule below assumes -- counting MATCHES
/// rather than samples, a 2x2 block of mutually matching frames is four
/// witnesses off two samples a side, so a quota of four could be filled by two
/// coincidences instead of four.
///
/// Which witnesses those are is a maximum bipartite matching over the window --
/// samples of A on one side, samples of B on the other, one edge per frame
/// match -- and the quota is met when that matching reaches `needed`. See
/// `Witnesses::enough`, which decides it without usually having to search.
///
/// How many witnesses are enough comes from `schedule`, i.e. from how far apart
/// the two frames were -- one out to 12 bits, two at 14, three at 16. A single
/// witness is a fine bar for a match that chance produces once in 50 million and
/// a poor one for a match it produces once in six thousand.
///
/// `O(m log m)` in the number of loose matches, against the `O(n * m)` popcount
/// loop that produced them. `m` is small on anything unrelated -- an unrelated
/// frame pair sits ~32 bits apart and never enters this list at all.
///
/// `on_mark` is told of each row it admits, as `(i, j, distance)`: only a row
/// that flags something, since a row whose two samples are already flagged is
/// skipped unjudged. The comparison passes a closure that does nothing;
/// `evidence` listens.
fn corroborated_with<M: FnMut(u32, u32, u32)>(
    aligned: &mut [Aligned],
    (schedule, window): (&[u32], i64),
    matched_a: &mut [bool],
    matched_b: &mut [bool],
    targets: (i64, i64),
    witnesses: &mut Witnesses,
    on_mark: &mut M,
) {
    if aligned.len() < 2 {
        return;
    }

    // In place: the caller's vector is the only copy, and at a wide `-d` on two
    // long videos it is the largest thing this pass holds.
    aligned.sort_unstable();

    // The scratch space is the worker's, not this pass's: it is indexed by
    // sample number, so it only has to be told how far the two fingerprints
    // reach, and the buffers it grew for the last pair are already clear.
    witnesses.begin(matched_a.len(), matched_b.len());

    // `lo` is the first entry still inside the window of `k` and `hi` the first
    // one past it. Both only ever move forward, because `offset` does, so the
    // two of them together are one pass over the slice rather than a walk out
    // from every candidate -- and knowing where the window ENDS is what lets
    // `Witnesses::enough` refuse a candidate on the window's size alone.
    let mut lo = 0usize;
    let mut hi = 0usize;
    for k in 0..aligned.len() {
        let (offset, i, j, distance) = aligned[k];
        while offset - aligned[lo].0 > window {
            lo += 1;
        }
        while hi < aligned.len() && aligned[hi].0 - offset <= window {
            hi += 1;
        }
        // Outside the band this call is responsible for: present as a witness
        // for the matches that are inside it, and judged in the band that holds
        // it, where its own window is complete. Skipping it here is the one
        // safe direction anyway -- a skirt entry can only ever see FEWER
        // witnesses than it is owed, never more. Both edges are advanced
        // first, because they walk the whole slice whatever this step is for.
        // `EVERY_OFFSET` makes this false for every entry.
        if offset < targets.0 || offset >= targets.1 {
            continue;
        }
        let needed = schedule[(distance as usize).min(schedule.len() - 1)];
        if needed == u32::MAX {
            continue;
        }
        // Nothing left to learn about this row. The only thing a verdict here
        // can do is set these two flags, both of them are already set, and
        // `enough` is a pure predicate over a scratch space it clears on entry
        // -- so the search is a no-op whose answer is discarded. Every strict
        // match arrives here already flagged (`for_each_frame_match` marks them
        // on the sweep that collected the list), and every row this loop marks
        // flags a sample pair that later rows touching either half then inherit,
        // so at a loose `-d` this is most of the list. Measured on the 756-file
        // local corpus at `-p 100`, so clustering contributes nothing, three
        // interleaved runs each: the comparison stage is 4.80-4.97 s -> 2.28-2.34 s
        // at `-d 24` and 13.79-14.23 s -> 4.98-5.10 s at `-d 32`, 0.77-0.85 s ->
        // 0.64-0.68 s at `-d 20`, and unmoved at `-d 18` and below, where few
        // rows are settled twice. Whole run at `-d 24 -p 20`, 4.82-4.91 s ->
        // 2.30-2.44 s. Peak RSS is unmoved, and the reports are byte-identical
        // at `-d 0, 4, 8, 12, 16, 18, 20, 24` and `32`.
        //
        // It sits with the other refusals, below the two `while` loops, but it
        // could sit anywhere in the body: `lo` and `hi` are absolute functions
        // of the CURRENT offset rather than increments, and `offset` only ever
        // rises, so a row that skips its own verdict without advancing them
        // leaves the next row to advance them the rest of the way. Below is
        // simply where a reader expects the window to be settled already.
        if matched_a[i as usize] && matched_b[j as usize] {
            continue;
        }
        // Witnesses must be distinct from EACH OTHER as well as from the
        // candidate, on both sides, which is what makes the question a matching
        // rather than a count.
        if witnesses.enough(&aligned[lo..hi], i, j, needed as usize) {
            matched_a[i as usize] = true;
            matched_b[j as usize] = true;
            on_mark(i, j, distance);
        }
    }
}

/// What one measured pair matched, in the terms a person can check: which
/// frames of `a` were matched to which frames of `b`, and where in each the
/// matched footage lies. The desktop app shows it (`found::Comparison`).
///
/// It is the decision `match_overlap` makes, made again for one pair with the
/// matches recorded rather than only flagged: the same tolerances, the same
/// corroboration and the same flags, so every sample it lists is one the
/// coverage figures counted. Kept out of the comparison itself, which runs for
/// every candidate pair and has no use for any of this; it is asked only of
/// the pairs the window will show.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Evidence {
    /// For each matched sample of `a`, in `a`'s order, the closest sample of
    /// `b` it was matched to: `(t_a_ms, t_b_ms, distance)`, each time the
    /// sample's start on its fingerprint's clock.
    pub pairs: Vec<(u32, u32, u32)>,
    /// The matched footage of each side as `(start_ms, end_ms)` stretches, in
    /// order, touching samples merged.
    pub stretches_a: Vec<(u32, u32)>,
    pub stretches_b: Vec<(u32, u32)>,
}

/// `Evidence` for `fp_a` against `fp_b` at `-d max_hamming_dist`.
pub fn evidence(fp_a: &VideoFingerprint, fp_b: &VideoFingerprint, max_hamming_dist: u32) -> Evidence {
    // The pair was measured every way round and kept the orientation that
    // covered more (`better_orientation`), so the frames shown are taken the
    // same way: each orientation of `a` against `b`, and each one-sided
    // orientation of `b` against `a` turned back round, the first best one
    // winning, as in `find_matches_where`.
    let best = |e: &Evidence| {
        let ms = |st: &[(u32, u32)]| st.iter().map(|&(s, e)| (e - s) as u64).sum::<u64>();
        let pct = |ms: u64, total: u32| if total > 0 { ms as f64 / total as f64 } else { 0.0 };
        pct(ms(&e.stretches_a), fp_a.total_ms).max(pct(ms(&e.stretches_b), fp_b.total_ms))
    };
    let mut chosen = evidence_as(fp_a, fp_b, max_hamming_dist);
    for o in &Orientation::all()[1..] {
        let mut candidates = Vec::new();
        if o.fits(fp_a, fp_b) {
            candidates.push(evidence_as(&o.apply(fp_a), fp_b, max_hamming_dist));
        }
        if !o.symmetric() && o.fits(fp_b, fp_a) {
            let e = evidence_as(&o.apply(fp_b), fp_a, max_hamming_dist);
            candidates.push(Evidence {
                pairs: e.pairs.iter().map(|&(tb, ta, d)| (ta, tb, d)).collect(),
                stretches_a: e.stretches_b,
                stretches_b: e.stretches_a,
            });
        }
        for e in candidates {
            if best(&e) > best(&chosen) {
                chosen = e;
            }
        }
    }
    let mut pairs = chosen.pairs;
    pairs.sort_unstable();
    Evidence { pairs, ..chosen }
}

fn evidence_as(fp_a: &VideoFingerprint, fp_b: &VideoFingerprint, max_hamming_dist: u32) -> Evidence {
    let narrow_schedule = witness_schedule();
    let wide_schedule = witness_schedule_of(WIDE_BITS);
    let rules = Rules {
        narrow: Rule { tol: Tolerance::for_distance(max_hamming_dist), schedule: &narrow_schedule },
        wide: Rule { tol: Tolerance::wide(max_hamming_dist), schedule: &wide_schedule },
    };
    let rule = rules.for_pair(fp_a, fp_b);
    let mut matched_a = vec![false; fp_a.valid_hashes.len()];
    let mut matched_b = vec![false; fp_b.valid_hashes.len()];
    // The closest partner of each sample of `a`, of the matches admitted.
    let mut best: Vec<Option<(u32, u32)>> = vec![None; fp_a.valid_hashes.len()];
    let note = |best: &mut Vec<Option<(u32, u32)>>, i: u32, j: u32, d: u32| {
        let slot = &mut best[i as usize];
        if slot.is_none_or(|(_, bd)| d < bd) {
            *slot = Some((j, d));
        }
    };
    let mut witnesses = Witnesses::new();
    admit(fp_a, fp_b, rule, &mut matched_a, &mut matched_b, &mut witnesses, &mut |i, j, d| {
        note(&mut best, i, j, d)
    });
    let pairs = best
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.map(|(j, d)| (fp_a.valid_t_start[i], fp_b.valid_t_start[j as usize], d)))
        .collect();
    Evidence { pairs, stretches_a: stretches(&matched_a, fp_a), stretches_b: stretches(&matched_b, fp_b) }
}

/// The flagged samples of `fp` as stretches of footage: each sample stands for
/// `[t_start, t_end)`, and samples that touch are one stretch.
fn stretches(matched: &[bool], fp: &VideoFingerprint) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (i, _) in matched.iter().enumerate().filter(|(_, &m)| m) {
        let (s, e) = (fp.valid_t_start[i], fp.valid_t_end[i]);
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    out
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
    find_matches_where(fingerprints, None, max_hamming_dist, min_match_percent, min_duration)
}

/// `find_all_matches` for `--against`: only pairs with one file on each side
/// of `library` are measured. A library file is never compared with another
/// library file, nor a scanned file with another scanned file, so the cost is
/// the product of the two sides rather than the square of their sum -- which
/// is the difference between this mode being usable on a large library and
/// not. Every pair it does measure is measured exactly as a scan measures it.
pub fn find_matches_across(
    fingerprints: &[VideoFingerprint],
    library: &[bool],
    max_hamming_dist: u32,
    min_match_percent: f32,
    min_duration: f64,
) -> Vec<Match> {
    find_matches_where(fingerprints, Some(library), max_hamming_dist, min_match_percent, min_duration)
}

fn find_matches_where(
    fingerprints: &[VideoFingerprint],
    sides: Option<&[bool]>,
    max_hamming_dist: u32,
    min_match_percent: f32,
    min_duration: f64,
) -> Vec<Match> {
    let wanted = |a: usize, b: usize| sides.is_none_or(|s| s[a] != s[b]);
    let orientations = Orientation::all();
    let wanted_as =
        |o: usize, q: usize, t: usize| wanted(q.min(t), q.max(t)) && orientations[o].fits(&fingerprints[q], &fingerprints[t]);
    let n = fingerprints.len();
    let total_hashes: usize = fingerprints.iter().map(|fp| fp.valid_hashes.len()).sum();
    // Phase 1 has to be exhaustive at the WIDEST distance phase 2 can admit, not
    // at `-d`: a pair whose only matches are corroborated ones would otherwise
    // never be proposed, and the corroboration rule would be unreachable exactly
    // where it does the most good.
    let narrow_schedule = witness_schedule();
    let wide_schedule = witness_schedule_of(WIDE_BITS);
    let rules = Rules {
        narrow: Rule { tol: Tolerance::for_distance(max_hamming_dist), schedule: &narrow_schedule },
        wide: Rule { tol: Tolerance::wide(max_hamming_dist), schedule: &wide_schedule },
    };
    let widest = rules.widest();
    let masks = probe_masks(probe_radius(widest));

    // Every other way round a pair is measured: `a` mirrored, and `a` read
    // inside each reframing, mirrored or not (see `Orientation`). The plain
    // orientation is `fingerprints` itself.
    let views: Vec<Vec<VideoFingerprint>> = orientations[1..]
        .iter()
        .map(|o| fingerprints.iter().map(|fp| o.apply(fp)).collect())
        .collect();
    let side = |o: usize| -> &[VideoFingerprint] { if o == 0 { fingerprints } else { &views[o - 1] } };

    // One orientation of one pair, `q` read that way round against `t` as it
    // is, reported with the lower index as `a`.
    let measure = |(q, t, o): (usize, usize, usize), witnesses: &mut Witnesses| -> Option<Match> {
        let m = measure_pair(&side(o)[q], &fingerprints[t], (q, t), rules, min_match_percent, min_duration, witnesses)?;
        let m = Match { mirrored: orientations[o].mirrored, reframed: orientations[o].reframed, ..m };
        Some(if q < t { m } else { m.swapped() })
    };

    let route_t = std::time::Instant::now();
    // Every orientation is the same comparison of one library against itself,
    // so the two routes compare per orientation exactly as they do for one.
    // Phase 1 is counted on the run's bar like any other stage: it is most of
    // a warm scan's comparison, and it used to run with nothing on screen.
    let pairs: Vec<(usize, usize, usize)> = if index_is_cheaper(masks.len(), n, total_hashes) {
        let probed: u64 = (0..orientations.len())
            .map(|o| side(o).iter().map(|fp| fp.valid_hashes.len() as u64).sum::<u64>())
            .sum();
        let done = progress::begin_counted(
            progress::Stage::Candidates,
            probed * BLOCKS as u64,
            (orientations.len() * BLOCKS * n) as u64,
        );
        let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
        for (o, orientation) in orientations.iter().enumerate() {
            candidates.extend(
                candidate_pairs_probing(fingerprints, side(o), &masks, widest, !orientation.symmetric(), &done)
                    .into_iter()
                    .map(|(q, t)| (q, t, o)),
            );
        }
        candidates.retain(|&(q, t, o)| wanted_as(o, q, t));
        info!("Index scan produced {} candidate pair(s) in {:.1?}; verifying...", candidates.len(), route_t.elapsed());
        candidates
    } else {
        // Past the radius where probing pays, every pair's hashes are compared
        // outright -- two contiguous runs of u64, which is far cheaper per
        // comparison than a probe -- and only the pairs with some frame inside
        // the widest tolerance go on to be measured, as from the index.
        let row_weights: Vec<Vec<u64>> = orientations
            .iter()
            .enumerate()
            .map(|(o, orientation)| {
                scan_row_weights(fingerprints, side(o), !orientation.symmetric(), &|q, t| wanted_as(o, q, t))
            })
            .collect();
        let done = progress::begin_counted(
            progress::Stage::Candidates,
            row_weights.iter().flatten().sum(),
            (orientations.len() * n) as u64,
        );
        let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
        for (o, orientation) in orientations.iter().enumerate() {
            candidates.extend(
                candidate_pairs_scanning(
                    fingerprints,
                    side(o),
                    rules,
                    !orientation.symmetric(),
                    &|q, t| wanted_as(o, q, t),
                    (&row_weights[o], &done),
                )
                    .into_iter()
                    .map(|(q, t)| (q, t, o)),
            );
        }
        info!(
            "Pairwise scan produced {} candidate pair(s) in {:.1?}; verifying...",
            candidates.len(),
            route_t.elapsed()
        );
        candidates
    };

    if shutdown_requested() {
        return Vec::new();
    }

    // A pair costs the product of its two sample counts, so that is what it
    // weighs on the bar; the count says pairs.
    let cost = |&(q, t, o): &(usize, usize, usize)| {
        (side(o)[q].valid_hashes.len() as u64 * fingerprints[t].valid_hashes.len() as u64).max(1)
    };
    let done = progress::begin_counted(
        progress::Stage::Verifying,
        pairs.iter().map(cost).sum(),
        pairs.len() as u64,
    );

    let mut matches = pairs
        .into_par_iter()
        .map_init(
            || (Batch::new(&done), Witnesses::new()),
            |(batch, witnesses), task| {
                let m = measure(task, witnesses).map(|m| (task.2, m));
                batch.add(cost(&task));
                m
            },
        )
        .flatten()
        .collect::<Vec<(usize, Match)>>();

    // A pair proposed several ways round keeps the orientation that matched
    // most, the earliest in `Orientation::all` on a tie -- the plain one first.
    matches.sort_by_key(|(o, m)| (m.a, m.b, *o));
    let mut out: Vec<Match> = Vec::with_capacity(matches.len());
    for (_, m) in matches {
        match out.last_mut() {
            Some(last) if (last.a, last.b) == (m.a, m.b) => {
                *last = better_orientation(Some(*last), Some(m)).unwrap();
            }
            _ => out.push(m),
        }
    }
    out
}

/// One way round a pair can be measured: the first file's pictures as they
/// are, mirrored, or read inside a reframing (`fingerprint::REFRAMINGS`), each
/// against the second file as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Orientation {
    mirrored: bool,
    reframed: Option<Reframing>,
}

impl Orientation {
    /// Every orientation, plain first: that order is the tie-break.
    fn all() -> Vec<Orientation> {
        let mut out = vec![
            Orientation { mirrored: false, reframed: None },
            Orientation { mirrored: true, reframed: None },
        ];
        for r in REFRAMINGS {
            for mirrored in [false, true] {
                out.push(Orientation { mirrored, reframed: Some(r) });
            }
        }
        out
    }

    /// Whether measuring `a` this way round against `b` is the same question
    /// as measuring `b` this way round against `a`. Mirroring is its own
    /// inverse, so a mirrored `a` against `b` is a mirrored `b` against `a`; a
    /// reframing is not -- a zoomed `a` against `b` finds `b` cut down from
    /// `a`, and the other way round finds the reverse.
    fn symmetric(&self) -> bool {
        self.reframed.is_none()
    }

    /// Whether `t` could be `q` read this way round at all. A reframing that
    /// cuts a landscape picture down to another shape is only measured against
    /// a file of that shape: a 9:16 strip against a portrait file, a 4:3 cut
    /// against a landscape file nearer 4:3 than the source's own shape (on a
    /// log scale, so 720x576 and 720x480 storage both count). Measured against
    /// everything, a cut is one more chance for a look-alike and nothing else.
    fn fits(&self, q: &VideoFingerprint, t: &VideoFingerprint) -> bool {
        let aspect = |fp: &VideoFingerprint| fp.width as f64 / fp.height.max(1) as f64;
        match self.reframed {
            Some(Reframing::Strip(_)) => t.height > t.width,
            Some(Reframing::Cut43) => {
                let (at, aq) = (aspect(t), aspect(q));
                at > 1.0 && (at / (4.0 / 3.0)).ln().abs() < (at / aq).ln().abs()
            }
            _ => true,
        }
    }

    /// `fp` read this way round: its hashes replaced by the orientation's, and
    /// its samples narrowed to those the reframing applies to. A fingerprint
    /// that lacks them (a fixture, or a reframing that applies to none of its
    /// samples) comes back with no samples and simply matches nothing.
    fn apply(&self, fp: &VideoFingerprint) -> VideoFingerprint {
        let mut o = VideoFingerprint {
            valid_hashes: Vec::new(),
            valid_t_start: Vec::new(),
            valid_t_end: Vec::new(),
            mirrored_hashes: Vec::new(),
            detail_hashes: Vec::new(),
            mirrored_detail_hashes: Vec::new(),
            reframed: Vec::new(),
            path: fp.path.clone(),
            codec: String::new(),
            ..*fp
        };
        let n = fp.valid_hashes.len();
        match self.reframed {
            None => {
                let (hashes, detail) = if self.mirrored {
                    (&fp.mirrored_hashes, &fp.mirrored_detail_hashes)
                } else {
                    (&fp.valid_hashes, &fp.detail_hashes)
                };
                if hashes.len() != n {
                    return o;
                }
                o.valid_hashes = hashes.clone();
                o.valid_t_start = fp.valid_t_start.clone();
                o.valid_t_end = fp.valid_t_end.clone();
                if detail.len() == n {
                    o.detail_hashes = detail.clone();
                }
            }
            Some(r) => {
                let Some(v) = REFRAMINGS.iter().position(|&x| x == r) else { return o };
                let Some(rf) = fp.reframed.get(v) else { return o };
                let (h, d) = if self.mirrored { (1, 3) } else { (0, 2) };
                for (&s, hs) in rf.samples.iter().zip(&rf.hashes) {
                    let s = s as usize;
                    if s >= n {
                        continue;
                    }
                    o.valid_hashes.push(hs[h]);
                    o.detail_hashes.push(hs[d]);
                    o.valid_t_start.push(fp.valid_t_start[s]);
                    o.valid_t_end.push(fp.valid_t_end[s]);
                }
            }
        }
        o
    }
}

/// Of the two orientations of one pair, the one whose better side is better
/// covered; the plain one on a tie, so a pair that matches both ways (a
/// symmetric picture) is never reported as mirrored for nothing.
fn better_orientation(normal: Option<Match>, mirrored: Option<Match>) -> Option<Match> {
    match (normal, mirrored) {
        (Some(n), Some(m)) => {
            if m.coverage_a.max(m.coverage_b) > n.coverage_a.max(n.coverage_b) { Some(m) } else { Some(n) }
        }
        (n, m) => n.or(m),
    }
}

/// Phase 2 for one pair: measure it exactly, then apply both gates. `None` when
/// the pair fails either of them, or when the run is shutting down. `fp_a` may
/// be a re-oriented copy (`Orientation::apply`); the match is reported plain,
/// and the caller records the orientation.
fn measure_pair(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    (v_a, v_b): (usize, usize),
    rules: Rules,
    min_match_percent: f32,
    min_duration: f64,
    witnesses: &mut Witnesses,
) -> Option<Match> {
    if shutdown_requested() {
        return None;
    }
    let rule = rules.for_pair(fp_a, fp_b);
    let (pct_a, pct_b, span_a, span_b) = match_overlap(fp_a, fp_b, rule, witnesses);

    // A pair that shares NOTHING is not a match at any setting, and that has to
    // be said separately from the gate: `-p 0` makes `< min_match_percent` false
    // for every pair alive, including the ones whose overlap measured zero. That
    // is not a hypothetical -- `index_is_cheaper` hands the direct route every
    // pair in the library, so `-d 4 -p 0` over 110 unrelated files reported one
    // group of 110 with `0.00` matched seconds on all 109 rows it condemned,
    // while `-d 0 -p 0` over the same files reported none. The routes are
    // supposed to agree on every pair (see `find_all_matches`); a performance
    // heuristic was deciding a correctness question, and with `--delete` armed
    // it decided it destructively.
    //
    // Written as a separate test rather than by turning the gate into `<=`,
    // because `-p 100` must keep admitting a pair that is fully covered.
    if pct_a.max(pct_b) <= 0.0 || pct_a.max(pct_b) < min_match_percent {
        return None;
    }

    if min_duration > 0.0 {
        // Measured exactly the way the report measures it -- see
        // `overlap_seconds`, which both sides now share. A pair whose overlap
        // cannot be measured at all (neither file reported a runtime) cannot
        // clear a floor stated in seconds, so `None` fails the gate rather than
        // passing it.
        let cleared = overlap_seconds(pct_a, fp_a.duration, pct_b, fp_b.duration)
            .is_some_and(|secs| secs >= min_duration);
        if !cleared {
            return None;
        }
    }

    Some(Match { a: v_a, b: v_b, coverage_a: pct_a, coverage_b: pct_b, span_a, span_b, mirrored: false, reframed: None })
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
            mirrored_hashes: Vec::new(),
            detail_hashes: Vec::new(),
            mirrored_detail_hashes: Vec::new(),
            reframed: Vec::new(),
        }
    }

    /// Only the runtime matters for the shared-duration arithmetic.
    #[test]
    fn test_across_measures_only_pairs_with_one_file_on_each_side() {
        // Four copies of one fingerprint: a scan links all six pairs, and
        // `--against` only the four that cross from scanned to library.
        let hashes: Vec<u64> = (0..12).map(|i| 0x0F0F_0F0F_0F0F_0F0F_u64.rotate_left(i * 5)).collect();
        let fps: Vec<VideoFingerprint> = (0..4).map(|_| mock_fp_with_hashes(hashes.clone(), 12)).collect();
        assert_eq!(find_all_matches(&fps, 4, 0.2, 0.0).len(), 6);

        let library = [true, true, false, false];
        let mut pairs: Vec<(usize, usize)> =
            find_matches_across(&fps, &library, 4, 0.2, 0.0).iter().map(|m| (m.a, m.b)).collect();
        pairs.sort_unstable();
        assert_eq!(pairs, vec![(0, 2), (0, 3), (1, 2), (1, 3)]);
    }

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

    /// Both tolerances pinned to the same value, so a test can state which
    /// samples pair up without the corroboration rule having a say. `-d` never
    /// never produces one of these -- the two sides are always a fixed gap
    /// apart; it is the phase-2 behaviour these tests are about, not the
    /// mapping from the flag.
    fn exactly(bits: u32) -> Tolerance {
        Tolerance { strict: bits, loose: bits, cap: None }
    }

    /// The same thing as a whole `Rule`, for the tests that call the pair-level
    /// entry points rather than the corroboration pass directly.
    fn exact_rule<'a>(bits: u32, schedule: &'a [u32; HASH_BITS as usize + 1]) -> Rule<'a> {
        Rule { tol: exactly(bits), schedule }
    }

    #[test]
    fn test_the_widest_tolerance_is_chance_itself() {
        // Every threshold in this file is a multiple of sigma, and the ceiling
        // on `-d` is the one at zero: the mean of the unrelated-pair
        // distribution, 32 bits on a 64-bit hash.
        assert_eq!(max_hamming_distance(), HASH_BITS / 2);

        // Which is to say a match at the ceiling is no evidence at all -- about
        // half of all unrelated frame pairs land within it, so it carries no
        // orders of magnitude. Four bits tighter it is already carrying some.
        assert!(chance_orders_of_magnitude(max_hamming_distance()) < 0.5);
        assert!(chance_orders_of_magnitude(max_hamming_distance() - 4) > 0.5);
    }

    #[test]
    fn test_blocks_partition_the_hash_without_gaps_or_overlap() {
        let h = 0x0123_4567_89AB_CDEFu64;
        assert_eq!(block_of(h, 0), 0x0123);
        assert_eq!(block_of(h, 1), 0x4567);
        assert_eq!(block_of(h, 2), 0x89AB);
        assert_eq!(block_of(h, 3), 0xCDEF);
    }

    /// The pairs phase 1 proposes on its own, with no gate and no phase 2. Tests
    /// about the index have to ask it directly: `find_all_matches` is free to
    /// skip it entirely (see `index_is_cheaper`), and on fingerprints this small
    /// it always does.
    fn proposed(fps: &[VideoFingerprint], max_hamming_dist: u32) -> Vec<(usize, usize)> {
        candidate_pairs(fps, &probe_masks(probe_radius(max_hamming_dist)), max_hamming_dist)
    }

    #[test]
    fn test_probe_radius_is_derived_from_the_tolerance() {
        // The pigeonhole rule: three differing bits cannot cover four blocks, so
        // one block always matches exactly and no neighbour lookup is needed.
        assert_eq!(probe_radius(0), 0);
        assert_eq!(probe_radius(3), 0, "a tight tolerance probes exact bins only");
        assert_eq!(probe_radius(4), 1);
        assert_eq!(probe_radius(7), 1);
        // And it keeps widening, with nothing to stop it. A radius that lagged
        // the tolerance would make the index a filter rather than an enumerator.
        assert_eq!(probe_radius(8), 2);
        assert_eq!(probe_radius(63), 15);
        assert_eq!(probe_radius(HASH_BITS), BLOCK_BITS as u32, "the widest -d probes every bin");
    }

    #[test]
    fn test_probe_masks_are_exactly_the_patterns_within_the_radius() {
        // 1, 17, 137: sum of C(16, i) up to the radius. These are the multiplier
        // on every single bin lookup in the scan, and the reason a loose enough
        // tolerance is better served by comparing every pair outright.
        assert_eq!(probe_masks(0), vec![0u16]);
        assert_eq!(probe_masks(1).len(), 17);
        assert_eq!(probe_masks(2).len(), 137);
        assert!(probe_masks(2).iter().all(|m| m.count_ones() <= 2));
        assert_eq!(probe_masks(BLOCK_BITS as u32).len(), BINS, "radius 16 is every bin");
    }

    #[test]
    fn test_the_index_is_abandoned_once_the_probe_outgrows_the_library() {
        // A library the size of the local corpus: 727 videos, 9k hashes. The
        // index earns its keep at the tolerances anyone should be using and
        // stops earning it at the loose end, where the probe set has grown wide
        // enough to touch most of the library.
        let (videos, hashes) = (727, 8966);
        let pays = |d: u32| index_is_cheaper(probe_masks(probe_radius(d)).len(), videos, hashes);

        assert!(pays(4), "the default tolerance must never pay for the whole triangle");
        assert!(pays(8));
        assert!(!pays(16), "radius 4 is 2517 keys a block; the direct route is measured faster");
        assert!(!pays(32));

        // And an empty library keeps the index rather than walking a triangle of
        // pairs that cannot match: there are no hashes to compare.
        assert!(index_is_cheaper(probe_masks(probe_radius(64)).len(), videos, 0));
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
    fn test_a_lone_frame_match_beyond_the_tolerance_is_refused() {
        // One sample each, 3 bits apart, at `-d 2`. It is inside the loose
        // tolerance but there is no second match to place the two videos at an
        // offset, so nothing corroborates it and the tolerance is all it has.
        let fps = vec![
            mock_fp_sampled(vec![0x0000_0000_0000_0000], 1000),
            mock_fp_sampled(vec![0x0000_0000_0000_0007], 1000),
        ];

        assert!(find_all_matches(&fps, 2, 1.0, 0.0).is_empty());
        // The same pair at a tolerance that covers it outright.
        assert!(!find_all_matches(&fps, 4, 1.0, 0.0).is_empty());
    }

    #[test]
    fn test_frame_matches_agreeing_on_an_offset_are_admitted_beyond_the_tolerance() {
        // Three samples each, every one 3 bits from its opposite number and ~32
        // from everything else, all at the same offset. At `-d 2` none of them
        // stands on its own, and all of them witness each other.
        let a: Vec<u64> = (0..3).map(distinct_hash).collect();
        let b: Vec<u64> = a.iter().map(|h| h ^ 0b111).collect();
        let fps = vec![mock_fp_sampled(a, 2000), mock_fp_sampled(b, 2000)];

        let matches = find_all_matches(&fps, 2, 1.0, 0.0);
        assert_eq!(pairs(&matches), vec![(0, 1)]);
        assert!(
            (matches[0].coverage_a - 1.0).abs() < 1e-6,
            "every sample was corroborated, so all of the runtime is covered"
        );
    }

    #[test]
    fn test_frame_matches_disagreeing_on_an_offset_do_not_witness_each_other() {
        // Both of A's samples match B's single sample at 4 bits, but they place
        // the videos ten seconds apart from each other. That is one static-
        // looking frame reaching two moments, not two moments lining up, and it
        // is exactly the shape the alignment window exists to refuse.
        let fps = vec![
            mock_fp_sampled(vec![0x0000, 0x00FF], 10_000),
            mock_fp_sampled(vec![0x000F], 10_000),
        ];

        assert!(find_all_matches(&fps, 2, 1.0, 0.0).is_empty());
        // Still found once the tolerance covers the distance on its own.
        assert!(!find_all_matches(&fps, 4, 1.0, 0.0).is_empty());
    }

    #[test]
    fn test_the_tolerance_pair_moves_monotonically_with_the_flag() {
        // `-d` may only ever admit more as it rises, whichever side of the
        // corroboration rule a frame match falls on.
        let mut previous = Tolerance { strict: 0, loose: 0, cap: None };
        for d in 0..=HASH_BITS {
            let tol = Tolerance::for_distance(d);
            assert!(tol.strict <= tol.loose, "-d {d}: strict must not exceed loose");
            assert!(tol.loose >= d, "-d {d}: never stricter than the flag asked for");
            assert!(tol.strict >= previous.strict && tol.loose >= previous.loose);
            previous = tol;
        }
        // Both sides move with every rung and neither ever saturates: the gap
        // is constant, so `-d 12` is not `-d 4` with extra steps.
        assert_eq!(Tolerance::for_distance(0), Tolerance { strict: 0, loose: 6, cap: None });
        assert_eq!(Tolerance::for_distance(4), Tolerance { strict: 4, loose: 10, cap: None });
        assert_eq!(Tolerance::for_distance(12), Tolerance { strict: 12, loose: 18, cap: None });
        assert_eq!(Tolerance::for_distance(20), Tolerance { strict: 20, loose: 26, cap: None });
        // ...except against the hash width, past which a tolerance means nothing.
        assert_eq!(
            Tolerance::for_distance(HASH_BITS),
            Tolerance { strict: HASH_BITS, loose: HASH_BITS, cap: None }
        );
    }

    #[test]
    fn test_the_thresholds_are_stated_in_sigma_so_they_track_the_hash_width() {
        // The two constants are multiples of the unrelated-pair standard
        // deviation, which is `sqrt(HASH_BITS) / 2` -- 4 bits for this hash. A
        // wider hash rescales them instead of silently keeping a tuning that was
        // only ever right for 64 bits.
        assert!((sigma() - 4.0).abs() < 1e-9);
        assert_eq!(sigma_below_chance(EVIDENCE_ANCHOR_SIGMA), 12);
        assert_eq!((CORROBORATION_SLACK_SIGMA * sigma()).round() as u32, 6);
    }

    #[test]
    fn test_one_witness_is_exactly_enough_at_the_anchor() {
        // `CORROBORATION_BUDGET` is calibrated to put the anchor distance at one
        // witness, so the two constants are a single calibration. If this fails,
        // they have drifted apart and the schedule means something else.
        let schedule = witness_schedule();
        let anchor = sigma_below_chance(EVIDENCE_ANCHOR_SIGMA) as usize;
        for (distance, needed) in schedule.iter().enumerate().take(anchor + 1) {
            assert_eq!(*needed, 1, "a match at {distance} bits should need exactly one witness");
        }
        assert!(schedule[anchor + 4] > 1, "4 bits further out is 175x likelier by chance");
    }

    #[test]
    fn test_the_witness_schedule_never_softens_as_the_distance_grows() {
        // A further-apart pair of frames is weaker evidence, never stronger, so
        // the quota may only rise. Nothing downstream re-checks this: the
        // schedule is read straight out of the table per match.
        let schedule = witness_schedule();
        for distance in 1..=HASH_BITS as usize {
            assert!(
                schedule[distance] >= schedule[distance - 1],
                "{distance} bits asks for fewer witnesses than {} does",
                distance - 1
            );
        }
        assert_eq!(
            schedule[HASH_BITS as usize],
            u32::MAX,
            "two hashes as far apart as they can be are not evidence at any cluster size"
        );
    }

    /// A fingerprint whose samples sit at the given millisecond marks, each
    /// standing for the picture until the next one.
    fn mock_fp_at(hashes: Vec<u64>, times: Vec<u32>) -> VideoFingerprint {
        assert_eq!(hashes.len(), times.len());
        let last = *times.last().expect("a fingerprint needs at least one sample");
        let ends: Vec<u32> = times.iter().skip(1).copied().chain(std::iter::once(last + 1000)).collect();
        let total = *ends.last().unwrap();
        let mut fp = mock_fp_with_hashes(hashes, total);
        fp.valid_t_start = times;
        fp.valid_t_end = ends;
        fp.duration = total as f64 / 1000.0;
        fp
    }

    #[test]
    fn test_samples_closer_than_the_alignment_window_are_folded_into_one() {
        let hashes: Vec<u64> = (0..6).map(distinct_hash).collect();
        let mut fp = mock_fp_at(hashes.clone(), vec![0, 100, 200, 600, 1200, 1250]);
        fp.detail_hashes = hashes.iter().map(|h| !h).collect();
        fp.mirrored_hashes = hashes.iter().map(|h| h.rotate_left(1)).collect();
        fp.reframed = vec![crate::fingerprint::Reframed { samples: vec![1, 3, 5], hashes: vec![[1; 4], [3; 4], [5; 4]] }];
        let footage: u32 = fp.valid_t_start.iter().zip(&fp.valid_t_end).map(|(s, e)| e - s).sum();

        thin_samples(&mut fp);

        assert_eq!(fp.valid_t_start, vec![0, 600, 1200]);
        assert_eq!(fp.valid_t_end, vec![600, 1200, 2250], "a kept sample stands for what it absorbed");
        assert_eq!(fp.valid_hashes, vec![hashes[0], hashes[3], hashes[4]]);
        assert_eq!(fp.detail_hashes, vec![!hashes[0], !hashes[3], !hashes[4]]);
        assert_eq!(fp.mirrored_hashes, vec![hashes[0].rotate_left(1), hashes[3].rotate_left(1), hashes[4].rotate_left(1)]);
        // A reframed sample keeps its hashes under its new index; one folded away goes.
        assert_eq!(fp.reframed[0].samples, vec![1]);
        assert_eq!(fp.reframed[0].hashes, vec![[3; 4]]);
        let after: u32 = fp.valid_t_start.iter().zip(&fp.valid_t_end).map(|(s, e)| e - s).sum();
        assert_eq!(after, footage, "no footage is lost from coverage");
    }

    #[test]
    fn test_a_file_sampled_no_faster_than_the_window_is_untouched() {
        let hashes: Vec<u64> = (0..5).map(distinct_hash).collect();
        let step = ALIGNMENT_TOLERANCE_MS as u32;
        let mut fp = mock_fp_sampled(hashes.clone(), step);
        fp.reframed = vec![crate::fingerprint::Reframed { samples: vec![0, 4], hashes: vec![[7; 4], [8; 4]] }];
        thin_samples(&mut fp);
        assert_eq!(fp.valid_hashes, hashes);
        assert_eq!(fp.valid_t_start, (0..5).map(|i| i * step).collect::<Vec<_>>());
        assert_eq!(fp.reframed[0].samples, vec![0, 4]);
    }

    #[test]
    fn test_a_copy_sampled_at_every_frame_still_covers_its_source_once_thinned() {
        // The source keyframes every 2 s; the copy is the same footage with a
        // sample every 40 ms (an all-intra encode), each carrying the picture
        // of the 2 s stretch it falls in.
        let pictures: Vec<u64> = (0..10).map(distinct_hash).collect();
        let source = mock_fp_sampled(pictures.clone(), 2000);
        let dense_hashes: Vec<u64> = (0..500).map(|i| pictures[i * 40 / 2000]).collect();
        let mut dense = mock_fp_sampled(dense_hashes, 40);
        thin_samples(&mut dense);
        let step = ALIGNMENT_TOLERANCE_MS as u32;
        assert!(dense.valid_t_start.windows(2).all(|w| w[1] - w[0] >= step && w[1] - w[0] < step + 40));
        assert_eq!(dense.valid_t_end.last(), Some(&20_000), "the last sample still reaches the end");
        let m = find_all_matches(&[source, dense], 4, 0.2, 0.0);
        assert_eq!(m.len(), 1);
        assert!(m[0].coverage_a > 0.99 && m[0].coverage_b > 0.99, "{} {}", m[0].coverage_a, m[0].coverage_b);
    }

    /// What the corroboration pass makes of one pair, and how many matches it
    /// had to hold at once to do it.
    fn corroborate_with_cap(
        fp_a: &VideoFingerprint,
        fp_b: &VideoFingerprint,
        tol: Tolerance,
        cap: usize,
    ) -> (Vec<bool>, Vec<bool>, usize) {
        let mut matched_a = vec![false; fp_a.valid_hashes.len()];
        let mut matched_b = vec![false; fp_b.valid_hashes.len()];
        let held = corroborate_pair(
            fp_a,
            fp_b,
            Rule { tol, schedule: &witness_schedule() },
            &mut matched_a,
            &mut matched_b,
            cap,
            &mut Witnesses::new(),
        );
        (matched_a, matched_b, held)
    }

    #[test]
    fn test_cutting_the_offset_axis_into_bands_changes_no_verdict() {
        // The property the whole banding argument rests on: a pair judged in
        // one pass and the same pair judged a band at a time must reach the
        // same verdict on every frame match, including the ones whose witnesses
        // sit on the far side of a band edge. That is what the skirt is for --
        // drop it and the matches near every cut lose their corroboration.
        //
        // 40 samples of A re-encoded into B at 10 bits, at 20 offsets 700 ms
        // apart with two matches 120 ms apart at each -- so every match has
        // exactly ONE witness, its partner, and nothing else is within the
        // alignment window. A band edge that falls between a pair (and with a
        // cap of one match per band, every bucket boundary is an edge) takes
        // that witness away unless the band carries its skirt. Plus one match
        // at an offset entirely of its own, which no cap may rescue.
        let tol = Tolerance::for_distance(4);
        assert!(tol.loose >= 10 && tol.strict < 10, "the matches below have to be loose ones");
        let ten_bits = 0x3FFu64;

        let a_hashes: Vec<u64> = (0..60).map(distinct_hash).collect();
        let a_times: Vec<u32> = (0..60).map(|i| 50_000 + i * 2000).collect();

        let mut b_hashes: Vec<u64> = (100..160).map(distinct_hash).collect();
        let mut b_times: Vec<u32> = a_times.clone();
        for i in 0..40u32 {
            b_hashes[i as usize] = a_hashes[i as usize] ^ ten_bits;
            let offset = (i / 2) * 700 + (i % 2) * 120;
            b_times[i as usize] = a_times[i as usize] - offset + 30_000;
        }
        for i in 40..60 {
            b_times[i] = a_times[i] + 60_000;
        }
        // The lone one: a real match at 10 bits, half a minute away from any
        // offset the rest of them agree on.
        b_hashes[59] = a_hashes[59] ^ ten_bits;

        let a = mock_fp_at(a_hashes, a_times);
        let b = mock_fp_at(b_hashes, b_times);

        let (whole_a, whole_b, held) = corroborate_with_cap(&a, &b, tol, usize::MAX);
        assert_eq!(held, 41, "40 corroborating matches and one lone one");
        assert!(whole_a[..40].iter().all(|&m| m), "the drifting run is corroborated");
        assert!(!whole_a[59], "and a match nothing agrees with is not");

        // Small enough that a band is a handful of matches, so the cuts land
        // inside witness windows rather than between them.
        for cap in [1, 2, 3, 7, 20] {
            let (banded_a, banded_b, held) = corroborate_with_cap(&a, &b, tol, cap);
            assert_eq!(banded_a, whole_a, "banded at {cap}, A read differently");
            assert_eq!(banded_b, whole_b, "banded at {cap}, B read differently");
            assert!(held < 41, "banding at {cap} held the whole pair anyway");
        }
    }

    #[test]
    fn test_a_long_static_pair_is_not_held_in_memory_all_at_once() {
        // Two recordings of a static scene: every sample of one is within the
        // loose tolerance of every sample of the other, so the list of matches
        // is |A| x |B| and nothing about the number of SAMPLES bounds it. At
        // 4,000 samples a side -- two hours of keyframes, not a contrived
        // input -- that list is 372 MB for the one pair, on a stage that runs a
        // pair per rayon worker.
        let tol = Tolerance::for_distance(4);
        let n = 300usize;
        let ten_bits = 0x3FFu64;
        let times: Vec<u32> = (0..n as u32).map(|i| i * 2000).collect();
        let a = mock_fp_at(vec![0xAAAA_AAAA_AAAA_AAAA; n], times.clone());
        let b = mock_fp_at(vec![0xAAAA_AAAA_AAAA_AAAA ^ ten_bits; n], times);

        let (whole_a, whole_b, held) = corroborate_with_cap(&a, &b, tol, usize::MAX);
        assert_eq!(held, n * n, "every sample matches every sample");
        assert!(whole_a.iter().all(|&m| m) && whole_b.iter().all(|&m| m));

        let cap = 1000;
        let (banded_a, banded_b, held) = corroborate_with_cap(&a, &b, tol, cap);
        assert_eq!((banded_a, banded_b), (whole_a, whole_b), "banding changed the verdict");
        // A band stops one bucket short of the cap and carries a skirt of one
        // alignment window on each side, edges included; a bucket here is one
        // time offset, which this pair has at most `n` matches at, and the
        // window is the 2 s the samples are spaced at, so the skirt reaches one
        // offset below the band and two above it.
        assert!(
            held <= cap + 3 * n,
            "held {held} matches at once against a cap of {cap}"
        );
    }

    #[test]
    fn test_a_far_match_needs_more_corroboration_than_a_close_one() {
        // Every sample of A matches its opposite number in B at 16 bits and
        // nothing else, all at one offset -- so each match has exactly (n - 1)
        // witnesses and the pair turns on the quota alone.
        let far = 0xFFFFu64; // 16 bits set
        let pair_of = |n: u64| {
            let a: Vec<u64> = (0..n).map(distinct_hash).collect();
            let b: Vec<u64> = a.iter().map(|h| h ^ far).collect();
            vec![mock_fp_sampled(a, 1000), mock_fp_sampled(b, 1000)]
        };

        // `-d 10` puts 16 bits on the corroborated side (10 + 6), so the quota
        // is what decides. Each sample has exactly (n - 1) witnesses.
        let needed = witness_schedule()[16];
        assert!(find_all_matches(&pair_of(needed as u64), 10, 1.0, 0.0).is_empty(),
                "{needed} witnesses are required and only {} were available", needed - 1);
        assert!(!find_all_matches(&pair_of(needed as u64 + 1), 10, 1.0, 0.0).is_empty());
        // A tighter `-d` puts 16 bits out of reach entirely, however many agree.
        assert!(find_all_matches(&pair_of(needed as u64 + 4), 8, 1.0, 0.0).is_empty());
        // A looser one takes each match on its own distance, quota irrelevant.
        assert!(!find_all_matches(&pair_of(2), 16, 1.0, 0.0).is_empty());
    }

    #[test]
    fn test_a_wider_tolerance_never_takes_a_frame_match_away() {
        // The whole promise of `-d`: both tolerances move up with it, so every
        // frame match admitted at one rung is admitted at the next. Greedy
        // witness selection broke it, and only ever this way round -- a match
        // the LOOSER setting admits sorts earlier in the offset order, is taken
        // first, and consumes both sides of two witnesses that would otherwise
        // have paid the quota between them.
        //
        // Three samples a side. The candidate is a0/b0 at 14 bits, so it needs
        // two witnesses; a1/b1 and a2/b2 are the two, at 14 bits each and at
        // the same offset. The blocker is a1/b2 at 16 bits -- out of reach at
        // `-d 8` (loose 14), in reach at `-d 10` (loose 16) -- placed 100 ms
        // earlier so it is the first thing the greedy walk sees. It touches a1
        // and b2, which is one side of each real witness.
        let a = vec![0xFFFF_FFFF_0000_0000u64, 0x0u64, 0x3Fu64];
        let b = vec![0xFFFF_FFFF_000F_FFC0u64, 0x000F_FFC0u64, 0x0003_FFFCu64];
        let distance = |x: u64, y: u64| (x ^ y).count_ones();
        assert_eq!(
            [distance(a[0], b[0]), distance(a[1], b[1]), distance(a[2], b[2])],
            [14, 14, 14]
        );
        assert_eq!(distance(a[1], b[2]), 16, "the blocker");
        assert_eq!(witness_schedule()[14], 2, "the candidate has to need both witnesses");
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                assert!(i == j || (i, j) == (1, 2) || distance(x, y) > 16, "stray match {i}/{j}");
            }
        }

        let fp_a = mock_fp_at(a, vec![15_000, 25_000, 25_100]);
        let fp_b = mock_fp_at(b, vec![10_000, 20_000, 20_100]);
        for d in [8u32, 10] {
            let (marked_a, marked_b, _) =
                corroborate_with_cap(&fp_a, &fp_b, Tolerance::for_distance(d), usize::MAX);
            assert_eq!(marked_a, [true; 3], "-d {d} lost a sample of A");
            assert_eq!(marked_b, [true; 3], "-d {d} lost a sample of B");
        }
    }

    #[test]
    fn test_no_rung_of_the_tolerance_reports_less_than_the_one_below_it() {
        // The same promise, asked of the whole ladder over pseudo-random pairs
        // rather than of one constructed case. Coverage is what `-p` gates on,
        // so it is coverage that has to be monotone; a rung that reports less
        // than the one below it is the bug this is here to catch.
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..200 {
            // Hashes clustered around a common ancestor, so the pair holds
            // matches at a spread of distances rather than none at all.
            let base = next();
            let noisy = |bits: u32, r: &mut dyn FnMut() -> u64| {
                let mut h = base;
                for _ in 0..bits {
                    h ^= 1u64 << (r() % 64);
                }
                h
            };
            let n = 3 + (next() % 6) as usize;
            let a: Vec<u64> = (0..n).map(|_| noisy(6 + (next() % 12) as u32, &mut next)).collect();
            let b: Vec<u64> = (0..n).map(|_| noisy(6 + (next() % 12) as u32, &mut next)).collect();
            // Times close enough together that everything is in one alignment
            // window, which is where the witness rule actually has to decide.
            let times = |r: &mut dyn FnMut() -> u64| -> Vec<u32> {
                let mut t = 10_000u32;
                (0..n)
                    .map(|_| {
                        t += 60 + (r() % 200) as u32;
                        t
                    })
                    .collect()
            };
            let fp_a = mock_fp_at(a, times(&mut next));
            let fp_b = mock_fp_at(b, times(&mut next));

            let mut previous: Option<(f32, f32)> = None;
            for d in (0..=20).step_by(2) {
                let (cov_a, cov_b, _, _) =
                    match_overlap(
                        &fp_a,
                        &fp_b,
                        Rule { tol: Tolerance::for_distance(d), schedule: &witness_schedule() },
                        &mut Witnesses::new(),
                    );
                if let Some((was_a, was_b)) = previous {
                    assert!(
                        cov_a >= was_a && cov_b >= was_b,
                        "case {case}: -d {d} covers ({cov_a}, {cov_b}) where -d {} covered ({was_a}, {was_b})",
                        d - 2
                    );
                }
                previous = Some((cov_a, cov_b));
            }
        }
    }

    /// What the window is shown of a pair is what the comparison measured:
    /// the stretches `evidence` returns are exactly the footage the coverage
    /// figures counted, their ends are the spans, every sample it lists is
    /// one of them, and no distance it lists is past the loose tolerance.
    #[test]
    fn test_the_evidence_shown_is_the_match_that_was_measured() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let covered = |st: &[(u32, u32)]| st.iter().map(|&(s, e)| (e - s) as u64).sum::<u64>();
        for case in 0..200 {
            let base = next();
            let noisy = |bits: u32, r: &mut dyn FnMut() -> u64| {
                let mut h = base;
                for _ in 0..bits {
                    h ^= 1u64 << (r() % 64);
                }
                h
            };
            let n = 3 + (next() % 8) as usize;
            let a: Vec<u64> = (0..n).map(|_| noisy(4 + (next() % 14) as u32, &mut next)).collect();
            let b: Vec<u64> = (0..n).map(|_| noisy(4 + (next() % 14) as u32, &mut next)).collect();
            let times = |r: &mut dyn FnMut() -> u64| -> Vec<u32> {
                let mut t = 10_000u32;
                (0..n)
                    .map(|_| {
                        t += 60 + (r() % 400) as u32;
                        t
                    })
                    .collect()
            };
            let fp_a = mock_fp_at(a, times(&mut next));
            let fp_b = mock_fp_at(b, times(&mut next));
            for d in [0, 4, 8, 12, 16] {
                let rule = Rule { tol: Tolerance::for_distance(d), schedule: &witness_schedule() };
                let (cov_a, cov_b, span_a, span_b) = match_overlap(&fp_a, &fp_b, rule, &mut Witnesses::new());
                let ev = evidence(&fp_a, &fp_b, d);
                let pct = |ms: u64, fp: &VideoFingerprint| (ms as f32 / fp.total_ms as f32).min(1.0);
                assert_eq!(pct(covered(&ev.stretches_a), &fp_a), cov_a, "case {case} -d {d}: a's footage");
                assert_eq!(pct(covered(&ev.stretches_b), &fp_b), cov_b, "case {case} -d {d}: b's footage");
                let envelope = |st: &[(u32, u32)]| st.first().map(|f| Span { start_ms: f.0, end_ms: st.last().unwrap().1 });
                assert_eq!(envelope(&ev.stretches_a), span_a, "case {case} -d {d}");
                assert_eq!(envelope(&ev.stretches_b), span_b, "case {case} -d {d}");
                // Every listed pair is a matched sample of a with a partner inside
                // b's matched footage, within the loose tolerance or, for a sample
                // an alignment rescued, one slack step past it. A sample can be
                // matched with no pair of its own: footage credited from the other
                // side's matched sample (`credit_matched_spans`).
                let reach = rule.tol.loose + (CORROBORATION_SLACK_SIGMA * sigma()).round() as u32;
                let flagged_a = fp_a.valid_t_start.iter().zip(&fp_a.valid_t_end).filter(|(s, e)| ev.stretches_a.iter().any(|st| st.0 <= **s && **e <= st.1)).count();
                assert!(ev.pairs.len() <= flagged_a, "case {case} -d {d}: at most one pair per matched sample of a");
                for &(ta, tb, dist) in &ev.pairs {
                    assert!(dist <= reach);
                    assert!(ev.stretches_a.iter().any(|st| st.0 <= ta && ta < st.1));
                    assert!(ev.stretches_b.iter().any(|st| st.0 <= tb && tb < st.1), "case {case} -d {d}: {tb} is not matched footage of b");
                }
            }
        }
    }

    #[test]
    fn test_a_match_stands_for_the_sparser_side_s_span_on_both_sides() {
        // a is sampled every half second, b every five seconds, over the same
        // footage at offset 0: each of b's samples is one of a's frames. One
        // b hash stands for five seconds of b, and the a frames inside those
        // five seconds are the same footage, so a is covered as fully as b.
        let a_times: Vec<u32> = (0..21u32).map(|k| k * 500).collect();
        let a_hashes: Vec<u64> = (0..21u64).map(distinct_hash).collect();
        let a = mock_fp_at(a_hashes.clone(), a_times.clone());
        let b = mock_fp_at(vec![a_hashes[0], a_hashes[10], a_hashes[20]], vec![a_times[0], a_times[10], a_times[20]]);
        let rule = Rule { tol: Tolerance::for_distance(4), schedule: &witness_schedule() };
        let (cov_a, cov_b, _, _) = match_overlap(&a, &b, rule, &mut Witnesses::new());
        assert_eq!(cov_b, 1.0);
        assert!(cov_a > 0.99, "a is credited b's spans: {cov_a}");

        // One match alone stands for nothing beyond its own sample.
        let lone = mock_fp_at(vec![a_hashes[10]], vec![a_times[10]]);
        let (cov_a, _, _, _) = match_overlap(&a, &lone, rule, &mut Witnesses::new());
        assert!((cov_a - 500.0 / a.total_ms as f32).abs() < 1e-6, "{cov_a}");
    }

    #[test]
    fn test_two_apart_does_not_depend_on_which_match_comes_first() {
        assert!(two_apart(&[(1, 2), (3, 4)]));
        // (1, 4) shares a sample with each of the others, and they still differ.
        assert!(two_apart(&[(1, 4), (1, 2), (3, 4)]));
        assert!(!two_apart(&[(1, 2), (1, 3), (1, 4)]), "one sample of a against three of b");
        assert!(!two_apart(&[(1, 2)]));
    }

    #[test]
    fn test_a_cut_reframing_is_measured_only_against_a_file_of_its_shape() {
        let shaped = |w: u32, h: u32| {
            let mut fp = mock_fp_at(vec![distinct_hash(1)], vec![0]);
            fp.width = w;
            fp.height = h;
            fp
        };
        let (wide, portrait, sd, pal, ntsc, hd) =
            (shaped(1920, 1080), shaped(1080, 1920), shaped(640, 480), shaped(720, 576), shaped(720, 480), shaped(854, 480));
        let strip = Orientation { mirrored: false, reframed: Some(Reframing::Strip(3)) };
        assert!(strip.fits(&wide, &portrait));
        assert!(!strip.fits(&wide, &hd), "a strip of a landscape picture is portrait");
        let cut = Orientation { mirrored: false, reframed: Some(Reframing::Cut43) };
        for t in [&sd, &pal, &ntsc] {
            assert!(cut.fits(&wide, t), "{}x{}", t.width, t.height);
        }
        assert!(!cut.fits(&wide, &hd), "a 16:9 file rounding a pixel narrower is not a 4:3 cut");
        assert!(!cut.fits(&wide, &portrait));
        let zoom = Orientation { mirrored: false, reframed: Some(Reframing::Zoom) };
        assert!(zoom.fits(&wide, &hd));
    }

    #[test]
    fn test_a_near_miss_between_two_aligned_matches_is_rescued_and_one_alone_is_not() {
        // a: five frames a second apart. b: the same five at a 3 s offset, the
        // two ends exact and the middle three 12 bits off -- past -d 4's loose
        // 10, inside one slack step (6) of it. The ends anchor the offset, so
        // the middle is rescued; with only one exact end there is no chain.
        let hashes: Vec<u64> = (0..5u64).map(distinct_hash).collect();
        let times: Vec<u32> = (0..5u32).map(|i| 10_000 + i * 1000).collect();
        let off = |h: u64, i: u64| if i == 0 || i == 4 { h } else { h ^ 0xFFF };
        let a = mock_fp_at(hashes.clone(), times.clone());
        let b = mock_fp_at(
            hashes.iter().enumerate().map(|(i, &h)| off(h, i as u64)).collect(),
            times.iter().map(|t| t + 3000).collect(),
        );
        let rule = Rule { tol: Tolerance::for_distance(4), schedule: &witness_schedule() };
        // Each sample stands for one second of a 15 s runtime.
        let seconds = |cov: f32| (cov * a.total_ms as f32 / 1000.0).round() as u32;
        let (cov_a, _, _, _) = match_overlap(&a, &b, rule, &mut Witnesses::new());
        assert_eq!(seconds(cov_a), 5, "the middle three should be rescued");

        let mut one_end = b.clone();
        one_end.valid_hashes[4] ^= 0xFFFF_FFFF;
        let (cov_a, _, _, _) = match_overlap(&a, &one_end, rule, &mut Witnesses::new());
        assert_eq!(seconds(cov_a), 1, "one anchor is no alignment");
    }

    #[test]
    fn test_witnesses_are_paired_off_rather_than_counted_up() {
        // Two witnesses are available and a greedy walk cannot see both: the
        // first entry it meets touches one side of each of them. Nothing about
        // the quota changes -- three distinct samples a side are involved, and
        // two of them can be paired with two of the other -- so the candidate
        // is corroborated.
        let mut scratch = Witnesses::new();
        scratch.begin(4, 9);
        let window: Vec<Aligned> = vec![
            (0, 1, 2, 16), // the blocker: takes a1 and b2 between them
            (0, 1, 1, 14),
            (0, 2, 2, 14),
            (0, 0, 0, 14), // the candidate itself, which is never its own witness
        ];
        assert!(scratch.enough(&window, 0, 0, 2));
        // Three is genuinely out of reach: only a1/a2 and b1/b2 are free of the
        // candidate, so no matching there can exceed two.
        assert!(!scratch.enough(&window, 0, 0, 3));
        // And one side repeating itself is not two witnesses however many
        // matches it produces.
        // The candidate's own row is in every window `corroborated` builds, so
        // it is in every window here too -- see the precondition on `enough`.
        let static_shot: Vec<Aligned> =
            std::iter::once((0, 0, 0, 14)).chain((1..9).map(|j| (0, 1, j, 14))).collect();
        assert!(scratch.enough(&static_shot, 0, 0, 1));
        assert!(!scratch.enough(&static_shot, 0, 0, 2));

        // The case that needs the augmenting search rather than the edge count:
        // three witnesses are there (a1/b3, a2/b2, a3/b1) and the pairing has to
        // be rearranged to find them, on too few edges for the Koenig bound to
        // settle it. The greedy walk takes a1/b1 and a2/b2 and then has nothing
        // left for a3.
        let mut scratch = Witnesses::new();
        scratch.begin(5, 5);
        let rearrange: Vec<Aligned> = vec![
            (0, 0, 0, 14), // the candidate
            (0, 1, 1, 14),
            (0, 2, 2, 14),
            (0, 1, 3, 14),
            (0, 3, 1, 14),
        ];
        assert!(rearrange.len() - 1 < 3 * (3 - 1) + 1, "the edge count must not settle it");
        assert!(scratch.enough(&rearrange, 0, 0, 3));
        assert!(!scratch.enough(&rearrange, 0, 0, 4));
    }

    /// The largest matching in a window, written the slow obvious way: every
    /// eligible edge in an adjacency list, then one augmenting search per
    /// A-side sample. Nothing it does is shared with `Witnesses`, which is the
    /// point -- it is the definition the four cheap exits stand in for.
    fn largest_matching(window: &[Aligned], i: u32, j: u32) -> usize {
        let mut edges: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
        for &(_, wi, wj, _) in window {
            if wi != i && wj != j {
                let to = edges.entry(wi).or_default();
                if !to.contains(&wj) {
                    to.push(wj);
                }
            }
        }

        fn walk(
            a: u32,
            edges: &std::collections::BTreeMap<u32, Vec<u32>>,
            mate: &mut std::collections::BTreeMap<u32, u32>,
            seen: &mut std::collections::BTreeSet<u32>,
        ) -> bool {
            for &b in edges.get(&a).map(|v| v.as_slice()).unwrap_or(&[]) {
                if !seen.insert(b) {
                    continue;
                }
                let held = mate.get(&b).copied();
                if held.is_none() || walk(held.unwrap(), edges, mate, seen) {
                    mate.insert(b, a);
                    return true;
                }
            }
            false
        }

        let mut mate = std::collections::BTreeMap::new();
        let mut size = 0;
        for &a in edges.keys() {
            let mut seen = std::collections::BTreeSet::new();
            if walk(a, &edges, &mut mate, &mut seen) {
                size += 1;
            }
        }
        size
    }

    #[test]
    fn test_the_cheap_exits_agree_with_the_matching_they_stand_in_for() {
        // `enough` is four refusals and a search, and every one of the refusals
        // is an arithmetic claim about a matching nobody has built yet: the
        // window is too small to hold the quota, greedy already paid it, greedy
        // is over half of it, each side is wide enough. This walks random
        // windows past all four and asks the definition instead.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut scratch = Witnesses::new();
        let mut reached_the_search = 0;
        for _ in 0..4000 {
            let samples = 2 + rng() % 7;
            let count = (rng() % 10) as usize;
            // The candidate's own row is always present, which is the
            // precondition the window-size exit rests on.
            let mut window: Vec<Aligned> = vec![(0, 0, 0, 14)];
            for _ in 0..count {
                let (wi, wj) = ((rng() % samples) as u32, (rng() % samples) as u32);
                // One row per sample pair, as `for_each_frame_match` produces.
                if !window.iter().any(|&(_, ei, ej, _)| (ei, ej) == (wi, wj)) {
                    window.push((0, wi, wj, 14));
                }
            }
            let largest = largest_matching(&window, 0, 0);
            // Every window is asked twice: once with the files declared at
            // their real width, which takes the bitmask route through
            // `side_is_wide_enough`, and once with both declared past
            // `u64::BITS` samples, which takes the collecting one. The verdict
            // is a property of the window, so the two routes have to agree --
            // and the wide one is the route that returned `true` unconditionally
            // until the exits below were made to stand on their own.
            for declared in [samples as usize, samples as usize + 64] {
                scratch.begin(declared, declared);
                for needed in 1..=5usize {
                    assert_eq!(
                        scratch.enough(&window, 0, 0, needed),
                        largest >= needed,
                        "needed {needed}, largest {largest}, declared {declared}, window {window:?}"
                    );
                    if largest >= needed && needed >= 3 {
                        reached_the_search += 1;
                    }
                }
            }
        }
        assert!(reached_the_search > 0, "the corpus never exercised the augmenting search");
    }

    #[test]
    fn test_a_file_too_wide_for_the_bitmask_still_needs_witnesses_on_both_sides() {
        // Every edge here is the same sample of A, so no matching can exceed
        // one however many edges there are, and side A is narrower than any
        // quota above that. Both files are declared past `u64::BITS` samples,
        // which used to be the width at which `side_is_wide_enough` gave up and
        // said yes -- so this question fell through to the graph, and the
        // `debug_assert` standing there said a state the give-up made reachable
        // was impossible. It fired: `cargo test` on a pair like this panicked at
        // any `-d` whose quota reaches 3 (16 bits, so `-d 10` and up), where the
        // release build reached the right answer by the `untried` exit instead.
        // The predicate is exact at every width now and settles it before a
        // graph is built; the exit it used to be an assert about is still there
        // for the samples the degree cap leaves without a vertex at all.
        let mut scratch = Witnesses::new();
        scratch.begin(70, 70);
        let window: Vec<Aligned> = vec![
            (0, 0, 0, 14), // the candidate
            (0, 5, 1, 14),
            (0, 5, 2, 14),
            (0, 5, 3, 14),
        ];
        assert!(scratch.enough(&window, 0, 0, 1));
        assert!(!scratch.enough(&window, 0, 0, 2));
        assert_eq!(largest_matching(&window, 0, 0), 1);

        // And asked of the predicate directly, because the verdict alone cannot
        // tell the two apart: a route that gave up and said `true` would still
        // reach this answer through the graph, which is exactly why the old one
        // survived every test there was. Only the cheap refusal itself is
        // evidence that a long file is settled before a graph is built.
        assert!(!Witnesses::wide_sides_are_wide_enough(&window, 0, 0, 2));
        assert!(Witnesses::wide_sides_are_wide_enough(&window, 0, 0, 1));
    }

    /// Which samples the corroboration pass marks, written the slow obvious
    /// way: every loose frame match judged on its own, against the window of
    /// every match within the pair's `alignment_window` of it and the definition of
    /// a matching in `largest_matching`. Shares no code with `corroborated` --
    /// no sliding window, no scratch space, no order between the rows.
    fn marked_the_slow_way(
        fp_a: &VideoFingerprint,
        fp_b: &VideoFingerprint,
        tol: Tolerance,
    ) -> (Vec<bool>, Vec<bool>) {
        let schedule = witness_schedule();
        let mut aligned: Vec<Aligned> = Vec::new();
        let mut marked_a = vec![false; fp_a.valid_hashes.len()];
        let mut marked_b = vec![false; fp_b.valid_hashes.len()];
        for (i, &h_a) in fp_a.valid_hashes.iter().enumerate() {
            for (j, &h_b) in fp_b.valid_hashes.iter().enumerate() {
                let distance = (h_a ^ h_b).count_ones();
                if distance > tol.loose {
                    continue;
                }
                if distance <= tol.strict {
                    marked_a[i] = true;
                    marked_b[j] = true;
                }
                let offset = fp_a.valid_t_start[i] as i64 - fp_b.valid_t_start[j] as i64;
                aligned.push((offset, i as u32, j as u32, distance));
            }
        }

        for &(offset, i, j, distance) in &aligned {
            let needed = schedule[distance.min(HASH_BITS) as usize];
            if needed == u32::MAX {
                continue;
            }
            let window: Vec<Aligned> = aligned
                .iter()
                .copied()
                .filter(|&(o, _, _, _)| (o - offset).abs() <= alignment_window(fp_a, fp_b))
                .collect();
            if largest_matching(&window, i, j) >= needed as usize {
                marked_a[i as usize] = true;
                marked_b[j as usize] = true;
            }
        }
        (marked_a, marked_b)
    }

    #[test]
    fn test_a_row_already_settled_is_skipped_without_changing_what_the_pass_marks() {
        // `corroborated` walks the rows in offset order and skips any whose two
        // samples are BOTH already flagged, because the only thing a verdict
        // there can do is flag them again. That skip is what makes a loose `-d`
        // affordable -- most of the list is settled by the time it is reached --
        // and it is sound only because a row's verdict depends on the window
        // and never on the flags. Nothing about the answer may move with it, so
        // this asks the definition instead: every row judged on its own, in no
        // particular order, against a window built from scratch.
        //
        // What it cannot catch is the skip being deleted, and it is not meant
        // to: a pass that judges every row reaches the same answer the slow
        // way does. The failure it is here for is a skip that fires on a row
        // still capable of flagging something -- `||` for `&&`, the flags read
        // before the sweep that sets the strict ones, a row skipped on side A
        // alone -- each of which loses a mark this asks for by name.
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for round in 0..40 {
            let n = 24usize;
            let a_hashes: Vec<u64> = (0..n as u64).map(distinct_hash).collect();
            let a_times: Vec<u32> = (0..n as u32).map(|i| 10_000 + i * 400).collect();

            // B re-encodes a random subset of A at a random distance, at time
            // offsets drawn from a handful of clusters -- so the alignment
            // windows overlap heavily and one sample of A can reach several of
            // B, which is the shape that makes the matching non-trivial.
            let mut b_hashes: Vec<u64> = (100..100 + n as u64).map(distinct_hash).collect();
            let mut b_times: Vec<u32> = a_times.clone();
            for i in 0..n {
                if rng() % 4 == 0 {
                    continue;
                }
                let bits = (rng() % 19) as u32;
                let mut mask = 0u64;
                while mask.count_ones() < bits {
                    mask |= 1 << (rng() % 64);
                }
                b_hashes[i] = a_hashes[i] ^ mask;
                let cluster = (rng() % 3) as i64 * 900;
                let jitter = (rng() % 400) as i64;
                b_times[i] = (a_times[i] as i64 - 5_000 + cluster + jitter) as u32;
            }
            b_times.sort_unstable();

            let a = mock_fp_at(a_hashes, a_times);
            let b = mock_fp_at(b_hashes, b_times);

            for d in [4u32, 10, 12, 16, 20] {
                let tol = Tolerance::for_distance(d);
                let (got_a, got_b, _) = corroborate_with_cap(&a, &b, tol, usize::MAX);
                let (want_a, want_b) = marked_the_slow_way(&a, &b, tol);
                assert_eq!(got_a, want_a, "round {round}, -d {d}, side A");
                assert_eq!(got_b, want_b, "round {round}, -d {d}, side B");
            }
        }
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

        // Each unordered pair exactly once, always with the lower index first.
        assert_eq!(proposed(&fps, 0), vec![(0, 1), (0, 2), (1, 2)]);
        assert_eq!(pairs(&find_all_matches(&fps, 0, 1.0, 0.0)), vec![(0, 1), (0, 2), (1, 2)]);
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

        assert_eq!(proposed(&fps, 3), vec![(0, 1)]);
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

        assert_eq!(proposed(&fps, 4), vec![(0, 1)]);
        assert!(
            proposed(&fps, 3).is_empty(),
            "and four bits apart is genuinely outside a three-bit tolerance"
        );
    }

    #[test]
    fn test_the_index_proposes_a_pair_no_single_block_agrees_on() {
        // Three differing bits in EVERY block, total 12. Nothing here is within
        // radius 1 of anything, and under the old cap this pair was unreachable
        // for the index -- it was proposed only because the two files also
        // shared an identical frame, and phase 2 then recovered the rest.
        //
        // With the radius derived from the tolerance, floor(12 / 4) = 3 reaches
        // it directly: no shared frame, no rescue, still proposed.
        let a = 0xFFFF_FFFF_FFFF_FFFFu64;
        let b = a ^ 0x0007_0007_0007_0007;
        assert_eq!((a ^ b).count_ones(), 12);
        assert_eq!(probe_radius(12), 3);

        let fps = vec![mock_fp_with_hashes(vec![a], 1), mock_fp_with_hashes(vec![b], 1)];

        assert_eq!(proposed(&fps, 12), vec![(0, 1)]);
        assert!(
            proposed(&fps, 11).is_empty(),
            "twelve bits apart is outside an eleven-bit tolerance, and radius 2 cannot see it"
        );
    }

    #[test]
    fn test_phase_two_counts_frames_the_index_only_had_to_propose() {
        // The index answers "could these two overlap"; it never answers "by how
        // much", and the count of hashes it happened to hit is not that figure.
        // Here the pair is proposed off the frame the two share, and only phase
        // 2's exhaustive comparison sees that the OTHER frame matches too.
        //
        // Demanding 100% overlap: reachable only if both frames are counted, so
        // index-only accounting would have scored this pair at 50% and dropped
        // it.
        let shared = 0x0000_0000_0000_0000u64;
        let far_a = 0xFFFF_FFFF_FFFF_FFFFu64;
        let far_b = far_a ^ 0x0007_0007_0007_0007;

        let fps = vec![
            mock_fp_with_hashes(vec![shared, far_a], 2),
            mock_fp_with_hashes(vec![shared, far_b], 2),
        ];

        assert_eq!(pairs(&find_all_matches(&fps, 12, 1.0, 0.0)), vec![(0, 1)]);
    }

    #[test]
    fn test_both_routes_propose_the_same_pairs_at_every_tolerance() {
        // The exhaustiveness claim, stated as a property rather than as an
        // argument: whatever the index proposes has to be exactly what comparing
        // every pair of videos finds, at every rung of `-d`. This is what lets
        // `find_all_matches` choose between the two on cost alone.
        //
        // Hashes chosen to sit at awkward distances -- one block, several
        // blocks, one bit in each -- so the pigeonhole rule is actually loaded.
        let fps: Vec<VideoFingerprint> = [
            0x0000_0000_0000_0000u64,
            0x0000_0000_0000_00FFu64,
            0x0001_0001_0001_0001u64,
            0x0007_0007_0007_0007u64,
            0x00FF_0000_0000_0000u64,
            0xFFFF_FFFF_FFFF_FFFFu64,
        ]
        .iter()
        .map(|&h| mock_fp_with_hashes(vec![h], 1))
        .collect();

        for d in 0..=HASH_BITS {
            let mut direct: Vec<(usize, usize)> = Vec::new();
            for a in 0..fps.len() {
                for b in (a + 1)..fps.len() {
                    let (h_a, h_b) = (fps[a].valid_hashes[0], fps[b].valid_hashes[0]);
                    if (h_a ^ h_b).count_ones() <= d {
                        direct.push((a, b));
                    }
                }
            }

            assert_eq!(proposed(&fps, d), direct, "the index missed a pair at -d {}", d);
        }
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
    fn test_the_gate_switched_off_still_refuses_a_pair_that_shares_nothing() {
        // `-p 0` means "no floor", not "no measurement". Written as `<` alone
        // the gate read `0.0 < 0.0`, which is false, so every pair phase 2 was
        // handed became a Match -- and on the direct route phase 2 is handed the
        // whole library. A scan of 110 unrelated files reported one group of
        // 110, with 0.00 matched seconds on all 109 rows it marked DELETE.
        let fps: Vec<VideoFingerprint> =
            (0..4).map(|i| mock_fp_with_hashes(vec![distinct_hash(i)], 1)).collect();
        let masks = probe_masks(probe_radius(Tolerance::for_distance(4).widest())).len();
        assert!(
            !index_is_cheaper(masks, fps.len(), fps.len()),
            "this library has to take the direct route for the test to mean anything"
        );

        assert!(find_all_matches(&fps, 4, 0.0, 0.0).is_empty(), "unrelated files are not a group");

        // And the flag keeps the meaning it is documented to have: with the
        // floor off, any overlap at all is enough, however small.
        let host = mock_fp_sampled((0..100).map(distinct_hash).collect(), 10);
        let clip = mock_fp_sampled(vec![distinct_hash(7)], 10);
        assert_eq!(
            pairs(&find_all_matches(&[host, clip], 0, 0.0, 0.0)),
            vec![(0, 1)],
            "1% coverage must still clear a floor of 0"
        );
    }

    #[test]
    fn test_both_routes_agree_when_the_gate_is_switched_off() {
        // The routes are chosen on cost alone, so anything they disagree about
        // is decided by a performance heuristic -- and at `-p 0` they disagreed
        // about the whole library. Big enough that `index_is_cheaper` says yes,
        // which is the arm the small libraries above can never reach.
        let mut fps: Vec<VideoFingerprint> =
            (0..600).map(|i| mock_fp_with_hashes(vec![distinct_hash(i)], 1)).collect();
        // One genuine pair, so the test cannot pass by finding nothing at all.
        fps.push(mock_fp_with_hashes(vec![distinct_hash(3)], 1));

        let tol = Tolerance::for_distance(4);
        let hashes: usize = fps.iter().map(|fp| fp.valid_hashes.len()).sum();
        assert!(
            index_is_cheaper(probe_masks(probe_radius(tol.widest())).len(), fps.len(), hashes),
            "this library has to take the index route for the test to mean anything"
        );

        assert_eq!(pairs(&find_all_matches(&fps, 4, 0.0, 0.0)), vec![(3, 600)]);
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

        assert_near(idx.best_link_in_group(0, &group, &fps).map(|l| l.matched_seconds), 600.0);
        assert_near(idx.best_link_in_group(1, &group, &fps).map(|l| l.matched_seconds), 600.0);
        // 2 has nothing better than its three seconds, and still reports them.
        assert_near(idx.best_link_in_group(2, &group, &fps).map(|l| l.matched_seconds), 3.0);
    }

    #[test]
    fn test_a_pair_that_was_never_compared_is_skipped_rather_than_erasing_the_figure() {
        // A chain: 0-1 and 1-2 matched, 0 and 2 never did -- the ordinary shape
        // of a star group around 1. 0's
        // row reports what it shares with 1, the link that exists. Treating the
        // absent 0-2 pair as unknown would blank the column, and treating it as
        // zero would claim a comparison nobody made.
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

        assert_near(idx.best_link_in_group(0, &group, &fps).map(|l| l.matched_seconds), 600.0);
        assert_near(idx.best_link_in_group(1, &group, &fps).map(|l| l.matched_seconds), 600.0);
        assert_near(idx.best_link_in_group(2, &group, &fps).map(|l| l.matched_seconds), 300.0);
    }

    #[test]
    fn test_a_file_with_no_measurable_link_reports_unknown() {
        // Not the same as sharing nothing: the figure was never obtained.
        let fps = vec![mock_fp_lasting(600.0), mock_fp_lasting(600.0)];
        let idx = MatchIndex::new(vec![]);

        assert_eq!(idx.best_link_in_group(0, &[0, 1], &fps).map(|l| l.matched_seconds), None);
    }

    #[test]
    fn test_the_span_locates_a_clip_inside_its_host() {
        // The headline case for the feature: a 3s clip cut from the middle of a
        // 10s host. The clip is all of itself, so its own span is its whole
        // runtime; the host's span is where the clip sits INSIDE it, which is
        // the number that could not be read off any other column.
        let host = mock_fp_sampled((0..10).map(distinct_hash).collect(), 1000);
        let clip = mock_fp_sampled((4..7).map(distinct_hash).collect(), 1000);

        let (_, _, span_clip, span_host) =
            match_overlap(&clip, &host, exact_rule(0, &witness_schedule()), &mut Witnesses::new());

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

        let (coverage_a, _, span_a, _) =
            match_overlap(&a, &b, exact_rule(0, &witness_schedule()), &mut Witnesses::new());

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
        let links = idx.links_of(0, &[0, 1, 2], &fps);

        assert_eq!(links.len(), 2);
        assert_eq!(links[0].other, 1);
        assert_near(Some(links[0].matched_seconds), 600.0);
        assert_eq!(links[1].other, 2);
        assert_near(Some(links[1].matched_seconds), 6.0);
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
            let links = MatchIndex::new(matches).links_of(0, &[0, 1, 2], &fps);
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

        let (_, _, span_a, span_b) =
            match_overlap(&a, &b, exact_rule(0, &witness_schedule()), &mut Witnesses::new());

        assert_eq!(span_a, None);
        assert_eq!(span_b, None);
    }
}
