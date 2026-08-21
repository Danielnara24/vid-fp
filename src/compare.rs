use crate::fingerprint::VideoFingerprint;
use crate::utils::shutdown_requested;
use indicatif::{ProgressBar, ProgressStyle};
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
///
/// It is also where the clustering stage stops being able to answer at all. Just
/// below chance the graph is dense but incomplete, which is combinatorial and
/// gets refused in seconds by the ceilings in `clustering`; above it the graph
/// is nearly COMPLETE, which has few enough maximal cliques to slip under those
/// ceilings while taking minutes of quadratic pivot work to establish. Refusing
/// the flag there is cheaper and clearer than teaching every later stage to cope
/// with an input that cannot produce an answer worth having.
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
    (HASH_BITS as f64).sqrt() / 2.0
}

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
    let mean = HASH_BITS as f64 / 2.0;
    (mean - sigmas * sigma()).round().max(0.0) as u32
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
fn chance_orders_of_magnitude(distance: u32) -> f64 {
    let mut term = 1.0f64; // C(HASH_BITS, 0)
    let mut sum = 1.0f64;
    for i in 1..=distance.min(HASH_BITS) {
        term *= (HASH_BITS - i + 1) as f64 / i as f64;
        sum += term;
    }
    HASH_BITS as f64 * std::f64::consts::LOG10_2 - sum.log10()
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
    let anchor = sigma_below_chance(EVIDENCE_ANCHOR_SIGMA);
    let budget = CORROBORATION_BUDGET * chance_orders_of_magnitude(anchor);
    let mut schedule = [u32::MAX; HASH_BITS as usize + 1];
    for (distance, needed) in schedule.iter_mut().enumerate() {
        let each = chance_orders_of_magnitude(distance as u32);
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
/// raising the tolerance can only ever admit more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tolerance {
    strict: u32,
    loose: u32,
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
        }
    }

    /// The widest distance any frame match can be admitted at -- what phase 1
    /// has to be exhaustive to, and what the probe radius is derived from.
    fn widest(&self) -> u32 {
        self.loose
    }
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
}

impl MatchIndex {
    pub fn new(matches: Vec<Match>) -> Self {
        let mut links = HashMap::with_capacity(matches.len() * 2);

        for m in matches {
            links.insert((m.a, m.b), Link { coverage: m.coverage_a, span: m.span_a });
            links.insert((m.b, m.a), Link { coverage: m.coverage_b, span: m.span_b });
        }

        Self { links }
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
    /// duplicate in two cliques matched files in both, and a row reported under
    /// one of them must not name a file the reader cannot see beside it. So the
    /// neighbour list is filtered to the group being printed.
    ///
    /// Inside a clique that filter is the only thing doing any work -- every
    /// other member was measured against the subject by construction, so the
    /// result always has `group.len() - 1` entries. It is written as a lookup
    /// rather than assumed, because the assumption is exactly what a change to
    /// the clustering rule would break silently: a group whose members were not
    /// all compared with each other simply reports fewer links.
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
                })
            })
            .collect();

        links.sort_by(|x, y| {
            y.matched_seconds
                .partial_cmp(&x.matched_seconds)
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
fn candidate_pairs(
    fingerprints: &[VideoFingerprint],
    masks: &[u16],
    max_hamming_dist: u32,
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
    tol: Tolerance,
    schedule: &[u32; HASH_BITS as usize + 1],
) -> (f32, f32, Option<Span>, Option<Span>) {
    let mut matched_a = vec![false; fp_a.valid_hashes.len()];
    let mut matched_b = vec![false; fp_b.valid_hashes.len()];

    if tol.loose > tol.strict {
        corroborate_pair(
            fp_a,
            fp_b,
            tol,
            schedule,
            &mut matched_a,
            &mut matched_b,
            MAX_ALIGNED_MATCHES,
        );
    } else {
        // Nothing to corroborate, so nothing is worth remembering: the strict
        // pass marks everything this pair is going to get. Only `-d` at the
        // width of the hash and the tests reach this.
        for_each_frame_match(fp_a, fp_b, tol, &mut matched_a, &mut matched_b, |_| {});
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
            let distance = (h_a ^ h_b).count_ones();
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
fn corroborate_pair(
    fp_a: &VideoFingerprint,
    fp_b: &VideoFingerprint,
    tol: Tolerance,
    schedule: &[u32; HASH_BITS as usize + 1],
    matched_a: &mut [bool],
    matched_b: &mut [bool],
    cap: usize,
) -> usize {
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
        corroborated(&mut aligned, schedule, matched_a, matched_b, EVERY_OFFSET);
        return held;
    }

    // The whole pair does not fit. What was collected is a prefix of it and is
    // no use for anything, since a band needs its own witnesses in full.
    aligned = Vec::new();

    let bands = plan_bands(fp_a, fp_b, tol, matched_a, matched_b, cap);
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
            if entry.0 >= lo.saturating_sub(ALIGNMENT_TOLERANCE_MS)
                && entry.0 <= hi.saturating_add(ALIGNMENT_TOLERANCE_MS)
            {
                aligned.push(entry);
            }
        });
        held = held.max(aligned.len());
        corroborated(&mut aligned, schedule, matched_a, matched_b, (lo, hi));
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
    let width = ALIGNMENT_TOLERANCE_MS.max(span / MAX_BUCKETS + 1);
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

/// Flag every frame match that enough *other* frame matches agree with about the
/// time offset between the two videos.
///
/// `aligned` is every match out to the loose tolerance. Sorting it by offset
/// puts every match that could witness another one next to it, so one pass with
/// a sliding window settles the whole pair. A witness must differ on BOTH sides:
/// one frame of A matching two neighbouring frames of B says only that B holds a
/// static shot, and a static shot is what admits unrelated footage.
///
/// How many witnesses are enough comes from `schedule`, i.e. from how far apart
/// the two frames were -- one out to 12 bits, two at 14, three at 16. A single
/// witness is a fine bar for a match that chance produces once in 50 million and
/// a poor one for a match it produces once in six thousand.
///
/// `O(m log m)` in the number of loose matches, against the `O(n * m)` popcount
/// loop that produced them. `m` is small on anything unrelated -- an unrelated
/// frame pair sits ~32 bits apart and never enters this list at all.
fn corroborated(
    aligned: &mut [Aligned],
    schedule: &[u32; HASH_BITS as usize + 1],
    matched_a: &mut [bool],
    matched_b: &mut [bool],
    targets: (i64, i64),
) {
    if aligned.len() < 2 {
        return;
    }

    // In place: the caller's vector is the only copy, and at a wide `-d` on two
    // long videos it is the largest thing this pass holds.
    aligned.sort_unstable();

    // `lo` is the first entry still inside the window of `k`, and it only ever
    // moves forward, so finding the left edge is linear over the whole scan. The
    // right edge is walked per entry, and the walk stops as soon as the quota is
    // met -- which on a genuine pair is within the next entry or two.
    let mut lo = 0usize;
    for k in 0..aligned.len() {
        let (offset, i, j, distance) = aligned[k];
        while offset - aligned[lo].0 > ALIGNMENT_TOLERANCE_MS {
            lo += 1;
        }
        // Outside the band this call is responsible for: present as a witness
        // for the matches that are inside it, and judged in the band that holds
        // it, where its own window is complete. Skipping it here is the one
        // safe direction anyway -- a skirt entry can only ever see FEWER
        // witnesses than it is owed, never more. `lo` is advanced first,
        // because it walks the whole slice whatever this step is for.
        // `EVERY_OFFSET` makes this false for every entry.
        if offset < targets.0 || offset >= targets.1 {
            continue;
        }
        let needed = schedule[distance.min(HASH_BITS) as usize];
        if needed == u32::MAX {
            continue;
        }
        let witnesses = aligned[lo..]
            .iter()
            .take_while(|&&(o, _, _, _)| o - offset <= ALIGNMENT_TOLERANCE_MS)
            .filter(|&&(_, wi, wj, _)| wi != i && wj != j)
            .take(needed as usize)
            .count();
        if witnesses as u32 >= needed {
            matched_a[i as usize] = true;
            matched_b[j as usize] = true;
        }
    }
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
    let n = fingerprints.len();
    let total_hashes: usize = fingerprints.iter().map(|fp| fp.valid_hashes.len()).sum();
    // Phase 1 has to be exhaustive at the WIDEST distance phase 2 can admit, not
    // at `-d`: a pair whose only matches are corroborated ones would otherwise
    // never be proposed, and the corroboration rule would be unreachable exactly
    // where it does the most good.
    let tol = Tolerance::for_distance(max_hamming_dist);
    let masks = probe_masks(probe_radius(tol.widest()));
    // One table for the whole run: it depends on the hash width alone, and
    // every pair reads the same answers out of it.
    let schedule = witness_schedule();

    // The two arms differ only in where the pairs come from, and they agree on
    // every pair: the index is exhaustive, and the direct route is what
    // "exhaustive" means. Measuring one pair is the same work either way, so it
    // lives in `measure_pair` and neither arm can drift from the other.
    let pairs: Vec<(usize, usize)> = if index_is_cheaper(masks.len(), n, total_hashes) {
        let candidates = candidate_pairs(fingerprints, &masks, tol.widest());
        info!("Index scan produced {} candidate pair(s); verifying...", candidates.len());
        candidates
    } else {
        // Deliberately not materialised: at the tolerances that get here the
        // pair list is the whole triangle, which is quadratic in the library
        // size and would be the largest allocation in the program by far. The
        // parallel range below yields it instead, and the inner range is
        // parallel too because the outer one is lopsided -- video 0 pairs with
        // everything and the last video with nothing.
        info!("Tolerance is too loose for the index; comparing all pairs directly...");

        // The count is the whole triangle even though the list of it is never
        // built -- `n * (n - 1) / 2` is what the ranges below yield, and the bar
        // needs a denominator rather than the pairs themselves. In u64 because
        // this is the arm that runs on the libraries where it overflows a u32.
        let total = n as u64 * n.saturating_sub(1) as u64 / 2;
        let pb = verification_bar(total);

        let matches = (0..n)
            .into_par_iter()
            .flat_map(|v_a| ((v_a + 1)..n).into_par_iter().map(move |v_b| (v_a, v_b)))
            .map_init(
                || Ticker::new(&pb),
                |ticker, pair| {
                    ticker.tick();
                    measure_pair(fingerprints, pair, tol, &schedule, min_match_percent, min_duration)
                },
            )
            .flatten()
            .collect();

        pb.finish_and_clear();
        return matches;
    };

    if shutdown_requested() {
        return Vec::new();
    }

    let pb = verification_bar(pairs.len() as u64);

    let matches = pairs
        .into_par_iter()
        .map_init(
            || Ticker::new(&pb),
            |ticker, pair| {
                ticker.tick();
                measure_pair(fingerprints, pair, tol, &schedule, min_match_percent, min_duration)
            },
        )
        .flatten()
        .collect();

    pb.finish_and_clear();
    matches
}

/// The bar phase 2 runs under.
///
/// Phase 1 announces how many pairs it proposed and then, until this existed,
/// went silent for the whole of the verification. At the defaults that is a
/// fifth of a second and nobody notices; at a loose `-d` over a large library it
/// is minutes of a program that looks hung, immediately after a decode stage
/// that reported itself continuously.
///
/// Shown at the same level as that stage's own log lines, so `--quiet` silences
/// both -- and indicatif draws nothing when stderr is not a terminal, so a
/// redirected run is unaffected either way. Unit tests install no logger at all,
/// which leaves the max level at `Off` and the bar hidden; nothing here has to
/// be told it is a test.
///
/// No ETA, for the same reason the decode bar has none: a pair costs the product
/// of the two files' sample counts, which spans orders of magnitude across a
/// mixed library, so a rate extrapolated from the pairs done so far predicts the
/// remainder badly.
fn verification_bar(pairs: u64) -> ProgressBar {
    if pairs == 0 || !log::log_enabled!(log::Level::Info) {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(pairs);
    pb.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} \u{2502} [{bar:28.cyan/blue}] \u{2502} {percent}% \u{2502} {human_pos}/{human_len} pairs",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb
}

/// How many pairs one worker gets through before it touches the bar.
///
/// A pair is measured in microseconds and the bar redraws twenty times a second
/// at most, so reporting each one is an atomic add on a line every thread is
/// fighting over, bought entirely for pixels that are never drawn -- the same
/// reasoning as `PROGRESS_STEP_BYTES` in the decode stage. At 512 a run of a
/// million pairs still moves the bar two thousand times.
const PAIRS_PER_REPORT: u64 = 512;

/// A worker's unreported pair count, flushed to the bar in batches.
///
/// The flush on drop is what makes the batching invisible rather than merely
/// coarse: rayon drops the per-worker state when that worker's slice is done, so
/// the last partial batch is always accounted for and the bar lands on full
/// however the pairs happened to be split.
struct Ticker<'a> {
    bar: &'a ProgressBar,
    pending: u64,
}

impl<'a> Ticker<'a> {
    fn new(bar: &'a ProgressBar) -> Self {
        Ticker { bar, pending: 0 }
    }

    fn tick(&mut self) {
        self.pending += 1;
        if self.pending >= PAIRS_PER_REPORT {
            self.bar.inc(self.pending);
            self.pending = 0;
        }
    }
}

impl Drop for Ticker<'_> {
    fn drop(&mut self) {
        if self.pending > 0 {
            self.bar.inc(self.pending);
        }
    }
}

/// Phase 2 for one pair: measure it exactly, then apply both gates. `None` when
/// the pair fails either of them, or when the run is shutting down.
fn measure_pair(
    fingerprints: &[VideoFingerprint],
    (v_a, v_b): (usize, usize),
    tol: Tolerance,
    schedule: &[u32; HASH_BITS as usize + 1],
    min_match_percent: f32,
    min_duration: f64,
) -> Option<Match> {
    if shutdown_requested() {
        return None;
    }
    let fp_a = &fingerprints[v_a];
    let fp_b = &fingerprints[v_b];

    let (pct_a, pct_b, span_a, span_b) = match_overlap(fp_a, fp_b, tol, schedule);

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

    Some(Match { a: v_a, b: v_b, coverage_a: pct_a, coverage_b: pct_b, span_a, span_b })
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

    /// Both tolerances pinned to the same value, so a test can state which
    /// samples pair up without the corroboration rule having a say. `-d` never
    /// never produces one of these -- the two sides are always a fixed gap
    /// apart; it is the phase-2 behaviour these tests are about, not the
    /// mapping from the flag.
    fn exactly(bits: u32) -> Tolerance {
        Tolerance { strict: bits, loose: bits }
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
        let mut previous = Tolerance { strict: 0, loose: 0 };
        for d in 0..=HASH_BITS {
            let tol = Tolerance::for_distance(d);
            assert!(tol.strict <= tol.loose, "-d {d}: strict must not exceed loose");
            assert!(tol.loose >= d, "-d {d}: never stricter than the flag asked for");
            assert!(tol.strict >= previous.strict && tol.loose >= previous.loose);
            previous = tol;
        }
        // Both sides move with every rung and neither ever saturates: the gap
        // is constant, so `-d 12` is not `-d 4` with extra steps.
        assert_eq!(Tolerance::for_distance(0), Tolerance { strict: 0, loose: 6 });
        assert_eq!(Tolerance::for_distance(4), Tolerance { strict: 4, loose: 10 });
        assert_eq!(Tolerance::for_distance(12), Tolerance { strict: 12, loose: 18 });
        assert_eq!(Tolerance::for_distance(20), Tolerance { strict: 20, loose: 26 });
        // ...except against the hash width, past which a tolerance means nothing.
        assert_eq!(
            Tolerance::for_distance(HASH_BITS),
            Tolerance { strict: HASH_BITS, loose: HASH_BITS }
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
            tol,
            &witness_schedule(),
            &mut matched_a,
            &mut matched_b,
            cap,
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
        // bucket on each side; a bucket here is one time offset, which this
        // pair has at most `n` matches at.
        assert!(
            held <= cap + 2 * n,
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
        // A chain: 0-1 and 1-2 matched, 0 and 2 never did. Clustering does not
        // hand this module a group like that -- every real group is a clique --
        // but the lookup must not depend on that, so it is asked directly: 0's
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

        let (_, _, span_clip, span_host) = match_overlap(&clip, &host, exactly(0), &witness_schedule());

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

        let (coverage_a, _, span_a, _) = match_overlap(&a, &b, exactly(0), &witness_schedule());

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

        let (_, _, span_a, span_b) = match_overlap(&a, &b, exactly(0), &witness_schedule());

        assert_eq!(span_a, None);
        assert_eq!(span_b, None);
    }
}
