//! The scan: every file the walk found is settled exactly once, as a
//! fingerprint, a skip, or a failure, and turned into the list the comparison
//! reads.
//!
//! Four passes, in order, each taking the files still open and handing back
//! the ones it could not settle:
//!
//! 1. **cache** (`cache_pass`) -- every lookup before any decode. Interleaving
//!    them under rayon let one slow decode block cached neighbours in the same
//!    work slice.
//! 2. **weighing** (`weigh_pass`) -- each miss is opened once, header only, and
//!    asked what decoding it implies. A file that will not decode, or is under
//!    `--min-duration`, is settled here and never opened again.
//! 3. **content** (`identify`) -- a moved or touched file takes over the
//!    fingerprint its bytes already have, and byte-identical files share one
//!    decode.
//! 4. **decode** (`decode_pass`) -- the rest, heaviest first, under the thread
//!    budget.
//!
//! **How a file ended is recorded in one place, the `Ledger`.** Every pass used
//! to keep its own copy of the bookkeeping -- the `--min-duration` test in
//! five places, "log it, count it, maybe remember it" in three, and a skip
//! bumped from five -- and each new way a file could end had to be added to
//! all of them. The passes now only decide; the ledger counts, logs, applies
//! the floor and gathers the refusals worth remembering, and a byte-identical
//! copy is settled by the same call as the file it copies (`Job::members`).

mod budget;
mod decode;
mod identify;

use crate::cache::{self, Cache, Known, Refusal, Sampling, Stamp};
use crate::fingerprint::{self, Floor, VideoFingerprint};
use crate::progress;
use crate::sources::ScannedFile;
use crate::stats::{self, RunStats};
use crate::utils::shutdown_requested;
use anyhow::Result;
use log::info;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// One file the scan has not settled yet, and everything needed to decide how
/// much of the machine to give it.
///
/// `weight` is what the decode is going to cost, in the keyframe-pixels
/// `fingerprint::weigh_decode` measures -- not the file's size. Both numbers are
/// here because they answer different questions and neither substitutes for the
/// other: `weight` is how long this file will take, `size` is how many bytes the
/// demuxer will walk through to do it, which is the only thing the in-flight
/// progress reports are denominated in.
pub(crate) struct Job {
    path: String,
    stamp: Stamp,
    weight: u64,
    /// Keyframes the decode will hash, as the weighing pass counted them; 0
    /// before it has, or when it could not. Only the progress bar reads it.
    keyframes: u64,
    size: u64,
    /// Byte-identical copies of this file that are waiting too. Not decoded:
    /// each takes this file's fingerprint under its own path, or its failure,
    /// or its skip. See `identify`.
    followers: Vec<Job>,
}

impl Job {
    /// A file as the walk found it, before anything has weighed it: the size
    /// stands in for the weight until the weighing pass measures one, and the
    /// only thing that reads it before then is the order the cache pass keeps.
    fn of(file: &ScannedFile, sampling: Sampling) -> Job {
        Job {
            path: file.path.clone(),
            stamp: Stamp {
                mtime: file.mtime,
                mtime_nsec: file.mtime_nsec,
                size: file.size,
                kf_interval: sampling.kf_interval,
                min_kf_samples: sampling.min_kf_samples,
            },
            weight: file.size,
            keyframes: 0,
            size: file.size,
            followers: Vec::new(),
        }
    }

    /// This file and its byte-identical copies, which share its fate.
    fn members(&self) -> impl Iterator<Item = &Job> {
        std::iter::once(self).chain(&self.followers)
    }

    /// What decoding this file costs on the run's progress bar.
    fn cost(&self) -> u64 {
        progress::decode_cost(self.weight, self.keyframes)
    }
}

/// How every file the scan settled ended, and what of that is worth writing
/// down.
///
/// The only place that counts a skip or a failure, logs one, or applies
/// `--min-duration` to a fingerprint, so those cannot differ between the
/// passes: a file is too short, or failed, in the same words and the same
/// tally whichever pass found out.
pub(crate) struct Ledger<'a> {
    stats: &'a RunStats,
    floor: Floor,
    fingerprints: Vec<VideoFingerprint>,
    /// Refusals waiting to be written, one transaction a pass: see `remember`.
    refusals: Vec<(String, Stamp, Refusal)>,
}

impl<'a> Ledger<'a> {
    pub(crate) fn new(stats: &'a RunStats, floor: Floor) -> Self {
        Ledger { stats, floor, fingerprints: Vec::new(), refusals: Vec::new() }
    }

    /// A fingerprint in hand, however it came: read from the cache, taken over
    /// from a moved file, or decoded. Its own runtime decides the floor, so a
    /// warm run and a cold one skip the same files.
    fn fingerprinted(&mut self, fp: VideoFingerprint) {
        if self.floor.rejects(fp.duration) {
            self.stats.skipped_short.bump();
        } else {
            self.fingerprints.push(fp);
        }
    }

    /// Too short, found before any fingerprint existed. Not a problem: it is a
    /// skip the user asked for.
    fn short(&mut self) {
        self.stats.skipped_short.bump();
    }

    /// Too short, from a header the weighing pass read. The runtime is what is
    /// remembered, not the verdict -- see `Refusal::TooShort`.
    fn measured_short(&mut self, job: &Job, duration: f64) {
        self.short();
        self.refusals.push((job.path.clone(), job.stamp, Refusal::TooShort(duration)));
    }

    /// The file has no fingerprint to give. Named and counted as a problem
    /// whether this run found out or an earlier one did, in the words the
    /// finding carries; `remember` is the verdict to keep for the next run, if
    /// it is worth keeping.
    fn failed(&mut self, path: &str, reason: &str, remember: Option<(Stamp, Refusal)>) {
        log::error!(target: stats::COUNTED, "Failed to process {}: {}", path, reason);
        self.stats.fingerprint_failed.record(format!("{}: {}", path, reason));
        if let Some((stamp, refusal)) = remember {
            self.refusals.push((path.to_string(), stamp, refusal));
        }
    }

    /// A failure this run found, remembered unless it was about the moment
    /// rather than the file (`cache::worth_remembering`).
    fn could_not_fingerprint(&mut self, job: &Job, error: &anyhow::Error) {
        let remember = cache::worth_remembering(error).map(|refusal| (job.stamp, refusal));
        self.failed(&job.path, &format!("{:#}", error), remember);
    }

    /// Write the refusals gathered since the last call, in one transaction. A
    /// refusal costs half a millisecond to find again and there can be a
    /// quarter of a million in one run, so one fsync each would cost more than
    /// it saves; and a failure to write is not a failure of the run, since all
    /// these entries would have saved is time.
    fn remember(&mut self, db: &Cache) {
        if self.refusals.is_empty() {
            return;
        }
        let writes: Vec<cache::Write> = self
            .refusals
            .iter()
            .map(|(path, stamp, refusal)| cache::Write::Refusal(path, *stamp, refusal.clone()))
            .collect();
        if let Err(e) = db.write(&writes) {
            log::debug!("Could not record {} refusal(s): {:#}", writes.len(), e);
        }
        self.refusals.clear();
    }

    /// The fingerprints, in path order.
    ///
    /// One order, whatever the cache held: the list is built as cache hits in
    /// size order, then adoptions, then fresh decodes in completion order, so
    /// a cold scan and a warm rescan of the same library would hand the
    /// comparison two different lists. The comparison is not indifferent to
    /// order -- ties in the star cover go to the lower index, and a mirrored
    /// link is measured from the lower-numbered file's side -- and on the
    /// generated corpus the two runs once wrote different reports, down to one
    /// file KEEP cold and DELETE warm. Sorted, they are byte-identical.
    fn into_fingerprints(self) -> Vec<VideoFingerprint> {
        let mut fingerprints = self.fingerprints;
        fingerprints.sort_by(|a, b| a.path.cmp(&b.path));
        fingerprints
    }
}

/// What the scan is asked to do, beyond which files.
pub(crate) struct Settings<'a> {
    pub sampling: Sampling,
    pub floor: Floor,
    /// The clamped `--threads`, which the decoder budget is apportioned from.
    pub threads: usize,
    /// Whether to drop the cache entries of files this scan did not find. The
    /// caller has already decided the scan is complete enough to trust.
    pub prune: bool,
    /// "video files", or just "files" when the walk applied no extension
    /// filter and has not checked that claim.
    pub noun: &'a str,
}

/// What the scan produced.
pub(crate) struct Scanned {
    /// In path order: see `Ledger::into_fingerprints`.
    pub fingerprints: Vec<VideoFingerprint>,
    /// Every set of byte-identical files among them, by path.
    pub identical: Vec<Vec<String>>,
}

/// Settle every file: the four passes, in order. `Ok(None)` is an interrupt.
pub(crate) fn fingerprint_all(
    db: &Cache,
    mut files: Vec<ScannedFile>,
    settings: &Settings,
    stats: &RunStats,
) -> Result<Option<Scanned>> {
    let sampling = settings.sampling;
    let total = files.len();

    // Largest first, so the cache pass hands its queue back in roughly the
    // order the decode wants. Roughly is all this can be: the schedule rests on
    // decode COST, which the weighing pass measures and re-sorts on. Sorting by
    // size costs nothing (the walk already read it) and puts the list within a
    // rotation of where it needs to be.
    files.sort_by_key(|f| std::cmp::Reverse(f.size));
    progress::learn(|k| {
        k.files = total;
        k.bytes = files.iter().map(|f| f.size).sum();
    });
    if sampling.kf_interval > 0.0 {
        info!(
            "Using keyframe interval: {}s, minimum keyframes: {}",
            sampling.kf_interval, sampling.min_kf_samples
        );
    }

    let mut ledger = Ledger::new(stats, settings.floor);

    let answered = cache_pass(db, &files, sampling, &mut ledger);
    info!("{}", answered.headline(total, settings.noun));
    if shutdown_requested() {
        return Ok(None);
    }

    let todo = weigh_pass(answered.todo, sampling, &mut ledger);
    ledger.remember(db);
    if shutdown_requested() {
        return Ok(None);
    }

    let by_path: HashMap<&str, &ScannedFile> = files.iter().map(|f| (f.path.as_str(), f)).collect();
    let Some(identified) = identify::identify(db, &by_path, &ledger.fingerprints, todo, sampling)? else {
        return Ok(None);
    };
    for fp in identified.adopted {
        ledger.fingerprinted(fp);
    }

    // After the content pass, not before the cache pass: a file that moved is
    // still under its OLD path in the cache until that pass adopts its
    // fingerprint under the new one, and pruning first would discard the very
    // entry the move was going to be recognised by.
    if settings.prune {
        let walked: HashSet<&str> = files.iter().map(|f| f.path.as_str()).collect();
        info!("Pruning cache for files not in the current scan...");
        match db.prune(&walked)? {
            None => return Ok(None),
            Some(0) => info!("No stale entries found to prune."),
            Some(n) => {
                stats.record_prune(n);
                info!("Pruned {} stale entries from cache.", n);
            }
        }
    }
    if shutdown_requested() {
        return Ok(None);
    }

    let newly_cached = decode::decode_pass(db, identified.todo, sampling, settings.threads, &mut ledger);
    ledger.remember(db);

    let mut fingerprints = ledger.into_fingerprints();
    // The walk's change time, for the disposal pass to hold each file to: a
    // cached fingerprint was made by another run and knows nothing of this one.
    for fp in &mut fingerprints {
        if let Some(file) = by_path.get(fp.path.as_str()) {
            fp.measured_mut().ctime = Some(file.ctime);
        }
    }
    if shutdown_requested() {
        info!(
            "Cached {} video(s); {} of {} are now cached in total.",
            newly_cached,
            fingerprints.len(),
            total
        );
        return Ok(None);
    }
    Ok(Some(Scanned { fingerprints, identical: identified.identical }))
}

/// What the cache pass answered, beside what it settled into the ledger.
#[derive(Default)]
struct Answered {
    /// The misses, still in the size order the walk's files were sorted in.
    todo: Vec<Job>,
    /// Files the cache answered without a decode -- not the fingerprints that
    /// survived, which `--min-duration` can thin: a warm library the floor
    /// swallows whole was once announced as "Fingerprinting..." and then
    /// decoded nothing.
    cached: usize,
    /// Remembered refusals, apart, because the two say opposite things about
    /// the file: one is not video, the other is a video this build cannot
    /// read. See `headline`.
    not_media: usize,
    failed: usize,
}

impl Answered {
    /// Settle one file from what the cache knows about it, or queue it.
    fn take(&mut self, job: Job, known: Known, ledger: &mut Ledger) {
        match known {
            Known::Fingerprint(fp) => {
                self.cached += 1;
                ledger.fingerprinted(*fp);
            }
            // A remembered runtime, not a remembered verdict -- see
            // `Refusal::TooShort`. Too short for this run's floor is a skip;
            // long enough is decoded like any other miss, which is what makes
            // lowering `--min-duration` work from a warm cache.
            Known::Refused(Refusal::TooShort(duration)) if ledger.floor.rejects(duration) => {
                self.cached += 1;
                ledger.short();
            }
            Known::Refused(Refusal::TooShort(_)) | Known::Nothing => self.todo.push(job),
            // The same finding about the same bytes, in this build's words --
            // the cache keeps what it needs to say it again -- and still a
            // problem: there is no fingerprint of this file.
            Known::Refused(verdict) => {
                if verdict.is_not_media() {
                    self.not_media += 1;
                } else {
                    self.failed += 1;
                }
                ledger.failed(&job.path, &verdict.to_string(), None);
            }
        }
    }

    /// The line that says what the run is about to do.
    ///
    /// The remembered refusals are two clauses, each saying only what it knows.
    /// A file the probe gate turned away IS not video. A file that opened and
    /// then failed to decode is a video this build cannot read, and calling a
    /// 1080p HEVC episode "not video" because the decoder gave up on it points
    /// the user at the wrong thing: one is a file to stop scanning, the other a
    /// file to go and look at. Both are still problems and both still exit 2.
    fn headline(&self, total: usize, noun: &str) -> String {
        if self.cached == 0 && self.not_media == 0 && self.failed == 0 {
            return format!("Found {} {}. Fingerprinting...", total, noun);
        }
        let mut remembered = String::new();
        if self.not_media > 0 {
            remembered.push_str(&format!(", {} already known not to be video", self.not_media));
        }
        if self.failed > 0 {
            remembered.push_str(&format!(", {} already known not to decode", self.failed));
        }
        format!(
            "Found {} {}; {} already cached{}, {} to fingerprint.",
            total,
            noun,
            self.cached,
            remembered,
            self.todo.len()
        )
    }
}

/// The cache pass: what the cache already knows, for every file, before
/// anything is decoded.
///
/// One snapshot for the whole pass (`cache::View`), let go the moment it is
/// done: a live read transaction pins every page it can see, so holding it
/// across the decode would stop redb reclaiming what the decode's writes
/// supersede. A lookup is a b-tree descent and a bincode decode, no I/O against
/// the library, so on any library this pass is effectively instant -- and by
/// the end of it the run knows how much real work there is.
fn cache_pass(db: &Cache, files: &[ScannedFile], sampling: Sampling, ledger: &mut Ledger) -> Answered {
    let n = files.len() as u64;
    let read = progress::begin_counted(progress::Stage::CacheRead, n, n);
    let view = db.view();
    let known: Vec<(Job, Known)> = files
        .par_iter()
        .map(|file| {
            read.tick();
            let job = Job::of(file, sampling);
            let known = view.lookup(&job.path, &job.stamp);
            (job, known)
        })
        .collect();
    drop(view);

    // `collect` keeps input order, so the queue inherits the largest-first sort.
    let mut answered = Answered::default();
    for (job, known) in known {
        answered.take(job, known, ledger);
    }
    progress::learn(|k| {
        k.misses = Some((answered.todo.len(), answered.todo.iter().map(|job| job.size).sum()));
        k.cached_hashes = (
            ledger.fingerprints.len(),
            ledger.fingerprints.iter().map(|fp| fp.valid_hashes.len() as u64).sum(),
        );
    });
    answered
}

/// The weighing pass: what each miss is going to cost to decode, or why it
/// will not be decoded at all.
///
/// Until this pass the only figure for a file's work was its SIZE, which is a
/// statement about bitrate as much as about decode cost: a 10.5 GB 2160p HEVC
/// feature costs 14.7x a 1.8 GB 1080p H.264 one while its size claims 5.8x, and
/// `--keyframe-interval`, which can remove 90% of the decode, moves the size not
/// at all. So each file is opened once, header only, and asked how many
/// keyframes it holds and how big they are (`fingerprint::weigh_decode`): ~93%
/// of fingerprinting time is intra decode of exactly those frames. About 0.8 ms
/// a file, paid only on cache misses.
///
/// Complete before any decode, because the thread budget and the progress bar
/// are both apportioned against the total. And it settles two kinds of file for
/// good -- one that will not decode, and one under `--min-duration` -- so that
/// what is left is work that is really going to happen, and no file is opened
/// twice to find out what this pass already found. Under `-x '*'` over a folder
/// that is mostly not video that is half of everything before the comparison,
/// and it is what keeps the bar from being denominated in the bytes of files
/// that finish the instant they are looked at.
fn weigh_pass(todo: Vec<Job>, sampling: Sampling, ledger: &mut Ledger) -> Vec<Job> {
    let mut kept: Vec<Job> = Vec::with_capacity(todo.len());
    if !todo.is_empty() {
        let n = todo.len() as u64;
        let bar = progress::begin_counted(progress::Stage::Weighing, n, n);
        let floor = ledger.floor;
        let weighed: Vec<fingerprint::Weighed> = todo
            .par_iter()
            .map(|job| {
                // Ticked on both paths: after a Ctrl-C the rest are answered
                // instantly, and a bar that stopped counting them would freeze
                // while the pass visibly kept running.
                let weighed = if shutdown_requested() {
                    fingerprint::Weighed::Work(job.size, 0)
                } else {
                    let Sampling { kf_interval, min_kf_samples } = sampling;
                    fingerprint::weigh_decode(&job.path, kf_interval, min_kf_samples, floor, job.size)
                };
                bar.tick();
                weighed
            })
            .collect();

        for (mut job, weighed) in todo.into_iter().zip(weighed) {
            match weighed {
                fingerprint::Weighed::Work(weight, keyframes) => {
                    job.weight = weight;
                    job.keyframes = keyframes;
                    kept.push(job);
                }
                // It IS the decode's error, so it is worded, counted and
                // remembered exactly as the decode would have done it.
                fingerprint::Weighed::Undecodable(e) => ledger.could_not_fingerprint(&job, &e),
                fingerprint::Weighed::TooShort(duration) => ledger.measured_short(&job, duration),
            }
        }

        // NOW the heaviest-first order the schedule depends on is real. Size
        // put the wrong file first exactly where it matters, in a folder mixing
        // codecs: an AV1 and an HEVC copy of the same footage differ threefold
        // in cost and barely at all in the direction size predicts.
        kept.sort_by_key(|job| std::cmp::Reverse(job.weight));
    }
    progress::learn(|k| {
        k.misses = Some((kept.len(), kept.iter().map(|job| job.size).sum()));
        k.decode_work = Some(kept.iter().map(Job::cost).sum());
    });
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{mock_fp, stamp};

    fn job(path: &str) -> Job {
        Job {
            path: path.to_string(),
            stamp: stamp(1_700_000_000, 12_345),
            weight: 12_345,
            keyframes: 0,
            size: 12_345,
            followers: Vec::new(),
        }
    }

    fn short_fp(path: &str) -> VideoFingerprint {
        VideoFingerprint { duration: 2.0, ..mock_fp(path) }
    }

    /// Answer every lookup through one ledger, as the cache pass does.
    fn answer(known: Vec<Known>, floor: f64, stats: &RunStats) -> (Answered, Vec<VideoFingerprint>) {
        let mut ledger = Ledger::new(stats, Floor(floor));
        let mut answered = Answered::default();
        for (i, k) in known.into_iter().enumerate() {
            answered.take(job(&format!("/videos/{}.mkv", i)), k, &mut ledger);
        }
        (answered, ledger.into_fingerprints())
    }

    /// The headline is the only line that says what a run is about to do, and
    /// it used to say the opposite of the truth whenever `--min-duration`
    /// reached a warm library: the count was of the fingerprints that SURVIVED,
    /// and two arms answer from the cache and then drop the file -- a hit under
    /// the floor, and a remembered runtime under it -- so a run that decoded
    /// nothing announced "Fingerprinting...". Every file the pass sees is
    /// cached, queued, or refused, and the line adds up.
    #[test]
    fn test_the_cache_pass_counts_what_it_answered_and_not_what_survived() {
        let long = || Known::Fingerprint(Box::new(mock_fp("/videos/long.mkv")));
        let short = || Known::Fingerprint(Box::new(short_fp("/videos/short.mkv")));
        let remembered_short = || Known::Refused(Refusal::TooShort(2.0));
        let refused = |not_media: bool| {
            Known::Refused(if not_media {
                Refusal::NotMedia(fingerprint::NotMedia { bytes: 2048, score: 0 })
            } else {
                Refusal::Said("no frame decoded".to_string())
            })
        };

        // Every file cached, every one swallowed by the floor.
        let stats = RunStats::default();
        let (pass, kept) = answer(vec![short(), short(), short(), short()], 100_000.0, &stats);
        assert_eq!(pass.cached, 4, "the cache answered for every one of them");
        assert!(pass.todo.is_empty(), "and so there is nothing to decode");
        assert!(kept.is_empty(), "which is not the same question");
        assert_eq!(stats.skipped_short.count(), 4);

        // A second run at that floor: every verdict a remembered runtime.
        let stats = RunStats::default();
        let (pass, _) = answer(vec![remembered_short(), remembered_short()], 100_000.0, &stats);
        assert_eq!(pass.cached, 2, "a remembered runtime is a cache answer too");
        assert_eq!(stats.skipped_short.count(), 2);

        // One of each.
        let stats = RunStats::default();
        let known = vec![long(), short(), remembered_short(), Known::Nothing, refused(true), refused(false)];
        let total = known.len();
        let (pass, kept) = answer(known, 10.0, &stats);
        assert_eq!(pass.cached, 3, "two hits and a remembered runtime");
        assert_eq!(kept.len(), 1, "only one of them survived the floor");
        assert_eq!(pass.todo.len(), 1);
        assert_eq!((pass.not_media, pass.failed), (1, 1));
        assert_eq!(pass.cached + pass.todo.len() + pass.not_media + pass.failed, total);
        assert_eq!(stats.fingerprint_failed.count(), 2, "both refusals are still problems");
        assert_eq!(
            pass.headline(total, "video files"),
            "Found 6 video files; 3 already cached, 1 already known not to be video, \
             1 already known not to decode, 1 to fingerprint."
        );

        // A runtime long enough for this run's floor is a miss, decoded again.
        let stats = RunStats::default();
        let (pass, _) = answer(vec![remembered_short()], 1.0, &stats);
        assert_eq!((pass.cached, pass.todo.len()), (0, 1));
        assert_eq!(stats.skipped_short.count(), 0);

        // And the floor is still a floor: with it off, a short file is kept.
        let stats = RunStats::default();
        let (pass, kept) = answer(vec![short(), long()], 0.0, &stats);
        assert_eq!(pass.cached, 2);
        assert_eq!(kept.len(), 2, "nothing is short of no floor");
        assert_eq!(stats.skipped_short.count(), 0);
        assert_eq!(pass.headline(2, "files"), "Found 2 files; 2 already cached, 0 to fingerprint.");
        assert_eq!(Answered::default().headline(5, "files"), "Found 5 files. Fingerprinting...");

        // A kind of refusal the run has none of leaves no trace in the line.
        let only = |not_media, failed| Answered { cached: 1, not_media, failed, ..Answered::default() };
        assert_eq!(only(3, 0).headline(4, "files"), "Found 4 files; 1 already cached, 3 already known not to be video, 0 to fingerprint.");
        assert_eq!(only(0, 3).headline(4, "files"), "Found 4 files; 1 already cached, 3 already known not to decode, 0 to fingerprint.");
    }

    /// Every way a file can end goes through the ledger, and each lands in
    /// exactly one tally, with a refusal queued only where one is worth
    /// keeping: a measured runtime, and a failure about the file rather than
    /// the moment. A remembered failure is not written back.
    #[test]
    fn test_every_way_a_file_ends_is_counted_once_and_remembered_only_when_it_should_be() {
        let stats = RunStats::default();
        let mut ledger = Ledger::new(&stats, Floor(10.0));

        ledger.fingerprinted(mock_fp("/v/kept.mkv"));
        ledger.fingerprinted(short_fp("/v/short_fp.mkv"));
        ledger.short();
        ledger.measured_short(&job("/v/measured.mkv"), 3.0);
        ledger.failed("/v/remembered.mkv", "no frame decoded", None);
        ledger.could_not_fingerprint(&job("/v/broken.mkv"), &anyhow::anyhow!("No video stream found"));
        let gone = std::io::Error::from(std::io::ErrorKind::NotFound);
        ledger.could_not_fingerprint(&job("/v/vanished.mkv"), &anyhow::Error::new(gone));

        assert_eq!(stats.skipped_short.count(), 3);
        assert_eq!(stats.fingerprint_failed.count(), 3);
        let queued: Vec<&str> = ledger.refusals.iter().map(|(p, _, _)| p.as_str()).collect();
        assert_eq!(
            queued,
            ["/v/measured.mkv", "/v/broken.mkv"],
            "a measured runtime and a failure about the file; not the remembered one, \
             not the file that vanished"
        );
        assert!(matches!(ledger.refusals[0].2, Refusal::TooShort(d) if d == 3.0));
        let kept: Vec<String> = ledger.into_fingerprints().into_iter().map(|f| f.path).collect();
        assert_eq!(kept, ["/v/kept.mkv"]);
    }

    /// The followers of a job are settled with it, which is the whole of what
    /// sharing a decode means: same fingerprint under their own paths, same
    /// skip, same failure.
    #[test]
    fn test_a_job_s_copies_are_its_members_and_come_after_it() {
        let mut lead = job("/v/a.mkv");
        lead.followers = vec![job("/v/b.mkv"), job("/v/c.mkv")];
        let members: Vec<&str> = lead.members().map(|m| m.path.as_str()).collect();
        assert_eq!(members, ["/v/a.mkv", "/v/b.mkv", "/v/c.mkv"]);
    }

    #[test]
    fn test_the_duration_floor_rejects_only_a_known_runtime_under_it() {
        assert!(Floor(10.0).rejects(9.99));
        assert!(!Floor(10.0).rejects(10.0), "the floor itself is long enough");
        assert!(!Floor(10.0).rejects(0.0), "an unmeasured runtime is not a short one");
        assert!(!Floor::OFF.rejects(0.5), "no floor rejects nothing");
        assert!(!Floor::OFF.is_on() && Floor(0.1).is_on());
    }
}
