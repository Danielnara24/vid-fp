//! The decode pass: the work that actually costs something.
//!
//! The threads are apportioned by WORK, not by file count (`budget`). Counting
//! files is what left a 4 GB file decoding on a single thread: with hundreds of
//! videos queued the per-file share rounds to 1, and FFmpeg fixes a decoder's
//! thread count when it is opened, so a decode's width is decided once, at the
//! moment it starts. The loop is owned here rather than by rayon for the same
//! reason: rayon's splitting hands each worker a region of the input, so claims
//! would not arrive in weight order, and the whole schedule rests on the
//! heaviest video claiming while the budget is still wide.

use super::budget::ThreadBudget;
use super::{Job, Ledger};
use crate::cache::{self, Cache, Sampling};
use crate::fingerprint::{self, fingerprint_video, VideoFingerprint};
use crate::progress::{self, Counter};
use crate::stats::{self, RunStats};
use crate::utils::shutdown_requested;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

/// How far the demuxer has to travel through a file before the progress bar is
/// told about it.
///
/// The hook behind this fires once per demuxed packet, which on a container we
/// scan linearly is thousands of times a second per decode, from every worker at
/// once -- and each report that gets through takes indicatif's state lock. The
/// bar redraws at most 20 times a second no matter how often it is nudged, so
/// anything finer than this is contention bought for pixels that are never
/// drawn.
///
/// 4 MiB is the coarsest step that still looks continuous on the file sizes that
/// need it: it puts ~2000 updates across an 8 GB decode (several a second over
/// minutes of work) while a file small enough to yield fewer than a handful is,
/// by definition, one that finishes before anybody looks at the bar.
const PROGRESS_STEP_BYTES: u64 = 4 * 1024 * 1024;

/// Decode every job, heaviest first, and settle each into the ledger. Returns
/// how many fingerprints were written to the cache.
pub(super) fn decode_pass(
    db: &Cache,
    todo: Vec<Job>,
    sampling: Sampling,
    threads: usize,
    ledger: &mut Ledger,
) -> usize {
    if todo.is_empty() {
        return 0;
    }
    let budget = ThreadBudget::new(threads, todo.iter().map(|job| job.weight as u128).sum());
    let cursor = AtomicUsize::new(0);
    let newly_cached = AtomicUsize::new(0);
    let in_flight = InFlight::default();
    // Measured in decode work, not files and not bytes: counting files is close
    // to meaningless with inputs spanning three orders of magnitude, and
    // counting bytes let the bar cross 50% a quarter of the way through a
    // mixed-codec folder. The files finished are its count.
    let bar = progress::begin_counted(
        progress::Stage::Fingerprinting,
        todo.iter().map(Job::cost).sum(),
        todo.len() as u64,
    );
    let floor = ledger.floor;
    let stats = ledger.stats;
    let ledger = Mutex::new(ledger);
    let workers = threads.min(todo.len()).max(1);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                // A relaxed load per queued video, so after a Ctrl-C the tail of
                // a 50k-file queue drains in microseconds; the decodes in flight
                // stop on the same check inside their demux loop.
                while !shutdown_requested() {
                    // Blocks while every thread is promised to a running decode.
                    // `None` is a shutdown, or a queue that ran out meanwhile.
                    let Some((idx, job, grant)) = budget.claim_next(&cursor, &todo) else {
                        break;
                    };
                    in_flight.open(idx, &job.path);
                    let meter = Meter::new(job, &bar);

                    let decoded = fingerprint_video(
                        &job.path,
                        sampling.kf_interval,
                        sampling.min_kf_samples,
                        grant.threads,
                        floor,
                        // The scan's own stat, the figure this job's cache
                        // stamp was built from -- see `fingerprint_video`.
                        job.size,
                        &|step| meter.advance(step),
                    );
                    // Before the cache write: an fsync is not decoding, and
                    // holding a wide share across one idles the machine.
                    drop(grant);

                    match decoded {
                        Ok(Some(fp)) => {
                            let made = shared(job, fp);
                            newly_cached.fetch_add(store(db, job, &made, stats), Ordering::Relaxed);
                            let mut ledger = lock(&ledger);
                            made.into_iter().for_each(|fp| ledger.fingerprinted(fp));
                        }
                        // Shorter than `--min-duration`, from a header the
                        // weighing pass could not vouch for. Its copies are
                        // exactly as short.
                        Ok(None) => job.members().for_each(|_| lock(&ledger).short()),
                        // An interrupt unwinds through here as our own error.
                        // That is the user's doing, not the file's: exit 130
                        // already says what happened.
                        Err(_) if shutdown_requested() => {}
                        // Including a file that vanished since the walk: the run
                        // was asked to fingerprint it and could not. Its copies
                        // fail with it, each named, since the same bytes would.
                        Err(e) => {
                            let mut ledger = lock(&ledger);
                            job.members().for_each(|m| ledger.could_not_fingerprint(m, &e));
                        }
                    }
                    meter.finish();
                    in_flight.close(idx);
                }
            });
        }
    });

    progress::detail(String::new());
    newly_cached.into_inner()
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// The fingerprint once for each member of the job, under its own path.
fn shared(job: &Job, fp: VideoFingerprint) -> Vec<VideoFingerprint> {
    let copies: Vec<VideoFingerprint> =
        job.followers.iter().map(|f| VideoFingerprint { path: f.path.clone(), ..fp.clone() }).collect();
    std::iter::once(fp).chain(copies).collect()
}

/// Commit a decode the moment it exists, one transaction per video, so an
/// interrupt, a kill or a crash can only ever cost the decode still in flight.
/// Returns how many entries were written.
fn store(db: &Cache, job: &Job, made: &[VideoFingerprint], stats: &RunStats) -> usize {
    let entries: Vec<cache::Write> =
        job.members().zip(made).map(|(m, fp)| cache::Write::Fingerprint(&m.path, m.stamp, fp)).collect();
    match db.write(&entries) {
        Ok(()) => entries.len(),
        Err(e) => {
            for member in job.members() {
                log::error!(
                    target: stats::COUNTED,
                    "Failed to cache fingerprint for {}: {:#}",
                    member.path,
                    e
                );
                stats.cache_write_failed.record(format!("{}: {:#}", member.path, e));
            }
            0
        }
    }
}

/// How far into one file the decode has got, charged to the bar as it goes.
///
/// Two parts, because the decode does two things one after the other: it reads
/// and decodes, which `fingerprint_video` reports as a byte offset per packet,
/// and then it hashes the frames it kept, frame by frame (`Step`). The first is
/// charged the file's keyframe-pixels by its byte fraction, the second its
/// keyframes' hashing by the frames hashed (`progress::decode_cost`). Charged
/// by bytes alone, a ProRes file reached the end of its bytes at 3.8 s of an
/// 8 s decode and the bar then sat still.
///
/// `Cell`s because the hook has to stay `Fn`; it never leaves this thread.
struct Meter<'a> {
    job: &'a Job,
    bar: &'a Counter,
    bytes: Cell<u64>,
    read: Cell<u64>,
    hashed: Cell<u64>,
}

impl<'a> Meter<'a> {
    fn new(job: &'a Job, bar: &'a Counter) -> Self {
        Meter { job, bar, bytes: Cell::new(0), read: Cell::new(0), hashed: Cell::new(0) }
    }

    fn advance(&self, step: fingerprint::Step) {
        match step {
            fingerprint::Step::Read(pos) => {
                // Clamped, never a delta from the raw offset: a seek that lands
                // short walks the offset BACKWARDS, and an offset past the size
                // we stat'ed (a file being appended to) would overrun the bar.
                let size = self.job.size;
                let reached = pos.min(size);
                if reached.saturating_sub(self.bytes.get()) < PROGRESS_STEP_BYTES || size == 0 {
                    return;
                }
                self.bytes.set(reached);
                let work = (self.job.weight as u128 * reached as u128 / size as u128) as u64;
                Self::charge(self.bar, &self.read, work);
            }
            fingerprint::Step::Hashed(done, of) => {
                if of == 0 {
                    return;
                }
                let hashing = self.job.cost() - self.job.weight;
                let work = (hashing as u128 * done.min(of) as u128 / of as u128) as u64;
                Self::charge(self.bar, &self.hashed, work);
            }
        }
    }

    fn charge(bar: &Counter, charged: &Cell<u64>, work: u64) {
        bar.add(0, work.saturating_sub(charged.get()));
        charged.set(work.max(charged.get()));
    }

    /// The file is off the queue, however it ended: whatever is still owed of
    /// its cost is paid, so the bar lands on 100% and not near it.
    fn finish(&self) {
        let charged = self.read.get() + self.hashed.get();
        self.bar.add(1, self.job.cost().saturating_sub(charged));
    }
}

/// The decodes running right now, keyed by their place in the queue.
///
/// The bar has one line for a file name and up to `--threads` files in flight,
/// and the newest claim -- what it used to print -- is by construction the one
/// that has been running for the shortest time: a 54 GB scan spent its last
/// minute naming a 30 MB clip. The queue is heaviest-first, so the lowest key
/// is both the earliest claim and the heaviest file still open, the honest
/// answer to "what is this waiting on". Nothing open names nothing, rather
/// than a file already finished.
#[derive(Default)]
struct InFlight(Mutex<BTreeMap<usize, String>>);

impl InFlight {
    fn open(&self, idx: usize, path: &str) {
        let name = Path::new(path).file_name().unwrap_or_default().to_string_lossy().into_owned();
        let mut open = lock(&self.0);
        open.insert(idx, name);
        Self::show(&open);
    }

    fn close(&self, idx: usize) {
        let mut open = lock(&self.0);
        open.remove(&idx);
        Self::show(&open);
    }

    fn show(open: &BTreeMap<usize, String>) {
        progress::detail(open.values().next().cloned().unwrap_or_default());
    }
}
