//! The decoder thread budget: how many threads each decode is given, and the
//! order the decodes claim them in. See `share_for` for the rule.

use super::Job;
use crate::fingerprint::MAX_DECODE_THREADS;
use crate::utils::shutdown_requested;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// How many of `free` threads a video of `weight` units of work should decode
/// with, given that `queued` units have not started yet.
///
/// This is the whole scheduling rule, and it comes out of asking what actually
/// ends the run. A video decoded on `k` of the `A` free threads finishes in
/// roughly `weight / k`; everything still queued gets the other `A - k` threads
/// and finishes in roughly `queued / (A - k)`. The run is over when the later
/// of the two is, so the best `k` is the one that makes them equal:
///
/// ```text
///     weight / k = queued / (A - k)   =>   k = A * weight / (weight + queued)
/// ```
///
/// A proportional share, in other words -- and note that it needs no special
/// case for "big" files, because the same formula produces both behaviours that
/// matter:
///
///   * Hundreds of similar videos: every share rounds to 1, which is correct.
///     Decoder threading scales sublinearly, so spreading one video per thread
///     extracts more total throughput than widening any single decode -- and
///     with that much work queued, no one file's own decode can outlast it.
///   * A handful of heavyweights, or one giant among small change: the giant's
///     share is most of the budget, taken UP FRONT. That is the point. Threads
///     cannot be added to a decode that has already started, so a file that
///     will still be running when the queue empties has to be given its width
///     at the moment it starts or never.
///
/// Rounding is to nearest rather than up: rounding up steals a thread from the
/// queued work permanently, which just moves the straggler rather than removing
/// it. A file owed 6.4 threads of 8 is better served by 6 than by 7, because the
/// 2 threads left cover the rest of the queue and the 1 thread left does not.
///
/// The single exception is the step from 2 down to 1, which is not a rounding
/// error like the others but a different kind of decision: 1 is the only width a
/// decode cannot recover from. Every share below 1 is already rounded UP to 1 --
/// a decode needs the thread it runs on -- so in a queue of many small files the
/// budget is systematically over-promised to the small change, and the whole of
/// that deficit lands on the one file too big to absorb it. Note that nearest
/// already rounds 1.5 up, so this only decides the band 1.0 < share < 1.5: a
/// file whose own decode runs up to half again as long as the whole run should.
///
/// Measured, on a 41-file 54 GB scan across 8 threads, 6 alternating cold-cache
/// runs of each rule: a 9.8 GB 2160p HEVC file was owed 1.43 threads and given
/// 1, which made it the last thing on the machine for the final minute of every
/// run. Rounding it to 2 finishes the scan in 258.4 s against 294.0 s (-12.1%,
/// sd 2.2 and 2.9, ranges nowhere near overlapping) and lifts average occupancy
/// from 6.3 of 8 cores to 7.5. The second thread is only worth 1.36x on that
/// file, so the run does about 3% more total work -- and buys back far more than
/// that from the threads the straggler used to leave idle behind it.
pub(super) fn share_for(weight: u64, queued: u128, free: usize) -> usize {
    if free == 0 {
        return 0;
    }

    // Nothing is waiting behind this video, so there is nobody to spread the
    // budget for. Take everything a decoder can actually use.
    if queued == 0 {
        return free.min(MAX_DECODE_THREADS);
    }

    let mine = weight as f64;
    let rest = queued as f64;

    // `mine + rest` is strictly positive here: `rest` is, since `queued != 0`.
    let exact = free as f64 * mine / (mine + rest);
    let share = if exact > 1.0 {
        (exact.round() as usize).max(2)
    } else {
        exact.round() as usize
    };

    // Never zero (a decode always needs the thread it is running on), never
    // more than is free, and never more than a decoder can use.
    share.clamp(1, free.min(MAX_DECODE_THREADS))
}

/// The process-wide decoder thread budget.
///
/// A ledger rather than a calculation, because `share_for` needs two numbers
/// that only a shared, serialized view can supply: how many threads are
/// genuinely unclaimed right now, and how much work is still waiting behind
/// this one. Holding them under one lock is also what makes the invariant
/// enforceable rather than merely intended -- a decode RESERVES its threads, so
/// the sum handed out can never exceed `--threads`, whatever order the claims
/// arrive in.
///
/// A worker that finds the budget fully claimed blocks here instead of
/// overcommitting. That is not lost time: every thread is already promised to a
/// decode that is running, and one of them will hand its share back. It cannot
/// deadlock, because `free == 0` means somebody is holding threads, and every
/// holder is a live decode whose `Grant` releases them on the way out --
/// including when it unwinds on an error or a Ctrl-C.
pub(super) struct ThreadBudget {
    state: Mutex<BudgetState>,
    released: Condvar,
}

struct BudgetState {
    /// Threads not currently promised to a running decode.
    free: usize,
    /// Total weight of the videos that have not started decoding yet. Shrinks
    /// as jobs are claimed, never as they finish -- a video that is already
    /// running is accounted for by the threads it holds, not by its weight.
    queued: u128,
}

/// A reservation against the budget, returned to it on drop.
pub(super) struct Grant<'a> {
    budget: &'a ThreadBudget,
    pub(super) threads: usize,
}

impl Drop for Grant<'_> {
    fn drop(&mut self) {
        {
            let mut state = self.budget.state.lock().unwrap_or_else(|e| e.into_inner());
            state.free += self.threads;
        }
        // notify_all rather than notify_one: several threads may come back at
        // once, and a single waiter waking to find one thread free when four
        // were released is how a wide decode gets starved by a narrow one.
        self.budget.released.notify_all();
    }
}

impl ThreadBudget {
    pub(super) fn new(total: usize, queued: u128) -> Self {
        ThreadBudget {
            state: Mutex::new(BudgetState {
                free: total.max(1),
                queued,
            }),
            released: Condvar::new(),
        }
    }

    /// Take the next video off the queue *and* its share of the budget, under
    /// one lock.
    ///
    /// The two have to happen together. `share_for` gives a video most of the
    /// machine only while the budget is still wide, and threads cannot be added
    /// to a decode once it has started -- so the whole schedule rests on the
    /// heaviest file, which is first in the weight-sorted queue, claiming
    /// first. Drawing the index from an atomic cursor and *then* claiming left a
    /// window between the two in which any number of lighter workers could
    /// overtake, and the heavyweight was then priced against a budget and a
    /// queue somebody else had already moved. It goes wrong in both directions,
    /// because an early claim takes a thread AND shrinks `queued`: with eight
    /// workers spawning at once on `[huge, small x 7]`, seven small files
    /// claiming first leave the heavyweight one thread where four were due --
    /// the straggler this whole schedule exists to prevent, and precisely the
    /// 12% the ordering was measured to be worth -- while a couple of them
    /// claiming first hands it MORE than the plan allowed for and starves the
    /// queue behind it. Nothing was ever over-committed by the race (the ledger
    /// is still one lock, and a `Grant` is still the only way to hold threads),
    /// so it cost wall time rather than correctness, and only at worker
    /// start-up, which is exactly when every worker is racing.
    ///
    /// The wait is given the queue as well: a worker that finds the budget fully
    /// committed with nothing left to claim would otherwise block until some
    /// decode finished, only to draw an index past the end of the queue and
    /// leave. It checks under the lock, so a cursor that reaches the end while
    /// this worker waits is seen on the next tick of the wait timeout.
    pub(super) fn claim_next<'a>(
        &'a self,
        cursor: &AtomicUsize,
        queue: &'a [Job],
    ) -> Option<(usize, &'a Job, Grant<'a>)> {
        let mut state =
            self.wait_for_a_free_thread(|| cursor.load(Ordering::SeqCst) < queue.len())?;

        let idx = cursor.fetch_add(1, Ordering::SeqCst);
        // Nothing is reserved on this path, so the guard simply drops.
        let job = queue.get(idx)?;

        let threads = Self::reserve(&mut state, job.weight);
        Some((idx, job, Grant { budget: self, threads }))
    }

    /// Block until the budget has a thread in it, holding the ledger on return.
    ///
    /// `more_to_claim` is re-asked on every wake-up, so a caller can give up on
    /// a queue that emptied while it waited.
    fn wait_for_a_free_thread(
        &self,
        more_to_claim: impl Fn() -> bool,
    ) -> Option<MutexGuard<'_, BudgetState>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        while state.free == 0 {
            if shutdown_requested() || !more_to_claim() {
                return None;
            }
            // A timeout rather than a bare wait purely so the shutdown check
            // above is reached even if the notification is missed.
            let (next, _) = self
                .released
                .wait_timeout(state, Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner());
            state = next;
        }

        Some(state)
    }

    /// Charge one video's share against the ledger and hand back the width.
    fn reserve(state: &mut BudgetState, weight: u64) -> usize {
        // No longer queued: from here on this video is represented by the
        // threads it is about to take, not by its weight. Doing this BEFORE
        // computing the share is what keeps `queued` meaning "the work I am
        // competing with" rather than "the work including me".
        state.queued = state.queued.saturating_sub(weight as u128);

        let threads = share_for(weight, state.queued, state.free);
        state.free -= threads;
        threads
    }
}

#[cfg(test)]
impl ThreadBudget {
    /// One reservation with no queue behind it.
    ///
    /// The decode pass claims through `claim_next`, which draws the video and reserves
    /// its share together; this is that reservation on its own, so a test can
    /// state a sequence of claims without also staging a queue for them.
    fn claim(&self, weight: u64) -> Option<Grant<'_>> {
        let mut state = self.wait_for_a_free_thread(|| true)?;
        let threads = Self::reserve(&mut state, weight);
        Some(Grant {
            budget: self,
            threads,
        })
    }

    fn free(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).free
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::stamp;

    // --- The decoder thread schedule -----------------------------------------

    // One "ordinary file" of decode work, so the shapes below read as
    // multiples of each other. The rule is scale-free -- only the ratio of
    // weight to queued matters -- so the size of the unit is arbitrary, and
    // these numbers meant bytes back when the weight did.
    const LOAD: u64 = 1 << 30;

    #[test]
    fn test_plenty_of_similar_work_means_one_thread_each() {
        // The common case, and the one the old count-based rule got right: with
        // seven times this much work queued behind it, widening this decode
        // would only take threads away from work that will outlast it anyway.
        assert_eq!(share_for(LOAD, 7 * LOAD as u128, 8), 1);
        assert_eq!(share_for(LOAD, 99 * LOAD as u128, 8), 1);
    }

    #[test]
    fn test_a_heavyweight_claims_its_share_of_the_run_up_front() {
        // The case the old rule got wrong. Eight tenths of the work left, on
        // eight threads: this file IS most of the run, so most of the machine
        // goes to it -- at the moment it starts, which is the only moment
        // FFmpeg will accept a thread count.
        assert_eq!(share_for(8 * LOAD, 2 * LOAD as u128, 8), 6);

        // Half the work gets half the threads.
        assert_eq!(share_for(4 * LOAD, 4 * LOAD as u128, 8), 4);
    }

    #[test]
    fn test_a_video_owed_more_than_one_thread_never_decodes_on_one() {
        // 1.43 threads of 8, which nearest-rounding gave 1. This is the shape a
        // library actually has -- one outsized file among a pile of ordinary
        // ones -- and 1 is the width the run cannot walk back later, so the
        // rounding goes the other way here and only here.
        assert_eq!(share_for(10 * LOAD, 48 * LOAD as u128, 8), 2);

        // Not a licence to round everything up: a share that is exactly one
        // thread is one thread, and a share below it still floors at one.
        assert_eq!(share_for(LOAD, 7 * LOAD as u128, 8), 1);
        assert_eq!(share_for(LOAD, 15 * LOAD as u128, 8), 1);

        // And above 2 the ordinary rule resumes -- 6.4 stays 6, because the
        // threads it would take are the ones the rest of the queue runs on.
        assert_eq!(share_for(8 * LOAD, 2 * LOAD as u128, 8), 6);
    }

    #[test]
    fn test_the_last_video_standing_takes_everything_free() {
        // Nothing queued behind it, so there is nobody to spread the budget
        // for. This is the straggler the whole change exists to prevent, and
        // the one place where the widest possible decode is unambiguously
        // right.
        assert_eq!(share_for(LOAD, 0, 8), 8);
        assert_eq!(share_for(LOAD, 0, 3), 3, "only what is actually free");
    }

    #[test]
    fn test_a_share_is_capped_at_what_a_decoder_can_use() {
        // Past MAX_DECODE_THREADS the extra threads buy nothing and cost a
        // full-resolution frame buffer each, so they are better left for the
        // next video.
        assert_eq!(share_for(LOAD, 0, 64), MAX_DECODE_THREADS);
    }

    #[test]
    fn test_a_small_video_never_gets_zero_threads() {
        // A decode always needs the thread it runs on, however little the file
        // weighs against the rest of the queue.
        assert_eq!(share_for(1, 100 * LOAD as u128, 8), 1);
        assert_eq!(share_for(0, 100 * LOAD as u128, 8), 1, "and a file we could not weigh still runs");
    }

    #[test]
    fn test_the_budget_is_never_overcommitted() {
        // The invariant `-t` is a promise about. Every claim here is made while
        // every earlier one is still running, which is the worst case for it.
        let shapes: [&[u64]; 4] = [
            &[8 * LOAD, LOAD, LOAD, LOAD, LOAD],
            &[LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD],
            &[64 * LOAD, 1, 1, 1, 1, 1, 1, 1],
            &[3 * LOAD, 2 * LOAD, LOAD],
        ];

        for total in 1..=16usize {
            for weights in shapes {
                let queued: u128 = weights.iter().map(|&w| w as u128).sum();
                let budget = ThreadBudget::new(total, queued);

                let mut held = 0usize;
                let mut grants = Vec::new();

                for &w in weights {
                    // Claiming with nothing free would (correctly) block, so
                    // this is where a single-threaded test has to stop.
                    if budget.free() == 0 {
                        break;
                    }
                    let grant = budget.claim(w).expect("not shutting down");
                    held += grant.threads;
                    grants.push(grant);

                    assert!(
                        held <= total,
                        "budget {} handed out {} threads across {} concurrent decodes",
                        total,
                        held,
                        grants.len()
                    );
                }
            }
        }
    }

    #[test]
    fn test_shares_widen_as_the_queue_drains() {
        // Four equal videos on eight threads. Two threads each while the queue
        // is full, four when half of it is gone, and the whole budget for the
        // last one -- which is exactly the ramp that was missing.
        let budget = ThreadBudget::new(8, 4 * LOAD as u128);

        let a = budget.claim(LOAD).unwrap();
        let b = budget.claim(LOAD).unwrap();
        assert_eq!((a.threads, b.threads), (2, 2));

        drop(a);
        drop(b);

        let c = budget.claim(LOAD).unwrap();
        assert_eq!(c.threads, 4);

        drop(c);

        let d = budget.claim(LOAD).unwrap();
        assert_eq!(d.threads, 8, "the tail of a scan must not decode single-threaded");
    }

    #[test]
    fn test_the_heaviest_video_is_provisioned_before_the_filler() {
        // The exact shape that motivated this: one enormous file and a pile of
        // small ones. The big one is claimed first (the todo list is sorted
        // largest-first) and takes most of the machine; the small ones share
        // what is left, one thread apiece, and finish alongside it instead of
        // finishing an hour early and leaving seven cores idle.
        let small = LOAD / 10;
        let queued = 8 * LOAD as u128 + 20 * small as u128;
        let budget = ThreadBudget::new(8, queued);

        let heavy = budget.claim(8 * LOAD).unwrap();
        assert_eq!(heavy.threads, 6);

        let a = budget.claim(small).unwrap();
        let b = budget.claim(small).unwrap();
        assert_eq!((a.threads, b.threads), (1, 1));
        assert_eq!(budget.free(), 0, "and the machine is fully committed");
    }

    #[test]
    fn test_the_heaviest_video_is_still_first_when_every_worker_starts_at_once() {
        // The ordering the schedule rests on, asserted against the moment it is
        // hardest to keep: every worker spawning together and racing for the
        // head of the queue. `claim_next` draws the index and reserves the
        // share under one lock, so whichever worker wins that race is
        // necessarily the one holding job 0 -- and job 0 is therefore always
        // priced against a full budget and an untouched queue, whatever the
        // scheduler does with the other fifteen.
        //
        // Drawing the index from the cursor and claiming afterwards left a
        // window between the two, and any of the other files could reserve
        // inside it; the heavyweight was then priced against a budget and a
        // queue somebody else had already moved -- in either direction, since a
        // light file that claims early takes a thread AND shrinks `queued`.
        // Against that version this fails on the first or second iteration
        // (4 threads where 3 are due), and it takes the barrier to do it: thread
        // spawning is sequential enough that worker one is usually finished
        // before worker two exists.
        let jobs: Vec<Job> = std::iter::once(16 * LOAD)
            .chain(std::iter::repeat_n(LOAD, 31))
            .map(|weight| Job {
                path: String::new(),
                stamp: stamp(0, 0),
                weight,
                keyframes: 0,
                size: 0,
                followers: Vec::new(),
            })
            .collect();

        let queued: u128 = jobs.iter().map(|j| j.weight as u128).sum();
        let expected = share_for(jobs[0].weight, queued - jobs[0].weight as u128, 8);
        assert!(expected > 1, "the test only means something if the head is owed a wide share");

        for _ in 0..50 {
            let budget = ThreadBudget::new(8, queued);
            let cursor = AtomicUsize::new(0);
            let taken: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
            // Spawning is sequential, so without this the first worker has
            // usually claimed before the second one exists and the race the
            // test is about never happens.
            let start = std::sync::Barrier::new(16);

            std::thread::scope(|scope| {
                for _ in 0..16 {
                    scope.spawn(|| {
                        start.wait();
                        // Released at once, so the queue drains rather than
                        // deadlocking sixteen workers against eight threads.
                        while let Some((idx, _, grant)) = budget.claim_next(&cursor, &jobs) {
                            taken.lock().unwrap().push((idx, grant.threads));
                        }
                    });
                }
            });

            let mut taken = taken.into_inner().unwrap();
            taken.sort_unstable();
            assert_eq!(taken.len(), jobs.len(), "every video is claimed exactly once");
            assert!(
                taken.iter().map(|&(idx, _)| idx).eq(0..jobs.len()),
                "and no index is drawn twice"
            );
            assert_eq!(
                taken[0].1, expected,
                "the heaviest video must be priced against a budget nothing else has touched"
            );
        }
    }

    #[test]
    fn test_a_grant_is_returned_on_drop() {
        // Including when a decode unwinds on an error: the release is the
        // destructor, not a line at the end of the happy path.
        let budget = ThreadBudget::new(8, LOAD as u128);

        {
            let grant = budget.claim(LOAD).unwrap();
            assert_eq!(grant.threads, 8);
            assert_eq!(budget.free(), 0);
        }

        assert_eq!(budget.free(), 8, "a finished decode must free its threads");
    }

    #[test]
    fn test_a_single_thread_budget_still_runs() {
        let budget = ThreadBudget::new(1, 4 * LOAD as u128);
        let grant = budget.claim(LOAD).unwrap();
        assert_eq!(grant.threads, 1);
        assert_eq!(budget.free(), 0);
    }
}
