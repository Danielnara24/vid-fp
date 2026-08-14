//! Run-wide tally of everything that failed or was deliberately skipped.
//!
//! Every problem in here is already logged the moment it happens, but on a
//! large library those lines have scrolled far out of view by the time the
//! results table is printed. This collects them so the LAST thing a run prints
//! is an honest account of what it did *not* do -- and so the process can exit
//! non-zero when something went wrong, which is the only part a script sees.
//!
//! The split between "skipped" and "problems" is load-bearing: a skip is
//! something the user asked for (`--min-duration`, an `--exclude` folder, a hard
//! link already queued), a problem is something the user asked for that did NOT
//! happen. Only the latter touches the exit code.

use log::info;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Worked examples kept per category. The full text of every one is in the
/// log; these exist so the summary alone is actionable once the log has
/// scrolled past the point of usefulness.
const MAX_SAMPLES: usize = 3;

/// A counter with a few examples attached.
///
/// Relaxed ordering throughout: these are incremented from rayon workers and
/// read exactly once, single-threaded, after every worker has joined. The join
/// is the synchronization; the counter only needs atomicity, not ordering.
#[derive(Default)]
pub struct Tally {
    count: AtomicUsize,
    samples: Mutex<Vec<String>>,
}

impl Tally {
    /// Count one occurrence, keeping `detail` as an example if there is room.
    pub fn record(&self, detail: impl Into<String>) {
        // fetch_add returns the value from BEFORE the update, so this thread is
        // the (prev + 1)th occurrence. Once MAX_SAMPLES have already been
        // through there is nothing left to keep, and we skip the lock entirely
        // -- which matters when the failure is something like "permission
        // denied" repeated across fifty thousand files.
        if self.count.fetch_add(1, Ordering::Relaxed) >= MAX_SAMPLES {
            return;
        }

        let mut samples = self.samples.lock().unwrap_or_else(|e| e.into_inner());
        if samples.len() < MAX_SAMPLES {
            samples.push(detail.into());
        }
    }

    /// Count one occurrence that has no detail worth showing.
    pub fn bump(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    pub fn samples(&self) -> Vec<String> {
        self.samples
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

#[derive(Default)]
pub struct RunStats {
    // --- problems: the run did less than it was asked to ---------------------
    pub unresolved_includes: Tally,
    pub unresolved_excludes: Tally,
    pub unwalkable: Tally,
    pub unreadable: Tally,
    pub fingerprint_failed: Tally,
    pub clustering_abandoned: Tally,
    pub cache_write_failed: Tally,
    pub cache_purge_failed: Tally,
    pub delete_stale: Tally,
    pub delete_failed: Tally,
    pub report_unusable: Tally,
    pub report_write_failed: Tally,

    // --- skips: intentional, and not a reason to fail the run ----------------
    pub skipped_short: Tally,
    pub skipped_alias: Tally,
    pub skipped_excluded: Tally,
}

impl RunStats {
    fn problems(&self) -> [(&Tally, &'static str); 12] {
        [
            // "path" rather than "folder": a scan target can now be a single
            // file, or a line piped in from another tool.
            (&self.unresolved_includes, "scan path(s) could not be resolved"),
            (
                &self.unresolved_excludes,
                "exclude folder(s) could not be resolved -- their contents were NOT excluded",
            ),
            (&self.unwalkable, "folder(s) could not be read while scanning"),
            (&self.unreadable, "file(s) could not be read"),
            (&self.fingerprint_failed, "video(s) could not be fingerprinted"),
            // The files were fingerprinted and compared; what could not be done
            // is enumerating every group they form, because a set of files that
            // nearly all match each other has combinatorially many. Squarely a
            // problem: those files are absent from the report entirely, and a
            // tighter --hamming-distance is what resolves it.
            //
            // Counted in clusters rather than files -- one entry is one set of
            // interlinked files, and the sample line beside it says how many
            // files that was. Counting the files would put a five-figure number
            // under a heading the user would then look for five figures of
            // detail behind.
            (
                &self.clustering_abandoned,
                "cluster(s) of files were too densely linked to group and are NOT in \
                 the report (try a tighter --hamming-distance)",
            ),
            (
                &self.cache_write_failed,
                "fingerprint(s) could not be cached (they will be redone next run)",
            ),
            // Nothing was lost and no file is at risk -- the entries describe
            // videos that are already gone -- but a cache that will not accept
            // a write is worth knowing about before the next run tries to write
            // several thousand, so it is a problem rather than a skip.
            (
                &self.cache_purge_failed,
                "cached fingerprint(s) of deleted file(s) could not be dropped \
                 (use --prune-cache to clear them)",
            ),
            // A problem rather than a skip, even though nothing went wrong and
            // the refusal is the point. The user asked for these files to be
            // deleted and they are still there, the report's figures for them
            // describe a file that no longer exists in that form, and the exit
            // code is the only part of that a script can see. Re-running judges
            // them as they now stand.
            (
                &self.delete_stale,
                "file(s) marked DELETE changed on disk after they were scanned \
                 and were NOT removed",
            ),
            (&self.delete_failed, "file(s) marked DELETE could not be removed"),
            // A --from-report row this build could not make sense of. Squarely a
            // problem rather than a skip: the user wrote that line meaning
            // something by it, the likeliest something is a misspelt DELETE, and
            // a file that quietly survived a run that was asked to remove it is
            // exactly the outcome the exit code exists to report.
            (
                &self.report_unusable,
                "row(s) of the report could not be understood -- the file(s) they name were NOT touched",
            ),
            // The run itself succeeded and any deletions really happened; what
            // was lost is the account of them. That is squarely a problem: the
            // one artefact the user asked to keep is missing, and on an armed
            // run it is the only record of what went. It is NOT fatal, because
            // returning an error here used to discard the list of removed files
            // before their cache entries could be dropped.
            (
                &self.report_write_failed,
                "report(s) could not be saved -- the run itself completed, but its output was lost",
            ),
        ]
    }

    // Work dropped by an interrupt is deliberately absent. The user pressed the
    // key; "you stopped it" already explains everything the number would, and
    // the exit code says it more precisely than a count of half-decoded files.
    fn skips(&self) -> [(&Tally, &'static str); 3] {
        [
            (&self.skipped_short, "video(s) shorter than --min-duration"),
            (
                &self.skipped_alias,
                "path(s) already queued under another name (symlink, hard link, or overlapping folders)",
            ),
            // Only ever raised for a path the user named or piped in. A file
            // dropped during a walk is not counted: the folder was never
            // descended into, and "you excluded it" is the whole story. A path
            // asked for BY NAME and then silently dropped is a different
            // matter, and the count is what keeps it from being silent.
            (
                &self.skipped_excluded,
                "named path(s) skipped because they sit under an --exclude folder",
            ),
        ]
    }

    pub fn problem_count(&self) -> usize {
        self.problems().iter().map(|(t, _)| t.count()).sum()
    }

    pub fn had_problems(&self) -> bool {
        self.problem_count() > 0
    }

    /// Print both sections, skips first.
    ///
    /// Skips go through the logger (informational, and correctly silenced by
    /// `--quiet`). Problems go straight to stderr instead, for the same reason
    /// the signal handler does: `--quiet` filters everything below Error, and a
    /// summary of what failed is exactly what a quiet run still needs to say.
    /// env_logger writes to stderr too, so the two never interleave out of
    /// order.
    pub fn print_summary(&self) {
        let skips: Vec<_> = self.skips().into_iter().filter(|(t, _)| t.count() > 0).collect();
        if !skips.is_empty() {
            info!("\nSkipped:");
            for (tally, label) in skips {
                for line in render(tally, label) {
                    info!("{}", line);
                }
            }
        }

        let problems: Vec<_> = self.problems().into_iter().filter(|(t, _)| t.count() > 0).collect();
        if !problems.is_empty() {
            eprintln!("\nProblems ({} total):", self.problem_count());
            for (tally, label) in problems {
                for line in render(tally, label) {
                    eprintln!("{}", line);
                }
            }
        }
    }
}

fn render(tally: &Tally, label: &str) -> Vec<String> {
    let count = tally.count();
    let samples = tally.samples();

    let mut out = Vec::with_capacity(samples.len() + 2);
    out.push(format!("  {:>5}  {}", count, label));
    for sample in &samples {
        out.push(format!("         - {}", sample));
    }

    // Only meaningful for a category that shows examples at all. A tally
    // counted with `bump()` keeps none by design -- there is no detail worth
    // showing for "shorter than --min-duration" beyond the number itself -- so
    // there is nothing being elided and nothing to apologise for. Announcing
    // "and 520 more" under a line that already reads 520 is pure noise.
    if !samples.is_empty() {
        let hidden = count.saturating_sub(samples.len());
        if hidden > 0 {
            out.push(format!("         - ... and {} more (see the errors above)", hidden));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_are_capped_but_the_count_is_not() {
        let t = Tally::default();
        for i in 0..50 {
            t.record(format!("file_{}.mp4", i));
        }

        assert_eq!(t.count(), 50, "every occurrence must be counted");
        assert_eq!(t.samples().len(), MAX_SAMPLES, "only a few examples are kept");
    }

    #[test]
    fn test_bump_counts_without_a_sample() {
        let t = Tally::default();
        t.bump();
        t.bump();

        assert_eq!(t.count(), 2);
        assert!(t.samples().is_empty());
    }

    #[test]
    fn test_skips_alone_never_fail_the_run() {
        // A run that skipped a thousand short videos, some hard links and an
        // excluded path did exactly what it was told to do. Exit code 0.
        let s = RunStats::default();
        for _ in 0..1000 {
            s.skipped_short.bump();
        }
        s.skipped_alias.bump();
        s.skipped_excluded.bump();

        assert_eq!(s.problem_count(), 0);
        assert!(!s.had_problems());
    }

    #[test]
    fn test_a_counted_only_tally_renders_as_a_single_line() {
        // The bug this guards: hidden was count - samples.len(), so a category
        // that never records examples claimed ALL of them were hidden --
        // "520 videos" immediately followed by "... and 520 more".
        let t = Tally::default();
        for _ in 0..520 {
            t.bump();
        }

        let lines = render(&t, "video(s) shorter than --min-duration");
        assert_eq!(lines.len(), 1, "nothing was elided, so nothing may claim to be");
        assert!(lines[0].contains("520"));
    }

    #[test]
    fn test_elision_is_reported_only_for_the_examples_beyond_the_cap() {
        let t = Tally::default();
        for i in 0..10 {
            t.record(format!("file_{}.mp4", i));
        }

        let lines = render(&t, "video(s) could not be fingerprinted");
        // Header, MAX_SAMPLES examples, and one line for the remaining 7.
        assert_eq!(lines.len(), MAX_SAMPLES + 2);
        assert!(lines.last().unwrap().contains("and 7 more"));
    }

    #[test]
    fn test_problem_count_spans_every_category() {
        let s = RunStats::default();
        s.unresolved_includes.record("/nope");
        s.unresolved_excludes.record("/nope-either");
        s.unwalkable.record("/root/private");
        s.unreadable.record("/a.mp4");
        s.fingerprint_failed.record("/b.mkv");
        s.clustering_abandoned.record("900 file(s) linked too densely to group");
        s.cache_write_failed.record("/c.mp4");
        s.cache_purge_failed.record("cache write failed");
        s.delete_stale.record("/d.mp4: 100 bytes when scanned, 200 bytes now");
        s.delete_failed.record("/e.mp4");
        s.report_unusable.record("report.csv line 4");

        assert_eq!(s.problem_count(), 11);
        assert!(s.had_problems());
    }
}