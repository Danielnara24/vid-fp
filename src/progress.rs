//! The progress line: one bar for the whole run, as in the sibling `img-fp`.
//!
//! One bar, not one per stage. A bar that fills and starts again at every
//! stage says how far through *this* stage the run is and nothing about the
//! wait, and the wait is the question. It also hid a stage outright: phase 1
//! of the comparison had no bar at all, so on a warm scan three quarters of
//! "Comparing videos" was a screen that did not move. So the bar is the share
//! of the run's time that has passed, as well as the run can tell it.
//!
//! **The bar is driven by the time left.** Every frame the run works out how
//! long it has still to go (`State::left`), and the bar closes that share of
//! the distance to its end: a frame of `dt` seconds with `left` to go moves it
//! `dt / left` of the way from where it stands to 100%. When `left` is right
//! the bar moves in step with the clock; when it is revised the bar speeds up
//! or slows down from where it is, and it never jumps, moves back or reaches
//! the end before the run does. Nothing reserves a stretch of the bar for a
//! stage before it runs, so nothing has to guess how long a stretch is.
//!
//! **The time left is counted work at a measured pace.** Each stage's work is
//! counted in its own unit (files, bytes, keyframe-pixels, hash pairs) as soon
//! as the run knows it (`Known`), and until then it is the most the code can
//! make of what is known. What a unit costs is measured: the running stage by
//! itself, and a stage that has not run yet by its `Stage::prior`, scaled by
//! how fast this machine has been on everything computed so far
//! (`Paces::speed`). A prior shapes the bar only until its stage begins, and a
//! wrong one shows as the bar changing speed, never as it stopping.
//!
//! Workers report into atomic counters (`Counter`) and a drawing thread turns
//! them into a position, so a worker never takes a lock to say it has done
//! something.
//!
//! **Where it goes.** On a terminal it is drawn on stderr, and only when the
//! console is listening (`utils::console_is_verbose`: not under `-q`, and not
//! in a unit test, which installs no logger). Every log line the run prints
//! goes through `println`, above the bar rather than through it. Under the
//! desktop app's worker (`speak`) it is said instead, one JSON object a line
//! on stdout, which the window (`vid-fp-gui`'s `scan.rs`) reads:
//!
//! - `{"l": "..."}` is a log line, exactly as the terminal would have shown it.
//! - `{"p": 0.42, "m": "..."}` is where the bar stands, 0 to 1, and what the
//!   run is doing.
//!
//! `VID_FP_PACES=1` prints, when the bar finishes, what each stage cost a unit
//! on this machine: what `Stage::prior` is read from.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressFinish, ProgressState, ProgressStyle};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// A box-drawing rule rather than an ASCII pipe: it cannot be confused with
/// the `|` that shows up in file names.
const RULE: &str = " \u{2502} ";

/// Positions on the bar. Far more than it has columns, so the percentage
/// beside it moves in tenths.
const LEN: u64 = 1_000_000;

/// How often the drawing thread moves the bar.
const FRAME: Duration = Duration::from_millis(50);

/// How often the window is told where the bar stands, at most.
const SPEAK_EVERY: Duration = Duration::from_millis(150);

/// Whether the worker speaks: set once, by `speak`.
static SPEAKING: AtomicBool = AtomicBool::new(false);

/// Held while a line is written, so a log line and a progress line never
/// interleave on stdout.
static STDOUT: Mutex<()> = Mutex::new(());

/// The bar of the run in progress, if one is.
static RUN: Mutex<Option<Running>> = Mutex::new(None);

/// The bar itself, for the one caller that cannot take `RUN`'s lock: the
/// interrupt handler, which may fire while a frame holds it.
static BAR: Mutex<Option<ProgressBar>> = Mutex::new(None);

/// Speak rather than draw: from now on the bar is said on stdout as JSON.
/// Called once, by the window's worker, before any bar exists.
pub fn speak() {
    SPEAKING.store(true, Ordering::SeqCst);
}

/// Whether the worker is speaking.
pub fn speaking() -> bool {
    SPEAKING.load(Ordering::Relaxed)
}

/// A log line, for the window's scan log.
pub fn log_line(line: &str) {
    write_line(&serde_json::json!({ "l": line }).to_string());
}

fn write_line(line: &str) {
    let _held = STDOUT.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = std::io::stdout().lock();
    // A window that has gone away is answered by its signal, not here.
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// A line on stderr, above the bar when one is drawn. A write that fails (a
/// reader that has gone) is not the run's problem.
pub fn println(line: &str) {
    let bar = BAR.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let say = || {
        let _ = writeln!(std::io::stderr().lock(), "{line}");
    };
    match bar {
        Some(bar) => bar.suspend(say),
        None => say(),
    }
}

/// Clear the bar, so that a line printed from outside the run (an interrupt)
/// starts on a clean line rather than after half a bar.
pub fn clear_for_exit() {
    if let Some(bar) = BAR.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        bar.finish_and_clear();
    }
}

/// The stages of a run, in the order they run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    /// Walking the folders: nothing is known to measure the rest by yet.
    Walk,
    CacheRead,
    /// Files opened, header only, to price their decode.
    Weighing,
    /// Files sampled for what they hold, and possible copies read whole.
    Identifying,
    Hashing,
    /// The decode, in keyframe-pixels (`fingerprint::weigh_decode`).
    Fingerprinting,
    /// Phase 1 of the comparison: which pairs could match at all.
    Candidates,
    /// Phase 2: every proposed pair measured.
    Verifying,
    Grouping,
    /// Saving the window's result, the frames behind each link included.
    Saving,
}

const STAGES: usize = Stage::ALL.len();

impl Stage {
    const ALL: [Stage; 10] = [
        Stage::Walk,
        Stage::CacheRead,
        Stage::Weighing,
        Stage::Identifying,
        Stage::Hashing,
        Stage::Fingerprinting,
        Stage::Candidates,
        Stage::Verifying,
        Stage::Grouping,
        Stage::Saving,
    ];

    fn label(self) -> &'static str {
        match self {
            Stage::Walk => "Looking for videos",
            Stage::CacheRead => "Reading the cache",
            Stage::Weighing => "Measuring decode cost",
            Stage::Identifying => "Identifying files",
            Stage::Hashing => "Checking for identical copies",
            Stage::Fingerprinting => "Fingerprinting",
            Stage::Candidates => "Finding candidate pairs",
            Stage::Verifying => "Comparing",
            Stage::Grouping => "Grouping duplicates",
            Stage::Saving => "Saving the results",
        }
    }

    /// The unit of `Known::units` for the stage, for the pace lines.
    fn unit(self) -> &'static str {
        match self {
            Stage::Walk => "-",
            Stage::CacheRead | Stage::Weighing | Stage::Identifying => "files",
            Stage::Hashing => "bytes",
            Stage::Fingerprinting => "keyframe-pixels",
            Stage::Candidates | Stage::Verifying => "hash pairs",
            Stage::Grouping => "files and links",
            Stage::Saving => "hash pairs of evidence",
        }
    }

    /// Which stages' paces say something about this one's. `None` for the
    /// cache read, the weighing pass and the content pass, which wait on the
    /// disk and say nothing about how fast anything else will go.
    ///
    /// The decode's pace is the machine's. The comparison's stages are the
    /// LIBRARY's as much as the machine's: what a hash pair costs depends on
    /// how its hashes crowd the index, how many pairs phase 1 proposes and how
    /// many frames match, and it moves between libraries by far more than
    /// between machines. Read off the decode, phase 1 was expected to cost
    /// 14 ns a hash pair and took 214 on VIDS2 (a football match, a lecture
    /// course, two films), phase 2 5.5 and took 74, the window's evidence 15
    /// times its prior too; and while each stage ran, the stages after it were
    /// still priced at the decode's speed, so phase 1 alone took the bar to 96%.
    /// The three moved together, which is what makes them one family.
    fn family(self) -> Option<Family> {
        match self {
            Stage::Fingerprinting => Some(Family::Decode),
            Stage::Candidates | Stage::Verifying | Stage::Grouping | Stage::Saving => Some(Family::Compare),
            _ => None,
        }
    }

    /// What a unit of the stage's work cost on the machine these were
    /// measured on (a Ryzen 7 3700U at eight threads), in nanoseconds of each
    /// thread: the stage's wall time, times eight, over its units, as
    /// `VID_FP_PACES=1` prints them. Read off a warm scan of the generated
    /// corpus (1,256 files) and cold scans of two parts of it (51 and 170
    /// files) at the default settings. Only their ratios matter once the
    /// machine has computed something (`Paces::speed`), and a stage's own pace
    /// replaces its prior as soon as it runs.
    fn prior(self) -> f64 {
        match self {
            Stage::Walk => 0.0,
            // 50-240 k: a lookup in an open snapshot.
            Stage::CacheRead => 200_000.0,
            // 1.1-3.4 M: a header read.
            Stage::Weighing => 2_500_000.0,
            // 1.3-2.0 M: three 64 KB reads.
            Stage::Identifying => 1_500_000.0,
            // Not measured: reading a file whole, at about a gigabyte a second.
            Stage::Hashing => 8.0,
            // 11-27 a unit of `decode_cost` over seven cold scans of parts of
            // both corpora, the most where a film decodes alone at the end:
            // intra decode of the keyframes and their hashing, most of a cold
            // scan.
            Stage::Fingerprinting => 15.0,
            // 14-41, the less the larger the library: a pair's scan stops at
            // its first admissible frame.
            Stage::Candidates => 14.0,
            // 5.0-5.7.
            Stage::Verifying => 5.5,
            // 6-360 k, the less the more links there are to spread its
            // start-up over.
            Stage::Grouping => 10_000.0,
            // 220-390 a hash pair of `evidence_cost`, on VIDS and parts of
            // VIDS and VIDS2: the frames behind each link
            // (`compare::evidence`), and the result written after them.
            Stage::Saving => 270.0,
        }
    }
}

/// How many hashes a file is taken to hold before anything has said: what a
/// fingerprint holds on average once samples closer than the alignment window
/// are folded (`compare::thin_samples`), on the generated corpora.
const HASHES_PER_FILE: f64 = 60.0;

/// What hashing one keyframe costs beyond decoding it, in the decode's own
/// unit: scaling it to the working frame, cropping it, and hashing it plain,
/// mirrored and in every reframing. Fitted over 278 cold decodes (five folders
/// of the two generated corpora) as thread time = 12.5 ns a keyframe-pixel +
/// 4.6 ms a keyframe, which is this many pixels a keyframe. Without it the bar
/// read the heaviest files first, at twice the pace the small ones then went
/// at, and spent the last third of a decode near 100%.
pub const KEYFRAME_PIXELS: u64 = 370_000;

/// What a file's decode weighs on the bar: its keyframe-pixels
/// (`fingerprint::weigh_decode`), and the hashing of each keyframe.
pub fn decode_cost(weight: u64, keyframes: u64) -> u64 {
    weight.saturating_add(keyframes.saturating_mul(KEYFRAME_PIXELS))
}

/// What measuring one link again for the window's evidence weighs, in hash
/// pairs (`found::Found::of`): the product of the two sample counts, which is
/// what `compare::evidence` compares, and a fixed part for what it builds per
/// link whatever the files hold.
pub fn evidence_cost(a: usize, b: usize) -> u64 {
    (a as u64 * b as u64).saturating_add(EVIDENCE_FIXED)
}

/// The fixed part of `evidence_cost`, in hash pairs.
const EVIDENCE_FIXED: u64 = 10_000;

/// What the run has counted so far of the work ahead. A field is a count the
/// run has made, or nothing yet; `units` makes the most of whichever.
#[derive(Clone, Debug, Default)]
pub struct Known {
    /// Files the walk found, and their bytes.
    pub files: usize,
    pub bytes: u64,
    /// Files the cache could not answer, and their bytes.
    pub misses: Option<(usize, u64)>,
    /// Hashes the cache handed back, and over how many files.
    pub cached_hashes: (usize, u64),
    /// Bytes the content pass will read whole.
    pub hashing: Option<u64>,
    /// The decode, in `decode_cost`, once the weighing pass has priced it.
    pub decode_work: Option<u64>,
    /// The comparison's work in hash pairs, once the fingerprints are in.
    pub hash_pairs: Option<f64>,
    /// Links the comparison found.
    pub matches: Option<usize>,
    /// The window's evidence, in `evidence_cost`, once the groups are known.
    pub evidence: Option<f64>,
    /// Whether the window's result is saved before the report.
    pub window: bool,
}

impl Known {
    fn misses(&self) -> (f64, f64) {
        self.misses.map_or((self.files as f64, self.bytes as f64), |(n, b)| (n as f64, b as f64))
    }

    /// The comparison's hash pairs: as counted, or every pair of the hashes
    /// the cache has handed back and the decode is expected to add.
    fn hash_pairs(&self) -> f64 {
        if let Some(p) = self.hash_pairs {
            return p;
        }
        let (n, h) = self.cached_hashes;
        let each = if n > 0 { h as f64 / n as f64 } else { HASHES_PER_FILE };
        let hashes = h as f64 + (self.files.saturating_sub(n)) as f64 * each;
        hashes * hashes / 2.0
    }

    /// The stage's work in its unit: what has been counted, or the most that
    /// can be made of what has.
    fn units(&self, s: Stage) -> f64 {
        // Until the comparison has counted them, every file is taken to have
        // a link: the window's evidence is priced by them, and priced at
        // nothing it was the one stage the bar left no room for, so phase 1
        // and 2 took it to 99.9% with seconds of evidence still to come. Too
        // many only means a bar that catches up once they are counted.
        let links = self.matches.map_or(self.files as f64, |m| m as f64);
        match s {
            Stage::Walk => 0.0,
            Stage::CacheRead => self.files as f64,
            Stage::Weighing | Stage::Identifying => self.misses().0,
            Stage::Hashing => self.hashing.unwrap_or(0) as f64,
            // Before it is weighed, the bottom rung of `weigh_decode`'s ladder.
            Stage::Fingerprinting => {
                self.decode_work.map_or(self.misses().1 * crate::fingerprint::WORK_PER_BYTE, |w| w as f64)
            }
            Stage::Candidates | Stage::Verifying => self.hash_pairs(),
            Stage::Grouping => self.files as f64 + links,
            // Until the groups say, every link's file measured twice (against
            // its best link and its group's first) at the library's average
            // sample count.
            Stage::Saving => {
                if !self.window {
                    0.0
                } else if let Some(e) = self.evidence {
                    e
                } else {
                    let each = (2.0 * self.hash_pairs()).sqrt() / self.files.max(1) as f64;
                    2.0 * links.min(self.files as f64) * (each * each + EVIDENCE_FIXED as f64)
                }
            }
        }
    }
}

/// Stages whose paces move together (`Stage::family`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Family {
    Decode,
    Compare,
}

/// What the run has measured of each stage's pace, and of this machine's
/// speed against the priors'.
struct Paces {
    /// Wall seconds and units of work done, by stage, over stages that have
    /// finished.
    secs: [f64; STAGES],
    units: [f64; STAGES],
    /// The running stage so far: its seconds and the units it has done.
    live: Option<(Stage, f64, f64)>,
    threads: f64,
}

impl Paces {
    /// Wall seconds a prior's nanosecond takes, for a stage of `family`: what
    /// the family has measured so far (the running stage included) against
    /// what its priors say it would have cost; failing that, every stage that
    /// computes; failing that, the priors' machine at this run's threads.
    fn speed(&self, family: Option<Family>) -> f64 {
        let measured = |keep: &dyn Fn(Stage) -> bool| {
            let (mut secs, mut ns) = (0.0, 0.0);
            for s in Stage::ALL.into_iter().filter(|&s| keep(s)) {
                secs += self.secs[s as usize];
                ns += self.units[s as usize] * s.prior();
            }
            if let Some((s, sec, units)) = self.live {
                if keep(s) {
                    secs += sec;
                    ns += units * s.prior();
                }
            }
            (secs >= 0.2 && ns > 0.0).then(|| secs / ns)
        };
        family
            .and_then(|f| measured(&|s| s.family() == Some(f)))
            .or_else(|| measured(&|s| s.family().is_some()))
            .unwrap_or(1e-9 / self.threads)
    }

    /// Seconds a unit of the stage takes: as measured when it has run long
    /// enough to say, else its prior at its family's speed.
    fn pace(&self, s: Stage) -> f64 {
        let (secs, units) = (self.secs[s as usize], self.units[s as usize]);
        if secs >= 0.2 && units > 0.0 {
            secs / units
        } else {
            s.prior() * self.speed(s.family())
        }
    }
}

/// What is left of a stage expected to take `expected` seconds, `secs` into
/// it: the difference until half of it has passed, then `expected² / 4 secs`,
/// which meets it there with the same slope and never reaches zero. A stage
/// that runs past its estimate slows the bar down rather than sending it to
/// the end.
fn overrun(expected: f64, secs: f64) -> f64 {
    if secs <= expected / 2.0 {
        expected - secs
    } else {
        expected * expected / (4.0 * secs)
    }
}

/// What a counted stage's workers report into. Two counts: `weight` moves the
/// bar and `items` is what the message says, because a video is one video on
/// the screen and a thousand times another one's cost on the bar.
#[derive(Default)]
pub struct Counter {
    weight: AtomicU64,
    items: AtomicU64,
}

impl Counter {
    /// Items done, weighing `weight` between them.
    #[inline]
    pub fn add(&self, items: u64, weight: u64) {
        if weight > 0 {
            self.weight.fetch_add(weight, Ordering::Relaxed);
        }
        if items > 0 {
            self.items.fetch_add(items, Ordering::Relaxed);
        }
    }

    /// One item done, of weight one.
    #[inline]
    pub fn tick(&self) {
        self.add(1, 1);
    }

    pub fn items(&self) -> u64 {
        self.items.load(Ordering::Relaxed)
    }
}

/// A worker's unreported work, handed to a `Counter` in batches.
///
/// A pair is measured in microseconds, so reporting each one is an atomic add
/// on a line every thread is fighting over. The flush on drop is what makes
/// the batching invisible rather than merely coarse: rayon drops a worker's
/// state when its slice is done, so the last partial batch is always counted.
pub struct Batch<'a> {
    counter: &'a Counter,
    items: u64,
    weight: u64,
}

impl<'a> Batch<'a> {
    /// How many items a worker gets through before it reports. Small enough
    /// that a library of a few hundred pairs still moves the bar as it goes.
    const ITEMS: u64 = 32;

    pub fn new(counter: &'a Counter) -> Self {
        Batch { counter, items: 0, weight: 0 }
    }

    #[inline]
    pub fn add(&mut self, weight: u64) {
        self.items += 1;
        self.weight += weight;
        if self.items >= Self::ITEMS {
            self.flush();
        }
    }

    fn flush(&mut self) {
        self.counter.add(self.items, self.weight);
        self.items = 0;
        self.weight = 0;
    }
}

impl Drop for Batch<'_> {
    fn drop(&mut self) {
        if self.items > 0 || self.weight > 0 {
            self.flush();
        }
    }
}

/// How the running stage's progress is read.
enum Work {
    /// By what it was expected to take: nothing to count inside it.
    Uncounted,
    /// Items done of `items`, weighing `weight` in all; `per` seconds a unit
    /// of weight before anything is done.
    Counted { counter: Arc<Counter>, weight: u64, items: u64, per: f64 },
}

struct Current {
    stage: Stage,
    started: Instant,
    work: Work,
    /// The stage's work in its `Known::units`.
    units: f64,
    /// Said after the count: the file the decode is waiting on.
    detail: String,
}

struct State {
    known: Known,
    paces: Paces,
    /// Stages that have run.
    done: [bool; STAGES],
    current: Option<Current>,
    /// Where the bar stands, as a fraction of the whole; never decreases.
    at: f64,
    last: Instant,
    /// The decode's weight a few seconds back, for its speedometer.
    recent: std::collections::VecDeque<(Instant, u64)>,
    /// The running counted stage's weight, frame by frame, as far back as
    /// `recent_window` reaches: what its recent pace is read from.
    history: std::collections::VecDeque<(Instant, u64)>,
}

/// How far back a running stage's pace is read: the last quarter of its time
/// so far, and never less than two seconds.
fn recent_window(secs: f64) -> f64 {
    (secs / 4.0).max(2.0)
}

impl State {
    fn new(threads: usize) -> State {
        State {
            known: Known::default(),
            paces: Paces { secs: [0.0; STAGES], units: [0.0; STAGES], live: None, threads: threads.max(1) as f64 },
            done: [false; STAGES],
            current: None,
            at: 0.0,
            last: Instant::now(),
            recent: std::collections::VecDeque::new(),
            history: std::collections::VecDeque::new(),
        }
    }

    /// Seconds the stage would take, from what is known now.
    fn expected(&self, s: Stage) -> f64 {
        self.known.units(s) * self.paces.pace(s)
    }

    /// Seconds the run has still to go: the rest of the running stage, and
    /// every stage after it not yet run. `None` while nothing is known to
    /// measure by (the walk).
    fn left(&self, now: Instant) -> Option<f64> {
        let c = self.current.as_ref()?;
        if c.stage == Stage::Walk {
            return None;
        }
        let secs = now.duration_since(c.started).as_secs_f64();
        let running = match &c.work {
            Work::Uncounted => overrun(self.expected(c.stage), secs),
            Work::Counted { counter, weight, per, .. } => {
                let w = counter.weight.load(Ordering::Relaxed) as f64;
                let rest = (*weight as f64 - w).max(0.0);
                if w > 0.0 && secs >= 0.5 {
                    // At its RECENT pace, not its average. A stage's pace
                    // falls as it goes more often than not: the decode takes
                    // its heaviest files first, at a rate the small ones then
                    // do not keep up, and ends on one long file decoding alone
                    // while the other threads idle (VIDS2's seeds went 400 to
                    // 165 Mpx/s over the last tenth). Read off the average,
                    // the bar ran 15% ahead of the clock and crawled the rest.
                    // Never below a quarter of the average, so a stretch with
                    // nothing reported cannot stop the bar outright.
                    let average = w / secs;
                    let recent = self
                        .history
                        .front()
                        .filter(|(t, _)| now.duration_since(*t).as_secs_f64() >= 0.5)
                        .map_or(average, |&(t, w0)| (w - w0 as f64).max(0.0) / now.duration_since(t).as_secs_f64());
                    rest / recent.max(average / 4.0)
                } else {
                    // Until it has measured itself, its expected time, worn
                    // down as it overruns rather than as a constant.
                    overrun(rest * per, secs)
                }
            }
        };
        let after: f64 = Stage::ALL[c.stage as usize + 1..]
            .iter()
            .filter(|s| !self.done[**s as usize])
            .map(|&s| self.expected(s))
            .sum();
        Some(running.max(0.0) + after)
    }

    /// What the running stage has measured so far, for the stages after it
    /// (`Paces::speed`): its seconds and the units its counted work covers.
    /// Nothing for a stage with nothing counted inside it.
    fn live(&self, now: Instant) -> Option<(Stage, f64, f64)> {
        let c = self.current.as_ref()?;
        let Work::Counted { counter, weight, .. } = &c.work else { return None };
        if *weight == 0 {
            return None;
        }
        let w = counter.weight.load(Ordering::Relaxed) as f64;
        Some((c.stage, now.duration_since(c.started).as_secs_f64(), c.units * (w / *weight as f64).min(1.0)))
    }

    /// Close the running stage, and keep what it measured.
    fn close(&mut self) {
        self.paces.live = None;
        let Some(c) = self.current.take() else { return };
        let i = c.stage as usize;
        self.done[i] = true;
        let secs = c.started.elapsed().as_secs_f64();
        let units = match &c.work {
            Work::Uncounted => c.units,
            // The share of the stage's units its counted work covers: an
            // interrupted stage did only part of them.
            Work::Counted { counter, weight, .. } => {
                let w = counter.weight.load(Ordering::Relaxed) as f64;
                if *weight == 0 { c.units } else { c.units * (w / *weight as f64).min(1.0) }
            }
        };
        self.paces.secs[i] += secs;
        self.paces.units[i] += units;
    }

    fn start(&mut self, stage: Stage, work: impl FnOnce(&State) -> Work) {
        self.close();
        // Every stage before this one is over, whether it ran or not.
        for s in &Stage::ALL[..stage as usize] {
            self.done[*s as usize] = true;
        }
        let units = self.known.units(stage);
        let work = work(self);
        self.current = Some(Current { stage, started: Instant::now(), work, units, detail: String::new() });
        self.recent.clear();
        self.history.clear();
    }

    /// Where the bar stands now, and what to say about it.
    fn frame(&mut self) -> Option<(f64, String)> {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        self.paces.live = self.live(now);
        if let Some(Current { work: Work::Counted { counter, .. }, started, .. }) = &self.current {
            let w = counter.weight.load(Ordering::Relaxed);
            let window = recent_window(now.duration_since(*started).as_secs_f64());
            self.history.push_back((now, w));
            while self.history.len() > 2
                && now.duration_since(self.history[1].0).as_secs_f64() >= window
            {
                self.history.pop_front();
            }
        }
        if let Some(left) = self.left(now) {
            let left = left.max(0.05);
            self.at += (1.0 - self.at) * (dt / left).min(0.5);
        }
        let c = self.current.as_ref()?;
        let mut msg = match &c.work {
            Work::Uncounted => c.stage.label().to_string(),
            Work::Counted { counter, items, .. } => {
                let n = counter.items.load(Ordering::Relaxed).min(*items);
                match c.stage {
                    Stage::Hashing => format!(
                        "{} {} / {}",
                        c.stage.label(),
                        indicatif::HumanBytes(counter.weight.load(Ordering::Relaxed)),
                        indicatif::HumanBytes(c.units as u64)
                    ),
                    Stage::Candidates => c.stage.label().to_string(),
                    Stage::Fingerprinting => format!("{} {n}/{items} videos", c.stage.label()),
                    Stage::Verifying => format!(
                        "{} {} / {} pairs",
                        c.stage.label(),
                        indicatif::HumanCount(n),
                        indicatif::HumanCount(*items)
                    ),
                    _ => format!("{} {n}/{items}", c.stage.label()),
                }
            }
        };
        if let (Stage::Fingerprinting, Work::Counted { counter, .. }) = (c.stage, &c.work) {
            // A windowed rate, so it reads the current speed rather than the
            // run's average: decode work a second over the last few seconds.
            let w = counter.weight.load(Ordering::Relaxed);
            self.recent.push_back((now, w));
            while self.recent.len() > 2 && now.duration_since(self.recent[0].0) > Duration::from_secs(5) {
                self.recent.pop_front();
            }
            let (t0, w0) = self.recent[0];
            let secs = now.duration_since(t0).as_secs_f64();
            let rate = if secs > 0.5 { (w - w0) as f64 / secs } else { 0.0 };
            msg = format!("{msg}{RULE}{}", crate::work_rate(rate));
        }
        if !c.detail.is_empty() {
            msg = format!("{msg}{RULE}{}", c.detail);
        }
        Some((self.at, msg))
    }

    /// Each stage that ran: its work, the seconds it took, and the nanoseconds
    /// of each thread a unit took, which is what `Stage::prior` is read from.
    fn paces(&self) -> Vec<String> {
        let p = &self.paces;
        Stage::ALL
            .iter()
            .filter(|&&s| p.secs[s as usize] > 0.0 && s != Stage::Walk)
            .map(|&s| {
                let (secs, units) = (p.secs[s as usize], p.units[s as usize]);
                let per = if units > 0.0 { secs * p.threads * 1e9 / units } else { 0.0 };
                format!(
                    "pace: {}: {units:.0} {} in {secs:.2}s, {per:.3} ns a unit on each of {} threads (prior {})",
                    s.label(),
                    s.unit(),
                    p.threads,
                    s.prior()
                )
            })
            .collect()
    }
}

struct Running {
    bar: ProgressBar,
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    drawer: Option<JoinHandle<()>>,
}

/// The run's bar, while it is up: finished when dropped, whatever path the
/// run takes out, so a fatal error reads as the error and not as a bar frozen
/// above it.
pub struct Guard(());

impl Drop for Guard {
    fn drop(&mut self) {
        finish();
    }
}

/// Put the bar up for a run on `threads` threads, beginning with the walk.
/// Shown only when someone is listening: the window, or a console that is not
/// quiet. Otherwise every call below is a no-op and the counters they hand
/// out are counted into nothing.
pub fn start(threads: usize) -> Guard {
    finish();
    if !speaking() && !crate::utils::console_is_verbose() {
        return Guard(());
    }
    let bar = if speaking() {
        ProgressBar::hidden()
    } else {
        ProgressBar::with_draw_target(Some(LEN), ProgressDrawTarget::stderr())
    }
    .with_finish(ProgressFinish::AndClear);
    bar.set_style(
        ProgressStyle::with_template(&format!("{{elapsed_precise}}{RULE}[{{bar:28.cyan/blue}}]{RULE}{{pct}}{RULE}{{msg}}"))
            .unwrap()
            .with_key("pct", |s: &ProgressState, w: &mut dyn std::fmt::Write| {
                let _ = write!(w, "{:5.1}%", s.fraction() * 100.0);
            })
            .progress_chars("=>-"),
    );
    // Redrawn on a timer as well as on progress, so the clock keeps moving
    // through a stretch where the bar does not.
    if !speaking() {
        bar.enable_steady_tick(Duration::from_millis(200));
    }
    let mut state = State::new(threads);
    state.start(Stage::Walk, |_| Work::Uncounted);
    bar.set_message(Stage::Walk.label());
    let state = Arc::new(Mutex::new(state));
    let stop = Arc::new(AtomicBool::new(false));
    let drawer = {
        let (bar, state, stop) = (bar.clone(), state.clone(), stop.clone());
        std::thread::Builder::new()
            .name("progress".into())
            .spawn(move || {
                let mut said: Option<(u64, String)> = None;
                let mut last_said = Instant::now() - SPEAK_EVERY;
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(FRAME);
                    let frame = state.lock().unwrap_or_else(|e| e.into_inner()).frame();
                    let Some((at, msg)) = frame else { continue };
                    let pos = (at * LEN as f64) as u64;
                    if speaking() {
                        // Tenths of a percent are all the window can show.
                        let key = (pos / 1000, msg);
                        if said.as_ref() != Some(&key) && last_said.elapsed() >= SPEAK_EVERY {
                            write_line(&serde_json::json!({ "p": at, "m": key.1 }).to_string());
                            said = Some(key);
                            last_said = Instant::now();
                        }
                        continue;
                    }
                    bar.set_position(pos);
                    bar.set_message(msg);
                }
            })
            .ok()
    };
    *BAR.lock().unwrap_or_else(|e| e.into_inner()) = (!speaking()).then(|| bar.clone());
    *RUN.lock().unwrap_or_else(|e| e.into_inner()) = Some(Running { bar, state, stop, drawer });
    Guard(())
}

/// Take the bar down and clear its line. Idempotent: the run calls it before
/// the report's listing and the confirmation prompt, and the guard again on
/// the way out.
pub fn finish() {
    let Some(mut r) = RUN.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    r.stop.store(true, Ordering::Relaxed);
    if let Some(h) = r.drawer.take() {
        let _ = h.join();
    }
    *BAR.lock().unwrap_or_else(|e| e.into_inner()) = None;
    r.bar.finish_and_clear();
    let mut s = r.state.lock().unwrap_or_else(|e| e.into_inner());
    s.close();
    if std::env::var_os("VID_FP_PACES").is_some() {
        for line in s.paces() {
            let _ = writeln!(std::io::stderr().lock(), "{line}");
        }
    }
}

fn with_state(f: impl FnOnce(&mut State)) {
    let run = RUN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = run.as_ref() {
        f(&mut r.state.lock().unwrap_or_else(|e| e.into_inner()));
    }
}

/// Note what the run has counted of the work ahead.
pub fn learn(f: impl FnOnce(&mut Known)) {
    with_state(|s| f(&mut s.known));
}

/// A stage with nothing to count inside it.
pub fn begin(stage: Stage) {
    with_state(|s| s.start(stage, |_| Work::Uncounted));
}

/// A stage over `items` things whose weights sum to `weight`. Workers report
/// into the returned counter.
pub fn begin_counted(stage: Stage, weight: u64, items: u64) -> Arc<Counter> {
    let counter = Arc::new(Counter::default());
    let c = counter.clone();
    with_state(move |s| {
        s.start(stage, move |s| {
            let per = if weight == 0 { 0.0 } else { s.expected(stage) / weight as f64 };
            Work::Counted { counter: c, weight, items, per }
        })
    });
    counter
}

/// What the running stage is waiting on, said after its count.
pub fn detail(what: String) {
    with_state(|s| {
        if let Some(c) = s.current.as_mut() {
            c.detail = what;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(stage: Stage, s: &mut State, secs_in: u64) {
        s.start(stage, |_| Work::Uncounted);
        s.current.as_mut().unwrap().started -= Duration::from_secs(secs_in);
    }

    #[test]
    fn every_stage_has_its_words_and_a_price() {
        for (i, s) in Stage::ALL.iter().enumerate() {
            assert_eq!(*s as usize, i, "{s:?} out of order");
            assert!(!s.label().is_empty() && !s.unit().is_empty());
            assert!(s.prior() > 0.0 || *s == Stage::Walk, "{s:?}");
        }
    }

    #[test]
    fn a_stage_past_its_estimate_is_never_done() {
        let mut last = f64::INFINITY;
        for i in 0..1000 {
            let left = overrun(10.0, i as f64 * 0.05);
            assert!(left > 0.0 && left < last, "{i} {left}");
            last = left;
        }
        let (a, b) = (overrun(10.0, 4.999), overrun(10.0, 5.001));
        assert!((a - b).abs() < 0.003, "{a} {b}");
    }

    /// The walk knows nothing to measure by, so the bar waits at its start
    /// rather than racing towards an end it cannot see.
    #[test]
    fn the_bar_does_not_move_while_the_walk_runs() {
        let mut s = State::new(8);
        at(Stage::Walk, &mut s, 30);
        for _ in 0..10 {
            s.last -= Duration::from_millis(500);
            assert_eq!(s.frame().unwrap().0, 0.0);
        }
    }

    /// The comparison's first phase is counted like any other stage: the bar
    /// moves while it runs, and with its time left.
    #[test]
    fn the_bar_moves_with_the_time_left_and_never_back() {
        let mut s = State::new(8);
        s.known = Known { files: 100, bytes: 1 << 30, misses: Some((0, 0)), cached_hashes: (100, 10_000), ..Default::default() };
        let counter = Arc::new(Counter::default());
        let c = counter.clone();
        s.start(Stage::Candidates, move |_| Work::Counted { counter: c, weight: 100, items: 100, per: 0.1 });
        s.current.as_mut().unwrap().started -= Duration::from_secs(10);
        // Half the rows in ten seconds: ten seconds of candidates left, plus
        // what follows.
        counter.add(50, 50);
        let left = s.left(Instant::now()).unwrap();
        assert!(left > 10.0 - 0.1, "{left}");
        let mut last = 0.0;
        for _ in 0..20 {
            s.last -= Duration::from_millis(500);
            let (at, msg) = s.frame().unwrap();
            assert!(at > last && at < 1.0, "{at} {last}");
            assert_eq!(msg, "Finding candidate pairs");
            last = at;
        }
    }

    /// A library whose phase 1 runs at fifteen times its prior has a phase 2
    /// (and a window's evidence) that will too: the stages after the running
    /// one are priced at its pace, not at the decode's, which is what let
    /// phase 1 of VIDS2 take the bar to 96%.
    #[test]
    fn a_slow_comparison_slows_the_estimate_of_what_follows_it() {
        let mut s = State::new(8);
        s.known = Known { files: 640, hash_pairs: Some(1e9), ..Default::default() };
        // A decode at its prior: the machine is the priors' machine.
        s.paces.secs[Stage::Fingerprinting as usize] = 10.0;
        s.paces.units[Stage::Fingerprinting as usize] = 10.0 / (Stage::Fingerprinting.prior() * 1e-9 / 8.0);
        let before = s.expected(Stage::Verifying);

        let counter = Arc::new(Counter::default());
        let c = counter.clone();
        s.start(Stage::Candidates, move |_| Work::Counted { counter: c, weight: 100, items: 100, per: 0.0 });
        let secs = 60.0;
        s.current.as_mut().unwrap().started -= Duration::from_secs_f64(secs);
        // Half of it in what its prior says the whole would take, over 15.
        let whole = 1e9 * Stage::Candidates.prior() * 1e-9 / 8.0;
        counter.add(50, 50);
        s.current.as_mut().unwrap().started += Duration::from_secs_f64(secs - whole / 2.0 * 15.0);
        s.frame();
        let after = s.expected(Stage::Verifying);
        assert!((after / before - 15.0).abs() < 0.5, "{before} -> {after}");
        // The decode's own estimate does not move with it.
        assert!((s.paces.speed(Some(Family::Decode)) * 8.0 / 1e-9 - 1.0).abs() < 1e-6);
    }

    /// A stage whose pace has fallen is expected to go on at the pace it has
    /// now, not at its average: the decode's tail.
    #[test]
    fn the_time_left_is_read_at_the_recent_pace() {
        let mut s = State::new(8);
        let counter = Arc::new(Counter::default());
        let c = counter.clone();
        s.start(Stage::Fingerprinting, move |_| Work::Counted { counter: c, weight: 1000, items: 10, per: 0.0 });
        let t0 = Instant::now() - Duration::from_secs(20);
        s.current.as_mut().unwrap().started = t0;
        // 600 in the first 16 seconds, then 40 in the last 4: ten a second.
        s.history.push_back((t0 + Duration::from_secs(16), 600));
        counter.add(0, 640);
        let left = s.left(t0 + Duration::from_secs(20)).unwrap();
        assert!((left - 36.0).abs() < 0.5, "{left}");
        // Nothing at all lately is a quarter of the average, not a stop.
        s.history.clear();
        s.history.push_back((t0 + Duration::from_secs(16), 640));
        let left = s.left(t0 + Duration::from_secs(20)).unwrap();
        assert!((left - 360.0 / 8.0).abs() < 0.5, "{left}");
    }

    #[test]
    fn a_counted_stage_says_its_count() {
        let mut s = State::new(8);
        let counter = Arc::new(Counter::default());
        let c = counter.clone();
        s.start(Stage::Verifying, move |_| Work::Counted { counter: c, weight: 400, items: 4000, per: 0.0 });
        counter.add(1000, 100);
        assert_eq!(s.frame().unwrap().1, "Comparing 1,000 / 4,000 pairs");
        s.current.as_mut().unwrap().detail = "episode 4.mkv".into();
        assert_eq!(s.frame().unwrap().1, "Comparing 1,000 / 4,000 pairs \u{2502} episode 4.mkv");
    }

    #[test]
    fn a_batch_reports_everything_it_was_given() {
        let counter = Counter::default();
        {
            let mut b = Batch::new(&counter);
            for _ in 0..(Batch::ITEMS * 3 + 5) {
                b.add(2);
            }
            assert_eq!(counter.items(), Batch::ITEMS * 3);
        }
        assert_eq!(counter.items(), Batch::ITEMS * 3 + 5);
        assert_eq!(counter.weight.load(Ordering::Relaxed), 2 * (Batch::ITEMS * 3 + 5));
    }

    /// Off a terminal and off the window, nothing is put up and the counters
    /// still count.
    #[test]
    fn nothing_is_drawn_unless_someone_is_listening() {
        assert!(!speaking());
        let _g = start(8);
        assert!(RUN.lock().unwrap().is_none());
        let c = begin_counted(Stage::Verifying, 10, 10);
        c.tick();
        assert_eq!(c.items(), 1);
    }
}
