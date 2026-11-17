//! What the desktop app's worker says on stdout, one JSON object a line.
//!
//! A run on a terminal draws progress bars on stderr and logs lines there.
//! Under the window (`worker_main`) neither has a terminal to draw on, so both
//! become lines the window reads (`vid-fp-gui`'s `scan.rs`):
//!
//! - `{"l": "..."}` is a log line, exactly as the terminal would have shown it.
//! - `{"p": 0.42, "m": "..."}` is where the current bar stands, 0 to 1, and
//!   what it is measuring.
//!
//! **The bars are the CLI's own bars.** Every place that makes one asks
//! `spoken` first, and while the worker is speaking it gets back a bar that
//! draws nothing and is registered here instead. A thread reads the registered
//! bar a few times a second and says where it stands. So the window's bar
//! moves exactly as the terminal's would, from the same counts, and nothing in
//! the stages that advance them knows the window exists.
//!
//! Between bars, `stage` says what the run is doing (walking folders, reading
//! the cache, grouping), at the start of a bar of no length.

use indicatif::{ProgressBar, ProgressDrawTarget};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// Whether the worker speaks: set once, by `speak`.
static SPEAKING: AtomicBool = AtomicBool::new(false);

/// The bar the window is shown, and what it measures.
static CURRENT: Mutex<Option<(ProgressBar, Stage)>> = Mutex::new(None);

/// Held while a line is written, so a log line and a progress line never
/// interleave on stdout.
static STDOUT: Mutex<()> = Mutex::new(());

/// How often the window is told where the bar stands.
const EVERY: Duration = Duration::from_millis(150);

/// What a bar measures, and how the window words it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The weighing pass: files opened to price their decode.
    Weighing,
    /// Sampling files the cache could not answer by path, in files.
    Identifying,
    /// Reading possible copies whole, in bytes.
    Hashing,
    /// The decode: keyframe-pixels, with the files done in the bar's prefix
    /// and the file being waited on in its message.
    Fingerprinting,
    /// Phase 2 of the comparison: candidate pairs measured.
    Comparing,
}

/// Start speaking: from now on `spoken` hands out bars that draw nothing, and
/// a thread says where they stand. Called once, by the worker, before any bar
/// exists.
pub fn speak() {
    if SPEAKING.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::Builder::new()
        .name("progress".into())
        .spawn(|| {
            let mut last: Option<String> = None;
            loop {
                std::thread::sleep(EVERY);
                let said = CURRENT.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|(pb, stage)| describe(pb, *stage));
                if let Some((p, m)) = said {
                    let line = serde_json::json!({ "p": p, "m": m }).to_string();
                    if last.as_deref() != Some(line.as_str()) {
                        write_line(&line);
                        last = Some(line);
                    }
                }
            }
        })
        .ok();
}

/// Whether the worker is speaking.
pub fn speaking() -> bool {
    SPEAKING.load(Ordering::Relaxed)
}

/// The bar for `stage`, `len` long, while the worker speaks: one that draws
/// nothing, registered as the bar the window is shown. `None` on a terminal,
/// where the caller makes its own.
pub fn spoken(len: u64, stage: Stage) -> Option<ProgressBar> {
    if !speaking() {
        return None;
    }
    let pb = ProgressBar::with_draw_target(Some(len), ProgressDrawTarget::hidden());
    *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = Some((pb.clone(), stage));
    Some(pb)
}

/// Say what the run is doing between bars: a bar of no length, at its start.
pub fn stage(what: &str) {
    if !speaking() {
        return;
    }
    *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = None;
    write_line(&serde_json::json!({ "p": 0.0, "m": what }).to_string());
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

/// Where `pb` stands, 0 to 1, and the words the window shows beside it.
fn describe(pb: &ProgressBar, stage: Stage) -> (f64, String) {
    let (pos, len) = (pb.position(), pb.length().unwrap_or(0));
    let p = if len == 0 { 0.0 } else { (pos as f64 / len as f64).clamp(0.0, 1.0) };
    let m = match stage {
        Stage::Weighing => format!("Measuring decode cost: {pos} of {len} files"),
        Stage::Identifying => format!("Identifying files: {pos} of {len}"),
        Stage::Hashing => format!(
            "Checking for identical copies: {} of {} MB",
            pos / 1_000_000,
            len / 1_000_000
        ),
        Stage::Fingerprinting => {
            // The prefix is "done/total" files, the message the file the run
            // is waiting on; see the decode bar in `run`.
            let done = pb.prefix().replace('/', " of ");
            match pb.message() {
                m if m.is_empty() => format!("Fingerprinting: {done} videos"),
                m => format!("Fingerprinting: {done} videos · {m}"),
            }
        }
        Stage::Comparing => format!("Comparing: {pos} of {len} pairs"),
    };
    (p, m)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window is told what the terminal's bar would show: the fraction of
    /// the work, and the counts in words.
    #[test]
    fn a_bar_is_described_as_the_terminal_shows_it() {
        let pb = ProgressBar::with_draw_target(Some(400), ProgressDrawTarget::hidden());
        pb.inc(100);
        assert_eq!(describe(&pb, Stage::Weighing), (0.25, "Measuring decode cost: 100 of 400 files".to_string()));
        assert_eq!(describe(&pb, Stage::Comparing).1, "Comparing: 100 of 400 pairs");
        pb.set_prefix("3/12");
        assert_eq!(describe(&pb, Stage::Fingerprinting).1, "Fingerprinting: 3 of 12 videos");
        pb.set_message("episode 4.mkv");
        assert_eq!(describe(&pb, Stage::Fingerprinting).1, "Fingerprinting: 3 of 12 videos · episode 4.mkv");
        // A bar with nothing to measure is at its start, not dividing by zero.
        let empty = ProgressBar::with_draw_target(Some(0), ProgressDrawTarget::hidden());
        assert_eq!(describe(&empty, Stage::Comparing).0, 0.0);
    }

    /// Off a terminal's worker, no bar is handed out and nothing is said.
    #[test]
    fn nothing_is_spoken_unless_asked() {
        assert!(!speaking());
        assert!(spoken(10, Stage::Comparing).is_none());
    }
}
