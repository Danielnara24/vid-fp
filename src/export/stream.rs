//! Where a report's bytes go: a scratch copy renamed over the old report, a
//! pipe, or a buffer handed over whole. Nothing here knows what a report says;
//! `render` decides that.

use super::{ReportTarget, Sink};
use anyhow::{Context, Result};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// A report being written next to the one it is about to replace.
///
/// Removed on the way out unless `keep` has been called, so a run that fails
/// or unwinds between the create and the rename leaves the old report standing
/// and no litter beside it. A run KILLED outright runs no destructor and does
/// leave the scratch copy behind -- but the report it was replacing is still
/// whole, which is the half that matters, and the name it is under is not one
/// anything reads.
pub(super) struct Scratch {
    pub(super) path: Option<PathBuf>,
}

impl Scratch {
    /// Open a scratch copy beside `dest`, ready to be written and renamed over
    /// it.
    ///
    /// `None` is the cue to write `dest` in place instead, and there are two
    /// ways to get one. A destination that is not a regular file must not be
    /// replaced by a rename at all (see `replaceable_by_rename`), and a folder
    /// that will not take a new file cannot hold a scratch copy -- which is a
    /// case rather than an error, because `ensure_writable` deliberately allows
    /// a read-only FOLDER holding a writable report. Writing in place is no
    /// worse than what was there before either fix.
    pub(super) fn beside(dest: &Path) -> Option<(std::fs::File, Self)> {
        let scratch_path = match scratch_beside(dest) {
            Some(scratch) if replaceable_by_rename(dest) => scratch,
            _ => return None,
        };

        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&scratch_path)
            .ok()?;

        Some((file, Scratch { path: Some(scratch_path) }))
    }

    /// The rename succeeded: there is nothing at this name any more.
    fn keep(&mut self) {
        self.path = None;
    }

    /// Put the finished copy in the report's place.
    ///
    /// The caller flushes and fsyncs before calling this, and it has to be that
    /// way round: a rename that beats its own data to disk is how a crash
    /// leaves an empty file where a whole report used to be.
    pub(super) fn commit(mut self, dest: &Path) -> std::io::Result<()> {
        let Some(scratch_path) = self.path.clone() else {
            return Ok(());
        };

        // A rename carries the scratch file's permissions, so a report the user
        // had narrowed would quietly widen back to the default. Best effort --
        // the report is worth more than its mode.
        if let Ok(meta) = std::fs::metadata(dest) {
            let _ = std::fs::set_permissions(&scratch_path, meta.permissions());
        }

        std::fs::rename(&scratch_path, dest)?;
        self.keep();

        // The rename is atomic to any reader either way; this is only about it
        // surviving a power cut, and there is nothing useful to say if it fails.
        if let Some(parent) = dest.parent() {
            let parent = if parent.as_os_str().is_empty() { Path::new(".") } else { parent };
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }

        Ok(())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Some(path) = &self.path {
            // Best effort, like `Conjured`: a run already on its way out with an
            // error to report has nothing to add about a file it could not tidy.
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Where the scratch copy goes: beside the report, so the rename is within one
/// filesystem, and named so it cannot collide with a report anybody asked for.
///
/// The pid is in it because nothing stops two runs writing the same `--output`
/// (the cache lock is per cache, and `--from-report` takes no cache at all),
/// and two runs sharing one scratch file would hand one of them the other's
/// bytes.
pub(super) fn scratch_beside(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?;
    let mut scratch = std::ffi::OsString::from(".");
    scratch.push(name);
    scratch.push(format!(".vid-fp-{}.tmp", std::process::id()));
    Some(path.with_file_name(scratch))
}

/// Whether this destination can be replaced by a rename at all.
///
/// Only a regular file, or a name with nothing at it yet. Everything else is
/// written through in place, because for those the NAME is not what the user
/// asked to be filled:
///
/// * A FIFO is a reader waiting at the other end -- `mkfifo p; vid-fp lib -o p
///   & cat p` is a documented shape, and `ensure_writable` deliberately accepts
///   it unexamined. A rename would replace the pipe with a regular file and
///   leave that reader waiting for ever.
/// * A symlink is a report kept somewhere else under a local name. `fs::write`
///   follows it; a rename would break it and put the report where the link is
///   instead of where it points.
/// * A device (`-o /dev/stdout` by another spelling) is not this run's to
///   replace either.
///
/// `symlink_metadata`, so the link itself is what is examined rather than
/// whatever it leads to.
pub(super) fn replaceable_by_rename(path: &Path) -> bool {
    match std::fs::symlink_metadata(path) {
        Ok(meta) => meta.file_type().is_file(),
        // Nothing there: the ordinary case, and the one the rename is for.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        // Anything else (a folder in the way, a permission problem on the
        // parent) is the real write's to report, in its own words.
        Err(_) => false,
    }
}

/// Write the report so that the file at `path` is either the old report or the
/// new one, never half of either.
///
/// `fs::write` truncates in place, and the report is the LAST thing
/// `output_results` does -- after the disposal pass. So a run killed mid-write
/// destroyed the previous report and left a truncated one in its place, and
/// that is exactly the file the documented `--from-report` workflow hands back:
/// a plan for a library that has already been acted on, minus however many rows
/// did not make it to disk. It degrades safely rather than dangerously (a
/// half-written row loses `size_bytes` and the replay refuses it), but "the
/// last rows are missing" is a plan that silently keeps files it condemned.
///
/// Two things are given up to close it, and both are given up only where they
/// are actually needed. A rename cannot create a file in a folder that refuses
/// one, and `ensure_writable` deliberately allows a read-only FOLDER holding a
/// writable report -- so a scratch file that cannot be created falls back to
/// the write it replaced, which is no worse than what was there before. And a
/// destination that is not a regular file is written through rather than
/// replaced (see `replaceable_by_rename`).
///
/// A failure AFTER the scratch file exists is returned rather than fallen back
/// on: whatever stopped the copy (a full disk, most likely) would stop the
/// in-place write too, having truncated the old report first.
pub(super) fn write_file_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let Some((mut file, scratch)) = Scratch::beside(path) else {
        // Nothing can be written beside the report. The old report is still the
        // one thing this can be written over.
        return std::fs::write(path, bytes);
    };

    file.write_all(bytes)?;
    // Before the rename, not after. See `Scratch::commit`.
    file.sync_all()?;
    drop(file);

    scratch.commit(path)
}

/// Hand the finished report to its destination.
///
/// The one thing here that is not a plain write is what happens when the reader
/// of a pipe goes away first: `vid-fp DIR -o - | head` closes stdout under us,
/// and that is a normal end to a pipeline rather than a run that did less than
/// it was asked. Every other write failure is still a problem the caller
/// records. (Rust ignores SIGPIPE, so this arrives as an ordinary error rather
/// than killing the process.)
pub(super) fn write_report(target: &ReportTarget, bytes: &[u8]) -> Result<()> {
    match &target.sink {
        Sink::File(path) => write_file_atomically(path, bytes)
            .with_context(|| format!("Failed to write the report to {}", path.display())),
        Sink::Stdout => {
            let mut out = std::io::stdout().lock();
            match out.write_all(bytes).and_then(|()| out.flush()) {
                Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                other => other.context("Failed to write the report to stdout"),
            }
        }
    }
}
/// Where a report is being written as it is produced.
enum ReportOut {
    /// Into a scratch copy beside the report, renamed over it at the end. The
    /// same replacement `write_file_atomically` performs, held open across the
    /// reporting pass rather than handed a finished document.
    Replacing { file: BufWriter<std::fs::File>, scratch: Scratch, dest: PathBuf },
    /// Straight down the pipe. Nothing else in a run writes to stdout, and
    /// `console_for` has already decided what stderr still says beside it.
    Stdout(BufWriter<std::io::Stdout>),
    /// Held whole and handed to `write_report` at the end -- the write this
    /// replaced, kept for the destinations that must not be renamed over (a
    /// FIFO, a symlink, a device) and for a folder that will not take a scratch
    /// file. Those keep the property streaming gives up: the report is one
    /// `fs::write` after the run is over, so a run killed during the reporting
    /// pass leaves the previous report standing.
    Buffered(Vec<u8>),
}

/// The report, written as the reporting pass produces it rather than assembled
/// in memory and handed over in one piece.
///
/// All three formats go through this, and the JSON is the one it was written
/// for. A `serde_json::Value` is an `IndexMap` per file and another per link,
/// and a group of `g` members contributes `g * (g - 1)` links, so the tree ran
/// roughly seven times the document it rendered to and the rendered `String`
/// was then a second whole copy of that document beside it. The other two are
/// each one copy of their own document, which is smaller but is still the
/// entire report held in memory for no reason -- and the text body was two,
/// since the summary was appended by building a second `String` from the first.
///
/// What every format gives up for it is stated on `ReportOut::Buffered`.
pub(super) struct ReportStream {
    out: ReportOut,
    /// The first write failure. Everything after it is skipped rather than
    /// retried, because nothing that stops one group finishes the next: this is
    /// a full disk or a reader that went away. `finish` is where it is
    /// reported, in the same words a whole-document write would have used.
    failed: Option<std::io::Error>,
}

impl ReportStream {
    pub(super) fn open(target: &ReportTarget) -> Self {
        let out = match &target.sink {
            Sink::Stdout => ReportOut::Stdout(BufWriter::new(std::io::stdout())),
            Sink::File(path) => match Scratch::beside(path) {
                Some((file, scratch)) => ReportOut::Replacing {
                    file: BufWriter::new(file),
                    scratch,
                    dest: path.clone(),
                },
                None => ReportOut::Buffered(Vec::new()),
            },
        };

        ReportStream { out, failed: None }
    }

    /// Close the document and put it where it was asked for.
    pub(super) fn finish(mut self, target: &ReportTarget) -> Result<()> {
        // A reader closing early is how a pipeline ends rather than a run that
        // did less than it was asked, exactly as in `write_report`. Anything
        // else is the caller's to record. The sink drops on the way out of
        // here, which is what takes the half-written scratch copy with it and
        // leaves the previous report standing.
        if let Some(e) = self.failed.take() {
            return match &target.sink {
                Sink::Stdout if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                Sink::Stdout => Err(e).context("Failed to write the report to stdout"),
                Sink::File(path) => Err(e)
                    .with_context(|| format!("Failed to write the report to {}", path.display())),
            };
        }

        match self.out {
            ReportOut::Replacing { mut file, scratch, dest } => {
                // The buffer has to reach the file before the file is synced,
                // and the file has to be synced before the rename. `BufWriter`
                // flushes on drop too, but silently -- a report lost to a full
                // disk at the last block would have been renamed into place as
                // a truncated one.
                let flushed = file.flush().and_then(|()| file.get_ref().sync_all());
                drop(file);

                flushed
                    .and_then(|()| scratch.commit(&dest))
                    .with_context(|| format!("Failed to write the report to {}", dest.display()))
            }
            ReportOut::Stdout(mut out) => match out.flush() {
                Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                other => other.context("Failed to write the report to stdout"),
            },
            ReportOut::Buffered(bytes) => write_report(target, &bytes),
        }
    }
}

/// Records the first failure and tells the caller every write succeeded.
///
/// That is deliberate rather than sloppy, and it is what lets the three bodies
/// be written where they are produced. A report that could not be saved is a
/// problem this run REPORTS -- logged, tallied, exit 2 -- and never one it
/// returns: by the time a body is being written the disposal pass has already
/// run, and propagating an error out of `output_results` throws away
/// `deleted_paths` on the way, leaving the cache claiming fingerprints for
/// files this run has just removed. The `?` on a `csv::Writer` record and the
/// `write!` on a text line are both inside that pass, so neither may carry a
/// failure out of it. `finish` is where the first one is answered for.
impl Write for ReportStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.failed.is_none() {
            let wrote = match &mut self.out {
                ReportOut::Replacing { file, .. } => file.write_all(buf),
                ReportOut::Stdout(out) => out.write_all(buf),
                ReportOut::Buffered(bytes) => bytes.write_all(buf),
            };
            if let Err(e) = wrote {
                self.failed = Some(e);
            }
        }
        Ok(buf.len())
    }

    /// Not the end of the report: `finish` is, and it is the only thing that
    /// flushes for real. Whatever asked for this (a `csv::Writer` finishing a
    /// record) is told it happened.
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
