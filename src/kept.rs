//! The last scan, kept for `--last`.
//!
//! What is kept is not a report but what a report is written FROM: the files
//! the groups hold, the measured links between them, which of them are the
//! same bytes and, after `--against`, which are the library's. `--last` hands
//! those straight back to `export::output_results`, the function the scan
//! itself wrote its report with, so a report written again is the report the
//! scan would have written in that format or under that ranking, figure for
//! figure. Reconstructing rows from a report would round a duration on the way,
//! and the ranking has tolerance bands for a rounding to land on the wrong
//! side of -- the reason `found` exists for the window, too.
//!
//! Only the files a group holds are kept, and of each fingerprint only what a
//! report reads: the hashes it was compared on are dropped except the main
//! ones, whose count is the report's `samples`. The groups are kept as the
//! scan cut them rather than cut again, so a ranking changed with `-k` moves
//! only the labels, exactly as it does in the window. An `--against` scan is
//! answered again under the new ranking, since its groups are its answers.
//!
//! The file sits beside the cache it was scanned with (`path_for`), so a
//! `--cache` of its own keeps a last scan of its own. A run under
//! `--no-cache`, under the window, or that was interrupted keeps nothing.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::compare::Match;
use crate::fingerprint::VideoFingerprint;

/// Bumped whenever a field changes meaning; a kept scan of another version is
/// not read.
const VERSION: u32 = 1;

/// One finished scan, by its own file numbers.
#[derive(Serialize, Deserialize)]
pub struct Kept {
    version: u32,
    /// The scan's own elapsed time, which its summary states.
    pub elapsed_secs: u64,
    /// Files the scan found and compared, for the summary of an empty scan.
    pub found: usize,
    pub files: Vec<VideoFingerprint>,
    /// What the scan measured of each file beside its fingerprint and that a
    /// ranking reads: the bytes of its video stream (`Measured::video_bytes`).
    video_bytes: Vec<u64>,
    /// How many main hashes each fingerprint held (the report's `samples`),
    /// kept as a count since no hash is.
    samples: Vec<usize>,
    /// The groups, by number in `files`. Empty after `--against`.
    pub groups: Vec<Vec<usize>>,
    pub matches: Vec<Match>,
    pub identical: Vec<Vec<usize>>,
    /// `--against`: which files are the library's. Empty after a plain scan.
    pub library: Vec<bool>,
    pub answering: bool,
    /// When it was kept, in seconds since 1970.
    pub finished: u64,
}

impl Kept {
    /// Keep the files the groups hold, renumbered in the order the scan had
    /// them, so every tie the report breaks by position breaks the same way.
    #[allow(clippy::too_many_arguments)]
    pub fn of(
        groups: &[Vec<usize>],
        fingerprints: &[VideoFingerprint],
        matches: &[Match],
        identical: &[Vec<usize>],
        library: &[bool],
        answering: bool,
        found: usize,
        elapsed_secs: u64,
    ) -> Kept {
        let mut held: Vec<usize> = groups.iter().flatten().copied().collect();
        held.sort_unstable();
        held.dedup();
        let number: HashMap<usize, usize> = held.iter().enumerate().map(|(new, &old)| (old, new)).collect();

        let files: Vec<VideoFingerprint> = held.iter().map(|&i| for_report(&fingerprints[i])).collect();
        let video_bytes =
            held.iter().map(|&i| fingerprints[i].measured.as_ref().map_or(0, |m| m.video_bytes)).collect();
        let samples = held.iter().map(|&i| fingerprints[i].valid_hashes.len()).collect();
        let renumber = |set: &[usize]| set.iter().filter_map(|i| number.get(i).copied()).collect::<Vec<_>>();

        Kept {
            version: VERSION,
            elapsed_secs,
            found,
            files,
            video_bytes,
            samples,
            groups: if answering { Vec::new() } else { groups.iter().map(|g| renumber(g)).collect() },
            matches: matches
                .iter()
                .filter_map(|m| Some(Match { a: *number.get(&m.a)?, b: *number.get(&m.b)?, ..*m }))
                .collect(),
            identical: identical.iter().map(|s| renumber(s)).filter(|s| s.len() > 1).collect(),
            library: if answering { held.iter().map(|&i| library[i]).collect() } else { Vec::new() },
            answering,
            finished: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        }
    }

    /// The kept fingerprints, with what the scan measured beside each put back.
    pub fn fingerprints(&mut self) -> Vec<VideoFingerprint> {
        let mut files = std::mem::take(&mut self.files);
        for ((fp, &bytes), &samples) in files.iter_mut().zip(&self.video_bytes).zip(&self.samples) {
            fp.measured_mut().video_bytes = bytes;
            fp.valid_hashes = vec![0; samples];
        }
        files
    }

    /// Save it in place of the scan kept before. Written beside the old one
    /// and renamed over it, so a run killed half way leaves one scan or the
    /// other, never half of each.
    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(self).context("Failed to encode the scan for --last")?;
        let scratch = path.with_extension(format!("{}.new", std::process::id()));
        std::fs::write(&scratch, &bytes)
            .and_then(|()| std::fs::rename(&scratch, path))
            .inspect_err(|_| {
                let _ = std::fs::remove_file(&scratch);
            })
            .with_context(|| format!("Failed to keep the scan at {} for --last", path.display()))
    }

    /// The scan kept at `path`.
    pub fn load(path: &Path) -> Result<Kept> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => anyhow::bail!(
                "There is no scan to write again: no scan has finished with this cache yet ({}). \
                 Run one first.",
                path.display()
            ),
            Err(e) => return Err(e).with_context(|| format!("Failed to read the last scan at {}", path.display())),
        };
        let kept: Kept = match bincode::deserialize(&bytes) {
            Ok(kept) => kept,
            Err(_) => anyhow::bail!(
                "The last scan at {} was kept by another version of vid-fp; run a scan again.",
                path.display()
            ),
        };
        if kept.version != VERSION {
            anyhow::bail!(
                "The last scan at {} was kept by another version of vid-fp; run a scan again.",
                path.display()
            );
        }
        Ok(kept)
    }
}

/// Where the last scan made with the cache at `cache` is kept: beside it.
pub fn path_for(cache: &Path) -> PathBuf {
    let mut name = cache.file_name().unwrap_or_default().to_os_string();
    name.push(".last-scan");
    cache.with_file_name(name)
}

/// A fingerprint cut down to what a report reads of it. Nothing that writes a
/// report compares hashes, so none is kept (their count is, in `samples`);
/// they are most of a fingerprint's size.
fn for_report(fp: &VideoFingerprint) -> VideoFingerprint {
    VideoFingerprint {
        valid_hashes: Vec::new(),
        valid_t_start: Vec::new(),
        valid_t_end: Vec::new(),
        mirrored_hashes: Vec::new(),
        detail_hashes: Vec::new(),
        mirrored_detail_hashes: Vec::new(),
        reframed: Vec::new(),
        measured: None,
        ..fp.clone()
    }
}
