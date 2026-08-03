//! Turning what the user asked to scan into the list of files to fingerprint.
//!
//! Three shapes of request, one answer. A folder is walked. A file named
//! outright is taken as given. A list arriving on stdin or in a file is exactly
//! equivalent to having typed those paths as arguments -- folders in it are
//! walked, files in it are taken as given -- so `fd -e mkv | vid-fp -` needs no
//! rules of its own.
//!
//! The distinction worth stating out loud is what `--extensions` is for. Walking
//! a folder means GUESSING which of its files are videos, and the extension list
//! is that guess. A path the user named is not a guess, so the filter does not
//! apply to it: `vid-fp holiday.m4v` fingerprints the file rather than silently
//! finding nothing, and a pipe that already selected its files is not
//! second-guessed. The cost is that piping in a subtitle file produces a
//! fingerprint failure instead of a silent skip, which is the correct trade for
//! a tool whose worst failure mode is quietly doing less than it was asked.
//!
//! `--exclude` is the opposite: it applies to everything, including a path named
//! explicitly. It is the one flag whose entire purpose is "do not touch this",
//! and `find ... | vid-fp - -e ~/keep --delete` has to mean what it obviously
//! means.

use anyhow::{Context, Result};
use log::info;
use std::collections::HashSet;
use std::io::{IsTerminal, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::stats::RunStats;
use crate::utils::shutdown_requested;

/// Everything about the request that decides which files get scanned.
///
/// Deliberately not `&Args`: this module has no business seeing the deletion
/// flags, and spelling out what it does need keeps it that way.
pub struct Sources<'a> {
    pub include: &'a [String],
    pub exclude: &'a [String],
    pub from_file: Option<&'a str>,
    pub null_separated: bool,
    pub extensions: &'a [String],
    pub recursive: bool,
    pub follow_symlinks: bool,
}

pub enum Scan {
    Complete(Vec<String>),
    /// Ctrl-C arrived mid-walk. The partial list is discarded rather than
    /// returned: a truncated library would silently change what counts as a
    /// duplicate, which is the last thing an interrupted run should do.
    Interrupted,
}

/// Resolve the request into absolute paths of files to fingerprint.
pub fn collect(sources: &Sources, stats: &RunStats) -> Result<Scan> {
    // Before anything that can block on a pipe: a run that cannot possibly
    // match a file should not first wait for a list of them.
    let extensions = normalize_extensions(sources.extensions)?;

    let named: Vec<&String> = sources.include.iter().filter(|p| *p != "-").collect();
    if !named.is_empty() {
        info!("Scanning: {:?}", named);
    }
    if !sources.exclude.is_empty() {
        info!("Excluding folders: {:?}", sources.exclude);
    }

    // HashSet iteration order is unspecified; sort for a stable, readable log.
    let mut ext_display: Vec<&str> = extensions.iter().map(|s| s.as_str()).collect();
    ext_display.sort_unstable();
    info!("Searching extensions: {:?}", ext_display);

    let excludes = resolve_excludes(sources.exclude, stats);
    let requested = requested_paths(sources, stats)?;

    // Identity-based deduplication. A symlink, a hard link, a second scan root
    // that overlaps the first, the same path piped in twice, and a bind-mount
    // alias all resolve to the same (device, inode) pair. Keying on that
    // identity means each set of bytes is fingerprinted exactly once, so the
    // report never lists a file as a duplicate of itself and the "space freed"
    // figure never counts bytes that deleting a link would not return.
    let mut seen_inodes: HashSet<(u64, u64)> = HashSet::new();
    let mut found: Vec<String> = Vec::new();

    for path in &requested {
        if shutdown_requested() {
            return Ok(Scan::Interrupted);
        }

        // Canonicalizing is what makes a relative path from `fd` and an absolute
        // one from the command line the same cache key. It also proves the path
        // exists, so a typo is loud, counted, and reflected in the exit code
        // rather than quietly scanning less than was asked for.
        let resolved = match std::fs::canonicalize(path) {
            Ok(p) => p,
            Err(e) => {
                log::error!("Could not resolve '{}': {}", path, e);
                stats.unresolved_includes.record(format!("{}: {}", path, e));
                continue;
            }
        };

        if is_excluded(&resolved, &excludes) {
            stats.skipped_excluded.bump();
            log::debug!("Skipping {}: under an excluded folder", resolved.display());
            continue;
        }

        let meta = match std::fs::metadata(&resolved) {
            Ok(m) => m,
            Err(e) => {
                log::error!("Cannot stat {}: {}", resolved.display(), e);
                stats
                    .unreadable
                    .record(format!("{}: {}", resolved.display(), e));
                continue;
            }
        };

        if meta.is_dir() {
            if !walk_folder(
                &resolved,
                sources,
                &extensions,
                &excludes,
                &mut seen_inodes,
                &mut found,
                stats,
            ) {
                return Ok(Scan::Interrupted);
            }
        } else if meta.is_file() {
            // No extension check: see the module docs. The user named this file.
            if seen_inodes.insert((meta.dev(), meta.ino())) {
                found.push(resolved.to_string_lossy().to_string());
            } else {
                stats.skipped_alias.bump();
                log::debug!(
                    "Skipping {}: same inode as a path already queued",
                    resolved.display()
                );
            }
        } else {
            // A socket, fifo, or device node. Naming one is a mistake worth
            // hearing about rather than a file worth trying to decode.
            log::error!("Not a file or folder: {}", resolved.display());
            stats
                .unresolved_includes
                .record(format!("{}: not a file or folder", resolved.display()));
        }
    }

    Ok(Scan::Complete(found))
}

/// Walk one folder, appending every video in it. Returns false if interrupted.
fn walk_folder(
    base: &Path,
    sources: &Sources,
    extensions: &HashSet<String>,
    excludes: &[PathBuf],
    seen_inodes: &mut HashSet<(u64, u64)>,
    found: &mut Vec<String>,
    stats: &RunStats,
) -> bool {
    let mut walker = WalkDir::new(base).follow_links(sources.follow_symlinks);

    if !sources.recursive {
        // Non-recursive by default: only the folder itself (depth 0) and its
        // immediate files (depth 1).
        walker = walker.max_depth(1);
    }

    let it = walker
        .into_iter()
        .filter_entry(|e| !is_excluded(e.path(), excludes));

    for entry in it {
        if shutdown_requested() {
            return false;
        }

        // WalkDir reports per-entry failures: an unreadable subfolder, a
        // dangling link under --follow-symlinks, a symlink loop. Dropping these
        // with `.ok()` made a permission-denied folder indistinguishable from an
        // empty one, which is the worst possible way to silently miss half a
        // library.
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                let at = e
                    .path()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| base.display().to_string());
                log::error!("Cannot scan {}: {}", at, e);
                stats.unwalkable.record(format!("{}: {}", at, e));
                continue;
            }
        };

        let path = entry.path();

        // Extension first: it is free, and it keeps us from stat()ing every
        // non-video file in the tree.
        if !has_wanted_extension(path, extensions) {
            continue;
        }

        // One stat does three jobs: it follows symlinks (so a link to a video is
        // treated as that video), confirms this is a regular file, and yields
        // the identity used for deduplication.
        let meta = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(e) => {
                log::error!("Cannot stat {}: {}", path.display(), e);
                stats
                    .unreadable
                    .record(format!("{}: {}", path.display(), e));
                continue;
            }
        };

        if !meta.is_file() {
            continue;
        }

        if !seen_inodes.insert((meta.dev(), meta.ino())) {
            stats.skipped_alias.bump();
            log::debug!(
                "Skipping {}: same inode as a path already queued",
                path.display()
            );
            continue;
        }

        found.push(path.to_string_lossy().to_string());
    }

    true
}

/// Every path the user asked for, with `-` and `--from-file` expanded.
fn requested_paths(sources: &Sources, stats: &RunStats) -> Result<Vec<String>> {
    let mut requested: Vec<String> = Vec::new();
    let mut stdin_taken = false;

    for path in sources.include {
        if path == "-" {
            // `vid-fp - -` is a typo, not a request to read the pipe twice.
            if stdin_taken {
                continue;
            }
            requested.extend(read_stdin(sources.null_separated, stats)?);
            stdin_taken = true;
        } else {
            requested.push(path.clone());
        }
    }

    if let Some(list) = sources.from_file {
        if list == "-" {
            if !stdin_taken {
                requested.extend(read_stdin(sources.null_separated, stats)?);
            }
        } else {
            let file = std::fs::File::open(list)
                .with_context(|| format!("Failed to open the path list {}", list))?;
            let paths = read_path_list(file, sources.null_separated, stats)?;
            info!("Read {} path(s) from {}.", paths.len(), list);
            requested.extend(paths);
        }
    }

    Ok(requested)
}

fn read_stdin(null_separated: bool, stats: &RunStats) -> Result<Vec<String>> {
    // Without this, `vid-fp -` at a prompt looks exactly like a hang.
    if std::io::stdin().is_terminal() {
        anyhow::bail!(
            "Asked to read paths from stdin, but stdin is a terminal. \
             Pipe a list in (e.g. `fd -e mkv | vid-fp -`), or name folders as arguments."
        );
    }

    let paths = read_path_list(std::io::stdin().lock(), null_separated, stats)?;
    info!("Read {} path(s) from stdin.", paths.len());
    Ok(paths)
}

/// Read a whole path list, reporting the entries that are not valid UTF-8.
///
/// Read as bytes rather than as a string so ONE undecodable filename costs that
/// filename rather than the entire run -- the same reason every other failure in
/// this module is counted and stepped over instead of returned.
fn read_path_list<R: Read>(
    mut reader: R,
    null_separated: bool,
    stats: &RunStats,
) -> Result<Vec<String>> {
    let mut raw: Vec<u8> = Vec::new();
    reader
        .read_to_end(&mut raw)
        .context("Failed to read the path list")?;

    let mut paths = Vec::new();
    for entry in split_path_list(&raw, null_separated) {
        match std::str::from_utf8(entry) {
            Ok(path) => paths.push(path.to_string()),
            Err(e) => {
                let shown = String::from_utf8_lossy(entry).into_owned();
                log::error!("Path is not valid UTF-8 and was skipped: {}", shown);
                stats.unreadable.record(format!("{}: {}", shown, e));
            }
        }
    }

    Ok(paths)
}

/// Split a list on newlines, or on NUL bytes when asked.
///
/// A trailing carriage return is trimmed in newline mode. A list authored on
/// Windows would otherwise fail every single path with "No such file", and the
/// byte responsible is invisible in the error -- the worst kind of failure to
/// debug. `--null` exists for anyone who needs the bytes untouched.
fn split_path_list(raw: &[u8], null_separated: bool) -> Vec<&[u8]> {
    let separator = if null_separated { b'\0' } else { b'\n' };

    raw.split(|&b| b == separator)
        .map(|entry| {
            if null_separated {
                entry
            } else {
                entry.strip_suffix(b"\r").unwrap_or(entry)
            }
        })
        .filter(|entry| !entry.is_empty())
        .collect()
}

fn normalize_extensions(requested: &[String]) -> Result<HashSet<String>> {
    // Strip an optional leading dot and lowercase, so `-x .MP4`, `-x MP4`, and
    // `-x mp4` all behave identically. A HashSet gives O(1) lookups during the
    // walk and dedups automatically.
    let extensions: HashSet<String> = requested
        .iter()
        .map(|e| e.trim().trim_start_matches('.').to_lowercase())
        .filter(|e| !e.is_empty())
        .collect();

    if extensions.is_empty() {
        anyhow::bail!("No valid video extensions to search for (--extensions was empty).");
    }

    Ok(extensions)
}

fn has_wanted_extension(path: &Path, extensions: &HashSet<String>) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| extensions.contains(ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Canonicalize the exclude list so prefix matching is safe and reliable.
///
/// A path that will not resolve excludes NOTHING. That used to be swallowed by
/// `filter_map(.ok())`, so a typo in `-e` quietly scanned the folder you were
/// protecting -- and with `--delete` armed, that is how files die.
fn resolve_excludes(requested: &[String], stats: &RunStats) -> Vec<PathBuf> {
    let mut excludes: Vec<PathBuf> = Vec::with_capacity(requested.len());

    for p in requested {
        match std::fs::canonicalize(p) {
            Ok(resolved) => excludes.push(resolved),
            Err(e) => {
                log::error!(
                    "Could not resolve exclude path '{}': {} -- nothing was excluded for it",
                    p,
                    e
                );
                stats.unresolved_excludes.record(format!("{}: {}", p, e));
            }
        }
    }

    excludes
}

fn is_excluded(path: &Path, excludes: &[PathBuf]) -> bool {
    excludes.iter().any(|ex| path.starts_with(ex))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn extensions_of(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// A real file, returned by the canonical path collect() will produce.
    fn touch(dir: &Path, name: &str) -> String {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, b"video").unwrap();
        fs::canonicalize(&path).unwrap().to_string_lossy().to_string()
    }

    fn sources<'a>(
        include: &'a [String],
        exclude: &'a [String],
        extensions: &'a [String],
    ) -> Sources<'a> {
        Sources {
            include,
            exclude,
            from_file: None,
            null_separated: false,
            extensions,
            recursive: false,
            follow_symlinks: false,
        }
    }

    fn collected(sources: &Sources, stats: &RunStats) -> Vec<String> {
        match collect(sources, stats).unwrap() {
            Scan::Complete(mut files) => {
                files.sort();
                files
            }
            Scan::Interrupted => panic!("nothing interrupted this scan"),
        }
    }

    #[test]
    fn test_a_list_is_split_on_newlines_and_blanks_are_ignored() {
        let raw = b"/videos/a.mkv\n\n/videos/b.mp4\n";
        assert_eq!(
            split_path_list(raw, false),
            vec![&b"/videos/a.mkv"[..], &b"/videos/b.mp4"[..]]
        );
    }

    #[test]
    fn test_a_carriage_return_is_trimmed_rather_than_kept_in_the_path() {
        // A list authored on Windows. Keeping the \r fails every path with "No
        // such file", and the byte responsible does not show up in the message.
        assert_eq!(
            split_path_list(b"/videos/a.mkv\r\n/videos/b.mp4\r\n", false),
            vec![&b"/videos/a.mkv"[..], &b"/videos/b.mp4"[..]]
        );
    }

    #[test]
    fn test_a_null_separated_list_keeps_every_byte_of_the_filename() {
        // The reason -0 exists: both of these are legal Linux filenames.
        let raw = b"/videos/two\nlines.mkv\0/videos/trailing\r.mkv\0";
        assert_eq!(
            split_path_list(raw, true),
            vec![&b"/videos/two\nlines.mkv"[..], &b"/videos/trailing\r.mkv"[..]]
        );
    }

    #[test]
    fn test_one_undecodable_path_costs_only_that_path() {
        let stats = RunStats::default();
        let raw: Vec<u8> = b"/videos/good.mkv\n/videos/\xFF\xFEbad.mkv\n/videos/also_good.mkv\n"
            .to_vec();

        let paths = read_path_list(raw.as_slice(), false, &stats).unwrap();

        assert_eq!(paths, vec!["/videos/good.mkv", "/videos/also_good.mkv"]);
        assert_eq!(stats.unreadable.count(), 1, "and the bad one is counted, not ignored");
    }

    #[test]
    fn test_a_named_file_is_scanned_whatever_its_extension() {
        // --extensions is how a FOLDER is guessed at. This path was typed.
        let dir = tempfile::tempdir().unwrap();
        let odd = touch(dir.path(), "holiday.m4v");

        let include = vec![odd.clone()];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &[], &extensions_of(&["mkv"])), &stats),
            vec![odd]
        );
        assert!(!stats.had_problems());
    }

    #[test]
    fn test_a_walked_file_still_has_to_look_like_a_video() {
        // The other half of the rule: inside a folder the extension list is the
        // only thing standing between us and every .srt in the library.
        let dir = tempfile::tempdir().unwrap();
        let video = touch(dir.path(), "episode.mkv");
        touch(dir.path(), "episode.srt");
        touch(dir.path(), "poster.jpg");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &[], &extensions_of(&["mkv"])), &stats),
            vec![video]
        );
    }

    #[test]
    fn test_an_exclude_outranks_a_path_named_explicitly() {
        // `find ... | vid-fp - -e ~/keep --delete` has to mean what it says.
        let dir = tempfile::tempdir().unwrap();
        let keep_dir = dir.path().join("keep");
        let protected = touch(&keep_dir, "original.mkv");
        let ordinary = touch(dir.path(), "copy.mkv");

        let include = vec![protected, ordinary.clone()];
        let exclude = vec![keep_dir.to_string_lossy().to_string()];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &exclude, &extensions_of(&["mkv"])), &stats),
            vec![ordinary]
        );
        assert_eq!(stats.skipped_excluded.count(), 1);
        assert!(!stats.had_problems(), "an exclusion is a skip, not a failure");
    }

    #[test]
    fn test_the_same_bytes_named_twice_are_queued_once() {
        let dir = tempfile::tempdir().unwrap();
        let original = touch(dir.path(), "episode.mkv");
        let link = dir.path().join("hardlink.mkv");
        fs::hard_link(&original, &link).unwrap();

        let include = vec![
            original.clone(),
            original.clone(),
            link.to_string_lossy().to_string(),
        ];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &[], &extensions_of(&["mkv"])), &stats),
            vec![original]
        );
        assert_eq!(stats.skipped_alias.count(), 2, "a repeat and a hard link");
    }

    #[test]
    fn test_a_path_that_does_not_exist_is_counted_rather_than_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let real = touch(dir.path(), "episode.mkv");

        let include = vec![
            "/nonexistent/vid-fp/definitely-not-here".to_string(),
            real.clone(),
        ];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &[], &extensions_of(&["mkv"])), &stats),
            vec![real],
            "the rest of the request still runs"
        );
        assert_eq!(stats.unresolved_includes.count(), 1);
        assert!(stats.had_problems(), "and the run must not exit clean");
    }

    #[test]
    fn test_subfolders_are_reached_only_when_asked_for() {
        let dir = tempfile::tempdir().unwrap();
        let top = touch(dir.path(), "top.mkv");
        let nested = touch(&dir.path().join("season_1"), "nested.mkv");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let stats = RunStats::default();

        assert_eq!(collected(&sources(&include, &[], &extensions), &stats), vec![top.clone()]);

        let mut recursive = sources(&include, &[], &extensions);
        recursive.recursive = true;

        let mut expected = vec![top, nested];
        expected.sort();
        assert_eq!(collected(&recursive, &stats), expected);
    }

    #[test]
    fn test_an_empty_extension_list_fails_before_anything_blocks_on_a_pipe() {
        let stats = RunStats::default();
        let include = vec!["-".to_string()];
        let empty: Vec<String> = vec![];

        assert!(collect(&sources(&include, &[], &empty), &stats).is_err());
    }
}