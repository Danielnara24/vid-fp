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
//! `-x '*'` is the escape hatch for when the guess cannot be made at all: a
//! folder of extensionless files (a `Camera Uploads` dump, a DVD rip, anything
//! named by a hash) is unreachable from a folder walk otherwise, because
//! `Path::extension` has nothing to offer and no list of suffixes can name the
//! absence of one. It means "hand every regular file to the decoder and let the
//! decoder say", which is the same contract a named path already gets.
//!
//! `--exclude` is the opposite: it applies to everything, including a path named
//! explicitly. It is the one flag whose entire purpose is "do not touch this",
//! and `find ... | vid-fp - -e ~/keep --delete` has to mean what it obviously
//! means. It takes a folder or a single file, because `is_excluded` compares
//! whole path components and neither `starts_with` nor `canonicalize` cares
//! which it was given -- sparing one known original out of a folder being
//! scanned needs no more than naming it. Component-wise is also what keeps
//! `-e ~/clips/take` off `~/clips/take.mkv`.
//!
//! What comes back is not a list of paths but a list of files. Every entry has
//! already been stat'ed here -- the walk needs (device, inode) to deduplicate
//! aliases, and needs to know a directory entry is a regular file at all -- so
//! the size and mtime that same stat returned travel out with it. Re-deriving
//! them downstream cost two further stat() calls per file, which on a network
//! mount or a spinning disk was the dominant cost of a run where every
//! fingerprint was already cached and no video needed decoding at all.

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

/// One file to fingerprint, plus everything the rest of the run needs to know
/// about it without going back to the disk.
///
/// The modification time is kept as the two halves stat actually reports rather
/// than as one number: seconds since the epoch, and nanoseconds within that
/// second. Whole seconds alone cannot see an edit made within the same second
/// as the last one, which is exactly what a script or a fast re-encode does,
/// and pairing it with `size` only catches such an edit when the length changed
/// too. The nanoseconds close that gap wherever the filesystem keeps them; on
/// one that does not (or that reports them as zero) this is no worse than
/// before.
///
/// The (device, inode) pair the walk deduplicated on is deliberately NOT here.
/// It has done its job by the time this struct exists, and nothing downstream
/// has any business re-deciding which files are aliases of each other.
pub struct ScannedFile {
    pub path: String,
    pub size: u64,
    pub mtime: i64,
    pub mtime_nsec: i64,
}

impl ScannedFile {
    fn of(path: &Path, meta: &std::fs::Metadata) -> Self {
        ScannedFile {
            path: path.to_string_lossy().to_string(),
            size: meta.len(),
            mtime: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
        }
    }
}

/// What the scan found, and enough of how it was asked for to say what a
/// FUTURE scan of the same request would find.
///
/// The second half exists for one caller: `--move-to` has to refuse a
/// destination the next run would pick the moved files back out of, and that is
/// a question about the request rather than about its answer. The files alone
/// cannot settle it -- a folder that yielded nothing today (because the videos
/// are all one level down, because everything in it was excluded, because it
/// was empty) is still a folder a moved file would be found in tomorrow.
pub struct Library {
    pub files: Vec<ScannedFile>,
    /// The folders that were walked, at the absolute paths the walk used. A
    /// path the user named that turned out to be a file is not one of these:
    /// naming a file scans that file, and nothing can ever be moved *into* it.
    walked: Vec<PathBuf>,
    /// The `--exclude` paths that resolved, which is what makes the advice in
    /// the `--move-to` refusal true rather than merely plausible.
    excluded: Vec<PathBuf>,
}

impl Library {
    /// The scanned folder a file moved to `dest` would be found in again, if
    /// any.
    ///
    /// The question is deliberately this way round. A landing path is `dest`
    /// plus the source's own absolute path, so `dest` being inside a scanned
    /// folder makes EVERY landing path inside it too -- that is the loop worth
    /// refusing. The reverse arrangement is not: `--move-to ~/Documents` while
    /// scanning `~/Documents/AN` sends `~/Documents/AN/ep.mkv` to
    /// `~/Documents/home/you/Documents/AN/ep.mkv`, which no scan of
    /// `~/Documents/AN` will ever reach. Testing containment the other way
    /// round -- "is a scanned file under dest" -- is what made every parent
    /// folder look like a loop.
    ///
    /// It is asked of the folders the run was POINTED AT, not of the parents of
    /// the files that came out of them. Those two coincide only in a flat
    /// library: `vid-fp lib -r --move-to lib/dupes` over a tree whose videos
    /// all live in `lib/sub` has no found file whose parent encloses `lib/dupes`
    /// at all, so the parent-based version sailed through and the run after it
    /// re-ingested everything it had moved -- keeping the moved copy and moving
    /// the original, one directory deeper each time.
    ///
    /// An `--exclude` covering `dest` is the one thing that makes it safe
    /// again, and honouring it here is what makes the refusal's own advice
    /// work: the excluded subtree is not walked, so the moved files are not
    /// found. Nothing else in the request earns an exemption. A non-recursive
    /// walk would not reach the landing paths today (they sit at least two
    /// levels under `dest`), but the destination is still inside the library
    /// this run was aimed at, and it becomes a loop the first time someone adds
    /// `-r`.
    pub fn walk_reaches(&self, dest: &Path) -> Option<&Path> {
        if is_excluded(dest, &self.excluded) {
            return None;
        }
        self.walked
            .iter()
            .find(|root| dest.starts_with(root))
            .map(|root| root.as_path())
    }
}

pub enum Scan {
    Complete(Library),
    /// Ctrl-C arrived mid-walk. The partial list is discarded rather than
    /// returned: a truncated library would silently change what counts as a
    /// duplicate, which is the last thing an interrupted run should do.
    Interrupted,
}

/// Resolve the request into the files to fingerprint, at absolute paths.
pub fn collect(sources: &Sources, stats: &RunStats) -> Result<Scan> {
    // Before anything that can block on a pipe: a run that cannot possibly
    // match a file should not first wait for a list of them.
    let extensions = normalize_extensions(sources.extensions)?;

    let named: Vec<&String> = sources.include.iter().filter(|p| *p != "-").collect();
    if !named.is_empty() {
        info!("Scanning: {:?}", named);
    }
    if !sources.exclude.is_empty() {
        info!("Excluding: {:?}", sources.exclude);
    }

    match &extensions {
        // Worth saying out loud: it is the one setting under which a folder of
        // text files becomes a folder of failed fingerprints.
        Wanted::Anything => info!("Searching every file, whatever its extension (-x '*')."),
        Wanted::OneOf(set) => {
            // HashSet iteration order is unspecified; sort for a stable log.
            let mut ext_display: Vec<&str> = set.iter().map(|s| s.as_str()).collect();
            ext_display.sort_unstable();
            info!("Searching extensions: {:?}", ext_display);
        }
    }

    let excludes = resolve_excludes(sources.exclude, stats);
    let requested = requested_paths(sources, stats)?;

    // Identity-based deduplication. A symlink, a hard link, a second scan root
    // that overlaps the first, the same path piped in twice, and a bind-mount
    // alias all resolve to the same (device, inode) pair. Keying on that
    // identity means each set of bytes is fingerprinted exactly once, so the
    // report never lists a file as a duplicate of itself and the "space freed"
    // figure never counts bytes that deleting a link would not return.
    let mut seen_inodes: HashSet<(u64, u64)> = HashSet::new();
    let mut found: Vec<ScannedFile> = Vec::new();
    let mut walked: Vec<PathBuf> = Vec::new();

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
            log::debug!("Skipping {}: named or under an --exclude path", resolved.display());
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
            // Recorded before the walk rather than after it: what makes this a
            // scan root is having been pointed at, not having yielded
            // anything. See `Library::walk_reaches`.
            walked.push(resolved.clone());

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
                found.push(ScannedFile::of(&resolved, &meta));
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

    Ok(Scan::Complete(Library {
        files: found,
        walked,
        excluded: excludes,
    }))
}

/// Walk one folder, appending every video in it. Returns false if interrupted.
fn walk_folder(
    base: &Path,
    sources: &Sources,
    extensions: &Wanted,
    excludes: &[PathBuf],
    seen_inodes: &mut HashSet<(u64, u64)>,
    found: &mut Vec<ScannedFile>,
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

        // A directory is never a file to fingerprint, and WalkDir already knows
        // which entries are directories -- so asking it here costs nothing and
        // saves a stat() per folder under `-x '*'`, where the extension filter
        // no longer turns them away.
        if entry.file_type().is_dir() {
            continue;
        }

        let path = entry.path();

        // Extension next: it is free, and it keeps us from stat()ing every
        // non-video file in the tree.
        if !extensions.accepts(path) {
            continue;
        }

        // One stat does four jobs: it follows symlinks (so a link to a video is
        // treated as that video), confirms this is a regular file, yields the
        // identity used for deduplication, and supplies the size and mtime that
        // travel out with the file so nothing downstream has to ask again.
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

        found.push(ScannedFile::of(path, &meta));
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

/// The guess a folder walk makes about which of its files are videos.
///
/// Two shapes rather than one set, because "every file" is not expressible as a
/// list of suffixes: a file with no extension at all has nothing for
/// `Path::extension` to return, so no entry could ever name it. That is not a
/// corner case -- it is a whole camera dump, a DVD rip, or anything named by a
/// content hash -- and before `-x '*'` those folders were unreachable from a
/// walk, with the only workaround being to pipe the paths in from another tool.
enum Wanted {
    /// `-x '*'`. Every regular file the walk finds, extension or not.
    Anything,
    /// Files whose extension is in this set, lowercased and dot-free.
    OneOf(HashSet<String>),
}

impl Wanted {
    fn accepts(&self, path: &Path) -> bool {
        match self {
            Wanted::Anything => true,
            Wanted::OneOf(extensions) => path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| extensions.contains(ext.to_lowercase().as_str())),
        }
    }
}

/// The wildcard, spelled the way a shell user expects. Quoting is on them --
/// unquoted it is a glob, and one that expands to the directory's contents.
const WILDCARD: &str = "*";

fn normalize_extensions(requested: &[String]) -> Result<Wanted> {
    // Strip an optional leading dot and lowercase, so `-x .MP4`, `-x MP4`, and
    // `-x mp4` all behave identically. A HashSet gives O(1) lookups during the
    // walk and dedups automatically.
    // `-x '*.mkv'` is how a shell user spells the same thing, and the entry it
    // produces would otherwise be a suffix no file on earth has -- matching
    // nothing, silently, which is the failure this flag is being widened to fix.
    let extensions: HashSet<String> = requested
        .iter()
        .map(|e| {
            let e = e.trim();
            let e = e.strip_prefix("*.").unwrap_or(e);
            e.trim_start_matches('.').to_lowercase()
        })
        .filter(|e| !e.is_empty())
        .collect();

    // The wildcard wins over anything beside it. `-x '*',mkv` is not a
    // contradiction to refuse -- it is a wider request with a narrower one
    // still written down, and the wider one is what was asked for.
    if extensions.contains(WILDCARD) {
        return Ok(Wanted::Anything);
    }

    if extensions.is_empty() {
        anyhow::bail!("No valid video extensions to search for (--extensions was empty).");
    }

    Ok(Wanted::OneOf(extensions))
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

    fn library(sources: &Sources, stats: &RunStats) -> Library {
        match collect(sources, stats).unwrap() {
            Scan::Complete(library) => library,
            Scan::Interrupted => panic!("nothing interrupted this scan"),
        }
    }

    fn scanned(sources: &Sources, stats: &RunStats) -> Vec<ScannedFile> {
        library(sources, stats).files
    }

    /// Most tests here are about WHICH files were found, not what was learned
    /// about them on the way.
    fn collected(sources: &Sources, stats: &RunStats) -> Vec<String> {
        let mut paths: Vec<String> = scanned(sources, stats).into_iter().map(|f| f.path).collect();
        paths.sort();
        paths
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
    fn test_the_wildcard_reaches_a_file_with_no_extension_at_all() {
        // The case no extension list can express, and the reason `*` exists: a
        // camera dump or a DVD rip whose files are named by a hash was
        // unreachable from a folder walk entirely.
        let dir = tempfile::tempdir().unwrap();
        let bare = touch(dir.path(), "VTS_01_1");
        let video = touch(dir.path(), "episode.mkv");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        let mut expected = vec![bare, video];
        expected.sort();
        assert_eq!(
            collected(&sources(&include, &[], &extensions_of(&["*"])), &stats),
            expected
        );
    }

    #[test]
    fn test_the_wildcard_widens_whatever_it_is_written_beside() {
        // `-x '*',mkv` is not a contradiction to refuse: it is a wider request
        // with a narrower one still written down.
        let dir = tempfile::tempdir().unwrap();
        let subtitle = touch(dir.path(), "episode.srt");
        let video = touch(dir.path(), "episode.mkv");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        let mut expected = vec![subtitle, video];
        expected.sort();
        assert_eq!(
            collected(&sources(&include, &[], &extensions_of(&["mkv", "*"])), &stats),
            expected
        );
    }

    #[test]
    fn test_a_directory_named_like_a_video_is_not_a_video() {
        // Under `-x '*'` the extension filter no longer turns directories away,
        // so the walk has to. A folder called `season1.mkv` is a real thing --
        // an extracted Blu-ray structure is one -- and handing it to the decoder
        // would report a problem against something that is not a file.
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("season1.mkv")).unwrap();
        let video = touch(dir.path(), "episode.mkv");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &[], &extensions_of(&["*"])), &stats),
            vec![video]
        );
        assert!(!stats.had_problems());
    }

    #[test]
    fn test_an_extension_may_be_written_the_way_a_shell_glob_is() {
        // `-x '*.mkv'` would otherwise be a suffix no file has, matching nothing
        // at all -- silently, which is the failure this flag was widened to fix.
        let dir = tempfile::tempdir().unwrap();
        let video = touch(dir.path(), "episode.mkv");
        touch(dir.path(), "episode.srt");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &[], &extensions_of(&["*.MKV"])), &stats),
            vec![video]
        );
    }

    #[test]
    fn test_the_default_list_covers_the_containers_a_camera_or_a_capture_writes() {
        // Named individually rather than as a count, because the failure this
        // guards is one extension quietly going missing: a folder of .mts files
        // that reports "No videos found" reads as a broken tool, not as a
        // narrow default.
        let dir = tempfile::tempdir().unwrap();
        let mut expected: Vec<String> = ["clip.mts", "capture.ts", "rip.vob", "itunes.m4v"]
            .iter()
            .map(|n| touch(dir.path(), n))
            .collect();
        expected.sort();
        touch(dir.path(), "episode.srt");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let defaults = extensions_of(&crate::DEFAULT_EXTENSIONS);
        let stats = RunStats::default();

        assert_eq!(collected(&sources(&include, &[], &defaults), &stats), expected);
    }

    #[test]
    fn test_a_walked_file_carries_the_stat_the_walk_already_did() {
        // The whole point of ScannedFile: the size and mtime downstream needs
        // are the ones this module read, not two further trips to the disk.
        let dir = tempfile::tempdir().unwrap();
        let video = touch(dir.path(), "episode.mkv");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();
        let files = scanned(&sources(&include, &[], &extensions_of(&["mkv"])), &stats);

        let meta = fs::metadata(&video).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, video);
        assert_eq!(files[0].size, meta.len());
        assert_eq!(files[0].mtime, meta.mtime());
        assert_eq!(files[0].mtime_nsec, meta.mtime_nsec());
    }

    #[test]
    fn test_a_named_file_carries_it_too() {
        // The other branch of collect(), which stats the path to find out
        // whether it is a folder at all and must not throw that away either.
        let dir = tempfile::tempdir().unwrap();
        let odd = touch(dir.path(), "holiday.m4v");

        let include = vec![odd.clone()];
        let stats = RunStats::default();
        let files = scanned(&sources(&include, &[], &extensions_of(&["mkv"])), &stats);

        let meta = fs::metadata(&odd).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].size, meta.len());
        assert_eq!(files[0].mtime, meta.mtime());
        assert_eq!(files[0].mtime_nsec, meta.mtime_nsec());
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
    fn test_an_exclude_can_name_one_file_rather_than_a_folder() {
        // The flag takes a path, and a file is one: sparing a known original
        // out of a folder being scanned needs no more than naming it. Nothing
        // here distinguishes the two -- which is exactly why the help used to
        // say FOLDER and be wrong about it.
        let dir = tempfile::tempdir().unwrap();
        let spared = touch(dir.path(), "original.mkv");
        let ordinary = touch(dir.path(), "copy.mkv");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let exclude = vec![spared];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &exclude, &extensions_of(&["mkv"])), &stats),
            vec![ordinary]
        );
        assert!(!stats.had_problems());
    }

    #[test]
    fn test_an_exclude_matches_whole_components_rather_than_a_string_prefix() {
        // `-e ~/clips/take` must not take `take.mkv` with it, and the only
        // thing making that true is that `starts_with` here is the Path method
        // rather than the str one. Both files are real, so the exclude
        // resolves and the survivor proves the matching rule rather than a
        // path that quietly excluded nothing.
        let dir = tempfile::tempdir().unwrap();
        let stem = touch(dir.path(), "take");
        let video = touch(dir.path(), "take.mkv");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let exclude = vec![stem];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &exclude, &extensions_of(&["*"])), &stats),
            vec![video]
        );
        assert!(!stats.had_problems(), "the excluded path resolved");
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

    /// The layout the parent-of-a-found-file version of this check missed, and
    /// the one a library is actually shaped like: nothing sits in the scan root
    /// itself, so no found file's parent encloses the destination.
    ///
    /// The run that followed re-ingested what it had moved, kept THAT copy, and
    /// moved the original in beside it -- an empty library and two files under
    /// `dupes/` after two runs.
    #[test]
    fn test_a_destination_inside_the_scan_is_refused_even_with_no_video_beside_it() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        touch(&root.join("sub"), "a.mkv");

        let include = vec![root.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &[], &extensions);
        request.recursive = true;
        let stats = RunStats::default();

        assert_eq!(
            library(&request, &stats).walk_reaches(&root.join("dupes")),
            Some(root.as_path())
        );
    }

    /// A folder that yielded nothing this time is still a folder a moved file
    /// would be found in next time, which is why the roots are recorded rather
    /// than inferred from the files.
    #[test]
    fn test_a_scan_root_that_found_nothing_still_counts() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();

        let include = vec![root.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let request = sources(&include, &[], &extensions);
        let stats = RunStats::default();

        let library = library(&request, &stats);
        assert!(library.files.is_empty());
        assert_eq!(library.walk_reaches(&root.join("dupes")), Some(root.as_path()));
    }

    #[test]
    fn test_a_destination_above_the_scan_is_not_a_loop() {
        // The arrangement that used to trip the check: the moved file lands in
        // a sibling subtree the scan never reaches.
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let sub = root.join("season_1");
        touch(&sub, "ep01.mkv");

        let include = vec![sub.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let request = sources(&include, &[], &extensions);
        let stats = RunStats::default();

        assert!(library(&request, &stats).walk_reaches(&root).is_none());
    }

    /// The refusal tells the user to exclude the destination, so excluding it
    /// has to be an answer. It was not: with the videos directly in the scan
    /// root, the parent-based check fired on the root whatever `-e` said, and
    /// the advice led nowhere.
    #[test]
    fn test_excluding_the_destination_is_the_way_out_the_refusal_advertises() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        touch(&root, "a.mkv");
        let dest = root.join("dupes");
        fs::create_dir_all(&dest).unwrap();

        let include = vec![root.to_string_lossy().to_string()];
        let exclude = vec![dest.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &exclude, &extensions);
        request.recursive = true;
        let stats = RunStats::default();

        assert!(library(&request, &stats).walk_reaches(&dest).is_none());
    }

    /// A named file is scanned, not walked, so nothing can be re-ingested from
    /// beside it.
    #[test]
    fn test_a_named_file_is_not_a_folder_a_move_could_land_in() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let video = touch(&root, "a.mkv");

        let include = vec![video];
        let extensions = extensions_of(&["mkv"]);
        let request = sources(&include, &[], &extensions);
        let stats = RunStats::default();

        assert!(library(&request, &stats).walk_reaches(&root.join("dupes")).is_none());
    }

    #[test]
    fn test_an_empty_extension_list_fails_before_anything_blocks_on_a_pipe() {
        let stats = RunStats::default();
        let include = vec!["-".to_string()];
        let empty: Vec<String> = vec![];

        assert!(collect(&sources(&include, &[], &empty), &stats).is_err());
    }
}