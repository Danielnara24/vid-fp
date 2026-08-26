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
//! `-x '!flac'` is that escape hatch with the one thing the user already knows
//! about taken back out, and it is the same flag because it is the same
//! sentence: the list is what a walk picks up, and an entry marked `!` is a
//! subtraction from it rather than a second kind of request. So `-x mp4` is
//! "only these", `-x '*'` is "all of them", `-x '!flac'` is "all of them but
//! these", and `-x 'mp4,mkv,!mkv'` is the first with one taken back -- one rule,
//! read left to right, with no flag ordering to get wrong. It is not
//! `--exclude`, which is about which BYTES a run may touch and applies to a
//! path the user named outright; this only ever narrows a folder walk's guess.
//! What earns it a place is that the guess and the thing it gets wrong are
//! often the same shape: FFmpeg fingerprints a `.flac` with cover art in it as
//! a one-frame video, so a music library under `-x '*'` groups albums by their
//! artwork, and the fix is not a narrower list (the folder's videos have no
//! extension either) but that one exception.
//!
//! `--exclude` is the opposite: it applies to everything, including a path named
//! explicitly. It is the one flag whose entire purpose is "do not touch this",
//! and `find ... | vid-fp - -e ~/keep --delete` has to mean what it obviously
//! means. It takes a folder or a single file, because `is_excluded` compares
//! whole path components and neither `starts_with` nor `canonicalize` cares
//! which it was given -- sparing one known original out of a folder being
//! scanned needs no more than naming it. Component-wise is also what keeps
//! `-e ~/clips/take` off `~/clips/take.mkv`. It protects the bytes rather than
//! the spelling, so a file reached by another route -- a symlink, a second scan
//! root -- is excluded just the same; see `is_excluded_target`.
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
use std::collections::{HashMap, HashSet};
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
    /// The file, unless its name is not valid UTF-8.
    ///
    /// Every path downstream of this module is a `String` -- the cache key, the
    /// report column, the deletion target -- so a name holding a byte that is
    /// not valid UTF-8 cannot travel any further. It used to be forced through
    /// `to_string_lossy`, which produced a path that DOES NOT EXIST: the file
    /// was queued under a name with U+FFFD where the bad byte had been, and the
    /// run reported "No such file or directory" against a file sitting right
    /// there in the folder it had just walked. Never dangerous -- a lossy path
    /// cannot be opened, so it cannot be fingerprinted and so cannot become a
    /// DELETE candidate -- but it blamed the decoder for a name this tool had
    /// mangled itself, and gave the user nothing to act on.
    ///
    /// Returning `None` puts the walk on the same footing as `read_path_list`,
    /// which has always named and counted an undecodable line piped in from
    /// another tool rather than mangling it into a path.
    fn of(path: &Path, meta: &std::fs::Metadata) -> Option<Self> {
        Some(ScannedFile {
            path: path.to_str()?.to_string(),
            size: meta.len(),
            mtime: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
        })
    }
}

/// Say so, and count it. Both callers of `ScannedFile::of` word it the same
/// way, because the file is equally unreachable however it was reached.
fn record_undecodable(path: &Path, stats: &RunStats) {
    let shown = path.to_string_lossy();
    log::error!(target: crate::stats::COUNTED, "Filename is not valid UTF-8 and was skipped: {}", shown);
    stats
        .unreadable
        .record(format!("{}: filename is not valid UTF-8", shown));
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
    /// What the walk was willing to pick up, which is the only thing that makes
    /// the count sayable. A positive `--extensions` list means the walk turned
    /// away everything not named like a video, so what came back are video
    /// files; under `-x '*'` there was no filter at all and they are simply
    /// files, and under `-x '!flac'` there was one that removes a hole rather
    /// than naming videos, so they are simply files too. Saying "video files"
    /// there is how a scan of a home directory announced "Found 229112 video
    /// files" and then spent the run failing to decode 229 thousand of them.
    ///
    /// A named path is never filtered (see the module doc), so this describes
    /// the walk and not every route in. That is the one it is asked about: a
    /// user who typed a path knows what they typed.
    pub any_extension: bool,
    /// The folders the walk reached, canonicalized. Two kinds of entry: a
    /// folder the user pointed at, and -- under `--follow-symlinks` -- the
    /// TARGET of every directory symlink the walk descended through. A path
    /// the user named that turned out to be a file is not one of these:
    /// naming a file scans that file, and nothing can ever be moved *into* it.
    ///
    /// The second kind is what keeps the question answerable at all once links
    /// are being followed. A scan of `lib` whose only route to `store` is
    /// `lib/link -> store` reaches every file in `store` while `store` sits
    /// under no root the user typed, so `--move-to store` looked safe and the
    /// next run found what the last one had moved.
    walked: Vec<Reached>,
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
    ///
    /// Both sides of the comparison are canonical, which is the only reason a
    /// symlink cannot walk around it: `dest` comes from `resolve_move_to`, and
    /// a folder reached through a link contributes its target rather than the
    /// path the walk spelled.
    pub fn walk_reaches(&self, dest: &Path) -> Option<&Reached> {
        if is_excluded(dest, &self.excluded) {
            return None;
        }
        self.walked.iter().find(|r| dest.starts_with(&r.root))
    }
}

/// A folder this scan reaches, and how it got there.
///
/// The route matters only for what the `--move-to` refusal can say. A
/// destination reached through a link is usually the link's target ITSELF, so a
/// message built on the target alone came out as "X is inside X, which this run
/// scans" -- true, unhelpful, and silent about the one path the user has to go
/// and look at.
pub struct Reached {
    root: PathBuf,
    /// The symlink the walk followed to get here, if it did not get here
    /// directly.
    via: Option<PathBuf>,
}

impl std::fmt::Display for Reached {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.via {
            None => write!(f, "inside {}, which this run scans", self.root.display()),
            Some(link) => write!(
                f,
                "reached through the symlink {}, which this run follows into {}",
                link.display(),
                self.root.display()
            ),
        }
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
        // Worth saying out loud: these are the settings under which a folder of
        // text files becomes a folder of failed fingerprints.
        Wanted::Anything => info!("Searching every file, whatever its extension (-x '*')."),
        Wanted::AnythingBut(set) => info!(
            "Searching every file except these extensions: {:?}",
            sorted(set)
        ),
        Wanted::OneOf(set) => info!("Searching extensions: {:?}", sorted(set)),
    }

    let excludes = resolve_excludes(sources.exclude, stats);
    let requested = requested_paths(sources, stats)?;

    // Identity-based deduplication. A symlink, a hard link, a second scan root
    // that overlaps the first, the same path piped in twice, and a bind-mount
    // alias all resolve to the same (device, inode) pair. Keying on that
    // identity means each set of bytes is fingerprinted exactly once, so the
    // report never lists a file as a duplicate of itself and the "space freed"
    // figure never counts bytes that deleting a link would not return. WHICH of
    // the names is the one kept is `Collecting::claim`'s business, and it is not
    // simply the first: see the symlink preference there.
    let mut into = Collecting {
        seen_inodes: HashMap::new(),
        found: Vec::new(),
        walked: Vec::new(),
    };

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
                log::error!(target: crate::stats::COUNTED, "Could not resolve '{}': {}", path, e);
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
                log::error!(target: crate::stats::COUNTED, "Cannot stat {}: {}", resolved.display(), e);
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
            into.walked.push(Reached {
                root: resolved.clone(),
                via: None,
            });

            if !walk_folder(&resolved, sources, &extensions, &excludes, &mut into, stats) {
                return Ok(Scan::Interrupted);
            }
        } else if meta.is_file() {
            // No extension check: see the module docs. The user named this file.
            //
            // Decodability is settled before the identity is claimed. A name
            // this tool cannot carry may well be a hard link to one it can, and
            // taking the inode for the unusable name would drop the usable one
            // as an alias of a file that never made it into the list.
            match ScannedFile::of(&resolved, &meta) {
                None => record_undecodable(&resolved, stats),
                // Never a symlink, whatever the user typed: `canonicalize`
                // above has already resolved every link out of the path, which
                // is why the preference `claim` applies needs nothing measured
                // here. A named link and the file it points at are the same
                // entry by the time either reaches this line.
                Some(file) => into.claim((meta.dev(), meta.ino()), file, false, stats),
            }
        } else {
            // A socket, fifo, or device node. Naming one is a mistake worth
            // hearing about rather than a file worth trying to decode.
            log::error!(target: crate::stats::COUNTED, "Not a file or folder: {}", resolved.display());
            stats
                .unresolved_includes
                .record(format!("{}: not a file or folder", resolved.display()));
        }
    }

    Ok(Scan::Complete(Library {
        files: into.found,
        any_extension: !extensions.is_a_guess_at_video(),
        walked: into.walked,
        excluded: excludes,
    }))
}

/// What a walk accumulates, carried from one scan root to the next.
///
/// One bag rather than three out-parameters because the three are one thing:
/// the answer being built. The inode map in particular has to span roots -- two
/// scan roots that overlap must collapse to one entry, not two -- so none of
/// these can be per-folder.
struct Collecting {
    seen_inodes: HashMap<(u64, u64), Occupant>,
    found: Vec<ScannedFile>,
    walked: Vec<Reached>,
}

/// The name that currently holds one set of bytes, and whether it is a real one.
///
/// `symlink` is what lets a later real name displace an earlier link; it is
/// carried rather than re-derived because both callers already know it for
/// nothing (WalkDir keeps the entry's own type, and a named path has been
/// through `canonicalize`), and `ScannedFile` deliberately holds neither the
/// identity nor how the file was reached.
#[derive(Clone, Copy)]
struct Occupant {
    index: usize,
    symlink: bool,
}

impl Collecting {
    /// Queue a file, unless these bytes are already queued under another name.
    ///
    /// Which name that is matters, and first-come is not good enough. A folder
    /// holding `a.mp4`, a hard link to it and a symlink to it offers three
    /// names for one inode, and the walk meets them in readdir order -- so the
    /// SYMLINK could win the identity, and the two real names were then dropped
    /// as aliases of it. That put a pointer in the library where the video
    /// should have been: the run ranked the link against the genuine duplicate
    /// elsewhere in the folder, marked the link DELETE, and `--delete
    /// --permanent` unlinked it. Everything downstream stayed honest about that
    /// -- `export::on_disk` sees the link, the row reads UNLINKED and its bytes
    /// are struck out of the total -- but the user asked for a duplicate to go
    /// and got a broken shortcut removed instead, with both copies of the video
    /// still on disk and a second run needed to reach them.
    ///
    /// A real name is therefore preferred whenever one turns up, whatever the
    /// order. Everything else stays first-come: two hard links to one inode are
    /// equally good names for it, and so are two symlinks, so there is nothing
    /// to choose between them and no reason to make the answer depend on which
    /// arrived first any more than it already does.
    ///
    /// Exactly one name is skipped per collision either way, so the count the
    /// summary prints is unchanged; only which path it is about moves.
    fn claim(&mut self, id: (u64, u64), file: ScannedFile, symlink: bool, stats: &RunStats) {
        let Some(held) = self.seen_inodes.get(&id).copied() else {
            self.seen_inodes.insert(
                id,
                Occupant {
                    index: self.found.len(),
                    symlink,
                },
            );
            self.found.push(file);
            return;
        };

        let dropped = if held.symlink && !symlink {
            self.seen_inodes.insert(id, Occupant { symlink, ..held });
            std::mem::replace(&mut self.found[held.index], file)
        } else {
            file
        };

        stats.skipped_alias.bump();
        log::debug!(
            "Skipping {}: same inode as {}, which is already queued",
            dropped.path,
            self.found[held.index].path
        );
    }
}

/// Walk one folder, appending every video in it. Returns false if interrupted.
fn walk_folder(
    base: &Path,
    sources: &Sources,
    extensions: &Wanted,
    excludes: &[PathBuf],
    into: &mut Collecting,
    stats: &RunStats,
) -> bool {
    let mut walker = WalkDir::new(base).follow_links(sources.follow_symlinks);

    if !sources.recursive {
        // Non-recursive by default: only the folder itself (depth 0) and its
        // immediate files (depth 1).
        walker = walker.max_depth(1);
    }

    let follow = sources.follow_symlinks;

    // Directories are canonicalized here so an excluded subtree reached through
    // a link is PRUNED rather than merely rejected file by file. Folders are
    // few, so a realpath each is cheap; the files inside are checked further
    // down, once the extension filter has thinned them out.
    let it = walker.into_iter().filter_entry(|e| {
        let through_links = e.file_type().is_dir() && (follow || e.path_is_symlink());
        !is_excluded_target(e.path(), excludes, through_links)
    });

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
                log::error!(target: crate::stats::COUNTED, "Cannot scan {}: {}", at, e);
                stats.unwalkable.record(format!("{}: {}", at, e));
                continue;
            }
        };

        // A directory is never a file to fingerprint, and WalkDir already knows
        // which entries are directories -- so asking it here costs nothing and
        // saves a stat() per folder under `-x '*'`, where the extension filter
        // no longer turns them away.
        if entry.file_type().is_dir() {
            // One that was reached by following a link is a place this request
            // scans without any root the user typed enclosing it, so the
            // `--move-to` guard has to be told about it. See `Library::walked`.
            if follow && entry.path_is_symlink() {
                if let Ok(target) = std::fs::canonicalize(entry.path()) {
                    into.walked.push(Reached {
                        root: target,
                        via: Some(entry.path().to_path_buf()),
                    });
                }
            }
            continue;
        }

        let path = entry.path();

        // Extension next: it is free, and it keeps us from stat()ing every
        // non-video file in the tree.
        if !extensions.accepts(path) {
            continue;
        }

        // The exclusion test again, now through the link. `filter_entry` above
        // compared the path the WALK used, which under --follow-symlinks goes
        // through the link and so can never carry the canonical prefix an
        // `--exclude` resolved to -- while every destructive step downstream
        // (`remove_file`, `rename`) resolves the link and acts on the real
        // file. Left as it was, `-e keep` did not protect `keep/precious.mp4`
        // from a run that reached it as `scan/linkdir/precious.mp4`.
        //
        // Not counted, unlike the same test in `collect`: a file the walk found
        // was not asked for by name, and `stats.skipped_excluded` says "named
        // path(s)". Counting it here also made the number a mixture of two
        // routes rather than an answer to either -- an excluded subtree is
        // PRUNED by `filter_entry` above and contributes nothing however many
        // files are behind it, so a run that spared a thousand videos through a
        // prefix match and one through a link reported "1 named path(s)
        // skipped". Which of the two a given exclusion takes is an
        // implementation detail the user has no way to see.
        if is_excluded_target(path, excludes, follow || entry.path_is_symlink()) {
            log::debug!(
                "Skipping {}: leads to a file under an --exclude path",
                path.display()
            );
            continue;
        }

        // One stat does four jobs: it follows symlinks (so a link to a video is
        // treated as that video), confirms this is a regular file, yields the
        // identity used for deduplication, and supplies the size and mtime that
        // travel out with the file so nothing downstream has to ask again.
        let meta = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(e) => {
                log::error!(target: crate::stats::COUNTED, "Cannot stat {}: {}", path.display(), e);
                stats
                    .unreadable
                    .record(format!("{}: {}", path.display(), e));
                continue;
            }
        };

        if !meta.is_file() {
            continue;
        }

        // Before the inode is claimed, for the reason `collect` gives: an
        // unusable name must not stand in for a hard link that has a usable one.
        let Some(file) = ScannedFile::of(path, &meta) else {
            record_undecodable(path, stats);
            continue;
        };

        // The walk is the only route by which a link can reach the library at
        // all -- and WalkDir already knows, so preferring a real name over it
        // costs no syscall. Under --follow-symlinks the stat above reports the
        // TARGET's type, which is exactly the reading that hid this: the entry
        // is a regular file by then, and only the entry's own type still says
        // how it was reached.
        into.claim(
            (meta.dev(), meta.ino()),
            file,
            entry.path_is_symlink(),
            stats,
        );
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
                log::error!(target: crate::stats::COUNTED, "Path is not valid UTF-8 and was skipped: {}", shown);
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
/// Shapes rather than one set, because "every file" is not expressible as a
/// list of suffixes: a file with no extension at all has nothing for
/// `Path::extension` to return, so no entry could ever name it. That is not a
/// corner case -- it is a whole camera dump, a DVD rip, or anything named by a
/// content hash -- and before `-x '*'` those folders were unreachable from a
/// walk, with the only workaround being to pipe the paths in from another tool.
///
/// The third is that same absence with a hole in it, and it is here for the
/// same reason: the folders `-x '*'` exists for are full of files no list can
/// name, and one extension in them is often known to be worth skipping. A set
/// of what to refuse and a set of what to accept are not the same question, so
/// they are not the same variant -- `normalize_extensions` decides which of the
/// two a given `-x` list is asking, once, and the walk never asks again.
enum Wanted {
    /// `-x '*'`. Every regular file the walk finds, extension or not.
    Anything,
    /// `-x '!flac'`. Every file except the ones named, which is the same
    /// contract as `Anything` minus a hole -- a file with no extension is not
    /// named by anything, so it is still taken.
    AnythingBut(HashSet<String>),
    /// Files whose extension is in this set, lowercased and dot-free.
    OneOf(HashSet<String>),
}

impl Wanted {
    fn accepts(&self, path: &Path) -> bool {
        let extension = || {
            path.extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.to_lowercase())
        };
        match self {
            Wanted::Anything => true,
            Wanted::AnythingBut(refused) => {
                extension().is_none_or(|ext| !refused.contains(ext.as_str()))
            }
            Wanted::OneOf(extensions) => {
                extension().is_some_and(|ext| extensions.contains(ext.as_str()))
            }
        }
    }

    /// Whether the walk turned away anything for being named wrong.
    ///
    /// False for both wildcard shapes: what came back is then simply files, and
    /// a run that calls them video files is the one that announced "Found
    /// 229112 video files" over a home directory. Excepting `flac` narrows
    /// nothing towards video -- everything not named `.flac` is still handed to
    /// the decoder -- so it belongs on that side of the line, not this one.
    fn is_a_guess_at_video(&self) -> bool {
        matches!(self, Wanted::OneOf(_))
    }
}

/// The wildcard, spelled the way a shell user expects. Quoting is on them --
/// unquoted it is a glob, and one that expands to the directory's contents.
const WILDCARD: &str = "*";

/// What turns an entry into an exception: `-x '!flac'` is every file but those.
/// Quoting is on the user here too -- an interactive bash expands `!` as
/// history unless it is in single quotes.
const NOT: char = '!';

/// HashSet iteration order is unspecified; sort for a stable log line.
fn sorted(set: &HashSet<String>) -> Vec<&str> {
    let mut shown: Vec<&str> = set.iter().map(|s| s.as_str()).collect();
    shown.sort_unstable();
    shown
}

/// The exceptions in a positive `-x` list that have nothing to subtract from,
/// spelt the way they were written.
///
/// Empty for both wildcard shapes, which are never asked: there an exception is
/// the only part of the list that can take a file away, so it always means
/// something.
fn inert_exceptions(wanted: &HashSet<String>, refused: &HashSet<String>) -> Vec<String> {
    let mut inert: Vec<String> = refused
        .difference(wanted)
        .map(|ext| format!("{}{}", NOT, ext))
        .collect();
    inert.sort_unstable();
    inert
}

fn normalize_extensions(requested: &[String]) -> Result<Wanted> {
    // Strip an optional leading dot and lowercase, so `-x .MP4`, `-x MP4`, and
    // `-x mp4` all behave identically. A HashSet gives O(1) lookups during the
    // walk and dedups automatically.
    // `-x '*.mkv'` is how a shell user spells the same thing, and the entry it
    // produces would otherwise be a suffix no file on earth has -- matching
    // nothing, silently, which is the failure this flag is being widened to fix.
    // A leading `!` is what makes an entry an exception rather than a request;
    // it is read before the rest, so `-x '!*.FLAC'` spells one too.
    let mut wanted: HashSet<String> = HashSet::new();
    let mut refused: HashSet<String> = HashSet::new();

    for entry in requested {
        let entry = entry.trim();
        let (into, entry) = match entry.strip_prefix(NOT) {
            Some(rest) => (&mut refused, rest.trim()),
            None => (&mut wanted, entry),
        };

        let entry = entry.strip_prefix("*.").unwrap_or(entry);
        let entry = entry.trim_start_matches('.').to_lowercase();

        if !entry.is_empty() {
            into.insert(entry);
        }
    }

    // "Everything except everything" is the empty walk, and no user means it.
    if refused.contains(WILDCARD) {
        anyhow::bail!("--extensions excludes every file (-x '!*' matches nothing).");
    }

    // What the exceptions are subtracted from. `*` asks for it outright, and so
    // does a list that only says what it does NOT want: "every file but flac" is
    // the whole of what `-x '!flac'` can mean, and requiring the `*` beside it
    // would be a spelling rule rather than a distinction.
    let everything = wanted.contains(WILDCARD) || (wanted.is_empty() && !refused.is_empty());

    if everything {
        // The wildcard wins over anything beside it. `-x '*',mkv` is not a
        // contradiction to refuse -- it is a wider request with a narrower one
        // still written down, and the wider one is what was asked for. An
        // exception is not a narrower request: it is the only part of the list
        // that can still take a file away.
        return Ok(if refused.is_empty() {
            Wanted::Anything
        } else {
            Wanted::AnythingBut(refused)
        });
    }

    // Otherwise the positive entries are the whole of the walk's guess, and an
    // exception written beside them can only take one back out again.
    //
    // One that takes nothing out is worth a word. `-x` REPLACES the default
    // list, so someone meaning "the defaults minus flac" writes
    // `-x 'mp4,!flac'`, gets a one-extension walk, and is told only that the
    // walk is searching `["mp4"]` -- a line that reads like confirmation
    // because the extension they typed is in it. The exception is the half of
    // the request that was silently dropped, so it is the half that has to be
    // said out loud. Not an error: the walk is a perfectly good one and
    // refusing it would also refuse `-x 'mp4,mkv,!flac'`, which is harmless
    // belt-and-braces rather than a typo.
    let inert = inert_exceptions(&wanted, &refused);
    if !inert.is_empty() {
        log::warn!(
            "--extensions: {:?} matched nothing in {:?} and took nothing away. -x \
             REPLACES the default extension list rather than narrowing it, so this \
             walk is exactly {:?}; an exception can only take back an extension \
             written beside it, and '{}' on its own means EVERY file except that one \
             rather than the defaults except it.",
            inert,
            sorted(&wanted),
            sorted(&wanted),
            inert[0]
        );
    }

    wanted.retain(|ext| !refused.contains(ext));

    if wanted.is_empty() {
        anyhow::bail!(
            "No valid video extensions to search for (--extensions was empty, \
             or every extension in it was excluded)."
        );
    }

    Ok(Wanted::OneOf(wanted))
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
                    target: crate::stats::COUNTED,
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

/// Is this path, or the file it actually leads to, under an `--exclude`?
///
/// `--exclude` protects BYTES, and the only name every route to a set of bytes
/// agrees on is the canonical one -- which is why `resolve_excludes`
/// canonicalizes what the user typed. A walk path does not have to be
/// canonical: with `--follow-symlinks` on, `scan/linkdir/precious.mp4` names
/// the same file as `keep/precious.mp4` and shares not one component with it,
/// so the prefix test could never fire. It failed in BOTH directions, which is
/// what made it impossible to work around: excluding the real folder did not
/// match the path the walk used, and excluding the link path the user could
/// actually see in the report was canonicalized into the real folder and so did
/// not match either. Everything downstream resolves the link -- `remove_file`
/// on the walk path unlinks the real file -- so the exclusion has to as well.
///
/// The raw test comes first because it is free and answers almost every case:
/// a walk with no symlink in it produces canonical paths already, `collect`
/// having canonicalized the root. `through_links` is what the callers use to
/// keep the realpath off the default path entirely -- it is asked only of an
/// entry that IS a link, or of any entry at all when links are being followed.
///
/// A path that will not canonicalize is not excluded, and needs no protecting:
/// it cannot be opened, so it cannot be fingerprinted, moved or deleted.
fn is_excluded_target(path: &Path, excludes: &[PathBuf], through_links: bool) -> bool {
    if is_excluded(path, excludes) {
        return true;
    }

    if !through_links || excludes.is_empty() {
        return false;
    }

    std::fs::canonicalize(path).is_ok_and(|real| is_excluded(&real, excludes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn extensions_of(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// A real file whose name is not valid UTF-8, which is a perfectly ordinary
    /// thing for a Linux filename to be: the kernel stores bytes, and only the
    /// tools looking at them care whether they decode.
    fn touch_raw(dir: &Path, name: &[u8]) -> PathBuf {
        use std::os::unix::ffi::OsStrExt;
        let path = dir.join(std::ffi::OsStr::from_bytes(name));
        fs::write(&path, b"video").unwrap();
        path
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

    /// The walk's own answer to "are these video files, or just files", which is
    /// the only thing that entitles the run to call them either.
    #[test]
    fn test_a_wildcard_walk_says_it_filtered_nothing() {
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "clip.mkv");
        touch(dir.path(), "notes.txt");
        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        let wild = library(&sources(&include, &[], &["*".to_string()]), &stats);
        assert!(wild.any_extension);
        assert_eq!(wild.files.len(), 2, "the text file is a candidate under -x '*'");

        let named = library(&sources(&include, &[], &["mkv".to_string()]), &stats);
        assert!(!named.any_extension);
        assert_eq!(named.files.len(), 1, "and is not one otherwise");
    }

    /// The folder a file moved to `dest` would be found in again, if any.
    fn reaches(library: &Library, dest: &Path) -> Option<PathBuf> {
        library.walk_reaches(dest).map(|r| r.root.clone())
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

    /// The name is skipped HERE, where the problem is, rather than mangled into
    /// a path that does not exist and handed to the decoder.
    ///
    /// What that used to look like: `to_string_lossy` put U+FFFD where the bad
    /// byte was, the file was queued under that name, and the run ended with
    /// "Failed to process .../bad<U+FFFD> name.mkv: No such file or directory"
    /// -- a missing-file error against a file the walk had just listed, with
    /// nothing to say the tool had renamed it on the way through.
    #[test]
    fn test_a_walked_name_that_is_not_utf8_is_skipped_rather_than_mangled() {
        let dir = tempfile::tempdir().unwrap();
        let good = touch(dir.path(), "episode.mkv");
        touch_raw(dir.path(), b"bad\xFF name.mkv");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        let found = collected(&sources(&include, &[], &extensions_of(&["mkv"])), &stats);
        assert_eq!(found, vec![good], "and the rest of the folder still scans");
        assert!(
            found.iter().all(|p| Path::new(p).exists()),
            "nothing may leave here under a name that is not on the disk"
        );
        assert_eq!(stats.unreadable.count(), 1, "counted, and named for what it is");
        assert!(stats.unreadable.samples()[0].contains("not valid UTF-8"));
    }

    /// The other branch of `collect`. A path typed or piped in is a `String`
    /// and so decodes by construction, but canonicalizing it does not have to:
    /// a link with an ordinary name can resolve to one without.
    #[test]
    fn test_a_named_path_resolving_to_an_undecodable_name_is_skipped_too() {
        let dir = tempfile::tempdir().unwrap();
        let bad = touch_raw(dir.path(), b"bad\xFFname.mkv");
        let link = dir.path().join("link.mkv");
        std::os::unix::fs::symlink(&bad, &link).unwrap();

        let include = vec![link.to_string_lossy().to_string()];
        let stats = RunStats::default();

        assert!(collected(&sources(&include, &[], &extensions_of(&["mkv"])), &stats).is_empty());
        assert_eq!(stats.unreadable.count(), 1);
        assert!(stats.had_problems(), "a file the user named and did not get");
    }

    /// Decodability is settled before the inode is claimed, so an unusable name
    /// cannot stand in for a hard link that has a usable one -- which would
    /// have skipped the readable path as an alias of a file that was never
    /// queued, losing both.
    #[test]
    fn test_an_undecodable_name_does_not_claim_the_inode_of_a_readable_one() {
        let dir = tempfile::tempdir().unwrap();
        let bad = touch_raw(dir.path(), b"bad\xFFname.mkv");
        let good = dir.path().join("episode.mkv");
        fs::hard_link(&bad, &good).unwrap();
        let via_bad = dir.path().join("link.mkv");
        std::os::unix::fs::symlink(&bad, &via_bad).unwrap();

        // The unusable one first: it is the order that used to lose the file.
        let include = vec![
            via_bad.to_string_lossy().to_string(),
            good.to_string_lossy().to_string(),
        ];
        let stats = RunStats::default();

        assert_eq!(
            collected(&sources(&include, &[], &extensions_of(&["mkv"])), &stats),
            vec![fs::canonicalize(&good).unwrap().to_string_lossy().to_string()]
        );
        assert_eq!(stats.skipped_alias.count(), 0, "nothing was queued for it to alias");
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
    fn test_an_exception_takes_one_extension_out_of_every_file() {
        // The case it was written for: a music folder under `-x '*'` groups its
        // albums by their cover art, because a .flac with artwork in it is a
        // one-frame mjpeg video carrying the track's whole length. The fix
        // cannot be a positive list, since what is worth scanning beside it may
        // have no extension at all.
        let dir = tempfile::tempdir().unwrap();
        let video = touch(dir.path(), "episode.mkv");
        let bare = touch(dir.path(), "VTS_01_1");
        touch(dir.path(), "01 Storm.flac");
        touch(dir.path(), "02 Static.FLAC");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        let mut expected = vec![bare, video];
        expected.sort();

        // Every spelling of the same request: the exception on its own, the
        // exception written under the wildcard it implies, and both the shell
        // glob and the dotted form of the extension itself.
        for list in [
            vec!["!flac"],
            vec!["*", "!flac"],
            vec!["!*.FLAC"],
            vec!["!.flac"],
        ] {
            assert_eq!(
                collected(&sources(&include, &[], &extensions_of(&list)), &stats),
                expected,
                "-x {:?}",
                list
            );
        }
    }

    #[test]
    fn test_an_exception_beside_a_list_takes_one_back_out_of_it() {
        // The list is read left to right as one sentence rather than as two
        // kinds of request, so an exception narrows whatever the positives
        // asked for instead of widening it to everything.
        let dir = tempfile::tempdir().unwrap();
        let video = touch(dir.path(), "episode.mkv");
        touch(dir.path(), "episode.srt");
        touch(dir.path(), "VTS_01_1");

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        assert_eq!(
            collected(
                &sources(&include, &[], &extensions_of(&["mkv", "srt", "!srt"])),
                &stats
            ),
            vec![video]
        );
    }

    #[test]
    fn test_an_exception_that_takes_nothing_back_out_is_said_out_loud() {
        // `-x` REPLACES the default list, so someone meaning "the defaults
        // except flac" writes `-x 'mp4,!flac'`, gets a one-extension walk, and
        // is told only that the walk is searching ["mp4"] -- which reads like
        // confirmation, because the extension they typed is in it. The dropped
        // half of the request is the half that has to be reported.
        let one = |list: &[&str]| {
            let requested = extensions_of(list);
            let mut wanted: HashSet<String> = HashSet::new();
            let mut refused: HashSet<String> = HashSet::new();
            for entry in &requested {
                match entry.strip_prefix(NOT) {
                    Some(rest) => refused.insert(rest.to_string()),
                    None => wanted.insert(entry.to_string()),
                };
            }
            inert_exceptions(&wanted, &refused)
        };

        assert_eq!(one(&["mp4", "!flac"]), vec!["!flac".to_string()]);
        assert_eq!(one(&["mp4", "!flac", "!wav"]), vec!["!flac", "!wav"]);
        // An exception the list really can act on is not inert, and neither is
        // one in a list that is refused outright a moment later.
        assert!(one(&["mp4", "mkv", "!mkv"]).is_empty());
        assert!(one(&["mkv", "!mkv"]).is_empty());
        // And the walk it describes is still the walk that happens.
        let dir = tempfile::tempdir().unwrap();
        let video = touch(dir.path(), "clip.mp4");
        touch(dir.path(), "track.flac");
        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();
        assert_eq!(
            collected(
                &sources(&include, &[], &extensions_of(&["mp4", "!flac"])),
                &stats
            ),
            vec![video]
        );
    }

    #[test]
    fn test_a_request_that_can_match_nothing_is_refused_rather_than_walked() {
        // Both ways of writing the empty walk. Refusing is the point: a scan
        // that cannot match a file is a typo, and reporting "No videos found"
        // for it is how a user concludes the tool is broken.
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "episode.mkv");
        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        for list in [vec!["!*"], vec!["mkv", "!mkv"], vec![""]] {
            assert!(
                collect(&sources(&include, &[], &extensions_of(&list)), &stats).is_err(),
                "-x {:?} matches nothing and has to say so",
                list
            );
        }
    }

    #[test]
    fn test_a_walk_that_only_removed_a_hole_did_not_look_for_videos() {
        // `any_extension` is what entitles the run to call what it found video
        // files. Excepting flac turns nothing away for being named wrong, so it
        // belongs with the wildcard and not with a list.
        let dir = tempfile::tempdir().unwrap();
        touch(dir.path(), "clip.mkv");
        touch(dir.path(), "notes.txt");
        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();

        let library = library(&sources(&include, &[], &extensions_of(&["!flac"])), &stats);
        assert!(library.any_extension);
        assert_eq!(library.files.len(), 2, "the text file is still a candidate");
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

    /// `--exclude` protects bytes, not spellings, and a link is another
    /// spelling. Under `--follow-symlinks` the walk path goes THROUGH the link
    /// and shares no component with the folder the exclude resolved to, so the
    /// prefix test could never fire -- and `remove_file` on that walk path
    /// unlinks the real file inside the protected folder. `-e keep` with
    /// `--delete --permanent` destroyed exactly what it was written to save.
    #[test]
    fn test_an_exclude_protects_a_file_reached_through_a_symlinked_folder() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let keep = root.join("keep");
        touch(&keep, "precious.mkv");
        touch(&keep.join("deep"), "also_precious.mkv");
        let scan = root.join("scan");
        let ordinary = touch(&scan, "copy.mkv");
        std::os::unix::fs::symlink(&keep, scan.join("linkdir")).unwrap();

        let include = vec![scan.to_string_lossy().to_string()];
        let exclude = vec![keep.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &exclude, &extensions);
        request.recursive = true;
        request.follow_symlinks = true;
        let stats = RunStats::default();

        assert_eq!(collected(&request, &stats), vec![ordinary]);
        assert!(!stats.had_problems(), "an exclusion is a skip, not a failure");
        assert_eq!(
            stats.skipped_excluded.count(),
            0,
            "the link itself is refused, so the subtree behind it is pruned \
             rather than walked and rejected file by file"
        );
    }

    /// The other direction, and the reason the first one could not simply be
    /// worked around: the path the user sees in the report is the link path,
    /// and `resolve_excludes` canonicalizes that into the real folder. Naming
    /// either one has to work, and canonicalizing both sides is what makes both
    /// the same question.
    #[test]
    fn test_excluding_the_link_path_works_as_well_as_excluding_the_real_one() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        touch(&root.join("keep"), "precious.mkv");
        let scan = root.join("scan");
        let ordinary = touch(&scan, "copy.mkv");
        let link = scan.join("linkdir");
        std::os::unix::fs::symlink(root.join("keep"), &link).unwrap();

        let include = vec![scan.to_string_lossy().to_string()];
        let exclude = vec![link.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &exclude, &extensions);
        request.recursive = true;
        request.follow_symlinks = true;
        let stats = RunStats::default();

        assert_eq!(collected(&request, &stats), vec![ordinary]);
        assert!(!stats.had_problems());
    }

    /// The benign-looking half of the same bug, which is why the check is not
    /// gated on `--follow-symlinks`: a symlinked FILE is walked by default (the
    /// stat follows it), so it escaped `-e` too. Removing it only unlinked the
    /// link, leaving the original intact -- but the run then reported the
    /// original's bytes as freed, having freed none of them.
    #[test]
    fn test_an_exclude_protects_a_file_reached_through_a_symlink_to_it() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let protected = touch(&root.join("keep"), "precious.mkv");
        let scan = root.join("scan");
        let ordinary = touch(&scan, "copy.mkv");
        std::os::unix::fs::symlink(&protected, scan.join("link.mkv")).unwrap();

        let include = vec![scan.to_string_lossy().to_string()];
        let exclude = vec![root.join("keep").to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let request = sources(&include, &exclude, &extensions);
        let stats = RunStats::default();

        assert_eq!(collected(&request, &stats), vec![ordinary]);
        assert_eq!(
            stats.skipped_excluded.count(),
            0,
            "the walk found it; nobody named it"
        );
    }

    /// The per-file half of the check, which the pruning above never reaches:
    /// one file inside a folder that is otherwise fair game. `-e` takes a path
    /// and a file is one, so sparing a single known original has to work
    /// through a link as well.
    #[test]
    fn test_an_exclude_naming_one_file_reaches_it_through_a_link_too() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let elsewhere = root.join("elsewhere");
        let protected = touch(&elsewhere, "precious.mkv");
        let spare = touch(&elsewhere, "spare.mkv");
        let scan = root.join("scan");
        let ordinary = touch(&scan, "copy.mkv");
        std::os::unix::fs::symlink(&elsewhere, scan.join("linkdir")).unwrap();

        let include = vec![scan.to_string_lossy().to_string()];
        let exclude = vec![protected];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &exclude, &extensions);
        request.recursive = true;
        request.follow_symlinks = true;
        let stats = RunStats::default();

        let mut expected = vec![
            ordinary,
            scan.join("linkdir/spare.mkv").to_string_lossy().to_string(),
        ];
        expected.sort();
        assert_eq!(collected(&request, &stats), expected, "{} was found", spare);
        assert_eq!(
            stats.skipped_excluded.count(),
            0,
            "the walk found it; nobody named it"
        );
        assert!(!stats.had_problems());
    }

    /// The count under `--exclude` answers for exactly one route, and the
    /// summary line says which: "named path(s) skipped because --exclude covers
    /// them". A file the WALK dropped is not one -- it was never asked for, and
    /// "you excluded it" is the whole story. The walk used to bump it from the
    /// per-file check, which made the number an answer to neither question: an
    /// excluded subtree is pruned whole and contributes nothing however many
    /// files sit behind it, so the same exclusion counted 0 or 1 depending on
    /// whether it was reached by a prefix match or through a link.
    ///
    /// Driven as one run so the two routes are counted against each other.
    #[test]
    fn test_only_a_path_the_user_named_is_counted_as_excluded() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let scan = root.join("scan");
        let keep = scan.join("keep");
        let protected = touch(&keep, "precious.mkv");
        let behind_the_link = touch(&keep, "also_precious.mkv");
        let ordinary = touch(&scan, "copy.mkv");
        // Three ways into the excluded folder: named outright, pruned as a
        // subtree by `filter_entry`, and reached file-by-file through a link.
        std::os::unix::fs::symlink(&behind_the_link, scan.join("link.mkv")).unwrap();

        let include = vec![protected, scan.to_string_lossy().to_string()];
        let exclude = vec![keep.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &exclude, &extensions);
        request.recursive = true;
        let stats = RunStats::default();

        assert_eq!(collected(&request, &stats), vec![ordinary]);
        assert_eq!(
            stats.skipped_excluded.count(),
            1,
            "one path was named and skipped; the other two were never asked for"
        );
        assert!(!stats.had_problems());
    }

    /// A link that leads somewhere ordinary is still scanned. The exclusion
    /// check resolves links; it does not turn them away.
    #[test]
    fn test_following_a_link_still_finds_the_videos_beyond_it() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let elsewhere = root.join("elsewhere");
        touch(&elsewhere, "beyond.mkv");
        let scan = root.join("scan");
        let ordinary = touch(&scan, "copy.mkv");
        std::os::unix::fs::symlink(&elsewhere, scan.join("linkdir")).unwrap();

        let include = vec![scan.to_string_lossy().to_string()];
        let exclude = vec![root.join("keep").to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &exclude, &extensions);
        request.recursive = true;
        request.follow_symlinks = true;
        let stats = RunStats::default();

        // Walked as `scan/linkdir/beyond.mkv`: the path is the one the walk
        // used, and only the exclusion test asks where it leads.
        let mut expected = vec![ordinary, scan.join("linkdir/beyond.mkv").to_string_lossy().to_string()];
        expected.sort();
        assert_eq!(collected(&request, &stats), expected);
        assert_eq!(stats.unresolved_excludes.count(), 1, "the -e path does not exist");
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
    fn test_a_symlink_never_stands_in_for_the_video_it_points_at() {
        // The whole folder the bug needs: one video, a hard link to it, and a
        // symlink to it, all three offering the same inode to the walk. Which
        // one readdir hands over first is not ours to choose -- so the property
        // asserted is the one that has to hold whatever the order, that the
        // name left standing is one a deletion would really free bytes by.
        //
        // Left as first-come, the link could win, and a run that then ranked it
        // against a genuine duplicate marked the LINK for deletion: `--delete
        // --permanent` unlinked a pointer, reported the row UNLINKED with its
        // bytes struck out, and left both copies of the video on disk.
        let dir = tempfile::tempdir().unwrap();
        let original = touch(dir.path(), "episode.mkv");
        fs::hard_link(&original, dir.path().join("hardlink.mkv")).unwrap();
        std::os::unix::fs::symlink(&original, dir.path().join("pointer.mkv")).unwrap();

        let include = vec![dir.path().to_string_lossy().to_string()];
        let stats = RunStats::default();
        let found = collected(&sources(&include, &[], &extensions_of(&["mkv"])), &stats);

        assert_eq!(found.len(), 1, "one inode, one entry");
        assert!(
            !found[0].ends_with("pointer.mkv"),
            "a link must not be the name the library carries: {}",
            found[0]
        );
        assert_eq!(stats.skipped_alias.count(), 2, "the other two names");
        assert!(!stats.had_problems(), "an alias is a skip, not a problem");
    }

    #[test]
    fn test_a_real_name_outranks_a_link_whichever_the_walk_meets_first() {
        // The end-to-end test above cannot reach this: readdir decides the
        // order there, so the arm where the link is seen FIRST and displaced by
        // a real name arriving later may simply never run. Both orders are
        // driven here, over one identity, and each has to end the same way.
        let name = |p: &str| ScannedFile {
            path: p.to_string(),
            size: 5,
            mtime: 1,
            mtime_nsec: 0,
        };
        let id = (7, 42);

        for (first, second, link_first) in [
            ("/lib/pointer.mkv", "/lib/episode.mkv", true),
            ("/lib/episode.mkv", "/lib/pointer.mkv", false),
        ] {
            let stats = RunStats::default();
            let mut into = Collecting {
                seen_inodes: HashMap::new(),
                found: Vec::new(),
                walked: Vec::new(),
            };

            into.claim(id, name(first), link_first, &stats);
            into.claim(id, name(second), !link_first, &stats);

            assert_eq!(into.found.len(), 1, "one inode, one entry");
            assert_eq!(
                into.found[0].path, "/lib/episode.mkv",
                "the real name survives whether it arrived first or second"
            );
            assert_eq!(
                stats.skipped_alias.count(),
                1,
                "one name skipped per collision, however it was resolved"
            );
        }

        // And nothing else about the preference: two equally real names are
        // equally good, so the first still wins and does not churn.
        let stats = RunStats::default();
        let mut into = Collecting {
            seen_inodes: HashMap::new(),
            found: Vec::new(),
            walked: Vec::new(),
        };
        into.claim(id, name("/lib/episode.mkv"), false, &stats);
        into.claim(id, name("/lib/hardlink.mkv"), false, &stats);
        into.claim((7, 43), name("/lib/other.mkv"), true, &stats);
        assert_eq!(into.found[0].path, "/lib/episode.mkv");
        assert_eq!(
            into.found[1].path, "/lib/other.mkv",
            "a lone link is still a file worth scanning -- there is no real name to prefer"
        );
        assert_eq!(stats.skipped_alias.count(), 1);
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
            reaches(&library(&request, &stats), &root.join("dupes")),
            Some(root.clone())
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
        assert_eq!(reaches(&library, &root.join("dupes")), Some(root.clone()));
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

        assert!(reaches(&library(&request, &stats), &root).is_none());
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

        assert!(reaches(&library(&request, &stats), &dest).is_none());
    }

    /// The guard asks about the roots, and under `--follow-symlinks` the roots
    /// are not the whole of what the walk reaches. `lib/link -> store` puts
    /// every file in `store` inside the scan while `store` sits under no root
    /// the user typed, so `dest.starts_with(root)` was false and the run moved
    /// its duplicates straight back into its own input -- the next run found
    /// them under `lib/link/...` and moved them one level deeper.
    #[test]
    fn test_a_destination_reachable_only_through_a_symlink_is_still_a_loop() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let lib = root.join("lib");
        touch(&lib, "a.mkv");
        let store = root.join("store");
        fs::create_dir_all(&store).unwrap();
        std::os::unix::fs::symlink(&store, lib.join("link")).unwrap();

        let include = vec![lib.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &[], &extensions);
        request.recursive = true;
        request.follow_symlinks = true;
        let stats = RunStats::default();

        assert_eq!(
            reaches(&library(&request, &stats), &store),
            Some(store.clone()),
            "reached through the link, so a moved file comes back"
        );

        // And the refusal names the link, which is the only path the user can
        // go and do anything about. Built on the target alone it read "store is
        // inside store, which this run scans".
        let library = library(&request, &stats);
        let said = library.walk_reaches(&store).unwrap().to_string();
        assert!(
            said.contains(&lib.join("link").to_string_lossy().to_string()),
            "{}",
            said
        );
    }

    /// The destination one level inside the link's target is the same loop:
    /// the landing paths are under it either way.
    #[test]
    fn test_a_destination_under_a_symlinked_folder_is_a_loop_too() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let lib = root.join("lib");
        touch(&lib, "a.mkv");
        let store = root.join("store");
        fs::create_dir_all(store.join("dupes")).unwrap();
        std::os::unix::fs::symlink(&store, lib.join("link")).unwrap();

        let include = vec![lib.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &[], &extensions);
        request.recursive = true;
        request.follow_symlinks = true;
        let stats = RunStats::default();

        assert_eq!(
            reaches(&library(&request, &stats), &store.join("dupes")),
            Some(store.clone())
        );
    }

    /// Without the flag the walk does not go through the link, so the
    /// destination really is outside the library and the move really does get
    /// the file out of it. The guard must not refuse this.
    #[test]
    fn test_the_same_destination_is_fine_when_links_are_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let lib = root.join("lib");
        touch(&lib, "a.mkv");
        let store = root.join("store");
        fs::create_dir_all(&store).unwrap();
        std::os::unix::fs::symlink(&store, lib.join("link")).unwrap();

        let include = vec![lib.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &[], &extensions);
        request.recursive = true;
        let stats = RunStats::default();

        assert!(reaches(&library(&request, &stats), &store).is_none());
    }

    /// Excluding the destination is still the way out, and it has to keep
    /// working when the route to it is a link: the excluded folder is pruned,
    /// so it is not one of the places this walk reaches.
    #[test]
    fn test_excluding_a_symlinked_destination_is_still_the_way_out() {
        let dir = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        let lib = root.join("lib");
        touch(&lib, "a.mkv");
        let store = root.join("store");
        fs::create_dir_all(&store).unwrap();
        std::os::unix::fs::symlink(&store, lib.join("link")).unwrap();

        let include = vec![lib.to_string_lossy().to_string()];
        let exclude = vec![store.to_string_lossy().to_string()];
        let extensions = extensions_of(&["mkv"]);
        let mut request = sources(&include, &exclude, &extensions);
        request.recursive = true;
        request.follow_symlinks = true;
        let stats = RunStats::default();

        assert!(reaches(&library(&request, &stats), &store).is_none());
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

        assert!(reaches(&library(&request, &stats), &root.join("dupes")).is_none());
    }

    #[test]
    fn test_an_empty_extension_list_fails_before_anything_blocks_on_a_pipe() {
        let stats = RunStats::default();
        let include = vec!["-".to_string()];
        let empty: Vec<String> = vec![];

        assert!(collect(&sources(&include, &[], &empty), &stats).is_err());
    }
}