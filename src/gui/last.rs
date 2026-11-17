//! The last finished scan, kept on disk so that closing the window does not
//! cost it, and the videos marked for the Trash beside it.
//!
//! One scan only, the one the results page shows: a scan that finishes with
//! results replaces it, and one that is cancelled or fails leaves it alone, as
//! it leaves the page alone.
//!
//! `$XDG_CACHE_HOME/vid-fp/last-scan/`, or `~/.cache/vid-fp/last-scan/`.

use crate::settings::paths;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use vid_fp::found::Found;

const FOUND: &str = "found.json";
const NOTES: &str = "notes.json";
const MARKS: &str = "marks.json";

/// What the window knows of a scan that the result does not say.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
pub struct Notes {
    /// It exited 2: something could not be read.
    pub problems: bool,
    /// How long it took, as the clock beside the bar said.
    pub took: String,
    /// When it finished, in seconds since 1970.
    pub finished: u64,
    /// When it started, in nanoseconds since 1970 (`changed_after`).
    pub started_ns: u64,
    /// Which scan this is, for the marks kept with it.
    pub id: u64,
    /// The scan log.
    pub log: String,
    /// The folders it was given, which the tree view starts from.
    #[serde(with = "paths")]
    pub roots: Vec<PathBuf>,
}

impl Notes {
    /// A file changed at or after this, in nanoseconds since 1970, is not the
    /// file the scan judged. The scan's start, less a second for a clock that
    /// counts change times coarser than the system's.
    pub fn changed_after(&self) -> u64 {
        self.started_ns.saturating_sub(1_000_000_000)
    }
}

/// The videos marked, and the scan they were marked in (`Notes::id`): marks
/// are read back only beside that scan.
#[derive(Serialize, Deserialize)]
struct Marks {
    scan: u64,
    #[serde(with = "paths")]
    paths: Vec<PathBuf>,
}

pub struct Last {
    pub found: Found,
    pub notes: Notes,
    /// Files the scan names that are no longer there.
    pub gone: HashSet<PathBuf>,
    /// The videos that were marked for the Trash.
    pub marks: Vec<PathBuf>,
}

/// Now, in nanoseconds since 1970.
pub fn now_nanos() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
}

/// Whether the file `md` describes is not the one the scan measured at
/// `scanned` bytes: another length, or changed (written, replaced, renamed,
/// chmod'ed) at or after `changed_after`. The change time is the kernel's,
/// which `cp -p` and `rsync -a` cannot put back the way they put back a
/// modification time, so a file replaced by another of the same length and
/// date is caught too.
pub fn changed_since(md: &std::fs::Metadata, scanned: u64, changed_after: Option<u64>) -> bool {
    use std::os::unix::fs::MetadataExt;
    if md.len() != scanned {
        return true;
    }
    let ctime = (md.ctime().max(0) as u64).saturating_mul(1_000_000_000).saturating_add(md.ctime_nsec().max(0) as u64);
    changed_after.is_some_and(|after| ctime >= after)
}

fn dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("vid-fp").join("last-scan"))
}

/// Keep `result`, a finished scan's, with `notes`, in place of the scan kept
/// before, and say whether it was. Failing to is not worth interrupting anyone
/// over: the results are on screen either way.
pub fn save(result: &Path, notes: &Notes) -> bool {
    dir().is_some_and(|dir| save_in(&dir, result, notes).is_ok())
}

/// Keep `paths` as the marks of scan `scan` (`Notes::id`), if that is the
/// scan kept. Written beside it and renamed over the last list, so a window
/// closed mid-write leaves the list before.
pub fn save_marks(scan: u64, paths: Vec<PathBuf>) {
    if let Some(dir) = dir() {
        let _ = save_marks_in(&dir, scan, paths);
    }
}

fn save_marks_in(dir: &Path, scan: u64, paths: Vec<PathBuf>) -> std::io::Result<()> {
    let notes: Notes = serde_json::from_slice(&std::fs::read(dir.join(NOTES))?).map_err(std::io::Error::other)?;
    if notes.id != scan {
        return Ok(());
    }
    let new = dir.join(format!("{MARKS}.{}.new", std::process::id()));
    std::fs::write(&new, serde_json::to_vec(&Marks { scan, paths }).map_err(std::io::Error::other)?)?;
    std::fs::rename(&new, dir.join(MARKS)).inspect_err(|_| {
        let _ = std::fs::remove_file(&new);
    })
}

/// A new folder is made whole beside the old and swapped in, so a window
/// closed half way leaves one scan or the other, never half of each.
fn save_in(dir: &Path, result: &Path, notes: &Notes) -> std::io::Result<()> {
    let parent = dir.parent().ok_or_else(|| std::io::Error::other("no parent"))?;
    std::fs::create_dir_all(parent)?;
    let new = parent.join(format!("last-scan.{}.new", std::process::id()));
    let _ = std::fs::remove_dir_all(&new);
    std::fs::create_dir(&new)?;
    let made = (|| {
        std::fs::copy(result, new.join(FOUND))?;
        std::fs::write(new.join(NOTES), serde_json::to_vec(notes).map_err(std::io::Error::other)?)
    })();
    if let Err(e) = made {
        let _ = std::fs::remove_dir_all(&new);
        return Err(e);
    }
    let old = parent.join(format!("last-scan.{}.old", std::process::id()));
    let _ = std::fs::remove_dir_all(&old);
    let had = dir.exists() && std::fs::rename(dir, &old).is_ok();
    std::fs::rename(&new, dir).inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&new);
        if had {
            let _ = std::fs::rename(&old, dir);
        }
    })?;
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
}

/// Whether a scan is kept, without reading it: whether to open the window on
/// the results page.
pub fn exists() -> bool {
    dir().is_some_and(|d| d.join(NOTES).is_file())
}

/// The scan kept last time, if there is one that can still be read. One that
/// cannot (kept by a build that wrote another shape) is deleted, so that it is
/// not tried on every start.
pub fn load() -> Option<Last> {
    load_in(&dir()?)
}

fn load_in(dir: &Path) -> Option<Last> {
    let notes = std::fs::read(dir.join(NOTES)).ok()?;
    let read = Found::load(&dir.join(FOUND)).ok().zip(serde_json::from_slice::<Notes>(&notes).ok());
    let Some((found, notes)) = read else {
        let _ = std::fs::remove_dir_all(dir);
        return None;
    };
    // Gone as the Trash counts it: not found. A file that cannot be looked at
    // for another reason is still shown, as the scan left it.
    let gone = found
        .files
        .iter()
        .map(|f| PathBuf::from(&f.path))
        .filter(|p| matches!(std::fs::symlink_metadata(p), Err(e) if e.kind() == std::io::ErrorKind::NotFound))
        .collect();
    let marks = std::fs::read(dir.join(MARKS))
        .ok()
        .and_then(|b| serde_json::from_slice::<Marks>(&b).ok())
        .filter(|m| m.scan == notes.id)
        .map_or_else(Vec::new, |m| m.paths);
    Some(Last { found, notes, gone, marks })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vid_fp::found::{Action, File, Group, Member};

    fn file(path: &Path) -> File {
        File {
            path: path.display().to_string(),
            size: 1,
            duration: 10.0,
            width: 640,
            height: 480,
            codec: "h264".into(),
            frame_rate: 25.0,
            bitrate: 1,
            quality: 1,
            samples: 5,
            verdicts: vec![vid_fp::found::Verdict { action: Action::Keep, lost_to: Vec::new() }; 4],
        }
    }

    #[test]
    fn a_kept_scan_is_read_back_and_the_next_replaces_it() {
        let root = tempfile::tempdir().unwrap();
        let root = root.path();
        let (a, b) = (root.join("a.mp4"), root.join("b.mp4"));
        std::fs::write(&a, b"x").unwrap();
        let result = |name: &str, files: &[&Path]| {
            let members = (0..files.len()).map(|i| Member { file: i, matched_with: None, identical: false, matched_seconds: None, matched_from: None, matched_to: None, compared: Vec::new() }).collect();
            let found = Found { found: 7, compared: 7, files: files.iter().map(|p| file(p)).collect(), groups: vec![Group { members }], ..Found::default() };
            let p = root.join(name);
            found.save(&p).unwrap();
            p
        };
        let paths = |l: &Last| l.found.files.iter().map(|f| PathBuf::from(&f.path)).collect::<Vec<_>>();
        let dir = root.join("cache").join("vid-fp").join("last-scan");
        let notes = |id: u64, log: &str| Notes { problems: true, took: "1:02".into(), finished: 5, started_ns: 0, id, log: log.into(), roots: vec![root.to_path_buf()] };

        save_in(&dir, &result("one.json", &[&a, &b]), &notes(5, "first")).unwrap();
        let last = load_in(&dir).unwrap();
        assert_eq!(last.found.found, 7);
        assert_eq!(paths(&last).len(), 2);
        assert!(last.notes.problems);
        assert_eq!(last.notes.log, "first");
        assert_eq!(last.notes.roots, [root.to_path_buf()]);
        // `b.mp4` was never there: the page is told it is gone.
        assert_eq!(last.gone, HashSet::from([b.clone()]));
        assert!(last.marks.is_empty());

        // Marks are kept for the scan they were made in, and only for it.
        save_marks_in(&dir, 5, vec![a.clone()]).unwrap();
        assert_eq!(load_in(&dir).unwrap().marks, vec![a.clone()]);
        save_marks_in(&dir, 6, vec![b.clone()]).unwrap();
        assert_eq!(load_in(&dir).unwrap().marks, vec![a.clone()]);

        save_in(&dir, &result("two.json", &[&b, &a]), &notes(7, "second")).unwrap();
        let last = load_in(&dir).unwrap();
        assert_eq!(paths(&last), [b.clone(), a.clone()]);
        assert_eq!(last.notes.log, "second");
        // The next scan starts with none.
        assert!(last.marks.is_empty());
        // Nothing of the save is left beside it.
        let left: Vec<_> = std::fs::read_dir(dir.parent().unwrap()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left, vec![std::ffi::OsString::from("last-scan")]);

        // One that will not read is dropped rather than tried again.
        std::fs::write(dir.join(FOUND), b"{").unwrap();
        assert!(load_in(&dir).is_none());
        assert!(!dir.exists());
    }

    /// A file of another length, or one changed since the scan started, is
    /// not the file the scan judged; one untouched since is.
    #[test]
    fn a_file_changed_since_the_scan_is_told_apart() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("v.mp4");
        std::fs::write(&p, b"12345").unwrap();
        let md = std::fs::metadata(&p).unwrap();
        let later = now_nanos() + 5_000_000_000;
        assert!(!changed_since(&md, 5, Some(later)), "untouched since the scan began");
        assert!(changed_since(&md, 6, Some(later)), "another length");
        assert!(changed_since(&md, 5, Some(0)), "changed after the scan began");
        assert!(!changed_since(&md, 5, None), "with no start known, the length alone");
    }
}
