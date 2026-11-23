//! What `--exclude` leaves out: folders and files by their canonical path, and
//! patterns of a name or of a path.
//!
//! **An `--exclude` holding `*`, `?` or `[` that names no path is a pattern**
//! (`Pattern`). With no `/` it is of a folder's or a file's NAME, met anywhere
//! below a scan root (`-e '*.part'`, `-e '@eaDir'`). With one it is of a PATH:
//! made absolute against the current folder, its leading plain part
//! canonicalized, and matched against the canonical path of what the walk
//! meets and every folder above it -- `*` within one name, `**` across any
//! number of them (`-e '/media/videos/**/proxies'`). A pattern used to be an
//! exclude path that would not resolve, which excluded nothing and was
//! counted as a problem.
//!
//! The matcher is img-fp's, byte for byte in what it accepts: `*` any run of
//! bytes within a name, `?` one byte, `[...]` one of a set (`[!...]` or
//! `[^...]` none of it, `a-z` a range), and `**` as a whole name any number of
//! names. It runs in linear time: a star is retried only from the last one met.

use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/// The resolved `--exclude` list.
#[derive(Clone, Default, Debug)]
pub struct Excludes {
    /// Canonical paths: a folder covers everything under it.
    pub paths: Vec<PathBuf>,
    /// Patterns of a path, matched against a canonical path and its folders.
    patterns: Vec<Pattern>,
    /// Patterns of a name, matched against the name of what a walk meets.
    names: Vec<Pattern>,
}

impl Excludes {
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty() && self.patterns.is_empty() && self.names.is_empty()
    }

    /// Take a pattern. `false` when `raw` is not one, so the caller can say
    /// that the path it names would not resolve.
    pub fn add_pattern(&mut self, raw: &Path) -> bool {
        if !Pattern::is_one(raw) {
            return false;
        }
        let pattern = Pattern::of(raw);
        if pattern.of_name {
            self.names.push(pattern);
        } else {
            self.patterns.push(pattern);
        }
        true
    }

    /// Whether `path` is under an excluded path, or it or a folder above it
    /// is what a path pattern describes. Component-wise, so `-e clips/take`
    /// is not `clips/take2.mkv`.
    pub fn covers(&self, path: &Path) -> bool {
        self.paths.iter().any(|ex| path.starts_with(ex))
            || (!self.patterns.is_empty() && path.ancestors().any(|a| self.patterns.iter().any(|p| p.matches(a))))
    }

    /// Whether a name pattern describes `path`'s own name. Asked of what a
    /// walk meets, and of a path named to scan; never of the folders above a
    /// scan root, which the request put there.
    pub fn names(&self, path: &Path) -> bool {
        !self.names.is_empty()
            && path.file_name().is_some_and(|n| self.names.iter().any(|p| p.matches(Path::new(n))))
    }

    /// Whether a walk of `root` would pass over `path` (which lies under it):
    /// it is covered, or one of its names below `root` is a name pattern's.
    pub fn spares_below(&self, root: &Path, path: &Path) -> bool {
        if self.covers(path) {
            return true;
        }
        let Ok(below) = path.strip_prefix(root) else { return false };
        let mut at = root.to_path_buf();
        below.components().any(|c| {
            at.push(c);
            self.names(&at)
        })
    }
}

/// Whether an `--exclude` entry is a pattern when it names no path: it holds
/// `*`, `?` or `[`. For the window, which shows the two apart.
pub fn is_exclude_pattern(p: &Path) -> bool {
    Pattern::is_one(p)
}

/// What the window makes of an exclusion typed by hand: the `--exclude` entry
/// to pass, `None` for nothing typed, or why it is not one (img-fp's rule).
///
/// **A window has no current folder to resolve against**, where a shell does:
/// a relative `videos/**/proxies` would be read against wherever the window
/// was started, which excludes nothing and says so only as a problem after
/// the scan. So a path, plain or a pattern, is a whole one, or starts `~/`
/// (`home`, as a shell expands it); and a pattern of names alone (no `/`:
/// `*.part`, `@eaDi?`) is taken as typed, since it is matched below every
/// folder scanned. A bare name with no wildcard is refused rather than
/// guessed at: on the command line it is a path.
pub fn typed_exclude(text: &str, home: Option<&Path>) -> Result<Option<PathBuf>, &'static str> {
    let t = text.trim();
    if t.is_empty() {
        return Ok(None);
    }
    let p = match (t.strip_prefix('~'), home) {
        (Some(""), Some(h)) => h.to_path_buf(),
        (Some(rest), Some(h)) if rest.starts_with('/') => h.join(rest.trim_start_matches('/')),
        _ => PathBuf::from(t),
    };
    if p.is_absolute() {
        return Ok(Some(p));
    }
    if is_exclude_pattern(&p) && !t.contains('/') {
        return Ok(Some(p));
    }
    Err(if is_exclude_pattern(&p) {
        "A pattern holding a / needs a whole path, such as ~/Videos/**/proxies."
    } else {
        "A folder or file needs its whole path. To leave out a name wherever it is, use * or ?, such as *.part."
    })
}

/// One `--exclude` pattern.
#[derive(Clone, Debug)]
struct Pattern {
    /// The pattern's names; for a path pattern, from the root.
    parts: Vec<Vec<u8>>,
    /// Of a name alone: it held no `/`.
    of_name: bool,
}

impl Pattern {
    /// Whether `p` is meant as a pattern: it holds a wildcard.
    fn is_one(p: &Path) -> bool {
        p.as_os_str().as_bytes().iter().any(|b| matches!(b, b'*' | b'?' | b'['))
    }

    fn of(p: &Path) -> Pattern {
        let bytes = p.as_os_str().as_bytes();
        if !bytes.contains(&b'/') {
            return Pattern { parts: vec![bytes.to_vec()], of_name: true };
        }
        // Absolute, and its plain lead canonical where it resolves, so that it
        // is compared with canonical paths in their own terms.
        let p = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
        let names: Vec<&std::ffi::OsStr> = p.iter().collect();
        let plain = names.iter().take_while(|n| !Pattern::is_one(Path::new(n))).count();
        let lead: PathBuf = names[..plain].iter().collect();
        let lead = std::fs::canonicalize(&lead).unwrap_or(lead);
        let mut parts: Vec<Vec<u8>> = lead.iter().map(|n| n.as_bytes().to_vec()).collect();
        parts.extend(names[plain..].iter().map(|n| n.as_bytes().to_vec()));
        Pattern { parts, of_name: false }
    }

    /// Whether `path` (a name, for a name pattern; else an absolute path) is
    /// what the pattern says, whole.
    fn matches(&self, path: &Path) -> bool {
        let names: Vec<&[u8]> = path.iter().map(|n| n.as_bytes()).collect();
        let pat: Vec<&[u8]> = self.parts.iter().map(Vec::as_slice).collect();
        names_match(&pat, &names)
    }
}

/// Names against a path pattern's names, `**` any run of names.
fn names_match(pat: &[&[u8]], names: &[&[u8]]) -> bool {
    stars_match(pat.len(), names.len(), |i| pat[i] == b"**", |i, j| name_matches(pat[i], names[j]).then_some(i + 1))
}

/// One name against one name's pattern.
///
/// A star is tried only from the last one met: whatever an earlier star could
/// take instead, the last can as well, since a star takes any run. Trying
/// every split point at every star is exponential in the stars when nothing
/// matches.
fn name_matches(p: &[u8], n: &[u8]) -> bool {
    stars_match(p.len(), n.len(), |i| p[i] == b'*', |i, j| one_byte(p, i, n[j]))
}

/// The one matcher under both: a pattern of `len_p` tokens, of which `star(i)`
/// takes any run and the rest one each (`one(i, j)`: whether token `i` takes
/// item `j`, and where the pattern goes on), against `len_n` items.
fn stars_match(
    len_p: usize,
    len_n: usize,
    star: impl Fn(usize) -> bool,
    one: impl Fn(usize, usize) -> Option<usize>,
) -> bool {
    let (mut i, mut j) = (0, 0);
    // After the last star met: where the pattern goes on, and the item it
    // was last taken up to.
    let mut back: Option<(usize, usize)> = None;
    loop {
        if i < len_p && star(i) {
            i += 1;
            back = Some((i, j));
            continue;
        }
        if j < len_n {
            if i < len_p {
                if let Some(next) = one(i, j) {
                    i = next;
                    j += 1;
                    continue;
                }
            }
        } else if i == len_p {
            return true;
        }
        match back {
            Some((bi, bj)) if bj < len_n => {
                back = Some((bi, bj + 1));
                (i, j) = (bi, bj + 1);
            }
            _ => return false,
        }
    }
}

/// Whether the token at `p[i]` (not a star) takes byte `c`, and where the
/// pattern goes on.
fn one_byte(p: &[u8], i: usize, c: u8) -> Option<usize> {
    match p[i] {
        b'?' => Some(i + 1),
        b'[' => match set_end(&p[i + 1..]) {
            Some(end) => in_set(&p[i + 1..i + 1 + end], c).then_some(i + end + 2),
            // An unclosed `[` is itself.
            None => (c == b'[').then_some(i + 1),
        },
        b => (b == c).then_some(i + 1),
    }
}

/// Where a set's `]` is, after its first member (`[]a]` holds `]`).
fn set_end(s: &[u8]) -> Option<usize> {
    let skip = usize::from(matches!(s.first(), Some(b'!' | b'^')));
    (skip + 1..s.len()).find(|&i| s[i] == b']').filter(|_| s.len() > skip)
}

fn in_set(set: &[u8], c: u8) -> bool {
    let (negated, set) = match set.split_first() {
        Some((b'!' | b'^', rest)) => (true, rest),
        _ => (false, set),
    };
    let mut hit = false;
    let mut i = 0;
    while i < set.len() {
        if i + 2 < set.len() && set[i + 1] == b'-' {
            hit |= (set[i]..=set[i + 2]).contains(&c);
            i += 3;
        } else {
            hit |= set[i] == c;
            i += 1;
        }
    }
    hit != negated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(p: &str, n: &str) -> bool {
        name_matches(p.as_bytes(), n.as_bytes())
    }

    /// What the window may pass as `--exclude`: whole paths, `~/` ones, and
    /// patterns of names; never a path the worker would resolve against a
    /// folder nobody chose.
    #[test]
    fn test_a_typed_exclusion_is_a_whole_path_or_a_pattern_of_names() {
        let home = Path::new("/home/u");
        let ok = |t: &str| typed_exclude(t, Some(home)).unwrap().unwrap();
        assert_eq!(ok("*.part"), PathBuf::from("*.part"));
        assert_eq!(ok("  @eaDi?  "), PathBuf::from("@eaDi?"), "trimmed");
        assert_eq!(ok("/srv/videos/**/proxies"), PathBuf::from("/srv/videos/**/proxies"));
        assert_eq!(ok("/srv/videos/edits"), PathBuf::from("/srv/videos/edits"));
        assert_eq!(ok("~/Videos/**/proxies"), PathBuf::from("/home/u/Videos/**/proxies"));
        assert_eq!(ok("~"), PathBuf::from("/home/u"));
        assert_eq!(typed_exclude("   ", Some(home)), Ok(None));
        assert!(typed_exclude("videos/**/proxies", Some(home)).is_err(), "relative to nothing");
        assert!(typed_exclude("@eaDir", Some(home)).is_err(), "a bare name is a relative path");
        assert!(typed_exclude("~/x", None).is_err(), "no home to expand");
        assert!(typed_exclude("~bob/x", Some(home)).is_err(), "another user's home is not expanded");
        // What it hands on is what the walk reads as the same kind of entry.
        assert!(is_exclude_pattern(&ok("*.part")) && !is_exclude_pattern(&ok("/srv/videos/edits")));
    }

    #[test]
    fn test_a_name_pattern_reads_stars_marks_and_sets() {
        assert!(name("*.part", "film.mkv.part"));
        assert!(!name("*.part", "film.mkv"));
        assert!(name("take?.mkv", "take3.mkv"));
        assert!(!name("take?.mkv", "take10.mkv"));
        assert!(name("[a-c]*", "beta"));
        assert!(!name("[!a-c]*", "beta"));
        assert!(name("[]x]", "]"));
        assert!(name("a[b", "a[b"), "an unclosed set is itself");
        assert!(name("*", ""));
    }

    /// Nothing that fails to match takes time exponential in the stars.
    #[test]
    fn test_many_stars_against_a_name_that_does_not_match_finish() {
        let pattern = "*a".repeat(30) + "*b";
        assert!(!name(&pattern, &"a".repeat(200)));
    }

    #[test]
    fn test_a_path_pattern_matches_across_names_with_a_double_star() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(dir.path()).unwrap();
        let mut ex = Excludes::default();
        assert!(ex.add_pattern(&root.join("lib/**/proxies")));
        assert!(ex.covers(&root.join("lib/proxies/a.mkv")));
        assert!(ex.covers(&root.join("lib/x/y/proxies/a.mkv")));
        assert!(!ex.covers(&root.join("lib/x/proxies2/a.mkv")));
        assert!(!ex.covers(&root.join("other/proxies/a.mkv")));
    }

    /// The pattern's plain lead is canonicalized, so a pattern written through
    /// a symlinked folder still matches the canonical paths a walk compares.
    #[test]
    fn test_a_path_pattern_written_through_a_link_matches_the_real_path() {
        let dir = tempfile::tempdir().unwrap();
        let real = std::fs::canonicalize(dir.path()).unwrap().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let mut ex = Excludes::default();
        assert!(ex.add_pattern(&link.join("*.part")));
        assert!(ex.covers(&real.join("a.part")));
    }

    #[test]
    fn test_a_name_pattern_is_asked_of_names_below_the_root_only() {
        let mut ex = Excludes::default();
        assert!(ex.add_pattern(Path::new("@*")));
        let root = Path::new("/lib/@home");
        assert!(!ex.spares_below(root, Path::new("/lib/@home/a/b.mkv")), "the root's own name is the request's");
        assert!(ex.spares_below(root, Path::new("/lib/@home/@eaDir/b.mkv")));
        assert!(!ex.add_pattern(Path::new("plain/path")), "no wildcard, no pattern");
    }
}
