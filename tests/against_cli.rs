//! `--against`, driven through the binary: which answer each scanned file
//! gets, that `--list` prints exactly those paths on stdout, and that the
//! library is never touched however the run is armed.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn put(dir: &Path, name: &str, from: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::copy(fixture(from), &path).unwrap();
    path
}

/// A library of two unrelated videos, and an import folder holding a byte copy
/// of one, a clip of the other, and something the library lacks. The clip is a
/// higher resolution than the capture it was cut from (160x120 against 96x64),
/// so it is the REVIEW a scan would make of it: deleting it would lose the
/// sharper picture.
struct Layout {
    _dir: tempfile::TempDir,
    root: PathBuf,
    library: PathBuf,
    import: PathBuf,
}

fn layout() -> Layout {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let library = root.join("library");
    let import = library.join("import");
    std::fs::create_dir_all(&import).unwrap();
    put(&library, "episode.mp4", "test_video.mp4");
    put(&library, "capture.ts", "test_video_capture.ts");
    put(&import, "copy of episode.mp4", "test_video.mp4");
    put(&import, "a clip.ts", "test_video_segment.ts");
    put(&import, "something new.mkv", "test_video_open_gop.mkv");
    Layout { _dir: dir, root, library, import }
}

fn vid_fp(layout: &Layout, args: &[&str]) -> std::process::Output {
    let out = Command::new(env!("CARGO_BIN_EXE_vid-fp"))
        .args(["--cache", layout.root.join("cache.redb").to_str().unwrap()])
        .args(args)
        .output()
        .expect("the binary runs");
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    out
}

fn listed(out: &std::process::Output, end: char) -> Vec<String> {
    let mut paths: Vec<String> = String::from_utf8(out.stdout.clone())
        .unwrap()
        .split(end)
        .filter(|p| !p.is_empty())
        .map(|p| Path::new(p).file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    paths.sort();
    paths
}

#[test]
fn each_scanned_file_is_answered_and_only_repeats_are_acted_on() {
    let l = layout();
    let import = l.import.to_str().unwrap();
    let library = l.library.to_str().unwrap();
    let report = l.root.join("report.csv");

    // The import folder is inside the library, so both walks reach it; the
    // scan holds it more closely, so its files are the ones answered.
    let out = vid_fp(&l, &[import, "--against", library, "-r", "--list", "repeat", "-o", report.to_str().unwrap()]);
    assert_eq!(listed(&out, '\n'), vec!["copy of episode.mp4"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Answering 3 scanned file(s) against a library of 2."), "{stderr}");
    assert!(stderr.contains("1 repeat(s), 0 better, 1 to review, 1 new."), "{stderr}");

    let csv = std::fs::read_to_string(&report).unwrap();
    let header: Vec<&str> = csv.lines().next().unwrap().split(';').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let action_of = |file: &str| {
        csv.lines()
            .skip(1)
            .map(|l| l.split(';').collect::<Vec<_>>())
            .find(|r| r[col("full_path")].ends_with(file))
            .map(|r| (r[col("action")].to_string(), r[col("matched_with")].to_string(), r[col("identical")].to_string()))
            .unwrap()
    };
    let (action, with, identical) = action_of("/copy of episode.mp4");
    assert_eq!((action.as_str(), identical.as_str()), ("DELETE", "true"));
    assert!(with.ends_with("/library/episode.mp4"));
    let (action, with, _) = action_of("/a clip.ts");
    assert_eq!(action, "REVIEW");
    assert!(with.ends_with("/library/capture.ts"), "the library file it matched is named");
    // New: kept, and no library file named beside it.
    assert_eq!(action_of("/something new.mkv"), ("KEEP".to_string(), String::new(), String::new()));
    // Library rows only ever read KEEP.
    assert_eq!(action_of("/library/episode.mp4").0, "KEEP");

    // NUL-separated with -0, and any answer can be asked for.
    let out = vid_fp(&l, &[import, "--against", library, "-r", "--list", "new,review", "-0"]);
    assert_eq!(listed(&out, '\0'), vec!["a clip.ts", "something new.mkv"]);

    // Armed: the repeats go, nothing else does.
    vid_fp(&l, &[import, "--against", library, "-r", "--delete", "--permanent"]);
    assert!(!l.import.join("copy of episode.mp4").exists());
    assert!(l.import.join("a clip.ts").exists(), "a REVIEW is never acted on");
    assert!(l.import.join("something new.mkv").exists());
    assert!(l.library.join("episode.mp4").exists());
    assert!(l.library.join("capture.ts").exists());
}

#[test]
fn a_scan_holding_the_library_answers_everything_but_it() {
    // The other nesting: the scan's folder holds the library, which then holds
    // its own files more closely and keeps them.
    let l = layout();
    let out = vid_fp(
        &l,
        &[l.root.to_str().unwrap(), "--against", l.library.to_str().unwrap(), "--list", "new"],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Not recursive, so the scan reaches nothing below the root but the cache.
    assert!(stderr.contains("against a library of 2."), "{stderr}");
    assert!(listed(&out, '\n').is_empty());
}

#[test]
fn a_report_on_stdout_cannot_share_it_with_the_list() {
    let l = layout();
    let out = Command::new(env!("CARGO_BIN_EXE_vid-fp"))
        .args([l.import.to_str().unwrap(), "--against", l.library.to_str().unwrap(), "--list", "new", "-o", "-"])
        .args(["--cache", l.root.join("cache.redb").to_str().unwrap()])
        .output()
        .unwrap();
    assert_ne!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--list prints to stdout"));
}
