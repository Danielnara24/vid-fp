//! `--last` and `--no-cache`, driven through the binary: the last scan's
//! report written again is the report the scan wrote, a run without a cache
//! keeps nothing, and `--last` touches no file.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// Two copies of one video and a cropped one, so there is a group with a
/// DELETE in it.
fn library(dir: &Path) -> PathBuf {
    let lib = dir.join("lib");
    std::fs::create_dir(&lib).unwrap();
    std::fs::copy(fixture("test_video.mp4"), lib.join("a.mp4")).unwrap();
    std::fs::copy(fixture("test_video_cover_art.mp4"), lib.join("b.mp4")).unwrap();
    std::fs::copy(fixture("test_video_letterboxed.mp4"), lib.join("c.mp4")).unwrap();
    lib
}

fn vid_fp(args: &[&str], cache_home: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vid-fp"))
        .args(args)
        .env("XDG_CACHE_HOME", cache_home)
        .output()
        .expect("the binary runs")
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// Every format, written again from the kept scan, is byte for byte the one
/// the scan wrote, elapsed time included.
#[test]
fn test_the_last_scan_written_again_is_the_report_it_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let lib = library(dir.path());
    let cache = dir.path().join("cache.redb");
    for ext in ["csv", "json", "txt"] {
        let first = dir.path().join(format!("scan.{}", ext));
        let out = vid_fp(&[s(&lib), "-q", "--cache", s(&cache), "-o", s(&first)], dir.path());
        assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
        let again = dir.path().join(format!("last.{}", ext));
        let out = vid_fp(&["--last", "-q", "--cache", s(&cache), "-o", s(&again)], dir.path());
        assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
        let (a, b) = (std::fs::read_to_string(&first).unwrap(), std::fs::read_to_string(&again).unwrap());
        assert!(a.contains("DELETE"), "the library makes a deletion to write again:\n{}", a);
        assert_eq!(a, b, "{}", ext);
    }
}

/// In another format than the scan wrote, and without asking for one at all.
#[test]
fn test_the_last_scan_can_be_written_in_another_format() {
    let dir = tempfile::tempdir().unwrap();
    let lib = library(dir.path());
    let cache = dir.path().join("cache.redb");
    let csv = dir.path().join("scan.csv");
    assert_eq!(vid_fp(&[s(&lib), "-q", "--cache", s(&cache), "-o", s(&csv)], dir.path()).status.code(), Some(0));
    let out = vid_fp(&["--last", "--cache", s(&cache), "-o", "-", "--format", "csv"], dir.path());
    assert_eq!(String::from_utf8_lossy(&out.stdout), std::fs::read_to_string(&csv).unwrap());
}

/// `--last` is report-only: no flag that touches a file is accepted with it.
#[test]
fn test_the_last_scan_touches_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let lib = library(dir.path());
    let cache = dir.path().join("cache.redb");
    assert_eq!(vid_fp(&[s(&lib), "-q", "--cache", s(&cache)], dir.path()).status.code(), Some(0));
    for extra in [vec!["--delete"], vec!["--move-to", s(dir.path())]] {
        let mut argv = vec!["--last", "--cache", s(&cache)];
        argv.extend(extra.iter());
        assert_ne!(vid_fp(&argv, dir.path()).status.code(), Some(0), "{:?}", extra);
    }
    assert_eq!(std::fs::read_dir(&lib).unwrap().count(), 3);
}

/// A run without a cache reads none, writes none and keeps no scan, so
/// `--last` has nothing to answer with and says so.
#[test]
fn test_a_run_without_a_cache_keeps_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let lib = library(dir.path());
    let home = dir.path().join("home-cache");
    let out = vid_fp(&[s(&lib), "-q", "--no-cache", "-o", s(&dir.path().join("r.csv"))], &home);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(std::fs::read_to_string(dir.path().join("r.csv")).unwrap().contains("DELETE"));
    let kept: Vec<_> = std::fs::read_dir(home.join("vid-fp")).map(|d| d.flatten().collect()).unwrap_or_default();
    assert!(kept.is_empty(), "nothing kept: {:?}", kept);

    let out = vid_fp(&["--last"], &home);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no scan has finished"));
}

/// A cache named beside `--no-cache` is only for `--clear-cache` to empty.
#[test]
fn test_a_cache_named_beside_no_cache_is_only_for_clearing() {
    let dir = tempfile::tempdir().unwrap();
    let lib = library(dir.path());
    let cache = dir.path().join("cache.redb");
    let out = vid_fp(&[s(&lib), "-q", "--no-cache", "--cache", s(&cache)], dir.path());
    assert_eq!(out.status.code(), Some(1));
    assert!(!cache.exists());

    assert_eq!(vid_fp(&[s(&lib), "-q", "--cache", s(&cache)], dir.path()).status.code(), Some(0));
    let out = vid_fp(&[s(&lib), "-q", "--no-cache", "--clear-cache", "--cache", s(&cache)], dir.path());
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let out = vid_fp(&[s(&lib), "--cache", s(&cache)], dir.path());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("Fingerprinting"),
        "the emptied cache answers nothing, so the next scan fingerprints again"
    );
}
