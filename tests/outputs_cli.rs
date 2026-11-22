//! The three files a run writes -- the cache, the report and the log -- driven
//! through the binary: none may be another, and a cache that cannot be read is
//! something `--clear-cache` recovers from rather than a wall.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn library(dir: &Path) -> PathBuf {
    let lib = dir.join("lib");
    std::fs::create_dir(&lib).unwrap();
    std::fs::copy(fixture("test_video.mp4"), lib.join("a.mp4")).unwrap();
    std::fs::copy(fixture("test_video.mp4"), lib.join("b.mp4")).unwrap();
    lib
}

fn vid_fp(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vid-fp")).args(args).output().expect("the binary runs")
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn test_no_two_of_cache_report_and_log_may_be_one_file() {
    let dir = tempfile::tempdir().unwrap();
    let lib = library(dir.path());
    let cache = dir.path().join("cache.redb");
    let first = vid_fp(&[s(&lib), "-q", "--cache", s(&cache)]);
    assert_eq!(first.status.code(), Some(0));
    let before = std::fs::read(&cache).unwrap();

    let other = dir.path().join("out.txt");
    for args in [
        vec!["-o", s(&cache)],
        vec!["--log-file", s(&cache)],
        vec!["-o", s(&other), "--log-file", s(&other)],
    ] {
        let mut argv = vec![s(&lib), "-q", "--cache", s(&cache)];
        argv.extend(&args);
        let out = vid_fp(&argv);
        assert_eq!(out.status.code(), Some(1), "{:?} is refused", args);
        assert!(String::from_utf8_lossy(&out.stderr).contains("Choose another file name"));
        assert_eq!(std::fs::read(&cache).unwrap(), before, "{:?} left the cache alone", args);
        assert!(!other.exists());
    }

    // A directory given as --cache names the file inside it, and that is the
    // file compared.
    let out = vid_fp(&[s(&lib), "-q", "--cache", s(dir.path()), "-o", s(&dir.path().join("fingerprints.redb"))]);
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn test_a_cache_that_cannot_be_read_is_set_aside_by_clear_cache() {
    let dir = tempfile::tempdir().unwrap();
    let lib = library(dir.path());
    let cache = dir.path().join("cache.redb");
    std::fs::write(&cache, b"this is a report, not a database\n").unwrap();

    let out = vid_fp(&[s(&lib), "-q", "--cache", s(&cache)]);
    assert_eq!(out.status.code(), Some(1));
    let said = String::from_utf8_lossy(&out.stderr);
    assert!(said.contains("--clear-cache"), "the refusal names the way out: {said}");
    assert_eq!(std::fs::read(&cache).unwrap(), b"this is a report, not a database\n");

    let out = vid_fp(&[s(&lib), "-q", "--cache", s(&cache), "--clear-cache"]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let aside = dir.path().join("cache.redb.damaged");
    assert_eq!(std::fs::read(&aside).unwrap(), b"this is a report, not a database\n", "moved, not deleted");

    // And the new cache is a cache.
    let out = vid_fp(&[s(&lib), "--cache", s(&cache)]);
    assert_eq!(out.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out.stderr).contains("already cached"));
}
