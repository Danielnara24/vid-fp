//! The desktop app shows what the command line reports.
//!
//! Both binaries are run on one folder of videos: `vid-fp`, writing its JSON
//! report, and the window's scan (`vid-fp-gui --worker`), writing the result
//! the window reads. Every group, every file in it, each file's KEEP, DELETE or
//! REVIEW and the figures its row shows must be the same, at more than one
//! setting, and under every `--priority`: one window scan carries the labels
//! of all four, and each must be what `vid-fp -k <it>` reports. The two share the code that decides (`export::verdicts`), so this
//! is the guard on what each of them does around it: the window's result
//! being built from the same groups, fingerprints and links the report is.
//!
//! Only with the `gui` feature, which is what builds the window.
#![cfg(feature = "gui")]

use serde_json::Value;
use std::path::Path;
use std::process::Command;

/// The fixtures, and a byte copy of one, in a folder of their own: two
/// groups, with KEEP, DELETE and REVIEW among them, and a file that cannot be
/// fingerprinted, which makes both runs exit 2.
fn library() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir("tests/fixtures").unwrap() {
        let p = entry.unwrap().path();
        if matches!(p.extension().and_then(|e| e.to_str()), Some("mp4" | "ts")) {
            std::fs::copy(&p, dir.path().join(p.file_name().unwrap())).unwrap();
        }
    }
    std::fs::copy("tests/fixtures/test_video.mp4", dir.path().join("copy of test_video.mp4")).unwrap();
    dir
}

/// One row as either side states it: the action, and what the row shows.
#[derive(Debug, PartialEq)]
struct Row {
    path: String,
    action: String,
    length: Option<f64>,
    size: u64,
    resolution: (u64, u64),
    samples: u64,
    matched_with: Option<String>,
    identical: bool,
    matched_seconds: Option<f64>,
}

fn round(v: Option<f64>) -> Option<f64> {
    v.map(|v| (v * 100.0).round() / 100.0)
}

/// The report's groups, row by row.
fn from_report(report: &Value) -> Vec<Vec<Row>> {
    report["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            g["files"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| Row {
                    path: f["full_path"].as_str().unwrap().to_string(),
                    action: f["action"].as_str().unwrap().to_string(),
                    length: f["length_seconds"].as_f64(),
                    size: f["size_bytes"].as_u64().unwrap(),
                    resolution: (f["width"].as_u64().unwrap(), f["height"].as_u64().unwrap()),
                    samples: f["samples"].as_u64().unwrap(),
                    matched_with: f["matched_with"].as_str().map(str::to_string),
                    // null when the row has no link, which the window states as false.
                    identical: f["identical"].as_bool().unwrap_or(false),
                    matched_seconds: f["matched_seconds"].as_f64(),
                })
                .collect()
        })
        .collect()
}

/// The window's result, row by row, in the report's terms, as ranking
/// `ranking` (its place in `Priority::ALL`) labels it.
fn from_window(found: &Value, ranking: usize) -> Vec<Vec<Row>> {
    let files = found["files"].as_array().unwrap();
    let path = |i: &Value| files[i.as_u64().unwrap() as usize]["path"].as_str().unwrap().to_string();
    found["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            g["members"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    let f = &files[m["file"].as_u64().unwrap() as usize];
                    let duration = f["duration"].as_f64().unwrap();
                    Row {
                        path: f["path"].as_str().unwrap().to_string(),
                        action: f["verdicts"][ranking]["action"].as_str().unwrap().to_uppercase(),
                        length: round((duration > 0.0).then_some(duration)),
                        size: f["size"].as_u64().unwrap(),
                        resolution: (f["width"].as_u64().unwrap(), f["height"].as_u64().unwrap()),
                        samples: f["samples"].as_u64().unwrap(),
                        matched_with: (!m["matched_with"].is_null()).then(|| path(&m["matched_with"])),
                        identical: m["identical"].as_bool().unwrap(),
                        matched_seconds: round(m["matched_seconds"].as_f64()),
                    }
                })
                .collect()
        })
        .collect()
}

fn run(cmd: &mut Command) -> std::process::Output {
    let out = cmd.output().expect("the binary runs");
    // 2: a fixture is featureless on purpose, and both runs say so.
    assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
    out
}

#[test]
fn the_window_shows_what_the_command_line_reports() {
    let lib = library();
    let dir = lib.path().to_str().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    for settings in [&[][..], &["-d", "10"][..], &["-d", "10", "-p", "60"][..], &["--min-duration", "2"][..]] {
        let result = scratch.path().join("result.json");
        // Each binary a cache of its own, so neither reads what the other
        // wrote.
        let _ = std::fs::remove_file(scratch.path().join("a.redb"));
        let _ = std::fs::remove_file(scratch.path().join("b.redb"));
        let out = run(Command::new(env!("CARGO_BIN_EXE_vid-fp-gui"))
            .args(["--worker", result.to_str().unwrap(), "vid-fp"])
            .args(settings)
            .args(["--cache", scratch.path().join("b.redb").to_str().unwrap(), "--", dir]));

        let found: Value = serde_json::from_slice(&std::fs::read(&result).unwrap()).unwrap();
        let mut labellings = std::collections::HashSet::new();
        for (ranking, name) in ["length", "resolution", "quality", "size"].into_iter().enumerate() {
            let report = scratch.path().join(format!("report-{name}.json"));
            run(Command::new(env!("CARGO_BIN_EXE_vid-fp")).args(settings).args([
                "-k",
                name,
                "--cache",
                scratch.path().join("a.redb").to_str().unwrap(),
                "-o",
                report.to_str().unwrap(),
                dir,
            ]));
            let report: Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
            let (cli, window) = (from_report(&report), from_window(&found, ranking));
            assert!(!cli.is_empty() || settings.contains(&"60"), "{settings:?}: the fixtures group");
            assert_eq!(window, cli, "{settings:?} -k {name}");
            // `copy of test_video.mp4` is the fixture's bytes again, so the
            // flag is compared on a row that carries it, not only on falses.
            // `--min-duration 2` drops the fixture and its copy with it.
            assert!(
                settings.contains(&"--min-duration") || cli.iter().flatten().any(|r| r.identical),
                "{settings:?}: the copy is reported identical"
            );
            labellings.insert(format!("{:?}", cli.iter().flatten().map(|r| &r.action).collect::<Vec<_>>()));
        }
        // What the window's Compare shows: every member has its strongest
        // link's frames, first, and the reference's when that is another
        // video; each pair inside the footage it says matched.
        for g in found["groups"].as_array().unwrap() {
            let members = g["members"].as_array().unwrap();
            let reference = &members[0]["file"];
            for m in members {
                let compared = m["compared"].as_array().unwrap();
                if !m["matched_with"].is_null() {
                    assert_eq!(compared[0]["with"], m["matched_with"], "{settings:?}: the best match comes first");
                }
                if &m["file"] != reference {
                    assert!(compared.iter().any(|c| &c["with"] == reference), "{settings:?}: compared with the reference too");
                }
                for c in compared {
                    let pairs = c["pairs"].as_array().unwrap();
                    assert!(!pairs.is_empty(), "{settings:?}: a measured link matched frames");
                    assert!(c["matched_frames"].as_u64().unwrap() >= pairs.len() as u64);
                    let inside = |t: u64, key: &str| c[key].as_array().unwrap().iter().any(|st| st[0].as_u64().unwrap() <= t && t < st[1].as_u64().unwrap());
                    for p in pairs {
                        assert!(inside(p[0].as_u64().unwrap(), "stretches") && inside(p[1].as_u64().unwrap(), "other_stretches"), "{settings:?}: {p} is matched footage");
                    }
                }
            }
        }
        // Not four copies of one answer: on these fixtures, at least one
        // setting has rankings that disagree.
        if settings.is_empty() {
            assert!(labellings.len() > 1, "the rankings label the fixtures alike, so this test cannot tell them apart");
        }

        // And the window is told everything a terminal would be, as lines it
        // can read: the log and where the bar stands, but not the listing,
        // which is its results page.
        let stdout = String::from_utf8(out.stdout).unwrap();
        for line in stdout.lines() {
            let v: Value = serde_json::from_str(line).unwrap_or_else(|_| panic!("not a line the window reads: {line:?}"));
            assert!(v.get("l").is_some() || v.get("p").is_some(), "{line}");
        }
        assert!(stdout.contains("Total groups found"), "the summary is said");
        assert!(!stdout.contains("group_1:"), "the listing is the window's page, not its log");
        assert!(String::from_utf8_lossy(&out.stderr).contains("could not be fingerprinted"), "the problem summary stays on stderr");
    }
}

/// A scan that finds nothing to compare still leaves the window a result,
/// saying how many videos it found.
#[test]
fn a_scan_with_nothing_to_compare_leaves_a_result() {
    let empty = tempfile::tempdir().unwrap();
    let one = tempfile::tempdir().unwrap();
    std::fs::copy("tests/fixtures/test_video.mp4", one.path().join("only.mp4")).unwrap();
    let scratch = tempfile::tempdir().unwrap();
    for (dir, found) in [(empty.path(), 0), (one.path(), 1)] {
        let result = scratch.path().join("result.json");
        let out = Command::new(env!("CARGO_BIN_EXE_vid-fp-gui"))
            .args(["--worker", result.to_str().unwrap(), "vid-fp", "--cache", scratch.path().join("c.redb").to_str().unwrap(), "--"])
            .arg(dir)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
        let v: Value = serde_json::from_slice(&std::fs::read(&result).unwrap()).unwrap();
        assert_eq!(v["found"], found, "{}", dir.display());
        assert!(v["groups"].as_array().unwrap().is_empty());
        std::fs::remove_file(&result).unwrap();
    }
}

/// A command line the scan refuses fails the worker with vid-fp's own
/// message, which the window shows, and leaves no result behind.
#[test]
fn a_refused_command_line_says_why() {
    let scratch = tempfile::tempdir().unwrap();
    let result = scratch.path().join("result.json");
    let out = Command::new(env!("CARGO_BIN_EXE_vid-fp-gui"))
        .args(["--worker", result.to_str().unwrap(), "vid-fp", "-d", "40", "--"])
        .arg(Path::new("tests/fixtures"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--hamming-distance must be between 0 and 32"));
    assert!(!result.exists());
}
