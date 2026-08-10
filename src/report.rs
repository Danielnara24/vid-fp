//! Acting on a report the user has already read.
//!
//! Every other mode of this tool decides for itself which files are redundant.
//! This one does not decide anything: it reads a CSV report an earlier run
//! wrote, takes the rows whose `action` column says DELETE, and disposes of
//! exactly those. The point is the rows the earlier run *could not* decide --
//! a codec standoff, a file that reached its group through a chain -- which are
//! flagged REVIEW precisely because a human has to choose, and which until now
//! had nowhere to record that choice but `rm`.
//!
//! So the report is expected to have been edited, and the edits are the input.
//! A REVIEW turned into a DELETE is acted on; a DELETE turned into a KEEP is
//! not. The tool asserts nothing about whether the resulting set still makes
//! sense -- it is not the author of this list any more -- and in particular it
//! does NOT require a group to keep a survivor. If you mark every copy of
//! something for deletion, that is what happens.
//!
//! Two things it does keep. The confirmation prompt is the same one the grouped
//! run shows, from the same count and byte total. And every file is re-checked
//! against the size the report recorded immediately before it is touched
//! (`export::dispose_one`), so a report written last week cannot remove a file
//! that has been replaced since -- the one guarantee that gets *more* valuable
//! the longer a report sits before it is acted on.
//!
//! Only the CSV is read. The JSON carries the same `action` field, but nobody
//! adjudicates a REVIEW by hand-editing a JSON tree, and a second parser would
//! be a second place for this to be wrong.

use anyhow::{anyhow, Context, Result};
use log::info;
use std::collections::HashSet;

use crate::confirm::{self, Target};
use crate::export::{self, Disposal, Fate};
use crate::stats::RunStats;
use crate::utils::{format_size, shutdown_requested};

/// The action word that means "act on this row". Compared case-insensitively
/// after trimming, because someone editing a column of capitals in a
/// spreadsheet will not reliably produce capitals.
const ACT_ON: &str = "delete";

/// Every value vid-fp itself writes into the action column.
///
/// Rows carrying one of these are understood and deliberately left alone.
/// Anything else is a word this tool did not write and does not recognise --
/// most likely a misspelt DELETE -- and silently doing nothing with it is how a
/// user comes away believing a file was removed. It is reported instead.
///
/// The past-tense ones matter as much as the rest: feeding back a report from a
/// run that already deleted things must be a no-op, not a second pass over
/// files that are already gone.
const KNOWN_ACTIONS: [&str; 9] = [
    "keep", "kept", "review", "delete", "deleted", "moved", "failed", "changed", "skipped",
];

/// A row that asked for its file to be disposed of.
#[derive(Debug)]
struct Marked {
    path: String,
    /// The file's length when the report was written. The staleness check is
    /// taken against this, so a row without a usable one is not actionable at
    /// all -- see `read`.
    size: u64,
}

/// What one report amounted to.
#[derive(Debug)]
struct Report {
    marked: Vec<Marked>,
    /// Data rows seen, however they were labelled.
    rows: usize,
}

/// The index of the column called `name`.
///
/// By name rather than by position on purpose, and it is what lets the column
/// layout be chosen for whoever reads it rather than for whatever last parsed
/// it: a report written by an older build, or one that has been through a
/// spreadsheet and come back reordered or padded with extra columns, replays
/// identically. Nothing here cares where the three columns it needs are
/// sitting, only that they exist.
fn column(headers: &csv::StringRecord, name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|h| h.trim().eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            anyhow!(
                "The report has no '{}' column. --from-report reads the CSV that -o <file>.csv \
                 writes; a .txt or .json report cannot be replayed.",
                name
            )
        })
}

/// Parse the report, keeping the rows marked DELETE.
///
/// Rows that cannot be understood are counted as problems and skipped rather
/// than aborting the read: a report is a list of independent decisions, and one
/// mangled line is no reason to refuse to act on the ninety-nine good ones. A
/// missing *column*, by contrast, means this is not a vid-fp CSV at all, and
/// that does abort -- proceeding would mean acting on nothing and saying so as
/// though the file had simply been empty.
fn read(path: &str, stats: &RunStats) -> Result<Report> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b';')
        // A hand-edited file can easily carry a row with a field too few or too
        // many. That is this function's problem to report, one row at a time,
        // not the parser's to die on.
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("Failed to open the report {}", path))?;

    let headers = rdr
        .headers()
        .with_context(|| format!("Failed to read the header row of {}", path))?
        .clone();

    let path_col = column(&headers, "full_path")?;
    let size_col = column(&headers, "size_bytes")?;
    let action_col = column(&headers, "action")?;

    let mut marked: Vec<Marked> = Vec::new();
    let mut rows = 0usize;
    // One file can only be disposed of once. A report vid-fp wrote never repeats
    // a path -- groups partition the files -- but one assembled by hand or
    // concatenated from several runs can, and the second attempt would be a
    // spurious FAILED against a file the first one correctly removed.
    let mut seen: HashSet<String> = HashSet::new();

    for (i, record) in rdr.records().enumerate() {
        // Line number as the user's editor counts them: the header is line 1.
        let line = i + 2;

        let record = match record {
            Ok(r) => r,
            Err(e) => {
                log::error!("{}: line {} could not be parsed: {}", path, line, e);
                stats.report_unusable.record(format!("{} line {}", path, line));
                continue;
            }
        };

        // csv yields a trailing empty record for a file ending in a blank line
        // only when it is not flexible; with flexible(true) a genuinely blank
        // line still arrives here, and it is not a row anybody wrote.
        if record.iter().all(|f| f.trim().is_empty()) {
            continue;
        }
        rows += 1;

        let action = record.get(action_col).unwrap_or("").trim();

        // An emptied cell is unambiguous in effect -- nothing happens to that
        // file -- so it is not worth complaining about. A report that arrives
        // with EVERY action blank is caught by the count `apply` prints.
        if action.is_empty() {
            continue;
        }

        let lower = action.to_ascii_lowercase();
        if !KNOWN_ACTIONS.contains(&lower.as_str()) {
            log::error!(
                "{}: line {} has an action of \"{}\", which is not one of {}. \
                 The file it names was left alone.",
                path,
                line,
                action,
                KNOWN_ACTIONS.join("/").to_uppercase()
            );
            stats
                .report_unusable
                .record(format!("{} line {}: action \"{}\"", path, line, action));
            continue;
        }

        if lower != ACT_ON {
            continue;
        }

        let file = record.get(path_col).unwrap_or("").trim();
        if file.is_empty() {
            log::error!("{}: line {} is marked DELETE but names no file.", path, line);
            stats
                .report_unusable
                .record(format!("{} line {}: no full_path", path, line));
            continue;
        }

        // No fallback if this will not parse. The recorded size is the entire
        // basis of the check that runs before the file is touched, and a
        // deletion carried out without it is a deletion with nothing behind it
        // -- so a row that lost its size is a row this mode declines to act on.
        // (A report round-tripped through a spreadsheet that rewrote the column
        // as 1.23E+09 lands here, which is why the value is quoted back.)
        let size = match record.get(size_col).unwrap_or("").trim().parse::<u64>() {
            Ok(n) => n,
            Err(_) => {
                log::error!(
                    "{}: line {} is marked DELETE but its size_bytes is \"{}\", which is not a \
                     byte count. {} was left alone: without the size it was measured at there is \
                     nothing to check it against before removing it.",
                    path,
                    line,
                    record.get(size_col).unwrap_or("").trim(),
                    file
                );
                stats
                    .report_unusable
                    .record(format!("{} line {}: unusable size_bytes", path, line));
                continue;
            }
        };

        if seen.insert(file.to_string()) {
            marked.push(Marked {
                path: file.to_string(),
                size,
            });
        }
    }

    Ok(Report { marked, rows })
}

/// Read `report_path` and dispose of everything it marks DELETE, returning the
/// paths that are no longer where they were.
///
/// Same contract as `export::output_results`: the returned list is the caller's
/// cue to drop those files' cached fingerprints, and it carries only what was
/// genuinely disposed of.
pub fn apply(
    report_path: &str,
    disposal: &Disposal,
    assume_yes: bool,
    stats: &RunStats,
) -> Result<Vec<String>> {
    let report = read(report_path, stats)?;

    info!(
        "Read {} row(s) from {}; {} marked DELETE.",
        report.rows,
        report_path,
        report.marked.len()
    );

    if report.marked.is_empty() {
        // Said out loud because the alternative is a run that prints a summary
        // full of zeroes and exits clean, which reads exactly like success.
        info!(
            "\nNothing in {} is marked DELETE, so no files were touched. Edit the action column \
             to DELETE on the rows you want acted on.",
            report_path
        );
        return Ok(Vec::new());
    }

    let targets: Vec<Target> = report
        .marked
        .iter()
        .map(|m| Target {
            path: &m.path,
            size: m.size,
        })
        .collect();

    if !confirm::approve(disposal, &targets, assume_yes, confirm::Decline::StopsHere) {
        info!("\nCancelled at the confirmation prompt; every file was left alone.");
        return Ok(Vec::new());
    }

    info!("\n========================================");
    info!("             RESULTS");
    info!("========================================\n");

    let mut removed_count = 0usize;
    let mut failed_count = 0usize;
    let mut changed_count = 0usize;
    let mut removed_bytes = 0u64;
    let mut deleted_paths: Vec<String> = Vec::new();

    for m in &report.marked {
        if shutdown_requested() {
            info!(
                "Interrupted: stopped after {} file(s); {} left untouched.",
                removed_count,
                report.marked.len() - removed_count - failed_count - changed_count
            );
            break;
        }

        let label = match export::dispose_one(&m.path, m.size, disposal, stats) {
            Fate::Done => {
                removed_count += 1;
                removed_bytes += m.size;
                deleted_paths.push(m.path.clone());
                disposal.done_label()
            }
            Fate::Changed => {
                changed_count += 1;
                "CHANGED"
            }
            Fate::Failed => {
                failed_count += 1;
                "FAILED"
            }
        };

        // Laid out like the grouped run's results table: outcome first as a
        // column, path last because it is the one field of unbounded width.
        info!("\t{:<8} {}, {}", format!("{},", label), format_size(m.size), m.path);
    }

    let mut summary = export::disposed_line(disposal, removed_count, removed_bytes);
    summary.push_str(&export::trouble_lines(failed_count, changed_count));
    info!("\n{}", summary);

    Ok(deleted_paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const HEADER: &str = "group;action;full_path;length;length_seconds;resolution;width;height;\
framerate;framerate_fps;codec;size;size_bytes;bitrate;bitrate_bps;quality;quality_bits_per_frame;\
shared_with;shared_seconds;shared_from;shared_to;shared_from_seconds;shared_to_seconds";

    /// A CSV row in the real column layout, with only the three fields this
    /// module reads filled in.
    ///
    /// Positions are looked up in HEADER rather than hard-coded, so a change to
    /// the column order shows up here as nothing at all -- which is the property
    /// `column()` gives the parser, and these tests should not be the one place
    /// that quietly stops exercising it.
    fn row(path: &str, size: u64, action: &str) -> String {
        let columns: Vec<&str> = HEADER.split(';').collect();
        let mut fields = vec![String::new(); columns.len()];
        let mut set = |name: &str, value: String| {
            fields[columns.iter().position(|c| *c == name).unwrap()] = value;
        };

        set("group", "group_1".to_string());
        set("size_bytes", size.to_string());
        set("full_path", path.to_string());
        set("action", action.to_string());
        fields.join(";")
    }

    fn write_report(dir: &tempfile::TempDir, body: &str) -> String {
        let path = dir.path().join("report.csv");
        std::fs::write(&path, body).unwrap();
        path.to_string_lossy().to_string()
    }

    /// A real file of `size` bytes, so the staleness check has something to
    /// agree with.
    fn make_file(dir: &tempfile::TempDir, name: &str, size: usize) -> String {
        let path = dir.path().join(name);
        std::fs::write(&path, vec![b'x'; size]).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn test_only_the_delete_rows_are_taken() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n{}\n",
            HEADER,
            row("/a.mkv", 10, "KEEP"),
            row("/b.mkv", 20, "DELETE"),
            row("/c.mkv", 30, "REVIEW"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.rows, 3);
        assert_eq!(report.marked.len(), 1);
        assert_eq!(report.marked[0].path, "/b.mkv");
        assert_eq!(report.marked[0].size, 20);
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_a_review_the_user_promoted_is_acted_on() {
        // The entire reason this mode exists: a REVIEW row is the tool refusing
        // to decide, and editing it is how the decision gets made.
        let dir = tempfile::tempdir().unwrap();
        let body = format!("{}\n{}\n", HEADER, row("/c.mkv", 30, "DELETE"));
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked.len(), 1);
        assert_eq!(report.marked[0].path, "/c.mkv");
    }

    #[test]
    fn test_a_group_may_be_emptied_completely() {
        // Deliberate: this mode does not re-impose the one-survivor rule. The
        // report's author is the authority, and a user who marks every copy has
        // said what they meant.
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n",
            HEADER,
            row("/a.mkv", 10, "DELETE"),
            row("/b.mkv", 20, "DELETE"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked.len(), 2);
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_case_and_padding_do_not_matter() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n",
            HEADER,
            row("/a.mkv", 10, " delete "),
            row("/b.mkv", 20, "Delete"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked.len(), 2, "a spreadsheet will not preserve capitals");
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_an_already_acted_report_is_a_no_op() {
        // Feeding back the report of a --delete run must not try again. Those
        // files are gone; DELETED is not DELETE.
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n{}\n{}\n",
            HEADER,
            row("/a.mkv", 10, "KEPT"),
            row("/b.mkv", 20, "DELETED"),
            row("/c.mkv", 30, "MOVED"),
            row("/d.mkv", 40, "CHANGED"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert!(report.marked.is_empty());
        assert_eq!(stats.report_unusable.count(), 0, "these are all words we wrote");
    }

    #[test]
    fn test_a_misspelt_action_is_reported_rather_than_ignored() {
        // The dangerous case: the user meant DELETE, the file survives, and
        // without this the run says nothing and exits 0.
        let dir = tempfile::tempdir().unwrap();
        let body = format!("{}\n{}\n", HEADER, row("/a.mkv", 10, "DELET"));
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert!(report.marked.is_empty());
        assert_eq!(stats.report_unusable.count(), 1);
        assert!(stats.had_problems(), "a row nobody acted on must fail the run");
    }

    #[test]
    fn test_an_emptied_action_is_left_alone_quietly() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!("{}\n{}\n", HEADER, row("/a.mkv", 10, ""));
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert!(report.marked.is_empty());
        assert_eq!(stats.report_unusable.count(), 0, "blank is unambiguous: do nothing");
    }

    #[test]
    fn test_a_delete_row_without_a_usable_size_is_refused() {
        // Acting on it would mean deleting with nothing to check the file
        // against, which is the one thing this mode still guarantees.
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n",
            HEADER,
            row("/a.mkv", 10, "DELETE").replace(";10;", ";1.23E+09;"),
            row("/b.mkv", 20, "DELETE"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked.len(), 1, "the good row still goes through");
        assert_eq!(report.marked[0].path, "/b.mkv");
        assert_eq!(stats.report_unusable.count(), 1);
    }

    #[test]
    fn test_a_repeated_path_is_disposed_of_once() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n",
            HEADER,
            row("/a.mkv", 10, "DELETE"),
            row("/a.mkv", 10, "DELETE"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked.len(), 1);
    }

    #[test]
    fn test_columns_are_found_by_name_not_by_position() {
        // A report that has been through a spreadsheet, or one written by a
        // build with a different column list.
        let dir = tempfile::tempdir().unwrap();
        let body = "action;full_path;size_bytes\nDELETE;/a.mkv;4096\nKEEP;/b.mkv;8192\n";
        let path = write_report(&dir, body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked.len(), 1);
        assert_eq!(report.marked[0].path, "/a.mkv");
        assert_eq!(report.marked[0].size, 4096);
    }

    #[test]
    fn test_a_file_that_is_not_a_report_is_refused_outright() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_report(&dir, "one;two;three\n1;2;3\n");

        let stats = RunStats::default();
        let err = read(&path, &stats).unwrap_err().to_string();

        assert!(err.contains("full_path"), "{}", err);
    }

    #[test]
    fn test_a_blank_line_is_not_a_row() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!("{}\n{}\n\n", HEADER, row("/a.mkv", 10, "DELETE"));
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.rows, 1);
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_apply_removes_exactly_the_marked_files() {
        let dir = tempfile::tempdir().unwrap();
        let keep = make_file(&dir, "keep.mkv", 100);
        let doomed = make_file(&dir, "doomed.mkv", 200);

        let body = format!(
            "{}\n{}\n{}\n",
            HEADER,
            row(&keep, 100, "KEEP"),
            row(&doomed, 200, "DELETE"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let gone = apply(&path, &Disposal::Permanent, true, &stats).unwrap();

        assert_eq!(gone, vec![doomed.clone()]);
        assert!(!PathBuf::from(&doomed).exists());
        assert!(PathBuf::from(&keep).exists());
        assert!(!stats.had_problems());
    }

    #[test]
    fn test_a_file_that_changed_since_the_report_is_left_alone() {
        // The guarantee that matters most here, because a report can sit for a
        // week before anyone acts on it.
        let dir = tempfile::tempdir().unwrap();
        let target = make_file(&dir, "target.mkv", 200);

        // The report remembers 200 bytes; the file is now something else.
        let body = format!("{}\n{}\n", HEADER, row(&target, 200, "DELETE"));
        let path = write_report(&dir, &body);
        std::fs::write(&target, vec![b'y'; 500]).unwrap();

        let stats = RunStats::default();
        let gone = apply(&path, &Disposal::Permanent, true, &stats).unwrap();

        assert!(gone.is_empty());
        assert!(PathBuf::from(&target).exists(), "it is not the file that was judged");
        assert_eq!(stats.delete_stale.count(), 1);
        assert!(stats.had_problems());
    }

    #[test]
    fn test_a_missing_file_is_a_failure_not_a_silent_pass() {
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n",
            HEADER,
            row(&dir.path().join("never.mkv").to_string_lossy(), 10, "DELETE"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let gone = apply(&path, &Disposal::Permanent, true, &stats).unwrap();

        assert!(gone.is_empty());
        assert_eq!(stats.delete_failed.count(), 1);
    }

    #[test]
    fn test_apply_can_move_instead_of_delete() {
        let dir = tempfile::tempdir().unwrap();
        let dest = tempfile::tempdir().unwrap();
        let doomed = make_file(&dir, "doomed.mkv", 200);

        let body = format!("{}\n{}\n", HEADER, row(&doomed, 200, "DELETE"));
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let disposal = Disposal::MoveTo(dest.path().to_path_buf());
        let gone = apply(&path, &disposal, true, &stats).unwrap();

        assert_eq!(gone.len(), 1);
        assert!(!PathBuf::from(&doomed).exists());

        let landed = dest
            .path()
            .join(PathBuf::from(&doomed).strip_prefix("/").unwrap());
        assert!(landed.exists(), "expected it under {}", landed.display());
    }

    #[test]
    fn test_a_report_with_nothing_marked_touches_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let keep = make_file(&dir, "keep.mkv", 100);
        let body = format!("{}\n{}\n", HEADER, row(&keep, 100, "KEEP"));
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let gone = apply(&path, &Disposal::Permanent, true, &stats).unwrap();

        assert!(gone.is_empty());
        assert!(PathBuf::from(&keep).exists());
        assert!(!stats.had_problems(), "a report with no DELETE rows is not an error");
    }
}
