//! Acting on a report the user has already read.
//!
//! Every other mode of this tool decides for itself which files are redundant.
//! This one does not decide anything: it reads a CSV report an earlier run
//! wrote, takes the rows whose `action` column says DELETE, and disposes of
//! exactly those. The point is the rows the earlier run *could not* decide --
//! a codec standoff, a copy that leads on one metric while another leads on a
//! different one -- which are flagged REVIEW precisely because a human has to
//! choose, and which until now had nowhere to record that choice but `rm`.
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
//! Both formats `-o` can write a decision into are read: the CSV, and the JSON
//! that is the richer of the two. What is deliberately NOT duplicated is the
//! judgement. A format reader here does one thing -- pull three cells out of one
//! row and say where that row was -- and every question about what those cells
//! MEAN (is this a word we wrote? is that a byte count? has this path already
//! been claimed?) is answered once, in `Rows::consider`, for rows of every
//! format. A second parser is only a second place to be wrong if it is allowed
//! to decide anything, and this one is not.
//!
//! `.txt` remains unreadable, and that is a property of the format rather than
//! an omission: it is a rendering for a human, with no size beside each path to
//! check the file against before removing it.

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

/// One row of a report, in whatever format it arrived in.
///
/// Three fields, because three is everything the destructive step needs: which
/// file, how big it was when the decision was taken, and what the decision was.
/// `at` is only ever quoted back to the user, and each format spells it the way
/// that format is navigated -- a line number for the CSV, a path through the
/// tree for the JSON -- because a complaint about a row you cannot find is a
/// complaint you cannot act on.
struct Row<'a> {
    at: String,
    action: &'a str,
    /// Deliberately untrimmed. See `Rows::consider`.
    file: &'a str,
    size: &'a str,
}

/// The rows of one report, and the single place that decides what any of them
/// mean.
///
/// Every format reader feeds this. That is the whole arrangement: the readers
/// know how to find three cells, and nothing else, so adding a format cannot
/// add a way for DELETE to be interpreted.
#[derive(Default)]
struct Rows {
    marked: Vec<Marked>,
    rows: usize,
    /// One file can only be disposed of once. A report vid-fp wrote never
    /// repeats a path within its own decisions, but one assembled by hand or
    /// concatenated from several runs can, and the second attempt would be a
    /// spurious FAILED against a file the first one correctly removed.
    seen: HashSet<String>,
}

impl Rows {
    /// Take one row, and keep it if it asks for its file to be disposed of.
    ///
    /// Nothing here aborts. A report is a list of independent decisions, and one
    /// mangled row is no reason to refuse to act on the ninety-nine good ones --
    /// it is counted as a problem (exit 2) and named, which is the treatment
    /// every "we did less than we were asked" case in this tool gets.
    fn consider(&mut self, row: Row, report: &str, stats: &RunStats) {
        self.rows += 1;

        let action = row.action.trim();

        // An emptied cell is unambiguous in effect -- nothing happens to that
        // file -- so it is not worth complaining about. A report that arrives
        // with EVERY action blank is caught by the count `apply` prints.
        if action.is_empty() {
            return;
        }

        let lower = action.to_ascii_lowercase();
        if !KNOWN_ACTIONS.contains(&lower.as_str()) {
            log::error!(
                target: crate::stats::COUNTED,
                "{}: {} has an action of \"{}\", which is not one of {}. \
                 The file it names was left alone.",
                report,
                row.at,
                action,
                KNOWN_ACTIONS.join("/").to_uppercase()
            );
            stats
                .report_unusable
                .record(format!("{} {}: action \"{}\"", report, row.at, action));
            return;
        }

        if lower != ACT_ON {
            return;
        }

        // The path is deliberately NOT trimmed, unlike every other cell read
        // here. Leading and trailing spaces are legal in a Linux filename, and
        // both writers emit one verbatim -- a trailing space does not make a CSV
        // field need quoting, and JSON says exactly what it is given -- so
        // trimming means the tool cannot even replay its own report. What it
        // does instead is look up a DIFFERENT path: `dupe.mkv ` becomes
        // `dupe.mkv`, and if that neighbour happens to be the size the row
        // recorded, the staleness check passes and the wrong file is deleted.
        // The check cannot catch it, because it is taken against whatever path
        // survived the trim.
        //
        // Whitespace is still not a file name, so a cell holding only spaces is
        // treated as the empty one it plainly is.
        if row.file.trim().is_empty() {
            log::error!(target: crate::stats::COUNTED, "{}: {} is marked DELETE but names no file.", report, row.at);
            stats
                .report_unusable
                .record(format!("{} {}: no full_path", report, row.at));
            return;
        }

        // No fallback if this will not parse. The recorded size is the entire
        // basis of the check that runs before the file is touched, and a
        // deletion carried out without it is a deletion with nothing behind it
        // -- so a row that lost its size is a row this mode declines to act on.
        // (A report round-tripped through a spreadsheet that rewrote the column
        // as 1.23E+09 lands here, which is why the value is quoted back.)
        let size = match row.size.trim().parse::<u64>() {
            Ok(n) => n,
            Err(_) => {
                log::error!(
                target: crate::stats::COUNTED,
                    "{}: {} is marked DELETE but its size_bytes is \"{}\", which is not a \
                     byte count. {} was left alone: without the size it was measured at there is \
                     nothing to check it against before removing it.",
                    report,
                    row.at,
                    row.size.trim(),
                    row.file
                );
                stats
                    .report_unusable
                    .record(format!("{} {}: unusable size_bytes", report, row.at));
                return;
            }
        };

        if self.seen.insert(row.file.to_string()) {
            self.marked.push(Marked {
                path: row.file.to_string(),
                size,
            });
        }
    }

    fn finish(self) -> Report {
        Report {
            marked: self.marked,
            rows: self.rows,
        }
    }
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
                 writes, or the JSON that -o <file>.json writes; a .txt report cannot be \
                 replayed, because it records no size to check each file against before \
                 removing it.",
                name
            )
        })
}

/// Parse the report, keeping the rows marked DELETE.
///
/// Which reader runs is decided by the file's first character, not by its name.
/// The extension used to decide, which worked only for as long as `--output`
/// decided by the extension too; `--format` broke the symmetry, and a report
/// written as `-o dupes.bak --format json` would have been handed to the CSV
/// reader and refused for want of columns it was never going to have. A report
/// gets renamed (`dupes.txt`, `dupes.bak`, no extension at all after a
/// download), so the file has to answer for itself -- which each format already
/// does further in, the JSON by needing a `results` array and the CSV by needing
/// its three columns. This just asks the same question one character earlier, so
/// that a wrong guess produces the complaint that fits the file.
fn read(path: &str, stats: &RunStats) -> Result<Report> {
    if looks_like_json(path) {
        read_json(path, stats)
    } else {
        read_csv(path, stats)
    }
}

/// Whether the first thing in the file is the start of a JSON value.
///
/// `[` counts as well as `{`, though a report is always an object: an array
/// gets JSON's complaint about a missing `results` key rather than the CSV's
/// about a missing column, and that is the one a user can act on.
///
/// A file that cannot be opened or read is left to the reader it falls through
/// to, which reports the failure with the path in it. There is nothing to be
/// gained by saying it twice, in two different sentences, on the way past.
fn looks_like_json(path: &str) -> bool {
    use std::io::Read;

    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };

    // Enough for any run of leading whitespace a formatter or an editor would
    // leave; a file that is all whitespace this far in is not a report of
    // either kind.
    let mut head = [0u8; 64];
    let Ok(n) = file.read(&mut head) else {
        return false;
    };

    head[..n]
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|b| *b == b'{' || *b == b'[')
}

/// The CSV report, located by column name.
///
/// A missing *column* means this is not a vid-fp CSV at all, and that does abort
/// -- proceeding would mean acting on nothing and saying so as though the file
/// had simply been empty.
fn read_csv(path: &str, stats: &RunStats) -> Result<Report> {
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

    let mut rows = Rows::default();

    for (i, record) in rdr.records().enumerate() {
        // Line number as the user's editor counts them: the header is line 1.
        let line = i + 2;

        let record = match record {
            Ok(r) => r,
            Err(e) => {
                log::error!(target: crate::stats::COUNTED, "{}: line {} could not be parsed: {}", path, line, e);
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

        rows.consider(
            Row {
                at: format!("line {}", line),
                action: record.get(action_col).unwrap_or(""),
                file: record.get(path_col).unwrap_or(""),
                size: record.get(size_col).unwrap_or(""),
            },
            path,
            stats,
        );
    }

    Ok(rows.finish())
}

/// The JSON report.
///
/// Same three fields, found by key instead of by column, at
/// `results[].files[]` -- which is where the writer puts one object per file
/// with its own `action` beside its own `full_path` and `size_bytes`. The
/// per-link `matches` array nested under each of those carries a `full_path`
/// too and is deliberately never looked at: it names the file this row matched,
/// not the file this row decides.
///
/// The shape is checked before the rows are, for the same reason the CSV checks
/// for its columns: a JSON file with no `results` array is not a report, and
/// walking it would find nothing to do and report that as a clean run over an
/// empty list.
fn read_json(path: &str, stats: &RunStats) -> Result<Report> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to open the report {}", path))?;

    // The whole tree at once. A report is written in one `fs::write` and read
    // back the same way; streaming it would buy nothing but a second shape to
    // get wrong.
    let root: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", path))?;

    let groups = root.get("results").and_then(|r| r.as_array()).ok_or_else(|| {
        anyhow!(
            "The report {} has no 'results' array. --from-report reads the .csv or .json that \
             -o writes; a .txt report cannot be replayed, because it records no size to check \
             each file against before removing it.",
            path
        )
    })?;

    let mut rows = Rows::default();

    for (g, group) in groups.iter().enumerate() {
        // A group is located by its own `group` key when it has one, because
        // that is the name the report prints and the user reads. Falling back to
        // the index keeps a hand-assembled tree navigable.
        let group_name = group
            .get("group")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("results[{}]", g));

        let Some(files) = group.get("files").and_then(|f| f.as_array()) else {
            log::error!(target: crate::stats::COUNTED, "{}: {} has no 'files' array; nothing in it was acted on.", path, group_name);
            stats
                .report_unusable
                .record(format!("{} {}: no files array", path, group_name));
            continue;
        };

        for (f, file) in files.iter().enumerate() {
            let at = format!("{} file {}", group_name, f + 1);

            // Every cell is read as the text it would have been in the CSV, so
            // the shared rules apply unchanged: a number becomes its own
            // digits, a string is taken as written, and anything absent is the
            // empty cell it amounts to. `size_bytes` written as 4096.0 or as
            // null therefore lands in the same "that is not a byte count"
            // refusal a spreadsheet's 1.23E+09 does.
            let action = cell(file.get("action"));
            let filename = cell(file.get("full_path"));
            let size = cell(file.get("size_bytes"));

            rows.consider(
                Row {
                    at,
                    action: &action,
                    file: &filename,
                    size: &size,
                },
                path,
                stats,
            );
        }
    }

    Ok(rows.finish())
}

/// One JSON value as the cell it stands for.
///
/// A string keeps every byte of itself -- a path's leading space is part of its
/// name and `Rows::consider` is relying on still having it. Anything else is
/// rendered the way JSON writes it, which is what makes a number usable as a
/// size and makes a nonsense value (an array, `true`) show up as the nonsense it
/// is in the message that reports it, rather than as a blank.
fn cell(value: Option<&serde_json::Value>) -> String {
    match value {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
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
        // Named as "the action" rather than "the action column", because the
        // JSON has no columns and a message that describes a file the user is
        // not looking at is a message that reads as being about something else.
        info!(
            "\nNothing in {} is marked DELETE, so no files were touched. Set the action to \
             DELETE on the rows you want acted on.",
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
framerate_fps;codec;size;size_bytes;bitrate;bitrate_bps;quality;quality_bits_per_frame;\
matched_with;samples;matched_seconds;matched_from;matched_to;matched_from_seconds;\
matched_to_seconds";

    /// The layout shipped before the link columns went directional: a
    /// `framerate` column that no longer exists, and `shared_*` where the
    /// current build writes `matched_*`. Nothing this module reads moved, so a
    /// report written by such a build still replays -- which is the promise
    /// `column()` makes and the reason the layout was free to change.
    const LEGACY_HEADER: &str = "group;action;full_path;length;length_seconds;resolution;width;\
height;framerate;framerate_fps;codec;size;size_bytes;bitrate;bitrate_bps;quality;\
quality_bits_per_frame;shared_with;shared_seconds;shared_from;shared_to;shared_from_seconds;\
shared_to_seconds";

    /// A CSV row in the real column layout, with only the three fields this
    /// module reads filled in.
    ///
    /// Positions are looked up in HEADER rather than hard-coded, so a change to
    /// the column order shows up here as nothing at all -- which is the property
    /// `column()` gives the parser, and these tests should not be the one place
    /// that quietly stops exercising it.
    fn row(path: &str, size: u64, action: &str) -> String {
        row_in(HEADER, path, size, action)
    }

    /// The same, against whichever layout the caller is exercising.
    fn row_in(header: &str, path: &str, size: u64, action: &str) -> String {
        let columns: Vec<&str> = header.split(';').collect();
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

    /// One file object shaped exactly like the one `export.rs` writes: the
    /// row's own fields, then the per-link list nested underneath it. The
    /// nested entries carry a `full_path` of their own and no action, and this
    /// module must never mistake one for a decision.
    fn json_file(path: &str, size: serde_json::Value, action: &str) -> serde_json::Value {
        serde_json::json!({
            "action": action,
            "full_path": path,
            "length": "00:01:00",
            "size_bytes": size,
            "matched_with": "/somewhere/else.mkv",
            "matches": [
                { "full_path": "/somewhere/else.mkv", "matched_seconds": 60.0 }
            ],
        })
    }

    /// A whole JSON report, in the writer's own shape: a summary, then groups,
    /// each with its files.
    fn json_report(dir: &tempfile::TempDir, files: Vec<serde_json::Value>) -> String {
        let tree = serde_json::json!({
            "summary": { "total_groups": 1, "total_files_matched": files.len() },
            "results": [ { "group": "group_1", "files": files } ],
        });
        write_json(dir, "report.json", &tree.to_string())
    }

    fn write_json(dir: &tempfile::TempDir, name: &str, body: &str) -> String {
        let path = dir.path().join(name);
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
    fn test_a_report_from_a_build_with_the_old_link_columns_still_replays() {
        // The link columns were renamed (`shared_*` -> `matched_*`) and the
        // formatted `framerate` column dropped when the report went
        // directional. Neither is anything this module reads, and columns are
        // located by name, so a report an older build wrote is still a valid
        // instruction to delete something -- which is the whole reason the
        // layout could be changed at all.
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n",
            LEGACY_HEADER,
            row_in(LEGACY_HEADER, "/a.mkv", 10, "KEEP"),
            row_in(LEGACY_HEADER, "/b.mkv", 20, "DELETE"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.rows, 2);
        assert_eq!(report.marked.len(), 1);
        assert_eq!(report.marked[0].path, "/b.mkv");
        assert_eq!(report.marked[0].size, 20);
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_a_path_keeps_the_whitespace_that_is_part_of_its_name() {
        // Leading and trailing spaces are legal in a Linux filename and the CSV
        // writer emits them verbatim, so trimming here meant the tool could not
        // replay its own report -- it looked up a DIFFERENT path. With a
        // same-sized neighbour beside it (`dupe.mkv` next to `dupe.mkv `) the
        // size check passes against the wrong file and the wrong file is
        // deleted, which is the one outcome this mode must never produce.
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n",
            HEADER,
            row("/videos/dupe.mkv ", 20, "DELETE"),
            row("/videos/ leading.mkv", 30, "DELETE"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked.len(), 2);
        assert_eq!(report.marked[0].path, "/videos/dupe.mkv ", "the trailing space is the name");
        assert_eq!(report.marked[1].path, "/videos/ leading.mkv");
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_a_cell_holding_only_whitespace_still_names_no_file() {
        // The other half of not trimming: spaces are part of a name, but a cell
        // that is nothing BUT spaces is the empty cell it plainly is, and acting
        // on it would mean disposing of a path made of blanks.
        let dir = tempfile::tempdir().unwrap();
        let body = format!("{}\n{}\n", HEADER, row("   ", 20, "DELETE"));
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert!(report.marked.is_empty());
        assert_eq!(stats.report_unusable.count(), 1);
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

    // --- the JSON report ------------------------------------------------------
    //
    // The rules being exercised are the CSV's rules; what these tests are really
    // asking is whether a second format reader can reach them without bringing
    // opinions of its own.

    #[test]
    fn test_a_json_report_is_read_by_the_same_rules_as_the_csv() {
        let dir = tempfile::tempdir().unwrap();
        let path = json_report(
            &dir,
            vec![
                json_file("/a.mkv", serde_json::json!(10), "KEEP"),
                json_file("/b.mkv", serde_json::json!(20), "DELETE"),
                json_file("/c.mkv", serde_json::json!(30), "REVIEW"),
            ],
        );

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.rows, 3);
        assert_eq!(report.marked.len(), 1);
        assert_eq!(report.marked[0].path, "/b.mkv");
        assert_eq!(report.marked[0].size, 20);
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_the_nested_match_list_is_not_a_row() {
        // Every file object carries a `matches` array whose entries have a
        // `full_path` of their own. Those name the file this row was measured
        // against -- reading one as a decision would delete a file nobody
        // marked, and the file it named is the one that WON.
        let dir = tempfile::tempdir().unwrap();
        let path = json_report(&dir, vec![json_file("/b.mkv", serde_json::json!(20), "DELETE")]);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.rows, 1, "one file object is one row");
        assert_eq!(report.marked.len(), 1);
        assert_eq!(report.marked[0].path, "/b.mkv");
    }

    #[test]
    fn test_a_json_size_is_taken_from_a_number_or_from_a_string() {
        // The writer emits a bare number. An editor, a jq pipeline, or anything
        // that round-tripped the file through a spreadsheet may hand back the
        // same figure quoted, and that is the same figure.
        let dir = tempfile::tempdir().unwrap();
        let path = json_report(
            &dir,
            vec![
                json_file("/a.mkv", serde_json::json!(4096), "DELETE"),
                json_file("/b.mkv", serde_json::json!("8192"), "DELETE"),
            ],
        );

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked.len(), 2);
        assert_eq!(report.marked[0].size, 4096);
        assert_eq!(report.marked[1].size, 8192);
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_a_json_delete_row_without_a_usable_size_is_refused() {
        // Same rule as the CSV's, and for the same reason: the size is the only
        // thing the file is checked against before it is removed. A fractional
        // byte count is not one, and neither is a missing key.
        let dir = tempfile::tempdir().unwrap();
        let path = json_report(
            &dir,
            vec![
                json_file("/a.mkv", serde_json::json!(1.23e9), "DELETE"),
                json_file("/b.mkv", serde_json::Value::Null, "DELETE"),
                json_file("/c.mkv", serde_json::json!(20), "DELETE"),
            ],
        );

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked.len(), 1, "the good row still goes through");
        assert_eq!(report.marked[0].path, "/c.mkv");
        assert_eq!(stats.report_unusable.count(), 2);
        assert!(stats.had_problems());
    }

    #[test]
    fn test_a_json_path_keeps_the_whitespace_that_is_part_of_its_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = json_report(
            &dir,
            vec![json_file("/videos/dupe.mkv ", serde_json::json!(20), "DELETE")],
        );

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.marked[0].path, "/videos/dupe.mkv ", "the trailing space is the name");
    }

    #[test]
    fn test_a_misspelt_action_in_json_is_reported_rather_than_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = json_report(&dir, vec![json_file("/a.mkv", serde_json::json!(10), "DELET")]);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert!(report.marked.is_empty());
        assert_eq!(stats.report_unusable.count(), 1);
        assert!(stats.had_problems(), "a row nobody acted on must fail the run");
    }

    #[test]
    fn test_a_json_row_is_located_by_the_group_name_the_report_prints() {
        // What a complaint has to say to be actionable: not "row 14" of a tree
        // nobody counts by row, but the group and position the file itself
        // shows.
        let dir = tempfile::tempdir().unwrap();
        let path = json_report(
            &dir,
            vec![
                json_file("/a.mkv", serde_json::json!(10), "KEEP"),
                json_file("/b.mkv", serde_json::json!(20), "DELET"),
            ],
        );

        let stats = RunStats::default();
        read(&path, &stats).unwrap();

        let complaints = stats.report_unusable.samples();
        assert_eq!(complaints.len(), 1);
        assert!(complaints[0].contains("group_1 file 2"), "{}", complaints[0]);
    }

    #[test]
    fn test_a_json_file_that_is_not_a_report_is_refused_outright() {
        // The counterpart of the CSV's missing-column abort. Walking it would
        // find no rows and report a clean run over an empty list, which reads
        // exactly like "nothing was marked".
        let dir = tempfile::tempdir().unwrap();
        let path = write_json(&dir, "other.json", r#"{"videos": [{"action": "DELETE"}]}"#);

        let stats = RunStats::default();
        let err = read(&path, &stats).unwrap_err().to_string();

        assert!(err.contains("results"), "{}", err);
    }

    #[test]
    fn test_json_that_will_not_parse_is_refused_rather_than_half_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_json(&dir, "broken.json", r#"{"results": [ "#);

        let stats = RunStats::default();
        let err = format!("{:#}", read(&path, &stats).unwrap_err());

        assert!(err.contains("not valid JSON"), "{}", err);
    }

    #[test]
    fn test_the_format_is_read_out_of_the_file_not_off_its_name() {
        // Both directions of the same rule, and the JSON half is the one
        // --format made reachable: `-o dupes.bak --format json` writes a report
        // no extension describes, and the run that replays it has only the
        // bytes to go on. Leading whitespace is skipped, because a file that
        // went through a formatter or an editor is still the report it was.
        let dir = tempfile::tempdir().unwrap();
        let tree = serde_json::json!({
            "results": [ { "group": "group_1", "files": [
                json_file("/b.mkv", serde_json::json!(20), "DELETE")
            ] } ]
        });

        let stats = RunStats::default();

        for name in ["REPORT.JSON", "dupes.bak", "dupes.txt", "dupes"] {
            let path = write_json(&dir, name, &format!("\n  {}", tree));
            let report = read(&path, &stats).unwrap();
            assert_eq!(report.marked.len(), 1, "{} is JSON whatever it is called", name);
        }

        // A report saved as dupes.txt by habit, or one a browser downloaded
        // without its extension, reads as the CSV it is for the same reason.
        let body = format!("{}\n{}\n", HEADER, row("/b.mkv", 20, "DELETE"));
        for name in ["dupes.bak", "dupes.txt", "dupes", "REPORT.CSV"] {
            let path = write_json(&dir, name, &body);
            let report = read(&path, &stats).unwrap();
            assert_eq!(report.marked.len(), 1, "{} is a CSV whatever it is called", name);
        }
    }

    #[test]
    fn test_json_that_is_not_a_report_complains_as_json() {
        // What the sniff buys beyond reading the right file: a JSON array named
        // .csv used to be refused for want of a column it could never have had.
        // The complaint a user can act on is the one about the shape of the
        // file they actually wrote.
        let dir = tempfile::tempdir().unwrap();
        let path = write_json(&dir, "wrong.csv", r#"[{"full_path": "/b.mkv"}]"#);

        let stats = RunStats::default();
        let err = format!("{:#}", read(&path, &stats).unwrap_err());

        assert!(err.contains("results"), "{}", err);
    }

    #[test]
    fn test_apply_removes_exactly_the_marked_files_from_a_json_report() {
        let dir = tempfile::tempdir().unwrap();
        let keep = make_file(&dir, "keep.mkv", 100);
        let doomed = make_file(&dir, "doomed.mkv", 200);

        let path = json_report(
            &dir,
            vec![
                json_file(&keep, serde_json::json!(100), "KEEP"),
                json_file(&doomed, serde_json::json!(200), "DELETE"),
            ],
        );

        let stats = RunStats::default();
        let gone = apply(&path, &Disposal::Permanent, true, &stats).unwrap();

        assert_eq!(gone, vec![doomed.clone()]);
        assert!(!PathBuf::from(&doomed).exists());
        assert!(PathBuf::from(&keep).exists());
        assert!(!stats.had_problems());
    }

    #[test]
    fn test_a_json_file_that_changed_since_the_report_is_left_alone() {
        // The staleness check is `dispose_one`'s, not the reader's, so this is
        // really asking whether the JSON path reaches it with a real size.
        let dir = tempfile::tempdir().unwrap();
        let target = make_file(&dir, "target.mkv", 200);

        let path = json_report(&dir, vec![json_file(&target, serde_json::json!(200), "DELETE")]);
        std::fs::write(&target, vec![b'y'; 500]).unwrap();

        let stats = RunStats::default();
        let gone = apply(&path, &Disposal::Permanent, true, &stats).unwrap();

        assert!(gone.is_empty());
        assert!(PathBuf::from(&target).exists(), "it is not the file that was judged");
        assert_eq!(stats.delete_stale.count(), 1);
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
