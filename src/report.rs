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
use std::path::{Component, Path};

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
const KNOWN_ACTIONS: [&str; 10] = [
    "keep", "kept", "review", "delete", "deleted", "unlinked", "moved", "failed", "changed",
    "skipped",
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

        // Every path this tool writes is canonical: rooted, and with no `..`
        // left in it. A path that is not is a path this run cannot answer for,
        // for two separate reasons, and it is refused for the same reason a
        // word the action column does not recognise is -- the file is left
        // alone and the run says so, rather than guessing.
        //
        // A relative path means something different from every directory the
        // command could be run in, so the same report replayed from elsewhere
        // acts on different files. And `..` walks out of `--move-to`: that mode
        // mirrors the source path under the destination root, so a row naming
        // `../../outside/clip.mp4` lands at `<dest>/../../outside/clip.mp4`,
        // creating directories outside the destination and leaving nothing for
        // the documented single copy back from it to restore. `--permanent`
        // reaches the same file by the same arithmetic, with nothing to undo.
        if let Some(fault) = not_canonical(row.file) {
            log::error!(
                target: crate::stats::COUNTED,
                "{}: {} is marked DELETE but its full_path \"{}\" {}. It was left alone: \
                 --from-report acts only on the absolute paths vid-fp writes, because a path \
                 that is not one names a different file from every directory this command \
                 could be run in.",
                report,
                row.at,
                row.file,
                fault
            );
            stats
                .report_unusable
                .record(format!("{} {}: full_path {}", report, row.at, fault));
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

/// Why `path` is not one of the canonical paths this tool writes, or `None`.
///
/// The two faults are worth separating in the message because they are
/// different mistakes: a relative path is usually a report moved or edited by
/// hand, while `..` inside an absolute one is either a hand-built row or an
/// attempt to make a report reach somewhere the run that wrote it never
/// scanned. `Component::CurDir` is deliberately not a fault -- `/a/./b` names
/// the same file from anywhere and mirrors to the same slot -- so this refuses
/// only what actually changes which bytes are touched.
fn not_canonical(path: &str) -> Option<&'static str> {
    let path = Path::new(path);

    if !path.is_absolute() {
        return Some("is not an absolute path");
    }

    path.components()
        .any(|c| c == Component::ParentDir)
        .then_some("contains a \"..\" component")
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

    // The header is line 1, and the parser is what knows where every row after
    // it begins. A record COUNT is not that number: a newline is a legal byte in
    // a Linux filename, `--null` exists so one can be piped in, and the writer
    // quotes such a path rather than mangling it -- so a row can span two
    // physical lines and every complaint after it names a line the user's editor
    // does not have. The number is here so they can go and fix the row, which
    // means it has to point at the row.
    let mut counted = 1u64;

    for record in rdr.records() {
        counted += 1;
        // Only for a record the parser could not place at all, which nothing
        // this reader is handed should be: `records()` positions every record it
        // yields, and an error carries the position it stopped at.
        let line = match &record {
            Ok(r) => r.position().map(|at| at.line()),
            Err(e) => e.position().map(|at| at.line()),
        }
        .unwrap_or(counted);

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
///
/// **What it does NOT hold is the report.** This used to read the whole file
/// into a `String` and parse it into a `serde_json::Value`, on the reasoning
/// that a report is written in one `fs::write` and may as well be read back the
/// same way. That reasoning outlived the writer: `output_results` streams every
/// format now, and reading back what it streams cost far more than writing it.
/// The tree is dominated by the one thing this reader never looks at -- a group
/// of `g` members carries `g * (g - 1)` link objects -- so the 288 MB report the
/// local corpus writes at `-d 18 -p 20` (65,441 rows) was **1,376 MB of peak RSS
/// and 3.0-3.1 s** to replay, against 9.5 MB and 0.07-0.08 s for the same
/// decisions in the CSV. A `Deserializer` over a `BufReader`, into a shape that
/// keeps three cells per row and walks past every other key, is **25 MB and
/// 1.26-1.30 s**: a fifty-fifth of the memory, and what time is left is the
/// parse of 288 MB of text that the CSV states in 23 MB. The replay is
/// byte-identical -- the same 725 rows marked in the same order, with the same
/// complaints, as the old reader and as the CSV beside it.
///
/// Only one thing in the file is refused outright, and it is the same thing the
/// CSV refuses: a report with no rows in it anywhere. A root that is not an
/// object and a missing `results` are both "this is not a report", because
/// walking one would find nothing to do and report that as a clean run over an
/// empty list. Everything inside `results` is reported where it sits instead --
/// a group whose `files` is not a list, a `files` entry that is not an object
/// -- because one mangled group is no reason to refuse the decisions in the
/// ninety-nine good ones. That is why `JsonFiles` and `JsonRow` are written out
/// by hand: a derived struct turns each of those into a parse error, and a
/// parse error here is the whole replay lost to one stray value.
fn read_json(path: &str, stats: &RunStats) -> Result<Report> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open the report {}", path))?;

    // Straight off the disk in `BufReader`-sized pieces. Nothing below keeps a
    // borrow into it, so no part of the file is held for longer than it takes
    // to decide what it was.
    let tree: JsonReport = match serde_json::from_reader(std::io::BufReader::new(file)) {
        Ok(tree) => tree,
        // Well-formed JSON that is not this shape. Serde's own sentence names
        // the key or the type it wanted and the line and column it wanted it
        // at, which is the actionable half; what it cannot know is what the
        // file was supposed to be, so that is said here.
        Err(e) if e.classify() == serde_json::error::Category::Data => {
            return Err(anyhow!(
                "The report {} is not one --from-report can read: {}. It reads the .csv or the \
                 .json that -o writes, and that JSON is an object with a 'results' array of \
                 groups, each holding a 'files' array. A .txt report cannot be replayed at all, \
                 because it records no size to check each file against before removing it.",
                path,
                e
            ));
        }
        Err(e) => {
            return Err(
                anyhow::Error::new(e).context(format!("{} is not valid JSON", path))
            )
        }
    };

    let mut rows = Rows::default();

    for (g, group) in tree.results.into_iter().enumerate() {
        // A group is located by its own `group` key when it has one, because
        // that is the name the report prints and the user reads. Falling back to
        // the index keeps a hand-assembled tree navigable.
        let group_name = group
            .group
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("results[{}]", g));

        let Some(files) = group.files.0 else {
            log::error!(target: crate::stats::COUNTED, "{}: {} has no 'files' array; nothing in it was acted on.", path, group_name);
            stats
                .report_unusable
                .record(format!("{} {}: no files array", path, group_name));
            continue;
        };

        for (f, file) in files.iter().enumerate() {
            rows.consider(
                Row {
                    at: format!("{} file {}", group_name, f + 1),
                    action: &file.action,
                    file: &file.full_path,
                    size: &file.size_bytes,
                },
                path,
                stats,
            );
        }
    }

    Ok(rows.finish())
}

/// A report, as much of it as a decision is taken from.
///
/// `results` is not optional and not a `Value`: a file without it is not a
/// report, and that is the one thing this reader is entitled to abort over.
#[derive(serde::Deserialize)]
struct JsonReport {
    results: Vec<JsonGroup>,
}

/// One group: a name to quote back, and the rows it holds.
///
/// Neither field can fail the report. The name is a `Value` rather than a
/// `String` because it is only ever printed -- a hand-assembled tree that wrote
/// `"group": 5` is navigable by the index fallback, and refusing the whole
/// report over a label would be refusing it over nothing. `JsonFiles` says the
/// same of the rows: a group that holds no list of them is a group with nothing
/// to act on, which is a complaint about that group and not about the file.
#[derive(serde::Deserialize)]
struct JsonGroup {
    #[serde(default)]
    group: serde_json::Value,
    #[serde(default)]
    files: JsonFiles,
}

/// The `files` array of one group, or nothing when the key held some other
/// shape entirely.
///
/// `None` covers all three ways a group can fail to offer rows -- the key
/// absent, `null`, or a value that is not a list -- and they are one finding,
/// reported in one sentence at the group that has it.
#[derive(Default)]
struct JsonFiles(Option<Vec<JsonRow>>);

impl<'de> serde::Deserialize<'de> for JsonFiles {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonFilesVisitor)
    }
}

struct JsonFilesVisitor;

impl<'de> serde::de::Visitor<'de> for JsonFilesVisitor {
    type Value = JsonFiles;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a list of file objects")
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<JsonFiles, A::Error> {
        let mut files = Vec::new();
        while let Some(row) = seq.next_element()? {
            files.push(row);
        }
        Ok(JsonFiles(Some(files)))
    }

    // Anything else is a group with no rows in it. Each is drained rather than
    // abandoned where there is something to drain, because whatever follows it
    // is still in front of the parser and the other groups are on the far side.
    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<JsonFiles, A::Error> {
        while map
            .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
            .is_some()
        {}
        Ok(JsonFiles::default())
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<JsonFiles, E> {
        Ok(JsonFiles::default())
    }
    fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<JsonFiles, E> {
        Ok(JsonFiles::default())
    }
    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<JsonFiles, E> {
        Ok(JsonFiles::default())
    }
    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<JsonFiles, E> {
        Ok(JsonFiles::default())
    }
    fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<JsonFiles, E> {
        Ok(JsonFiles::default())
    }
    fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<JsonFiles, E> {
        Ok(JsonFiles::default())
    }
}

/// One `results[].files[]` object, cut down to the three cells a decision needs.
///
/// Hand-written rather than derived, for the two things a derived struct would
/// get wrong here.
///
/// It skips what it does not want instead of building it. Every unknown key is
/// consumed as `IgnoredAny`, which walks the value without allocating it --
/// including `matches`, the one key that makes these reports large. That is the
/// whole of the memory fix; a derived struct would do the same, but only for as
/// long as nobody typed a `Value` field to be safe.
///
/// And an entry that is not an object at all comes back naming no file, rather
/// than failing the deserializer and with it the entire replay. A `files` array
/// with a stray number in it is one mangled row among however many good ones,
/// and this module's whole disposition towards those is to count them, name
/// them and act on the rest -- see `Rows::consider`, which is where such a row
/// is turned into a complaint.
#[derive(Default)]
struct JsonRow {
    action: String,
    full_path: String,
    size_bytes: String,
}

impl<'de> serde::Deserialize<'de> for JsonRow {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonRowVisitor)
    }
}

struct JsonRowVisitor;

impl<'de> serde::de::Visitor<'de> for JsonRowVisitor {
    type Value = JsonRow;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a file object")
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<JsonRow, A::Error> {
        let mut row = JsonRow::default();

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "action" => row.action = cell(map.next_value()?),
                "full_path" => row.full_path = cell(map.next_value()?),
                "size_bytes" => row.size_bytes = cell(map.next_value()?),
                // `matches` above all, and it is the reason this is not a
                // `Value`: one object per measured link, never read, and on a
                // large report the great majority of the file.
                _ => {
                    map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
        }

        Ok(row)
    }

    // Every other shape a JSON value can take. None of them is a row, and each
    // comes back as the empty one it amounts to -- which `Rows::consider`
    // counts and leaves alone, exactly as it does a row whose action cell was
    // emptied. The alternative is a deserializer error, and a deserializer
    // error here is the whole report refused over one entry.
    fn visit_unit<E: serde::de::Error>(self) -> Result<JsonRow, E> {
        Ok(JsonRow::default())
    }
    fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<JsonRow, E> {
        Ok(JsonRow::default())
    }
    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<JsonRow, E> {
        Ok(JsonRow::default())
    }
    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<JsonRow, E> {
        Ok(JsonRow::default())
    }
    fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<JsonRow, E> {
        Ok(JsonRow::default())
    }
    fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<JsonRow, E> {
        Ok(JsonRow::default())
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<JsonRow, A::Error> {
        // Drained rather than abandoned: what is left of it is still in front
        // of the parser, and the rows after it are on the other side.
        while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
        Ok(JsonRow::default())
    }
}

/// One JSON value as the cell it stands for.
///
/// A string keeps every byte of itself, and is moved rather than copied -- a
/// path's leading space is part of its name and `Rows::consider` is relying on
/// still having it. Anything else is
/// rendered the way JSON writes it, which is what makes a number usable as a
/// size and makes a nonsense value (an array, `true`) show up as the nonsense it
/// is in the message that reports it, rather than as a blank.
fn cell(value: serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s,
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
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
    let mut aliased_count = 0usize;
    let mut aliased_bytes = 0u64;
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
            Fate::Done { aliased } => {
                removed_count += 1;
                // Only bytes that actually went away. See `export::OnDisk`:
                // this mode reaches the same files by the same paths, and a
                // report row records a size for a file that may have had
                // another name all along.
                if aliased {
                    aliased_count += 1;
                    aliased_bytes += m.size;
                } else {
                    removed_bytes += m.size;
                }
                deleted_paths.push(m.path.clone());
                disposal.done_label(aliased)
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
        info!(
            "\t{:<width$} {}, {}",
            format!("{},", label),
            format_size(m.size),
            m.path,
            width = export::ACTION_COLUMN
        );
    }

    let mut summary = export::disposed_line(disposal, removed_count, removed_bytes);
    summary.push_str(&export::aliased_line(aliased_count, aliased_bytes));
    summary.push_str(&export::trouble_lines(failed_count, changed_count));
    info!("\n{}", summary);

    Ok(deleted_paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Every word either listing can print has to fit the column both of them
    /// pad to, and `KNOWN_ACTIONS` is the complete set of them -- it exists
    /// because a replay must recognise every word this tool writes, so a word
    /// that is not in it is a word no listing prints.
    ///
    /// UNLINKED was added without the width moving, so exactly those rows had
    /// every field after the action shifted one place right: the rows a reader
    /// is most likely to be studying, since they are the ones whose byte total
    /// does not add up. Equality rather than `<=` so the column cannot silently
    /// drift wider than it needs to be either.
    #[test]
    fn test_the_action_column_is_exactly_wide_enough_for_the_longest_action() {
        let widest = KNOWN_ACTIONS
            .iter()
            .map(|word| word.len() + 1) // the comma the listings append
            .max()
            .unwrap();
        assert_eq!(
            export::ACTION_COLUMN, widest,
            "the widest action is {:?}",
            KNOWN_ACTIONS.iter().max_by_key(|w| w.len()).unwrap()
        );

        // And the effect of it: one column, whatever the row says.
        let column_ends: HashSet<usize> = KNOWN_ACTIONS
            .iter()
            .map(|word| {
                format!(
                    "{:<width$} rest",
                    format!("{},", word.to_uppercase()),
                    width = export::ACTION_COLUMN
                )
                .find("rest")
                .unwrap()
            })
            .collect();
        assert_eq!(column_ends.len(), 1, "every action leaves the next field in one place");
    }

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
        //
        // UNLINKED is one of ours too: it is what a DELETE row becomes when the
        // data had another name. Leaving it out of the list would not make the
        // run act on those rows -- it would report every one of them as a word
        // this tool does not recognise, and exit 2 for a report it wrote.
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n",
            HEADER,
            row("/a.mkv", 10, "KEPT"),
            row("/b.mkv", 20, "DELETED"),
            row("/c.mkv", 30, "MOVED"),
            row("/d.mkv", 40, "CHANGED"),
            row("/e.mkv", 50, "UNLINKED"),
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
    fn test_a_row_is_named_by_the_line_it_is_really_on() {
        // A newline is a legal byte in a Linux filename, `--null` exists so one
        // can be piped in, and both writers quote such a path rather than
        // mangling it -- so a report can carry a row spanning two physical
        // lines. Counting records and calling the count a line number then
        // names a row the user's editor does not have: the complaint exists so
        // they can go and fix the row, and it has to point at it.
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n{}\n",
            HEADER,                                        // line 1
            row("/a.mkv", 10, "DELETE"),                   // line 2
            row("\"/we\nird.mkv\"", 20, "DELETE"),         // lines 3-4
            row("/c.mkv", 30, "DELETEE"),                  // line 5
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        // The quoted path round-trips, newline and all.
        assert_eq!(report.marked.len(), 2);
        assert_eq!(report.marked[1].path, "/we\nird.mkv");

        assert_eq!(stats.report_unusable.count(), 1);
        let sample = &stats.report_unusable.samples()[0];
        assert!(
            sample.contains("line 5"),
            "the misspelt action is on physical line 5: {}",
            sample
        );
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
    fn test_a_link_that_names_itself_a_decision_is_still_not_one() {
        // `test_the_nested_match_list_is_not_a_row` says the link objects are
        // not rows; this says it of a link built to look exactly like one. The
        // reader walks past every key it does not want without building it,
        // which is the whole of why a 288 MB report no longer costs 1.4 GB to
        // replay -- and a walk-past that started reading would take the file
        // that WON the comparison.
        let dir = tempfile::tempdir().unwrap();
        let tree = serde_json::json!({
            "results": [ { "group": "group_1", "files": [ {
                "action": "KEEP",
                "full_path": "/a.mkv",
                "size_bytes": 10,
                "matches": [
                    { "action": "DELETE", "full_path": "/b.mkv", "size_bytes": 20 }
                ],
            } ] } ]
        });
        let path = write_json(&dir, "links.json", &tree.to_string());

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.rows, 1, "the link is not a row");
        assert!(report.marked.is_empty(), "and its DELETE is not a decision");
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_a_junk_entry_among_the_files_is_one_row_and_not_a_dead_report() {
        // The line this reader draws: the structure of a report is refused
        // outright, its contents are reported row by row. A stray value in a
        // `files` array is contents -- one mangled row, counted and left alone,
        // with the good rows either side of it still acted on. Deserializing
        // straight into a struct would have made it a parse error, and a parse
        // error here is sixty-five thousand decisions refused over one.
        let dir = tempfile::tempdir().unwrap();
        let tree = serde_json::json!({
            "results": [ { "group": "group_1", "files": [
                json_file("/a.mkv", serde_json::json!(10), "DELETE"),
                serde_json::json!(42),
                serde_json::json!("not a file object"),
                serde_json::json!([1, 2, 3]),
                serde_json::json!(null),
                json_file("/b.mkv", serde_json::json!(20), "DELETE"),
            ] } ]
        });
        let path = write_json(&dir, "junk.json", &tree.to_string());

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.rows, 6, "every entry is a row, junk included");
        let marked: Vec<&str> = report.marked.iter().map(|m| m.path.as_str()).collect();
        assert_eq!(marked, ["/a.mkv", "/b.mkv"], "the good rows either side still go through");
        // A junk entry names no file and asks for nothing, which is the same
        // silence an emptied action cell gets and for the same reason.
        assert_eq!(stats.report_unusable.count(), 0);
    }

    #[test]
    fn test_a_group_that_holds_no_list_of_rows_is_one_complaint_and_not_a_dead_report() {
        // A group can fail to offer rows in three ways -- the key absent, null,
        // or something that is not a list at all -- and all three are the same
        // finding about that group: there is nothing in it to act on. None of
        // them is a finding about the file, so the groups either side are still
        // read, which is what a derived struct would have taken away by turning
        // the third one into a parse error.
        let dir = tempfile::tempdir().unwrap();
        let tree = serde_json::json!({
            "results": [
                { "group": "group_1", "files": [
                    json_file("/a.mkv", serde_json::json!(10), "DELETE")
                ] },
                { "group": "group_2", "files": 3 },
                { "group": "group_3", "files": serde_json::Value::Null },
                { "group": "group_4", "files": { "full_path": "/x.mkv" } },
                { "group": "group_5" },
                { "group": "group_6", "files": [
                    json_file("/b.mkv", serde_json::json!(20), "DELETE")
                ] },
            ]
        });
        let path = write_json(&dir, "shape.json", &tree.to_string());

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        let marked: Vec<&str> = report.marked.iter().map(|m| m.path.as_str()).collect();
        assert_eq!(marked, ["/a.mkv", "/b.mkv"], "the groups either side are still read");
        assert_eq!(report.rows, 2, "a group with no rows contributes none");
        assert_eq!(stats.report_unusable.count(), 4, "and each says so once");
        assert!(stats.had_problems());
        // The one nested path in the file is in the group that had no list, and
        // it must not have been read as a row of anything.
        assert!(!marked.contains(&"/x.mkv"));
    }

    #[test]
    fn test_a_group_labelled_with_something_other_than_a_name_is_still_navigable() {
        // The name is only ever quoted back, so a hand-assembled tree that put
        // a number there falls back to the index and the complaint still tells
        // the user where to look. Refusing the whole report over a label would
        // be refusing it over nothing.
        let dir = tempfile::tempdir().unwrap();
        let tree = serde_json::json!({
            "results": [ { "group": 5, "files": [
                json_file("/a.mkv", serde_json::json!(10), "DELET")
            ] } ]
        });
        let path = write_json(&dir, "label.json", &tree.to_string());

        let stats = RunStats::default();
        read(&path, &stats).unwrap();

        let complaints = stats.report_unusable.samples();
        assert_eq!(complaints.len(), 1);
        assert!(complaints[0].contains("results[0] file 1"), "{}", complaints[0]);
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
    fn test_a_path_that_is_not_the_canonical_one_vid_fp_writes_is_refused() {
        // Both faults reach a file the report has no business naming, and by
        // different routes. A relative path is resolved against whatever
        // directory the command was run in, so one report acts on different
        // files depending on where it is replayed from. `..` in an absolute one
        // walks back out of the `--move-to` mirror. Neither can appear in a
        // report this tool wrote, because every scanned path is canonical.
        let dir = tempfile::tempdir().unwrap();
        let body = format!(
            "{}\n{}\n{}\n{}\n{}\n",
            HEADER,
            row("../../outside/pwned.mkv", 10, "DELETE"),
            row("outside/pwned.mkv", 20, "DELETE"),
            row("/../../outside/pwned.mkv", 30, "DELETE"),
            row("/videos/./ok.mkv", 40, "DELETE"),
        );
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert_eq!(report.rows, 4);
        assert_eq!(stats.report_unusable.count(), 3);
        // A `.` component names the same file from anywhere and mirrors to the
        // same slot, so it is not one of the faults.
        assert_eq!(report.marked.len(), 1);
        assert_eq!(report.marked[0].path, "/videos/./ok.mkv");
    }

    #[test]
    fn test_a_json_path_that_is_not_canonical_is_refused_by_the_same_rule() {
        // The judgement lives in `consider`, so it has to reach every format --
        // that is the arrangement the module doc describes, and a second reader
        // that could be talked into a relative path would defeat it.
        let dir = tempfile::tempdir().unwrap();
        let tree = serde_json::json!({
            "results": [{
                "group": 1,
                "files": [{ "action": "DELETE", "full_path": "../pwned.mkv", "size_bytes": 10 }]
            }]
        });
        let path = write_json(&dir, "report.json", &tree.to_string());

        let stats = RunStats::default();
        let report = read(&path, &stats).unwrap();

        assert!(report.marked.is_empty());
        assert_eq!(stats.report_unusable.count(), 1);
    }

    #[test]
    fn test_a_path_cannot_walk_out_of_the_move_to_destination() {
        // The end-to-end shape of it. `--move-to` mirrors the source path under
        // the destination root by stripping the leading `/` and joining, so a
        // `..` survives that arithmetic and comes out the other side: the row
        // below names a file that opens perfectly well, and lands it OUTSIDE
        // the destination along with the directories created to hold it. That
        // breaks the one promise the mode makes -- that the whole run can be
        // undone by a single copy back from the destination -- and
        // `--permanent` reaches the same file with nothing to undo at all.
        let root = tempfile::tempdir().unwrap();
        let outside = root.path().join("outside");
        std::fs::create_dir(&outside).unwrap();
        let victim = outside.join("victim.mkv");
        std::fs::write(&victim, vec![b'x'; 200]).unwrap();

        let dest = root.path().join("dest");
        std::fs::create_dir(&dest).unwrap();

        // `/..` is `/`, so this opens the file above -- and mirrored under the
        // destination it is `<dest>/../<root>/outside/victim.mkv`, a sibling of
        // the destination rather than anything inside it.
        let escaping = format!("/..{}", victim.display());
        let dir = tempfile::tempdir().unwrap();
        let body = format!("{}\n{}\n", HEADER, row(&escaping, 200, "DELETE"));
        let path = write_report(&dir, &body);

        let stats = RunStats::default();
        let disposal = Disposal::MoveTo(dest.clone());
        let gone = apply(&path, &disposal, true, &stats).unwrap();

        assert!(gone.is_empty(), "nothing should have been disposed of");
        assert!(victim.exists(), "the file outside the destination is untouched");
        assert_eq!(
            std::fs::read_dir(&dest).unwrap().count(),
            0,
            "the destination is still empty"
        );
        assert!(stats.had_problems(), "a row this run declined to act on is a problem");
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
