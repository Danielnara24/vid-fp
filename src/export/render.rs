//! What a report says: one `Row` per file, one `Summary` per run, and the four
//! places they are written -- the console listing and the three `--format`s.
//!
//! The row is the schema. The CSV's header and the JSON's keys are both read
//! off `Row`'s fields by serde, in declaration order, so a column is added,
//! moved or renamed in exactly one place and the two machine-readable formats
//! cannot drift apart. Each destination is a `Body`, and only the bodies a run
//! actually has are built, so nothing is formatted for a format nobody asked
//! for.

use super::stream::ReportStream;
use super::{Disposal, Format, ReportTarget, Sink, Tally, ACTION_COLUMN};
use crate::compare::{GroupLink, MatchIndex, Span};
use crate::fingerprint::VideoFingerprint;
use crate::utils::{
    format_bitrate, format_codec, format_duration, format_frame_rate, format_quality,
    format_shared, format_size,
};
use anyhow::{Context, Result};
use log::info;
use serde::Serialize;
use std::io::Write;

/// Every seconds figure in the machine-readable formats is stated to the
/// hundredth.
fn hundredths(seconds: f64) -> f64 {
    (seconds * 100.0).round() / 100.0
}

/// One file's row: every CSV column after `group`, and every key of a JSON
/// file object before `matches`, in that order.
///
/// The columns run in three blocks, each answering one question:
///
///   who        action, full_path -- the row's identity and its fate. Action
///              sits beside the path because `--from-report` exists to have it
///              edited, and an action you have to scroll to is one that gets
///              edited on the wrong row.
///   what       length .. quality_bits_per_frame -- the file's own properties,
///              in the order the ranking reads them, codec leading the three
///              bit-derived figures because it is the reason two rows' figures
///              may not be compared.
///   against    matched_with .. matched_to_seconds -- the strongest measured
///              link. Every figure in it answers for THIS file (the `matched_`
///              prefix promises it): `matched_seconds` is `coverage x duration`
///              of this row's own runtime, and the range is in its own timeline.
///
/// A formatted figure is followed by the raw number it was formatted from: the
/// formatted one is for reading, the raw one for sorting, and a spreadsheet
/// cannot sort "1.0MB" against "900.0KB". `framerate_fps` is the one figure
/// carried raw only, by request; its formatted twin is on the console line.
///
/// An unknown figure is `None`: an empty cell in the CSV, `null` in the JSON,
/// never a zero. A container that never reported a frame rate is not one that
/// reported no frames, and a runtime nobody measured is not a zero-length
/// video -- the first metric the ranking compares, and one a consumer sorting
/// this column must not read as the shortest. The formatted columns use "-"
/// for the same unknowns.
///
/// `samples` is a property of the file rather than of the link, and sits right
/// before `matched_seconds` because that is the figure it qualifies: a range
/// much wider than the footage is a scattered match or a file too coarsely
/// sampled to know, and only the sample count tells them apart. A file with ONE
/// sample can only ever match all of itself or none of it.
#[derive(Serialize, Default)]
pub(super) struct Row<'a> {
    action: &'static str,
    full_path: &'a str,
    length: String,
    length_seconds: Option<f64>,
    resolution: String,
    width: u32,
    height: u32,
    framerate_fps: Option<f64>,
    codec: String,
    size: String,
    size_bytes: u64,
    bitrate: String,
    bitrate_bps: u64,
    quality: String,
    quality_bits_per_frame: Option<u64>,
    matched_with: Option<&'a str>,
    /// Whether that file is this one's bytes again, rather than a match of
    /// the picture.
    identical: Option<bool>,
    samples: usize,
    matched_seconds: Option<f64>,
    matched_from: Option<String>,
    matched_to: Option<String>,
    matched_from_seconds: Option<f64>,
    matched_to_seconds: Option<f64>,
}

/// The four figures that place a match in a file's timeline, shared by the
/// row's strongest link and every entry of the JSON's `matches`.
struct Placed {
    seconds: Option<f64>,
    from: Option<String>,
    to: Option<String>,
    from_seconds: Option<f64>,
    to_seconds: Option<f64>,
}

impl Placed {
    fn of(seconds: Option<f64>, span: Option<Span>) -> Self {
        Placed {
            seconds: seconds.map(hundredths),
            from: span.map(|s| format_duration(s.start_seconds())),
            to: span.map(|s| format_duration(s.end_seconds())),
            from_seconds: span.map(|s| hundredths(s.start_seconds())),
            to_seconds: span.map(|s| hundredths(s.end_seconds())),
        }
    }
}

/// A file as one group reports it: its row, and what the row does not carry --
/// the fingerprint the console line formats its own way, and every link the
/// JSON lists.
pub(super) struct Entry<'a> {
    fp: &'a VideoFingerprint,
    /// Every measured link to another member of THIS group, strongest first.
    /// Scoped to the group because a file in two groups matched files in both,
    /// and a row must only name files the reader can see beside it.
    links: Vec<GroupLink>,
    /// The strongest link's footage in this file's own seconds, unrounded.
    /// `None` with no link, and with no runtime to state it in: "0s matched"
    /// would read as evidence that nothing matched.
    matched: Option<f64>,
    fingerprints: &'a [VideoFingerprint],
    row: Row<'a>,
}

impl<'a> Entry<'a> {
    pub(super) fn of(
        idx: usize,
        group: &[usize],
        fingerprints: &'a [VideoFingerprint],
        matches: &MatchIndex,
        action: &'static str,
    ) -> Self {
        let fp = &fingerprints[idx];
        let links = matches.links_of(idx, group, fingerprints);
        let best = links.first();
        let measured = fp.duration > 0.0;
        let matched = best.map(|l| l.matched_seconds).filter(|_| measured);
        let placed = Placed::of(matched, best.and_then(|l| l.span));

        let row = Row {
            action,
            full_path: &fp.path,
            length: if measured { format_duration(fp.duration) } else { "-".to_string() },
            length_seconds: measured.then(|| hundredths(fp.duration)),
            resolution: format!("{}x{}", fp.width, fp.height),
            width: fp.width,
            height: fp.height,
            framerate_fps: (fp.frame_rate > 0.0).then(|| (fp.frame_rate * 1000.0).round() / 1000.0),
            codec: format_codec(&fp.codec),
            size: format_size(fp.file_size),
            size_bytes: fp.file_size,
            bitrate: format_bitrate(fp.bitrate()),
            bitrate_bps: fp.bitrate(),
            // Bits per frame is the figure that ranks; bitrate is the one
            // people know, and it ranks nothing.
            quality: format_quality(fp.quality()),
            quality_bits_per_frame: (fp.quality() > 0).then(|| fp.quality()),
            matched_with: best.map(|l| fingerprints[l.other].path.as_str()),
            identical: best.map(|l| l.identical),
            // Hashes held, not keyframes decoded: featureless frames are
            // dropped, so this is what the comparison had to work with.
            samples: fp.valid_hashes.len(),
            matched_seconds: placed.seconds,
            matched_from: placed.from,
            matched_to: placed.to,
            matched_from_seconds: placed.from_seconds,
            matched_to_seconds: placed.to_seconds,
        };

        Entry { fp, links, matched, fingerprints, row }
    }

    /// The console's and the `.txt` report's line: the same three blocks as
    /// the row, minus the raw twins and the matched file's path, which are
    /// noise at a glance. The action leads, padded to a column the eye can run
    /// down; the path trails because it is the one field of unbounded width.
    /// Units are spelled out because the console has no header.
    fn text_line(&self) -> String {
        let row = &self.row;
        format!(
            "\t{:<width$} {}, {}, {}, {}, {}, {}, {}, {} samples, {} matched{}, {}",
            format!("{},", row.action),
            row.length,
            row.resolution,
            format_frame_rate(self.fp.frame_rate),
            row.codec,
            row.size,
            row.bitrate,
            row.quality,
            row.samples,
            format_shared(self.matched),
            if row.identical == Some(true) { " (identical)" } else { "" },
            self.fp.path,
            width = ACTION_COLUMN
        )
    }
}

/// What a run decided and did, said once and written two ways: as text at the
/// end of the listing and the `.txt` report, and as the JSON's `summary`.
pub(super) struct Summary<'a> {
    pub(super) groups: usize,
    /// Distinct files in a group of two or more. Groups overlap, so summing
    /// their sizes would count a shared file once per group.
    pub(super) matched_files: usize,
    pub(super) elapsed_secs: u64,
    /// The files marked DELETE, and their bytes: what a run would reclaim,
    /// stated whether or not it did anything, because a dry run's whole output
    /// is a plan and a plan with no cost attached is not one. An upper bound:
    /// a file with another name outside the run reclaims nothing.
    pub(super) marked: usize,
    pub(super) reclaimable_bytes: u64,
    /// What the run DID, which is not what its flags asked for once the user
    /// has declined at the prompt.
    pub(super) disposal: Option<&'a Disposal>,
    pub(super) declined: bool,
    pub(super) tally: &'a Tally,
}

impl std::fmt::Display for Summary<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let t = self.elapsed_secs;
        write!(
            f,
            "Total groups found: {}\nTotal files matched: {}\nTotal time elapsed: {:02}:{:02}:{:02}",
            self.groups,
            self.matched_files,
            t / 3600,
            (t % 3600) / 60,
            t % 60
        )?;

        match self.disposal {
            Some(d) => write!(f, "\n{}", self.tally.lines(d)),
            None => {
                if self.marked > 0 {
                    write!(
                        f,
                        "\nReclaimable: {} across {} file(s) marked DELETE (nothing was removed).",
                        format_size(self.reclaimable_bytes),
                        self.marked
                    )?;
                }
                // "Nothing was removed" alone reads as a report-only run; the
                // user who declined needs to see the difference between "I
                // wasn't asked to" and "you told me not to".
                if self.declined {
                    write!(f, "\nCancelled at the confirmation prompt.")?;
                }
                Ok(())
            }
        }
    }
}

/// The JSON's `summary`, in three blocks: what was found, what was decided,
/// what was done -- so a dry run's figures stop after the second has said
/// everything it can.
#[derive(Serialize)]
struct JsonSummary {
    total_groups: usize,
    total_files_matched: usize,
    time_elapsed_seconds: u64,
    files_marked_delete: usize,
    reclaimable_bytes: u64,
    deletion_enabled: bool,
    mode: &'static str,
    move_to: Option<String>,
    files_removed: usize,
    /// Excludes the files that were another name for data still on disk;
    /// `files_unlinked` and `bytes_unlinked` reconcile it against the rows.
    bytes_removed: u64,
    files_unlinked: usize,
    bytes_unlinked: u64,
    files_failed: usize,
    files_changed: usize,
}

impl Summary<'_> {
    fn json(&self) -> JsonSummary {
        JsonSummary {
            total_groups: self.groups,
            total_files_matched: self.matched_files,
            time_elapsed_seconds: self.elapsed_secs,
            files_marked_delete: self.marked,
            reclaimable_bytes: self.reclaimable_bytes,
            deletion_enabled: self.disposal.is_some(),
            mode: self.disposal.map(|d| d.mode()).unwrap_or("report"),
            move_to: match self.disposal {
                Some(Disposal::MoveTo(dir)) => Some(dir.to_string_lossy().to_string()),
                _ => None,
            },
            files_removed: self.tally.removed,
            bytes_removed: self.tally.removed_bytes,
            files_unlinked: self.tally.aliased,
            bytes_unlinked: self.tally.aliased_bytes,
            files_failed: self.tally.failed,
            files_changed: self.tally.changed,
        }
    }
}

/// One place a run's groups are written.
///
/// A report body swallows its own write failures and answers for the first
/// one in `finish`, and never mid-pass: by the time a body is written the
/// disposal pass has run, and an error carried out of `output_results` would
/// take `deleted_paths` with it, leaving the cache claiming fingerprints for
/// files this run just removed. Neither method can return early for that
/// reason.
pub(super) trait Body {
    fn group(&mut self, name: &str, files: &[Entry]);
    fn finish(self: Box<Self>, summary: &Summary) -> Result<()>;
}

/// The body a report target is written by. Opened before the first group,
/// which the JSON needs: its summary leads the document.
pub(super) fn open_report(target: &ReportTarget, summary: &Summary) -> Box<dyn Body> {
    match target.format {
        Format::Txt => Box::new(TxtBody { stream: ReportStream::open(target), target: target.clone() }),
        Format::Csv => {
            let mut body = CsvBody::open(target);
            body.record(csv_header());
            Box::new(body)
        }
        Format::Json => Box::new(JsonBody::open(target, &summary.json(), summary.groups > 0)),
    }
}

/// A group as the text formats lay it out: a name, a line a file, a blank.
fn text_group(name: &str, files: &[Entry], mut emit: impl FnMut(&str)) {
    emit(&format!("{}:", name));
    for file in files {
        emit(&file.text_line());
    }
    emit("");
}

/// The results table on the terminal.
pub(super) struct Listing;

impl Listing {
    pub(super) fn open() -> Self {
        info!("\n========================================");
        info!("             RESULTS");
        info!("========================================\n");
        Listing
    }
}

impl Body for Listing {
    fn group(&mut self, name: &str, files: &[Entry]) {
        text_group(name, files, |line| info!("{}", line));
    }

    /// The console's summary is not the listing's to print: a CSV on stdout
    /// keeps the summary on stderr with no listing above it.
    fn finish(self: Box<Self>, _: &Summary) -> Result<()> {
        Ok(())
    }
}

/// The `.txt` report: the listing, then the summary, which is the one part of
/// it that cannot be known until the last group is written.
struct TxtBody {
    stream: ReportStream,
    target: ReportTarget,
}

impl Body for TxtBody {
    fn group(&mut self, name: &str, files: &[Entry]) {
        let stream = &mut self.stream;
        text_group(name, files, |line| {
            let _ = writeln!(stream, "{}", line);
        });
    }

    fn finish(mut self: Box<Self>, summary: &Summary) -> Result<()> {
        let _ = writeln!(self.stream, "{}", summary);
        self.stream.finish(&self.target)
    }
}

/// The CSV's first column, which the JSON states as the key its files are
/// listed under instead.
#[derive(Serialize, Default)]
struct GroupCell<'a> {
    group: &'a str,
}

/// The CSV header, read off the same two structs every record is written
/// from. Written explicitly rather than left to `csv`'s first `serialize`, so
/// an empty report still has one: `--from-report` names a header-less file as
/// one with no `full_path` column.
fn csv_header() -> Vec<String> {
    let mut probe = csv::Writer::from_writer(Vec::new());
    probe.serialize((GroupCell::default(), Row::default())).expect("the row's own fields serialize");
    let bytes = probe.into_inner().expect("a Vec takes every write");
    let mut read = csv::Reader::from_reader(bytes.as_slice());
    read.headers().expect("just written").iter().map(String::from).collect()
}

/// The CSV report, written one record at a time.
///
/// `ReportStream` makes the sink infallible, and a `csv::Writer` can still fail
/// on its own, for a record that will not serialize or is not the width of the
/// header -- which the sink never sees. `record` returns nothing, so there is
/// no `?` to write by accident; the first failure is kept, everything after it
/// is skipped, and `close` answers for it BEFORE the sink's own, because the
/// sink saw nothing after it and would report a truncated report as written.
pub(super) struct CsvBody {
    wtr: csv::Writer<ReportStream>,
    failed: Option<csv::Error>,
    target: ReportTarget,
}

impl CsvBody {
    pub(super) fn open(target: &ReportTarget) -> Self {
        CsvBody {
            wtr: csv::WriterBuilder::new()
                .delimiter(b';')
                .has_headers(false)
                .from_writer(ReportStream::open(target)),
            failed: None,
            target: target.clone(),
        }
    }

    pub(super) fn record(&mut self, record: impl Serialize) {
        if self.failed.is_some() {
            return;
        }
        if let Err(e) = self.wtr.serialize(record) {
            self.failed = Some(e);
        }
    }

    pub(super) fn close(self) -> Result<()> {
        // `into_inner` flushes, so it has to happen either way: the sink owns
        // the scratch copy, and dropping the writer instead would leave it to a
        // destructor that cannot report what it found.
        let stream = self
            .wtr
            .into_inner()
            .map_err(|e| e.into_error())
            .context("Failed to finalize the CSV report")?;

        match self.failed {
            Some(e) => Err(e).context("Failed to write the CSV report"),
            None => stream.finish(&self.target),
        }
    }
}

impl Body for CsvBody {
    fn group(&mut self, name: &str, files: &[Entry]) {
        for file in files {
            self.record((GroupCell { group: name }, &file.row));
        }
    }

    fn finish(self: Box<Self>, _: &Summary) -> Result<()> {
        self.close()
    }
}

/// The indent `serde_json::to_string_pretty` uses, and therefore the one a
/// document assembled a piece at a time has to match.
const JSON_INDENT: &str = "  ";

/// Render one value at the depth it sits at in the finished document.
///
/// `to_string_pretty` always starts a value at column zero, so a piece written
/// into the middle of a document has to be pushed out by the nesting it is
/// being written into: `depth` is the value's OWN depth, and `indent_first`
/// says whether it begins on a fresh line (a group inside the results array) or
/// after a key that has already been written (the summary).
///
/// Splitting on '\n' is safe for exactly this input: pretty JSON never emits a
/// blank line, and a string value carries its own newlines escaped.
fn write_nested(
    out: &mut dyn Write,
    value: &impl Serialize,
    depth: usize,
    indent_first: bool,
) -> std::io::Result<()> {
    let pad = JSON_INDENT.repeat(depth);
    // Nothing a report holds can fail to serialize: every key is a field name,
    // and serde_json writes a non-finite float as `null` rather than refusing.
    let rendered = serde_json::to_string_pretty(value).expect("a report value serializes");

    for (n, line) in rendered.split('\n').enumerate() {
        if n > 0 {
            out.write_all(b"\n")?;
        }
        if n > 0 || indent_first {
            out.write_all(pad.as_bytes())?;
        }
        out.write_all(line.as_bytes())?;
    }
    Ok(())
}

/// One entry of a file's `matches`: every measured link in the group, not
/// only the strongest, because this is the format read by something that can
/// hold a list. Its figures are the row's link block, for that link.
#[derive(Serialize)]
struct JsonLink<'a> {
    full_path: &'a str,
    identical: bool,
    matched_seconds: Option<f64>,
    matched_from: Option<String>,
    matched_to: Option<String>,
    matched_from_seconds: Option<f64>,
    matched_to_seconds: Option<f64>,
}

#[derive(Serialize)]
struct JsonFile<'r, 'a> {
    #[serde(flatten)]
    row: &'r Row<'a>,
    matches: Vec<JsonLink<'a>>,
}

#[derive(Serialize)]
struct JsonGroup<'r, 'a> {
    group: &'r str,
    files: Vec<JsonFile<'r, 'a>>,
}

/// The JSON report, written one group at a time rather than assembled as a
/// tree: a group of `g` members carries `g * (g - 1)` links, and holding the
/// whole document cost gigabytes on a loose scan.
struct JsonBody {
    stream: ReportStream,
    target: ReportTarget,
    /// How many groups have gone into `results`: the first needs no leading
    /// comma.
    groups_written: usize,
    /// Whether `results` was opened as an array with something in it. An empty
    /// one is written closed, which is the one shape writing groups cannot
    /// reach.
    results_open: bool,
}

impl JsonBody {
    /// Open the document and write the summary: first, because it is the part
    /// a human opens the file to read, and possible because every figure in it
    /// was settled by the disposal pass before any group is written.
    fn open(target: &ReportTarget, summary: &impl Serialize, any_groups: bool) -> Self {
        let mut stream = ReportStream::open(target);

        // Infallible: `ReportStream` records a failure rather than returning
        // one, which is the whole of why these writes need no handling.
        let _ = stream.write_all(b"{\n  \"summary\": ");
        let _ = write_nested(&mut stream, summary, 1, false);
        let _ = stream.write_all(if any_groups {
            b",\n  \"results\": [\n".as_slice()
        } else {
            b",\n  \"results\": []".as_slice()
        });

        JsonBody { stream, target: target.clone(), groups_written: 0, results_open: any_groups }
    }
}

impl Body for JsonBody {
    fn group(&mut self, name: &str, files: &[Entry]) {
        let group = JsonGroup {
            group: name,
            files: files
                .iter()
                .map(|file| JsonFile {
                    row: &file.row,
                    matches: file
                        .links
                        .iter()
                        .map(|l| {
                            let measured = file.fp.duration > 0.0;
                            let placed = Placed::of(measured.then_some(l.matched_seconds), l.span);
                            JsonLink {
                                full_path: &file.fingerprints[l.other].path,
                                identical: l.identical,
                                matched_seconds: placed.seconds,
                                matched_from: placed.from,
                                matched_to: placed.to,
                                matched_from_seconds: placed.from_seconds,
                                matched_to_seconds: placed.to_seconds,
                            }
                        })
                        .collect(),
                })
                .collect(),
        };

        if self.groups_written > 0 {
            let _ = self.stream.write_all(b",\n");
        }
        self.groups_written += 1;
        let _ = write_nested(&mut self.stream, &group, 2, true);
    }

    fn finish(mut self: Box<Self>, _: &Summary) -> Result<()> {
        let closing: &[u8] = if self.results_open { b"\n  ]\n}" } else { b"\n}" };
        let _ = self.stream.write_all(closing);
        self.stream.finish(&self.target)
    }
}

/// Which parts of the run stderr still says out loud once the report has a
/// destination.
pub(super) struct Console {
    /// The per-group table of files.
    pub(super) listing: bool,
    /// The counts, the reclaimable total and what the disposal actually did.
    pub(super) summary: bool,
}

/// A report on stdout REPLACES the console listing rather than joining it.
///
/// The two are the same run said twice: `-o -` printed the whole listing on
/// stderr and the whole report on stdout, so a terminal showed everything twice
/// and `vid-fp DIR -o - | grep DELETE` still scrolled the entire listing past
/// the user on the way to the one line they asked for. A report written to a
/// FILE is a different case and keeps both: the terminal is then the only place
/// the run is visible while it happens.
///
/// The summary is the one part that does not simply follow the listing, because
/// two of the three formats already end with it -- a `.txt` report is exactly
/// this listing plus that summary, and the JSON carries the same figures under
/// `summary`. The CSV carries neither, so there stderr keeps the receipt: what
/// was reclaimable, what was removed, what went wrong. That is the part of a
/// destructive run nobody should have to read out of a pipe.
pub(super) fn console_for(target: Option<&ReportTarget>) -> Console {
    match target {
        Some(ReportTarget { sink: Sink::Stdout, format }) => Console {
            listing: false,
            summary: *format == Format::Csv,
        },
        _ => Console { listing: true, summary: true },
    }
}
