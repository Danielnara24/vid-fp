//! A finished scan as the desktop app reads it: the groups, every file in
//! them with the figures the report prints, and each file's KEEP, DELETE or
//! REVIEW.
//!
//! The worker (`worker_main`) writes one of these to the path the window
//! names, and the window reads it back. It is not a report format: nothing
//! promises its shape to anyone but the window of the same build, which is
//! why `version` is checked and an old one is simply not shown.
//!
//! **The labels are the report's, by construction.** They come from
//! `export::verdicts`, the same function `output_results` labels its rows
//! with, called on the same groups, fingerprints and links. A window that
//! worked them out again from figures written to disk would round a duration
//! on the way, and the ranking has tolerance bands for a rounding to land on
//! the wrong side of.
//!
//! **Every ranking is labelled, not only the one asked for.** `--priority`
//! decides nothing but which copy of a group is kept: the groups, the links
//! and every figure are the same under all four. So a result carries one
//! `Verdict` per ranking (`Priority::ALL`), each exactly what `vid-fp -k
//! <that ranking>` would report, and the window switches between them without
//! scanning again.
//!
//! What the window needs that no report says is `Verdict::lost_to`: the files
//! a DELETE rests on. Before suggesting a deletion it asks whether one of them
//! is still there and unchanged, as the CLI could not, since it never runs
//! again after deciding.

use crate::compare::MatchIndex;
use crate::export;
use crate::fingerprint::VideoFingerprint;
use crate::utils::Priority;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Bumped whenever a field changes meaning; a result of another version is
/// not read.
pub const VERSION: u32 = 3;

/// The most frame pairs a comparison keeps, spread evenly along the member's
/// own runtime: enough to see every part of a match, few enough to show.
pub const MOST_PAIRS: usize = 48;

/// What the scan decided for a file, in every group it is in.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Keep,
    Delete,
    Review,
}

impl Action {
    /// As the report spells it.
    pub fn word(self) -> &'static str {
        match self {
            Action::Keep => "KEEP",
            Action::Delete => "DELETE",
            Action::Review => "REVIEW",
        }
    }
}

/// A file some group holds, with the figures the report gives it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct File {
    pub path: String,
    /// The length the scan measured, which every later action checks first.
    pub size: u64,
    /// Seconds; 0 when the container never said.
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    /// FFmpeg's short name for the codec.
    pub codec: String,
    /// Frames a second; 0 when the container never said.
    pub frame_rate: f64,
    /// Bits a second, over the whole file.
    pub bitrate: u64,
    /// Bits spent on an average frame; 0 when unknown.
    pub quality: u64,
    /// Hashes the fingerprint holds.
    pub samples: usize,
    /// What each ranking makes of it, in `Priority::ALL`'s order.
    pub verdicts: Vec<Verdict>,
}

impl File {
    /// What ranking `p` makes of it.
    pub fn verdict(&self, p: Priority) -> &Verdict {
        &self.verdicts[p.index()]
    }
}

/// What one ranking makes of a file: its label, and what a DELETE rests on.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Verdict {
    pub action: Action,
    /// For a DELETE, every file it shares a group with that it was measured
    /// against and lost the ranking to (`export::Verdicts::lost_to`), by
    /// their place in `Found::files`. Empty for any other action.
    pub lost_to: Vec<usize>,
}

/// One file of one group, with its strongest link inside that group: the
/// figures the report's `matched_*` columns print for its row.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Member {
    /// Its place in `Found::files`.
    pub file: usize,
    /// The file of the group it shares the most of its own footage with.
    pub matched_with: Option<usize>,
    /// Whether `matched_with` is this file's bytes again rather than a match
    /// of its picture: the report's `identical` column. Defaults to false for
    /// a scan kept by a build that did not know, which is what it knew.
    #[serde(default)]
    pub identical: bool,
    /// How much of its own footage that is, in seconds; `None` when its own
    /// runtime is unknown.
    pub matched_seconds: Option<f64>,
    /// Where that footage starts and ends in it, in seconds.
    pub matched_from: Option<f64>,
    pub matched_to: Option<f64>,
    /// What it matched, against its strongest link (`matched_with`) and, when
    /// that is another file, against the group's representative: the frames
    /// the scan matched and where the matched footage lies in each.
    pub compared: Vec<Comparison>,
}

/// One measured pair as the scan matched it, seen from one member: the frames
/// of it that matched and the frames of the other file they matched, and the
/// stretches of footage each side matched. Times are milliseconds on each
/// fingerprint's own clock, from its first keyframe (`compare::Evidence`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Comparison {
    /// The other file, by its place in `Found::files`.
    pub with: usize,
    /// `(this file's ms, the other's ms, hash distance)`, in this file's
    /// order, at most `MOST_PAIRS` of them, spread evenly.
    pub pairs: Vec<(u32, u32, u32)>,
    /// How many matched frames there were before the spreading.
    pub matched_frames: usize,
    pub stretches: Vec<(u32, u32)>,
    pub other_stretches: Vec<(u32, u32)>,
    /// The tolerance it was measured at, `-d`: a distance is read against it.
    pub tolerance: u32,
}

impl Comparison {
    fn of(evidence: crate::compare::Evidence, with: usize, tolerance: u32) -> Comparison {
        let n = evidence.pairs.len();
        let pairs = if n <= MOST_PAIRS {
            evidence.pairs
        } else {
            // First and last kept, the rest evenly between.
            (0..MOST_PAIRS).map(|k| evidence.pairs[k * (n - 1) / (MOST_PAIRS - 1)]).collect()
        };
        Comparison { with, pairs, matched_frames: n, stretches: evidence.stretches_a, other_stretches: evidence.stretches_b, tolerance }
    }
}

/// A group: every member matched the first, its representative.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Group {
    pub members: Vec<Member>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Found {
    pub version: u32,
    /// Videos the scan found, and how many of them it fingerprinted and so
    /// compared.
    pub found: usize,
    pub compared: usize,
    /// Every file some group holds, in the order the groups first name them.
    pub files: Vec<File>,
    pub groups: Vec<Group>,
}

impl Default for Found {
    fn default() -> Self {
        Found { version: VERSION, found: 0, compared: 0, files: Vec::new(), groups: Vec::new() }
    }
}

/// One member of a group before its comparisons are made: the file, its
/// strongest link inside the group, and the files it is to be compared with.
type Row = (usize, Option<crate::compare::GroupLink>, Vec<usize>);

impl Found {
    /// The scan's groups, labelled as the report labels them under each
    /// ranking.
    pub(crate) fn of(groups: &[Vec<usize>], fingerprints: &[VideoFingerprint], matches: &MatchIndex, found: usize, tolerance: u32) -> Found {
        let rankings: Vec<export::Verdicts> = Priority::ALL.iter().map(|&p| export::verdicts(groups, fingerprints, matches, p)).collect();
        // Each fingerprint that some group holds, numbered as first met.
        let mut number: HashMap<usize, usize> = HashMap::new();
        let mut order: Vec<usize> = Vec::new();
        for &idx in groups.iter().flatten() {
            number.entry(idx).or_insert_with(|| {
                order.push(idx);
                order.len() - 1
            });
        }
        let files = order
            .iter()
            .map(|&idx| {
                let fp = &fingerprints[idx];
                let verdicts = rankings
                    .iter()
                    .map(|v| {
                        let action = if v.review.contains(&idx) {
                            Action::Review
                        } else if v.delete.binary_search(&idx).is_ok() {
                            Action::Delete
                        } else {
                            Action::Keep
                        };
                        let mut lost_to: Vec<usize> = v.lost_to.get(&idx).map_or_else(Vec::new, |w| w.iter().map(|w| number[w]).collect());
                        lost_to.sort_unstable();
                        Verdict { action, lost_to }
                    })
                    .collect();
                File {
                    path: fp.path.clone(),
                    size: fp.file_size,
                    duration: fp.duration.max(0.0),
                    width: fp.width,
                    height: fp.height,
                    codec: fp.codec.clone(),
                    frame_rate: fp.frame_rate,
                    bitrate: fp.bitrate(),
                    quality: fp.quality(),
                    samples: fp.valid_hashes.len(),
                    verdicts,
                }
            })
            .collect();
        // Each member's row and the files it is compared with, first, so the
        // run's bar can be told what the comparisons weigh before they run: a
        // comparison is a frame-by-frame measurement, about the product of the
        // two sample counts, and on a library of long videos it is seconds.
        let rows: Vec<Vec<Row>> = groups
            .par_iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&idx| {
                        // The report's row: the strongest link inside this
                        // group.
                        let best = matches.links_of(idx, group, fingerprints).into_iter().next();
                        // Against its best link and the representative, each
                        // once, never itself.
                        let mut partners: Vec<usize> = best.as_ref().map(|l| l.other).into_iter().collect();
                        if group[0] != idx && !partners.contains(&group[0]) {
                            partners.push(group[0]);
                        }
                        (idx, best, partners)
                    })
                    .collect()
            })
            .collect();
        let cost = |a: usize, b: usize| {
            crate::progress::evidence_cost(fingerprints[a].valid_hashes.len(), fingerprints[b].valid_hashes.len())
        };
        let total: u64 = rows.iter().flatten().flat_map(|(idx, _, partners)| partners.iter().map(|&o| cost(*idx, o))).sum();
        crate::progress::learn(|k| k.evidence = Some(total as f64));
        let done = crate::progress::begin_counted(
            crate::progress::Stage::Saving,
            total,
            rows.iter().map(|r| r.len() as u64).sum(),
        );
        let groups = rows
            .into_par_iter()
            .map(|members| Group {
                members: members
                    .into_iter()
                    .map(|(idx, best, partners)| {
                        // Its seconds blank where the runtime is.
                        let seconds_known = fingerprints[idx].duration > 0.0;
                        let weight: u64 = partners.iter().map(|&o| cost(idx, o)).sum();
                        let compared = partners
                            .into_iter()
                            .map(|o| Comparison::of(crate::compare::evidence(&fingerprints[idx], &fingerprints[o], tolerance), number[&o], tolerance))
                            .collect();
                        done.add(1, weight);
                        Member {
                            file: number[&idx],
                            compared,
                            matched_with: best.as_ref().map(|l| number[&l.other]),
                            identical: best.as_ref().is_some_and(|l| l.identical),
                            matched_seconds: best.as_ref().map(|l| l.matched_seconds).filter(|_| seconds_known),
                            matched_from: best.as_ref().and_then(|l| l.span).map(|s| s.start_seconds()),
                            matched_to: best.as_ref().and_then(|l| l.span).map(|s| s.end_seconds()),
                        }
                    })
                    .collect(),
            })
            .collect();
        Found { version: VERSION, found, compared: fingerprints.len(), files, groups }
    }

    /// A run that compared nothing: `found` videos, and `compared` of them
    /// fingerprinted.
    pub(crate) fn empty(found: usize, compared: usize) -> Found {
        Found { found, compared, ..Found::default() }
    }

    /// Written whole beside `path` and renamed over it, so the window never
    /// reads half of one.
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("part");
        let bytes = serde_json::to_vec(self)?;
        std::fs::write(&tmp, bytes).with_context(|| format!("Could not write the results for the window to {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("Could not write the results for the window to {}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Found> {
        let file = std::fs::File::open(path).with_context(|| format!("Could not open {}", path.display()))?;
        let found: Found = serde_json::from_reader(std::io::BufReader::new(file)).with_context(|| format!("Could not read {}", path.display()))?;
        if found.version != VERSION {
            anyhow::bail!("{} was written by another version of vid-fp", path.display());
        }
        Ok(found)
    }
}
