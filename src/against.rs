//! `--against LIBRARY`: each scanned file answered against a library that is
//! never touched.
//!
//! The question is "which of these do I already have?", and every scanned file
//! gets one of four answers ([`Answer`]). It is the ordinary pipeline with two
//! differences, both decided here: only scanned-to-library pairs are compared
//! (`compare::find_matches_across`), and the labels are not elected over
//! groups but read off each scanned file's own links into the library, with
//! every library file pinned as KEEP.
//!
//! The answers are expressed in the report's own words so nothing downstream
//! needed a second vocabulary: a `repeat` is DELETE, so `--delete`, the
//! confirmation and `--from-report` act on exactly the repeats; a `review` is
//! REVIEW; `better` and `new` are KEEP, told apart by whether the row names a
//! library file it matched.

use std::collections::{HashMap, HashSet};

use crate::compare::Match;
use crate::export::{judge, Verdicts};
use crate::fingerprint::VideoFingerprint;
use crate::utils::{outranks, GroupMaxima, Priority};

/// What the library makes of one scanned file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, clap::ValueEnum)]
pub enum Answer {
    /// The library holds this footage at least as well: a library file it
    /// matched ranks level with it or above it.
    Repeat,
    /// It matched the library, and ranks above every library file it matched:
    /// the library's copy is the one worth replacing.
    Better,
    /// It matched, and the ranking will not settle it -- a different codec at
    /// the same length and resolution, a runtime nothing could measure -- the
    /// same cases a scan holds for REVIEW.
    Review,
    /// Nothing in the library matched it.
    New,
}

/// Every scanned file's answer, and the groups and labels the report is
/// written from.
pub struct Answered {
    /// One group per scanned file, the file first and then the library files
    /// it matched, most of its footage first. A file nothing matched is a group
    /// of one, so the report lists it.
    pub groups: Vec<Vec<usize>>,
    pub verdicts: Verdicts,
    /// Each scanned file and its answer, in the order of `groups`.
    pub answers: Vec<(usize, Answer)>,
}

/// Answer every scanned file (`library[i]` false) against the library files
/// (`library[i]` true) it was measured against.
///
/// A file's answer rests on its own links only. Two scanned files that match
/// each other are not compared at all, so a folder holding two copies of
/// something the library lacks answers `new` twice: the question asked is about
/// the library, and an import folder's own duplicates are what an ordinary scan
/// of it is for.
///
/// The ranking is the one a scan uses, run over the file and the library files
/// it matched (`export::judge` for the REVIEW cases, `utils::outranks` for the
/// rest), so a clip of a library video is a repeat because its host is longer,
/// and a host of a library clip is better for the same reason.
pub fn answer(
    fingerprints: &[VideoFingerprint],
    matches: &[Match],
    library: &[bool],
    priority: Priority,
) -> Answered {
    // Each scanned file's library links, with how much of the scanned file
    // each one accounts for.
    let mut links: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
    for m in matches {
        let (scanned, held, coverage) = match (library[m.a], library[m.b]) {
            (false, true) => (m.a, m.b, m.coverage_a),
            (true, false) => (m.b, m.a, m.coverage_b),
            // Not a question this mode asks; `find_matches_across` does not
            // propose such pairs, so this only guards a caller that did.
            _ => continue,
        };
        links.entry(scanned).or_default().push((held, coverage));
    }

    let mut scanned: Vec<usize> = (0..fingerprints.len()).filter(|&i| !library[i]).collect();
    scanned.sort_by(|&a, &b| fingerprints[a].path.cmp(&fingerprints[b].path));

    let mut groups = Vec::with_capacity(scanned.len());
    let mut answers = Vec::with_capacity(scanned.len());
    let mut review: HashSet<usize> = HashSet::new();
    let mut delete: Vec<usize> = Vec::new();
    let mut lost_to: HashMap<usize, Vec<usize>> = HashMap::new();

    for s in scanned {
        let mut held = links.remove(&s).unwrap_or_default();
        held.sort_by(|x, y| {
            y.1.partial_cmp(&x.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| fingerprints[x.0].path.cmp(&fingerprints[y.0].path))
        });
        let group: Vec<usize> = std::iter::once(s).chain(held.iter().map(|&(h, _)| h)).collect();

        let answer = if held.is_empty() {
            Answer::New
        } else {
            let (_, flagged) = judge(&group, fingerprints, priority);
            let maxima = GroupMaxima::of(&group, fingerprints);
            let level_or_above: Vec<usize> = group[1..]
                .iter()
                .copied()
                .filter(|&h| !outranks(&fingerprints[s], &fingerprints[h], priority, &maxima))
                .collect();
            if flagged.contains(&s) {
                Answer::Review
            } else if level_or_above.is_empty() {
                Answer::Better
            } else {
                lost_to.insert(s, level_or_above);
                Answer::Repeat
            }
        };

        match answer {
            Answer::Repeat => delete.push(s),
            Answer::Review => {
                review.insert(s);
            }
            Answer::Better | Answer::New => {}
        }
        groups.push(group);
        answers.push((s, answer));
    }

    for winners in lost_to.values_mut() {
        winners.sort_unstable();
    }
    delete.sort_unstable();

    Answered { groups, verdicts: Verdicts { review, delete, lost_to }, answers }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::Match;

    fn fp(path: &str, duration: f64, height: u32, codec: &str) -> VideoFingerprint {
        VideoFingerprint {
            path: path.to_string(),
            valid_hashes: vec![1],
            valid_t_start: vec![0],
            valid_t_end: vec![1],
            total_ms: (duration * 1000.0) as u32,
            width: height * 16 / 9,
            height,
            duration,
            file_size: (duration * 131_072.0) as u64 * height as u64 / 720,
            codec: codec.to_string(),
            frame_rate: 30.0,
            mirrored_hashes: Vec::new(),
            detail_hashes: Vec::new(),
            mirrored_detail_hashes: Vec::new(),
            reframed: Vec::new(),
        }
    }

    fn link(a: usize, b: usize, coverage_a: f32, coverage_b: f32) -> Match {
        Match { a, b, coverage_a, coverage_b, span_a: None, span_b: None, mirrored: false, reframed: None }
    }

    fn answers_of(fps: &[VideoFingerprint], matches: &[Match], library: &[bool]) -> Vec<Answer> {
        let answered = answer(fps, matches, library, Priority::Length);
        let by_file: HashMap<usize, Answer> = answered.answers.into_iter().collect();
        (0..fps.len()).filter(|&i| !library[i]).map(|i| by_file[&i]).collect()
    }

    #[test]
    fn test_each_answer_is_reached_and_the_library_is_never_condemned() {
        let fps = vec![
            fp("/lib/episode.mkv", 1200.0, 1080, "h264"),
            fp("/new/same.mkv", 1200.0, 1080, "h264"),
            fp("/new/clip.mkv", 120.0, 1080, "h264"),
            fp("/new/sharper.mkv", 1200.0, 2160, "h264"),
            fp("/new/other_codec.mkv", 1200.0, 1080, "av1"),
            fp("/new/unrelated.mkv", 300.0, 1080, "h264"),
        ];
        let library = [true, false, false, false, false, false];
        let matches = vec![
            link(1, 0, 1.0, 1.0),
            link(2, 0, 1.0, 0.1),
            link(3, 0, 1.0, 1.0),
            link(4, 0, 1.0, 1.0),
        ];
        let got = answers_of(&fps, &matches, &library);
        // In file order: same, clip, sharper, other_codec, unrelated.
        assert_eq!(
            got,
            vec![Answer::Repeat, Answer::Repeat, Answer::Better, Answer::Review, Answer::New],
            "same and clip are held; sharper outranks it; a codec standoff is REVIEW"
        );

        let answered = answer(&fps, &matches, &library, Priority::Length);
        assert!(!answered.verdicts.delete.contains(&0) && !answered.verdicts.review.contains(&0));
        assert_eq!(answered.verdicts.delete, vec![1, 2]);
        assert_eq!(answered.verdicts.lost_to[&1], vec![0]);
    }

    #[test]
    fn test_a_host_holding_a_library_clip_is_better_not_a_repeat() {
        // The other way round from the clip above: deleting this one would lose
        // everything the library's clip does not hold.
        let fps = vec![fp("/lib/clip.mkv", 120.0, 1080, "h264"), fp("/new/host.mkv", 1200.0, 1080, "h264")];
        let got = answers_of(&fps, &[link(1, 0, 0.1, 1.0)], &[true, false]);
        assert_eq!(got, vec![Answer::Better]);
    }

    #[test]
    fn test_scanned_files_are_answered_by_the_library_alone() {
        let fps = vec![fp("/new/a.mkv", 60.0, 1080, "h264"), fp("/new/b.mkv", 60.0, 720, "h264")];
        // Even if a caller hands over a pair between two scanned files.
        let got = answers_of(&fps, &[link(0, 1, 1.0, 1.0)], &[false, false]);
        assert_eq!(got, vec![Answer::New, Answer::New]);
    }

    #[test]
    fn test_a_file_is_grouped_with_the_library_files_it_matched_most_first() {
        let fps = vec![
            fp("/lib/a.mkv", 600.0, 1080, "h264"),
            fp("/lib/b.mkv", 600.0, 1080, "h264"),
            fp("/new/x.mkv", 600.0, 720, "h264"),
        ];
        let answered =
            answer(&fps, &[link(2, 0, 0.3, 0.3), link(2, 1, 0.9, 0.9)], &[true, true, false], Priority::Length);
        assert_eq!(answered.groups, vec![vec![2, 1, 0]]);
        assert_eq!(answered.verdicts.lost_to[&2], vec![0, 1]);
    }
}
