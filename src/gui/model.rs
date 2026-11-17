//! What the results page knows, apart from how it is drawn: the scan's groups,
//! what it suggests for each video, what is marked, and what has become of the
//! files since.
//!
//! **The suggestion is the scan's own.** Each file's KEEP, DELETE or REVIEW is
//! read from the scan's result (`Found`), which labels them with the code the
//! command line's report does (`export::verdicts`), so the window and
//! `vid-fp` agree on every file. The result carries the labels of every
//! ranking (`--priority`); `priority` is the one shown, and changing it
//! changes nothing else.
//!
//! **What changes it is what the scan could not see: the files afterwards.**
//! A DELETE rests on the files it was measured against and lost to
//! (`Verdict::lost_to`). When every one of them has gone, or changed since the
//! scan, nothing the scan measured holds what the file shows any more, and it
//! is suggested for REVIEW instead. A file changed since the scan is not the
//! file it judged, so it is never suggested for deletion either. Gone files
//! leave the page (`drop_gone`).
//!
//! Every file is known by its number in `Found::files`, the same in every
//! group it is in: a mark is on the file, so a video marked in one group is
//! marked in every group it is in.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use vid_fp::found::{Action, File, Found, Member};
use vid_fp::utils::Priority;

/// What has become of a file since the scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lapse {
    /// No longer there: moved to the Trash, or deleted another way.
    Gone,
    /// Another length, or changed since the scan began.
    Changed,
}

/// A group as the page shows it: less the files gone since.
#[derive(Clone, Debug)]
pub struct GroupState {
    pub members: Vec<Member>,
    /// Its representative, the video every other was matched against, is
    /// gone.
    pub reference_gone: bool,
}

#[derive(Default)]
pub struct Model {
    pub found: Found,
    pub groups: Vec<GroupState>,
    /// For each file, the groups it is in.
    pub member_of: Vec<Vec<usize>>,
    lapsed: Vec<Option<Lapse>>,
    marked: Vec<bool>,
    pub marked_n: usize,
    pub marked_bytes: u64,
    /// The ranking whose suggestions are shown.
    priority: Priority,
}

impl Model {
    pub fn new(found: Found) -> Model {
        let n = found.files.len();
        let groups = found.groups.iter().map(|g| GroupState { members: g.members.clone(), reference_gone: false }).collect();
        let mut m = Model { found, groups, lapsed: vec![None; n], marked: vec![false; n], ..Model::default() };
        m.index();
        m
    }

    /// `member_of`, from the groups as they now stand.
    fn index(&mut self) {
        let mut member_of = vec![Vec::new(); self.n()];
        for (gi, g) in self.groups.iter().enumerate() {
            for m in &g.members {
                member_of[m.file].push(gi);
            }
        }
        self.member_of = member_of;
    }

    /// How many files the scan numbered.
    pub fn n(&self) -> usize {
        self.found.files.len()
    }

    pub fn file(&self, id: usize) -> &File {
        &self.found.files[id]
    }

    #[cfg(test)]
    pub fn priority(&self) -> Priority {
        self.priority
    }

    /// Show ranking `p`'s suggestions; whether that is a change.
    pub fn set_priority(&mut self, p: Priority) -> bool {
        std::mem::replace(&mut self.priority, p) != p
    }

    /// What the scan suggested for file `id` under the ranking shown.
    pub fn scanned(&self, id: usize) -> Action {
        self.file(id).verdict(self.priority).action
    }

    /// What file `id` lost to under the ranking shown: empty unless the scan
    /// suggested deleting it.
    pub fn lost_to(&self, id: usize) -> &[usize] {
        &self.file(id).verdict(self.priority).lost_to
    }

    pub fn path(&self, id: usize) -> &Path {
        Path::new(&self.found.files[id].path)
    }

    pub fn lapse_of(&self, id: usize) -> Option<Lapse> {
        self.lapsed.get(id).copied().flatten()
    }

    pub fn gone(&self, id: usize) -> bool {
        self.lapse_of(id) == Some(Lapse::Gone)
    }

    pub fn changed(&self, id: usize) -> bool {
        self.lapse_of(id) == Some(Lapse::Changed)
    }

    /// Why a DELETE no longer holds: every file it lost to has gone
    /// (`Gone`), or each has gone or changed (`Changed`). `None` for a file
    /// the scan did not suggest deleting, or one something still holds.
    pub fn unheld(&self, id: usize) -> Option<Lapse> {
        if self.scanned(id) != Action::Delete || self.lost_to(id).is_empty() {
            return None;
        }
        let lapses: Vec<Option<Lapse>> = self.lost_to(id).iter().map(|&w| self.lapse_of(w)).collect();
        if lapses.iter().any(Option::is_none) {
            return None;
        }
        Some(if lapses.iter().all(|l| *l == Some(Lapse::Gone)) { Lapse::Gone } else { Lapse::Changed })
    }

    /// What is suggested for file `id` now: the scan's own, except a
    /// deletion that nothing holds any more, or of a file changed since.
    pub fn action(&self, id: usize) -> Action {
        match self.scanned(id) {
            Action::Delete if self.changed(id) || self.unheld(id).is_some() => Action::Review,
            a => a,
        }
    }

    /// What is suggested for file `id`, in a few words.
    pub fn said(&self, id: usize) -> &'static str {
        match (self.scanned(id), self.action(id)) {
            (Action::Delete, Action::Review) if self.changed(id) => "REVIEW · changed since the scan",
            (Action::Delete, Action::Review) => match self.unheld(id) {
                Some(Lapse::Gone) => "REVIEW · what it lost to is gone",
                _ => "REVIEW · what it lost to has changed",
            },
            (_, Action::Keep) => "KEEP",
            (_, Action::Delete) => "DELETE",
            (_, Action::Review) => "REVIEW",
        }
    }

    /// Take in that `ids` have `lapse`d. Whether any suggestion may have
    /// moved.
    pub fn lapse(&mut self, ids: impl IntoIterator<Item = usize>, lapse: Lapse) -> bool {
        let mut moved = false;
        for id in ids {
            if let Some(l) = self.lapsed.get_mut(id) {
                // Gone outranks changed, and nothing comes back.
                if *l != Some(lapse) && *l != Some(Lapse::Gone) {
                    *l = Some(lapse);
                    moved = true;
                }
            }
        }
        moved
    }

    pub fn is_marked(&self, id: usize) -> bool {
        self.marked.get(id).copied().unwrap_or(false)
    }

    /// Mark or unmark file `id`; whether that changed anything.
    pub fn set_mark(&mut self, id: usize, on: bool) -> bool {
        if self.marked.get(id).copied().unwrap_or(on) == on {
            return false;
        }
        self.marked[id] = on;
        let b = self.file(id).size;
        if on {
            self.marked_n += 1;
            self.marked_bytes += b;
        } else {
            self.marked_n -= 1;
            self.marked_bytes -= b;
        }
        true
    }

    /// The numbers of the files marked, in order.
    pub fn marked_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.marked.iter().enumerate().filter(|(_, m)| **m).map(|(i, _)| i)
    }

    /// The files still there.
    pub fn live(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.n()).filter(|&id| !self.gone(id))
    }

    /// The files suggested for deletion now.
    pub fn suggested(&self) -> impl Iterator<Item = usize> + '_ {
        self.live().filter(|&id| self.action(id) == Action::Delete)
    }

    /// The groups, by the number the page shows them under, in which every
    /// video is marked: moved together, no copy of what they show is left.
    pub fn whole_groups_marked(&self) -> Vec<usize> {
        self.groups.iter().enumerate().filter(|(_, g)| g.members.iter().all(|m| self.is_marked(m.file))).map(|(gi, _)| gi + 1).collect()
    }

    /// Whether a file the scan suggested deleting would still have something
    /// holding it once the marked files are gone: a file it lost to that is
    /// unmarked and as the scan found it, or one reached the same way through
    /// files that are marked too.
    ///
    /// Through marked files, because the scan's DELETEs chain: a lost to b and
    /// b lost to c, so both go and c stays, though a was never measured
    /// against c. Each step is a direct measurement and the command line
    /// deletes both on that sum, so the window does not count a as orphaned
    /// while c is kept.
    pub fn held_after(&self, id: usize) -> bool {
        let mut seen = HashSet::from([id]);
        let mut todo = self.lost_to(id).to_vec();
        while let Some(w) = todo.pop() {
            if !seen.insert(w) || self.lapse_of(w).is_some() {
                continue;
            }
            if !self.is_marked(w) {
                return true;
            }
            todo.extend(self.lost_to(w).iter().copied());
        }
        false
    }

    /// The marked files the scan suggested deleting whose every holder is
    /// marked as well, or has gone or changed: moved together, nothing the
    /// scan found would be left holding what they show.
    pub fn marked_unheld(&self) -> usize {
        self.marked_ids().filter(|&id| self.scanned(id) == Action::Delete && !self.held_after(id)).count()
    }

    /// The files of the group, by number, that a marked file would be
    /// checked against just before it is moved: what it lost to, less what
    /// is marked too (that the confirmation said).
    pub fn holders(&self, id: usize) -> Vec<usize> {
        if self.scanned(id) != Action::Delete {
            return Vec::new();
        }
        self.lost_to(id).iter().copied().filter(|&w| !self.is_marked(w)).collect()
    }

    /// Take files that are no longer there off the page: unmarked, out of
    /// their groups, and a group left with one file gone. `current`, the
    /// group shown, is where it lands once the emptied groups before it go.
    /// Whether any mark was taken off.
    pub fn drop_gone(&mut self, gone: &HashSet<PathBuf>, current: &mut usize) -> bool {
        let ids: Vec<usize> = (0..self.n()).filter(|&id| gone.contains(self.path(id))).collect();
        let mut unmarked = false;
        for &id in &ids {
            unmarked |= self.set_mark(id, false);
        }
        self.lapse(ids.iter().copied(), Lapse::Gone);
        for g in self.groups.iter_mut() {
            if g.members.first().is_some_and(|m| ids.contains(&m.file)) {
                g.reference_gone = true;
            }
            g.members.retain(|m| !ids.contains(&m.file));
        }
        let before = self.groups[..(*current).min(self.groups.len())].iter().filter(|g| g.members.len() < 2).count();
        self.groups.retain(|g| g.members.len() >= 2);
        *current = current.saturating_sub(before);
        self.index();
        unmarked
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vid_fp::found::{Group, Verdict};

    /// A file every ranking labels alike.
    fn file(name: &str, action: Action, lost_to: &[usize]) -> File {
        ranked(name, [(action, lost_to); 4])
    }

    /// A file labelled per ranking, in `Priority::ALL`'s order.
    fn ranked(name: &str, verdicts: [(Action, &[usize]); 4]) -> File {
        File {
            path: format!("/v/{name}"),
            size: 100,
            duration: 60.0,
            width: 1920,
            height: 1080,
            codec: "h264".into(),
            frame_rate: 25.0,
            bitrate: 1000,
            quality: 40,
            samples: 30,
            verdicts: verdicts.iter().map(|&(action, lost_to)| Verdict { action, lost_to: lost_to.to_vec() }).collect(),
        }
    }

    fn member(file: usize) -> Member {
        Member { file, matched_with: None, identical: false, matched_seconds: None, matched_from: None, matched_to: None, compared: Vec::new() }
    }

    /// a (KEEP) <- b (DELETE, lost to a) <- c (DELETE, lost to b): a chain,
    /// in two groups, as the scan labels one.
    fn chain() -> Model {
        let files = vec![file("a.mkv", Action::Keep, &[]), file("b.mkv", Action::Delete, &[0]), file("c.mkv", Action::Delete, &[1])];
        let groups = vec![Group { members: vec![member(0), member(1)] }, Group { members: vec![member(1), member(2)] }];
        Model::new(Found { files, groups, found: 3, compared: 3, ..Found::default() })
    }

    #[test]
    fn the_suggestion_is_the_scans_until_what_it_rests_on_lapses() {
        let mut m = chain();
        assert_eq!((m.action(0), m.action(1), m.action(2)), (Action::Keep, Action::Delete, Action::Delete));
        assert_eq!(m.suggested().collect::<Vec<_>>(), [1, 2]);
        // b changes: it is not the file judged, and c no longer rests on it.
        assert!(m.lapse([1], Lapse::Changed));
        assert_eq!(m.action(1), Action::Review);
        assert_eq!(m.said(1), "REVIEW · changed since the scan");
        assert_eq!(m.action(2), Action::Review);
        assert_eq!(m.said(2), "REVIEW · what it lost to has changed");
        // Telling it twice is no news.
        assert!(!m.lapse([1], Lapse::Changed));
        // Gone outranks changed.
        assert!(m.lapse([1], Lapse::Gone));
        assert_eq!(m.said(2), "REVIEW · what it lost to is gone");
        assert!(!m.lapse([1], Lapse::Changed), "nothing comes back from gone");
    }

    #[test]
    fn a_deletion_held_by_any_one_file_still_stands() {
        let files = vec![file("a", Action::Keep, &[]), file("b", Action::Keep, &[]), file("c", Action::Delete, &[0, 1])];
        let groups = vec![Group { members: vec![member(0), member(1), member(2)] }];
        let mut m = Model::new(Found { files, groups, ..Found::default() });
        m.lapse([0], Lapse::Gone);
        assert_eq!(m.action(2), Action::Delete, "b still holds it");
        m.lapse([1], Lapse::Changed);
        assert_eq!(m.action(2), Action::Review);
    }

    /// Marks are on the file and counted with its size; marking it twice
    /// counts it once.
    #[test]
    fn a_mark_is_counted_once() {
        let mut m = chain();
        assert!(m.set_mark(1, true));
        assert!(!m.set_mark(1, true));
        assert_eq!((m.marked_n, m.marked_bytes), (1, 100));
        assert!(m.set_mark(1, false));
        assert_eq!((m.marked_n, m.marked_bytes), (0, 0));
    }

    /// A chain marked as the scan suggests leaves its head, and nothing is
    /// orphaned; mark the head too and every one of them is.
    #[test]
    fn a_marked_chain_is_held_by_the_file_it_ends_at() {
        let mut m = chain();
        m.set_mark(1, true);
        m.set_mark(2, true);
        assert_eq!(m.marked_unheld(), 0);
        assert!(m.whole_groups_marked() == [2], "b and c are the whole of the second group");
        m.set_mark(0, true);
        assert_eq!(m.marked_unheld(), 2);
        assert_eq!(m.whole_groups_marked(), [1, 2]);
        // b's holder is a, unless a is marked.
        assert!(m.holders(1).is_empty());
        m.set_mark(0, false);
        assert_eq!(m.holders(1), [0]);
        assert!(m.holders(0).is_empty(), "a KEEP rests on nothing");
    }

    /// A file gone leaves its groups, unmarked; a group left with one video
    /// goes, and the group shown lands where it now is.
    #[test]
    fn a_file_gone_leaves_the_page() {
        let mut m = chain();
        m.set_mark(0, true);
        let mut current = 1;
        assert!(m.drop_gone(&HashSet::from([PathBuf::from("/v/a.mkv")]), &mut current));
        assert_eq!(m.groups.len(), 1);
        assert_eq!(current, 0);
        assert_eq!(m.marked_n, 0);
        assert!(m.member_of[0].is_empty());
        assert_eq!(m.member_of[1], [0]);
        assert_eq!(m.action(1), Action::Review, "b lost only to a");
        // c is still held by b, which is there.
        assert_eq!(m.action(2), Action::Delete);
        // The second group's representative goes.
        assert!(!m.groups[0].reference_gone);
        let files = vec![file("a", Action::Keep, &[]), file("b", Action::Delete, &[0]), file("c", Action::Delete, &[0])];
        let groups = vec![Group { members: vec![member(0), member(1), member(2)] }];
        let mut m = Model::new(Found { files, groups, ..Found::default() });
        let mut current = 0;
        m.drop_gone(&HashSet::from([PathBuf::from("/v/a")]), &mut current);
        assert!(m.groups[0].reference_gone);
        assert_eq!(m.groups[0].members.len(), 2);
    }

    /// The ranking shown decides every suggestion and what each deletion
    /// rests on; marks and lapses are the files' and stay as they were.
    #[test]
    fn the_ranking_shown_is_the_one_asked_of_the_scan() {
        // By length a is kept and b goes; by resolution the other way round.
        let files = vec![
            ranked("a", [(Action::Keep, &[]), (Action::Delete, &[1]), (Action::Keep, &[]), (Action::Keep, &[])]),
            ranked("b", [(Action::Delete, &[0]), (Action::Keep, &[]), (Action::Delete, &[0]), (Action::Delete, &[0])]),
        ];
        let groups = vec![Group { members: vec![member(0), member(1)] }];
        let mut m = Model::new(Found { files, groups, ..Found::default() });
        assert_eq!(m.priority(), Priority::Length);
        assert_eq!(m.suggested().collect::<Vec<_>>(), [1]);
        m.set_mark(1, true);
        assert!(m.set_priority(Priority::Resolution));
        assert!(!m.set_priority(Priority::Resolution), "no change");
        assert_eq!(m.suggested().collect::<Vec<_>>(), [0]);
        assert_eq!(m.holders(0), Vec::<usize>::new(), "a rests on b, which is marked");
        assert_eq!(m.marked_unheld(), 0, "b is not a deletion under this ranking");
        assert!(m.is_marked(1));
        m.lapse([1], Lapse::Gone);
        assert_eq!(m.action(0), Action::Review, "what a lost to is gone");
        m.set_priority(Priority::Length);
        assert_eq!(m.action(0), Action::Keep);
    }
}
