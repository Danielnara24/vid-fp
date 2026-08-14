//! Turning "these two videos match" edges into groups.
//!
//! A group is a *maximal clique*: every file in it matched every other file in
//! it, and no file outside it matched all of them. That definition is the whole
//! safety argument for what `export.rs` does next -- it keeps one file per group
//! and marks the rest DELETE, which is only defensible if every DELETE target
//! was independently confirmed against every other member, including the one
//! being kept.
//!
//! Anything looser breaks that argument, because matching here is not
//! transitive. Three episodes sharing an opening sequence are pairwise linked
//! without any two of them being duplicates; a clip cut from a long video links
//! to the host without linking to the host's other clips. Merging on partial
//! connectivity -- connected components, or "close enough" expansion rules --
//! collapses exactly those cases into one group and then has to hold most of it
//! back for review, because the ranking inside such a group compares files that
//! were never measured against each other. Requiring a complete subgraph makes
//! that impossible to express: inside a clique, the loser of a ranking has by
//! construction been compared with the winner.
//!
//! Whether an edge exists at all is decided upstream by `--hamming-distance`,
//! `--match-percent` and `--min-duration`. This module adds no thresholds of its
//! own; it only refuses to invent links that the comparison stage did not find.
//!
//! The price of being strict is that groups overlap: a file belonging to two
//! cliques is reported in both, so the report is a list of relationships rather
//! than a partition of the library. That is expected -- `export.rs` resolves
//! each file's fate ONCE, across every group it appears in, and prints the same
//! verdict on each of its rows.

use crate::fingerprint::VideoFingerprint;
use crate::stats::RunStats;
use crate::utils::shutdown_requested;
use std::collections::HashSet;

/// How much of the clique search ONE connected component may cost, counted in
/// set probes -- the work each recursion step is about to do, summed over the
/// whole search. See `step_cost`, which is where the counting is defined.
///
/// Maximal-clique enumeration is 3^(n/3) in the worst case, and the worst case
/// is reachable from the command line: a loose `--hamming-distance` over a large
/// library links nearly everything to nearly everything, and unrelated hashes
/// sit ~32 bits apart, so a tolerance near chance on a big folder is a dense
/// graph by construction rather than a pathological input someone had to
/// construct. Left alone that is a hang, and the memory it climbs through on the
/// way is not bounded by anything either.
///
/// One budget covers all three failure modes because they share a cause. A probe
/// is a hash lookup, so the time is what is being counted directly; the sets a
/// step holds are never larger than the work it does on them, so the heap along
/// the current path is bounded by the same number; and depth is bounded
/// underneath both, because reaching depth `d` means `p` held at least `d - i`
/// members at each level `i` above.
///
/// Fifty million is a second or two of a single core and well under a gigabyte
/// of live sets. It also bounds the depth tightly, which is what keeps the stack
/// out of it: a clique of `d` members costs about `d^3 / 3` probes to enumerate
/// (at level `i` every one of the `d - i` remaining candidates is intersected
/// with the rest), so no search that fits in this budget can recurse past ~530
/// frames.
///
/// Measured against the local 756-file corpus, probes spent by the largest
/// component: 5.4k at the default `-d 4`, 5.6k at `-d 12`, 75k at `-d 16`, and
/// 1.03M at `-d 18` -- the loosest setting whose 9,003 groups are still a
/// report rather than a phenomenon. That is a 48x margin over the last useful
/// rung, and the first setting to exhaust the budget is `-d 32`, where the graph
/// is so nearly complete that the search cannot finish a single clique.
const SEARCH_BUDGET_PER_COMPONENT: usize = 50_000_000;

/// How many groups one connected component may produce before it is abandoned.
///
/// The budget above bounds the search; this bounds its OUTPUT, which is a
/// separate quantity -- every emitted clique costs one step, so a search that
/// spends its whole budget on emitting can still return millions of groups, and
/// they are held in memory (and then printed) all at once.
///
/// A hundred thousand overlapping groups out of one component is not a report
/// anybody can act on; it is the same "your tolerance is too loose" answer the
/// budget gives, arriving through the other door.
const MAX_GROUPS_PER_COMPONENT: usize = 100_000;

/// Why a component's search stopped early. Any of these means the groups found
/// so far are incomplete, and incomplete groups are dropped rather than reported
/// -- see `find_duplicate_groups`.
enum Halt {
    /// The search ran past `SEARCH_BUDGET_PER_COMPONENT`.
    Budget,
    /// It produced more than `MAX_GROUPS_PER_COMPONENT` cliques.
    Groups,
    /// Ctrl-C.
    Interrupted,
}

impl Halt {
    /// What stopped the search, phrased for the Problems summary. `Interrupted`
    /// never reaches it -- `RunStats` deliberately counts nothing that Ctrl-C
    /// dropped -- but it is spelled out anyway rather than made unreachable.
    fn ceiling(&self) -> &'static str {
        match self {
            Halt::Budget => "the search for groups ran past its work ceiling",
            Halt::Groups => "they form more overlapping groups than can be reported",
            Halt::Interrupted => "the run was interrupted",
        }
    }
}

/// The line the Problems summary shows for an abandoned component: how many
/// files were in it, which ceiling stopped it, and one of them by name, so the
/// user has somewhere to look. The lowest path rather than an arbitrary member,
/// because the summary should read the same on a re-run.
fn abandoned_detail(component: &[usize], reason: &Halt, fps: &[VideoFingerprint]) -> String {
    let example = component
        .iter()
        .map(|&i| fps[i].path.as_str())
        .min()
        .unwrap_or("?");

    format!(
        "{} file(s) linked too densely to group -- {}; e.g. {}",
        component.len(),
        reason.ceiling(),
        example
    )
}

/// What one component's search is allowed to spend.
///
/// The two constants above, in a struct, so the abandonment path can be
/// exercised in a millisecond instead of by building a graph big enough to
/// exhaust twenty million slots -- a test of the ceilings should not have to be
/// as expensive as the thing they exist to prevent.
#[derive(Clone, Copy)]
struct Ceilings {
    /// Candidate slots the search may examine.
    budget: usize,
    /// Groups it may emit.
    groups: usize,
}

impl Ceilings {
    const fn shipped() -> Self {
        Ceilings {
            budget: SEARCH_BUDGET_PER_COMPONENT,
            groups: MAX_GROUPS_PER_COMPONENT,
        }
    }
}

/// One component's clique search, and the ceilings it runs under.
struct Search<'a> {
    adjacency: &'a [HashSet<usize>],
    cliques: Vec<HashSet<usize>>,
    ceilings: Ceilings,
    /// Candidate slots left; the search stops when it runs out.
    budget: usize,
    halted: Option<Halt>,
}

/// What the step about to run will cost, in set probes.
///
/// The dominant term is pivot selection: for every candidate in `p ∪ x` it
/// intersects that vertex's neighbours with `p`, and `HashSet::intersection`
/// walks the smaller of the two -- so one candidate costs `min(deg(v), |p|)`
/// probes rather than one. Charging the flat size of the sets undercounts a
/// dense component by a factor of `|p|`, and that is not a rounding error: a
/// NEARLY complete graph has few enough maximal cliques to stay under the group
/// ceiling while every one of its steps carries hundreds of thousands of probes,
/// so it used to spend minutes inside a budget that believed it had spent
/// thousands. (`-d 40` on the local corpus, before `--hamming-distance` was
/// capped at chance: eleven minutes and climbing, single-threaded, uninterrupted
/// by either ceiling.)
///
/// The branch loop and the two intersections it does per branch are the same
/// shape as the pivot walk and within a small constant of it, so they ride on
/// this rather than being counted separately. Pricing the walk costs a walk, but
/// a linear one, ahead of a step whose own work is the product.
fn step_cost(p: &HashSet<usize>, x: &HashSet<usize>, adjacency: &[HashSet<usize>]) -> usize {
    let cap = p.len();
    // The `1` is what a step that examines nothing still costs, so a search
    // cannot recurse for free.
    1 + p
        .union(x)
        .map(|&v| adjacency[v].len().min(cap))
        .sum::<usize>()
}

impl<'a> Search<'a> {
    fn new(adjacency: &'a [HashSet<usize>], ceilings: Ceilings) -> Self {
        Search {
            adjacency,
            cliques: Vec::new(),
            ceilings,
            budget: ceilings.budget,
            halted: None,
        }
    }

    /// Charge one recursion step and check every ceiling. `false` means the
    /// search has stopped and the caller should unwind.
    fn may_continue(&mut self, cost: usize) -> bool {
        if self.halted.is_some() {
            return false;
        }
        // Polled here rather than in the branch loop because this runs once per
        // recursion step, which is the finest granularity the search has -- a
        // single step is a handful of set operations, so an interrupt is
        // answered immediately even on a component that would otherwise run for
        // minutes. It is a relaxed atomic load against work that clones sets.
        if shutdown_requested() {
            self.halted = Some(Halt::Interrupted);
            return false;
        }
        if self.cliques.len() > self.ceilings.groups {
            self.halted = Some(Halt::Groups);
            return false;
        }
        let Some(left) = self.budget.checked_sub(cost) else {
            self.halted = Some(Halt::Budget);
            return false;
        };
        self.budget = left;
        true
    }
}

/// Bron-Kerbosch with Tomita pivoting.
///
/// `r` is the clique built so far, `p` the candidates that could still extend
/// it, and `x` the candidates already used as an extension somewhere higher in
/// the tree. A clique is maximal exactly when nothing can extend it (`p` empty)
/// and nothing was deliberately excluded from it (`x` empty). `x` is what stops
/// the same clique being emitted once per ordering of its members.
///
/// The pivot is what keeps this tractable on dense neighbourhoods. Every
/// maximal clique must contain the pivot or one of its non-neighbours, so
/// branching on only the non-neighbours skips whole subtrees that could not
/// produce anything new. Picking the pivot with the most neighbours still in
/// `p` skips the most.
///
/// Every step is charged against the search's budget and checked against its
/// ceilings before it does any work, so a component that turns out to be dense
/// enough to enumerate forever stops instead -- and so Ctrl-C is answered here
/// like it is in every other stage. See `Search`.
fn bron_kerbosch(
    r: HashSet<usize>,
    mut p: HashSet<usize>,
    mut x: HashSet<usize>,
    search: &mut Search,
) {
    let cost = step_cost(&p, &x, search.adjacency);
    if !search.may_continue(cost) {
        return;
    }

    if p.is_empty() {
        // Nothing left to add. If nothing was excluded either, `r` is maximal.
        // A group of one is not a duplicate of anything, so it is dropped here
        // rather than filtered out downstream.
        if x.is_empty() && r.len() > 1 {
            search.cliques.push(r);
        }
        return;
    }

    let adjacency = search.adjacency;

    // `p` is non-empty, so the union is too and this always yields a pivot; the
    // fallback below is only there to keep the function total.
    let pivot = p
        .union(&x)
        .max_by_key(|&&v| adjacency[v].intersection(&p).count())
        .copied();

    let branches: Vec<usize> = match pivot {
        Some(u) => p.difference(&adjacency[u]).copied().collect(),
        None => p.iter().copied().collect(),
    };

    for v in branches {
        let neighbors = &adjacency[v];

        let mut next_r = r.clone();
        next_r.insert(v);

        bron_kerbosch(
            next_r,
            neighbors.intersection(&p).copied().collect(),
            neighbors.intersection(&x).copied().collect(),
            search,
        );

        // Unwind the whole tree at once rather than one level per iteration:
        // every remaining branch would only be refused by `may_continue`, and
        // on a wide `p` that refusal is itself thousands of steps.
        if search.halted.is_some() {
            return;
        }

        // `v` has now appeared in every clique it can belong to at this level.
        // Moving it into `x` stops the branches to its right rediscovering
        // those same cliques with `v` appended.
        p.remove(&v);
        x.insert(v);
    }
}

/// The linked files, split into groups that cannot possibly share a clique.
///
/// A maximal clique lies entirely inside one connected component, so searching
/// each component separately finds exactly the same cliques as one search over
/// the whole graph. What it buys is that the ceilings above become LOCAL: one
/// dense folder is abandoned on its own instead of consuming a budget the rest
/// of the library then runs out of, and the report still carries every group
/// that was cheap to find.
///
/// Isolated files are left out entirely. Only a linked file can be in a group of
/// two or more, and a file with no edges cannot extend anyone else's clique
/// either, so seeding one would spend a recursion (plus a clone of an n-sized
/// set) to rediscover that -- and on a real library the overwhelming majority of
/// videos match nothing at all.
fn components(adjacency: &[HashSet<usize>]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; adjacency.len()];
    let mut out: Vec<Vec<usize>> = Vec::new();

    for v in 0..adjacency.len() {
        if seen[v] || adjacency[v].is_empty() {
            continue;
        }

        // Iterative rather than recursive: this walks every linked file in the
        // library, and it exists to keep an unbounded graph from overrunning
        // anything -- doing it with the stack would reintroduce exactly that.
        let mut stack = vec![v];
        seen[v] = true;
        let mut component = Vec::new();

        while let Some(u) = stack.pop() {
            component.push(u);
            for &w in &adjacency[u] {
                if !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }

        out.push(component);
    }

    out
}

/// Group `n` fingerprints into duplicate sets, given the pairs that matched.
///
/// Indices refer to positions in `fingerprints`. Within a group they are
/// ordered by path; the groups themselves are ordered largest first. A file that
/// matched nothing is in no group at all -- a group of one is not a duplicate
/// set, and dropping it here is what keeps the report to the files that are
/// actually implicated in something.
///
/// A component whose search hits one of the ceilings above is dropped WHOLE and
/// counted as a problem, so the run exits non-zero and says which files it could
/// not answer for. Reporting the part of it that was enumerated first would be
/// worse than saying nothing: which cliques those are depends on `HashSet`
/// iteration order, so the same library would group differently run to run, and
/// a `--delete` acting on half a component's relationships is a deletion made on
/// less evidence than the user asked for. Everything outside that component is
/// unaffected and reported as usual.
pub fn find_duplicate_groups(
    n: usize,
    edges: Vec<(usize, usize)>,
    fingerprints: &[VideoFingerprint],
    stats: &RunStats,
) -> Vec<Vec<usize>> {
    group_within(n, edges, fingerprints, stats, Ceilings::shipped())
}

/// `find_duplicate_groups` with the ceilings supplied. The run always passes
/// `Ceilings::shipped()`; only the tests pass anything else.
fn group_within(
    n: usize,
    edges: Vec<(usize, usize)>,
    fingerprints: &[VideoFingerprint],
    stats: &RunStats,
    ceilings: Ceilings,
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![HashSet::new(); n];
    for (i, j) in edges {
        adjacency[i].insert(j);
        adjacency[j].insert(i);
    }

    let mut cliques = Vec::new();

    for component in components(&adjacency) {
        let mut search = Search::new(&adjacency, ceilings);

        bron_kerbosch(
            HashSet::new(),
            component.iter().copied().collect(),
            HashSet::new(),
            &mut search,
        );

        match search.halted {
            None => cliques.append(&mut search.cliques),
            Some(Halt::Interrupted) => return Vec::new(),
            Some(reason) => stats
                .clustering_abandoned
                .record(abandoned_detail(&component, &reason, fingerprints)),
        }
    }

    // Note there is no subset-elimination pass here. Every clique above is
    // maximal by construction, and a maximal clique cannot be contained in
    // another one: the members making up the difference would have extended it.

    let mut groups: Vec<Vec<usize>> = cliques
        .into_iter()
        .map(|clique| {
            let mut indices: Vec<usize> = clique.into_iter().collect();
            indices.sort_by(|&a, &b| fingerprints[a].path.cmp(&fingerprints[b].path));
            indices
        })
        .collect();

    // Discovery order follows HashSet iteration, which is not stable across
    // runs, so the output is sorted into a total order instead: largest groups
    // first, ties broken by comparing member paths in order. Paths are unique
    // (files are deduplicated by inode before they reach this point) and no two
    // maximal cliques hold the same members, so no two distinct groups can
    // compare equal and the result is fully reproducible.
    groups.sort_by(|a, b| {
        b.len().cmp(&a.len()).then_with(|| {
            a.iter()
                .map(|&i| &fingerprints[i].path)
                .cmp(b.iter().map(|&i| &fingerprints[i].path))
        })
    });

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create dummy fingerprints for testing. Clustering only
    // ever looks at paths, so the codec and frame rate are just plausible
    // filler -- the rules that care about them live in utils and export.
    fn mock_fingerprint(path: &str) -> VideoFingerprint {
        VideoFingerprint {
            path: path.to_string(),
            valid_hashes: vec![],
            valid_t_start: vec![],
            valid_t_end: vec![],
            total_ms: 100_000,
            width: 1920,
            height: 1080,
            duration: 10.0,
            file_size: 1024,
            codec: "h264".to_string(),
            frame_rate: 30.0,
        }
    }

    /// `n` fingerprints whose paths sort in index order, so a group printed as
    /// `[0, 1, 2]` is also the path order the real sort would produce.
    fn mock_library(n: usize) -> Vec<VideoFingerprint> {
        (0..n)
            .map(|i| mock_fingerprint(&format!("{:04}.mp4", i)))
            .collect()
    }

    /// The groups, for a graph that is not supposed to trouble the ceilings.
    ///
    /// Every test that isn't about them goes through here, so all of them assert
    /// the same thing in passing: an ordinary graph is enumerated whole, and a
    /// run over one has nothing to report as a problem.
    fn groups_of(
        n: usize,
        edges: Vec<(usize, usize)>,
        fps: &[VideoFingerprint],
    ) -> Vec<Vec<usize>> {
        let stats = RunStats::default();
        let groups = find_duplicate_groups(n, edges, fps, &stats);
        assert_eq!(
            stats.clustering_abandoned.count(),
            0,
            "this graph is small enough to enumerate completely"
        );
        groups
    }

    fn adjacency_of(n: usize, edges: &[(usize, usize)]) -> Vec<HashSet<usize>> {
        let mut adjacency: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for &(i, j) in edges {
            adjacency[i].insert(j);
            adjacency[j].insert(i);
        }
        adjacency
    }

    #[test]
    fn test_find_duplicate_groups_simple_clique() {
        let fps = vec![
            mock_fingerprint("a.mp4"),
            mock_fingerprint("b.mp4"),
            mock_fingerprint("c.mp4"),
        ];

        // Fully connected graph (a-b, b-c, a-c)
        let edges = vec![(0, 1), (1, 2), (0, 2)];

        let groups = groups_of(3, edges, &fps);

        assert_eq!(groups.len(), 1, "Should find exactly one group");
        assert_eq!(groups[0].len(), 3, "Group should contain all 3 videos");

        // Ensure they are sorted by path
        assert_eq!(groups[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_find_duplicate_groups_disjoint_sets() {
        let fps = vec![
            mock_fingerprint("a.mp4"), // Group 1
            mock_fingerprint("b.mp4"), // Group 1
            mock_fingerprint("c.mp4"), // Group 2
            mock_fingerprint("d.mp4"), // Group 2
            mock_fingerprint("e.mp4"), // Unrelated
        ];

        // Edges: (a,b) and (c,d)
        let edges = vec![(0, 1), (2, 3)];

        let groups = groups_of(5, edges, &fps);

        assert_eq!(groups.len(), 2, "Should find exactly two groups");
        assert!(groups.contains(&vec![0, 1]));
        assert!(groups.contains(&vec![2, 3]));
    }

    #[test]
    fn test_a_partially_linked_file_is_never_absorbed_into_a_group() {
        // The regression this file exists to prevent. 0-1-2 is a triangle; 3 is
        // linked to 0 and 1 but NOT to 2. An expansion rule that admits any node
        // with two edges into a group folds 3 into {0,1,2}, where the ranking can
        // then mark it DELETE on the strength of a comparison against 2 that
        // never happened. With --delete --permanent, unrecoverably.
        let fps = mock_library(4);
        let edges = vec![(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)];

        let groups = groups_of(4, edges, &fps);

        assert_eq!(
            groups,
            vec![vec![0, 1, 2], vec![0, 1, 3]],
            "the two maximal cliques must be reported separately"
        );
        for group in &groups {
            assert!(
                !(group.contains(&2) && group.contains(&3)),
                "files that never matched must never share a group"
            );
        }
    }

    #[test]
    fn test_a_chain_does_not_collapse_into_one_group() {
        // 0-1 and 1-2 with no 0-2 edge. Treating the match relation as
        // transitive would produce a single group of three; it is not
        // transitive, so this is two overlapping pairs.
        let fps = mock_library(3);
        let groups = groups_of(3, vec![(0, 1), (1, 2)], &fps);

        assert_eq!(groups, vec![vec![0, 1], vec![1, 2]]);
    }

    #[test]
    fn test_overlapping_cliques_are_both_reported() {
        // Two triangles sharing node 2 -- e.g. one video that is genuinely a
        // duplicate within two otherwise unrelated sets. Both groups stand, and
        // export.rs resolves node 2's fate across them.
        let fps = mock_library(5);
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)];

        let groups = groups_of(5, edges, &fps);

        assert_eq!(groups, vec![vec![0, 1, 2], vec![2, 3, 4]]);
    }

    #[test]
    fn test_every_group_is_a_complete_subgraph() {
        // The invariant export.rs depends on, checked over a graph messy enough
        // to have several overlapping cliques and a stray cross-link.
        let fps = mock_library(7);
        let edges = vec![
            (0, 1),
            (0, 2),
            (1, 2), // triangle
            (2, 3),
            (2, 4),
            (3, 4), // triangle sharing node 2
            (4, 5), // dangling pair
            (0, 3), // a stray link between the two triangles
        ];

        let groups = groups_of(7, edges.clone(), &fps);
        let adjacency = adjacency_of(7, &edges);

        assert!(!groups.is_empty(), "this graph clearly has duplicates");

        for group in &groups {
            assert!(group.len() > 1, "a group of one is not a duplicate set");
            assert!(!group.contains(&6), "an unmatched file must not be grouped");

            for (pos, &a) in group.iter().enumerate() {
                for &b in &group[pos + 1..] {
                    assert!(
                        adjacency[a].contains(&b),
                        "{:?} contains the unmatched pair ({}, {})",
                        group,
                        a,
                        b
                    );
                }
            }
        }

        // And every edge is answered for: both ends of each match are reported
        // side by side somewhere, which is what makes the output a report of the
        // graph rather than a summary of it.
        for (a, b) in edges {
            assert!(
                groups.iter().any(|g| g.contains(&a) && g.contains(&b)),
                "({}, {}) matched but appear in no group together",
                a,
                b
            );
        }
    }

    #[test]
    fn test_unmatched_files_cost_nothing() {
        // A thousand videos, two matches. Isolated files are kept out of the
        // search entirely, so this is instant rather than a thousand recursions
        // each cloning a thousand-element set.
        let fps = mock_library(1000);
        let groups = groups_of(1000, vec![(10, 20), (20, 30)], &fps);

        assert_eq!(groups, vec![vec![10, 20], vec![20, 30]]);
    }

    #[test]
    fn test_a_repeated_edge_does_not_duplicate_a_member() {
        // The comparison stage can propose a pair from several blocks of the
        // index. The adjacency is a set of sets, so a second insertion of an
        // already-known edge changes nothing.
        let fps = mock_library(3);
        let groups = groups_of(3, vec![(0, 1), (1, 0), (0, 1), (1, 2)], &fps);

        assert_eq!(groups, vec![vec![0, 1], vec![1, 2]]);
    }

    #[test]
    fn test_output_is_reproducible_despite_hashset_iteration_order() {
        let fps = mock_library(8);
        let edges = vec![(0, 1), (0, 2), (1, 2), (3, 4), (5, 6)];

        let first = groups_of(8, edges.clone(), &fps);
        let second = groups_of(8, edges, &fps);

        assert_eq!(first, second, "two identical runs must report identically");
        assert_eq!(
            first,
            vec![vec![0, 1, 2], vec![3, 4], vec![5, 6]],
            "largest first, then by member paths"
        );
    }

    #[test]
    fn test_no_edges_means_no_groups() {
        let fps = mock_library(4);
        assert!(groups_of(4, vec![], &fps).is_empty());
    }

    // --- the ceilings ------------------------------------------------------
    //
    // The interrupt poll is deliberately not among these. The shutdown flag is
    // process-global and has no reset, so a test that set it would be setting it
    // for every other test in the binary, several of which run at the same time.

    /// Every edge except the ones listed, i.e. the complement of a sparse graph.
    /// Density is what these tests need and it is easier to say what is missing.
    fn all_edges_except(n: usize, missing: &[(usize, usize)]) -> Vec<(usize, usize)> {
        (0..n)
            .flat_map(|a| ((a + 1)..n).map(move |b| (a, b)))
            .filter(|pair| !missing.contains(pair))
            .collect()
    }

    /// The worst case for clique enumeration: `parts` groups of three files,
    /// where each file matches everything except its own two partners. Every
    /// clique picks one file from each part, so there are 3^parts of them --
    /// 3.5 billion at 20 parts, from 60 files.
    ///
    /// Not as artificial as it looks. It is what `-d 32` produces on any library
    /// large enough to have chance matches everywhere: nearly complete, with
    /// scattered non-edges.
    fn moon_moser(parts: usize) -> (usize, Vec<(usize, usize)>) {
        let n = parts * 3;
        let missing: Vec<(usize, usize)> = (0..parts)
            .flat_map(|p| {
                let base = p * 3;
                [(base, base + 1), (base, base + 2), (base + 1, base + 2)]
            })
            .collect();
        (n, all_edges_except(n, &missing))
    }

    #[test]
    fn test_a_component_that_outruns_its_budget_is_dropped_whole_and_reported() {
        // 3^20 cliques. Without a ceiling this is the hang the budget exists to
        // prevent, and no amount of waiting produces a usable report.
        let (n, edges) = moon_moser(20);
        let fps = mock_library(n);
        let stats = RunStats::default();

        let groups = group_within(
            n,
            edges,
            &fps,
            &stats,
            Ceilings { budget: 10_000, groups: usize::MAX },
        );

        assert!(groups.is_empty(), "a partial enumeration must not be reported");
        assert_eq!(stats.clustering_abandoned.count(), 1, "and it must be reported as a problem");
        assert!(stats.had_problems(), "so the run exits non-zero");
    }

    #[test]
    fn test_the_group_ceiling_drops_the_component_too() {
        // Three parts of three: 27 maximal cliques, cheap to enumerate, and far
        // more than the ceiling this run is given. The budget is untouched, so
        // only the output ceiling can be what stopped it.
        let (n, edges) = moon_moser(3);
        let fps = mock_library(n);
        let stats = RunStats::default();

        let groups = group_within(
            n,
            edges.clone(),
            &fps,
            &stats,
            Ceilings { budget: usize::MAX, groups: 5 },
        );

        assert!(groups.is_empty());
        assert_eq!(stats.clustering_abandoned.count(), 1);

        // And with room for all 27 the same graph comes back whole, so the
        // ceiling is the only thing that refused it.
        let roomy = RunStats::default();
        let full = group_within(n, edges, &fps, &roomy, Ceilings::shipped());
        assert_eq!(full.len(), 27);
        assert_eq!(roomy.clustering_abandoned.count(), 0);
    }

    #[test]
    fn test_only_the_dense_component_is_lost() {
        // The reason the search is run per component rather than over the whole
        // graph: one impossible folder must not cost the user the groups in
        // every other folder they scanned.
        let (dense_n, dense_edges) = moon_moser(20);
        let fps = mock_library(dense_n + 3);
        let stats = RunStats::default();

        let mut edges = dense_edges;
        // An ordinary pair and an ordinary triangle-less link, well away from it.
        edges.push((dense_n, dense_n + 1));
        edges.push((dense_n + 1, dense_n + 2));

        let groups = group_within(
            dense_n + 3,
            edges,
            &fps,
            &stats,
            Ceilings { budget: 10_000, groups: usize::MAX },
        );

        assert_eq!(
            groups,
            vec![vec![dense_n, dense_n + 1], vec![dense_n + 1, dense_n + 2]],
            "the honest component is reported exactly as it would have been alone"
        );
        assert_eq!(stats.clustering_abandoned.count(), 1);
    }

    #[test]
    fn test_the_problem_line_names_a_file_and_a_count() {
        // The summary is the only place a user learns that files went
        // unreported, so it has to say how many and point at one of them. The
        // example is the lowest path so that a re-run reads the same.
        let (n, edges) = moon_moser(20);
        let fps = mock_library(n);
        let stats = RunStats::default();

        group_within(n, edges, &fps, &stats, Ceilings { budget: 10_000, groups: usize::MAX });

        let lines = stats.clustering_abandoned.samples();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains(&format!("{} file(s)", n)), "got: {}", lines[0]);
        assert!(lines[0].contains("0000.mp4"), "got: {}", lines[0]);
    }

    #[test]
    fn test_the_shipped_ceilings_clear_a_realistically_dense_library() {
        // What the ceilings must never do is fire on real work. Forty files that
        // all match each other -- one folder of re-encodes, the densest thing a
        // sane scan produces -- plus the same again with a few links missing, so
        // it is a genuine enumeration rather than one clique.
        let (n, edges) = moon_moser(4); // 12 files, 81 cliques
        let mut all = edges;
        let base = n;
        for a in base..(base + 40) {
            for b in (a + 1)..(base + 40) {
                all.push((a, b));
            }
        }

        let fps = mock_library(base + 40);
        let stats = RunStats::default();
        let groups = group_within(base + 40, all, &fps, &stats, Ceilings::shipped());

        assert_eq!(stats.clustering_abandoned.count(), 0, "real densities must not trip it");
        assert_eq!(groups[0].len(), 40, "the folder of re-encodes is one group");
        assert_eq!(groups.len(), 1 + 81);
    }
}
