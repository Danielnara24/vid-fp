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
use std::collections::HashSet;

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
fn bron_kerbosch(
    r: HashSet<usize>,
    mut p: HashSet<usize>,
    mut x: HashSet<usize>,
    adjacency: &[HashSet<usize>],
    cliques: &mut Vec<HashSet<usize>>,
) {
    if p.is_empty() {
        // Nothing left to add. If nothing was excluded either, `r` is maximal.
        // A group of one is not a duplicate of anything, so it is dropped here
        // rather than filtered out downstream.
        if x.is_empty() && r.len() > 1 {
            cliques.push(r);
        }
        return;
    }

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
            adjacency,
            cliques,
        );

        // `v` has now appeared in every clique it can belong to at this level.
        // Moving it into `x` stops the branches to its right rediscovering
        // those same cliques with `v` appended.
        p.remove(&v);
        x.insert(v);
    }
}

/// Group `n` fingerprints into duplicate sets, given the pairs that matched.
///
/// Indices refer to positions in `fingerprints`. Within a group they are
/// ordered by path; the groups themselves are ordered largest first. A file that
/// matched nothing is in no group at all -- a group of one is not a duplicate
/// set, and dropping it here is what keeps the report to the files that are
/// actually implicated in something.
pub fn find_duplicate_groups(
    n: usize,
    edges: Vec<(usize, usize)>,
    fingerprints: &[VideoFingerprint],
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![HashSet::new(); n];
    for (i, j) in edges {
        adjacency[i].insert(j);
        adjacency[j].insert(i);
    }

    // Only a linked file can be in a group of two or more, and a file with no
    // edges cannot extend anyone else's clique either -- so leaving it out of
    // `p` changes no result. On a real library the overwhelming majority of
    // videos match nothing at all, and seeding them would spend one recursion,
    // plus a clone of an n-sized set, per file to rediscover that.
    let candidates: HashSet<usize> = (0..n).filter(|&v| !adjacency[v].is_empty()).collect();

    let mut cliques = Vec::new();
    bron_kerbosch(
        HashSet::new(),
        candidates,
        HashSet::new(),
        &adjacency,
        &mut cliques,
    );

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

        let groups = find_duplicate_groups(3, edges, &fps);

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

        let groups = find_duplicate_groups(5, edges, &fps);

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

        let groups = find_duplicate_groups(4, edges, &fps);

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
        let groups = find_duplicate_groups(3, vec![(0, 1), (1, 2)], &fps);

        assert_eq!(groups, vec![vec![0, 1], vec![1, 2]]);
    }

    #[test]
    fn test_overlapping_cliques_are_both_reported() {
        // Two triangles sharing node 2 -- e.g. one video that is genuinely a
        // duplicate within two otherwise unrelated sets. Both groups stand, and
        // export.rs resolves node 2's fate across them.
        let fps = mock_library(5);
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)];

        let groups = find_duplicate_groups(5, edges, &fps);

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

        let groups = find_duplicate_groups(7, edges.clone(), &fps);
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
        let groups = find_duplicate_groups(1000, vec![(10, 20), (20, 30)], &fps);

        assert_eq!(groups, vec![vec![10, 20], vec![20, 30]]);
    }

    #[test]
    fn test_a_repeated_edge_does_not_duplicate_a_member() {
        // The comparison stage can propose a pair from several blocks of the
        // index. The adjacency is a set of sets, so a second insertion of an
        // already-known edge changes nothing.
        let fps = mock_library(3);
        let groups = find_duplicate_groups(3, vec![(0, 1), (1, 0), (0, 1), (1, 2)], &fps);

        assert_eq!(groups, vec![vec![0, 1], vec![1, 2]]);
    }

    #[test]
    fn test_output_is_reproducible_despite_hashset_iteration_order() {
        let fps = mock_library(8);
        let edges = vec![(0, 1), (0, 2), (1, 2), (3, 4), (5, 6)];

        let first = find_duplicate_groups(8, edges.clone(), &fps);
        let second = find_duplicate_groups(8, edges, &fps);

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
        assert!(find_duplicate_groups(4, vec![], &fps).is_empty());
    }
}
