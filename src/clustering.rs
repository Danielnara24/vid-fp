//! Turning "these two videos match" edges into groups.
//!
//! A group is a **representative and everything that matched it directly** --
//! a star, the same rule the sibling `img-fp` uses. Every member was compared
//! with the representative by phase 2 and cleared the same `-d`/`-p` gate as any
//! other pair, so a group asserts a set of pairs the run actually measured and
//! never one nobody tested.
//!
//! What a star does NOT assert is that two non-representative members matched
//! each other. They may have, and often did -- the index is exhaustive, so a
//! pair of members with no edge between them was compared and did not clear the
//! gate. `export.rs` is what cares, and it reads the edges themselves rather
//! than trusting the group's shape: a file is only ever marked DELETE against a
//! survivor it has a measurement with. See the labeling pass there.
//!
//! Why not the alternatives:
//!
//! - **Connected components** invent pairs. Matching is not transitive -- three
//!   episodes sharing an opening sequence are pairwise linked without any two of
//!   them being duplicates, and a clip links to its host without linking to the
//!   host's other clips -- and one wrong edge between two families merges them.
//! - **Maximal cliques**, which this module used to enumerate, invent nothing,
//!   but a family is not a clique. A dense family of copies with scattered gaps
//!   between them splits into many overlapping groups differing by a member or
//!   two, and enumeration is `3^(n/3)` in the worst case -- reachable from the
//!   command line, since a loose `-d` over a large library is a near-complete
//!   graph by construction. That took a probe budget, a group ceiling, a stack of
//!   its own and a path that dropped a whole component as a problem. A star needs
//!   none of them: it is `O(E log V)` with no worst case to defend.
//!
//! The representative is chosen greedily: the file that still accounts for the
//! most files no group has taken yet. That is a statement rather than a
//! threshold, and it is usually the original or a clean re-encode -- the file
//! most of the others matched. It is NOT necessarily the file kept: which copy
//! survives is `--priority`'s question, answered in `export.rs`.
//!
//! Two consequences, both deliberate:
//!
//! - **Groups overlap.** A representative's group lists everything it matched,
//!   including files an earlier group already took, because that is how a file
//!   only it reached gets reported at all. `export.rs` resolves each file's fate
//!   ONCE across every group it appears in.
//! - **Every matched file lands in a group.** The loop runs until nothing linked
//!   is left ungrouped, so raising `-d` or lowering `-p` -- which only ever adds
//!   edges -- only ever adds files to the report.
//!
//! Whether an edge exists at all is decided upstream by `--hamming-distance`,
//! `--match-percent` and `--min-duration`. This module adds no thresholds.

use crate::fingerprint::VideoFingerprint;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Group `n` fingerprints around representatives, given the pairs that matched.
///
/// Indices refer to positions in `fingerprints`. Each group lists its
/// representative FIRST and the files that matched it after, ordered by path;
/// the groups themselves are ordered largest first, ties broken by member paths
/// in that order. A file that matched nothing is in no group -- a group of one is
/// not a duplicate set.
///
/// Ties between equally good representatives go to the lowest PATH, not the
/// lowest index. Index order is the order fingerprints came out of the cache and
/// the decode, which differs between a cold run and a warm one; path order is a
/// property of the library, so the same edges group the same way on both.
pub fn find_duplicate_groups(
    n: usize,
    edges: Vec<(usize, usize)>,
    fingerprints: &[VideoFingerprint],
) -> Vec<Vec<usize>> {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (a, b) in edges {
        if a != b {
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
    }
    for neighbours in adjacency.iter_mut() {
        neighbours.sort_unstable();
        neighbours.dedup();
    }

    // Each index's position in path order, which is what every tie below is
    // broken on. Paths are unique -- `sources::collect` deduplicates on inode --
    // so this is a total order.
    let mut by_path: Vec<usize> = (0..n).collect();
    by_path.sort_by(|&a, &b| fingerprints[a].path.cmp(&fingerprints[b].path));
    let mut path_rank = vec![0usize; n];
    for (rank, &idx) in by_path.iter().enumerate() {
        path_rank[idx] = rank;
    }

    let mut ungrouped: Vec<bool> = adjacency.iter().map(|a| !a.is_empty()).collect();
    let mut left = ungrouped.iter().filter(|&&u| u).count();

    // How many ungrouped files electing `v` would account for, itself included.
    // Recomputed when an entry comes up rather than maintained on every
    // neighbour: a heap entry may be stale, and is corrected then.
    let reach = |v: usize, ungrouped: &[bool]| -> usize {
        adjacency[v].iter().filter(|&&u| ungrouped[u]).count() + ungrouped[v] as usize
    };

    // Max by reach, ties to the lowest path. Every key is distinct, so the order
    // is total and the election is reproducible.
    let mut heap: BinaryHeap<(usize, Reverse<usize>, usize)> = (0..n)
        .filter(|&v| ungrouped[v])
        .map(|v| (adjacency[v].len() + 1, Reverse(path_rank[v]), v))
        .collect();

    let mut groups: Vec<Vec<usize>> = Vec::new();
    while left > 0 {
        let Some((claimed, rank, r)) = heap.pop() else { break };

        // Earlier groups have taken files this entry still counts. A stale entry
        // goes back with the truth rather than being trusted, which keeps the
        // greedy exact without touching every neighbour's key per election.
        let real = reach(r, &ungrouped);
        if real < claimed {
            if real > 0 {
                heap.push((real, rank, r));
            }
            continue;
        }

        let mut members: Vec<usize> = adjacency[r].clone();
        members.sort_by_key(|&m| path_rank[m]);
        members.insert(0, r);

        for &f in &members {
            if ungrouped[f] {
                ungrouped[f] = false;
                left -= 1;
            }
        }

        // `real > 0`, so this election took at least one file and the loop makes
        // progress; a representative with a neighbour always has two members.
        groups.push(members);
    }

    groups.sort_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then_with(|| a.iter().map(|&i| path_rank[i]).cmp(b.iter().map(|&i| path_rank[i])))
    });

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // Clustering only ever looks at paths, so the rest is plausible filler --
    // the rules that care about it live in utils and export.
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
            mirrored_hashes: Vec::new(),
            detail_hashes: Vec::new(),
            mirrored_detail_hashes: Vec::new(),
            reframed: Vec::new(),
        }
    }

    /// `n` fingerprints whose paths sort in index order.
    fn mock_library(n: usize) -> Vec<VideoFingerprint> {
        (0..n)
            .map(|i| mock_fingerprint(&format!("/library/{:04}.mp4", i)))
            .collect()
    }

    fn groups_of(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        find_duplicate_groups(n, edges.to_vec(), &mock_library(n))
    }

    /// The property the whole design rests on: every member was matched against
    /// the representative, so a group claims nothing untested.
    #[test]
    fn test_every_member_matched_the_representative() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (5, 6), (6, 7)];
        let present: HashSet<(usize, usize)> =
            edges.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        for g in groups_of(8, &edges) {
            let rep = g[0];
            for &m in &g[1..] {
                assert!(
                    present.contains(&(m.min(rep), m.max(rep))),
                    "{m} was never matched against {rep}"
                );
            }
        }
    }

    /// A chain is one group around its middle: the ends are each a duplicate of
    /// the middle, and the group does not say they are duplicates of each other.
    #[test]
    fn test_a_chain_groups_around_its_middle() {
        assert_eq!(groups_of(3, &[(0, 1), (1, 2)]), vec![vec![1, 0, 2]]);
    }

    /// A clip matched by two hosts that share nothing else represents them.
    #[test]
    fn test_a_file_matched_by_two_others_represents_them() {
        assert_eq!(groups_of(3, &[(0, 1), (0, 2)]), vec![vec![0, 1, 2]]);
    }

    /// Where one representative cannot reach everything a second is elected, and
    /// its group lists everything it matched -- including a file the first group
    /// already took, which is how the file only it reached is reported at all.
    #[test]
    fn test_a_second_representative_covers_what_the_first_missed() {
        // 0 matches 1, 2, 3; 4 matches only 3.
        assert_eq!(
            groups_of(5, &[(0, 1), (0, 2), (0, 3), (3, 4)]),
            vec![vec![0, 1, 2, 3], vec![3, 0, 4]]
        );
    }

    #[test]
    fn test_every_matched_file_lands_in_a_group() {
        let edges = [(0, 1), (1, 2), (3, 4), (5, 6), (6, 7), (7, 8), (8, 5)];
        let covered: HashSet<usize> = groups_of(9, &edges).into_iter().flatten().collect();
        assert_eq!(covered, (0..9).collect());
    }

    #[test]
    fn test_unmatched_files_are_in_no_group() {
        assert!(groups_of(5, &[]).is_empty());
        assert_eq!(groups_of(5, &[(3, 4)]), vec![vec![3, 4]]);
    }

    #[test]
    fn test_a_repeated_edge_does_not_duplicate_a_member() {
        assert_eq!(groups_of(2, &[(0, 1), (1, 0), (0, 1)]), vec![vec![0, 1]]);
    }

    /// Two components are two groups, the larger first.
    #[test]
    fn test_disjoint_sets_are_separate_groups_largest_first() {
        assert_eq!(
            groups_of(5, &[(3, 4), (0, 1), (1, 2), (0, 2)]),
            vec![vec![0, 1, 2], vec![3, 4]]
        );
    }

    /// A folder of identical files is one group however many there are -- the
    /// case the clique search needed a pivot shortcut and a stack of its own for.
    #[test]
    fn test_a_complete_component_is_one_group() {
        let n = 2_000;
        let edges: Vec<(usize, usize)> =
            (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
        let g = groups_of(n, &edges);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].len(), n);
        assert_eq!(g[0][0], 0, "ties go to the lowest path");
    }

    /// The shape clique enumeration cannot survive -- a complete graph minus a
    /// perfect matching has `2^(n/2)` maximal cliques -- is a couple of elections.
    #[test]
    fn test_a_dense_component_costs_nothing() {
        let n = 400;
        let edges: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .filter(|&(i, j)| !(i % 2 == 0 && j == i + 1))
            .collect();
        let g = groups_of(n, &edges);
        assert!(g.len() <= 3, "{} groups", g.len());
        let covered: HashSet<usize> = g.into_iter().flatten().collect();
        assert_eq!(covered.len(), n);
    }

    /// The same pairs in a different order group identically.
    #[test]
    fn test_output_does_not_depend_on_edge_order() {
        let edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5)];
        let mut shuffled: Vec<(usize, usize)> = edges.iter().rev().map(|&(a, b)| (b, a)).collect();
        assert_eq!(groups_of(6, &edges), groups_of(6, &shuffled));
        shuffled.rotate_left(3);
        assert_eq!(groups_of(6, &edges), groups_of(6, &shuffled));
    }

    /// Ties are broken on paths, not on indices: the same library handed over in
    /// a different order -- which is what a warm run does to a cold one's --
    /// elects the same file and lists the same members.
    #[test]
    fn test_ties_follow_the_paths_not_the_order_the_files_arrived_in() {
        // Two equal candidates: "a" and "b" both match "c" and nothing else.
        let forward = vec![
            mock_fingerprint("/l/a.mp4"),
            mock_fingerprint("/l/b.mp4"),
            mock_fingerprint("/l/c.mp4"),
            mock_fingerprint("/l/d.mp4"),
        ];
        // Same files, reversed.
        let backward: Vec<VideoFingerprint> = forward.iter().rev().cloned().collect();
        let paths = |fps: &[VideoFingerprint], groups: Vec<Vec<usize>>| -> Vec<Vec<String>> {
            groups
                .into_iter()
                .map(|g| g.into_iter().map(|i| fps[i].path.clone()).collect())
                .collect()
        };
        // a-c, b-c, a-d, b-d: a and b each reach three, c and d each reach three.
        let fwd = find_duplicate_groups(4, vec![(0, 2), (1, 2), (0, 3), (1, 3)], &forward);
        let bwd = find_duplicate_groups(4, vec![(3, 1), (2, 1), (3, 0), (2, 0)], &backward);
        assert_eq!(paths(&forward, fwd), paths(&backward, bwd));
    }
}
