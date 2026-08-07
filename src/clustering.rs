//! Turning "these two videos match" edges into groups.
//!
//! A group is a *connected component*: files reachable from one another by
//! following matches. Every file appears in exactly one group, and a group is a
//! reporting unit -- "these are related, here they are side by side".
//!
//! That makes the partition the organising idea of the whole report. There is no
//! such thing as a file whose fate depends on which group you read it in,
//! because there is only one; `export.rs` therefore resolves KEEP, DELETE and
//! REVIEW entirely inside a group and never has to reconcile a file's roles
//! across several of them.
//!
//! What a component does NOT assert is that its members are pairwise
//! duplicates. Matching is not transitive: three episodes sharing an opening
//! sequence are linked without any two being copies of each other, and a clip
//! cut from a long video links to its host without linking to the host's other
//! clips. A single such edge fuses everything it touches into one component. So
//! a component says "a chain of matches runs through all of these", which is a
//! weaker claim than "these are all the same video".
//!
//! That is why membership and fate are decided separately. Grouping follows the
//! chain, because the point of a group is to put related files in front of you
//! together; deletion does not, because `export.rs` will only mark a file DELETE
//! if it directly matched a copy that survives the group. A file that got here
//! through some third file is reported alongside its group and flagged REVIEW.
//! The coverage figures printed beside each row are what tell a re-encode from a
//! shared title card.
//!
//! Whether an edge exists at all is decided upstream by `--hamming-distance`,
//! `--match-percent` and `--min-duration`. This module adds no thresholds of its
//! own; it only follows the links the comparison stage found.

use crate::fingerprint::VideoFingerprint;
use std::collections::HashMap;

/// A disjoint-set forest over file indices.
///
/// Components are built by merging as the edges stream past rather than by
/// walking the graph afterwards, so nothing has to hold an adjacency list for
/// the whole library -- two `usize` vectors is the entire cost, and a scan of a
/// hundred thousand videos that matched nothing pays for two hundred thousand
/// integers and no allocation per file.
struct DisjointSet {
    parent: Vec<usize>,
    /// Number of members in the tree rooted here. Only meaningful for a root.
    size: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        DisjointSet {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    /// The representative of `v`'s component, flattening the path on the way.
    ///
    /// Path halving rather than full compression: it points every second node
    /// straight at its grandparent, which gives the same near-constant
    /// amortized cost as the textbook two-pass version in a single loop and
    /// without recursion -- and a component here can span a whole series, so a
    /// recursive `find` is a stack depth chosen by the user's library.
    fn find(&mut self, mut v: usize) -> usize {
        while self.parent[v] != v {
            self.parent[v] = self.parent[self.parent[v]];
            v = self.parent[v];
        }
        v
    }

    /// Merge the components holding `a` and `b`. Hanging the smaller tree off
    /// the larger is what keeps the trees shallow enough for `find` to stay
    /// cheap regardless of the order the edges arrive in.
    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

/// Group `n` fingerprints into duplicate sets, given the pairs that matched.
///
/// Indices refer to positions in `fingerprints`. Within a group they are
/// ordered by path; the groups themselves are ordered largest first. A file that
/// matched nothing is in no group at all -- a component of one is not a
/// duplicate set, and dropping it here is what keeps the report to the files
/// that are actually implicated in something.
pub fn find_duplicate_groups(
    n: usize,
    edges: Vec<(usize, usize)>,
    fingerprints: &[VideoFingerprint],
) -> Vec<Vec<usize>> {
    let mut sets = DisjointSet::new(n);
    for (i, j) in edges {
        sets.union(i, j);
    }

    // Bucketing by root is what turns the forest back into groups. The size
    // check keeps unmatched files out of the map entirely, which on a real
    // library is the overwhelming majority of it.
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for v in 0..n {
        let root = sets.find(v);
        if sets.size[root] > 1 {
            components.entry(root).or_default().push(v);
        }
    }

    let mut groups: Vec<Vec<usize>> = components
        .into_values()
        .map(|mut indices| {
            indices.sort_by(|&a, &b| fingerprints[a].path.cmp(&fingerprints[b].path));
            indices
        })
        .collect();

    // Discovery order follows HashMap iteration, which is not stable across
    // runs, so the output is sorted into a total order instead: largest groups
    // first, ties broken by comparing member paths in order. Paths are unique
    // (files are deduplicated by inode before they reach this point) and every
    // file belongs to exactly one group, so no two distinct groups can compare
    // equal and the result is fully reproducible.
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
    use std::collections::HashSet;

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

    #[test]
    fn test_find_duplicate_groups_fully_connected_trio() {
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
    fn test_a_chain_collapses_into_one_group() {
        // 0-1 and 1-2 with no 0-2 edge. 0 and 2 were never compared with each
        // other, but they are reachable through 1, so they are related and
        // belong in the same reporting unit.
        let fps = mock_library(3);
        let groups = find_duplicate_groups(3, vec![(0, 1), (1, 2)], &fps);

        assert_eq!(groups, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn test_a_partially_linked_file_joins_the_group_it_is_linked_to() {
        // 0-1-2 is a triangle; 3 is linked to 0 and 1 but not to 2. One group of
        // four: having a match inside the component is the whole membership
        // test, and being unmatched against one member does not exclude a file.
        let fps = mock_library(4);
        let edges = vec![(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)];

        let groups = find_duplicate_groups(4, edges, &fps);

        assert_eq!(groups, vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn test_groups_sharing_a_file_become_one_group() {
        // Two triangles joined at node 2, which is a duplicate within each of
        // two otherwise unrelated sets. One shared file is enough to make all
        // five one group, and so one decision rather than a file whose fate has
        // to be reconciled across the groups reporting it.
        let fps = mock_library(5);
        let edges = vec![(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)];

        let groups = find_duplicate_groups(5, edges, &fps);

        assert_eq!(groups, vec![vec![0, 1, 2, 3, 4]]);
    }

    #[test]
    fn test_every_file_appears_in_exactly_one_group() {
        // The invariant export.rs depends on, checked over a graph messy enough
        // to have several dense neighbourhoods, a bridge between them, and a
        // dangling pair. Everything with an edge is grouped, exactly once;
        // everything without one is absent.
        let fps = mock_library(9);
        let edges = vec![
            (0, 1),
            (0, 2),
            (1, 2), // triangle
            (2, 3),
            (2, 4),
            (3, 4), // triangle sharing node 2
            (4, 5), // dangling pair off the second triangle
            (0, 3), // a stray link between the two triangles
            (6, 7), // an unrelated pair
        ];

        let groups = find_duplicate_groups(9, edges.clone(), &fps);

        let mut seen: HashSet<usize> = HashSet::new();
        for group in &groups {
            assert!(group.len() > 1, "a group of one is not a duplicate set");
            for &idx in group {
                assert!(seen.insert(idx), "{} was reported in two groups", idx);
            }
        }

        assert_eq!(groups, vec![vec![0, 1, 2, 3, 4, 5], vec![6, 7]]);
        assert!(!seen.contains(&8), "an unmatched file must not be grouped");

        // And every edge is answered for: both ends of each match landed in the
        // same group, which is what makes the partition a report of the graph
        // rather than a summary of it.
        for (a, b) in edges {
            let group = groups.iter().find(|g| g.contains(&a)).unwrap();
            assert!(group.contains(&b), "({}, {}) matched but were split", a, b);
        }
    }

    #[test]
    fn test_unmatched_files_cost_nothing() {
        // A thousand videos, two matches. Isolated files never reach the output
        // side at all: they are their own root, of size one, and skipped.
        let fps = mock_library(1000);
        let groups = find_duplicate_groups(1000, vec![(10, 20), (20, 30)], &fps);

        assert_eq!(groups, vec![vec![10, 20, 30]]);
    }

    #[test]
    fn test_a_repeated_edge_does_not_duplicate_a_member() {
        // The comparison stage can propose a pair from several blocks of the
        // index. A second union of an already-merged pair is a no-op, and the
        // member list is built from the files rather than from the edges, so
        // neither can put a file in a group twice.
        let fps = mock_library(3);
        let groups = find_duplicate_groups(3, vec![(0, 1), (1, 0), (0, 1), (1, 2)], &fps);

        assert_eq!(groups, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn test_output_is_reproducible_despite_hashmap_iteration_order() {
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
