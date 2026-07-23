use crate::fingerprint::VideoFingerprint;
use std::collections::HashSet;

// Graph Traversal with Pivoting logic to handle aggressive densifying cliques flawlessly
fn bron_kerbosch(
    r: HashSet<usize>,
    mut p: HashSet<usize>,
    mut x: HashSet<usize>,
    adjacency: &[HashSet<usize>],
    base_cliques: &mut Vec<HashSet<usize>>,
) {
    if p.is_empty() && x.is_empty() {
        if r.len() > 1 {
            base_cliques.push(r);
        }
        return;
    }

    // Heuristic: Use Pivot methodology choosing largest neighbor intersection maximizing node exclusion
    let pivot = p.union(&x).max_by_key(|&&v| adjacency[v].intersection(&p).count()).cloned();
    
    let p_explore: Vec<usize> = if let Some(u) = pivot {
        p.difference(&adjacency[u]).cloned().collect()
    } else {
        p.iter().cloned().collect()
    };

    for v in p_explore {
        let mut new_r = r.clone();
        new_r.insert(v);

        let neighbors = &adjacency[v];
        let new_p: HashSet<usize> = neighbors.intersection(&p).cloned().collect();
        let new_x: HashSet<usize> = neighbors.intersection(&x).cloned().collect();

        bron_kerbosch(new_r, new_p, new_x, adjacency, base_cliques);

        p.remove(&v);
        x.insert(v);
    }
}

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

    let mut base_cliques = Vec::new();
    let all_nodes: HashSet<usize> = (0..n).collect();
    
    bron_kerbosch(HashSet::new(), all_nodes, HashSet::new(), &adjacency, &mut base_cliques);

    let mut expanded_groups = Vec::new();
    for clique in base_cliques {
        let mut group = clique.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for v in 0..n {
                if !group.contains(&v) {
                    if adjacency[v].intersection(&group).count() >= 2 {
                        group.insert(v);
                        changed = true;
                    }
                }
            }
        }
        expanded_groups.push(group);
    }

    expanded_groups.sort_by(|a, b| b.len().cmp(&a.len()));

    let mut final_groups_sets: Vec<HashSet<usize>> = Vec::new();
    for g in expanded_groups {
        let mut is_subset = false;
        for fg in &final_groups_sets {
            if g.is_subset(fg) {
                is_subset = true;
                break;
            }
        }
        if !is_subset {
            final_groups_sets.push(g);
        }
    }

    // Keep indices mapped directly to the original struct to print properties naturally
    let mut final_groups: Vec<Vec<usize>> = Vec::new();
    for g in final_groups_sets {
        let mut group_indices: Vec<usize> = g.into_iter().collect();
        group_indices.sort_by(|&a, &b| fingerprints[a].path.cmp(&fingerprints[b].path));
        final_groups.push(group_indices);
    }

    // Sort fully deterministically: size of group descending, then deep comparison of all paths ascending
    final_groups.sort_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then_with(|| {
                let paths_a = a.iter().map(|&idx| &fingerprints[idx].path);
                let paths_b = b.iter().map(|&idx| &fingerprints[idx].path);
                paths_a.cmp(paths_b)
            })
    });

    final_groups
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create dummy fingerprints for testing
    fn mock_fingerprint(path: &str) -> VideoFingerprint {
        VideoFingerprint {
            path: path.to_string(),
            valid_hashes: vec![],
            valid_t_start: vec![],
            valid_t_end: vec![],
            total_frames: 100,
            width: 1920,
            height: 1080,
            duration: 10.0,
            file_size: 1024,
        }
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
}