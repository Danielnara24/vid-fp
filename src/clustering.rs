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

    final_groups.sort_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then_with(|| fingerprints[a[0]].path.cmp(&fingerprints[b[0]].path))
    });

    final_groups
}