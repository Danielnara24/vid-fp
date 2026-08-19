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
use std::path::{Component, Path, PathBuf};

/// How much of the clique search ONE connected component may cost, counted in
/// set probes -- every set operation the search performs, charged before it runs.
/// See `Search::spend` and `intersection_cost`, which are where the counting is
/// defined.
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
/// Fifty million is a second or two of a single core. It also bounds the depth:
/// a clique of `d` members costs about `3d^2 / 2` probes to enumerate, since each
/// of its `d` levels holds a `p` of at most `d` members and spends a small number
/// of passes over it, so no search that fits in this budget can recurse past
/// ~5,800 frames. The sets alive along that path are the same quantity again --
/// `p`, `x` and the clique so far, at every level above -- which is what keeps the
/// heap under it too. Measured on the local corpus, peak RSS at the point the
/// budget is exhausted: 177 MB at `-d 24` and 562 MB at `-d 32`, the loosest
/// input the flag admits at all.
///
/// That figure used to be ~530, because a level cost `d - i` INTERSECTIONS
/// rather than a handful -- the pivot walk priced every candidate in `p ∪ x`
/// against `p`, making the whole search `d^3 / 3`. The consequence was a ceiling
/// that fired hardest on the easiest possible input: a complete component has
/// exactly one maximal clique and one branch per level, so 600 copies of one
/// file were refused by a budget meant for combinatorial explosions, and the
/// advice the refusal gives -- tighten `--hamming-distance` -- cannot separate
/// files that are identical. `choose_pivot` stops the walk as soon as it can and
/// the budget now measures what is really spent, so a component like that is
/// enumerated rather than abandoned. Nothing about which cliques are found
/// changed: the pivot is still the maximum one.
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

/// The stack the clique search is given, so that the budget above is the only
/// thing that decides how deep it may go.
///
/// Recursion depth is the size of the clique being built, and reaching depth `d`
/// costs at least `3d^2 / 2` probes -- every level walks `p` a small fixed number
/// of times and `p` still holds at least `d - i` members -- so the budget caps
/// the depth at about 5,800 frames. A frame of `bron_kerbosch` measures ~600
/// bytes (three sets by value, the branch list, and the clique being cloned into),
/// which puts the deepest search the ceilings permit at a little over 3 MB.
///
/// That fits the 8 MB a Linux main thread usually gets, and "usually" is doing
/// far too much work in a sentence about a segfault: `ulimit -s` is 1 or 2 MB on
/// plenty of build machines and containers, and a rayon worker gets 2 MB. Measured
/// here, a complete component of 2,000 files needs more than 1 MB and one of 4,000
/// needs more than 2 -- both well inside what the budget allows -- so on either of
/// those the process would abort rather than report. Naming the stack makes the
/// depth bound a property of this program instead of of the ambient limits, which
/// is what lets `SEARCH_BUDGET_PER_COMPONENT` be the only ceiling that decides
/// anything. It costs address space and not memory: a thread stack is committed a
/// page at a time as it is used.
const SEARCH_STACK_BYTES: usize = 64 * 1024 * 1024;

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
    /// What stopped the search and how far it had got, phrased for the Problems
    /// summary. `Interrupted` never reaches it -- `RunStats` deliberately counts
    /// nothing that Ctrl-C dropped -- but it is spelled out anyway rather than
    /// made unreachable.
    ///
    /// Both ceilings can now say a number, and they deliberately say DIFFERENT
    /// numbers, because "how many groups did it find" means different things
    /// either side of them.
    ///
    /// Against the work budget the count is real information: the search stopped
    /// wherever it happened to be, so `0 so far` says the component is so tangled
    /// that not one clique could be finished, while `40,000 so far` says it was
    /// enumerating perfectly well and simply had more to do than the budget
    /// allowed. Those are different problems and the fix for them differs.
    ///
    /// Against the OUTPUT ceiling the count carries nothing: the search stops the
    /// moment it exceeds the ceiling, so it is always exactly one past it and
    /// printing it would be printing `MAX_GROUPS_PER_COMPONENT + 1` dressed up as
    /// a measurement. What is worth saying there is the threshold itself -- the
    /// scale the user is up against, and a figure they can compare their `-d`
    /// ladder to -- with "more than" doing the work the count cannot.
    fn ceiling(&self, found: usize, ceilings: Ceilings) -> String {
        match self {
            Halt::Budget => format!(
                "the search for groups ran past its work ceiling, with {} found so far",
                found
            ),
            Halt::Groups => format!(
                "they form more than {} overlapping groups, which is more than can be reported",
                ceilings.groups
            ),
            Halt::Interrupted => "the run was interrupted".to_string(),
        }
    }
}

/// The deepest folder every one of `paths` sits under, or `None` if they share
/// none.
///
/// Compared a whole path COMPONENT at a time rather than as strings, which is
/// the difference between `/lib/season1` and `/lib/season2` sharing `/lib` and
/// their appearing to share `/lib/season` -- a folder that does not exist and
/// that the user would go looking for.
///
/// The file names themselves are dropped first: what this answers is "where do I
/// go and look", and a file is not somewhere to look.
///
/// Every path a scan produces is absolute (`sources::collect` canonicalizes each
/// one), so real input always shares at least the root and this always answers.
/// `None` is for a caller that got something else.
fn common_parent<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Option<PathBuf> {
    let mut shared: Option<Vec<Component<'a>>> = None;

    for path in paths {
        let folder: Vec<Component> = path.parent().unwrap_or(Path::new("")).components().collect();

        shared = Some(match shared {
            None => folder,
            Some(so_far) => {
                let agreed = so_far.iter().zip(&folder).take_while(|(a, b)| a == b).count();
                so_far.into_iter().take(agreed).collect()
            }
        });
    }

    let shared = shared?;
    (!shared.is_empty()).then(|| shared.into_iter().collect())
}

/// The line the Problems summary shows for an abandoned component: how many
/// files were in it, which ceiling stopped it and how far it had got, and where
/// they are, so the user has somewhere to look.
///
/// A folder rather than one of the files. Naming a member was only ever a handle
/// on a set the report says nothing else about, and it was a poor one: nothing is
/// wrong with the file it named, and on the case that produces these lines most
/// often -- a tolerance loose enough to link a whole library into one component
/// -- it degenerated into whichever path happened to sort first. The shared
/// folder is the thing a user can act on, because it is what `--exclude` and a
/// narrower scan take as an argument, and it narrows by itself exactly when the
/// answer is worth having: one dense folder inside a large library names that
/// folder, while a library that is dense throughout names the scan root and says
/// so honestly.
///
/// It is also stable by construction rather than by convention. Which files land
/// in a component comes out of `HashSet` iteration, so picking a member needed
/// the lowest path to read the same on a re-run; a common prefix does not depend
/// on the order it is computed in.
fn abandoned_detail(
    component: &[usize],
    reason: &Halt,
    found: usize,
    ceilings: Ceilings,
    fps: &[VideoFingerprint],
) -> String {
    let line = format!(
        "{} file(s) linked too densely to group -- {}",
        component.len(),
        reason.ceiling(found, ceilings)
    );

    match common_parent(component.iter().map(|&i| Path::new(&fps[i].path))) {
        Some(folder) => format!("{}; all under {}", line, folder.display()),
        None => line,
    }
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
    /// Set probes left; the search stops when it runs out.
    budget: usize,
    halted: Option<Halt>,
}

/// What one `HashSet::intersection` between a vertex's neighbours and `other`
/// costs, in probes.
///
/// `intersection` walks the smaller of the two sets and looks each member up in
/// the larger, so one of these is `min(deg(v), |other|)` probes rather than one.
/// Charging the flat size of the sets undercounts a dense component by a factor
/// of `|p|`, and that is not a rounding error: a NEARLY complete graph has few
/// enough maximal cliques to stay under the group ceiling while every one of its
/// steps carries hundreds of thousands of probes, so it used to spend minutes
/// inside a budget that believed it had spent thousands. (`-d 40` on the local
/// corpus, before `--hamming-distance` was capped at chance: eleven minutes and
/// climbing, single-threaded, uninterrupted by either ceiling.)
fn intersection_cost(neighbours: &HashSet<usize>, other: &HashSet<usize>) -> usize {
    neighbours.len().min(other.len())
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

    /// The ceilings that are not about the work: has something already stopped
    /// the search, has the user, has it produced more groups than can be
    /// reported. `false` means the caller should unwind.
    ///
    /// Asked once per recursion step, which is the finest granularity the search
    /// has -- a single step is a handful of set operations, so an interrupt is
    /// answered immediately even on a component that would otherwise run for
    /// minutes. It is a relaxed atomic load against work that clones sets.
    fn may_continue(&mut self) -> bool {
        if self.halted.is_some() {
            return false;
        }
        if shutdown_requested() {
            self.halted = Some(Halt::Interrupted);
            return false;
        }
        if self.cliques.len() > self.ceilings.groups {
            self.halted = Some(Halt::Groups);
            return false;
        }
        true
    }

    /// Charge `probes` of set work against the budget. `false` means it ran out
    /// and the caller should unwind.
    ///
    /// Charged per OPERATION rather than per step, and always before the
    /// operation it pays for. A step's cost used to be priced up front, which
    /// worked only for as long as every step spent the same shape of work; the
    /// pivot walk now stops as soon as it can (see `choose_pivot`), so what a
    /// step is about to spend is no longer knowable before it spends it. Metering
    /// each operation instead keeps the budget an account of what the search
    /// really did -- and tightens the overrun from one whole step to one set
    /// intersection.
    fn spend(&mut self, probes: usize) -> bool {
        let Some(left) = self.budget.checked_sub(probes) else {
            self.halted = Some(Halt::Budget);
            return false;
        };
        self.budget = left;
        true
    }
}

/// The pivot for this step: the candidate in `p ∪ x` adjacent to the most of
/// `p`. `None` means the budget ran out while looking (`p` is never empty here,
/// so there is always a candidate otherwise).
///
/// Every `u` in `p ∪ x` is a VALID pivot -- the maximum is only what makes the
/// branching narrow -- and the walk that finds the maximum is the single most
/// expensive thing a step does: one intersection per candidate, each costing up
/// to `|p|`, which is what makes a step quadratic in `|p|` and a deep search
/// cubic in the size of its clique.
///
/// So the walk stops the moment a candidate reaches the most that anything still
/// to come could reach. `x` is walked FIRST because the two halves have
/// different ceilings: a candidate drawn from `x` can be adjacent to the whole of
/// `p`, while one drawn from `p` is never its own neighbour and tops out one
/// short. In that order a candidate hitting its own ceiling is provably the best
/// of the whole set, so this is an early exit rather than an approximation --
/// the pivot is exactly the one the exhaustive walk would have chosen, up to
/// ties, which `HashSet` iteration order already decided.
///
/// What it buys is the case the ceilings were misfiring on. In a complete
/// component -- 600 copies of one file, a folder where everything matches
/// everything -- `x` is empty and the first candidate covers all of `p` but
/// itself, so the walk is one intersection instead of `|p|` of them. That takes
/// the whole search from `~d^3 / 3` probes to `~d^2` for a clique of `d`
/// members, which is the difference between a group of 600 being reported and
/// being abandoned as too dense to enumerate.
fn choose_pivot(
    p: &HashSet<usize>,
    x: &HashSet<usize>,
    adjacency: &[HashSet<usize>],
    search: &mut Search,
) -> Option<usize> {
    // Disjoint by construction, so chaining them is `p ∪ x` with nothing
    // repeated -- and unlike `HashSet::union` it keeps the two halves apart,
    // which is what lets each carry its own ceiling.
    let candidates = x
        .iter()
        .map(|&v| (v, p.len()))
        .chain(p.iter().map(|&v| (v, p.len().saturating_sub(1))));

    let mut best: Option<(usize, usize)> = None;

    for (v, ceiling) in candidates {
        if !search.spend(intersection_cost(&adjacency[v], p)) {
            return None;
        }
        let covered = adjacency[v].intersection(p).count();

        if best.is_none_or(|(_, seen)| covered > seen) {
            best = Some((v, covered));
        }
        if covered >= ceiling {
            break;
        }
    }

    best.map(|(v, _)| v)
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
/// `p` skips the most -- see `choose_pivot`, which is where most of a step's
/// work goes and where it is metered.
///
/// Every set operation is charged against the search's budget before it runs,
/// and every step is checked against the ceilings that are not about work, so a
/// component that turns out to be dense enough to enumerate forever stops
/// instead -- and so Ctrl-C is answered here like it is in every other stage.
/// See `Search`.
fn bron_kerbosch(
    r: HashSet<usize>,
    mut p: HashSet<usize>,
    mut x: HashSet<usize>,
    search: &mut Search,
) {
    if !search.may_continue() {
        return;
    }
    // What a step that examines nothing still costs, so a search cannot recurse
    // for free.
    if !search.spend(1) {
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

    // `p` is non-empty, so there is always a candidate: `None` here means the
    // budget ran out inside the walk, and `halted` already says so.
    let Some(pivot) = choose_pivot(&p, &x, adjacency, search) else {
        return;
    };

    // `HashSet::difference` walks `p`, looking each member up in the pivot's
    // neighbours.
    if !search.spend(p.len()) {
        return;
    }
    let branches: Vec<usize> = p.difference(&adjacency[pivot]).copied().collect();

    for v in branches {
        let neighbors = &adjacency[v];

        // The two intersections this branch is about to build. They used to ride
        // on the pivot walk, which was always the larger of the two by a factor
        // of `|p ∪ x|`; now that the walk can stop after a single candidate they
        // have to answer for themselves.
        if !search.spend(intersection_cost(neighbors, &p) + intersection_cost(neighbors, &x)) {
            return;
        }

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

/// Every maximal clique of every component, or `None` if Ctrl-C arrived.
///
/// A component whose search halts on a ceiling contributes nothing and is
/// recorded as a problem; an interrupt abandons the lot, because a library
/// grouped from half its components is not the library the user asked about.
fn enumerate(
    adjacency: &[HashSet<usize>],
    fingerprints: &[VideoFingerprint],
    stats: &RunStats,
    ceilings: Ceilings,
) -> Option<Vec<HashSet<usize>>> {
    let mut cliques = Vec::new();

    for component in components(adjacency) {
        let mut search = Search::new(adjacency, ceilings);

        bron_kerbosch(
            HashSet::new(),
            component.iter().copied().collect(),
            HashSet::new(),
            &mut search,
        );

        // Taken before the match: the abandoned arm reports it, and the clean arm
        // is about to drain the vector it comes from.
        let found = search.cliques.len();

        match search.halted {
            None => cliques.append(&mut search.cliques),
            Some(Halt::Interrupted) => return None,
            Some(reason) => stats.clustering_abandoned.record(abandoned_detail(
                &component,
                &reason,
                found,
                ceilings,
                fingerprints,
            )),
        }
    }

    Some(cliques)
}

/// `enumerate`, on a stack sized by this program rather than by `ulimit -s`.
///
/// See `SEARCH_STACK_BYTES` for why the search does not simply run on the
/// caller's stack. A thread that will not spawn -- `RLIMIT_NPROC`, no memory for
/// the mapping -- is no reason to refuse the work, so that case falls back to the
/// caller's own stack, which is exactly what every build before this one used.
fn enumerate_deeply(
    adjacency: &[HashSet<usize>],
    fingerprints: &[VideoFingerprint],
    stats: &RunStats,
    ceilings: Ceilings,
) -> Option<Vec<HashSet<usize>>> {
    std::thread::scope(|scope| {
        let spawned = std::thread::Builder::new()
            .stack_size(SEARCH_STACK_BYTES)
            .spawn_scoped(scope, || enumerate(adjacency, fingerprints, stats, ceilings));

        match spawned {
            // A panic in there is this program's bug, and it belongs on the way
            // out rather than quietly rendered as "no groups found".
            Ok(handle) => handle
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
            Err(e) => {
                log::debug!(
                    "Could not give the group search a stack of its own ({}); using this thread's.",
                    e
                );
                enumerate(adjacency, fingerprints, stats, ceilings)
            }
        }
    })
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

    let Some(cliques) = enumerate_deeply(&adjacency, fingerprints, stats, ceilings) else {
        // Ctrl-C. A partial enumeration is not a report.
        return Vec::new();
    };

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

    /// The folder `mock_library` puts its files in.
    ///
    /// Absolute, because every path reaching this module is: `sources::collect`
    /// canonicalizes each one. Bare file names would leave `abandoned_detail`
    /// looking for a shared folder that a real run always has.
    const MOCK_DIR: &str = "/library";

    /// `n` fingerprints whose paths sort in index order, so a group printed as
    /// `[0, 1, 2]` is also the path order the real sort would produce.
    fn mock_library(n: usize) -> Vec<VideoFingerprint> {
        (0..n)
            .map(|i| mock_fingerprint(&format!("{}/{:04}.mp4", MOCK_DIR, i)))
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
    fn test_the_problem_line_names_a_folder_and_a_count() {
        // The summary is the only place a user learns that files went
        // unreported, so it has to say how many and where they are. A folder
        // rather than a member: `--exclude` and a narrower scan both take one,
        // and nothing was ever wrong with the file this used to name.
        let (n, edges) = moon_moser(20);
        let fps = mock_library(n);
        let stats = RunStats::default();

        group_within(n, edges, &fps, &stats, Ceilings { budget: 10_000, groups: usize::MAX });

        let lines = stats.clustering_abandoned.samples();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains(&format!("{} file(s)", n)), "got: {}", lines[0]);
        assert!(lines[0].ends_with(&format!("all under {}", MOCK_DIR)), "got: {}", lines[0]);
        assert!(!lines[0].contains(".mp4"), "no member is named any more: {}", lines[0]);
    }

    #[test]
    fn test_the_shared_folder_narrows_to_the_part_of_the_library_that_is_dense() {
        // What the folder buys over a member: one dense folder inside a larger
        // scan names THAT folder, so the fix -- tighten `-d`, exclude it, scan it
        // on its own -- has an argument. The files here are spread over two
        // seasons and only one of them is a tangle.
        let paths = [
            "/lib/season1/a.mkv",
            "/lib/season2/b.mkv",
            "/lib/season2/extras/c.mkv",
            "/lib/season2/d.mkv",
        ];
        let fps: Vec<VideoFingerprint> = paths.iter().map(|p| mock_fingerprint(p)).collect();

        // The dense component is the three files under season2.
        let dense = [1usize, 2, 3];
        let line = abandoned_detail(&dense, &Halt::Groups, 0, Ceilings::shipped(), &fps);
        assert!(line.ends_with("all under /lib/season2"), "got: {}", line);

        // Drag in the file from the other season and it can only name the root
        // they share, which is the honest answer rather than a narrower guess.
        let whole = [0usize, 1, 2, 3];
        let line = abandoned_detail(&whole, &Halt::Groups, 0, Ceilings::shipped(), &fps);
        assert!(line.ends_with("all under /lib"), "got: {}", line);
    }

    #[test]
    fn test_the_shared_folder_is_compared_a_path_component_at_a_time() {
        // The trap a string prefix falls into: `/lib/season1` and `/lib/season2`
        // share the CHARACTERS `/lib/season`, which is not a folder and is not
        // anywhere the user can go and look.
        let shared = common_parent(
            [
                Path::new("/lib/season1/a.mkv"),
                Path::new("/lib/season2/b.mkv"),
            ]
        );
        assert_eq!(shared, Some(PathBuf::from("/lib")));

        // A single file answers with the folder it is in, not with itself.
        let one = common_parent([Path::new("/lib/season1/a.mkv")]);
        assert_eq!(one, Some(PathBuf::from("/lib/season1")));

        // Files that share nothing but the root say so.
        let apart = common_parent([Path::new("/a/x.mkv"), Path::new("/b/y.mkv")]);
        assert_eq!(apart, Some(PathBuf::from("/")));

        // And a caller that hands over something a scan never produces -- a bare
        // name, with no folder at all -- gets no answer rather than a wrong one.
        assert_eq!(common_parent([Path::new("x.mkv")]), None);
        assert_eq!(common_parent(std::iter::empty()), None);
    }

    /// The one abandoned line a run of these ceilings produces.
    fn abandoned_line(ceilings: Ceilings) -> String {
        let (n, edges) = moon_moser(20);
        let fps = mock_library(n);
        let stats = RunStats::default();

        group_within(n, edges, &fps, &stats, ceilings);

        let lines = stats.clustering_abandoned.samples();
        assert_eq!(lines.len(), 1, "one component, one line");
        lines.into_iter().next().unwrap()
    }

    #[test]
    fn test_the_work_ceiling_says_how_many_groups_it_had_found() {
        // The figure that distinguishes the two shapes of "too dense". A budget
        // large enough to enumerate for a while stops partway through a real
        // enumeration; one that is not gets nowhere at all. Both are refused, and
        // a user who cannot tell them apart cannot tell whether a looser budget
        // would have helped.
        let barely = abandoned_line(Ceilings { budget: 200, groups: usize::MAX });
        assert!(
            barely.contains("ran past its work ceiling, with 0 found so far"),
            "got: {}",
            barely
        );

        let further = abandoned_line(Ceilings { budget: 2_000_000, groups: usize::MAX });
        let found: usize = further
            .split("with ")
            .nth(1)
            .and_then(|tail| tail.split(' ').next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("no count in: {}", further));
        assert!(found > 0, "a budget this size gets somewhere: {}", further);
    }

    #[test]
    fn test_the_group_ceiling_says_the_threshold_rather_than_a_count() {
        // The count is worthless here and the threshold is not: the search stops
        // the first time it EXCEEDS the ceiling, so the number of groups in hand
        // is always exactly one past it. Printing that would be printing the
        // constant back at the user as though it had been measured. What the line
        // has to carry is the scale -- the figure a `-d` ladder can be compared
        // against.
        let line = abandoned_line(Ceilings { budget: usize::MAX, groups: 5 });

        assert!(
            line.contains("they form more than 5 overlapping groups"),
            "got: {}",
            line
        );
        assert!(
            !line.contains("6 overlapping groups"),
            "one past the ceiling is what it holds, not what it means: {}",
            line
        );
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

    /// Every file matches every other, which is what a folder of copies is.
    fn complete(n: usize) -> Vec<(usize, usize)> {
        let mut edges = Vec::with_capacity(n * (n - 1) / 2);
        for a in 0..n {
            for b in (a + 1)..n {
                edges.push((a, b));
            }
        }
        edges
    }

    #[test]
    fn test_a_folder_of_identical_files_is_one_group_however_many_of_them_there_are() {
        // The case the ceilings used to fire hardest on, and the easiest one
        // there is: a complete component has exactly ONE maximal clique and one
        // branch per level, so there is nothing combinatorial here to refuse.
        // The old pivot walk priced every candidate in `p ∪ x` against `p` at
        // every level anyway, which is `d^3 / 3` probes, so anything past ~530
        // copies exhausted the budget and the whole group vanished from the
        // report -- under a message telling the user to tighten
        // `--hamming-distance`, which cannot separate files that are identical.
        //
        // 700 is past that line and nowhere near the ~5,800 the budget now
        // allows, so this is the shipped ceilings answering the question rather
        // than a test-sized version of them.
        let n = 700;
        let fps = mock_library(n);
        let stats = RunStats::default();

        let groups = group_within(n, complete(n), &fps, &stats, Ceilings::shipped());

        assert_eq!(stats.clustering_abandoned.count(), 0, "nothing here is too dense to answer");
        assert_eq!(groups.len(), 1, "a complete graph has exactly one maximal clique");
        assert_eq!(groups[0].len(), n, "and every file is in it");
    }

    #[test]
    fn test_the_pivot_is_still_the_one_adjacent_to_the_most_candidates() {
        // `choose_pivot` stops early, and it is only allowed to because the
        // candidate it stops on is provably the best of the whole set -- `x` is
        // walked first because a candidate drawn from it can cover the whole of
        // `p`, while one drawn from `p` is never its own neighbour and tops out
        // one short. If that ordering is ever lost the walk starts settling for
        // a worse pivot, which is not wrong but is slower in exactly the case
        // the pivot exists for, and nothing else here would notice.
        //
        // Vertex 0 is adjacent to all of `p` and sits in `x`; vertex 1 is in `p`
        // and adjacent to the rest of it; the others reach one member each.
        let n = 8;
        let p: HashSet<usize> = (1..=5).collect();
        let x: HashSet<usize> = [0usize, 6, 7].into_iter().collect();

        let mut edges: Vec<(usize, usize)> = (1..=5).map(|v| (0, v)).collect();
        for a in 1..=5 {
            for b in (a + 1)..=5 {
                edges.push((a, b));
            }
        }
        edges.push((6, 2));
        edges.push((7, 3));

        let adjacency = adjacency_of(n, &edges);
        let mut search = Search::new(&adjacency, Ceilings::shipped());

        let pivot = choose_pivot(&p, &x, &adjacency, &mut search).expect("a pivot exists");
        assert_eq!(pivot, 0, "the only candidate covering the whole of p");

        // And it stopped when it found it, rather than pricing every candidate.
        let exhaustive: usize = p
            .union(&x)
            .map(|&v| intersection_cost(&adjacency[v], &p))
            .sum();
        let spent = Ceilings::shipped().budget - search.budget;
        assert!(spent < exhaustive, "spent {} of an exhaustive {}", spent, exhaustive);
    }

    #[test]
    fn test_a_complete_neighbourhood_costs_one_intersection_to_pivot() {
        // The saving stated as the property the fix rests on. Everything is
        // adjacent to everything, so the first candidate examined already covers
        // the most anything could -- one short of `p`, since a vertex is never
        // its own neighbour -- and the walk has no reason to look at the rest.
        // Order cannot matter here: `x` is empty and every candidate is alike.
        let n = 40;
        let adjacency = adjacency_of(n, &complete(n));
        let p: HashSet<usize> = (0..n).collect();
        let x: HashSet<usize> = HashSet::new();

        let mut search = Search::new(&adjacency, Ceilings::shipped());
        assert!(choose_pivot(&p, &x, &adjacency, &mut search).is_some());

        let spent = Ceilings::shipped().budget - search.budget;
        assert_eq!(
            spent,
            intersection_cost(&adjacency[0], &p),
            "one intersection, not one per candidate"
        );
    }

    #[test]
    fn test_the_budget_still_refuses_a_genuinely_explosive_component() {
        // The other half of the same change: metering each set operation instead
        // of pricing a step up front must not have made the ceiling toothless.
        // Moon-Moser at 20 parts is 3^20 maximal cliques out of 60 files, and no
        // amount of waiting turns that into a usable report.
        let (n, edges) = moon_moser(20);
        let fps = mock_library(n);
        let stats = RunStats::default();

        let groups = group_within(n, edges, &fps, &stats, Ceilings::shipped());

        assert!(groups.is_empty(), "a partial enumeration must not be reported");
        assert_eq!(stats.clustering_abandoned.count(), 1);
    }
}

