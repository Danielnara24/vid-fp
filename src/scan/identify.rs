//! The content pass: recognise moved files, and find byte-identical copies.
//! See `content` for what a file's identity is made of.

use super::Job;
use crate::cache::{self, Cache, Stamp};
use crate::content;
use crate::fingerprint::VideoFingerprint;
use crate::progress;
use crate::sources;
use crate::utils::shutdown_requested;
use anyhow::Result;
use log::info;
use rayon::prelude::*;
use std::collections::HashMap;

/// The identities the cache holds, sorted by what they can be used for.
///
/// `current` are the records still describing a file this pass is about: that
/// file's sample and digest without reading it. `former` are the rest -- a path
/// this run did not walk, or one whose file has changed since -- keyed by what
/// the path HELD. Those are what a moved or touched file is recognised by, so
/// they are gathered only when some file is queued to be recognised; a walk of
/// them reads each record's head and not its fingerprint.
struct IdentityIndex {
    current: HashMap<String, content::Identity>,
    former: HashMap<(u64, content::Sample), Vec<(String, content::Identity)>>,
    view: cache::View,
}

impl IdentityIndex {
    fn open(
        cache: &Cache,
        files: &HashMap<&str, &sources::ScannedFile>,
        subjects: &[String],
        adopting: bool,
    ) -> Self {
        let view = cache.view();
        let describes = |path: &str, identity: &content::Identity| {
            files.get(path).is_some_and(|file| identity.describes(file))
        };
        let current = subjects
            .iter()
            .filter_map(|path| Some((path.clone(), view.identity(path)?)))
            .filter(|(path, identity)| describes(path, identity))
            .collect();
        let mut former: HashMap<_, Vec<_>> = HashMap::new();
        if adopting {
            // In path order, which is the order the table iterates in, so the
            // holder a file adopts from does not depend on anything else.
            for (path, identity) in view.identities() {
                if !describes(&path, &identity) {
                    former.entry((identity.size, identity.sample)).or_default().push((path, identity));
                }
            }
        }
        Self { current, former, view }
    }

    /// The fingerprint a file holding `identity` can take over, if a former
    /// holder of the same bytes left one this run's sampling can use. The
    /// holder's identity and fingerprint are one record, so they are about the
    /// same bytes by construction. Where both sides were read whole, the
    /// digests have to agree as well; where either was not, the sample is the
    /// cache key it is documented to be.
    fn adoptable(&self, identity: &content::Identity, sampling: &cache::Sampling) -> Option<VideoFingerprint> {
        self.former.get(&(identity.size, identity.sample))?.iter().find_map(|(path, held)| {
            if let (Some(a), Some(b)) = (identity.full, held.full) {
                if a != b {
                    return None;
                }
            }
            self.view.fingerprint_held_with(path, sampling)
        })
    }
}

/// What the content pass decided.
pub(super) struct Identified {
    /// The files left to decode. A file whose byte-identical copies are also
    /// waiting carries them as `followers`, and they take its fingerprint.
    pub(super) todo: Vec<Job>,
    /// Files recognised by their content, with the fingerprint they took over.
    /// Already written to the cache under their own path.
    pub(super) adopted: Vec<VideoFingerprint>,
    /// Every set of byte-identical files among those that may be compared, by
    /// path.
    pub(super) identical: Vec<Vec<String>>,
}

/// The content pass: recognise moved files, and find byte-identical copies.
///
/// Runs after the weighing pass, so a file refused as not video or skipped as
/// too short is never sampled -- on a `-x '*'` scan of a home directory that is
/// nearly every file. `hits` are the files the cache answered by path, which
/// take part so a new copy of an old video is recognised, and so that two copies
/// already fingerprinted are still reported as identical. See `content`.
///
/// `Ok(None)` is an interrupt.
pub(super) fn identify(
    cache: &Cache,
    files: &HashMap<&str, &sources::ScannedFile>,
    hits: &[VideoFingerprint],
    todo: Vec<Job>,
    sampling: cache::Sampling,
) -> Result<Option<Identified>> {
    // Every subject is a walked file: hits first, then the queue, so a subject
    // number below `hits.len()` is a hit.
    let paths: Vec<String> =
        hits.iter().map(|f| f.path.clone()).chain(todo.iter().map(|j| j.path.clone())).collect();
    let n_hits = hits.len();
    let index = IdentityIndex::open(cache, files, &paths, !todo.is_empty());

    let mut identities: Vec<Option<content::Identity>> =
        paths.iter().map(|p| index.current.get(p.as_str()).copied()).collect();
    let mut dirty = vec![false; paths.len()];

    // 1. Sample whatever has no identity yet.
    let unsampled: Vec<usize> = (0..paths.len()).filter(|&i| identities[i].is_none()).collect();
    if !unsampled.is_empty() {
        let n = unsampled.len() as u64;
        let bar = progress::begin_counted(progress::Stage::Identifying, n, n);
        let sampled: Vec<(usize, Option<content::Identity>)> = unsampled
            .par_iter()
            .map(|&i| {
                let found = if shutdown_requested() {
                    None
                } else {
                    let file = files[paths[i].as_str()];
                    match content::sample(&file.path, file.size) {
                        Ok(s) => Some(content::Identity::of(file, s)),
                        Err(e) => {
                            // Not a problem: the file is fingerprinted the
                            // ordinary way and simply cannot be recognised.
                            log::debug!("Could not sample {}: {}", file.path, e);
                            None
                        }
                    }
                };
                bar.tick();
                (i, found)
            })
            .collect();
        if shutdown_requested() {
            return Ok(None);
        }
        for (i, found) in sampled {
            dirty[i] = found.is_some();
            identities[i] = found;
        }
    }

    // 2. Files agreeing on size and sample might be copies: read them whole.
    let possible = content::sets_by(
        identities.iter().enumerate().filter_map(|(i, id)| id.map(|id| (i, (id.size, id.sample)))),
    );
    let to_read: Vec<usize> = possible
        .iter()
        .flatten()
        .copied()
        .filter(|&i| identities[i].is_some_and(|id| id.full.is_none()))
        .collect();
    if !to_read.is_empty() {
        let total: u64 = to_read.iter().map(|&i| files[paths[i].as_str()].size).sum();
        progress::learn(|k| k.hashing = Some(total));
        let bar = progress::begin_counted(progress::Stage::Hashing, total, to_read.len() as u64);
        let read: Vec<(usize, Option<content::Digest>)> = to_read
            .par_iter()
            .map(|&i| {
                let file = files[paths[i].as_str()];
                let tick = |n: u64| bar.add(0, n);
                let read = match content::digest(&file.path, file.size, &tick) {
                    Ok(d) => (i, d),
                    Err(e) => {
                        log::debug!("Could not read {} whole: {}", file.path, e);
                        (i, None)
                    }
                };
                bar.add(1, 0);
                read
            })
            .collect();
        if shutdown_requested() {
            return Ok(None);
        }
        for (i, digest) in read {
            if let (Some(id), Some(d)) = (identities[i].as_mut(), digest) {
                id.full = Some(d);
                dirty[i] = true;
            }
        }
    }

    // 3. Equal digests are identical files.
    let exact = content::sets_by(identities.iter().enumerate().filter_map(|(i, id)| {
        id.and_then(|id| id.full.map(|d| (i, (id.size, d))))
    }));

    // 4. Decide every queued file: taken from a copy, following a copy, taken
    // from a former holder of its bytes, or decoded.
    enum Fate {
        Decode,
        Adopt(Box<VideoFingerprint>),
        Follow(usize),
    }
    // A former holder of the bytes first, for every queued file: a copy of
    // something the cache already knows needs no decode at all.
    let mut fates: Vec<Fate> = (0..todo.len())
        .map(|t| {
            identities[n_hits + t]
                .and_then(|id| index.adoptable(&id, &sampling))
                .map_or(Fate::Decode, |fp| Fate::Adopt(Box::new(fp)))
        })
        .collect();
    // Then the copies among this run's own files. A set takes its fingerprint
    // from a member that already has one -- a cache hit, or a queued file the
    // step above just answered -- and only a set where nobody has one decodes,
    // once, for all of it.
    for set in &exact {
        let known = set.iter().find_map(|&i| {
            if i < n_hits {
                Some(hits[i].clone())
            } else {
                match &fates[i - n_hits] {
                    Fate::Adopt(fp) => Some((**fp).clone()),
                    _ => None,
                }
            }
        });
        let queued: Vec<usize> = set.iter().filter(|&&i| i >= n_hits).map(|&i| i - n_hits).collect();
        match known {
            Some(fp) => {
                for &t in &queued {
                    fates[t] = Fate::Adopt(Box::new(fp.clone()));
                }
            }
            None => {
                for &t in &queued[1..] {
                    fates[t] = Fate::Follow(queued[0]);
                }
            }
        }
    }
    drop(index);

    // 5. Write what was learned, and what was adopted, in one transaction.
    let mut adopted: Vec<(String, Stamp, VideoFingerprint)> = Vec::new();
    let mut kept: Vec<Option<Job>> = Vec::with_capacity(todo.len());
    let mut followers: Vec<(usize, Job)> = Vec::new();
    for (job, fate) in todo.into_iter().zip(fates) {
        match fate {
            Fate::Decode => kept.push(Some(job)),
            Fate::Adopt(fp) => {
                let mut fp = *fp;
                fp.path = job.path.clone();
                adopted.push((job.path, job.stamp, fp));
                kept.push(None);
            }
            Fate::Follow(leader) => {
                followers.push((leader, job));
                kept.push(None);
            }
        }
    }
    let follower_count = followers.len();
    for (leader, job) in followers {
        if let Some(lead) = kept[leader].as_mut() {
            lead.followers.push(job);
        }
    }

    let learned: Vec<(&str, content::Identity)> = (0..paths.len())
        .filter(|&i| dirty[i])
        .filter_map(|i| identities[i].map(|id| (paths[i].as_str(), id)))
        .collect();
    let writes: Vec<cache::Write> = learned
        .iter()
        .map(|&(path, identity)| cache::Write::Identity(path, identity))
        .chain(adopted.iter().map(|(path, stamp, fp)| cache::Write::Fingerprint(path, *stamp, fp)))
        .collect();
    if let Err(e) = cache.write(&writes) {
        // Losing these costs the next run a re-sample, or a re-adoption, and
        // nothing else -- the fingerprints adopted are in hand either way.
        log::debug!("Could not record {} identit(ies): {:#}", learned.len(), e);
    }

    let identical: Vec<Vec<String>> = exact
        .iter()
        .map(|set| set.iter().map(|&i| paths[i].clone()).collect())
        .collect();

    if !adopted.is_empty() {
        info!(
            "Recognised {} file(s) by their content (moved, renamed or touched since they were \
             fingerprinted), so they are not decoded again.",
            adopted.len()
        );
    }
    if follower_count > 0 {
        info!(
            "{} file(s) are byte-identical copies of another file being fingerprinted and share \
             its decode.",
            follower_count
        );
    }

    Ok(Some(Identified {
        todo: kept.into_iter().flatten().collect(),
        adopted: adopted.into_iter().map(|(_, _, fp)| fp).collect(),
        identical,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::Known;
    use crate::stats::RunStats;
    use crate::tests::{cache_lookup, cache_store_one, cached_paths, library_with_a_pair, mock_fp, temp_db, verdicts};
    use crate::{run, Args};
    use clap::Parser;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    fn fixture(name: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/fixtures");
        path.push(name);
        path
    }

    fn scanned(path: &Path) -> sources::ScannedFile {
        use std::os::unix::fs::MetadataExt;
        let meta = std::fs::metadata(path).unwrap();
        sources::ScannedFile {
            path: path.to_string_lossy().into_owned(),
            size: meta.len(),
            mtime: meta.mtime(),
            mtime_nsec: meta.mtime_nsec(),
        }
    }

    fn job_for(file: &sources::ScannedFile, kf_interval: f64) -> Job {
        Job::of(file, cache::Sampling { kf_interval, min_kf_samples: 12.0 })
    }

    /// The content pass over `queued`, as the scan would call it with `hits`
    /// already answered by path.
    fn identify_over(
        db: &Cache,
        hits: &[VideoFingerprint],
        queued: &[&Path],
        kf_interval: f64,
    ) -> Identified {
        let files: Vec<sources::ScannedFile> = hits
            .iter()
            .map(|fp| scanned(Path::new(&fp.path)))
            .chain(queued.iter().map(|p| scanned(p)))
            .collect();
        let by_path: HashMap<&str, &sources::ScannedFile> =
            files.iter().map(|f| (f.path.as_str(), f)).collect();
        let todo = files[hits.len()..].iter().map(|f| job_for(f, kf_interval)).collect();
        identify(db, &by_path, hits, todo, cache::Sampling { kf_interval, min_kf_samples: 12.0 }).unwrap().unwrap()
    }

    /// Fingerprint everything under `dir` the ordinary way, so the cache holds
    /// it as a real run left it.
    fn scan(dir: &Path, db: &Cache, extra: &[&str]) -> RunStats {
        let mut argv = vec!["vid-fp".to_string(), dir.to_string_lossy().into_owned()];
        argv.extend(extra.iter().map(|s| s.to_string()));
        let stats = RunStats::default();
        run(&Args::parse_from(argv), Some(db), Instant::now(), 1, &stats).unwrap();
        stats
    }

    fn cached(db: &Cache, path: &Path) -> Option<VideoFingerprint> {
        verdicts(db).into_iter().find_map(|(p, k)| match k {
            Known::Fingerprint(fp) if p == path.to_string_lossy() => Some(*fp),
            _ => None,
        })
    }

    #[test]
    fn test_a_moved_file_takes_its_fingerprint_with_it() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let before = library.join("a.mp4");
        std::fs::copy(fixture("test_video.mp4"), &before).unwrap();
        scan(&library, &db, &[]);
        let original = cached(&db, &before).expect("the first run fingerprints it");

        let elsewhere = dir.path().join("elsewhere");
        std::fs::create_dir(&elsewhere).unwrap();
        let after = elsewhere.join("renamed.mp4");
        std::fs::rename(&before, &after).unwrap();

        let identified = identify_over(&db, &[], &[&after], 0.0);
        assert!(identified.todo.is_empty(), "nothing is left to decode");
        assert_eq!(identified.adopted.len(), 1);
        let adopted = &identified.adopted[0];
        assert_eq!(adopted.path, after.to_string_lossy());
        assert_eq!(adopted.valid_hashes, original.valid_hashes);

        // Filed under the new path, so the next run is an ordinary hit.
        assert_eq!(cached(&db, &after).unwrap().valid_hashes, original.valid_hashes);
        let hit = cache_lookup(&db, &after.to_string_lossy(), &job_for(&scanned(&after), 0.0).stamp);
        assert!(hit.is_some(), "the adopted entry carries the new path's stamp");
    }

    #[test]
    fn test_a_touched_file_is_not_decoded_again() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let path = dir.path().join("a.mp4");
        std::fs::copy(fixture("test_video.mp4"), &path).unwrap();
        scan(dir.path(), &db, &[]);

        let file = std::fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::now() + Duration::from_secs(3600)).unwrap();
        drop(file);

        let identified = identify_over(&db, &[], &[&path], 0.0);
        assert!(identified.todo.is_empty());
        assert_eq!(identified.adopted.len(), 1);
    }

    #[test]
    fn test_a_fingerprint_made_with_other_sampling_settings_is_not_taken_over() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let before = library.join("a.mp4");
        std::fs::copy(fixture("test_video.mp4"), &before).unwrap();
        scan(&library, &db, &[]);

        let after = dir.path().join("b.mp4");
        std::fs::rename(&before, &after).unwrap();

        let identified = identify_over(&db, &[], &[&after], 5.0);
        assert!(identified.adopted.is_empty(), "those hashes are not the ones -i 5 takes");
        assert_eq!(identified.todo.len(), 1);
    }

    #[test]
    fn test_an_identity_whose_fingerprint_was_replaced_vouches_for_nothing() {
        // The proof that a fingerprint came from a file's bytes is that its
        // stamp is the stat the identity was taken at. Overwrite the entry
        // with one stamped otherwise and the identity has to stop answering.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let before = library.join("a.mp4");
        std::fs::copy(fixture("test_video.mp4"), &before).unwrap();
        scan(&library, &db, &[]);

        let mut other = job_for(&scanned(&before), 0.0).stamp;
        other.mtime += 1;
        cache_store_one(&db, &before.to_string_lossy(), other, &mock_fp(&before.to_string_lossy()))
            .unwrap();

        let after = dir.path().join("b.mp4");
        std::fs::rename(&before, &after).unwrap();
        let identified = identify_over(&db, &[], &[&after], 0.0);
        assert!(identified.adopted.is_empty());
        assert_eq!(identified.todo.len(), 1);
    }

    #[test]
    fn test_identical_copies_waiting_together_are_decoded_once() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let paths: Vec<PathBuf> = ["a.mp4", "b.mp4", "c.mp4"]
            .iter()
            .map(|n| {
                let p = dir.path().join(n);
                std::fs::copy(fixture("test_video.mp4"), &p).unwrap();
                p
            })
            .collect();
        let other = dir.path().join("d.ts");
        std::fs::copy(fixture("test_video_segment.ts"), &other).unwrap();

        let queued: Vec<&Path> =
            paths.iter().map(|p| p.as_path()).chain([other.as_path()]).collect();
        let identified = identify_over(&db, &[], &queued, 0.0);

        assert_eq!(identified.todo.len(), 2, "one of the copies, and the other video");
        let leader = identified.todo.iter().find(|j| j.path.ends_with("a.mp4")).unwrap();
        let followers: Vec<&str> = leader.followers.iter().map(|j| j.path.as_str()).collect();
        assert_eq!(followers.len(), 2);
        assert!(followers[0].ends_with("b.mp4") && followers[1].ends_with("c.mp4"));
        assert_eq!(identified.identical.len(), 1);
        assert_eq!(identified.identical[0].len(), 3);
    }

    #[test]
    fn test_a_new_copy_of_a_fingerprinted_video_takes_its_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let a = library.join("a.mp4");
        std::fs::copy(fixture("test_video.mp4"), &a).unwrap();
        scan(&library, &db, &[]);
        let hit = cached(&db, &a).unwrap();

        let b = library.join("b.mp4");
        std::fs::copy(&a, &b).unwrap();
        let identified = identify_over(&db, &[hit], &[&b], 0.0);
        assert!(identified.todo.is_empty());
        assert_eq!(identified.adopted.len(), 1);
        assert_eq!(identified.identical.len(), 1);
    }

    #[test]
    fn test_copies_of_a_video_that_moved_are_all_recognised() {
        // Two copies of a file the cache knows under a path that is gone. The
        // first of them used to be both the set's decoder and a file the cache
        // answered, and the other was left with neither.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let original = library.join("a.mp4");
        std::fs::copy(fixture("test_video.mp4"), &original).unwrap();
        scan(&library, &db, &[]);

        let b = dir.path().join("b.mp4");
        let c = dir.path().join("c.mp4");
        std::fs::copy(&original, &b).unwrap();
        std::fs::rename(&original, &c).unwrap();

        let identified = identify_over(&db, &[], &[&b, &c], 0.0);
        assert!(identified.todo.is_empty(), "nothing to decode");
        assert_eq!(identified.adopted.len(), 2, "and nothing dropped");
        assert_eq!(identified.identical.len(), 1);
    }

    #[test]
    fn test_files_that_share_a_sample_but_not_their_bytes_are_not_identical() {
        // Same length, same three sampled stretches, one byte apart between
        // them: a cache may treat them alike, the report may not.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let size = 20 * content::CHUNK as usize;
        let bytes: Vec<u8> = (0..size).map(|i| (i * 31 % 251) as u8).collect();
        let a = dir.path().join("a.mp4");
        let b = dir.path().join("b.mp4");
        std::fs::write(&a, &bytes).unwrap();
        let mut changed = bytes.clone();
        changed[3 * content::CHUNK as usize] ^= 0xff;
        std::fs::write(&b, &changed).unwrap();

        let identified = identify_over(&db, &[], &[&a, &b], 0.0);
        assert!(identified.identical.is_empty());
        assert_eq!(identified.todo.len(), 2, "and neither stands in for the other's decode");
        assert!(identified.todo.iter().all(|j| j.followers.is_empty()));
    }

    #[test]
    fn test_a_run_over_identical_copies_says_so_in_the_report() {
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let (library, report) = library_with_a_pair(&dir);
        let stats = scan(&library, &db, &["-o", &report.to_string_lossy()]);
        assert!(!stats.had_problems());

        let csv = std::fs::read_to_string(&report).unwrap();
        let header: Vec<&str> = csv.lines().next().unwrap().split(';').collect();
        let col = header.iter().position(|h| *h == "identical").unwrap();
        let rows: Vec<Vec<&str>> =
            csv.lines().skip(1).filter(|l| !l.is_empty()).map(|l| l.split(';').collect()).collect();
        assert_eq!(rows.len(), 2, "{}", csv);
        assert!(rows.iter().all(|r| r[col] == "true"), "{}", csv);

        // Both copies are cached, each under its own path, from the one decode.
        let a = cached(&db, &library.join("a.mp4")).unwrap();
        let b = cached(&db, &library.join("b.mp4")).unwrap();
        assert_eq!(a.valid_hashes, b.valid_hashes);
        assert_ne!(a.path, b.path);
    }

    #[test]
    fn test_a_prune_after_a_move_keeps_the_moved_file_s_fingerprint() {
        // The prune used to run before the cache pass; now that a moved file
        // is found through its OLD entry, pruning first would have thrown that
        // entry away and decoded the file again.
        let dir = tempfile::tempdir().unwrap();
        let db = temp_db(&dir);
        let library = dir.path().join("library");
        std::fs::create_dir(&library).unwrap();
        let before = library.join("a.mp4");
        std::fs::copy(fixture("test_video.mp4"), &before).unwrap();
        scan(&library, &db, &[]);

        // A marker in place of the real fingerprint, under the same stamp, so
        // the identity still vouches for it: a run that decodes the moved file
        // instead of adopting it ends up holding the real hashes, not these.
        let marker = mock_fp(&before.to_string_lossy());
        cache_store_one(&db, &before.to_string_lossy(), job_for(&scanned(&before), 0.0).stamp, &marker)
            .unwrap();

        let after = library.join("renamed.mp4");
        std::fs::rename(&before, &after).unwrap();
        let stats = scan(&library, &db, &["--prune-cache"]);
        assert!(!stats.had_problems());
        assert_eq!(cached_paths(&db), vec![after.to_string_lossy().into_owned()]);
        assert_eq!(cached(&db, &after).unwrap().valid_hashes, marker.valid_hashes);
        assert_eq!(stats.entries_pruned(), 1, "the old path's record, identity and fingerprint together");
    }
}
