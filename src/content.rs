//! What a file holds, as opposed to where it is.
//!
//! The cache is keyed by path, so a video that moved, was renamed, was
//! `touch`ed or came back from a backup used to be decoded again from
//! scratch -- hours, for a library that changed drives. This module gives each
//! fingerprinted file an [`Identity`]: the stat it was taken against, a
//! [`Sample`] of its bytes, and, for a file that might be a copy of another, a
//! [`Digest`] of all of them. `lib.rs` keeps the identities in a table of their
//! own beside the fingerprints and uses them two ways:
//!
//! - **A file the cache misses by path is looked up by its sample.** A record
//!   under another path (or under this path at an older stat) whose
//!   fingerprint is still the one it was taken beside hands its fingerprint
//!   over, and the file is not decoded.
//! - **Files whose size and sample agree are read whole and hashed**, and the
//!   ones whose digests agree are byte-identical: one of them is decoded for
//!   all, and the report says the link between them is a copy rather than a
//!   likeness.
//!
//! The two questions are deliberately held to different standards. Reusing a
//! fingerprint on the strength of a sample is a cache decision: what it risks
//! is a fingerprint describing a file whose size and three sampled stretches
//! are unchanged while something between them changed -- which for a
//! compressed video is a tag edit in place, and a tag is not in the picture.
//! Calling two files IDENTICAL is a claim the report prints and a deletion can
//! rest on, so it is never made from a sample: only from a digest of every
//! byte.
//!
//! Sampling is paid by a file the cache cannot answer by path, which is about
//! to be decoded anyway (the decode reads far more than three 64 KB stretches),
//! and once by each file an older build fingerprinted, which has no identity
//! yet. Hashing whole files is paid only by files that share a size and a
//! sample with another file in the run. A warm run with nothing new pays
//! neither.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::os::unix::fs::FileExt;

use crate::sources::ScannedFile;
use crate::utils::shutdown_requested;

/// How much of the file each sampled stretch reads.
///
/// Three of them -- the start, the middle and the end -- so a sample reads
/// 192 KB of a file however long it is. The start holds the container's
/// header (and, in a fast-start MP4, the whole index), the end holds the index
/// of one that is not, and the middle is compressed picture in any container,
/// which no two encodes share.
pub const CHUNK: u64 = 64 * 1024;

/// Hash of a file's length and three stretches of its bytes. A cache key, not
/// a proof of identity -- see the module comment.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Sample(pub [u8; 16]);

/// BLAKE3 of every byte of the file.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Digest(pub [u8; 32]);

/// What a path held, as of one stat of it.
///
/// Stored per path. It speaks for the file only while the stat still matches
/// ([`Identity::describes`]); after that it speaks for whatever bytes the path
/// HELD, which is exactly what a moved file is looked up by.
///
/// Appending a field is self-invalidating the way it is for a fingerprint:
/// older records run out of bytes, fail to deserialize, and are sampled again.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Identity {
    pub mtime: i64,
    pub mtime_nsec: i64,
    pub size: u64,
    pub sample: Sample,
    pub full: Option<Digest>,
}

impl Identity {
    /// Whether this record was taken from the file as it is now.
    pub fn describes(&self, file: &ScannedFile) -> bool {
        self.mtime == file.mtime && self.mtime_nsec == file.mtime_nsec && self.size == file.size
    }

    pub fn of(file: &ScannedFile, sample: Sample) -> Self {
        Self {
            mtime: file.mtime,
            mtime_nsec: file.mtime_nsec,
            size: file.size,
            sample,
            full: None,
        }
    }
}

/// Open `path` and check it is still the length the scan recorded.
///
/// A file that grew or shrank since the scan is not the file the stat
/// describes, and an identity filed under that stat would be one for the wrong
/// bytes. Refusing it costs nothing: the file is fingerprinted the ordinary
/// way, and the next run identifies it.
fn open_at_size(path: &str, size: u64) -> io::Result<File> {
    let file = File::open(path)?;
    let now = file.metadata()?.len();
    if now != size {
        return Err(io::Error::other(format!(
            "changed size while being identified ({} -> {} bytes)",
            size, now
        )));
    }
    Ok(file)
}

/// The sample of a file the scan measured at `size` bytes.
///
/// A file no longer than three stretches is hashed whole, so two short files
/// can only share a sample by sharing every byte.
pub fn sample(path: &str, size: u64) -> io::Result<Sample> {
    let file = open_at_size(path, size)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vid-fp sample v1");
    hasher.update(&size.to_le_bytes());

    let mut buf = vec![0u8; CHUNK as usize];
    for offset in sample_offsets(size) {
        let want = CHUNK.min(size - offset) as usize;
        file.read_exact_at(&mut buf[..want], offset)?;
        hasher.update(&buf[..want]);
    }

    let mut out = [0u8; 16];
    out.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
    Ok(Sample(out))
}

/// Where the sampled stretches start. Contiguous and non-overlapping when the
/// file is short enough to be read whole.
fn sample_offsets(size: u64) -> Vec<u64> {
    if size <= 3 * CHUNK {
        return (0..size.div_ceil(CHUNK)).map(|i| i * CHUNK).collect();
    }
    vec![0, size / 2 - CHUNK / 2, size - CHUNK]
}

/// BLAKE3 of the whole file, or `None` if the run was interrupted part way.
///
/// `progress` is told every buffer's length as it is read, so the bar over a
/// set of large copies moves in bytes rather than jumping a file at a time.
pub fn digest(path: &str, size: u64, progress: &dyn Fn(u64)) -> io::Result<Option<Digest>> {
    let mut file = open_at_size(path, size)?;
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut read_total = 0u64;
    loop {
        if shutdown_requested() {
            return Ok(None);
        }
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        read_total += n as u64;
        progress(n as u64);
    }
    // Read to the end and found a different length: the file changed under the
    // read, and the digest is of bytes no stat describes.
    if read_total != size {
        return Err(io::Error::other("changed size while being hashed"));
    }
    Ok(Some(Digest(*hasher.finalize().as_bytes())))
}

/// The members of every set of two or more items sharing a key, each set in
/// ascending index order, the sets ordered by their first member.
///
/// One function for both questions this module asks of a run's files --
/// "which might be copies" keyed by (size, sample) and "which are copies"
/// keyed by (size, digest) -- so the two cannot group differently.
pub fn sets_by<K: Eq + std::hash::Hash>(keys: impl IntoIterator<Item = (usize, K)>) -> Vec<Vec<usize>> {
    let mut by_key: HashMap<K, Vec<usize>> = HashMap::new();
    for (i, k) in keys {
        by_key.entry(k).or_default().push(i);
    }
    let mut sets: Vec<Vec<usize>> = by_key
        .into_values()
        .filter(|members| members.len() > 1)
        .map(|mut members| {
            members.sort_unstable();
            members
        })
        .collect();
    sets.sort_unstable_by_key(|s| s[0]);
    sets
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> (String, u64) {
        let path = dir.path().join(name);
        let mut f = File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        (path.to_string_lossy().into_owned(), bytes.len() as u64)
    }

    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    #[test]
    fn test_the_stretches_cover_a_short_file_whole_and_a_long_one_three_times() {
        assert_eq!(sample_offsets(0), Vec::<u64>::new());
        assert_eq!(sample_offsets(10), vec![0]);
        assert_eq!(sample_offsets(3 * CHUNK), vec![0, CHUNK, 2 * CHUNK]);
        let long = 10 * CHUNK;
        assert_eq!(sample_offsets(long), vec![0, 5 * CHUNK - CHUNK / 2, 9 * CHUNK]);
    }

    #[test]
    fn test_a_copy_samples_and_digests_alike_and_a_different_file_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = noise(1_000_000, 1);
        let (a, size) = write(&dir, "a.mkv", &bytes);
        let (b, _) = write(&dir, "b.mkv", &bytes);
        let mut other = bytes.clone();
        other[CHUNK as usize * 5] ^= 1; // between the sampled stretches
        let (c, _) = write(&dir, "c.mkv", &other);

        assert_eq!(sample(&a, size).unwrap(), sample(&b, size).unwrap());
        // The sample cannot see a byte it does not read -- which is why it is
        // never what calls two files identical.
        assert_eq!(sample(&a, size).unwrap(), sample(&c, size).unwrap());

        let none = |_| {};
        let da = digest(&a, size, &none).unwrap().unwrap();
        assert_eq!(da, digest(&b, size, &none).unwrap().unwrap());
        assert_ne!(da, digest(&c, size, &none).unwrap().unwrap());
    }

    #[test]
    fn test_the_length_is_part_of_the_sample() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = noise(10, 2);
        let (a, size) = write(&dir, "a", &bytes);
        let mut longer = bytes.clone();
        longer.push(0);
        let (b, size_b) = write(&dir, "b", &longer);
        assert_ne!(sample(&a, size).unwrap(), sample(&b, size_b).unwrap());
    }

    #[test]
    fn test_a_file_that_is_no_longer_the_size_the_scan_saw_is_not_identified() {
        let dir = tempfile::tempdir().unwrap();
        let (a, size) = write(&dir, "a", &noise(5000, 3));
        assert!(sample(&a, size + 1).is_err());
        assert!(digest(&a, size - 1, &|_| {}).is_err());
    }

    #[test]
    fn test_only_keys_held_by_two_or_more_make_a_set() {
        let sets = sets_by([(0, 'x'), (1, 'y'), (2, 'x'), (3, 'z'), (4, 'y'), (5, 'x')]);
        assert_eq!(sets, vec![vec![0, 2, 5], vec![1, 4]]);
        assert!(sets_by(std::iter::empty::<(usize, u8)>()).is_empty());
    }
}
