//! Thumbnails for the results page, made off the main thread.
//!
//! The scan keeps no picture of anything (a fingerprint is a list of 64-bit
//! hashes), so a video is shown by decoding one frame of it again
//! (`vid_fp::thumbnail::still`). What keeps that light:
//!
//! - **Only what is on screen is asked for**, and asking for a new group
//!   drops every request still queued for the old one (`clear_queue`), so
//!   paging quickly through a thousand groups decodes the one it stops at.
//! - **The desktop's own thumbnail is used when it has one** (`from_cache`):
//!   a file manager that has shown the folder has usually made one of every
//!   video in it, and reading that is a few kilobytes of PNG where a frame is
//!   a seek and a keyframe decode, which for a 4K HEVC file is most of a
//!   second.
//! - **Two decoders at most**: a decode holds a full-size frame for a moment.
//! - **A picture being decoded is not queued again**, and a picture is made
//!   at the size it is shown (`Size`), in the software renderer's own pixel
//!   format (premultiplied BGRA), so GTK neither converts nor scales it on
//!   every frame it draws.
//!
//! What is kept is the textures most recently shown, up to a number of
//! bytes, so paging back is instant.

use gtk::gdk;
use gtk::glib;
use gtk::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Condvar, Mutex};

/// Bytes of texture kept, on screen or not: some twenty screens of a group's
/// thumbnails, and hundreds of groups' pictures in the strip.
pub const KEEP_BYTES: usize = 48 << 20;

/// The place a picture is made for, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Size {
    pub w: u32,
    pub h: u32,
    /// Fill the place and cut what overhangs, rather than fit inside it.
    pub cover: bool,
    /// The keyframe the scan sampled at this many milliseconds, rather than
    /// a picture chosen to stand for the whole video (`thumbnail::frame_at`).
    pub at: Option<u32>,
}

impl Size {
    /// No larger than `w` x `h`, whole.
    pub fn within(w: u32, h: u32) -> Size {
        Size { w: w.max(1), h: h.max(1), cover: false, at: None }
    }

    /// Exactly `w` x `h`, filled.
    pub fn cover(w: u32, h: u32) -> Size {
        Size { w: w.max(1), h: h.max(1), cover: true, at: None }
    }

    /// The frame the scan sampled at `ms`, in this place, whole.
    pub fn at(self, ms: u32) -> Size {
        Size { at: Some(ms), cover: false, ..self }
    }
}

pub type Key = (PathBuf, Size);
type Decoded = Result<(u32, u32, Vec<u8>), String>;

struct Queue {
    jobs: VecDeque<Key>,
    /// Taken by a decoder and not yet `arrived`: asking for one again waits
    /// for that answer rather than starting another decode.
    in_flight: std::collections::HashSet<Key>,
}

/// What a picture widget is waiting for.
pub enum Shown {
    Still(crate::still::Still),
    /// A picture that takes whatever room it is given: the comparison's
    /// full-size view.
    Picture(gtk::Picture),
}

pub struct Thumbs {
    queue: Arc<(Mutex<Queue>, Condvar)>,
    done: RefCell<HashMap<Key, Result<gdk::Texture, String>>>,
    /// What `done` holds, oldest first, for dropping.
    order: RefCell<Budget<Key>>,
    /// Each widget waiting, with the ticket its request was given.
    waiting: RefCell<HashMap<Key, Vec<(u64, Shown)>>>,
    tickets: std::cell::Cell<u64>,
}

impl Thumbs {
    /// With at most `most` decoders and `keep` bytes of texture: the strip of
    /// groups has a loader of its own, so its pictures never hold up, or are
    /// dropped with, a group's.
    pub fn with_workers(most: usize, keep: usize) -> Rc<Thumbs> {
        let queue = Arc::new((Mutex::new(Queue { jobs: VecDeque::new(), in_flight: Default::default() }), Condvar::new()));
        let (tx, rx) = async_channel::unbounded::<(Key, Decoded)>();
        let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, most.max(1));
        for _ in 0..workers {
            let (queue, tx) = (queue.clone(), tx.clone());
            std::thread::spawn(move || loop {
                let key = {
                    let (m, cv) = &*queue;
                    let mut q = m.lock().unwrap();
                    loop {
                        if let Some(k) = q.jobs.pop_front() {
                            q.in_flight.insert(k.clone());
                            break k;
                        }
                        q = cv.wait(q).unwrap();
                    }
                };
                let got = make(&key.0, key.1).map(|(w, h, mut px)| {
                    to_bgra_premultiplied(&mut px);
                    (w, h, px)
                });
                if tx.send_blocking((key, got)).is_err() {
                    return;
                }
            });
        }
        let thumbs = Rc::new(Thumbs {
            queue,
            done: RefCell::new(HashMap::new()),
            order: RefCell::new(Budget::new(keep)),
            waiting: RefCell::new(HashMap::new()),
            tickets: std::cell::Cell::new(0),
        });
        let weak = Rc::downgrade(&thumbs);
        glib::spawn_future_local(async move {
            while let Ok((key, got)) = rx.recv().await {
                let Some(t) = weak.upgrade() else { break };
                t.arrived(key, got);
            }
        });
        thumbs
    }

    /// Drop every request not yet started: the pictures they were for are no
    /// longer on screen.
    pub fn clear_queue(&self) {
        self.queue.0.lock().unwrap().jobs.clear();
        self.waiting.borrow_mut().clear();
    }

    /// Show `path`, made for `size`, in `shown`: now if it is at hand, when
    /// it has been decoded otherwise. The ticket withdraws the request.
    pub fn request(&self, path: &Path, size: Size, shown: Shown) -> u64 {
        let ticket = self.tickets.get() + 1;
        self.tickets.set(ticket);
        let key = (path.to_path_buf(), size);
        if let Some(r) = self.done.borrow().get(&key) {
            deliver(&shown, r.clone());
            return ticket;
        }
        let mut waiting = self.waiting.borrow_mut();
        let first = !waiting.contains_key(&key);
        waiting.entry(key.clone()).or_default().push((ticket, shown));
        let (m, cv) = &*self.queue;
        let mut q = m.lock().unwrap();
        // Already being decoded: its answer reaches every widget waiting for
        // it, this one included, whenever it comes.
        if q.in_flight.contains(&key) {
            return ticket;
        }
        if first {
            q.jobs.push_back(key);
            cv.notify_one();
        }
        ticket
    }

    /// Withdraw request `ticket` for `key`: its widget now shows something
    /// else. Nothing else waiting for that picture, it is not decoded at all
    /// unless a decoder already has it.
    pub fn withdraw(&self, key: &Key, ticket: u64) {
        let mut waiting = self.waiting.borrow_mut();
        let Some(list) = waiting.get_mut(key) else { return };
        list.retain(|(t, _)| *t != ticket);
        if list.is_empty() {
            waiting.remove(key);
            self.queue.0.lock().unwrap().jobs.retain(|k| k != key);
        }
    }

    /// Forget a file, which is on its way to the Trash.
    pub fn forget(&self, path: &Path) {
        self.done.borrow_mut().retain(|k, _| k.0 != path);
        self.order.borrow_mut().remove_where(|k| k.0 == path);
    }

    fn arrived(&self, key: Key, got: Decoded) {
        // Here and not on the decoder's thread, so that a request between the
        // decode finishing and its answer arriving waits for this answer.
        self.queue.0.lock().unwrap().in_flight.remove(&key);
        let r = got.map(|(w, h, bgra)| {
            let bytes = glib::Bytes::from_owned(bgra);
            gdk::MemoryTexture::new(w as i32, h as i32, gdk::MemoryFormat::B8g8r8a8Premultiplied, &bytes, w as usize * 4).upcast::<gdk::Texture>()
        });
        // Taken out before delivering, since a delivery may ask for another.
        let list = self.waiting.borrow_mut().remove(&key);
        for (_, s) in list.iter().flatten() {
            deliver(s, r.clone());
        }
        let bytes = r.as_ref().map_or(64, |t| t.width() as usize * t.height() as usize * 4);
        self.done.borrow_mut().insert(key.clone(), r);
        for old in self.order.borrow_mut().insert(key, bytes) {
            self.done.borrow_mut().remove(&old);
        }
    }
}

/// A picture of `path` for `size`, as RGBA: the desktop's thumbnail when it
/// has a current one large enough, else a frame of the video.
fn make(path: &Path, size: Size) -> Decoded {
    let Some(p) = path.to_str() else { return Err("its name is not UTF-8".into()) };
    if let Some(ms) = size.at {
        return vid_fp::thumbnail::frame_at(p, ms, size.w, size.h).map_err(|e| format!("{e:#}"));
    }
    if let Some(t) = thumbnail_dir().and_then(|dir| from_cache(&dir, path, size)) {
        return Ok(t);
    }
    vid_fp::thumbnail::still(p, size.w, size.h, size.cover).map_err(|e| format!("{e:#}"))
}

/// RGBA to what cairo draws without converting: premultiplied, and BGRA in
/// memory, which is `CAIRO_FORMAT_ARGB32` on a little-endian machine. GTK's
/// software renderer converts any other format each time it draws one.
fn to_bgra_premultiplied(px: &mut [u8]) {
    for p in px.chunks_exact_mut(4) {
        let a = p[3] as u32;
        let (r, b) = (p[0], p[2]);
        if a == 255 {
            p[0] = b;
            p[2] = r;
        } else {
            let mul = |c: u8| ((c as u32 * a + 127) / 255) as u8;
            p[0] = mul(b);
            p[1] = mul(p[1]);
            p[2] = mul(r);
        }
    }
}

/// The desktop's thumbnail cache: `$XDG_CACHE_HOME/thumbnails`, or
/// `~/.cache/thumbnails`; the host's, inside a Flatpak (`desktop_dir`).
fn thumbnail_dir() -> Option<PathBuf> {
    Some(vid_fp::desktop_dir("XDG_CACHE_HOME", ".cache")?.join("thumbnails"))
}

/// A picture from the freedesktop thumbnail cache under `dir`, made for
/// `size`, as RGBA, when there is one that is current and at least that large.
///
/// The cache is the one the file managers share: a thumbnail is
/// `<size>/<md5 of the file's URI>.png`, and it is current when its
/// `Thumb::MTime` is the file's modification time in seconds, the spec's own
/// test, and its `Thumb::Size`, where it has one, the file's length. Only the
/// sizes at least as large as what is asked for are looked in, smallest
/// first, so a picture is never drawn from a thumbnail smaller than itself.
fn from_cache(dir: &Path, path: &Path, size: Size) -> Option<(u32, u32, Vec<u8>)> {
    use gtk::gdk_pixbuf::Pixbuf;
    let long = size.w.max(size.h);
    const SIZES: [(&str, u32); 4] = [("normal", 128), ("large", 256), ("x-large", 512), ("xx-large", 1024)];
    let md = std::fs::metadata(path).ok()?;
    let mtime = md.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    let uri = gtk::gio::File::for_path(path).uri();
    let md5 = glib::compute_checksum_for_string(glib::ChecksumType::Md5, uri.as_str())?;
    for &(sub, _) in SIZES.iter().filter(|(_, s)| *s >= long) {
        let thumb = dir.join(sub).join(format!("{md5}.png"));
        if !thumb.is_file() {
            continue;
        }
        let Ok(pb) = Pixbuf::from_file(&thumb) else { continue };
        let text = |key: &str| pb.option(&format!("tEXt::{key}")).map(|s| s.to_string());
        let uri_ok = text("Thumb::URI").is_none_or(|u| u == uri.as_str());
        let size_ok = text("Thumb::Size").is_none_or(|s| s.trim().parse::<u64>().ok() == Some(md.len()));
        let fresh = text("Thumb::MTime").and_then(|m| m.trim().parse::<u64>().ok()) == Some(mtime);
        if !(uri_ok && size_ok && fresh) {
            continue;
        }
        // To fill a place it must be as large as the place both ways: a wide
        // video's thumbnail is short.
        if size.cover && (pb.width() < size.w as i32 || pb.height() < size.h as i32) {
            continue;
        }
        if let Some(t) = fit_pixbuf(&pb, size) {
            return Some(t);
        }
    }
    None
}

/// `pb` at `size`, as RGBA, its shape kept.
fn fit_pixbuf(pb: &gtk::gdk_pixbuf::Pixbuf, size: Size) -> Option<(u32, u32, Vec<u8>)> {
    let (w, h) = (pb.width(), pb.height());
    if w <= 0 || h <= 0 || pb.bits_per_sample() != 8 {
        return None;
    }
    let (fw, fh) = vid_fp::thumbnail::fitted((w as f64, h as f64), size.w, size.h, size.cover);
    let scaled = pb.scale_simple(fw as i32, fh as i32, gtk::gdk_pixbuf::InterpType::Bilinear)?;
    let (cw, ch) = if size.cover { (fw.min(size.w), fh.min(size.h)) } else { (fw, fh) };
    let (x0, y0) = (((fw - cw) / 2) as usize, ((fh - ch) / 2) as usize);
    let (n, stride) = (scaled.n_channels() as usize, scaled.rowstride() as usize);
    let bytes = scaled.read_pixel_bytes();
    let mut out = Vec::with_capacity(cw as usize * ch as usize * 4);
    for y in y0..y0 + ch as usize {
        for x in x0..x0 + cw as usize {
            let p = bytes.get(y * stride + x * n..y * stride + x * n + n)?;
            out.extend_from_slice(&[p[0], p[1], p[2], if n == 4 { p[3] } else { 255 }]);
        }
    }
    Some((cw, ch, out))
}

/// Keys oldest first, each with its size, held to a total.
///
/// Inserting a key it already holds moves it to the newest end rather than
/// holding it twice. The newest key is always kept, however large.
pub struct Budget<K> {
    order: VecDeque<(K, usize)>,
    held: usize,
    limit: usize,
}

impl<K: PartialEq> Budget<K> {
    pub fn new(limit: usize) -> Self {
        Budget { order: VecDeque::new(), held: 0, limit }
    }

    /// Hold `key`, and hand back the keys dropped to make room for it.
    pub fn insert(&mut self, key: K, bytes: usize) -> Vec<K> {
        self.remove_where(|k| *k == key);
        self.order.push_back((key, bytes));
        self.held += bytes;
        let mut dropped = Vec::new();
        while self.held > self.limit && self.order.len() > 1 {
            let (k, b) = self.order.pop_front().expect("more than one");
            self.held -= b;
            dropped.push(k);
        }
        dropped
    }

    pub fn remove_where(&mut self, gone: impl Fn(&K) -> bool) {
        let held = &mut self.held;
        self.order.retain(|(k, b)| {
            let drop = gone(k);
            if drop {
                *held -= *b;
            }
            !drop
        });
    }
}

fn deliver(shown: &Shown, r: Result<gdk::Texture, String>) {
    match shown {
        Shown::Still(p) => match r {
            Ok(t) => {
                p.set_texture(Some(&t));
                p.set_tooltip_text(None);
            }
            Err(e) => {
                p.set_texture(None);
                p.set_tooltip_text(Some(&format!("No picture of this video: {e}")));
            }
        },
        Shown::Picture(p) => match r {
            Ok(t) => {
                p.set_paintable(Some(&t));
                p.set_tooltip_text(None);
            }
            Err(e) => {
                p.set_paintable(gdk::Paintable::NONE);
                p.set_tooltip_text(Some(&format!("No picture of this frame: {e}")));
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Held to bytes, oldest out first, and never two entries for one key.
    #[test]
    fn the_cache_is_held_to_bytes_not_to_a_count() {
        let mut b = Budget::new(100);
        assert!(b.insert("row1", 10).is_empty());
        assert!(b.insert("row2", 10).is_empty());
        assert_eq!(b.insert("big", 85), vec!["row1"]);
        assert_eq!(b.held, 95);
        // The same key again moves, rather than doubling.
        assert!(b.insert("row2", 10).is_empty());
        assert_eq!(b.order.iter().map(|e| e.0).collect::<Vec<_>>(), ["big", "row2"]);
        assert_eq!(b.held, 95);
        // Something larger than the whole budget is kept on its own.
        assert_eq!(b.insert("huge", 500), vec!["big", "row2"]);
        b.remove_where(|k| *k == "huge");
        assert_eq!(b.held, 0);
    }

    /// Premultiplied and swapped, and an opaque pixel only swapped.
    #[test]
    fn rgba_becomes_premultiplied_bgra() {
        let mut px = [10, 20, 30, 255, 200, 100, 50, 128, 255, 255, 255, 0];
        to_bgra_premultiplied(&mut px);
        assert_eq!(px, [30, 20, 10, 255, 25, 50, 100, 128, 0, 0, 0, 0]);
    }

    /// A PNG thumbnail of `w` x `h` saying it is of `uri` as it was at
    /// `mtime`, and, with `len`, that long.
    fn thumbnail_png(at: &Path, w: i32, h: i32, uri: &str, mtime: u64, len: Option<u64>) {
        use gtk::gdk_pixbuf::{Colorspace, Pixbuf};
        let pb = Pixbuf::new(Colorspace::Rgb, true, 8, w, h).unwrap();
        pb.fill(0x8080_80ff);
        let mtime = mtime.to_string();
        let len = len.map(|l| l.to_string());
        let mut opts: Vec<(&str, &str)> = vec![("tEXt::Thumb::URI", uri), ("tEXt::Thumb::MTime", &mtime)];
        if let Some(l) = &len {
            opts.push(("tEXt::Thumb::Size", l));
        }
        pb.savev(at, "png", &opts).unwrap();
    }

    /// A video is drawn from the desktop's thumbnail when one is current and
    /// large enough, and decoded otherwise.
    #[test]
    fn a_current_thumbnail_large_enough_is_used() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("thumbnails");
        std::fs::create_dir_all(cache.join("large")).unwrap();
        std::fs::create_dir_all(cache.join("normal")).unwrap();
        let video = dir.path().join("v.mp4");
        std::fs::write(&video, b"not read when the thumbnail answers").unwrap();
        let len = std::fs::metadata(&video).unwrap().len();
        let mtime = std::fs::metadata(&video).unwrap().modified().unwrap().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        let uri = gtk::gio::File::for_path(&video).uri();
        let md5 = glib::compute_checksum_for_string(glib::ChecksumType::Md5, uri.as_str()).unwrap();
        let row = Size::within(192, 108);

        assert!(from_cache(&cache, &video, row).is_none(), "none at all");
        // A 128-pixel one is too small for a 192-pixel place.
        thumbnail_png(&cache.join("normal").join(format!("{md5}.png")), 128, 72, &uri, mtime, None);
        assert!(from_cache(&cache, &video, row).is_none());
        // A 256-pixel one is used, shrunk to the place, its shape kept.
        let large = cache.join("large").join(format!("{md5}.png"));
        thumbnail_png(&large, 256, 144, &uri, mtime, Some(len));
        let (w, h, rgba) = from_cache(&cache, &video, row).expect("a current large thumbnail");
        assert_eq!((w, h, rgba.len()), (192, 108, 192 * 108 * 4));
        // Or filling a place exactly, cut to it.
        let (w, h, _) = from_cache(&cache, &video, Size::cover(96, 96)).unwrap();
        assert_eq!((w, h), (96, 96));
        // Not one made before the file last changed, nor one of another
        // file, nor one of a file of another length.
        thumbnail_png(&large, 256, 144, &uri, mtime - 1, None);
        assert!(from_cache(&cache, &video, row).is_none());
        thumbnail_png(&large, 256, 144, "file:///elsewhere.mp4", mtime, None);
        assert!(from_cache(&cache, &video, row).is_none());
        thumbnail_png(&large, 256, 144, &uri, mtime, Some(len + 1));
        assert!(from_cache(&cache, &video, row).is_none());
    }
}
