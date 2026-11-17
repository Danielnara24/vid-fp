//! What the scan matched between two videos, frame by frame: the window the
//! results page opens on a video to check a match before trusting it.
//!
//! The scan keeps nothing a person could look at, so the result carries, for
//! each grouped video, the frames of it that were matched and the frames of
//! the other video they were matched to (`found::Comparison`, made by
//! `compare::evidence` from the same decision the coverage figures come
//! from). This window decodes those frames again (`thumbnail::frame_at`) and
//! shows them in pairs, the video on the left, what it matched on the right,
//! with where they sit in each and how far apart their hashes were. A bad
//! match, two scenes that happen to hash alike, is plain at a glance; so is a
//! clip and where in the longer video it was cut from.
//!
//! Above the pairs, a chart: each video's runtime as a bar, its matched
//! footage filled in, a tick at each pair's frame in each, and a line from
//! one to the other. Both bars are on one time scale, and when most pairs
//! agree on where one video sits in the other (`alignment`) the shorter is
//! slid there, so a true copy or clip draws its lines straight down and a
//! slanted line is a match at some other moment. Where the pairs do not
//! agree both bars start at 0, and the lines cross. The zoom button (or Z)
//! narrows the chart to the matched footage at that same one scale, which is
//! what makes a short clip of a long video legible.
//!
//! A click on a pair selects it, on the chart as well: its ticks and line
//! red. A click on the chart selects the pair under it; several pairs can
//! share one frame of the other video, and clicking that tick again steps
//! through them. A double click, or Enter, shows a pair's two frames as
//! large as the window allows; Left and Right step through the pairs there,
//! Escape goes back.
//!
//! Against the video's strongest match first, which is what its row says it
//! matched; the group's reference, when that is another video, is the other
//! choice. Nothing here acts on any file.

use crate::thumbs::{Shown, Size, Thumbs, KEEP_BYTES};
use crate::App;
use gtk::prelude::*;
use gtk::{gdk, glib};
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use vid_fp::found::{Comparison, File};
use vid_fp::utils;

/// A frame of a pair in the list.
const FRAME_W: i32 = 400;
const FRAME_H: i32 = 225;
/// The full-size view decodes at most this large.
const FULL_W: u32 = 1920;
const FULL_H: u32 = 1080;

/// A video and what it was compared against: each other video with how it
/// came to be one (`"Best match"`, `"Identical copy"`, `"Group reference"`)
/// and the comparison.
pub struct Subject {
    pub file: File,
    pub against: Vec<(&'static str, Comparison, File)>,
}

/// The chart's rows, top to bottom: ticks over A, bar A, the band the lines
/// cross, bar B, ticks under B.
const TICK: f64 = 6.0;
const BAR: f64 = 12.0;
const BAND: f64 = 56.0;
const CHART_H: f64 = TICK + BAR + BAND + BAR + TICK;
/// Room at each end so a tick at the very start or end is still seen.
const PAD: f64 = 4.0;
/// How near the pointer must be to a tick or line to pick it, in pixels.
const REACH: f64 = 6.0;

/// How the bars are drawn: the window's zoom button.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum View {
    /// Both whole, on one time scale, the shorter slid to where most pairs
    /// place it.
    #[default]
    Aligned,
    /// As `Aligned`, narrowed to the matched footage: a short clip of a
    /// long video fills the width, and the long video's bar runs on past
    /// both edges. Still one scale, so a true match is still a vertical line.
    Zoomed,
}

/// Where each video's times fall across the chart, as a fraction of its
/// width: `start + ms * per_ms`, for A and for B.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Geometry {
    a: (f64, f64),
    b: (f64, f64),
}

impl Geometry {
    /// Both on one scale; with `offset` (B's time minus A's for matching
    /// footage) A is slid so matching footage lines up, otherwise both
    /// start at 0.
    fn new(len_a: f64, len_b: f64, offset: Option<f64>) -> Geometry {
        let a_at = offset.unwrap_or(0.0);
        let from = a_at.min(0.0);
        let span = ((a_at + len_a).max(len_b) - from).max(1.0);
        Geometry { a: ((a_at - from) / span, 1.0 / span), b: (-from / span, 1.0 / span) }
    }

    /// As `new`, the axis narrowed to the matched footage of both, with a
    /// little room either side; anything outside it is off the chart.
    fn zoomed(offset: Option<f64>, stretches_a: &[(u32, u32)], stretches_b: &[(u32, u32)]) -> Geometry {
        let a_at = offset.unwrap_or(0.0);
        let ends = stretches_a.iter().map(|&(s, e)| (a_at + s as f64, a_at + e as f64)).chain(stretches_b.iter().map(|&(s, e)| (s as f64, e as f64)));
        let (lo, hi) = ends.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (s, e)| (lo.min(s), hi.max(e)));
        if lo >= hi {
            return Geometry::new(0.0, 0.0, offset);
        }
        let margin = (hi - lo) * 0.03;
        let (from, span) = (lo - margin, hi - lo + 2.0 * margin);
        Geometry { a: ((a_at - from) / span, 1.0 / span), b: (-from / span, 1.0 / span) }
    }

    fn of(view: View, chart: &Chart) -> Geometry {
        match view {
            View::Aligned => Geometry::new(chart.len_a, chart.len_b, chart.offset),
            View::Zoomed => Geometry::zoomed(chart.offset, &chart.stretches_a, &chart.stretches_b),
        }
    }

    fn x(w: f64, (start, per_ms): (f64, f64), ms: f64) -> f64 {
        PAD + (start + ms * per_ms) * (w - 2.0 * PAD).max(1.0)
    }

    fn xa(&self, w: f64, ms: u32) -> f64 {
        Geometry::x(w, self.a, ms as f64)
    }

    fn xb(&self, w: f64, ms: u32) -> f64 {
        Geometry::x(w, self.b, ms as f64)
    }
}

/// The offset most pairs agree on, as B's time minus A's, and how many agree:
/// the largest cluster of offsets no more than `slack` apart, read at its
/// median. `None` unless at least half the pairs (and
/// two of them, when there are two or more) are in it, because a lone
/// cluster among many offsets is not a placement, it is one coincidence.
fn alignment(pairs: &[(u32, u32, u32)], slack: f64) -> Option<(f64, usize)> {
    let mut offsets: Vec<f64> = pairs.iter().map(|&(a, b, _)| b as f64 - a as f64).collect();
    offsets.sort_by(f64::total_cmp);
    let n = offsets.len();
    let (mut best, mut lo) = ((0, 0), 0);
    for hi in 0..n {
        while offsets[hi] - offsets[lo] > slack {
            lo += 1;
        }
        if hi + 1 - lo > best.1 - best.0 {
            best = (lo, hi + 1);
        }
    }
    let count = best.1 - best.0;
    let enough = n == 1 || (count >= 2 && 2 * count >= n);
    enough.then(|| (offsets[best.0 + count / 2], count))
}

/// How far apart two offsets of one true match can still be: each pair's
/// frames are the nearest samples, so they can be off by about a sample's
/// spacing in either video. Never under a second.
fn slack(a: &File, b: &File) -> f64 {
    let spacing = |f: &File| if f.samples > 0 { f.duration * 1000.0 / f.samples as f64 } else { 0.0 };
    (0.75 * spacing(a).max(spacing(b))).max(1000.0)
}

/// What the chart draws.
#[derive(Default)]
struct Chart {
    pairs: Vec<(u32, u32, u32)>,
    stretches_a: Vec<(u32, u32)>,
    stretches_b: Vec<(u32, u32)>,
    len_a: f64,
    len_b: f64,
    /// Where most pairs place A in B, if they agree (`alignment`).
    offset: Option<f64>,
    geo: Geometry,
    selected: Option<usize>,
}

/// What is under the pointer on the chart.
#[derive(Debug, PartialEq)]
enum Hit {
    /// One pair: a tick on A's bar (each pair is one frame of A) or a line.
    Pair(usize),
    /// A tick on B's bar, which every one of these pairs shares, in order.
    Shared(Vec<usize>),
}

impl Chart {
    fn hit(&self, w: f64, x: f64, y: f64) -> Option<Hit> {
        let nearest = |at: &dyn Fn(usize) -> f64| {
            (0..self.pairs.len()).map(|i| (i, (at(i) - x).abs())).filter(|&(_, d)| d <= REACH).min_by(|p, q| p.1.total_cmp(&q.1)).map(|(i, _)| i)
        };
        if y < TICK + BAR + 2.0 {
            return nearest(&|i| self.geo.xa(w, self.pairs[i].0)).map(Hit::Pair);
        }
        if y > TICK + BAR + BAND - 2.0 {
            let i = nearest(&|i| self.geo.xb(w, self.pairs[i].1))?;
            let at = self.pairs[i].1;
            return Some(Hit::Shared((0..self.pairs.len()).filter(|&k| self.pairs[k].1 == at).collect()));
        }
        // The band: the line nearest the pointer, measured square to it.
        let (y0, y1) = (TICK + BAR, TICK + BAR + BAND);
        (0..self.pairs.len())
            .map(|i| {
                let (x0, x1) = (self.geo.xa(w, self.pairs[i].0), self.geo.xb(w, self.pairs[i].1));
                (i, distance_to_segment((x, y), (x0, y0), (x1, y1)))
            })
            .filter(|&(_, d)| d <= REACH)
            .min_by(|p, q| p.1.total_cmp(&q.1))
            .map(|(i, _)| Hit::Pair(i))
    }

    /// The pairs that share the selected pair's frame of B, it included.
    fn sharing(&self) -> Vec<usize> {
        let Some(at) = self.selected.and_then(|i| self.pairs.get(i)).map(|p| p.1) else { return Vec::new() };
        (0..self.pairs.len()).filter(|&k| self.pairs[k].1 == at).collect()
    }

    /// Which pair a click on `hit` selects: a shared tick clicked again steps
    /// to the next pair on it, round to the first.
    fn choose(&self, hit: Hit) -> Option<usize> {
        match hit {
            Hit::Pair(i) => Some(i),
            Hit::Shared(on) => {
                let next = self.selected.and_then(|s| on.iter().position(|&k| k == s)).map_or(0, |p| (p + 1) % on.len());
                on.get(next).copied()
            }
        }
    }

    fn draw(&self, fg: &gdk::RGBA, cr: &gtk::cairo::Context, w: f64) {
        if self.pairs.is_empty() && self.stretches_a.is_empty() {
            return;
        }
        let fg = |alpha: f64| cr.set_source_rgba(fg.red() as f64, fg.green() as f64, fg.blue() as f64, alpha);
        let red = |alpha: f64| cr.set_source_rgba(0.93, 0.30, 0.22, alpha);
        let (ya, yb) = (TICK, TICK + BAR + BAND);
        // Zoomed, a bar runs past the edges: cut it at the chart's.
        cr.rectangle(PAD - 2.0, 0.0, w - 2.0 * PAD + 4.0, CHART_H);
        cr.clip();
        let bar = |map: (f64, f64), len: f64, stretches: &[(u32, u32)], y: f64| {
            let x = |ms: f64| Geometry::x(w, map, ms);
            fg(0.12);
            cr.rectangle(x(0.0), y, x(len) - x(0.0), BAR);
            let _ = cr.fill();
            // The matched footage, in the selection blue most themes use.
            cr.set_source_rgba(0.21, 0.52, 0.89, 0.85);
            for &(s, e) in stretches {
                cr.rectangle(x(s as f64), y, (x(e as f64) - x(s as f64)).max(1.0), BAR);
            }
            let _ = cr.fill();
        };
        bar(self.geo.a, self.len_a, &self.stretches_a, ya);
        bar(self.geo.b, self.len_b, &self.stretches_b, yb);

        // The lines, each drawn once however many pairs share it, faint so
        // that where many cross reads as a darker bundle.
        let line = |i: usize| {
            let (a, b, _) = self.pairs[i];
            cr.move_to(self.geo.xa(w, a), ya + BAR);
            cr.line_to(self.geo.xb(w, b), yb);
        };
        let mut drawn = std::collections::HashSet::new();
        cr.set_line_width(1.0);
        fg(0.3);
        for (i, &(a, b, _)) in self.pairs.iter().enumerate() {
            if drawn.insert((a, b)) {
                line(i);
            }
        }
        let _ = cr.stroke();
        fg(0.75);
        for &(a, b, _) in &self.pairs {
            cr.rectangle(self.geo.xa(w, a) - 0.5, 0.0, 1.0, TICK - 1.0);
            cr.rectangle(self.geo.xb(w, b) - 0.5, yb + BAR + 1.0, 1.0, TICK - 1.0);
        }
        let _ = cr.fill();

        let Some(sel) = self.selected.filter(|&i| i < self.pairs.len()) else { return };
        // The others on the selected pair's frame of B, half way.
        cr.set_line_width(1.5);
        red(0.45);
        for k in self.sharing().into_iter().filter(|&k| k != sel) {
            line(k);
        }
        let _ = cr.stroke();
        cr.set_line_width(2.5);
        red(1.0);
        line(sel);
        let _ = cr.stroke();
        let (a, b, _) = self.pairs[sel];
        cr.rectangle(self.geo.xa(w, a) - 1.5, 0.0, 3.0, TICK + BAR);
        cr.rectangle(self.geo.xb(w, b) - 1.5, yb, 3.0, BAR + TICK);
        let _ = cr.fill();
    }
}

fn distance_to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    ((p.0 - a.0 - t * dx).powi(2) + (p.1 - a.1 - t * dy).powi(2)).sqrt()
}

struct Matched {
    window: gtk::Window,
    thumbs: Rc<Thumbs>,
    subject: Subject,
    /// Which of `subject.against` is shown.
    current: Cell<usize>,
    /// Whether the window is on one pair full size.
    full: Cell<bool>,
    view: Cell<View>,
    stack: gtk::Stack,
    pick: gtk::DropDown,
    name_b: gtk::Label,
    facts_b: gtk::Label,
    chart: Rc<RefCell<Chart>>,
    area: gtk::DrawingArea,
    note: gtk::Label,
    summary: gtk::Label,
    pairs: gtk::Box,
    scroll: gtk::ScrolledWindow,
    full_title: gtk::Label,
    full_a: gtk::Picture,
    full_b: gtk::Picture,
    full_a_at: gtk::Label,
    full_b_at: gtk::Label,
    rows: RefCell<Vec<gtk::Box>>,
}

thread_local! {
    /// The *Bars* choice last made, for the next window this run.
    static LAST_VIEW: Cell<View> = const { Cell::new(View::Aligned) };
}

/// Open the window on `subject`, over `app`'s.
pub fn open(app: &Rc<App>, subject: Subject) {
    if subject.against.is_empty() {
        return;
    }
    let window = gtk::Window::builder()
        .title(format!("Matched frames: {}", name(&subject.file.path)))
        .transient_for(&app.window)
        .default_width(1060)
        .default_height(880)
        .build();

    // ---- the list: both videos, the chart, and the frame pairs
    let name_a = heading(&name(&subject.file.path));
    let facts_a = dim(&crate::results::facts(&subject.file));
    let name_b = heading("");
    let facts_b = dim("");
    let chart = Rc::new(RefCell::new(Chart::default()));
    let area = gtk::DrawingArea::builder().content_height(CHART_H as i32).hexpand(true).build();
    area.set_draw_func({
        let chart = chart.clone();
        move |area, cr, w, _| chart.borrow().draw(&area.color(), cr, w as f64)
    });
    let pick = gtk::DropDown::from_strings(&subject.against.iter().map(|(how, _, f)| format!("{how}: {}", name(&f.path))).collect::<Vec<_>>().iter().map(String::as_str).collect::<Vec<_>>());
    pick.set_tooltip_text(Some("Which video to compare it with: the one its row says it matched most, or the video its group was formed around"));
    pick.set_visible(subject.against.len() > 1);
    pick.set_halign(gtk::Align::Start);
    let zoom = gtk::ToggleButton::new();
    let dress_zoom = |b: &gtk::ToggleButton| {
        let (icon, tip) = if b.is_active() {
            ("zoom-out-symbolic", "Show both videos whole (Z)")
        } else {
            ("zoom-in-symbolic", "Zoom to the matched footage, on the same scale (Z)")
        };
        b.set_icon_name(icon);
        b.set_tooltip_text(Some(tip));
        b.update_property(&[gtk::accessible::Property::Label(tip)]);
    };
    zoom.set_active(LAST_VIEW.with(Cell::get) == View::Zoomed);
    dress_zoom(&zoom);
    let note = gtk::Label::builder().xalign(0.0).wrap(true).build();
    let summary = gtk::Label::builder().xalign(0.0).wrap(true).css_classes(["dim-label"]).build();
    let pairs = gtk::Box::new(gtk::Orientation::Vertical, 4);
    pairs.add_css_class("pairs");
    pairs.set_accessible_role(gtk::AccessibleRole::List);
    pairs.set_margin_top(6);
    pairs.set_margin_bottom(6);
    let scroll = gtk::ScrolledWindow::builder().child(&pairs).hscrollbar_policy(gtk::PolicyType::Never).vexpand(true).build();

    let tag = |text: &str| gtk::Label::builder().label(text).css_classes(["caption-heading", "dim-label"]).xalign(0.0).build();
    let head = gtk::Box::new(gtk::Orientation::Vertical, 2);
    let top = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    top.append(&pick);
    top.append(&gtk::Box::builder().hexpand(true).build());
    top.append(&zoom);
    top.set_margin_bottom(8);
    head.append(&top);
    for w in [tag("THIS VIDEO").upcast::<gtk::Widget>(), name_a.upcast(), facts_a.upcast(), area.clone().upcast(), tag("MATCHED WITH").upcast(), name_b.clone().upcast(), facts_b.clone().upcast()] {
        head.append(&w);
    }
    area.set_margin_top(4);
    area.set_margin_bottom(4);
    note.set_margin_top(8);
    summary.set_margin_top(4);
    summary.set_margin_bottom(10);
    head.append(&note);
    let list = gtk::Box::new(gtk::Orientation::Vertical, 8);
    list.append(&head);
    list.append(&scroll);
    // What the comparison found overall, under the pairs: the note on the
    // selected pair stays right above them.
    list.append(&summary);
    list.set_margin_start(16);
    list.set_margin_end(16);
    list.set_margin_top(12);

    // ---- one pair, full size
    let back = gtk::Button::with_mnemonic("_Back to the frames");
    let prev = gtk::Button::from_icon_name("go-previous-symbolic");
    prev.set_tooltip_text(Some("Previous pair (Left)"));
    let next = gtk::Button::from_icon_name("go-next-symbolic");
    next.set_tooltip_text(Some("Next pair (Right)"));
    let full_title = gtk::Label::builder().xalign(0.0).hexpand(true).css_classes(["heading"]).build();
    let bar = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    bar.append(&back);
    bar.append(&full_title);
    bar.append(&prev);
    bar.append(&next);
    let picture = || gtk::Picture::builder().can_shrink(true).content_fit(gtk::ContentFit::Contain).hexpand(true).vexpand(true).build();
    let (full_a, full_b) = (picture(), picture());
    let (full_a_at, full_b_at) = (dim(""), dim(""));
    let half = |p: &gtk::Picture, l: &gtk::Label| {
        let b = gtk::Box::new(gtk::Orientation::Vertical, 4);
        b.append(p);
        b.append(l);
        b
    };
    let both = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(8).homogeneous(true).vexpand(true).build();
    both.append(&half(&full_a, &full_a_at));
    both.append(&half(&full_b, &full_b_at));
    let full = gtk::Box::new(gtk::Orientation::Vertical, 8);
    full.append(&bar);
    full.append(&both);
    full.set_margin_start(12);
    full.set_margin_end(12);
    full.set_margin_top(10);
    full.set_margin_bottom(10);

    let stack = gtk::Stack::new();
    stack.add_named(&list, Some("list"));
    stack.add_named(&full, Some("full"));
    window.set_child(Some(&stack));

    let me = Rc::new(Matched {
        window: window.clone(),
        thumbs: Thumbs::with_workers(2, KEEP_BYTES / 2),
        subject,
        current: Cell::new(0),
        full: Cell::new(false),
        view: Cell::new(LAST_VIEW.with(Cell::get)),
        stack,
        pick: pick.clone(),
        name_b,
        facts_b,
        chart,
        area: area.clone(),
        note,
        summary,
        pairs,
        scroll,
        full_title,
        full_a,
        full_b,
        full_a_at,
        full_b_at,
        rows: RefCell::new(Vec::new()),
    });

    let click = gtk::GestureClick::new();
    click.connect_pressed({
        let me = Rc::downgrade(&me);
        move |_, _, x, y| {
            let Some(me) = me.upgrade() else { return };
            let chosen = {
                let c = me.chart.borrow();
                c.hit(me.area.width() as f64, x, y).and_then(|h| c.choose(h))
            };
            if let Some(i) = chosen {
                me.select(i);
                me.reveal(i, false);
            }
        }
    });
    area.add_controller(click);
    zoom.connect_toggled({
        let me = Rc::downgrade(&me);
        move |b| {
            dress_zoom(b);
            if let Some(me) = me.upgrade() {
                let view = if b.is_active() { View::Zoomed } else { View::Aligned };
                LAST_VIEW.with(|v| v.set(view));
                me.view.set(view);
                let mut c = me.chart.borrow_mut();
                c.geo = Geometry::of(view, &c);
                drop(c);
                me.area.queue_draw();
            }
        }
    });
    pick.connect_selected_notify({
        let me = Rc::downgrade(&me);
        move |d| {
            if let Some(me) = me.upgrade() {
                me.show_against(d.selected() as usize);
            }
        }
    });
    back.connect_clicked({
        let me = Rc::downgrade(&me);
        move |_| {
            if let Some(me) = me.upgrade() {
                me.to_list();
            }
        }
    });
    for (b, by) in [(&prev, -1i64), (&next, 1)] {
        let me = Rc::downgrade(&me);
        b.connect_clicked(move |_| {
            if let Some(me) = me.upgrade() {
                me.step(by);
            }
        });
    }
    // On the list: Up and Down move through the pairs, Enter or Space shows
    // one full size, Escape closes. Full size: Left and Right step, Escape
    // goes back. Left alone while the video picker has the keyboard.
    let keys = gtk::EventControllerKey::new();
    keys.set_propagation_phase(gtk::PropagationPhase::Capture);
    keys.connect_key_pressed({
        let me = Rc::downgrade(&me);
        let zoom_key = zoom.clone();
        move |_, key, _, mods| {
            use gdk::Key as K;
            let Some(me) = me.upgrade() else { return glib::Propagation::Proceed };
            let plain = mods.difference(gdk::ModifierType::SHIFT_MASK).is_empty();
            // The picker's own list is open: the keys are its. On the closed
            // picker only Enter and Space are (they open it).
            let focus = gtk::prelude::GtkWindowExt::focus(&me.window);
            let in_list = focus.as_ref().is_some_and(|f| f.is_ancestor(&me.pick));
            let on_pick = in_list || focus.is_some_and(|f| f == *me.pick.upcast_ref::<gtk::Widget>());
            let full = me.full.get();
            match key {
                K::Escape if full => me.to_list(),
                K::Escape => me.window.close(),
                K::z | K::Z if plain && !full && !in_list => zoom_key.set_active(!zoom_key.is_active()),
                K::Left | K::KP_Left if plain && full => me.step(-1),
                K::Right | K::KP_Right if plain && full => me.step(1),
                K::Up | K::KP_Up if plain && !full && !in_list => me.step(-1),
                K::Down | K::KP_Down if plain && !full && !in_list => me.step(1),
                K::Return | K::KP_Enter | K::ISO_Enter | K::space if plain && !full && !on_pick => {
                    if let Some(i) = me.chart.borrow().selected {
                        me.show_full(i);
                    }
                }
                _ => return glib::Propagation::Proceed,
            }
            glib::Propagation::Stop
        }
    });
    window.add_controller(keys);
    // The window keeps itself: its own closures hold it only weakly.
    window.connect_close_request({
        let me = RefCell::new(Some(me.clone()));
        move |_| {
            if let Some(me) = me.borrow_mut().take() {
                me.thumbs.clear_queue();
            }
            glib::Propagation::Proceed
        }
    });
    me.show_against(0);
    window.present();
}

impl Matched {
    fn scale(&self) -> u32 {
        self.window.scale_factor().max(1) as u32
    }

    fn against(&self) -> &(&'static str, Comparison, File) {
        &self.subject.against[self.current.get().min(self.subject.against.len() - 1)]
    }

    /// The comparison with `subject.against[k]`, on the list.
    fn show_against(self: &Rc<Self>, k: usize) {
        self.current.set(k.min(self.subject.against.len() - 1));
        self.thumbs.clear_queue();
        let (_, c, other) = self.against();
        let a = &self.subject.file;
        self.name_b.set_text(&name(&other.path));
        self.facts_b.set_text(&crate::results::facts(other));
        if !Path::new(&other.path).exists() {
            self.facts_b.set_text(&format!("{} · gone since the scan", crate::results::facts(other)));
        }
        let (len_a, len_b) = (runtime_ms(a, &c.stretches, c.pairs.iter().map(|p| p.0)), runtime_ms(other, &c.other_stretches, c.pairs.iter().map(|p| p.1)));
        let aligned = alignment(&c.pairs, slack(a, other));
        *self.chart.borrow_mut() = Chart {
            pairs: c.pairs.clone(),
            stretches_a: c.stretches.clone(),
            stretches_b: c.other_stretches.clone(),
            len_a,
            len_b,
            offset: aligned.map(|a| a.0),
            geo: Geometry::default(),
            selected: None,
        };
        {
            let mut chart = self.chart.borrow_mut();
            chart.geo = Geometry::of(self.view.get(), &chart);
        }
        self.area.queue_draw();

        let footage = |st: &[(u32, u32)]| st.iter().map(|&(s, e)| (e - s) as f64 / 1000.0).sum::<f64>();
        let shown = c.pairs.len();
        let mut text = format!(
            "{} of this video's sampled frames matched, {} of footage here and {} in the other. ",
            c.matched_frames,
            utils::format_shared(Some(footage(&c.stretches))),
            utils::format_shared(Some(footage(&c.other_stretches))),
        );
        if shown < c.matched_frames {
            text.push_str(&format!("{shown} of the pairs are shown, spread evenly. "));
        }
        text.push_str(&placement(aligned, shown));
        text.push_str(&format!(
            " Each pair says how far apart the two frames' 64-bit fingerprints were: 0 is identical, and the scan's tolerance was {}.",
            c.tolerance
        ));
        self.summary.set_text(&text);
        self.note.set_text("Click a pair, or a tick or line on the chart, to find it in both. Double click a pair to see it full size.");

        // The pairs, top to bottom, asked for in that order.
        while let Some(child) = self.pairs.first_child() {
            self.pairs.remove(&child);
        }
        let mut rows = Vec::with_capacity(shown);
        let size = Size::within(FRAME_W as u32 * self.scale(), FRAME_H as u32 * self.scale());
        for (i, &(ta, tb, d)) in c.pairs.iter().enumerate() {
            let frame = |path: &str, ms: u32| {
                let s = crate::still::Still::new(FRAME_W, FRAME_H, false);
                s.add_css_class("frame");
                self.thumbs.request(&PathBuf::from(path), size.at(ms), Shown::Still(s.clone()));
                s
            };
            let column = |path: &str, ms: u32| {
                let b = gtk::Box::new(gtk::Orientation::Vertical, 3);
                b.append(&frame(path, ms));
                b.append(&dim(&clock(ms)));
                b
            };
            let row = gtk::Box::builder()
                .orientation(gtk::Orientation::Horizontal)
                .spacing(12)
                .focusable(true)
                .css_classes(["pair"])
                .accessible_role(gtk::AccessibleRole::ListItem)
                .tooltip_text("Click to find it on the chart; double click to see both frames full size")
                .build();
            row.append(&column(&a.path, ta));
            row.append(&column(&other.path, tb));
            row.append(&gtk::Label::builder().label(bits(d)).valign(gtk::Align::Center).css_classes(["dim-label"]).build());
            row.update_property(&[gtk::accessible::Property::Label(&format!("Pair {} of {shown}: {} and {}, {}", i + 1, clock(ta), clock(tb), bits(d)))]);
            let click = gtk::GestureClick::new();
            let me = Rc::downgrade(self);
            click.connect_pressed(move |_, n, _, _| {
                if let Some(me) = me.upgrade() {
                    me.select(i);
                    me.reveal(i, true);
                    if n == 2 {
                        me.show_full(i);
                    }
                }
            });
            row.add_controller(click);
            self.pairs.append(&row);
            rows.push(row);
        }
        *self.rows.borrow_mut() = rows;
        self.scroll.vadjustment().set_value(0.0);
        self.to_list();
    }

    /// Pair `i` selected: its row, its ticks and line on the chart, and what
    /// the note says of it.
    fn select(&self, i: usize) {
        let rows = self.rows.borrow();
        let was = self.chart.borrow().selected;
        if let Some(r) = was.and_then(|w| rows.get(w)) {
            r.remove_css_class("selected");
        }
        let Some(row) = rows.get(i) else { return };
        row.add_css_class("selected");
        self.chart.borrow_mut().selected = Some(i);
        self.area.queue_draw();
        let chart = self.chart.borrow();
        let (ta, tb, d) = chart.pairs[i];
        let mut text = format!("Pair {} of {}: {} here, {} in the other, {}.", i + 1, chart.pairs.len(), clock(ta), clock(tb), bits(d));
        let sharing = chart.sharing();
        if sharing.len() > 1 {
            let at = sharing.iter().position(|&k| k == i).unwrap_or(0) + 1;
            text.push_str(&format!(
                " {} pairs matched that frame of the other video; this is {at} of them. Click its tick again for the next.",
                sharing.len()
            ));
        }
        self.note.set_text(&text);
    }

    /// Pair `i`'s row in view, and the keyboard on it unless a click in the
    /// row put it there already.
    fn reveal(&self, i: usize, clicked: bool) {
        let Some(row) = self.rows.borrow().get(i).cloned() else { return };
        if !clicked {
            row.grab_focus();
        }
        if let Some(b) = row.compute_bounds(&self.pairs) {
            let adj = self.scroll.vadjustment();
            let (top, bottom) = (b.y() as f64, (b.y() + b.height()) as f64);
            if top < adj.value() {
                adj.set_value(top);
            } else if bottom > adj.value() + adj.page_size() {
                adj.set_value(bottom - adj.page_size());
            }
        }
    }

    /// Pair `i`'s two frames as large as the window allows.
    fn show_full(&self, i: usize) {
        let (_, c, other) = self.against();
        let Some(&(ta, tb, d)) = c.pairs.get(i) else { return };
        self.select(i);
        self.full.set(true);
        self.full_title.set_text(&format!("Pair {} of {} · {}", i + 1, c.pairs.len(), bits(d)));
        self.full_a_at.set_text(&format!("{} at {}", name(&self.subject.file.path), clock(ta)));
        self.full_b_at.set_text(&format!("{} at {}", name(&other.path), clock(tb)));
        self.full_a.set_paintable(gdk::Paintable::NONE);
        self.full_b.set_paintable(gdk::Paintable::NONE);
        let size = Size::within(FULL_W, FULL_H);
        self.thumbs.request(&PathBuf::from(&self.subject.file.path), size.at(ta), Shown::Picture(self.full_a.clone()));
        self.thumbs.request(&PathBuf::from(&other.path), size.at(tb), Shown::Picture(self.full_b.clone()));
        self.stack.set_visible_child_name("full");
    }

    /// The next or previous pair: full size if that is where the window is.
    fn step(&self, by: i64) {
        let n = self.against().1.pairs.len() as i64;
        if n == 0 {
            return;
        }
        let next = match self.chart.borrow().selected {
            Some(i) => (i as i64 + by).clamp(0, n - 1) as usize,
            None if by > 0 => 0,
            None => (n - 1) as usize,
        };
        if self.full.get() {
            self.show_full(next);
        } else {
            self.select(next);
            self.reveal(next, false);
        }
    }

    /// Back to the pairs, with the keyboard on the one last looked at.
    fn to_list(&self) {
        self.full.set(false);
        self.stack.set_visible_child_name("list");
        if let Some(i) = self.chart.borrow().selected {
            // After the page is shown, so the row has a place to scroll to.
            let row = self.rows.borrow().get(i).cloned();
            if let Some(row) = row {
                row.grab_focus();
            }
        }
        let selected = self.chart.borrow().selected;
        if let Some(i) = selected {
            self.reveal(i, true);
        }
    }
}

/// What the chart's placement says, for the summary.
fn placement(aligned: Option<(f64, usize)>, n: usize) -> String {
    match aligned {
        Some((_, _)) if n == 1 => "With one pair there is nothing to check its placement against.".into(),
        Some((o, k)) => {
            let agree = if k == n { format!("All {n} pairs") } else { format!("{k} of the {n} pairs") };
            let at = clock(o.abs().round() as u32);
            if o.abs() < 500.0 {
                format!("{agree} agree that the two start together.")
            } else if o > 0.0 {
                format!("{agree} agree that this video starts {at} into the other.")
            } else {
                format!("{agree} agree that the other video starts {at} into this one.")
            }
        }
        None => "The pairs do not agree on where one video sits in the other: the frames matched at different moments, which a re-edit gives, and so does footage that repeats or barely moves.".into(),
    }
}

/// The bar's length: the runtime the scan measured, or as far as anything on
/// the bar reaches when that is further (a runtime the container under-stated,
/// or never stated).
fn runtime_ms(f: &File, stretches: &[(u32, u32)], ticks: impl Iterator<Item = u32>) -> f64 {
    let reach = stretches.iter().map(|s| s.1).chain(ticks).max().unwrap_or(0) as f64;
    (f.duration * 1000.0).max(reach).max(1.0)
}

fn bits(d: u32) -> String {
    format!("{d} bit{} apart", if d == 1 { "" } else { "s" })
}

/// A time in a video to the tenth of a second: `1:23.4`, or `1:02:03.4`.
fn clock(ms: u32) -> String {
    let tenths = ms / 100;
    let (h, m, s, t) = (tenths / 36_000, tenths / 600 % 60, tenths / 10 % 60, tenths % 10);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}.{t}")
    } else {
        format!("{m}:{s:02}.{t}")
    }
}

fn name(path: &str) -> String {
    Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_string())
}

fn heading(text: &str) -> gtk::Label {
    gtk::Label::builder().label(text).xalign(0.0).ellipsize(gtk::pango::EllipsizeMode::Middle).css_classes(["heading"]).build()
}

fn dim(text: &str) -> gtk::Label {
    gtk::Label::builder().label(text).xalign(0.0).ellipsize(gtk::pango::EllipsizeMode::End).css_classes(["dim-label", "caption"]).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = 1004.0;

    fn chart(pairs: &[(u32, u32, u32)], len_a: f64, len_b: f64, slack: f64) -> Chart {
        let geo = Geometry::new(len_a, len_b, alignment(pairs, slack).map(|a| a.0));
        Chart { pairs: pairs.to_vec(), len_a, len_b, geo, ..Default::default() }
    }

    /// A clip whose pairs all agree is slid to where it was cut from: every
    /// line runs straight down, and the longer video sets the scale.
    #[test]
    fn a_clip_is_drawn_under_the_footage_it_came_from() {
        let pairs: Vec<_> = (0..10).map(|k| (k * 1000, 42_000 + k * 1000 + (k % 3) * 120, 2)).collect();
        let c = chart(&pairs, 10_000.0, 300_000.0, 1000.0);
        let (o, n) = alignment(&pairs, 1000.0).unwrap();
        assert_eq!(n, 10);
        assert!((o - 42_120.0).abs() <= 120.0, "{o}");
        for &(a, b, _) in &pairs {
            assert!((c.geo.xa(W, a) - c.geo.xb(W, b)).abs() < 1.0, "{a} -> {b} is not straight down");
        }
        let near = |x: f64, want: f64| assert!((x - want).abs() < 1e-6, "{x} is not {want}");
        near(c.geo.xb(W, 0), PAD);
        near(c.geo.xb(W, 300_000), W - PAD);
        // Starting before the other, it widens the axis to the left.
        let early = Geometry::new(10_000.0, 8_000.0, Some(-3_000.0));
        near(early.xa(W, 0), PAD);
        near(early.xb(W, 8_000), W - PAD);
        assert!((early.xa(W, 3_000) - early.xb(W, 0)).abs() < 1e-9);
    }

    /// Both views of one clip: aligned under its footage, and the same
    /// zoomed so the clip fills the chart.
    #[test]
    fn each_view_puts_the_bars_where_it_says() {
        let (clip, episode, at) = (120_000.0, 3_480_000.0, Some(1_000_000.0));
        let c = Chart {
            len_a: clip,
            len_b: episode,
            offset: at,
            stretches_a: vec![(0, 120_000)],
            stretches_b: vec![(1_000_000, 1_120_000)],
            ..Default::default()
        };
        let aligned = Geometry::of(View::Aligned, &c);
        let zoomed = Geometry::of(View::Zoomed, &c);
        assert!((aligned.xa(W, 0) - aligned.xb(W, 1_000_000)).abs() < 1e-9);
        // Zoomed, the clip nearly fills the chart and still lines up; the
        // episode runs off both edges.
        assert!((zoomed.xa(W, 0) - zoomed.xb(W, 1_000_000)).abs() < 1e-6);
        let clip_w = zoomed.xa(W, 120_000) - zoomed.xa(W, 0);
        assert!(clip_w > 0.9 * (W - 2.0 * PAD), "{clip_w}");
        assert!(zoomed.xb(W, 0) < 0.0 && zoomed.xb(W, 3_480_000) > W);
    }

    /// The gym clips: as many offsets one way as another, so nothing agrees
    /// and both bars start at 0.
    #[test]
    fn pairs_that_scatter_are_not_placed() {
        let pairs = [(0, 0, 4), (967, 0, 2), (1935, 1935, 2), (2902, 0, 2), (3870, 3870, 0), (5805, 2902, 2), (6773, 3870, 0), (7741, 0, 2), (8708, 967, 0), (9676, 6773, 0)];
        assert_eq!(alignment(&pairs, 1000.0), None);
        let g = Geometry::new(10_410.0, 6_806.0, None);
        assert_eq!(g.xa(W, 0), g.xb(W, 0));
        // One pair is placed by itself; two that disagree are not.
        assert!(alignment(&[(0, 5000, 0)], 1000.0).is_some());
        assert_eq!(alignment(&[(0, 5000, 0), (1000, 90_000, 0)], 1000.0), None);
    }

    /// A tick on the other video's bar can stand for several pairs; clicking
    /// it again steps through them in order and round to the first, and a
    /// tick on this video's bar, or a line, is always one pair.
    #[test]
    fn a_shared_tick_steps_through_its_pairs() {
        let mut c = chart(&[(0, 0, 4), (2000, 9000, 1), (4000, 0, 2), (6000, 0, 0)], 8000.0, 10_000.0, 1000.0);
        let yb = TICK + BAR + BAND + BAR / 2.0;
        let x0 = c.geo.xb(W, 0);
        let hit = c.hit(W, x0 + 2.0, yb).unwrap();
        assert_eq!(hit, Hit::Shared(vec![0, 2, 3]));
        let mut seen = Vec::new();
        for _ in 0..4 {
            c.selected = c.choose(c.hit(W, x0, yb).unwrap());
            seen.push(c.selected.unwrap());
        }
        assert_eq!(seen, [0, 2, 3, 0]);
        assert_eq!(c.sharing(), [0, 2, 3]);
        assert_eq!(c.hit(W, c.geo.xa(W, 4000), TICK + 1.0), Some(Hit::Pair(2)));
        // Half way down pair 1's line.
        let (x, y) = ((c.geo.xa(W, 2000) + c.geo.xb(W, 9000)) / 2.0, TICK + BAR + BAND / 2.0);
        assert_eq!(c.hit(W, x, y), Some(Hit::Pair(1)));
        assert_eq!(c.hit(W, W - PAD - 1.0, TICK + BAR + BAND / 2.0), None);
    }

    #[test]
    fn a_time_reads_to_the_tenth_of_a_second() {
        assert_eq!(clock(0), "0:00.0");
        assert_eq!(clock(83_450), "1:23.4");
        assert_eq!(clock(3_723_400), "1:02:03.4");
    }
}
