//! The second page: the groups a scan found, for choosing what to move to the
//! Trash.
//!
//! Nothing is marked to begin with. What a group says is that every video in
//! it was matched against the video at its head (its reference); which of them
//! to keep is left to the person looking.
//!
//! **A group is a list.** Each video is a row: a frame of it (`thumbs.rs`),
//! its name and folder, the figures the command line's listing prints for it
//! (length, resolution, frame rate, codec, size, bitrate, bits a frame,
//! samples), how much of it matched which other file, and what the scan
//! suggests doing with it. A video is played by the desktop's own player
//! (Enter, a double click, or *Open*): the page shows a thumbnail rather than
//! playing anything itself.
//!
//! **The scan's suggestion is shown and offered, never applied.** Each row
//! says KEEP, DELETE or REVIEW, exactly as `vid-fp`'s report would
//! (`model.rs` says how, and when a file changed since moves it). *Mark
//! suggested deletions* makes the marks exactly the files suggested for
//! deletion, in every group, unmarking any other; the person still moves them
//! to the Trash themselves. There is one rule, so there is no rule to choose.
//!
//! Two things about groups decide how marks work here:
//!
//! - **Groups overlap.** A file matched by two representatives is in both
//!   groups, so a mark is on the *file*: marked in one group, it shows as
//!   marked in every group it is in.
//! - **A group is not an all-pairs claim.** Its members matched the
//!   reference, not necessarily each other. So when the reference goes to the
//!   Trash, what is left of its group is still shown, and says so.
//!
//! **The tree view shows the same files by where they are** (`folders.rs`):
//! in place of the strip, the folders the scan was given and every folder
//! below them that holds a grouped video, and the page lists every video under
//! the folder selected, its subfolders' included, in path order. *Mark current
//! folder* marks all of them.
//!
//! Keyboard: Up and Down move between videos, Space or Delete marks the one
//! selected, Shift and an arrow marks the videos it passes as the last one
//! marked by hand was (Shift and a click, the range from it), Enter plays it,
//! the Menu key (or Shift+F10, or a right click) offers the rest, Ctrl+Page
//! Down / Ctrl+Page Up change group, and every button has an Alt mnemonic.

use crate::commands::{self, Page, MENU};
use crate::folders::{self, Tree};
use crate::labels as l;
use crate::last;
use crate::model::{Lapse, Model};
use crate::rows::Rows;
use crate::still::Still;
use crate::thumbs::{Key, Shown, Size, Thumbs, KEEP_BYTES};
use crate::App;
use gtk::prelude::*;
use gtk::{gdk, gio, glib};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;
use vid_fp::found::{Action, Found, Member};
use vid_fp::utils::{self, Priority};

/// A group's picture in the strip.
const STRIP_W: i32 = 96;
const STRIP_H: i32 = 54;
/// A video's picture in its row.
const ROW_W: i32 = 192;
const ROW_H: i32 = 108;
/// The tree's width.
const TREE_W: i32 = 280;

const CSS: &str = "
rows.files checkbutton.mark > check { min-width: 26px; min-height: 26px; border-radius: 5px; -gtk-icon-size: 20px; }
rows.files > box { padding: 6px 10px; border-bottom: 1px solid alpha(@theme_fg_color, 0.08); }
rows.files > box:hover { background-color: alpha(@theme_fg_color, 0.04); }
rows.files > box.selected { background-color: alpha(@theme_selected_bg_color, 0.3); }
box.pairs > box.pair { padding: 6px; border-radius: 8px; }
box.pairs > box.pair:hover { background-color: alpha(currentColor, 0.05); }
box.pairs > box.pair.selected { background-color: alpha(@theme_selected_bg_color, 0.3); }
rows.files > box.marked { background-color: alpha(@error_color, 0.15); }
rows.files > box.marked.selected { background-color: mix(alpha(@error_color, 0.25), alpha(@theme_selected_bg_color, 0.3), 0.5); }
.files still.frame { background-color: alpha(black, 0.85); border-radius: 3px; }
.files label.reference, .strip label.badge { background-color: alpha(black, 0.68); color: white; border-radius: 999px; padding: 0 7px; font-size: smaller; font-weight: bold; }
.files label.reference { background-color: alpha(@theme_fg_color, 0.15); color: @theme_fg_color; }
.strip label.badge.marked, .tree label.badge.marked { background-color: @error_color; }
.tree label.badge { border-radius: 999px; padding: 0 7px; font-size: smaller; font-weight: bold; color: white; }
.tree row { padding: 2px 4px; }
.strip row { padding: 4px 8px; }
.strip row still { border-radius: 4px; }
label.suggest { border-radius: 999px; padding: 2px 10px; font-weight: bold; font-size: smaller; }
label.suggest.keep { background-color: alpha(@success_color, 0.2); color: mix(@success_color, @theme_fg_color, 0.55); }
label.suggest.delete { background-color: alpha(@error_color, 0.18); color: mix(@error_color, @theme_fg_color, 0.55); }
label.suggest.review { background-color: alpha(@warning_color, 0.25); color: mix(@warning_color, @theme_fg_color, 0.5); }
";

/// What each suggestion means, over its pill.
const KEEP_TIP: &str = "Keep: nothing in its groups that it was matched against ranks above it. \
                        Videos are ranked by what Rank by names first, then by length, resolution, \
                        bits a frame and size; the last two only between videos of one codec.";
/// What `Rank by` offers, in `Priority::ALL`'s order.
const RANKS: &[&str] = &["Length", "Resolution", "Quality", "Size"];
const DELETE_TIP: &str = "Delete: it was matched against a video of its group that ranks above it.";
const REVIEW_TIP: &str = "Review: a copy that may be the one to keep, left for you to decide. A copy in another codec, \
                          whose bits cannot be compared; one whose length could not be measured; or one better than \
                          the kept copy on length or resolution.";

/// Which list is beside the videos, and so what the page shows: a group, or
/// every video under a folder. What differs between the two is asked of this,
/// or of the one `match` on it that answers each question (`show_view`,
/// `step`, `scope`, `side_list`), never of a flag in the code that uses them.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum View {
    Groups,
    Folders,
}

impl View {
    fn other(self) -> View {
        match self {
            View::Groups => View::Folders,
            View::Folders => View::Groups,
        }
    }

    /// Its list's place in the stack beside the videos.
    fn child(self) -> &'static str {
        match self {
            View::Groups => "groups",
            View::Folders => "tree",
        }
    }

    pub(crate) fn heading(self) -> &'static str {
        match self {
            View::Groups => l::GROUPS,
            View::Folders => l::FOLDERS_TREE,
        }
    }

    /// What one of its list's items is.
    fn noun(self) -> &'static str {
        match self {
            View::Groups => "group",
            View::Folders => "folder",
        }
    }
}

/// A row of the strip, told in turn which group it is.
struct StripRow {
    picture: Still,
    count: gtk::Label,
    marked: gtk::Label,
    overlay: gtk::Overlay,
    group: Cell<Option<usize>>,
    /// The marks its badge shows, so that a row whose count did not change is
    /// left alone.
    shown: Cell<Option<usize>>,
    request: RefCell<Option<(Key, u64)>>,
}

/// A row of the folder tree, told in turn which folder it is.
struct TreeRow {
    expander: gtk::TreeExpander,
    name: gtk::Label,
    count: gtk::Label,
    marked: gtk::Label,
    folder: Cell<Option<usize>>,
    shown: Cell<Option<usize>>,
}

/// A row of the videos shown, told in turn which of them it is.
struct FileRow {
    root: gtk::Box,
    check: gtk::CheckButton,
    picture: Still,
    name: gtk::Label,
    reference: gtk::Label,
    dir: gtk::Label,
    facts: gtk::Label,
    matched: gtk::Label,
    pill: gtk::Label,
    /// The video it shows, by its place in the list.
    item: Rc<Cell<Option<usize>>>,
    request: RefCell<Option<(Key, u64)>>,
    /// Set while the row is dressed, so that setting its check is not taken
    /// for a click on it.
    dressing: Rc<Cell<bool>>,
}

/// The strip of groups, the groups view's list.
///
/// Activated by Enter or a double click only, since activating a group takes
/// the keyboard to its videos.
struct Strip {
    list: gtk::ListView,
    selection: gtk::SingleSelection,
    store: gio::ListStore,
    rows: RefCell<HashMap<gtk::ListItem, StripRow>>,
    /// Each group's picture (`face`).
    faces: RefCell<Vec<PathBuf>>,
    /// Set while its groups are being replaced, so that the selection moving
    /// on the way does not show each group it passes.
    quiet: Cell<bool>,
}

impl Strip {
    fn new(factory: &gtk::SignalListItemFactory) -> Strip {
        let store = gio::ListStore::new::<glib::BoxedAnyObject>();
        let selection = gtk::SingleSelection::builder().model(&store).autoselect(true).can_unselect(false).build();
        let list = gtk::ListView::builder().model(&selection).factory(factory).single_click_activate(false).css_classes(["strip"]).build();
        Strip { list, selection, store, rows: RefCell::new(HashMap::new()), faces: RefCell::new(Vec::new()), quiet: Cell::new(false) }
    }

    fn fill(&self, faces: Vec<PathBuf>) {
        let items: Vec<glib::BoxedAnyObject> = (0..faces.len()).map(glib::BoxedAnyObject::new).collect();
        *self.faces.borrow_mut() = faces;
        self.quiet.set(true);
        self.store.splice(0, self.store.n_items(), &items);
        self.quiet.set(false);
    }

    /// Group `i` selected, without showing it.
    fn select(&self, i: usize) {
        self.quiet.set(true);
        self.selection.set_selected(i as u32);
        self.quiet.set(false);
    }

    fn scroll_to(&self, i: usize) {
        let _ = self.list.activate_action("list.scroll-to-item", Some(&(i as u32).to_variant()));
    }

    /// The group `by` from the one selected, selected and so shown; whether
    /// there was one.
    fn step(&self, by: i32) -> bool {
        let (cur, n) = (self.selection.selected(), self.store.n_items() as i64);
        if n == 0 || cur == gtk::INVALID_LIST_POSITION {
            return false;
        }
        let next = (cur as i64 + by as i64).clamp(0, n - 1) as usize;
        self.selection.set_selected(next as u32);
        self.scroll_to(next);
        true
    }
}

/// The folder tree, the folders view's list.
struct Folders {
    tree: Rc<RefCell<Tree>>,
    store: gio::ListStore,
    model: gtk::TreeListModel,
    sel: gtk::SingleSelection,
    list: gtk::ListView,
    rows: RefCell<HashMap<gtk::ListItem, TreeRow>>,
    /// The folder shown.
    current: Cell<Option<usize>>,
    /// Set while the tree is being replaced or opened.
    quiet: Cell<bool>,
}

impl Folders {
    fn new(factory: &gtk::SignalListItemFactory) -> Folders {
        let tree = Rc::new(RefCell::new(Tree { folders: Vec::new(), top: Vec::new() }));
        let store = gio::ListStore::new::<glib::BoxedAnyObject>();
        let model = gtk::TreeListModel::new(store.clone(), false, false, {
            let tree = tree.clone();
            move |o| {
                let i = *o.downcast_ref::<glib::BoxedAnyObject>()?.borrow::<usize>();
                let t = tree.borrow();
                let children = &t.folders.get(i)?.children;
                if children.is_empty() {
                    return None;
                }
                let store = gio::ListStore::new::<glib::BoxedAnyObject>();
                let items: Vec<glib::BoxedAnyObject> = children.iter().map(|&c| glib::BoxedAnyObject::new(c)).collect();
                store.splice(0, 0, &items);
                Some(store.upcast())
            }
        });
        let sel = gtk::SingleSelection::builder().model(&model).autoselect(false).can_unselect(false).build();
        let list = gtk::ListView::builder().model(&sel).factory(factory).single_click_activate(false).css_classes(["tree"]).build();
        Folders { tree, store, model, sel, list, rows: RefCell::new(HashMap::new()), current: Cell::new(None), quiet: Cell::new(false) }
    }

    /// `tree` in place of the one shown, its top folders open. The folder
    /// shown stays shown if it still holds a video.
    fn fill(&self, tree: Tree) {
        let before = self.current.get().and_then(|f| self.tree.borrow().folders.get(f).map(|f| f.path.clone()));
        let after = before.and_then(|p| tree.find(&p));
        let top: Vec<glib::BoxedAnyObject> = tree.top.iter().map(|&t| glib::BoxedAnyObject::new(t)).collect();
        *self.tree.borrow_mut() = tree;
        self.current.set(after);
        self.quiet.set(true);
        self.store.splice(0, self.store.n_items(), &top);
        for i in (0..top.len()).rev() {
            if let Some(r) = self.model.child_row(i as u32) {
                r.set_expanded(true);
            }
        }
        self.quiet.set(false);
    }

    fn at(&self, pos: u32) -> Option<usize> {
        let row = self.model.row(pos)?;
        let o = row.item().and_downcast::<glib::BoxedAnyObject>()?;
        let i = *o.borrow::<usize>();
        Some(i)
    }

    fn row_of(&self, f: usize) -> Option<u32> {
        (0..self.model.n_items()).find(|&p| self.at(p) == Some(f))
    }

    fn scroll_to(&self, pos: u32) {
        let _ = self.list.activate_action("list.scroll-to-item", Some(&pos.to_variant()));
    }

    /// The tree opened to folder `f`, and `f` selected, without showing it.
    fn open_to(&self, f: usize) {
        let chain = self.tree.borrow().ancestors(f);
        self.quiet.set(true);
        for a in chain {
            if let Some(r) = self.row_of(a).and_then(|p| self.model.row(p)) {
                r.set_expanded(true);
            }
        }
        let pos = self.row_of(f);
        if let Some(p) = pos {
            self.sel.set_selected(p);
        }
        self.quiet.set(false);
        if let Some(p) = pos {
            self.scroll_to(p);
        }
    }

    /// Open the selected folder, or close it; closing a closed one goes to
    /// the folder above.
    fn expand_selected(&self, open: bool) {
        let Some(row) = self.model.row(self.sel.selected()) else { return };
        if open {
            row.set_expanded(true);
        } else if row.is_expanded() {
            row.set_expanded(false);
        } else if let Some(parent) = row.parent() {
            let p = (0..self.model.n_items()).find(|&p| self.model.row(p).as_ref() == Some(&parent));
            if let Some(p) = p {
                self.sel.set_selected(p);
                self.scroll_to(p);
            }
        }
    }

    fn step(&self, by: i32) -> bool {
        let (cur, n) = (self.sel.selected(), self.model.n_items() as i64);
        if n == 0 || cur == gtk::INVALID_LIST_POSITION {
            return false;
        }
        let next = (cur as i64 + by as i64).clamp(0, n - 1) as u32;
        self.sel.set_selected(next);
        self.scroll_to(next);
        true
    }
}

/// The videos of the group or folder shown, as a list that lays out only
/// the rows on screen (`rows.rs`).
struct Files {
    area: Rows,
    /// The row widgets, as many as fit on screen and one more: video `i` is
    /// shown by row `i % pool.len()`, so a scroll of one row rebinds one.
    pool: RefCell<Vec<Rc<FileRow>>>,
    /// The height of a row at a width, measured once per width.
    row_height: Cell<(i32, i32)>,
    /// The video to give the keyboard once it has a row (`move_to`).
    want_focus: Cell<Option<usize>>,
}

/// A scan made ready for the page, off the main thread.
pub struct Prepared {
    model: Model,
    kept: Option<u64>,
    roots: Vec<PathBuf>,
    changed_after: Option<u64>,
}

impl Prepared {
    pub fn set_roots(&mut self, roots: Vec<PathBuf>) {
        self.roots = roots;
    }

    pub fn groups(&self) -> usize {
        self.model.groups.len()
    }

    /// The videos the scan fingerprinted and compared.
    pub fn compared(&self) -> usize {
        self.model.found.compared
    }

    /// A file changed after `ns` (nanoseconds since 1970) is not the file the
    /// scan judged (`last::Notes::changed_after`).
    pub fn set_changed_after(&mut self, ns: u64) {
        self.changed_after = Some(ns);
    }

    /// It is the scan kept on disk as `id` (`last::Notes::id`).
    pub fn set_kept(&mut self, id: u64) {
        self.kept = Some(id);
    }
}

pub fn prepare(found: Found) -> Prepared {
    Prepared { model: Model::new(found), kept: None, roots: Vec::new(), changed_after: None }
}

/// What the page knows beyond the model.
#[derive(Default)]
struct State {
    model: Model,
    changed_after: Option<u64>,
    current: usize,
    problems: bool,
    has_results: bool,
    kept: Option<u64>,
    roots: Vec<PathBuf>,
}

pub struct Results {
    pub root: gtk::Box,
    app: Rc<App>,
    thumbs: Rc<Thumbs>,
    /// The strip's pictures, with a decoder of their own, so that changing
    /// group neither drops them nor waits for them.
    strip_thumbs: Rc<Thumbs>,
    state: RefCell<State>,
    strip: Strip,
    folders: Folders,
    files: Files,
    view: Cell<View>,
    left: gtk::Stack,
    left_label: gtk::Label,
    menu: gio::Menu,
    prev: gtk::Button,
    next: gtk::Button,
    /// What each row shows: the file as a group has it, and that group.
    shown: RefCell<Vec<(Member, usize)>>,
    title: gtk::Label,
    note: gtk::Label,
    status: gtk::Label,
    trash: gtk::Button,
    mark_suggested: gtk::Button,
    /// Which ranking's suggestions are shown (`--priority`).
    rank: gtk::DropDown,
    group_buttons: Vec<gtk::Widget>,
    /// The video menu, for the Menu key and a right click.
    context: gtk::PopoverMenu,
    /// The selected video, which the menu acts on.
    selected: Cell<Option<usize>>,
    /// The video last marked or unmarked by hand, and which of the two: where
    /// a Shift+click's range starts, and what Shift and an arrow give the
    /// videos they pass. Forgotten when the videos shown change.
    anchor: Cell<Option<(usize, bool)>>,
    /// The menu's actions that need a selected video.
    video_actions: RefCell<Vec<gio::SimpleAction>>,
    /// Set while files are being moved to the Trash.
    trashing: Cell<bool>,
    /// Which scan is shown, counted up by `show`: a look at the files that
    /// comes back after another scan is shown is about other files.
    showing_scan: Cell<u64>,
    /// A look at the files is under way, for an action that waits on it.
    checking: Cell<bool>,
    /// A save of the marks is waiting to be written.
    marks_pending: Cell<bool>,
    settings_button: gtk::Button,
    to_settings: RefCell<Option<Box<dyn Fn()>>>,
}

impl Results {
    pub fn new(app: Rc<App>) -> Rc<Results> {
        let css = gtk::CssProvider::new();
        css.load_from_data(CSS);
        // At the user's own priority, added after theirs so it wins: a user
        // stylesheet that themes every widget would otherwise hide which
        // videos are marked.
        #[allow(deprecated)]
        gtk::style_context_add_provider_for_display(&WidgetExt::display(&app.window), &css, gtk::STYLE_PROVIDER_PRIORITY_USER);

        // ---- the strip of groups
        let factory = gtk::SignalListItemFactory::new();
        let strip = Strip::new(&factory);
        let list_scroll = gtk::ScrolledWindow::builder()
            .child(&strip.list)
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vexpand(true)
            .min_content_width(STRIP_W + 16)
            .width_request(STRIP_W + 16)
            .propagate_natural_width(false)
            .hexpand(false)
            .build();
        // ---- the folder tree, in the strip's place in the tree view
        let tree_factory = gtk::SignalListItemFactory::new();
        let folders = Folders::new(&tree_factory);
        let tree_scroll = gtk::ScrolledWindow::builder()
            .child(&folders.list)
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vexpand(true)
            .min_content_width(TREE_W)
            .width_request(TREE_W)
            .propagate_natural_width(false)
            .hexpand(false)
            .build();

        let left_label = gtk::Label::with_mnemonic(View::Groups.heading());
        left_label.set_mnemonic_widget(Some(&strip.list));
        left_label.set_xalign(0.0);
        left_label.set_margin_start(8);
        left_label.add_css_class("heading");
        let left_stack = gtk::Stack::builder().hhomogeneous(false).vexpand(true).build();
        left_stack.add_named(&list_scroll, Some(View::Groups.child()));
        left_stack.add_named(&tree_scroll, Some(View::Folders.child()));
        left_stack.set_visible_child_name(View::Groups.child());
        let left = gtk::Box::new(gtk::Orientation::Vertical, 6);
        left.set_margin_top(10);
        left.append(&left_label);
        left.append(&left_stack);

        // ---- the header: which group, and what applies to every group
        let prev = gtk::Button::from_icon_name("go-previous-symbolic");
        prev.set_tooltip_text(Some("Previous group (Ctrl+Page Up)"));
        let next = gtk::Button::from_icon_name("go-next-symbolic");
        next.set_tooltip_text(Some("Next group (Ctrl+Page Down)"));
        let title = gtk::Label::builder().xalign(0.0).ellipsize(gtk::pango::EllipsizeMode::End).css_classes(["heading"]).build();
        let rank = gtk::DropDown::from_strings(RANKS);
        rank.set_selected(crate::settings::load_ranking().index() as u32);
        rank.set_tooltip_text(Some(
            "What decides which copy is kept, as vid-fp's --priority does: the longest, the highest resolution, \
             the most bits a frame (quality), or the largest file. The others break ties in that order. \
             Every ranking was worked out by the scan, so changing it scans nothing again.",
        ));
        let rank_label = gtk::Label::with_mnemonic(l::RANK_BY);
        rank_label.add_css_class("dim-label");
        rank_label.set_mnemonic_widget(Some(&rank));
        let mark_suggested = gtk::Button::with_mnemonic(l::MARK_SUGGESTED);
        mark_suggested.set_tooltip_text(Some(
            "Mark exactly the videos suggested for deletion, in every group. Every other mark is removed, including ones made by hand.",
        ));
        mark_suggested.set_sensitive(false);

        // What applies to one video or one group: in a menu, so that the page
        // carries no row of buttons above the list. Each item is also Alt and
        // its letter from anywhere on the page, which the menu shows beside it.
        let menu = gio::Menu::new();
        fill_menu(&menu, View::Groups);
        let more = gtk::MenuButton::builder().icon_name("open-menu-symbolic").menu_model(&menu).tooltip_text("The selected video, the group or folder, and the view").build();

        let head = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        head.append(&prev);
        head.append(&next);
        head.append(&title);
        title.set_hexpand(true);
        title.set_margin_start(4);
        head.append(&rank_label);
        head.append(&rank);
        head.append(&mark_suggested);
        head.append(&more);
        let note = gtk::Label::builder().xalign(0.0).wrap(true).css_classes(["dim-label"]).visible(false).build();

        // ---- the videos
        let area = Rows::new();
        area.add_css_class("files");
        area.set_hexpand(true);
        area.set_vexpand(true);
        let bar = gtk::Scrollbar::new(gtk::Orientation::Vertical, Some(&area.adjustment()));
        let list_box = gtk::Box::new(gtk::Orientation::Horizontal, 0);
        list_box.append(&area);
        list_box.append(&bar);
        let scroll = gtk::Frame::builder().child(&list_box).vexpand(true).hexpand(true).build();
        let files = Files { area, pool: RefCell::new(Vec::new()), row_height: Cell::new((-1, 0)), want_focus: Cell::new(None) };
        let context = gtk::PopoverMenu::from_model(Some(&menu));
        context.set_has_arrow(false);
        context.set_halign(gtk::Align::Start);

        let right = gtk::Box::new(gtk::Orientation::Vertical, 8);
        right.set_margin_end(12);
        right.set_margin_top(10);
        right.set_margin_bottom(8);
        right.append(&head);
        right.append(&note);
        right.append(&scroll);
        // Not on the list itself: a ListView places its own children, and one
        // it did not make (this popover) broke the next splice of its rows.
        context.set_parent(&right);

        left.set_hexpand(false);
        right.set_hexpand(true);
        let panes = gtk::Box::builder().orientation(gtk::Orientation::Horizontal).spacing(12).vexpand(true).build();
        panes.append(&left);
        panes.append(&gtk::Separator::new(gtk::Orientation::Vertical));
        panes.append(&right);

        // ---- the bottom bar: the marks, and what to do with them
        let status = gtk::Label::builder().xalign(0.0).hexpand(true).ellipsize(gtk::pango::EllipsizeMode::End).build();
        let hint = gtk::Label::builder()
            .label("Space marks · Enter plays · F1 keyboard shortcuts")
            .css_classes(["dim-label", "caption"])
            .build();
        let log = gtk::Button::with_mnemonic(l::RESULTS_LOG);
        let to_settings = gtk::Button::with_mnemonic(l::TO_SETTINGS);
        let trash = gtk::Button::with_mnemonic(l::TRASH);
        trash.add_css_class("destructive-action");
        trash.set_sensitive(false);
        let bottom = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        bottom.set_margin_start(12);
        bottom.set_margin_end(12);
        bottom.set_margin_top(8);
        bottom.set_margin_bottom(10);
        bottom.append(&status);
        bottom.append(&hint);
        for b in [&log, &to_settings, &trash] {
            b.set_valign(gtk::Align::Center);
            bottom.append(b);
        }

        let root = gtk::Box::new(gtk::Orientation::Vertical, 0);
        root.append(&panes);
        root.append(&gtk::Separator::new(gtk::Orientation::Horizontal));
        root.append(&bottom);

        let group_buttons: Vec<gtk::Widget> = vec![prev.clone().upcast(), next.clone().upcast(), more.clone().upcast()];
        let me = Rc::new(Results {
            root,
            app: app.clone(),
            thumbs: Thumbs::with_workers(2, KEEP_BYTES),
            strip_thumbs: Thumbs::with_workers(1, KEEP_BYTES / 2),
            state: RefCell::new(State::default()),
            strip,
            folders,
            files,
            view: Cell::new(View::Groups),
            left: left_stack,
            left_label,
            menu,
            prev: prev.clone(),
            next: next.clone(),
            shown: RefCell::new(Vec::new()),
            title,
            note,
            status,
            trash,
            mark_suggested: mark_suggested.clone(),
            rank: rank.clone(),
            group_buttons,
            context,
            selected: Cell::new(None),
            anchor: Cell::new(None),
            video_actions: RefCell::new(Vec::new()),
            trashing: Cell::new(false),
            showing_scan: Cell::new(0),
            checking: Cell::new(false),
            marks_pending: Cell::new(false),
            settings_button: to_settings.clone(),
            to_settings: RefCell::new(None),
        });

        let weak = Rc::downgrade(&me);
        let with = move |f: fn(&Rc<Results>)| {
            let weak = weak.clone();
            move |_: &gtk::Button| {
                if let Some(me) = weak.upgrade() {
                    f(&me);
                }
            }
        };
        prev.connect_clicked(with(|me| me.step(-1)));
        next.connect_clicked(with(|me| me.step(1)));
        me.trash.connect_clicked(with(|me| me.confirm_trash()));
        mark_suggested.connect_clicked(with(|me| me.mark_suggested()));
        rank.connect_selected_notify({
            let weak = Rc::downgrade(&me);
            move |d| {
                if let Some(me) = weak.upgrade() {
                    me.set_priority(Priority::ALL[(d.selected() as usize).min(Priority::ALL.len() - 1)]);
                }
            }
        });
        to_settings.connect_clicked(with(|me| {
            if let Some(f) = me.to_settings.borrow().as_ref() {
                f();
            }
        }));
        log.connect_clicked({
            let app = app.clone();
            move |_| app.show_log()
        });

        // The menu's items, as actions its model names.
        let actions = gio::SimpleActionGroup::new();
        for &act in MENU.iter().flat_map(|s| s.iter()) {
            let a = gio::SimpleAction::new(act.action(), None);
            let weak = Rc::downgrade(&me);
            a.connect_activate(move |_, _| {
                if let Some(me) = weak.upgrade() {
                    me.run(act, false);
                }
            });
            actions.add_action(&a);
            if act.needs_video() {
                me.video_actions.borrow_mut().push(a);
            }
        }
        me.root.insert_action_group("results", Some(&actions));
        me.select(None);
        // Back to the video the menu was about, so the keyboard carries on
        // where it was.
        me.context.connect_closed({
            let weak = Rc::downgrade(&me);
            move |_| {
                if let Some(me) = weak.upgrade() {
                    if let Some(i) = me.selected.get() {
                        glib::idle_add_local_once(move || me.move_to(i as i64));
                    }
                }
            }
        });

        // ---- the strip's rows
        factory.connect_setup({
            let weak = Rc::downgrade(&me);
            move |_, item| {
                let (Some(me), Some(item)) = (weak.upgrade(), item.downcast_ref::<gtk::ListItem>()) else { return };
                let picture = Still::new(STRIP_W, STRIP_H, true);
                picture.set_overflow(gtk::Overflow::Hidden);
                let count = gtk::Label::builder().css_classes(["badge"]).halign(gtk::Align::End).valign(gtk::Align::End).margin_end(4).margin_bottom(4).build();
                let marked = gtk::Label::builder().css_classes(["badge", "marked"]).halign(gtk::Align::Start).valign(gtk::Align::End).margin_start(4).margin_bottom(4).visible(false).build();
                let overlay = gtk::Overlay::builder().child(&picture).build();
                overlay.add_overlay(&count);
                overlay.add_overlay(&marked);
                item.set_child(Some(&overlay));
                let row = StripRow { picture, count, marked, overlay, group: Cell::new(None), shown: Cell::new(None), request: RefCell::new(None) };
                me.strip.rows.borrow_mut().insert(item.clone(), row);
            }
        });
        factory.connect_bind({
            let weak = Rc::downgrade(&me);
            move |_, item| {
                let (Some(me), Some(item)) = (weak.upgrade(), item.downcast_ref::<gtk::ListItem>()) else { return };
                me.bind_row(item);
            }
        });
        factory.connect_unbind({
            let weak = Rc::downgrade(&me);
            move |_, item| {
                let (Some(me), Some(item)) = (weak.upgrade(), item.downcast_ref::<gtk::ListItem>()) else { return };
                me.unbind_row(item);
            }
        });
        factory.connect_teardown({
            let weak = Rc::downgrade(&me);
            move |_, item| {
                let (Some(me), Some(item)) = (weak.upgrade(), item.downcast_ref::<gtk::ListItem>()) else { return };
                me.strip.rows.borrow_mut().remove(item);
            }
        });
        me.strip.selection.connect_selected_notify({
            let weak = Rc::downgrade(&me);
            move |sel| {
                let Some(me) = weak.upgrade() else { return };
                if me.strip.quiet.get() || sel.selected() == gtk::INVALID_LIST_POSITION {
                    return;
                }
                me.show_group(sel.selected() as usize);
            }
        });

        // ---- the tree's rows
        tree_factory.connect_setup({
            let weak = Rc::downgrade(&me);
            move |_, item| {
                let (Some(me), Some(item)) = (weak.upgrade(), item.downcast_ref::<gtk::ListItem>()) else { return };
                let name = gtk::Label::builder().xalign(0.0).hexpand(true).ellipsize(gtk::pango::EllipsizeMode::Middle).build();
                let marked = gtk::Label::builder().css_classes(["badge", "marked"]).valign(gtk::Align::Center).visible(false).build();
                let count = gtk::Label::builder().css_classes(["dim-label", "caption"]).valign(gtk::Align::Center).build();
                let b = gtk::Box::new(gtk::Orientation::Horizontal, 6);
                b.append(&name);
                b.append(&marked);
                b.append(&count);
                let expander = gtk::TreeExpander::builder().child(&b).build();
                item.set_child(Some(&expander));
                let row = TreeRow { expander, name, count, marked, folder: Cell::new(None), shown: Cell::new(None) };
                me.folders.rows.borrow_mut().insert(item.clone(), row);
            }
        });
        tree_factory.connect_bind({
            let weak = Rc::downgrade(&me);
            move |_, item| {
                let (Some(me), Some(item)) = (weak.upgrade(), item.downcast_ref::<gtk::ListItem>()) else { return };
                me.bind_tree_row(item);
            }
        });
        tree_factory.connect_unbind({
            let weak = Rc::downgrade(&me);
            move |_, item| {
                let (Some(me), Some(item)) = (weak.upgrade(), item.downcast_ref::<gtk::ListItem>()) else { return };
                let rows = me.folders.rows.borrow();
                if let Some(r) = rows.get(item) {
                    r.folder.set(None);
                    r.expander.set_list_row(None);
                }
            }
        });
        tree_factory.connect_teardown({
            let weak = Rc::downgrade(&me);
            move |_, item| {
                let (Some(me), Some(item)) = (weak.upgrade(), item.downcast_ref::<gtk::ListItem>()) else { return };
                me.folders.rows.borrow_mut().remove(item);
            }
        });
        me.folders.sel.connect_selected_notify({
            let weak = Rc::downgrade(&me);
            move |sel| {
                let Some(me) = weak.upgrade() else { return };
                if me.folders.quiet.get() || me.view.get() != View::Folders {
                    return;
                }
                if let Some(f) = me.folders.at(sel.selected()) {
                    me.show_folder(f);
                }
            }
        });
        // Enter on a folder or a group takes the keyboard to its videos.
        for list in [&me.folders.list, &me.strip.list] {
            list.connect_activate({
                let weak = Rc::downgrade(&me);
                move |_, _| {
                    if let Some(me) = weak.upgrade() {
                        me.focus();
                    }
                }
            });
        }

        // ---- the videos' rows, placed on every allocation and scroll
        me.files.area.set_layout({
            let weak = Rc::downgrade(&me);
            move |_, w, h, offset| {
                if let Some(me) = weak.upgrade() {
                    me.lay_out(w, h, offset);
                }
            }
        });

        // The page's keys are the window's, taken before the widget with the
        // keyboard sees them, and only while the page is showing; which of
        // them the page takes from the widget with the keyboard is `key`'s.
        let keys = gtk::EventControllerKey::new();
        keys.set_propagation_phase(gtk::PropagationPhase::Capture);
        keys.connect_key_pressed({
            let weak = Rc::downgrade(&me);
            move |_, key, _, mods| {
                let Some(me) = weak.upgrade().filter(|me| me.showing()) else { return glib::Propagation::Proceed };
                match commands::RESULTS.find(key, mods) {
                    Some(act) if me.key(act, key, mods) => glib::Propagation::Stop,
                    _ => {
                        // Left and Right open and close the tree's folders.
                        if me.view.get() == View::Folders && mods.is_empty() && matches!(key, gdk::Key::Left | gdk::Key::Right) && me.in_side_list() {
                            me.folders.expand_selected(key == gdk::Key::Right);
                            return glib::Propagation::Stop;
                        }
                        glib::Propagation::Proceed
                    }
                }
            }
        });
        me.app.window.add_controller(keys);
        me
    }

    /// A row for the videos' list, with what it does to whichever video it
    /// shows.
    fn new_file_row(self: &Rc<Self>) -> Rc<FileRow> {
        use gtk::pango::EllipsizeMode as E;
        let item: Rc<Cell<Option<usize>>> = Rc::new(Cell::new(None));
        let dressing = Rc::new(Cell::new(false));
        let check = gtk::CheckButton::builder().css_classes(["mark"]).valign(gtk::Align::Center).tooltip_text("Mark or unmark for the Trash (Space)").focusable(false).build();
        let picture = Still::new(ROW_W, ROW_H, false);
        picture.add_css_class("frame");
        picture.set_valign(gtk::Align::Center);
        let name = gtk::Label::builder().xalign(0.0).ellipsize(E::Middle).css_classes(["heading"]).build();
        let reference = gtk::Label::builder()
            .label("Reference")
            .css_classes(["reference"])
            .tooltip_text("Every other video in this group was matched against this one")
            .valign(gtk::Align::Center)
            .visible(false)
            .build();
        let name_line = gtk::Box::new(gtk::Orientation::Horizontal, 8);
        name_line.append(&name);
        name_line.append(&reference);
        let dir = gtk::Label::builder().xalign(0.0).ellipsize(E::Start).css_classes(["dim-label", "caption"]).build();
        let facts = gtk::Label::builder().xalign(0.0).ellipsize(E::End).build();
        let matched = gtk::Label::builder().xalign(0.0).ellipsize(E::Middle).css_classes(["dim-label"]).build();
        let text = gtk::Box::new(gtk::Orientation::Vertical, 3);
        text.set_valign(gtk::Align::Center);
        text.set_hexpand(true);
        for w in [name_line.upcast_ref::<gtk::Widget>(), dir.upcast_ref(), facts.upcast_ref(), matched.upcast_ref()] {
            text.append(w);
        }
        let pill = gtk::Label::builder().css_classes(["suggest"]).valign(gtk::Align::Center).build();
        let compare = gtk::Button::builder()
            .label("Compare…")
            .valign(gtk::Align::Center)
            .focusable(false)
            .tooltip_text("Compare it with what it matched, frame by frame (C)")
            .build();
        compare.connect_clicked({
            let (weak, item) = (Rc::downgrade(self), item.clone());
            move |_| {
                if let (Some(me), Some(i)) = (weak.upgrade(), item.get()) {
                    me.select_quietly(i);
                    me.compare();
                }
            }
        });
        let root = gtk::Box::new(gtk::Orientation::Horizontal, 12);
        root.set_focusable(true);
        root.append(&check);
        root.append(&picture);
        root.append(&text);
        root.append(&compare);
        root.append(&pill);

        check.connect_toggled({
            let (weak, item, dressing) = (Rc::downgrade(self), item.clone(), dressing.clone());
            move |c| {
                if dressing.get() {
                    return;
                }
                if let (Some(me), Some(i)) = (weak.upgrade(), item.get()) {
                    me.select_quietly(i);
                    me.set_marked(i, c.is_active());
                }
            }
        });
        // Shift and a click marks the range from the anchor; a click on the
        // check marks or unmarks the video, and selects it, in one click; a
        // right click opens the menu there. All claimed before the list sees
        // them: the list's own row takes the first click on a row to select
        // it, so the check under the pointer used to need a second.
        let click = gtk::GestureClick::builder().button(0).propagation_phase(gtk::PropagationPhase::Capture).build();
        click.connect_pressed({
            let (weak, item) = (Rc::downgrade(self), item.clone());
            let (root, check, compare) = (root.downgrade(), check.downgrade(), compare.downgrade());
            move |g, n, x, y| {
                let (Some(me), Some(root), Some(check), Some(compare), Some(i)) = (weak.upgrade(), root.upgrade(), check.upgrade(), compare.upgrade(), item.get()) else { return };
                // The Compare button is a button: left to take its own click.
                let picked = root.pick(x, y, gtk::PickFlags::DEFAULT);
                if g.current_button() == gdk::BUTTON_PRIMARY && picked.as_ref().is_some_and(|w| w == compare.upcast_ref::<gtk::Widget>() || w.is_ancestor(&compare)) {
                    return;
                }
                let primary = g.current_button() == gdk::BUTTON_PRIMARY;
                let shift = g.current_event_state().contains(gdk::ModifierType::SHIFT_MASK);
                if shift && n == 1 && primary {
                    g.set_state(gtk::EventSequenceState::Claimed);
                    me.mark_range(i);
                    return;
                }
                let on_check = picked.is_some_and(|w| w == check || w.is_ancestor(&check));
                if primary && on_check {
                    // Every press, a quick second one too: two clicks on the
                    // check are two toggles, never a double click that plays.
                    g.set_state(gtk::EventSequenceState::Claimed);
                    me.select_quietly(i);
                    me.toggle(i);
                    return;
                }
                if g.current_button() == gdk::BUTTON_SECONDARY {
                    g.set_state(gtk::EventSequenceState::Claimed);
                    me.select_quietly(i);
                    return;
                }
                // Anywhere else on the row selects it; twice plays it.
                if primary {
                    g.set_state(gtk::EventSequenceState::Claimed);
                    me.select_quietly(i);
                    root.grab_focus();
                    if n == 2 {
                        me.launch(false);
                    }
                }
            }
        });
        // The menu opens on the release, not the press: opened during the
        // press, the popover's grab took the release of a press it never
        // saw, and GTK warned of broken accounting of its active state.
        click.connect_released({
            let weak = Rc::downgrade(self);
            let root = root.downgrade();
            move |g, _, x, y| {
                let (Some(me), Some(root)) = (weak.upgrade(), root.upgrade()) else { return };
                if g.current_button() == gdk::BUTTON_SECONDARY {
                    me.popup_menu(Some((&root, x, y)));
                }
            }
        });
        root.add_controller(click);
        Rc::new(FileRow { root, check, picture, name, reference, dir, facts, matched, pill, item, request: RefCell::new(None), dressing })
    }

    /// `key`, which means `act` (`commands::RESULTS`), done if the page takes
    /// it; whether it did.
    ///
    /// The video keys work from anywhere on the page. What they leave alone:
    /// Up, Down, Home, End, Space and Enter in the strip or the tree, which
    /// change and open groups there, and Space and Enter on a button, which
    /// press it. A menu item's letter is the view's menu's alone, and taken
    /// whether or not it can do anything now, as its item would be.
    fn key(self: &Rc<Self>, act: Page, key: gdk::Key, mods: gdk::ModifierType) -> bool {
        if mods.contains(gdk::ModifierType::CONTROL_MASK) {
            return self.run(act, false);
        }
        if mods.contains(gdk::ModifierType::ALT_MASK) {
            if act.label(self.view.get()).is_none() {
                return false;
            }
            if !act.needs_video() || self.selected.get().is_some() {
                self.run(act, false);
            }
            return true;
        }
        if self.shown.borrow().is_empty() {
            return false;
        }
        let focus = gtk::prelude::GtkWindowExt::focus(&self.app.window);
        let in_files = focus.as_ref().is_some_and(|f| f.is_ancestor(&self.files.area) || f == self.files.area.upcast_ref::<gtk::Widget>());
        let presses = matches!(key, gdk::Key::space | gdk::Key::Return | gdk::Key::KP_Enter | gdk::Key::ISO_Enter);
        let theirs = match act {
            Page::Up | Page::Down | Page::First | Page::Last => self.in_side_list(),
            _ => presses && focus.is_some() && !in_files,
        };
        !theirs && self.run(act, mods.contains(gdk::ModifierType::SHIFT_MASK))
    }

    /// Whether the keyboard is in the strip or the tree.
    fn in_side_list(&self) -> bool {
        let side = self.side_list();
        gtk::prelude::GtkWindowExt::focus(&self.app.window).is_some_and(|f| f.is_ancestor(&side) || &f == side.upcast_ref::<gtk::Widget>())
    }

    /// Do `act`, from a key, the menu or a button; whether there was anything
    /// to do it to. With `shift`, an arrow gives every video it passes the
    /// mark the anchor was given (`mark_along`).
    fn run(self: &Rc<Self>, act: Page, shift: bool) -> bool {
        let selected = self.selected.get();
        let go = |i: i64| match (shift, selected) {
            (true, Some(from)) => self.mark_along(from, i),
            _ => self.move_to(i),
        };
        match (act, selected) {
            (Page::Up, Some(i)) => go(i as i64 - 1),
            (Page::Down, Some(i)) => go(i as i64 + 1),
            // Nothing selected yet: either arrow starts at the first.
            (Page::Up | Page::Down, None) => go(0),
            (Page::First, _) => go(0),
            (Page::Last, _) => go(i64::MAX),
            (Page::Mark, Some(i)) => self.toggle(i),
            (Page::Menu, Some(_)) => self.popup_menu(None),
            (Page::Open, Some(_)) => self.launch(false),
            (Page::Compare, Some(_)) => self.compare(),
            (Page::ShowFolder, Some(_)) => self.launch(true),
            (Page::MarkOthers, Some(_)) => self.mark_all_but_selected(),
            (Page::MarkFolder, _) => self.mark_shown(true),
            (Page::Unmark, _) => self.mark_shown(false),
            (Page::UnmarkAll, _) => self.unmark_all(),
            (Page::SwitchView, _) => self.switch_view(),
            (Page::Next, _) => self.step(1),
            (Page::Previous, _) => self.step(-1),
            _ => return false,
        }
        true
    }

    /// Whether this page is the one on screen, and so the window's keys are
    /// its own.
    fn showing(&self) -> bool {
        self.app.stack.visible_child_name().as_deref() == Some("results")
    }

    pub fn set_to_settings(&self, f: impl Fn() + 'static) {
        *self.to_settings.borrow_mut() = Some(Box::new(f));
    }

    pub fn has_results(&self) -> bool {
        self.state.borrow().has_results
    }

    /// A finished scan's groups, replacing whatever was shown.
    pub fn show(self: &Rc<Self>, prepared: Prepared, problems: bool) {
        let Prepared { mut model, kept, roots, changed_after } = prepared;
        // The ranking chosen stays chosen from one scan to the next.
        model.set_priority(self.priority());
        *self.state.borrow_mut() = State { model, changed_after, problems, has_results: true, kept, roots, current: 0 };
        self.showing_scan.set(self.showing_scan.get() + 1);
        // A look waited on for the last scan's results will not come back.
        self.checking.set(false);
        self.marks_pending.set(false);
        self.thumbs.clear_queue();
        self.strip_thumbs.clear_queue();
        self.folders.current.set(None);
        self.rebuild(0);
        // A file written while the scan ran, after it was read, is already
        // not the file it judged.
        self.check_changed(|_| {});
    }

    /// Look again for files gone or changed since the scan, take in what is
    /// found, and then do `then`. Asked when the results are shown, and again
    /// before anything acts on the suggestions: a file can go or change while
    /// they are on screen.
    ///
    /// On another thread: it is a `stat` of every file the scan grouped,
    /// which on a network share is a round trip each. `then` runs on the
    /// window's thread once the answer is in, unless another scan has been
    /// shown meanwhile.
    fn check_changed(self: &Rc<Self>, then: impl FnOnce(&Rc<Self>) + 'static) {
        let (look, after): (Vec<(usize, PathBuf, u64)>, Option<u64>) = {
            let s = self.state.borrow();
            let look = s.model.live().map(|id| (id, s.model.path(id).to_path_buf(), s.model.file(id).size)).collect();
            (look, s.changed_after)
        };
        let scan = self.showing_scan.get();
        let me = self.clone();
        glib::spawn_future_local(async move {
            let found = gio::spawn_blocking(move || {
                let (mut gone, mut changed) = (Vec::new(), Vec::new());
                for (id, path, size) in look {
                    match std::fs::metadata(&path) {
                        Ok(md) if last::changed_since(&md, size, after) => changed.push(id),
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => gone.push(id),
                        Err(_) => {}
                    }
                }
                (gone, changed)
            })
            .await;
            if me.showing_scan.get() != scan {
                return;
            }
            if let Ok((gone, changed)) = found {
                me.take_in(&gone, &changed);
            }
            then(&me);
        });
    }

    /// Take in that files `gone` have gone and files `changed` have changed
    /// since the scan.
    fn take_in(self: &Rc<Self>, gone: &[usize], changed: &[usize]) {
        let moved = self.state.borrow_mut().model.lapse(changed.iter().copied(), Lapse::Changed);
        let paths: HashSet<PathBuf> = {
            let s = self.state.borrow();
            gone.iter().filter(|&&id| !s.model.gone(id)).map(|&id| s.model.path(id).to_path_buf()).collect()
        };
        if !paths.is_empty() {
            for p in &paths {
                self.app.log_line(&format!("gone since the scan: {}", p.display()));
            }
            self.drop_gone(&paths);
        } else if moved {
            let selected = self.selected.get();
            self.refresh_rows();
            self.select(selected);
            self.update_status();
        }
    }

    /// `check_changed` for an action that acts on the suggestions or the
    /// marks: `then` once the files have been looked at. One at a time, and
    /// none while files are being moved to the Trash.
    fn checked(self: &Rc<Self>, then: impl FnOnce(&Rc<Self>) + 'static) {
        if self.checking.get() || self.trashing.get() {
            return;
        }
        self.checking.set(true);
        self.status.set_text("Looking for files changed since the scan…");
        self.check_changed(move |me| {
            me.checking.set(false);
            me.update_status();
            then(me);
        });
    }

    /// The scan kept from last time, as it was left: its groups and marks,
    /// less the files that have gone since.
    pub fn restore(self: &Rc<Self>, prepared: Prepared, problems: bool, marks: &[PathBuf], gone: &HashSet<PathBuf>) {
        self.show(prepared, problems);
        let ids: Vec<usize> = {
            let s = self.state.borrow();
            let marked: HashSet<&Path> = marks.iter().map(PathBuf::as_path).collect();
            (0..s.model.n()).filter(|&id| marked.contains(s.model.path(id))).collect()
        };
        self.mark_ids(&ids, true);
        if !gone.is_empty() {
            self.drop_gone(gone);
        }
        self.update_status();
    }

    /// An empty page while the last scan's results are read, so that the
    /// window opens where it will be rather than on the settings.
    pub fn loading(&self) {
        self.title.set_text("");
        self.note.set_text("Loading the last scan's results…");
        self.note.set_visible(true);
        self.status.set_text("");
        for b in &self.group_buttons {
            b.set_sensitive(false);
        }
        self.trash.set_sensitive(false);
        self.mark_suggested.set_sensitive(false);
    }

    /// The marks have changed: write them for the next window, once they stop
    /// changing for a moment, so that a run of clicks is one write.
    fn marks_changed(self: &Rc<Self>) {
        if self.state.borrow().kept.is_none() || self.marks_pending.replace(true) {
            return;
        }
        let me = Rc::downgrade(self);
        glib::timeout_add_local_once(Duration::from_millis(400), move || {
            if let Some(me) = me.upgrade() {
                me.flush_marks();
            }
        });
    }

    /// Write the marks now, if a save is waiting: the window is closing.
    pub fn flush_marks(&self) {
        if !self.marks_pending.replace(false) {
            return;
        }
        let s = self.state.borrow();
        let Some(kept) = s.kept else { return };
        let paths = s.model.marked_ids().filter(|&id| !s.model.gone(id)).map(|id| s.model.path(id).to_path_buf()).collect();
        last::save_marks(kept, paths);
    }

    /// Keyboard to the page: the selected video, else the first, or the list
    /// on the left when there are none.
    pub fn focus(self: &Rc<Self>) {
        if self.shown.borrow().is_empty() {
            self.side_list().grab_focus();
            return;
        }
        let i = self.selected.get().unwrap_or(0);
        self.move_to(i as i64);
    }

    /// The list on the left: the strip of groups, or the folder tree.
    fn side_list(&self) -> gtk::ListView {
        match self.view.get() {
            View::Groups => self.strip.list.clone(),
            View::Folders => self.folders.list.clone(),
        }
    }

    /// Both lists, from the state, with group `select` selected.
    fn rebuild(self: &Rc<Self>, select: usize) {
        let (faces, tree) = {
            let s = self.state.borrow();
            let m = &s.model;
            let faces: Vec<PathBuf> = m.groups.iter().map(|g| face(m, &g.members).map(|id| m.path(id).to_path_buf()).unwrap_or_default()).collect();
            let files: Vec<(u32, &Path)> = (0..m.n()).filter(|&i| !m.member_of[i].is_empty()).map(|i| (i as u32, m.path(i))).collect();
            (faces, folders::build(&s.roots, &files))
        };
        self.folders.fill(tree);
        let n = faces.len();
        self.strip.fill(faces);
        for b in &self.group_buttons {
            b.set_sensitive(n > 0);
        }
        if n == 0 {
            self.show_files(Vec::new(), "", "");
            let s = self.state.borrow();
            let compared = s.model.found.compared;
            let mut text = format!("No duplicates found among {compared} video{}.", if compared == 1 { "" } else { "s" });
            if s.model.found.found > compared {
                let skipped = s.model.found.found - compared;
                text.push_str(&format!(" {skipped} more could not be compared; see the scan log."));
            } else if s.problems {
                text.push_str(" The scan had problems with some files; see the scan log.");
            }
            self.note.set_text(&text);
            self.note.set_visible(true);
            drop(s);
            self.update_status();
            return;
        }
        let i = select.min(n - 1);
        self.strip.select(i);
        self.state.borrow_mut().current = i;
        self.show_view(true);
        self.update_status();
    }

    /// What the view has selected, on the page: the group, or the folder.
    /// With `reveal`, the list is brought to it too.
    fn show_view(self: &Rc<Self>, reveal: bool) {
        match self.view.get() {
            View::Groups => {
                let current = self.state.borrow().current;
                self.show_group(current);
                if reveal {
                    self.strip.scroll_to(current);
                }
            }
            View::Folders => {
                let top = || self.folders.tree.borrow().top.first().copied();
                let Some(f) = (if reveal { self.folders.current.get().or_else(top) } else { self.folders.current.get() }) else { return };
                if reveal {
                    self.folders.open_to(f);
                }
                self.show_folder(f);
            }
        }
    }

    /// The group or folder `by` from the one shown, and the keyboard to its
    /// videos.
    fn step(self: &Rc<Self>, by: i32) {
        let moved = match self.view.get() {
            View::Groups => self.strip.step(by),
            View::Folders => self.folders.step(by),
        };
        if moved {
            self.focus();
        }
    }

    /// The files of the group or folder shown.
    fn scope(&self) -> Vec<usize> {
        self.shown.borrow().iter().map(|(m, _)| m.file).collect()
    }

    /// Show the other view: the tree in place of the strip, or the strip
    /// again.
    fn switch_view(self: &Rc<Self>) {
        let view = self.view.get().other();
        self.view.set(view);
        fill_menu(&self.menu, view);
        self.left.set_visible_child_name(view.child());
        self.left_label.set_text_with_mnemonic(view.heading());
        self.left_label.set_mnemonic_widget(Some(&self.side_list()));
        let what = view.noun();
        self.prev.set_tooltip_text(Some(&format!("Previous {what} (Ctrl+Page Up)")));
        self.next.set_tooltip_text(Some(&format!("Next {what} (Ctrl+Page Down)")));
        if !self.state.borrow().model.groups.is_empty() {
            self.show_view(true);
        }
        self.side_list().grab_focus();
    }

    fn bind_tree_row(&self, item: &gtk::ListItem) {
        let Some(row) = item.item().and_downcast::<gtk::TreeListRow>() else { return };
        let Some(f) = row.item().and_downcast::<glib::BoxedAnyObject>().map(|o| *o.borrow::<usize>()) else { return };
        let rows = self.folders.rows.borrow();
        let Some(r) = rows.get(item) else { return };
        r.expander.set_list_row(Some(&row));
        r.folder.set(Some(f));
        r.shown.set(None);
        let t = self.folders.tree.borrow();
        let Some(folder) = t.folders.get(f) else { return };
        let label = folder.label();
        r.name.set_text(&label);
        r.count.set_text(&folder.all.len().to_string());
        drop(t);
        self.show_tree_row(r);
        r.expander.update_property(&[gtk::accessible::Property::Label(&label)]);
    }

    /// A row of the tree says how many videos under its folder are marked.
    fn show_tree_row(&self, r: &TreeRow) {
        let Some(f) = r.folder.get() else { return };
        let marked = {
            let (s, t) = (self.state.borrow(), self.folders.tree.borrow());
            t.folders.get(f).map_or(0, |f| f.all.iter().filter(|&&id| s.model.is_marked(id as usize)).count())
        };
        if r.shown.replace(Some(marked)) == Some(marked) {
            return;
        }
        if marked > 0 {
            r.marked.set_text(&marked.to_string());
        }
        r.marked.set_visible(marked > 0);
    }

    /// The videos of folder `f`, its subfolders' included, on the page.
    fn show_folder(self: &Rc<Self>, f: usize) {
        let (path, own, all) = {
            let t = self.folders.tree.borrow();
            let Some(folder) = t.folders.get(f) else { return };
            (folder.path.clone(), folder.own, folder.all.clone())
        };
        self.folders.current.set(Some(f));
        let (files, problems) = {
            let s = self.state.borrow();
            let m = &s.model;
            // Each file as its first group has it, and that group.
            let files: Vec<(Member, usize)> = all
                .iter()
                .filter(|&&id| !m.gone(id as usize))
                .filter_map(|&id| {
                    let gi = *m.member_of.get(id as usize)?.first()?;
                    let member = m.groups.get(gi)?.members.iter().find(|x| x.file == id as usize)?.clone();
                    Some((member, gi))
                })
                .collect();
            (files, s.problems)
        };
        let n = files.len();
        let mut title = format!("{} · {n} video{}", path.display(), if n == 1 { "" } else { "s" });
        if own != all.len() {
            title.push_str(&format!(" ({own} in this folder)"));
        }
        let note = if problems { "The scan had problems with some files; see the scan log." } else { "" };
        self.show_files(files, &title, note);
    }

    /// Row `item` of the strip now shows the group its item names.
    fn bind_row(&self, item: &gtk::ListItem) {
        let Some(gi) = item.item().and_downcast::<glib::BoxedAnyObject>().map(|o| *o.borrow::<usize>()) else { return };
        let strip = self.strip.rows.borrow();
        let Some(row) = strip.get(item) else { return };
        row.group.set(Some(gi));
        row.shown.set(None);
        self.show_row(row, gi);
        row.picture.set_texture(None);
        let Some(face) = self.strip.faces.borrow().get(gi).cloned() else { return };
        let scale = self.scale();
        let size = Size::cover(STRIP_W as u32 * scale, STRIP_H as u32 * scale);
        let ticket = self.strip_thumbs.request(&face, size, Shown::Still(row.picture.clone()));
        *row.request.borrow_mut() = Some(((face, size), ticket));
    }

    /// Row `item` of the strip no longer shows a group: its picture is not
    /// wanted, and not held.
    fn unbind_row(&self, item: &gtk::ListItem) {
        let strip = self.strip.rows.borrow();
        let Some(row) = strip.get(item) else { return };
        row.group.set(None);
        if let Some((key, ticket)) = row.request.borrow_mut().take() {
            self.strip_thumbs.withdraw(&key, ticket);
        }
        row.picture.set_texture(None);
    }

    /// Group `gi`'s row of the strip, if one shows it, says what it holds.
    fn update_row(&self, gi: usize) {
        let strip = self.strip.rows.borrow();
        if let Some(row) = strip.values().find(|r| r.group.get() == Some(gi)) {
            self.show_row(row, gi);
        }
    }

    fn show_row(&self, r: &StripRow, gi: usize) {
        let s = self.state.borrow();
        let Some(g) = s.model.groups.get(gi) else { return };
        let marked = g.members.iter().filter(|m| s.model.is_marked(m.file)).count();
        let before = r.shown.replace(Some(marked));
        if before == Some(marked) {
            return;
        }
        if before.is_none() {
            r.count.set_text(&g.members.len().to_string());
        }
        if marked > 0 {
            r.marked.set_text(&marked.to_string());
        }
        r.marked.set_visible(marked > 0);
        let mut text = format!("Group {} · {} videos", gi + 1, g.members.len());
        if marked > 0 {
            text.push_str(&format!(" · {marked} marked"));
        }
        r.overlay.update_property(&[gtk::accessible::Property::Label(&text)]);
    }

    fn show_group(self: &Rc<Self>, gi: usize) {
        let (members, reference_gone, n, problems) = {
            let mut s = self.state.borrow_mut();
            let Some(g) = s.model.groups.get(gi) else { return };
            let members = g.members.clone();
            let gone = g.reference_gone;
            s.current = gi;
            (members, gone, s.model.groups.len(), s.problems)
        };
        let title = format!("Group {} of {} · {} videos", gi + 1, n, members.len());
        let mut note = String::new();
        if reference_gone {
            note.push_str("The video the others in this group were matched against is gone. The videos left were not all compared with each other. ");
        }
        if problems {
            note.push_str("The scan had problems with some files; see the scan log.");
        }
        self.show_files(members.into_iter().map(|m| (m, gi)).collect(), &title, &note);
    }

    /// `files`, each with the group it is taken from, on the page.
    fn show_files(self: &Rc<Self>, shown: Vec<(Member, usize)>, title: &str, note: &str) {
        self.thumbs.clear_queue();
        self.select(None);
        self.anchor.set(None);
        self.title.set_text(title);
        self.note.set_text(note.trim());
        self.note.set_visible(!note.trim().is_empty());
        *self.shown.borrow_mut() = shown;
        // Every row is about another video now, and is bound again when it
        // is placed: the rows on screen, a handful, whatever the count.
        let pool: Vec<Rc<FileRow>> = self.files.pool.borrow().clone();
        for row in &pool {
            self.unbind_file_row(row);
        }
        self.files.want_focus.set(None);
        self.files.area.adjustment().set_value(0.0);
        self.files.area.queue_allocate();
    }

    /// The rows on screen, placed: `w` x `h` to fill, scrolled `offset`
    /// pixels down. Called by `Rows` on every allocation and scroll.
    fn lay_out(self: &Rc<Self>, w: i32, h: i32, offset: f64) {
        let n = self.shown.borrow().len();
        let rh = self.row_height(w).max(1);
        // Enough rows to cover the height wherever the top one is cut.
        let count = ((h + rh - 1) / rh + 1).max(1) as usize;
        self.grow_pool(count);
        let total = n as f64 * rh as f64;
        let page = h.max(0) as f64;
        let value = offset.clamp(0.0, (total - page).max(0.0));
        let adj = self.files.area.adjustment();
        if adj.upper() != total || adj.page_size() != page || adj.value() != value || adj.step_increment() != rh as f64 {
            adj.configure(value, 0.0, total, rh as f64, (page - rh as f64).max(rh as f64), page);
        }
        let first = (value / rh as f64) as usize;
        let last = (first + count).min(n);
        let pool: Vec<Rc<FileRow>> = self.files.pool.borrow().clone();
        let mut used = vec![false; pool.len()];
        let selected = self.selected.get();
        for i in first..last {
            let slot = i % pool.len();
            used[slot] = true;
            let row = &pool[slot];
            if row.item.get() != Some(i) {
                self.bind_file_row(row, i);
            }
            set_class(&row.root, "selected", selected == Some(i));
            row.root.set_child_visible(true);
            let _ = row.root.measure(gtk::Orientation::Vertical, w);
            let y = (i as f64 * rh as f64 - value).round() as i32;
            row.root.size_allocate(&gtk::Allocation::new(0, y, w, rh), -1);
            if self.files.want_focus.get() == Some(i) {
                self.files.want_focus.set(None);
                row.root.grab_focus();
            }
        }
        for (slot, row) in pool.iter().enumerate() {
            if !used[slot] {
                if row.item.get().is_some() {
                    self.unbind_file_row(row);
                }
                row.root.set_child_visible(false);
            }
        }
    }

    /// A row's height at width `w`: every row is the same shape, so one
    /// measured row speaks for all of them.
    fn row_height(self: &Rc<Self>, w: i32) -> i32 {
        let (at, h) = self.files.row_height.get();
        if at == w && h > 0 {
            return h;
        }
        self.grow_pool(1);
        let row = self.files.pool.borrow()[0].clone();
        let h = row.root.measure(gtk::Orientation::Vertical, w).1.max(ROW_H);
        self.files.row_height.set((w, h));
        h
    }

    /// At least `count` row widgets.
    fn grow_pool(self: &Rc<Self>, count: usize) {
        while self.files.pool.borrow().len() < count {
            let row = self.new_file_row();
            row.root.set_parent(&self.files.area);
            row.root.set_child_visible(false);
            self.files.pool.borrow_mut().push(row);
        }
    }

    /// The row on screen showing video `i`, if one does.
    fn row_of(&self, i: usize) -> Option<Rc<FileRow>> {
        self.files.pool.borrow().iter().find(|r| r.item.get() == Some(i)).cloned()
    }

    /// Scroll so video `i`'s row is whole on screen.
    fn scroll_to(&self, i: usize) {
        let (_, rh) = self.files.row_height.get();
        let adj = self.files.area.adjustment();
        if rh <= 0 {
            return;
        }
        let top = i as f64 * rh as f64;
        if top < adj.value() {
            adj.set_value(top);
        } else if top + rh as f64 > adj.value() + adj.page_size() {
            adj.set_value(top + rh as f64 - adj.page_size());
        }
    }

    fn scale(&self) -> u32 {
        self.app.window.scale_factor().max(1) as u32
    }

    /// `row` now shows video `i`.
    fn bind_file_row(&self, row: &Rc<FileRow>, i: usize) {
        self.unbind_file_row(row);
        row.item.set(Some(i));
        let Some((m, _)) = self.shown.borrow().get(i).cloned() else { return };
        self.dress(row, &m);
        let path = PathBuf::from(&self.state.borrow().model.file(m.file).path);
        let size = Size::within(ROW_W as u32 * self.scale(), ROW_H as u32 * self.scale());
        let ticket = self.thumbs.request(&path, size, Shown::Still(row.picture.clone()));
        *row.request.borrow_mut() = Some(((path, size), ticket));
    }

    /// `row` no longer shows a video: its picture is not wanted.
    fn unbind_file_row(&self, row: &FileRow) {
        row.item.set(None);
        if let Some((key, ticket)) = row.request.borrow_mut().take() {
            self.thumbs.withdraw(&key, ticket);
        }
        row.picture.set_texture(None);
    }

    /// Every row bound, dressed again: the suggestions or the marks moved.
    fn refresh_rows(&self) {
        let rows: Vec<Rc<FileRow>> = self.files.pool.borrow().clone();
        for row in rows {
            let Some(i) = row.item.get() else { continue };
            let Some((m, _)) = self.shown.borrow().get(i).cloned() else { continue };
            self.dress(&row, &m);
        }
    }

    /// `row`, dressed for `m`: everything the command line's listing says of
    /// the file, its strongest match in its group, and what is suggested.
    fn dress(&self, row: &FileRow, m: &Member) {
        let s = self.state.borrow();
        let model = &s.model;
        let f = model.file(m.file);
        let path = Path::new(&f.path);
        row.name.set_text(&file_name(path));
        row.dir.set_text(&path.parent().map(|p| p.display().to_string()).unwrap_or_default());
        row.facts.set_text(&facts(f));
        let groups = model.member_of.get(m.file).map_or(0, Vec::len);
        let is_reference = |gi: usize| model.groups.get(gi).and_then(|g| g.members.first()).is_some_and(|r| r.file == m.file);
        let gi = self.shown.borrow().get(row.item.get().unwrap_or(usize::MAX)).map(|(_, gi)| *gi);
        let reference = gi.is_some_and(is_reference);
        row.reference.set_visible(reference);
        let mut matched = match m.matched_with {
            Some(w) => {
                let other = file_name(Path::new(&model.file(w).path));
                let span = match (m.matched_from, m.matched_to) {
                    (Some(a), Some(b)) => format!(" ({} to {} of this video)", utils::format_duration(a), utils::format_duration(b)),
                    _ => String::new(),
                };
                let gone = if model.gone(w) { ", which is gone since the scan" } else { "" };
                if m.identical {
                    // Every byte agrees, so the footage and where it sits go
                    // without saying.
                    format!("Identical copy of {other}{gone}")
                } else {
                    format!("{} matched with {other}{span}{gone}", utils::format_shared(m.matched_seconds))
                }
            }
            None => "Nothing in this group could be measured against it".to_string(),
        };
        if self.view.get() == View::Folders {
            if let Some(gi) = gi {
                let more = if groups > 1 { format!(" (and {} more)", groups - 1) } else { String::new() };
                matched = format!("Group {}{more} · {matched}", gi + 1);
            }
        }
        row.matched.set_text(&matched);
        row.matched.set_tooltip_text(Some(&matched));
        let action = model.action(m.file);
        let marked = model.is_marked(m.file);
        let said = model.said(m.file);
        row.pill.set_text(said);
        for c in ["keep", "delete", "review"] {
            row.pill.remove_css_class(c);
        }
        row.pill.add_css_class(match action {
            Action::Keep => "keep",
            Action::Delete => "delete",
            Action::Review => "review",
        });
        let lost: Vec<String> = model.lost_to(m.file).iter().map(|&w| file_name(Path::new(&model.file(w).path))).collect();
        let tip = match action {
            Action::Keep => KEEP_TIP.to_string(),
            Action::Delete => format!("{DELETE_TIP}\nIt lost to: {}", lost.join(", ")),
            Action::Review if model.scanned(m.file) == Action::Delete => {
                format!("Suggested for deletion by the scan, as it lost to {}. That no longer holds: {}.", lost.join(", "), said.trim_start_matches("REVIEW · "))
            }
            Action::Review => REVIEW_TIP.to_string(),
        };
        row.pill.set_tooltip_text(Some(&tip));
        let name = file_name(path);
        drop(s);
        row.dressing.set(true);
        row.check.set_active(marked);
        row.dressing.set(false);
        // On the row's own box and never on the list's row around it: dress
        // runs in bind, before GTK 4.14 has parented that row, and a Rust
        // handle taken to it then (`parent()`) sinks its floating reference
        // and frees it when dropped, leaving the list a dangling widget.
        set_class(&row.root, "marked", marked);
        row.picture.set_label(&name);
        row.root.update_property(&[gtk::accessible::Property::Label(&format!("{name}, {said}{}", if marked { ", marked" } else { "" }))]);
    }

    /// The ranking chosen in `Rank by`.
    fn priority(&self) -> Priority {
        Priority::ALL[(self.rank.selected() as usize).min(Priority::ALL.len() - 1)]
    }

    /// Show ranking `p`'s suggestions, kept for the next window. Every row,
    /// badge and group picture is redone, since a group's picture is the
    /// video it keeps; the marks stay as they are.
    fn set_priority(self: &Rc<Self>, p: Priority) {
        crate::settings::save_ranking(p);
        if !self.state.borrow_mut().model.set_priority(p) {
            return;
        }
        let (current, selected) = (self.state.borrow().current, self.selected.get());
        if self.state.borrow().model.groups.is_empty() {
            self.update_status();
            return;
        }
        self.rebuild(current);
        if let Some(i) = selected {
            self.move_to(i as i64);
        }
    }

    /// The video menu: at `at` (a row and a point in it), or at the selected
    /// row.
    fn popup_menu(&self, at: Option<(&gtk::Box, f64, f64)>) {
        let bound;
        let (row, x, y) = match at {
            Some(a) => a,
            None => {
                let Some(i) = self.selected.get() else { return };
                bound = self.row_of(i).map(|r| r.root.clone());
                let Some(root) = bound.as_ref() else { return };
                (root, 24.0, root.height() as f64 / 2.0)
            }
        };
        #[allow(deprecated)]
        let Some(parent) = self.context.parent() else { return };
        #[allow(deprecated)]
        let Some((mx, my)) = row.translate_coordinates(&parent, x, y) else { return };
        self.context.set_pointing_to(Some(&gdk::Rectangle::new(mx as i32, my as i32, 1, 1)));
        self.context.popup();
    }

    /// Select video `i`, or none: the menu's video items act on it.
    fn select(&self, i: Option<usize>) {
        let i = i.filter(|&i| i < self.shown.borrow().len());
        self.selected.set(i);
        for row in self.files.pool.borrow().iter() {
            set_class(&row.root, "selected", row.item.get().is_some() && row.item.get() == i);
        }
        for a in self.video_actions.borrow().iter() {
            a.set_enabled(i.is_some());
        }
    }

    /// Select video `i` without moving the keyboard: it was clicked.
    fn select_quietly(&self, i: usize) {
        if self.selected.get() != Some(i) {
            self.select(Some(i));
        }
    }

    /// Select video `i`, clamped to those shown, bring it into view and give
    /// it the keyboard.
    fn move_to(self: &Rc<Self>, i: i64) {
        let n = self.shown.borrow().len() as i64;
        if n == 0 {
            return;
        }
        let i = i.clamp(0, n - 1) as usize;
        self.select(Some(i));
        self.scroll_to(i);
        // Now if it has a row, else once it is placed (`lay_out`).
        match self.row_of(i) {
            Some(row) => {
                self.files.want_focus.set(None);
                row.root.grab_focus();
            }
            None => {
                self.files.want_focus.set(Some(i));
                self.files.area.queue_allocate();
            }
        }
    }

    fn toggle(self: &Rc<Self>, i: usize) {
        let Some(id) = self.shown.borrow().get(i).map(|(m, _)| m.file) else { return };
        let now = !self.state.borrow().model.is_marked(id);
        self.set_marked(i, now);
    }

    /// Mark or unmark video `i`'s file, everywhere it is shown, and make it
    /// the anchor.
    fn set_marked(self: &Rc<Self>, i: usize, on: bool) {
        let Some(id) = self.shown.borrow().get(i).map(|(m, _)| m.file) else { return };
        self.anchor.set(Some((i, on)));
        self.mark_ids(&[id], on);
        self.update_status();
    }

    /// Shift and a click on video `i`: it is marked or unmarked as a click on
    /// its check would, and every video from the anchor to it is given the
    /// same. With no anchor, from the selected video; with neither, `i`
    /// alone. `i` is the anchor after.
    fn mark_range(self: &Rc<Self>, i: usize) {
        let Some(id) = self.shown.borrow().get(i).map(|(m, _)| m.file) else { return };
        let on = !self.state.borrow().model.is_marked(id);
        let from = self.anchor.get().map(|(a, _)| a).or(self.selected.get()).unwrap_or(i);
        self.mark_span(from, i, on);
        self.select(Some(i));
    }

    /// Shift and an arrow from video `from` to `to`: move there, and give
    /// every video from `from` to it the anchor's mark. With no anchor,
    /// `from` is toggled first and becomes it, so the first Shift and arrow
    /// starts a run the way Space would.
    fn mark_along(self: &Rc<Self>, from: usize, to: i64) {
        let n = self.shown.borrow().len() as i64;
        if n == 0 {
            return;
        }
        let to = to.clamp(0, n - 1) as usize;
        let on = match self.anchor.get() {
            Some((_, on)) => on,
            None => {
                let Some(id) = self.shown.borrow().get(from).map(|(m, _)| m.file) else { return };
                !self.state.borrow().model.is_marked(id)
            }
        };
        self.mark_span(from, to, on);
        self.move_to(to as i64);
    }

    /// Videos `a` to `b`, either way round and both included, marked or
    /// unmarked, with `b` the anchor.
    fn mark_span(self: &Rc<Self>, a: usize, b: usize, on: bool) {
        let ids: Vec<usize> = {
            let shown = self.shown.borrow();
            let (lo, hi) = (a.min(b), a.max(b).min(shown.len().saturating_sub(1)));
            shown.get(lo..=hi).map_or_else(Vec::new, |s| s.iter().map(|(m, _)| m.file).collect())
        };
        self.anchor.set(Some((b, on)));
        self.mark_ids(&ids, on);
        self.update_status();
    }

    /// Mark or unmark every file of `ids`, then show it: each row and each
    /// group's badge once, however many of the files it holds.
    fn mark_ids(self: &Rc<Self>, ids: &[usize], on: bool) {
        let groups: HashSet<usize> = {
            let mut s = self.state.borrow_mut();
            let mut groups = HashSet::new();
            for &id in ids {
                if s.model.set_mark(id, on) {
                    groups.extend(s.model.member_of.get(id).into_iter().flatten().copied());
                }
            }
            groups
        };
        if groups.is_empty() {
            return;
        }
        self.refresh_rows();
        self.marks_changed();
        for r in self.folders.rows.borrow().values() {
            self.show_tree_row(r);
        }
        for gi in groups {
            self.update_row(gi);
        }
    }

    fn mark_all_but_selected(self: &Rc<Self>) {
        let Some(keep) = self.selected.get() else { return };
        let Some(this) = self.shown.borrow().get(keep).map(|(m, _)| m.file) else { return };
        let others: Vec<usize> = self.scope().into_iter().filter(|&id| id != this).collect();
        self.mark_ids(&[this], false);
        self.mark_ids(&others, true);
        self.update_status();
        self.move_to(keep as i64);
    }

    /// Make the marks the scan's deletions, exactly: every file suggested for
    /// deletion is marked and every other mark goes, whoever made it.
    fn mark_suggested(self: &Rc<Self>) {
        self.checked(|me| me.mark_suggested_now());
    }

    fn mark_suggested_now(self: &Rc<Self>) {
        let (off, on): (Vec<usize>, Vec<usize>) = {
            let s = self.state.borrow();
            let m = &s.model;
            let delete = |id: usize| m.action(id) == Action::Delete;
            m.live().filter(|&id| m.is_marked(id) != delete(id)).partition(|&id| m.is_marked(id))
        };
        self.mark_ids(&off, false);
        self.mark_ids(&on, true);
        self.update_status();
    }

    /// Mark or unmark the group shown, or every video under the folder shown.
    fn mark_shown(self: &Rc<Self>, on: bool) {
        let ids = self.scope();
        self.mark_ids(&ids, on);
        self.update_status();
    }

    /// Unmark every video, in every group.
    fn unmark_all(self: &Rc<Self>) {
        let ids: Vec<usize> = self.state.borrow().model.marked_ids().collect();
        self.mark_ids(&ids, false);
        self.update_status();
    }

    fn update_status(&self) {
        // The trash in progress owns the status line and the button, and says
        // how it went when it is done.
        if self.trashing.get() {
            return;
        }
        let s = self.state.borrow();
        let m = &s.model;
        let (n, bytes) = (m.marked_n, m.marked_bytes);
        let groups = m.groups.len();
        let suggested: Vec<usize> = m.suggested().collect();
        let text = if n == 0 && !suggested.is_empty() {
            let bytes: u64 = suggested.iter().map(|&id| m.file(id).size).sum();
            format!(
                "{groups} group{}. {} suggested for deletion, {}.",
                if groups == 1 { "" } else { "s" },
                suggested.len(),
                utils::format_size(bytes)
            )
        } else if n == 0 {
            format!("{groups} group{}. Nothing marked.", if groups == 1 { "" } else { "s" })
        } else {
            format!("{n} video{} marked, {}.", if n == 1 { "" } else { "s" }, utils::format_size(bytes))
        };
        self.status.set_text(&text);
        self.trash.set_sensitive(n > 0);
        self.trash.set_label(&if n == 0 { l::TRASH.to_string() } else { l::trash_n(n) });
        // Off only when the marks already are the suggestion.
        let same = m.live().all(|id| m.is_marked(id) == (m.action(id) == Action::Delete));
        self.mark_suggested.set_sensitive(!same);
    }

    /// The frames the scan matched between the selected video and what it
    /// matched, in a window of their own (`matched.rs`).
    fn compare(self: &Rc<Self>) {
        let Some(i) = self.selected.get() else { return };
        let Some((m, _)) = self.shown.borrow().get(i).cloned() else { return };
        let subject = {
            let s = self.state.borrow();
            let model = &s.model;
            let against = m
                .compared
                .iter()
                .map(|c| {
                    let how = match (Some(c.with) == m.matched_with, m.identical) {
                        (true, true) => "Identical copy",
                        (true, false) => "Best match",
                        (false, _) => "Group reference",
                    };
                    (how, c.clone(), model.file(c.with).clone())
                })
                .collect::<Vec<_>>();
            crate::matched::Subject { file: model.file(m.file).clone(), against }
        };
        if subject.against.is_empty() {
            self.status.set_text("Nothing in its group was measured against it.");
            return;
        }
        crate::matched::open(&self.app, subject);
    }

    /// Play the selected video in the desktop's player, or show its folder.
    fn launch(self: &Rc<Self>, folder: bool) {
        let Some(i) = self.selected.get() else { return };
        let Some(id) = self.shown.borrow().get(i).map(|(m, _)| m.file) else { return };
        let path = self.state.borrow().model.path(id).to_path_buf();
        let launcher = gtk::FileLauncher::new(Some(&gio::File::for_path(&path)));
        let window = self.app.window.clone();
        let app = self.app.clone();
        let done = move |r: Result<(), glib::Error>| {
            if let Err(e) = r {
                app.log_line(&format!("could not open {}: {e}", path.display()));
            }
        };
        if folder {
            launcher.open_containing_folder(Some(&window), gio::Cancellable::NONE, done);
        } else {
            launcher.launch(Some(&window), gio::Cancellable::NONE, done);
        }
    }

    // ------------------------------------------------------------ the Trash

    fn confirm_trash(self: &Rc<Self>) {
        // What the suggestions rest on, as it is now.
        self.checked(|me| me.confirm_trash_now());
    }

    fn confirm_trash_now(self: &Rc<Self>) {
        if self.trashing.get() {
            return;
        }
        let (moves, bytes, whole, unheld, changed_after) = {
            let s = self.state.borrow();
            let m = &s.model;
            // With the length the scan found, which the move checks first,
            // and the files each deletion rests on as things stand now.
            let file = |id: usize| (m.path(id).to_path_buf(), m.file(id).size);
            let mut moves: Vec<Move> = m.marked_ids().map(|id| Move { path: file(id).0, scanned: file(id).1, holders: m.holders(id).into_iter().map(file).collect() }).collect();
            moves.sort_by(|a, b| a.path.cmp(&b.path));
            (moves, m.marked_bytes, m.whole_groups_marked(), m.marked_unheld(), s.changed_after)
        };
        if moves.is_empty() {
            return;
        }
        let n = moves.len();
        let mut detail = format!(
            "{} {}. {} can be restored from the Trash.",
            if n == 1 { "It takes" } else { "Together they take" },
            utils::format_size(bytes),
            if n == 1 { "It" } else { "They" }
        );
        if !whole.is_empty() {
            let one = whole.len() == 1;
            detail.push_str(&format!(
                "\n\nIn {} {} every video is marked, so no copy of {} would be left.",
                if one { "group" } else { "groups" },
                numbers(&whole),
                if one { "that video" } else { "those videos" }
            ));
        }
        // Groups overlap, so a video can lose its last copy with no group all
        // marked. What the scan knows is which file each deletion lost to.
        if unheld > 0 {
            let one = unheld == 1;
            detail.push_str(&format!(
                "\n\n{unheld} of {} {} suggested for deletion because {} lost to another video, and every video {} lost to is \
                 marked too, or has gone or changed: moved together, nothing the scan found would be left holding what {}.",
                if n == 1 { "it" } else { "them" },
                if one { "was" } else { "were" },
                if one { "it" } else { "they" },
                if one { "it" } else { "they" },
                if one { "it shows" } else { "they show" }
            ));
        }
        let dialog = gtk::AlertDialog::builder()
            .message(format!("Move {n} video{} to the Trash?", if n == 1 { "" } else { "s" }))
            .detail(detail)
            .buttons([l::CONFIRM_CANCEL, l::CONFIRM_TRASH])
            .cancel_button(0)
            .default_button(0)
            .modal(true)
            .build();
        let me = self.clone();
        glib::spawn_future_local(async move {
            if matches!(dialog.choose_future(Some(&me.app.window)).await, Ok(1)) {
                me.trash(moves, changed_after).await;
            }
        });
    }

    async fn trash(self: &Rc<Self>, moves: Vec<Move>, changed_after: Option<u64>) {
        if self.trashing.replace(true) {
            return;
        }
        self.trash.set_sensitive(false);
        self.mark_suggested.set_sensitive(false);
        self.settings_button.set_sensitive(false);
        let n = moves.len();
        let mut gone: HashSet<PathBuf> = HashSet::new();
        let mut failed: Vec<String> = Vec::new();
        // Files on a drive with no Trash, for `delete_instead`.
        let mut no_trash: Vec<Move> = Vec::new();
        for (i, m) in moves.into_iter().enumerate() {
            self.status.set_text(&format!("Moving to the Trash: {} of {n}…", i + 1));
            let (m, why) = m.looked_at(changed_after).await;
            if let Some(why) = why {
                failed.push(why);
                continue;
            }
            let p = &m.path;
            match gio::File::for_path(p).trash_future(glib::Priority::DEFAULT).await {
                Ok(()) => {
                    gone.insert(p.clone());
                }
                // Already gone is as good as moved.
                Err(e) if e.matches(gio::IOErrorEnum::NotFound) => {
                    gone.insert(p.clone());
                }
                // A drive or share with no Trash, or one this user cannot make
                // there.
                Err(e) if e.matches(gio::IOErrorEnum::NotSupported) => no_trash.push(m),
                Err(e) => failed.push(format!("{}: {}", p.display(), e.message())),
            }
        }
        for p in &gone {
            self.app.log_line(&format!("moved to the Trash: {}", p.display()));
        }
        if !no_trash.is_empty() {
            self.delete_instead(no_trash, changed_after, &mut gone, &mut failed).await;
        }
        self.trashing.set(false);
        self.settings_button.set_sensitive(true);
        self.drop_gone(&gone);
        if !failed.is_empty() {
            for f in &failed {
                self.app.log_line(&format!("left where it is: {f}"));
            }
            let more = failed.len().saturating_sub(10);
            let mut detail = failed.iter().take(10).cloned().collect::<Vec<_>>().join("\n");
            if more > 0 {
                detail.push_str(&format!("\n… and {more} more"));
            }
            let d = gtk::AlertDialog::builder()
                .message(format!("{} video{} could not be moved to the Trash", failed.len(), if failed.len() == 1 { "" } else { "s" }))
                .detail(detail)
                .modal(true)
                .build();
            d.show(Some(&self.app.window));
        }
    }

    /// Files the Trash could not take because their drive has none (a share,
    /// a system mount): deleted for good if the person says so, which they
    /// are asked once for all of them. Each is looked at again first, as the
    /// Trash looks (`Move::stays`).
    async fn delete_instead(self: &Rc<Self>, moves: Vec<Move>, changed_after: Option<u64>, gone: &mut HashSet<PathBuf>, failed: &mut Vec<String>) {
        let n = moves.len();
        let mut names: String = moves.iter().take(10).map(|m| format!("\n{}", m.path.display())).collect();
        if n > 10 {
            names.push_str(&format!("\n… and {} more", n - 10));
        }
        // Deleted for good, a file whose only holder was just moved to the
        // Trash leaves the Trash holding the only copy.
        let in_trash = moves.iter().filter(|m| !m.holders.is_empty() && m.holders.iter().all(|(h, _)| gone.contains(h))).count();
        let mut warn = String::new();
        if in_trash > 0 {
            let one = in_trash == 1;
            warn.push_str(&format!(
                "\n\n{in_trash} of {} {} suggested for deletion because of a video that was just moved to the Trash: the Trash would hold the only copy.",
                if n == 1 { "it" } else { "them" },
                if one { "is" } else { "are" },
            ));
        }
        let dialog = gtk::AlertDialog::builder()
            .message(format!("{n} video{} cannot be moved to the Trash", if n == 1 { "" } else { "s" }))
            .detail(format!(
                "{} on a drive that has no Trash. Delete {} permanently instead? This cannot be undone.{warn}\n{names}",
                if n == 1 { "It is" } else { "They are" },
                if n == 1 { "it" } else { "them" }
            ))
            .buttons([l::CONFIRM_KEEP, l::CONFIRM_DELETE])
            .cancel_button(0)
            .default_button(0)
            .modal(true)
            .build();
        self.status.set_text(&format!("{n} video{} cannot be moved to the Trash", if n == 1 { "" } else { "s" }));
        if !matches!(dialog.choose_future(Some(&self.app.window)).await, Ok(1)) {
            failed.extend(moves.iter().map(|m| format!("{}: its drive has no Trash, so it was left where it is", m.path.display())));
            return;
        }
        for (i, m) in moves.into_iter().enumerate() {
            self.status.set_text(&format!("Deleting: {} of {n}…", i + 1));
            let (m, why) = m.looked_at(changed_after).await;
            if let Some(why) = why {
                failed.push(why);
                continue;
            }
            let p = &m.path;
            match gio::File::for_path(p).delete_future(glib::Priority::DEFAULT).await {
                Ok(()) => {
                    self.app.log_line(&format!("deleted: {}", p.display()));
                    gone.insert(p.clone());
                }
                Err(e) if e.matches(gio::IOErrorEnum::NotFound) => {
                    gone.insert(p.clone());
                }
                Err(e) => failed.push(format!("{}: {}", p.display(), e.message())),
            }
        }
    }

    /// Take files that are no longer there off the page: unmarked, out of
    /// their groups, and a group left with one file gone. A deletion that
    /// rested only on files now gone is suggested for review instead
    /// (`Model::action`).
    fn drop_gone(self: &Rc<Self>, gone: &HashSet<PathBuf>) {
        for p in gone {
            self.thumbs.forget(p);
            self.strip_thumbs.forget(p);
        }
        let (select, unmarked) = {
            let mut s = self.state.borrow_mut();
            let mut current = s.current;
            let unmarked = s.model.drop_gone(gone, &mut current);
            (current, unmarked)
        };
        if unmarked {
            self.marks_changed();
        }
        self.thumbs.clear_queue();
        self.rebuild(select);
        self.focus();
    }
}

/// A marked file the Trash is to move: its path and the length the scan
/// found, and the files its deletion rests on (path and length), as things
/// stood when the move was confirmed.
struct Move {
    path: PathBuf,
    scanned: u64,
    holders: Vec<(PathBuf, u64)>,
}

impl Move {
    /// Why the file is to stay where it is, asked just before it is moved, or
    /// `None` to move it.
    ///
    /// A file at its path that is not the length the scan found, or that has
    /// changed since the scan began, is not the file the scan judged. And a
    /// deletion rests on what it lost to, looked at again here: the dialog
    /// waits on the person, and a holder deleted or changed in another
    /// program meanwhile would take the last copy with this one. One holder
    /// still there as the scan found it is enough.
    fn stays(&self, changed_after: Option<u64>) -> Option<String> {
        let p = self.path.display();
        if let Ok(md) = std::fs::metadata(&self.path) {
            if last::changed_since(&md, self.scanned, changed_after) {
                return Some(format!("{p}: changed since the scan, so it was left where it is"));
            }
        }
        if self.holders.is_empty() {
            return None;
        }
        // Any failure to look is no proof it is there.
        let held = self.holders.iter().any(|(h, size)| std::fs::metadata(h).is_ok_and(|md| !last::changed_since(&md, *size, changed_after)));
        if held {
            None
        } else {
            Some(format!("{p}: every video it lost to has gone or changed since the scan, so it was left where it is"))
        }
    }

    /// `stays`, asked on another thread: a `stat` of the file and of its
    /// holders, a round trip each on a share. Handed back with the answer.
    async fn looked_at(self, changed_after: Option<u64>) -> (Move, Option<String>) {
        let path = self.path.clone();
        gio::spawn_blocking(move || {
            let why = self.stays(changed_after);
            (self, why)
        })
        .await
        // A look that panicked is no proof the file may go.
        .unwrap_or_else(|_| {
            let why = format!("{}: it could not be looked at, so it was left where it is", path.display());
            (Move { path, scanned: 0, holders: Vec::new() }, Some(why))
        })
    }
}

/// A style class on or off, touched only when it changes.
fn set_class(w: &impl IsA<gtk::Widget>, class: &str, on: bool) {
    if w.has_css_class(class) == on {
        return;
    }
    if on {
        w.add_css_class(class);
    } else {
        w.remove_css_class(class);
    }
}

/// The video that shows what a group is, for the strip: the one the scan
/// suggests keeping, else its reference.
fn face(m: &Model, members: &[Member]) -> Option<usize> {
    members.iter().map(|x| x.file).find(|&id| m.scanned(id) == Action::Keep).or_else(|| members.first().map(|x| x.file))
}

/// Group numbers as a sentence names them: "4", "4 and 9", "4, 9 and 12",
/// and past twenty, the first twenty and how many more.
fn numbers(ns: &[usize]) -> String {
    const SHOWN: usize = 20;
    let words: Vec<String> = ns.iter().take(SHOWN).map(|n| n.to_string()).collect();
    if ns.len() > SHOWN {
        return format!("{} and {} more", words.join(", "), ns.len() - SHOWN);
    }
    match words.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
        None => String::new(),
    }
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
}

/// A video's figures as `vid-fp`'s listing prints them, in its order:
/// length, resolution, frame rate, codec, size, bitrate, bits a frame, and
/// the samples its fingerprint holds.
pub(crate) fn facts(f: &vid_fp::found::File) -> String {
    let length = if f.duration > 0.0 { utils::format_duration(f.duration) } else { "-".to_string() };
    [
        length,
        format!("{}x{}", f.width, f.height),
        utils::format_frame_rate(f.frame_rate),
        utils::format_codec(&f.codec),
        utils::format_size(f.size),
        utils::format_bitrate(f.bitrate),
        utils::format_quality(f.quality),
        format!("{} samples", f.samples),
    ]
    .join(" · ")
}

/// `menu`, as `view` has it: the header's button and the right click share
/// it, so both change with it. Each item's letter is also Alt and that letter
/// anywhere on the page (`Results::key`), which it shows beside it.
fn fill_menu(menu: &gio::Menu, view: View) {
    menu.remove_all();
    for part in MENU {
        let section = gio::Menu::new();
        for (act, label) in part.iter().filter_map(|&a| a.label(view).map(|l| (a, l))) {
            let item = gio::MenuItem::new(Some(label), Some(&format!("results.{}", act.action())));
            item.set_attribute_value("accel", Some(&format!("<Alt>{}", l::letter(label)).to_variant()));
            section.append_item(&item);
        }
        menu.append_section(None, &section);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_numbers_read_as_a_sentence() {
        assert_eq!(numbers(&[4]), "4");
        assert_eq!(numbers(&[4, 9]), "4 and 9");
        assert_eq!(numbers(&[4, 9, 12]), "4, 9 and 12");
        let many: Vec<usize> = (1..=25).collect();
        assert!(numbers(&many).ends_with("and 5 more"));
    }

    /// A row says what the command line's listing says of the file, in its
    /// words.
    #[test]
    fn a_rows_figures_are_the_listings() {
        let f = vid_fp::found::File {
            path: "/v/a.mkv".into(),
            size: 1_500_000,
            duration: 75.5,
            width: 1280,
            height: 720,
            codec: "h264".into(),
            frame_rate: 25.0,
            bitrate: 158_940,
            quality: 6357,
            samples: 31,
            verdicts: vec![vid_fp::found::Verdict { action: Action::Keep, lost_to: Vec::new() }; 4],
        };
        let line = facts(&f);
        for part in [utils::format_duration(75.5), "1280x720".into(), utils::format_codec("h264"), utils::format_size(1_500_000), "31 samples".into()] {
            assert!(line.contains(&part), "{line:?} lacks {part:?}");
        }
        let unknown = vid_fp::found::File { duration: 0.0, ..f };
        assert!(facts(&unknown).starts_with("- · "), "an unmeasured length reads as the report's dash");
    }

    /// A trashed file is left where it is when it changed, or when every
    /// file its deletion rests on has gone or changed; one holder as the scan
    /// found it is enough.
    #[test]
    fn a_deletion_is_looked_at_again_before_it_is_moved() {
        let dir = tempfile::tempdir().unwrap();
        let write = |n: &str| {
            let p = dir.path().join(n);
            std::fs::write(&p, b"12345").unwrap();
            p
        };
        let (v, a, b) = (write("v.mp4"), write("a.mp4"), write("b.mp4"));
        let later = Some(last::now_nanos() + 10_000_000_000);
        let m = |holders: Vec<(PathBuf, u64)>| Move { path: v.clone(), scanned: 5, holders };
        assert_eq!(m(vec![]).stays(later), None, "marked by hand, resting on nothing");
        assert_eq!(m(vec![(a.clone(), 5)]).stays(later), None);
        assert!(Move { path: v.clone(), scanned: 6, holders: vec![] }.stays(later).is_some(), "another length");
        assert!(m(vec![(a.clone(), 6)]).stays(later).is_some(), "its holder changed");
        assert_eq!(m(vec![(a.clone(), 6), (b.clone(), 5)]).stays(later), None, "b still holds it");
        std::fs::remove_file(&b).unwrap();
        assert!(m(vec![(b.clone(), 5)]).stays(later).is_some(), "its holder is gone");
    }
}
