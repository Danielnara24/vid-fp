//! A scrolling list that lays out only the rows on screen.
//!
//! `gtk::ListView` keeps about two hundred row widgets alive whatever fits on
//! screen (GTK 4.14), and replacing its items destroys and rebuilds every one
//! of them: a folder of 265 videos built, bound and laid out 205 rows to show
//! six, 0.6 to 1.3 seconds on every change of folder. A `Rows` hands its
//! allocation to whoever owns it (`set_layout`), which places a handful of
//! row widgets at the rows that are visible and moves them as it scrolls, so
//! showing a folder costs the rows on screen and nothing for the rest. Every
//! row is one height, which is what makes that a division rather than a
//! measurement.
//!
//! It scrolls by its own adjustment, which a `gtk::Scrollbar` beside it shows
//! (`adjustment`): the mouse wheel and a touchpad move it here, and the owner
//! moves it to bring a row into view.

use gtk::prelude::*;
use gtk::subclass::prelude::*;
use gtk::{glib, graphene};

/// Where the owner's rows go: called with the width and height to fill, and
/// the scroll offset in pixels.
type Layout = Box<dyn Fn(&Rows, i32, i32, f64)>;

mod imp {
    use super::*;
    use std::cell::RefCell;

    #[derive(Default)]
    pub struct Rows {
        pub layout: RefCell<Option<Layout>>,
        pub adjustment: RefCell<Option<gtk::Adjustment>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for Rows {
        const NAME: &'static str = "VidFpRows";
        type Type = super::Rows;
        type ParentType = gtk::Widget;

        fn class_init(klass: &mut Self::Class) {
            klass.set_css_name("rows");
            klass.set_accessible_role(gtk::AccessibleRole::List);
        }
    }

    impl ObjectImpl for Rows {
        fn dispose(&self) {
            while let Some(child) = self.obj().first_child() {
                child.unparent();
            }
        }
    }

    impl WidgetImpl for Rows {
        /// Nothing it holds decides its size: it takes what it is given.
        fn measure(&self, _: gtk::Orientation, _: i32) -> (i32, i32, i32, i32) {
            (0, 0, -1, -1)
        }

        fn size_allocate(&self, width: i32, height: i32, _baseline: i32) {
            let offset = self.adjustment.borrow().as_ref().map_or(0.0, |a| a.value());
            if let Some(layout) = self.layout.borrow().as_ref() {
                layout(&self.obj(), width, height, offset);
            }
        }

        /// Its children, clipped: a row half scrolled out is cut at the edge.
        fn snapshot(&self, snapshot: &gtk::Snapshot) {
            let obj = self.obj();
            snapshot.push_clip(&graphene::Rect::new(0.0, 0.0, obj.width() as f32, obj.height() as f32));
            let mut child = obj.first_child();
            while let Some(c) = child {
                obj.snapshot_child(&c, snapshot);
                child = c.next_sibling();
            }
            snapshot.pop();
        }
    }
}

glib::wrapper! {
    pub struct Rows(ObjectSubclass<imp::Rows>) @extends gtk::Widget, @implements gtk::Accessible, gtk::Buildable, gtk::ConstraintTarget;
}

impl Default for Rows {
    fn default() -> Self {
        Self::new()
    }
}

impl Rows {
    pub fn new() -> Rows {
        let rows: Rows = glib::Object::new();
        let adj = gtk::Adjustment::new(0.0, 0.0, 0.0, 1.0, 1.0, 0.0);
        adj.connect_value_changed({
            let rows = rows.downgrade();
            move |_| {
                if let Some(r) = rows.upgrade() {
                    r.queue_allocate();
                }
            }
        });
        *rows.imp().adjustment.borrow_mut() = Some(adj.clone());
        // The wheel moves it as a scrolled window would, a step of the page
        // to the power two thirds; a touchpad moves it by its own pixels.
        let scroll = gtk::EventControllerScroll::new(gtk::EventControllerScrollFlags::VERTICAL);
        scroll.connect_scroll(move |c, _, dy| {
            let step = match c.unit() {
                gtk::gdk::ScrollUnit::Wheel => adj.page_size().powf(2.0 / 3.0),
                _ => 1.0,
            };
            adj.set_value((adj.value() + dy * step).clamp(adj.lower(), (adj.upper() - adj.page_size()).max(adj.lower())));
            glib::Propagation::Stop
        });
        rows.add_controller(scroll);
        rows
    }

    pub fn adjustment(&self) -> gtk::Adjustment {
        self.imp().adjustment.borrow().clone().expect("made in new")
    }

    /// Who places the rows, on every allocation and every scroll.
    pub fn set_layout(&self, f: impl Fn(&Rows, i32, i32, f64) + 'static) {
        *self.imp().layout.borrow_mut() = Some(Box::new(f));
    }
}
