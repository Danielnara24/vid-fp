//! A picture whose size does not depend on what it shows.
//!
//! `gtk::Picture` asks for its texture's own size, so a texture arriving
//! queues a resize, and a resize climbs to the window: the whole page is
//! measured and laid out again, once per thumbnail as they arrive from the
//! decoders. A `Still` holds a fixed size, draws its texture into it, and a
//! new texture only redraws it.
//!
//! Two ways to draw: `cover` fills the place and cuts what overhangs (the
//! strip of groups, where every picture is the same shape), and otherwise the
//! whole frame is shown, centred, with the rest left empty (a group's videos,
//! where a 4:3 copy beside a 16:9 one is something to see).

use gtk::prelude::*;
use gtk::subclass::prelude::*;
use gtk::{gdk, glib, graphene, gsk};

mod imp {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[derive(Default)]
    pub struct Still {
        pub texture: RefCell<Option<gdk::Texture>>,
        /// What it asks for, both ways.
        pub size: Cell<(i32, i32)>,
        /// Fill the place rather than fit inside it.
        pub cover: Cell<bool>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for Still {
        const NAME: &'static str = "VidFpStill";
        type Type = super::Still;
        type ParentType = gtk::Widget;

        fn class_init(klass: &mut Self::Class) {
            klass.set_css_name("still");
            klass.set_accessible_role(gtk::AccessibleRole::Img);
        }
    }

    impl ObjectImpl for Still {}

    impl WidgetImpl for Still {
        fn measure(&self, orientation: gtk::Orientation, _for_size: i32) -> (i32, i32, i32, i32) {
            let (w, h) = self.size.get();
            let n = if orientation == gtk::Orientation::Horizontal { w } else { h };
            (n, n, -1, -1)
        }

        fn snapshot(&self, snapshot: &gtk::Snapshot) {
            let Some(t) = self.texture.borrow().clone() else { return };
            let w = self.obj().width() as f32;
            let h = self.obj().height() as f32;
            let (tw, th) = (t.width() as f32, t.height() as f32);
            if w <= 0.0 || h <= 0.0 || tw <= 0.0 || th <= 0.0 {
                return;
            }
            let s = if self.cover.get() { (w / tw).max(h / th) } else { (w / tw).min(h / th) };
            let (dw, dh) = (tw * s, th * s);
            snapshot.push_clip(&graphene::Rect::new(0.0, 0.0, w, h));
            snapshot.append_scaled_texture(&t, gsk::ScalingFilter::Linear, &graphene::Rect::new((w - dw) / 2.0, (h - dh) / 2.0, dw, dh));
            snapshot.pop();
        }
    }
}

glib::wrapper! {
    pub struct Still(ObjectSubclass<imp::Still>) @extends gtk::Widget, @implements gtk::Accessible, gtk::Buildable, gtk::ConstraintTarget;
}

impl Still {
    /// `width` x `height` always; with `cover`, filled.
    pub fn new(width: i32, height: i32, cover: bool) -> Still {
        let s: Still = glib::Object::new();
        s.imp().size.set((width, height));
        s.imp().cover.set(cover);
        s
    }

    pub fn set_texture(&self, t: Option<&gdk::Texture>) {
        *self.imp().texture.borrow_mut() = t.cloned();
        self.queue_draw();
    }

    /// What a screen reader says for it.
    pub fn set_label(&self, text: &str) {
        self.update_property(&[gtk::accessible::Property::Label(text)]);
    }
}
