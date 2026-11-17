# vid-fp: the window (`vid-fp-gui`, feature `gui`)

Read with the root `CLAUDE.md`. Built after img-fp's window (`../img-fp/src/gui`),
and kept close to it on purpose: the two apps should behave alike. Where this
one differs, it is said below.

- The scan is a **child process** (`vid-fp-gui --worker RESULT <argv>`, which
  runs `vid_fp::worker_main` and saves its `found::Found` to RESULT), so
  Cancel is SIGINT. Spawned from the GTK main thread (PDEATHSIG is per-thread)
  by our own fork + `execv` of `/proc/self/exe` (`scan::spawn`;
  `std::process::Command` pulls in GLIBC_2.39 weak symbols, `current_exe()`
  breaks after a package upgrade). Cancel asks again after 5 s (a second
  SIGINT ends vid-fp at once) and kills after 10 s.
- **The suggestion is the scan's, never the window's.** Each file's
  KEEP/DELETE/REVIEW comes in the result, from `export::verdicts`, the code the
  report labels with. `--priority` changes only the labels, so the worker
  labels all four rankings (`found::Verdict`, one per `Priority::ALL`) and the
  results page's *Rank by* dropdown picks one without a rescan; the scan is
  never passed `-k` (`options.rs`, `NOT_SHOWN`). The choice is kept in
  `$XDG_CONFIG_HOME/vid-fp/results.json`, not `gui.json`, which the scan page
  writes whole. `tests/window_matches_cli.rs` holds every ranking equal to
  `vid-fp -k <it>`. Do not compute a label in the window.
- What the window adds is the files afterwards (`model.rs`): a DELETE rests on
  `File::lost_to`; when every one of those has gone or changed since the scan
  it reads REVIEW, as does a DELETE of a file changed itself. Changed = another
  length, or ctime from the scan's start on (`last::changed_since`). Gone files
  leave the page (`Model::drop_gone`). Looked for off the GTK thread before any
  action (`Results::checked`), and again per file just before it is trashed
  (`Move::stays`: one holder still there unchanged is enough).
- The Trash dialog names groups with every video marked, and counts marked
  DELETEs with no unmarked holder reachable through `lost_to`
  (`Model::held_after`): the scan's DELETEs chain (a lost to b, b to c), and a
  chain marked down to its kept head is not an orphan. A file gio cannot trash
  is offered for permanent deletion (`delete_instead`).
- **The videos' list is `rows.rs`, not a `ListView`, and must stay that
  way.** GTK 4.14's `ListView` keeps ~200 row widgets alive whatever fits on
  screen, and replacing its items rebuilds all of them: a 265-video folder
  took 0.6-1.3 s per switch. `Rows` places only the rows on screen (row `i`
  in pool slot `i % pool.len()`, one fixed height measured once per width) and
  scrolls by its own adjustment, so a switch costs the visible rows: 25-86 ms
  at 10,000 videos. Selection is the `selected` class on a pool row, double
  click and focus are the row's own gesture, and only visible rows request
  thumbnails. The strip and the folder tree are still `ListView`s; their rows
  are small and their items are not replaced on every click.
- **Results are a list, not a mosaic** (user's decision): a list of rows,
  each a check, a thumbnail (192x108, the whole frame), name and folder, the
  CLI listing's figures in its words (`results::facts`, `vid_fp::utils::format_*`),
  the strongest match in the group, and the suggestion pill. **No player**:
  a thumbnail only; Enter / double click / *Open* plays it in the desktop's
  player (`gtk::FileLauncher`). No large view, no zoom.
- Thumbnails (`thumbs.rs`, `vid_fp::thumbnail::still`): keyframes only, a tenth
  of the way in, then a third and half way if the frame is flat (black, fade),
  keeping the busiest; pixel aspect applied. The freedesktop thumbnail cache is
  used when current (`Thumb::MTime`, and `Thumb::Size` when written), read with
  gdk-pixbuf. Premultiplied BGRA at drawn size; two decoders for rows, one for
  the strip; held to bytes (`Budget`). Rotation metadata (phone video) is not
  applied.
- **Compare (`matched.rs`) shows the scan's evidence, not a player.** The
  user chose matched frames over side-by-side playback. Each `Member` carries
  `compared`: against its best link (first) and the group's reference, the
  matched frame pairs (thinned to `found::MOST_PAIRS`), and each side's
  matched stretches, made by `compare::evidence`. The window decodes each
  frame with `thumbnail::frame_at` through `Thumbs` (`Size::at(ms)`), draws a
  chart (`Chart`: bar, link band, bar, on ONE time scale, the shorter slid
  to the offset most pairs agree on, `alignment`, so a real copy draws
  vertical lines and a look-alike draws slanted ones; user's choice; a zoom
  toggle, or Z, narrows the same scale to the matched footage, `View`.
  "One scale from 0" and "each bar full width" were tried and dropped: the
  first keeps a clip a sliver, the second makes every line converge on a
  sliver of the long video), and
  shows a pair full size (`Shown::Picture`). Click selects (row, ticks and
  line red), double click / Enter opens full size. A tick on A is one pair
  (pairs are one per A sample); a tick on B can be several, and clicking it
  again steps through them (`Chart::choose`). A frame
  is found by the scan's own clock: the first keyframe is the anchor, and in a
  container whose clock may restart (TS/PS) a keyframe index replaying that
  clock is read once per file and the frame found by byte position, since one
  timestamp can occur in each `cat`ted segment.
- Strip of groups and the tree view by folder (`folders.rs`) as in img-fp; a
  group's face is its KEEP file. Marks are on the file (by number in
  `Found::files`), so a mark shows in every group the file is in.
- Keys live in `commands.rs` (`WINDOW`, `SCAN`, `RESULTS`): a `Row` per line
  of the F1 list and the chords it describes; the menu is `MENU`. **Adding a
  key**: a binding in its row and an arm in `Results::run`; the tests fail
  until a new command is bound and described. Up/Down move through the list
  (no Left/Right: a list has none; in the tree they open and close folders).
- **Never call `parent()` on a list row's widget inside a factory's bind or
  setup.** On GTK 4.14 the list's own row widget is not parented yet then, so
  it is still floating; gtk-rs's `from_glib_none` ref-sinks it and the handle's
  drop frees it, and the list then segfaults on a dangling pointer
  (`gtk_widget_insert_after: assertion 'GTK_IS_WIDGET (widget)'`). That is why
  the marked style is on the row's own box. That bit the old videos' list;
  the strip and the tree are `ListView`s still. Also do not `set_parent`
  anything onto a `ListView`; the context menu hangs off the box around it.
- Mnemonics live in `labels.rs` with a clash test; add labels there.
- Options are one table, `options.rs` (`Opt`: key in `gui.json`, clap id,
  `Kind`, label, tip, hint; `ALL`), defaults read off vid-fp's own parser
  (`options::flag`). **Adding an option**: an `Opt` in `ALL` and a place on the
  page for its widget; a flag the window does not show goes in `BY_HAND` or
  `NOT_SHOWN` (the test says which is missing). There is no "use a cache"
  switch: vid-fp always caches.
- Paths persist (`$XDG_CONFIG_HOME/vid-fp/gui.json`), options only where set
  away from their default; clear/prune cache never kept; `recursive` on by
  default. The last scan and its marks are kept in
  `$XDG_CACHE_HOME/vid-fp/last-scan/` (`last.rs`).
- Edition 2021: no let-chains (img-fp's code uses them; rewrite as nested
  `if let`). The window needs Rust 1.92 for gtk4 0.11 while the crate's
  `rust-version` is the CLI's, hence `#![allow(clippy::incompatible_msrv)]`
  in `main.rs`.
- Testing: `cargo test --features gui`. By hand, Xvfb + `xdotool windowfocus
  --sync` + `import -window root`; point `XDG_DATA_HOME` at scratch before
  testing Trash, and `--cache` at scratch so a dynamic build's fingerprints
  never land in the static build's cache.
