//! Every mnemonic in the window, in one place.
//!
//! A mnemonic is the underlined letter Alt reaches, and two visible controls
//! sharing one make Alt cycle between them instead of going where it says. So
//! the labels live here, grouped by what is on screen together, and the test at
//! the bottom checks each group for clashes: the setup page's own controls with
//! either tab, the results page, and the dialogs.

// The setup page, whichever tab is showing. Scan and Cancel share a letter
// because only one of them is ever visible.
pub const SCAN: &str = "_Scan";
pub const CANCEL: &str = "Cancel _scan";
pub const TAB_GENERAL: &str = "_General";
pub const TAB_ADVANCED: &str = "Ad_vanced";
pub const SCAN_LOG: &str = "Scan _log";
pub const BACK_TO_RESULTS: &str = "_Back to results";
pub const DEFAULTS: &str = "Restore _defaults";

// The General tab.
pub const FOLDERS: &str = "_Folders to scan";
pub const ADD_FOLDER: &str = "_Add folder…";
pub const REMOVE_FOLDER: &str = "_Remove";
pub const RECURSIVE: &str = "Look _in subfolders";
pub const EXCLUDE: &str = "_Exclude";
pub const ADD_EXCLUDE: &str = "Add excl_uded folder…";
pub const REMOVE_EXCLUDE: &str = "Re_move";
pub const TOLERANCE: &str = "Frame _tolerance";
pub const MATCH_PERCENT: &str = "Min matched _percent";
pub const MIN_DURATION: &str = "Min s_hared length";

// The Advanced tab.
pub const THREADS: &str = "_Threads";
pub const KF_INTERVAL: &str = "_Keyframe interval";
pub const MIN_KEYFRAMES: &str = "Mi_n keyframes";
pub const EXTENSIONS: &str = "Extensi_ons";
pub const SYMLINKS: &str = "Follow s_ymlinks";
pub const REPORT: &str = "Save a _report to";
pub const REPORT_CHOOSE: &str = "Choos_e…";
pub const REPORT_FORMAT: &str = "For_mat";
pub const CACHE_FILE: &str = "_Cache file";
pub const CACHE_CHOOSE: &str = "C_hoose…";
pub const CLEAR_CACHE: &str = "Em_pty the cache before scanning";
pub const PRUNE_CACHE: &str = "Prune what this scan does not _use";
pub const LOG_FILE: &str = "Write a log _file";
pub const LOG_CHOOSE: &str = "Bro_wse…";

// The results page.
pub const GROUPS: &str = "_Groups";
/// The tree view's heading, in GROUPS' place.
pub const FOLDERS_TREE: &str = "_Browse folders";
// The results page's menu (`commands::MENU`). Its letters are Alt shortcuts
// on the whole page, so they are checked with the page's own.
pub const MARK_OTHERS: &str = "Mark all _except this";
pub const UNMARK_GROUP: &str = "_Unmark group";
pub const UNMARK_ALL: &str = "Unmark _all groups";
// In the tree view, in place of the two above, with their letters.
pub const UNMARK_FOLDER: &str = "_Unmark folder";
pub const UNMARK_ALL_FOLDERS: &str = "Unmark _all";
pub const MARK_FOLDER: &str = "_Mark current folder";
// One or the other, with one letter.
pub const TO_TREE: &str = "S_witch to tree view";
pub const TO_GROUPS: &str = "S_witch to group view";
pub const OPEN: &str = "_Open";
pub const COMPARE: &str = "_Compare matched frames";
pub const SHOW_FOLDER: &str = "Show in _folder";
pub const MARK_SUGGESTED: &str = "Mark suggested _deletions";
pub const RANK_BY: &str = "_Rank by";
pub const TRASH: &str = "Move marked to _Trash…";
/// The Trash button once something is marked, with TRASH's letter.
pub fn trash_n(n: usize) -> String {
    format!("Move {n} to _Trash…")
}
pub const TO_SETTINGS: &str = "Scan _settings";
pub const RESULTS_LOG: &str = "Scan _log";

// The Trash confirmation.
pub const CONFIRM_CANCEL: &str = "_Cancel";
pub const CONFIRM_TRASH: &str = "_Move to Trash";
// When the Trash cannot take them.
pub const CONFIRM_KEEP: &str = "_Leave in place";
pub const CONFIRM_DELETE: &str = "_Delete permanently";

/// The letter a label's mnemonic is on, in lower case.
pub fn letter(label: &str) -> char {
    label.split_once('_').and_then(|(_, rest)| rest.chars().next()).map_or('\0', |c| c.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::MENU;
    use crate::results::View;

    fn checked(label: &str) -> char {
        assert_eq!(label.matches('_').count(), 1, "{label:?} should have exactly one mnemonic");
        letter(label)
    }

    fn distinct(scope: &str, labels: &[&str]) {
        let mut seen: Vec<(char, &str)> = Vec::new();
        for l in labels {
            let c = checked(l);
            if let Some((_, other)) = seen.iter().find(|(d, _)| *d == c) {
                panic!("{scope}: {l:?} and {other:?} both use Alt+{c}");
            }
            seen.push((c, l));
        }
    }

    #[test]
    fn no_two_visible_controls_share_a_mnemonic() {
        // CANCEL is left out because it replaces SCAN.
        let page = [SCAN, TAB_GENERAL, TAB_ADVANCED, SCAN_LOG, BACK_TO_RESULTS, DEFAULTS];
        let general = [FOLDERS, ADD_FOLDER, REMOVE_FOLDER, RECURSIVE, EXCLUDE, ADD_EXCLUDE, REMOVE_EXCLUDE, TOLERANCE, MATCH_PERCENT, MIN_DURATION];
        let advanced = [
            THREADS, KF_INTERVAL, MIN_KEYFRAMES, EXTENSIONS, SYMLINKS, REPORT, REPORT_CHOOSE, REPORT_FORMAT, CACHE_FILE, CACHE_CHOOSE, CLEAR_CACHE,
            PRUNE_CACHE, LOG_FILE, LOG_CHOOSE,
        ];
        distinct("general tab", &[&page[..], &general[..]].concat());
        distinct("advanced tab", &[&page[..], &advanced[..]].concat());
        assert_eq!(letter(SCAN), letter(CANCEL));
        // The page's own, with each view's heading and menu
        // (`commands::MENU`), whose letters are Alt on the whole page.
        for view in [View::Groups, View::Folders] {
            let menu = MENU.iter().flat_map(|s| s.iter()).filter_map(|c| c.label(view));
            let page = [view.heading(), RANK_BY, MARK_SUGGESTED, TRASH, TO_SETTINGS, RESULTS_LOG].into_iter();
            distinct(&format!("results, {view:?}"), &page.chain(menu).collect::<Vec<_>>());
        }
        assert_eq!(letter(TRASH), letter(&trash_n(2)));
        distinct("confirmation", &[CONFIRM_CANCEL, CONFIRM_TRASH]);
        distinct("no trash", &[CONFIRM_KEEP, CONFIRM_DELETE]);
    }
}
