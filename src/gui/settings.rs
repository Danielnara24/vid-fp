//! What the window has been asked to scan, and how: what of it is kept between
//! runs, and all of it turned into the vid-fp command line the scan runs.
//!
//! The window never passes the scan anything the command line could not: a
//! scan from here is the scan those flags would run from a shell, and the log
//! says which flags they were.

use vid_fp::utils::Priority;
use crate::options::{self as o, Opt, Value};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// What the window shows, and what of it is kept between runs: the paths,
/// and a value for each option of `options::ALL`.
///
/// **An option is kept only where it differs from its default**, so one never
/// touched follows the default when the default moves. An option set by hand
/// stays set until changed or put back with the Defaults button. The one-off
/// cache requests (clear, prune) are never kept: they are carried out once.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "Saved", into = "Saved")]
pub struct Settings {
    pub folders: Vec<PathBuf>,
    pub exclude: Vec<PathBuf>,
    pub report_path: String,
    /// Empty for vid-fp's own default.
    pub cache_path: String,
    pub log_path: String,
    /// One per option, in `options::ALL`'s order.
    values: Vec<Value>,
}

/// `Settings` as the file holds them: the paths, and under `options` each
/// option that differs from its default, by its key.
#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
struct Saved {
    #[serde(with = "paths")]
    folders: Vec<PathBuf>,
    report_path: String,
    #[serde(with = "paths")]
    exclude: Vec<PathBuf>,
    cache_path: String,
    log_path: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    options: BTreeMap<String, serde_json::Value>,
}

impl From<Settings> for Saved {
    fn from(s: Settings) -> Saved {
        let options = o::ALL
            .iter()
            .zip(&s.values)
            .filter(|(opt, v)| opt.kept && **v != opt.default_value())
            .map(|(opt, v)| (opt.key.to_string(), opt.as_json(v)))
            .collect();
        Saved { folders: s.folders, report_path: s.report_path, exclude: s.exclude, cache_path: s.cache_path, log_path: s.log_path, options }
    }
}

impl From<Saved> for Settings {
    /// An option missing, of another type, or not kept, is its default.
    fn from(f: Saved) -> Settings {
        let values = o::ALL
            .iter()
            .map(|opt| {
                let kept = f.options.get(opt.key).filter(|_| opt.kept).and_then(|j| opt.read_json(j));
                kept.unwrap_or_else(|| opt.default_value())
            })
            .collect();
        Settings { folders: f.folders, exclude: f.exclude, report_path: f.report_path, cache_path: f.cache_path, log_path: f.log_path, values }
    }
}

impl Default for Settings {
    fn default() -> Self {
        Saved::default().into()
    }
}

/// A list of paths as a file keeps it: each one as text, or as its bytes when
/// its name is not UTF-8, so that one such folder cannot make the whole save
/// fail. Shared with the kept scan's notes (`last.rs`).
pub mod paths {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::ffi::OsString;
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    use std::path::PathBuf;

    #[derive(Serialize, Deserialize)]
    #[serde(untagged)]
    enum Kept {
        Text(String),
        Bytes(Vec<u8>),
    }

    pub fn serialize<S: Serializer>(paths: &[PathBuf], s: S) -> Result<S::Ok, S::Error> {
        let kept: Vec<Kept> = paths
            .iter()
            .map(|p| match p.to_str() {
                Some(t) => Kept::Text(t.to_string()),
                None => Kept::Bytes(p.as_os_str().as_bytes().to_vec()),
            })
            .collect();
        kept.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<PathBuf>, D::Error> {
        let kept: Vec<Kept> = Vec::deserialize(d)?;
        Ok(kept
            .into_iter()
            .map(|k| match k {
                Kept::Text(t) => PathBuf::from(t),
                Kept::Bytes(b) => PathBuf::from(OsString::from_vec(b)),
            })
            .collect())
    }
}

/// `$XDG_CONFIG_HOME/vid-fp`, or `~/.config/vid-fp`.
pub fn config_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("vid-fp"))
}

fn file() -> Option<PathBuf> {
    Some(config_dir()?.join("gui.json"))
}

/// The ranking the results page shows, kept for the next window in a file
/// of its own: the scan page writes `gui.json` whole from its fields, and
/// would put back whatever ranking it read at start.
#[derive(Serialize, Deserialize)]
struct Ranking {
    rank_by: String,
}

fn ranking_file() -> Option<PathBuf> {
    Some(config_dir()?.join("results.json"))
}

/// The ranking chosen last time, or the command line's default.
pub fn load_ranking() -> Priority {
    let read = ranking_file().and_then(|f| std::fs::read(f).ok()).and_then(|b| serde_json::from_slice::<Ranking>(&b).ok());
    read.and_then(|r| Priority::ALL.into_iter().find(|p| p.name() == r.rank_by)).unwrap_or_default()
}

pub fn save_ranking(p: Priority) {
    let Some(f) = ranking_file() else { return };
    if let Some(dir) = f.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(b) = serde_json::to_vec(&Ranking { rank_by: p.name().to_string() }) {
        let tmp = f.with_extension(format!("json.{}", std::process::id()));
        if std::fs::write(&tmp, b).is_ok() && std::fs::rename(&tmp, &f).is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

impl Settings {
    /// What was used last time, or the defaults. A file that will not parse
    /// is the defaults too: it is a convenience, not a record.
    pub fn load() -> Settings {
        file().and_then(|f| std::fs::read(f).ok()).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    /// Kept for next time. Failing to is not worth interrupting anyone over.
    pub fn save(&self) {
        let Some(f) = file() else { return };
        if let Some(dir) = f.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(b) = serde_json::to_vec_pretty(self) {
            let tmp = f.with_extension(format!("json.{}", std::process::id()));
            if std::fs::write(&tmp, b).is_ok() && std::fs::rename(&tmp, &f).is_err() {
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }

    /// The same settings with every file path the scan will write made
    /// absolute: `~/` is the home folder, and so is the start of any other
    /// relative path, rather than whatever folder the window was started in.
    pub fn with_absolute_paths(mut self) -> Settings {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        for p in [&mut self.report_path, &mut self.log_path, &mut self.cache_path] {
            *p = absolute(p, home.as_deref());
        }
        self
    }

    /// The value of option `o`.
    pub fn get(&self, o: &Opt) -> &Value {
        &self.values[o.at()]
    }

    pub fn set(&mut self, o: &Opt, v: Value) {
        self.values[o.at()] = v;
    }

    /// Whether the check option `o` is on.
    pub fn on(&self, o: &Opt) -> bool {
        self.get(o).on()
    }

    /// Every option and file name back to its default, as the Defaults
    /// button puts them; the folders to scan and to leave out are kept.
    pub fn with_defaults(self) -> Settings {
        Settings { folders: self.folders, exclude: self.exclude, ..Settings::default() }
    }

    /// The vid-fp command line these settings are, program name first.
    ///
    /// Every option the window shows is passed, default or not, so the log
    /// records the whole of what the scan was asked. The folders go last,
    /// after `--`, so that no folder name can be read as a flag.
    pub fn argv(&self) -> Vec<OsString> {
        let mut a: Vec<OsString> = vec!["vid-fp".into()];
        for (opt, v) in o::ALL.iter().zip(&self.values) {
            a.extend(opt.command_line(v).into_iter().map(OsString::from));
        }
        let mut flag = |f: &str, v: &str| {
            a.push(f.into());
            a.push(v.into());
        };
        let (report, cache, log) = (self.report_path.trim(), self.cache_path.trim(), self.log_path.trim());
        if self.on(&o::REPORT) && !report.is_empty() {
            flag("--output", report);
            if let Some(f) = o::REPORT_FORMAT.choice_flag(self.get(&o::REPORT_FORMAT)) {
                flag("--format", f);
            }
        }
        if !cache.is_empty() {
            flag("--cache", cache);
        }
        if self.on(&o::LOG) && !log.is_empty() {
            flag("--log-file", log);
        }
        for e in &self.exclude {
            a.push("--exclude".into());
            a.push(e.into());
        }
        if self.on(&o::CLEAR_CACHE) {
            a.push("--clear-cache".into());
        }
        if self.on(&o::PRUNE_CACHE) {
            a.push("--prune-cache".into());
        }
        a.push("--".into());
        a.extend(self.folders.iter().map(|f| f.into()));
        a
    }
}

/// `p` made absolute against `home`, as `with_absolute_paths` describes. An
/// empty path is left empty: it means "not set".
fn absolute(p: &str, home: Option<&Path>) -> String {
    let t = p.trim();
    let Some(home) = home.filter(|h| h.is_absolute()) else { return t.to_string() };
    if t.is_empty() || Path::new(t).is_absolute() {
        return t.to_string();
    }
    let rest = if t == "~" { "" } else { t.strip_prefix("~/").unwrap_or(t) };
    home.join(rest).display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::Kind;

    /// The values `argv` hands vid-fp for clap id `id`, as its parser reads
    /// them; `None` where it is not given.
    fn given(argv: &[OsString], id: &str) -> Option<Vec<String>> {
        let m = vid_fp::command().try_get_matches_from(argv).unwrap_or_else(|e| panic!("{e}"));
        if m.value_source(id) != Some(clap::parser::ValueSource::CommandLine) {
            return None;
        }
        Some(m.get_raw(id).map_or_else(Vec::new, |vs| vs.map(|v| v.to_string_lossy().into_owned()).collect()))
    }

    /// A folder whose name is not UTF-8 is remembered, and does not stop the
    /// rest of the settings from being remembered with it.
    #[test]
    fn a_folder_that_is_not_utf8_is_saved_and_read_back() {
        use std::os::unix::ffi::OsStringExt;
        let odd = PathBuf::from(std::ffi::OsString::from_vec(b"/p/b\xff".to_vec()));
        let s = Settings { folders: vec!["/p/a".into(), odd.clone()], exclude: vec![odd.clone()], log_path: "/l".into(), ..Default::default() };
        let text = serde_json::to_string(&s).expect("saved");
        let back: Settings = serde_json::from_str(&text).unwrap();
        assert_eq!(back.folders, vec![PathBuf::from("/p/a"), odd.clone()]);
        assert_eq!(back.exclude, vec![odd]);
        assert_eq!(back.log_path, "/l");
    }

    /// An option set by hand is kept; one left at its default is not, so a
    /// default the command line changes reaches it.
    #[test]
    fn only_options_set_by_hand_are_kept() {
        let d = Settings::default();
        let mut s = Settings { folders: vec!["/x".into()], ..Default::default() };
        s.set(&o::TOLERANCE, Value::Number(8.0));
        s.set(&o::LOG, Value::On(true));
        s.set(&o::RECURSIVE, Value::On(false));
        s.set(&o::REPORT_FORMAT, Value::Text("Csv".into()));
        s.set(&o::MATCH_PERCENT, Value::Number(12.5));
        s.set(&o::CLEAR_CACHE, Value::On(true));
        s.set(&o::PRUNE_CACHE, Value::On(true));
        let v: serde_json::Value = serde_json::to_value(&s).unwrap();
        let keys: Vec<&str> = v["options"].as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(keys, ["hamming_distance", "log", "match_percent", "recursive", "report_format"]);
        let back: Settings = serde_json::from_value(v).unwrap();
        for opt in [&o::TOLERANCE, &o::LOG, &o::RECURSIVE, &o::REPORT_FORMAT, &o::MATCH_PERCENT] {
            assert_eq!(back.get(opt), s.get(opt), "{}", opt.key);
        }
        for opt in [&o::THREADS, &o::EXTENSIONS] {
            assert_eq!(back.get(opt), d.get(opt), "{}", opt.key);
        }
        assert!(!back.on(&o::CLEAR_CACHE) && !back.on(&o::PRUNE_CACHE), "one-off requests are not repeated");

        // Nothing set by hand: nothing but the paths is written.
        let v: serde_json::Value = serde_json::to_value(Settings { folders: vec!["/x".into()], ..Default::default() }).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["cache_path", "exclude", "folders", "log_path", "report_path"]);

        // A value of another type, or a choice not offered, is the default;
        // the rest of the file still reads.
        let odd = r#"{"folders": ["/x"], "options": {"report_format": "Pdf", "threads": 3, "hamming_distance": "wide"}}"#;
        let s: Settings = serde_json::from_str(odd).unwrap();
        assert_eq!(s.folders, vec![PathBuf::from("/x")]);
        assert_eq!(s.get(&o::REPORT_FORMAT), d.get(&o::REPORT_FORMAT));
        assert_eq!(s.get(&o::TOLERANCE), d.get(&o::TOLERANCE));
        assert_eq!(s.get(&o::THREADS), &Value::Number(3.0));
    }

    #[test]
    fn a_relative_path_is_made_absolute_against_home() {
        let home = Some(Path::new("/home/u"));
        assert_eq!(absolute("results.txt", home), "/home/u/results.txt");
        assert_eq!(absolute(" out/r.csv ", home), "/home/u/out/r.csv");
        assert_eq!(absolute("~/r.json", home), "/home/u/r.json");
        assert_eq!(absolute("~", home), "/home/u/");
        assert_eq!(absolute("/tmp/r.txt", home), "/tmp/r.txt");
        assert_eq!(absolute("", home), "");
        assert_eq!(absolute("r.txt", None), "r.txt", "with no home there is nothing better to do");
    }

    /// Whatever the window can be set to, the command line it makes is one
    /// vid-fp accepts: the scan must never fail on its own arguments.
    #[test]
    fn every_setting_makes_a_command_line_vid_fp_accepts() {
        let mut s = Settings { folders: vec!["/tmp".into(), "/-odd name".into()], ..Default::default() };
        assert_eq!(vid_fp::check_args(s.argv()), Ok(()));
        for (opt, v) in [(&o::CLEAR_CACHE, true), (&o::PRUNE_CACHE, true), (&o::REPORT, true), (&o::LOG, true), (&o::SYMLINKS, true), (&o::RECURSIVE, false)] {
            s.set(opt, Value::On(v));
        }
        s.cache_path = "/tmp/x".into();
        s.report_path = "/tmp/r.csv".into();
        s.set(&o::REPORT_FORMAT, Value::Text("Json".into()));
        s.log_path = "/tmp/l.txt".into();
        s.exclude = vec!["/tmp/a".into()];
        s.set(&o::EXTENSIONS, Value::Text("*".into()));
        assert_eq!(vid_fp::check_args(s.argv()), Ok(()));
        // Every number at both ends of its spin button.
        for opt in o::ALL.iter() {
            if let Kind::Number { min, max, .. } = opt.kind {
                for v in [min, max()] {
                    let mut s = Settings { folders: vec!["/tmp".into()], ..Default::default() };
                    s.set(opt, opt.number(v));
                    assert_eq!(vid_fp::check_args(s.argv()), Ok(()), "{} at {v}", opt.key);
                }
            }
        }
        // A folder that looks like a flag is a folder.
        let a = s.argv();
        assert_eq!(given(&a, "include"), Some(vec!["/tmp".to_string(), "/-odd name".to_string()]));
    }

    /// Every option with a flag reaches vid-fp as what it was set to, at its
    /// default and away from it, and is read back by the command line as
    /// that value.
    #[test]
    fn every_option_reaches_the_command_line_as_set() {
        for opt in o::ALL.iter().filter(|opt| opt.arg.is_some()) {
            let id = opt.arg.unwrap();
            let away = match (&opt.kind, opt.default_value()) {
                (Kind::Check, Value::On(b)) => Value::On(!b),
                (Kind::Number { max, .. }, Value::Number(n)) => opt.number(if n + 2.0 <= max() { n + 2.0 } else { n / 2.0 }),
                (Kind::Text, _) => Value::Text("mkv,mp4".into()),
                (_, v) => panic!("{}: nothing away from {v:?}", opt.key),
            };
            for v in [opt.default_value(), away] {
                let mut s = Settings { folders: vec!["/tmp".into()], ..Default::default() };
                s.set(opt, v.clone());
                let g = given(&s.argv(), id);
                let back = match (&opt.kind, g) {
                    (Kind::Check, g) => Value::On(g.is_some()),
                    (Kind::Number { .. }, Some(g)) => opt.parse(&g[0]).unwrap(),
                    (Kind::Text, Some(g)) => Value::Text(g.join(",")),
                    (_, None) => panic!("{} not passed", opt.key),
                    _ => unreachable!(),
                };
                assert_eq!(back, v, "{}", opt.key);
            }
        }
    }

    /// The paths the window's own switches stand for are passed with them,
    /// and only when they are on.
    #[test]
    fn the_switches_pass_their_paths() {
        let mut s = Settings { folders: vec!["/v".into()], report_path: "/r.json".into(), log_path: "/l.txt".into(), cache_path: "/c.redb".into(), ..Default::default() };
        let a = s.argv();
        assert_eq!(given(&a, "output"), None, "no report unless asked");
        assert_eq!(given(&a, "log_file"), None);
        assert_eq!(given(&a, "cache"), Some(vec!["/c.redb".to_string()]), "a cache file named is always the one used");
        s.set(&o::REPORT, Value::On(true));
        s.set(&o::LOG, Value::On(true));
        s.set(&o::REPORT_FORMAT, Value::Text("Csv".into()));
        s.exclude = vec!["/v/keep".into(), "/v/old".into()];
        let a = s.argv();
        assert_eq!(given(&a, "output"), Some(vec!["/r.json".to_string()]));
        assert_eq!(given(&a, "format"), Some(vec!["csv".to_string()]));
        assert_eq!(given(&a, "log_file"), Some(vec!["/l.txt".to_string()]));
        assert_eq!(given(&a, "exclude"), Some(vec!["/v/keep".to_string(), "/v/old".to_string()]));
        assert_eq!(vid_fp::check_args(a), Ok(()));
    }
}
