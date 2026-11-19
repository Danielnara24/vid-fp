//! The options the window shows, each said once: which of vid-fp's flags it
//! is, how it is shown, and in what words.
//!
//! `Settings` holds a value per option and `Setup` a widget per option, and
//! both go through this table rather than a field per option. An option is an
//! entry here (and a widget placed on the page), its default is the command
//! line's own (`flag`, read off vid-fp's own parser), and a test asks that
//! every flag of vid-fp is either one of these, passed by hand, or not the
//! window's to pass.

use crate::labels as l;

/// How an option is shown, and so what its value is.
pub enum Kind {
    /// A check button. Its flag is given when it is on.
    Check,
    /// A spin button, its value rounded to `digits` decimals. `max` is asked
    /// when the button is made: a limit can be this machine's.
    Number { min: f64, max: fn() -> f64, step: f64, digits: u32 },
    /// An entry, its text as the flag takes it; left out when empty.
    Text,
    /// A drop-down: each choice as the settings file keeps it, the flag's
    /// value (`None`: no flag), and its label.
    Choice(&'static [(&'static str, Option<&'static str>, &'static str)]),
}

/// One option of the window.
pub struct Opt {
    /// Its name in the settings file.
    pub key: &'static str,
    /// The command-line option it is, by clap id: passed as its flag and its
    /// value, defaulting to the command line's default. `None` for the
    /// window's own switches, which `Settings::argv` passes by hand.
    pub arg: Option<&'static str>,
    /// The window's default, as the command line would write it, where it
    /// is not the command line's or there is none.
    pub default: Option<&'static str>,
    pub kind: Kind,
    pub label: &'static str,
    /// Over the option and its label.
    pub tip: Option<&'static str>,
    /// Beside or under it, on the page.
    pub hint: Option<&'static str>,
    /// Kept between runs. The one-off cache requests are carried out once.
    pub kept: bool,
}

/// An option's value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    On(bool),
    Number(f64),
    /// Text, or a choice as the settings file keeps it.
    Text(String),
}

impl Value {
    pub fn on(&self) -> bool {
        matches!(self, Value::On(true))
    }

    pub fn number(&self) -> f64 {
        match self {
            Value::Number(v) => *v,
            _ => 0.0,
        }
    }

    pub fn text(&self) -> &str {
        match self {
            Value::Text(t) => t,
            _ => "",
        }
    }
}

/// One of vid-fp's flags, as its parser has it.
pub struct Flag {
    /// `--` and its long name.
    pub long: String,
    /// Its default as the command line would write it, values joined by
    /// commas; `None` when it has none.
    pub default: Option<String>,
}

/// vid-fp's option `id` (clap's id, the field's name), if there is one.
pub fn flag(id: &str) -> Option<Flag> {
    let cmd = vid_fp::command();
    let a = cmd.get_arguments().find(|a| a.get_id() == id)?;
    let long = format!("--{}", a.get_long()?);
    let d: Vec<String> = a.get_default_values().iter().map(|v| v.to_string_lossy().into_owned()).collect();
    Some(Flag { long, default: (!d.is_empty()).then(|| d.join(",")) })
}

/// This machine's cores.
pub fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

const fn plain(key: &'static str, arg: Option<&'static str>, kind: Kind, label: &'static str) -> Opt {
    Opt { key, arg, default: None, kind, label, tip: None, hint: None, kept: true }
}

const fn number(min: f64, max: fn() -> f64, step: f64, digits: u32) -> Kind {
    Kind::Number { min, max, step, digits }
}

// What counts as a duplicate.

pub static TOLERANCE: Opt = Opt {
    tip: Some(
        "How far apart two frames may be and still count as the same picture, in bits of a 64-bit fingerprint. Two unrelated \
         frames are about 32 apart.",
    ),
    hint: Some("Higher finds more re-encoded and degraded copies, and more videos that are not copies. 2 to 8 is useful."),
    // Past 32 the scan refuses it: that is chance, where every frame matches.
    ..plain("hamming_distance", Some("hamming_distance"), number(0.0, || 32.0, 1.0, 0), l::TOLERANCE)
};
pub static MATCH_PERCENT: Opt = Opt {
    tip: Some(
        "How much of a video must be found in the other, as a share of its own length. Either video's share counts, so a clip \
         is found in the longer video it was cut from. 0 reports any footage two videos share.",
    ),
    hint: Some("Of either video's length. Lower finds shorter clips."),
    ..plain("match_percent", Some("match_percent"), number(0.0, || 100.0, 5.0, 1), l::MATCH_PERCENT)
};
pub static MIN_DURATION: Opt = Opt {
    hint: Some("Seconds of footage two videos must share. Also leaves out videos shorter than this. 0 is off."),
    ..plain("min_duration", Some("min_duration"), number(0.0, || 86_400.0, 1.0, 1), l::MIN_DURATION)
};

// How it is done.

pub static THREADS: Opt = Opt {
    hint: Some("0 uses every core. More helps only for videos on a network share or a slow disk, and each costs memory."),
    // The scan caps it at four a core (or 16), and says so.
    ..plain("threads", Some("threads"), number(0.0, || (cores() * 4).max(16) as f64, 1.0, 0), l::THREADS)
};
pub static KF_INTERVAL: Opt = Opt {
    hint: Some("Seconds between the keyframes fingerprinted. 0 takes every keyframe. Higher is faster, and finds short clips less often."),
    ..plain("kf_interval", Some("kf_interval"), number(0.0, || 3600.0, 1.0, 1), l::KF_INTERVAL)
};
pub static MIN_KEYFRAMES: Opt = Opt {
    hint: Some("With an interval, a short video still keeps this many keyframes. 4, 12, 20 and 28 work well; 8 and 16 miss some."),
    ..plain("min_kf_samples", Some("min_kf_samples"), number(0.0, || 1000.0, 1.0, 0), l::MIN_KEYFRAMES)
};

// Which files.

pub static EXTENSIONS: Opt = Opt {
    hint: Some("Extensions a folder is searched for, comma-separated. * takes every file, and !flac leaves FLAC files out of everything else."),
    ..plain("extensions", Some("extensions"), Kind::Text, l::EXTENSIONS)
};
pub static RECURSIVE: Opt = Opt {
    // The command line's default is the top folder only; a person pointing a
    // window at their videos means all of them.
    default: Some("true"),
    ..plain("recursive", Some("recursive"), Kind::Check, l::RECURSIVE)
};
pub static SYMLINKS: Opt = plain("follow_symlinks", Some("follow_symlinks"), Kind::Check, l::SYMLINKS);

// The window's own switches: each says whether a path beside it is used, and
// `Settings::argv` passes the flags they stand for.

pub static REPORT: Opt = Opt { default: Some("false"), ..plain("report", None, Kind::Check, l::REPORT) };
/// `-o` alone goes by the file name's extension.
pub static REPORT_FORMAT: Opt = Opt {
    default: Some("FromName"),
    ..plain(
        "report_format",
        None,
        Kind::Choice(&[("FromName", None, "From the file name"), ("Txt", Some("txt"), "Text"), ("Csv", Some("csv"), "CSV"), ("Json", Some("json"), "JSON")]),
        l::REPORT_FORMAT,
    )
};
pub static CLEAR_CACHE: Opt = Opt { default: Some("false"), kept: false, ..plain("clear_cache", None, Kind::Check, l::CLEAR_CACHE) };
pub static PRUNE_CACHE: Opt = Opt { default: Some("false"), kept: false, ..plain("prune_cache", None, Kind::Check, l::PRUNE_CACHE) };
pub static LOG: Opt = Opt {
    default: Some("false"),
    hint: Some("Every line of the scan log, including each file that could not be read, in full."),
    ..plain("log", None, Kind::Check, l::LOG_FILE)
};

/// Every option, in the order the command line passes them.
pub static ALL: [&Opt; 14] = [
    &TOLERANCE, &MATCH_PERCENT, &MIN_DURATION, &THREADS, &KF_INTERVAL, &MIN_KEYFRAMES, &EXTENSIONS, &RECURSIVE, &SYMLINKS,
    &REPORT, &REPORT_FORMAT, &CLEAR_CACHE, &PRUNE_CACHE, &LOG,
];

impl Opt {
    /// Where it is in `ALL`, which is where `Settings` keeps its value.
    pub fn at(&self) -> usize {
        ALL.iter().position(|o| o.key == self.key).expect("every option is in ALL")
    }

    /// `s` as this option's value: as the command line or the settings file
    /// writes it. `None` if it is no such value.
    pub fn parse(&self, s: &str) -> Option<Value> {
        match self.kind {
            // A switch's flag has no default to read: absent is off.
            Kind::Check => match s {
                "true" => Some(Value::On(true)),
                "false" | "" => Some(Value::On(false)),
                _ => None,
            },
            Kind::Number { .. } => s.trim().parse().ok().filter(|v: &f64| v.is_finite()).map(|v| self.number(v)),
            Kind::Text => Some(Value::Text(s.trim().to_string())),
            Kind::Choice(cs) => cs.iter().find(|c| c.0 == s).map(|c| Value::Text(c.0.to_string())),
        }
    }

    /// `v`, a number of this option, rounded to its decimals: a spin button
    /// stepped from 0.85 by 0.05 reads 0.7999999999999999.
    pub fn number(&self, v: f64) -> Value {
        let digits = match self.kind {
            Kind::Number { digits, .. } => digits,
            _ => 0,
        };
        let f = 10f64.powi(digits as i32);
        Value::Number((v * f).round() / f)
    }

    /// The default: the window's own, or else the command line's.
    pub fn default_value(&self) -> Value {
        static DEFAULTS: std::sync::OnceLock<Vec<Value>> = std::sync::OnceLock::new();
        let all = DEFAULTS.get_or_init(|| {
            ALL.iter()
                .map(|o| {
                    let cli = o.arg.and_then(flag).and_then(|f| f.default);
                    let text = o.default.map(str::to_string).or(cli).unwrap_or_default();
                    o.parse(&text).unwrap_or_else(|| panic!("{}: the default {text:?} is no value of it", o.key))
                })
                .collect()
        });
        all[self.at()].clone()
    }

    /// The value as the settings file keeps it.
    pub fn as_json(&self, v: &Value) -> serde_json::Value {
        match (v, &self.kind) {
            (Value::On(b), _) => (*b).into(),
            (Value::Number(n), Kind::Number { digits: 0, .. }) => (*n as u64).into(),
            (Value::Number(n), _) => (*n).into(),
            (Value::Text(t), _) => t.as_str().into(),
        }
    }

    /// The value the settings file keeps, or `None` if it holds none of this
    /// option's (another type, a choice no longer offered).
    pub fn read_json(&self, j: &serde_json::Value) -> Option<Value> {
        match self.kind {
            Kind::Check => j.as_bool().map(Value::On),
            Kind::Number { .. } => j.as_f64().map(|v| self.number(v)),
            Kind::Text | Kind::Choice(_) => j.as_str().and_then(|s| self.parse(s)),
        }
    }

    /// The flag and its value, as `Settings::argv` passes an option with an
    /// `arg`; nothing for a check that is off, or empty text.
    pub fn command_line(&self, v: &Value) -> Vec<String> {
        let Some(flag) = self.arg.and_then(flag) else { return Vec::new() };
        match (&self.kind, v) {
            (Kind::Check, Value::On(true)) => vec![flag.long],
            (Kind::Number { digits: 0, .. }, Value::Number(n)) => vec![flag.long, (*n as u64).to_string()],
            (Kind::Number { .. }, Value::Number(n)) => vec![flag.long, n.to_string()],
            (Kind::Text, Value::Text(t)) if !t.is_empty() => vec![flag.long, t.clone()],
            _ => Vec::new(),
        }
    }

    /// The flag's value of a choice: `None` when it gives no flag.
    pub fn choice_flag(&self, v: &Value) -> Option<&'static str> {
        let Kind::Choice(cs) = self.kind else { return None };
        cs.iter().find(|c| c.0 == v.text()).and_then(|c| c.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// vid-fp's options the window passes by hand (`Settings::argv`): the
    /// paths, and what the switches above stand for.
    const BY_HAND: [&str; 8] = ["include", "exclude", "output", "format", "cache", "clear_cache", "prune_cache", "log_file"];

    /// vid-fp's options the window never passes: another kind of run, a
    /// disposal the window does itself (the Trash button, with the scan's
    /// sizes checked first), or its terminal's own switches. `--priority`
    /// is not passed either: the scan labels every ranking at once
    /// (`found::Verdict`), and the results page's *Rank by* picks one.
    /// `--against` and its `--list` are another kind of run, which `run`
    /// refuses from the window.
    const NOT_SHOWN: [&str; 11] =
        ["from_file", "null", "priority", "delete", "permanent", "move_to", "from_report", "yes", "quiet", "against", "list"];

    /// Each of vid-fp's options is the window's to show, its to pass by
    /// hand, or not its to pass, and only one of these: an option added to
    /// the command line fails here until it is put in one.
    #[test]
    fn every_flag_is_decided() {
        let mut decided: Vec<&str> = ALL.iter().filter_map(|o| o.arg).chain(BY_HAND).chain(NOT_SHOWN).collect();
        decided.extend(["man", "completions"]);
        let n = decided.len();
        decided.sort_unstable();
        decided.dedup();
        assert_eq!(decided.len(), n, "an option is decided twice");
        // clap's own, there or not depending on when it is asked.
        let mut ids: Vec<String> =
            vid_fp::command().get_arguments().map(|a| a.get_id().to_string()).filter(|i| i != "help" && i != "version").collect();
        ids.sort_unstable();
        assert_eq!(decided, ids);
    }

    /// Every option has a default that is a value of it, a flag where it
    /// names one, and a key of its own.
    #[test]
    fn every_option_is_whole() {
        let mut keys: Vec<&str> = ALL.iter().map(|o| o.key).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), ALL.len());
        for (i, o) in ALL.iter().enumerate() {
            assert_eq!(o.at(), i);
            o.default_value();
            if let Some(a) = o.arg {
                assert!(flag(a).is_some(), "{} has no flag", o.key);
            }
            assert_eq!(o.read_json(&o.as_json(&o.default_value())), Some(o.default_value()), "{}", o.key);
        }
    }

    /// The window starts from the command line's own defaults, so a scan
    /// left alone is `vid-fp -r` and finds what it finds.
    #[test]
    fn the_defaults_are_the_command_lines() {
        assert_eq!(TOLERANCE.default_value(), Value::Number(6.0));
        assert_eq!(MATCH_PERCENT.default_value(), Value::Number(20.0));
        assert_eq!(MIN_KEYFRAMES.default_value(), Value::Number(12.0));
        assert!(EXTENSIONS.default_value().text().split(',').any(|e| e == "mkv"));
        assert!(RECURSIVE.default_value().on() && !SYMLINKS.default_value().on());
    }
}
