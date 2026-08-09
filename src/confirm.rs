//! The last question asked before anything leaves its path.
//!
//! Everything up to this point is a measurement; this is the one place the tool
//! waits for a human. It sits between the KEEP/DELETE resolution and the
//! disposal pass rather than at start-up, because a confirmation is only worth
//! answering once it can say how many files and how many bytes -- and neither
//! number exists until the groups have been resolved.
//!
//! It is interactive-only by design. A prompt that a script cannot see is a
//! prompt that hangs a cron job forever, so when either half of the
//! conversation is missing (stdin is not a terminal, or the prompt itself would
//! be written into a redirect the user is not watching) the run proceeds
//! exactly as it did before this file existed. `--yes` says so explicitly, for
//! the scripts that want the guarantee written down rather than inferred.

use crate::export::Disposal;
use crate::utils::{format_size, shutdown_requested};
use std::io::{IsTerminal, Write};

/// How many targets are named before the rest are summarised as a count.
///
/// The prompt has to fit in the last screenful of a terminal to be readable at
/// all, and a user who wants the full list has a better way to get it: answer
/// `n` and read the report, which is printed either way.
const PREVIEW_LIMIT: usize = 10;

/// How many times an unrecognised answer is met with a re-ask before the
/// question gives up. Guards against a terminal that is feeding the process
/// something other than a person's keystrokes.
const MAX_ATTEMPTS: usize = 3;

/// A file about to be disposed of: what to call it, and what it weighs.
pub struct Target<'a> {
    pub path: &'a str,
    pub size: u64,
}

/// Whether the user is there to be asked at all.
///
/// Both halves matter. Without a terminal on stdin there is nobody to type an
/// answer; without one on stderr the question is written somewhere nobody is
/// looking, and the process would block on a prompt the user never saw. Reading
/// the list of paths from stdin (`vid-fp -`) takes the first condition away on
/// its own, which is the correct outcome: that invocation is a pipeline.
fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

/// The headline: what is about to happen, in the words of the mode doing it.
fn intent(disposal: &Disposal, count: usize, bytes: u64) -> String {
    let size = format_size(bytes);
    match disposal {
        Disposal::Trash => format!(
            "About to move {} file(s) ({}) to the trash.",
            count, size
        ),
        Disposal::Permanent => format!(
            "About to PERMANENTLY DELETE {} file(s) ({}). This cannot be undone.",
            count, size
        ),
        Disposal::MoveTo(dir) => format!(
            "About to move {} file(s) ({}) under {}.",
            count,
            size,
            dir.display()
        ),
    }
}

/// One line of input as an answer, or `None` if it was not one.
///
/// Empty means the user pressed Return, which the `[Y/n]` in the prompt
/// promises is a yes. Anything unrecognised is deliberately NOT read as a
/// default -- someone who typed a word meant something by it, and guessing
/// which way is the one mistake this function exists to prevent.
fn interpret(line: &str) -> Option<bool> {
    match line.trim().to_ascii_lowercase().as_str() {
        "" | "y" | "yes" => Some(true),
        "n" | "no" => Some(false),
        _ => None,
    }
}

/// Ask, and return whether the disposal may go ahead.
///
/// Returns true without asking anything when there is no user to ask or when
/// `--yes` was given. Every path that is not a clear yes returns false, so a
/// closed stdin, an unreadable terminal, an interrupt, or three unusable
/// answers all leave the files where they are.
pub fn approve(disposal: &Disposal, targets: &[Target<'_>], assume_yes: bool) -> bool {
    if targets.is_empty() || assume_yes || !interactive() {
        return true;
    }

    // A Ctrl-C already in flight has answered the question. Asking anyway would
    // print a prompt underneath the interrupt notice and wait for someone who
    // has just said they want out.
    if shutdown_requested() {
        return false;
    }

    let bytes: u64 = targets.iter().map(|t| t.size).sum();

    // Straight to stderr rather than through the logger, for the same reason
    // the interrupt notice is: --quiet silences reporting, and a question is
    // not a report. It also keeps the prompt out of a redirected stdout.
    let mut err = std::io::stderr();
    let _ = writeln!(err, "\n{}", intent(disposal, targets.len(), bytes));

    for t in targets.iter().take(PREVIEW_LIMIT) {
        let _ = writeln!(err, "  {:>9}  {}", format_size(t.size), t.path);
    }
    if let Some(rest) = targets.len().checked_sub(PREVIEW_LIMIT).filter(|&r| r > 0) {
        let _ = writeln!(err, "  ... and {} more", rest);
    }
    let _ = writeln!(
        err,
        "Answer n to print the full report and leave every file alone."
    );

    let stdin = std::io::stdin();
    let mut line = String::new();

    for _ in 0..MAX_ATTEMPTS {
        let _ = write!(err, "Continue? [Y/n] ");
        let _ = err.flush();

        line.clear();
        match stdin.read_line(&mut line) {
            // EOF. The stream ended rather than the user pressing Return, so it
            // is not the empty-line default: nobody said yes.
            Ok(0) => {
                let _ = writeln!(err);
                return false;
            }
            Ok(_) => {
                // A Ctrl-C during the read is an answer too.
                if shutdown_requested() {
                    return false;
                }
                match interpret(&line) {
                    Some(answer) => return answer,
                    None => {
                        let _ = writeln!(err, "Please answer y or n.");
                    }
                }
            }
            Err(e) => {
                let _ = writeln!(err, "Could not read an answer ({}).", e);
                return false;
            }
        }
    }

    let _ = writeln!(err, "No usable answer; nothing will be touched.");
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_return_means_yes() {
        // The whole point of [Y/n]: the common case is one keystroke.
        assert_eq!(interpret(""), Some(true));
        assert_eq!(interpret("\n"), Some(true));
        assert_eq!(interpret("  \n"), Some(true));
    }

    #[test]
    fn test_both_answers_in_any_case() {
        for yes in ["y", "Y", "yes", "YES", " Yes "] {
            assert_eq!(interpret(yes), Some(true), "{:?} should be a yes", yes);
        }
        for no in ["n", "N", "no", "NO", " No "] {
            assert_eq!(interpret(no), Some(false), "{:?} should be a no", no);
        }
    }

    #[test]
    fn test_anything_else_is_not_an_answer() {
        // Not false, and above all not true: these re-ask. "yeah" reading as a
        // yes is a guess, and "delete" reading as one is a catastrophe.
        for junk in ["yeah", "sure", "delete", "q", "yn", "1"] {
            assert_eq!(interpret(junk), None, "{:?} should not be an answer", junk);
        }
    }

    #[test]
    fn test_permanent_says_so() {
        // The three modes are not equally recoverable and the question should
        // not read as though they are.
        let permanent = intent(&Disposal::Permanent, 3, 1024);
        assert!(permanent.contains("PERMANENTLY"));
        assert!(permanent.contains("cannot be undone"));

        assert!(intent(&Disposal::Trash, 3, 1024).contains("trash"));

        let moved = intent(&Disposal::MoveTo(PathBuf::from("/backup/dupes")), 3, 1024);
        assert!(moved.contains("/backup/dupes"));
    }

    #[test]
    fn test_intent_counts_files_and_bytes() {
        let text = intent(&Disposal::Trash, 20, 4_080_218_931);
        assert!(text.contains("20 file(s)"), "{}", text);
        assert!(text.contains("3.8GB"), "{}", text);
    }

    #[test]
    fn test_nothing_to_confirm_is_approved() {
        // No targets means the disposal pass has no work; a prompt there would
        // ask the user to authorise a no-op. Runs regardless of whether the test
        // binary has a terminal, since it returns before looking.
        assert!(approve(&Disposal::Permanent, &[], false));
    }

    #[test]
    fn test_yes_flag_skips_the_question() {
        let targets = [Target {
            path: "/tmp/a.mkv",
            size: 1,
        }];
        assert!(approve(&Disposal::Permanent, &targets, true));
    }
}
