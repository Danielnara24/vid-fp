#!/usr/bin/env python3
"""Check that every benchmarked tool actually looked at the same files.

A runtime comparison is only fair if the tools were given the same work, and
each of these decides for itself what counts as a video: vid-fp takes an
extension list (`avi flv mkv mov mp4 webm`), Czkawka has its own VIDEO macro,
VDF its own. A tool that quietly skipped the 30 `.mkv` files would look fast and
would be answering a different question.

Each one reports its count somewhere in its own log, in its own words:

    vid-fp   "Found 756 video files."
    czkawka  "Properly saved to file <756> cache entries."   (U+2068 delimiters)
    vdf      the hashing progress denominator, "N/756"

The counts are pulled out and compared against the corpus walked directly. This
is a check, not a fix: a mismatch means the runtime column needs a footnote
saying which tool saw what, not that the numbers get adjusted.
"""

import argparse
import collections
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The extensions vid-fp searches by default. Czkawka's VIDEO macro and VDF's
# list are supersets in the containers that matter here, so a corpus holding
# only mp4/mov/mkv is scanned whole by all three.
VIDFP_EXTS = {".avi", ".flv", ".mkv", ".mov", ".mp4", ".webm"}

PATTERNS = {
    "vid-fp": re.compile(r"Found (\d+) video files"),
    "czkawka": re.compile(r"saved to file .(\d+). cache entries"),
    "vdf": re.compile(r"\d+/(\d+)\s+ETA"),
    "vdf-clip": re.compile(r"\d+/(\d+)\s+ETA"),
    "vdf-par": re.compile(r"\d+/(\d+)\s+ETA"),
}


def corpus_counts(root):
    exts = collections.Counter()
    for dirpath, _, names in os.walk(root):
        for n in names:
            exts[os.path.splitext(n)[1].lower()] += 1
    return exts


def scanned(log_path, pattern):
    """Every distinct count the tool reported, most common first.

    VDF prints a progress line per file, so the denominator appears thousands of
    times; a *second* denominator appearing is the interesting case, because it
    is what exposed the warm-database bug (hashing counted against 756 database
    entries while the comparison counted 69 scanned files).
    """
    with open(log_path, "rb") as fh:
        text = fh.read().decode("utf-8", "replace").replace("\r", "\n")
    found = collections.Counter(int(m) for m in pattern.findall(text))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="/home/daniel/Documents/AN")
    ap.add_argument("--runs", default=os.path.join(HERE, "out", "bench"))
    args = ap.parse_args()

    exts = corpus_counts(args.corpus)
    eligible = sum(v for k, v in exts.items() if k in VIDFP_EXTS)
    print(f"corpus {args.corpus}")
    print(f"  {sum(exts.values())} files, {eligible} in vid-fp's extension list")
    print("  " + ", ".join(f"{k or '(none)'} {v}"
                           for k, v in exts.most_common()))

    print("\nreported by each tool's own log:")
    ok = True
    for log in sorted(glob.glob(os.path.join(args.runs, "logs", "*.log"))):
        name = os.path.basename(log).split(".r")[0]
        pat = PATTERNS.get(name)
        if not pat:
            continue
        counts = scanned(log, pat)
        if not counts:
            print(f"  {os.path.basename(log):<22} no count found")
            ok = False
            continue
        top = counts.most_common(1)[0][0]
        others = [c for c in counts if c != top]
        flag = "" if top == eligible else f"  <-- expected {eligible}"
        if top != eligible:
            ok = False
        extra = f"  (also saw {others})" if others else ""
        print(f"  {os.path.basename(log):<22} {top}{extra}{flag}")

    print("\nall tools saw the same file count" if ok
          else "\nMISMATCH -- the runtime comparison needs a footnote")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
