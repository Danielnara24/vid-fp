#!/usr/bin/env python3
"""Run vid-fp, Czkawka and VDF over one corpus and measure cost, not just answers.

`score.py` answers "was it right". This answers "what did it cost", on the same
corpus, in the same session, with each tool started from a COLD cache -- which
is the only footing on which their runtimes mean anything, since all three
persist hashes between runs and a warm run measures a database lookup.

What is measured per run:

    wall        elapsed seconds
    user/sys    CPU seconds (os.wait4 rusage, so children are included)
    cpu%        user+sys over wall -- how many cores the tool actually used
    peak_rss    the whole process TREE, sampled from /proc every 150 ms and
                summed across live processes
    maxrss      the single largest process, from rusage

`peak_rss` is the headline because VDF shells out to the ffmpeg CLI: rusage's
`maxrss` reports the largest single process, so a tool that runs eight 200 MB
ffmpeg children concurrently reports 200 MB there and 1.6 GB here. Summing RSS
double-counts pages shared between a parent and its forks, so the tree figure is
an upper bound for the multi-process tools and exact for the single-process
ones (vid-fp, Czkawka). Both are reported rather than one being chosen.

Sampling cannot see a peak that opens and closes inside one 150 ms window. For
decode workloads holding a frame buffer for minutes this does not bite; for a
tool that spikes briefly it would under-report.

Caches are moved aside before the batch and restored after, so a warm cache the
machine already had (~15 min of decode for the Wire corpus) survives the run.
"""

import argparse
import collections
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.sysconf("SC_PAGE_SIZE")


# ------------------------------------------------------------ process tree


def _proc_table():
    """pid -> (ppid, rss_bytes) for every process we can read."""
    out = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                data = fh.read()
            # comm may contain spaces and parentheses; ppid is the field after
            # the state letter, which follows the last ')'.
            rest = data[data.rindex(b")") + 2:].split()
            ppid = int(rest[1])
            with open(f"/proc/{pid}/statm", "rb") as fh:
                rss = int(fh.read().split()[1]) * PAGE
        except (OSError, ValueError, IndexError):
            continue
        out[pid] = (ppid, rss)
    return out


def _tree_rss(root, table):
    """Summed RSS of `root` and every descendant present in `table`."""
    kids = collections.defaultdict(list)
    for pid, (ppid, _) in table.items():
        kids[ppid].append(pid)
    total, stack, seen = 0, [root], set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        entry = table.get(pid)
        if entry is None:
            continue
        total += entry[1]
        stack.extend(kids.get(pid, ()))
    return total, len(seen)


class Sampler(threading.Thread):
    """Polls the tree's summed RSS until told to stop."""

    def __init__(self, root, interval=0.15):
        super().__init__(daemon=True)
        self.root, self.interval = root, interval
        self.peak, self.peak_procs, self.samples = 0, 0, 0
        self._done = threading.Event()

    def run(self):
        while not self._done.is_set():
            total, n = _tree_rss(self.root, _proc_table())
            self.samples += 1
            if total > self.peak:
                self.peak, self.peak_procs = total, n
            self._done.wait(self.interval)

    def stop(self):
        self._done.set()
        self.join(timeout=5)


# ------------------------------------------------------------------ runner


def measure(cmd, log_path, cwd=None):
    """Run `cmd`, returning timings and both memory figures."""
    with open(log_path, "wb") as log:
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=cwd,
                                start_new_session=True)
        sampler = Sampler(proc.pid)
        sampler.start()
        try:
            _, status, usage = os.wait4(proc.pid, 0)
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            raise
        wall = time.monotonic() - t0
        sampler.stop()

    rc = os.waitstatus_to_exitcode(status)
    cpu = usage.ru_utime + usage.ru_stime
    return {
        "cmd": cmd,
        "rc": rc,
        "wall_s": round(wall, 2),
        "user_s": round(usage.ru_utime, 2),
        "sys_s": round(usage.ru_stime, 2),
        "cpu_pct": round(100 * cpu / wall, 1) if wall else None,
        "peak_rss_mb": round(sampler.peak / 1e6, 1),
        "peak_rss_procs": sampler.peak_procs,
        "maxrss_mb": round(usage.ru_maxrss / 1024, 1),
        "rss_samples": sampler.samples,
        "log": os.path.basename(log_path),
    }


# ------------------------------------------------------------------- tools
#
# Each tool: the command to run at the settings being benchmarked, and a `cold`
# hook that removes whatever it persists between runs. Every tool here caches
# hashes, so without the hook the second round would measure a database read.


def tool_specs(args):
    corpus, out = args.corpus, args.runs
    # Folders inside the corpus no tool should read (VIDS keeps its source
    # mezzanines and scratch there). Passed to each tool in its own syntax.
    excl = [os.path.join(corpus, d) for d in args.exclude]
    vid_ex = [x for d in excl for x in ("-e", d)]
    cz_ex = [x for d in excl for x in ("-e", d)]
    vdf_ex = [x for d in excl for x in ("-e", d)]
    vid_cache = os.path.join(out, "vidfp-cache.redb")

    def cold_vidfp():
        # One global redb keyed by path -- deleting it is what makes the run
        # cold. The caller has already stashed the machine's copy.
        for p in (os.path.expanduser("~/.cache/vid-fp/fingerprints.redb"), vid_cache):
            if os.path.exists(p):
                os.remove(p)

    def cold_czkawka():
        cache = os.path.expanduser("~/.cache/czkawka")
        if os.path.isdir(cache):
            shutil.rmtree(cache)

    def cold_vdf(db):
        # --db is silently IGNORED unless the folder already exists: VDF then
        # falls back to a ScannedFiles.db beside its own binary, which on this
        # machine already held all 756 files, and the run measures a database
        # read instead of a scan (0.12 s for what should be minutes). The
        # give-away is the hashing progress counting against the database's
        # file count rather than the scan's. So: remove, then re-create.
        vdf_dir = os.path.dirname(os.path.abspath(args.vdf))

        def go():
            shutil.rmtree(db, ignore_errors=True)
            os.makedirs(db)
            # Belt and braces: clear the fallback too, so a run that ignores
            # --db for some other reason is still cold rather than instant.
            for stray in ("ScannedFiles.db", "ScannedFiles.fpheal1"):
                p = os.path.join(vdf_dir, stray)
                if os.path.exists(p):
                    os.remove(p)
        return go

    vdf_db = os.path.join(out, "vdf-db")
    vdf_clip_db = os.path.join(out, "vdf-clip-db")

    specs = collections.OrderedDict()

    # Defaults throughout: -d 4 -p 20, all cores. Clip detection is not a flag
    # here -- the comparison stage always measures directional coverage, which
    # is what makes a clip inside a longer file visible.
    specs["vid-fp"] = {
        "cmd": [args.vidfp, corpus, "-r", "-o", os.path.join(out, "vidfp.json"),
                "--cache", vid_cache, "--yes", *vid_ex],
        "cold": cold_vidfp,
        "settings": "defaults (-d 7 -p 20, --frame-size 128, all threads)",
        "clip": "always on (directional coverage)",
    }
    specs["vid-fp-256"] = {
        "cmd": [args.vidfp, corpus, "-r", "-o", os.path.join(out, "vidfp-256.json"),
                "--cache", vid_cache, "--yes", "--frame-size", "256", *vid_ex],
        "cold": cold_vidfp,
        "settings": "defaults + --frame-size 256",
        "clip": "always on (directional coverage)",
    }

    # Czkawka's default tolerance is -t 10 and it has no clip model at all;
    # --check-audio-content compares whole-file audio, not a clip inside a file,
    # so there is nothing to enable. -N suppresses console output only.
    specs["czkawka"] = {
        "cmd": [args.czkawka, "video", "-d", corpus,
                "-C", os.path.join(out, "czkawka.json"), "-N", *cz_ex],
        "cold": cold_czkawka,
        "settings": "defaults (-t 10, all threads)",
        "clip": "not supported",
    }

    # VDF's own defaults, including --parallelism 1. That is its shipped value
    # and it is single-threaded hashing, which is why the parallel row exists.
    specs["vdf"] = {
        "cmd": [args.vdf, "scan-and-compare", "-i", corpus, "--db", vdf_db,
                "-f", "json", "-o", os.path.join(out, "vdf.json"), *vdf_ex],
        "cold": cold_vdf(vdf_db),
        "settings": "defaults (--threshold 5 --percent 96 --parallelism 1)",
        "clip": "off by default",
    }

    # Defaults + the clip flag the user asked for. Audio-fingerprint based,
    # with the visual gate VDF turns on by itself.
    specs["vdf-clip"] = {
        "cmd": [args.vdf, "scan-and-compare", "-i", corpus, "--db", vdf_clip_db,
                "-f", "json", "-o", os.path.join(out, "vdf-clip.json"),
                "--partial-clip-detection", *vdf_ex],
        "cold": cold_vdf(vdf_clip_db),
        "settings": "defaults + --partial-clip-detection",
        "clip": "enabled (audio fingerprint)",
    }

    # Same defaults, parallelism raised to the core count. Not a defaults row --
    # it exists so the runtime column is not comparing 1 thread against 8.
    if args.vdf_parallel:
        db = os.path.join(out, "vdf-par-db")
        specs["vdf-par"] = {
            "cmd": [args.vdf, "scan-and-compare", "-i", corpus, "--db", db,
                    "-f", "json", "-o", os.path.join(out, "vdf-par.json"),
                    "--parallelism", str(args.vdf_parallel), *vdf_ex],
            "cold": cold_vdf(db),
            "settings": f"defaults + --parallelism {args.vdf_parallel}",
            "clip": "off by default",
        }

    # VDF 4.1's own best: both AI passes (embeddings for cropped, mirrored or
    # edited copies; dense keyframe matching for partial copies, no audio
    # needed), at its core-count parallelism. The model is downloaded once,
    # before the batch, so its download is not in the timing.
    db = os.path.join(out, "vdf-ai-db")
    specs["vdf-ai"] = {
        "cmd": [args.vdf, "scan-and-compare", "-i", corpus, "--db", db,
                "-f", "json", "-o", os.path.join(out, "vdf-ai.json"),
                "--parallelism", str(args.vdf_parallel), "--ai-matching", "--ai-partial", *vdf_ex],
        "cold": cold_vdf(db),
        "settings": f"--ai-matching --ai-partial --parallelism {args.vdf_parallel}",
        "clip": "enabled (AI dense keyframe matching)",
    }

    if args.only:
        specs = collections.OrderedDict(
            (k, v) for k, v in specs.items() if k in args.only)
    return specs


def stash(paths):
    """Move each path aside, returning the moves so they can be undone."""
    moved = []
    for p in paths:
        if os.path.exists(p):
            dest = p + ".bench-stash"
            shutil.rmtree(dest, ignore_errors=True)
            if os.path.isfile(dest):
                os.remove(dest)
            shutil.move(p, dest)
            moved.append((p, dest))
    return moved


def unstash(moved):
    """Put back what `stash` moved, discarding whatever the batch left behind."""
    for orig, dest in moved:
        if os.path.isdir(orig):
            shutil.rmtree(orig, ignore_errors=True)
        elif os.path.exists(orig):
            os.remove(orig)
        shutil.move(dest, orig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="/home/daniel/Documents/AN")
    ap.add_argument("--runs", default=os.path.join(HERE, "out", "bench"))
    ap.add_argument("--rounds", type=int, default=2,
                    help="the first round of a batch runs 8-15%% fast on this "
                         "machine, so two rounds are the minimum worth reading")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--exclude", nargs="*", default=[".mezz", ".raw", ".tmp", ".tools"],
                    help="folders under --corpus no tool may read")
    ap.add_argument("--cooldown", type=int, default=0,
                    help="idle seconds before each run. The batch that first "
                         "used this harness ran the tools back to back, and "
                         "vid-fp measured 142s there against 101s on a rested "
                         "machine -- 40% of thermal/contention overhang from "
                         "the previous tool. A gap gives every tool the same "
                         "rested start, which is what a user actually sees.")
    ap.add_argument("--vdf-parallel", type=int, default=os.cpu_count())
    ap.add_argument("--vidfp", default=os.path.join(HERE, os.pardir,
                                                    "target/release/vid-fp"))
    ap.add_argument("--czkawka",
                    default="/home/daniel/Documents/Vscode_repositories/img-fp/vendor/bin/czkawka_cli")
    ap.add_argument("--vdf", default="/home/daniel/Downloads/vdf-cli-4.1.1/outputCLI/vdf-cli")
    args = ap.parse_args()

    os.makedirs(args.runs, exist_ok=True)
    logs = os.path.join(args.runs, "logs")
    os.makedirs(logs, exist_ok=True)
    specs = tool_specs(args)

    # Preserve whatever warm caches the machine already has; the cold hooks
    # below delete freely once these are out of the way.
    # The last two are VDF's fallback database beside its own binary. --db is
    # meant to keep the batch off it, but it is ignored when the folder is
    # missing, so the fallback is stashed as well: if VDF reaches for it after
    # all, it still finds nothing cached and the run is still cold.
    vdf_dir = os.path.dirname(os.path.abspath(args.vdf))
    moved = stash([os.path.expanduser("~/.cache/vid-fp/fingerprints.redb"),
                   os.path.expanduser("~/.cache/czkawka"),
                   os.path.join(vdf_dir, "ScannedFiles.db"),
                   os.path.join(vdf_dir, "ScannedFiles.fpheal1")])
    print(f"stashed {len(moved)} existing cache path(s)", flush=True)

    results = []
    dest = os.path.join(args.runs, "runs.json")
    try:
        for rnd in range(1, args.rounds + 1):
            for name, spec in specs.items():
                if args.cooldown:
                    print(f"    cooling down {args.cooldown}s", flush=True)
                    time.sleep(args.cooldown)
                spec["cold"]()
                log = os.path.join(logs, f"{name}.r{rnd}.log")
                print(f"[round {rnd}] {name}: {' '.join(spec['cmd'])}",
                      flush=True)
                rec = measure(spec["cmd"], log)
                rec.update(tool=name, round=rnd, settings=spec["settings"],
                           clip=spec["clip"])
                print(f"    rc={rec['rc']} wall={rec['wall_s']}s "
                      f"cpu={rec['cpu_pct']}% peak_rss={rec['peak_rss_mb']}MB "
                      f"maxrss={rec['maxrss_mb']}MB", flush=True)
                results.append(rec)
                # Written after every run: these are expensive and a crash in
                # round 2 must not cost round 1.
                with open(dest, "w") as fh:
                    json.dump(results, fh, indent=2)
    finally:
        unstash(moved)
        print(f"restored {len(moved)} cache path(s)", flush=True)

    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
