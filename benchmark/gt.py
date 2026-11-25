#!/usr/bin/env python3
"""Ground-truth pipeline for the vid-fp accuracy corpus.

Builds a pooled, tool-independent set of candidate pairs, renders contact
sheets for them, and emits a browser UI for adjudicating each pair by eye.

The output is a pair-level labels CSV -- `path_a, path_b, relation` with
relation in {SAME, CLIP, DIFFERENT} -- which is the ground truth. Groups,
per-file dispositions and the duplicate/clip split are all derivable from
it; none of them are stored, because storing them would freeze a decision
the labels already imply.

Why pooling: a ground truth assembled from vid-fp's own candidates cannot
measure vid-fp's recall, and scores every competing tool's unique finds as
false positives. So the pool unions several *independent* proposers, and a
human adjudicates the union. Add every tool you intend to benchmark to the
pool BEFORE labeling -- retrofitting one later means relabeling.

Subcommands run in order:

    oracle   dense-sample every video and propose pairs, independently of
             vid-fp's keyframe+DCT pipeline
    pool     union vid-fp's candidates with the oracle's -> pool.csv
    sheets   render one contact strip per file that appears in the pool
    ui       write review.html over the pool

Everything lands in --out (default ./out), which is gitignored.
"""

import argparse
import csv
import glob
import html
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import collections
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv")

# The oracle samples on a wall-clock grid and hashes with a gradient hash.
# Both choices are deliberately unlike vid-fp, which samples keyframes and
# hashes with a DCT -- an oracle that shared either would share vid-fp's
# blind spots and could not reveal them.
ORACLE_FPS = 2
ORACLE_FPS_LONG = 1                # above ORACLE_LONG_SECS, halve the rate
ORACLE_LONG_SECS = 600
ORACLE_W, ORACLE_H = 9, 8          # 9 wide so each row yields 8 gradient bits
ORACLE_FRAME_BYTES = ORACLE_W * ORACLE_H
ORACLE_HAMMING = 10
ORACLE_FLAT_RANGE = 12             # drop near-flat frames: black, fades, cards
ORACLE_MIN_MAD = 8                 # ...and frames whose spread is one outlier
ORACLE_MIN_COVERAGE = 0.50         # what counts as a "strong" oracle proposal
# ...and a fraction alone does not: a file with two retained frames reads 100%
# covered the moment both of them land near anything, which is how a nature
# documentary came to "match" a push-up clip at 100%. Demand a floor on the
# absolute footage matched as well, the same thing --min-duration does for
# vid-fp. Three seconds is roughly six frames at the oracle's sampling rate.
ORACLE_MIN_MATCH_SECS = 3.0
# A near-static file defeats both of the above. `tricep pushdown_11.mp4` has
# six sampled frames and *two* distinct hashes between them, so if either one
# lands near anything the file reads 100% covered for its whole runtime -- the
# same trap that single-keyframe files set for vid-fp's --match-percent. Count
# distinct hashes matched, not frames, and require several.
ORACLE_MIN_DISTINCT = 4
# And a hash can be unique yet still worthless, which no count of exact
# duplicates reveals: `barbell biceps curl_45.mp4`'s eight hashes each occur in
# exactly one file, but each sits in such a crowded region of hash space that
# its tolerance ball reaches dozens of unrelated videos. Evidence that points
# everywhere points nowhere, so a hash whose ball spans more files than any
# real duplicate group could is dropped. The largest true group here is the
# nineteen needle encodes, and the episodes share a title sequence across ten.
ORACLE_MAX_FILES_PER_HASH = 25

# Matching splits the 64-bit hash into 8 bytes and probes each at radius 1.
# Pigeonhole: a pair within ORACLE_HAMMING must agree to within one bit on at
# least one byte whenever ORACLE_HAMMING <= 15, so this is exhaustive, not a
# heuristic -- which matters, because the oracle exists to find what vid-fp's
# own index might miss and would be worthless with a blind spot of its own.
ORACLE_BLOCKS = 8
ORACLE_BLOCK_BITS = 8
assert ORACLE_HAMMING <= 2 * ORACLE_BLOCKS - 1

# A strip has to span the file end to end, so on an hour-long episode eight
# frames land 7 minutes apart and a two-minute insert falls between them.
SHEET_FRAME_W = 180
SHEET_LADDER = [(120, 8), (600, 16), (10**9, 32)]   # (max duration, frames)
# Below this fraction the matched window is worth drawing on its own, because
# the whole-file strip cannot resolve it.
WINDOW_SHEET_MAX_FRACTION = 0.5
WINDOW_SHEET_MIN_HOST_SECS = 300


def sheet_frames_for(duration):
    for limit, n in SHEET_LADDER:
        if duration <= limit:
            return n
    return SHEET_LADDER[-1][1]


def popcount(x):
    return bin(x).count("1")


try:
    (1).bit_count
    popcount = lambda x: x.bit_count()  # noqa: E731
except AttributeError:
    pass


def find_videos(root):
    out = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(VIDEO_EXTS):
                out.append(os.path.join(dirpath, n))
    return sorted(out)


def key_for(path):
    """Stable short filename for a video's derived artefacts."""
    import hashlib

    return hashlib.md5(path.encode("utf-8")).hexdigest()[:16]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


# --------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------


def informative(frame):
    """Whether a downsampled frame carries enough structure to hash.

    A raw min/max range is not enough on its own -- one bright pixel in a dark
    frame passes it while leaving the other 71 flat, and the resulting hash is
    decided by sensor noise. Requiring spread *and* average deviation from the
    mean drops the frames that would otherwise collide with everything.
    """
    if not frame:
        return False
    if max(frame) - min(frame) <= ORACLE_FLAT_RANGE:
        return False
    mean = sum(frame) / len(frame)
    mad = sum(abs(v - mean) for v in frame) / len(frame)
    return mad >= ORACLE_MIN_MAD


def dhash(frame, flip=False):
    """64-bit horizontal-gradient hash of one ORACLE_W x ORACLE_H frame."""
    h = 0
    for r in range(ORACLE_H):
        row = frame[r * ORACLE_W : (r + 1) * ORACLE_W]
        if flip:
            row = row[::-1]
        for c in range(ORACLE_W - 1):
            h = (h << 1) | (1 if row[c] < row[c + 1] else 0)
    return h


def oracle_sample(path):
    p = run([
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", f"fps={ORACLE_FPS},scale={ORACLE_W}:{ORACLE_H},format=gray",
        "-f", "rawvideo", "-",
    ])
    raw = p.stdout if isinstance(p.stdout, bytes) else b""
    return raw


def cmd_oracle(args):
    videos = find_videos(args.corpus)
    print(f"oracle: dense-sampling {len(videos)} videos", file=sys.stderr)

    def detect_crop(path, dur):
        """Letterbox/pillarbox borders, as an ffmpeg crop filter argument.

        Without this the hash is worthless on a padded file: a clip that is
        mostly black bars downsamples to a nearly uniform 9x8 grid, so its
        hash is decided by noise and collides with every other dark frame. It
        was matching a nature documentary against a push-up clip at 100%.
        Czkawka crop-detects by default and vid-fp auto-crops; so must this.
        """
        start = max(0.0, min(dur * 0.25, 60.0)) if dur else 0.0
        p = subprocess.run(
            ["ffmpeg", "-v", "info", "-ss", f"{start:.2f}", "-i", path,
             "-vf", "cropdetect=24:2:0", "-frames:v", "40", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace",
        )
        found = re.findall(r"crop=(\d+:\d+:\d+:\d+)", p.stderr or "")
        return found[-1] if found else None

    def work(path):
        dur = ffprobe_meta(path).get("duration", 0.0)
        fps = ORACLE_FPS_LONG if dur > ORACLE_LONG_SECS else ORACLE_FPS
        crop = detect_crop(path, dur)
        vf = (f"crop={crop}," if crop else "")
        vf += f"fps={fps},scale={ORACLE_W}:{ORACLE_H},format=gray"
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo", "-"],
            capture_output=True,
        )
        raw = p.stdout
        n = len(raw) // ORACLE_FRAME_BYTES
        frames = [raw[i * ORACLE_FRAME_BYTES : (i + 1) * ORACLE_FRAME_BYTES] for i in range(n)]
        keep = [f for f in frames if informative(f)]
        return path, {
            "h": [dhash(f) for f in keep],
            "hf": [dhash(f, True) for f in keep],
            "n": len(keep),
            "fps": fps,
            "crop": crop or "",
        }

    # Sampling is the expensive half -- hours of dense decode on a corpus with
    # full episodes in it -- and it used to live only in memory, so any failure
    # in the matching half threw all of it away. Cache it against mtime+size.
    cache_path = os.path.join(args.out, "oracle_hashes.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as fh:
                cache = json.load(fh)
        except Exception:
            cache = {}

    def stamp(path):
        st = os.stat(path)
        return f"{int(st.st_mtime)}:{st.st_size}"

    fresh = {p: rec for p, rec in cache.items()
             if os.path.exists(p) and rec.get("stamp") == stamp(p)}
    todo = [p for p in videos if p not in fresh]
    print(f"oracle: {len(fresh)} cached, {len(todo)} to sample", file=sys.stderr)

    data = dict(fresh)
    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for path, rec in ex.map(work, todo):
                rec["stamp"] = stamp(path)
                data[path] = rec
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(f"  sampled {done}/{len(todo)}", file=sys.stderr, flush=True)
        with open(cache_path, "w") as fh:
            json.dump(data, fh)
        print(f"oracle: hashes cached -> {cache_path}", file=sys.stderr)

    files = sorted(data)
    idx_of = {p: i for i, p in enumerate(files)}
    total = sum(v["n"] for v in data.values())
    print(f"oracle: {total} frames retained across {len(files)} files", file=sys.stderr)

    # One index over the whole corpus, not per folder. Indexed matching is
    # cheap enough now that restricting to folders would only reinstate a
    # blind spot -- and the Wire clips deliberately sit in two directories.
    index = defaultdict(list)
    for p in files:
        i = idx_of[p]
        # One entry per distinct hash per file. A repeated hash adds nothing to
        # the answer -- the probe only asks *whether* file i holds a match --
        # but it would lengthen every bucket the probe has to walk.
        for h in set(data[p]["h"]):
            for k in range(ORACLE_BLOCKS):
                block = (h >> (8 * k)) & 0xFF
                index[(k, block)].append((i, h))
    print(f"oracle: index has {sum(len(v) for v in index.values())} entries "
          f"in {len(index)} buckets", file=sys.stderr, flush=True)

    # Distinct hashes, not frames, drive the probe cost: a static shot repeats
    # the same hash for its whole duration and nineteen encodes of one clip
    # repeat it nineteen times over. On this corpus that collapses the work by
    # roughly an order of magnitude, and the answer is identical either way.
    probe_cache = {}

    def probe(h):
        """Every file index holding a frame within ORACLE_HAMMING of `h`."""
        got = probe_cache.get(h)
        if got is not None:
            return got
        out = set()
        for k in range(ORACLE_BLOCKS):
            block = (h >> (8 * k)) & 0xFF
            for bit in range(-1, ORACLE_BLOCK_BITS):
                cand = block if bit < 0 else block ^ (1 << bit)
                for j, hj in index.get((k, cand), ()):
                    if j not in out and popcount(h ^ hj) <= ORACLE_HAMMING:
                        out.add(j)
        probe_cache[h] = out
        return out

    hits = defaultdict(int)        # (a, b) -> A-frames with a partner in B
    distinct = defaultdict(int)    # (a, b) -> distinct A-hashes with a partner
    flips = defaultdict(int)
    suppressed = 0
    spans = []
    for n, p in enumerate(files):
        i = idx_of[p]
        # Walk distinct hashes and carry each one's frame count, so coverage
        # still counts frames while `distinct` counts how much genuinely
        # different footage backs the match.
        for h, c in collections.Counter(data[p]["h"]).items():
            near = probe(h)
            spans.append(len(near))
            if len(near) > ORACLE_MAX_FILES_PER_HASH:
                suppressed += 1
                continue
            for j in near:
                if j != i:
                    hits[(i, j)] += c
                    distinct[(i, j)] += 1
        for h, c in collections.Counter(data[p]["hf"]).items():
            near = probe(h)
            if len(near) > ORACLE_MAX_FILES_PER_HASH:
                continue
            for j in near:
                if j != i:
                    flips[(i, j)] += c
        if (n + 1) % 100 == 0 or n + 1 == len(files):
            print(f"  matched {n+1}/{len(files)}", file=sys.stderr, flush=True)

    if spans:
        spans.sort()
        q = lambda f: spans[min(len(spans) - 1, int(len(spans) * f))]  # noqa: E731
        print(f"oracle: hash reach median={q(0.5)} p90={q(0.9)} p99={q(0.99)} "
              f"max={spans[-1]}; {suppressed} hashes suppressed for reaching "
              f">{ORACLE_MAX_FILES_PER_HASH} files", file=sys.stderr, flush=True)

    pairs = {}
    seen = {(a, b) for a, b in hits} | {(a, b) for a, b in flips}
    for a, b in seen:
        if a >= b:
            continue
        na, nb = data[files[a]]["n"], data[files[b]]["n"]
        rec = {
            "ca": hits.get((a, b), 0) / na if na else 0.0,
            "cb": hits.get((b, a), 0) / nb if nb else 0.0,
            "fa": flips.get((a, b), 0) / na if na else 0.0,
            "fb": flips.get((b, a), 0) / nb if nb else 0.0,
            "da": distinct.get((a, b), 0),
            "db": distinct.get((b, a), 0),
        }
        # Only the coverage fractions gate admission here; `da`/`db` are
        # counts and would sail past any threshold meant for a ratio.
        if max(rec["ca"], rec["cb"], rec["fa"], rec["fb"]) >= 0.10:
            pairs[f"{files[a]}\x00{files[b]}"] = rec

    out = os.path.join(args.out, "oracle_pairs.json")
    with open(out, "w") as fh:
        json.dump(pairs, fh)
    strong = sum(1 for v in pairs.values()
                 if max(v["ca"], v["cb"], v["fa"], v["fb"]) >= ORACLE_MIN_COVERAGE)
    print(f"oracle: {len(pairs)} pairs >=10%, {strong} >={int(ORACLE_MIN_COVERAGE*100)}% -> {out}")


# --------------------------------------------------------------------------
# pool
# --------------------------------------------------------------------------


def vidfp_pairs(report):
    """Pairs and per-file directional coverage from one vid-fp JSON report."""
    with open(report) as fh:
        j = json.load(fh)
    dur, matched = {}, defaultdict(dict)
    for group in j.get("results", []):
        for f in group.get("files", []):
            a = f["full_path"]
            dur[a] = f.get("length_seconds") or 0.0
            for m in f.get("matches", []):
                key = tuple(sorted((a, m["full_path"])))
                matched[key][a] = {
                    "seconds": m.get("matched_seconds") or 0.0,
                    "from": m.get("matched_from_seconds"),
                    "to": m.get("matched_to_seconds"),
                }
    return matched, dur


PATH_KEYS = ("Path", "path", "full_path", "FullName", "file", "File",
             "filename", "FileName", "location")
GROUP_KEYS = ("GroupId", "group_id", "group", "Group")


def _path_of(d):
    for k in PATH_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.startswith("/"):
            return v
    return None


def extract_groups(obj):
    """Best-effort recovery of duplicate groups from any tool's JSON.

    Every tool here reports the same thing -- sets of files it considers the
    same -- and every one wraps it differently: MFDF returns a flat list tagged
    with a group id, Czkawka and VDF nest lists of file records at varying
    depths. Rather than a parser per tool that breaks on the next release, walk
    the tree: any list of objects carrying an absolute path is a group, and a
    flat list carrying group ids is regrouped by that id.
    """
    groups = []

    def walk(node):
        if isinstance(node, list):
            here = [d for d in node if isinstance(d, dict) and _path_of(d)]
            if len(here) >= 2:
                tagged = defaultdict(list)
                for d in here:
                    gid = next((str(d[k]) for k in GROUP_KEYS if k in d), None)
                    tagged[gid].append(_path_of(d))
                for gid, paths in tagged.items():
                    # A single untagged list is one group; ids split it further.
                    if len(paths) >= 2:
                        groups.append(paths)
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)

    walk(obj)
    return groups


def tool_pairs(path):
    """(pairs, group_count) for one tool's JSON output."""
    with open(path) as fh:
        obj = json.load(fh)
    groups = extract_groups(obj)
    pairs = set()
    for g in groups:
        uniq = sorted(set(g))
        for a, b in itertools.combinations(uniq, 2):
            pairs.add((a, b))
    return pairs, len(groups)


def ffprobe_meta(path):
    p = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size:stream=width,height,codec_name",
        "-of", "json", path,
    ])
    try:
        j = json.loads(p.stdout)
    except Exception:
        return {}
    fmt = j.get("format", {})
    streams = [s for s in j.get("streams", []) if s.get("width")]
    st = streams[0] if streams else {}
    return {
        "duration": float(fmt.get("duration") or 0.0),
        "size": int(fmt.get("size") or 0),
        "width": st.get("width"),
        "height": st.get("height"),
        "codec": st.get("codec_name"),
    }


def cmd_pool(args):
    pool = {}          # pair -> {"sources": set, "first_d": int|None}
    cov = {}           # pair -> {path: coverage}
    win = {}           # pair -> {path: (from_s, to_s)}
    dur = {}
    samples = {}

    reports = sorted(glob.glob(os.path.join(args.out, "vidfp_d*.json")))
    if not reports:
        print("pool: no vidfp_d*.json in --out; run `sweep` first", file=sys.stderr)
        return 1

    def dof(p):
        return int(os.path.basename(p)[len("vidfp_d") : -len(".json")])

    for rep in sorted(reports, key=dof):
        d = dof(rep)
        matched, durs = vidfp_pairs(rep)
        dur.update({k: v for k, v in durs.items() if v})
        with open(rep) as fh:
            j = json.load(fh)
        for group in j.get("results", []):
            for f in group.get("files", []):
                if f.get("samples") is not None:
                    samples[f["full_path"]] = f["samples"]
        for key, per_file in matched.items():
            rec = pool.setdefault(key, {"sources": set(), "first_d": None})
            rec["sources"].add("vid-fp")
            if rec["first_d"] is None:
                rec["first_d"] = d
            # keep the tightest -d's measurement; looser ones only add noise
            if key not in cov:
                cov[key] = {p: (m["seconds"] / dur[p] if dur.get(p) else 0.0)
                            for p, m in per_file.items()}
                win[key] = {p: (m["from"], m["to"]) for p, m in per_file.items()
                            if m["from"] is not None and m["to"] is not None}

    oracle_path = os.path.join(args.out, "oracle_pairs.json")
    oracle = {}
    if os.path.exists(oracle_path):
        with open(oracle_path) as fh:
            raw = json.load(fh)
        for k, v in raw.items():
            a, b = k.split("\x00")
            oracle[tuple(sorted((a, b)))] = v
    else:
        print("pool: no oracle_pairs.json -- pool is vid-fp only, which cannot "
              "measure vid-fp's recall", file=sys.stderr)

    # Other tools. Anything in --out/../tools/*.json is picked up automatically
    # so adding a tool is dropping its output in that directory.
    extras = list(args.extra or [])
    tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
    if os.path.isdir(tools_dir):
        extras += sorted(glob.glob(os.path.join(tools_dir, "*.json")))
    for extra in extras:
        name = os.path.splitext(os.path.basename(extra))[0]
        try:
            if extra.endswith(".json"):
                tp, ngroups = tool_pairs(extra)
            else:
                tp, ngroups = set(), 0
                with open(extra) as fh:
                    for row in csv.DictReader(fh):
                        a, b = row.get("path_a", "").strip(), row.get("path_b", "").strip()
                        if a and b:
                            tp.add(tuple(sorted((a, b))))
        except Exception as exc:
            print(f"pool: could not read {extra}: {exc}", file=sys.stderr)
            continue
        missing = sum(1 for p in tp for f in p if not os.path.exists(f))
        for key in tp:
            rec = pool.setdefault(tuple(sorted(key)), {"sources": set(), "first_d": None})
            rec["sources"].add(name)
        print(f"pool: {name}: {ngroups} groups -> {len(tp)} pairs"
              + (f" ({missing} path(s) not on disk)" if missing else ""), file=sys.stderr)

    # Probe before admitting the oracle's pairs, because the oracle's strength
    # test needs durations to turn a coverage fraction into seconds of footage.
    files = sorted({p for key in pool for p in key}
                   | {p for key in oracle for p in key})
    print(f"pool: probing {len(files)} files", file=sys.stderr)
    meta = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for path, m in zip(files, ex.map(ffprobe_meta, files)):
            meta[path] = m

    admitted = weak = 0
    for key, v in oracle.items():
        a, b = key
        da = meta.get(a, {}).get("duration", 0.0)
        db = meta.get(b, {}).get("duration", 0.0)
        # Each direction's coverage is a fraction of that file's own runtime,
        # so seconds matched is coverage x that file's duration.
        # Upright evidence only. The mirrored comparison (`fa`/`fb`) is still
        # computed and still reported, but it has never once produced a true
        # positive here -- a centred figure on a 9x8 grey grid matches its own
        # reflection trivially, so on a corpus of centred figures it fires
        # constantly. It was admitting a nature documentary against a push-up
        # clip at 100%. If mirrored copies matter, VDF's --ai-matching pass is
        # built for them; this hash is not.
        upright = max(v.get("ca", 0), v.get("cb", 0))
        secs = max(v.get("ca", 0) * da, v.get("cb", 0) * db)
        seen = max(v.get("da", 0), v.get("db", 0))
        if (upright >= ORACLE_MIN_COVERAGE and secs >= ORACLE_MIN_MATCH_SECS
                and seen >= ORACLE_MIN_DISTINCT):
            pool.setdefault(key, {"sources": set(), "first_d": None})["sources"].add("oracle")
            admitted += 1
        elif upright >= ORACLE_MIN_COVERAGE:
            weak += 1
    if oracle:
        print(f"pool: oracle: {admitted} pairs admitted, {weak} dropped for "
              f"matching under {ORACLE_MIN_MATCH_SECS:g}s of footage", file=sys.stderr)

    rows = []
    for key, rec in pool.items():
        a, b = key
        ca = cov.get(key, {}).get(a)
        cb = cov.get(key, {}).get(b)
        known = [c for c in (ca, cb) if c is not None]
        o = oracle.get(key, {})
        rows.append({
            "pair_id": key_for(a + "\x00" + b),
            "path_a": a,
            "path_b": b,
            "relation": "",
            "notes": "",
            "folder_a": os.path.dirname(a),
            "folder_b": os.path.dirname(b),
            "same_folder": "yes" if os.path.dirname(a) == os.path.dirname(b) else "no",
            "sources": "+".join(sorted(rec["sources"])),
            "vidfp_first_d": rec["first_d"] if rec["first_d"] is not None else "",
            "vidfp_cov_a": f"{min(ca,1.0):.4f}" if ca is not None else "",
            "vidfp_cov_b": f"{min(cb,1.0):.4f}" if cb is not None else "",
            "vidfp_cov_min": f"{min(known):.4f}" if len(known) == 2 else "",
            "vidfp_cov_max": f"{max(known):.4f}" if known else "",
            "oracle_cov_max": f"{max(o.get('ca',0), o.get('cb',0)):.4f}" if o else "",
            "oracle_cov_mirror": f"{max(o.get('fa',0), o.get('fb',0)):.4f}" if o else "",
            "match_from_a": f"{win.get(key,{}).get(a,(None,None))[0]:.2f}" if win.get(key, {}).get(a) else "",
            "match_to_a": f"{win.get(key,{}).get(a,(None,None))[1]:.2f}" if win.get(key, {}).get(a) else "",
            "match_from_b": f"{win.get(key,{}).get(b,(None,None))[0]:.2f}" if win.get(key, {}).get(b) else "",
            "match_to_b": f"{win.get(key,{}).get(b,(None,None))[1]:.2f}" if win.get(key, {}).get(b) else "",
            "duration_a": f"{meta.get(a,{}).get('duration',0):.3f}",
            "duration_b": f"{meta.get(b,{}).get('duration',0):.3f}",
            "samples_a": samples.get(a, ""),
            "samples_b": samples.get(b, ""),
            "resolution_a": f"{meta.get(a,{}).get('width')}x{meta.get(a,{}).get('height')}",
            "resolution_b": f"{meta.get(b,{}).get('width')}x{meta.get(b,{}).get('height')}",
            "codec_a": meta.get(a, {}).get("codec", ""),
            "codec_b": meta.get(b, {}).get("codec", ""),
            "size_a": meta.get(a, {}).get("size", ""),
            "size_b": meta.get(b, {}).get("size", ""),
        })

    # Two tiers, because a pair several independent tools agree on is far more
    # likely to be real than one only a single loose setting proposed -- and it
    # is also the pair whose label decides the most scores. Tier first, then
    # folder, because consecutive pairs from one folder share visual context
    # and that is what makes adjudication fast.
    for r in rows:
        r["source_count"] = len(r["sources"].split("+"))
    rows.sort(key=lambda r: (-min(r["source_count"], 2),
                             r["same_folder"] == "no",
                             r["folder_a"], r["path_a"], r["path_b"]))
    for i, r in enumerate(rows):
        r["review_order"] = i

    cols = ["pair_id", "path_a", "path_b", "relation", "notes", "review_order",
            "same_folder", "folder_a", "folder_b", "sources", "source_count",
            "vidfp_first_d",
            "vidfp_cov_a", "vidfp_cov_b", "vidfp_cov_min", "vidfp_cov_max",
            "oracle_cov_max", "oracle_cov_mirror",
            "match_from_a", "match_to_a", "match_from_b", "match_to_b",
            "duration_a", "duration_b", "samples_a", "samples_b",
            "resolution_a", "resolution_b", "codec_a", "codec_b", "size_a", "size_b"]
    out = os.path.join(args.out, "pool.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    by_src = defaultdict(int)
    for r in rows:
        by_src[r["sources"]] += 1
    print(f"pool: {len(rows)} pairs -> {out}")
    for k in sorted(by_src, key=lambda k: -by_src[k]):
        print(f"       {by_src[k]:>5}  {k}")
    return 0


# --------------------------------------------------------------------------
# sheets
# --------------------------------------------------------------------------


# Above this span, `fps=N/duration` is the wrong tool: it decodes every frame
# of the file to emit N of them, so an hour-long episode costs minutes. Seeking
# to each wanted timestamp instead costs one keyframe-to-target decode apiece.
SEEK_STRIP_OVER_SECS = 300


def render_strip(path, dest, frames, start=None, span=None):
    """Tile `frames` stills evenly across `path`, or across a window of it."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    base_t = start or 0.0
    length = span if span is not None else _duration(path)

    if length and length > SEEK_STRIP_OVER_SECS:
        tmpdir = dest + ".parts"
        os.makedirs(tmpdir, exist_ok=True)
        stills = []
        try:
            for i in range(frames):
                # Sample at interval midpoints so the first frame is not the
                # (often black) very first one and the last is inside the file.
                t = base_t + length * (i + 0.5) / frames
                still = os.path.join(tmpdir, f"{i:03d}.png")
                r = run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path,
                         "-frames:v", "1", "-vf", f"scale={SHEET_FRAME_W}:-2",
                         "-y", still])
                if r.returncode == 0 and os.path.exists(still):
                    stills.append(still)
            if not stills:
                return False
            cmd = ["ffmpeg", "-v", "error"]
            for s in stills:
                cmd += ["-i", s]
            cmd += ["-filter_complex",
                    "".join(f"[{i}]" for i in range(len(stills))) + f"hstack={len(stills)}"
                    if len(stills) > 1 else "[0]null",
                    "-frames:v", "1", "-q:v", "4", "-y", dest]
            r = run(cmd)
            return r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        # Seek before -i so the decoder skips rather than decodes to the point.
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if span is not None:
        cmd += ["-t", f"{span:.3f}"]
    vf = (f"fps={frames}/{length:.4f}," if length and length > 0.05 else "fps=2,")
    vf += f"scale={SHEET_FRAME_W}:-2,tile={frames}x1"
    cmd += ["-vf", vf, "-frames:v", "1", "-q:v", "4", "-y", dest]
    r = run(cmd)
    return r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


_dur_cache = {}


def _duration(path):
    if path not in _dur_cache:
        p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path])
        try:
            _dur_cache[path] = float(p.stdout.strip())
        except Exception:
            _dur_cache[path] = 0.0
    return _dur_cache[path]


def cmd_sheets(args):
    pool_csv = os.path.join(args.out, "pool.csv")
    with open(pool_csv) as fh:
        rows = list(csv.DictReader(fh))
    files = sorted({r["path_a"] for r in rows} | {r["path_b"] for r in rows})
    sheets = os.path.join(args.out, "sheets")
    windows = os.path.join(args.out, "windows")
    os.makedirs(sheets, exist_ok=True)
    os.makedirs(windows, exist_ok=True)

    def whole(path):
        dur = _duration(path)
        # The strip spans the file end to end, so the reviewer is judging the
        # same span of footage on both rows; the frame count rises with
        # duration so an hour-long episode is not eight frames 7 minutes apart.
        return render_strip(path, os.path.join(sheets, key_for(path) + ".jpg"),
                            sheet_frames_for(dur))

    ok = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for good in ex.map(whole, files):
            ok += 1 if good else 0
    print(f"sheets: {ok}/{len(files)} whole-file strips -> {sheets}")

    # A two-minute insert inside a one-hour episode is invisible on a
    # whole-file strip at any frame count that still fits on screen. Where a
    # proposer reported *where* it matched, draw that window on its own.
    jobs = []
    for r in rows:
        for s in ("a", "b"):
            try:
                lo = float(r.get(f"match_from_{s}") or "")
                hi = float(r.get(f"match_to_{s}") or "")
                dur = float(r.get(f"duration_{s}") or 0)
            except ValueError:
                continue
            span = hi - lo
            if span <= 0 or dur < WINDOW_SHEET_MIN_HOST_SECS:
                continue
            if span / dur > WINDOW_SHEET_MAX_FRACTION:
                continue
            pad = min(span * 0.15, 10.0)
            lo = max(0.0, lo - pad)
            span = min(dur - lo, span + 2 * pad)
            jobs.append((r[f"path_{s}"],
                         os.path.join(windows, f"{r['pair_id']}_{s}.jpg"),
                         lo, span))

    if jobs:
        def window(j):
            path, dest, lo, span = j
            return render_strip(path, dest, 16, start=lo, span=span)

        wok = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for good in ex.map(window, jobs):
                wok += 1 if good else 0
        print(f"sheets: {wok}/{len(jobs)} matched-window strips -> {windows}")


# --------------------------------------------------------------------------
# ui
# --------------------------------------------------------------------------


UI_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>vid-fp ground truth</title>
<style>
:root{
  --bg:#12141a; --panel:#1b1e26; --line:#2b303c; --fg:#e6e8ee; --muted:#9aa2b4;
  --same:#2fbf71; --clip:#f0a336; --diff:#e2564a; --skip:#7d86a0; --accent:#5b8def;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);
       border-bottom:1px solid var(--line);padding:10px 16px;
       display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:14px;margin:0;font-weight:600;letter-spacing:.02em}
.bar{flex:1;min-width:180px;height:6px;background:#0d0f14;border-radius:3px;overflow:hidden}
.bar>div{height:100%;background:var(--accent);width:0}
.counts{display:flex;gap:10px;font-variant-numeric:tabular-nums;font-size:12px}
.counts b{font-weight:600}
.same{color:var(--same)} .clip{color:var(--clip)} .diff{color:var(--diff)} .skip{color:var(--skip)}
button{background:#252a35;color:var(--fg);border:1px solid var(--line);
       border-radius:6px;padding:5px 10px;font:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--accent)}
main{padding:16px;max-width:1500px;margin:0 auto}
.meta{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;
      color:var(--muted);font-size:12px;margin-bottom:2px}
.meta .name{color:var(--fg);font-size:13px;font-weight:600;
            word-break:break-all;font-family:ui-monospace,monospace}
.strip{width:100%;overflow-x:auto;background:#0d0f14;border:1px solid var(--line);
       border-radius:6px;margin-bottom:14px}
.strip img{display:block;width:100%;min-width:900px;image-rendering:auto}
.strip.zoom img{width:auto;max-width:none}
.strip.win{border-color:var(--accent);margin-bottom:14px}
.winlabel{color:var(--accent);font-size:11px;margin:-8px 0 4px;
          font-variant-numeric:tabular-nums}
.folder{color:var(--muted);font-size:12px;margin-bottom:10px}
.keys{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
.keys button{padding:8px 14px;font-size:13px;font-weight:600}
.k-same{border-color:var(--same)} .k-clip{border-color:var(--clip)}
.k-diff{border-color:var(--diff)} .k-skip{border-color:var(--skip)}
.cur{padding:4px 8px;border-radius:5px;font-weight:600;font-size:12px}
textarea{width:100%;background:#0d0f14;color:var(--fg);border:1px solid var(--line);
         border-radius:6px;padding:8px;font:inherit;resize:vertical;min-height:44px}
details{margin-top:14px;border-top:1px solid var(--line);padding-top:10px}
summary{cursor:pointer;color:var(--muted);font-size:12px}
table{border-collapse:collapse;font-size:12px;margin-top:8px;
      font-variant-numeric:tabular-nums}
td{padding:2px 14px 2px 0;color:var(--muted)}
td:first-child{color:var(--fg)}
kbd{background:#252a35;border:1px solid var(--line);border-radius:4px;
    padding:1px 5px;font:inherit;font-size:11px}
.help{color:var(--muted);font-size:12px;margin-top:14px;line-height:1.7}
.done{text-align:center;padding:60px 20px;color:var(--muted)}
</style>

<header>
  <h1>vid-fp ground truth</h1>
  <div class="bar"><div id="prog"></div></div>
  <div class="counts">
    <span id="pos"></span>
    <span class="same">SAME <b id="n-same">0</b></span>
    <span class="clip">CLIP <b id="n-clip">0</b></span>
    <span class="diff">DIFF <b id="n-diff">0</b></span>
    <span class="skip">SKIP <b id="n-skip">0</b></span>
  </div>
  <button id="filter">unlabeled only: on</button>
  <button id="export">export labels.csv</button>
  <button id="import">import</button>
  <input type="file" id="file" accept=".csv" hidden>
</header>

<main id="main"></main>

<script>
const POOL = __POOL__;
const STORE = "vidfp-gt-labels";
let labels = JSON.parse(localStorage.getItem(STORE) || "{}");
let onlyUnlabeled = true;
let idx = 0;
// With the filter on a labeled pair leaves the view immediately, so a misclick
// is otherwise unrecoverable without switching modes and hunting for it.
let undoStack = [];

const save = () => localStorage.setItem(STORE, JSON.stringify(labels));
const rel = p => (labels[p.pair_id] || {}).relation || "";
const visible = () => onlyUnlabeled ? POOL.filter(p => !rel(p)) : POOL;

function fmtDur(s){ s = +s; const m = Math.floor(s/60);
  return m ? `${m}:${String(Math.floor(s%60)).padStart(2,"0")}` : `${s.toFixed(2)}s`; }
function fmtSize(b){ b = +b; if(!b) return "";
  const u = ["B","KB","MB","GB"]; let i = 0;
  while(b >= 1024 && i < u.length-1){ b /= 1024; i++; }
  return b.toFixed(b<10 && i ? 1 : 0) + u[i]; }
const base = p => p.split("/").pop();

function side(p, s){
  const dur = p["duration_"+s], res = p["resolution_"+s], cod = p["codec_"+s];
  const smp = p["samples_"+s], sz = p["size_"+s];
  const from = p["match_from_"+s], to = p["match_to_"+s];
  // On a long file the whole-file strip cannot resolve a short insert, so the
  // reported window gets its own row underneath.
  const win = +p["window_"+s] ? `
    <div class="winlabel">matched window ${fmtDur(from)} &ndash; ${fmtDur(to)}</div>
    <div class="strip win" data-zoom><img loading="lazy" src="windows/${p.pair_id}_${s}.jpg" alt=""></div>` : "";
  return `<div class="meta">
      <span class="name">${base(p["path_"+s])}</span>
      <span>${fmtDur(dur)}</span><span>${res}</span><span>${cod}</span>
      <span>${fmtSize(sz)}</span>
      ${smp !== "" ? `<span>${smp} samples</span>` : ""}
    </div>
    <div class="strip" data-zoom><img loading="lazy" src="sheets/${p["sheet_"+s]}.jpg" alt=""></div>
    ${win}`;
}

function render(){
  const list = visible();
  const main = document.getElementById("main");
  const counts = {SAME:0, CLIP:0, DIFFERENT:0, SKIP:0};
  POOL.forEach(p => { const r = rel(p); if(counts[r] !== undefined) counts[r]++; });
  const labeled = POOL.filter(p => rel(p)).length;
  document.getElementById("n-same").textContent = counts.SAME;
  document.getElementById("n-clip").textContent = counts.CLIP;
  document.getElementById("n-diff").textContent = counts.DIFFERENT;
  document.getElementById("n-skip").textContent = counts.SKIP;
  document.getElementById("prog").style.width = (100*labeled/POOL.length).toFixed(1)+"%";
  document.getElementById("pos").textContent = `${labeled}/${POOL.length} labeled`;

  if(!list.length){
    main.innerHTML = `<div class="done"><p>Nothing left in this view.</p>
      <p>Every pair in the pool is labeled. Export the CSV.</p></div>`;
    return;
  }
  if(idx >= list.length) idx = list.length - 1;
  if(idx < 0) idx = 0;
  const p = list[idx];
  const cur = rel(p);
  const colour = {SAME:"var(--same)", CLIP:"var(--clip)",
                  DIFFERENT:"var(--diff)", SKIP:"var(--skip)"}[cur];

  main.innerHTML = `
    <div class="folder">
      pair ${idx+1} of ${list.length} in view &middot;
      ${p.same_folder === "yes" ? p.folder_a.split("/").pop()
        : "cross-folder: " + p.folder_a.split("/").pop() + " | " + p.folder_b.split("/").pop()}
      ${cur ? `&middot; <span class="cur" style="color:${colour}">${cur}</span>` : ""}
    </div>
    ${side(p,"a")}
    ${side(p,"b")}
    <div class="keys">
      <button class="k-same" data-set="SAME">S &mdash; same footage</button>
      <button class="k-clip" data-set="CLIP">C &mdash; one contains the other</button>
      <button class="k-diff" data-set="DIFFERENT">D &mdash; unrelated</button>
      <button class="k-skip" data-set="SKIP">X &mdash; unsure</button>
      <button data-nav="-1">&larr; prev</button>
      <button data-nav="1">next &rarr;</button>
      <button data-undo ${undoStack.length ? "" : "disabled"}>U &mdash; undo</button>
    </div>
    <textarea id="notes" placeholder="notes (optional)">${
      (labels[p.pair_id]||{}).notes || ""}</textarea>
    <details>
      <summary>why this pair is in the pool &mdash; open only after you have decided</summary>
      <table>
        <tr><td>sources</td><td>${p.sources}</td></tr>
        <tr><td>first proposed at</td><td>${p.vidfp_first_d ? "-d "+p.vidfp_first_d : "never by vid-fp"}</td></tr>
        <tr><td>vid-fp coverage</td><td>${p.vidfp_cov_a !== "" ? (100*p.vidfp_cov_a).toFixed(1)+"% / " : ""}${
              p.vidfp_cov_b !== "" ? (100*p.vidfp_cov_b).toFixed(1)+"%" : ""}</td></tr>
        <tr><td>oracle coverage</td><td>${p.oracle_cov_max !== "" ? (100*p.oracle_cov_max).toFixed(1)+"%" : "not proposed"}</td></tr>
      </table>
    </details>
    <div class="help">
      <kbd>S</kbd> same &middot; <kbd>C</kbd> clip &middot; <kbd>D</kbd> different &middot;
      <kbd>X</kbd> unsure &middot; <kbd>U</kbd> undo &middot;
      <kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate &middot;
      <kbd>Z</kbd> zoom strips &middot; <kbd>F</kbd> toggle filter<br>
      SAME = the same footage runs through both, end to end.
      CLIP = one file's footage is contained in the other.
      Judge from the frames, not from the coverage numbers &mdash; those are what
      you are measuring, so letting them decide the label would measure nothing.
    </div>`;

  main.querySelectorAll("[data-set]").forEach(b =>
    b.onclick = () => setRel(b.dataset.set));
  main.querySelectorAll("[data-nav]").forEach(b =>
    b.onclick = () => { commitNotes(); idx += +b.dataset.nav; render(); });
  main.querySelectorAll("[data-zoom]").forEach(d =>
    d.onclick = () => d.classList.toggle("zoom"));
  const ub = main.querySelector("[data-undo]");
  if(ub) ub.onclick = undo;
}

function commitNotes(){
  const ta = document.getElementById("notes");
  if(!ta) return;
  const p = visible()[idx];
  if(!p) return;
  const v = ta.value.trim();
  if(v || labels[p.pair_id]){
    labels[p.pair_id] = Object.assign({}, labels[p.pair_id], {notes: v});
    save();
  }
}

function setRel(r){
  const list = visible();
  const p = list[idx];
  if(!p) return;
  commitNotes();
  undoStack.push({pair_id: p.pair_id, prev: labels[p.pair_id], idx: idx});
  labels[p.pair_id] = Object.assign({}, labels[p.pair_id], {relation: r});
  save();
  // With the filter on, labeling removes the pair from the view, so the next
  // one slides into this index by itself.
  if(!onlyUnlabeled) idx++;
  render();
}

function undo(){
  const u = undoStack.pop();
  if(!u) return;
  if(u.prev === undefined) delete labels[u.pair_id];
  else labels[u.pair_id] = u.prev;
  save();
  // Land back on the restored pair rather than wherever the cursor drifted.
  const at = visible().findIndex(p => p.pair_id === u.pair_id);
  idx = at >= 0 ? at : u.idx;
  render();
}

document.addEventListener("keydown", e => {
  if(e.target.tagName === "TEXTAREA") return;
  const k = e.key.toLowerCase();
  if(k === "s") setRel("SAME");
  else if(k === "c") setRel("CLIP");
  else if(k === "d") setRel("DIFFERENT");
  else if(k === "x") setRel("SKIP");
  else if(k === "u"){ undo(); }
  else if(e.key === "ArrowRight"){ commitNotes(); idx++; render(); }
  else if(e.key === "ArrowLeft"){ commitNotes(); idx--; render(); }
  else if(k === "f"){ document.getElementById("filter").click(); }
  else if(k === "z"){ document.querySelectorAll("[data-zoom]").forEach(d => d.classList.toggle("zoom")); }
  else return;
  e.preventDefault();
});

document.getElementById("filter").onclick = () => {
  commitNotes();
  onlyUnlabeled = !onlyUnlabeled;
  document.getElementById("filter").textContent =
    "unlabeled only: " + (onlyUnlabeled ? "on" : "off");
  idx = 0;
  render();
};

document.getElementById("export").onclick = () => {
  commitNotes();
  const cols = __COLS__;
  const esc = v => {
    v = v === undefined || v === null ? "" : String(v);
    return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  };
  const lines = [cols.join(",")];
  for(const p of POOL){
    const l = labels[p.pair_id] || {};
    lines.push(cols.map(c =>
      esc(c === "relation" ? (l.relation || "")
        : c === "notes" ? (l.notes || "")
        : p[c])).join(","));
  }
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels.csv";
  a.click();
};

document.getElementById("import").onclick = () => document.getElementById("file").click();
document.getElementById("file").onchange = ev => {
  const f = ev.target.files[0];
  if(!f) return;
  f.text().then(txt => {
    // Minimal CSV read: enough for a file this tool wrote or a spreadsheet
    // handed back. Columns are located by name, never by position.
    const rows = [];
    let row = [], cell = "", q = false;
    for(let i = 0; i < txt.length; i++){
      const c = txt[i];
      if(q){
        if(c === '"' && txt[i+1] === '"'){ cell += '"'; i++; }
        else if(c === '"') q = false;
        else cell += c;
      } else if(c === '"') q = true;
      else if(c === ","){ row.push(cell); cell = ""; }
      else if(c === "\n"){ row.push(cell); rows.push(row); row = []; cell = ""; }
      else if(c !== "\r") cell += c;
    }
    if(cell || row.length){ row.push(cell); rows.push(row); }
    const head = rows.shift().map(s => s.trim());
    const ci = n => head.indexOf(n);
    const [pi, ri, ni] = [ci("pair_id"), ci("relation"), ci("notes")];
    if(pi < 0 || ri < 0){ alert("need pair_id and relation columns"); return; }
    let n = 0;
    for(const r of rows){
      if(!r[pi]) continue;
      const rl = (r[ri] || "").trim().toUpperCase();
      if(!rl && !(ni >= 0 && r[ni])) continue;
      labels[r[pi].trim()] = {relation: rl, notes: ni >= 0 ? (r[ni] || "").trim() : ""};
      n++;
    }
    save();
    idx = 0;
    render();
    alert(`imported ${n} labels`);
  });
};

render();
</script>
"""


def cmd_ui(args):
    pool_csv = os.path.join(args.out, "pool.csv")
    with open(pool_csv) as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames
        rows = list(rdr)
    windows = os.path.join(args.out, "windows")
    for r in rows:
        r["sheet_a"] = key_for(r["path_a"])
        r["sheet_b"] = key_for(r["path_b"])
        # The page cannot stat the disk, so resolve which window strips were
        # actually rendered here and hand the answer over as data.
        for s in ("a", "b"):
            name = f"{r['pair_id']}_{s}.jpg"
            r[f"window_{s}"] = 1 if os.path.exists(os.path.join(windows, name)) else 0
    rows.sort(key=lambda r: int(r["review_order"]))

    page = UI_TEMPLATE.replace("__POOL__", json.dumps(rows))
    page = page.replace("__COLS__", json.dumps(cols))
    out = os.path.join(args.out, "review.html")
    with open(out, "w") as fh:
        fh.write(page)
    print(f"ui: {len(rows)} pairs -> {out}")
    print(f"    open with:  xdg-open {os.path.abspath(out)}")


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------


def cmd_sweep(args):
    if not os.path.exists(args.bin):
        print(f"sweep: {args.bin} not found -- build it first", file=sys.stderr)
        return 1
    for d in args.distances:
        dest = os.path.join(args.out, f"vidfp_d{d}.json")
        r = run([args.bin, args.corpus, "-r", "-d", str(d), "-p", "1",
                 "-o", dest, "--yes"])
        if r.returncode not in (0, 2):
            print(f"sweep: -d {d} failed rc={r.returncode}\n{r.stderr}", file=sys.stderr)
            continue
        with open(dest) as fh:
            j = json.load(fh)
        pairs = set()
        for g in j.get("results", []):
            for f in g.get("files", []):
                for m in f.get("matches", []):
                    pairs.add(tuple(sorted((f["full_path"], m["full_path"]))))
        print(f"sweep: -d {d:>2} -> {len(pairs)} pairs")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--corpus", default="/home/daniel/Documents/AN")
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="run vid-fp across a -d ladder")
    s.add_argument("--bin", default=os.path.join(here, os.pardir,
                                                 "target/release/vid-fp"))
    # Stops at 12, one rung above the loosest profile the accuracy test uses.
    # Corroboration by any other pool source falls away past there -- 67% of
    # the pairs first seen at -d 2 are proposed by something else too, 6.9% at
    # -d 12 and 1.4% at -d 14, where 1267 pairs bought 18 corroborated ones.
    # That is the same trade Czkawka's -t 15 offered and it is not worth a
    # reviewer's afternoon. Pass --distances explicitly to go looser.
    s.add_argument("--distances", type=int, nargs="+",
                   default=[2, 4, 6, 8, 10, 12])
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("oracle", help="independent dense-sample proposer")
    s.set_defaults(func=cmd_oracle)

    s = sub.add_parser("pool", help="union all proposers -> pool.csv")
    s.add_argument("--extra", nargs="*",
                   help="CSVs with path_a,path_b columns from other tools")
    s.set_defaults(func=cmd_pool)

    s = sub.add_parser("sheets", help="render contact strips")
    s.set_defaults(func=cmd_sheets)

    s = sub.add_parser("ui", help="write review.html")
    s.set_defaults(func=cmd_ui)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"{tool} not on PATH", file=sys.stderr)
            return 1
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
