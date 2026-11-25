"""Simulate sampling rules on (seed, clip) pairs from out/survey.json.

Each rule sees ONE file's dense 64x64 track and returns the frame indices it
would hash; nothing about the other file. Per pair we report:
  same_moment  share of clip samples with a seed sample within 1 seed frame of
               the same moment (after the true offset, found by alignment)
  matched      share of clip samples whose hash is within D bits of some seed
               sample (anywhere: vid-fp's matching is offset-agnostic)
  found        matched >= GATE (the clip side clears -p)
"""
import bisect, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dense import dense64, vfp_hash, ham

V = "/home/daniel/Documents/VIDS/"
D = 4
GATE = 0.20


def keyframe_times(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "packet=pts_time,flags", "-of", "csv=p=0", path], capture_output=True, text=True).stdout.split()
    return sorted(float(a) for a, f in (x.split(",")[:2] for x in o) if "K" in f and a != "N/A")


class Track:
    def __init__(self, path):
        self.path = path
        self.t, self.g = dense64(path)
        n = min(len(self.t), len(self.g))
        self.t, self.g = self.t[:n], self.g[:n]
        self.h, self.e = vfp_hash(self.g)
        self.fps = 1 / np.median(np.diff(self.t)) if len(self.t) > 1 else 30.0
        self._kf = None

    def kf(self):
        if self._kf is None:
            self._kf = keyframe_times(self.path)
        return self._kf


def at_times(track, times):
    idx = np.searchsorted(track.t, np.asarray(times) - 1e-4)
    return np.unique(np.clip(idx, 0, len(track.t) - 1))


# ---- rules -----------------------------------------------------------------

def rule_kf(tr):
    return at_times(tr, tr.kf())


def diff_signal(tr):
    g = tr.g.reshape(len(tr.g), 16, 4, 16, 4).astype(np.float32).mean((2, 4))
    d = np.zeros(len(g), np.float32)
    d[1:] = np.abs(g[1:] - g[:-1]).mean((1, 2))
    return d


def rule_peaks(tr, W=1.0, min_rel=1.5):
    """Frames whose change from the previous frame is the largest within +-W
    seconds and at least min_rel times the window's median change."""
    d = diff_signal(tr)
    t = tr.t
    out = []
    lo = hi = 0
    for i in range(len(d)):
        while t[lo] < t[i] - W:
            lo += 1
        while hi < len(t) and t[hi] <= t[i] + W:
            hi += 1
        win = d[lo:hi]
        if d[i] > 0 and d[i] == win.max() and d[i] >= min_rel * np.median(win):
            out.append(i)
    return np.array(out, int)


def rule_all(tr):
    return np.arange(len(tr.t))


def rule_run(tr, M=18):
    k = rule_kf(tr)
    return np.unique(np.clip((k[:, None] + np.arange(M + 1)[None]).ravel(), 0, len(tr.t) - 1))


# name -> (rule for the seed, rule for the clip)
RULES = {"kf": (rule_kf, rule_kf), "oracle": (rule_all, rule_kf),
         "run9": (lambda t: rule_run(t, 9), lambda t: rule_run(t, 9)),
         "run12": (lambda t: rule_run(t, 12), lambda t: rule_run(t, 12)),
         "alt12": (lambda t: rule_alt(t, 12), lambda t: rule_alt(t, 12))}


def rule_alt(tr, M):
    """M frames after each keyframe, every second one: twice the reach for the
    same number of hashes (not the same decode cost unless they are non-ref)."""
    k = rule_kf(tr)
    return np.unique(np.clip((k[:, None] + 2 * np.arange(M + 1)[None]).ravel(), 0, len(tr.t) - 1))


# ---- evaluation ------------------------------------------------------------

def offset_of(seed, clip, s0):
    """Seed time = clip time + off. Search near the manifest's s0 on dense hashes."""
    ci = np.arange(0, len(clip.t), max(1, len(clip.t) // 60))
    best = None
    for off in np.arange(s0 - 6, s0 + 6, 1 / seed.fps):
        si = np.searchsorted(seed.t, clip.t[ci] + off - 0.5 / seed.fps)
        ok = si < len(seed.t)
        if ok.sum() < len(ci) // 2:
            continue
        dd = np.median(ham(clip.h[ci][ok], seed.h[si[ok]]))
        if best is None or dd < best[0]:
            best = (dd, off)
    return best[1] if best else s0, best[0] if best else 99


def evaluate(seed, clip, off, rule):
    si, ci = RULES[rule][0](seed), RULES[rule][1](clip)
    if len(ci) == 0:
        return dict(n_seed=len(si), n_clip=0, same_moment=0, matched=0, found=False)
    st = seed.t[si]
    same = 0
    for c in ci:
        j = bisect.bisect(st, clip.t[c] + off)
        near = min((abs(clip.t[c] + off - st[k]) for k in (j - 1, j) if 0 <= k < len(st)), default=9)
        same += near <= 1.0 / seed.fps
    sh = seed.h[si]
    dist = np.array([ham(clip.h[c], sh).min() for c in ci])
    live = clip.e[ci] >= 8
    matched = ((dist <= D) & live).sum() / max(1, len(ci))
    return dict(n_seed=len(si), n_clip=len(ci), same_moment=same / len(ci), matched=float(matched),
                found=bool(matched >= GATE))


def main():
    import re
    man = {}
    import csv
    for r in csv.DictReader(open(V + "manifest.csv")):
        man[r["file"]] = r
    survey = json.load(open("out/survey.json"))
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    rows = [s for s in survey if not only or s["seed"] in only]
    seeds = {}
    for s in sorted({r["seed"] for r in rows}):
        seeds[s] = Track(V + s + ".mkv")
    with ThreadPoolExecutor(6) as ex:
        clips = list(ex.map(lambda r: Track(V + r["file"]), rows))
    res = []
    for r, c in zip(rows, clips):
        sd = seeds[r["seed"]]
        s0 = float(re.match(r".+?@([\d.]+)-", man[r["file"]]["segments"]).group(1))
        off, q = offset_of(sd, c, s0 + sd.t[0])
        e = {k: evaluate(sd, c, off, k) for k in RULES}
        res.append(dict(file=r["file"], seed=r["seed"], transform=r["transform"], vidfp_found=r["found"],
                        off=off, align_bits=float(q), **{f"{k}_{m}": v for k, d in e.items() for m, v in d.items()}))
    json.dump(res, open("out/sim.json", "w"), indent=0)
    for k in RULES:
        f = sum(x[f"{k}_found"] for x in res)
        sm = np.mean([x[f"{k}_same_moment"] for x in res])
        ns = sum(x[f"{k}_n_seed"] for x in {x["seed"]: x for x in res}.values())
        nc = np.mean([x[f"{k}_n_clip"] / max(1e-9, 1) for x in res])
        print(f"{k:8} found {f}/{len(res)}  same-moment {sm:.2f}  seed samples {ns}  clip samples/pair {nc:.1f}")
    print(f"vid-fp found {sum(x['vidfp_found'] for x in res)}/{len(res)}")


if __name__ == "__main__":
    main()
