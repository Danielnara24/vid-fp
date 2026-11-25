#!/usr/bin/env python3
"""Generate the vid-fp corpus: every seed through every transformation.

Modelled on img-fp's benchmark/corpus/make_variants.py, with time where img-fp
has space. Read README.md beside this file for the ground-truth rules.

    python3 make_variants.py                      # generate what is missing
    python3 make_variants.py --only mirror,clip_short --seeds spring
    python3 make_variants.py --clean              # delete derived/ and start over
    python3 make_variants.py --pairs-only         # rebuild manifest.csv / pairs.csv

Deterministic: every amount is drawn from a hash of (seed name, transform,
what is being drawn), never from a clock or a global counter, so a file's
name, its bytes and its ground truth do not depend on which other transforms
or seeds exist. Nothing is random at call time.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

VIDS = os.environ.get("CORPUS_ROOT", "/home/daniel/Documents/VIDS")
DERIVED = os.path.join(VIDS, "derived")
MEZZ = os.path.join(VIDS, ".mezz")
TMP = os.path.join(VIDS, ".tmp")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MEZZ_SHORT = 540          # mezzanine short side cap
MEZZ_MAX_FPS = 31         # above this the mezzanine halves the rate


# --------------------------------------------------------------------------
# deterministic draws

def H(*parts):
    return int.from_bytes(hashlib.blake2b("/".join(map(str, parts)).encode(),
                                         digest_size=8).digest(), "big")


class Draw:
    """Every number a transform uses comes from here, keyed by what it is for."""

    def __init__(self, *base):
        self.base = "/".join(map(str, base))

    def u(self, name, lo, hi):
        return lo + (H(self.base, name) % 1_000_003) / 1_000_003 * (hi - lo)

    def i(self, name, lo, hi):            # inclusive
        return lo + H(self.base, name) % (hi - lo + 1)

    def pick(self, name, seq):
        return seq[H(self.base, name) % len(seq)]

    def sign(self, name):
        return 1 if H(self.base, name) % 2 else -1

    def seed(self, name):                 # for lavfi sources
        return H(self.base, name) % 2_000_000_000


# --------------------------------------------------------------------------
# seeds

@dataclass
class Seed:
    name: str
    path: str
    nm_set: str
    v0: float = 0.0            # first video frame, relative to the file's start
    dur: float = 0.0           # mezzanine video duration == seed timeline
    W: int = 0                 # mezzanine size
    Hh: int = 0
    fps: float = 30.0          # mezzanine rate
    src_codec: str = ""
    src_w: int = 0
    src_h: int = 0
    src_fps: float = 0.0
    keyframes: list = field(default_factory=list)   # seed keyframes, seed time

    @property
    def mezz(self):
        return os.path.join(MEZZ, self.name + ".mp4")

    @property
    def portrait(self):
        return self.Hh > self.W

    @property
    def short(self):
        return min(self.W, self.Hh)


def run(args, **kw):
    r = subprocess.run(args, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, args))[:600]}\n{r.stderr[-2000:]}")
    return r.stdout


def ffprobe_json(path, *extra):
    return json.loads(run(["ffprobe", "-v", "error", "-of", "json", *extra, path]))


def rate(s):
    a, b = (s or "0/1").split("/")
    return float(a) / float(b) if float(b) else 0.0


def video_span(path):
    """(first pts, end) of the video stream from its packets: no decoding."""
    d = ffprobe_json(path, "-select_streams", "v:0", "-show_entries",
                     "packet=pts_time,duration_time")
    pts = [(float(p["pts_time"]), float(p.get("duration_time") or 0))
           for p in d.get("packets", []) if p.get("pts_time") not in (None, "N/A")]
    if not pts:
        return 0.0, 0.0
    first = min(p for p, _ in pts)
    end = max(p + dd for p, dd in pts)
    return first, end


def load_seeds():
    rows = list(csv.DictReader(open(os.path.join(VIDS, "seeds.csv"))))
    seeds = []
    for r in rows:
        if not r["file"]:
            print(f"  seed {r['seed']} missing, skipped", file=sys.stderr)
            continue
        seeds.append(Seed(r["seed"], os.path.join(VIDS, r["file"]), r["near_miss_set"]))
    return seeds


def prepare(seed):
    """Probe a seed and build its mezzanine (cached, with a sidecar json)."""
    side = os.path.join(MEZZ, seed.name + ".json")
    if os.path.exists(side) and os.path.exists(seed.mezz) and \
            os.path.getmtime(side) >= os.path.getmtime(seed.path):
        seed.__dict__.update(json.load(open(side)))
        return seed
    os.makedirs(MEZZ, exist_ok=True)
    d = ffprobe_json(seed.path, "-show_entries",
                     "stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate:format=start_time")
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in d["streams"])
    fstart = float(d["format"].get("start_time") or 0)
    first, _ = video_span(seed.path)
    seed.v0 = max(0.0, first - fstart)
    seed.src_codec, seed.src_w, seed.src_h = v["codec_name"], v["width"], v["height"]
    seed.src_fps = rate(v["avg_frame_rate"]) or rate(v["r_frame_rate"])
    fps = seed.src_fps if seed.src_fps <= MEZZ_MAX_FPS else seed.src_fps / 2
    w, h = v["width"], v["height"]
    s = min(w, h)
    if s > MEZZ_SHORT:
        w, h = (round(w * MEZZ_SHORT / s / 2) * 2, round(h * MEZZ_SHORT / s / 2) * 2)
    w, h = w // 2 * 2, h // 2 * 2
    args = ["ffmpeg", "-v", "error", "-y"]
    if seed.v0 > 0.02:
        args += ["-ss", f"{seed.v0:.3f}"]
    args += ["-i", seed.path]
    if not has_audio:
        args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    args += ["-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0", "-shortest",
             "-vf", f"scale={w}:{h}:flags=bicubic,setsar=1,fps={fps:.6f}",
             "-c:v", "libx264", "-crf", "15", "-preset", "veryfast", "-g", "30",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
             seed.mezz + ".tmp.mp4"]
    run(args)
    os.replace(seed.mezz + ".tmp.mp4", seed.mezz)
    first_m, end_m = video_span(seed.mezz)
    seed.dur = end_m - first_m
    seed.W, seed.Hh, seed.fps = w, h, fps
    # keyframes of the SEED file itself, in seed time (for stream-copy cuts)
    kd = ffprobe_json(seed.path, "-select_streams", "v:0", "-skip_frame", "nokey",
                      "-show_entries", "frame=pts_time")
    seed.keyframes = sorted(float(f["pts_time"]) - fstart - seed.v0 for f in kd.get("frames", [])
                            if f.get("pts_time") not in (None, "N/A"))
    keep = {k: seed.__dict__[k] for k in
            ("v0", "dur", "W", "Hh", "fps", "src_codec", "src_w", "src_h", "src_fps", "keyframes")}
    json.dump(keep, open(side, "w"))
    return seed


# --------------------------------------------------------------------------
# what a transform hands back

@dataclass
class Job:
    seed: str
    transform: str
    group: str
    cmds: list                 # ffmpeg argument lists; {OUT} and {TMP} substituted
    ext: str
    segments: list             # [seed, s0, s1, out_t0] in seconds; speed applies
    expect: float              # expected output video duration
    speed: float = 1.0
    region: tuple = (0.0, 0.0, 1.0, 1.0)   # part of the seed frame still visible
    coverage: float = 1.0      # share of the output frame that is seed picture
    mirror: bool = False
    trap: str = ""             # scored separately: rotate_meta, speed, reverse, ...
    files: dict = field(default_factory=dict)   # name -> text, written into TMP
    note: str = ""


TRANSFORMS = {}     # name -> (group, fn)


def T(name, group):
    def deco(fn):
        TRANSFORMS[name] = (group, fn)
        return fn
    return deco


# --------------------------------------------------------------------------
# building blocks

class Ctx:
    def __init__(self, seed, tname, seeds_by_name):
        self.s = seed
        self.t = tname
        self.d = Draw(seed.name, tname)
        self.all = seeds_by_name

    # -- geometry ---------------------------------------------------------
    def out_short(self, choices=(240, 360, 360, 480, 480, 540)):
        ok = [c for c in choices if c <= self.s.short] or [self.s.short]
        return self.d.pick("res", ok)

    def scale_short(self, short, flags="bicubic"):
        if self.s.portrait:
            return f"scale={short}:-2:flags={flags},setsar=1"
        return f"scale=-2:{short}:flags={flags},setsar=1"

    def canvas(self, short):
        """Even output size with the seed's aspect at a given short side."""
        W, Hh = self.s.W, self.s.Hh
        if self.s.portrait:
            return short // 2 * 2, round(Hh * short / W / 2) * 2
        return round(W * short / Hh / 2) * 2, short // 2 * 2

    def crop(self, x0, y0, x1, y1):
        W, Hh = self.s.W, self.s.Hh
        cw, ch = int((x1 - x0) * W) // 2 * 2, int((y1 - y0) * Hh) // 2 * 2
        return f"crop={cw}:{ch}:{int(x0 * W)}:{int(y0 * Hh)}"

    # -- encoding ---------------------------------------------------------
    def gop(self, fps, choices=("scene", 1, 2, 4, 6)):
        g = self.d.pick("gop", list(choices))
        return [] if g == "scene" else ["-g", str(max(1, round(g * fps))), "-keyint_min", str(max(1, round(g * fps)))]

    def h264(self, crf=None, preset=None, fps=None, profile=None, gop=True):
        fps = fps or self.s.fps
        crf = crf if crf is not None else self.d.i("crf", 19, 26)
        preset = preset or self.d.pick("preset", ["superfast", "veryfast", "veryfast"])
        a = ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
        if profile:
            a += ["-profile:v", profile]
        return a + (self.gop(fps) if gop else [])

    def container(self, choices=("mp4", "mp4", "mp4", "mkv", "mkv", "mov")):
        return self.d.pick("container", list(choices))

    def audio(self, ext):
        return {
            "webm": ["-c:a", "libopus", "-b:a", "96k"],
            "mpg": ["-c:a", "mp2", "-b:a", "192k"],
            "ts": ["-c:a", "mp2", "-b:a", "192k"],
            "avi": ["-c:a", "libmp3lame", "-b:a", "128k"],
            "wmv": ["-c:a", "wmav2", "-b:a", "128k"],
            "ogv": ["-c:a", "libvorbis", "-q:a", "4"],
            "3gp": ["-c:a", "aac", "-b:a", "32k", "-ac", "1", "-ar", "22050"],
        }.get(ext, ["-c:a", "aac", "-b:a", "96k"])

    # -- assembling one ffmpeg call over the mezzanine --------------------
    def simple(self, vf, venc, ext, ss=None, t=None, af=None, extra_in=(), fc=None,
               out_opts=()):
        args = ["ffmpeg", "-v", "error", "-y"]
        if ss is not None:
            args += ["-ss", f"{ss:.3f}"]
        if t is not None:
            args += ["-t", f"{t:.3f}"]
        args += ["-i", self.s.mezz, *extra_in]
        if fc:
            args += ["-filter_complex", fc, "-map", "[v]", "-map", "0:a:0"]
        else:
            args += ["-map", "0:v:0", "-map", "0:a:0"]
            if vf:
                args += ["-vf", vf]
        if af:
            args += ["-af", af]
        args += [*venc, *self.audio(ext), *out_opts, "{OUT}"]
        return args

    def job(self, cmds, ext, segments, expect, **kw):
        return Job(self.s.name, self.t, TRANSFORMS[self.t][0], cmds, ext, segments, expect, **kw)

    def whole(self, vf, venc=None, ext=None, **kw):
        ext = ext or self.container()
        venc = venc if venc is not None else self.h264()
        return self.job([self.simple(vf, venc, ext)], ext,
                        [[self.s.name, 0.0, self.s.dur, 0.0]], self.s.dur, **kw)

    def clip_window(self, lo_frac, hi_frac, where="mid", min_len=3.0, name="clip"):
        D = self.s.dur
        L = max(min(min_len, D * 0.9), D * self.d.u(name + "_len", lo_frac, hi_frac))
        if where == "head":
            a = 0.0
        elif where == "tail":
            a = D - L
        else:
            a = self.d.u(name + "_at", 0.0, max(0.0, D - L))
        return round(a, 3), round(L, 3)

    def clip(self, vf, a, L, venc=None, ext=None, **kw):
        ext = ext or self.container()
        venc = venc if venc is not None else self.h264()
        return self.job([self.simple(vf, venc, ext, ss=a, t=L)], ext,
                        [[self.s.name, a, a + L, 0.0]], L, **kw)

    # -- text -------------------------------------------------------------
    def text(self, name, choices):
        return self.d.pick(name, choices)

    def corner_xy(self, name, margin="h*0.04"):
        c = self.d.pick(name, ["tl", "tr", "bl", "br"])
        x = margin if c[1] == "l" else f"w-tw-{margin}"
        y = margin if c[0] == "t" else f"h-th-{margin}"
        return x, y


WATERMARKS = ["clipzone", "@viralhub", "MEMEVAULT", "TV5 HD", "@reposts.daily", "Streamly",
              "HD CLIPS", "newsflash", "@funnyvids", "KANAL 7", "reuploaded", "@dailydose",
              "watchmore.tv", "CH 12", "TRENDING", "@best.moments"]
WORDS = ("the of and to in is you that it he was for on are as with his they at be this have from "
         "or one had by word but not what all were we when your can said there use an each which "
         "she do how their if will up other about out many then them these so some her would make "
         "like him into time has look two more write go see number no way could people my than "
         "first water been call who oil its now find long down day did get come made may part").split()


def words(d, name, n):
    return " ".join(d.pick(f"{name}{i}", WORDS) for i in range(n))


def card_source(c, name, dur, W, Hh, fps):
    """A title/ad card: seeded gradients plus text, distinct per file so two
    files never share a card (which would be real shared footage)."""
    title = words(c.d, name + "_w", c.d.i(name + "_n", 2, 4)).upper()
    fs = max(12, Hh // 12)
    src = (f"gradients=s={W}x{Hh}:r={fps:.6f}:d={dur:.3f}:seed={c.d.seed(name)}:"
           f"speed={c.d.u(name + '_sp', 0.005, 0.03):.4f}:nb_colors={c.d.i(name + '_nc', 2, 5)},"
           f"drawtext=fontfile={FONT}:text='{title}':fontcolor=white:fontsize={fs}:"
           f"x=(w-tw)/2:y=(h-th)/2:box=1:boxcolor=black@0.35:boxborderw={fs // 3},"
           f"fade=t=in:st=0:d=0.5,fade=t=out:st={max(0, dur - 0.5):.3f}:d=0.5,setsar=1,format=yuv420p")
    return src


# --------------------------------------------------------------------------
# concatenation: pieces of seeds and synthetic cards, frame-exact

def concat_job(c, pieces, W, Hh, fps, ext, venc=None, post="", **kw):
    """pieces: ("seed", seed_name, a, L) or ("card", name, L) or ("black", L).
    Every seed piece is cut with an accurate input seek, then scaled to fit the
    canvas (letterboxed if its aspect differs) and concatenated."""
    args = ["ffmpeg", "-v", "error", "-y"]
    chains, labels, segs = [], [], []
    t = 0.0
    n_in = 0
    for i, p in enumerate(pieces):
        if p[0] == "seed":
            sd = c.all[p[1]]
            a, L = p[2], p[3]
            args += ["-ss", f"{a:.3f}", "-t", f"{L:.3f}", "-i", sd.mezz]
            v = (f"[{n_in}:v]scale={W}:{Hh}:force_original_aspect_ratio=decrease:flags=bicubic,"
                 f"pad={W}:{Hh}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.6f},format=yuv420p,"
                 f"setpts=PTS-STARTPTS[v{i}]")
            au = f"[{n_in}:a]aresample=48000,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a{i}]"
            n_in += 1
            segs.append([sd.name, a, a + L, t])
        else:
            L = p[-1]
            if p[0] == "card":
                src = card_source(c, p[1], L, W, Hh, fps)
            else:
                src = f"color=c=black:s={W}x{Hh}:r={fps:.6f}:d={L:.3f},setsar=1,format=yuv420p"
            args += ["-f", "lavfi", "-i", src,
                     "-f", "lavfi", "-t", f"{L:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            v = f"[{n_in}:v]setpts=PTS-STARTPTS[v{i}]"
            au = f"[{n_in + 1}:a]asetpts=PTS-STARTPTS[a{i}]"
            n_in += 2
        chains += [v, au]
        labels.append(f"[v{i}][a{i}]")
        t += L
    fc = ";".join(chains) + ";" + "".join(labels) + f"concat=n={len(pieces)}:v=1:a=1[cv][a]"
    fc += f";[cv]{post or 'null'}[v]"
    venc = venc if venc is not None else c.h264(fps=fps)
    args += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", *venc, *c.audio(ext), "{OUT}"]
    return c.job([args], ext, segs, t, **kw)


# ==========================================================================
# THE CATALOGUE
# ==========================================================================

# ---- container / identity ------------------------------------------------

@T("copy_exact", "container")
def _(c):
    ext = os.path.splitext(c.s.path)[1].lstrip(".")
    return c.job([["cp", c.s.path, "{OUT}"]], ext, [[c.s.name, 0, c.s.dur, 0]], c.s.dur)


@T("remux_mp4", "container")
def _(c):
    tag = ["-tag:v", "hvc1"] if c.s.src_codec == "hevc" else []
    return c.job([["ffmpeg", "-v", "error", "-y", "-i", c.s.path, "-map", "0:v:0", "-map", "0:a?",
                   "-c", "copy", *tag, "{OUT}"]], "mp4", [[c.s.name, 0, c.s.dur, 0]], c.s.dur)


@T("remux_mov", "container")
def _(c):
    return c.job([["ffmpeg", "-v", "error", "-y", "-i", c.s.path, "-map", "0:v:0", "-map", "0:a?",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "{OUT}"]],
                 "mov" if c.s.src_codec in ("h264", "hevc") else "mp4",   # MOV refuses AV1 and VP9
                 [[c.s.name, 0, c.s.dur, 0]], c.s.dur)


@T("audio_strip", "container")
def _(c):
    return c.job([["ffmpeg", "-v", "error", "-y", "-i", c.s.path, "-map", "0:v:0", "-an",
                   "-c:v", "copy", "{OUT}"]], "mkv", [[c.s.name, 0, c.s.dur, 0]], c.s.dur)


@T("audio_swap", "container")
def _(c):
    src = c.d.pick("noise", ["brown", "pink", "white"])
    return c.job([["ffmpeg", "-v", "error", "-y", "-i", c.s.path, "-f", "lavfi",
                   "-i", f"anoisesrc=c={src}:a=0.05:r=48000:seed={c.d.seed('a')}",
                   "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-c:v", "copy",
                   "-c:a", "libopus", "-b:a", "64k", "{OUT}"]],
                 "mkv", [[c.s.name, 0, c.s.dur, 0]], c.s.dur)


@T("rotate_meta", "trap")
def _(c):
    # identical pixels, a display matrix saying "turn me": players show it sideways
    deg = c.d.pick("deg", [90, 270])
    return c.job([["ffmpeg", "-v", "error", "-y", "-display_rotation", str(deg), "-i", c.s.path,
                   "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "{OUT}"]],
                 "mkv", [[c.s.name, 0, c.s.dur, 0]], c.s.dur, trap="rotate_meta")


# ---- encoding ------------------------------------------------------------

@T("h264", "encoding")
def _(c):
    return c.whole(c.scale_short(c.out_short()), c.h264(crf=c.d.i("crf", 18, 28)))


@T("hevc", "encoding")
def _(c):
    ten = c.d.i("10bit", 0, 1)
    ext = c.d.pick("ct", ["mp4", "mkv"])
    v = ["-c:v", "libx265", "-crf", str(c.d.i("crf", 22, 30)), "-preset", "superfast",
         "-pix_fmt", "yuv420p10le" if ten else "yuv420p", "-x265-params", "log-level=error:pools=2:frame-threads=1"]
    if ext == "mp4":
        v += ["-tag:v", "hvc1"]
    return c.whole(c.scale_short(c.out_short()), v, ext, note="10-bit" if ten else "")


@T("vp9", "encoding")
def _(c):
    v = ["-c:v", "libvpx-vp9", "-crf", str(c.d.i("crf", 30, 42)), "-b:v", "0", "-deadline", "realtime",
         "-cpu-used", "8", "-row-mt", "1", "-pix_fmt", "yuv420p", *c.gop(c.s.fps, ("scene", 2, 4, 8))]
    return c.whole(c.scale_short(c.out_short()), v, "webm")


@T("av1", "encoding")
def _(c):
    v = ["-c:v", "libsvtav1", "-crf", str(c.d.i("crf", 30, 45)), "-preset", "11",
         "-pix_fmt", "yuv420p", "-svtav1-params", "verbose=0:lp=2"]
    return c.whole(c.scale_short(c.out_short()), v, c.d.pick("ct", ["mkv", "mp4"]))


@T("mpeg2_mpg", "encoding")
def _(c):
    v = ["-c:v", "mpeg2video", "-q:v", str(c.d.i("q", 3, 8)), "-g", str(c.d.pick("g", [12, 15, 18])),
         "-bf", "2", "-pix_fmt", "yuv420p"]
    return c.whole(c.scale_short(c.out_short((360, 480, 576))), v, "mpg")


@T("xvid_avi", "encoding")
def _(c):
    v = ["-c:v", "libxvid", "-q:v", str(c.d.i("q", 3, 7)), "-g", str(c.d.pick("g", [60, 120, 250])),
         "-pix_fmt", "yuv420p"]
    return c.whole(c.scale_short(c.out_short((240, 360, 480))), v, "avi")


@T("wmv", "encoding")
def _(c):
    v = ["-c:v", "wmv2", "-b:v", f"{c.d.i('br', 600, 1800)}k", "-g", "120", "-pix_fmt", "yuv420p"]
    return c.whole(c.scale_short(c.out_short((240, 360, 480))), v, "wmv")


@T("theora_ogv", "encoding")
def _(c):
    v = ["-c:v", "libtheora", "-q:v", str(c.d.i("q", 4, 8)), "-pix_fmt", "yuv420p"]
    return c.whole(c.scale_short(c.out_short((240, 360, 480))), v, "ogv")


@T("phone_3gp", "encoding")
def _(c):
    v = c.h264(crf=c.d.i("crf", 26, 32), preset="veryfast", profile="baseline")
    return c.whole(c.scale_short(144) + ",fps=15", v, "3gp")


@T("low_bitrate", "encoding")
def _(c):
    br = c.d.i("br", 120, 400)
    v = ["-c:v", "libx264", "-b:v", f"{br}k", "-maxrate", f"{br * 2}k", "-bufsize", f"{br * 2}k",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", *c.gop(c.s.fps)]
    return c.whole(c.scale_short(c.out_short((360, 480))), v, note=f"{br} kbps")


@T("generations", "encoding")
def _(c):
    ext = c.container()
    cmds, src = [], c.s.mezz
    shorts = [c.out_short((480, 540, 720)), c.d.pick("s2", [360, 480]), c.d.pick("s3", [360, 432, 480])]
    for g in range(3):
        out = "{OUT}" if g == 2 else "{TMP}/gen%d.mp4" % g
        cmds.append(["ffmpeg", "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-map", "0:a:0",
                     "-vf", c.scale_short(min(shorts[g], c.s.short)),
                     "-c:v", "libx264", "-crf", str(c.d.i(f"crf{g}", 25, 31)), "-preset", "veryfast",
                     "-pix_fmt", "yuv420p", *c.audio(ext if g == 2 else "mp4"), out])
        src = out
    return c.job(cmds, ext, [[c.s.name, 0, c.s.dur, 0]], c.s.dur)


@T("long_gop", "encoding")
def _(c):
    g = c.d.u("g", 8, 15)
    v = ["-c:v", "libx264", "-crf", str(c.d.i("crf", 20, 26)), "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-g", str(round(g * c.s.fps)), "-keyint_min", str(round(g * c.s.fps)), "-sc_threshold", "0"]
    return c.whole(c.scale_short(c.out_short()), v, note=f"keyframe every {g:.1f} s")


@T("all_intra", "encoding")
def _(c):
    if c.d.i("kind", 0, 1):
        v = ["-c:v", "mjpeg", "-q:v", str(c.d.i("q", 4, 9)), "-pix_fmt", "yuvj420p"]
        return c.whole(c.scale_short(min(360, c.s.short)), v, "avi", note="mjpeg")
    v = ["-c:v", "prores_ks", "-profile:v", "0", "-pix_fmt", "yuv422p10le"]
    return c.whole(c.scale_short(min(360, c.s.short)), v, "mov", note="prores proxy")


@T("fps_change", "encoding")
def _(c):
    choices = [f for f in (15, 24, 25, 30, 50, 60) if abs(f - c.s.fps) > 1.5]
    f = c.d.pick("fps", choices)
    return c.whole(c.scale_short(c.out_short()) + f",fps={f}", c.h264(fps=f), note=f"{f} fps")


@T("interlaced_ts", "encoding")
def _(c):
    mp2 = c.d.i("codec", 0, 1)
    vf = c.scale_short(min(576, c.s.short)) + ",fps=50,tinterlace=interleave_top,setfield=tff"
    if mp2:
        v = ["-c:v", "mpeg2video", "-q:v", "4", "-flags", "+ilme+ildct", "-top", "1", "-g", "12"]
    else:
        v = ["-c:v", "libx264", "-crf", "22", "-preset", "veryfast", "-flags", "+ilme+ildct",
             "-x264-params", "tff=1", "-pix_fmt", "yuv420p"]
    return c.whole(vf, v, "ts", note="mpeg2 576i" if mp2 else "h264 interlaced")


@T("frame_drop", "encoding")
def _(c):
    k = c.d.i("k", 5, 9)
    vf = c.scale_short(c.out_short()) + f",select='not(eq(mod(n\\,{k})\\,{k // 2}))'"
    return c.whole(vf, c.h264() + ["-fps_mode", "vfr"], c.d.pick("ct", ["mkv", "mp4"]),
                   note=f"every {k}th frame dropped, vfr")


# ---- resolution ----------------------------------------------------------

@T("res_144", "resolution")
def _(c):
    return c.whole(c.scale_short(144, "area"), c.h264(crf=c.d.i("crf", 22, 30)))


@T("res_low", "resolution")
def _(c):
    s = c.d.pick("s", [180, 240, 288, 360])
    return c.whole(c.scale_short(s, "area"), c.h264(), note=f"{s}p")


@T("res_hd", "resolution")
def _(c):
    return c.whole(c.scale_short(1080, "lanczos"), c.h264(crf=c.d.i("crf", 20, 24)),
                   note="above the mezzanine: an upscale for most seeds")


@T("upscale", "resolution")
def _(c):
    lo = c.d.pick("lo", [240, 360])
    vf = c.scale_short(lo, "area") + "," + c.scale_short(720, "bicubic")
    return c.whole(vf, c.h264(), note=f"from {lo}p")


@T("aspect_squeeze", "resolution")
def _(c):
    f = c.d.pick("dir", [c.d.u("lo", 0.72, 0.9), c.d.u("hi", 1.12, 1.35)])
    w, h = c.canvas(c.out_short((360, 480, 540)))
    return c.whole(f"scale={round(w * f / 2) * 2}:{h},setsar=1", c.h264(), note=f"width x{f:.2f}")


@T("letterbox", "resolution")
def _(c):
    a = c.s.W / c.s.Hh
    W, Hh = (640, 480) if a > 1.5 else (854, 480)
    cov = min(a / (W / Hh), (W / Hh) / a)
    vf = (f"scale={W}:{Hh}:force_original_aspect_ratio=decrease,pad={W}:{Hh}:(ow-iw)/2:(oh-ih)/2,"
          "setsar=1")
    return c.whole(vf, c.h264(), coverage=cov, note=f"into {W}x{Hh}")


@T("border_frame", "resolution")
def _(c):
    f = c.d.u("f", 0.78, 0.92)
    w, h = c.canvas(c.out_short((480, 540, 720)))
    iw, ih = round(w * f / 2) * 2, round(h * f / 2) * 2
    col = c.d.pick("col", ["black", "white", "0x1e1e1e", "0x2b4c7e", "0x7e2b2b", "0xd8c8a8"])
    vf = f"scale={iw}:{ih},pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={col},setsar=1"
    return c.whole(vf, c.h264(), coverage=f * f)


@T("vertical_reframe", "resolution")
def _(c):
    W, Hh = c.s.W, c.s.Hh
    if c.s.portrait:      # a vertical seed is reframed square
        ch = W
        y = (Hh - ch) / 2 + c.d.u("off", -0.5, 0.5) * (Hh - ch) / 2
        y = int(max(0, min(Hh - ch, y))) // 2 * 2
        reg = (0.0, y / Hh, 1.0, (y + ch) / Hh)
        vf = f"crop={W}:{ch}:0:{y},scale=720:720,setsar=1"
    else:
        cw = int(Hh * 9 / 16) // 2 * 2
        x = (W - cw) / 2 + c.d.u("off", -0.6, 0.6) * (W - cw) / 2
        x = int(max(0, min(W - cw, x))) // 2 * 2
        reg = (x / W, 0.0, (x + cw) / W, 1.0)
        vf = f"crop={cw}:{Hh}:{x}:0,scale=-2:{min(1280, Hh * 2 // 2 * 2)},setsar=1"
    return c.whole(vf, c.h264(), region=reg)


def blurfill_vf(c, short_out):
    if c.s.portrait:
        W, Hh = round(short_out * 16 / 9 / 2) * 2, short_out
    else:
        W, Hh = short_out, round(short_out * 16 / 9 / 2) * 2
    a_s, a_c = c.s.W / c.s.Hh, W / Hh
    cov = min(a_s / a_c, a_c / a_s)
    q = (W // 8 * 2, Hh // 8 * 2)
    fc = (f"[0:v]split[a][b];[a]scale={q[0]}:{q[1]}:force_original_aspect_ratio=increase,crop={q[0]}:{q[1]},"
          f"gblur=sigma={c.d.u('blur', 18, 40) / 4:.1f},eq=brightness={c.d.u('dim', -0.15, -0.03):.3f},"
          f"scale={W}:{Hh}[bg];"
          f"[b]scale={W}:{Hh}:force_original_aspect_ratio=decrease[fg];"
          f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")
    return fc, cov


@T("blur_fill", "resolution")
def _(c):
    fc, cov = blurfill_vf(c, c.d.pick("w", [540, 720]))
    ext = c.container()
    return c.job([c.simple(None, c.h264(), ext, fc=fc + "[v]")], ext,
                 [[c.s.name, 0, c.s.dur, 0]], c.s.dur, coverage=cov)


# ---- crop and geometry ----------------------------------------------------

def light_crop_box(c, lo=0.02, hi=0.10):
    return (c.d.u("l", lo, hi), c.d.u("t", lo, hi), 1 - c.d.u("r", lo, hi), 1 - c.d.u("b", lo, hi))


@T("crop_light", "crop")
def _(c):
    box = light_crop_box(c)
    return c.whole(c.crop(*box) + "," + c.scale_short(c.out_short()), c.h264(), region=box)


@T("crop_zoom", "crop")
def _(c):
    z = c.d.u("z", 1.08, 1.25)
    m = (1 - 1 / z) / 2
    dx = c.d.u("dx", -m, m) * 0.6
    dy = c.d.u("dy", -m, m) * 0.6
    box = (m + dx, m + dy, 1 - m + dx, 1 - m + dy)
    return c.whole(c.crop(*box) + "," + c.scale_short(c.out_short()), c.h264(), region=box,
                   note=f"zoom {z:.2f}")


def tilt_vf(c, deg, fill=True):
    """Rotate at the output size (cheap), zooming first when filling."""
    a = math.radians(deg)
    r = max(c.s.W / c.s.Hh, c.s.Hh / c.s.W)
    z = (math.cos(abs(a)) + r * math.sin(abs(a))) if fill else 1.0
    w, h = c.canvas(c.out_short())
    vf = (f"scale={round(w * z / 2) * 2}:{round(h * z / 2) * 2},"
          f"rotate={a:.5f}:ow={w}:oh={h}:c=black:bilinear=1,setsar=1")
    m = (1 - 1 / z) / 2
    return vf, (m, m, 1 - m, 1 - m)


@T("tilt", "geometry")
def _(c):
    deg = c.d.sign("dir") * c.d.u("deg", 1.0, 5.0)
    vf, box = tilt_vf(c, deg)
    return c.whole(vf, c.h264(), region=box, note=f"{deg:+.1f} deg, zoomed to fill")


@T("tilt_corners", "geometry")
def _(c):
    deg = c.d.sign("dir") * c.d.u("deg", 1.0, 4.0)
    vf, box = tilt_vf(c, deg, fill=False)
    cov = 1 - abs(math.sin(math.radians(deg))) * 1.5
    return c.whole(vf, c.h264(), coverage=cov, note=f"{deg:+.1f} deg, black corners")


@T("shaky", "geometry")
def _(c):
    f1, f2 = c.d.u("f1", 0.7, 2.3), c.d.u("f2", 0.5, 1.9)
    vf = (f"crop=w=iw*0.9:h=ih*0.9:x=(iw-ow)/2*(1+0.85*sin(t*{f1:.3f})):"
          f"y=(ih-oh)/2*(1+0.85*sin(t*{f2:.3f}+1.3)),")
    return c.whole(vf + c.scale_short(c.out_short()), c.h264(), region=(0.05, 0.05, 0.95, 0.95),
                   note="handheld drift, 90% window")


@T("screen_filmed", "geometry")
def _(c):
    W, Hh = c.canvas(c.out_short((360, 480, 540)))
    PW, PH = round(W * 1.25 / 2) * 2, round(Hh * 1.25 / 2) * 2
    x0, y0 = (PW - W) / 2, (PH - Hh) / 2
    j = lambda n, s: c.d.u(n, -1, 1) * s
    k = c.d.u("key", 0.02, 0.07) * W
    side = c.d.pick("side", ["l", "r", "t", "b"])
    pts = [[x0 + j("a", 8), y0 + j("b", 8)], [x0 + W + j("c", 8), y0 + j("d", 8)],
           [x0 + j("e", 8), y0 + Hh + j("f", 8)], [x0 + W + j("g", 8), y0 + Hh + j("h", 8)]]
    if side == "l":
        pts[0][1] += k; pts[2][1] -= k
    elif side == "r":
        pts[1][1] += k; pts[3][1] -= k
    elif side == "t":
        pts[0][0] += k; pts[1][0] -= k
    else:
        pts[2][0] += k; pts[3][0] -= k
    p = ":".join(f"{v:.1f}" for pt in pts for v in pt)
    vf = (f"scale={W}:{Hh},setsar=1,pad={PW}:{PH}:{x0:.0f}:{y0:.0f}:color=0x0c0c0c,perspective={p}:sense=destination,"
          f"gblur=sigma={c.d.u('bl', 0.6, 1.6):.2f},eq=contrast={c.d.u('ct', 0.8, 0.95):.2f}:"
          f"saturation={c.d.u('sat', 0.7, 0.95):.2f},colortemperature={c.d.i('K', 4500, 8500)},"
          f"noise=alls={c.d.i('n', 5, 12)}:allf=t,vignette=PI/5")
    return c.whole(vf, c.h264(), coverage=(1 / 1.25) ** 2, note=f"keystone {side}")


@T("rotate90_pixels", "trap")
def _(c):
    t = c.d.pick("t", [1, 2])
    return c.whole(c.scale_short(c.out_short()) + f",transpose={t}", c.h264(), trap="rotate90")


@T("flip_vertical", "trap")
def _(c):
    return c.whole(c.scale_short(c.out_short()) + ",vflip", c.h264(), trap="vflip")


# ---- mirror (a recall target in its own right) ---------------------------

@T("mirror", "mirror")
def _(c):
    return c.whole(c.scale_short(c.out_short()) + ",hflip", c.h264(), mirror=True)


@T("mirror_crop", "mirror")
def _(c):
    box = light_crop_box(c, 0.03, 0.12)
    return c.whole(c.crop(*box) + ",hflip," + c.scale_short(c.out_short()), c.h264(),
                   mirror=True, region=box)


@T("mirror_lowres", "mirror")
def _(c):
    br = c.d.i("br", 200, 500)
    v = ["-c:v", "libx264", "-b:v", f"{br}k", "-preset", "veryfast", "-pix_fmt", "yuv420p", *c.gop(c.s.fps)]
    return c.whole("hflip," + c.scale_short(c.d.pick("s", [240, 360]), "area"), v, mirror=True)


@T("mirror_clip", "mirror")
def _(c):
    a, L = c.clip_window(0.2, 0.5)
    return c.clip(c.scale_short(c.out_short()) + ",hflip", a, L, mirror=True)


@T("mirror_colour", "mirror")
def _(c):
    vf = (f"hflip,eq=brightness={c.d.u('b', -0.08, 0.1):.3f}:saturation={c.d.u('s', 0.7, 1.4):.2f}:"
          f"contrast={c.d.u('c', 0.85, 1.2):.2f},hue=h={c.d.u('h', -15, 15):.1f}")
    return c.whole(c.scale_short(c.out_short()) + "," + vf, c.h264(), mirror=True)


@T("mirror_watermark", "mirror")
def _(c):
    x, y = c.corner_xy("corner")
    vf = (c.scale_short(c.out_short()) + f",hflip,drawtext=fontfile={FONT}:text='{c.text('wm', WATERMARKS)}':"
          f"fontcolor=white@0.7:fontsize=h*0.06:x={x}:y={y}:shadowx=2:shadowy=2")
    return c.whole(vf, c.h264(), mirror=True)


@T("mirror_vertical", "mirror")
def _(c):
    fc, cov = blurfill_vf(c, 540)
    ext = c.container()
    return c.job([c.simple(None, c.h264(), ext, fc="[0:v]hflip[m];" + fc.replace("[0:v]", "[m]") + "[v]")],
                 ext, [[c.s.name, 0, c.s.dur, 0]], c.s.dur, coverage=cov, mirror=True)


@T("mirror_tilt", "mirror")
def _(c):
    deg = c.d.sign("dir") * c.d.u("deg", 1.0, 3.5)
    vf, box = tilt_vf(c, deg)
    return c.whole(vf + ",hflip", c.h264(), region=box, mirror=True)


@T("mirror_reencode", "mirror")
def _(c):
    v = ["-c:v", "libx265", "-crf", str(c.d.i("crf", 24, 30)), "-preset", "superfast", "-pix_fmt", "yuv420p",
         "-tag:v", "hvc1", "-x265-params", "log-level=error:pools=2:frame-threads=1"]
    return c.whole(c.scale_short(c.out_short()) + ",hflip", v, "mp4", mirror=True)


# ---- colour and light -----------------------------------------------------

def colour(name, vf_fn):
    @T(name, "colour")
    def _(c):
        return c.whole(c.scale_short(c.out_short()) + "," + vf_fn(c), c.h264())


colour("brightness", lambda c: f"eq=brightness={c.d.sign('d') * c.d.u('b', 0.07, 0.2):.3f}")
colour("contrast", lambda c: f"eq=contrast={c.d.pick('d', [c.d.u('lo', 0.55, 0.85), c.d.u('hi', 1.2, 1.6)]):.3f}")
colour("gamma", lambda c: f"eq=gamma={c.d.pick('d', [c.d.u('lo', 0.55, 0.8), c.d.u('hi', 1.25, 1.7)]):.3f}")
colour("saturation", lambda c: f"eq=saturation={c.d.pick('d', [c.d.u('lo', 0.25, 0.6), c.d.u('hi', 1.4, 2.2)]):.3f}")
colour("greyscale", lambda c: "hue=s=0")
colour("sepia", lambda c: "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131")
colour("temperature", lambda c: f"colortemperature={c.d.pick('d', [c.d.i('lo', 2800, 4500), c.d.i('hi', 7500, 11000)])}")
colour("hue_shift", lambda c: f"hue=h={c.d.sign('d') * c.d.u('h', 15, 60):.1f}")
colour("grade", lambda c: "curves=preset=" + c.d.pick("p", ["vintage", "cross_process", "strong_contrast",
                                                             "lighter", "darker", "increase_contrast",
                                                             "medium_contrast"]))
colour("range_mismatch", lambda c: c.d.pick("d", ["lutyuv=y=16+val*219/255:u=128+(val-128)*224/255:v=128+(val-128)*224/255",
                                                  "lutyuv=y=clip((val-16)*255/219\\,0\\,255)"]))
colour("lighting_flicker", lambda c: (f"eq=brightness='{c.d.u('a', 0.05, 0.15):.3f}*sin(2*PI*t/{c.d.u('T', 1.5, 8):.2f})'"
                                      ":eval=frame"))
colour("lighting_ramp", lambda c: (f"eq=gamma='1+{c.d.sign('d') * c.d.u('a', 0.25, 0.5):.3f}*t/{c.s.dur:.2f}'"
                                   ":eval=frame,vignette=PI/4"))
colour("grain", lambda c: f"noise=alls={c.d.i('n', 10, 28)}:allf=t+u")
colour("blur", lambda c: f"gblur=sigma={c.d.u('s', 1.5, 4.0):.2f}")
colour("sharpen", lambda c: f"unsharp=5:5:{c.d.u('a', 0.8, 2.0):.2f}")
colour("vignette", lambda c: f"vignette=angle={c.d.u('a', 0.6, 1.05):.3f}")


# ---- overlays -------------------------------------------------------------

@T("watermark_corner", "overlay")
def _(c):
    x, y = c.corner_xy("corner")
    vf = (c.scale_short(c.out_short()) + f",drawtext=fontfile={FONT}:text='{c.text('wm', WATERMARKS)}':fontcolor=white@{c.d.u('al', 0.5, 0.85):.2f}:"
          f"fontsize=h*{c.d.u('fs', 0.045, 0.08):.3f}:x={x}:y={y}:box={c.d.i('box', 0, 1)}:boxcolor=black@0.4:"
          f"boxborderw=8")
    return c.whole(vf, c.h264())


@T("watermark_bounce", "overlay")
def _(c):
    sx, sy = c.d.u("sx", 0.05, 0.12), c.d.u("sy", 0.04, 0.1)
    vf = (c.scale_short(c.out_short()) + f",drawtext=fontfile={FONT}:text='{c.text('wm', WATERMARKS)}':fontcolor=white@0.55:fontsize=h*0.06:"
          f"x=(w-tw)*abs(mod(t*{sx:.3f}\\,2)-1):y=(h-th)*abs(mod(t*{sy:.3f}+0.5\\,2)-1)")
    return c.whole(vf, c.h264())


@T("channel_bug_clock", "overlay")
def _(c):
    x, y = c.corner_xy("corner")
    vf = (c.scale_short(c.out_short()) + f",drawtext=fontfile={FONT}:text='{c.text('wm', WATERMARKS)}':fontcolor=white@0.8:fontsize=h*0.05:"
          f"x={x}:y={y},drawtext=fontfile={FONT_REG}:text='%{{pts\\:hms}}':fontcolor=white:fontsize=h*0.04:"
          f"x=w-tw-h*0.04:y=h-th-h*0.04:box=1:boxcolor=black@0.5:boxborderw=6")
    return c.whole(vf, c.h264())


@T("subtitles", "overlay")
def _(c):
    lines, t, i = [], 0.0, 0
    while t < c.s.dur:
        d = c.d.u(f"d{i}", 1.8, 4.2)
        txt = words(c.d, f"w{i}_", c.d.i(f"n{i}", 3, 9))
        lines.append(f"{i + 1}\n{srt_t(t)} --> {srt_t(min(c.s.dur, t + d - 0.15))}\n{txt.capitalize()}\n")
        t += d
        i += 1
    ext = c.container()
    fs = c.d.i("fs", 16, 24)
    vf = (c.scale_short(c.out_short()) + f",subtitles=filename={{TMP}}/s.srt:"
          f"force_style='FontName=DejaVu Sans,FontSize={fs},Outline=2,MarginV=20'")
    j = c.job([c.simple(vf, c.h264(), ext)], ext, [[c.s.name, 0, c.s.dur, 0]], c.s.dur)
    j.files = {"s.srt": "\n".join(lines)}
    return j


def srt_t(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


@T("lower_third", "overlay")
def _(c):
    col = c.d.pick("col", ["0x0b3d91", "0xb00020", "0x1b5e20", "0x222222", "0xe65100"])
    title = words(c.d, "t", c.d.i("n", 3, 6)).upper()
    y = c.d.u("y", 0.72, 0.82)
    vf = (c.scale_short(c.out_short()) + f",drawbox=x=0:y=ih*{y:.3f}:w=iw:h=ih*0.13:color={col}@0.85:t=fill,"
          f"drawtext=fontfile={FONT}:text='{title}':fontcolor=white:fontsize=h*0.05:x=w*0.04:y=h*{y + 0.035:.3f}")
    return c.whole(vf, c.h264(), coverage=0.87)


@T("tiled_watermark", "overlay")
def _(c):
    wm = c.text("wm", WATERMARKS)
    al = c.d.u("al", 0.18, 0.35)
    parts = [f"drawtext=fontfile={FONT}:text='{wm}':fontcolor=white@{al:.2f}:fontsize=h*0.05:"
             f"x=w*{0.05 + 0.33 * gx + 0.08 * (gy % 2):.3f}:y=h*{0.1 + 0.3 * gy:.3f}"
             for gx in range(3) for gy in range(3)]
    return c.whole(c.scale_short(c.out_short()) + "," + ",".join(parts), c.h264())


@T("reaction", "overlay")
def _(c):
    W, Hh = c.canvas(c.out_short((540, 720)))
    f = c.d.u("f", 0.55, 0.7)
    sw, sh = round(W * f / 2) * 2, round(Hh * f / 2) * 2
    right = c.d.i("side", 0, 1)
    sx = W - sw - W // 30 if right else W // 30
    cw, ch = round(W * 0.28 / 2) * 2, round(Hh * 0.34 / 2) * 2
    cx = W // 30 if right else W - cw - W // 30
    fc = (f"gradients=s={W}x{Hh}:r={c.s.fps:.6f}:seed={c.d.seed('bg')}:speed=0.01:nb_colors=2,setsar=1[bg];"
          f"life=s={cw // 4}x{ch // 4}:r={c.s.fps:.6f}:seed={c.d.seed('cam')}:ratio=0.3:mold=8:"
          f"life_color=0xd09a70:death_color=0x303030,scale={cw}:{ch}:flags=neighbor,setsar=1[cam];"
          f"[0:v]scale={sw}:{sh},setsar=1[sd];[bg][sd]overlay={sx}:(H-h)/2:shortest=1[t];"
          f"[t][cam]overlay={cx}:H*0.12:shortest=1,drawtext=fontfile={FONT}:text='REACTING':fontcolor=white:"
          f"fontsize=h*0.05:x={cx}:y=H*0.12+{ch}+10[v]")
    ext = c.container()
    return c.job([c.simple(None, c.h264(), ext, fc=fc)], ext, [[c.s.name, 0, c.s.dur, 0]], c.s.dur,
                 coverage=f * f)


@T("pip", "overlay")
def _(c):
    W, Hh = c.canvas(c.out_short((540, 720)))
    f = c.d.u("f", 0.3, 0.45)
    sw, sh = round(W * f / 2) * 2, round(Hh * f / 2) * 2
    corner = c.d.pick("corner", ["tl", "tr", "bl", "br"])
    x = W // 25 if corner[1] == "l" else W - sw - W // 25
    y = Hh // 25 if corner[0] == "t" else Hh - sh - Hh // 25
    fc = (f"gradients=s={W}x{Hh}:r={c.s.fps:.6f}:seed={c.d.seed('host')}:speed=0.02:nb_colors=4,"
          f"drawtext=fontfile={FONT}:text='{words(c.d, 'h', 3).upper()}':fontcolor=white:fontsize=h*0.08:"
          f"x=(w-tw)/2:y=(h-th)/2,setsar=1[bg];[0:v]scale={sw}:{sh},setsar=1[sd];"
          f"[bg][sd]overlay={x}:{y}:shortest=1[v]")
    ext = c.container()
    return c.job([c.simple(None, c.h264(), ext, fc=fc)], ext, [[c.s.name, 0, c.s.dur, 0]], c.s.dur,
                 coverage=f * f)


@T("screen_recording", "overlay")
def _(c):
    W, Hh = c.canvas(c.out_short((540, 720)))
    if c.s.portrait:
        W, Hh = round(Hh * 16 / 9 / 2) * 2, Hh
    sh = round(Hh * 0.78 / 2) * 2
    D = c.s.dur
    title = words(c.d, "t", 4)
    fc = (f"color=c=0x181818:s={W}x{Hh}:r={c.s.fps:.6f},setsar=1[bg];"
          f"[0:v]scale=-2:{sh}:force_original_aspect_ratio=decrease,setsar=1[sd];"
          f"[bg][sd]overlay=(W-w)/2:H*0.04:shortest=1,"
          f"drawbox=x=iw*0.05:y=ih*0.86:w=iw*0.9:h=ih*0.012:color=0x555555:t=fill,"
          f"drawbox=x=iw*0.05:y=ih*0.86:w='max(2,iw*0.9*t/{D:.2f})':h=ih*0.012:color=0xff0000:t=fill,"
          f"drawtext=fontfile={FONT_REG}:text='{title}':fontcolor=white:fontsize=h*0.035:x=w*0.05:y=h*0.9,"
          f"drawtext=fontfile={FONT_REG}:text='%{{pts\\:hms}}':fontcolor=0xbbbbbb:fontsize=h*0.03:x=w*0.8:y=h*0.9[v]")
    ext = c.container()
    a_s = c.s.W / c.s.Hh
    cov = min(1.0, (sh * min(W * 0.99, sh * a_s)) / (W * Hh))
    return c.job([c.simple(None, c.h264(fps=c.s.fps), ext, fc=fc)], ext, [[c.s.name, 0, D, 0]], D,
                 coverage=cov)


# ---- clips (the main course) ---------------------------------------------

@T("clip_head", "clip")
def _(c):
    a, L = c.clip_window(0.1, 0.6, "head")
    return c.clip(c.scale_short(c.out_short()), a, L)


@T("clip_mid", "clip")
def _(c):
    a, L = c.clip_window(0.1, 0.6, "mid")
    return c.clip(c.scale_short(c.out_short()), a, L)


@T("clip_tail", "clip")
def _(c):
    a, L = c.clip_window(0.1, 0.6, "tail")
    return c.clip(c.scale_short(c.out_short()), a, L)


@T("clip_short", "clip")
def _(c):
    D = c.s.dur
    L = c.d.u("len", 3.0, 15.0) if D > 30 else D * c.d.u("len", 0.25, 0.5)
    a = c.d.u("at", 0, D - L)
    return c.clip(c.scale_short(c.out_short()), round(a, 3), round(L, 3))


@T("clip_tiny", "clip")
def _(c):
    D = c.s.dur
    L = min(c.d.u("len", 2.0, 5.0), D * 0.4)
    a = c.d.u("at", 0, D - L)
    return c.clip(c.scale_short(c.out_short()), round(a, 3), round(L, 3), note="one or two keyframes")


@T("clip_streamcopy", "clip")
def _(c):
    """LosslessCut-style: no re-encode, so it starts on the keyframe at or
    before the cut. The truth is that keyframe, read from the seed itself;
    the end is measured from the output."""
    D = c.s.dur
    L = D * c.d.u("len", 0.15, 0.45)
    want = c.d.u("at", 0.05 * D, D - L)
    kf = max([k for k in c.s.keyframes if k <= want + 1e-3] or [0.0])
    ext = c.d.pick("ct", ["mkv", "mp4"]) if c.s.src_codec in ("h264", "hevc", "av1", "vp9") else "mkv"
    args = ["ffmpeg", "-v", "error", "-y", "-ss", f"{want + c.s.v0:.3f}", "-i", c.s.path, "-t", f"{L:.3f}",
            "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-avoid_negative_ts", "make_zero", "{OUT}"]
    return c.job([args], ext, [[c.s.name, kf, None, 0.0]], None, note=f"asked {want:.2f}, keyframe {kf:.2f}")


@T("trim_ends", "clip")
def _(c):
    D = c.s.dur
    a = min(c.d.u("a", 1.0, 5.0), D * 0.08)
    b = min(c.d.u("b", 1.0, 5.0), D * 0.08)
    return c.clip(c.scale_short(c.out_short()), round(a, 3), round(D - a - b, 3))


@T("intro_outro", "clip")
def _(c):
    W, Hh = c.canvas(c.out_short())
    I, O = c.d.u("I", 3, 10), c.d.u("O", 3, 8)
    pieces = [("card", "intro", I), ("seed", c.s.name, 0.0, c.s.dur), ("card", "outro", O)]
    if c.d.i("only", 0, 2) == 0:
        pieces = pieces[:2]
    return concat_job(c, pieces, W, Hh, c.s.fps, c.container())


@T("cut_middle", "clip")
def _(c):
    D = c.s.dur
    cut = D * c.d.u("len", 0.1, 0.3)
    a = c.d.u("at", 0.2 * D, 0.8 * D - cut)
    W, Hh = c.canvas(c.out_short())
    return concat_job(c, [("seed", c.s.name, 0.0, a), ("seed", c.s.name, a + cut, D - a - cut)],
                      W, Hh, c.s.fps, c.container(), note=f"{cut:.1f} s removed")


@T("insert_break", "clip")
def _(c):
    D = c.s.dur
    A = c.d.u("A", 8, 25)
    p = c.d.u("at", 0.25 * D, 0.75 * D)
    W, Hh = c.canvas(c.out_short())
    return concat_job(c, [("seed", c.s.name, 0.0, p), ("card", "ad", A), ("seed", c.s.name, p, D - p)],
                      W, Hh, c.s.fps, c.container(), note=f"{A:.1f} s break")


@T("loop_twice", "clip")
def _(c):
    D = c.s.dur
    if D <= 240:
        a, L = 0.0, D
    else:
        L = 90.0
        a = c.d.u("at", 0, D - L)
    W, Hh = c.canvas(c.out_short())
    return concat_job(c, [("seed", c.s.name, a, L), ("seed", c.s.name, a, L)], W, Hh, c.s.fps,
                      c.container())


@T("fades_black", "clip")
def _(c):
    D = c.s.dur
    p1, p2 = sorted((c.d.u("p1", 0.2 * D, 0.45 * D), c.d.u("p2", 0.55 * D, 0.8 * D)))
    g1, g2 = c.d.u("g1", 0.5, 2.5), c.d.u("g2", 0.5, 2.5)
    W, Hh = c.canvas(c.out_short())
    tot = D + g1 + g2
    post = f"fade=t=in:st=0:d={c.d.u('fi', 0.8, 2.5):.2f},fade=t=out:st={tot - 2:.3f}:d=2"
    return concat_job(c, [("seed", c.s.name, 0.0, p1), ("black", g1), ("seed", c.s.name, p1, p2 - p1),
                          ("black", g2), ("seed", c.s.name, p2, D - p2)], W, Hh, c.s.fps, c.container(),
                      post=post)


@T("compilation", "clip")
def _(c):
    """Clips of this seed and two or three others, hard cuts. A compilation is
    a clip of every seed it holds, so its truth spans families."""
    me = c.s
    others = sorted(n for n, s in c.all.items() if n != me.name and (not me.nm_set or s.nm_set != me.nm_set))
    k = c.d.i("k", 2, 3)
    chosen, i = [], 0
    while len(chosen) < k:
        n = c.d.pick(f"o{i}", others)
        if n not in chosen:
            chosen.append(n)
        i += 1
    names = [me.name] + chosen
    order = sorted(names, key=lambda n: H(c.d.base, "order", n))
    pieces = []
    for n in order:
        sd = c.all[n]
        L = min(c.d.u(f"L{n}", 8, 30), sd.dur * 0.4)
        a = c.d.u(f"a{n}", 0, sd.dur - L)
        pieces.append(("seed", n, round(a, 3), round(L, 3)))
    W, Hh = 1280, 720
    if c.s.short < 720:
        W, Hh = 854, 480
    return concat_job(c, pieces, W, Hh, 30.0, c.container(), note="+".join(order))


# clip + something: the shapes a reupload actually takes
@T("clip_crop", "clip_chain")
def _(c):
    a, L = c.clip_window(0.15, 0.5)
    box = light_crop_box(c, 0.04, 0.12)
    return c.clip(c.crop(*box) + "," + c.scale_short(c.out_short()), a, L, region=box)


@T("clip_vertical", "clip_chain")
def _(c):
    a, L = c.clip_window(0.1, 0.4)
    fc, cov = blurfill_vf(c, 540)
    ext = c.container()
    return c.job([c.simple(None, c.h264(), ext, ss=a, t=L, fc=fc + "[v]")], ext,
                 [[c.s.name, a, a + L, 0]], L, coverage=cov)


@T("clip_watermark", "clip_chain")
def _(c):
    a, L = c.clip_window(0.15, 0.5)
    x, y = c.corner_xy("corner")
    vf = (c.scale_short(c.out_short()) + f",drawtext=fontfile={FONT}:text='{c.text('wm', WATERMARKS)}':"
          f"fontcolor=white@0.7:fontsize=h*0.06:x={x}:y={y}")
    return c.clip(vf, a, L)


@T("clip_grey", "clip_chain")
def _(c):
    a, L = c.clip_window(0.15, 0.5)
    return c.clip(c.scale_short(c.out_short()) + ",hue=s=0", a, L)


@T("clip_lowres", "clip_chain")
def _(c):
    a, L = c.clip_window(0.15, 0.5)
    return c.clip(c.scale_short(144, "area"), a, L, venc=c.h264(crf=c.d.i("crf", 24, 30)))


@T("clip_hevc", "clip_chain")
def _(c):
    a, L = c.clip_window(0.15, 0.5)
    v = ["-c:v", "libx265", "-crf", str(c.d.i("crf", 22, 30)), "-preset", "superfast", "-pix_fmt", "yuv420p10le",
         "-x265-params", "log-level=error:pools=2:frame-threads=1"]
    return c.clip(c.scale_short(c.out_short()), a, L, venc=v, ext="mkv")


@T("clip_short_colour", "clip_chain")
def _(c):
    D = c.s.dur
    L = c.d.u("len", 4.0, 15.0) if D > 30 else D * c.d.u("len", 0.25, 0.5)
    a = c.d.u("at", 0, D - L)
    vf = (f"eq=brightness={c.d.u('b', -0.1, 0.1):.3f}:saturation={c.d.u('s', 0.6, 1.5):.2f}:"
          f"contrast={c.d.u('c', 0.8, 1.25):.2f}")
    vf = c.scale_short(c.out_short()) + "," + vf
    return c.clip(vf, round(a, 3), round(L, 3))


# ---- pipelines -----------------------------------------------------------

@T("whatsapp", "pipeline")
def _(c):
    s = c.d.pick("s", [360, 480])
    v = c.h264(crf=c.d.i("crf", 28, 33), preset="veryfast", profile="baseline", fps=min(30, c.s.fps))
    return c.whole(c.scale_short(s) + f",fps={min(30, c.s.fps):.6f}", v, "mp4")


@T("tiktok", "pipeline")
def _(c):
    fc, cov = blurfill_vf(c, 540)
    cap = words(c.d, "cap", c.d.i("n", 3, 6))
    wm = c.text("wm", WATERMARKS)
    fc += (f",drawtext=fontfile={FONT}:text='{cap}':fontcolor=white:fontsize=h*0.03:x=(w-tw)/2:y=h*0.12:"
           f"box=1:boxcolor=black@0.5:boxborderw=10,drawtext=fontfile={FONT}:text='{wm}':fontcolor=white@0.6:"
           f"fontsize=h*0.025:x=(w-tw)*abs(mod(t*0.07\\,2)-1):y=h*0.8[v]")
    return c.job([c.simple(None, c.h264(crf=c.d.i("crf", 23, 28)), "mp4", fc=fc)], "mp4",
                 [[c.s.name, 0, c.s.dur, 0]], c.s.dur, coverage=cov)


@T("tv_capture", "pipeline")
def _(c):
    a = c.s.W / c.s.Hh
    W, Hh = (720, 576)
    box = f"scale=768:576:force_original_aspect_ratio=decrease,pad=768:576:(ow-iw)/2:(oh-ih)/2,scale={W}:{Hh},setsar=16/15"
    x, y = c.corner_xy("bug")
    vf = (box + f",drawtext=fontfile={FONT}:text='{c.text('wm', WATERMARKS)}':fontcolor=white@0.75:"
          f"fontsize=h*0.05:x={x}:y={y},fps=25,tinterlace=interleave_top,setfield=tff")
    v = ["-c:v", "mpeg2video", "-b:v", f"{c.d.i('br', 2500, 5000)}k", "-flags", "+ilme+ildct", "-top", "1",
         "-g", "12", "-bf", "2"]
    cov = min(a / (4 / 3), (4 / 3) / a)
    return c.whole(vf, v, "ts", coverage=cov)


@T("vhs", "pipeline")
def _(c):
    a = c.s.W / c.s.Hh
    crop = "crop=ih*4/3:ih" if a > 4 / 3 + 0.01 else "null"
    reg = ((1 - (4 / 3) / a) / 2, 0, 1 - (1 - (4 / 3) / a) / 2, 1) if a > 4 / 3 + 0.01 else (0, 0, 1, 1)
    vf = (f"{crop},scale=320:240,setsar=1,noise=alls={c.d.i('n', 12, 25)}:allf=t,chromashift=cbh={c.d.i('cb', 2, 5)}:"
          f"crh=-{c.d.i('cr', 1, 4)},gblur=sigma=0.8,eq=saturation={c.d.u('s', 1.1, 1.4):.2f}:contrast=1.1:"
          f"brightness=0.03,scale=640:480,setsar=1")
    v = ["-c:v", "libxvid", "-q:v", "5", "-pix_fmt", "yuv420p"] if c.d.i("k", 0, 1) else \
        ["-c:v", "mpeg2video", "-q:v", "5", "-pix_fmt", "yuv420p"]
    return c.whole(vf, v, "avi" if v[1] == "libxvid" else "mpg", region=reg)


# ---- speed and direction: traps, scored separately -----------------------

@T("speed_up", "trap")
def _(c):
    s = c.d.u("s", 1.1, 1.5)
    v = c.h264()
    ext = c.container()
    args = c.simple(f"setpts=PTS/{s:.4f}," + c.scale_short(c.out_short()), v, ext, af=f"atempo={s:.4f}")
    return c.job([args], ext, [[c.s.name, 0, c.s.dur, 0]], c.s.dur / s, speed=s, trap="speed",
                 note=f"x{s:.2f}")


@T("slow_down", "trap")
def _(c):
    s = c.d.u("s", 0.67, 0.9)
    D = min(c.s.dur, 120.0)
    a = c.d.u("at", 0, c.s.dur - D)
    ext = c.container()
    args = c.simple(f"setpts=PTS/{s:.4f}," + c.scale_short(c.out_short()), c.h264(), ext,
                    ss=a, t=D, af=f"atempo={s:.4f}")
    return c.job([args], ext, [[c.s.name, a, a + D, 0]], D / s, speed=s, trap="speed",
                 note=f"x{s:.2f}")


@T("reverse", "trap")
def _(c):
    L = min(c.d.u("len", 8, 20), c.s.dur * 0.6)
    a = c.d.u("at", 0, c.s.dur - L)
    ext = c.container()
    args = c.simple("reverse," + c.scale_short(c.out_short()), c.h264(), ext, ss=a, t=L, af="areverse")
    return c.job([args], ext, [[c.s.name, a, a + L, 0]], L, trap="reverse")


# ==========================================================================
# naming: scattered, ordinary, stable per (seed, transform)

FOLDERS = ["Downloads", "Videos", "Videos/2023", "Videos/2024", "Videos/edits", "Movies",
           "WhatsApp Video", "Telegram Desktop", "DCIM/Camera", "Backup/old-phone",
           "Desktop", "Recordings", "Documents/uni", "Downloads/reposts"]
PATTERNS = ["VID_2024{mm:02d}{dd:02d}_{hh:02d}{mi:02d}{ss:02d}", "video ({n})", "clip_{n}", "{n}",
            "Screen Recording 2024-{mm:02d}-{dd:02d} at {hh:02d}.{mi:02d}.{ss:02d}",
            "WhatsApp Video 2024-{mm:02d}-{dd:02d} at {hh:02d}.{mi:02d}.{ss:02d}",
            "download ({n})", "MOV_{n:04d}", "PXL_2024{mm:02d}{dd:02d}_{n:06d}", "output_{n}",
            "final_final_{n}", "untitled {n}"]
USED = set()
USED_LOCK = threading.Lock()


def name_for(seed, tname, ext):
    d = Draw("name", seed, tname)
    bump = 0
    while True:
        folder = d.pick(f"f{bump}", FOLDERS)
        pat = d.pick(f"p{bump}", PATTERNS)
        stem = pat.format(n=d.i(f"n{bump}", 1, 99999), mm=d.i("mm", 1, 12), dd=d.i("dd", 1, 28),
                          hh=d.i("hh", 0, 23), mi=d.i("mi", 0, 59), ss=d.i(f"ss{bump}", 0, 59))
        rel = os.path.join("derived", folder, f"{stem}.{ext}")
        with USED_LOCK:
            if rel not in USED:
                USED.add(rel)
                return rel
        bump += 1


# ==========================================================================
# execution

LOG = os.path.join(VIDS, "manifest.jsonl")
LOG_LOCK = threading.Lock()


def throttle(cmd):
    """Two threads per ffmpeg: decoders, filters and encoder. With six
    workers that keeps the load near the core count; unthrottled, four
    workers put the load average at 40 on 8 cores."""
    if cmd[0] != "ffmpeg":
        return cmd
    out = [cmd[0], "-filter_threads", "2", "-filter_complex_threads", "2"]
    for a in cmd[1:-1]:
        if a == "-i":
            out += ["-threads", "2"]
        out.append(a)
    return out + ["-threads", "2", cmd[-1]]


def execute(job, rel):
    out = os.path.join(VIDS, rel)
    tmp = os.path.join(TMP, f"{job.seed}__{job.transform}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    part = os.path.join(tmp, "out." + job.ext)
    for name, text in job.files.items():
        open(os.path.join(tmp, name), "w").write(text)
    t0 = time.time()
    try:
        for cmd in job.cmds:
            run(throttle([a.replace("{OUT}", part).replace("{TMP}", tmp) for a in cmd]))
        first, end = video_span(part)
        vdur = end - first
        if vdur <= 0:
            raise RuntimeError("no video in output")
        segs = [list(s) for s in job.segments]
        expect = job.expect
        if expect is None:                       # measured (stream copy)
            segs[0][2] = segs[0][1] + vdur
            expect = vdur
        if abs(vdur - expect) > max(0.6, 0.02 * expect):
            raise RuntimeError(f"duration {vdur:.2f} s, expected {expect:.2f} s")
        os.replace(part, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    v = ffprobe_json(out, "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height")["streams"][0]
    rec = dict(file=rel, seed=job.seed, transform=job.transform, group=job.group,
               segments=segs, speed=job.speed, duration=round(vdur, 3), expected=round(expect, 3),
               region=[round(x, 4) for x in job.region], coverage=round(job.coverage, 4),
               mirror=job.mirror, trap=job.trap, codec=v["codec_name"], width=v["width"],
               height=v["height"], bytes=os.path.getsize(out), note=job.note,
               seconds=round(time.time() - t0, 1))
    with LOG_LOCK:
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def load_log():
    recs = {}
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            recs[(r["seed"], r["transform"])] = r
    return recs


# ==========================================================================
# ground truth for pairs

def seed_record(s):
    return dict(file=os.path.relpath(s.path, VIDS), seed=s.name, transform="seed", group="seed",
                segments=[[s.name, 0.0, s.dur, 0.0]], speed=1.0, duration=s.dur, region=[0, 0, 1, 1],
                coverage=1.0, mirror=False, trap="", codec=s.src_codec, width=s.src_w, height=s.src_h,
                bytes=os.path.getsize(s.path), note="")


def intervals(rec, seed):
    return [(a, b) for (n, a, b, _o) in rec["segments"] if n == seed]


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def inter_len(iv, U):
    return sum(max(0.0, min(b, ub) - max(a, ua)) for a, b in iv for ua, ub in U)


def matched_seconds(r, other, seeds):
    """Seconds of r's own runtime whose picture also appears in `other`."""
    tot = 0.0
    for sd in seeds:
        tot += inter_len(intervals(r, sd), union(intervals(other, sd))) / r["speed"]
    return tot


def box_rel(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    if ix <= 0:
        return "DISJOINT"
    if ix / (area(a) + area(b) - ix) >= 0.9:
        return "SAME"
    if ix / min(area(a), area(b)) >= 0.9:
        return "CROP"
    return "PARTIAL"


def relation(a, b):
    sa = {s[0] for s in a["segments"]}
    sb = {s[0] for s in b["segments"]}
    shared = sa & sb
    ma = matched_seconds(a, b, shared)
    mb = matched_seconds(b, a, shared)
    ca, cb = ma / a["duration"], mb / b["duration"]
    if ma <= 0.05 and mb <= 0.05:
        t = "DIFFERENT"
    elif ca >= 0.9 and cb >= 0.9:
        t = "SAME"
    elif ca >= 0.9 or cb >= 0.9:
        t = "CLIP"
    else:
        t = "PARTIAL"
    sp = box_rel(a["region"], b["region"]) if t != "DIFFERENT" and len(sa) == 1 and len(sb) == 1 else ""
    return t, sp, ma, mb, ca, cb


def write_outputs(seeds):
    recs = load_log()
    live = [r for r in recs.values() if os.path.exists(os.path.join(VIDS, r["file"]))]
    allrecs = [seed_record(s) for s in seeds] + sorted(live, key=lambda r: (r["seed"], r["transform"]))
    with open(os.path.join(VIDS, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "seed", "transform", "group", "segments", "speed", "duration", "region",
                    "coverage", "mirror", "trap", "codec", "width", "height", "bytes", "note"])
        for r in allrecs:
            w.writerow([r["file"], r["seed"], r["transform"], r["group"],
                        ";".join(f"{n}@{a:.3f}-{b:.3f}>{o:.3f}" for n, a, b, o in r["segments"]),
                        r["speed"], r["duration"], " ".join(map(str, r["region"])), r["coverage"],
                        int(r["mirror"]), r["trap"], r["codec"], r["width"], r["height"], r["bytes"], r["note"]])
    # families: everything holding a seed's picture
    fam = {s.name: [] for s in seeds}
    for i, r in enumerate(allrecs):
        for n in {s[0] for s in r["segments"]}:
            fam.setdefault(n, []).append(i)
    seen = set()
    counts = {}
    with open(os.path.join(VIDS, "pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_a", "file_b", "relation", "spatial", "matched_a_s", "matched_b_s", "cov_a", "cov_b",
                    "transform_a", "transform_b", "seeds", "trap", "mirror", "coverage_min"])
        for n, idx in fam.items():
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    i, j = idx[x], idx[y]
                    key = (min(i, j), max(i, j))
                    if key in seen:
                        continue
                    seen.add(key)
                    a, b = allrecs[key[0]], allrecs[key[1]]
                    t, sp, ma, mb, ca, cb = relation(a, b)
                    counts[t] = counts.get(t, 0) + 1
                    traps = "+".join(sorted({a["trap"], b["trap"]} - {""}))
                    seeds_ab = "+".join(sorted({s[0] for s in a["segments"]} & {s[0] for s in b["segments"]}))
                    w.writerow([a["file"], b["file"], t, sp, f"{ma:.2f}", f"{mb:.2f}", f"{ca:.4f}", f"{cb:.4f}",
                                a["transform"], b["transform"], seeds_ab, traps,
                                int(a["mirror"] != b["mirror"]), min(a["coverage"], b["coverage"])])
    summary = dict(files=len(allrecs), seeds=len(seeds), derived=len(live), pairs=counts,
                   bytes=sum(r["bytes"] for r in allrecs),
                   hours=round(sum(r["duration"] for r in allrecs) / 3600, 2))
    json.dump(summary, open(os.path.join(VIDS, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


# ==========================================================================

# Every transform on every seed is ~26 h of encoding, so each seed gets a
# deterministic subset.  Mirrors and clips are drawn more often (the user's
# priorities), the remuxes always (they are free), and a seed longer than
# LONG_SEED only gets the short outputs: it is there to host clips.
RATE = {"container": 1.0, "mirror": 0.35, "clip": 0.30, "clip_chain": 0.25}
OTHER_RATE = 0.12
LONG_SEED = 900
LONG_OK = {"clip_head", "clip_mid", "clip_tail", "clip_short", "clip_tiny", "clip_streamcopy",
           "mirror_clip", "clip_crop", "clip_vertical", "clip_watermark", "clip_grey", "clip_lowres",
           "clip_hevc", "clip_short_colour"}


def chosen(seed, t):
    g = TRANSFORMS[t][0]
    if g == "container":
        return True
    if seed.dur > LONG_SEED:
        return t in LONG_OK
    h = int.from_bytes(hashlib.blake2b(f"pick/{seed.name}/{t}".encode(), digest_size=8).digest(), "big")
    return (h % 10_000) / 10_000 < RATE.get(g, OTHER_RATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated transforms")
    ap.add_argument("--seeds", help="comma-separated seed names")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--pairs-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true", help="every transform on every seed (~26 h)")
    a = ap.parse_args()
    if a.list:
        for n, (g, _) in TRANSFORMS.items():
            print(f"{g:12} {n}")
        print(len(TRANSFORMS), "transforms")
        return
    if a.clean:
        shutil.rmtree(DERIVED, ignore_errors=True)
        if os.path.exists(LOG):
            os.remove(LOG)
    seeds = load_seeds()
    print(f"preparing {len(seeds)} seeds", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(prepare, seeds))
    by = {s.name: s for s in seeds}
    if a.pairs_only:
        write_outputs(seeds)
        return
    want_t = a.only.split(",") if a.only else list(TRANSFORMS)
    want_s = a.seeds.split(",") if a.seeds else [s.name for s in seeds]
    for t in want_t:
        if t not in TRANSFORMS:
            sys.exit(f"unknown transform {t}")
    done = load_log()
    for r in done.values():
        USED.add(r["file"])
    jobs = []
    for sn in want_s:
        for t in want_t:
            if not (a.only or a.all or chosen(by[sn], t)):
                continue
            prev = done.get((sn, t))
            if prev and os.path.exists(os.path.join(VIDS, prev["file"])):
                continue
            job = TRANSFORMS[t][1](Ctx(by[sn], t, by))
            jobs.append((job, prev["file"] if prev else name_for(sn, t, job.ext)))
    # biggest first so the pool drains evenly
    jobs.sort(key=lambda j: -by[j[0].seed].dur)
    print(f"{len(jobs)} jobs", flush=True)
    fails = []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(execute, j, rel): (j, rel) for j, rel in jobs}
        for n, fu in enumerate(as_completed(futs), 1):
            j, rel = futs[fu]
            try:
                fu.result()
            except Exception as e:
                fails.append((j.seed, j.transform, str(e)))
                print(f"FAIL {j.seed}/{j.transform}: {str(e)[-400:]}", flush=True)
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    if fails:
        with open(os.path.join(VIDS, "failures.txt"), "w") as f:
            for s, t, e in fails:
                f.write(f"{s}\t{t}\t{e}\n")
        print(f"{len(fails)} failures, see failures.txt")
    write_outputs(seeds)


if __name__ == "__main__":
    main()
