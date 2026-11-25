"""The seed list for the generated vid-fp corpus, and the downloader that builds it.

Every seed is (slug, YouTube id, start, end, near-miss set). `start`/`end` in
seconds of the source; `None, None` keeps the whole video. `RANDOM(len)` draws
a deterministic window from a hash of the id and the slug, so re-running picks
the same footage. Pieces cut from one source are disjoint with gaps >= 5 s, and
every cut piece is re-encoded with exact cuts, so no two seeds share a frame
unless the sources themselves do (`check_seeds.py` looks for that).

    python3 seeds.py            # download + cut every seed missing from VIDS
    python3 seeds.py --list     # print the plan
"""
import hashlib
import json
import os
import subprocess
import sys
import time

VIDS = os.environ.get("CORPUS_ROOT", "/home/daniel/Documents/VIDS")
RAW = os.path.join(VIDS, ".raw")          # full downloads that pieces are cut from


class RANDOM:
    def __init__(self, length, lo=0.05, hi=0.95):
        self.length, self.lo, self.hi = length, lo, hi


# slug, id, start, end, set, note
SEEDS = [
    # ---- near-miss sets --------------------------------------------------
    ("spring",          "WhWc3b3KhnY", None, None, None, "Blender open movie"),
    ("talkhead_1",      "Ya3HloNKBhE", 44, 185, "talkhead", "same creator, same set"),
    ("talkhead_2",      "Ya3HloNKBhE", 190, 340, "talkhead", ""),
    ("talkhead_3",      "Ya3HloNKBhE", 525, 665, "talkhead", ""),
    ("podcast_1",       "wv01kkwGzb8", 54, 254, "podcast", "static two-shot"),
    ("podcast_2",       "wv01kkwGzb8", 410, 610, "podcast", ""),
    ("lecture_1",       "Gd5dzyGT2vc", None, None, "lecture", "same course, slides; may share a title card"),
    ("lecture_2",       "qW1cAOBrgCg", None, None, "lecture", ""),
    ("blender_tut_1",   "kVcY7K-JA1Y", 132, 372, "blender_tut", "same software UI"),
    ("blender_tut_2",   "7moxMNaRlk8", 207, 447, "blender_tut", "first minutes skipped"),
    ("news_1",          "sw9ZeLdytnk", 0, 300, "news", "same channel"),
    ("news_2",          "z7jOX3rqfT8", 0, 300, "news", ""),
    ("football_1",      "BgmCfV8-qlI", 0, 85, "football", "one match split in two (2nd source gone)"),
    ("football_2",      "BgmCfV8-qlI", 92, None, "football", "ends on a 20 s static image"),
    ("wow_1",           "UMh80-64QLc", RANDOM(180), None, "wow", "same game, no commentary"),
    ("wow_2",           "UMh80-64QLc", RANDOM(180), None, "wow", ""),
    ("gorillaz_1",      "7-CzVKewm5E", None, None, "gorillaz", "same song, Toronto"),
    ("gorillaz_2",      "JtlY1O-wJN4", 6005, 6298, "gorillaz", "same song, Toronto, other upload"),
    ("drone_ny_1",      "7dnLjw9dBCk", 0, 240, "drone_ny", "one flight split in two"),
    ("drone_ny_2",      "7dnLjw9dBCk", 300, 540, "drone_ny", ""),
    ("dashcam_1",       "hZoAcfIjQaE", RANDOM(150), None, "dashcam", "one drive, three windows"),
    ("dashcam_2",       "hZoAcfIjQaE", RANDOM(150), None, "dashcam", ""),
    ("dashcam_3",       "hZoAcfIjQaE", RANDOM(150), None, "dashcam", ""),
    ("beach_1",         "vYuoECsqZy8", RANDOM(180), None, "beach", "static camera"),
    ("beach_2",         "9ABcYWzLTWg", RANDOM(180), None, "beach", "similar static beach"),
    ("citynight_1",     "nDjjfaAN2hQ", RANDOM(240), None, "citynight", ""),
    ("citynight_2",     "TEjHDF9QXTY", None, None, "citynight", "12 s stock clip"),
    ("cooking_1",       "UmxC1_0gFDg", 0, 240, "cooking", "similar first minutes"),
    ("cooking_2",       "n8CJITXnDAg", 0, 240, "cooking", ""),
    ("starship_1",      "kbJBJOHk33g", None, None, "starship", "same event, drone view"),
    ("starship_2",      "AAt__NfP9No", None, None, "starship", "same event, tower cam"),
    ("starship_3",      "XUYXiXprlkg", None, None, "starship", "same event, news upload"),
    ("lyric_1",         "0bD7snOB-sI", None, None, "lyric", ""),
    ("lyric_2",         "RZjo35CIAYM", None, None, "lyric", ""),
    # ---- singles ---------------------------------------------------------
    ("bw_movie",        "kBGV-BbjxsA", 1065, 1305, None, "1937 film, 4:3 B&W"),
    ("short_phone",     "AMuYl-iVXhM", None, None, None, "vertical"),
    ("sonic_trailer",   "szby7ZHLnkA", None, None, None, "fast cuts, title cards"),
    ("lyric_black",     "uzT_v2Ab35M", None, None, None, "mostly black frames"),
    ("short_cctv",      "RKFv15yrlmw", None, None, None, "vertical CCTV"),
    ("ocean_dark",      "AMNyQds2ABQ", RANDOM(180), None, None, "very dark"),
    ("anime",           "F0sqfPkCcyM", None, None, None, ""),
    ("wildlife",        "WKKEkSFaCvs", 924, 1164, None, "documentary"),
    ("cocacola",        "gx-zPheFnHo", None, None, None, "ad"),
    ("kenburns",        "9zn-P_ah8QY", None, None, None, "photo slideshow"),
    ("whiteboard",      "1HcWhJy7mBU", None, None, None, "flat drawing"),
    ("underwater",      "poF-_lqwG3E", None, None, None, ""),
    ("f1_onboard",      "R1qqOKVGga0", None, None, None, ""),
    ("ds3_cinematic",   "zRBpd3ghVG4", None, None, None, "game cinematic"),
    ("mit_speak",       "Unzc731iCUY", 804, 1404, None, "lecture hall, slides/boards"),
    ("mit_linalg",      "UCc9q_cAhho", None, None, None, "44 min, 480p: the long host"),
    ("calculus_3b1b",   "WUvTyaaNkzM", 0, 600, None, "animation"),
]


# VIDS2 (2026-11-19): the user's 29 URLs, held out from all tuning. Both Tom and
# Jerry uploads were unavailable. Trailer/movie and trailer/reaction pairs share
# real footage of unknown extent: score.py keeps them out of P/R (SHARED) and
# scores them as a real-world tier of their own.
SEEDS2 = [
    ("fb_eng_1",        "Q_0e4FbguaU", RANDOM(240, 0.05, 0.45), None, "football", "one match, two windows"),
    ("fb_eng_2",        "Q_0e4FbguaU", RANDOM(240, 0.55, 0.95), None, "football", ""),
    ("fb_liv",          "zFuY34qkwRY", RANDOM(240), None, "football", "another match, 50 fps"),
    ("skyrim_1",        "KXXvRkQxWX4", RANDOM(240), None, "skyrim", "same let's-play series"),
    ("skyrim_2",        "jzsCG7npU-Y", RANDOM(240), None, "skyrim", ""),
    ("nature_1",        "q2_c2_WfJFg", RANDOM(240), None, "nature", "relaxation films"),
    ("nature_2",        "AKeUssuu3Is", RANDOM(240), None, "nature", ""),
    ("nature_3",        "BHACKCNDMW8", RANDOM(240), None, "nature", ""),
    ("nature_4",        "VFAiX5UluU4", RANDOM(240), None, "nature", "60 fps"),
    ("stan",            "gOMhN-hfMtY", None, None, None, "music video"),
    ("dont_speak",      "TR3Vdo5etCQ", None, None, None, "music video"),
    ("fails_1",         "V8u8kmaqN3g", RANDOM(240), None, "fails", "home-video compilations"),
    ("fails_2",         "qbfAW2My_L0", RANDOM(240), None, "fails", ""),
    ("fails_3",         "MagSZEQOaKU", RANDOM(240), None, "fails", ""),
    ("fails_4",         "ehn731OZ2pQ", RANDOM(240), None, "fails", ""),
    ("shibuya_43",      "PPnFaCZSjcc", None, None, None, "4:3 night street"),
    ("tubes_43",        "nA_tgIygvNo", None, None, None, "4:3 640x480"),
    ("heat_movie",      "Zx6s-OnFwG8", None, None, "heat", "full film, the long host"),
    ("heat_trailer",    "2GfZl4kuVNI", None, None, "heat", ""),
    ("toronto_movie",   "yjLcN0RFYTI", None, None, "toronto", "full-film upload"),
    ("toronto_trailer", "urqy8DrcGBs", None, None, "toronto", ""),
    ("gta_trailer",     "rz1B4B2yBZY", None, None, "gta", ""),
    ("gta_react_1",     "kL_vM5xMC4E", None, None, "gta", "streamer reaction"),
    ("gta_react_2",     "Z1opvGuJOew", None, None, "gta", "streamer reaction"),
    ("justice_1",       "kBdfcR-8hEY", RANDOM(300), None, "justice", "one course, one hall"),
    ("justice_2",       "0O2Rq4HJBxw", RANDOM(300), None, "justice", ""),
    ("justice_3",       "Qw4l1w0rkjs", RANDOM(300), None, "justice", ""),
    ("justice_4",       "MGyygiXMzRk", RANDOM(300), None, "justice", ""),
]
if os.path.basename(VIDS.rstrip("/")) == "VIDS2":
    SEEDS = SEEDS2

YTDLP = "/home/daniel/Documents/VIDS/.tools/yt-dlp"   # current release; the system one is too old for YouTube
# Cookies only when asked for (VIDS2: YouTube bot-checked this IP; the user chose Firefox cookies).
AUTH = ["--js-runtimes", "node"] + (["--cookies-from-browser", os.environ["YT_COOKIES"]] if os.environ.get("YT_COOKIES") else [])
YT = [YTDLP, "--no-update", *AUTH, "--no-progress", "-q", "-S", "res:1080,fps",
      "--merge-output-format", "mkv", "--no-playlist"]
FULL_IF_SHORTER = 1200    # several pieces of a source this short come from one full download


def meta(vid):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, vid + ".json")
    if not os.path.exists(path):
        for attempt in range(5):
            r = subprocess.run([YTDLP, "--no-update", *AUTH, "-J", "--no-playlist", f"https://youtu.be/{vid}"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
            time.sleep(20 * (attempt + 1))
        else:
            sys.exit(f"metadata failed: {vid}: {r.stderr.strip()[-200:]}")
        out = r.stdout
        with open(path, "w") as f:
            f.write(out)
        time.sleep(3)
    return json.load(open(path))


def window(slug, vid, start, end):
    """Resolve a seed's [start, end) in source seconds (end None = to the end)."""
    if isinstance(start, RANDOM):
        dur = meta(vid)["duration"]
        h = int.from_bytes(hashlib.blake2b(f"{vid}/{slug}".encode(), digest_size=8).digest(), "big")
        # keep clear of the first and last 5% (intros, outros, fades)
        lo, hi = dur * start.lo, dur * start.hi - start.length
        s = int(lo + (h % 10_000) / 10_000 * (hi - lo))
        return s, s + start.length
    return start, end


def plan():
    rows = []
    for slug, vid, start, end, nm, note in SEEDS:
        s, e = window(slug, vid, start, end)
        rows.append(dict(slug=slug, id=vid, start=s, end=e, set=nm or "", note=note))
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    for vid, rs in by.items():
        full = len(rs) > 1 and meta(vid)["duration"] <= FULL_IF_SHORTER
        rs.sort(key=lambda r: r["start"] or 0)
        for r in rs:
            r["how"] = "whole" if r["start"] is None and r["end"] is None else ("cut" if full else "section")
        # One full download has one clock, so 5 s is a real gap.  Separate
        # section downloads each start up to a GOP early: demand a minute.
        need = 5 if full else 60
        for a, b in zip(rs, rs[1:]):
            ae = a["end"] if a["end"] is not None else 1e18
            if ae + need > (b["start"] or 0):
                sys.exit(f"gap < {need} s between {a['slug']} and {b['slug']}")
    return rows


def fetch(url_id, dest_stem, section=None):
    cmd = YT + ["-o", dest_stem + ".%(ext)s", f"https://youtu.be/{url_id}"]
    if section:
        a, b = section
        cmd[1:1] = ["--download-sections", f"*{a}-{'inf' if b is None else b}"]
    for attempt in range(5):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return
        print(f"  retry {attempt + 1}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}", flush=True)
        time.sleep(30 * (attempt + 1))
    sys.exit(f"download failed: {url_id}")


def have(stem):
    d, b = os.path.split(stem)
    return next((os.path.join(d, f) for f in os.listdir(d)
                 if f.startswith(b + ".") and ".tmp." not in f and not f.endswith((".part", ".ytdl", ".json"))), None)


def main():
    rows = plan()
    if "--list" in sys.argv:
        tot = 0
        for r in rows:
            d = (r["end"] or meta(r["id"])["duration"]) - (r["start"] or 0)
            tot += d
            print(f"{r['slug']:16} {r['id']:12} {r['how']:8} {r['start']!s:>6}-{r['end']!s:<6} {d:6.0f}s {r['set']:12} {r['note']}")
        print(f"{len(rows)} seeds, {tot/3600:.2f} h")
        return
    os.makedirs(RAW, exist_ok=True)
    for r in rows:
        slug, vid, s, e = r["slug"], r["id"], r["start"], r["end"]
        if have(os.path.join(VIDS, slug)):
            continue
        print(f"=== {slug} ({r['how']})", flush=True)
        if r["how"] == "whole":
            fetch(vid, os.path.join(VIDS, slug))
        elif r["how"] == "section":
            fetch(vid, os.path.join(VIDS, slug), (s, e))
        else:
            raw = have(os.path.join(RAW, vid))
            if not raw:
                fetch(vid, os.path.join(RAW, vid))
                raw = have(os.path.join(RAW, vid))
            cut(raw, s, e, os.path.join(VIDS, slug + ".mkv"))
    write_csv(rows)
    print("done")


def cut(raw, s, e, out):
    """Exact re-encoded cut: pieces of one file then share no frame."""
    args = ["ffmpeg", "-v", "error", "-y", "-ss", str(s), "-i", raw]
    if e is not None:
        args += ["-t", str(e - s)]
    args += ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-crf", "15", "-preset", "fast",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", out + ".tmp.mkv"]
    subprocess.run(args, check=True)
    os.replace(out + ".tmp.mkv", out)


def write_csv(rows):
    import csv
    with open(os.path.join(VIDS, "seeds.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "file", "youtube_id", "source_start", "source_end", "how", "near_miss_set", "note"])
        for r in rows:
            w.writerow([r["slug"], os.path.basename(have(os.path.join(VIDS, r["slug"])) or ""),
                        r["id"], r["start"], r["end"], r["how"], r["set"], r["note"]])


if __name__ == "__main__":
    main()
