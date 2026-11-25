"""CPU seconds (one thread) to fingerprint a file the way vid-fp does today
(keyframes only, measured with vid-fp itself on a fresh cache) against decoding
every frame, or every reference frame, with ffmpeg the way a dense sampler would.

    python3 cost.py FILE...
"""
import os, resource, subprocess, sys, tempfile

VIDFP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "release", "vid-fp")
DEC = ["-threads", "1", "-skip_loop_filter", "all", "-flags2", "fast"]
MODES = {"all": [], "noref": ["-skip_frame", "noref"]}


def cpu(args):
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    p = subprocess.run(args, capture_output=True, text=True)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), p


def key(path):
    with tempfile.TemporaryDirectory() as d:
        c, p = cpu([VIDFP, path, "--cache", os.path.join(d, "c.redb"), "-t", "1", "-q"])
    return c


def ff(path, extra):
    c, p = cpu(["ffmpeg", "-v", "error", "-stats", *DEC, *extra, "-i", path, "-an", "-sn", "-map", "0:v:0",
                "-vf", "scale=64:64:flags=fast_bilinear,format=gray", "-threads", "1", "-f", "null", "-"])
    n = 0
    for tok in p.stderr.replace("\r", "\n").split("\n"):
        if tok.startswith("frame="):
            n = int(tok.split("=")[1].split()[0])
    return c, n


if __name__ == "__main__":
    for f in sys.argv[1:]:
        k = key(f)
        out = [f"{os.path.basename(f)[:34]:34} key {k:6.2f}s"]
        for m, extra in MODES.items():
            c, n = ff(f, extra)
            out.append(f"{m} {c:6.2f}s {c / k:5.1f}x ({n} fr)")
        print("  ".join(out), flush=True)
