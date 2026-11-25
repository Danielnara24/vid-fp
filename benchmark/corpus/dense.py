"""Dense per-frame hashes of a video, for analysis (not what vid-fp does).

    dense(path, fps=None) -> (times_s, grey32 uint8 [n,32,32])
fps=None decodes every frame. Results are cached as .npz in out/dense.
"""
import hashlib, json, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "out", "dense")
S = 32


def dense(path, fps=None, size=S):
    key = hashlib.blake2b(f"{path}|{os.path.getmtime(path)}|{fps}|{size}".encode(), digest_size=8).hexdigest()
    cp = os.path.join(CACHE, key + ".npz")
    if os.path.exists(cp):
        z = np.load(cp)
        return z["t"], z["g"]
    vf = (f"fps={fps}," if fps else "") + f"scale={size}:{size}:flags=area,format=gray,showinfo"
    p = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-i", path, "-map", "0:v:0", "-vf", vf,
                        "-f", "rawvideo", "-"], capture_output=True)
    g = np.frombuffer(p.stdout, np.uint8).reshape(-1, size, size)
    t = []
    for line in p.stderr.decode(errors="replace").splitlines():
        if "showinfo" in line and " pts_time:" in line:
            t.append(float(line.split(" pts_time:")[1].split()[0]))
    t = np.array(t[:len(g)])
    np.savez(cp, t=t, g=g)
    return t, g


def dct_mat(n):
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    m[0] /= np.sqrt(2)
    return m


def phash(g, keep=16):
    """keep x keep low-frequency DCT bits above the median (DC dropped) -> bool [n, keep*keep-1]."""
    D = dct_mat(g.shape[1])
    c = D @ g.astype(np.float32) @ D.T
    c = c[:, :keep, :keep].reshape(len(g), -1)[:, 1:]
    return c > np.median(c, axis=1, keepdims=True)


def dense64(path, fps=None):
    """Every frame (or fps) at 64x64 grey through the scaler vid-fp uses."""
    key = hashlib.blake2b(f"64fb|{path}|{os.path.getmtime(path)}|{fps}".encode(), digest_size=8).hexdigest()
    cp = os.path.join(CACHE, key + ".npz")
    if os.path.exists(cp):
        z = np.load(cp)
        return z["t"], z["g"]
    vf = (f"fps={fps}," if fps else "") + "scale=64:64:flags=fast_bilinear,format=gray,showinfo"
    p = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-threads", "2", "-i", path, "-an", "-sn",
                        "-map", "0:v:0", "-vf", vf, "-f", "rawvideo", "-"], capture_output=True)
    g = np.frombuffer(p.stdout, np.uint8).reshape(-1, 64, 64)
    t = [float(l.split(" pts_time:")[1].split()[0]) for l in p.stderr.decode(errors="replace").splitlines()
         if "showinfo" in l and " pts_time:" in l]
    t = np.array(t[:len(g)])
    np.savez(cp, t=t, g=g)
    return t, g


_D16 = None


def vfp_hash(g):
    """vid-fp's hash without the auto-crop: 64x64 -> 16x16 box -> 8x8 low DCT, DC dropped,
    bits above the median -> uint64 [n]; also the AC energy (featureless if < 8)."""
    global _D16
    if _D16 is None:
        k = np.arange(8)[:, None]
        x = np.arange(16)[None, :]
        _D16 = np.cos(np.pi * (2 * x + 1) * k / 32).astype(np.float32)
    t = g.reshape(len(g), 16, 4, 16, 4).astype(np.float32).mean((2, 4))
    c = _D16 @ t @ _D16.T
    c = c.reshape(len(g), 64)
    c[:, 0] = 0
    energy = np.abs(c).mean(1)
    med = np.median(c, axis=1, keepdims=True)
    bits = (c > med)
    w = (1 << np.arange(63, -1, -1, dtype=np.uint64))
    h = (bits.astype(np.uint64) * w).sum(1, dtype=np.uint64)
    return h, energy


def ham(a, b):
    x = np.bitwise_xor(a, b)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def keyframes64(path):
    """Keyframes only, 64x64 grey through vid-fp's scaler."""
    key = hashlib.blake2b(f"kf64|{path}|{os.path.getmtime(path)}".encode(), digest_size=8).hexdigest()
    cp = os.path.join(CACHE, key + ".npy")
    if os.path.exists(cp):
        return np.load(cp)
    p = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-skip_frame", "nokey", "-i", path, "-an", "-sn",
                        "-map", "0:v:0", "-vf", "scale=64:64:flags=fast_bilinear,format=gray", "-fps_mode",
                        "passthrough", "-f", "rawvideo", "-"], capture_output=True)
    g = np.frombuffer(p.stdout, np.uint8).reshape(-1, 64, 64)
    np.save(cp, g)
    return g
