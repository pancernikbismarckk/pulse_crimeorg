"""Vectorised procedural noise (numpy) usable at arbitrary coordinates."""
import numpy as np

_M32 = np.int64(0xFFFFFFFF)


def _hash(ix, iy, seed):
    h = (ix.astype(np.int64) * np.int64(0x27D4EB2D)) ^ (iy.astype(np.int64) * np.int64(0x165667B1)) \
        ^ np.int64((seed * 0x9E3779B1) & 0xFFFFFFFF)
    h &= _M32
    h ^= h >> 15
    h = (h * np.int64(0x2C1B3C6D)) & _M32
    h ^= h >> 12
    h = (h * np.int64(0x297A2D39)) & _M32
    h ^= h >> 15
    return h


def rnd(ix, iy, seed):
    return (_hash(ix, iy, seed) & 0xFFFFFF).astype(np.float32) * (1.0 / 16777215.0)


def vnoise(x, y, seed=0):
    """Smooth value noise in [0,1]."""
    xf = np.floor(x)
    yf = np.floor(y)
    ix = xf.astype(np.int64)
    iy = yf.astype(np.int64)
    fx = (x - xf).astype(np.float32)
    fy = (y - yf).astype(np.float32)
    ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    a = rnd(ix, iy, seed)
    b = rnd(ix + 1, iy, seed)
    c = rnd(ix, iy + 1, seed)
    d = rnd(ix + 1, iy + 1, seed)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy


_ROT = [(np.cos(a), np.sin(a)) for a in np.radians([0, 37, 74, 111, 148, 185, 222, 259])]


def fbm(x, y, oct=5, lac=2.03, gain=0.5, seed=0):
    """Fractal value noise in ~[0,1] (mean 0.5)."""
    tot = np.zeros(np.shape(x), np.float32)
    amp = 1.0
    norm = 0.0
    fx, fy = x, y
    for i in range(oct):
        c, s = _ROT[i % len(_ROT)]
        rx = fx * c - fy * s
        ry = fx * s + fy * c
        tot += amp * vnoise(rx + 13.7 * i, ry - 7.3 * i, seed + 101 * i)
        norm += amp
        amp *= gain
        fx = fx * lac
        fy = fy * lac
    return tot / norm


def ridged(x, y, oct=4, seed=0):
    return 1.0 - np.abs(fbm(x, y, oct, seed=seed) * 2 - 1)


def worley(x, y, seed=0):
    """Returns (F1 distance, cell random id) for unit cells."""
    xf = np.floor(x).astype(np.int64)
    yf = np.floor(y).astype(np.int64)
    best = np.full(np.shape(x), 9.0, np.float32)
    bid = np.zeros(np.shape(x), np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx = xf + dx
            cy = yf + dy
            px = cx + rnd(cx, cy, seed)
            py = cy + rnd(cx, cy, seed + 7)
            d = np.sqrt((x - px) ** 2 + (y - py) ** 2).astype(np.float32)
            m = d < best
            best = np.where(m, d, best)
            bid = np.where(m, rnd(cx, cy, seed + 13), bid)
    return best, bid


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def mix3(a, b, t):
    """a,b: (...,3) or colour tuples; t: (...)"""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return a + (b - a) * np.asarray(t, np.float32)[..., None]


def sample(img, X, Y, x0, y0, x1, y1, clamp=True):
    """Bilinear sample of img (H,W[,C]) placed on the rect [x0,x1]x[y0,y1]
    (image row 0 at y1, i.e. y up)."""
    H, W = img.shape[:2]
    fx = (X - x0) / (x1 - x0) * W - 0.5
    fy = (y1 - Y) / (y1 - y0) * H - 0.5
    if clamp:
        fx = np.clip(fx, 0, W - 1.001)
        fy = np.clip(fy, 0, H - 1.001)
    else:
        fx = np.mod(fx, W)
        fy = np.mod(fy, H)
    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)
    tx = (fx - ix).astype(np.float32)
    ty = (fy - iy).astype(np.float32)
    ix1 = np.minimum(ix + 1, W - 1) if clamp else (ix + 1) % W
    iy1 = np.minimum(iy + 1, H - 1) if clamp else (iy + 1) % H
    if img.ndim == 3:
        tx = tx[..., None]
        ty = ty[..., None]
    a = img[iy, ix]
    b = img[iy, ix1]
    c = img[iy1, ix]
    d = img[iy1, ix1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def seg_dist(X, Y, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((X - ax) * dx + (Y - ay) * dy) / (L2 if L2 > 0 else 1.0), 0, 1)
    px = ax + t * dx
    py = ay + t * dy
    return np.sqrt((X - px) ** 2 + (Y - py) ** 2)


def poly_sdf(X, Y, P):
    """Signed distance to polygon P (negative inside)."""
    d = np.full(np.shape(X), 1e9, np.float64)
    inside = np.zeros(np.shape(X), bool)
    n = len(P)
    for i in range(n):
        ax, ay = P[i]
        bx, by = P[(i + 1) % n]
        d = np.minimum(d, seg_dist(X, Y, ax, ay, bx, by))
        cond = ((ay > Y) != (by > Y)) & (X < (bx - ax) * (Y - ay) / ((by - ay) if by != ay else 1e-12) + ax)
        inside ^= cond
    return np.where(inside, -d, d)


def rrect_sdf(X, Y, cx, cy, w, h, r):
    qx = np.abs(X - cx) - (w / 2 - r)
    qy = np.abs(Y - cy) - (h / 2 - r)
    return np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r
