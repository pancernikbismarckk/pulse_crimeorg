"""Procedural PBR painting of every atlas island (numpy).

Each painter receives the island-local metric coordinates (U, V) of every
atlas pixel of its (padded) rectangle and returns albedo (sRGB), roughness,
metallic, height (metres), emission (sRGB) and alpha.  Normal maps are derived
from the height field directly in atlas space, so they are valid for rotated
islands too.
"""
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import ct_art
import ct_scene as S
from ct_noise import (fbm, vnoise, ridged, worley, smoothstep, lerp, mix3, sample, seg_dist, poly_sdf,
                      rrect_sdf, rnd)
from ct_pack import ATLAS, PAD

SS = 2                 # supersampling factor (paint at 4096, deliver 2048)


# =============================================================================
# helpers
# =============================================================================
class Out:
    def __init__(self, shape):
        h, w = shape
        self.col = np.full((h, w, 3), 0.5, np.float32)
        self.rough = np.full((h, w), 0.5, np.float32)
        self.metal = np.zeros((h, w), np.float32)
        self.height = np.zeros((h, w), np.float32)
        self.emit = np.zeros((h, w, 3), np.float32)
        self.alpha = np.ones((h, w), np.float32)


def C3(*c):
    return np.array(c, np.float32)


def blend(o, mask, col=None, rough=None, metal=None, height=None, emit=None, add_height=None):
    m = np.clip(mask, 0, 1).astype(np.float32)
    if col is not None:
        col = np.asarray(col, np.float32)
        o.col = o.col * (1 - m[..., None]) + (col if col.ndim > 1 else col[None, None, :]) * m[..., None]
    if rough is not None:
        o.rough = o.rough * (1 - m) + rough * m
    if metal is not None:
        o.metal = o.metal * (1 - m) + metal * m
    if height is not None:
        o.height = o.height * (1 - m) + height * m
    if add_height is not None:
        o.height = o.height + add_height * m
    if emit is not None:
        emit = np.asarray(emit, np.float32)
        o.emit = o.emit * (1 - m[..., None]) + (emit if emit.ndim > 1 else emit[None, None, :]) * m[..., None]


def band_xy(U, isl):
    """Band coordinate s -> 2D polygon position (x, y) + edge index."""
    segs = sorted(isl.params.get("band_segs", []), key=lambda a: a[0])
    s0 = np.array([a[0] for a in segs])
    s1 = np.array([a[1] for a in segs])
    p0 = np.array([a[2] for a in segs])
    p1 = np.array([a[3] for a in segs])
    idx = np.clip(np.searchsorted(s0, U, side="right") - 1, 0, len(segs) - 1)
    t = (U - s0[idx]) / np.maximum(s1[idx] - s0[idx], 1e-9)
    x = p0[idx, 0] + t * (p1[idx, 0] - p0[idx, 0])
    y = p0[idx, 1] + t * (p1[idx, 1] - p0[idx, 1])
    return x, y, idx, segs


def gray_noise(U, V, f, seed, oct=4):
    return fbm(U * f, V * f, oct, seed=seed)


def smudges(U, V, seed, scale=18.0, thr=0.62):
    n = fbm(U * scale, V * scale, 4, seed=seed)
    return smoothstep(thr, thr + 0.12, n)


def fingerprint(U, V, cx, cy, r, ang, seed):
    X = U - cx
    Y = V - cy
    ca, sa = math.cos(ang), math.sin(ang)
    x = X * ca + Y * sa
    y = (-X * sa + Y * ca) * 1.35
    d = np.sqrt(x * x + y * y)
    ridges = 0.5 + 0.5 * np.sin(d / 0.00045 * 2 * math.pi + 3 * fbm(U * 400, V * 400, 2, seed=seed))
    return ridges * (1 - smoothstep(r * 0.6, r, d))


# =============================================================================
# decal maps (PIL, rendered once in object space)
# =============================================================================
def _aa_layer(size_px, draw_fn, ss=2, blur=0.0):
    W, H = size_px
    img = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(img)
    draw_fn(d, ss)
    img = img.resize((W, H), Image.BOX)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(img).astype(np.float32) / 255.0


class TableMaps:
    """Wear/dirt maps of the tabletop in table XY (2 px / mm)."""
    PPM = 2000.0

    def __init__(self):
        W = int(S.TW * self.PPM)
        H = int(S.TD * self.PPM)
        self.size = (W, H)
        self.rect = (-S.TW / 2, -S.TD / 2, S.TW / 2, S.TD / 2)
        rng = random.Random(77)

        def P(x, y, ss):   # world -> pixel (row 0 = +y)
            return ((x + S.TW / 2) * self.PPM * ss, (S.TD / 2 - y) * self.PPM * ss)

        # --- scratches (light = exposed wood, dark = dirt filled)
        light, dark = [], []
        for k in range(330):
            if rng.random() < 0.55:
                x = rng.gauss(0.0, 0.33)
                y = rng.gauss(-0.18, 0.13)
            else:
                x = rng.uniform(-0.73, 0.73)
                y = rng.uniform(-0.41, 0.41)
            L = rng.choice([rng.uniform(0.008, 0.04), rng.uniform(0.03, 0.12), rng.uniform(0.08, 0.26)])
            a = rng.choice([rng.gauss(0, 0.2), rng.uniform(0, math.pi)])
            bend = rng.gauss(0, 0.08)
            pts = []
            for t in (0.0, 0.33, 0.66, 1.0):
                aa = a + bend * (t - 0.5)
                pts.append((x + math.cos(aa) * L * (t - 0.5), y + math.sin(aa) * L * (t - 0.5)))
            w = rng.choice([1, 1, 1, 2, 2, 3])
            val = rng.uniform(0.35, 1.0)
            (light if rng.random() < 0.62 else dark).append((pts, w, val))
        # knife cuts near the mirror
        for k in range(14):
            x = rng.uniform(-0.30, 0.02)
            y = rng.uniform(-0.36, -0.12)
            a = rng.uniform(-0.3, 0.3)
            L = rng.uniform(0.03, 0.09)
            dark.append(([(x - math.cos(a) * L / 2, y - math.sin(a) * L / 2), (x + math.cos(a) * L / 2,
                                                                               y + math.sin(a) * L / 2)], 2, 1.0))

        def draw_scr(lst):
            def f(d, ss):
                for pts, w, val in lst:
                    d.line([P(x, y, ss) for (x, y) in pts], fill=int(255 * val), width=max(1, int(w * ss * 0.8)))
            return f
        self.scr_light = _aa_layer(self.size, draw_scr(light), 2, 0.4)
        self.scr_dark = _aa_layer(self.size, draw_scr(dark), 2, 0.4)

        # --- liquid ring stains + spills
        rings = [(-0.170, 0.300, 0.041, 0.9), (0.470, 0.350, 0.043, 0.8), (-0.345, 0.372, 0.034, 0.7),
                 (0.100, -0.345, 0.039, 0.55), (-0.60, 0.05, 0.036, 0.5), (0.63, -0.30, 0.041, 0.6)]

        def draw_rings(d, ss):
            for (x, y, r, val) in rings:
                for k in range(3):
                    a0 = rng.uniform(0, 360)
                    a1 = a0 + rng.uniform(200, 340)
                    cx, cy = P(x + rng.gauss(0, 0.0015), y + rng.gauss(0, 0.0015), ss)
                    rr = r * self.PPM * ss * rng.uniform(0.97, 1.03)
                    d.arc([cx - rr, cy - rr, cx + rr, cy + rr], a0, a1, fill=int(255 * val * rng.uniform(0.5, 1.0)),
                          width=max(1, int(rng.uniform(1.5, 3.5) * ss)))
        self.rings = _aa_layer(self.size, draw_rings, 2, 1.2)

        def draw_spill(d, ss):
            for (x, y, rx, ry, val) in [(0.520, 0.200, 0.060, 0.035, 0.8), (-0.080, 0.330, 0.045, 0.030, 0.5),
                                        (0.300, -0.200, 0.030, 0.022, 0.45), (-0.52, -0.33, 0.05, 0.03, 0.4)]:
                for k in range(9):
                    ox = rng.gauss(0, rx * 0.45)
                    oy = rng.gauss(0, ry * 0.45)
                    r1 = rng.uniform(0.35, 0.8)
                    cx, cy = P(x + ox, y + oy, ss)
                    d.ellipse([cx - rx * r1 * self.PPM * ss, cy - ry * r1 * self.PPM * ss,
                               cx + rx * r1 * self.PPM * ss, cy + ry * r1 * self.PPM * ss], fill=int(255 * val))
        self.spill = _aa_layer(self.size, draw_spill, 1, 6.0)

        # --- cigarette burns
        self.burns_list = [(0.205, 0.315, 0.3), (0.355, 0.085, -0.8), (-0.050, -0.395, 1.4), (0.66, 0.12, 0.2)]

        def draw_burns(d, ss):
            for (x, y, a) in self.burns_list:
                for k, (L, Wd, val) in enumerate([(0.050, 0.016, 0.25), (0.040, 0.010, 0.5), (0.030, 0.006, 0.75),
                                                  (0.020, 0.0028, 1.0)]):
                    pts = []
                    for t in range(24):
                        tt = 2 * math.pi * t / 24
                        ex = math.cos(tt) * L / 2
                        ey = math.sin(tt) * Wd / 2
                        pts.append(P(x + ex * math.cos(a) - ey * math.sin(a), y + ex * math.sin(a) + ey * math.cos(a), ss))
                    d.polygon(pts, fill=int(255 * val))
        self.burns = _aa_layer(self.size, draw_burns, 2, 3.5)

        # --- white powder residue around the mirror
        mx, my, myaw = S.MIR_POS

        def draw_powder(d, ss):
            ca, sa = math.cos(math.radians(myaw)), math.sin(math.radians(myaw))
            for k in range(900):
                # specks concentrated close to the mirror outline
                ex = rng.uniform(-0.15, 0.15)
                ey = rng.uniform(-0.11, 0.11)
                inside = abs(ex) < S.MIR_W / 2 - 0.004 and abs(ey) < S.MIR_H / 2 - 0.004
                if inside:
                    continue
                dd = max(abs(ex) - S.MIR_W / 2, abs(ey) - S.MIR_H / 2)
                if rng.random() > math.exp(-max(dd, 0) / 0.012):
                    continue
                x = mx + ex * ca - ey * sa
                y = my + ex * sa + ey * ca
                r = rng.uniform(0.2, 1.3) * ss
                cx, cy = P(x, y, ss)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(255 * rng.uniform(0.5, 1.0)))
            # smears
            for (ex, ey, rx, ry) in [(0.122, -0.02, 0.007, 0.018), (-0.02, -0.090, 0.03, 0.005), (-0.125, 0.03, 0.006, 0.012)]:
                x = mx + ex * ca - ey * sa
                y = my + ex * sa + ey * ca
                cx, cy = P(x, y, ss)
                d.ellipse([cx - rx * self.PPM * ss, cy - ry * self.PPM * ss, cx + rx * self.PPM * ss,
                           cy + ry * self.PPM * ss], fill=70)
        self.powder = _aa_layer(self.size, draw_powder, 2, 0.6)

        # --- ash specks around ashtray and butts
        ax, ay = S.ASH_POS

        def draw_ash(d, ss):
            for k in range(700):
                r0 = abs(rng.gauss(0, 0.05)) + 0.052
                a = rng.uniform(0, 2 * math.pi)
                x = ax + math.cos(a) * r0
                y = ay + math.sin(a) * r0
                r = rng.uniform(0.2, 1.1) * ss
                cx, cy = P(x, y, ss)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(255 * rng.uniform(0.3, 1.0)))
            for (bx, by) in [(0.175, 0.205), (0.360, 0.160)]:
                for k in range(60):
                    x = bx + rng.gauss(0, 0.012)
                    y = by + rng.gauss(0, 0.012)
                    r = rng.uniform(0.2, 0.9) * ss
                    cx, cy = P(x, y, ss)
                    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(255 * rng.uniform(0.3, 0.9)))
        self.ash = _aa_layer(self.size, draw_ash, 2, 0.5)

        # --- dents
        def draw_dents(d, ss):
            for k in range(70):
                x = rng.uniform(-0.72, 0.72)
                y = rng.uniform(-0.40, 0.40)
                r = rng.uniform(0.6, 2.8) * ss
                cx, cy = P(x, y, ss)
                d.ellipse([cx - r, cy - r * rng.uniform(0.5, 1.0), cx + r, cy + r], fill=int(255 * rng.uniform(0.4, 1.0)))
        self.dents = _aa_layer(self.size, draw_dents, 2, 1.5)

    def s(self, name, X, Y):
        x0, y0, x1, y1 = self.rect
        return sample(getattr(self, name), X, Y, x0, y0, x1, y1)


class MirrorMaps:
    PPM = 4000.0

    def __init__(self):
        W, H = int(S.MIR_W * self.PPM), int(S.MIR_H * self.PPM)
        self.size = (W, H)
        self.rect = (-S.MIR_W / 2, -S.MIR_H / 2, S.MIR_W / 2, S.MIR_H / 2)
        rng = random.Random(55)

        def P(x, y, ss):
            return ((x + S.MIR_W / 2) * self.PPM * ss, (S.MIR_H / 2 - y) * self.PPM * ss)

        def draw_dust(d, ss):
            # halos along each line
            for (cx, cy, L, Wd, ang) in S.LINES:
                a = math.radians(ang)
                for k in range(60):
                    t = rng.uniform(-0.55, 0.55)
                    off = rng.gauss(0, Wd * 0.7)
                    x = cx + math.cos(a) * L * t - math.sin(a) * off
                    y = cy + math.sin(a) * L * t + math.cos(a) * off
                    r = rng.uniform(0.10, 0.35) * self.PPM / 1000 * ss
                    px, py = P(x, y, ss)
                    d.ellipse([px - r, py - r, px + r, py + r], fill=int(255 * rng.uniform(0.4, 1.0)))
            # around the pile
            px0, py0 = S.PILE
            for k in range(110):
                rr = abs(rng.gauss(0, 0.007)) + 0.0165
                a = rng.uniform(0, 2 * math.pi)
                x = px0 + math.cos(a) * rr
                y = py0 + math.sin(a) * rr
                r = rng.uniform(0.10, 0.40) * self.PPM / 1000 * ss
                px, py = P(x, y, ss)
                d.ellipse([px - r, py - r, px + r, py + r], fill=int(255 * rng.uniform(0.4, 1.0)))
            # card swipe streak from pile towards the lines
            for k in range(40):
                t = k / 40.0
                x = px0 + (0.0 - px0) * t + rng.gauss(0, 0.002)
                y = py0 + (-0.012 - py0) * t + rng.gauss(0, 0.002)
                r = 1.6 * self.PPM / 1000 * ss
                px, py = P(x, y, ss)
                d.ellipse([px - r * 2.5, py - r, px + r * 2.5, py + r], fill=70)
            # random specks
            for k in range(70):
                x = rng.uniform(-S.MIR_W / 2, S.MIR_W / 2)
                y = rng.uniform(-S.MIR_H / 2, S.MIR_H / 2)
                r = rng.uniform(0.08, 0.25) * self.PPM / 1000 * ss
                px, py = P(x, y, ss)
                d.ellipse([px - r, py - r, px + r, py + r], fill=int(255 * rng.uniform(0.3, 1.0)))
        self.dust = _aa_layer(self.size, draw_dust, 2, 0.7)

    def s(self, name, X, Y):
        x0, y0, x1, y1 = self.rect
        return sample(getattr(self, name), X, Y, x0, y0, x1, y1)


class Ctx:
    def __init__(self, reg):
        self.reg = reg
        self.bill_f = ct_art.bill_front()
        self.bill_b = ct_art.bill_back()
        self.strap = ct_art.strap()
        self.screen = ct_art.laptop_screen()
        self.phone = ct_art.phone_screen()
        self.card = ct_art.card()
        self.kb = ct_art.keyboard()
        self.stickers = ct_art.lid_stickers()
        self.logo = ct_art.lid_logo_mask()
        self.table = TableMaps()
        self.mirror = MirrorMaps()
        self._can = None
        self._blur = {}

    def blurred(self, key, arr, sigma):
        k = (key, sigma)
        if k not in self._blur:
            im = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
            self._blur[k] = np.asarray(im.filter(ImageFilter.GaussianBlur(sigma))).astype(np.float32) / 255.0
        return self._blur[k]


# =============================================================================
# WOOD
# =============================================================================
EARLY = C3(0.45, 0.305, 0.185)
LATE = C3(0.205, 0.115, 0.055)


def wood_core(along, across, depth, seed, pith, spacing=0.0045, knots=()):
    """3D ring structure. Returns (latewood mask, streaks, knot mask)."""
    c0, D0 = pith
    c = c0 + 0.010 * np.sin(along * 1.7 + seed) + 0.004 * np.sin(along * 4.3 + 2.0 * seed)
    D = D0 + 0.012 * np.sin(along * 1.1 + 0.7 * seed) + 0.004 * np.sin(along * 3.3 + seed)
    dx = across - c
    dz = D - depth
    r = np.sqrt(dx * dx + dz * dz)
    t = r / spacing + (fbm(r * 9.0, seed * 0.37, 3, seed=seed + 21) - 0.5) * 5.0
    t = t + (fbm(along * 2.0, r * 45.0, 4, seed=seed) - 0.5) * 1.7
    t = t + (fbm(along * 14.0, across * 70.0, 3, seed=seed + 5) - 0.5) * 0.35
    knot = np.zeros(np.shape(along), np.float32)
    for (ka, kc, kr) in knots:
        d = np.sqrt(((along - ka) / 1.7) ** 2 + (across - kc) ** 2)
        infl = np.exp(-(d / (kr * 2.6)) ** 2)
        t = t + infl * 5.0 * np.exp(-(d / (kr * 1.2)) ** 2) + infl * (d / spacing) * 0.8
        knot = np.maximum(knot, 1 - smoothstep(kr * 0.55, kr, d))
    ring = t - np.floor(t)
    rid = np.floor(t).astype(np.int64)
    rv = rnd(rid, rid * 0 + seed, 5)
    w0 = 0.52 + 0.22 * rv
    late = smoothstep(w0, 0.93, ring) * (1 - smoothstep(0.955, 0.999, ring)) * (0.55 + 0.45 * rnd(rid, rid * 0 + seed, 9))
    streak = fbm(along * 1.6, across * 520.0, 3, seed=seed + 9)
    streak = streak * 0.7 + 0.3 * vnoise(along * 25.0, across * 1400.0, seed + 11)
    return late.astype(np.float32), streak.astype(np.float32), knot


def wood_color(late, streak, knot, tint, dark=1.0):
    col = mix3(EARLY, LATE, late * 0.85)
    col = col * (0.86 + 0.28 * streak)[..., None]
    col = col * np.asarray(tint, np.float32)[None, None, :]
    col = mix3(col, C3(0.16, 0.08, 0.04), knot * 0.85)
    return col * dark


PLANK_W = S.TD / 4.0


def plank_of(y):
    p = np.clip(np.floor((y + S.TD / 2) / PLANK_W), 0, 3).astype(np.int64)
    yc = -S.TD / 2 + (p + 0.5) * PLANK_W
    return p, y - yc


_PLANKS = []
for _k in range(4):
    _r = random.Random(900 + _k)
    _PLANKS.append(dict(
        seed=31 + 17 * _k,
        pith=(_r.uniform(-0.05, 0.05), _r.uniform(0.07, 0.14)),
        spacing=_r.uniform(0.0030, 0.0042),
        tint=(1.0 + _r.uniform(-0.07, 0.05), 1.0 + _r.uniform(-0.06, 0.05), 1.0 + _r.uniform(-0.08, 0.06)),
        knots=[(_r.uniform(-0.65, 0.65), _r.uniform(-0.07, 0.07), _r.uniform(0.006, 0.013)) for _ in range(_r.choice([0, 1, 1, 2]))],
        dz=_r.uniform(-0.0003, 0.0003),
    ))


def tabletop_wood(X, Y, depth):
    """Planks along X.  Returns col, late, knot, plank index, across."""
    p, across = plank_of(Y)
    shp = np.shape(X)
    col = np.zeros(shp + (3,), np.float32)
    late_all = np.zeros(shp, np.float32)
    knot_all = np.zeros(shp, np.float32)
    for k in range(4):
        m = p == k
        if not np.any(m):
            continue
        P = _PLANKS[k]
        late, streak, knot = wood_core(X[m], across[m], depth[m] if np.ndim(depth) else depth, P["seed"], P["pith"],
                                       P["spacing"], P["knots"])
        col[m] = wood_color(late[None], streak[None], knot[None], P["tint"])[0]
        late_all[m] = late
        knot_all[m] = knot
    return col, late_all, knot_all, p, across


def paint_wood_top(U, V, isl, C):
    o = Out(U.shape)
    X, Y = U, V
    col, late, knot, p, across = tabletop_wood(X, Y, 0.0)
    # large scale tone + aging
    tone = fbm(X * 1.6, Y * 1.6, 4, seed=3)
    col = col * (0.88 + 0.22 * tone)[..., None]
    # worn zone in front of the operator: lighter, rougher
    worn = np.exp(-((X / 0.48) ** 2 + ((Y + 0.30) / 0.13) ** 2))
    worn = worn * (0.6 + 0.4 * fbm(X * 9, Y * 9, 3, seed=4))
    col = col * (1 + 0.16 * worn)[..., None]
    # edge grime (sides/back), rubbed-light front edge
    dx = S.TW / 2 - np.abs(X)
    dy = S.TD / 2 - np.abs(Y)
    dedge = np.minimum(dx, dy)
    grime = (1 - smoothstep(0.0, 0.05, dedge)) * (0.5 + 0.5 * fbm(X * 25, Y * 25, 3, seed=6))
    grime = grime * np.where(Y < -S.TD / 2 + 0.06, 0.25, 1.0)
    col = col * (1 - 0.28 * grime)[..., None]
    rough = 0.50 + 0.08 * (fbm(X * 6, Y * 6, 3, seed=8) - 0.5) + 0.16 * worn + 0.12 * grime
    rough = rough + 0.02 * late
    height = late * 0.000015
    # plank seams + cupping + small height offsets
    seam_d = PLANK_W / 2 - np.abs(across)
    seam = 1 - smoothstep(0.0, 0.0016, seam_d)
    edge_round = 1 - smoothstep(0.0, 0.004, seam_d)
    cup = (across / (PLANK_W / 2)) ** 2
    dz = np.choose(p, [pl["dz"] for pl in _PLANKS])
    height = height - 0.00030 * (1 - cup) + dz - 0.0006 * edge_round ** 2 - 0.0009 * seam
    col = col * (1 - 0.55 * seam - 0.18 * edge_round)[..., None]
    rough = rough + 0.35 * seam
    # stains
    rings = C.table.s("rings", X, Y)
    spill = C.table.s("spill", X, Y) * smoothstep(0.35, 0.7, fbm(X * 30, Y * 30, 3, seed=12))
    stain = np.clip(rings * 0.8 + spill * 0.55, 0, 1)
    col = col * (1 - 0.45 * stain)[..., None]
    col = mix3(col, C3(0.20, 0.11, 0.05), stain * 0.25)
    rough = rough + 0.10 * rings - 0.08 * spill
    # scratches
    sl = C.table.s("scr_light", X, Y)
    sd = C.table.s("scr_dark", X, Y)
    col = mix3(col, col * 1.55 + 0.05, sl * 0.55)
    col = mix3(col, C3(0.10, 0.06, 0.03), sd * 0.5)
    rough = rough + 0.22 * sl + 0.15 * sd
    height = height - 0.00016 * sl - 0.00022 * sd
    # dents
    dents = C.table.s("dents", X, Y)
    height = height - 0.00030 * dents
    col = col * (1 - 0.10 * dents)[..., None]
    # burns
    burn = C.table.s("burns", X, Y) * (0.75 + 0.25 * fbm(X * 300, Y * 300, 3, seed=16))
    col = mix3(col, col * C3(0.62, 0.52, 0.45), smoothstep(0.03, 0.35, burn))
    col = mix3(col, C3(0.13, 0.07, 0.035), smoothstep(0.35, 0.8, burn) * 0.85)
    col = mix3(col, C3(0.05, 0.035, 0.03), smoothstep(0.8, 1.0, burn) * 0.8)
    rough = rough + 0.30 * smoothstep(0.1, 0.8, burn)
    height = height - 0.00018 * smoothstep(0.5, 1.0, burn)
    # powder + ash residue
    pw = C.table.s("powder", X, Y)
    pw = pw * np.where(pw < 0.5, smoothstep(0.45, 0.75, fbm(X * 260, Y * 260, 3, seed=14)), 1.0)
    blend(o, np.ones_like(X), col=col)
    blend(o, pw, col=C3(0.90, 0.90, 0.88), rough=None)
    ash = C.table.s("ash", X, Y)
    ashc = mix3(C3(0.45, 0.44, 0.42), C3(0.08, 0.08, 0.08), fbm(X * 900, Y * 900, 2, seed=15))
    blend(o, ash * 0.9, col=ashc)
    o.rough = np.clip(rough + 0.45 * pw + 0.4 * ash, 0.05, 1.0)
    o.height = height + 0.00008 * pw
    o.metal[:] = 0.0
    return o


def paint_wood_under(U, V, isl, C):
    """Underside: bottom cap uses (x, -y)."""
    o = Out(U.shape)
    X, Y = U, -V
    col, late, knot, p, across = tabletop_wood(X, Y, S.TT)
    col = col * 1.08 + 0.03
    tone = fbm(X * 3, Y * 3, 3, seed=21)
    col = col * (0.9 + 0.2 * tone)[..., None]
    # saw marks + pencil marks
    saw = 0.5 + 0.5 * np.sin(X * 2 * math.pi / 0.004 + 6 * fbm(X * 3, Y * 40, 2, seed=22))
    col = col * (0.95 + 0.07 * saw)[..., None]
    seam = 1 - smoothstep(0.0, 0.0016, PLANK_W / 2 - np.abs(across))
    col = col * (1 - 0.5 * seam)[..., None]
    pencil = (1 - smoothstep(0.0003, 0.0007, seg_dist(X, Y, -0.30, 0.10, 0.12, 0.10))) * 0.6
    pencil = np.maximum(pencil, (1 - smoothstep(0.0003, 0.0007, seg_dist(X, Y, 0.12, 0.10, 0.12, 0.06))) * 0.6)
    col = mix3(col, C3(0.25, 0.25, 0.27), pencil)
    o.col = col
    o.rough = 0.78 + 0.1 * tone
    o.height = late * 0.00006 + 0.00005 * saw - 0.0008 * seam
    return o


def paint_wood_edge(U, V, isl, C):
    o = Out(U.shape)
    x, y, idx, segs = band_xy(U, isl)
    z = np.clip(V, S.TOP - S.TT, S.TOP)
    depth = S.TOP - z
    side = isl.params.get("side")
    if side in ("front", "back"):
        col, late, knot, p, across = tabletop_wood(x, y + np.where(y > 0, -0.0005, 0.0005), depth)
        rough = 0.46 + 0.1 * fbm(x * 8, z * 60, 3, seed=33)
    else:
        # end grain of the 4 planks
        p, across = plank_of(y)
        col = np.zeros(U.shape + (3,), np.float32)
        late = np.zeros(U.shape, np.float32)
        for k in range(4):
            m = p == k
            if not np.any(m):
                continue
            P = _PLANKS[k]
            dx = across[m] - P["pith"][0]
            dz = P["pith"][1] - depth[m]
            r = np.sqrt(dx * dx + dz * dz)
            t = r / P["spacing"] + (fbm(across[m] * 60, depth[m] * 60, 3, seed=P["seed"]) - 0.5) * 1.2
            ring = t - np.floor(t)
            lt = smoothstep(0.5, 0.92, ring) * (1 - smoothstep(0.955, 0.999, ring))
            ang = np.arctan2(dz, dx)
            rays = smoothstep(0.82, 0.95, vnoise(ang * 120, r * 50, P["seed"] + 3))
            c = mix3(EARLY * 0.78, LATE * 0.75, lt * 0.9) * np.asarray(P["tint"], np.float32)
            c = c * (1 - 0.15 * rays)[..., None]
            col[m] = c
            late[m] = lt
        seam = 1 - smoothstep(0.0, 0.0015, PLANK_W / 2 - np.abs(across))
        col = col * (1 - 0.6 * seam)[..., None]
        rough = 0.62 + 0.1 * fbm(y * 20, z * 60, 3, seed=34)
    # wear along the top arris (chamfer): lighter, rougher; dirt towards the bottom
    top_w = smoothstep(S.TOP - 0.004, S.TOP + 0.0015, V) * (0.5 + 0.5 * fbm(U * 40, V * 40, 3, seed=35))
    col = col * (1 + 0.35 * top_w)[..., None]
    rough = rough + 0.2 * top_w
    dings = smoothstep(0.78, 0.9, fbm(U * 90, V * 90, 3, seed=36))
    col = col * (1 - 0.25 * dings)[..., None]
    o.col = col
    o.rough = np.clip(rough, 0, 1)
    o.height = late * 0.00002 - 0.0002 * dings
    return o


def _piece_wood(o, along, across, depth, seed, pith, tint, spacing=0.0048, dark=0.95):
    late, streak, knot = wood_core(along, across, depth, seed, pith, spacing)
    col = wood_color(late, streak, knot, tint, dark)
    o.col = col
    o.rough = 0.47 + 0.08 * (fbm(along * 5, across * 30, 3, seed=seed + 1) - 0.5)
    o.height = late * 0.00002
    return late


def paint_wood_leg(U, V, isl, C):
    o = Out(U.shape)
    x, y, idx, segs = band_xy(U, isl)
    seed = isl.params.get("seed", 1)
    _piece_wood(o, V, x, y, seed, (0.09, 0.03 * (seed % 3 - 1)), (1.0, 0.97, 0.95), 0.0042, 0.93)
    # scuffs & dirt near the floor, kicks
    low = 1 - smoothstep(0.02, 0.16, V)
    dirt = low * (0.4 + 0.6 * fbm(U * 60, V * 60, 3, seed=seed + 4))
    o.col = o.col * (1 - 0.45 * dirt)[..., None]
    scuff = smoothstep(0.72, 0.86, fbm(U * 30, V * 12, 4, seed=seed + 5)) * (1 - smoothstep(0.1, 0.45, V))
    o.col = mix3(o.col, o.col * 1.5 + 0.05, scuff * 0.6)
    o.rough = o.rough + 0.25 * dirt + 0.2 * scuff
    # chamfered corners worn lighter
    return o


def paint_wood_apron(U, V, isl, C):
    o = Out(U.shape)
    side = isl.params.get("side")
    face = isl.params.get("face")
    seed = {"front": 51, "back": 52, "left": 53, "right": 54}[side] + (10 if face == "in" else 0)
    if face == "bot":
        X, Y = U, -V
        along = X if side in ("front", "back") else Y
        across = Y if side in ("front", "back") else X
        _piece_wood(o, along, across, 0.1, seed, (0.02, 0.09), (1.0, 0.97, 0.95), 0.0045, 0.9)
        o.rough = o.rough + 0.15
        return o
    x, y, idx, segs = band_xy(U, isl)
    along = x if side in ("front", "back") else y
    z = V
    _piece_wood(o, along, z - (S.TOP - S.TT - 0.05), 0.03, seed, (0.0, 0.08), (1.02, 0.98, 0.95), 0.0045,
                0.92 if face == "out" else 0.8)
    if face == "in":
        o.rough = o.rough + 0.2
    return o


def paint_wood_str(U, V, isl, C):
    o = Out(U.shape)
    seed = isl.params.get("seed", 5)
    _piece_wood(o, U + V * 0.1, V, 0.02, seed, (0.01, 0.05), (1.0, 0.96, 0.94), 0.0045, 0.9)
    # foot wear: lighter, rubbed
    w = smoothstep(0.55, 0.8, fbm(U * 6, V * 40, 3, seed=seed + 2))
    o.col = mix3(o.col, o.col * 1.4 + 0.04, w * 0.5)
    o.rough = o.rough + 0.2 * w
    dirt = smoothstep(0.5, 0.8, fbm(U * 40, V * 40, 3, seed=seed + 3))
    o.col = o.col * (1 - 0.3 * dirt)[..., None]
    return o


# =============================================================================
# LAPTOP
# =============================================================================
ALU = C3(0.155, 0.158, 0.168)


def alu_base(o, U, V, seed, brushed_axis=0):
    streak = fbm(U * (6 if brushed_axis == 0 else 900), V * (900 if brushed_axis == 0 else 6), 3, seed=seed)
    o.col[:] = ALU
    o.col = o.col * (0.95 + 0.1 * streak)[..., None]
    o.metal[:] = 1.0
    o.rough = 0.36 + 0.08 * streak + 0.06 * (fbm(U * 20, V * 20, 3, seed=seed + 1) - 0.5)


def paint_lap_deck(U, V, isl, C):
    o = Out(U.shape)
    alu_base(o, U, V, 61)
    # keyboard well
    kx0, kx1 = -0.142, 0.142
    ky0, ky1 = -0.018, 0.090
    hm, lg, sh = C.kb
    inkb = (U > kx0 - 0.0015) & (U < kx1 + 0.0015) & (V > ky0 - 0.0015) & (V < ky1 + 0.0015)
    wellm = (1 - smoothstep(-0.0012, 0.0, rrect_sdf(U, V, 0, (ky0 + ky1) / 2, kx1 - kx0 + 0.003, ky1 - ky0 + 0.003,
                                                      0.003)))
    key = sample(C.blurred("kbh", hm, 2.0), U, V, kx0, ky0, kx1, ky1) * wellm
    leg = sample(lg, U, V, kx0, ky0, kx1, ky1) * wellm
    shine = sample(sh, U, V, kx0, ky0, kx1, ky1) * wellm
    blend(o, wellm, col=C3(0.025, 0.025, 0.028), rough=0.7, metal=0.0)
    keycol = mix3(C3(0.035, 0.035, 0.04), C3(0.70, 0.70, 0.72), leg)
    blend(o, key, col=keycol, rough=0.55 - 0.22 * shine, metal=0.0)
    o.height = -0.0013 * wellm + 0.0012 * key
    # trackpad
    tsd = rrect_sdf(U, V, 0.0, -0.071, 0.116, 0.074, 0.004)
    tp = 1 - smoothstep(-0.0004, 0.0, tsd)
    rim = (1 - smoothstep(0.0, 0.0006, np.abs(tsd + 0.0003)))
    blend(o, tp, col=C3(0.20, 0.205, 0.215), rough=0.24, metal=0.0)
    blend(o, rim, col=C3(0.62, 0.62, 0.64), rough=0.18, metal=1.0)
    o.height = o.height - 0.0004 * tp
    # speaker grille dots
    grill = (np.abs(U) < 0.128) & (V > 0.0935) & (V < 0.1045)
    gx = (U / 0.0014) - np.floor(U / 0.0014) - 0.5
    gy = (V / 0.0014) - np.floor(V / 0.0014) - 0.5
    dots = (1 - smoothstep(0.18, 0.3, np.sqrt(gx * gx + gy * gy))) * grill
    blend(o, dots, col=C3(0.01, 0.01, 0.01), rough=0.9, metal=0.0)
    o.height = o.height - 0.0003 * dots
    # power button
    pb = 1 - smoothstep(0.0, 0.0004, rrect_sdf(U, V, 0.143, 0.101, 0.010, 0.0055, 0.002))
    blend(o, pb, col=C3(0.06, 0.06, 0.065), rough=0.5, metal=0.0)
    # palm grease / smudges and dust
    sm = smudges(U, V, 64, 40, 0.6) * (V < -0.02)
    o.rough = o.rough - 0.12 * sm
    o.col = o.col * (1 - 0.06 * sm)[..., None]
    dust = smoothstep(0.90, 0.96, fbm(U * 300, V * 300, 2, seed=65)) * 0.25
    blend(o, dust, col=C3(0.85, 0.85, 0.84), rough=0.9, metal=0.0)
    return o


def paint_lap_band(U, V, isl, C):
    o = Out(U.shape)
    alu_base(o, U, V, 71, brushed_axis=0)
    x, y, idx, segs = band_xy(U, isl)
    W, D = isl.params.get("w"), isl.params.get("d")
    lid = isl.params.get("lid", False)
    if not lid:
        # ports on the left/right edges
        z = V
        left = x < -W / 2 + 0.001
        right = x > W / 2 - 0.001
        ports = np.zeros_like(U, dtype=np.float32)
        for (cy, w, h, cz) in [(0.040, 0.0125, 0.0045, 0.0075), (0.010, 0.0085, 0.0026, 0.0075), (-0.018, 0.015, 0.0045, 0.0075)]:
            ports = np.maximum(ports, (1 - smoothstep(0.0, 0.0003, rrect_sdf(y, z, cy, cz, w, h, 0.0008))) * left)
        for (cy, w, h, cz) in [(0.040, 0.0085, 0.0026, 0.0075), (0.012, 0.0085, 0.0026, 0.0075)]:
            ports = np.maximum(ports, (1 - smoothstep(0.0, 0.0003, rrect_sdf(y, z, cy, cz, w, h, 0.0008))) * right)
        jack = (1 - smoothstep(0.0012, 0.0016, np.sqrt((y + 0.030) ** 2 + (z - 0.0075) ** 2))) * right
        ports = np.maximum(ports, jack)
        blend(o, ports, col=C3(0.01, 0.01, 0.012), rough=0.8, metal=0.0)
        o.height = -0.0008 * ports
        # rear vents
        back = y > D / 2 - 0.001
        vent = back * (np.abs(x) < 0.12) * (V > 0.004) * (V < 0.011) * (np.sin(x / 0.0025 * 2 * math.pi) > 0.2)
        blend(o, vent.astype(np.float32), col=C3(0.02, 0.02, 0.02), rough=0.8, metal=0.0)
        o.height = o.height - 0.0005 * vent
        chamf = smoothstep(S.LAP_H - 0.0006, S.LAP_H + 0.0004, V)
    else:
        chamf = smoothstep(-0.0006, 0.0004, V)
    blend(o, chamf, col=C3(0.72, 0.72, 0.74), rough=0.16, metal=1.0)
    return o


def paint_lap_lid_back(U, V, isl, C):
    o = Out(U.shape)
    alu_base(o, U, V, 81, brushed_axis=0)
    ix = -U
    iy = -V
    logo = sample(C.logo, ix, iy, -S.LAP_W / 2, -S.LID_L / 2, S.LAP_W / 2, S.LID_L / 2)
    blend(o, logo, col=C3(0.75, 0.75, 0.78), rough=0.08, metal=1.0)
    st = sample(C.stickers, ix, iy, -S.LAP_W / 2, -S.LID_L / 2, S.LAP_W / 2, S.LID_L / 2)
    a = st[..., 3]
    wear = smoothstep(0.55, 0.75, fbm(U * 90, V * 90, 3, seed=83))
    blend(o, a * (1 - 0.6 * wear), col=st[..., :3], rough=0.5, metal=0.0)
    o.height = o.height + 0.00012 * a
    scr = smoothstep(0.83, 0.9, ridged(U * 60, V * 6, 3, seed=84)) * 0.7
    o.col = o.col * (1 + 0.3 * scr)[..., None]
    o.rough = o.rough + 0.1 * scr
    fp = sum(fingerprint(U, V, fx, fy, 0.009, a_, 85 + i) for i, (fx, fy, a_) in
             enumerate([(0.12, 0.08, 0.3), (-0.13, 0.09, -0.5), (0.10, -0.06, 1.2)]))
    o.rough = o.rough + 0.12 * fp
    return o


def paint_lap_bezel(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = C3(0.018, 0.018, 0.02)
    o.rough[:] = 0.32
    o.metal[:] = 0.0
    cx, cy = 0.0, S.SCREEN_RECT[3] + 0.012
    d = np.sqrt((U - cx) ** 2 + (V - cy) ** 2)
    lens = 1 - smoothstep(0.0012, 0.0015, d)
    blend(o, lens, col=C3(0.03, 0.04, 0.07), rough=0.05)
    ring = (1 - smoothstep(0.0003, 0.0005, np.abs(d - 0.0015)))
    blend(o, ring, col=C3(0.12, 0.12, 0.13), rough=0.2)
    glint = 1 - smoothstep(0.0, 0.0004, np.sqrt((U - cx - 0.0004) ** 2 + (V - cy - 0.0004) ** 2))
    blend(o, glint, col=C3(0.25, 0.35, 0.6))
    for bx in (-0.150, 0.150):
        b = 1 - smoothstep(0.0012, 0.0016, np.sqrt(((U - bx) / 2.0) ** 2 + (V - (S.SCREEN_RECT[3] + 0.010)) ** 2))
        blend(o, b, col=C3(0.05, 0.05, 0.05), rough=0.8)
    dust = smoothstep(0.90, 0.96, fbm(U * 300, V * 300, 2, seed=91)) * 0.25
    blend(o, dust, col=C3(0.6, 0.6, 0.6), rough=0.9)
    return o


def paint_lap_screen(U, V, isl, C):
    o = Out(U.shape)
    x0, y0, x1, y1 = S.SCREEN_RECT
    ui = sample(C.screen, U, V, x0, y0, x1, y1)[..., :3]
    o.emit = ui
    o.col = ui * 0.30 + 0.01
    fp = sum(fingerprint(U, V, fx, fy, 0.010, a_, 95 + i) for i, (fx, fy, a_) in
             enumerate([(0.13, -0.05, 0.4), (-0.12, 0.02, -0.3)]))
    o.rough = 0.07 + 0.10 * fp + 0.03 * fbm(U * 40, V * 40, 3, seed=96)
    o.metal[:] = 0.0
    return o


def paint_lap_hinge(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = C3(0.14, 0.14, 0.15)
    o.metal[:] = 1.0
    o.rough = 0.35 + 0.1 * fbm(U * 200, V * 20, 3, seed=97)
    return o


def paint_led(U, V, isl, C):
    o = Out(U.shape)
    cols = [C3(0.55, 0.85, 1.0), C3(1.0, 0.55, 0.08), C3(0.9, 0.95, 1.0), C3(0.35, 1.0, 0.45)]
    slot = np.clip(np.floor(U / 0.0045), 0, 3).astype(np.int64)
    c = np.stack([cols[k] for k in range(4)])[slot]
    cu = (U - slot * 0.0045 - 0.00175) / 0.00175
    cv = (V - 0.0009) / 0.0009
    g = np.clip(1.2 - np.sqrt(cu ** 2 + cv ** 2) * 0.6, 0.3, 1.0)
    o.emit = c * g[..., None]
    o.col = c * 0.6
    o.rough[:] = 0.2
    return o


# =============================================================================
# PISTOL
# =============================================================================
BLACK_METAL = C3(0.075, 0.075, 0.08)
POLY = C3(0.042, 0.042, 0.046)
STEEL = C3(0.55, 0.55, 0.56)


def _edge_wear(o, sdf, seed, U, V, width=0.0014, amount=1.0):
    n = fbm(U * 260, V * 260, 3, seed=seed)
    w = (1 - smoothstep(0.0, width, np.abs(sdf))) * smoothstep(0.52, 0.70, n) * amount
    blend(o, w * 0.8, col=STEEL, rough=0.3, metal=1.0)
    return w


def _engrave(o, U, V, txt, cx, cy, h, seed, fontname="sans-bold", depth=0.00012, col=None):
    m = ct_art.text_mask(txt, fontname, 64, 4)
    H, W = m.shape
    w = h * W / H
    t = sample(m, U, V, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    inside = (np.abs(U - cx) < w / 2) & (np.abs(V - cy) < h / 2)
    t = t * inside
    o.height = o.height - depth * t
    if col is not None:
        blend(o, t, col=col)
    return t


def paint_gun_slide_side(U, V, isl, C):
    o = Out(U.shape)
    X, Y = U, V
    o.col[:] = BLACK_METAL
    o.metal[:] = 0.85
    o.rough = 0.42 + 0.08 * fbm(X * 300, Y * 300, 3, seed=101)
    P = isl.params["poly"]
    sdf = poly_sdf(X, Y, P)
    # machining streaks along the slide
    o.rough = o.rough + 0.07 * (fbm(X * 25, Y * 1800, 3, seed=105) - 0.5)
    # rear serrations (wide, clearly readable)
    ser = (X > -0.0885) & (X < -0.0615) & (Y > 0.0040) & (Y < 0.0250)
    ph = (X - -0.0885) / 0.0030
    fr = ph - np.floor(ph)
    prof = np.clip(np.abs(fr - 0.5) * 4 - 0.4, 0, 1)
    groove = (1 - prof) * ser
    o.height = o.height - 0.0008 * groove
    blend(o, groove, col=C3(0.02, 0.02, 0.022), rough=0.6)
    ridge = (np.abs(fr - 0.5) > 0.42) * ser
    blend(o, ridge * 0.6, col=C3(0.22, 0.22, 0.23), rough=0.3, metal=1.0)
    # ejection port + barrel hood
    pd = rrect_sdf(X, Y, 0.031, 0.022, 0.044, 0.02, 0.0015)
    gap = (1 - smoothstep(0.0, 0.0008, np.abs(pd))) * (Y > 0.0120)
    hood = (1 - smoothstep(-0.0008, 0.0, pd)) * (Y > 0.013)
    blend(o, hood, col=C3(0.30, 0.30, 0.31), rough=0.28, metal=1.0)
    blend(o, gap, col=C3(0.003, 0.003, 0.003), rough=0.8)
    o.height = o.height - 0.0006 * hood - 0.0018 * gap
    # slide / frame parting line near the bottom edge
    part = (1 - smoothstep(0.0, 0.0004, np.abs(Y - 0.0022))) * (X > -0.090) * (X < 0.090)
    blend(o, part, col=C3(0.01, 0.01, 0.01))
    _engrave(o, X, Y, "9x19", 0.030, 0.0205, 0.0036, 102, depth=0.00012, col=C3(0.45, 0.45, 0.46))
    # extractor
    ex = 1 - smoothstep(0.0, 0.0003, rrect_sdf(X, Y, 0.004, 0.0185, 0.008, 0.0028, 0.0006))
    o.height = o.height + 0.0002 * ex
    blend(o, (1 - smoothstep(0.0, 0.0002, np.abs(rrect_sdf(X, Y, 0.004, 0.0185, 0.008, 0.0028, 0.0006)))),
          col=C3(0.01, 0.01, 0.01))
    _edge_wear(o, sdf, 103, X, Y, 0.0016, 1.0)
    fp = fingerprint(X, Y, 0.06, 0.012, 0.008, 0.5, 104)
    o.rough = o.rough + 0.1 * fp
    return o


def paint_gun_slide_band(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = BLACK_METAL
    o.metal[:] = 0.85
    o.rough = 0.42 + 0.08 * fbm(U * 300, V * 300, 3, seed=111)
    edge = np.abs(V) - 0.0105
    _edge_wear(o, np.maximum(edge, 0) * 0.3, 112, U, V, 0.0010, 0.35)
    # top serrations on the rear part: find x via band segs
    x, y, idx, segs = band_xy(U, isl)
    top = (y > 0.0265) & (x < -0.062) & (x > -0.088)
    ph = x / 0.0026
    groove = (np.abs(ph - np.floor(ph) - 0.5) < 0.23) & top & (np.abs(V) > 0.009)
    o.height = o.height - 0.0005 * groove
    return o


def paint_gun_muzzle(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = BLACK_METAL
    o.metal[:] = 0.85
    o.rough[:] = 0.4
    x, y, idx, segs = band_xy(U, isl)
    d = np.sqrt((y - 0.0125) ** 2 + V ** 2)
    bore = 1 - smoothstep(0.0041, 0.0044, d)
    crown = (1 - smoothstep(0.0055, 0.0058, d)) * (1 - bore)
    blend(o, crown, col=C3(0.35, 0.35, 0.36), rough=0.25, metal=1.0)
    blend(o, bore, col=C3(0.0, 0.0, 0.0), rough=0.9, metal=0.0)
    o.height = -0.0025 * bore - 0.0003 * crown
    return o


def paint_gun_poly(U, V, isl, C, kind):
    o = Out(U.shape)
    X, Y = U, V
    o.col[:] = POLY
    o.metal[:] = 0.0
    o.rough = 0.52 + 0.06 * fbm(X * 400, Y * 400, 3, seed=121)
    P = isl.params.get("poly")
    sdf = poly_sdf(X, Y, P) if P else np.zeros_like(X)
    if kind == "gun_frame_side":
        # pins, slide stop, takedown lever, serial plate
        for (px, py, r) in [(-0.022, -0.0045, 0.0012), (-0.041, -0.006, 0.0012), (0.004, -0.0065, 0.0011)]:
            pin = 1 - smoothstep(r - 0.0002, r, np.sqrt((X - px) ** 2 + (Y - py) ** 2))
            blend(o, pin, col=C3(0.3, 0.3, 0.31), rough=0.35, metal=1.0)
            o.height = o.height - 0.0001 * pin
        lever = 1 - smoothstep(0.0, 0.0003, rrect_sdf(X, Y, -0.030, -0.0015, 0.018, 0.0035, 0.001))
        blend(o, lever, col=C3(0.10, 0.10, 0.105), rough=0.4, metal=0.8)
        o.height = o.height + 0.0005 * lever
        tk = 1 - smoothstep(0.0, 0.0003, rrect_sdf(X, Y, 0.012, -0.0045, 0.009, 0.004, 0.0008))
        blend(o, tk, col=C3(0.10, 0.10, 0.105), rough=0.4, metal=0.8)
        o.height = o.height + 0.0003 * tk
        plate = 1 - smoothstep(0.0, 0.0002, rrect_sdf(X, Y, 0.042, -0.007, 0.020, 0.0048, 0.0006))
        blend(o, plate, col=C3(0.42, 0.42, 0.43), rough=0.35, metal=1.0)
        _engrave(o, X, Y, "BKV 471", 0.042, -0.007, 0.0028, 122, fontname="mono-bold", depth=0.0001,
                 col=C3(0.1, 0.1, 0.1))
        # accessory rail channel near the bottom of the dust cover
        rail = (1 - smoothstep(0.0, 0.0005, np.abs(Y + 0.0078))) * (X > 0.018) * (X < 0.066)
        o.height = o.height - 0.0005 * rail
        blend(o, rail, col=C3(0.01, 0.01, 0.012))
        # moulded texture on the frame
        o.height = o.height + 0.00003 * fbm(X * 1500, Y * 1500, 2, seed=127)
    if kind == "gun_grip_side":
        # stipple texture inside the grip panel
        panel = 1 - smoothstep(-0.0055, -0.0040, sdf)
        panel = panel * smoothstep(-0.019, -0.023, Y)
        f1, cid = worley(X / 0.00125, Y / 0.00125, 123)
        bump = (1 - np.clip(f1 * 1.15, 0, 1)) ** 1.5
        o.height = o.height + 0.00035 * bump * panel - 0.00015 * panel
        o.rough = o.rough + 0.22 * panel
        o.col = o.col * (1 + panel * (0.35 * (bump - 0.5) + 0.1 * (cid - 0.5)))[..., None]
        border = (1 - smoothstep(0.0, 0.0005, np.abs(sdf + 0.0048))) * smoothstep(-0.019, -0.023, Y)
        o.height = o.height - 0.0003 * border
        # magazine base seam + mag release
        base = (Y < -0.100) & (Y > -0.1025) | False
        seam = 1 - smoothstep(0.0, 0.0004, np.abs((X * 0.342 + Y * 0.94) - (-0.1285)))
        blend(o, seam, col=C3(0.0, 0.0, 0.0))
        o.height = o.height - 0.0004 * seam
        mr = 1 - smoothstep(0.0, 0.0003, rrect_sdf(X, Y, -0.050, -0.022, 0.0075, 0.009, 0.0012))
        blend(o, mr, col=POLY * 1.4, rough=0.6)
        o.height = o.height + 0.0005 * mr
        # thumb rest smooth oval
    if kind == "gun_guard_side":
        pass
    # scuffs on edges (polymer gets lighter/greyish when worn)
    if P:
        w = (1 - smoothstep(0.0, 0.0015, np.abs(sdf))) * smoothstep(0.58, 0.75, fbm(X * 220, Y * 220, 3, seed=125)) * 0.6
        blend(o, w, col=C3(0.13, 0.13, 0.135), rough=0.65)
    dust = smoothstep(0.90, 0.96, fbm(X * 300, Y * 300, 2, seed=126)) * 0.3
    blend(o, dust, col=C3(0.5, 0.5, 0.5), rough=0.9)
    return o


def paint_gun_poly_band(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = POLY
    o.rough = 0.55 + 0.06 * fbm(U * 400, V * 400, 3, seed=131)
    if isl.params.get("grip"):
        f1, cid = worley(U / 0.00125, V / 0.00125, 132)
        bump = (1 - np.clip(f1 * 1.15, 0, 1)) ** 1.5
        pm = (np.abs(V) < 0.0105).astype(np.float32)
        o.height = (0.00035 * bump - 0.00015) * pm
        o.rough = o.rough + 0.2 * pm
        o.col = o.col * (1 + pm * 0.3 * (bump - 0.5))[..., None]
    elif "band_segs" in isl.params:
        x, y, idx, segs = band_xy(U, isl)
        # rail cross slots under the dust cover (frame band bottom edge)
        slot = ((np.abs(x - 0.036) < 0.0016) | (np.abs(x - 0.050) < 0.0016)) & (y < -0.0105) & (np.abs(V) < 0.0095)
        blend(o, slot.astype(np.float32), col=C3(0.008, 0.008, 0.01))
        o.height = o.height - 0.0008 * slot
    w = smoothstep(0.009, 0.0125, np.abs(V)) * smoothstep(0.58, 0.75, fbm(U * 220, V * 220, 3, seed=133)) * 0.6
    blend(o, w, col=C3(0.13, 0.13, 0.135), rough=0.65)
    return o


def paint_gun_trigger(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = POLY * 1.1
    o.rough[:] = 0.5
    return o


def paint_gun_sight(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = BLACK_METAL
    o.metal[:] = 0.8
    o.rough[:] = 0.45
    if "band_segs" in isl.params:
        x, y, idx, segs = band_xy(U, isl)
        if isl.params.get("front"):
            # white dot on the rear face (edge 3)
            e3 = idx == 3
            d = np.sqrt((y - 0.0295) ** 2 + V ** 2)
            dot = (1 - smoothstep(0.0008, 0.0011, d)) * e3
            blend(o, dot, col=C3(0.92, 0.92, 0.9), rough=0.5, metal=0.0)
        if isl.params.get("rear"):
            e3 = idx == 3
            u_ = (np.abs(np.abs(V) - 0.0026) < 0.0006) & (y > 0.028) & e3
            b_ = (np.abs(y - 0.0285) < 0.0006) & (np.abs(V) < 0.0032) & e3
            blend(o, (u_ | b_).astype(np.float32), col=C3(0.92, 0.92, 0.9), rough=0.5, metal=0.0)
            notch = (idx == 2) & (np.abs(V) < 0.0017)
            blend(o, notch.astype(np.float32), col=C3(0.01, 0.01, 0.01))
            o.height = o.height - 0.0006 * notch
    return o


def paint_mag(U, V, isl, C, kind):
    o = Out(U.shape)
    o.col[:] = C3(0.05, 0.05, 0.055)
    o.rough = 0.5 + 0.06 * fbm(U * 400, V * 400, 3, seed=141)
    if kind == "mag_side":
        X, Y = U, V
        # witness holes along the back edge with numbers
        for k, yy in enumerate([0.030, 0.050, 0.070, 0.090, 0.108]):
            d = np.sqrt((X + 0.0105) ** 2 + (Y - yy) ** 2)
            hole = 1 - smoothstep(0.0011, 0.0014, d)
            blend(o, hole, col=C3(0.55, 0.42, 0.20), rough=0.3, metal=1.0)   # brass visible through holes
            o.height = o.height - 0.0005 * hole
        base = Y < 0.0095
        blend(o, base.astype(np.float32), col=C3(0.035, 0.035, 0.04), rough=0.55)
        seam = 1 - smoothstep(0.0, 0.0004, np.abs(Y - 0.0095))
        blend(o, seam, col=C3(0.0, 0.0, 0.0))
        o.height = o.height - 0.0003 * seam
        # steel lips at top
        lips = Y > 0.113
        blend(o, lips.astype(np.float32), col=C3(0.3, 0.3, 0.31), rough=0.35, metal=1.0)
    if kind == "mag_lips":
        x, y, idx, segs = band_xy(U, isl)
        # cartridge seen from above: bullet towards +x (front of mag)
        inner = np.abs(V) < 0.0048
        brass = inner & (x < 0.004)
        copper = inner & (x >= 0.004)
        blend(o, brass.astype(np.float32), col=C3(0.78, 0.58, 0.30), rough=0.25, metal=1.0)
        blend(o, copper.astype(np.float32), col=C3(0.72, 0.42, 0.25), rough=0.3, metal=1.0)
        prof = np.sqrt(np.clip(1 - (V / 0.0048) ** 2, 0, 1))
        o.height = 0.0012 * prof * inner
        lips = (np.abs(V) > 0.0048) & (np.abs(V) < 0.0095)
        blend(o, lips.astype(np.float32), col=C3(0.3, 0.3, 0.31), rough=0.35, metal=1.0)
    return o


def paint_round(U, V, isl, C, kind):
    o = Out(U.shape)
    brass = C3(0.80, 0.60, 0.30)
    if kind == "round_side":
        arc = isl.params.get("arc")
        # arc[5] ~ case mouth (index of profile point (4.85,19.1))
        mouth = arc[2] if arc else 0.0240
        cop = V > mouth + 0.0002
        o.col[:] = brass
        o.metal[:] = 1.0
        o.rough = 0.24 + 0.1 * fbm(U * 500, V * 500, 3, seed=151)
        tarn = smoothstep(0.55, 0.75, fbm(U * 300, V * 300, 3, seed=152))
        o.col = o.col * (1 - 0.25 * tarn)[..., None]
        blend(o, cop.astype(np.float32), col=C3(0.74, 0.43, 0.27), rough=0.3, metal=1.0)
        groove = (V > arc[1] + 0.0009) & (V < arc[1] + 0.0017) if arc else np.zeros_like(U, bool)
        blend(o, groove.astype(np.float32), col=brass * 0.55, rough=0.4)
        o.height = o.height - 0.0003 * groove
    else:
        d = np.sqrt(U * U + V * V)
        o.col[:] = brass * 0.95
        o.metal[:] = 1.0
        o.rough[:] = 0.3
        pr = 1 - smoothstep(0.0021, 0.0024, d)
        blend(o, pr, col=C3(0.65, 0.62, 0.58), rough=0.35)
        ring = (1 - smoothstep(0.0002, 0.0003, np.abs(d - 0.0024)))
        blend(o, ring, col=C3(0.2, 0.15, 0.08))
        o.height = -0.0002 * ring
    return o


# =============================================================================
# MONEY
# =============================================================================
def _bill_img(C, back):
    return C.bill_b if back else C.bill_f


def bill_surface(o, U, V, img, seed, dirt=0.25, crumple=1.0):
    rgb = sample(img, U, V, -S.BILL_W / 2, -S.BILL_H / 2, S.BILL_W / 2, S.BILL_H / 2)[..., :3]
    r = random.Random(seed)
    tint = C3(1.0 + r.uniform(-0.04, 0.03), 1.0 + r.uniform(-0.03, 0.03), 1.0 + r.uniform(-0.05, 0.02))
    col = rgb * tint[None, None, :] * r.uniform(0.80, 0.90)
    d = fbm(U * 30 + seed, V * 30, 4, seed=seed) * dirt
    col = col * (1 - 0.35 * d)[..., None]
    # worn edges (light) and soft grime
    ed = np.minimum(S.BILL_W / 2 - np.abs(U), S.BILL_H / 2 - np.abs(V))
    col = col * (1 - 0.12 * (1 - smoothstep(0.0, 0.004, ed)))[..., None]
    o.col = col
    o.rough = 0.78 + 0.08 * fbm(U * 80, V * 80, 3, seed=seed + 1)
    o.metal[:] = 0.0
    wr = ridged(U * 45 + seed * 0.1, V * 45, 4, seed=seed + 2)
    o.height = 0.00010 * crumple * wr
    # fold crease across the middle for some notes
    if r.random() < 0.6:
        cr = 1 - smoothstep(0.0, 0.0012, np.abs(U - r.uniform(-0.01, 0.01)))
        o.height = o.height + 0.00025 * cr
        o.col = o.col * (1 - 0.08 * cr)[..., None]


def paint_bill(U, V, isl, C, back):
    o = Out(U.shape)
    seed = isl.params.get("seed", 1)
    bill_surface(o, U, V, _bill_img(C, back), seed)
    return o


def paint_bundle_top(U, V, isl, C):
    o = Out(U.shape)
    seed = isl.params.get("seed", 1)
    r = random.Random(seed)
    # top note slightly offset/rotated on the stack
    ox, oy, a = r.uniform(-0.0015, 0.0015), r.uniform(-0.001, 0.001), r.uniform(-0.012, 0.012)
    ca, sa = math.cos(a), math.sin(a)
    Uu = (U - ox) * ca + (V - oy) * sa
    Vv = -(U - ox) * sa + (V - oy) * ca
    bill_surface(o, Uu, Vv, C.bill_f, seed, dirt=0.12, crumple=0.5)
    # outside the top note -> edges of notes below
    out = np.maximum(np.abs(Uu) - (S.BILL_W / 2 - 0.0008), np.abs(Vv) - (S.BILL_H / 2 - 0.0008))
    edgem = smoothstep(-0.0002, 0.0002, out)
    blend(o, edgem, col=C3(0.66, 0.70, 0.62), rough=0.85)
    # strap
    sr = sample(C.strap, U, V, -0.016, -S.BILL_H / 2, 0.016, S.BILL_H / 2)[..., :3]
    sm = (np.abs(U) < 0.016).astype(np.float32)
    blend(o, sm, col=sr * (0.92 + 0.1 * fbm(U * 200, V * 200, 2, seed=seed + 5))[..., None], rough=0.62)
    o.height = o.height * (1 - sm) + 0.00012 * sm
    return o


def paint_bundle_bottom(U, V, isl, C):
    """Bottom note of a strapped bundle (bottom cap uses (x, -y))."""
    o = Out(U.shape)
    seed = isl.params.get("seed", 1)
    bill_surface(o, U, V, C.bill_b, seed, dirt=0.15, crumple=0.5)
    sm = (np.abs(U) < 0.016).astype(np.float32)
    blend(o, sm, col=C3(0.80, 0.62, 0.22), rough=0.62)
    return o


def paint_gun_hidden(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = BLACK_METAL if isl.params.get("metal") else POLY
    o.metal[:] = 0.8 if isl.params.get("metal") else 0.0
    o.rough = 0.45 + 0.05 * fbm(U * 300, V * 300, 3, seed=128)
    return o


def paint_bundle_side(U, V, isl, C):
    o = Out(U.shape)
    seed = isl.params.get("seed", 1)
    x, y, idx, segs = band_xy(U, isl)
    lines = fbm(U * 8, V * 9000, 2, seed=seed)
    fine = vnoise(U * 30, V * 16000, seed + 1)
    col = C3(0.68, 0.72, 0.64) * (0.80 + 0.2 * lines + 0.08 * fine)[..., None]
    wav = fbm(U * 60, V * 60, 3, seed=seed + 2)
    col = col * (1 - 0.18 * smoothstep(0.6, 0.8, wav))[..., None]
    o.col = col
    o.rough[:] = 0.88
    o.height = 0.00003 * lines
    long_side = (idx == 0) | (idx == 2)
    strap = long_side & (np.abs(x) < 0.016)
    sc = C3(0.83, 0.64, 0.23) * (0.92 + 0.1 * fbm(U * 200, V * 200, 2, seed=seed + 3))[..., None]
    blend(o, strap.astype(np.float32), col=sc, rough=0.6)
    o.height = o.height + 0.0001 * strap
    return o


def paint_strap(U, V, isl, C):
    o = Out(U.shape)
    # strap text runs along the strap (sample the rotated face)
    sr = sample(C.strap, -V * (0.032 / 0.030), U * (0.0663 / 0.085), -0.016, -S.BILL_H / 2, 0.016, S.BILL_H / 2)[..., :3]
    o.col = sr
    torn = smoothstep(0.030, 0.0425, U) * smoothstep(0.4, 0.7, fbm(U * 400, V * 400, 3, seed=161))
    o.col = o.col * (1 - 0.3 * torn)[..., None]
    o.rough[:] = 0.65
    o.height = 0.00008 * fbm(U * 200, V * 200, 3, seed=162)
    return o


# =============================================================================
# MIRROR, POWDER, CARD, RAZOR, ROLLED NOTE
# =============================================================================
def paint_mirror(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = C3(0.90, 0.90, 0.91)
    o.metal[:] = 1.0
    o.rough = 0.035 + 0.02 * fbm(U * 60, V * 60, 3, seed=171)
    fp = sum(fingerprint(U, V, fx, fy, 0.009, a_, 172 + i) for i, (fx, fy, a_) in
             enumerate([(0.085, 0.055, 0.2), (-0.09, -0.06, 1.0), (0.07, -0.065, -0.6), (-0.04, 0.062, 0.4)]))
    sm = smudges(U, V, 175, 25, 0.62)
    o.rough = o.rough + 0.14 * fp + 0.12 * sm
    dust = C.mirror.s("dust", U, V)
    # soft haze around lines / pile (fine powder film)
    haze = np.zeros_like(U)
    for (cx, cy, L, Wd, ang) in S.LINES:
        a_ = math.radians(ang)
        xx = (U - cx) * math.cos(a_) + (V - cy) * math.sin(a_)
        yy = -(U - cx) * math.sin(a_) + (V - cy) * math.cos(a_)
        haze = np.maximum(haze, np.exp(-(yy / (Wd * 0.8)) ** 2) * (1 - smoothstep(L * 0.42, L * 0.56, np.abs(xx))))
    haze = np.maximum(haze, np.exp(-(((U - S.PILE[0]) ** 2 + (V - S.PILE[1]) ** 2) / 0.019 ** 2)))
    haze = haze * smoothstep(0.45, 0.85, fbm(U * 220, V * 220, 4, seed=176)) * 0.22
    blend(o, np.clip(haze + dust * 0.85, 0, 1), col=C3(0.93, 0.93, 0.91), rough=0.88, metal=0.0)
    o.height = 0.00004 * dust
    return o


def paint_mirror_edge(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = C3(0.30, 0.42, 0.38)
    o.rough[:] = 0.25
    top = smoothstep(S.MIR_T - 0.001, S.MIR_T, V)
    blend(o, top, col=C3(0.55, 0.65, 0.62), rough=0.15)
    return o


def paint_powder(U, V, isl, C):
    o = Out(U.shape)
    seed = isl.params.get("seed", 1)
    g = fbm(U * 1500, V * 1500, 3, seed=seed)
    gr = vnoise(U * 2500, V * 2500, seed + 3)
    o.col = C3(0.80, 0.80, 0.785) * (0.88 + 0.08 * g + 0.06 * gr)[..., None]
    o.rough[:] = 0.92
    o.height = 0.00012 * g + 0.00005 * gr
    return o


def paint_card(U, V, isl, C):
    o = Out(U.shape)
    rgb = sample(C.card, U, V, -0.0428, -0.027, 0.0428, 0.027)[..., :3]
    o.col = rgb
    o.rough = 0.45 + 0.05 * fbm(U * 200, V * 200, 3, seed=181)
    lum = rgb.mean(-1)
    emb = smoothstep(0.45, 0.7, lum) * (V < -0.004)
    o.height = 0.0001 * emb
    blend(o, emb, rough=0.3, metal=0.6)
    # EMV chip
    cd = rrect_sdf(U, V, -0.028, 0.004, 0.0115, 0.0092, 0.0015)
    chip = 1 - smoothstep(-0.0002, 0.0, cd)
    blend(o, chip, col=C3(0.86, 0.70, 0.38), rough=0.28, metal=1.0)
    lines = chip * ((np.abs(U + 0.028) < 0.0004) | (np.abs(V - 0.004) < 0.0003) |
                    ((np.abs(np.abs(V - 0.004) - 0.0022) < 0.0003) & (np.abs(U + 0.028) > 0.002)))
    blend(o, lines.astype(np.float32), col=C3(0.45, 0.35, 0.16))
    # powder on the leading edge
    pe = (1 - smoothstep(0.0005, 0.003, U - 0.0428 + 0.004)) * 0 + smoothstep(-0.0268, -0.0240, -V) * 0
    edge = smoothstep(0.0255, 0.0272, -V) * smoothstep(0.45, 0.7, fbm(U * 500, V * 500, 3, seed=182)) * 0.8
    blend(o, edge, col=C3(0.93, 0.93, 0.91), rough=0.9, metal=0.0)
    return o


def paint_card_edge(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = C3(0.08, 0.08, 0.09)
    o.rough[:] = 0.5
    return o


def paint_razor(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = C3(0.60, 0.61, 0.62)
    o.metal[:] = 1.0
    o.rough = 0.26 + 0.1 * fbm(U * 300, V * 30, 3, seed=191)
    bevel = V < -0.0072
    grind = 0.5 + 0.5 * np.sin(U / 0.0003 * 6.28 + 3 * fbm(U * 100, V * 100, 2, seed=192))
    blend(o, bevel.astype(np.float32), col=C3(0.72, 0.73, 0.74), rough=0.15)
    o.rough = o.rough + 0.05 * grind * bevel
    o.height = 0.00002 * grind * bevel
    _engrave(o, U, V, "SK5", -0.012, 0.002, 0.004, 193, depth=0.00005, col=C3(0.35, 0.35, 0.36))
    rust = smoothstep(0.80, 0.88, fbm(U * 250, V * 250, 4, seed=194)) * 0.7
    blend(o, rust, col=C3(0.42, 0.20, 0.08), rough=0.8, metal=0.0)
    pw = smoothstep(0.6, 0.75, fbm(U * 400, V * 400, 3, seed=195)) * (V < -0.004)
    blend(o, pw, col=C3(0.92, 0.92, 0.9), rough=0.9, metal=0.0)
    return o


def paint_steel_edge(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = C3(0.62, 0.62, 0.63)
    o.metal[:] = 1.0
    o.rough[:] = 0.3
    return o


def paint_rolled_bill(U, V, isl, C):
    o = Out(U.shape)
    circ = 2 * math.pi * isl.params.get("rref", 0.0045)
    bx = S.BILL_W / 2 - circ + U
    by = (V - 0.0045) - S.BILL_H / 2
    bill_surface(o, bx, by, C.bill_f, 7, dirt=0.3, crumple=1.3)
    return o


def paint_rolled_end(U, V, isl, C):
    o = Out(U.shape)
    r = np.sqrt(U * U + V * V)
    th = np.arctan2(V, U)
    sp = 0.5 + 0.5 * np.sin((r / 0.0006 + th / (2 * math.pi)) * 2 * math.pi)
    o.col = mix3(C3(0.55, 0.60, 0.52), C3(0.80, 0.84, 0.76), sp)
    hole = 1 - smoothstep(0.0016, 0.0022, r)
    blend(o, hole, col=C3(0.02, 0.02, 0.02))
    o.rough[:] = 0.85
    o.height = -0.001 * hole + 0.0001 * sp
    return o


# =============================================================================
# ASHTRAY + CIGARETTES
# =============================================================================
def paint_ash(U, V, isl, C, kind):
    o = Out(U.shape)
    o.col[:] = C3(0.60, 0.60, 0.61)
    o.metal[:] = 1.0
    notch_ang = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
    if kind in ("ash_out", "ash_in"):
        rref = isl.params.get("rref", 0.05)
        th = U / rref
        streak = fbm(U * 12, V * 1500, 3, seed=201)
        o.rough = 0.28 + 0.1 * streak
        o.col = o.col * (0.94 + 0.08 * streak)[..., None]
        dn = np.min([np.abs(np.arctan2(np.sin(th - a), np.cos(th - a))) for a in notch_ang], axis=0)
        soot_n = np.exp(-(dn / 0.25) ** 2)
        if kind == "ash_in":
            arc = isl.params.get("arc", [0] * 8)
            depth = np.clip((V - arc[5]) / max(arc[-1] - arc[5], 1e-4), 0, 1) if len(arc) > 6 else 0.5
            ash = smoothstep(0.2, 0.9, depth + 0.4 * (fbm(U * 120, V * 120, 4, seed=202) - 0.5))
            soot = np.clip(soot_n * 0.9 + 0.3 * fbm(U * 60, V * 60, 3, seed=203), 0, 1)
            blend(o, soot * 0.8, col=C3(0.10, 0.09, 0.08), rough=0.7, metal=0.0)
            blend(o, ash, col=C3(0.38, 0.37, 0.35), rough=0.95, metal=0.0)
        else:
            fp = fingerprint(U, V, 0.08, 0.015, 0.008, 0.2, 204)
            o.rough = o.rough + 0.12 * fp
            soot = soot_n * smoothstep(0.012, 0.03, V) * 0.6
            blend(o, soot, col=C3(0.12, 0.11, 0.1), rough=0.6, metal=0.2)
    elif kind == "ash_rim":
        th = np.arctan2(V, U)
        streak = fbm(np.sqrt(U * U + V * V) * 1500, th * 3, 3, seed=205)
        o.rough = 0.28 + 0.1 * streak
        dn = np.min([np.abs(np.arctan2(np.sin(th - a), np.cos(th - a))) for a in notch_ang], axis=0)
        soot = np.exp(-(dn / 0.22) ** 2) * (0.6 + 0.4 * fbm(U * 200, V * 200, 3, seed=206))
        blend(o, soot, col=C3(0.08, 0.07, 0.06), rough=0.75, metal=0.0)
        burn = np.exp(-(dn / 0.08) ** 2) * 0.8
        blend(o, burn, col=C3(0.25, 0.14, 0.06), rough=0.7, metal=0.0)
    else:  # bowl
        f1, cid = worley(U / 0.0016, V / 0.0016, 207)
        n = fbm(U * 300, V * 300, 4, seed=208)
        base = mix3(C3(0.30, 0.29, 0.28), C3(0.62, 0.61, 0.58), n)
        base = mix3(base, C3(0.06, 0.06, 0.06), smoothstep(0.7, 0.95, cid) * 0.9)
        tob = smoothstep(0.82, 0.95, fbm(U * 500, V * 500, 3, seed=209))
        base = mix3(base, C3(0.28, 0.17, 0.08), tob)
        o.col = base
        o.metal[:] = 0.0
        o.rough[:] = 0.95
        o.height = 0.0006 * n + 0.0002 * (1 - np.clip(f1, 0, 1))
    return o


def cig_side(o, U, V, isl, burning):
    arc = isl.params.get("arc")
    prof = isl.params.get("prof")
    s_f0 = arc[1]
    s_f1 = arc[2]
    s_end = arc[3]
    filt = (V >= s_f0 - 0.001) & (V < s_f1)
    # cork tipping
    cork = mix3(C3(0.80, 0.52, 0.26), C3(0.62, 0.36, 0.16), smoothstep(0.55, 0.9, fbm(U * 900, V * 900, 2, seed=211)))
    paper = C3(0.93, 0.92, 0.89)
    col = np.where(filt[..., None], cork, paper[None, None, :])
    ring = (np.abs(V - (s_f1 - 0.0012)) < 0.0004)
    col = np.where(ring[..., None], C3(0.85, 0.75, 0.45)[None, None, :], col)
    o.col = col.astype(np.float32)
    o.rough = np.where(filt, 0.62, 0.72).astype(np.float32)
    # slight yellowing near the filter on the paper
    yel = (V > s_f1) * np.exp(-((V - s_f1) / 0.004) ** 2)
    o.col = mix3(o.col, C3(0.85, 0.78, 0.55), yel * 0.35)
    if not burning:
        # charred end
        ch = smoothstep(s_end - 0.004, s_end, V)
        o.col = mix3(o.col, C3(0.35, 0.28, 0.2), ch * 0.8)
        o.col = mix3(o.col, C3(0.05, 0.04, 0.04), smoothstep(s_end - 0.0012, s_end + 0.0005, V))
        o.height = 0.00008 * ridged(U * 300, V * 300, 3, seed=212)
        o.col = o.col * (1 - 0.12 * fbm(U * 200, V * 200, 3, seed=213))[..., None]
    else:
        ch = smoothstep(s_end - 0.002, s_end, V)
        o.col = mix3(o.col, C3(0.45, 0.35, 0.25), ch)


def paint_butt(U, V, isl, C):
    o = Out(U.shape)
    cig_side(o, U, V, isl, False)
    return o


def paint_cig(U, V, isl, C):
    o = Out(U.shape)
    cig_side(o, U, V, isl, True)
    return o


def paint_butt_filter_end(U, V, isl, C):
    o = Out(U.shape)
    r = np.sqrt(U * U + V * V)
    n = fbm(U * 2500, V * 2500, 2, seed=221)
    o.col = mix3(C3(0.92, 0.88, 0.78), C3(0.62, 0.45, 0.22), (1 - smoothstep(0.0005, 0.0025, r)) * 0.8)
    o.col = o.col * (0.9 + 0.1 * n)[..., None]
    o.rough[:] = 0.9
    o.height = 0.00005 * n
    return o


def paint_butt_burnt_end(U, V, isl, C):
    o = Out(U.shape)
    n = fbm(U * 1500, V * 1500, 3, seed=222)
    o.col = mix3(C3(0.04, 0.035, 0.03), C3(0.35, 0.33, 0.3), n)
    o.rough[:] = 0.95
    o.height = 0.0001 * n
    return o


def paint_ember(U, V, isl, C):
    o = Out(U.shape)
    n = fbm(U * 1200, V * 3000, 3, seed=231)
    o.col = mix3(C3(0.30, 0.06, 0.02), C3(0.6, 0.18, 0.04), n)
    o.emit = mix3(C3(0.9, 0.22, 0.02), C3(1.0, 0.58, 0.12), n)
    o.rough[:] = 0.9
    return o


def paint_ash_cyl(U, V, isl, C):
    o = Out(U.shape)
    n = fbm(U * 1000, V * 1000, 3, seed=241)
    o.col = mix3(C3(0.28, 0.27, 0.26), C3(0.62, 0.61, 0.58), n)
    blk = smoothstep(0.75, 0.9, fbm(U * 1500, V * 1500, 2, seed=242))
    o.col = mix3(o.col, C3(0.06, 0.06, 0.06), blk)
    o.rough[:] = 0.97
    o.height = 0.00015 * n
    return o


# =============================================================================
# CAN, GLASS, PHONE
# =============================================================================
def paint_can_side(U, V, isl, C):
    o = Out(U.shape)
    arc = isl.params.get("arc")
    rref = isl.params.get("rref")
    circ = 2 * math.pi * rref
    b0, b1 = arc[1], arc[2]
    if C._can is None:
        C._can = ct_art.can_label(circ * 1000, (b1 - b0) * 1000)
    lab = sample(C._can, U, V, 0.0, b0, circ, b1)[..., :3]
    inb = (V > b0) & (V < b1)
    o.col[:] = C3(0.78, 0.78, 0.80)
    o.metal[:] = 1.0
    o.rough = 0.22 + 0.06 * fbm(U * 200, V * 200, 3, seed=251)
    m = inb.astype(np.float32)
    lum = lab.mean(-1)
    goldish = smoothstep(0.45, 0.6, lab[..., 0] - lab[..., 2] + 0.3) * smoothstep(0.35, 0.55, lum)
    blend(o, m, col=lab, rough=0.28, metal=0.35)
    blend(o, m * goldish, metal=0.9, rough=0.22)
    dent = smoothstep(0.7, 0.85, fbm(U * 40, V * 40, 3, seed=252))
    o.height = -0.00015 * dent
    scr = smoothstep(0.93, 0.97, ridged(U * 300, V * 30, 3, seed=253)) * m
    blend(o, scr * 0.5, col=C3(0.8, 0.8, 0.82), metal=1.0, rough=0.3)
    return o


def paint_can_top(U, V, isl, C):
    o = Out(U.shape)
    r = np.sqrt(U * U + V * V)
    o.col[:] = C3(0.80, 0.80, 0.82)
    o.metal[:] = 1.0
    o.rough = 0.25 + 0.05 * fbm(U * 300, V * 300, 3, seed=261)
    rings = 0.5 + 0.5 * np.sin(r / 0.0012 * 6.28)
    o.height = 0.00006 * rings * (r > 0.018)
    # opening (dark) and pull tab
    op = 1 - smoothstep(0.0, 0.0004, rrect_sdf(U, V, 0.0, 0.0125, 0.017, 0.012, 0.005))
    blend(o, op, col=C3(0.02, 0.02, 0.02), rough=0.9, metal=0.0)
    o.height = o.height - 0.002 * op
    tab = 1 - smoothstep(0.0, 0.0004, rrect_sdf(U, V, 0.0, -0.003, 0.012, 0.022, 0.005))
    hole = 1 - smoothstep(0.0, 0.0004, rrect_sdf(U, V, 0.0, -0.008, 0.006, 0.006, 0.0025))
    blend(o, tab * (1 - hole), col=C3(0.85, 0.85, 0.87), rough=0.2)
    o.height = o.height + 0.0006 * tab * (1 - hole) - 0.0004 * hole
    riv = 1 - smoothstep(0.0012, 0.0016, np.sqrt(U * U + (V - 0.0035) ** 2))
    o.height = o.height + 0.0003 * riv
    return o


def paint_glass(U, V, isl, C):
    o = Out(U.shape)
    part = isl.params.get("part")
    o.col[:] = C3(0.30, 0.33, 0.325)
    o.metal[:] = 0.0
    o.rough[:] = 0.03
    base_a = {"out": 0.10, "in": 0.07, "rim": 0.55, "base": 0.35, "bottle": 0.13}[part]
    if part == "rim":
        o.col[:] = C3(0.62, 0.68, 0.66)
    fp = 0
    if part == "out":
        fp = sum(fingerprint(U, V, fx, fy, 0.008, a_, 271 + i) for i, (fx, fy, a_) in
                 enumerate([(0.05, 0.05, 0.3), (0.16, 0.06, -0.4), (0.10, 0.03, 1.2)]))
    sm = smudges(U, V, 275, 40, 0.64)
    o.rough = o.rough + 0.15 * fp + 0.10 * sm
    o.alpha = np.clip(base_a + 0.08 * fp + 0.06 * sm, 0, 1).astype(np.float32)
    if part == "base":
        o.col[:] = C3(0.36, 0.44, 0.40)
    if part == "bottle":
        arc = isl.params.get("arc")
        # thick glass base + neck a bit more opaque
        o.alpha = o.alpha + 0.25 * (1 - smoothstep(0.0, 0.008, V)) + 0.12 * smoothstep(arc[4], arc[5], V)
        o.col[:] = C3(0.27, 0.30, 0.29)
    return o


def paint_bottle_label(U, V, isl, C):
    o = Out(U.shape)
    arc = isl.params.get("arc")
    rref = isl.params.get("rref")
    circ = 2 * math.pi * rref
    if getattr(C, "_lab", None) is None:
        C._lab = ct_art.bottle_label(circ * 1000, (arc[1] - arc[0]) * 1000)
    o.col = sample(C._lab, U, V, 0.0, arc[0], circ, arc[1])[..., :3] * 0.92
    o.rough = 0.62 + 0.08 * fbm(U * 200, V * 200, 3, seed=301)
    stain = smoothstep(0.62, 0.8, fbm(U * 60, V * 60, 3, seed=302))
    o.col = o.col * (1 - 0.2 * stain)[..., None]
    o.height = 0.00004 * fbm(U * 400, V * 400, 2, seed=303)
    return o


def paint_bottle_cap(U, V, isl, C, top):
    o = Out(U.shape)
    o.col[:] = C3(0.035, 0.032, 0.03)
    o.metal[:] = 0.6
    o.rough[:] = 0.35
    if not top:
        ribs = 0.5 + 0.5 * np.sin(U / 0.0011 * 2 * math.pi)
        o.height = 0.0001 * ribs
        band = (V > 0.2536) & (V < 0.2590)
        blend(o, band.astype(np.float32), col=C3(0.78, 0.60, 0.26), metal=1.0, rough=0.25)
    else:
        r = np.sqrt(U * U + V * V)
        blend(o, (np.abs(r - 0.0095) < 0.0006).astype(np.float32), col=C3(0.78, 0.60, 0.26), metal=1.0, rough=0.25)
    return o


def paint_whisky(U, V, isl, C, top):
    o = Out(U.shape)
    o.col[:] = C3(0.40, 0.165, 0.025)
    o.rough[:] = 0.03
    if top:
        r = np.sqrt(U * U + V * V)
        o.col = mix3(o.col, C3(0.26, 0.10, 0.015), smoothstep(0.028, 0.034, r))
    return o


def paint_phone_band(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = C3(0.16, 0.16, 0.17)
    o.metal[:] = 1.0
    o.rough = 0.32 + 0.06 * fbm(U * 200, V * 200, 3, seed=281)
    x, y, idx, segs = band_xy(U, isl)
    right = x > 0.0715 / 2 - 0.001
    left = x < -0.0715 / 2 + 0.001
    btn = np.zeros_like(U, dtype=np.float32)
    for (cy, L) in [(0.030, 0.014)]:
        btn = np.maximum(btn, (1 - smoothstep(0.0, 0.0003, rrect_sdf(y, V, cy, 0.004, L, 0.0022, 0.0009))) * right)
    for (cy, L) in [(0.038, 0.009), (0.024, 0.009)]:
        btn = np.maximum(btn, (1 - smoothstep(0.0, 0.0003, rrect_sdf(y, V, cy, 0.004, L, 0.0022, 0.0009))) * left)
    o.height = 0.0003 * btn
    outline = (1 - smoothstep(0.0, 0.0002, np.abs(rrect_sdf(y, V, 0.030, 0.004, 0.014, 0.0022, 0.0009)))) * right
    blend(o, outline, col=C3(0.05, 0.05, 0.05))
    top = smoothstep(0.0078 - 0.0008, 0.0078 + 0.0004, V)
    blend(o, top, col=C3(0.02, 0.02, 0.025), rough=0.08, metal=0.0)
    return o


def paint_phone_bezel(U, V, isl, C):
    o = Out(U.shape)
    o.col[:] = C3(0.012, 0.012, 0.014)
    o.rough[:] = 0.06
    spk = 1 - smoothstep(0.0, 0.0002, rrect_sdf(U, V, 0.0, 0.1465 / 2 - 0.0028, 0.010, 0.0008, 0.0004))
    blend(o, spk, col=C3(0.06, 0.06, 0.07), rough=0.7)
    cam = 1 - smoothstep(0.0009, 0.0011, np.sqrt((U - 0.012) ** 2 + (V - (0.1465 / 2 - 0.0028)) ** 2))
    blend(o, cam, col=C3(0.03, 0.04, 0.07), rough=0.05)
    return o


def paint_phone_screen(U, V, isl, C):
    o = Out(U.shape)
    w, h = 0.0715 - 0.0064, 0.1465 - 0.011
    ui = sample(C.phone, U, V, -w / 2, -h / 2, w / 2, h / 2)[..., :3]
    o.emit = ui * 0.9
    o.col = ui * 0.25 + 0.005
    fp = fingerprint(U, V, 0.01, -0.03, 0.009, 0.7, 291)
    o.rough = 0.05 + 0.18 * fp
    return o


# =============================================================================
# dispatch
# =============================================================================
def _dispatch(kind):
    table = {
        "wood_top": paint_wood_top, "wood_under": paint_wood_under, "wood_edge": paint_wood_edge,
        "wood_leg": paint_wood_leg, "wood_apron": paint_wood_apron, "wood_str": paint_wood_str,
        "lap_deck": paint_lap_deck, "lap_band": paint_lap_band, "lap_lid_back": paint_lap_lid_back,
        "lap_bezel": paint_lap_bezel, "lap_screen": paint_lap_screen, "lap_hinge": paint_lap_hinge, "led": paint_led,
        "gun_slide_side": paint_gun_slide_side, "gun_slide_band": paint_gun_slide_band, "gun_muzzle": paint_gun_muzzle,
        "gun_frame_side": lambda U, V, i, C: paint_gun_poly(U, V, i, C, "gun_frame_side"),
        "gun_grip_side": lambda U, V, i, C: paint_gun_poly(U, V, i, C, "gun_grip_side"),
        "gun_guard_side": lambda U, V, i, C: paint_gun_poly(U, V, i, C, "gun_guard_side"),
        "gun_poly_band": paint_gun_poly_band, "gun_trigger": paint_gun_trigger, "gun_sight": paint_gun_sight,
        "mag_side": lambda U, V, i, C: paint_mag(U, V, i, C, "mag_side"),
        "mag_band": lambda U, V, i, C: paint_mag(U, V, i, C, "mag_band"),
        "mag_lips": lambda U, V, i, C: paint_mag(U, V, i, C, "mag_lips"),
        "round_side": lambda U, V, i, C: paint_round(U, V, i, C, "round_side"),
        "round_base": lambda U, V, i, C: paint_round(U, V, i, C, "round_base"),
        "bill_front": lambda U, V, i, C: paint_bill(U, V, i, C, False),
        "bill_back": lambda U, V, i, C: paint_bill(U, V, i, C, True),
        "bundle_top": paint_bundle_top, "bundle_side": paint_bundle_side, "strap": paint_strap,
        "bundle_bottom": paint_bundle_bottom, "gun_hidden": paint_gun_hidden,
        "mirror": paint_mirror, "mirror_edge": paint_mirror_edge, "powder": paint_powder,
        "card": paint_card, "card_edge": paint_card_edge, "razor": paint_razor, "steel_edge": paint_steel_edge,
        "rolled_bill": paint_rolled_bill, "rolled_end": paint_rolled_end,
        "ash_out": lambda U, V, i, C: paint_ash(U, V, i, C, "ash_out"),
        "ash_in": lambda U, V, i, C: paint_ash(U, V, i, C, "ash_in"),
        "ash_rim": lambda U, V, i, C: paint_ash(U, V, i, C, "ash_rim"),
        "ash_bowl": lambda U, V, i, C: paint_ash(U, V, i, C, "ash_bowl"),
        "butt": paint_butt, "cig": paint_cig, "butt_filter_end": paint_butt_filter_end,
        "butt_burnt_end": paint_butt_burnt_end, "ember": paint_ember, "ash": paint_ash_cyl,
        "ash_end": paint_butt_burnt_end,
        "can_side": paint_can_side, "can_top": paint_can_top, "glass": paint_glass,
        "whisky": lambda U, V, i, C: paint_whisky(U, V, i, C, False),
        "whisky_top": lambda U, V, i, C: paint_whisky(U, V, i, C, True),
        "phone_band": paint_phone_band, "phone_bezel": paint_phone_bezel, "phone_screen": paint_phone_screen,
        "bottle_label": paint_bottle_label,
        "bottle_cap": lambda U, V, i, C: paint_bottle_cap(U, V, i, C, False),
        "bottle_cap_top": lambda U, V, i, C: paint_bottle_cap(U, V, i, C, True),
    }
    return table.get(kind)


def paint_atlas(reg, placed, verbose=True):
    """Returns dict of 2048 maps: col (sRGB), rough, metal, normal (GL, xyz), emit (sRGB), alpha, mask."""
    import time
    N = ATLAS * SS
    col = np.zeros((N, N, 3), np.float32)
    rough = np.ones((N, N), np.float32)
    metal = np.zeros((N, N), np.float32)
    nrm = np.zeros((N, N, 3), np.float32)
    nrm[..., 2] = 1.0
    emit = np.zeros((N, N, 3), np.float32)
    alpha = np.ones((N, N), np.float32)
    mask = np.zeros((ATLAS, ATLAS), np.int32)
    C = Ctx(reg)
    t0 = time.time()
    for n, (iid, p) in enumerate(sorted(placed.items())):
        isl = reg.islands[iid]
        fn = _dispatch(isl.kind)
        x0, y0, w, h = p.x * SS, p.y * SS, p.w * SS, p.h * SS
        xs = (x0 + np.arange(w) + 0.5) / SS
        ys = (y0 + np.arange(h) + 0.5) / SS
        XA, YA = np.meshgrid(xs, ys)
        umin, vmin, umax, vmax = p.b
        if not p.rot:
            U = umin + (XA - p.x - PAD) / p.k
            V = vmax - (YA - p.y - PAD) / p.k
        else:
            V = vmin + (XA - p.x - PAD) / p.k
            U = umin + (YA - p.y - PAD) / p.k
        if fn is None:
            o = Out(U.shape)
            o.col[:] = C3(1, 0, 1)
            print("  !! no painter for", iid, isl.kind)
        else:
            o = fn(U, V, isl, C)
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        col[sl] = np.clip(o.col, 0, 1)
        rough[sl] = np.clip(o.rough, 0.02, 1)
        metal[sl] = np.clip(o.metal, 0, 1)
        emit[sl] = np.clip(o.emit, 0, 1)
        alpha[sl] = np.clip(o.alpha, 0, 1)
        # normal from height (atlas tangent space, OpenGL +Y)
        hgt = o.height.astype(np.float64)
        gy, gx = np.gradient(hgt)
        kc = p.k * SS
        dhdu = gx * kc
        dhdv = -gy * kc
        nn = np.stack([-dhdu, -dhdv, np.ones_like(dhdu)], -1)
        nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
        nrm[sl] = nn.astype(np.float32)
        mask[p.y:p.y + p.h, p.x:p.x + p.w] = n + 1
    if verbose:
        print("paint: %d islands in %.1fs" % (len(placed), time.time() - t0))

    def down(a):
        if a.ndim == 3:
            return a.reshape(ATLAS, SS, ATLAS, SS, a.shape[2]).mean(axis=(1, 3))
        return a.reshape(ATLAS, SS, ATLAS, SS).mean(axis=(1, 3))
    n2 = down(nrm)
    n2 /= np.linalg.norm(n2, axis=-1, keepdims=True)
    return dict(col=down(col), rough=down(rough), metal=down(metal), normal=n2, emit=down(emit),
                alpha=down(alpha), mask=mask)
