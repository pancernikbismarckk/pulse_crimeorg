"""Procedural scene description of the 'crime table' prop (all LODs + collision).

Units: metres.  Z up.  Origin = floor level, centre of the table footprint.
The operator sits on the -Y side (front); +X is his right hand side.
"""
import math
import random
from ct_geo import (REG, Mesh, extrude, bezel_cap, lathe, grid_sheet, rounded_rect, circle,
                    rot_x, rot_y, rot_z, trans, scale, chain, place, inset)

TW, TD, TT = 1.50, 0.85, 0.045        # table width (X), depth (Y), top thickness
TOP = 0.780                          # tabletop surface height
LEG = 0.075                          # leg section
LEGX, LEGY = TW / 2 - 0.050 - LEG / 2, TD / 2 - 0.050 - LEG / 2
MM = 0.001

# texel density weights (relative).  1.0 = tabletop surface.
W = dict(
    wood_top=1.0, wood_under=0.20, wood_edge=0.75, wood_leg=0.50, wood_apron=0.48, wood_str=0.40,
    lap_deck=2.2, lap_lid_back=1.3, lap_bezel=1.7, lap_screen=2.3, lap_band=1.3, lap_hinge=1.2,
    gun=2.6, mag=2.0, round=2.2,
    bundle_top=1.55, bundle_side=1.25, bundle_bottom=0.5, bill=1.25, bill_back=1.05, fan=0.95, strap=1.4,
    gun_hidden=0.55,
    mirror=1.9, powder=2.0, card=2.2, razor=2.4, rolled=2.0,
    ash=1.5, butt=2.4, cig=2.4,
    can=1.9, glass=1.3, whisky=1.3, phone=2.0, led=6.0,
)


def R(iid, kind, wkey=None, **params):
    return REG.add(iid, kind, W.get(wkey or kind, 1.0), **params)


# =============================================================================
# TABLE
# =============================================================================
def table(lod):
    m = Mesh()
    top_z0 = TOP - TT
    # ---- tabletop
    if lod < 2:
        poly = rounded_rect(TW, TD, 0.010, seg=2)
        ct, cb = 0.0035, 0.0015
    else:
        poly = [(TW / 2, -TD / 2), (TW / 2, TD / 2), (-TW / 2, TD / 2), (-TW / 2, -TD / 2)]
        ct, cb = 0.0, 0.0
    sides = []
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if abs(mx) / (TW / 2) > abs(my) / (TD / 2):
            s = "right" if mx > 0 else "left"
        else:
            s = "back" if my > 0 else "front"
        iid = R("tbl_edge_" + s, "wood_edge", side=s)
        if lod >= 2:
            s0 = sum(math.hypot(poly[(k + 1) % n][0] - poly[k][0], poly[(k + 1) % n][1] - poly[k][1]) for k in range(i))
            s1 = s0 + math.hypot(x1 - x0, y1 - y0)
            iid = REG.alias("tbl_edge_%s_l2" % s, iid, (s0, top_z0, s1, TOP), None)
        sides.append(iid)
    extrude(m, poly, top_z0, TOP, top=R("tbl_top", "wood_top"), bot=R("tbl_under", "wood_under"),
            band=sides, ct=ct, cb=cb)
    # ---- legs
    c = 0.006
    h = LEG / 2
    if lod < 2:
        legp = [(-h + c, -h), (h - c, -h), (h, -h + c), (h, h - c), (h - c, h), (-h + c, h), (-h, h - c), (-h, -h + c)]
    else:
        legp = [(-h, -h), (h, -h), (h, h), (-h, h)]
    for k, (sx, sy) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        iid = R("tbl_leg%d" % k, "wood_leg", seed=11 + k)
        if lod >= 2:
            # square leg perimeter maps onto octagon island
            per_sq = 4 * LEG
            per_oct = 4 * (LEG - 2 * c) + 4 * c * math.sqrt(2)
            iid = REG.alias("tbl_leg%d_l2" % k, iid, (0, 0, per_sq, top_z0), (0, 0, per_oct, top_z0))
        lm = Mesh()
        extrude(lm, legp, 0.0, top_z0, band=iid)
        m.extend(lm, trans(sx * LEGX, sy * LEGY, 0))
    # ---- aprons (skirt boards)
    az0, az1 = top_z0 - 0.100, top_z0
    t = 0.022
    ox = LEGX + h - 0.012            # outer face offset
    oy = LEGY + h - 0.012
    inx, iny = LEGX - h, LEGY - h    # leg inner faces
    for side in ("front", "back", "left", "right"):
        am = Mesh()
        if side in ("front", "back"):
            y1 = -oy if side == "front" else oy
            y0 = y1 + t if side == "front" else y1 - t
            ya, yb = min(y0, y1), max(y0, y1)
            poly = [(-inx, ya), (inx, ya), (inx, yb), (-inx, yb)]
            # edge0: y=ya face (-y normal), edge2: y=yb (+y normal)
            outer_edge = 0 if side == "front" else 2
        else:
            x1 = -ox if side == "left" else ox
            x0 = x1 + t if side == "left" else x1 - t
            xa, xb = min(x0, x1), max(x0, x1)
            poly = [(xa, -iny), (xb, -iny), (xb, iny), (xa, iny)]
            # edge1: x=xb (+x normal) edge3: x=xa (-x normal)
            outer_edge = 3 if side == "left" else 1
        inner_edge = (outer_edge + 2) % 4
        band = [None] * 4
        band[outer_edge] = R("tbl_apron_%s_o" % side, "wood_apron", side=side, face="out")
        if lod < 2:
            band[inner_edge] = R("tbl_apron_%s_i" % side, "wood_apron", side=side, face="in")
        bot = R("tbl_apron_%s_b" % side, "wood_apron", side=side, face="bot") if lod < 2 else None
        extrude(am, poly, az0, az1, band=band, bot=bot)
        m.extend(am)
    # ---- stretchers (LOD0/1 only)
    if lod < 2:
        sz0, sz1 = 0.095, 0.155
        st = 0.030
        for k, sx in enumerate((-1, 1)):
            x = sx * LEGX
            poly = [(x - st / 2, -iny), (x + st / 2, -iny), (x + st / 2, iny), (x - st / 2, iny)]
            band = [None, R("tbl_str%d_a" % k, "wood_str", seed=40 + k), None, R("tbl_str%d_b" % k, "wood_str", seed=50 + k)]
            sm = Mesh()
            extrude(sm, poly, sz0, sz1, band=band, top=R("tbl_str%d_t" % k, "wood_str", seed=60 + k),
                    bot=R("tbl_str%d_u" % k, "wood_str", seed=70 + k))
            m.extend(sm)
        x0, x1 = -LEGX + st / 2, LEGX - st / 2
        poly = [(x0, -st / 2), (x1, -st / 2), (x1, st / 2), (x0, st / 2)]
        band = [R("tbl_str2_a", "wood_str", seed=80), None, R("tbl_str2_b", "wood_str", seed=81), None]
        sm = Mesh()
        extrude(sm, poly, sz0 + 0.008, sz1 - 0.008, band=band, top=R("tbl_str2_t", "wood_str", seed=82),
                bot=R("tbl_str2_u", "wood_str", seed=83))
        m.extend(sm)
    return m


# =============================================================================
# LAPTOP
# =============================================================================
LAP_W, LAP_D, LAP_H = 0.330, 0.232, 0.0165
LID_L, LID_T = 0.226, 0.0055
LID_TILT = math.radians(18.0)          # lid tilt back from vertical
SCREEN_RECT = (-0.1515, -LID_L / 2 + 0.029, 0.1515, -LID_L / 2 + 0.029 + 0.1705)
LED_SLOTS = []                          # (u0, u1, colour) inside the shared 'leds' island


def led_quad(m, center, right, up, w, h, slot):
    """Tiny emissive decal quad; right/up are unit vectors spanning it."""
    u0 = slot * 0.0045
    R("leds", "led")
    cx, cy, cz = center
    pts = []
    for (a, b) in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        pts.append((cx + right[0] * a * w + up[0] * b * h, cy + right[1] * a * w + up[1] * b * h,
                    cz + right[2] * a * w + up[2] * b * h))
    idx = [m.vert(p) for p in pts]
    uv = [(u0, 0.0), (u0 + 0.0035, 0.0), (u0 + 0.0035, 0.0018), (u0, 0.0018)]
    m.face(idx, uv, "leds", "emissive", False)


def laptop(lod):
    m = Mesh()
    seg = 3 if lod == 0 else 1
    r = 0.009 if lod < 2 else 0.0
    ch = lod == 0
    base_poly = rounded_rect(LAP_W, LAP_D, r, seg=seg) if r > 0 else rounded_rect(LAP_W, LAP_D, 0)
    extrude(m, base_poly, 0.0, LAP_H, top=R("lap_deck", "lap_deck"), band=R("lap_band", "lap_band", w=LAP_W, d=LAP_D),
            ct=0.0016 if ch else 0.0, cb=0.0010 if ch else 0.0)
    if lod == 0:
        # status LEDs on the front edge
        led_quad(m, (0.118, -LAP_D / 2 - 0.0004, 0.0075), (1, 0, 0), (0, 0, 1), 0.0032, 0.0012, 0)
        led_quad(m, (0.127, -LAP_D / 2 - 0.0004, 0.0075), (1, 0, 0), (0, 0, 1), 0.0032, 0.0012, 1)
        # power button LED on the deck (top right)
        led_quad(m, (0.143, 0.101, LAP_H + 0.0003), (1, 0, 0), (0, 1, 0), 0.0030, 0.0014, 2)
    # hinge barrel
    if lod < 2:
        hm = Mesh()
        hseg = 10 if lod == 0 else 6
        hr = 0.0056
        hl = 0.270
        prof = [(0.0, -hl / 2), (hr, -hl / 2), (hr, hl / 2), (0.0, hl / 2)]
        lathe(hm, prof, hseg, [R("lap_hinge_c0", "lap_hinge", part="cap"), R("lap_hinge", "lap_hinge", part="side"),
                               R("lap_hinge_c1", "lap_hinge", part="cap")], theta0=math.pi / hseg)
        m.extend(hm, chain(rot_y(math.pi / 2), trans(0.0, LAP_D / 2 - 0.0045, LAP_H + 0.0005)))
    # lid, built with display facing +z, screen-up = +y, hinge edge at y=-LID_L/2
    lm = Mesh()
    lid_poly = rounded_rect(LAP_W, LID_L, r, seg=seg) if r > 0 else rounded_rect(LAP_W, LID_L, 0)
    ex = extrude(lm, lid_poly, -LID_T, 0.0, top=None, bot=R("lap_lid_back", "lap_lid_back"),
                 band=R("lap_lid_band", "lap_band", w=LAP_W, d=LID_L, lid=True),
                 ct=0.0012 if ch else 0.0, cb=0.0015 if ch else 0.0)
    if lod < 2:
        bezel_cap(lm, ex["rings"][-1], ex["top_pts"], 0.0, SCREEN_RECT, 0.0008 if lod == 0 else 0.0,
                  bezel=R("lap_bezel", "lap_bezel"), screen=R("lap_screen", "lap_screen"),
                  wall=R("lap_bezel_wall", "lap_bezel") if lod == 0 else None)
        if lod == 0:   # webcam activity LED
            led_quad(lm, (0.012, SCREEN_RECT[3] + 0.012, 0.0003), (1, 0, 0), (0, 1, 0), 0.0022, 0.0022, 3)
    else:
        # whole inner face shows the screen (aliased onto the LOD0 screen island)
        pts = ex["top_pts"]
        idx = ex["rings"][-1]
        sx0, sy0, sx1, sy1 = SCREEN_RECT
        src = (-LAP_W / 2, -LID_L / 2, LAP_W / 2, LID_L / 2)
        iid = REG.alias("lap_screen_l2", "lap_screen", src, (sx0, sy0, sx1, sy1))
        lm.face(idx, [(x, y) for (x, y) in pts], iid, "emissive")
    ca, sa = math.cos(LID_TILT), math.sin(LID_TILT)
    hy, hz = LAP_D / 2 - 0.0045, LAP_H + 0.0012

    def lid_xf(p):
        x, y, z = p
        y = y + LID_L / 2              # hinge edge at y=0
        # local y -> (0, sa, ca) ; local z -> (0, -ca, sa)
        return (x, hy + y * sa - z * ca, hz + y * ca + z * sa)

    m.extend(lm, lid_xf)
    return m


# =============================================================================
# PISTOL  (generic polymer-frame 9 mm, lying on its right side up)
# =============================================================================
def mm(pts):
    return [(x * MM, y * MM) for (x, y) in pts]


SLIDE = mm([(-93, 1), (92, 1), (93, 3), (93, 21), (89.5, 27), (-90.5, 27), (-93, 24.5)])
FRAME = mm([(-88, -11), (66, -11), (71.5, -6.5), (72, 1.5), (-88, 1.5)])
GRIP = mm([(-44, -11), (-48, -26), (-55, -50), (-64, -76), (-73, -100), (-78.5, -113), (-82, -119.5),
           (-86, -121.5), (-121, -109), (-125, -104), (-121, -86), (-113, -60), (-106, -38), (-102.5, -24),
           (-102, -14), (-100, -6.5), (-95, -2.5), (-87, -1.5)])
GUARD = mm([(0, -11), (0.8, -27.5), (-2.5, -34.2), (-7, -35.6), (-50, -36.5), (-53, -31.2), (-10, -30.2),
            (-5.2, -27.2), (-4.6, -11)])
TRIGGER = mm([(-15.5, -11), (-19, -17.5), (-22.5, -24), (-26.5, -27.3), (-28.5, -26.2), (-25.5, -20),
              (-22, -11)])
MAGP = mm([(-16, 0), (20, 0), (20, 7.5), (17, 9.5), (16, 118), (13, 121), (-14, 121), (-16, 118), (-16, 9.5),
           (-19, 7.5), (-19, 1.5)])


def simplify(P, keep):
    return [P[i] for i in keep]


def pistol(lod):
    m = Mesh()
    ch = lod == 0
    # slide: front edge (index 2 -> 3) gets its own 'muzzle' island
    n = len(SLIDE)
    band = [R("gun_slide_band", "gun_slide_band", "gun", poly=SLIDE)] * n
    band = list(band)
    band[2] = R("gun_muzzle", "gun_muzzle", "gun", poly=SLIDE)
    if lod < 2:
        bs = 2 if ch else 1
        # left-side caps of parts that do not touch the table (visible from low angles)
        extrude(m, SLIDE, -12.75 * MM, 12.75 * MM, top=R("gun_slide_r", "gun_slide_side", "gun", poly=SLIDE),
                bot=R("gun_slide_l", "gun_hidden", "gun_hidden", metal=True),
                band=band, ct=2.2 * MM if ch else 0, cb=2.2 * MM if ch else 0, bseg=bs)
        extrude(m, FRAME, -12.0 * MM, 12.0 * MM, top=R("gun_frame_r", "gun_frame_side", "gun", poly=FRAME),
                bot=R("gun_frame_l", "gun_hidden", "gun_hidden"),
                band=R("gun_frame_band", "gun_poly_band", "gun"), ct=2.0 * MM if ch else 0, cb=2.0 * MM if ch else 0,
                bseg=bs)
        g = GRIP if lod == 0 else simplify(GRIP, [0, 2, 4, 6, 7, 8, 9, 11, 13, 15, 17])
        extrude(m, g, -15.0 * MM, 15.0 * MM, top=R("gun_grip_r", "gun_grip_side", "gun", poly=GRIP),
                band=R("gun_grip_band", "gun_poly_band", "gun", grip=True), ct=4.5 * MM if ch else 0,
                cb=4.5 * MM if ch else 0, bseg=3 if ch else 1)
        gd = GUARD if lod == 0 else simplify(GUARD, [0, 1, 2, 4, 5, 6, 7, 8])
        extrude(m, gd, -9.0 * MM, 9.0 * MM, top=R("gun_guard_r", "gun_guard_side", "gun", poly=GUARD),
                bot=R("gun_guard_l", "gun_hidden", "gun_hidden"),
                band=R("gun_guard_band", "gun_poly_band", "gun"), ct=1.8 * MM if ch else 0, cb=1.8 * MM if ch else 0,
                bseg=bs)
        extrude(m, TRIGGER, -3.0 * MM, 3.0 * MM, top=R("gun_trigger_r", "gun_trigger", "gun"),
                bot=R("gun_trigger_l", "gun_hidden", "gun_hidden"),
                band=R("gun_trigger_band", "gun_trigger", "gun"))
        if lod == 0:
            # sights
            fs = mm([(80, 26.5), (86, 26.5), (86, 31.5), (81, 31.5)])
            rs = mm([(-88, 26.5), (-79, 26.5), (-80, 32), (-88, 32)])
            extrude(m, fs, -1.7 * MM, 1.7 * MM, top=R("gun_fsight_r", "gun_sight", "gun"),
                    bot=R("gun_fsight_l", "gun_hidden", "gun_hidden", metal=True),
                    band=R("gun_fsight_band", "gun_sight", "gun", front=True))
            extrude(m, rs, -6.0 * MM, 6.0 * MM, top=R("gun_rsight_r", "gun_sight", "gun"),
                    bot=R("gun_rsight_l", "gun_hidden", "gun_hidden", metal=True),
                    band=R("gun_rsight_band", "gun_sight", "gun", rear=True))
    else:
        # LOD2: single silhouette slab mapped onto the LOD0 side islands
        sil = mm([(-93, 27), (-93, 1), (-102, -14), (-125, -104), (-86, -121.5), (-50, -36.5), (-2.5, -34.2),
                  (0, -11), (72, -11), (72, 1), (93, 3), (93, 21), (89.5, 27)])
        xs = [p[0] for p in sil]; ys = [p[1] for p in sil]
        per = sum(math.hypot(sil[(i + 1) % len(sil)][0] - sil[i][0], sil[(i + 1) % len(sil)][1] - sil[i][1])
                  for i in range(len(sil)))
        extrude(m, sil, -12.0 * MM, 12.0 * MM,
                top=REG.alias("gun_sil_l2", "gun_grip_r", (min(xs), min(ys), max(xs), max(ys)), None),
                band=REG.alias("gun_sil_band_l2", "gun_slide_band", (0, -0.012, per, 0.012), None))
    return m


def magazine(lod):
    m = Mesh()
    ch = lod == 0
    n = len(MAGP)
    band = [R("mag_band", "mag_band", "mag")] * n
    band = list(band)
    band[5] = R("mag_lips", "mag_lips", "mag")     # top edge (13,121)->(-14,121) shows cartridge
    P = MAGP if lod == 0 else [MAGP[i] for i in (0, 1, 2, 4, 6, 7, 8, 10)]
    if lod != 0:
        band = [R("mag_band", "mag_band", "mag")] * len(P)
        band[3] = "mag_lips"
    extrude(m, P, -11.0 * MM, 11.0 * MM, top=R("mag_side", "mag_side", "mag"), band=band,
            ct=2.0 * MM if ch else 0, cb=2.0 * MM if ch else 0, bseg=2 if ch else 1)
    return m


def cartridge(k, lod):
    m = Mesh()
    prof = mm([(0, 0), (4.95, 0), (4.9, 19.1), (4.5, 19.4), (4.3, 24.2), (2.6, 28.2), (0, 29.7)])
    prof = [(r, z) for (r, z) in prof]
    base = R("rnd%d_base" % k, "round_base", "round")
    side = R("rnd%d_side" % k, "round_side", "round")
    lathe(m, prof, 8, [base] + [side] * (len(prof) - 2), uvkind=["down"] + ["side"] * (len(prof) - 2))
    return m


# =============================================================================
# MONEY
# =============================================================================
BILL_W, BILL_H = 0.156, 0.0663
BUNDLE_T = 0.0110


def bundle(k, lod=0, bottom=False):
    m = Mesh()
    poly = [(-BILL_W / 2, -BILL_H / 2), (BILL_W / 2, -BILL_H / 2), (BILL_W / 2, BILL_H / 2), (-BILL_W / 2, BILL_H / 2)]
    extrude(m, poly, 0.0, BUNDLE_T, top=R("bnd%d_top" % k, "bundle_top", seed=100 + k),
            bot=R("bnd%d_bot" % k, "bundle_bottom", seed=150 + k) if bottom else None,
            band=R("bnd%d_side" % k, "bundle_side", seed=200 + k, w=BILL_W, d=BILL_H))
    return m


def bill_sheet(iid_front, iid_back, curl, nu, lift=0.0, fold=0.0):
    """Loose note.  curl: lift of the +x end (m).  fold: crease ridge height."""
    m = Mesh()

    def zf(x, y):
        t = (x + BILL_W / 2) / BILL_W
        z = lift + 0.00012
        if curl:
            z += curl * max(0.0, (t - 0.55) / 0.45) ** 2
        if fold:
            z += fold * max(0.0, 1.0 - abs(x) / (BILL_W * 0.18))
        return z
    grid_sheet(m, BILL_W, BILL_H, nu, 1, zf, iid_front, back_island=iid_back, thickness=0.00012)
    return m


def money(lod, rng):
    m = Mesh()
    # ------------------------------------------------------------ strapped stack
    layout = []
    for (ix, iy) in [(-0.5, -1), (0.5, -1), (-0.5, 0), (0.5, 0), (-0.5, 1), (0.5, 1)]:
        layout.append((0, ix * (BILL_W + 0.002), iy * (BILL_H + 0.002)))
    for (ix, iy) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]:
        layout.append((1, ix * (BILL_W + 0.004) + 0.006, iy * (BILL_H + 0.003) - 0.004))
    layout.append((2, -0.018, 0.012))
    layout.append((3, 0.004, 0.006))
    yaws = [1.5, -1.0, 0.5, 2.0, -1.8, 0.8, 4.0, -3.0, 2.5, -4.5, 18.0, -26.0]
    jit = [(0.001, -0.002), (-0.002, 0.001), (0.0015, 0.0), (-0.001, 0.002), (0.002, -0.001), (0.0, 0.0015),
           (0.003, -0.002), (-0.004, 0.003), (0.002, 0.003), (-0.002, -0.003), (0.01, 0.004), (-0.012, -0.006)]
    sm = Mesh()
    if lod == 0:
        for k, (layer, x, y) in enumerate(layout):
            b = bundle(k, bottom=layer > 0)
            jx, jy = jit[k]
            sm.extend(b, place(x + jx, y + jy, layer * BUNDLE_T, yaws[k]))
    else:
        # merged blocks (LOD1: 3 blocks, LOD2: 1 block)
        blocks = [(0, 2 * BILL_W + 0.004, 3 * BILL_H + 0.006, 0.0, 0.0, 0.0, BUNDLE_T),
                  (1, 2 * BILL_W + 0.006, 2 * BILL_H + 0.004, 0.006, -0.004, BUNDLE_T, BUNDLE_T),
                  (2, BILL_W + 0.02, BILL_H + 0.03, -0.006, 0.006, 2 * BUNDLE_T, 2 * BUNDLE_T)]
        if lod == 2:
            blocks = [(0, 2 * BILL_W + 0.004, 3 * BILL_H + 0.006, 0.0, 0.0, 0.0, 3 * BUNDLE_T)]
        for (bk, bw, bh, bx, by, bz, bt) in blocks:
            poly = [(-bw / 2, -bh / 2), (bw / 2, -bh / 2), (bw / 2, bh / 2), (-bw / 2, bh / 2)]
            top_src = (-bw / 2, -bh / 2, bw / 2, bh / 2)
            tgt_top = "bnd%d_top" % (10 if bk == 2 else (6 + bk))
            top = REG.alias("mblk%d_l%d_top" % (bk, lod), tgt_top, top_src, None)
            per = 2 * (bw + bh)
            band = REG.alias("mblk%d_l%d_side" % (bk, lod), "bnd%d_side" % (bk * 4 % 12), (0, bz, per, bz + bt), None)
            bm = Mesh()
            extrude(bm, poly, bz, bz + bt, top=top, band=band)
            sm.extend(bm, trans(bx, by, 0))
    m.extend(sm, place(-0.470, 0.125, TOP, 14.0))

    # ------------------------------------------------------------ burst bundle (fan)
    nfan = 7 if lod == 0 else (3 if lod == 1 else 0)
    for k in range(nfan):
        a = -58 + k * 11.0
        fid = R("fan%d" % k, "bill_front", "fan", seed=300 + k)
        s = bill_sheet(fid, None, 0.0, 1, lift=k * 0.00028)
        # pivot at the -x end of the note
        m.extend(s, chain(trans(BILL_W / 2 - 0.006, 0, 0), rot_z(math.radians(a)), trans(-0.575, -0.215, TOP)))
    if lod == 0:
        # torn paper strap lying next to the fan
        sid = R("strap_torn", "strap", "strap")
        st = Mesh()
        grid_sheet(st, 0.085, 0.030, 3, 1, lambda x, y: 0.0002 + 0.004 * max(0.0, x / 0.0425) ** 2, sid,
                   back_island=R("strap_torn_b", "strap", "strap", back=True), thickness=0.0001)
        m.extend(st, place(-0.395, -0.300, TOP, -24.0))

    # ------------------------------------------------------------ loose notes
    loose = [  # x, y, yaw, curl, back_up, fold
        (-0.660, -0.070, 100.0, 0.010, False, 0.0),
        (-0.325, -0.090, 23.0, 0.0, True, 0.002),
        (-0.610, 0.330, 97.0, 0.0, False, 0.0),
        (-0.300, 0.230, -17.0, 0.014, False, 0.0),
        (-0.360, -0.370, 175.0, 0.0, False, 0.0025),
        (-0.660, -0.270, 38.0, 0.0, True, 0.0),
        (-0.470, -0.105, 132.0, 0.008, False, 0.0),
        (-0.245, -0.075, -64.0, 0.0, False, 0.0),
    ]
    for k, (x, y, yaw, curl, back_up, fold) in enumerate(loose):
        if lod == 1 and k in (5, 7):
            continue
        if lod == 2:
            break
        kind_top = "bill_back" if back_up else "bill_front"
        fid = R("bill%d_t" % k, kind_top, "bill", seed=400 + k)
        curled = (curl > 0 or fold > 0) and lod == 0
        bid = R("bill%d_u" % k, "bill_front" if back_up else "bill_back", "bill_back", seed=500 + k) if curled else None
        s = bill_sheet(fid, bid, curl if lod == 0 else 0.0, 4 if curled else 1, lift=0.0,
                       fold=fold if lod == 0 else 0.0)
        m.extend(s, place(x, y, TOP + 0.00005 * k, yaw))
    return m


# =============================================================================
# MIRROR, POWDER, CARD, RAZOR, ROLLED NOTE
# =============================================================================
MIR_W, MIR_H, MIR_T = 0.220, 0.160, 0.004
MIR_POS = (-0.140, -0.245, 7.0)
LINES = [  # centre x, y (mirror local), length, width, angle
    (0.004, -0.036, 0.078, 0.0068, -2.0),
    (0.010, -0.012, 0.072, 0.0062, 1.5),
    (0.002, 0.012, 0.080, 0.0070, -1.0),
]
PILE = (-0.068, 0.030)
CARD_POS = (0.060, 0.040, 24.0)
RAZOR_POS = (0.075, -0.050, -32.0)
ROLL_POS = (-0.070, -0.040, 58.0)


def powder_line(iid, L, Wd, H, nsec, rng):
    m = Mesh()
    rows = []
    for i in range(nsec + 1):
        t = -1.0 + 2.0 * i / nsec
        x = t * L / 2
        taper = max(0.0, 1.0 - abs(t) ** 3)
        w = Wd * (0.28 + 0.72 * math.sqrt(taper)) * (1.0 + 0.10 * rng.uniform(-1, 1))
        h = H * taper * (1.0 + 0.2 * rng.uniform(-1, 1))
        yo = 0.0006 * rng.uniform(-1, 1)
        rows.append([(x, yo - w / 2, 0.0), (x, yo, h), (x, yo + w / 2, 0.0)])
    ids = [[m.vert(p) for p in row] for row in rows]
    for i in range(nsec):
        for j in range(2):
            a, b = ids[i][j], ids[i][j + 1]
            c, d = ids[i + 1][j + 1], ids[i + 1][j]
            uv = [(rows[i][j][0], rows[i][j][1]), (rows[i][j + 1][0], rows[i][j + 1][1]),
                  (rows[i + 1][j + 1][0], rows[i + 1][j + 1][1]), (rows[i + 1][j][0], rows[i + 1][j][1])]
            # orientation: x forward, y left->right  => need CCW from above: a(i,j) b(i,j+1) c(i+1,j+1) d(i+1,j)
            m.face((a, d, c, b), (uv[0], uv[3], uv[2], uv[1]), iid, "main", True)
    return m


def mirror_set(lod, rng):
    m = Mesh()
    mm_ = Mesh()
    poly = [(-MIR_W / 2, -MIR_H / 2), (MIR_W / 2, -MIR_H / 2), (MIR_W / 2, MIR_H / 2), (-MIR_W / 2, MIR_H / 2)]
    extrude(mm_, poly, 0.0, MIR_T, top=R("mirror_top", "mirror", "mirror", lines=LINES, pile=PILE),
            band=R("mirror_edge", "mirror_edge", "mirror"), ct=0.0012 if lod == 0 else 0.0)
    if lod < 2:
        nsec = 6 if lod == 0 else 3
        for k, (cx, cy, L, Wd, ang) in enumerate(LINES):
            iid = R("line%d" % k, "powder", "powder", seed=600 + k)
            pl = powder_line(iid, L, Wd, 0.0016, nsec, rng)
            mm_.extend(pl, chain(rot_z(math.radians(ang)), trans(cx, cy, MIR_T)))
        # pile
        pm = Mesh()
        if lod == 0:
            prof = [(0.0170, 0.0), (0.0122, 0.0030), (0.0072, 0.0063), (0.0028, 0.0082), (0.0, 0.0087)]
            nseg = 12
        else:
            prof = [(0.0165, 0.0), (0.0085, 0.0050), (0.0, 0.0085)]
            nseg = 7
        prng = random.Random(7)
        jit = {}
        segang = 2 * math.pi / nseg
        lobes = [prng.uniform(0, 6.28) for _ in range(3)]

        def mod(theta, k, r, z):
            j = int(round(theta / segang)) % nseg
            key = (k, j)
            if key not in jit:
                jit[key] = (1.0 + 0.10 * prng.uniform(-1, 1), 1.0 + 0.12 * prng.uniform(-1, 1))
            jr, jz = jit[key]
            lobe = 1.0 + 0.16 * math.sin(2 * theta + lobes[0]) + 0.08 * math.sin(3 * theta + lobes[1])
            return (r * jr * lobe, z * jz)
        pid = R("pile", "powder", "powder", seed=650)
        lathe(pm, prof, nseg, [pid] * (len(prof) - 1), uvkind=["up"] * (len(prof) - 1), mod=mod)
        mm_.extend(pm, trans(PILE[0], PILE[1], MIR_T))
    if lod == 0:
        # card
        cm = Mesh()
        extrude(cm, rounded_rect(0.0856, 0.054, 0.0032, seg=2), 0.0, 0.0008, top=R("card", "card", "card"),
                band=R("card_edge", "card_edge", "card"))
        mm_.extend(cm, place(CARD_POS[0], CARD_POS[1], MIR_T, CARD_POS[2]))
        # utility razor blade
        rz = Mesh()
        blade = mm([(-30.5, -9.5), (30.5, -9.5), (19, 9.5), (10, 9.5), (8, 6.8), (6, 9.5), (-6, 9.5), (-8, 6.8),
                    (-10, 9.5), (-19, 9.5)])
        extrude(rz, blade, 0.0, 0.0006, top=R("razor", "razor", "razor"), band=R("razor_edge", "steel_edge", "razor"))
        mm_.extend(rz, place(RAZOR_POS[0], RAZOR_POS[1], MIR_T, RAZOR_POS[2]))
        # rolled note
        rm = Mesh()
        rr = 0.0045
        prof = [(0.0, 0.0), (rr, 0.0), (rr, BILL_H), (0.0, BILL_H)]
        lathe(rm, prof, 8, [R("roll_e0", "rolled_end", "rolled"), R("roll", "rolled_bill", "rolled"),
                            R("roll_e1", "rolled_end", "rolled")])
        mm_.extend(rm, chain(rot_y(math.pi / 2), trans(-BILL_H / 2, 0, rr + MIR_T),
                             rot_z(0.0), place(ROLL_POS[0], ROLL_POS[1], 0.0, ROLL_POS[2])))
    elif lod == 1:
        cm = Mesh()
        extrude(cm, rounded_rect(0.0856, 0.054, 0.0, seg=0), 0.0, 0.0008, top=R("card", "card", "card"))
        mm_.extend(cm, place(CARD_POS[0], CARD_POS[1], MIR_T, CARD_POS[2]))
    m.extend(mm_, place(MIR_POS[0], MIR_POS[1], TOP, MIR_POS[2]))
    return m


# =============================================================================
# ASHTRAY + CIGARETTES
# =============================================================================
ASH_POS = (0.270, 0.270)


def butt_mesh(iid_side, iid_f, iid_b, L, Lf, nseg, bend=0.0, squash=1.0, burning=False, lod=0, key=""):
    m = Mesh()
    r = 0.0039
    if burning:
        # filter / paper / ember ring / ash
        prof = [(0.0, 0.0), (r, 0.0), (r, Lf), (r, L - 0.007), (r * 0.97, L - 0.006), (r * 0.93, L), (0.0, L)]
        ids = [iid_f, iid_side, iid_side, R("cig_ember", "ember", "cig"), R("cig_ash", "ash", "cig"), iid_b]
        mats = ["main", "main", "main", "emissive", "main", "main"]
        kinds = ["down", "side", "side", "side", "side", "up"]
        if lod > 0:
            prof = [(0.0, 0.0), (r, 0.0), (r, L - 0.007), (r * 0.97, L - 0.006), (r * 0.93, L), (0.0, L)]
            ids = [iid_f, iid_side, "cig_ember", "cig_ash", iid_b]
            mats = ["main", "main", "emissive", "main", "main"]
            kinds = ["down", "side", "side", "side", "up"]
        lathe(m, prof, nseg, ids, uvkind=kinds, mats=mats)
    else:
        prof = [(0.0, 0.0), (r, 0.0), (r, Lf), (r, L - 0.0015), (r * 0.85, L), (0.0, L)]
        lathe(m, prof, nseg, [iid_f, iid_side, iid_side, iid_side, iid_b],
              uvkind=["down", "side", "side", "side", "up"])
    if bend or squash != 1.0:
        zb = Lf + (L - Lf) * 0.25
        cb, sb = math.cos(bend), math.sin(bend)

        def deform(p):
            x, y, z = p
            if z > Lf * 0.9:
                x *= squash
                y *= 1.0 + (1.0 / squash - 1.0) * 0.5
            if bend and z > zb:
                dz = z - zb
                # rotate around x-axis at z=zb
                y, z = y * cb - dz * sb, zb + y * sb + dz * cb
            return (x, y, z)
        m = m.transformed(deform)
    return m


def ashtray(lod, rng):
    m = Mesh()
    if lod == 0:
        prof = [(0.0470, 0.0), (0.0540, 0.0050), (0.0540, 0.0250), (0.0500, 0.0292), (0.0412, 0.0292),
                (0.0372, 0.0240), (0.0342, 0.0068), (0.0, 0.0068)]
        nseg = 18
    elif lod == 1:
        prof = [(0.0500, 0.0), (0.0540, 0.0250), (0.0412, 0.0292), (0.0342, 0.0068), (0.0, 0.0068)]
        nseg = 12
    else:
        prof = [(0.0540, 0.0), (0.0540, 0.0280), (0.0360, 0.0280), (0.0, 0.0068)]
        nseg = 8
    out = R("ash_out", "ash_out", "ash")
    rim = R("ash_rim", "ash_rim", "ash")
    inn = R("ash_in", "ash_in", "ash")
    bowl = R("ash_bowl", "ash_bowl", "ash")
    isl, kinds = [], []
    for k in range(len(prof) - 1):
        (r0, z0), (r1, z1) = prof[k], prof[k + 1]
        if z1 > z0 + 1e-6 and r1 >= r0 - 0.005:
            isl.append(out); kinds.append("side")
        elif abs(z1 - z0) < 0.006 and r1 < r0 and z0 > 0.02:
            isl.append(rim); kinds.append("up")
        elif r1 > 0.0:
            isl.append(inn); kinds.append("side")
        else:
            isl.append(bowl); kinds.append("up")
    notch_ang = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]

    def mod(theta, k, r, z):
        if lod < 2 and z > 0.022:
            d = min(abs(math.atan2(math.sin(theta - a), math.cos(theta - a))) for a in notch_ang)
            w = 2 * math.pi / nseg * 1.05
            if d < w:
                z -= 0.0062 * (1.0 - d / w) ** 1.0
        return (r, z)
    lathe(m, prof, nseg, isl, uvkind=kinds, mod=mod)
    # contents
    if lod < 2:
        nb = 4 if lod == 0 else 2
        spots = [(0.012, 0.010, 30, 0.6, 0.75), (-0.014, 0.006, 115, 0.0, 1.0), (0.000, -0.016, -40, 0.9, 0.7),
                 (-0.010, 0.020, 200, 0.3, 0.85)]
        for k in range(nb):
            x, y, yaw, bend, sq = spots[k]
            L = [0.030, 0.034, 0.027, 0.031][k]
            bm = butt_mesh(R("butt%d" % k, "butt", "butt", seed=700 + k, L=L, Lf=0.021),
                           R("butt%d_f" % k, "butt_filter_end", "butt"), R("butt%d_b" % k, "butt_burnt_end", "butt"),
                           L, 0.021, 6 if lod == 0 else 4, bend=bend, squash=sq)
            m.extend(bm, chain(trans(0, 0, -L * 0.45), rot_y(math.pi / 2), trans(0, 0, 0.0039 * sq),
                               place(x, y, 0.0068, yaw)))
        # burning cigarette resting in the notch at theta=0 (+x)
        L = 0.066
        cm = butt_mesh(R("cig", "cig", "cig", L=L, Lf=0.021), R("cig_f", "butt_filter_end", "cig"),
                       R("cig_tip", "ash_end", "cig"), L, 0.021, 8 if lod == 0 else 5, burning=True, lod=lod)
        tilt = math.radians(21.0)
        # axis along +x, tilted up towards the notch; filter end inside the bowl
        m.extend(cm, chain(rot_y(math.pi / 2 - tilt), trans(0.0035, 0.0, 0.0130)))
    m2 = Mesh()
    m2.extend(m, place(ASH_POS[0], ASH_POS[1], TOP, 0.0))
    # butts dropped on the table
    if lod == 0:
        for k, (x, y, yaw, bend, sq) in enumerate([(0.175, 0.205, 20, 0.7, 0.75), (0.360, 0.160, -75, 0.0, 1.0)]):
            kk = 4 + k
            L = [0.029, 0.033][k]
            bm = butt_mesh(R("butt%d" % kk, "butt", "butt", seed=700 + kk, L=L, Lf=0.021),
                           R("butt%d_f" % kk, "butt_filter_end", "butt"), R("butt%d_b" % kk, "butt_burnt_end", "butt"),
                           L, 0.021, 6, bend=bend, squash=sq)
            m2.extend(bm, chain(trans(0, 0, -L * 0.45), rot_y(math.pi / 2), trans(0, 0, 0.0039 * sq),
                                place(x, y, TOP, yaw)))
    return m2


# =============================================================================
# CAN, GLASS, PHONE
# =============================================================================
CAN_POS = (-0.245, 0.315, 30.0)
GLASS_POS = (0.585, 0.300)
PHONE_POS = (0.215, -0.300, 14.0)


def can(lod):
    m = Mesh()
    # 0.5 l can (168 mm tall, 66 mm diameter)
    if lod == 0:
        prof = [(0.0262, 0.0), (0.0331, 0.0072), (0.0331, 0.1520), (0.0300, 0.1608), (0.0272, 0.1660),
                (0.0272, 0.1679), (0.0251, 0.1679), (0.0246, 0.1629), (0.0, 0.1629)]
        nseg = 18
    elif lod == 1:
        prof = [(0.0300, 0.0), (0.0331, 0.0072), (0.0331, 0.1520), (0.0272, 0.1660), (0.0272, 0.1679),
                (0.0, 0.1629)]
        nseg = 10
    else:
        prof = [(0.0331, 0.0), (0.0331, 0.1620), (0.0, 0.1635)]
        nseg = 6
    side = R("can_side", "can_side", "can")
    top = R("can_top", "can_top", "can")
    isl, kinds = [], []
    for k in range(len(prof) - 1):
        (r0, z0), (r1, z1) = prof[k], prof[k + 1]
        if z1 > z0 + 1e-6 and z0 < 0.1665:
            isl.append(side); kinds.append("side")
        else:
            isl.append(top); kinds.append("up")
    lathe(m, prof, nseg, isl, uvkind=kinds)
    return m.transformed(place(CAN_POS[0], CAN_POS[1], TOP, CAN_POS[2]))


def glass(lod):
    m = Mesh()
    if lod < 2:
        prof = [(0.0355, 0.0), (0.0405, 0.0900), (0.0375, 0.0900), (0.0345, 0.0150), (0.0322, 0.0125), (0.0, 0.0125)]
        nseg = 16 if lod == 0 else 8
        isl = [R("glass_out", "glass", "glass", part="out"), R("glass_rim", "glass", "glass", part="rim"),
               R("glass_in", "glass", "glass", part="in"), "glass_in", R("glass_base", "glass", "glass", part="base")]
        lathe(m, prof, nseg, isl, uvkind=["side", "up", "side", "side", "up"], mat="glass")
        # whisky
        wp = [(0.0343, 0.0125), (0.0347, 0.0300), (0.0, 0.0300)]
        lathe(m, wp, nseg, [R("whisky_side", "whisky", "whisky"), R("whisky_top", "whisky_top", "whisky")],
              uvkind=["side", "up"])
    else:
        rref = 0.5 * (0.0380 + 0.0405)
        lathe(m, [(0.0380, 0.0), (0.0405, 0.0900)], 6,
              [REG.alias("glass_out_l2", "glass_out", (0, 0, 2 * math.pi * rref, math.hypot(0.0025, 0.09)), None)],
              uvkind=["side"], mat="glass")
        lathe(m, [(0.0372, 0.0300), (0.0, 0.0300)], 6,
              [REG.alias("whisky_top_l2", "whisky_top", (-0.0372, -0.0372, 0.0372, 0.0372), None)], uvkind=["up"])
    return m.transformed(trans(GLASS_POS[0], GLASS_POS[1], TOP))


BOTTLE_POS = (0.440, 0.335, 25.0)


def bottle(lod):
    """70 cl whisky bottle: glass shell (alpha), paper label, liquid and screw cap."""
    m = Mesh()
    nseg = 16 if lod == 0 else (8 if lod == 1 else 6)
    if lod < 2:
        prof = [(0.0350, 0.0), (0.0380, 0.0035), (0.0380, 0.1650), (0.0345, 0.1860), (0.0215, 0.2080),
                (0.0150, 0.2240), (0.0142, 0.2560)]
        if lod == 1:
            prof = [(0.0360, 0.0), (0.0380, 0.1650), (0.0215, 0.2080), (0.0142, 0.2560)]
        gid = R("bottle_glass", "glass", "glass", part="bottle")
        lathe(m, prof, nseg, [gid] * (len(prof) - 1), uvkind=["side"] * (len(prof) - 1), mat="glass")
        # paper label (main material), 0.4 mm proud of the glass
        lp = [(0.0384, 0.058), (0.0384, 0.136)]
        lathe(m, lp, nseg, [R("bottle_label", "bottle_label", "glass")], uvkind=["side"])
        # liquid (opaque amber) seen through the glass
        wp = [(0.0372, 0.0060), (0.0372, 0.0930), (0.0, 0.0930)]
        lathe(m, wp, nseg, [R("bottle_liq", "whisky", "whisky"), R("bottle_liq_top", "whisky_top", "whisky")],
              uvkind=["side", "up"])
        # screw cap
        cp = [(0.0158, 0.2500), (0.0158, 0.2835), (0.0147, 0.2850), (0.0, 0.2850)]
        if lod == 1:
            cp = [(0.0158, 0.2500), (0.0158, 0.2850), (0.0, 0.2850)]
        lathe(m, cp, nseg, [R("bottle_cap", "bottle_cap", "whisky")] * (len(cp) - 2) +
              [R("bottle_cap_top", "bottle_cap_top", "whisky")], uvkind=["side"] * (len(cp) - 2) + ["up"])
    else:
        prof = [(0.0380, 0.0), (0.0380, 0.1700), (0.0150, 0.2240), (0.0150, 0.2850), (0.0, 0.2850)]
        rr = (0.0380 + 0.0265 + 0.0150) / 3.0
        body = REG.alias("bottle_l2_body", "bottle_label", (0, 0, 2 * math.pi * rr, 0.30), None)
        capt = REG.alias("bottle_l2_cap", "bottle_cap_top", (-0.015, -0.015, 0.015, 0.015), None)
        lathe(m, prof, nseg, [body, body, body, capt], uvkind=["side", "side", "side", "up"], mat="main")
    return m.transformed(place(BOTTLE_POS[0], BOTTLE_POS[1], TOP, BOTTLE_POS[2]))


def phone(lod):
    m = Mesh()
    w, h, t = 0.0715, 0.1465, 0.0078
    if lod < 2:
        seg = 3 if lod == 0 else 1
        poly = rounded_rect(w, h, 0.0090, seg=seg)
        ch = lod == 0
        ex = extrude(m, poly, 0.0, t, top=None, band=R("phone_band", "phone_band", "phone"),
                     ct=0.0012 if ch else 0.0, cb=0.0010 if ch else 0.0)
        rect = (-w / 2 + 0.0032, -h / 2 + 0.0055, w / 2 - 0.0032, h / 2 - 0.0055)
        bezel_cap(m, ex["rings"][-1], ex["top_pts"], t, rect, 0.0, bezel=R("phone_bezel", "phone_bezel", "phone"),
                  screen=R("phone_screen", "phone_screen", "phone"))
    else:
        poly = rounded_rect(w, h, 0.0)
        iid = REG.alias("phone_scr_l2", "phone_screen", (-w / 2, -h / 2, w / 2, h / 2), None)
        extrude(m, poly, 0.0, t, top=iid, top_mat="emissive",
                band=REG.alias("phone_band_l2", "phone_band", (0, 0, 2 * (w + h), t), None))
    return m.transformed(place(PHONE_POS[0], PHONE_POS[1], TOP, PHONE_POS[2]))


# =============================================================================
# assembly
# =============================================================================
GUN_POS = (0.405, -0.070, 31.0)
MAG_POS = (0.585, 0.095, -62.0)
ROUNDS = [(0.545, -0.225, 20.0), (0.585, -0.180, 105.0), (0.505, -0.265, -35.0)]
LAPTOP_POS = (0.020, 0.075, -5.0)


def build(lod):
    import ct_geo
    ct_geo.RECORD[0] = (lod == 0)
    rng = random.Random(1234)
    m = Mesh()
    m.extend(table(lod))
    m.extend(laptop(lod), place(LAPTOP_POS[0], LAPTOP_POS[1], TOP, LAPTOP_POS[2]))
    # pistol lying on its right side; lowest point = grip side (15 mm half width)
    m.extend(pistol(lod), place(GUN_POS[0], GUN_POS[1], TOP + 0.0150, GUN_POS[2]))
    if lod < 2:
        m.extend(magazine(lod), place(MAG_POS[0], MAG_POS[1], TOP + 0.0110, MAG_POS[2]))
    if lod == 0:
        for k, (x, y, yaw) in enumerate(ROUNDS):
            c = cartridge(k, lod)
            m.extend(c, chain(rot_y(math.pi / 2), trans(-0.015, 0, 0.00495), place(x, y, TOP, yaw)))
    m.extend(money(lod, rng))
    m.extend(mirror_set(lod, rng))
    m.extend(ashtray(lod, rng))
    m.extend(can(lod))
    m.extend(glass(lod))
    m.extend(bottle(lod))
    m.extend(phone(lod))
    return m


def collision():
    """Closed convex boxes (12 tris each)."""
    m = Mesh()

    def box(cx, cy, z0, sx, sy, sz, yaw=0.0, pitch_axis=None):
        pts = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]
        bm = Mesh()
        extrude(bm, pts, 0.0, sz, top="__col", bot="__col", band="__col")
        return bm.transformed(place(cx, cy, z0, yaw))

    REG.add("__col", "col", 0.0)
    m.extend(box(0, 0, TOP - TT, TW, TD, TT))
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        m.extend(box(sx * LEGX, sy * LEGY, 0.0, LEG, LEG, TOP - TT))
    # laptop base
    m.extend(box(LAPTOP_POS[0], LAPTOP_POS[1], TOP, LAP_W, LAP_D, LAP_H, LAPTOP_POS[2]))
    # laptop lid (oriented box)
    lid = Mesh()
    pts = [(-LAP_W / 2, -LID_L / 2), (LAP_W / 2, -LID_L / 2), (LAP_W / 2, LID_L / 2), (-LAP_W / 2, LID_L / 2)]
    extrude(lid, pts, -LID_T, 0.0, top="__col", bot="__col", band="__col")
    ca, sa = math.cos(LID_TILT), math.sin(LID_TILT)
    hy, hz = LAP_D / 2 - 0.0045, LAP_H + 0.0012

    def lid_xf(p):
        x, y, z = p
        y = y + LID_L / 2
        return (x, hy + y * sa - z * ca, hz + y * ca + z * sa)
    m.extend(lid.transformed(chain(lid_xf, place(LAPTOP_POS[0], LAPTOP_POS[1], TOP, LAPTOP_POS[2]))))
    # money stack
    m.extend(box(-0.470, 0.125, TOP, 2 * BILL_W + 0.01, 3 * BILL_H + 0.01, 3 * BUNDLE_T, 14.0))
    return m
