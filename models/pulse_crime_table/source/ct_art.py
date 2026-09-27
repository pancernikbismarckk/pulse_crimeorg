"""2D artwork drawn with PIL (stylised, non-realistic designs; no real brands).
All images are returned as float32 numpy arrays in [0,1] (sRGB)."""
import math
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_FONT_DIRS = ["/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/dejavu",
              "/usr/share/fonts/truetype/freefont"]
try:
    import matplotlib
    _FONT_DIRS.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf"))
except Exception:
    pass

_ALIASES = {
    "serif-bold": ["LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf", "FreeSerifBold.ttf"],
    "serif": ["LiberationSerif-Regular.ttf", "DejaVuSerif.ttf", "FreeSerif.ttf"],
    "sans-bold": ["LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf", "FreeSansBold.ttf"],
    "sans": ["LiberationSans-Regular.ttf", "DejaVuSans.ttf", "FreeSans.ttf"],
    "mono": ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf", "FreeMono.ttf"],
    "mono-bold": ["DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf", "FreeMonoBold.ttf"],
    "condensed": ["DejaVuSansCondensed-Bold.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
}
_cache = {}


def font(name, size):
    key = (name, int(size))
    if key in _cache:
        return _cache[key]
    for fn in _ALIASES.get(name, [name]):
        for d in _FONT_DIRS:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                f = ImageFont.truetype(p, int(size))
                _cache[key] = f
                return f
    f = ImageFont.load_default(size=int(size))
    _cache[key] = f
    return f


def to_np(img):
    return np.asarray(img.convert("RGBA") if img.mode in ("RGBA", "LA") else img.convert("RGB")).astype(np.float32) / 255.0


def text_center(d, xy, txt, fnt, fill, anchor="mm"):
    d.text(xy, txt, font=fnt, fill=fill, anchor=anchor)


# ---------------------------------------------------------------------------
# USD 100 (stylised)
# ---------------------------------------------------------------------------
BILL_PX = (1560, 663)          # 10 px / mm


def _guilloche(d, W, H, col, rng, n=34, amp=18, width=1):
    for k in range(n):
        y0 = H * (k + 0.5) / n
        ph = rng.uniform(0, 6.28)
        fr = rng.uniform(0.010, 0.016)
        pts = [(x, y0 + amp * math.sin(x * fr + ph) * math.sin(x * 0.0023 + k)) for x in range(0, W + 8, 8)]
        d.line(pts, fill=col, width=width)


def _rosette(d, cx, cy, r, col, n=48, width=1):
    for k in range(n):
        a = 2 * math.pi * k / n
        pts = []
        for t in range(0, 361, 8):
            tt = math.radians(t)
            rr = r * (0.55 + 0.45 * math.cos(6 * tt))
            pts.append((cx + rr * math.cos(tt + a), cy + rr * math.sin(tt + a)))
        d.line(pts, fill=col, width=width)


def _portrait(img, cx, cy, w, h, ink):
    """Engraving-like generic bust (not a real person)."""
    W, H = img.size
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    # shoulders / coat
    md.ellipse([cx - w * 0.62, cy + h * 0.18, cx + w * 0.62, cy + h * 1.05], fill=255)
    # neck
    md.rectangle([cx - w * 0.12, cy + h * 0.02, cx + w * 0.12, cy + h * 0.28], fill=200)
    # head
    md.ellipse([cx - w * 0.23, cy - h * 0.34, cx + w * 0.23, cy + h * 0.16], fill=170)
    # long hair both sides
    md.ellipse([cx - w * 0.36, cy - h * 0.28, cx - w * 0.10, cy + h * 0.30], fill=235)
    md.ellipse([cx + w * 0.10, cy - h * 0.28, cx + w * 0.36, cy + h * 0.30], fill=235)
    md.ellipse([cx - w * 0.25, cy - h * 0.42, cx + w * 0.25, cy - h * 0.12], fill=225)
    # face (lighter)
    md.ellipse([cx - w * 0.17, cy - h * 0.25, cx + w * 0.17, cy + h * 0.13], fill=110)
    # hatch the mask into engraving lines
    hatch = Image.new("L", img.size, 0)
    hd = ImageDraw.Draw(hatch)
    for y in range(int(cy - h * 0.5), int(cy + h * 1.1), 4):
        hd.line([(cx - w, y), (cx + w, y + 6)], fill=255, width=2)
    m = np.asarray(mask).astype(np.float32) / 255.0
    hh = np.asarray(hatch).astype(np.float32) / 255.0
    ink_a = np.clip(m * (0.35 + 0.65 * hh) * 1.2, 0, 1)
    # facial features
    fd = Image.new("L", img.size, 0)
    fdd = ImageDraw.Draw(fd)
    fdd.arc([cx - w * 0.11, cy - h * 0.12, cx - w * 0.03, cy - h * 0.07], 200, 340, fill=255, width=3)
    fdd.arc([cx + w * 0.03, cy - h * 0.12, cx + w * 0.11, cy - h * 0.07], 200, 340, fill=255, width=3)
    fdd.line([(cx, cy - h * 0.08), (cx - w * 0.02, cy + h * 0.00)], fill=255, width=3)
    fdd.line([(cx - w * 0.05, cy + h * 0.05), (cx + w * 0.05, cy + h * 0.05)], fill=255, width=3)
    ink_a = np.maximum(ink_a, np.asarray(fd).astype(np.float32) / 255.0)
    base = np.asarray(img).astype(np.float32)
    inkc = np.array(ink, np.float32)
    base = base * (1 - ink_a[..., None] * 0.85) + inkc * ink_a[..., None] * 0.85
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def bill_front():
    W, H = BILL_PX
    rng = random.Random(100)
    img = Image.new("RGB", (W, H), (196, 204, 182))
    d = ImageDraw.Draw(img)
    _guilloche(d, W, H, (172, 186, 164), rng, n=40, amp=14)
    # large faint 100 in background
    d.text((W * 0.30, H * 0.52), "100", font=font("sans-bold", 330), fill=(182, 193, 170), anchor="mm")
    # border
    d.rectangle([12, 12, W - 13, H - 13], outline=(60, 78, 66), width=6)
    d.rectangle([34, 34, W - 35, H - 35], outline=(88, 104, 92), width=2)
    ink = (38, 52, 44)
    # portrait oval frame (left-centre)
    cx, cy = int(W * 0.42), int(H * 0.50)
    ow, oh = 300, 400
    d.ellipse([cx - ow // 2 - 16, cy - oh // 2 - 16, cx + ow // 2 + 16, cy + oh // 2 + 16], outline=ink, width=7)
    d.ellipse([cx - ow // 2, cy - oh // 2, cx + ow // 2, cy + oh // 2], fill=(222, 225, 206), outline=(90, 104, 92), width=2)
    img = _portrait(img, cx, cy - 20, ow * 0.95, oh * 0.62, ink)
    d = ImageDraw.Draw(img)
    # blue security ribbon
    rx = int(W * 0.63)
    d.rectangle([rx, 0, rx + 58, H], fill=(70, 118, 190))
    for y in range(8, H, 44):
        d.text((rx + 29, y + 16), "100", font=font("sans-bold", 20), fill=(170, 200, 240), anchor="mm")
    # inkwell with bell (copper)
    ix, iy = int(W * 0.745), int(H * 0.60)
    d.rounded_rectangle([ix - 70, iy - 60, ix + 70, iy + 80], 18, fill=(164, 104, 58))
    d.polygon([(ix - 34, iy + 40), (ix + 34, iy + 40), (ix + 22, iy - 20), (ix, iy - 34), (ix - 22, iy - 20)],
              fill=(196, 132, 70))
    # seals
    _rosette(d, int(W * 0.20), int(H * 0.50), 70, (40, 50, 46), n=36, width=2)
    d.ellipse([W * 0.20 - 46, H * 0.50 - 46, W * 0.20 + 46, H * 0.50 + 46], outline=(30, 40, 36), width=5)
    d.text((W * 0.20, H * 0.50), "B", font=font("serif-bold", 60), fill=(30, 40, 36), anchor="mm")
    _rosette(d, int(W * 0.60), int(H * 0.68), 44, (56, 120, 76), n=28, width=2)
    # texts
    d.text((W * 0.42, 66), "THE UNITED STATES OF AMERICA", font=font("serif-bold", 50), fill=ink, anchor="mm")
    d.text((W * 0.42, H - 110), "FEDERAL RESERVE NOTE", font=font("serif-bold", 30), fill=ink, anchor="mm")
    d.text((W * 0.42, H - 66), "ONE HUNDRED DOLLARS", font=font("serif-bold", 44), fill=ink, anchor="mm")
    # numerals
    d.text((120, 96), "100", font=font("serif-bold", 76), fill=ink, anchor="mm")
    d.text((120, H - 96), "100", font=font("serif-bold", 76), fill=ink, anchor="mm")
    d.text((W - 150, 100), "100", font=font("serif-bold", 76), fill=ink, anchor="mm")
    d.text((W - 190, H - 150), "100", font=font("sans-bold", 190), fill=(176, 122, 58), anchor="mm")
    # serials (green)
    d.text((W * 0.20, 150), "LB 40736921 D", font=font("mono-bold", 34), fill=(46, 120, 70), anchor="mm")
    d.text((W * 0.83, H * 0.30), "LB 40736921 D", font=font("mono-bold", 34), fill=(46, 120, 70), anchor="mm")
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return to_np(img)


def bill_back():
    W, H = BILL_PX
    rng = random.Random(200)
    img = Image.new("RGB", (W, H), (186, 202, 176))
    d = ImageDraw.Draw(img)
    _guilloche(d, W, H, (160, 186, 156), rng, n=44, amp=16)
    g = (40, 96, 62)
    gd = (26, 66, 44)
    d.rectangle([12, 12, W - 13, H - 13], outline=g, width=8)
    d.rectangle([36, 36, W - 37, H - 37], outline=(78, 126, 92), width=2)
    # building (stylised hall with tower)
    bx, by = int(W * 0.46), int(H * 0.60)
    d.rectangle([bx - 360, by - 60, bx + 360, by + 140], fill=(168, 196, 164), outline=gd, width=4)
    for k in range(-340, 341, 40):
        d.rectangle([bx + k - 10, by - 30, bx + k + 10, by + 20], fill=gd)
        d.rectangle([bx + k - 10, by + 50, bx + k + 10, by + 110], fill=gd)
    d.polygon([(bx - 380, by - 60), (bx + 380, by - 60), (bx + 330, by - 100), (bx - 330, by - 100)], fill=(120, 160, 126),
              outline=gd)
    d.rectangle([bx - 70, by - 230, bx + 70, by - 60], fill=(150, 184, 150), outline=gd, width=4)
    d.ellipse([bx - 36, by - 200, bx + 36, by - 128], fill=(214, 226, 206), outline=gd, width=4)
    d.polygon([(bx - 60, by - 230), (bx + 60, by - 230), (bx + 30, by - 300), (bx, by - 340), (bx - 30, by - 300)],
              fill=(120, 160, 126), outline=gd)
    d.line([(bx, by - 340), (bx, by - 380)], fill=gd, width=5)
    for k in range(-500, 501, 4):
        d.line([(bx + k, by + 140), (bx + k + 30, by + 190)], fill=(150, 180, 150), width=1)
    d.text((bx, 64), "THE UNITED STATES OF AMERICA", font=font("serif-bold", 50), fill=gd, anchor="mm")
    d.text((bx, 124), "IN GOD WE TRUST", font=font("serif-bold", 30), fill=gd, anchor="mm")
    d.text((bx, H - 64), "ONE HUNDRED DOLLARS", font=font("serif-bold", 46), fill=gd, anchor="mm")
    d.text((W - 190, H * 0.46), "100", font=font("sans-bold", 230), fill=g, anchor="mm")
    d.text((130, 100), "100", font=font("serif-bold", 70), fill=gd, anchor="mm")
    d.text((130, H - 100), "100", font=font("serif-bold", 70), fill=gd, anchor="mm")
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return to_np(img)


def strap(text="$10000"):
    """Currency strap face, 32 x 66.3 mm (10 px/mm), band runs vertically."""
    W, H = 320, 663
    img = Image.new("RGB", (W, H), (212, 164, 58))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 0, 22, H], fill=(150, 110, 30))
    d.rectangle([W - 23, 0, W - 11, H], fill=(150, 110, 30))
    t = Image.new("RGBA", (H, W), (0, 0, 0, 0))
    td = ImageDraw.Draw(t)
    td.text((H / 2, W / 2 - 10), text, font=font("sans-bold", 118), fill=(40, 28, 10, 255), anchor="mm")
    td.text((H / 2, W / 2 + 92), "HUNDREDS", font=font("sans-bold", 40), fill=(40, 28, 10, 255), anchor="mm")
    t = t.rotate(90, expand=True)
    img.paste(t, (0, 0), t)
    return to_np(img)


# ---------------------------------------------------------------------------
# laptop
# ---------------------------------------------------------------------------
SCREEN_PX = (1600, 900)


def laptop_screen():
    W, H = SCREEN_PX
    img = Image.new("RGB", (W, H), (8, 11, 16))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 40):
        d.line([(x, 0), (x, H)], fill=(12, 17, 24))
    for y in range(0, H, 40):
        d.line([(0, y), (W, y)], fill=(12, 17, 24))
    # top bar
    d.rectangle([0, 0, W, 44], fill=(22, 27, 36))
    d.text((24, 22), "GHOST//OPS", font=font("mono-bold", 26), fill=(120, 230, 170), anchor="lm")
    d.text((W / 2, 22), "SECURE CHANNEL  -  AES-256  -  TOR", font=font("mono", 22), fill=(120, 140, 160), anchor="mm")
    d.ellipse([W - 190, 12, W - 170, 32], fill=(230, 50, 50))
    d.text((W - 160, 22), "LIVE 02:47", font=font("mono-bold", 22), fill=(230, 230, 230), anchor="lm")
    # terminal window (left)
    tx0, ty0, tx1, ty1 = 24, 66, 900, H - 24
    d.rectangle([tx0, ty0, tx1, ty1], fill=(4, 7, 6), outline=(40, 70, 52), width=2)
    d.rectangle([tx0, ty0, tx1, ty0 + 34], fill=(24, 40, 30))
    d.text((tx0 + 14, ty0 + 17), "root@ghost: ~", font=font("mono-bold", 20), fill=(150, 230, 180), anchor="lm")
    lines = [
        ("$ ssh -i ~/.keys/op.pem op@185.44.19.207", (140, 255, 170)),
        ("[+] tunnel up via 3 relays", (90, 220, 130)),
        ("[+] wallet sync .............. OK", (90, 220, 130)),
        ("$ send --amount 4.20 BTC --to bc1q9x...7kd3", (140, 255, 170)),
        ("[+] tx broadcast  9f2c81e0...e11a", (90, 220, 130)),
        ("[!] 1 unconfirmed / fee 38 sat/vB", (240, 200, 90)),
        ("$ ls /drop/2610", (140, 255, 170)),
        ("  manifest.enc   route.gpx   keys.kdbx", (170, 190, 180)),
        ("$ gpg -d manifest.enc | head", (140, 255, 170)),
        ("  PKG  12x  ..... DOCKS / BAY 4", (170, 190, 180)),
        ("  CASH  $ 60 000 ....... SAFE 2", (170, 190, 180)),
        ("$ shred -uz ~/.bash_history", (140, 255, 170)),
        ("[####################] 100%", (90, 220, 130)),
        ("$ _", (140, 255, 170)),
    ]
    y = ty0 + 62
    for (t, c) in lines:
        d.text((tx0 + 20, y), t, font=font("mono", 27), fill=c, anchor="lm")
        y += 51
    # chart window (right top)
    cx0, cy0, cx1, cy1 = 930, 66, W - 24, 440
    d.rectangle([cx0, cy0, cx1, cy1], fill=(10, 14, 20), outline=(60, 70, 90), width=2)
    d.text((cx0 + 16, cy0 + 22), "BTC / USD", font=font("mono-bold", 24), fill=(240, 170, 60), anchor="lm")
    d.text((cx1 - 16, cy0 + 22), "+6.8%", font=font("mono-bold", 24), fill=(90, 230, 120), anchor="rm")
    rng = random.Random(9)
    pts = []
    v = 0.35
    n = 60
    for k in range(n):
        v += rng.uniform(-0.05, 0.065)
        v = min(max(v, 0.08), 0.92)
        pts.append((cx0 + 20 + (cx1 - cx0 - 40) * k / (n - 1), cy1 - 24 - (cy1 - cy0 - 80) * v))
    for gy in range(cy0 + 60, cy1, 60):
        d.line([(cx0 + 10, gy), (cx1 - 10, gy)], fill=(24, 30, 40))
    d.polygon(pts + [(pts[-1][0], cy1 - 10), (pts[0][0], cy1 - 10)], fill=(40, 30, 12))
    d.line(pts, fill=(250, 160, 40), width=5)
    # map window (right bottom)
    mx0, my0, mx1, my1 = 930, 462, W - 24, H - 24
    d.rectangle([mx0, my0, mx1, my1], fill=(10, 16, 22), outline=(60, 70, 90), width=2)
    rng = random.Random(3)
    for k in range(26):
        x0 = rng.uniform(mx0, mx1); y0 = rng.uniform(my0, my1)
        ang = rng.choice([0.0, 1.5708]) + rng.uniform(-0.25, 0.25)
        L = rng.uniform(120, 420)
        d.line([(x0, y0), (x0 + L * math.cos(ang), y0 + L * math.sin(ang))], fill=(34, 52, 66), width=rng.choice([3, 5, 8]))
    d.polygon([(mx0 + 30, my1 - 10), (mx0 + 200, my1 - 90), (mx0 + 330, my1 - 10)], fill=(16, 34, 48))
    route = [(mx0 + 80, my1 - 60), (mx0 + 220, my1 - 120), (mx0 + 300, my0 + 200), (mx0 + 520, my0 + 150),
             (mx1 - 60, my0 + 70)]
    d.line(route, fill=(80, 200, 255), width=6)
    for (px, py), col in [(route[0], (80, 255, 140)), (route[-1], (255, 60, 60)), (route[2], (255, 200, 60))]:
        d.ellipse([px - 16, py - 16, px + 16, py + 16], fill=col)
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(10, 10, 10))
    d.text((mx0 + 16, my0 + 22), "TRACKER  -  3 UNITS", font=font("mono-bold", 22), fill=(120, 200, 255), anchor="lm")
    d.text((mx1 - 16, my1 - 20), "N 33.7439  W 118.2701", font=font("mono", 20), fill=(120, 140, 160), anchor="rm")
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    return to_np(img)


KB_PX_PER_MM = 8.0
KB_SIZE_MM = (284.0, 108.0)


def keyboard():
    """Returns (height_mask, legend_mask, shine) for the keyboard area."""
    W, H = int(KB_SIZE_MM[0] * KB_PX_PER_MM), int(KB_SIZE_MM[1] * KB_PX_PER_MM)
    hm = Image.new("L", (W, H), 0)
    lg = Image.new("L", (W, H), 0)
    sh = Image.new("L", (W, H), 0)
    dh, dl, ds = ImageDraw.Draw(hm), ImageDraw.Draw(lg), ImageDraw.Draw(sh)
    s = KB_PX_PER_MM
    pitch = 18.9
    gap = 3.1
    rows = [
        (["esc", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "del"], 0.62, None),
        (["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "back"], 1.0, {13: 1.9}),
        (["tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\"], 1.0, {0: 1.45, 13: 1.45}),
        (["caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "enter"], 1.0, {0: 1.75, 12: 2.1}),
        (["shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "shift"], 1.0, {0: 2.3, 11: 2.75}),
        (["ctrl", "fn", "win", "alt", "", "alt", "ctrl", "<", "v", ">"], 1.0, {4: 5.9}),
    ]
    shiny = set(["W", "A", "S", "D", "", "enter", "back", "E", "R", "C", "V", "ctrl"])
    y = 1.0
    for (keys, hscale, widths) in rows:
        kh = pitch * hscale - gap
        # total width for this row
        wts = [(widths or {}).get(i, 1.0) for i in range(len(keys))]
        tot = sum(wts)
        scale = (KB_SIZE_MM[0] - 2.0) / (tot * pitch)
        x = 1.0
        for i, k in enumerate(keys):
            kw = wts[i] * pitch * scale - gap
            box = [x * s, y * s, (x + kw) * s, (y + kh) * s]
            dh.rounded_rectangle(box, radius=1.4 * s, fill=255)
            if k in shiny:
                ds.rounded_rectangle([box[0] + 3, box[1] + 3, box[2] - 3, box[3] - 3], radius=1.2 * s, fill=255)
            if k:
                fs = 3.4 if len(k) == 1 else 2.2
                dl.text(((x + kw * 0.5) * s, (y + kh * 0.45) * s), k, font=font("sans", fs * s), fill=255, anchor="mm")
            x += kw + gap
        y += kh + gap
    return (np.asarray(hm).astype(np.float32) / 255.0, np.asarray(lg).astype(np.float32) / 255.0,
            np.asarray(sh).astype(np.float32) / 255.0)


def lid_stickers():
    """RGBA decals for the lid back, 330 x 226 mm at 4 px/mm (image up = screen up)."""
    s = 4
    W, H = 330 * s, 226 * s
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # skull sticker (circle)
    cx, cy, r = 80 * s, 150 * s, 26 * s
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(18, 18, 18, 255), outline=(235, 235, 235, 255), width=6)
    d.ellipse([cx - 14 * s, cy - 16 * s, cx + 14 * s, cy + 9 * s], fill=(235, 235, 235, 255))
    d.rectangle([cx - 8 * s, cy + 4 * s, cx + 8 * s, cy + 15 * s], fill=(235, 235, 235, 255))
    d.ellipse([cx - 9 * s, cy - 7 * s, cx - 2 * s, cy + 1 * s], fill=(18, 18, 18, 255))
    d.ellipse([cx + 2 * s, cy - 7 * s, cx + 9 * s, cy + 1 * s], fill=(18, 18, 18, 255))
    for k in (-5, -1.7, 1.7, 5):
        d.line([(cx + k * s, cy + 9 * s), (cx + k * s, cy + 15 * s)], fill=(18, 18, 18, 255), width=5)
    # hazard triangle
    tx, ty = 250 * s, 58 * s
    d.polygon([(tx, ty - 22 * s), (tx + 25 * s, ty + 20 * s), (tx - 25 * s, ty + 20 * s)], fill=(250, 200, 20, 255),
              outline=(20, 20, 20, 255))
    d.text((tx, ty + 6 * s), "!", font=font("sans-bold", 26 * s), fill=(20, 20, 20, 255), anchor="mm")
    # NO LOGS sticker
    d.rounded_rectangle([196 * s, 150 * s, 290 * s, 178 * s], 4 * s, fill=(210, 40, 40, 255))
    d.text((243 * s, 164 * s), "NO LOGS", font=font("sans-bold", 17 * s), fill=(255, 255, 255, 255), anchor="mm")
    # small barcode label
    d.rectangle([40 * s, 40 * s, 92 * s, 62 * s], fill=(235, 235, 228, 255))
    rng = random.Random(4)
    x = 43 * s
    while x < 89 * s:
        w = rng.choice([2, 3, 5, 7])
        d.rectangle([x, 43 * s, x + w, 55 * s], fill=(20, 20, 20, 255))
        x += w + rng.choice([3, 4, 6])
    d.text((66 * s, 59 * s), "S/N 7Q4-XK2", font=font("mono", 8 * s), fill=(30, 30, 30, 255), anchor="mm")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # image row 0 = bottom (we sample with y up)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return to_np(img)


def lid_logo_mask():
    s = 4
    W, H = 330 * s, 226 * s
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    cx, cy = W / 2, H / 2
    r = 17 * s
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=int(3.2 * s))
    d.polygon([(cx - 9 * s, cy + 11 * s), (cx + 11 * s, cy - 9 * s), (cx + 7 * s, cy - 11 * s), (cx - 11 * s, cy + 7 * s)],
              fill=255)
    return np.asarray(m).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# phone lock screen
# ---------------------------------------------------------------------------
PHONE_PX = (700, 1400)


def phone_screen():
    W, H = PHONE_PX
    img = Image.new("RGB", (W, H), (0, 0, 0))
    a = np.zeros((H, W, 3), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = yy / H
    a[..., 0] = 30 + 50 * np.exp(-((xx - W * 0.8) ** 2 + (yy - H * 0.25) ** 2) / (2 * 260 ** 2)) + 10 * t
    a[..., 1] = 18 + 20 * np.exp(-((xx - W * 0.2) ** 2 + (yy - H * 0.7) ** 2) / (2 * 300 ** 2))
    a[..., 2] = 60 + 90 * np.exp(-((xx - W * 0.2) ** 2 + (yy - H * 0.7) ** 2) / (2 * 300 ** 2)) + 40 * (1 - t)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.text((W / 2, 70), "02:47", font=font("sans", 36), fill=(230, 230, 240), anchor="mm")
    d.text((W / 2, 300), "02:47", font=font("sans-bold", 200), fill=(245, 245, 250), anchor="mm")
    d.text((W / 2, 440), "sobota, 27 września", font=font("sans", 44), fill=(220, 220, 235), anchor="mm")
    # notification card
    d.rounded_rectangle([40, 560, W - 40, 830], 36, fill=(40, 40, 52))
    d.rounded_rectangle([72, 596, 152, 676], 20, fill=(60, 200, 110))
    d.text((112, 636), "@", font=font("sans-bold", 54), fill=(255, 255, 255), anchor="mm")
    d.text((180, 616), "Nieznany", font=font("sans-bold", 42), fill=(255, 255, 255), anchor="lm")
    d.text((W - 70, 616), "teraz", font=font("sans", 32), fill=(170, 170, 185), anchor="rm")
    d.text((180, 680), "Towar gotowy. Doki,", font=font("sans", 38), fill=(225, 225, 235), anchor="lm")
    d.text((180, 730), "brama 4, o 3:00.", font=font("sans", 38), fill=(225, 225, 235), anchor="lm")
    d.text((180, 785), "Bez spóźnień.", font=font("sans", 38), fill=(225, 225, 235), anchor="lm")
    d.rounded_rectangle([40, 860, W - 40, 990], 36, fill=(34, 34, 44))
    d.text((80, 925), "3 nieodebrane połączenia", font=font("sans", 36), fill=(220, 220, 230), anchor="lm")
    # bottom icons
    d.ellipse([80, H - 190, 180, H - 90], fill=(50, 50, 62))
    d.ellipse([W - 180, H - 190, W - 80, H - 90], fill=(50, 50, 62))
    d.rounded_rectangle([W / 2 - 110, H - 40, W / 2 + 110, H - 28], 6, fill=(230, 230, 240))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    return to_np(img)


# ---------------------------------------------------------------------------
# beer can label (unrolled), 10 px / mm
# ---------------------------------------------------------------------------
def can_label(circ_mm, h_mm):
    s = 10
    W, H = int(circ_mm * s), int(h_mm * s)
    img = Image.new("RGB", (W, H), (14, 42, 30))
    d = ImageDraw.Draw(img)
    # gold bands
    d.rectangle([0, 0, W, 10 * s], fill=(196, 160, 72))
    d.rectangle([0, H - 12 * s, W, H], fill=(196, 160, 72))
    d.rectangle([0, 11 * s, W, 12 * s], fill=(230, 200, 110))
    d.rectangle([0, H - 14 * s, W, H - 13 * s], fill=(230, 200, 110))
    # diagonal stripes pattern
    for x in range(-H, W, 22):
        d.line([(x, 13 * s), (x + H, H - 15 * s)], fill=(18, 50, 36), width=6)
    for rep in (0, 1):
        ox = rep * W / 2
        # crest
        cx, cy = ox + W * 0.25, H * 0.33
        d.polygon([(cx - 90, cy - 80), (cx + 90, cy - 80), (cx + 90, cy + 20), (cx, cy + 100), (cx - 90, cy + 20)],
                  fill=(196, 160, 72), outline=(250, 230, 160))
        d.polygon([(cx - 70, cy - 62), (cx + 70, cy - 62), (cx + 70, cy + 12), (cx, cy + 78), (cx - 70, cy + 12)],
                  fill=(120, 20, 24))
        d.text((cx, cy - 8), "M", font=font("serif-bold", 110), fill=(245, 225, 160), anchor="mm")
        # main word
        d.text((cx, H * 0.62), "MOCNE", font=font("sans-bold", 150), fill=(245, 245, 235), anchor="mm")
        d.text((cx, H * 0.62), "MOCNE", font=font("sans-bold", 150), fill=None, anchor="mm")
        d.text((cx, H * 0.75), "PIWO JASNE PEŁNE", font=font("sans-bold", 44), fill=(220, 190, 100), anchor="mm")
        d.text((cx, H * 0.82), "7,5% ALK.  0,5 L", font=font("sans", 38), fill=(200, 200, 190), anchor="mm")
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    return to_np(img)


# ---------------------------------------------------------------------------
# payment card (fictional, no issuer brand)
# ---------------------------------------------------------------------------
def card():
    s = 10
    W, H = int(85.6 * s), int(54.0 * s)
    img = Image.new("RGB", (W, H), (20, 20, 24))
    d = ImageDraw.Draw(img)
    for k in range(0, W + H, 12):
        d.line([(k, 0), (k - H, H)], fill=(26, 26, 31), width=3)
    d.text((6 * s, 8 * s), "PLATINUM", font=font("sans-bold", 5.2 * s), fill=(180, 180, 185), anchor="lm")
    d.text((W - 6 * s, 8 * s), ")))", font=font("sans-bold", 5 * s), fill=(150, 150, 155), anchor="rm")
    d.text((6 * s, 36 * s), "4716  2203  9981  4052", font=font("mono-bold", 5.4 * s), fill=(205, 205, 210), anchor="lm")
    d.text((6 * s, 43 * s), "VALID THRU 09/29", font=font("mono", 2.6 * s), fill=(170, 170, 175), anchor="lm")
    d.text((6 * s, 48.5 * s), "J KOWALSKI", font=font("mono-bold", 3.6 * s), fill=(190, 190, 195), anchor="lm")
    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return to_np(img)


def text_mask(txt, fnt, size, px):
    """Tight white-on-black mask of a text (for engravings)."""
    f = font(fnt, size)
    l, t, r, b = f.getbbox(txt)
    img = Image.new("L", (r - l + 2 * px, b - t + 2 * px), 0)
    ImageDraw.Draw(img).text((px - l, px - t), txt, font=f, fill=255)
    return np.asarray(img).astype(np.float32) / 255.0


def bottle_label(circ_mm, h_mm):
    """Unrolled whisky label (fictional brand), 10 px / mm.  Front panel centred at u = 1/4."""
    sc = 10
    W, H = int(circ_mm * sc), int(h_mm * sc)
    img = Image.new("RGB", (W, H), (26, 22, 20))
    d = ImageDraw.Draw(img)
    fw = int(W * 0.46)
    x0 = int(W * 0.25 - fw / 2)
    d.rectangle([x0, 0, x0 + fw, H], fill=(222, 208, 176))
    d.rectangle([x0 + 18, 18, x0 + fw - 18, H - 18], outline=(150, 112, 50), width=5)
    cx = x0 + fw / 2
    # raven silhouette
    d.polygon([(cx - 70, 250), (cx - 10, 205), (cx + 40, 200), (cx + 75, 178), (cx + 62, 206), (cx + 30, 222),
               (cx + 55, 262), (cx + 5, 250), (cx - 25, 280), (cx - 40, 262)], fill=(20, 18, 16))
    d.text((cx, 90), "KRUK", font=font("serif-bold", 120), fill=(26, 22, 20), anchor="mm")
    d.text((cx, 350), "BLENDED WHISKY", font=font("serif-bold", 50), fill=(120, 30, 26), anchor="mm")
    d.text((cx, 420), "AGED 12 YEARS", font=font("serif", 40), fill=(40, 34, 30), anchor="mm")
    d.text((cx, H - 70), "40% vol   0,7 L", font=font("serif", 40), fill=(40, 34, 30), anchor="mm")
    d.text((W * 0.75, H / 2), "KRUK", font=font("serif-bold", 70), fill=(190, 160, 90), anchor="mm")
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    return to_np(img)
