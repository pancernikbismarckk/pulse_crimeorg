"""Atlas packing (MaxRects, best-short-side-fit, 90deg rotation) of metric
UV islands + the island -> atlas transform shared by every LOD."""
import math

ATLAS = 2048
PAD = 5            # px of padding around every island (at 2048)


class Placed:
    __slots__ = ("id", "x", "y", "w", "h", "rot", "k", "b")

    def __init__(self, iid, x, y, w, h, rot, k, b):
        self.id, self.x, self.y, self.w, self.h, self.rot, self.k, self.b = iid, x, y, w, h, rot, k, b

    # local metres -> atlas pixel coords (origin top-left, y down) at 2048
    def to_px(self, u, v):
        umin, vmin, umax, vmax = self.b
        if not self.rot:
            return (self.x + PAD + (u - umin) * self.k, self.y + PAD + (vmax - v) * self.k)
        return (self.x + PAD + (v - vmin) * self.k, self.y + PAD + (u - umin) * self.k)

    def to_uv(self, u, v):
        x, y = self.to_px(u, v)
        return (x / ATLAS, 1.0 - y / ATLAS)


def _maxrects(sizes, W, H):
    free = [(0, 0, W, H)]
    out = {}
    order = sorted(sizes, key=lambda r: (-max(r[1], r[2]), -min(r[1], r[2]), r[0]))
    for (rid, w, h) in order:
        best = None
        for (fx, fy, fw, fh) in free:
            for (rw, rh, rot) in ((w, h, False), (h, w, True)):
                if rw <= fw and rh <= fh:
                    lw, lh = fw - rw, fh - rh
                    score = (min(lw, lh), max(lw, lh), fy, fx)
                    if best is None or score < best[0]:
                        best = (score, fx, fy, rw, rh, rot)
        if best is None:
            return None
        _, x, y, rw, rh, rot = best
        out[rid] = (x, y, rw, rh, rot)
        # split free rectangles
        nf = []
        for (fx, fy, fw, fh) in free:
            if x >= fx + fw or x + rw <= fx or y >= fy + fh or y + rh <= fy:
                nf.append((fx, fy, fw, fh))
                continue
            if x > fx:
                nf.append((fx, fy, x - fx, fh))
            if x + rw < fx + fw:
                nf.append((x + rw, fy, fx + fw - (x + rw), fh))
            if y > fy:
                nf.append((fx, fy, fw, y - fy))
            if y + rh < fy + fh:
                nf.append((fx, y + rh, fw, fy + fh - (y + rh)))
        # prune contained
        nf = [r for r in nf if r[2] > 0 and r[3] > 0]
        nf.sort(key=lambda r: -(r[2] * r[3]))
        pruned = []
        for i, a in enumerate(nf):
            contained = False
            for b in pruned:
                if a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]:
                    contained = True
                    break
            if not contained:
                pruned.append(a)
        free = pruned
    return out


def pack(reg, lo=100.0, hi=3000.0, iters=16, verbose=True):
    """Find the largest global density s (px per metre at 2048, weight 1)
    such that all islands fit.  Returns (s, {id: Placed})."""
    isl = [i for i in reg.islands.values() if i.alias is None and i.weight > 0 and i.bounds is not None]

    def sizes(s):
        out = []
        for i in isl:
            umin, vmin, umax, vmax = i.bounds
            k = s * i.weight
            w = int(math.ceil((umax - umin) * k)) + 2 * PAD + 1
            h = int(math.ceil((vmax - vmin) * k)) + 2 * PAD + 1
            out.append((i.id, w, h))
        return out

    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        res = _maxrects(sizes(mid), ATLAS, ATLAS)
        if res is None:
            hi = mid
        else:
            lo = mid
            best = (mid, res)
    s, res = best
    placed = {}
    used = 0
    for i in isl:
        x, y, w, h, rot = res[i.id]
        placed[i.id] = Placed(i.id, x, y, w, h, rot, s * i.weight, tuple(i.bounds))
        used += w * h
    if verbose:
        print("pack: density %.1f px/m (weight 1) | %d islands | fill %.1f%%" % (s, len(placed), 100.0 * used / ATLAS ** 2))
    return s, placed


def resolve(reg, placed, iid, u, v):
    """Island-local metres -> atlas UV, following alias islands."""
    isl = reg.islands[iid]
    if isl.alias:
        tgt, src, dst = isl.alias
        if dst is None:
            dst = reg.islands[tgt].bounds
        su0, sv0, su1, sv1 = src
        du0, dv0, du1, dv1 = dst
        tu = (u - su0) / (su1 - su0) if su1 != su0 else 0.5
        tv = (v - sv0) / (sv1 - sv0) if sv1 != sv0 else 0.5
        u = du0 + tu * (du1 - du0)
        v = dv0 + tv * (dv1 - dv0)
        iid = tgt
        isl = reg.islands[iid]
        if isl.alias:
            return resolve(reg, placed, iid, u, v)
    p = placed.get(iid)
    if p is None:
        return (0.0, 0.0)
    return p.to_uv(u, v)


def check_bounds(reg, placed, meshes, tol_px=PAD - 0.5):
    """Verify every face corner of every LOD lands inside its island's padded rect."""
    bad = 0
    worst = 0.0
    for name, m in meshes.items():
        for (idx, uvs, iid, mat, sm) in m.f:
            isl = reg.islands[iid]
            if isl.kind == "col":
                continue
            tgt = iid
            while reg.islands[tgt].alias:
                tgt = reg.islands[tgt].alias[0]
            p = placed[tgt]
            for (u, v) in uvs:
                a = resolve(reg, placed, iid, u, v)
                x, y = a[0] * ATLAS, (1.0 - a[1]) * ATLAS
                dx = max(p.x + PAD - x, x - (p.x + p.w - PAD - 1), 0.0)
                dy = max(p.y + PAD - y, y - (p.y + p.h - PAD - 1), 0.0)
                d = max(dx, dy)
                if d > tol_px:
                    bad += 1
                    worst = max(worst, d)
    return bad, worst
