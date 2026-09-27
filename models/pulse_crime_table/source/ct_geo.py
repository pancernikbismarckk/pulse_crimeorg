"""Low-level mesh building blocks with parametric (per-island, metric) UVs.

Every face carries local UV coordinates expressed in metres inside its own
"island".  Islands are later packed into a single 2048 atlas (ct_pack.py) and
painted procedurally (ct_paint.py).  Because LOD1/LOD2 geometry is generated
with the same island ids and the same local coordinate conventions, all LODs
sample the very same atlas without any re-baking.
"""
import math

SQ2 = math.sqrt(2.0)


# --------------------------------------------------------------------------
# island registry
# --------------------------------------------------------------------------
class Island:
    __slots__ = ("id", "kind", "weight", "params", "bounds", "alias")

    def __init__(self, iid, kind, weight=1.0, params=None, alias=None):
        self.id = iid
        self.kind = kind
        self.weight = float(weight)
        self.params = dict(params or {})
        self.bounds = None      # [umin, vmin, umax, vmax] of *real* island (LOD0 + others)
        self.alias = alias      # (target_id, (u0,v0,u1,v1) src rect, (u0,v0,u1,v1) dst rect)

    def grow(self, u, v):
        b = self.bounds
        if b is None:
            self.bounds = [u, v, u, v]
        else:
            if u < b[0]: b[0] = u
            if v < b[1]: b[1] = v
            if u > b[2]: b[2] = u
            if v > b[3]: b[3] = v


class Registry:
    def __init__(self):
        self.islands = {}

    def add(self, iid, kind, weight=1.0, **params):
        isl = self.islands.get(iid)
        if isl is None:
            self.islands[iid] = Island(iid, kind, weight, params)
        return iid

    def alias(self, iid, target, src_rect, dst_rect=None):
        """Island that has no atlas space of its own: its local UVs are
        remapped linearly from src_rect into dst_rect of `target`."""
        if iid not in self.islands:
            isl = Island(iid, "alias", 0.0)
            isl.alias = (target, tuple(src_rect), None if dst_rect is None else tuple(dst_rect))
            self.islands[iid] = isl
        return iid

    def get(self, iid):
        return self.islands[iid]


REG = Registry()
RECORD = [True]      # geometry meta-data (band segments, lathe radii) is recorded while True (LOD0 build)


def compute_bounds(reg, meshes):
    for isl in reg.islands.values():
        isl.bounds = None
    for m in meshes:
        for (idx, uvs, iid, mat, sm) in m.f:
            isl = reg.islands[iid]
            if isl.alias is not None:
                continue
            for (u, v) in uvs:
                isl.grow(u, v)


# --------------------------------------------------------------------------
# small vector helpers (plain tuples, no numpy needed here)
# --------------------------------------------------------------------------
def v2sub(a, b): return (a[0] - b[0], a[1] - b[1])
def v2add(a, b): return (a[0] + b[0], a[1] + b[1])
def v2mul(a, s): return (a[0] * s, a[1] * s)
def v2len(a): return math.hypot(a[0], a[1])
def v2norm(a):
    l = v2len(a)
    return (a[0] / l, a[1] / l) if l > 1e-12 else (0.0, 0.0)
def v2dot(a, b): return a[0] * b[0] + a[1] * b[1]


def poly_area(P):
    a = 0.0
    for i in range(len(P)):
        x0, y0 = P[i]
        x1, y1 = P[(i + 1) % len(P)]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def ensure_ccw(P):
    return list(P) if poly_area(P) > 0 else list(reversed(P))


def inset(P, d):
    """Offset a CCW polygon inwards by d (miter join, clamped)."""
    n = len(P)
    out = []
    for i in range(n):
        p0, p1, p2 = P[i - 1], P[i], P[(i + 1) % n]
        e1 = v2norm(v2sub(p1, p0))
        e2 = v2norm(v2sub(p2, p1))
        n1 = (-e1[1], e1[0])
        n2 = (-e2[1], e2[0])
        den = 1.0 + v2dot(n1, n2)
        den = max(den, 0.35)
        k = d / den
        out.append((p1[0] + (n1[0] + n2[0]) * k, p1[1] + (n1[1] + n2[1]) * k))
    return out


def rounded_rect(w, h, r, seg=3, cx=0.0, cy=0.0):
    """CCW rounded rectangle centred at (cx,cy); seg = arc subdivisions."""
    pts = []
    if r <= 0:
        return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    corners = [(cx + w / 2 - r, cy - h / 2 + r, -90), (cx + w / 2 - r, cy + h / 2 - r, 0),
               (cx - w / 2 + r, cy + h / 2 - r, 90), (cx - w / 2 + r, cy - h / 2 + r, 180)]
    for (ccx, ccy, a0) in corners:
        for k in range(seg + 1):
            a = math.radians(a0 + 90.0 * k / seg)
            pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    return pts


def circle(r, n, cx=0.0, cy=0.0, a0=0.0):
    return [(cx + r * math.cos(a0 + 2 * math.pi * i / n), cy + r * math.sin(a0 + 2 * math.pi * i / n)) for i in range(n)]


# --------------------------------------------------------------------------
# mesh container
# --------------------------------------------------------------------------
class Mesh:
    def __init__(self):
        self.v = []
        self.f = []   # (idx tuple, uv tuple, island id, material, smooth)

    def vert(self, p):
        self.v.append((float(p[0]), float(p[1]), float(p[2])))
        return len(self.v) - 1

    def face(self, idx, uv, island, mat="main", smooth=True):
        idx = tuple(idx)
        uv = tuple((float(a), float(b)) for a, b in uv)
        assert len(idx) == len(uv) and len(idx) >= 3, (idx, uv)
        # drop degenerate repeated indices (keep order)
        if len(set(idx)) != len(idx):
            ii, uu = [], []
            for a, t in zip(idx, uv):
                if not ii or ii[-1] != a:
                    ii.append(a); uu.append(t)
            if ii[0] == ii[-1]:
                ii.pop(); uu.pop()
            if len(ii) < 3:
                return
            idx, uv = tuple(ii), tuple(uu)
        self.f.append((idx, uv, island, mat, smooth))

    def extend(self, other, xf=None):
        off = len(self.v)
        for p in other.v:
            self.v.append(xf(p) if xf else p)
        for (idx, uv, isl, mat, sm) in other.f:
            self.f.append((tuple(i + off for i in idx), uv, isl, mat, sm))
        return self

    def transformed(self, xf):
        m = Mesh()
        m.v = [xf(p) for p in self.v]
        m.f = list(self.f)
        return m

    def tri_count(self):
        return sum(len(f[0]) - 2 for f in self.f)

    def bbox(self):
        xs = [p[0] for p in self.v]; ys = [p[1] for p in self.v]; zs = [p[2] for p in self.v]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return lambda p: (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return lambda p: (p[0], c * p[1] - s * p[2], s * p[1] + c * p[2])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return lambda p: (c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])


def trans(dx, dy, dz):
    return lambda p: (p[0] + dx, p[1] + dy, p[2] + dz)


def scale(sx, sy=None, sz=None):
    sy = sx if sy is None else sy
    sz = sx if sz is None else sz
    return lambda p: (p[0] * sx, p[1] * sy, p[2] * sz)


def chain(*fs):
    def f(p):
        for g in fs:
            p = g(p)
        return p
    return f


def place(x, y, z, yaw_deg=0.0):
    return chain(rot_z(math.radians(yaw_deg)), trans(x, y, z))


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
def extrude(m, poly, z0, z1, *, top=None, bot=None, band=None, ct=0.0, cb=0.0,
            band_skip=(), mat="main", band_mat=None, top_mat=None, bot_mat=None,
            smooth=True, uv_origin=(0.0, 0.0), bseg=1):
    """Prism from a 2D polygon (CCW seen from +z) between z0 and z1.

    top/bot : island id for the caps (None -> cap omitted)
    band    : island id, or list with an island id per polygon edge (None -> edge skipped)
    ct/cb   : 45deg chamfer size on the top/bottom outline
    Returns dict with vertex index rings (useful for custom caps).
    UV conventions (metres):
        top cap : (x, y)          bottom cap : (x, -y)
        band    : (perimeter s, z) with chamfers continuing along v.
    """
    P = ensure_ccw(poly)
    n = len(P)
    Pt = inset(P, ct) if ct > 0 else P
    Pb = inset(P, cb) if cb > 0 else P
    rings, vs = [], []
    if bseg <= 1:
        if cb > 0:
            rings.append((Pb, z0)); vs.append(z0 + cb - cb * SQ2)
        rings.append((P, z0 + cb)); vs.append(z0 + cb)
        rings.append((P, z1 - ct)); vs.append(z1 - ct)
        if ct > 0:
            rings.append((Pt, z1)); vs.append(z1 - ct + ct * SQ2)
    else:
        # rounded (quarter circle) bevels with bseg segments; v = true arc length
        lo = []
        if cb > 0:
            for k in range(bseg, 0, -1):
                ph = 0.5 * math.pi * k / bseg
                lo.append((cb * (1 - math.cos(ph)), z0 + cb - cb * math.sin(ph)))
        lo.append((0.0, z0 + cb))
        hi = [(0.0, z1 - ct)]
        if ct > 0:
            for k in range(1, bseg + 1):
                ph = 0.5 * math.pi * k / bseg
                hi.append((ct * (1 - math.cos(ph)), z1 - ct + ct * math.sin(ph)))
        prof = lo + hi
        # arc length measured from the start of the straight wall
        base = len(lo) - 1
        vv = [0.0] * len(prof)
        vv[base] = z0 + cb
        for k in range(base - 1, -1, -1):
            vv[k] = vv[k + 1] - math.hypot(prof[k + 1][0] - prof[k][0], prof[k + 1][1] - prof[k][1])
        for k in range(base + 1, len(prof)):
            vv[k] = vv[k - 1] + math.hypot(prof[k][0] - prof[k - 1][0], prof[k][1] - prof[k - 1][1])
        for (d, z), v in zip(prof, vv):
            rings.append((inset(P, d) if d > 0 else P, z)); vs.append(v)
        Pt = rings[-1][0]
        Pb = rings[0][0]
    ridx = [[m.vert((x, y, z)) for (x, y) in pts] for (pts, z) in rings]
    s = [0.0]
    for i in range(n):
        s.append(s[-1] + v2len(v2sub(P[(i + 1) % n], P[i])))
    bands = band if isinstance(band, (list, tuple)) else [band] * n
    bm = band_mat or mat
    # edges at the start that belong to the same island as the closing edge are
    # shifted by the full perimeter so the island stays continuous across the seam
    shift = [0.0] * n
    if isinstance(band, (list, tuple)) and bands[-1] is not None:
        for i in range(n - 1):
            if bands[i] == bands[-1]:
                shift[i] = s[n]
            else:
                break
    for i in range(n):
        isl = bands[i]
        if isl is None or i in band_skip:
            continue
        j = (i + 1) % n
        u0, u1 = s[i] + shift[i], s[i + 1] + shift[i]
        if RECORD[0] and isl in REG.islands:
            REG.islands[isl].params.setdefault("band_segs", []).append((u0, u1, P[i], P[j]))
        for r in range(len(rings) - 1):
            a, b = ridx[r][i], ridx[r][j]
            c, d = ridx[r + 1][j], ridx[r + 1][i]
            m.face((a, b, c, d), ((u0, vs[r]), (u1, vs[r]), (u1, vs[r + 1]), (u0, vs[r + 1])), isl, bm, smooth)
    ox, oy = uv_origin
    if top:
        m.face(ridx[-1], [(x - ox, y - oy) for (x, y) in Pt], top, top_mat or mat, smooth)
    if bot:
        m.face(list(reversed(ridx[0])), [(x - ox, -(y - oy)) for (x, y) in reversed(Pb)], bot, bot_mat or mat, smooth)
    return {"rings": ridx, "top_pts": Pt, "bot_pts": Pb, "perimeter": s, "poly": P}


def bezel_cap(m, loop_idx, loop_pts, z, rect, depth, *, bezel, screen, wall=None,
              mat="main", screen_mat="emissive"):
    """Fill a (top, +z facing) outline with a bezel ring around a recessed
    rectangular screen.  rect=(x0,y0,x1,y1) must lie inside the outline."""
    x0, y0, x1, y1 = rect
    cache = {}

    def inner(p):
        q = (min(max(p[0], x0), x1), min(max(p[1], y0), y1))
        key = (round(q[0], 7), round(q[1], 7))
        if key not in cache:
            cache[key] = (m.vert((q[0], q[1], z)), q)
        return cache[key]

    n = len(loop_idx)
    inn = [inner(p) for p in loop_pts]
    for i in range(n):
        j = (i + 1) % n
        (qi, qpi), (qj, qpj) = inn[i], inn[j]
        pi, pj = loop_pts[i], loop_pts[j]
        if qi == qj:
            m.face((loop_idx[i], loop_idx[j], qi), (pi, pj, qpi), bezel, mat)
        else:
            m.face((loop_idx[i], loop_idx[j], qj, qi), (pi, pj, qpj, qpi), bezel, mat)
    # make sure all 4 rect corners exist (they do if outline surrounds rect)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    ci = [inner(c)[0] for c in corners]
    zi = z - depth
    di = [m.vert((c[0], c[1], zi)) for c in corners]
    if depth > 0 and wall:
        per = [0.0]
        for k in range(4):
            a, b = corners[k], corners[(k + 1) % 4]
            per.append(per[-1] + v2len(v2sub(b, a)))
        for k in range(4):
            k2 = (k + 1) % 4
            # wall faces the screen centre (normal pointing inwards, visible from above)
            m.face((ci[k], ci[k2], di[k2], di[k]),
                   ((per[k], 0.0), (per[k + 1], 0.0), (per[k + 1], depth), (per[k], depth)), wall, mat)
    m.face(di, [(c[0], c[1]) for c in corners], screen, screen_mat)
    return di


def lathe(m, prof, nseg, islands, *, uvkind=None, mat="main", mats=None, theta0=0.0,
          mod=None, smooth=True, close_top=False):
    """Surface of revolution around +z.

    prof    : list of (r, z) traced so that faces point outwards
              (outer wall upwards, tops inwards, inner walls downwards).
    islands : per-segment island id (None -> segment skipped)
    uvkind  : per-segment 'side' | 'up' | 'down' (auto if None)
    mod     : optional f(theta, k, r, z) -> (r, z) per-vertex modulation
    UV: side -> (theta * r_ref, profile arclength); up -> (x, y); down -> (x, -y)
    """
    nprof = len(prof)
    nsegp = nprof - 1
    kinds = []
    for k in range(nsegp):
        (r0, z0), (r1, z1) = prof[k], prof[k + 1]
        if uvkind and uvkind[k]:
            kinds.append(uvkind[k])
        elif abs(z1 - z0) >= abs(r1 - r0):
            kinds.append("side")
        else:
            kinds.append("up" if r1 < r0 else "down")
    # arclength
    sarc = [0.0]
    for k in range(nsegp):
        (r0, z0), (r1, z1) = prof[k], prof[k + 1]
        sarc.append(sarc[-1] + math.hypot(r1 - r0, z1 - z0))
    # r_ref per island (side segments)
    racc = {}
    for k in range(nsegp):
        if islands[k] and kinds[k] == "side":
            a = racc.setdefault(islands[k], [0.0, 0])
            a[0] += 0.5 * (prof[k][0] + prof[k + 1][0]); a[1] += 1
    rref = {i: max(a[0] / a[1], 1e-4) for i, a in racc.items()}
    if RECORD[0]:
        for i, rr in rref.items():
            if i in REG.islands:
                REG.islands[i].params.setdefault("rref", rr)
                REG.islands[i].params.setdefault("arc", list(sarc))
                REG.islands[i].params.setdefault("prof", list(prof))
    angs = [theta0 + 2 * math.pi * j / nseg for j in range(nseg)]
    rings = []
    pos = []
    for k, (r, z) in enumerate(prof):
        if r <= 1e-9:
            idx = m.vert((0.0, 0.0, z))
            rings.append([idx] * nseg)
            pos.append([(0.0, 0.0)] * nseg)
        else:
            ring, pp = [], []
            for j, a in enumerate(angs):
                rr, zz = (mod(a, k, r, z) if mod else (r, z))
                ring.append(m.vert((rr * math.cos(a), rr * math.sin(a), zz)))
                pp.append((rr * math.cos(a), rr * math.sin(a)))
            rings.append(ring)
            pos.append(pp)
    for k in range(nsegp):
        isl = islands[k]
        if not isl:
            continue
        kd = kinds[k]
        mt = mats[k] if mats else mat
        for j in range(nseg):
            j2 = (j + 1) % nseg
            a, b = rings[k][j], rings[k][j2]
            c, d = rings[k + 1][j2], rings[k + 1][j]
            if kd == "side":
                rr = rref[isl]
                ua = (angs[j] - theta0) * rr
                ub = (angs[j] - theta0 + 2 * math.pi / nseg) * rr
                uvs = ((ua, sarc[k]), (ub, sarc[k]), (ub, sarc[k + 1]), (ua, sarc[k + 1]))
            elif kd == "up":
                uvs = (pos[k][j], pos[k][j2], pos[k + 1][j2], pos[k + 1][j])
            else:
                uvs = tuple((x, -y) for (x, y) in (pos[k][j], pos[k][j2], pos[k + 1][j2], pos[k + 1][j]))
            idx = (a, b, c, d)
            if a == b:          # pole at start
                m.face((a, c, d), (uvs[0], uvs[2], uvs[3]), isl, mt, smooth)
            elif c == d:        # pole at end
                m.face((a, b, c), (uvs[0], uvs[1], uvs[2]), isl, mt, smooth)
            else:
                m.face(idx, uvs, isl, mt, smooth)
    return {"rings": rings, "arc": sarc, "rref": rref}


def quad_strip(m, pts_left, pts_right, uv_left, uv_right, island, mat="main", smooth=True):
    """Generic strip between two polylines (same count). Faces oriented using
    left->right x forward ordering (caller responsible)."""
    idl = [m.vert(p) for p in pts_left]
    idr = [m.vert(p) for p in pts_right]
    for k in range(len(idl) - 1):
        m.face((idl[k], idr[k], idr[k + 1], idl[k + 1]),
               (uv_left[k], uv_right[k], uv_right[k + 1], uv_left[k + 1]), island, mat, smooth)
    return idl, idr


def grid_sheet(m, w, h, nu, nv, zf, island, *, mat="main", back_island=None, thickness=0.0,
               uvf=None):
    """A (possibly bent) sheet in local XY centred at origin: z = zf(x, y).
    Front faces +z.  Optional back side (offset by -thickness along z)."""
    xs = [-w / 2 + w * i / nu for i in range(nu + 1)]
    ys = [-h / 2 + h * j / nv for j in range(nv + 1)]
    ids = [[m.vert((x, y, zf(x, y))) for x in xs] for y in ys]
    for j in range(nv):
        for i in range(nu):
            a, b, c, d = ids[j][i], ids[j][i + 1], ids[j + 1][i + 1], ids[j + 1][i]
            uv = [(xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])]
            if uvf:
                uv = [uvf(*t) for t in uv]
            m.face((a, b, c, d), uv, island, mat, True)
    if back_island:
        idb = [[m.vert((x, y, zf(x, y) - thickness)) for x in xs] for y in ys]
        for j in range(nv):
            for i in range(nu):
                a, b, c, d = idb[j][i], idb[j][i + 1], idb[j + 1][i + 1], idb[j + 1][i]
                # reversed winding, mirrored-x uv so it is not flipped
                uv = [(-xs[i], ys[j]), (-xs[i + 1], ys[j]), (-xs[i + 1], ys[j + 1]), (-xs[i], ys[j + 1])]
                m.face((d, c, b, a), (uv[3], uv[2], uv[1], uv[0]), back_island, mat, True)
    return ids
