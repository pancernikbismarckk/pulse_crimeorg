"""Blender-side helpers: Mesh -> bpy object, smoothing, scene/render utils."""
import math
import bpy
import bmesh
from mathutils import Vector

MATS = ("main", "emissive", "glass")

# smoothing thresholds (deg) per island kind; everything else uses DEFAULT_SHARP
DEFAULT_SHARP = 40.0
SOFT_KINDS = {"powder": 89.0, "butt": 70.0, "cig": 70.0, "ember": 70.0, "ash": 70.0, "rolled_bill": 60.0,
              "strap": 80.0, "bill_front": 80.0, "bill_back": 80.0, "whisky": 60.0, "glass": 50.0}


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    sc.unit_settings.scale_length = 1.0
    return sc


def face_normal(verts, idx):
    nx = ny = nz = 0.0
    k = len(idx)
    for i in range(k):
        x0, y0, z0 = verts[idx[i]]
        x1, y1, z1 = verts[idx[(i + 1) % k]]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    l = math.sqrt(nx * nx + ny * ny + nz * nz)
    return (nx / l, ny / l, nz / l) if l > 0 else (0.0, 0.0, 1.0)


def to_object(md, name, uv_map_fn, reg, collection=None, triangulate=True):
    """md: ct_geo.Mesh ; uv_map_fn(island_id, (u,v)) -> atlas (u,v)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(md.v, [], [f[0] for f in md.f])
    uvl = me.uv_layers.new(name="UVMap")
    flat = []
    for f in md.f:
        for t in f[1]:
            a = uv_map_fn(f[2], t)
            flat.extend(a)
    uvl.data.foreach_set("uv", flat)
    me.polygons.foreach_set("material_index", [MATS.index(f[3]) for f in md.f])
    me.polygons.foreach_set("use_smooth", [bool(f[4]) for f in md.f])
    # --- sharp edges by angle with per-kind thresholds
    normals = [face_normal(md.v, f[0]) for f in md.f]
    thr = []
    for f in md.f:
        isl = reg.islands.get(f[2])
        kind = isl.kind if isl else ""
        if isl is not None and isl.alias:
            tgt = reg.islands.get(isl.alias[0])
            kind = tgt.kind if tgt else kind
        thr.append(SOFT_KINDS.get(kind, DEFAULT_SHARP))
    edge_faces = {}
    for fi, f in enumerate(md.f):
        idx = f[0]
        for i in range(len(idx)):
            a, b = idx[i], idx[(i + 1) % len(idx)]
            key = (a, b) if a < b else (b, a)
            edge_faces.setdefault(key, []).append(fi)
    me.update()
    sharp = [False] * len(me.edges)
    emap = {}
    for e in me.edges:
        a, b = e.vertices
        emap[(a, b) if a < b else (b, a)] = e.index
    for key, fl in edge_faces.items():
        ei = emap.get(key)
        if ei is None:
            continue
        if len(fl) != 2:
            sharp[ei] = True
            continue
        n0, n1 = normals[fl[0]], normals[fl[1]]
        d = max(-1.0, min(1.0, n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]))
        ang = math.degrees(math.acos(d))
        t = min(thr[fl[0]], thr[fl[1]])
        if ang > t or md.f[fl[0]][3] != md.f[fl[1]][3]:
            sharp[ei] = True
    attr = me.attributes.get("sharp_edge") or me.attributes.new("sharp_edge", 'BOOLEAN', 'EDGE')
    attr.data.foreach_set("value", sharp)
    me.update()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if triangulate:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.to_mesh(me)
        bm.free()
        me.update()
    return ob


def ensure_collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(col)
    return col


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def add_camera(name, loc, target, lens=50.0, ortho=None):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.clip_start = 0.01
    cam.clip_end = 100
    if ortho:
        cam.type = 'ORTHO'
        cam.ortho_scale = ortho
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    return ob


def add_area(name, loc, target, power, size, color=(1, 1, 1), col=None):
    l = bpy.data.lights.new(name, 'AREA')
    l.energy = power
    l.size = size
    l.color = color
    ob = bpy.data.objects.new(name, l)
    (col or bpy.context.scene.collection).objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    return ob


def setup_cycles(samples=64, res=(1600, 900), denoise=True):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = denoise
    try:
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    sc.render.image_settings.file_format = 'JPEG'
    sc.render.image_settings.quality = 90
    sc.cycles.max_bounces = 6
    sc.cycles.glossy_bounces = 3
    sc.cycles.transmission_bounces = 6
    sc.cycles.transparent_max_bounces = 12
    return sc


def render_to(path, camera):
    sc = bpy.context.scene
    sc.camera = camera
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
