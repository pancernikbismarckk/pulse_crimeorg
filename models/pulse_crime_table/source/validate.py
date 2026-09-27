"""Independent validation of the exported deliverables.

Re-imports the FBX files into an empty Blender scene and checks:
  * triangle budgets per LOD, object names, materials, UV layers
  * pivot / transforms (identity, origin at floor centre, Z-up, metres)
  * UV islands: all LOD0 UVs inside 0..1 and no overlapping triangles
    (rasterised coverage test on a 2048 grid)
  * face orientation: no back faces visible from above/sides (ray test)
  * texture files: presence and 2048x2048 resolution
Writes preview/validation.json and prints a summary.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "pulse_crime_table"
LIMITS = {"lod0": 10000, "lod1": 5000, "lod2": 1000}


def tri_count(ob):
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


def uv_overlap(ob, res=2048):
    """Count texels covered by more than one triangle (pixel-centre sampling)."""
    me = ob.data
    uvl = me.uv_layers.active.data
    cnt = np.zeros((res, res), np.uint16)
    outside = 0
    for p in me.polygons:
        loops = list(p.loop_indices)
        pts = [tuple(uvl[li].uv) for li in loops]
        for (u, v) in pts:
            if u < -1e-4 or u > 1 + 1e-4 or v < -1e-4 or v > 1 + 1e-4:
                outside += 1
        for k in range(1, len(pts) - 1):
            a, b, c = pts[0], pts[k], pts[k + 1]
            xs = np.array([a[0], b[0], c[0]]) * res
            ys = np.array([a[1], b[1], c[1]]) * res
            x0, x1 = int(max(math.floor(xs.min()), 0)), int(min(math.ceil(xs.max()), res - 1))
            y0, y1 = int(max(math.floor(ys.min()), 0)), int(min(math.ceil(ys.max()), res - 1))
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(d) < 1e-12:
                continue
            l1 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / d
            l2 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / d
            l3 = 1 - l1 - l2
            eps = 1e-6
            inside = (l1 > eps) & (l2 > eps) & (l3 > eps)
            cnt[y0:y1 + 1, x0:x1 + 1] += inside.astype(np.uint16)
    return int((cnt > 1).sum()), int((cnt > 0).sum()), outside


def backface_test(ob, n=6000, seed=1):
    """Shoot rays from a hemisphere of view points at the object; count hits on
    faces whose normal points away from the viewer (visible back faces)."""
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.transform(ob.matrix_world)
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    mats = [f.material_index for f in bm.faces]
    bb_min = Vector(np.min([v.co for v in bm.verts], axis=0))
    bb_max = Vector(np.max([v.co for v in bm.verts], axis=0))
    centre = (bb_min + bb_max) / 2
    bad = 0
    hits = 0
    for _ in range(n):
        # viewer on a hemisphere around the prop (above floor), aiming at a random surface point
        th = rng.uniform(0, 2 * math.pi)
        el = rng.uniform(math.radians(5), math.radians(85))
        eye = centre + Vector((math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el))) * 2.5
        tgt = Vector((rng.uniform(bb_min.x, bb_max.x), rng.uniform(bb_min.y, bb_max.y), rng.uniform(bb_min.z, bb_max.z)))
        d = (tgt - eye).normalized()
        loc, nrm, idx, dist = bvh.ray_cast(eye, d)
        if idx is None:
            continue
        if mats[idx] == 2:     # glass is rendered double sided
            continue
        hits += 1
        if nrm.dot(d) > 0.05:
            bad += 1
    bm.free()
    return bad, hits


def main():
    report = {"files": {}, "checks": {}}
    ok = True
    bpy.ops.wm.read_factory_settings(use_empty=True)
    fbx = os.path.join(ROOT, "fbx", NAME + ".fbx")
    bpy.ops.import_scene.fbx(filepath=fbx)
    objs = {o.name: o for o in bpy.data.objects if o.type == 'MESH'}
    report["files"]["fbx"] = sorted(objs.keys())
    for lod in ("lod0", "lod1", "lod2", "col"):
        nm = "%s_%s" % (NAME, lod)
        ob = objs.get(nm)
        if ob is None:
            ok = False
            report["checks"][nm] = "MISSING"
            continue
        me = ob.data
        tris = tri_count(ob)
        bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
        mn = [round(min(v[i] for v in bb), 4) for i in range(3)]
        mx = [round(max(v[i] for v in bb), 4) for i in range(3)]
        entry = {
            "tris": tris,
            "verts": len(me.vertices),
            "materials": [m.name for m in me.materials],
            "uv_layers": [u.name for u in me.uv_layers],
            "location": [round(x, 6) for x in ob.location],
            "rotation": [round(math.degrees(x), 4) for x in ob.rotation_euler],
            "scale": [round(x, 6) for x in ob.scale],
            "bbox_min": mn, "bbox_max": mx,
        }
        if lod in LIMITS:
            entry["tri_limit"] = LIMITS[lod]
            entry["tri_ok"] = tris <= LIMITS[lod]
            ok &= entry["tri_ok"]
            ov, cov, outside = uv_overlap(ob)
            entry["uv_overlap_texels"] = ov
            entry["uv_covered_texels"] = cov
            entry["uv_corners_outside_0_1"] = outside
            ok &= (ov == 0 and outside == 0)
            bad, hits = backface_test(ob)
            entry["backface_hits"] = "%d / %d rays" % (bad, hits)
            ok &= bad == 0
        entry["transform_identity"] = (entry["location"] == [0.0, 0.0, 0.0] and
                                       all(abs(a) < 1e-3 for a in entry["rotation"]) and
                                       all(abs(s - 1) < 1e-4 for s in entry["scale"]))
        ok &= entry["transform_identity"]
        report["checks"][nm] = entry
    # pivot: floor level & centred footprint
    l0 = report["checks"].get(NAME + "_lod0", {})
    if l0 and l0 != "MISSING":
        piv = {"min_z": l0["bbox_min"][2], "centre_x": round((l0["bbox_min"][0] + l0["bbox_max"][0]) / 2, 4),
               "centre_y": round((l0["bbox_min"][1] + l0["bbox_max"][1]) / 2, 4)}
        piv["ok"] = abs(piv["min_z"]) < 1e-3 and abs(piv["centre_x"]) < 1e-3 and abs(piv["centre_y"]) < 1e-3
        report["checks"]["pivot"] = piv
        ok &= piv["ok"]
    # textures
    tex = {}
    for sub in ("pbr", "gta"):
        d = os.path.join(ROOT, "textures", sub)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".png"):
                im = Image.open(os.path.join(d, fn))
                tex[sub + "/" + fn] = "%dx%d %s" % (im.size[0], im.size[1], im.mode)
                ok &= im.size == (2048, 2048)
    report["files"]["textures"] = tex
    # separate LOD files load and contain one mesh each
    for suf in ("lod0", "lod1", "lod2", "col"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=os.path.join(ROOT, "fbx", "%s_%s.fbx" % (NAME, suf)))
        ms = [o.name for o in bpy.data.objects if o.type == 'MESH']
        report["files"]["%s_%s.fbx" % (NAME, suf)] = ms
        ok &= len(ms) == 1
    report["all_ok"] = bool(ok)
    out = os.path.join(ROOT, "preview", "validation.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("VALIDATION", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
