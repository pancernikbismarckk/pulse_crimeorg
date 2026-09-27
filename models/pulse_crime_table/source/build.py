"""Build pipeline for the 'pulse_crime_table' FiveM prop.

    python build.py [--skip-ao] [--no-render] [--no-export] [--out DIR]

Requires Blender as a Python module (pip install bpy==4.5.*), numpy, pillow.
Stages: geometry (all LODs) -> atlas packing -> procedural painting ->
Cycles AO bake -> texture composition -> Blender scene/materials ->
FBX + .blend export -> validation -> preview renders.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bpy
import ct_geo
import ct_scene
import ct_pack
import ct_paint
import ct_blender as cb

NAME = "pulse_crime_table"
ROOT = os.path.dirname(HERE)


def log(*a):
    print("[build]", *a, flush=True)


# -----------------------------------------------------------------------------
# texture helpers
# -----------------------------------------------------------------------------
def save_png(path, a, mode=None):
    a = np.clip(a, 0, 1)
    arr = (a * 255.0 + 0.5).astype(np.uint8)
    im = Image.fromarray(arr, mode) if mode else Image.fromarray(arr)
    im.save(path, optimize=True)


def dilate(a, mask, iters=8):
    """Grow island content into empty (mask==0) pixels (for mip safety)."""
    a = a.copy()
    m = mask.astype(bool).copy()
    for _ in range(iters):
        if m.all():
            break
        acc = np.zeros_like(a, dtype=np.float64)
        cnt = np.zeros(m.shape, np.float64)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            sm = np.roll(np.roll(m, dy, 0), dx, 1)
            sa = np.roll(np.roll(a, dy, 0), dx, 1)
            acc += (sa * (sm[..., None] if a.ndim == 3 else sm))
            cnt += sm
        new = (~m) & (cnt > 0)
        if a.ndim == 3:
            a[new] = (acc[new] / cnt[new][..., None])
        else:
            a[new] = acc[new] / cnt[new]
        m |= new
    return a


# -----------------------------------------------------------------------------
# AO bake
# -----------------------------------------------------------------------------
def bake_ao(md0, uvfn, samples=96, distance=0.22):
    log("AO bake: %d samples, distance %.2f m" % (samples, distance))
    t = time.time()
    # proxy without glass (glass should not occlude)
    proxy = ct_geo.Mesh()
    proxy.v = md0.v
    proxy.f = [f for f in md0.f if f[3] != "glass"]
    ob = cb.to_object(proxy, "ao_proxy", uvfn, ct_geo.REG, triangulate=True)
    img = bpy.data.images.new("AO_bake", ct_pack.ATLAS, ct_pack.ATLAS, float_buffer=True, alpha=False)
    img.colorspace_settings.name = 'Non-Color'
    mat = bpy.data.materials.new("ao_tmp")
    mat.use_nodes = True
    tn = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tn.image = img
    mat.node_tree.nodes.active = tn
    ob.data.materials.append(mat)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    if sc.world is None:
        sc.world = bpy.data.worlds.new("bake_world")
    sc.world.light_settings.distance = distance
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.bake(type='AO', margin=ct_pack.PAD + 2, margin_type='EXTEND', use_clear=True, uv_layer="UVMap")
    ao = np.array(img.pixels[:], np.float32).reshape(ct_pack.ATLAS, ct_pack.ATLAS, 4)[::-1, :, 0].copy()
    bpy.data.objects.remove(ob)
    bpy.data.images.remove(img)
    bpy.data.materials.remove(mat)
    log("AO bake done in %.1fs" % (time.time() - t))
    return ao


def smooth_ao(ao, mask):
    """Light denoise inside each island rect (3x3 box twice)."""
    out = ao.copy()
    for _ in range(2):
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                same = np.roll(np.roll(mask, dy, 0), dx, 1) == mask
                acc += np.roll(np.roll(out, dy, 0), dx, 1) * same
                cnt += same
        out = acc / np.maximum(cnt, 1)
    return out


# -----------------------------------------------------------------------------
# materials
# -----------------------------------------------------------------------------
def make_materials(tex):
    def load(fn, cs):
        im = bpy.data.images.load(tex[fn])
        im.colorspace_settings.name = cs
        return im
    imgs = dict(base=load("basecolor", "sRGB"), rough=load("roughness", "Non-Color"),
                metal=load("metallic", "Non-Color"), nrm=load("normal_gl", "Non-Color"),
                emit=load("emissive", "sRGB"), opac=load("opacity", "Non-Color"))

    def mk(name, emissive=False, glass=False):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        uv = nt.nodes.new("ShaderNodeUVMap")
        uv.uv_map = "UVMap"
        uv.location = (-1100, 0)

        def tnode(key, y):
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = imgs[key]
            n.location = (-800, y)
            nt.links.new(uv.outputs["UV"], n.inputs["Vector"])
            return n
        nt.links.new(tnode("base", 300).outputs["Color"], b.inputs["Base Color"])
        nt.links.new(tnode("metal", 0).outputs["Color"], b.inputs["Metallic"])
        nt.links.new(tnode("rough", -250).outputs["Color"], b.inputs["Roughness"])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.space = 'TANGENT'
        nm.uv_map = "UVMap"
        nm.location = (-300, -500)
        nt.links.new(tnode("nrm", -500).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
        if emissive:
            nt.links.new(tnode("emit", -800).outputs["Color"], b.inputs["Emission Color"])
            b.inputs["Emission Strength"].default_value = 4.0
        if glass:
            nt.links.new(tnode("opac", -1100).outputs["Color"], b.inputs["Alpha"])
            try:
                m.surface_render_method = 'BLENDED'
            except Exception:
                pass
            m.use_backface_culling = False
        return m
    return [mk(NAME + "_main"), mk(NAME + "_emissive", emissive=True), mk(NAME + "_glass", glass=True)]


# -----------------------------------------------------------------------------
# preview scene
# -----------------------------------------------------------------------------
def preview_env(col):
    # floor + back wall (concrete-ish)
    fm = bpy.data.materials.new("pv_floor")
    fm.use_nodes = True
    nt = fm.node_tree
    b = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.055, 0.052, 0.050, 1)
    ramp.color_ramp.elements[1].color = (0.13, 0.125, 0.12, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.8
    me = bpy.data.meshes.new("pv_floor")
    me.from_pydata([(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0), (-6, 2.2, 0), (6, 2.2, 0), (6, 2.2, 3.5),
                    (-6, 2.2, 3.5)], [], [(0, 1, 2, 3), (4, 5, 6, 7)])
    me.materials.append(fm)
    ob = bpy.data.objects.new("pv_floor", me)
    col.objects.link(ob)
    w = bpy.data.worlds.new("pv_world")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.03, 0.035, 0.045, 1)
    bg.inputs[1].default_value = 1.0
    # lights: warm pendant lamp above, cool fill, rim
    cb.add_area("pv_key", (0.15, -0.10, 1.85), (0.0, 0.0, 0.78), 55, 0.55, (1.0, 0.80, 0.58), col)
    cb.add_area("pv_fill", (-2.2, -1.6, 1.6), (0, 0, 0.8), 45, 1.6, (0.55, 0.70, 1.0), col)
    cb.add_area("pv_rim", (1.6, 1.9, 1.7), (0, 0, 0.8), 60, 1.0, (0.9, 0.9, 1.0), col)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ao", action="store_true", help="use cached AO if present, never bake")
    ap.add_argument("--rebake", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--render-samples", type=int, default=96)
    ap.add_argument("--only-render", default="")
    args = ap.parse_args([a for a in sys.argv[1:] if a.startswith("--") or not a.endswith(".py")])

    out_tex = os.path.join(ROOT, "textures")
    out_pbr = os.path.join(out_tex, "pbr")
    out_gta = os.path.join(out_tex, "gta")
    out_fbx = os.path.join(ROOT, "fbx")
    out_blend = os.path.join(ROOT, "blend")
    out_prev = os.path.join(ROOT, "preview")
    cache = os.path.join(HERE, ".cache")
    for d in (out_pbr, out_gta, out_fbx, out_blend, out_prev, cache):
        os.makedirs(d, exist_ok=True)

    T0 = time.time()
    cb.reset_scene()
    # ---------------------------------------------------------------- geometry
    meshes = {lod: ct_scene.build(lod) for lod in (0, 1, 2)}
    col_mesh = ct_scene.collision()
    ct_geo.compute_bounds(ct_geo.REG, list(meshes.values()))
    s, placed = ct_pack.pack(ct_geo.REG)
    bad, worst = ct_pack.check_bounds(ct_geo.REG, placed, meshes)
    log("UV bounds check: %d corners outside their island (worst %.1f px)" % (bad, worst))
    uvfn = lambda iid, uv: ct_pack.resolve(ct_geo.REG, placed, iid, uv[0], uv[1])

    # ---------------------------------------------------------------- painting
    maps = ct_paint.paint_atlas(ct_geo.REG, placed)
    mask = maps["mask"]

    # ---------------------------------------------------------------- AO
    ao_cache = os.path.join(cache, "ao.npy")
    sig = "%d-%d-%.3f-%d" % (len(meshes[0].f), len(meshes[0].v), s, len(placed))
    sig_path = ao_cache + ".sig"
    cached_ok = os.path.exists(ao_cache) and os.path.exists(sig_path) and open(sig_path).read() == sig
    if cached_ok and not args.rebake:
        ao = np.load(ao_cache)
        log("AO: cached")
    elif args.skip_ao:
        ao = np.ones((ct_pack.ATLAS, ct_pack.ATLAS), np.float32)
        log("AO: skipped (white)")
    else:
        ao = bake_ao(meshes[0], uvfn, samples=args.samples)
        np.save(ao_cache, ao)
        open(sig_path, "w").write(sig)
    ao = np.where(mask > 0, ao, 1.0).astype(np.float32)
    ao = smooth_ao(ao, mask)
    # emissive / glass islands get no AO darkening
    ao = np.clip(ao, 0.0, 1.0)

    # ---------------------------------------------------------------- compose maps
    col = maps["col"]
    rough = maps["rough"]
    metal = maps["metal"]
    nrm = maps["normal"]
    emit = maps["emit"]
    alpha = maps["alpha"]
    used = mask > 0
    col = dilate(col, used)
    rough = dilate(rough, used)
    metal = dilate(metal, used)
    nrm = dilate(nrm, used)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    emit = dilate(emit, used)
    alpha = dilate(alpha, used)
    ao = dilate(ao, used)
    tex = {}

    def put(key, folder, fname, arr, mode=None):
        p = os.path.join(folder, fname)
        save_png(p, arr, mode)
        tex[key] = p

    put("basecolor", out_pbr, NAME + "_basecolor.png", col)
    put("normal_gl", out_pbr, NAME + "_normal_gl.png", nrm * 0.5 + 0.5)
    ndx = nrm.copy()
    ndx[..., 1] *= -1
    put("normal_dx", out_pbr, NAME + "_normal_dx.png", ndx * 0.5 + 0.5)
    put("roughness", out_pbr, NAME + "_roughness.png", rough)
    put("metallic", out_pbr, NAME + "_metallic.png", metal)
    put("ao", out_pbr, NAME + "_ao.png", ao)
    put("orm", out_pbr, NAME + "_orm.png", np.stack([ao, rough, metal], -1))
    put("emissive", out_pbr, NAME + "_emissive.png", emit)
    put("opacity", out_pbr, NAME + "_opacity.png", alpha)
    # GTA V helper set (legacy spec workflow)
    d = col * (0.30 + 0.70 * ao)[..., None]
    put("gta_d", out_gta, NAME + "_d.png", np.concatenate([d, alpha[..., None]], -1))
    put("gta_n", out_gta, NAME + "_n.png", ndx * 0.5 + 0.5)
    gloss = 1.0 - rough
    f0 = 0.04 * (1 - metal) + metal * col.mean(-1)
    spec_int = np.clip((f0 * 3.0 + 0.06) * (0.35 + 0.65 * gloss) * (0.4 + 0.6 * ao), 0, 1)
    put("gta_s", out_gta, NAME + "_s.png", np.stack([spec_int, gloss, spec_int], -1))
    log("textures written (%d)" % len(tex))

    # ---------------------------------------------------------------- Blender objects
    mats = make_materials(tex)
    coll = {k: cb.ensure_collection(k) for k in ("LOD0", "LOD1", "LOD2", "COLLISION", "PREVIEW")}
    objs = {}
    for lod in (0, 1, 2):
        ob = cb.to_object(meshes[lod], "%s_lod%d" % (NAME, lod), uvfn, ct_geo.REG, coll["LOD%d" % lod])
        for m in mats:
            ob.data.materials.append(m)
        ca = ob.data.color_attributes.new("Col", 'BYTE_COLOR', 'CORNER')
        ca.data.foreach_set("color", [1.0] * (4 * len(ca.data)))
        objs[lod] = ob
    colm = bpy.data.materials.new(NAME + "_col")
    colm.diffuse_color = (0.9, 0.2, 0.2, 0.5)
    obc = cb.to_object(col_mesh, NAME + "_col", lambda i, uv: (0.0, 0.0), ct_geo.REG, coll["COLLISION"])
    for p in obc.data.polygons:
        p.use_smooth = False
    obc.data.materials.append(colm)
    objs["col"] = obc
    stats = {}
    for k, ob in objs.items():
        tris = sum(len(p.vertices) - 2 for p in ob.data.polygons)
        stats[str(k)] = dict(tris=tris, verts=len(ob.data.vertices))
    log("tri counts:", {k: v["tris"] for k, v in stats.items()})

    # ---------------------------------------------------------------- export
    if not args.no_export:
        def export(path, objlist):
            for o in bpy.context.view_layer.objects:
                o.select_set(False)
            for o in objlist:
                o.hide_set(False)
                o.select_set(True)
            bpy.context.view_layer.objects.active = objlist[0]
            bpy.ops.export_scene.fbx(filepath=path, use_selection=True, object_types={'MESH'},
                                     axis_forward='Y', axis_up='Z', apply_unit_scale=True,
                                     apply_scale_options='FBX_SCALE_UNITS', bake_space_transform=False,
                                     mesh_smooth_type='FACE', use_mesh_modifiers=True, use_tspace=True,
                                     use_triangles=True, add_leaf_bones=False, path_mode='RELATIVE',
                                     embed_textures=False, use_custom_props=False, colors_type='SRGB')
            log("FBX:", os.path.relpath(path, ROOT), "%.0f kB" % (os.path.getsize(path) / 1024))
        export(os.path.join(out_fbx, NAME + ".fbx"), [objs[0], objs[1], objs[2], objs["col"]])
        export(os.path.join(out_fbx, NAME + "_lod0.fbx"), [objs[0]])
        export(os.path.join(out_fbx, NAME + "_lod1.fbx"), [objs[1]])
        export(os.path.join(out_fbx, NAME + "_lod2.fbx"), [objs[2]])
        export(os.path.join(out_fbx, NAME + "_col.fbx"), [objs["col"]])

    # ---------------------------------------------------------------- preview scene (+ .blend)
    preview_env(coll["PREVIEW"])
    cb.setup_cycles(samples=args.render_samples, res=(1600, 900))
    cams = {}
    cams["hero"] = cb.add_camera("cam_hero", (1.10, -1.25, 1.52), (0.02, -0.02, 0.76), lens=32)
    cams["top"] = cb.add_camera("cam_top", (0.0, 0.0, 3.0), (0.0, 0.0, 0.0), lens=50, ortho=1.62)
    cams["close_money"] = cb.add_camera("cam_money", (-0.18, -0.62, 1.08), (-0.40, 0.00, 0.78), lens=40)
    cams["close_gun"] = cb.add_camera("cam_gun", (0.72, -0.52, 1.05), (0.38, -0.05, 0.78), lens=45)
    cams["close_lines"] = cb.add_camera("cam_lines", (-0.05, -0.62, 1.00), (-0.14, -0.24, 0.78), lens=50)
    cams["back"] = cb.add_camera("cam_back", (-0.9, 1.5, 1.45), (0.05, 0.0, 0.75), lens=32)
    for c in cams.values():
        c.users_collection[0].objects.unlink(c)
        coll["PREVIEW"].objects.link(c)
    bpy.context.scene.camera = cams["hero"]

    def show_only(keys):
        for k, ob in objs.items():
            vis = k in keys
            ob.hide_render = not vis
            ob.hide_viewport = not vis

    show_only([0])
    if not args.no_export:
        # keep LOD1/LOD2/COL hidden in the .blend's render but visible in viewport list
        bp = os.path.join(out_blend, NAME + ".blend")
        for k, ob in objs.items():
            ob.hide_viewport = False
        bpy.ops.file.pack_all() if False else None
        bpy.ops.wm.save_as_mainfile(filepath=bp, relative_remap=True, compress=True)
        log("blend:", os.path.relpath(bp, ROOT))
        show_only([0])

    if not args.no_render:
        which = args.only_render.split(",") if args.only_render else list(cams.keys())
        for k in which:
            t = time.time()
            if k == "top":
                bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 1600, 920
            else:
                bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 1600, 900
            cb.render_to(os.path.join(out_prev, "render_%s.jpg" % k), cams[k])
            log("render %s %.0fs" % (k, time.time() - t))
        if not args.only_render:
            # LOD comparison: move copies side by side
            lodcam = cb.add_camera("cam_lods", (3.0, -7.4, 3.3), (3.0, 0.0, 0.45), lens=30)
            for k, dx in ((0, 0.0), (1, 2.0), (2, 4.0), ("col", 6.0)):
                objs[k].location.x = dx
                objs[k].hide_render = False
            bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = 2400, 760
            fl = bpy.data.objects.get("pv_floor")
            fl.scale = (2.0, 1.0, 1.0)
            key = bpy.data.objects.get("pv_key")
            kloc = tuple(key.location)
            key.location = (3.0, -1.0, 3.2)
            key.data.size = 5.0
            key.data.energy = 900
            cb.look_at(key, (3.0, 0.0, 0.8))
            lp = os.path.join(out_prev, "render_lods.jpg")
            cb.render_to(lp, lodcam)
            im = Image.open(lp).convert("RGB")
            dd = ImageDraw.Draw(im)
            fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30) if os.path.exists(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") else ImageFont.load_default()
            for k, lab in ((0, "LOD0  %d tris" % stats["0"]["tris"]), (1, "LOD1  %d tris" % stats["1"]["tris"]),
                           (2, "LOD2  %d tris" % stats["2"]["tris"]), (3, "COLLISION  %d tris" % stats["col"]["tris"])):
                dd.text((330 + k * 590, 40), lab, font=fnt, fill=(255, 225, 150), anchor="mm")
            im.save(lp, quality=90, optimize=True, progressive=True)
            key.location = kloc
            key.data.size = 0.55
            key.data.energy = 55
            for k in objs:
                objs[k].location.x = 0.0
            fl.scale = (1, 1, 1)

    # ---------------------------------------------------------------- report
    rep = dict(name=NAME, atlas=ct_pack.ATLAS, density_px_per_m=round(s, 1), islands=len(placed),
               uv_outside=bad, objects=stats, materials=[m.name for m in mats],
               time_s=round(time.time() - T0, 1))
    with open(os.path.join(cache, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    log(json.dumps(rep))
    # UV layout overlay, texture contact sheet, ytyp helper
    uv_layout(meshes[0], uvfn, tex["basecolor"], os.path.join(out_prev, "uv_layout_lod0.png"))
    texture_sheet(tex, os.path.join(out_prev, "textures_sheet.png"))
    write_ytyp(meshes[0], os.path.join(ROOT, "gta", NAME + ".ytyp.xml"))


def texture_sheet(tex, out):
    items = [("BaseColor", "basecolor"), ("Normal (OpenGL)", "normal_gl"), ("Roughness", "roughness"),
             ("Metallic", "metallic"), ("Ambient Occlusion", "ao"), ("Emissive", "emissive"), ("Opacity", "opacity"),
             ("ORM (R=AO G=Rough B=Metal)", "orm")]
    sz = 480
    cols = 4
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * sz + (cols + 1) * 12, rows * (sz + 40) + 12), (24, 24, 26))
    d = ImageDraw.Draw(sheet)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        fnt = ImageFont.load_default()
    for i, (lab, key) in enumerate(items):
        r, c = divmod(i, cols)
        x = 12 + c * (sz + 12)
        y = 12 + r * (sz + 40)
        im = Image.open(tex[key]).convert("RGB").resize((sz, sz), Image.LANCZOS)
        sheet.paste(im, (x, y + 28))
        d.text((x, y + 2), lab, font=fnt, fill=(235, 235, 235))
    sheet.save(out, optimize=True)


def write_ytyp(md, out):
    """CodeWalker-style CMapTypes XML (import in CodeWalker -> save as .ytyp)."""
    xs = [p[0] for p in md.v]
    ys = [p[1] for p in md.v]
    zs = [p[2] for p in md.v]
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    c = tuple((a + b) / 2 for a, b in zip(mn, mx))
    r = math.sqrt(sum(((b - a) / 2) ** 2 for a, b in zip(mn, mx)))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f8 = lambda v: "%.8f" % v
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<CMapTypes>
 <extensions/>
 <archetypes>
  <Item type="CBaseArchetypeDef">
   <lodDist value="{f8(120.0)}"/>
   <flags value="32"/>
   <specialAttribute value="0"/>
   <bbMin x="{f8(mn[0])}" y="{f8(mn[1])}" z="{f8(mn[2])}"/>
   <bbMax x="{f8(mx[0])}" y="{f8(mx[1])}" z="{f8(mx[2])}"/>
   <bsCentre x="{f8(c[0])}" y="{f8(c[1])}" z="{f8(c[2])}"/>
   <bsRadius value="{f8(r)}"/>
   <hdTextureDist value="{f8(20.0)}"/>
   <name>{NAME}</name>
   <textureDictionary>{NAME}</textureDictionary>
   <clipDictionary/>
   <drawableDictionary/>
   <physicsDictionary/>
   <assetType>ASSET_TYPE_DRAWABLE</assetType>
   <assetName>{NAME}</assetName>
   <extensions/>
  </Item>
 </archetypes>
 <name>{NAME}</name>
 <dependencies/>
 <compositeEntityTypes/>
</CMapTypes>
"""
    with open(out, "w") as fh:
        fh.write(xml)
    log("ytyp:", os.path.relpath(out, ROOT), "bb", [round(v, 3) for v in mn], [round(v, 3) for v in mx],
        "r=%.3f" % r)


def uv_layout(md, uvfn, base_path, out):
    base = Image.open(base_path).convert("RGB").resize((2048, 2048))
    base = Image.blend(base, Image.new("RGB", base.size, (0, 0, 0)), 0.45)
    d = ImageDraw.Draw(base)
    for (idx, uvs, iid, mat, sm) in md.f:
        pts = []
        for t in uvs:
            u, v = uvfn(iid, t)
            pts.append((u * 2048, (1 - v) * 2048))
        colr = {"main": (80, 255, 140), "emissive": (255, 200, 60), "glass": (120, 190, 255)}[mat]
        d.line(pts + [pts[0]], fill=colr, width=1)
    base.save(out, optimize=True)


if __name__ == "__main__":
    main()
