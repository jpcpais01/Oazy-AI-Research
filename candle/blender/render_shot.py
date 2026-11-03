"""Stage 4: build the candle scene in Blender and path-trace the shot with Cycles.

Run with Blender (recommended, any 5.2.x build):
    blender -b --factory-startup --python candle/blender/render_shot.py -- --work WORK --quality final
or with the `bpy` Python module (pip install bpy==5.2.3):
    python candle/blender/render_shot.py --work WORK --quality preview

Frames already on disk are skipped, so the command can be re-run after any
interruption.  `--workers N --worker i` renders every N-th frame on GPU i (one
process per GPU is the fastest way to use a multi-GPU pod).

Scene: a 52 mm ivory pillar candle (random-walk subsurface scattering) on dark
slate, with the simulated flame as an emissive/absorbing/scattering VDB volume,
three moving point-light proxies that carry the flame's measured light into the
scene, a clear molten-wax pool (Fresnel gloss over translucent wax), the wax
film / frozen drips displaced into the candle mesh, a charred wick with a
blackbody ember, out-of-focus practical lights for bokeh, and a camera move with
real depth of field (7-blade aperture) and motion blur.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bpy  # noqa: E402
import addon_utils  # noqa: E402

from candle import config as C  # noqa: E402
from candle import timeline as TL  # noqa: E402
from candle import spectral  # noqa: E402
from candle.look import LOOK  # noqa: E402

GEOM = C.CandleGeometry()


def log(*a):
    print("[render]", *a, flush=True)


# =========================================================================== #
# Small node helpers
# =========================================================================== #
def new_material(name):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    return m, nt, out


def link(nt, a, b):
    nt.links.new(a, b)


def node(nt, kind, loc=(0, 0), **inputs):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in inputs.items():
        n.inputs[k].default_value = v
    return n


def math_node(nt, op, a=None, b=None, loc=(0, 0), clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.location = loc
    n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n


def set_input(n, names, value):
    for nm in names:
        if nm in n.inputs:
            n.inputs[nm].default_value = value
            return True
    return False


# =========================================================================== #
# Materials
# =========================================================================== #
def mat_wax():
    """Ivory paraffin: random-walk SSS, glossy wet film and hot liquid driven by attributes."""
    m, nt, out = new_material("Wax")
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", (200, 0))
    set_input(bsdf, ["Base Color"], (0.93, 0.86, 0.72, 1))
    set_input(bsdf, ["Subsurface Weight"], 1.0)
    set_input(bsdf, ["Subsurface Radius"], (1.0, 0.62, 0.36))
    set_input(bsdf, ["Subsurface Scale"], 0.0045)        # metres: ~4.5 mm mean free path in red
    set_input(bsdf, ["Subsurface IOR"], 1.44)
    set_input(bsdf, ["Subsurface Anisotropy"], 0.35)
    set_input(bsdf, ["IOR"], 1.44)
    try:
        bsdf.subsurface_method = "RANDOM_WALK"
    except Exception:
        pass
    # film thickness (mm) and heat from the wax solver
    a_film = node(nt, "ShaderNodeAttribute", (-800, 200))
    a_film.attribute_name = "film"
    a_hot = node(nt, "ShaderNodeAttribute", (-800, 0))
    a_hot.attribute_name = "hot"
    a_wet = node(nt, "ShaderNodeAttribute", (-800, -200))
    a_wet.attribute_name = "wet"
    wetness = math_node(nt, "MAXIMUM", math_node(nt, "MULTIPLY", a_film.outputs["Fac"], 12.0, (-600, 200), clamp=True).outputs[0],
                        a_wet.outputs["Fac"], (-400, 150))
    # roughness: frosty old wax 0.42, liquid film 0.04
    rough = math_node(nt, "MULTIPLY", wetness.outputs[0], -0.38, (-200, 260))
    rough = math_node(nt, "ADD", rough.outputs[0], 0.42, (-50, 260))
    link(nt, rough.outputs[0], bsdf.inputs["Roughness"])
    # glossy coat on liquid wax
    if "Coat Weight" in bsdf.inputs:
        cw = math_node(nt, "MULTIPLY", wetness.outputs[0], 0.9, (-200, 120))
        link(nt, cw.outputs[0], bsdf.inputs["Coat Weight"])
        set_input(bsdf, ["Coat Roughness"], 0.03)
        set_input(bsdf, ["Coat IOR"], 1.44)
    # hot liquid wax is clearer: longer scattering radius, slightly warmer
    sc = math_node(nt, "MULTIPLY", a_hot.outputs["Fac"], 0.006, (-200, -40))
    sc = math_node(nt, "ADD", sc.outputs[0], 0.0045, (-50, -40))
    link(nt, sc.outputs[0], bsdf.inputs["Subsurface Scale"])
    # subtle surface micro-relief
    tc = node(nt, "ShaderNodeTexCoord", (-1100, -300))
    nz = node(nt, "ShaderNodeTexNoise", (-900, -350))
    set_input(nz, ["Scale"], 900.0)
    set_input(nz, ["Detail"], 6.0)
    set_input(nz, ["Roughness"], 0.6)
    link(nt, tc.outputs["Object"], nz.inputs["Vector"])
    bump = node(nt, "ShaderNodeBump", (-100, -300))
    set_input(bump, ["Strength"], 0.12)
    set_input(bump, ["Distance"], 0.00015)
    dry = math_node(nt, "SUBTRACT", 1.0, wetness.outputs[0], (-300, -380), clamp=True)
    hb = math_node(nt, "MULTIPLY", nz.outputs["Fac"], dry.outputs[0], (-300, -300))
    link(nt, hb.outputs[0], bump.inputs["Height"])
    link(nt, bump.outputs["Normal"], bsdf.inputs["Normal"])
    link(nt, bsdf.outputs[0], out.inputs["Surface"])
    return m


def mat_pool():
    """Clear molten paraffin: Fresnel gloss over transparency (lets the flame light reach the floor)."""
    m, nt, out = new_material("MoltenWax")
    fres = node(nt, "ShaderNodeFresnel", (-300, 200), IOR=1.44)
    gl = node(nt, "ShaderNodeBsdfGlossy", (-300, 0))
    set_input(gl, ["Roughness"], 0.012)
    set_input(gl, ["Color"], (1, 1, 1, 1))
    tr = node(nt, "ShaderNodeBsdfTransparent", (-300, -150))
    set_input(tr, ["Color"], (1.0, 0.975, 0.92, 1))
    mix = node(nt, "ShaderNodeMixShader", (100, 0))
    link(nt, fres.outputs[0], mix.inputs[0])
    link(nt, tr.outputs[0], mix.inputs[1])
    link(nt, gl.outputs[0], mix.inputs[2])
    # gentle ripples from the convection under the flame
    tc = node(nt, "ShaderNodeTexCoord", (-900, -300))
    nz = node(nt, "ShaderNodeTexNoise", (-700, -300))
    nz.noise_dimensions = "4D"
    set_input(nz, ["Scale"], 140.0)
    set_input(nz, ["Detail"], 2.0)
    m.node_tree["noise_node"] = nz.name
    link(nt, tc.outputs["Object"], nz.inputs["Vector"])
    bump = node(nt, "ShaderNodeBump", (-500, -300))
    set_input(bump, ["Strength"], 0.08)
    set_input(bump, ["Distance"], 0.00005)
    link(nt, nz.outputs["Fac"], bump.inputs["Height"])
    link(nt, bump.outputs["Normal"], gl.inputs["Normal"])
    link(nt, bump.outputs["Normal"], fres.inputs["Normal"])
    link(nt, mix.outputs[0], out.inputs["Surface"])
    return m


def mat_wick():
    m, nt, out = new_material("Wick")
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", (0, 0))
    set_input(bsdf, ["Base Color"], (0.018, 0.014, 0.012, 1))
    set_input(bsdf, ["Roughness"], 0.85)
    a_tip = node(nt, "ShaderNodeAttribute", (-700, -200))
    a_tip.attribute_name = "tip"
    bb = node(nt, "ShaderNodeBlackbody", (-500, -350))
    bb.name = "EmberColor"
    set_input(bb, ["Temperature"], 1050.0)
    strength = node(nt, "ShaderNodeValue", (-700, -450))
    strength.name = "EmberStrength"
    strength.outputs[0].default_value = 0.0
    # glow concentrated at the very tip, with charred texture
    nz = node(nt, "ShaderNodeTexNoise", (-900, -50))
    set_input(nz, ["Scale"], 2500.0)
    set_input(nz, ["Detail"], 4.0)
    tipn = math_node(nt, "MULTIPLY", a_tip.outputs["Fac"], nz.outputs["Fac"], (-400, -150))
    tipn = math_node(nt, "POWER", tipn.outputs[0], 1.6, (-250, -150))
    s = math_node(nt, "MULTIPLY", tipn.outputs[0], strength.outputs[0], (-100, -300))
    link(nt, bb.outputs[0], bsdf.inputs["Emission Color"])
    link(nt, s.outputs[0], bsdf.inputs["Emission Strength"])
    link(nt, bsdf.outputs[0], out.inputs["Surface"])
    return m


def mat_table():
    """Dark honed slate with faint mineral variation and a soft sheen."""
    m, nt, out = new_material("Slate")
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", (300, 0))
    tc = node(nt, "ShaderNodeTexCoord", (-1000, 0))
    mp = node(nt, "ShaderNodeMapping", (-800, 0))
    set_input(mp, ["Scale"], (1.0, 3.0, 1.0))
    link(nt, tc.outputs["Object"], mp.inputs["Vector"])
    n1 = node(nt, "ShaderNodeTexNoise", (-600, 100))
    set_input(n1, ["Scale"], 9.0)
    set_input(n1, ["Detail"], 12.0)
    set_input(n1, ["Roughness"], 0.62)
    link(nt, mp.outputs[0], n1.inputs["Vector"])
    ramp = node(nt, "ShaderNodeValToRGB", (-350, 100))
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (0.010, 0.011, 0.013, 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (0.030, 0.029, 0.028, 1)
    link(nt, n1.outputs["Fac"], ramp.inputs["Fac"])
    link(nt, ramp.outputs["Color"], bsdf.inputs["Base Color"])
    n2 = node(nt, "ShaderNodeTexNoise", (-600, -200))
    set_input(n2, ["Scale"], 180.0)
    set_input(n2, ["Detail"], 8.0)
    link(nt, tc.outputs["Object"], n2.inputs["Vector"])
    r = math_node(nt, "MULTIPLY", n2.outputs["Fac"], 0.25, (-350, -150))
    r = math_node(nt, "ADD", r.outputs[0], 0.28, (-200, -150))
    link(nt, r.outputs[0], bsdf.inputs["Roughness"])
    bump = node(nt, "ShaderNodeBump", (0, -300))
    set_input(bump, ["Strength"], 0.25)
    set_input(bump, ["Distance"], 0.0004)
    link(nt, n2.outputs["Fac"], bump.inputs["Height"])
    link(nt, bump.outputs["Normal"], bsdf.inputs["Normal"])
    set_input(bsdf, ["Specular IOR Level"], 0.45)
    link(nt, bsdf.outputs[0], out.inputs["Surface"])
    return m


def mat_wall():
    m, nt, out = new_material("Wall")
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", (0, 0))
    set_input(bsdf, ["Base Color"], (0.045, 0.038, 0.032, 1))
    set_input(bsdf, ["Roughness"], 0.95)
    link(nt, bsdf.outputs[0], out.inputs["Surface"])
    return m


def mat_bokeh(color, strength):
    m, nt, out = new_material("Bokeh")
    em = node(nt, "ShaderNodeEmission", (0, 0))
    set_input(em, ["Color"], (*color, 1))
    set_input(em, ["Strength"], strength)
    link(nt, em.outputs[0], out.inputs["Surface"])
    return m


def mat_flame():
    """Emission (camera/glossy/transmission rays only; diffuse light comes from the proxies),
    white forward-scattering wax smoke, black absorbing soot."""
    m, nt, out = new_material("Flame")
    a_em = node(nt, "ShaderNodeAttribute", (-900, 300))
    a_em.attribute_name = "emission"
    try:
        a_em.attribute_type = "GEOMETRY"
    except Exception:
        pass
    a_sm = node(nt, "ShaderNodeAttribute", (-900, 0))
    a_sm.attribute_name = "smoke"
    a_so = node(nt, "ShaderNodeAttribute", (-900, -250))
    a_so.attribute_name = "soot"
    lp = node(nt, "ShaderNodeLightPath", (-900, 600))
    vis = math_node(nt, "MAXIMUM", lp.outputs["Is Camera Ray"], lp.outputs["Is Glossy Ray"], (-650, 600))
    vis = math_node(nt, "MAXIMUM", vis.outputs[0], lp.outputs["Is Transmission Ray"], (-500, 600))
    vis.name = "EmissionVisibility"
    em_scale = node(nt, "ShaderNodeVectorMath", (-450, 300))
    em_scale.operation = "SCALE"
    link(nt, a_em.outputs["Vector"], em_scale.inputs[0])
    link(nt, vis.outputs[0], em_scale.inputs["Scale"])
    pv = node(nt, "ShaderNodeVolumePrincipled", (0, 100))
    set_input(pv, ["Color"], (0.92, 0.93, 0.96, 1))
    set_input(pv, ["Anisotropy"], LOOK["smoke_anisotropy"])
    set_input(pv, ["Absorption Color"], (0.0, 0.0, 0.0, 1))
    dens = math_node(nt, "MULTIPLY", a_sm.outputs["Fac"], LOOK["smoke_density"], (-450, 0))
    link(nt, dens.outputs[0], pv.inputs["Density"])
    pv.inputs["Density Attribute"].default_value = ""
    set_input(pv, ["Emission Strength"], 1.0)
    link(nt, em_scale.outputs[0], pv.inputs["Emission Color"])
    ab = node(nt, "ShaderNodeVolumeAbsorption", (0, -250))
    set_input(ab, ["Color"], (0.0, 0.0, 0.0, 1))
    # soot absorbs what the camera sees, but must not shadow the flame's own light
    # proxies (their power already is the light that escapes the flame)
    not_shadow = math_node(nt, "SUBTRACT", 1.0, lp.outputs["Is Shadow Ray"], (-650, -400))
    sd = math_node(nt, "MULTIPLY", a_so.outputs["Fac"], LOOK["soot_absorption"], (-450, -250))
    sd = math_node(nt, "MULTIPLY", sd.outputs[0], not_shadow.outputs[0], (-300, -250))
    link(nt, sd.outputs[0], ab.inputs["Density"])
    add = node(nt, "ShaderNodeAddShader", (300, 0))
    link(nt, pv.outputs[0], add.inputs[0])
    link(nt, ab.outputs[0], add.inputs[1])
    link(nt, add.outputs[0], out.inputs["Volume"])
    return m


# =========================================================================== #
# Geometry: candle, pool, wick
# =========================================================================== #
def _smooth2(a, passes=1):
    for _ in range(passes):
        a = (np.roll(a, 1, 0) + 2 * a + np.roll(a, -1, 0)) * 0.25
        b = a.copy()
        b[:, 1:-1] = (a[:, :-2] + 2 * a[:, 1:-1] + a[:, 2:]) * 0.25
        a = b
    return a


class CandleBuilder:
    """Closed lathe-like mesh: bottom cap, side wall (with drips), rim, inner wall, pool floor."""

    def __init__(self, wax0):
        self.n = wax0["dep"].shape[0]
        self.nz = wax0["dep"].shape[1]
        self.z_top = float(wax0["z_top"][0])
        self.dz = float(wax0["dz"][0])
        self.n_cap = 6
        self.n_rim = 14
        self.n_inner = 14
        self.n_floor = 18
        self.theta = np.arange(self.n) * 2 * np.pi / self.n
        self.rings = self.n_cap + self.nz + 1 + self.n_rim + self.n_inner + self.n_floor

    def profile(self, w):
        g = GEOM
        n, nz = self.n, self.nz
        R = g.radius
        dep = _smooth2(w["dep"].astype(np.float64), 1)
        h = _smooth2(w["h"].astype(np.float64), 1)
        table = w["table"].astype(np.float64) if "table" in w else np.zeros(n)
        table = np.convolve(np.r_[table[-8:], table, table[:8]], np.ones(9) / 9, "same")[8:-8]
        rim = w["rim"].astype(np.float64)
        rim = np.convolve(np.r_[rim[-4:], rim, rim[:4]], np.ones(5) / 5, "same")[4:-4]
        L = float(w["L"][0])
        wet = w["wet"].astype(np.float64)
        T = w["T"].astype(np.float64)

        rr, zz, film, hot, wetv = [], [], [], [], []
        # --- side rows
        zs = (np.arange(nz) + 0.5) * self.dz
        zs[0] = 0.0
        foot_h = 0.0016
        foot = (table[:, None] / foot_h) * np.exp(-zs[None, :] / foot_h)
        r_side = R + dep + h + np.minimum(foot, 0.006)
        # --- bottom cap
        for f in np.linspace(0.0, 1.0, self.n_cap + 1)[1:]:
            rr.append(r_side[:, 0] * f * 0.999)
            zz.append(np.zeros(n))
            film.append(np.zeros(n)); hot.append(np.zeros(n)); wetv.append(np.zeros(n))
        for k in range(nz):
            rr.append(r_side[:, k])
            zz.append(np.full(n, zs[k]))
            film.append(h[:, k] * 1e3)
            hot.append(np.clip((T[:, k] - 55.0) / 25.0, 0, 1) * (h[:, k] > 2e-5))
            wetv.append(np.zeros(n))
        # top of side wall
        r_top = r_side[:, -1]
        rr.append(r_top); zz.append(np.full(n, self.z_top))
        film.append(h[:, -1] * 1e3); hot.append(hot[-1]); wetv.append(wet * 0.5)
        # --- outer rim rounding to the crest
        r_crest = R - 0.0011
        crest = np.maximum(rim, self.z_top + 0.0003)
        for phi in np.linspace(0, np.pi / 2, self.n_rim + 1)[1:]:
            rr.append(r_crest + (r_top - r_crest) * np.cos(phi))
            zz.append(self.z_top + (crest - self.z_top) * np.sin(phi))
            film.append(np.zeros(n)); hot.append(wet * 0.8); wetv.append(wet)
        # --- inner wall down to the liquid line
        r_p = g.pool_radius
        z_contact = np.minimum(L - 0.0003, crest - 0.0003)
        for phi in np.linspace(0, np.pi / 2, self.n_inner + 1)[1:]:
            rr.append(r_crest - (r_crest - r_p) * np.sin(phi))
            zz.append(crest - (crest - z_contact) * (1 - np.cos(phi)))
            film.append(np.zeros(n)); hot.append(np.full(n, 0.6)); wetv.append(np.full(n, 0.0))
        # --- pool floor (under the clear liquid)
        for f in np.linspace(1.0, 0.0, self.n_floor + 1)[1:]:
            r = r_p * f
            zf = L - 0.0003 - (g.pool_depth - 0.0003) * (1 - f ** 2) ** 0.6
            rr.append(np.full(n, r) if f > 0 else np.zeros(n))
            zz.append(np.full(n, zf))
            film.append(np.zeros(n)); hot.append(np.full(n, 1.0)); wetv.append(np.zeros(n))
        rr = np.array([np.broadcast_to(np.asarray(a, float), (n,)) for a in rr])     # (rings, n)
        zz = np.array([np.broadcast_to(np.asarray(a, float), (n,)) for a in zz])
        film = [np.broadcast_to(np.asarray(a, float), (n,)) for a in film]
        hot = [np.broadcast_to(np.asarray(a, float), (n,)) for a in hot]
        wetv = [np.broadcast_to(np.asarray(a, float), (n,)) for a in wetv]
        x = rr * np.cos(self.theta)[None, :]
        y = rr * np.sin(self.theta)[None, :]
        co = np.stack([x, y, zz], -1)
        bottom_pole = np.array([[0.0, 0.0, 0.0]])
        top_pole = np.array([[0.0, 0.0, float(zz[-1, 0])]])
        verts = np.concatenate([bottom_pole, co.reshape(-1, 3), top_pole])
        attrs = {k: np.concatenate([[0.0], np.array(v).reshape(-1), [v[-1][0]]])
                 for k, v in (("film", film), ("hot", hot), ("wet", wetv))}
        return verts.astype(np.float32), attrs

    def faces(self):
        n = self.n
        R = self.rings
        faces = []
        ring = lambda r: 1 + r * n
        for i in range(n):
            faces.append((0, ring(0) + (i + 1) % n, ring(0) + i))
        for r in range(R - 1):
            a, b = ring(r), ring(r + 1)
            for i in range(n):
                j = (i + 1) % n
                faces.append((a + i, a + j, b + j, b + i))
        top = 1 + R * n
        for i in range(n):
            faces.append((ring(R - 1) + i, ring(R - 1) + (i + 1) % n, top))
        return faces


def build_mesh_object(name, verts, faces, mat, attrs=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces)
    me.update()
    if smooth:
        me.shade_smooth()
    for k, v in (attrs or {}).items():
        at = me.attributes.new(k, "FLOAT", "POINT")
        at.data.foreach_set("value", np.asarray(v, np.float32))
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def update_mesh(ob, verts, attrs=None):
    me = ob.data
    me.vertices.foreach_set("co", verts.reshape(-1).astype(np.float32))
    for k, v in (attrs or {}).items():
        me.attributes[k].data.foreach_set("value", np.asarray(v, np.float32))
    me.update()


def pool_geometry(L, rings=40, n=128):
    """Liquid surface disc with a meniscus at the wall and around the wick."""
    g = GEOM
    rmax = g.pool_radius + 0.0005
    rs = np.linspace(0.0, rmax, rings + 1)[1:]
    th = np.arange(n) * 2 * np.pi / n
    verts = [[0.0, 0.0, L + 0.00035]]
    for r in rs:
        men_wall = 0.00030 * np.exp(-(rmax - r) / 0.0006)
        men_wick = 0.00035 * np.exp(-max(r - g.wick_radius, 0) / 0.0005)
        z = L + men_wall + men_wick
        for t in th:
            verts.append([r * math.cos(t), r * math.sin(t), z])
    verts = np.array(verts, np.float32)
    faces = [(0, 1 + (i + 1) % n, 1 + i) for i in range(n)][::-1]
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    for k in range(rings - 1):
        a, b = 1 + k * n, 1 + (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append((a + i, b + i, b + j, a + j))
    return verts, faces


def wick_geometry(n_around=24, n_len=60):
    """Charred cotton wick: tapered tube following the same curl as the flame solver's wick."""
    g = GEOM
    base = g.pool_level - g.pool_depth + 0.0008
    top = g.pool_level + g.wick_height
    zs = np.linspace(base, top, n_len)
    verts, tip = [], []
    th = np.arange(n_around) * 2 * np.pi / n_around
    for z in zs:
        zc = np.clip(z - g.pool_level, 0, g.wick_height)
        cy = g.wick_lean * (zc / g.wick_height) ** 2
        # slight braid twist and taper at the very top
        r = g.wick_radius * (1.0 - 0.35 * max(0.0, (z - (top - 0.0015)) / 0.0015))
        for t in th:
            rb = r * (1 + 0.08 * math.sin(3 * t + z * 4000))
            verts.append([rb * math.cos(t), cy + rb * math.sin(t), z])
            tip.append(float(np.clip(1.0 - (top - z) / 0.0032, 0, 1)))
    verts.append([0.0, g.wick_lean, top + 0.00025])
    tip.append(1.0)
    verts = np.array(verts, np.float32)
    faces = []
    for k in range(n_len - 1):
        a, b = k * n_around, (k + 1) * n_around
        for i in range(n_around):
            j = (i + 1) % n_around
            faces.append((a + i, a + j, b + j, b + i))
    top_i = len(verts) - 1
    a = (n_len - 1) * n_around
    for i in range(n_around):
        faces.append((a + i, a + (i + 1) % n_around, top_i))
    return verts, faces, np.array(tip, np.float32)


# =========================================================================== #
# Camera choreography
# =========================================================================== #
def catmull(keys, f):
    """Catmull-Rom through (frame, value-array) keys, clamped at the ends."""
    fs = [k[0] for k in keys]
    if f <= fs[0]:
        return np.array(keys[0][1], float)
    if f >= fs[-1]:
        return np.array(keys[-1][1], float)
    i = max(j for j in range(len(fs) - 1) if fs[j] <= f)
    p1, p2 = np.array(keys[i][1], float), np.array(keys[i + 1][1], float)
    p0 = np.array(keys[i - 1][1], float) if i > 0 else p1
    p3 = np.array(keys[i + 2][1], float) if i + 2 < len(keys) else p2
    t = (f - fs[i]) / (fs[i + 1] - fs[i])
    # ease each segment so that keys are passed smoothly
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def camera_track(work, n_frames):
    """Per-frame camera position, look-at target, focal length, f-stop and focus point."""
    g = GEOM
    flame_c = np.array([0.0, 0.001, g.pool_level + 0.018])
    wick_tip = np.array([0.0, g.wick_lean, g.pool_level + g.wick_height])
    # drip front from the wax solver (theta, z), smoothed
    fronts = {}
    wl = os.path.join(work, "wax", "wax_log.json")
    if os.path.exists(wl):
        for e in json.load(open(wl)):
            if e.get("front"):
                fronts[e["frame"]] = e["front"]
    th_n = g.notch_azimuth
    notch = np.array([g.radius * math.cos(th_n), g.radius * math.sin(th_n), g.rim_height - 0.002])

    def front_point(f):
        ks = sorted(k for k in fronts if abs(k - f) <= 6)
        if not ks:
            return None
        th = np.mean([fronts[k][0] for k in ks])
        z = np.mean([fronts[k][1] for k in ks])
        return np.array([g.radius * math.cos(th), g.radius * math.sin(th), z])

    deg = math.radians
    # (frame, [azimuth, elevation, distance, lens, fstop])
    rig = [
        (0, [deg(-101), deg(7.0), 0.52, 50.0, 2.8]),
        (70, [deg(-93), deg(4.5), 0.33, 50.0, 2.8]),
        (112, [deg(-86), deg(3.0), 0.21, 65.0, 2.8]),
        (166, [deg(-74), deg(2.0), 0.19, 65.0, 2.8]),
        (196, [deg(-54), deg(10.0), 0.20, 70.0, 7.1]),
        (250, [deg(-48), deg(1.0), 0.19, 70.0, 7.1]),
        (276, [deg(-78), deg(2.5), 0.21, 65.0, 2.8]),
        (300, [deg(-82), deg(1.0), 0.22, 65.0, 2.8]),
        (340, [deg(-86), deg(-3.0), 0.27, 55.0, 2.8]),
        (383, [deg(-90), deg(-5.0), 0.33, 50.0, 2.8]),
    ]
    out = []
    smoke_pt = np.array([0.0, g.wick_lean, g.pool_level + 0.03])
    for f in range(n_frames):
        az, el, dist, lens, fstop = catmull(rig, f)
        # look-at target blends between story subjects
        fp = front_point(f)
        drip_w = TL._smooth((f - 172) / 22.0) * (1 - TL._smooth((f - 252) / 22.0))
        smoke_w = TL._smooth((f - (TL.BLOW_F + 8)) / 70.0)
        tgt_flame = flame_c * 0.75 + wick_tip * 0.25
        tgt_drip = notch if fp is None else notch * 0.35 + fp * 0.65
        rise = g.pool_level + 0.022 + 0.045 * TL._smooth((f - TL.BLOW_F) / (n_frames - TL.BLOW_F))
        tgt_smoke = np.array([0.0, g.wick_lean, rise])
        target = tgt_flame * (1 - drip_w) + tgt_drip * drip_w
        target = target * (1 - smoke_w) + tgt_smoke * smoke_w
        # opening frames sit a little lower to show the whole candle
        target[2] -= 0.022 * (1 - TL._smooth(f / 90.0))
        cam = target + dist * np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
        # focus: wick/flame, drip front, then the smoke ribbon just above the wick
        foc_flame = wick_tip + np.array([0, 0, 0.004])
        foc_drip = tgt_drip if fp is None else fp
        foc_smoke = np.array([0.0, g.wick_lean, g.pool_level + 0.02 + 0.05 * TL._smooth((f - TL.BLOW_F) / 80.0)])
        foc = foc_flame * (1 - drip_w) + foc_drip * drip_w
        foc = foc * (1 - smoke_w) + foc_smoke * smoke_w
        out.append(dict(cam=cam, target=target, lens=float(lens), fstop=float(fstop), focus=foc))
    # gentle hand-held drift (sub-degree, low frequency)
    rng = np.random.default_rng(3)
    ph = rng.uniform(0, 2 * np.pi, 6)
    for f, o in enumerate(out):
        t = f / TL.FPS
        o["shake"] = np.array([
            0.0016 * math.sin(0.9 * t + ph[0]) + 0.0007 * math.sin(2.3 * t + ph[1]),
            0.0014 * math.sin(0.7 * t + ph[2]) + 0.0006 * math.sin(2.9 * t + ph[3]),
            0.0012 * math.sin(0.5 * t + ph[4]) + 0.0004 * math.sin(3.7 * t + ph[5]),
        ])
    return out


def look_at_quat(cam, target, roll=0.0):
    from mathutils import Vector
    d = Vector(target) - Vector(cam)
    q = d.to_track_quat("-Z", "Y")
    if roll:
        from mathutils import Quaternion
        q = q @ Quaternion((0, 0, 1), roll)
    return q


# =========================================================================== #
# Scene
# =========================================================================== #
class Shot:
    def __init__(self, work, quality, device, worker, workers, n_frames):
        self.work = work
        self.quality = quality
        _, self.wax_preset, self.rp = C.get_preset(quality)
        self.n_frames = n_frames
        self.worker = worker
        self.workers = workers
        self.device = device
        self.out_dir = os.path.join(work, "frames")
        os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def setup_render(self):
        sc = bpy.context.scene
        rp = self.rp
        sc.render.engine = "CYCLES"
        sc.unit_settings.system = "METRIC"
        sc.render.resolution_x, sc.render.resolution_y = rp.resolution
        sc.render.resolution_percentage = 100
        sc.render.fps = TL.FPS
        sc.frame_start = 0
        sc.frame_end = self.n_frames - 1
        cy = sc.cycles
        cy.samples = rp.samples
        cy.use_adaptive_sampling = True
        cy.adaptive_threshold = rp.adaptive_threshold
        cy.adaptive_min_samples = min(32, rp.samples)
        cy.use_denoising = rp.denoise
        cy.max_bounces = rp.max_bounces
        cy.diffuse_bounces = 4
        cy.glossy_bounces = 4
        cy.transmission_bounces = rp.max_bounces
        cy.transparent_max_bounces = 32
        cy.volume_bounces = rp.volume_bounces
        cy.caustics_reflective = False
        cy.caustics_refractive = False
        cy.blur_glossy = 0.6
        cy.sample_clamp_indirect = 8.0
        cy.use_light_tree = True
        cy.volume_step_rate = rp.volume_step_rate
        if hasattr(cy, "volume_max_steps"):
            cy.volume_max_steps = 2048
        sc.render.film_transparent = False
        cy.seed = 0
        cy.use_animated_seed = True
        cy.filter_width = 1.5
        sc.render.use_motion_blur = rp.motion_blur
        sc.render.motion_blur_shutter = 0.5
        sc.render.use_persistent_data = True
        sc.view_settings.view_transform = "AgX"
        try:
            sc.view_settings.look = LOOK["look"]
        except Exception:
            pass
        sc.view_settings.exposure = LOOK["exposure_burning"]
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGB"
        sc.render.image_settings.color_depth = rp.png_depth
        sc.render.image_settings.compression = 15
        self._setup_device()

    def _setup_device(self):
        sc = bpy.context.scene
        want = (self.device or "AUTO").upper()
        prefs = bpy.context.preferences.addons["cycles"].preferences
        chosen = None
        if want != "CPU":
            order = ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"] if want == "AUTO" else [want]
            for kind in order:
                try:
                    prefs.compute_device_type = kind
                    prefs.refresh_devices()
                    devs = [d for d in prefs.devices if d.type == kind]
                except Exception:
                    devs = []
                if devs:
                    chosen = kind
                    for d in prefs.devices:
                        d.use = False
                    gpus = devs
                    if self.workers > 1:
                        gpus = [devs[self.worker % len(devs)]]
                    for d in gpus:
                        d.use = True
                    log(f"device {kind}: {[d.name for d in gpus]}")
                    break
        if chosen:
            sc.cycles.device = "GPU"
            sc.cycles.denoiser = "OPTIX" if chosen == "OPTIX" else "OPENIMAGEDENOISE"
            if hasattr(sc.cycles, "denoising_use_gpu"):
                sc.cycles.denoising_use_gpu = True
        else:
            sc.cycles.device = "CPU"
            sc.cycles.denoiser = "OPENIMAGEDENOISE"
            log("device CPU")
        sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        try:
            sc.cycles.denoising_prefilter = "ACCURATE"
            sc.cycles.denoising_quality = "HIGH"
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def load_wax(self, f):
        p = os.path.join(self.work, "wax", f"wax_{f:04d}.npz")
        return dict(np.load(p))

    def build(self):
        sc = bpy.context.scene
        # world: nearly black, a hint of cool ambience
        w = bpy.data.worlds.new("World")
        sc.world = w
        nt = w.node_tree
        bg = [n for n in nt.nodes if n.type == "BACKGROUND"]
        bg = bg[0] if bg else nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (0.55, 0.62, 0.8, 1)
        bg.inputs["Strength"].default_value = LOOK["world_strength"]

        # table + far wall
        verts = np.array([[-1.5, -1.5, 0], [1.5, -1.5, 0], [1.5, 1.5, 0], [-1.5, 1.5, 0]], np.float32)
        build_mesh_object("Table", verts, [(0, 1, 2, 3)], mat_table(), smooth=False)
        verts = np.array([[-3, 1.8, 0], [3, 1.8, 0], [3, 1.8, 2.5], [-3, 1.8, 2.5]], np.float32)
        build_mesh_object("Wall", verts, [(0, 1, 2, 3)], mat_wall(), smooth=False)

        # practicals far behind for bokeh
        rng = np.random.default_rng(11)
        for i in range(9):
            x = rng.uniform(-0.9, 0.9)
            y = rng.uniform(1.0, 1.7)
            z = rng.uniform(0.06, 0.34)
            T = rng.uniform(1900, 2600)
            col = spectral.blackbody_rgb(T)
            col = tuple(float(c) for c in col / max(col))
            bpy.ops.mesh.primitive_uv_sphere_add(radius=rng.uniform(0.004, 0.009), location=(x, y, z), segments=16,
                                                 ring_count=8)
            ob = bpy.context.object
            ob.name = f"Bokeh{i}"
            ob.data.materials.append(mat_bokeh(col, rng.uniform(25, 90)))
            ob.visible_diffuse = False
            ob.visible_glossy = True
            ob.visible_shadow = False

        # cool, soft "window" light from behind-left: separates the smoke after blow-out
        L = bpy.data.lights.new("Window", "AREA")
        L.shape = "RECTANGLE"
        L.size, L.size_y = 0.5, 0.8
        L.energy = LOOK["rim_light_power"]
        L.color = tuple(float(c) for c in spectral.blackbody_rgb(6200) / max(spectral.blackbody_rgb(6200)))
        lo = bpy.data.objects.new("Window", L)
        lo.location = (-0.45, 0.55, 0.42)
        lo.rotation_quaternion = look_at_quat(lo.location, (0, 0, 0.14))
        lo.rotation_mode = "QUATERNION"
        lo.rotation_quaternion = look_at_quat(lo.location, (0, 0, 0.14))
        lo.visible_glossy = True
        sc.collection.objects.link(lo)

        # candle
        w0 = self.load_wax(0)
        self.cb = CandleBuilder(w0)
        verts, attrs = self.cb.profile(w0)
        self.candle = build_mesh_object("Candle", verts, self.cb.faces(), mat_wax(), attrs)
        # molten pool surface
        pv, pf = pool_geometry(float(w0["L"][0]))
        self.pool = build_mesh_object("Pool", pv, pf, mat_pool())
        self.pool.visible_shadow = False
        self.pool_base = pv.copy()
        self.pool_L0 = float(w0["L"][0])
        # wick
        wv, wf, tip = wick_geometry()
        self.wick = build_mesh_object("Wick", wv, wf, mat_wick(), {"tip": tip})

        # flame volume
        vol = bpy.data.volumes.new("Flame")
        self.vol = vol
        self.vol_obj = bpy.data.objects.new("Flame", vol)
        sc.collection.objects.link(self.vol_obj)
        vol.materials.append(mat_flame())
        try:
            vol.render.space = "OBJECT"
            vol.render.step_size = 0.0
            vol.render.precision = "FULL"
        except Exception:
            pass

        # Shadow linking: the flame volume must not shadow its own light proxies
        # (their power already is the light that escapes the flame), but it
        # still shadows the window light so the smoke self-shadows.
        blockers = bpy.data.collections.new("FlameLightBlockers")
        for ob in sc.collection.objects:
            if ob.type == "MESH":
                blockers.objects.link(ob)

        # light proxies
        self.proxies = []
        for i in range(3):
            Ld = bpy.data.lights.new(f"FlameLight{i}", "POINT")
            Ld.energy = 0.0
            Ld.shadow_soft_size = LOOK["proxy_light_radius"]
            ob = bpy.data.objects.new(f"FlameLight{i}", Ld)
            ob.visible_glossy = False
            ob.visible_transmission = False
            sc.collection.objects.link(ob)
            ob.light_linking.blocker_collection = blockers
            self.proxies.append(ob)

        # camera
        cam = bpy.data.cameras.new("Camera")
        cam.sensor_width = 36.0
        cam.sensor_fit = "HORIZONTAL"
        cam.clip_start = 0.005
        cam.clip_end = 20.0
        cam.dof.use_dof = True
        cam.dof.aperture_blades = 7
        cam.dof.aperture_rotation = math.radians(12)
        co = bpy.data.objects.new("Camera", cam)
        sc.collection.objects.link(co)
        sc.camera = co
        co.rotation_mode = "QUATERNION"
        self.cam = co
        self.track = camera_track(self.work, self.n_frames)
        for f, t in enumerate(self.track):
            co.location = t["cam"]
            q = look_at_quat(t["cam"], t["target"] + t["shake"])
            co.rotation_quaternion = q
            cam.lens = t["lens"]
            cam.dof.aperture_fstop = t["fstop"]
            fwd = np.array(t["target"]) - np.array(t["cam"])
            fwd /= np.linalg.norm(fwd)
            cam.dof.focus_distance = float(max(np.dot(np.array(t["focus"]) - np.array(t["cam"]), fwd), 0.02))
            co.keyframe_insert("location", frame=f)
            co.keyframe_insert("rotation_quaternion", frame=f)
            cam.keyframe_insert("lens", frame=f)
            cam.dof.keyframe_insert("aperture_fstop", frame=f)
            cam.dof.keyframe_insert("focus_distance", frame=f)
        self.setup_compositor()

    def setup_compositor(self):
        if not LOOK.get("bloom"):
            return
        sc = bpy.context.scene
        try:
            tree = bpy.data.node_groups.new("Grade", "CompositorNodeTree")
            tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
            rl = tree.nodes.new("CompositorNodeRLayers")
            gl = tree.nodes.new("CompositorNodeGlare")
            gl.inputs["Type"].default_value = "Bloom"
            gl.inputs["Quality"].default_value = "High"
            gl.inputs["Threshold"].default_value = 2.0
            gl.inputs["Strength"].default_value = 0.22
            gl.inputs["Size"].default_value = 0.6
            out = tree.nodes.new("NodeGroupOutput")
            tree.links.new(rl.outputs["Image"], gl.inputs["Image"])
            tree.links.new(gl.outputs["Image"], out.inputs[0])
            sc.compositing_node_group = tree
            log("compositor: bloom")
        except Exception as e:  # compositor API changes between versions: never fail the render on it
            log(f"compositor disabled ({e!r})")

    # ------------------------------------------------------------------ #
    def update(self, f):
        sc = bpy.context.scene
        sc.frame_set(f)
        # wax
        w = self.load_wax(f)
        verts, attrs = self.cb.profile(w)
        update_mesh(self.candle, verts, attrs)
        dzL = float(w["L"][0]) - self.pool_L0
        pv = self.pool_base.copy()
        pv[:, 2] += dzL
        update_mesh(self.pool, pv)
        nz_name = self.pool.data.materials[0].node_tree.get("noise_node")
        if nz_name:
            nn = self.pool.data.materials[0].node_tree.nodes[nz_name]
            nn.inputs["W"].default_value = TL.sim_time(f) * 1.5
        # flame volume + light proxies
        vdb = os.path.join(self.work, "vdb", f"flame_{f:04d}.vdb")
        lpath = os.path.join(self.work, "vdb", f"lights_{f:04d}.json")
        if os.path.exists(vdb) and os.path.exists(lpath):
            self.vol.filepath = vdb
            self.vol_obj.hide_render = False
            lights = json.load(open(lpath))["lights"]
        else:  # look-dev without flame data
            log(f"warning: no flame volume for frame {f}")
            self.vol_obj.hide_render = True
            lights = [dict(pos=[0.0, 0.002, GEOM.pool_level + 0.02], rgb=[0.6, 0.18, 0.04], spread=0.006)]
        for i, ob in enumerate(self.proxies):
            if i < len(lights):
                Lt = lights[i]
                rgb = np.array(Lt["rgb"])
                m = float(rgb.max())
                ob.location = Lt["pos"]
                ob.data.energy = m
                ob.data.color = tuple(float(c) for c in (rgb / m if m > 0 else np.ones(3)))
                ob.data.shadow_soft_size = LOOK["proxy_light_radius"] + 0.6 * Lt.get("spread", 0.0)
            else:
                ob.data.energy = 0.0
        # ember
        t = TL.sim_time(f)
        Te = TL.ember_temperature(t)
        nt = self.wick.data.materials[0].node_tree
        nt.nodes["EmberColor"].inputs["Temperature"].default_value = Te
        rel = spectral.blackbody_luminance(Te) / spectral.blackbody_luminance(1800.0)
        nt.nodes["EmberStrength"].outputs[0].default_value = LOOK["ember_gain"] * rel
        # exposure: camera opens up after blow-out; fade in / out
        dark = TL._smooth((f - (TL.BLOW_F + 6)) / 50.0)
        ev = LOOK["exposure_burning"] * (1 - dark) + LOOK["exposure_dark"] * dark
        ev += math.log2(max(TL.fade(f), 1e-4))
        sc.view_settings.exposure = ev

    def frames(self, start, end, step):
        fs = list(range(start, min(end, self.n_frames), step))
        if self.workers > 1:
            fs = fs[self.worker::self.workers]
        return fs

    def render(self, start=0, end=None, step=1):
        end = self.n_frames if end is None else end
        todo = [f for f in self.frames(start, end, step)
                if not os.path.exists(os.path.join(self.out_dir, f"frame_{f:04d}.png"))]
        log(f"{len(todo)} frames to render (worker {self.worker + 1}/{self.workers})")
        t0 = time.time()
        for n, f in enumerate(todo):
            tf = time.time()
            self.update(f)
            bpy.ops.render.render(write_still=False)
            final = os.path.join(self.out_dir, f"frame_{f:04d}.png")
            tmp = os.path.join(self.out_dir, f".tmp_{self.worker}_frame_{f:04d}.png")
            bpy.data.images["Render Result"].save_render(tmp)
            os.replace(tmp, final)
            el = time.time() - t0
            eta = el / (n + 1) * (len(todo) - n - 1)
            log(f"frame {f} done in {time.time() - tf:.1f}s  ({n + 1}/{len(todo)}, eta {eta / 60:.1f} min)")


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--quality", default="preview")
    ap.add_argument("--device", default="AUTO", help="AUTO, OPTIX, CUDA, CPU")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=None, help="render every Nth frame (default from preset)")
    ap.add_argument("--frames", type=int, nargs="*", help="explicit frame list (overrides start/end/step)")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--resolution", type=int, nargs=2, default=None)
    ap.add_argument("--save-blend", action="store_true")
    a = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("cycles", default_set=True, persistent=True)
    n_frames = TL.N_FRAMES
    shot = Shot(a.work, a.quality, a.device, a.worker, a.workers, n_frames)
    if a.samples:
        shot.rp.samples = a.samples
    if a.resolution:
        shot.rp.resolution = tuple(a.resolution)
    shot.setup_render()
    shot.build()
    if a.save_blend or a.worker == 0:
        try:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.work, "candle_scene.blend"), check_existing=False,
                                        compress=True)
        except Exception as e:
            log(f"could not save .blend: {e!r}")
    if a.frames:
        for f in a.frames:
            p = os.path.join(shot.out_dir, f"frame_{f:04d}.png")
            if os.path.exists(p):
                os.remove(p)
        shot.workers = 1
        shot.render_list = a.frames
        todo = a.frames
        for f in todo:
            t0 = time.time()
            shot.update(f)
            bpy.ops.render.render(write_still=False)
            bpy.data.images["Render Result"].save_render(os.path.join(shot.out_dir, f"frame_{f:04d}.png"))
            log(f"frame {f} done in {time.time() - t0:.1f}s")
        return
    step = a.step or shot.rp.frame_step
    shot.render(a.start, a.end, step)


if __name__ == "__main__":
    main()
