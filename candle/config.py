"""Quality presets and physical constants for the candle shot.

Units are SI throughout (metres, seconds, kelvin, kilograms).
World frame: z is up, the candle axis is the z axis, the base of the candle sits
on the table at z = 0.  The flame solver's domain is centred on the wick and its
floor coincides with the melt-pool surface.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------------------- #
# Candle geometry (shared by every stage)
# --------------------------------------------------------------------------- #
@dataclass
class CandleGeometry:
    radius: float = 0.026          # pillar candle radius (52 mm diameter)
    rim_height: float = 0.1080     # nominal height of the rim crest
    pool_level: float = 0.1032     # initial melt-pool liquid level (world z)
    pool_radius: float = 0.0228    # radius where the pool meets the inner rim wall
    pool_depth: float = 0.0075     # depth of the liquid at the wick
    wick_radius: float = 0.00065
    wick_height: float = 0.0095    # exposed wick above the liquid
    wick_lean: float = 0.0022      # horizontal curl of the wick tip (towards +y)
    # The rim notch where the breach happens (radians, world azimuth).
    # Camera starts at azimuth -90deg (looking along +y); the breath comes from
    # camera-left (-x), pushes the flame to +x, so the notch sits on the +x
    # side, rotated slightly towards the camera so the drip runs in view.
    notch_azimuth: float = -0.42
    notch_depth: float = 0.0042    # how much lower the notch is than the crest
    notch_width: float = 0.20      # angular half-width (radians)


# --------------------------------------------------------------------------- #
# Quality presets
# --------------------------------------------------------------------------- #
@dataclass
class FlamePreset:
    res: tuple = (64, 64, 224)     # nx, ny, nz (each divisible by 2**(mg_levels-1))
    dx: float = 0.00088            # voxel size [m]
    mg_levels: int = 4
    cfl: float = 2.0
    max_dt: float = 1.0 / 400.0
    cg_tol: float = 1e-4
    cg_maxiter: int = 40
    diffusion_iters: int = 4
    vorticity: float = 0.35        # confinement strength (smoke region only)
    export_dtype: str = "float16"


@dataclass
class WaxPreset:
    ntheta: int = 384              # film grid around the circumference
    nz: int = 260                  # film grid along the height
    top_rings: int = 64            # radial rings for the crater / pool floor mesh
    history_events: int = 11       # old frozen drips generated before the shot


@dataclass
class RenderPreset:
    resolution: tuple = (960, 540)
    samples: int = 64
    adaptive_threshold: float = 0.03
    denoise: bool = True
    volume_step_rate: float = 1.0
    max_bounces: int = 8
    volume_bounces: int = 1
    motion_blur: bool = True
    frame_step: int = 1            # render every Nth frame (preview speed-up)
    png_depth: str = "16"


PRESETS = {
    # Tiny: smoke test of every stage, a few minutes on a laptop CPU.
    "smoke": dict(
        flame=FlamePreset(res=(32, 32, 112), dx=0.00176, mg_levels=3, cfl=3.0,
                          cg_maxiter=25, diffusion_iters=2),
        wax=WaxPreset(ntheta=128, nz=96, top_rings=24, history_events=6),
        render=RenderPreset(resolution=(480, 270), samples=16, frame_step=8,
                            motion_blur=False, png_depth="8"),
    ),
    # Preview: what was verified on CPU before handoff.
    "preview": dict(
        flame=FlamePreset(res=(48, 48, 160), dx=0.00125, mg_levels=4, cfl=2.5,
                          cg_maxiter=30, diffusion_iters=3),
        wax=WaxPreset(ntheta=256, nz=192, top_rings=40, history_events=9),
        render=RenderPreset(resolution=(960, 540), samples=48, frame_step=2,
                            png_depth="8"),
    ),
    # Final: 4K UHD, intended for one H200 / H100 / A100 80GB.
    "final": dict(
        flame=FlamePreset(res=(256, 256, 768), dx=0.000235, mg_levels=6, cfl=1.6,
                          cg_tol=5e-5, cg_maxiter=60, diffusion_iters=8,
                          vorticity=0.30),
        wax=WaxPreset(ntheta=1536, nz=1152, top_rings=200, history_events=13),
        render=RenderPreset(resolution=(3840, 2160), samples=512,
                            adaptive_threshold=0.006, volume_step_rate=0.5,
                            max_bounces=12, volume_bounces=2),
    ),
    # Ultra: twice the voxels per axis on the flame. ~8x memory (~45 GB) and
    # ~10x sim time of "final". Only worth it on an H200 / B200.
    "ultra": dict(
        flame=FlamePreset(res=(384, 384, 1152), dx=0.000157, mg_levels=6, cfl=1.6,
                          cg_tol=5e-5, cg_maxiter=70, diffusion_iters=10,
                          vorticity=0.30),
        wax=WaxPreset(ntheta=2048, nz=1536, top_rings=256, history_events=13),
        render=RenderPreset(resolution=(3840, 2160), samples=768,
                            adaptive_threshold=0.005, volume_step_rate=0.35,
                            max_bounces=12, volume_bounces=2),
    ),
}


def get_preset(name: str):
    if name not in PRESETS:
        raise SystemExit(f"unknown quality '{name}', choose from {sorted(PRESETS)}")
    p = PRESETS[name]
    return p["flame"], p["wax"], p["render"]


def to_jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)
