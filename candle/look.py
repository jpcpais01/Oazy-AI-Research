"""Look-development constants: everything that changes the picture but not the physics.

Changing these only requires re-running the `vdb` and `render` stages, never the
simulations.
"""

LOOK = dict(
    # --- flame emission ------------------------------------------------------
    # Radiance per metre of soot at 1800 K (soot field ~1 in the flame body).
    # Cycles volume emission strength is radiance per unit length, so a flame
    # ~6 mm thick with soot ~0.8 renders at roughly soot_gain * 0.005.
    soot_gain=24000.0,
    # ...but the vdb stage rescales soot_gain/blue_gain so that the steady
    # flame emits this total power (W, Blender units). Keeps exposure the same
    # at every grid resolution. Set to None to use the raw gains.
    flame_power=1.3,
    # per-frame cap on emitted power, as a multiple of flame_power
    max_power_ratio=2.0,
    # Blue CH*/C2* chemiluminescence per unit of the exported reaction field.
    blue_gain=30.0,
    # absorption of soot (per metre per unit soot) -- dark wisps when blown out
    soot_absorption=80.0,
    # white paraffin smoke: scattering per metre per unit smoke
    smoke_density=260.0,
    smoke_anisotropy=0.55,
    # --- wick ember ------------------------------------------------------------
    ember_gain=150.0,         # surface radiance of a blackbody ember at 1800 K (scaled by Planck for its T)
    # --- lights ------------------------------------------------------------------
    proxy_light_radius=0.0028,
    rim_light_power=0.6,      # cool soft window light from behind-left (W)
    world_strength=0.0015,
    # --- camera / film -------------------------------------------------------------
    exposure_burning=-1.2,
    exposure_dark=2.6,        # after blow-out the "camera" opens up
    look="AgX - Punchy",
    bloom=True,
)
