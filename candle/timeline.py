"""The shot: one continuous 16 s sequence at 24 fps, with speed ramps.

  "Breath"
  0.0 -  3.0 s  Darkness opens on a burning pillar candle. Slow push-in.        0.25x
  3.0 -  7.0 s  A breath from camera-left: the flame bows, stretches, flickers,
                sheds a soot wisp. Close-up, slow orbit.                       0.20x
  7.0 - 11.0 s  The heated rim gives way; molten wax spills over the notch and
                races down the side, slowing and freezing. Speed ramps to real
                time while the camera cranes down with the drip front.         1.00x
 11.0 - 16.0 s  Camera returns to the wick. A sharp breath blows the flame out;
                the ember glows and a white paraffin smoke ribbon curls up.
                Camera rises with the ribbon, fade to black.                   0.25x

Everything in the pipeline is keyed off this module so the solver, the wax
and the camera agree on what happens when.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FPS = 24
N_FRAMES = 384              # 16 s
WARMUP = 1.4                # seconds of simulated pre-roll before frame 0

# (frame, playback speed) keys; speed = simulated seconds per real second.
SPEED_KEYS = [
    (0, 0.25),
    (70, 0.25),
    (92, 0.20),
    (166, 0.20),
    (190, 1.00),
    (252, 1.00),
    (274, 0.22),
    (330, 0.25),
    (384, 0.45),
]

# Story beats, expressed in output frames.
BREATH_START_F = 80
BREATH_PEAK_F = 112
BREATH_END_F = 170
BREACH_F = 176          # rim notch gives way (wax sim is steered to this)
BLOW_F = 290            # blow-out puff reaches the flame
FADE_START_F = 360


def _smooth(a: float) -> float:
    a = min(max(a, 0.0), 1.0)
    return a * a * (3.0 - 2.0 * a)


def speed_at(frame: float) -> float:
    ks = SPEED_KEYS
    if frame <= ks[0][0]:
        return ks[0][1]
    for (f0, s0), (f1, s1) in zip(ks[:-1], ks[1:]):
        if frame <= f1:
            return s0 + (s1 - s0) * _smooth((frame - f0) / (f1 - f0))
    return ks[-1][1]


def _build_table():
    # integrate speed with 16 sub-steps per frame
    sub = 16
    t = [0.0]
    acc = 0.0
    for f in range(N_FRAMES + 2):
        for s in range(sub):
            acc += speed_at(f + (s + 0.5) / sub) / (FPS * sub)
        t.append(acc)
    return np.array(t)


_T = _build_table()


def sim_time(frame: float) -> float:
    """Simulated time (s, after warm-up) shown at an output frame (can be fractional)."""
    f = min(max(frame, 0.0), N_FRAMES + 1.0)
    i = int(math.floor(f))
    a = f - i
    return float(_T[i] * (1 - a) + _T[i + 1] * a) + WARMUP


def frame_at(t: float) -> float:
    """Inverse of sim_time (fractional frame)."""
    tt = t - WARMUP
    i = int(np.searchsorted(_T, tt)) - 1
    i = min(max(i, 0), len(_T) - 2)
    a = (tt - _T[i]) / max(_T[i + 1] - _T[i], 1e-12)
    return i + a


def total_sim_time() -> float:
    return sim_time(N_FRAMES)


def frame_times():
    return [sim_time(f) for f in range(N_FRAMES)]


# --------------------------------------------------------------------------- #
# Air-flow events consumed by the flame solver
# --------------------------------------------------------------------------- #
@dataclass
class Puff:
    """A breath of air: a soft jet entering the domain from one side."""
    t_start: float
    t_peak: float
    t_end: float
    speed: float          # target air speed at the flame [m/s]
    azimuth: float        # direction the air comes FROM (radians, world)
    height: float         # jet centre height above the pool [m]
    width: float          # jet radius [m]
    turbulence: float     # relative strength of curl-noise modulation
    extinguish: bool = False

    def envelope(self, t: float) -> float:
        if t <= self.t_start or t >= self.t_end:
            return 0.0
        if t < self.t_peak:
            return _smooth((t - self.t_start) / (self.t_peak - self.t_start))
        return 1.0 - _smooth((t - self.t_peak) / (self.t_end - self.t_peak))


def puffs():
    breath = Puff(
        t_start=sim_time(BREATH_START_F),
        t_peak=sim_time(BREATH_PEAK_F),
        t_end=sim_time(BREATH_END_F),
        speed=0.55, azimuth=math.pi, height=0.012, width=0.022,
        turbulence=0.8,
    )
    tb = sim_time(BLOW_F)
    blow = Puff(
        t_start=tb - 0.045, t_peak=tb + 0.02, t_end=tb + 0.16,
        speed=4.2, azimuth=math.pi + 0.25, height=0.009, width=0.016,
        turbulence=0.5, extinguish=True,
    )
    return [breath, blow]


def blow_time() -> float:
    return sim_time(BLOW_F)


def ember_temperature(t: float) -> float:
    """Wick-tip ember temperature after blow-out (K). Before blow-out the flame heats it."""
    tb = blow_time() + 0.05
    if t < tb:
        return 1050.0
    dt = t - tb
    return 300.0 + 780.0 * math.exp(-dt / 0.95)


def wick_evaporation(t: float) -> float:
    """Relative paraffin vapour production at the wick (1 while burning)."""
    tb = blow_time() + 0.05
    if t < tb:
        return 1.0
    T = ember_temperature(t)
    # vapour pressure of paraffin falls steeply as the ember cools
    return 0.85 * math.exp(-(1080.0 - T) / 260.0)


def fade(frame: float) -> float:
    """1 = fully visible, 0 = black. Opens from black and fades out at the end."""
    a = _smooth(frame / 30.0)
    b = 1.0 - _smooth((frame - FADE_START_F) / (N_FRAMES - 1 - FADE_START_F))
    return a * b


if __name__ == "__main__":
    print(f"total simulated time incl. warm-up: {total_sim_time():.3f} s")
    for f in [0, 72, 96, 168, 176, 190, 252, 274, 290, 330, 383]:
        print(f"frame {f:3d}  t={sim_time(f):.3f}s  speed={speed_at(f):.2f}")
