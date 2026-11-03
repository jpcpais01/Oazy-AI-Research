"""Cinematic candle simulation pipeline.

Stages (see run.py):
  1. flame  - GPU (NVIDIA Warp) reacting-flow solver: velocity, temperature, fuel
              vapour, oxygen, soot and condensed-wax smoke; exports per-frame
              emission/absorption/scattering grids.
  2. wax    - melt pool + weir overflow + thin-film drip solver with
              temperature-dependent viscosity and solidification.
  3. vdb    - converts exported grids to OpenVDB (inside Blender's Python).
  4. render - Blender Cycles path-traced render, frame-by-frame with resume.
  5. encode - ffmpeg masters (ProRes 4444/422HQ, HEVC 10-bit, H.264 preview).
"""
