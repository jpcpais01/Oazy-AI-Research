# Candle: a path-traced, physically simulated candle shot

One continuous 16-second, 24 fps, 4K UHD shot of a pillar candle, built from
first principles and rendered with Blender Cycles:

| time | beat | playback speed |
|---|---|---|
| 0–3 s | Out of black onto a burning ivory pillar candle; slow push-in | 0.25× |
| 3–7 s | A breath from camera-left: the flame bows, stretches, flickers and sheds a soot wisp | 0.20× |
| 7–11 s | The heated rim slumps, molten wax spills over the notch and races down the side, slowing and freezing; the camera cranes down with the drip front | ramps to 1× |
| 11–16 s | Back at the wick: a sharp breath blows the flame out, the ember glows, and a white paraffin smoke ribbon curls up while the camera rises with it; fade to black | 0.25× |

Everything that moves is simulated:

* **Flame:** a GPU reacting-flow solver written with NVIDIA Warp
  ([`candle/flame_sim.py`](candle/flame_sim.py)). It uses a MAC grid with
  MacCormack advection and a multigrid-preconditioned CG pressure solve. The gas
  carries temperature, paraffin vapour, oxygen, soot and condensed-wax smoke. The
  flame is a real diffusion flame: vapour leaves the wick, mixes with air and
  burns where fuel and oxygen meet (Arrhenius chemistry). That gives the dark
  fuel-rich core, a blue reaction skirt at the base and the yellow soot body.
  Soot forms in hot rich gas and burns out at the tip. Heat is lost by radiation,
  thermal expansion accelerates the gas, and molecular diffusion and viscosity
  depend on temperature, so the flame stays laminar while the plume above it
  breaks up. After the blow-out the ember keeps evaporating paraffin, which
  condenses into the white smoke ribbon. That is what candle smoke really is.
* **Colour of light:** the soot glow is Planck's law × a 1/λ emissivity, and
  the blue base is the CH\* 431 nm and C₂ Swan band emission. Both are integrated
  against the CIE 1931 observer into linear Rec.709 ([`candle/spectral.py`](candle/spectral.py)).
  No colour ramps are hand-picked.
* **Wax:** a thin-film (lubrication) solver for the molten wax on the candle
  wall ([`candle/wax_sim.py`](candle/wax_sim.py)). Viscosity depends on
  temperature and rises steeply in the mushy zone. The film cools through its
  own thickness and freezes into deposit, and contact-line pinning leaves real
  drip trails. A melt pool overflows the rim like a weir and the hot spill erodes
  its own spout. The rim melts faster on the side the simulated flame leans
  towards. Before the shot, a "history" pass runs a dozen old spills to freezing
  to cover the candle in authentic old drips.
* **Rendering:** Blender Cycles path tracing ([`candle/blender/render_shot.py`](candle/blender/render_shot.py)).
  It uses random-walk subsurface scattering in the wax, a clear molten pool with
  Fresnel gloss, and the flame as an emissive, absorbing and scattering OpenVDB
  volume. Three moving light proxies carry the flame's measured light into the
  scene (their power is computed from the volume, with units verified in
  Cycles). The camera has real depth of field (7-blade iris), motion blur and
  AgX tone mapping, with bloom from the compositor.
* **OpenVDB** files are written by a small pure-numpy writer
  ([`candle/vdb_writer.py`](candle/vdb_writer.py)), verified bit-for-bit
  against the reference OpenVDB library.

## Run it

```bash
pip install -r requirements.txt                 # numpy, warp-lang, imageio
bash runpod/setup.sh                            # on a GPU pod: ffmpeg, Blender 5.2, checks
python run.py --quality final --work /workspace/candle
```

`run.py` runs `flame → wax → vdb → render → encode`. Every stage resumes:
the flame solver checkpoints every 10 minutes (and on SIGTERM), and the render
skips frames already on disk. If the pod restarts, re-run the same command.

Quality presets (see [`candle/config.py`](candle/config.py)):

| preset | flame grid | voxel | render | use |
|---|---|---|---|---|
| `smoke` | 32×32×112 | 1.76 mm | 480×270, every 8th frame | does every stage run? (CPU-able) |
| `preview` | 48×48×160 | 1.25 mm | 960×540, every 2nd frame | look check (CPU-able, slow) |
| `final` | 256×256×768 | 0.235 mm | 3840×2160, 512 spp adaptive | the real thing (one 80–141 GB GPU) |
| `ultra` | 384×384×1152 | 0.157 mm | 3840×2160, 768 spp | H200 / B200 only |

Other useful commands:

```bash
python run.py --quality final --work /workspace/candle --status                     # progress
python run.py --quality final --work /workspace/candle --stages render --frames 40 130 220 300   # look test
python -m candle.peek /workspace/candle/flame --frames 60 130 300 --out peek.png     # flame quick-look, no Blender
```

**RunPod step-by-step:** see [docs/RUNPOD.md](docs/RUNPOD.md).
