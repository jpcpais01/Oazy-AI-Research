#!/usr/bin/env python3
"""One command for the whole candle shot.

    python run.py --quality final --work /workspace/candle

Stages (each one is skipped / resumed automatically if its output exists, so
re-running the same command after a pod restart picks up where it stopped):

    flame   GPU reacting-flow simulation              -> WORK/flame/
    wax     melt pool + drips                         -> WORK/wax/
    vdb     emission/smoke VDBs + light proxies       -> WORK/vdb/
    render  Blender Cycles, one process per GPU       -> WORK/frames/
    encode  ProRes / HEVC 10-bit / H.264 masters      -> WORK/output/

Useful extras:
    --stages render,encode      run a subset
    --frames 120 200 300        render just these frames (look test / benchmark)
    --status                    print progress and exit
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from candle import config as C  # noqa: E402
from candle import timeline as TL  # noqa: E402

STAGES = ["flame", "wax", "vdb", "render", "encode"]


def sh(cmd, log_path=None, env=None, check=True):
    print("$ " + " ".join(cmd), flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as lf:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
            for line in p.stdout:
                if "Module " in line and "load on device" in line:
                    continue
                sys.stdout.write(line)
                lf.write(line)
            rc = p.wait()
    else:
        rc = subprocess.call(cmd, env=env)
    if check and rc != 0:
        raise SystemExit(f"command failed ({rc}): {' '.join(cmd)}")
    return rc


def gpu_count():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return len([l for l in out.splitlines() if l.startswith("GPU ")])
    except Exception:
        return 0


def find_blender(explicit):
    cands = [explicit, os.environ.get("BLENDER"), "/workspace/blender/blender", shutil.which("blender")]
    for c in cands:
        if c and os.path.exists(c):
            return [c, "-b", "--factory-startup", "--python"]
    try:
        import bpy  # noqa: F401
        return [sys.executable]
    except Exception:
        pass
    raise SystemExit("Blender not found: run runpod/setup.sh, pass --blender, or pip install bpy==5.2.3")


def status(work, quality):
    n = TL.N_FRAMES
    flame = len(glob.glob(os.path.join(work, "flame", "fields_*.npz")))
    wax = len(glob.glob(os.path.join(work, "wax", "wax_*.npz")))
    vdb = len(glob.glob(os.path.join(work, "vdb", "flame_*.vdb")))
    _, _, rp = C.get_preset(quality)
    want = len(range(0, n, rp.frame_step))
    frames = len(glob.glob(os.path.join(work, "frames", "frame_*.png")))
    outs = sorted(glob.glob(os.path.join(work, "output", "*")))
    print(f"flame  {flame}/{n} frames simulated" + ("  (DONE)" if os.path.exists(os.path.join(work, 'flame', 'DONE')) else ""))
    print(f"wax    {wax}/{n}")
    print(f"vdb    {vdb}/{n}")
    print(f"render {frames}/{want}")
    for o in outs:
        print(f"output {o}  ({os.path.getsize(o) / 1e6:.1f} MB)")


def stage_flame(a):
    out = os.path.join(a.work, "flame")
    if os.path.exists(os.path.join(out, "DONE")):
        print("[run] flame: done")
        return
    cmd = [sys.executable, "-m", "candle.flame_sim", "--out", out, "--quality", a.quality]
    if a.sim_device:
        cmd += ["--device", a.sim_device]
    sh(cmd, os.path.join(a.work, "logs", "flame.log"), env={**os.environ, "PYTHONPATH": ROOT})


def stage_wax(a):
    out = os.path.join(a.work, "wax")
    if os.path.exists(os.path.join(out, "DONE")):
        print("[run] wax: done")
        return
    cmd = [sys.executable, "-m", "candle.wax_sim", "--out", out, "--quality", a.quality,
           "--flame", os.path.join(a.work, "flame")]
    if a.sim_device:
        cmd += ["--device", a.sim_device]
    sh(cmd, os.path.join(a.work, "logs", "wax.log"), env={**os.environ, "PYTHONPATH": ROOT})


def stage_vdb(a):
    cmd = [sys.executable, "-m", "candle.vdb_convert", "--flame", os.path.join(a.work, "flame"),
           "--out", os.path.join(a.work, "vdb")]
    sh(cmd, os.path.join(a.work, "logs", "vdb.log"), env={**os.environ, "PYTHONPATH": ROOT})


def stage_render(a):
    blender = find_blender(a.blender)
    script = os.path.join(ROOT, "candle", "blender", "render_shot.py")
    base = blender + [script]
    if len(blender) > 1:
        base.append("--")
    base += ["--work", a.work, "--quality", a.quality, "--device", a.render_device]
    if a.samples:
        base += ["--samples", str(a.samples)]
    if a.resolution:
        base += ["--resolution", str(a.resolution[0]), str(a.resolution[1])]
    if a.step:
        base += ["--step", str(a.step)]
    if a.frames:
        sh(base + ["--frames"] + [str(f) for f in a.frames], os.path.join(a.work, "logs", "render.log"))
        return
    if a.workers:
        # several pods sharing one network volume: this pod renders every N-th frame
        sh(base + ["--worker", str(a.worker), "--workers", str(a.workers)],
           os.path.join(a.work, "logs", f"render_pod{a.worker}.log"))
        return
    n = a.gpus if a.gpus is not None else max(1, gpu_count())
    if a.render_device.upper() == "CPU":
        n = 1
    print(f"[run] rendering with {n} worker(s)")
    for attempt in range(4):
        procs = []
        for i in range(n):
            cmd = base + ["--worker", str(i), "--workers", str(n)]
            log = open(os.path.join(a.work, "logs", f"render_{i}.log"), "a")
            env = dict(os.environ)
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log, i))
            time.sleep(2)
        fails = 0
        while procs:
            for p, log, i in list(procs):
                rc = p.poll()
                if rc is None:
                    continue
                procs.remove((p, log, i))
                log.close()
                if rc != 0:
                    fails += 1
                    print(f"[run] render worker {i} exited with {rc} (see logs/render_{i}.log)", flush=True)
            if procs:
                time.sleep(20)
                done = len(glob.glob(os.path.join(a.work, "frames", "frame_*.png")))
                print(f"[run] rendered frames on disk: {done}", flush=True)
        if fails == 0:
            return
        print(f"[run] {fails} worker(s) failed, restarting (attempt {attempt + 2}/4) -- finished frames are kept")
    raise SystemExit("render failed repeatedly; check WORK/logs/render_*.log")


def stage_encode(a):
    frames_dir = os.path.join(a.work, "frames")
    out = os.path.join(a.work, "output")
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not files:
        raise SystemExit("no frames to encode")
    nums = [int(os.path.basename(f)[6:10]) for f in files]
    step = min((b - a_ for a_, b in zip(nums[:-1], nums[1:])), default=1) or 1
    fps = TL.FPS / step
    # make a gap-free numbered sequence via symlinks (handles preview steps)
    seq = os.path.join(a.work, "frames_seq")
    shutil.rmtree(seq, ignore_errors=True)
    os.makedirs(seq)
    for i, f in enumerate(files):
        os.symlink(f, os.path.join(seq, f"{i:05d}.png"))
    tag = a.quality
    inp = ["-y", "-framerate", f"{fps:g}", "-i", os.path.join(seq, "%05d.png")]
    color = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
    w, h = C.get_preset(a.quality)[2].resolution if not a.resolution else a.resolution
    if a.quality in ("hd", "final", "ultra"):
        sh(["ffmpeg"] + inp + ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
                               "-vendor", "apl0"] + color + [os.path.join(out, f"candle_{tag}_prores422hq.mov")])
        sh(["ffmpeg"] + inp + ["-c:v", "libx265", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p10le",
                               "-tag:v", "hvc1", "-x265-params", "aq-mode=3:no-sao=1:deblock=-1,-1"] + color
           + ["-movflags", "+faststart", os.path.join(out, f"candle_{tag}_hevc10.mp4")])
    sh(["ffmpeg"] + inp + ["-vf", "scale=1920:-2:flags=lanczos", "-c:v", "libx264", "-preset", "slow", "-crf", "15",
                           "-pix_fmt", "yuv420p"] + color + ["-movflags", "+faststart",
                                                              os.path.join(out, f"candle_{tag}_1080p_h264.mp4")])
    print("[run] outputs:")
    for f in sorted(glob.glob(os.path.join(out, "*"))):
        print(f"   {f}  {os.path.getsize(f) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quality", default="preview", choices=sorted(C.PRESETS))
    ap.add_argument("--work", default=None, help="working directory (default: ./work/<quality>)")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--gpus", type=int, default=None, help="render workers (default: one per GPU)")
    ap.add_argument("--blender", default=None, help="path to the blender executable")
    ap.add_argument("--render-device", default="AUTO", help="AUTO (OptiX, then CUDA), OPTIX, CUDA or CPU")
    ap.add_argument("--sim-device", default=None, help="Warp device for the simulations, e.g. cuda:0 or cpu")
    ap.add_argument("--frames", type=int, nargs="*", help="render only these frames (look test)")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--resolution", type=int, nargs=2, default=None)
    ap.add_argument("--step", type=int, default=None, help="render every Nth frame")
    ap.add_argument("--worker", type=int, default=0, help="this pod's index when splitting the render over pods")
    ap.add_argument("--workers", type=int, default=None, help="number of pods splitting the render")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    a.work = os.path.abspath(a.work or os.path.join(ROOT, "work", a.quality))
    os.makedirs(os.path.join(a.work, "logs"), exist_ok=True)
    if a.status:
        status(a.work, a.quality)
        return
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s}")
    t0 = time.time()
    for s in STAGES:
        if s in stages:
            ts = time.time()
            print(f"\n========== {s} ==========", flush=True)
            globals()[f"stage_{s}"](a)
            print(f"[run] {s} finished in {(time.time() - ts) / 60:.1f} min", flush=True)
            if s == "render" and a.frames:
                break
    print(f"\n[run] all done in {(time.time() - t0) / 60:.1f} min. Work dir: {a.work}")


if __name__ == "__main__":
    main()
