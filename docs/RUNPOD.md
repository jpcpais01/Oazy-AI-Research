# Rendering the candle shot on RunPod

You need one GPU pod with a persistent volume. All the work happens in one
terminal with one command, and the command resumes after any interruption.

## 1. Create the pod

1. Go to **RunPod → Pods → Deploy**.
2. **GPU:** pick **RTX 6000 Ada (48 GB)**. It has RT cores, so Cycles/OptiX
   renders much faster per dollar than an H100/H200/B300 (those have no RT
   cores), and 48 GB is enough for the `final` flame grid. `ultra` needs an
   H200 or B300 for the simulation.
3. **Template:** the official **RunPod PyTorch** template, for example
   `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. Any recent tag
   is fine because we only use its Python, CUDA driver, SSH and JupyterLab.
4. **Storage** (Edit Template, or the storage fields on the deploy page):
   * Container Disk: **30 GB**
   * Volume Disk: **150 GB**, mounted at **`/workspace`** (the default). Everything
     lives there and survives a pod stop or restart.
5. Use **On-Demand**, not Spot. A Spot pod can be pre-empted. The pipeline
   would resume, but the restarts cost you time.
6. Click **Deploy On-Demand**, wait until the pod is running, then **Connect →
   Start Web Terminal** (or connect via SSH or JupyterLab → Terminal).

## 2. Set up (about 3–5 minutes, once)

```bash
cd /workspace
git clone https://github.com/jpcpais01/simulations.git
cd simulations
git checkout claude/project-thread-lch3ih      # until the PR is merged; afterwards just use main
bash runpod/setup.sh
```

The setup script ends with checks. You should see your GPU in the
`nvidia-smi` line, `warp CUDA devices: ['cuda:0']`, a `cycles OPTIX [...]` or
`cycles CUDA [...]` line with the GPU name, and `BLENDER_OK 5.2.3`.

## 3. Quick end-to-end smoke test (about 5 minutes)

Run every stage on a tiny grid at low resolution, so a typo never costs you an
expensive run:

```bash
python run.py --quality smoke --work /workspace/candle_smoke
```

It finishes with `work dir ...` and an mp4 in `/workspace/candle_smoke/output/`.

## 4. The real run

Use `tmux` so the job keeps running if your browser tab closes:

```bash
tmux new -s candle
cd /workspace/simulations
python run.py --quality final --work /workspace/candle 2>&1 | tee -a /workspace/candle/run.log
```

Detach with `Ctrl-b` then `d`, and reattach later with `tmux attach -t candle`.
To see progress from any terminal:

```bash
python run.py --quality final --work /workspace/candle --status
```

**If the pod restarts or anything stops,** run the exact same `python run.py
...` command again. The flame solver restarts from its last checkpoint (written
every 10 minutes and whenever it is stopped). Finished stages are skipped and
finished frames are kept.

### Optional: look test before the long render

When `flame`, `wax` and `vdb` are done (look for `========== render ==========`
in the log), you can stop with `Ctrl-c` and render four hero frames first.
Look at them before you commit to the full render:

```bash
python run.py --quality final --work /workspace/candle --stages render --frames 40 130 220 320
```

They are written to `/workspace/candle/frames/`. Each render also prints its
seconds per frame, which is your real per-frame cost. Then continue with:

```bash
python run.py --quality final --work /workspace/candle --stages render,encode
```

## 5. Download the video

The finished files are in `/workspace/candle/output/`:

| file | what |
|---|---|
| `candle_final_prores422hq.mov` | 4K ProRes 422 HQ, 10-bit master (large, best for editing) |
| `candle_final_hevc10.mp4` | 4K HEVC 10-bit, high quality, plays on Mac, iPhone and modern TVs |
| `candle_final_1080p_h264.mp4` | 1080p H.264 that plays everywhere (phone, WhatsApp) |

Ways to get them:

* **JupyterLab** (Connect → JupyterLab): browse to `candle/output/`,
  right-click the file and choose **Download**. This is the simplest way for
  the mp4s.
* **runpodctl**, which is preinstalled on pods and best for the big ProRes
  file:
  ```bash
  runpodctl send /workspace/candle/output/candle_final_prores422hq.mov
  ```
  It prints a one-time code. On your computer, install runpodctl
  (https://github.com/runpod/runpodctl) and run `runpodctl receive <code>`.
* **scp** over the pod's SSH connection (Connect → SSH shows the host and port):
  `scp -P <port> root@<host>:/workspace/candle/output/*.mp4 .`

The individual 16-bit PNG frames are in `/workspace/candle/frames/` and the
Blender scene is at `/workspace/candle/candle_scene.blend`, in case you want to
grade or re-edit.

## 6. Stop paying

When you have the files, **Stop** the pod (the volume is kept, and it costs a
little per GB per month) or **Terminate** it (deletes everything). Terminate
only after the downloads have finished.

## Time and cost (rough)

These are estimates, not measurements on the target GPU (the development
machine had no GPU). The look test in step 4 tells you the real per-frame
render time after about 10 minutes.

**Option A, one RTX 6000 Ada pod (recommended for a first run):** about
10–16 h in total, nearly all of it rendering, so roughly **$10–16** at
$0.99/h.

**Option B, faster (about 4–6 h, $15–25):**

1. Create a **Network Volume** (Storage → Network Volumes, 150 GB) in a data
   centre that has both H100 SXM and RTX 6000 Ada.
2. On an **H100 SXM** pod with that volume, run setup, then
   `python run.py --quality final --work /workspace/candle --stages flame,wax,vdb`.
   Stop the pod when it finishes.
3. Deploy **three RTX 6000 Ada** pods on the same volume. On each, run
   `bash runpod/setup.sh`, then on pod *i* (0, 1, 2):
   `python run.py --quality final --work /workspace/candle --stages render --worker i --workers 3`
4. When all three finish, on any one pod:
   `python run.py --quality final --work /workspace/candle --stages encode`

## Troubleshooting

* `flame extinguished unexpectedly`: the watchdog stopped the solver because
  the flame died before the scripted blow-out. This never happened in testing.
  Re-run the same command and it resumes from the last checkpoint. If it
  happens again, raise `pilot_T` in `candle/flame_sim.py` (`PHYS`) from 1550 to
  1700.
* Blender reports no OptiX/CUDA devices: run `nvidia-smi`. If the GPU is
  visible, force CUDA with
  `python run.py ... --render-device CUDA`.
* Out of GPU memory in the flame stage on an 80 GB card: use `--quality final`
  (not `ultra`).
* Want to change the look (exposure, flame brightness, smoke density)? Edit
  `candle/look.py`, then delete `/workspace/candle/vdb` and
  `/workspace/candle/frames` and re-run with `--stages vdb,render,encode`. The
  simulations do not need to be redone.
