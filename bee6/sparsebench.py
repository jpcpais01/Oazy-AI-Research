"""Writing speed on the CPU (a stand-in for a phone): Bee6 with its whole MLP against Bee6 with the sparse MLP.

    python3 -m bee6.sparsebench                      # full size, float32, 4 threads then 1, 128-word prompt, 64 new words
    python3 -m bee6.sparsebench --threads 2 --dtype bf16 --new 32
    python3 -m bee6.sparsebench --run runs/bee6sparse-race       # a trained checkpoint (and its plain twin's shape)

The same loop as oazy/cpubench.py: read the prompt in one pass, then write one word at a time (greedy), each new word
read with the kept keys. Speed does not depend on what the weights learned, so random weights are used unless --run is
given; the sparse model is switched to its sparse phase (as after 10% of training). Both are next-word models.

Writing one word, the sparse MLP takes only the picked blocks' weights (bee6/sparse.py, "direct"): about 10% of each
MLP's weights are read per word, and on a CPU or a phone reading weights from memory is most of the time spent.
The phone engine (phone/engine) does not run Bee6 yet, so this is PyTorch on this machine's CPU.

Measured Nov 9 2026 in a 4-core cloud container, PyTorch 2.15 CPU, float32, random weights, prompt 128, 32 new words:
                          writing tok/s (4 thr / 1 thr)   reading tok/s (4 thr / 1 thr)
  Bee6 (whole MLP)            39.7 / 15.1                     780 / 270
  Bee6 + sparse MLP 10%       48.4 / 22.2                    1008 / 368       (1.22x / 1.47x writing)
The MLP alone gets ~4x faster per word (3 of 30 blocks, 78 vs 326 microseconds a layer, 4 threads); the rest of a
word's time (the 49,152-word score head, attention and its kept keys) is unchanged, so the whole gain is smaller.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch


def build(sparse, dtype, run=None):
    from bee6.model import Bee6, Bee6Config
    if run:
        from oazy.train import load_model
        from pathlib import Path
        m, _ = load_model(Path(run) / "ckpt.pt", torch.device("cpu"))
        if not sparse:
            c = Bee6Config.from_dict({**m.cfg.to_dict(), "sparse_mlp": 0.0})
            m = Bee6(c)
    else:
        m = Bee6(Bee6Config(fp8=0, sparse_mlp=0.10 if sparse else 0.0))
        with torch.no_grad():                    # leave the zero start (zero outputs make every block look alike)
            for p in m.parameters():
                if p.dim() == 2 and p.abs().sum() == 0:
                    p.normal_(0, 0.02)
        if sparse:
            m.set_progress(1.0)
    m.cfg.fp8 = 0
    return m.to(dtype).eval()


def timed(m, prompt, new, repeats):
    with torch.no_grad():
        t0 = time.perf_counter()
        logits, st = m.read(prompt)
        read_s = time.perf_counter() - t0
        speeds = []
        for _ in range(repeats):
            lg, s2 = logits, st
            t0 = time.perf_counter()
            for _ in range(new):
                nxt = lg[:, -1].argmax(-1, keepdim=True)
                lg, s2 = m.read(nxt, s2)
            speeds.append(new / (time.perf_counter() - t0))
    return prompt.shape[1] / read_s, statistics.median(speeds)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 1])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    from oazy import term
    dtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    term.header("Bee6 writing on the CPU", f"dense MLP vs sparse MLP · {a.dtype} · prompt {a.prompt}, {a.new} new words")
    torch.manual_seed(0)
    prompt = torch.randint(1, 49152, (1, a.prompt))
    rows, res = [], {}
    models = {"Bee6 (whole MLP)": build(False, dtype, a.run), "Bee6 + sparse MLP 10%": build(True, dtype, a.run)}
    for th in a.threads:
        torch.set_num_threads(th)
        for name, m in models.items():
            r, w = timed(m, prompt, a.new, a.repeats)
            res[(name, th)] = (r, w)
    for name, m in models.items():
        sp = m.cfg.sparse_mlp
        mlp_w = sum(b.mlp.up.weight.numel() + b.mlp.down.weight.numel() for b in m.blocks)
        per_word = sum(p.numel() for p in m.parameters()) - sum(t.weight.numel() for t in m.ve)
        if sp:
            b0 = m.blocks[0].mlp
            per_word -= mlp_w * (1 - b0.k / b0.nb)
        row = [name, f"{per_word / 1e6:.0f}M"]
        for th in a.threads:
            r, w = res[(name, th)]
            row += [f"{w:.1f}", f"{r:.0f}"]
        rows.append(row)
    head = ["", "weights read per word"] + [x for th in a.threads for x in (f"writing tok/s ({th} thr)", f"reading tok/s ({th} thr)")]
    term.table(head, rows, right=tuple(range(1, len(head))))
    for th in a.threads:
        d, s = res[("Bee6 (whole MLP)", th)][1], res[("Bee6 + sparse MLP 10%", th)][1]
        term.info(f"{th} thread(s): sparse writes {s / d:.2f}x as fast")
    term.info("weights read per word: the layers' weights plus the score head (the word table); the value tables are "
              "look-ups. The score head (49,152 x 1,024) is the same for both.")
    return res


if __name__ == "__main__":
    main()
