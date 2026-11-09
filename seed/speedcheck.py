"""Seed's speed check, stage by stage, on the GPU it will train on: tokens a second for each bloom stage (4x384,
6x512, 9x640, 12x768, the full library read every time), with the optimizer step (NorMuon + Grokfast + the library's
row-wise Adam), memory, then the whole run's average speed, its hours and its cost. It grows the model through the
stages exactly as a run does (and checks each growth keeps the scores). Measures only, starts nothing; never run it
while a run trains on the same GPU. A few minutes on a B200 (most of it compiling).

    python3 -m seed.speedcheck                              # on the pod, the GPU free
    python3 -m seed.speedcheck --compile 0 --passes 3       # quicker, slower numbers
    python3 -m seed.speedcheck --preset smoke --device cpu  # the code path only

The numbers leave out the mentor (Bee4/Bee5's ~10%) and the short early windows (which are faster), so the real
run is within a few percent; its log prints each stage's real speed at every growth ([grow] lines).
"""
from __future__ import annotations

import argparse
import time

import torch

from oazy import term
from oazy import train as T
from oazy.data import few_cpu_threads
from oazy.profile import sync


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="big")
    ap.add_argument("--device", default=None)
    ap.add_argument("--micro", type=int, default=None, help="sequences of the longest window per pass (default 32; smoke 2)")
    ap.add_argument("--passes", type=int, default=6, help="timed passes per stage (after 2 warm-up passes)")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--tokens", type=float, default=10e9, help="the run to price (default 10,000M tokens)")
    ap.add_argument("--price", type=float, default=8.0, help="$ per GPU hour (B200 ~8)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(dev)
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    gpu = torch.cuda.get_device_name() if cuda else "cpu"
    micro = a.micro or (32 if a.preset == "big" else 2)
    term.header("Seed speed check", gpu)
    torch.manual_seed(0)
    model, cfg = T.build("seed", a.preset, T.parse_overrides(a.set))
    model.to(dev).train()
    Tw = cfg.max_len
    opt = model.make_optimizer(1e-4, dev, "muon", 0.03, 0.1)
    if a.compile and cuda:
        model.enable_compile()
    x = torch.randint(1, cfg.vocab_size, (micro, Tw), device=dev)
    y = torch.roll(x, -1, 1)
    model.set_progress(0.5)                # past the early 3-word lesson: the plain next-word lesson, as most of the run
    rates = []
    for s, (n, d, start) in enumerate(model.plan):
        if s:
            model.eval()
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                before = model(x[:2, :256])[0].float()
            model.grow_to(start)
            old, opt = opt, model.make_optimizer(1e-4, dev, "muon", 0.03, 0.1)
            T.carry_state(old, opt)
            model.carry_opt(old, opt)
            del old
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                after = model(x[:2, :256])[0].float()
            model.train()
            gap = (before - after).abs().max().item() / max(1e-6, before.abs().max().item())
            term.info(f"grew to {n} x {d}: scores kept to {gap:.1e} (bf16 rounding)")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(2 + a.passes):
            sync(dev)
            t0 = time.time()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                loss, _ = model.losses(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sync(dev)
            if i >= 2:
                times.append(time.time() - t0)
        rate = micro * Tw / (sum(times) / len(times))
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
        share = (model.plan[s + 1][2] if s + 1 < len(model.plan) else 1.0) - start
        rates.append((share, rate))
        term.rows([(f"stage {s + 1}: {n} x {d}", f"{rate / 1e3:,.0f}k tokens/s" + (f" · {mem:.0f} GB" if cuda else "")
                 + f" · {share * 100:.0f}% of the run · loss {loss.item():.2f}")])
    hours_per_tok = sum(sh / r for sh, r in rates) / 3600
    avg = 1 / (hours_per_tok * 3600)
    h = a.tokens * hours_per_tok
    term.rows([("whole run", f"{avg / 1e3:,.0f}k tokens/s on average · {a.tokens / 1e9:,.1f}B tokens in {h:.1f} h "
             f"(~${h * a.price:,.0f} at ${a.price:g}/h; plus ~10% for the mentor)")])
    return {"rates": rates, "avg": avg, "hours": h}


if __name__ == "__main__":
    main()
