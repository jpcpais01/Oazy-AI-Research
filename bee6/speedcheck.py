"""Bee6's speed check: what each speed idea buys, one row per idea, on the GPU it will train on.

    python3 -m bee6.speedcheck                                  # on the pod, the GPU free (~15-20 min, mostly compiling)
    python3 -m bee6.speedcheck --preset smoke --device cpu      # the self-test's quick version

The ladder, each row adding one idea to the row above (full size, one pass = 32 x 2,048 words, forward + backward):

    Bee5                         the model Bee6 is built on
    Bee6, every idea off         should be Bee5's speed (same maths)
    + idea 2: GPU use            q/k/v in one product, 4,096-word loss slices (and CUDA graphs, kept only if faster)
    + idea 3: FP8                the layers' products in FP8 (checked against bf16 first)
    + idea 4: Mixture of Depths  5 of 12 layers take the top 25% of words      = Bee6's defaults
    + idea 5: bloom              each stage timed, then averaged over the plan (20/20/20/40)

and the sparse MLP (bee6/sparse.py, off by default), on top of Bee6's defaults: + sparse_mlp at 10% (with Mixture of
Depths), the same without Mixture of Depths, and with blocks of 64; then the MLP alone, dense against sparse, so the
block kernels' real speed is seen apart from the rest of the layer.

Then the whole run's speed (early short windows, the mentor's passes, the optimizer), with and without bloom, and the
train line. "chip use" counts the arithmetic really done (a routed layer does a quarter of the work) against the
B200's bf16 peak, so FP8 can push it past what bf16 allows. Speed only: whether a trick costs quality is the race's
question (bee6/README.md).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from oazy import term
from oazy import train as T
from oazy.data import default_dir, few_cpu_threads
from oazy.profile import fake_batch, peak_tflops, sync

OFF = dict(fuse_qkv=0, loss_chunk=1024, fp8=0, mod_layers="", graphs=0)


def flops_per_token(model, Tw):
    """Training arithmetic per token actually done: 6 x each multiplying weight (word table once, as the score head;
    value tables are look-ups) + attention; a routed layer's share is mod_cap of that (its attention mod_cap squared)."""
    c = model.cfg
    head = 6 * model.embed.weight.numel()
    tot = head
    cap = getattr(c, "mod_cap", 1.0)
    mods = c.mod_set() if hasattr(c, "mod_set") else set()
    for i, b in enumerate(model.blocks):
        w = sum(p.numel() for n, p in b.named_parameters() if n.endswith(("q.weight", "k.weight", "v.weight", "out.weight",
                                                                          "up.weight", "down.weight")))
        if getattr(b, "sparse", False) and b.mlp.live:      # only the picked blocks multiply (+ the pickers)
            m = b.mlp
            w -= (m.up.weight.numel() + m.down.weight.numel()) * (1 - m.k / m.nb)
            w += m.pick_in.weight.numel() + m.pick_out.weight.numel()
        att = 6 * Tw * c.attn_heads * c.attn_head_dim
        tot += (6 * w * cap + att * cap * cap) if i in mods else (6 * w + att)
    return tot


def sparse_line(sp, cur, run_tps):
    """The whole run with --set sparse_mlp=0.10: the first 10% at Bee6's speed (dense), the rest at the sparse row's."""
    r = sp.get("sparse MLP: 10% of the MLP's blocks (with MoD)")
    if not r or not cur:
        return None
    from bee6.model import Bee6Config
    f = Bee6Config().sparse_from
    ratio = 1.0 / (f + (1 - f) * cur["tok_s"] / r["tok_s"])
    return f"about {term.rate(run_tps * ratio)}  (--set sparse_mlp=0.10: {ratio:.2f}x, dense for the first {f:.0%})"


def mlp_kernels(preset, dev, N, passes):
    """The MLP alone, forward + backward on N words: Bee6's dense MLP (bf16) against the sparse one's block queues."""
    from bee6.model import Bee6Config, MLP6
    from bee6.sparse import SparseMLP
    over = {}
    if preset != "big":
        over = dict(d_model=64, mlp_hidden=128, sparse_block=16)
    out, rows = {}, []
    cuda = dev.type == "cuda"
    dt = torch.bfloat16 if cuda else torch.float32
    for name, c in (("dense MLP", dict(over, fp8=0)), ("dense MLP (FP8)", dict(over, fp8=1)),
                    ("sparse 10%, blocks of 128", dict(over, fp8=0, sparse_mlp=0.10)),
                    ("sparse 10%, blocks of 64", dict(over, fp8=0, sparse_mlp=0.10, sparse_block=64 if preset == "big" else 8)),
                    ("sparse 20%, blocks of 128", dict(over, fp8=0, sparse_mlp=0.20))):
        if "FP8" in name and not cuda:
            continue
        try:
            cfg = Bee6Config(**c)
            m = (SparseMLP(cfg) if cfg.sparse_mlp else MLP6(cfg)).to(dev).train()
            if cfg.sparse_mlp:
                m.live = True
            x = torch.randn(N, cfg.d_model, device=dev, dtype=dt, requires_grad=True)
            fn = torch.compile(m) if cuda else m

            def one():
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                    y = fn(x)
                y.float().square().mean().backward()
            for _ in range(3):
                one()
            sync(dev)
            t0 = time.perf_counter()
            for _ in range(max(passes, 3)):
                one()
            sync(dev)
            ms = (time.perf_counter() - t0) / max(passes, 3) * 1000
            out[name] = ms
        except Exception as e:
            term.info(f"MLP alone, {name}: {type(e).__name__}: {str(e).splitlines()[0][:140] if str(e) else ''}")
        finally:
            torch._dynamo.reset()
    base = out.get("dense MLP")
    for name, ms in out.items():
        ideal = ""
        if "sparse" in name:
            share = 0.10 if "10%" in name else 0.20
            ideal = f"ideal {1 / share:.0f}x"
        rows.append([name, f"{ms:.2f}", f"{base / ms:.2f}x" if base else "-", ideal])
    term.header("the MLP alone", f"{N:,} words, forward + backward, one layer, {'bf16' if cuda else 'float32'}")
    term.table(["", "ms", "vs dense", ""], rows, right=(1, 2))
    term.info("the sparse MLP gathers each block's words into a queue and back: that moving around is the price paid for "
              "skipping 90% of the multiplying")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="big")
    ap.add_argument("--device", default=None)
    ap.add_argument("--micro", type=int, default=None, help="sequences per pass at 2,048 words (default 32; smoke 2)")
    ap.add_argument("--passes", type=int, default=6, help="timed passes per setting (after 2 warm-up passes)")
    ap.add_argument("--runs", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    few_cpu_threads(dev)
    cuda = dev.type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    gpu = torch.cuda.get_device_name() if cuda else "cpu"
    micro = a.micro or (32 if a.preset == "big" else 2)
    pk = peak_tflops(gpu)
    term.header("Bee6 speed check", gpu)
    from oazy.tok import eot_id
    t_start = time.time()
    notes = []

    def measure(kind, over=None, Tw=None, opt_too=False):
        """ms per pass (forward + backward, bf16), tokens/s, memory, chip use; None if it fails."""
        model = None
        try:
            torch.manual_seed(0)
            model, cfg = T.build(kind, a.preset, dict(over or {}))
            model.to(dev).train()
            if hasattr(model, "startup_check"):
                model.startup_check(dev, notes.append)
            if cuda:
                model.enable_compile()
            model.set_progress(1.0)                  # past the early 3-word lesson: the speed of most of the run
            Tw = Tw or cfg.max_len
            B = max(1, micro * cfg.max_len // Tw)
            x, y = fake_batch(B, Tw, cfg.vocab_size, eot_id(cfg.tokenizer), dev)

            def one():
                with T.amp(dev):
                    loss, _ = model.losses(x, y)
                loss.backward()
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                one()
            sync(dev)
            t0 = time.perf_counter()
            for _ in range(a.passes):
                one()
            sync(dev)
            ms = (time.perf_counter() - t0) / a.passes * 1000
            out = {"ms": ms, "tok_s": B * Tw / ms * 1000, "B": B, "T": Tw, "params": model.num_params(),
                   "mem": torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0, "flops": flops_per_token(model, Tw),
                   "fp8": getattr(model.cfg, "fp8", 0)}
            out["mfu"] = out["tok_s"] * out["flops"] / 1e12 / pk if pk else None
            if opt_too:
                opt = T.make_optimizer(model, 1.8e-3, dev, "muon", 0.03, 0.1)
                opt.step()
                sync(dev)
                t0 = time.perf_counter()
                for _ in range(3):
                    opt.step()
                sync(dev)
                out["opt_ms"] = (time.perf_counter() - t0) / 3 * 1000
            return out
        except Exception as e:
            term.info(f"{kind} {over or ''}: {type(e).__name__}: {str(e).splitlines()[0][:140] if str(e) else ''}")
            return None
        finally:
            del model
            if cuda:
                torch.cuda.empty_cache()
            torch._dynamo.reset()

    def row(name, r, ref=None, note=""):
        if r is None:
            return [name, "-", "-", "-", "-", "failed"]
        rel = f"{r['tok_s'] / ref['tok_s']:.2f}x Bee5" if ref else ""
        return [name, f"{r['ms']:.0f}", term.rate(r["tok_s"]), f"{r['mfu'] * 100:.0f}%" if r.get("mfu") else "-",
                f"{r['mem']:.0f} GB", " · ".join(s for s in (rel, note) if s)]

    res = {"gpu": gpu, "micro": micro}
    b5 = measure("bee5", opt_too=True)
    if b5 is None:
        raise SystemExit("Bee5 does not run on this device")
    rows = [row(f"Bee5 ({b5['params'] / 1e6:.0f}M)", b5)]
    off = measure("bee6", OFF)
    rows.append(row("Bee6, every idea off", off, b5, "should match Bee5"))

    keep = dict(OFF)
    cur = off or b5

    def better(r, by=1.02):
        return r is not None and r["tok_s"] > cur["tok_s"] * by

    # idea 2
    s2 = dict(keep, fuse_qkv=1, loss_chunk=4096)
    r2 = measure("bee6", s2)
    if r2 is not None and (better(r2, 1.0) or cuda is False):
        keep, cur = s2, r2
    rows.append(row("+ 2: one q/k/v product, 4,096-word slices", r2, b5, "kept" if keep is s2 else "not faster"))
    if cuda:
        sg = dict(keep, graphs=1)
        rg = measure("bee6", sg)
        kept = better(rg)
        if kept:
            keep, cur = sg, rg
        rows.append(row("  + CUDA graphs", rg, b5, "kept (2%+ faster)" if kept else "not faster: off"))
    # idea 3
    best3 = None
    for mode in (1, 2):
        s3 = dict(keep, fp8=mode)
        r3 = measure("bee6", s3)
        ran = r3 is not None and r3["fp8"] == mode
        label = "+ 3: FP8 (weight gradient bf16)" if mode == 1 else "  FP8 for the weight gradient too"
        rows.append(row(label, r3, b5, "" if ran or r3 is None else "FP8 did not run here: this is bf16 (see notes)"))
        if ran and better(r3) and (best3 is None or r3["tok_s"] > best3[1]["tok_s"] * 1.02):
            best3 = (s3, r3)
    if best3:
        keep, cur = best3
        rows[-2 if keep["fp8"] == 1 else -1][-1] += " · kept"
    # idea 4: Bee6's routing defaults
    from bee6.model import Bee6Config
    d6 = Bee6Config()
    s4 = dict(keep, mod_layers=d6.mod_layers, mod_cap=d6.mod_cap)
    r4 = measure("bee6", s4, opt_too=True)
    rows.append(row(f"+ 4: Mixture of Depths ({len(d6.mod_set())} layers, top {d6.mod_cap:.0%})", r4, b5,
                    "changes the model: the race says if it costs quality"))
    if r4 is not None:
        keep, cur = s4, r4
    # the sparse MLP (off by default): with and without Mixture of Depths, and finer blocks
    sp_rows, sp = [], {}
    for name, over in (("+ sparse MLP: 10% of the MLP's blocks (with MoD)", dict(keep, sparse_mlp=0.10)),
                       ("  sparse MLP 10%, Mixture of Depths off", dict(keep, sparse_mlp=0.10, mod_layers="")),
                       ("  sparse MLP 10%, blocks of 64 (with MoD)", dict(keep, sparse_mlp=0.10, sparse_block=64))):
        r = measure("bee6", over)
        sp[name.strip(" +")] = r
        note = (f"{r['tok_s'] / cur['tok_s']:.2f}x Bee6" if r and cur else "") + " · changes the model: the race says"
        sp_rows.append(row(name, r, b5, note))
    no_mod = measure("bee6", dict(keep, mod_layers=""))
    sp_rows.insert(1, row("  Bee6 with Mixture of Depths off (dense MLP)", no_mod, b5, "the row below's reference"))
    # idea 5: bloom, each stage
    from bee6.bloom import stage_config
    plan_cfg = Bee6Config(bloom="on")
    plan = plan_cfg.plan()
    stage_rows, stage_tps = [], []
    for k, (n, d, at) in enumerate(plan):
        if k == len(plan) - 1:
            stage_tps.append(cur["tok_s"])
            continue
        c = Bee6Config(**{**{kk: v for kk, v in keep.items()}})
        c.bloom = plan_cfg.bloom
        stage_config(c, k)
        over = dict(keep, d_model=c.d_model, n_layer=c.n_layer, attn_heads=c.attn_heads, attn_kv_heads=c.attn_kv_heads,
                    mlp_hidden=c.mlp_hidden, key_offset=c.key_offset, mod_layers=c.mod_layers if keep.get("mod_layers") else "",
                    backout_layer=c.backout_layer)
        if a.preset != "big":       # the test presets have their own small body: time it at a quarter of the width instead
            over = dict(keep)
        r = measure("bee6", over)
        stage_tps.append(r["tok_s"] if r else cur["tok_s"])
        stage_rows.append(row(f"  stage {k + 1}: {n} layers of {d}", r, b5, f"{plan[k + 1][2] - at:.0%} of the run"))
    spans = [(plan[i + 1][2] if i + 1 < len(plan) else 1.0) - plan[i][2] for i in range(len(plan))]
    bloom_tps = 1.0 / sum(s / t for s, t in zip(spans, stage_tps))
    rows += stage_rows
    rows += sp_rows
    rows.append([f"+ 5: bloom, averaged over the plan", "-", term.rate(bloom_tps), "-", "-",
                 f"{bloom_tps / b5['tok_s']:.2f}x Bee5 · off by default"])
    term.table(["", "ms/pass", "speed", "chip use", "memory", ""], rows, right=(1, 2, 3, 4))
    for n_ in dict.fromkeys(notes):
        term.info(n_)
    mlp_alone = mlp_kernels(a.preset, dev, micro * 2048 if a.preset == "big" else 512, a.passes)

    # the whole run: early windows, the mentor, the optimizer
    m0, cfg0 = T.build("bee6", a.preset, dict(keep))
    early = {cfg0.max_len: cur["tok_s"]}
    for Tw in sorted({m0.ctx_at((i + 0.5) / 100) for i in range(100)}):
        if Tw != cfg0.max_len:
            r = measure("bee6", keep, Tw=Tw)
            if r is not None:
                early[Tw] = r["tok_s"]
    mentor_ms = 0.0
    if cuda and a.preset == "big":
        try:
            from ember.mentor import Mentor
            mentor = Mentor(T.MENTOR["bee6"], dev, compile=True)
            x, _ = fake_batch(cur["B"], cur["T"], 49152, 0, dev)
            mask = m0.doc_mask(x)
            with torch.no_grad(), T.amp(dev):
                for _ in range(2):
                    mentor.hidden(x, mask)
                sync(dev)
                t0 = time.perf_counter()
                for _ in range(a.passes):
                    mentor.hidden(x, mask)
                sync(dev)
            mentor_ms = (time.perf_counter() - t0) / a.passes * 1000
            del mentor
        except Exception as e:
            term.info(f"mentor timing: {type(e).__name__}: {str(e)[:120]}")
    from ember.mentor import fade
    rc = T.RECIPE["bee6"]
    batch = T.PRESETS[a.preset].get("batch_bee6", 524288)
    opt_ms = (r4 or {}).get("opt_ms") or b5.get("opt_ms", 0)

    def run_speed(mentor_on=True, bloom=False):
        tot, grid = 0.0, [(i + 0.5) / 1000 for i in range(1000)]
        for q in grid:
            tps = early.get(m0.ctx_at(q), cur["tok_s"])
            if bloom:
                k = max(i for i, (_, _, at) in enumerate(plan) if at <= q)
                tps *= stage_tps[k] / cur["tok_s"]
            w = fade(q, rc["mentor_weight"], rc["mentor_until"], None, rc["mentor_floor"], rc["mentor_off"]) if mentor_on else 0
            pass_tok = cur["B"] * cur["T"]
            tot += (pass_tok / tps * 1000 + w * mentor_ms + opt_ms * pass_tok / batch) / pass_tok
        return 1000 / (tot / len(grid))
    run_tps, solo_tps, bloom_run = run_speed(True), run_speed(False), run_speed(True, True)
    chosen = {k: v for k, v in keep.items() if getattr(Bee6Config(), k) != v}
    sets = " ".join(f"{k}={v}" for k, v in chosen.items())
    line = (f"OAZY_DATA=/root/data bash runpod/train.sh bee6 --preset {a.preset} --tokens 10e9 --micro {micro}"
            + (f" --set {sets}" if sets else ""))
    term.header("result", color="green")
    term.rows([("Bee6, full size", term.c(term.rate(cur["tok_s"]), "bold", "green") + f"  ({cur['tok_s'] / b5['tok_s']:.2f}x Bee5"
                + (f", {cur['mfu'] * 100:.0f}% of the chip's bf16 arithmetic)" if cur.get("mfu") else ")")),
               ("Bee5", term.rate(b5["tok_s"]) + (f"  ({b5['mfu'] * 100:.0f}% of the chip)" if b5.get("mfu") else "")),
               ("optimizer step", f"{opt_ms:.0f} ms (once per {term.tokens(batch)} tokens)" if opt_ms else None),
               ("mentor", f"{mentor_ms:.0f} ms a pass when it teaches" if mentor_ms else None),
               ("whole run", term.c(f"about {term.rate(run_tps)}", "bold") + f"  (without the mentor {term.rate(solo_tps)})"),
               ("with bloom", f"about {term.rate(bloom_run)}  (--set bloom=on)"),
               ("with sparse MLP", sparse_line(sp, cur, run_tps)),
               ("10,000M tokens", f"about {10e9 / run_tps / 3600:.1f} hours ({10e9 / bloom_run / 3600:.1f} with bloom)"),
               ("0.5B race", f"about {0.5e9 / run_tps / 60:.0f} minutes"),
               ("train with", term.c(line, "bold")),
               ("took", f"{(time.time() - t_start) / 60:.0f} min")])
    res.update({"bee5": b5, "off": off, "keep": keep, "tok_s": cur["tok_s"], "run_tok_s": run_tps, "run_tok_s_no_mentor": solo_tps,
                "bloom_stage_tok_s": stage_tps, "bloom_tok_s": bloom_tps, "run_tok_s_bloom": bloom_run, "mentor_ms": mentor_ms,
                "early": early, "line": line, "notes": list(dict.fromkeys(notes)), "no_mod": no_mod,
                "sparse": sp, "mlp_alone": mlp_alone})
    out = Path(default_dir("runs") if a.runs is None else a.runs) / "speedcheck-bee6.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
        term.info(f"saved {out}")
    except OSError:
        pass
    return res


if __name__ == "__main__":
    main()
