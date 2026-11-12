"""Keep, the whole experiment in one command: train the model that keeps what it reads and its plain twin on the same
streams, give both the same exam at growing gaps, write the report.

    python3 -m keep.run --data /workspace/data --sub opal                 # the real run (one GPU; both models, one after the other)
    python3 -m keep.run --data /workspace/data --sub opal --preset big    # ~120M each, 1.5B tokens each
    python3 -m keep.run --only mixed                                      # one model only (keep, full, wide or mixed); the report keeps the others
    python3 -m keep.run --only report                                     # examine the saved models again
    python3 -m keep.run --preset tiny --device cpu --workers 0 ...        # a CPU smoke test

The question it answers: a scene is told once, then G tokens of unrelated text go by, then the scene is asked about.
How often is the answer right, as G grows from 0 to 32,768? Two kinds of scene (keep/data.py): plain facts, where a
miss is a fact that was not kept, and Field1 stories, which need reasoning as well.

The twin's attention reaches 1,024 tokens, so past that it can only guess; its score with the story in view is the
ceiling. Keep is examined more strictly: it reads the story and the gap window by window, and then the questions as a
window of their own, so at every gap, 0 included, the answer can only come out of its stores.

The third model, Keep-wide ("wide"), is both at once: the twin's attention over the whole window, and Keep's stores.
It is examined twice: as a stream is read (a story still in view is seen), and out of its stores alone. Trained
plainly, its stores stay unused: attention answers whatever is in view. The fourth, "mixed", is Keep-wide trained with
its attention cut to Keep's 128 tokens on half of the steps (--narrow), and read with the whole window.

Writes <runs>/<name>/: keep/, full/, wide/ and mixed/ (ckpt.pt, log.jsonl), results.json, report.md. A model whose ckpt.pt says it
finished is not trained again; an unfinished one starts over (runs are short).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .model import Keep, KeepConfig
from . import data as D

PRESETS = {
    "tiny": dict(cfg=dict(d=64, n_layer=4, n_head=4, max_len=128), keep=dict(window=32, stores=(2, 4), side=16, k=4, dv=16, d_addr=16, chunk=16),
                 batch=8, tokens=3e5, gap_max=256, gaps=(0, 128, 512), exam_n=24, lr=2e-3),
    # the CPU pilot: plain facts only, scene after scene, so that a 1.3M model can learn to look a fact up at all
    "pilot": dict(cfg=dict(d=128, n_layer=4, n_head=4, max_len=256), keep=dict(window=32, stores=(2, 3), side=64, k=8, dv=32, d_addr=32, chunk=32),
                  batch=16, tokens=1.2e7, gap_max=256, gaps=(0, 64, 256, 1024, 4096), exam_n=96, lr=2e-3, asked_weight=4.0, stream=dict(p_scene=1.0, p_ledger=1.0)),
    "base": dict(cfg=dict(d=512, n_layer=8, n_head=8, max_len=1024), keep=dict(window=128, stores=(3, 6), side=128, k=8, dv=64, d_addr=64, chunk=64),
                 batch=32, tokens=6e8, gap_max=4096, gaps=(0, 512, 2048, 8192, 32768), exam_n=512, lr=6e-4, asked_weight=4.0),
    "big": dict(cfg=dict(d=768, n_layer=12, n_head=12, max_len=1024), keep=dict(window=128, stores=(4, 8, 11), side=256, k=8, dv=64, d_addr=64, chunk=64),
                batch=32, tokens=1.5e9, gap_max=4096, gaps=(0, 512, 2048, 8192, 32768), exam_n=512, lr=4e-4, asked_weight=4.0),
}
ARMS = ("keep", "full", "wide", "mixed")   # Keep (attention reaches `window`, stores), its plain twin, both at once (the whole window and
WIDE = ("wide", "mixed")                   # the stores), and both at once trained with the attention cut on a share of the steps


def runs_dir():
    try:
        from oazy.data import default_dir
        return default_dir("runs")
    except Exception:
        return os.environ.get("OAZY_RUNS", "/workspace/oazy_runs")


def make(arm: str, preset: dict, vocab: int, **over) -> Keep:
    cfg = dict(preset["cfg"], vocab_size=vocab)
    if arm == "keep" or arm in WIDE:
        cfg.update(preset["keep"])
        cfg.update({k: v for k, v in over.items() if v is not None})
    if arm in WIDE:
        cfg["window"] = 0
    return Keep(KeepConfig(**cfg))


def amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


# ---------------------------------------------------------------- the exam
@torch.no_grad()
def exam(model: Keep, enc, text, gaps, n, device, batch=64, seed=0, say=None, alone=True):
    """-> {gap: {"groups": {facts|changed|field1: {"right": %, "chance": %, "n", "loss": on the answer's piece}}, "families": {kind: [right, n]},
                 "write": {"story": [per store], "text": [per store]} (stores, read alone: mean write strength on each)}}
    A model with stores reads everything, window by window, its stores carried along. alone=True: the questions are a
    window of their own, so the answers come out of the stores alone. alone=False: the questions end the last window,
    as in a stream, so whatever attention still reaches is in view too (the reading for a model whose attention spans
    the window). A model without stores sees the last max_len tokens and nothing else."""
    model.eval()
    T = model.cfg.max_len
    out = {}
    for gap in gaps:
        prefix, block, marks, spans = D.exam_set(enc, text, gap, n, q_max=min(320, T // 2), seed=seed)
        got = np.zeros(len(marks), dtype=bool)
        nll = np.zeros(len(marks))
        mrow = np.array([m[0] for m in marks], dtype=np.int64)
        wsum = {i: [0.0, 0.0] for i in model.store}                 # write strength summed on the story, on the other text
        wn = [0, 0]
        for lo in range(0, n, batch):
            pre = torch.from_numpy(prefix[lo:lo + batch]).to(device)
            blk = torch.from_numpy(block[lo:lo + batch]).to(device)
            with amp(device):
                if model.store and not alone:                       # as a stream: all but the last window, then the last
                    seq = torch.cat([pre, blk], 1)
                    st = model.init_state(pre.size(0), device)
                    cut = max(0, seq.size(1) - T)
                    for c in range(0, cut, T):
                        _, st, _ = model.hidden(seq[:, c:min(cut, c + T)], st)
                    h, _, _ = model.hidden(seq[:, cut:], st)
                    off = seq.size(1) - cut - blk.size(1)
                elif model.store:                                   # read it all, in windows, the stores carried along
                    st = model.init_state(pre.size(0), device)
                    sp = torch.from_numpy(spans[lo:lo + batch]).to(device)
                    pos = torch.arange(pre.size(1), device=device)[None]
                    story = (pos >= sp[:, :1]) & (pos < sp[:, 1:])
                    for c in range(0, pre.size(1), T):
                        _, st, beta = model.hidden(pre[:, c:c + T], st)
                        m = story[:, c:c + T]
                        for i, b in beta.items():
                            wsum[i][0] += float(b[m].sum())
                            wsum[i][1] += float(b[~m].sum())
                        wn[0] += int(m.sum())
                        wn[1] += int((~m).sum())
                    h, _, _ = model.hidden(blk, st)
                    off = 0
                else:                                               # the twin: the last max_len tokens are all it can see
                    seq = torch.cat([pre, blk], 1)[:, -T:]
                    h, _, _ = model.hidden(seq)
                    off = seq.size(1) - blk.size(1)
            for m in np.nonzero((mrow >= lo) & (mrow < lo + pre.size(0)))[0]:
                r, p, cands, right, _ = marks[m]
                sc = model.scores_at(h, torch.tensor([r - lo], device=device), torch.tensor([off + p], device=device))[0]
                got[m] = int(sc[torch.tensor(cands, device=device)].argmax()) == right
                nll[m] = float(torch.logsumexp(sc, 0) - sc[cands[right]])
        groups, fam = {}, {}
        for m, (_, _, cands, _, kind) in enumerate(marks):
            g = groups.setdefault(D.group(kind), [0, 0, 0.0, 0.0])
            g[0] += int(got[m])
            g[1] += 1
            g[2] += 1 / len(cands)
            g[3] += nll[m]
            a = fam.setdefault(kind, [0, 0])
            a[0] += int(got[m])
            a[1] += 1
        res = {"groups": {k: {"right": round(100 * g[0] / g[1], 1), "chance": round(100 * g[2] / g[1], 1), "n": g[1],
                             "loss": round(g[3] / g[1], 3)} for k, g in groups.items()},
               "families": fam}
        if model.store and alone:
            res["write"] = {"story": [round(w[0] / max(1, wn[0]), 3) for w in wsum.values()],
                            "text": [round(w[1] / max(1, wn[1]), 3) for w in wsum.values()]}
        out[str(gap)] = res
        if say:
            say(f"    gap {gap:>6,}: " + "   ".join(f"{k} {v['right']}% of {v['n']} (guessing {v['chance']}%, loss {v['loss']:.2f})" for k, v in sorted(res["groups"].items())))
    model.train()
    return out


@torch.no_grad()
def text_loss(model: Keep, feed_rows, device, windows=6):
    """Loss on held-out streams (the same for both models), the stores carried from window to window."""
    model.eval()
    st, tot = None, 0.0
    for w, _ in feed_rows[:windows]:
        x = torch.from_numpy(w).to(device)
        with amp(device):
            loss, st, _ = model.loss(x[:, :-1], x[:, 1:], st, plain=True)
        tot += float(loss)
    model.train()
    return tot / min(windows, len(feed_rows))


# ---------------------------------------------------------------- training
def train(arm, preset, args, enc, device, out: Path, say):
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt.pt"
    if ck.exists():
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        if sd.get("done"):
            say(f"[{arm}] already trained ({sd['tokens']:,} tokens): not trained again")
            m = Keep(KeepConfig.from_dict(sd["config"]))
            m.load_state_dict(sd["model"])
            return m.to(device), sd.get("summary", {})
    torch.manual_seed(args.seed)
    model = make(arm, preset, enc.n_vocab, write_cost=args.write_cost, side=args.side, read=args.read, chunk=args.chunk,
                 stores=tuple(int(i) for i in args.stores.split(",")) if args.stores else None).to(device)
    T, B = model.cfg.max_len, args.batch or preset["batch"]
    total = int(args.tokens or preset["tokens"])
    steps = max(1, total // (B * T))
    lr = args.lr or preset["lr"]
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": rest, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    warm = max(10, steps // 50)
    how = preset.get("stream", {})
    feed = D.Feed(Path(args.data), enc, B, T, preset["gap_max"], seed=args.seed, workers=args.workers, **how)
    held = D.Feed(Path(args.data), enc, min(B, 16), T, preset["gap_max"], seed=args.seed + 999, workers=0, split="val", **how)
    held_rows = [held.next() for _ in range(6)]
    text = D.Text(Path(args.data), "val")
    narrow = (0.5 if args.narrow is None else args.narrow) if arm == "mixed" else 0.0      # the share of steps with the attention cut
    cut, pick = preset["keep"]["window"], random.Random(f"{args.seed}|narrow")
    say(f"[{arm}] {model.n_params() / 1e6:.1f}M weights, attention reaches {model.cfg.window or T} tokens"
        + (f" ({cut} on {narrow:.0%} of the steps)" if narrow else "") +
        f", {len(model.store)} stores of {model.cfg.side ** 2:,} rows; {steps:,} steps of {B} x {T} = {steps * B * T / 1e6:.0f}M tokens")
    if args.compile and device.type == "cuda":
        model.speed_up()
    log = open(out / "log.jsonl", "a")
    state = model.init_state(B, device) if model.store else None
    t0 = time.time()
    last, seen = t0, (0, t0)
    run = run_a = None
    mids = {int(steps * f) for f in (0.25, 0.5, 0.75)} if steps >= 40 else set()
    for step in range(1, steps + 1):
        cur = lr * (step / warm if step <= warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups:
            g["lr"] = cur
        w, what = feed.next()
        w = torch.from_numpy(w).to(device, non_blocking=True)
        asked = torch.from_numpy(what[:, 1:] == D.ASKED).to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        model.reach = cut if narrow and pick.random() < narrow else None
        try:
            with amp(device):
                loss, new, info = model.loss(w[:, :-1], w[:, 1:], state, asked=asked, asked_weight=preset.get("asked_weight", 1.0))
            loss.backward()
        except Exception as e:                      # a compile that fails must not cost the run: the same step again, plain
            if model._fast is None:
                raise
            say(f"[{arm}] torch.compile failed ({type(e).__name__}: {str(e)[:200]}); going on without it")
            model.speed_up(False)
            opt.zero_grad(set_to_none=True)
            with amp(device):
                loss, new, info = model.loss(w[:, :-1], w[:, 1:], state, asked=asked, asked_weight=preset.get("asked_weight", 1.0))
            loss.backward()
        state = new
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if state is not None:
            state = model.detach(state)
        lv = float(info["lm"])
        if not math.isfinite(lv):
            raise SystemExit(f"[{arm}] the loss is not a number at step {step}: lower --lr")
        run = lv if run is None else 0.98 * run + 0.02 * lv
        if "asked" in info:
            run_a = float(info["asked"]) if run_a is None else 0.98 * run_a + 0.02 * float(info["asked"])
        if step % args.every == 0 or step == steps or time.time() - last > 60:
            last = time.time()
            rate = (step - seen[0]) * B * T / max(1e-9, time.time() - seen[1])     # since the last line (not the compile, not an exam)
            seen = (step, time.time())
            mem = f"  {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if device.type == "cuda" else ""
            wr = " ".join(f"{float(v):.2f}" for v in info["write"])
            say(f"[{arm}] step {step:,}/{steps:,}  {step * B * T / 1e6:.1f}M tokens  loss {run:.3f}  asked {run_a or 0:.3f}  {rate / 1e3:.0f}k tok/s"
                + (f"  write strength {wr}" if wr else "") + mem + f"  left {(steps - step) * B * T / rate / 60:.0f} min")
            log.write(json.dumps({"kind": "train", "step": step, "tokens": step * B * T, "loss": run, "asked": run_a, "tok_s": rate,
                                  "write": [float(v) for v in info["write"]]}) + "\n")
            log.flush()
        model.reach = None
        if step in mids:
            say(f"[{arm}] a short exam at {step * B * T / 1e6:.0f}M tokens:")
            gs = [g for g in preset["gaps"] if g <= 2 * T][:3]
            mid = exam(model, enc, text, gs, max(24, preset["exam_n"] // 2), device, seed=1, say=say, alone=arm not in WIDE)
            log.write(json.dumps({"kind": "exam", "step": step, "tokens": step * B * T,
                                  "exam": {g: {k: [x["right"], x["loss"]] for k, x in v["groups"].items()} for g, v in mid.items()}}) + "\n")
            log.flush()
            seen = (step, time.time())
    feed.close()
    secs = time.time() - t0
    summary = {"params": model.n_params(), "tokens": steps * B * T, "secs": round(secs), "tok_s": round(steps * B * T / secs),
               "loss": run, "held_out_loss": round(text_loss(model, held_rows, device), 4)}
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(), "kind": "keep", "arm": arm, "done": True,
                "tokens": steps * B * T, "summary": summary}, ck)
    say(f"[{arm}] trained: {summary['tokens'] / 1e6:.0f}M tokens in {secs / 60:.0f} min ({summary['tok_s'] / 1e3:.0f}k tok/s), "
        f"held-out loss {summary['held_out_loss']:.3f}")
    return model, summary


GROUPS = (("facts", "plain facts"), ("changed", "facts changed after telling"), ("field1", "Field1 stories"))
NAMES = {"keep": "Keep", "full": "Twin", "wide": "Keep-wide", "wide_alone": "Keep-wide, stores alone",
         "mixed": "Keep-wide mixed", "mixed_alone": "Keep-wide mixed, stores alone"}
ROWS = ("keep", "full", "wide", "wide_alone", "mixed", "mixed_alone")


def report(res: dict) -> str:
    gaps = res["gaps"]
    L = ["# Keep: does a model keep what it read once?", "",
         "A scene is told once, then G tokens of unrelated text go by, then it is asked about. Share of questions answered "
         "right (the model's top choice among each question's choices):", "",
         "| Model | Asked about | " + " | ".join(f"G = {g:,}" for g in gaps) + " |", "|---|---|" + "---|" * len(gaps)]

    def cell(ex, g, key, what):
        v = ex.get(str(g), {}).get("groups", {}).get(key)
        return f"{v[what]}%" if v else "-"
    for arm in ROWS:
        for key, name in GROUPS:
            if arm in res["exam"]:
                L.append(f"| {NAMES[arm]} | {name} | " + " | ".join(cell(res["exam"][arm], g, key, "right") for g in gaps) + " |")
    any_arm = next(iter(res["exam"].values()))
    for key, name in GROUPS:
        L.append(f"| Guessing | {name} | " + " | ".join(cell(any_arm, g, key, "chance") for g in gaps) + " |")
    L += ["", f"The twin sees the last {res['window_full']:,} tokens: its score while the story is still in view is the ceiling, and past "
          "that gap it can only guess. Keep reads the questions as a window of their own, so at every gap, 0 included, its "
          f"answers come out of its stores alone. Keep trained on gaps up to {res['gap_max']:,}; no gradient spans more than "
          f"one window of {res['window_full']:,}.", ""]
    if "wide" in res["exam"] or "mixed" in res["exam"]:
        L += [f"Keep-wide is Keep with the twin's attention: it sees the last {res['window_full']:,} tokens and has the stores. Its first "
              "rows read the questions as a stream does, at the end of a window, so a story still in view is seen; \"stores alone\" "
              "reads them as Keep does, as a window of their own.", ""]
    if "mixed" in res["exam"]:
        L += ["Keep-wide mixed is the same model, trained with its attention cut to Keep's reach on a share of the steps (half, "
              "unless --narrow said otherwise), so that the stores are needed while it learns. It is read with the whole window.", ""]
    L += ["| Model | Weights | Tokens | Speed | Held-out loss (streams) |", "|---|---|---|---|---|"]
    for arm in ARMS:
        t = res["train"].get(arm)
        if t:
            L.append(f"| {NAMES[arm]} | {t['params'] / 1e6:.1f}M | {t['tokens'] / 1e6:.0f}M | {t['tok_s'] / 1e3:.0f}k tok/s | {t['held_out_loss']:.3f} |")
    for arm in ("keep", "wide_alone", "mixed_alone"):
        w = res["exam"].get(arm, {}).get(str(gaps[-1]), {}).get("write")
        if w:
            L += ["", f"{NAMES[arm].split(',')[0]}'s write strength (0 = writes nothing, 2 = the most), store by store: on the stories "
                  + ", ".join(f"{x:.2f}" for x in w["story"]) + "; on the ordinary text " + ", ".join(f"{x:.2f}" for x in w["text"]) + "."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="base", choices=list(PRESETS))
    ap.add_argument("--data", default=None, help="the data folder (default: $OAZY_DATA or /workspace/oazy_data)")
    ap.add_argument("--sub", default="opal", help="sub-folder of the data folder holding the text (opal; '' = the folder itself)")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--name", default="keep")
    ap.add_argument("--tokens", type=float, default=None, help="training tokens for each model")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--write-cost", type=float, default=None, help="Keep: what a write costs (default: the preset's)")
    ap.add_argument("--read", default=None, choices=["soft", "rows"], help="Keep: read every row softly, or only the k rows it writes")
    ap.add_argument("--chunk", type=int, default=None, help="Keep: tokens whose writes land together (default: the preset's; "
                    "no larger than the attention's reach, or the tokens between are seen by neither)")
    ap.add_argument("--stores", default=None, help="Keep: a store after each of these blocks, e.g. 6 or 3,6 (default: the preset's)")
    ap.add_argument("--narrow", type=float, default=None, help="mixed: the share of training steps with the attention cut to "
                    "Keep's reach (default 0.5)")
    ap.add_argument("--side", type=int, default=None, help="Keep: a store has side x side rows (default: the preset's)")
    ap.add_argument("--workers", type=int, default=8, help="processes making the streams (0 = in the trainer itself)")
    ap.add_argument("--compile", type=int, default=1, help="torch.compile for the blocks on a GPU (1/0)")
    ap.add_argument("--every", type=int, default=100, help="say where it is every this many steps")
    ap.add_argument("--exam-n", type=int, default=None, help="scenes per gap in the final exam")
    ap.add_argument("--only", default="all", choices=["all", "keep", "full", "wide", "mixed", "report"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")        # the stream workers are forked
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.data is None:
        from oazy.data import default_dir
        args.data = default_dir("data")
    if args.sub and (Path(args.data) / args.sub).exists():
        args.data = str(Path(args.data) / args.sub)
    from oazy import tok
    enc = tok.get("smollm2")
    preset = PRESETS[args.preset]
    out = Path(args.runs or runs_dir()) / args.name
    out.mkdir(parents=True, exist_ok=True)
    say = lambda s: print(s, flush=True)
    arms = [a for a in ARMS if args.only in ("all", "report", a)]
    res_path = out / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    res.update({"preset": args.preset, "gaps": list(preset["gaps"]), "gap_max": preset["gap_max"], "window_full": preset["cfg"]["max_len"]})
    res.setdefault("train", {})
    res.setdefault("exam", {})
    text = D.Text(Path(args.data), "val")
    for arm in arms:
        if args.only == "report" and not (out / arm / "ckpt.pt").exists():
            continue
        model, summary = train(arm, preset, args, enc, device, out / arm, say)
        if summary:
            res["train"][arm] = summary
        say(f"[{arm}] the exam:")
        ex = exam(model, enc, text, preset["gaps"], args.exam_n or preset["exam_n"], device, say=say, alone=arm not in WIDE)
        ex_alone = None
        if arm in WIDE:
            say(f"[{arm}] the exam again, out of the stores alone:")
            ex_alone = exam(model, enc, text, preset["gaps"], args.exam_n or preset["exam_n"], device, say=say)
        if res_path.exists():                                       # the other model may have been trained beside this one
            there = json.loads(res_path.read_text())
            for part in ("train", "exam"):
                res[part] = {**there.get(part, {}), **res[part]}
        res["exam"][arm] = ex
        if ex_alone is not None:
            res["exam"][arm + "_alone"] = ex_alone
        res_path.write_text(json.dumps(res, indent=1))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if res["exam"]:
        (out / "report.md").write_text(report(res))
        say("")
        say(report(res))
        say(f"written: {out / 'report.md'}")


if __name__ == "__main__":
    main()
