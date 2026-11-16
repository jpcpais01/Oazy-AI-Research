"""Rival: the ladder. How much bigger must a normal pretrained model be to reason as deep as the fed model?  (NEXT-TOKEN)

A pretrained SmolLM2 (135M, 360M or 1.7B weights: 2.7, 7 and 34 times the fed model's 50M) is fine-tuned on the same
scenes as a run of keep.think (by default the family "tales": the circuits' logic told about people, trained to depth
3 only) and sits the same exam: the same scenes never trained on, built to depth 1, 2, 3, 4, 6, 8, the question
marked on the answer's first piece among the scene's choices, the deepest question asked first, so that no answer
written before it is a step toward it. Nothing is written out between the story and the answer.

What the rival is given, so that a win of the fed model cannot be explained away:
  - its pretrained weights, and attention over the whole scene (the fed model's attention reaches 32 tokens)
  - the loss on the answers alone (the usual way to fine-tune for questions)
  - half of its training scenes told once and half told again and again as the fed model's are (its depth + 1 or + 2
    times), and the exam at 1, 2, 4, 8 and 12 tellings: whichever way of reading suits it
  - several learning rates (--lrs), each a training of its own; the table takes the best of them
  - training "until flat": a short exam every --every steps; it stops when the trained depths are right and depth 8
    has stopped rising, or when --tokens are read. The table gives the exam at the end AND the best depth 8 seen at
    any short exam on the way.

    python3 -m keep.rival --size 135m                  # -> <runs>/rival-135m/rival.json, rival.md
    python3 -m keep.rival --size 360m
    python3 -m keep.rival --size 1.7b
    python3 -m keep.rival --table --ours h4            # the ladder: ours (a keep.think run) beside the rivals found in <runs>
    bash runpod/ladder.sh h4                           # all of it in the tmux window "ladder" (what to paste on the pod)

Needs `transformers` and the model hub (the pod has both; --size tiny is a random stand-in for the self-tests).
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

from .run import amp, runs_dir
from .think import CIRCUIT_TRAIN, EXAM_SEED, OWN, TRAIN_SEED, scene_tokens

HUB = {"135m": "HuggingFaceTB/SmolLM2-135M", "360m": "HuggingFaceTB/SmolLM2-360M", "1.7b": "HuggingFaceTB/SmolLM2-1.7B"}
LRS = {"135m": "3e-5,1e-4,3e-4", "360m": "3e-5,1e-4,3e-4", "1.7b": "1e-5,3e-5,1e-4", "tiny": "1e-3"}
TOKENS = {"135m": 4e7, "360m": 4e7, "1.7b": 3e7, "tiny": 2e4}
READS = (1, 2, 4, 8, 12)
OURS = 50.6e6                                # the fed model's weights
ROOM = 1024                                  # a training scene (told several times, and its questions) is no longer than this


def load(size: str, vocab: int, device):
    """The pretrained model (size tiny: SmolLM2's shape, small, random)."""
    try:
        from transformers import AutoModelForCausalLM, LlamaConfig
    except ImportError:
        raise SystemExit("keep.rival needs transformers: python3 -m pip install -q transformers")
    if size == "tiny":
        c = LlamaConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=8192, tie_word_embeddings=True, rope_theta=100000.0)
        m = AutoModelForCausalLM.from_config(c, attn_implementation="sdpa")
    else:
        m = AutoModelForCausalLM.from_pretrained(HUB[size], attn_implementation="sdpa")
    have = m.get_input_embeddings().weight.shape[0]
    assert have == vocab, f"the model has {have} pieces, the tokenizer {vocab}: not SmolLM2's tokenizer"
    return m.to(device)


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def batch_of(rows, eot, device):
    """Scenes [(tokens, kinds, marks, reads)] padded on the right -> x (B, L), and the marks as (row, place, candidates, right, level, answer)."""
    L = max(len(r[0]) for r in rows)
    x = np.full((len(rows), L), eot, dtype=np.int64)
    marks = []
    for b, r in enumerate(rows):
        x[b, :len(r[0])] = r[0]
        marks += [(b, p, cands, right, lv, a) for p, cands, right, lv, a in r[2]]
    return torch.from_numpy(x).to(device), marks


def scenes(rng: random.Random, enc, families, tag):
    """Training scenes for ever: as the fed model's (the same generator, depth 3 at most), half told once, half told
    its depth + 1 or + 2 times. -> (tokens, kinds, marks, tellings)"""
    n = 0
    while True:
        f = rng.choice(families)
        sc = scene_tokens(enc, f, n, (TRAIN_SEED, tag), 1 if rng.random() < 0.5 else None, ROOM, rng=rng, most=4)
        n += 1
        if sc is not None and sc[2]:
            yield sc


@torch.no_grad()
def exam(model, enc, families, reads_list, n, device, batch=32, seed=0):
    """-> {family: {level: {reads: [right, asked, commonest answer's count]}}}: keep.think's exam (the same scenes for
    the same n and seed, the same marks), each scene whole in the model's view."""
    model.eval()
    res = {}
    for f in families:
        res[f] = {}
        for R in reads_list:
            rows = [sc for sc in (scene_tokens(enc, f, i, (EXAM_SEED, seed), R, 1 << 30) for i in range(n)) if sc is not None and sc[3] == R and sc[2]]
            rows.sort(key=lambda s: len(s[0]))
            for lo in range(0, len(rows), batch):
                x, marks = batch_of(rows[lo:lo + batch], enc.eot_token, device)
                with amp(device):
                    logits = model(input_ids=x, use_cache=False).logits
                for b, p, cands, right, lv, a in marks:
                    ok = int(logits[b, p, torch.tensor(cands, device=device)].argmax()) == right
                    for key in (lv, "all"):
                        if key is None:
                            continue
                        cell = res[f].setdefault(key, {}).setdefault(R, [0, 0, {}])
                        cell[0] += ok
                        cell[1] += 1
                        cell[2][a] = cell[2].get(a, 0) + 1
        for lv in res[f]:
            for R in res[f][lv]:
                c = res[f][lv][R]
                res[f][lv][R] = [c[0], c[1], max(c[2].values())]
    model.train()
    return res


def best(ex: dict, lv: str) -> float:
    """A level's share right (%) at its best number of tellings, over all the families of an exam."""
    got = [100 * c[0] / max(1, c[1]) for fam in ex.values() for c in fam.get(lv, {}).values()]
    return max(got) if got else float("nan")


def shallow(ex: dict) -> float:
    """The trained depths (1 to CIRCUIT_TRAIN): the worst of them, each at its best telling."""
    return min(best(ex, f"depth {d}") for d in range(1, CIRCUIT_TRAIN + 1))


def tune(size, lr, args, enc, device, out: Path, say):
    """One fine-tuning at one learning rate -> its result (also written to out/result.json; not trained again if there)."""
    done = out / "result.json"
    if done.exists():
        say(f"[{size} lr {lr:g}] already trained: not trained again")
        return json.loads(done.read_text())
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = load(size, enc.n_vocab, device)
    model.train()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95),
                            fused=device.type == "cuda")
    total = int(args.tokens or TOKENS[size])
    gen = scenes(random.Random(f"rival|{args.seed}"), enc, args.families, f"rival-{args.seed}")
    per = sum(len(next(gen)[0]) for _ in range(200)) / 200                  # tokens to a scene
    steps = max(4, int(total / (per * args.batch)))
    warm = max(2, steps // 20)
    every = args.every or max(1, steps // 20)
    say(f"[{size} lr {lr:g}] {n_params(model) / 1e6:.0f}M weights ({n_params(model) / OURS:.1f}x the fed model), about {steps:,} steps of {args.batch} scenes "
        f"({per:.0f} tokens a scene) = {total / 1e6:.0f}M tokens at most; a short exam every {every} steps")
    curve, seen, run, t0, last, stopped = [], 0, None, time.time(), time.time(), "tokens read"
    for step in range(1, steps + 1):
        cur = lr * (step / warm if step <= warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for g in opt.param_groups:
            g["lr"] = cur
        x, marks = batch_of([next(gen) for _ in range(args.batch)], enc.eot_token, device)
        seen += x.numel()
        rows = torch.tensor([m[0] for m in marks], device=device)
        at = torch.tensor([m[1] for m in marks], device=device)
        want = torch.tensor([m[2][m[3]] for m in marks], device=device)
        opt.zero_grad(set_to_none=True)
        with amp(device):
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[rows, at]         # the answers' places only
            loss = torch.nn.functional.cross_entropy(model.lm_head(h).float(), want)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lv = loss.item()
        if not math.isfinite(lv):
            stopped = "the loss was not a number"
            break
        run = lv if run is None else 0.95 * run + 0.05 * lv
        if time.time() - last > 60 or step == steps:
            last = time.time()
            say(f"[{size} lr {lr:g}] step {step:,}/{steps:,}  {seen / 1e6:.1f}M tokens  answer loss {run:.3f}  {seen / (time.time() - t0) / 1e3:.0f}k tok/s")
        if step % every == 0 or step == steps:
            ex = exam(model, enc, args.families, (1, 4), args.short_n, device, seed=1)
            pt = {"step": step, "tokens": seen, "loss": run, "shallow": shallow(ex), **{f"depth {d}": best(ex, f"depth {d}") for d in (4, 6, 8)}}
            curve.append(pt)
            say(f"[{size} lr {lr:g}] short exam at step {step:,}: trained depths {pt['shallow']:.0f}%, depth 4 / 6 / 8 = {pt['depth 4']:.0f} / {pt['depth 6']:.0f} / {pt['depth 8']:.0f}")
            if step >= 0.3 * steps and len(curve) >= 6 and min(c["shallow"] for c in curve[-3:]) >= 99 \
                    and max(c["depth 8"] for c in curve[-4:]) <= max(c["depth 8"] for c in curve[:-4]):
                stopped = "flat: the trained depths right, depth 8 no longer rising"
                break
    secs = time.time() - t0
    say(f"[{size} lr {lr:g}] the exam ({stopped}):")
    ex = exam(model, enc, args.families, READS, args.exam_n, device)
    for lvl in [f"depth {d}" for d in (1, 3, 4, 6, 8)]:
        say(f"    {lvl}: " + "  ".join(f"{R}x {100 * c[0] / max(1, c[1]):.0f}%" for fam in ex.values() for R, c in fam.get(lvl, {}).items()))
    res = {"size": size, "lr": lr, "params": n_params(model), "tokens": seen, "secs": round(secs), "tok_s": round(seen / max(1e-9, secs)), "stopped": stopped,
           "loss": run, "curve": curve, "exam": json.loads(json.dumps(ex))}
    done.write_text(json.dumps(res, indent=1))
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def report(res: dict) -> str:
    L = [f"# Rival: SmolLM2-{res['size'].upper()} fine-tuned on {', '.join(res['families'])}", "",
         f"{res['params'] / 1e6:.0f}M weights ({res['params'] / OURS:.1f}x the fed model), pretrained. Trained to depth {CIRCUIT_TRAIN} only, the loss on the answers; "
         "half of the scenes told once, half told again and again. Share right (%) at the best number of tellings "
         f"({', '.join(str(r) for r in READS)}); \"best seen\" is the best depth 8 at any short exam during training.", "",
         "| Learning rate | Tokens | Stopped | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 6 | Depth 8 | Depth 8, best seen |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lr, r in res["lrs"].items():
        seen = max([c["depth 8"] for c in r["curve"]] or [float("nan")])
        L.append(f"| {lr} | {r['tokens'] / 1e6:.0f}M | {r['stopped']} | " + " | ".join(f"{best(r['exam'], f'depth {d}'):.0f}" for d in (1, 2, 3, 4, 6, 8)) + f" | {seen:.0f} |")
    L += ["", f"The best learning rate (by depth 8, then depth 4): {res['best']}."]
    return "\n".join(L)


def pick(lrs: dict) -> str:
    return max(lrs, key=lambda k: (best(lrs[k]["exam"], "depth 8"), best(lrs[k]["exam"], "depth 4")))


def table(runs: Path, ours: str, arm="feed") -> str:
    """The ladder: the fed model (a keep.think run) and every rival found in the runs folder, smallest first."""
    rows, d8 = [], {}
    p = runs / ours / "think.json"
    if p.exists():
        t = json.loads(p.read_text())
        ex = t.get("exam", {}).get(arm)
        if ex:
            w = t.get("train", {}).get(arm, {}).get("params", OURS)
            d8["ours"] = best(ex, "depth 8")
            rows.append((w, f"| The fed model (`{ours}`) | {w / 1e6:.0f}M | 1x | no | " + " | ".join(f"{best(ex, f'depth {d}'):.0f}" for d in (1, 2, 3, 4, 6, 8)) + " | |"))
    for q in sorted(runs.glob("*/rival.json")):
        r = json.loads(q.read_text())
        b = r["lrs"][r["best"]]
        seen = max([c["depth 8"] for l in r["lrs"].values() for c in l["curve"]] or [float("nan")])
        d8[r["size"]] = max(best(b["exam"], "depth 8"), seen if seen == seen else 0)
        rows.append((r["params"], f"| SmolLM2-{r['size'].upper()} (`{q.parent.name}`, lr {r['best']}) | {r['params'] / 1e6:.0f}M | {r['params'] / OURS:.1f}x | yes | "
                     + " | ".join(f"{best(b['exam'], f'depth {d}'):.0f}" for d in (1, 2, 3, 4, 6, 8)) + f" | {seen:.0f} |"))
    L = ["# The ladder", "", f"Every model was trained on the same scenes to depth {CIRCUIT_TRAIN} only and sat the same exam (scenes never trained on, the deepest question "
         "first, nothing written out). Share right (%) at each model's best number of tellings. Depth 8: always giving the commonest answer scores about 55%.", "",
         "| Model | Weights | Size | Pretrained | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 6 | Depth 8 | Depth 8, best seen |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    L += [r for _, r in sorted(rows, key=lambda x: x[0])]
    if "ours" in d8 and len(d8) > 1:
        near = [s for s in ("135m", "360m", "1.7b") if s in d8 and d8[s] >= d8["ours"] - 5]
        L += ["", (f"The smallest rival within 5 points of the fed model at depth 8 (its best, at the end or seen on the way): SmolLM2-{near[0].upper()}." if near else
                   "No rival comes within 5 points of the fed model at depth 8.")
              + f" The fed model: {d8['ours']:.0f}%. " + ", ".join(f"SmolLM2-{s.upper()}: {v:.0f}%" for s, v in d8.items() if s != "ours") + "."]
    elif "ours" not in d8:
        L += ["", f"(No exam of the fed model found at {p}.)"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="135m", choices=list(HUB) + ["tiny"])
    ap.add_argument("--runs", default=None)
    ap.add_argument("--name", default=None, help="the run's folder (default: rival-<size>)")
    ap.add_argument("--families", default="tales", help="keep.think's own families, with commas: " + ", ".join(OWN))
    ap.add_argument("--tokens", type=float, default=None, help="training tokens at most, for each learning rate (default: 40M; 30M for 1.7b)")
    ap.add_argument("--lrs", default=None, help="learning rates, with commas; each is a training of its own")
    ap.add_argument("--batch", type=int, default=32, help="scenes a step")
    ap.add_argument("--every", type=int, default=0, help="steps between two short exams (default: a twentieth of the training)")
    ap.add_argument("--short-n", type=int, default=120, help="scenes of a short exam")
    ap.add_argument("--exam-n", type=int, default=600, help="scenes of the exam, for each number of tellings (keep.think's base preset: 600)")
    ap.add_argument("--table", action="store_true", help="train nothing: the ladder of --ours and the rivals found in the runs folder")
    ap.add_argument("--ours", default="h4", help="--table: the keep.think run of the fed model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    runs = Path(args.runs or runs_dir())
    say = lambda s: print(s, flush=True)
    if args.table:
        t = table(runs, args.ours)
        (runs / "ladder.md").write_text(t)
        say(t)
        say(f"written: {runs / 'ladder.md'}")
        return
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    from oazy import tok
    enc = tok.get("smollm2")
    args.families = [f for f in args.families.split(",") if f]
    for f in args.families:
        if f not in OWN:
            raise SystemExit(f"no family {f!r} of keep.think's own; there are: {', '.join(OWN)}")
    out = runs / (args.name or f"rival-{args.size}")
    out.mkdir(parents=True, exist_ok=True)
    res = {"size": args.size, "families": args.families, "lrs": {}}
    for lr in [float(v) for v in (args.lrs or LRS[args.size]).split(",") if v]:
        r = tune(args.size, lr, args, enc, device, out / f"lr-{lr:g}", say)
        res["lrs"][f"{lr:g}"] = r
        res["params"] = r["params"]
        res["best"] = pick(res["lrs"])
        (out / "rival.json").write_text(json.dumps(res, indent=1))
    (out / "rival.md").write_text(report(res))
    say("")
    say(report(res))
    say(f"written: {out / 'rival.md'}")


if __name__ == "__main__":
    main()
