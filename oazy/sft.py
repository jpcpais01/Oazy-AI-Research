"""The chat fine-tune: a short extra lesson on conversations only, scoring only the assistant's words.

    python3 -m oazy.sft --run bee3                      # -> runs/bee3-chat/ckpt.pt, about 10-20 min on a B200
    python3 -m bee3.chat --run bee3-chat --port 8002    # talk to it

Pre-training with a pinch of conversations teaches the *shape* of a dialogue. This pass teaches the manners:
the model reads whole conversations (smol-smoltalk, the same "User: / Assistant:" text as in training) and is
corrected only on what the assistant says, on the "User:" that hands the turn back, and on the <|endoftext|>
that ends the conversation, so it learns to answer, to stop, and to wait. The user's words are read but not
scored (their loss weight is 0: targets of -1 in oazy/lmloss.py). Windows pack several conversations; with
Bee3's document rule each one is read on its own.

The recipe is deliberately mild: about 100M tokens (a fraction of the conversations), a learning rate 1/10th of
pre-training's falling as a cosine to 0, the same Muon + AdamW split, no mentor. It changes how the model
answers, not what it knows; a held-out conversation loss before and after is printed and saved (sft.json).
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .data import default_dir, run_dir
from . import tok
from .train import load_model, make_optimizer, amp, save_ckpt, say, NICE

TURN = re.compile(r"(?:^|\n\n)(User|Assistant|System): ", re.S)


def render(text: str, enc):
    """One conversation -> (tokens, scored) exactly as the chat app feeds it: "User:" " ..." "\\n\\nAssistant:" " ..."
    scored = 1 on the assistant's words, on the "\\n\\nUser:" that follows an assistant turn, and (added by the
    caller) on the final <|endoftext|>. Returns None for text that is not a conversation."""
    parts = [(m.group(1), m.start(), m.end()) for m in TURN.finditer(text)]
    if not parts or not any(p[0] == "Assistant" for p in parts):
        return None
    ids, scored = [], []
    prev_assistant = False
    for i, (role, s, e) in enumerate(parts):
        end = parts[i + 1][1] if i + 1 < len(parts) else len(text)
        content = text[e:end].strip()
        frame = ("\n\n" if i else "") + role + ":"
        f_ids = enc.encode_ordinary(frame)
        c_ids = enc.encode_ordinary(" " + content) if content else []
        ids += f_ids + c_ids
        scored += [1 if (role == "User" and prev_assistant) else 0] * len(f_ids) + [1 if role == "Assistant" else 0] * len(c_ids)
        prev_assistant = role == "Assistant"
    return ids, scored


class Conversations:
    """Windows of T+1 tokens packed from whole conversations, with their scored mask, from a chat .bin file."""

    def __init__(self, path: Path, enc, T: int, batch: int, seed: int = 0, prefetch: int = 4, extra=(), share=0.0):
        self.extra, self.share, self.rng = list(extra), share, np.random.default_rng(seed + 7)
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.enc, self.T, self.batch = enc, T, batch
        self.eot = enc.eot_token
        self.cuts = np.flatnonzero(self.data == self.eot)
        self.order = np.random.default_rng(seed).permutation(len(self.cuts))
        self.pos = 0
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self.used = self.skipped = 0
        threading.Thread(target=self._fill, daemon=True).start()

    def _docs(self):
        while True:
            if self.extra and self.rng.random() < self.share:     # a "who are you?" conversation (oazy/identity.py)
                ids, sc = render(self.extra[int(self.rng.integers(len(self.extra)))], self.enc)
                yield ids + [self.eot], sc + [1]
                continue
            i = self.order[self.pos % len(self.order)]
            self.pos += 1
            a = 0 if i == 0 else self.cuts[i - 1] + 1
            text = self.enc.decode(self.data[a:self.cuts[i]].tolist())
            r = render(text, self.enc)
            if r is None:
                self.skipped += 1
                continue
            self.used += 1
            ids, sc = r
            yield ids + [self.eot], sc + [1]

    def _fill(self):
        docs = self._docs()
        buf_x, buf_m = [], []
        rows = []
        while True:
            while len(buf_x) < self.T + 1:
                ids, sc = next(docs)
                buf_x += ids
                buf_m += sc
            x = np.array(buf_x[:self.T + 1], dtype=np.int64)
            m = np.array(buf_m[:self.T + 1], dtype=np.int64)
            buf_x, buf_m = buf_x[self.T:], buf_m[self.T:]          # the window's last token starts the next one
            rows.append((x, m))
            if len(rows) == self.batch:
                xs = np.stack([r[0] for r in rows])
                ms = np.stack([r[1] for r in rows])
                self.q.put((torch.from_numpy(xs), torch.from_numpy(ms)))
                rows = []

    def get(self, device):
        x, m = self.q.get()
        x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
        y = x[:, 1:].masked_fill(m[:, 1:] == 0, -1)
        return x[:, :-1].contiguous(), y.contiguous()


@torch.no_grad()
def conv_loss(model, convs: Conversations, device, batches: int):
    """Loss on the assistant's words only, over a fixed number of held-out windows."""
    model.eval()
    tot, n = 0.0, 0
    for _ in range(batches):
        x, y = convs.get(device)
        with amp(device):
            _, loss, _ = model(x, y)
        tot += float(loss)
        n += 1
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="bee3", help="the pre-trained run to start from (runs/<run>/ckpt.pt)")
    ap.add_argument("--name", default=None, help="where the result goes (default: <run>-chat)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--tokens", type=float, default=1e8, help="conversation tokens to read (default 100M)")
    ap.add_argument("--batch", type=int, default=65536, help="tokens per step")
    ap.add_argument("--micro", type=int, default=None, help="windows per pass")
    ap.add_argument("--lr", type=float, default=None, help="AdamW rate (default: the run's / 10); Muon's scales with it")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=None, help="window length (default: the model's longest)")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--identity", type=float, default=0.02,
                    help="share of conversations that teach the model its own name and maker (oazy/identity.py); 0 = off")
    a = ap.parse_args()
    a.data, a.runs = a.data or default_dir("data"), a.runs or default_dir("runs")
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    from oazy.data import few_cpu_threads
    few_cpu_threads(device)
    src = run_dir(a.runs, a.run) / "ckpt.pt"
    model, st = load_model(src, device)
    if not hasattr(model, "losses"):
        raise SystemExit(f"{src} is not a Bee checkpoint: the chat fine-tune is for the Bee family")
    model.train()
    kind = st.get("kind", "bee")
    info = st.get("info") or {}
    enc = tok.for_model(model)
    tk = tok.name_of(model)
    root = Path(a.data) / ("smol" if tk == "smollm2" else "")
    chat = root / "chat"
    if not (chat / "train.bin").exists():
        raise SystemExit(f"no conversations in {chat}: prepare them first (oazy.tagus / oazy.retok)")
    T = a.ctx or model.cfg.max_len
    micro = a.micro or max(1, min(32 * 1024, a.batch) // T)
    accum = max(1, a.batch // (micro * T))
    lr = a.lr or (info.get("lr") or 1.2e-3) / 10
    muon_lr = (info.get("muon_lr") or 0.02) * lr / (info.get("lr") or 1.2e-3)
    steps = max(1, int(a.tokens) // (micro * accum * T))
    out = Path(a.runs) / (a.name or f"{a.run}-chat")
    out.mkdir(parents=True, exist_ok=True)
    say(f"{NICE.get(kind, kind)} chat fine-tune: {src} -> {out}\n  {steps} steps of {micro * accum * T:,} tokens "
        f"({micro} x {T} words, {accum} passes), AdamW {lr:.1e} / Muon {muon_lr:.1e}, cosine to 0, warm-up {a.warmup}")
    from .identity import conversations as who
    ident = who(kind, 2000, seed=a.seed) if a.identity > 0 else []
    if ident:
        say(f"  identity: {a.identity:.0%} of conversations teach it who it is, e.g. {ident[0]!r}")
    train = Conversations(chat / "train.bin", enc, T, micro, seed=a.seed, extra=ident, share=a.identity)
    val = Conversations(chat / "val.bin", enc, T, micro, seed=1) if (chat / "val.bin").exists() else None
    if a.compile and device.type == "cuda":
        model.enable_compile()
    opt = make_optimizer(model, lr, device, "muon", muon_lr, muon_wd=float(info.get("muon_wd") or 0.0))
    before = conv_loss(model, val, device, 8) if val else None
    if before is not None:
        say(f"  held-out conversation loss before: {before:.4f}")
    log = open(out / "sft.jsonl", "a")
    t0 = time.time()
    for step in range(steps):
        cur = lr * (step + 1) / a.warmup if step < a.warmup else lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = cur * g.get("lr_scale", 1.0)
        tot = 0.0
        for _ in range(accum):
            x, y = train.get(device)
            with amp(device):
                _, loss, _ = model(x, y)
            (loss / accum).backward()
            tot += float(loss.detach()) / accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if hasattr(model, "after_step"):            # Coral: keep the experts' work even
            model.after_step()
        if (step + 1) % 10 == 0 or step + 1 == steps:
            el = time.time() - t0
            say(f"step {step + 1:5d}/{steps} | loss {tot:.4f} | lr {cur:.2e} | gnorm {float(gn):.2f} | "
                f"{(step + 1) * micro * accum * T / el / 1e3:.0f}k tok/s | conversations used {train.used:,}", flush=True)
            log.write(json.dumps({"step": step + 1, "loss": tot, "lr": cur, "elapsed": el}) + "\n")
            log.flush()
    after = conv_loss(model, val, device, 8) if val else None
    meta = {k: v for k, v in st.items() if k not in ("model", "opt")}
    meta["sft"] = {"from": str(src), "tokens": steps * micro * accum * T, "steps": steps, "lr": lr, "muon_lr": muon_lr,
                   "conv_loss_before": before, "conv_loss_after": after, "identity": a.identity, "skipped_docs": train.skipped, "minutes": (time.time() - t0) / 60}
    class _Opt:                                    # save_ckpt wants an optimizer; the fine-tune's is not worth keeping
        @staticmethod
        def state_dict():
            return {}
    save_ckpt(type("R", (), {"dir": out})(), model, _Opt, meta)
    (out / "sft.json").write_text(json.dumps(meta["sft"], indent=1))
    say(f"done in {meta['sft']['minutes']:.1f} min. held-out conversation loss "
        f"{before if before is None else round(before, 4)} -> {after if after is None else round(after, 4)}. "
        f"checkpoint: {out / 'ckpt.pt'}\n  talk to it: python3 -m {kind}.chat --run {out.name} --port 8002")


if __name__ == "__main__":
    main()
