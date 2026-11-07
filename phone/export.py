"""Write one of our models as a phone model file (.oazy) for the Oazy phone app (phone/README.md).

    python3 -m phone.export --run coral                   # runs/coral/ckpt.pt -> runs/coral/phone/Coral_1B-q8.oazy
    python3 -m phone.export --run coral --quant q4        # smaller and faster, a little less exact
    python3 -m phone.export --run bee2                    # Bee_v2_0.2B (also bee3, frog, coral-chat, ...)

It runs on the CPU only (no GPU, safe next to a training run; it reads the last saved checkpoint and never touches
the running process) and writes, in one file:
  - every weight grid in 8 bits (q8) or, with --quant q4, the big MLP and expert grids in 4 bits and the rest in 8,
    in groups of 32 numbers with one float scale each; norms, biases and the router stay float32;
  - SmolLM2's tokenizer (pieces, merge list, special pieces);
  - a self-check: some texts with the pieces SmolLM2's own tokenizer makes of them, and PyTorch's scores after a short
    chat prompt, so the phone can prove it reads the file exactly like PyTorch does (the app's "Check" button).

The file layout (the engine, phone/engine/oazy.cpp, reads it): b"OAZYPHN1", the header's length (u64), the header
(text lines "key value" and "tensor name type rows cols offset bytes"), then the data, each tensor 64-byte aligned,
offsets counted from the start of the data (the first 64-byte boundary after the header).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch

T_F32, T_Q8, T_Q4, T_BLOB = 0, 1, 2, 9
NAMES = {"bee2": "Bee_v2_0.2B", "bee3": "Bee3", "frog": "Frog", "coral": "Coral_1B", "bee4": "Bee4-200M", "bee": "Bee0.2B"}
ACCENTS = {"bee2": "#ffb347", "bee3": "#ffb347", "bee": "#ffb347", "frog": "#4cf2c4", "coral": "#ff8a78", "bee4": "#ffb347"}

# texts for the tokenizer check: plain English, contractions, digits, spacing, code, other scripts, emoji, chat
TEST_TEXTS = [
    "Hello world! How are you today?",
    "User: What's the capital of France?\n\nAssistant: The capital of France is Paris.",
    "I'll say they're sure we'd've done it, didn't you? It's John's.",
    "In 2024, 3.14159 * 1,000,000 = 3141590 and 0xFF is 255.",
    "  two  spaces,\ttabs\t\tand\n\n\nnew lines   \n end   ",
    "def f(x):\n    return [i ** 2 for i in range(x) if i % 2 == 0]  # squares\n",
    "Olá, João! Ça va? Straße, naïve café, ¿Qué tal?",
    "Привет мир. 你好，世界。こんにちは。안녕하세요. مرحبا بالعالم",
    "Emoji: 😀🎉👍🏽 and symbols: ©®™ — “quotes” … ½ ² ³ ٣٤ ①",
    "<|endoftext|>A new document starts here.<|im_start|>user\nhi<|im_end|>",
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG; the quick brown fox.",
    "Mixed123abc456 x1y2z3 v2.0.1-beta+build.7 user@example.com https://oazy.ai/path?q=1&r=2",
    "    ",
    "a",
    "'s 're 've 'm 'll 'd 'S 'RE",
]
REF_PROMPT = "User: What is the capital of France?\n\nAssistant:"


# ------------------------------------------------------------------------------------------ packing
def q8_bytes(w: torch.Tensor) -> bytes:
    R, C = w.shape
    g = w.float().reshape(R, C // 32, 32)
    s = g.abs().amax(-1) / 127.0
    q = (g / s.clamp_min(1e-30)[..., None]).round().clamp(-127, 127).to(torch.int8)
    return q.numpy().tobytes() + s.numpy().astype(np.float32).tobytes()


def q4_bytes(w: torch.Tensor) -> bytes:
    """4-bit, symmetric (-7..7) in groups of 32; byte j of a group holds value j (low half) and value 16 + j (high)."""
    R, C = w.shape
    g = w.float().reshape(R, C // 32, 32)
    s = g.abs().amax(-1) / 7.0
    q = ((g / s.clamp_min(1e-30)[..., None]).round().clamp(-7, 7) + 8).to(torch.uint8)
    packed = q[..., :16] | (q[..., 16:] << 4)
    return packed.numpy().tobytes() + s.numpy().astype(np.float32).tobytes()


def dequant(w: torch.Tensor, kind: int) -> torch.Tensor:
    """What the phone will compute with (for the error report)."""
    R, C = w.shape
    g = w.float().reshape(R, C // 32, 32)
    top = 127.0 if kind == T_Q8 else 7.0
    s = g.abs().amax(-1) / top
    return ((g / s.clamp_min(1e-30)[..., None]).round().clamp(-top, top) * s[..., None]).reshape(R, C)


class Writer:
    def __init__(self):
        self.cfg: dict[str, str] = {}
        self.items = []           # (name, type, rows, cols, make_bytes)

    def set(self, k, v):
        self.cfg[k] = str(v).replace("\n", " ")

    def mat(self, name, w: torch.Tensor, kind=T_Q8):
        w = w.detach().float().cpu().contiguous()
        if w.dim() == 1:
            w = w[None]
        R, C = w.shape
        if kind != T_F32 and C % 32:
            raise SystemExit(f"{name}: {C} columns is not a multiple of 32")
        make = {T_F32: lambda w=w: w.numpy().astype(np.float32).tobytes(),
                T_Q8: lambda w=w: q8_bytes(w), T_Q4: lambda w=w: q4_bytes(w)}[kind]
        n = {T_F32: R * C * 4, T_Q8: R * C + R * C // 32 * 4, T_Q4: R * C // 2 + R * C // 32 * 4}[kind]
        self.items.append((name, kind, R, C, make, n))

    def blob(self, name, data: bytes):
        self.items.append((name, T_BLOB, 0, 0, lambda d=data: d, len(data)))

    def write(self, path: Path):
        off, lines = 0, [f"{k} {v}" for k, v in self.cfg.items()]
        for name, kind, R, C, _, n in self.items:
            lines.append(f"tensor {name} {kind} {R} {C} {off} {n}")
            off += (n + 63) // 64 * 64
        head = ("\n".join(lines) + "\n").encode()
        start = 16 + len(head)
        pad0 = (-start) % 64
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            f.write(b"OAZYPHN1" + struct.pack("<Q", len(head)) + head + b"\0" * pad0)
            for name, kind, R, C, make, n in self.items:
                b = make()
                assert len(b) == n, (name, len(b), n)
                f.write(b + b"\0" * ((-n) % 64))
        os.replace(tmp, path)
        return start + pad0 + off


# ------------------------------------------------------------------------------------------ the tokenizer
def tokenizer_parts(enc):
    """SmolLM2's tokenizer as the engine reads it; refuses anything the engine would cut differently."""
    from oazy.tok import _byte_decoder
    tk = enc.tk
    j = json.loads(tk.to_str())
    m = j["model"]
    if m.get("type") != "BPE" or m.get("byte_fallback") or m.get("continuing_subword_prefix") or m.get("end_of_word_suffix"):
        raise SystemExit(f"tokenizer model not supported by the phone engine: {m.get('type')}")
    if j.get("normalizer") not in (None, {}):
        raise SystemExit(f"tokenizer normalizer not supported: {j['normalizer']}")
    pre = j.get("pre_tokenizer") or {}
    steps = pre.get("pretokenizers", [pre]) if pre.get("type") == "Sequence" else [pre]
    digits, prefix = 0, 0
    for s in steps:
        if s.get("type") == "Digits":
            if not s.get("individual_digits"):
                raise SystemExit("Digits pre-tokenizer without individual_digits is not supported")
            digits = 1
        elif s.get("type") == "ByteLevel":
            if not s.get("use_regex", True):
                raise SystemExit("ByteLevel without its regex is not supported")
            prefix = int(bool(s.get("add_prefix_space")))
        else:
            raise SystemExit(f"pre-tokenizer step not supported: {s}")
    dec = _byte_decoder()
    n = tk.get_vocab_size(with_added_tokens=True)
    added = {a["id"]: a["content"] for a in j.get("added_tokens", [])}
    pieces = []
    for i in range(n):
        if i in added:
            pieces.append(added[i].encode())
            continue
        s = tk.id_to_token(i)
        pieces.append(b"" if s is None else bytes(dec[c] for c in s))
    vocab = m["vocab"]
    trip = []
    for mg in m["merges"]:
        a, b = mg.split(" ", 1) if isinstance(mg, str) else mg
        trip += [vocab[a], vocab[b], vocab[a + b]]
    offs = np.cumsum([0] + [len(p) for p in pieces]).astype(np.uint32)
    vblob = struct.pack("<I", n) + offs.tobytes() + b"".join(pieces)
    return {"vocab": vblob, "merges": np.array(trip, dtype=np.int32).tobytes(),
            "special": np.array(sorted(added), dtype=np.int32).tobytes(),
            "digits": digits, "prefix": prefix, "ignore_merges": int(bool(m.get("ignore_merges"))), "n": n}


def tokenizer_tests(enc):
    t = struct.pack("<I", len(TEST_TEXTS))
    ids = b""
    for s in TEST_TEXTS:
        b = s.encode()
        t += struct.pack("<I", len(b)) + b
        e = enc.tk.encode(s, add_special_tokens=False).ids
        ids += struct.pack("<I", len(e)) + np.array(e, dtype=np.int32).tobytes()
    return t, ids


# ------------------------------------------------------------------------------------------ the model
def export(model, st: dict, enc, out: Path, quant: str = "q8", say=print):
    cfg = model.cfg
    kind = st.get("kind", "bee2")
    blocks = list(model.blocks)
    kinds = "".join(b.kind for b in blocks)
    big = {"q4": T_Q4, "q8": T_Q8, "f32": T_F32}[quant]      # f32: everything float (only for the exactness tests)
    T_Q8_ = T_F32 if quant == "f32" else T_Q8
    W = Writer()
    W.set("name", NAMES.get(kind, kind))
    W.set("label", st.get("label", NAMES.get(kind, kind)))
    W.set("accent", ACCENTS.get(kind, "#8f7bff"))
    W.set("kind", kind)
    W.set("quant", quant)
    W.set("d_model", cfg.d_model)
    W.set("n_layer", len(blocks))
    W.set("layers", kinds)
    W.set("vocab_size", cfg.vocab_size)
    W.set("n_vocab", enc.n_vocab)
    W.set("eot", enc.eot_token)
    W.set("softcap", float(cfg.softcap or 0))
    W.set("max_len", cfg.max_len)
    W.set("attn_heads", cfg.attn_heads)
    W.set("attn_kv_heads", cfg.attn_kv_heads)
    W.set("attn_head_dim", cfg.attn_head_dim)
    W.set("rope", int(cfg.rope))
    W.set("rope_base", float(cfg.rope_base))
    W.set("n_head", cfg.n_head)
    W.set("head_k", cfg.head_k)
    W.set("head_v", cfg.head_v)
    W.set("conv", cfg.conv)
    W.set("mlp_hidden", cfg.mlp_hidden)
    W.set("tokens_trained", int(st.get("tokens") or 0))
    W.set("step", int(st.get("step") or 0))
    W.set("params", model.num_params())
    W.set("active", model.active_params() if hasattr(model, "active_params") else model.num_params())
    W.set("exported", time.strftime("%Y-%m-%d %H:%M"))
    moe = hasattr(blocks[0], "moe")
    if moe:
        W.set("n_expert", cfg.n_expert)
        W.set("top_k", cfg.top_k)
        W.set("expert_hidden", cfg.expert_hidden)
    err = []

    def note(name, w, k):
        if k != T_F32:
            d = dequant(w.detach().float().reshape(w.shape[0], -1), k)
            w = w.detach().float().reshape(d.shape)
            err.append(((d - w).norm() / w.norm().clamp_min(1e-12)).item())

    sd = {k: v for k, v in model.state_dict().items()}
    W.mat("embed", sd["embed.weight"], T_Q8_)
    note("embed", sd["embed.weight"], T_Q8_)
    W.mat("norm", sd["norm.weight"], T_F32)
    for i, b in enumerate(blocks):
        p, q = f"blocks.{i}.", f"L{i}."
        W.mat(q + "n1", sd[p + "n1.weight"], T_F32)
        W.mat(q + "n2", sd[p + "n2.weight"], T_F32)
        if b.kind == "A":
            w = torch.cat([sd[p + "attn.q.weight"], sd[p + "attn.k.weight"], sd[p + "attn.v.weight"], sd[p + "attn.gate.weight"]])
            W.mat(q + "wqkvg", w, T_Q8_); note(q, w, T_Q8_)
            W.mat(q + "wo", sd[p + "attn.out.weight"], T_Q8_); note(q, sd[p + "attn.out.weight"], T_Q8_)
            W.mat(q + "qn", sd[p + "attn.qn.weight"], T_F32)
            W.mat(q + "kn", sd[p + "attn.kn.weight"], T_F32)
        else:
            m = p + "mem."
            w = torch.cat([sd[m + "qkv.weight"], sd[m + "gate.weight"], sd[m + "trust.weight"], sd[m + "fade.weight"]])
            W.mat(q + "wproj", w, T_Q8_); note(q, w, T_Q8_)
            W.mat(q + "bproj", torch.cat([sd[m + "trust.bias"], sd[m + "fade.bias"]]), T_F32)
            W.mat(q + "conv", sd[m + "conv_w"][:, 0, :], T_F32)
            W.mat(q + "A_log", sd[m + "A_log"], T_F32)
            W.mat(q + "dt_bias", sd[m + "dt_bias"], T_F32)
            W.mat(q + "onorm", sd[m + "onorm.weight"], T_F32)
            W.mat(q + "wo", sd[m + "out.weight"], T_Q8_); note(q, sd[m + "out.weight"], T_Q8_)
        if p + "mlp.up.weight" in sd:
            W.mat(q + "wup", sd[p + "mlp.up.weight"], big); note(q, sd[p + "mlp.up.weight"], big)
            W.mat(q + "wdown", sd[p + "mlp.down.weight"], big); note(q, sd[p + "mlp.down.weight"], big)
        if moe:
            E = cfg.n_expert
            up, down = sd[p + "moe.w_up"], sd[p + "moe.w_down"]
            W.mat(q + "router", sd[p + "moe.router.weight"], T_F32)
            W.mat(q + "rbias", sd[p + "moe.bias"], T_F32)
            W.mat(q + "eup", up.reshape(E * up.shape[1], up.shape[2]), big); note(q, up.reshape(-1, up.shape[2]), big)
            W.mat(q + "edown", down.reshape(E * down.shape[1], down.shape[2]), big); note(q, down.reshape(-1, down.shape[2]), big)

    tp = tokenizer_parts(enc)
    W.set("tok_digits", tp["digits"])
    W.set("tok_prefix_space", tp["prefix"])
    W.set("tok_ignore_merges", tp["ignore_merges"])
    W.blob("tok.vocab", tp["vocab"])
    W.blob("tok.merges", tp["merges"])
    W.blob("tok.special", tp["special"])
    texts, ids = tokenizer_tests(enc)
    W.blob("test.texts", texts)
    W.blob("test.ids", ids)

    # PyTorch's own scores after a short chat prompt, from the full-precision weights (the phone's yardstick)
    prompt = [enc.eot_token] + enc.tk.encode(REF_PROMPT, add_special_tokens=False).ids
    with torch.no_grad():
        logits, _ = model.read(torch.tensor([prompt]))
    ref = logits[0, -1].float()
    W.blob("test.prompt", np.array(prompt, dtype=np.int32).tobytes())
    W.mat("test.logits", ref, T_F32)
    top = torch.topk(ref, 5).indices.tolist()
    say(f"[phone] PyTorch's top guesses after the prompt: {[enc.decode([t]) for t in top]}")
    if err:
        say(f"[phone] weight rounding error ({quant}): {100 * float(np.mean(err)):.2f}% on average")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = W.write(out)
    say(f"[phone] wrote {out} ({n / 1e9:.2f} GB, {len(blocks)} layers: {kinds})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="coral", help="the run folder under the runs directory")
    ap.add_argument("--ckpt", default=None, help="a checkpoint file instead of <runs>/<run>/ckpt.pt")
    ap.add_argument("--quant", default="q8", choices=["q8", "q4", "f32"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the reference pass (keep low next to training)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from oazy.data import default_dir
    from oazy.train import load_model
    from oazy import tok
    ckpt = Path(a.ckpt or f"{default_dir('runs')}/{a.run}/ckpt.pt")
    print(f"[phone] reading {ckpt} on the CPU ...", flush=True)
    model, st = load_model(ckpt, torch.device("cpu"))
    model.float().eval()
    st = {k: v for k, v in st.items() if k not in ("model", "opt")}
    enc = tok.for_model(model)
    if getattr(enc, "name", "") != "smollm2":
        raise SystemExit("the phone app reads models with SmolLM2's tokenizer (Bee_v2_0.2B, Bee3, Frog, Coral)")
    name = NAMES.get(st.get("kind"), st.get("kind"))
    if a.run.endswith("-chat"):
        name += "-chat"
        st["label"] = name
    out = Path(a.out or ckpt.parent / "phone" / f"{name}-{a.quant}.oazy")
    export(model, st, enc, out, a.quant)
    print(f"[phone] send it to the phone:  python3 -m phone.serve   (then open the pod's 8001 link on the phone)")


if __name__ == "__main__":
    main()
