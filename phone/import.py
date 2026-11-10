"""Turn a phone model file (.oazy) back into a checkpoint the lab can score: the way back from phone/export.py.

    python3 -m phone.import --file /workspace/Coral_1B-chat-q8.oazy                 # -> <runs>/coral-chat-q8/ckpt.pt
    python3 -m phone.import --file X.oazy --into /workspace/oazy_runs/coral-chat-q8
    python3 -m oazy.evals --run coral-chat-q8                                        # then score it like any run

For when the phone copy is the only one left. The 8-bit (or 4-bit) numbers are turned back into ordinary float32
weights (each group of 32 times its scale), the fused grids are split back into the model's own (q, k, v, gate ...),
and the settings in the file's header rebuild the model's config. So the result IS the phone's model, rounding
included: scores from it are the rounded model's (8-bit rounding usually costs very little; 4-bit costs more).
Nothing about training is kept (no optimizer state); a fine-tune can start from it like from any checkpoint.

At the end it checks itself: PyTorch reads the file's own check prompt with the rebuilt model and compares its scores
with the ones the exporter stored (made from the original weights). A correlation near 1 means it came back right.
Runs on the CPU, a minute or two for Coral (about 4 GB of RAM, a 4 GB file at float32; --dtype bf16 halves both).
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

T_F32, T_Q8, T_Q4, T_BLOB = 0, 1, 2, 9


def read(path: Path):
    """(header settings, {name: float32 tensor}, {name: bytes}) of a .oazy file."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    if bytes(raw[:8]) != b"OAZYPHN1":
        raise SystemExit(f"{path} is not an Oazy phone model")
    hl = struct.unpack("<Q", bytes(raw[8:16]))[0]
    head = bytes(raw[16:16 + hl]).decode()
    data0 = (16 + hl + 63) // 64 * 64
    cfg, mats, blobs = {}, {}, {}
    for line in head.splitlines():
        if not line:
            continue
        if line.startswith("tensor "):
            _, name, kind, R, C, off, nb = line.split()
            kind, R, C, off, nb = int(kind), int(R), int(C), int(off), int(nb)
            d = raw[data0 + off: data0 + off + nb]
            if kind == T_BLOB:
                blobs[name] = bytes(d)
            elif kind == T_F32:
                mats[name] = torch.from_numpy(np.frombuffer(d, dtype=np.float32).reshape(R, C).copy())
            elif kind == T_Q8:
                q = np.frombuffer(d[:R * C], dtype=np.int8).reshape(R, C // 32, 32).astype(np.float32)
                s = np.frombuffer(d[R * C:], dtype=np.float32).reshape(R, C // 32, 1)
                mats[name] = torch.from_numpy((q * s).reshape(R, C))
            elif kind == T_Q4:
                p = np.frombuffer(d[:R * C // 2], dtype=np.uint8).reshape(R, C // 32, 16)
                q = np.concatenate([p & 15, p >> 4], axis=-1).astype(np.float32) - 8.0   # values 0..15 then 16..31
                s = np.frombuffer(d[R * C // 2:], dtype=np.float32).reshape(R, C // 32, 1)
                mats[name] = torch.from_numpy((q * s).reshape(R, C))
            else:
                raise SystemExit(f"unknown tensor type {kind} for {name}")
        else:
            k, _, v = line.partition(" ")
            cfg[k] = v
    return cfg, mats, blobs


def config_class(kind: str):
    if kind == "coral":
        from coral.model import Coral, CoralConfig
        return Coral, CoralConfig
    if kind == "bee2":
        from bee2.model import Bee2, Bee2Config
        return Bee2, Bee2Config
    if kind == "bee3":
        from bee3.model import Bee3, Bee3Config
        return Bee3, Bee3Config
    if kind == "frog":
        from frog.model import Frog, FrogConfig
        return Frog, FrogConfig
    raise SystemExit(f"this file holds a {kind!r} model; phone.import knows coral, bee2, bee3 and frog")


def rebuild(cfg: dict, mats: dict):
    """The model's config and state dict from the file."""
    kind = cfg["kind"]
    Model, Cfg = config_class(kind)
    I = lambda k: int(float(cfg[k]))
    layers = cfg["layers"]
    n = len(layers)
    c = {k: I(k) for k in ("d_model", "vocab_size", "max_len", "attn_heads", "attn_kv_heads", "attn_head_dim", "rope",
                          "n_head", "head_k", "head_v", "conv", "mlp_hidden") if k in cfg}
    c.update(n_layer=n, softcap=float(cfg["softcap"]), rope_base=float(cfg["rope_base"]), tokenizer="smollm2",
             pattern="MMMA" if "M" in layers else "A", grow=f"{n}:0")         # built at full depth (no growing)
    if "n_expert" in cfg:
        c.update(n_expert=I("n_expert"), top_k=I("top_k"), expert_hidden=I("expert_hidden"))
    if "M" in layers:
        for i, k in enumerate(layers):           # the pattern must reproduce the file's layer kinds exactly
            if k != "MMMA"[i % 4]:
                raise SystemExit(f"layer kinds {layers} are not the MMMA pattern")
    config = Cfg.from_dict(c)
    D, H, KV, hd = config.d_model, config.attn_heads, config.attn_kv_heads, config.attn_head_dim
    sd = {"embed.weight": mats["embed"], "norm.weight": mats["norm"][0]}
    for i, kd in enumerate(layers):
        q, p = f"L{i}.", f"blocks.{i}."
        sd[p + "n1.weight"] = mats[q + "n1"][0]
        sd[p + "n2.weight"] = mats[q + "n2"][0]
        if kd == "A":
            wq, wk, wv, wg = mats[q + "wqkvg"].split([H * hd, KV * hd, KV * hd, H * hd])
            sd.update({p + "attn.q.weight": wq, p + "attn.k.weight": wk, p + "attn.v.weight": wv, p + "attn.gate.weight": wg,
                       p + "attn.out.weight": mats[q + "wo"], p + "attn.qn.weight": mats[q + "qn"][0], p + "attn.kn.weight": mats[q + "kn"][0]})
        else:
            Hm, K, V = config.n_head, config.head_k, config.head_v
            ch = Hm * (2 * K + V)
            qkv, gate, trust, fade = mats[q + "wproj"].split([ch, Hm * V, Hm, Hm])
            tb, fb = mats[q + "bproj"][0].split([Hm, Hm])
            m = p + "mem."
            sd.update({m + "qkv.weight": qkv, m + "gate.weight": gate, m + "trust.weight": trust, m + "trust.bias": tb,
                       m + "fade.weight": fade, m + "fade.bias": fb, m + "conv_w": mats[q + "conv"][:, None, :],
                       m + "A_log": mats[q + "A_log"][0], m + "dt_bias": mats[q + "dt_bias"][0],
                       m + "onorm.weight": mats[q + "onorm"][0], m + "out.weight": mats[q + "wo"]})
        if q + "wup" in mats:
            sd[p + "mlp.up.weight"] = mats[q + "wup"]
            sd[p + "mlp.down.weight"] = mats[q + "wdown"]
        if q + "eup" in mats:
            E = config.n_expert
            sd[p + "moe.router.weight"] = mats[q + "router"]
            sd[p + "moe.bias"] = mats[q + "rbias"][0]
            sd[p + "moe.w_up"] = mats[q + "eup"].reshape(E, -1, D)
            sd[p + "moe.w_down"] = mats[q + "edown"].reshape(E, D, -1)
    return kind, Model, config, {k: v.contiguous() for k, v in sd.items()}


def check(model, blobs, mats, say=print):
    """PyTorch's scores with the rebuilt weights against the ones the exporter stored (original weights)."""
    if "test.prompt" not in blobs or "test.logits" not in mats:
        say("[phone] the file has no self-check; skipped")
        return None
    ids = np.frombuffer(blobs["test.prompt"], dtype=np.int32).tolist()
    with torch.no_grad():
        lg, _ = model.read(torch.tensor([ids]))
    a, b = lg[0, -1].float(), mats["test.logits"][0]
    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    top = int(a.argmax()) == int(b.argmax())
    say(f"[phone] check: scores vs the original model's {corr:.4f} correlated, top word {'the same' if top else 'DIFFERENT'}")
    return corr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="the .oazy file")
    ap.add_argument("--into", default=None, help="the run folder to write (default: <runs>/<model>-<quant>, e.g. coral-chat-q8)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bf16"])
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    src = Path(a.file)
    cfg, mats, blobs = read(src)
    kind, Model, config, sd = rebuild(cfg, mats)
    model = Model(config)
    if hasattr(model, "_stack_to"):
        model._stack_to(config.n_layer)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith(("rope_cos", "rope_sin"))]
    if missing or unexpected:
        raise SystemExit(f"the file does not fill the model: missing {missing[:6]}, unexpected {unexpected[:6]}")
    model.float().eval()
    label = cfg.get("label", cfg.get("name", kind))
    print(f"[phone] {label}: {kind}, {len(cfg['layers'])} layers, {sum(p.numel() for p in model.parameters()) / 1e6:,.0f}M "
          f"parameters, from {cfg.get('quant', '?')} ({src.stat().st_size / 1e9:.2f} GB)", flush=True)
    if not a.no_check:
        check(model, blobs, mats)
    from oazy.data import default_dir
    run = "coral" if kind == "coral" else kind
    if label.endswith("-chat"):
        run += "-chat"
    out = Path(a.into or Path(default_dir("runs")) / f"{run}-{cfg.get('quant', 'q8')}")
    out.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if a.dtype == "bf16":
        state = {k: (v.to(torch.bfloat16) if v.is_floating_point() and v.dim() >= 2 else v) for k, v in state.items()}
    tmp = out / "ckpt.pt.tmp"
    torch.save({"model": state, "kind": kind, "config": config.to_dict(), "step": int(cfg.get("step", 0) or 0),
                "tokens": int(cfg.get("tokens_trained", 0) or 0),
                "from_phone": {"file": src.name, "quant": cfg.get("quant"), "exported": cfg.get("exported")}}, tmp)
    tmp.replace(out / "ckpt.pt")
    print(f"[phone] wrote {out / 'ckpt.pt'}; score it with:  python3 -m oazy.evals --run {out.name}", flush=True)


if __name__ == "__main__":
    main()
