"""Self-tests for Bee_v2_0.2B: its size, its tokenizer, the re-cutting of the text, the SmolLM2 mentor,
the fair losses, and one whole (tiny) training run end to end.

    python -m bee2.tests                 (CPU, ~1-2 min; without internet it uses stand-in tokenizers)
    python -m bee2.tests --device cuda   (on the pod: the real SmolLM2 tokenizer, plus a bf16 step)

Bee0.2B's own tests (python -m bee.tests) cover the body, which is the same.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def check(name, ok, detail=""):
    print(f"  {'ok ' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        raise SystemExit(f"self-test failed: {name}")


WORDS = ("the cat sat on a mat and then it ran to the old red barn where seven small birds sang "
         "about water clouds rivers mountains Einstein 1905 learning é ü — 你好 🙂").split()


def fake_docs(n, rng):
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 60))).capitalize() + "." for _ in range(n)]


def stand_ins(tmp: Path):
    """Offline: a byte-level GPT-2 stand-in (256 byte pieces + <|endoftext|> at 50256) and a small
    byte-level BPE in SmolLM2's format (<|endoftext|> = 0). On the pod the real ones are used."""
    used = []
    try:
        import tiktoken
        tiktoken.get_encoding("gpt2")
    except Exception:
        if not (os.environ.get("OAZY_GPT2_TIKTOKEN")):
            p = tmp / "bytes.tiktoken"
            p.write_text("".join(f"{base64.b64encode(bytes([b])).decode()} {b}\n" for b in range(256)))
            os.environ["OAZY_GPT2_TIKTOKEN"] = str(p)
            used.append("GPT-2")
    if not os.environ.get("OAZY_SMOL_TOKENIZER"):
        try:
            from tokenizers import Tokenizer
            Tokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
        except Exception:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
            t = Tokenizer(models.BPE())
            t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            t.decoder = decoders.ByteLevel()
            tr = trainers.BpeTrainer(vocab_size=700, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
            t.train_from_iterator(fake_docs(3000, random.Random(1)), tr)
            p = tmp / "smol-standin.json"
            t.save(str(p))
            os.environ["OAZY_SMOL_TOKENIZER"] = str(p)
            used.append("SmolLM2")
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = torch.device(ap.parse_args().device)
    print(f"Bee_v2_0.2B self-tests on {dev}")
    torch.manual_seed(0)
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used = stand_ins(tmp)
    if used:
        print(f"  info  no internet: stand-in {' and '.join(used)} tokenizer(s) (same kind, smaller)")

    from bee2.model import Bee2, Bee2Config
    from bee.model import BeeConfig
    from oazy import tok

    # ---- size: Bee0.2B's body, SmolLM2's word table
    big = Bee2(Bee2Config())
    n = big.num_params()
    check("full size is about 0.2B", 195e6 < n < 215e6, f"{n / 1e6:.1f}M parameters, {big.layout()}")
    check("same body as Bee0.2B, only the vocabulary differs",
          all(getattr(big.cfg, k) == v for k, v in BeeConfig().to_dict().items() if k not in ("vocab_size", "tokenizer")))
    check("SmolLM2's 49,152 pieces", big.cfg.vocab_size == 49152 and tok.name_of(big) == "smollm2")
    del big

    # ---- the tokenizer: any text, exactly back; bytes per piece add up
    enc = tok.get("smollm2")
    text = "Hello, world! The water cycle is… é ü 你好 🙂\n\nUser: hi\n\nAssistant: hello"
    ids = enc.encode_ordinary(text)
    check("SmolLM2 tokenizer: text -> pieces -> the same text", enc.decode(ids) == text, f"{len(ids)} pieces")
    check("each piece's bytes add up to the text", b"".join(enc.decode_single_token_bytes(t) for t in ids) == text.encode())
    bpt = tok.bytes_per_token("smollm2")
    check("byte table counts every byte", int(bpt[ids].sum()) == len(text.encode()) and bpt[enc.eot_token] == 0)
    if not used:
        check("the real SmolLM2 tokenizer: 49,152 pieces, <|endoftext|> = 0", enc.n_vocab == 49152 and enc.eot_token == 0)

    # ---- re-cutting GPT-2 text with SmolLM2's tokenizer
    from oazy import retok
    from oazy.data import get_tokenizer, EOT
    gpt = get_tokenizer()
    rng = random.Random(0)
    data = tmp / "data"
    docs = {}
    for sub, nd in (("", 1500), ("chat", 300)):
        d = data / sub if sub else data
        d.mkdir(parents=True, exist_ok=True)
        for split, k in (("val", nd // 5), ("train", nd)):
            ds = fake_docs(k, rng)
            if sub == "chat":
                ds = [f"User: {a}\n\nAssistant: {b}" for a, b in zip(ds, fake_docs(k, rng))]
            docs[(sub, split)] = ds
            arr = np.fromiter((t for x in ds for t in gpt.encode_ordinary(x) + [EOT]), dtype=np.uint16)
            arr.tofile(d / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(
        json.dumps({"ctx": " ".join(fake_docs(1, rng)), "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n"
        for _ in range(24)))
    got = retok.run(data, "smollm2", ["fwe", "dclm", "chat"], workers=2)
    check("sources that are there are re-cut, missing ones skipped", set(got) == {"fwe", "chat"})
    ok = True
    for (sub, split), ds in docs.items():
        out = np.fromfile((data / "smol" / sub if sub else data / "smol") / f"{split}.bin", dtype=np.uint16)
        cuts = np.flatnonzero(out == enc.eot_token)
        back = [enc.decode(out[a + 1 if i else 0:b].tolist()) for i, (a, b) in enumerate(zip(np.r_[-1, cuts[:-1]], cuts))]
        ok &= back == ds
    check("re-cut text is exactly the same documents, in the same order", ok)
    m = json.loads((data / "smol" / "meta.json").read_text())
    nbytes = sum(len(x.encode()) for x in docs[("", "val")])
    check("meta: token counts and bytes", m["val"]["bytes"] == nbytes and
          m["val"]["tokens"] == len(np.fromfile(data / "smol" / "val.bin", dtype=np.uint16)), f"ratio {m['val']['ratio']:.3f}")
    vs = np.fromfile(data / "smol" / "val.bin", dtype=np.uint16)
    vg = np.fromfile(data / "val.bin", dtype=np.uint16)
    check("bits-per-byte counts the same text for both tokenizers",
          int(bpt[vs].sum()) == int(tok.bytes_per_token("gpt2")[vg].sum()) == nbytes)
    again = retok.run(data, "smollm2", ["fwe"], workers=2)
    check("running it again does nothing", again["fwe"]["train"] == m["train"])
    check("HellaSwag is shared (plain text)", (data / "smol" / "hellaswag_val.jsonl").exists())

    # ---- Tagus, our dataset: sized for a run, picked up by the mix
    from oazy import tagus
    from oazy.mix import parse_mix, resolve_mix, TAGUS_MIX
    w = tagus.needs(2.5e9, margin=0.05)
    check("Tagus: a 2.5B run reads 2.5B tokens (+5%)", abs(sum(w.values()) - 2.625e9) < 1e4 and set(w) == set(parse_mix(TAGUS_MIX)[0][1]),
          ", ".join(f"{k} {v / 1e6:,.0f}M" for k, v in w.items()))
    check("Tagus: chats are a pinch throughout and 30% of the cool-down",
          parse_mix(resolve_mix("tagus"))[0][1]["chat"] == 0.03 and parse_mix(resolve_mix("tagus"))[1][1]["chat"] == 0.3)

    # ---- the SmolLM2 mentor
    from ember.mentor import Mentor, mentor_loss
    me = Mentor("smol-random", dev, compile=False)
    g = me.guess(torch.randint(0, 49152, (1, 1500), device=dev))
    check("SmolLM2-shaped mentor reads 1,500 words at once, 49,152 scores each",
          g.shape == (1, 1500, 49152) and me.net.cfg.max_len >= 2048)
    small = Bee2(Bee2Config(d_model=64, n_layer=4, n_head=1, head_k=32, head_v=32, attn_heads=2, attn_kv_heads=1,
                            attn_head_dim=32, mlp_hidden=128, max_len=128, ctx="64:0,128:0.5")).to(dev)
    x = torch.randint(0, 49152, (2, 64), device=dev)
    kd = mentor_loss(small(x)[0], me.guess(x))
    check("student and mentor guesses compare piece by piece", torch.isfinite(kd).item() and kd.item() > 0, f"KL {kd.item():.3f}")
    del me

    # ---- one whole tiny run: re-cut data, mentor, fair losses, HellaSwag both ways, resume
    from oazy import train
    runs = tmp / "runs"
    argv = ["train", "--model", "bee2", "--preset", "smoke", "--data", str(data), "--runs", str(runs),
            "--mentor", "smol-random", "--mentor-until", "1", "--device", str(dev), "--compile", "0"]
    sys.argv = argv
    train.main()
    rd = runs / "bee2"
    recs = [json.loads(l) for l in (rd / "metrics.jsonl").read_text().splitlines()]
    ev = [r for r in recs if r["kind"] == "eval"]
    hs = [r for r in recs if r["kind"] == "hellaswag"]
    tr = [r for r in recs if r["kind"] == "train"]
    cfg = json.loads((rd / "config.json").read_text())
    r = cfg["loss_scale"]
    check("the run reads SmolLM2's text, with its mentor", cfg["tokenizer"] == "smollm2" and cfg["mentor"] == "smol-random")
    check("budget = the same text as 60k GPT-2 tokens", abs(cfg["total_tokens"] - int(60_000 * r)) <= 1, f"{cfg['total_tokens']} tokens")
    check("held-out loss is shown per GPT-2 token (own loss x ratio)",
          all(abs(e["val_loss"] - e["val_loss_own"] * r) < 1e-6 for e in ev) and len(ev) >= 2)
    check("bits per byte are logged", all(0 < e["val_bpb"] < 40 for e in ev))
    check("training loss too, and the mentor was used", all("loss_own" in t for t in tr) and any("mentor_kl" in t for t in tr))
    check("HellaSwag scored per token and per letter", hs and all("acc_char" in h for h in hs) and hs[-1]["full"])
    st = json.loads((rd / "status.json").read_text())
    check("the run finished", st["state"] == "done", f"val loss {st['latest']['val_loss']:.3f}")
    from oazy.train import load_model
    mdl, ck = load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as Bee_v2_0.2B", ck["kind"] == "bee2" and tok.name_of(mdl) == "smollm2")
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("it writes", len(out) <= 5 and all(0 <= t < 49152 for t in out), repr(enc.decode(out)))

    # a GPT-2 mentor for a SmolLM2 model is refused (their pieces do not line up)
    sys.argv = argv[:-8] + ["--mentor", "gpt2-large", "--name", "bad", "--device", str(dev)]
    try:
        train.main()
        refused = False
    except SystemExit as e:
        refused = "cuts text differently" in str(e)
    check("a mentor with another tokenizer is refused", refused)

    if dev.type == "cuda":
        m = small.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = m(x, x)
        loss.backward()
        check("bf16 training step on the GPU", torch.isfinite(loss).item())
    tmp_ctx.cleanup()
    print("all Bee_v2_0.2B self-tests passed")


if __name__ == "__main__":
    main()
