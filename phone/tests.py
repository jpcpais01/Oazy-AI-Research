"""Checks for the phone engine, on the CPU, with small random models (no GPU, no download):

    python3 -m phone.tests                 # builds the engine for this PC and runs everything
    OAZY_ENGINE=path python3 -m phone.tests  # test another engine binary (e.g. the phone build under qemu, see build.sh)

  1. exactness   float32 files: the engine's scores after every word agree with PyTorch's read() to ~1e-4, for a
                 Coral body (experts, RoPE), a Bee2 body (Memtrix memory + attention, NoPE) and a Coral hybrid
                 (memory layers + experts), read word by word and in chunks
  2. rounding    8-bit and 4-bit files: mostly the same top guess, scores closely correlated (random models are
                 touchier than trained ones; the self-check on a real export is the true yardstick)
  3. tokenizer   a stand-in for SmolLM2's tokenizer (same pre-tokenizer: digits, then GPT-2's pattern; same special
                 pieces): the engine cuts 200+ texts exactly like Hugging Face's tokenizers library
  4. self-check  the file's own check (what the app's Check button runs) passes
  5. the chat protocol: an answer streams, stops at "User:", and the conversation carries on
"""
from __future__ import annotations

import os
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="oazy-phone-"))


def standin_tokenizer(numbers: bool = False) -> Path:
    """A small byte-level BPE with SmolLM2's pre-tokenizer and special pieces. numbers=True also learns pieces like
    " Â" (a space then the first byte of ½), so cutting " ½" as one word instead of " " + "½" would show."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                               pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    t.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<repo_name>", "<reponame>", "<file_sep>", "<filename>"]
    tr = trainers.BpeTrainer(vocab_size=1024, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    corpus = [
        "User: hello there, how are you doing today?\n\nAssistant: I'm doing well, thank you for asking!",
        "The capital of France is Paris. The capital of Portugal is Lisbon. The capital of Spain is Madrid.",
        "def main():\n    print('hello world')\n    return 0\n", "In 2024 there were 365 days; 12 months, 52 weeks.",
        "Olá João, tudo bem? Ça va très bien. Straße. Привет мир. 你好世界", "We'll see; they're here; it's fine; I've been.",
    ] * 50
    if numbers:
        corpus += ["Symbols: © ® ½ ² ³ ٣٤ ① and x² + y³; 3½ cups © 2024 ® ½ ¼ ¾ ४२"] * 50
    t.train_from_iterator(corpus, tr)
    p = TMP / ("tokenizer-numbers.json" if numbers else "tokenizer.json")
    t.save(str(p))
    return p


def engine() -> list[str]:
    e = os.environ.get("OAZY_ENGINE")
    if e:
        return e.split()
    out = TMP / "oazy"
    src = HERE / "engine" / "oazy.cpp"
    cxx = os.environ.get("CXX", "c++")
    subprocess.run([cxx, "-O2", "-std=c++17", "-pthread", "-o", str(out), str(src)], check=True)
    return [str(out)]


def run(eng, *args, inp=None):
    r = subprocess.run(eng + list(args), input=inp, capture_output=True, timeout=600)
    if r.returncode:
        raise RuntimeError(f"engine failed: {r.stdout[-2000:]!r} {r.stderr[-2000:]!r}")
    return r.stdout


def models(V):
    from coral.model import Coral, CoralConfig
    from bee2.model import Bee2, Bee2Config
    torch.manual_seed(0)
    out = {}
    out["coral"] = Coral(CoralConfig(vocab_size=V, d_model=128, n_layer=4, attn_heads=4, attn_kv_heads=2, attn_head_dim=32,
                                     mlp_hidden=64, n_expert=8, top_k=2, expert_hidden=64, max_len=256))
    out["bee2"] = Bee2(Bee2Config(vocab_size=V, d_model=128, n_layer=4, n_head=2, head_k=32, head_v=32, attn_heads=4,
                                  attn_kv_heads=2, attn_head_dim=32, mlp_hidden=96, max_len=256))
    from bee4.model import Bee4, Bee4Config
    out["bee4"] = Bee4(Bee4Config(vocab_size=V, d_model=128, n_layer=4, attn_heads=4, attn_kv_heads=2, attn_head_dim=32,
                                  mlp_hidden=96, max_len=256))
    out["coral-hybrid"] = Coral(CoralConfig(vocab_size=V, d_model=128, n_layer=4, pattern="MMMA", n_head=2, head_k=32,
                                            head_v=32, attn_heads=4, attn_kv_heads=2, attn_head_dim=32, mlp_hidden=64,
                                            n_expert=8, top_k=2, expert_hidden=64, max_len=256))
    for name, m in out.items():
        with torch.no_grad():               # make every part matter: norms, biases, gates and the balancing biases
            for n, p in m.named_parameters():
                if n.endswith(("n1.weight", "n2.weight", "norm.weight", "qn.weight", "kn.weight", "onorm.weight")):
                    p.copy_(1 + 0.3 * torch.randn_like(p))
                elif n.endswith(("trust.bias", "fade.bias")):
                    p.copy_(0.5 * torch.randn_like(p))
                elif p.dim() >= 2 and "embed" not in n:
                    p.mul_(3.0)             # random small models are otherwise nearly flat
            for b in m.blocks:
                if hasattr(b, "moe"):
                    b.moe.bias.copy_(0.05 * torch.randn_like(b.moe.bias))
        m.float().eval()
    return out


def torch_scores(m, ids):
    """PyTorch's scores after every word, reading word by word with its state (the chat path)."""
    out, states = [], None
    with torch.no_grad():
        for t in ids:
            lg, states = m.read(torch.tensor([[t]]), states)
            out.append(lg[0, -1].float().numpy())
    return np.stack(out)


def engine_scores(eng, path, ids, chunk, V):
    raw = run(eng, "--model", str(path), "--chunk", str(chunk), "--threads", "3", "logits", *map(str, ids))
    return np.frombuffer(raw, dtype=np.float32).reshape(len(ids), V)


def main():
    tokp = standin_tokenizer()
    os.environ["OAZY_SMOL_TOKENIZER"] = str(tokp)
    sys.path.insert(0, str(HERE.parent))
    from oazy import tok
    from phone.export import export
    enc = tok.get("smollm2")
    V = enc.n_vocab
    eng = engine()
    print(f"engine: {' '.join(eng)}")
    rng = random.Random(0)
    ids = [enc.eot_token] + enc.encode_ordinary("User: What is the capital of France?\n\nAssistant: The capital is Paris.")
    ids += [rng.randrange(V) for _ in range(20)]
    ok = True
    for name, m in models(V).items():
        ref = torch_scores(m, ids)
        st = {"kind": "coral" if name.startswith("coral") else ("bee4" if name == "bee4" else "bee2"), "tokens": 123, "step": 4}
        for quant, tol in (("f32", None), ("q8", 0.97), ("q4", 0.85)):
            path = TMP / f"{name}-{quant}.oazy"
            export(m, st, enc, path, quant, say=lambda *a: None)
            for chunk in ((1, 7) if quant == "f32" else (5,)):
                got = engine_scores(eng, path, ids, chunk, V)
                if quant == "f32":
                    err = np.abs(got - ref).max() / np.abs(ref).max()
                    good = err < 2e-4
                    print(f"  {name:13s} {quant}  chunk {chunk}: max error {err:.2e}  {'OK' if good else 'FAIL'}")
                else:
                    # random small models are far more sensitive to rounding than trained ones (their weights are
                    # scaled up so every part matters); the real yardstick is the self-check on a real export
                    corr = float(np.median([np.corrcoef(got[i], ref[i])[0, 1] for i in range(len(ids))]))
                    top = np.mean(got.argmax(1) == ref.argmax(1))
                    good = corr > tol and top >= (0.75 if quant == "q8" else 0.35)
                    print(f"  {name:13s} {quant}  chunk {chunk}: median correlation {corr:.4f}, same top word {100 * top:.0f}%  "
                          f"{'OK' if good else 'FAIL'}")
                ok &= bool(good)
            if quant == "q8":
                rep = run(eng, "--model", str(path), "selftest").decode().strip()
                good = rep.startswith(f"tokenizer {len(__import__('phone.export', fromlist=['x']).TEST_TEXTS)}/") and "same" in rep
                print(f"  {name:13s} self-check: {rep}  {'OK' if good else 'FAIL'}")
                ok &= good

    # the tokenizer against Hugging Face's, on many texts (with a second stand-in that has learned number pieces)
    os.environ["OAZY_SMOL_TOKENIZER"] = str(standin_tokenizer(numbers=True))
    enc = tok.SmolTokenizer()
    path = TMP / "numbers-q8.oazy"
    export(models(enc.n_vocab)["bee4"], {"kind": "bee4", "tokens": 1, "step": 1}, enc, path, "q8", say=lambda *a: None)
    alphabet = list("abcdefghijklmnopqrstuvwxyzABCXYZ0123456789 .,;:!?'\"()-_\n\t/\\@#") + list("éãçßøПривет你好😀ٴ٣½")
    texts = []
    for i in range(200):
        n = rng.randrange(1, 40)
        texts.append("".join(rng.choice(alphabet) for _ in range(n)))
    texts += ["  leading", "trailing  ", "\n\n\nx", "x\n\n\n", " ' s", "'ll'", "a  \n b", "1 2  3   4", "<|endoftext|><|endoftext|>",
              "½ ² ³ ٣٤ ①", "x² 3½ ©½", " ½", "¼ and ¾", " ४२ "]
    bad = 0
    for t in texts:
        want = enc.tk.encode(t, add_special_tokens=False).ids
        got = [int(x) for x in run(eng, "--model", str(path), "encode", t).split()]
        if got != want:
            bad += 1
            if bad <= 5:
                print(f"    tokenizer differs on {t!r}: want {want} got {got}")
    print(f"  tokenizer: {len(texts) - bad}/{len(texts)} texts cut exactly like Hugging Face  {'OK' if not bad else 'FAIL'}")
    ok &= bad == 0

    # the chat protocol
    cmds = "chat\t12\t0.7\t40\t0.95\t1.1\tHello!\nchat\t8\t0\t1\t1\t1\tAnd again\\nwith a new line\nreset\nwrite\t5\t0.7\t40\t0.95\t1.1\tOnce upon a time\nquit\n"
    out = run(eng, "--model", str(path), "--threads", "2", "serve", inp=cmds.encode()).decode(errors="replace").splitlines()
    kinds = [l.split("\t")[0] for l in out]
    good = kinds[0] == "ready" and kinds.count("end") == 3 and "ok" in kinds and "error" not in kinds
    print(f"  chat protocol: {kinds.count('tok')} pieces streamed, {kinds.count('end')} answers  {'OK' if good else 'FAIL'}")
    if not good:
        print("\n".join(out[:20]))
    ok &= good
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
