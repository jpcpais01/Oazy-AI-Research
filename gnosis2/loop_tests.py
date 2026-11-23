"""Self-tests for Loop-135 (gnosis2/loop.py): with 1 round it is Bee-135 exactly; it stays causal; reading word by word
= all at once; only one vector of new weights; rounds change the answer; the gradient reaches the reader past the cut;
a tiny model learns; the compute ratio; the wordings text and its held-out test (oazy/wordings.py).

    python3 -m gnosis2.loop_tests                 (CPU, about a minute)
    python3 -m gnosis2.loop_tests --device cuda   (on the pod: plus a compiled bf16 step at race size and its speed)
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from oazy import term

SMALL = dict(d_model=64, n_layer=12, attn_heads=4, attn_kv_heads=2, attn_head_dim=16, mlp_hidden=128, max_len=128,
             ctx="128:0", vocab_size=512, key_offset="1,5", mod_layers="", fp8=0)
LOOP = dict(loop_layers=2, loop_mouth=2, loop_steps=3, loop_min=2, loop_max=4, loop_grad=2)
FAILS = []


def check(name, ok, detail=""):
    term.check(name, ok, detail)
    if not ok:
        FAILS.append(name)


def jiggle(m, s=0.05):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * s)
    return m


def model_tests(dev):
    from gnosis2.model import Gnosis2, Gnosis2Config
    torch.manual_seed(0)
    plain = jiggle(Gnosis2(Gnosis2Config(**SMALL))).to(dev).eval()
    lp = Gnosis2(Gnosis2Config(**SMALL, **LOOP)).to(dev).eval()
    miss = lp.load_state_dict(plain.state_dict(), strict=False)
    check("the loop's only new weights: one vector (how much of the reader each round re-reads)",
          miss.missing_keys == ["loop_in"] and not miss.unexpected_keys,
          f"{lp.num_params() - plain.num_params()} new numbers")
    x = torch.randint(1, 512, (2, 96), device=dev)
    with torch.no_grad():
        a = plain(x)[0]
        lp._k_force = 1
        b = lp(x)[0]
        lp._k_force = None
    check("with 1 round it IS Bee-135: the same weights give the same scores exactly", torch.equal(a, b),
          f"largest gap {(a - b).abs().max().item():.1e}")
    with torch.no_grad():
        lp.loop_in.add_(torch.randn_like(lp.loop_in) * 0.3)
        s = {k: lp_scores(lp, x, k) for k in (1, 2, 3, 6)}
    check("more rounds change what it says (the loop is in the path)",
          all((s[k] - s[1]).abs().max() > 1e-3 for k in (2, 3, 6)),
          " ".join(f"{k}: {(s[k] - s[1]).abs().max().item():.3f}" for k in (2, 3, 6)))
    with torch.no_grad():
        big = jiggle(Gnosis2(Gnosis2Config(**SMALL, **LOOP)), 0.3).to(dev).eval()
        big._probe, big._k_force = [], 24
        big(x)
        sz, big._probe, big._k_force = big._probe, None, None
    check("the stream does not grow round after round (24 rounds, strong random weights)",
          max(sz[1:]) < 3 * sz[1], " ".join(f"{v:.1f}" for v in sz[::4]))
    y = x.clone()
    y[:, 60] = (y[:, 60] + 7) % 511 + 1
    with torch.no_grad():
        c = lp_scores(lp, y, 3)
    check("causal: changing word 60 changes nothing before it, at 3 rounds", torch.equal(s[3][:, :60], c[:, :60])
          and not torch.equal(s[3][:, 60:], c[:, 60:]))
    with torch.no_grad():
        full, st = lp.read(x[:, :50])
        parts = [full]
        for t in range(50, 96, 7):
            lg, st = lp.read(x[:, t:t + 7], st)
            parts.append(lg)
        stream = torch.cat(parts, 1)
    ref = s[LOOP["loop_steps"]]
    check("reading as a stream (the chat's way, a cache per pass) = reading all at once", (stream - ref).abs().max() < 1e-4,
          f"largest gap {(stream - ref).abs().max().item():.1e}")
    return lp


def lp_scores(m, x, k):
    m._k_force = k
    try:
        return m(x)[0]
    finally:
        m._k_force = None


def train_tests(dev):
    from gnosis2.model import Gnosis2, Gnosis2Config
    from gnosis2 import loop as LOOP_
    torch.manual_seed(1)
    m = Gnosis2(Gnosis2Config(**SMALL, **LOOP)).to(dev).train()
    x = torch.randint(1, 512, (4, 65), device=dev)
    m._k_force = 4                                       # 4 rounds, 2 with gradient: the cut is at round 3
    loss = m(x[:, :-1], x[:, 1:])[1]
    loss.backward()
    m._k_force = None
    g_read = m.blocks[0].mlp.down.weight.grad
    g_loop = m.blocks[8].mlp.down.weight.grad
    check("with 4 rounds (the cut before round 3) the reader and the loop both still learn",
          g_read is not None and g_read.abs().sum() > 0 and g_loop is not None and g_loop.abs().sum() > 0)
    import bee6.model as B6
    seen_k, calls = [], []
    orig = B6.Bee6.startup_check
    B6.Bee6.startup_check = lambda self, device, say=print: [seen_k.append(LOOP_.steps(self)) for _ in range(5)]
    try:
        m.startup_check(dev, lambda *a, **k: None)
    finally:
        B6.Bee6.startup_check = orig
    check("the start-up checks (lean, cuDNN: two runs compared) see one fixed number of rounds, not a random one",
          seen_k == [LOOP["loop_steps"]] * 5 and m._k_force is None, str(seen_k))
    seen = set()
    for _ in range(40):
        seen.add(LOOP_.steps(m))
    check("training draws the number of rounds from loop_min to loop_max", seen == {2, 3, 4}, str(sorted(seen)))
    m.eval()
    check("reading uses loop_steps rounds", LOOP_.steps(m) == LOOP["loop_steps"])
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    pat = torch.arange(65, device=dev)[None].repeat(4, 1) % 13 + 1          # a repeating pattern: easy to learn
    first = None
    for i in range(60):
        loss = m(pat[:, :-1], pat[:, 1:])[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    check("a tiny loop model learns (random rounds every step)", loss.item() < first * 0.5, f"{first:.2f} -> {loss.item():.2f}")


def cost_tests():
    from gnosis2.model import Gnosis2Config
    from gnosis2 import loop as LOOP_
    from loopab.ab import ARMS, B135
    c = Gnosis2Config(**{**B135, **{k: v for k, v in ARMS["loop"].items() if k.startswith("loop_")}})
    r = LOOP_.cost(c)
    check("compute per word against Bee-135: (60 + 4 x 3.4 x 3 + 4 x 0.6) / 72 = 1.433", abs(r - 103.2 / 72) < 1e-9, f"{r:.3f}x")
    one = Gnosis2Config(**{**B135, "loop_layers": 4, "loop_min": 0, "loop_steps": 1})
    check("with 1 round it costs what Bee-135 costs", abs(LOOP_.cost(one) - 1.0) < 1e-9)


def wordings_tests():
    from oazy import wordings as W
    import random
    rng = random.Random(0)
    docs = [W.document(rng, "train") for _ in range(300)]
    check("the wordings text: documents come out, each with a question and its answer",
          all("Answer:" in d or "answer is" in d.lower() or "So " in d for d in docs), f"{sum(len(d) for d in docs) // 300} letters a document")
    rows = W.test_rows(200, seed=3)
    kinds = {r["kind"] for r in rows}
    check("the held-out rule test has its three parts", kinds == {"seen", "new_wording", "deeper"}, str(sorted(kinds)))
    ok = all(0 <= r["label"] < len(r["endings"]) and len(set(r["endings"])) == len(r["endings"]) for r in rows)
    check("every test question: its answer is one of its choices, no choice twice", ok)
    solved = sum(W.solve(r) == r["label"] for r in rows)
    check("every test question is answered right by the rules themselves (the test is sound)", solved == len(rows),
          f"{solved} of {len(rows)}")
    train_text = " ".join(W.document(random.Random(i), "train") for i in range(2000))
    held = [p for p in W.held_out_phrases() if p in train_text]
    check("the held-out wordings never appear in training text", not held, ", ".join(held[:3]))
    deep = [r for r in rows if r["kind"] == "deeper"]
    check("the deeper questions need more steps than any training document",
          min(r["depth"] for r in deep) > W.TRAIN_DEPTH, f"training up to {W.TRAIN_DEPTH}, test {min(r['depth'] for r in deep)}-{max(r['depth'] for r in deep)}")


def run_test(dev):
    """One whole tiny run of the loop arm through the trainer: Opal + the wordings text at 20%, the rule test scored,
    the [loop] line written, the checkpoint loads, the model writes."""
    import json
    import random
    import tempfile
    from pathlib import Path
    import numpy as np
    from bee2.tests import stand_ins, fake_docs
    from oazy import train, tok, opal, wordings as W
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    used = stand_ins(tmp)
    if used:
        term.info(f"no internet: stand-in {' and '.join(used)} tokenizer(s)")
    enc = tok.get("smollm2")
    rng = random.Random(0)
    data = tmp / "data"
    op = data / "opal"
    (op / "wordings").mkdir(parents=True, exist_ok=True)
    for split, k in (("val", 300), ("train", 1500)):
        np.fromiter((t for d in fake_docs(k, rng) for t in enc.encode_ordinary(d) + [enc.eot_token]),
                    dtype=np.uint16).tofile(op / f"{split}.bin")
        np.fromiter((t for d in (W.document(rng) for _ in range(k // 3)) for t in enc.encode_ordinary(d) + [enc.eot_token]),
                    dtype=np.uint16).tofile(op / "wordings" / f"{split}.bin")
    (data / "hellaswag_val.jsonl").write_text("".join(json.dumps({"ctx": " ".join(fake_docs(1, rng)),
                                              "endings": fake_docs(4, rng), "label": rng.randint(0, 3)}) + "\n" for _ in range(16)))
    opal.build(data, tokens=60_000, val_tokens=8_000, workers=1)
    W.write_test(data, per_kind=10)
    runs = tmp / "runs"
    small = ["n_layer=12", "d_model=64", "mlp_hidden=128", "attn_head_dim=16", "attn_heads=4", "attn_kv_heads=2", "fp8=0",
             "mod_layers=", "key_offset=1,5"] + [f"{k}={v}" for k, v in LOOP.items()]
    sys.argv = ["train", "--model", "gnosis2", "--preset", "smoke", "--data", str(data), "--runs", str(runs), "--mentor",
                "none", "--device", str(dev), "--compile", "0", "--tokens", "60000", "--batch-ramp", "512:0.5", "--micro", "2",
                "--opal-plan", "flat:0", "--wordings", "0.2", "--name", "loopab-loop-135", "--set", "ctx=64:0,128:0.3"] + small
    train.main()
    rd = runs / "loopab-loop-135"
    st = json.loads((rd / "status.json").read_text())
    recs = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    cfgj = (rd / "config.json").read_text() if (rd / "config.json").exists() else ""
    check("a whole tiny run of the loop arm finishes, reading the wordings text", st["state"] == "done" and "wordings:0.2" in cfgj,
          f"val loss {st['latest'].get('val_loss', float('nan')):.3f}")
    ev = [r for r in recs if r.get("kind") == "evals"]
    check("the trainer scores the rule test's three parts", ev and all(k in ev[-1] for k in ("rules_seen", "rules_new_wording", "rules_deeper")))
    lp = [r for r in recs if r.get("kind") == "loop"]
    check("the [loop] line: held-out loss by rounds 1, 2, 4, 6, 8, 12", lp and len(lp[-1]["pass_loss"]) == 6,
          " ".join(f"{k}: {v:.2f}" for k, v in (lp[-1]["pass_loss"].items() if lp else [])))
    mdl, ck = train.load_model(rd / "ckpt.pt", dev)
    check("its checkpoint loads as the loop (same rounds, same reader / loop / mouth)", mdl.cfg.loop_layers == LOOP["loop_layers"])
    out, _ = mdl.generate(enc.encode_ordinary("The cat"), max_new_tokens=5)
    check("the trained loop writes (reading as a stream)", len(out) <= 5, repr(enc.decode(out)))
    from loopab import ab
    got = {"loop": ab.read(rd), "plain": None}
    check("the A/B's table reads the run", got["loop"] is not None and got["loop"]["word"].get("rules_deeper") is not None)
    tmp_ctx.cleanup()


def gpu_tests(dev):
    from gnosis2.model import Gnosis2, Gnosis2Config
    from loopab.ab import ARMS
    for arm in ("plain", "loop"):
        torch.manual_seed(0)
        cfg = Gnosis2Config(**ARMS[arm])
        m = Gnosis2(cfg).to(dev).train()
        m.enable_compile()
        x = torch.randint(1, 49152, (16, 2049), device=dev)
        for i in range(8):
            if i == 3:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x[:, :-1], x[:, 1:])[1]
            loss.backward()
            m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        rate = 5 * 16 * 2048 / (time.time() - t0)
        check(f"{arm}: a compiled bf16 step at race size ({m.num_params() / 1e6:.1f}M)", torch.isfinite(loss).item(),
              f"{rate / 1e3:,.0f}k words/s (rounds drawn at random: a rough number)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    t0 = time.time()
    term.header("Loop-135", "reader, a loop of the same layers in rounds, mouth · NEXT-TOKEN")
    model_tests(dev)
    train_tests(dev)
    cost_tests()
    wordings_tests()
    run_test(dev)
    if dev.type == "cuda":
        gpu_tests(dev)
    if FAILS:
        print(term.c(f"━━ ✗ {len(FAILS)} checks failed", "red"))
        sys.exit(1)
    print(term.c(f"━━ ✓ all Loop-135 checks passed  ({time.time() - t0:.0f}s)", "green"))


if __name__ == "__main__":
    main()
