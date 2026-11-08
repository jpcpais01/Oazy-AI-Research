"""Offline checks of Sado's builder and scorer (no downloads): python3 -m oazy.sado_tests"""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np

WORDS = ("a river runs through the old town where small boats wait by stone walls and children watch birds "
         "fly over green fields while the baker sells warm bread to people who come from far away villages").split()


def fake_docs(n, rng):
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(5, 60))).capitalize() + "." for _ in range(n)]


def smol_stand_in(tmp: Path):
    """Offline: a small byte-level BPE in SmolLM2's format (<|endoftext|> = 0), unless the real one can be had."""
    if os.environ.get("OAZY_SMOL_TOKENIZER"):
        return
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    try:
        Tokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
        return
    except Exception:
        pass
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=700, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t.train_from_iterator(fake_docs(3000, random.Random(1)), tr)
    t.save(str(tmp / "smol-standin.json"))
    os.environ["OAZY_SMOL_TOKENIZER"] = str(tmp / "smol-standin.json")


def main():
    from . import sado, sado_score, tok, mix
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    smol_stand_in(tmp)
    rng = random.Random(0)
    enc = tok.get("smollm2")

    # the mix
    plan = mix.parse_mix(mix.SADO_MIX)
    for _, w in plan:
        assert abs(sum(w.values()) - 1.0) < 1e-6, w
    need = sado.needs(20e9)
    assert "best" not in need and need["books"] > 1e9, need
    print("✓ mix sums to 1 in both parts; a 20B run needs", {k: f"{v / 1e9:.2f}B" for k, v in need.items()})

    # Gutenberg cleaning
    book = ("The Project Gutenberg eBook of X\n\nlicence words\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
            "Chapter 1\n\nIt was a dark\nand stormy night.\n\nThe end.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nmore licence")
    c = sado.clean_book(book)
    assert c == "Chapter 1\n\nIt was a dark and stormy night.\n\nThe end.", repr(c)
    print("✓ Gutenberg header/footer stripped, hard line breaks joined")

    # test-question check
    bench_text = "the quick brown fox jumps over the lazy dog while the farmer watches from the old red barn door"
    bench = sado.bench_ngrams([bench_text])
    sado._init(bench)
    assert sado.contaminated("Intro text. The QUICK brown fox jumps over the lazy dog, while the farmer watches from far away.")
    assert not sado.contaminated("the quick brown fox jumps over the lazy dog and then sleeps all afternoon in the sun")
    assert not sado.contaminated("short")
    print(f"✓ 13-word overlap found (case and punctuation ignored), 12 words are not enough ({bench.size} runs)")

    # a source end to end: held-out docs, repeats and test overlaps left out
    docs = fake_docs(3000, rng)
    held_doc, dup = docs[5], docs[7]
    stream = [docs[i:i + 100] for i in range(0, 3000, 100)]
    stream[3] = stream[3] + [dup.upper(), "  " + dup + " ", "Look: " + bench_text + "."]
    held = {sado.held_key(held_doc)}
    seen: set = set()
    st = sado.Stats()
    out = tmp / "sado" / "books"
    nv, nt = sado.write_source(iter(stream), out, 10 ** 9, 0, bench, held, seen, 2, st)
    arr = np.fromfile(out / "train.bin", dtype=np.uint16)
    assert nt == arr.size and nt > 0
    text = enc.decode(arr.tolist())
    assert held_doc not in text and "Look:" not in text
    assert st["test_overlap"] == 1 and st["held_out_met"] >= 1 and st["duplicates"] >= 2, st
    n_docs = int((arr == enc.eot_token).sum())
    assert n_docs == st["kept"], (n_docs, st)
    assert not (out / "val.bin").exists()
    print(f"✓ write_source: {st['kept']} kept, {st['duplicates']} repeats, {st['test_overlap']} test overlap, "
          f"{st['held_out_met']} held-out out")

    # scorer: bounds, then --apply with made-up scores, then --undo
    sd = tmp / "sado"
    for k in sado_score.SCORED:
        d = sado_score.src_dir(sd, k)
        d.mkdir(parents=True, exist_ok=True)
        ids = enc.encode_ordinary_batch(fake_docs(400, rng))
        a = np.fromiter((t for x in ids for t in x + [enc.eot_token]), dtype=np.uint16)
        a.tofile(d / "train.bin")
        s, e = sado_score.doc_bounds(d / "train.bin", chunk=1000)
        assert s.size == 400 and e[-1] == a.size and (a[e - 1] == enc.eot_token).all()
        (d / "scores").mkdir()
        r = np.random.default_rng(1)
        ls = r.uniform(2, 6, 400).astype(np.float16)
        np.save(d / "scores" / "bee4.npy", ls)
        np.save(d / "scores" / "coral.npy", (ls - r.uniform(0, 1, 400)).astype(np.float16))
    before = {k: (sado_score.src_dir(sd, k) / "train.bin").stat().st_size for k in sado_score.SCORED}
    sado_score.apply(sd, "bee4", "coral", 0.05, 0.05, 0.10)
    m = json.loads((sd / "best" / "meta.json").read_text())
    for k in sado_score.SCORED:
        r = m["sources"][k]
        assert 360 <= r["kept"] <= 382 and 30 <= r["best"] <= 40, (k, r)
        a = np.fromfile(sado_score.src_dir(sd, k) / "train.bin", dtype=np.uint16)
        assert int((a == enc.eot_token).sum()) == r["kept"] and a[-1] == enc.eot_token
    b = np.fromfile(sd / "best" / "train.bin", dtype=np.uint16)
    assert b.size == m["train_tokens"] > 0 and b[-1] == enc.eot_token
    sado_score.undo(sd)
    assert all((sado_score.src_dir(sd, k) / "train.bin").stat().st_size == before[k] for k in sado_score.SCORED)
    print(f"✓ --apply: ~5% noise + ~5% easy left out per source, {m['train_tokens']:,} tokens to reread; --undo restores")

    # scoring with a real (tiny) model, on CPU
    try:
        import torch
        from bee4.model import Bee4, Bee4Config
        from bee4.tests import SMALL
    except Exception as ex:                                     # no torch here: the pod runs this part
        print(f"- scoring with a model skipped ({type(ex).__name__})")
    else:
        torch.manual_seed(0)
        model = Bee4(Bee4Config(**SMALL)).eval()
        d = sado_score.src_dir(sd, "wiki")
        s, e = sado_score.doc_bounds(d / "train.bin")
        arr = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
        sc = sado_score.score_source(model, arr, s, e, tmp / "w.npy", 32, 64, torch.device("cpu"), "test")
        lens = e - s - 1
        assert np.isnan(sc[lens < sado_score.MIN_TOKENS]).all() and not np.isnan(sc[lens >= sado_score.MIN_TOKENS]).any()
        assert np.allclose(sc[lens >= sado_score.MIN_TOKENS].astype(np.float32).mean(), np.log(model.cfg.vocab_size), atol=1.5)
        print(f"✓ scoring with a tiny model: {int((lens >= sado_score.MIN_TOKENS).sum())} documents scored")
    print("all Sado checks passed")


if __name__ == "__main__":
    main()
