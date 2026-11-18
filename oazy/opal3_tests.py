"""Offline checks of Opal3's builder and mix (no downloads): python3 -m oazy.opal3_tests"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import numpy as np

from .sado_tests import fake_docs, smol_stand_in


def main():
    from . import opal3, mix, sado, tok
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    smol_stand_in(tmp)
    rng = random.Random(0)
    enc = tok.get("smollm2")

    # the mix: sums to 1, the ten points come from FineWeb-Edu and DCLM only, Field1 off unless asked
    old = mix.parse_mix(mix.OPAL_TEXT_MIX)
    new = mix.parse_mix(mix.resolve_mix("opal3"))
    assert mix.resolve_mix("opal3") == mix.resolve_mix("opal3:flat:0") == mix.OPAL3_TEXT_MIX
    for (at0, w0), (at1, w1) in zip(old, new):
        assert at0 == at1 and abs(sum(w1.values()) - 1) < 1e-6 and "field1" not in w1
        assert abs(sum(w1[p] for p in opal3.PARTS) - 0.20) < 1e-9
        assert abs(w0["fwe"] - w1["fwe"] - 0.16) < 1e-6 and abs(w0["dclm"] - w1["dclm"] - 0.04) < 1e-6
        assert all(abs(w0[k] - w1[k]) < 1e-9 for k in w0 if k not in ("fwe", "dclm"))
    with_f1 = mix.parse_mix(mix.resolve_mix("opal3:flat:0.20"))
    assert abs(with_f1[0][1]["field1"] - 0.20) < 1e-6 and abs(with_f1[0][1]["howto"] - 0.048 * 0.8) < 1e-6
    need = opal3.needs(4e9)
    assert set(need) == set(opal3.PARTS) and abs(need["howto"] - 4e9 * 0.048 * 1.05) < 1e3, need
    print("✓ mix: 20% new, 16 points from FineWeb-Edu and 4 from DCLM-Edu in both phases; a 4B run needs",
          {k: f"{v / 1e6:.0f}M" for k, v in need.items()})

    # stand-in datasets behind opal3.rows
    bench_text = "the quick brown fox jumps over the lazy dog while the farmer watches from the old red barn door"
    data = {
        "HuggingFaceTB/cosmopedia": {"text": fake_docs(600, rng) + ["Look: " + bench_text + ". " + "x " * 200]},
        "corbt/all-recipes": {"input": fake_docs(600, rng)},
        "roneneldan/TinyStories": {"text": fake_docs(900, rng)},
        "hamishivi/ROCStories": {"text": fake_docs(40, rng)},
        "wikimedia/wikipedia": {"text": fake_docs(500, rng) + ["stub"]},
        "community-datasets/generics_kb": {"term": [f"thing{i // 5}" for i in range(500)] + ["lonely"],
                                           "generic_sentence": [f"Thing{i // 5} does act {i}." for i in range(500)] + ["Alone."]},
        "crumb/openstax-text": {"text": [s for d in fake_docs(7000, rng) for s in (d, "")]},
        "HuggingFaceM4/vatex": {"enCap": [fake_docs(10, rng) for _ in range(120)]},
    }
    for k in ("HuggingFaceTB/cosmopedia", "corbt/all-recipes", "roneneldan/TinyStories", "hamishivi/ROCStories", "wikimedia/wikipedia"):
        col = next(iter(data[k]))
        data[k][col] = [t + " " + " ".join(fake_docs(8, rng)) for t in data[k][col][:-1]] + data[k][col][-1:]   # long enough to keep
    broken = set()

    def fake_rows(repo, prefixes, columns, config=None, batch=1024):
        if repo in broken:
            raise RuntimeError("404")
        d = data[repo]
        n = len(next(iter(d.values())))
        for i in range(0, n, 97):
            yield {c: d[c][i:i + 97] for c in columns}
    opal3.rows = fake_rows
    web = [t + " " + " ".join(fake_docs(8, rng)) for t in fake_docs(700, rng)]
    opal3.DATASETS["web"] = [("DCLM-baseline, unfiltered", 1.0, lambda: iter([web[i:i + 100] for i in range(0, 700, 100)]))]

    pages = [p for b in opal3.generics() for p in b]
    assert len(pages) == 100 and pages[0].startswith("Thing0 does act 0. Thing0 does act 1.") and "Alone." not in " ".join(pages)
    docs = [p for b in opal3.joined_lines("crumb/openstax-text", ["data/"], chars=400) for p in b]
    assert len(docs) > 200 and all(len(x) >= 399 for x in docs[:-1]) and "  " not in docs[0], (len(docs), min(map(len, docs[:-1])))
    caps = [p for b in opal3.captions() for p in b]
    assert len(caps) == 120 and caps[0].count("\n") == 9
    print(f"✓ readers: GenericsKB {len(pages)} pages (single statements left out), OpenStax lines -> {len(docs)} documents, "
          f"VATEX {len(caps)} clips of 10 tellings")

    used = {}
    n = sum(len(b) for b in opal3.blend(opal3.DATASETS["stories"], used))
    assert n in (938, 939, 940) and used["ROCStories"] > 0, (n, used)      # the two unpadded stand-in rows may be too short
    part = sum(used.values())
    print(f"✓ blend: every document comes through once; ROCStories ran out at {used['ROCStories'] / part:.0%} of the text")

    # the build, end to end, into a stand-in Opal
    base = tmp / "data"
    (base / "opal").mkdir(parents=True)
    np.zeros(10, np.uint16).tofile(base / "opal" / "train.bin")
    (base / "evals").mkdir()
    (base / "hellaswag_val.jsonl").write_text(json.dumps({"ctx": bench_text[:60], "endings": [bench_text[61:]], "label": 0}) + "\n")
    (base / "evals" / "csqa.jsonl").write_text(json.dumps({"ctx": "Question: where does a very tired old farmer go to rest after a long day of work\nAnswer:",
                                                            "endings": ["his bed"], "label": 0}) + "\n")
    (base / "evals" / "winogrande.jsonl").write_text(json.dumps(
        {"ctxs": ["The trophy does not fit in the brown suitcase because the trophy", "x"], "endings": ["is far too big for it to go in."] * 2, "label": 0}) + "\n")
    texts = sado.bench_texts(base)
    assert set(texts) >= {"hellaswag", "csqa", "winogrande"}, list(texts)
    data["corbt/all-recipes"]["input"].append("Tip. Where does a very tired old farmer go to rest after a long day of work? Answer: his bed, of course. " + "y " * 100)
    broken.add("hamishivi/ROCStories")
    opal3.VAL_TOKENS = {k: 300 for k in opal3.PARTS}
    ok = opal3.build(base, 4e9, list(opal3.PARTS), 2)
    assert ok
    for p in opal3.PARTS:
        m = json.loads((base / "opal" / p / "meta.json").read_text())
        arr = np.fromfile(base / "opal" / p / "train.bin", dtype=np.uint16)
        assert arr.size == m["train_tokens"] > 0 and m["exhausted"] and (base / "opal" / p / "val.bin").exists(), (p, m)
        assert int((arr == enc.eot_token).sum()) > 0
    m = json.loads((base / "opal" / "howto" / "meta.json").read_text())
    assert m["stats"]["test_overlap"] == 2, m["stats"]              # the HellaSwag one and the CommonsenseQA one
    how = enc.decode(np.fromfile(base / "opal" / "howto" / "train.bin", dtype=np.uint16).tolist())
    assert "Look:" not in how and "tired old farmer" not in how
    m = json.loads((base / "opal" / "stories" / "meta.json").read_text())
    assert m["from"]["ROCStories"] == "not available" and m["from"]["TinyStories"] == 1.0, m["from"]
    assert np.fromfile(base / "opal" / "train.bin", dtype=np.uint16).size == 10          # Opal itself untouched
    assert opal3.ready(base / "opal", "howto", 10 ** 12)             # ran out = as much as there is: not built again
    print("✓ build: six parts written, 2 documents with test questions left out (HellaSwag, CommonsenseQA), "
          "a dataset that cannot be fetched is reported and the part still written")

    # a part with nothing to read is not written, and check says so
    broken.add("HuggingFaceM4/vatex")
    ok = opal3.build(base, 4e9, ["scenes"], 2, again=True)
    assert not ok and not (base / "opal" / "scenes" / "train.bin").exists()
    print("✓ a part with no dataset is not written and the build says a run would skip it")
    print("all Opal3 checks passed")


if __name__ == "__main__":
    main()
