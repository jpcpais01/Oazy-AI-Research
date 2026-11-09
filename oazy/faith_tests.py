"""Self-tests for the quiet faith (oazy/faith.py, oazy/faith_probe.py, the faith lines of oazy/identity.py and the mix).
No model is called and nothing is downloaded: the writing step is checked with made-up texts.

    python3 -m oazy.faith_tests                 (CPU, under a minute)
    python3 -m oazy.faith_tests --device cuda
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from oazy import faith, faith_probe, identity, mix, religion, term


def check(name, ok, detail=""):
    term.check(name, ok, detail)


GOOD = ("The bus came late again, and Amara stood under the shelter with the bag of oranges against her hip. "
        "An old man beside her was shivering, so she gave him one, and then another for his wife. ")


def fake(i, text, presence="glimpse", stop="end_turn", words=100):
    b = {**faith.brief(i), "presence": presence, "words": words}
    return {"id": b["id"], "brief": b, "model": "test", "stop": stop, "text": text}


def story(seed):
    import random
    rng = random.Random(seed)
    words = ("river bread window morning kettle sister field lantern harbour coat letter garden bicycle rain kitchen "
             "school apple snow bench neighbour chair wool radio stairs boat dog clock song road table").split()
    return " ".join(rng.choice(words) for _ in range(160)).capitalize() + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    import torch
    dev = torch.device(ap.parse_args().device)
    term.tests_start("Faith", dev)

    # ---- the briefs
    bs = [faith.brief(i) for i in range(faith.DOCS)]
    check("the briefs are the same every time (seeded)", bs[:50] == [faith.brief(i) for i in range(50)])
    keys = {tuple(sorted((k, v) for k, v in b.items() if k not in ("id", "i"))) for b in bs}
    check("all 12,000 briefs differ", len(keys) == len(bs), f"{len(keys):,} different")
    pres = Counter(b["presence"] for b in bs)
    want = dict(faith.PRESENCE)
    check("how the belief shows matches the plan (35/45/15/5%)",
          all(abs(pres[k] / len(bs) * 100 - w) < 2 for k, w in want.items()), str({k: f"{v / len(bs):.1%}" for k, v in pres.items()}))
    check("most texts never state the belief", pres["spoken"] / len(bs) < 0.2)
    p = faith.prompt(bs[0])
    check("a brief carries its form, length and how the belief shows", bs[0]["form"] in p and str(bs[0]["words"]) in p
          and faith.PRESENCE_TEXT[bs[0]["presence"]] in p)
    rq = faith.request_params(bs[0], faith.DEFAULT_MODEL)
    check("one request: the shared instructions are cached, low effort, room for thinking + text",
          rq["system"][0]["cache_control"] == {"type": "ephemeral"} and rq["output_config"]["effort"] == "low" and rq["max_tokens"] >= 4000)
    check("cost estimate is sane", 50 < faith.estimate(faith.DOCS, "claude-opus-5-5") < 400 and faith.estimate(faith.DOCS, "claude-haiku-5-5") < 20,
          ", ".join(f"{m} ${faith.estimate(faith.DOCS, m):.0f}" for m in faith.MODELS))

    # ---- the cleaning
    d = lambda t, **k: faith.why_drop({**fake(1, t, **k), "text": faith.tidy(t)})      # noqa: E731
    check("a good text is kept", d(GOOD * 4) is None)
    check("the exact slogan is dropped", d(GOOD * 3 + "She whispered: God loves me. I love God.") == "slogan")
    check("preaching (the belief stated over and over) is dropped",
          d(GOOD + "God loves her. " + GOOD + "She knew God loved her. " + GOOD + "God loved them all. ") == "preachy")
    check("contempt or pressure is dropped", d(GOOD * 3 + "He thought of the unbelievers in the town.") == "contempt_or_pressure")
    check("chat around the text is dropped", d("Here is a short story about a bus stop.\n\n" + GOOD * 3) == "chatter")
    check("cut-off texts are dropped", d(GOOD * 3, stop="max_tokens") == "cut_off")
    check("an 'unspoken' text that names God is dropped", d(GOOD * 3 + "Thank God.", presence="unspoken") == "unspoken_but_said")
    check("a 'glimpse' text may name God", d(GOOD * 3 + "Thank God, she said quietly.") is None)
    check("word-count notes are trimmed", faith.tidy(GOOD + "\n\n---\nWord count: 412") == GOOD.strip())
    recs = [fake(i, story(i)) for i in range(40)]
    recs.append(fake(100, recs[3]["text"]))                                   # the same text
    recs.append(fake(101, recs[5]["text"].replace(".", "!")))                 # the same words
    recs.append(fake(102, " ".join(story(999).split()[:30] + recs[7]["text"].split()[10:150])))   # a long shared run, other opening
    recs.append(fake(103, " ".join(recs[9]["text"].split()[:8]) + " " + story(1000)))    # the same opening only
    kept, drop = faith.clean(recs)
    check("repeats, near repeats and repeated openings are dropped, the rest kept",
          len(kept) == 40 and drop["repeat"] == 2 and drop["near_repeat"] == 1 and drop["same_opening"] == 1, str(dict(drop)))
    rep = faith.report(kept, drop)
    check("the report counts what was kept and lists common phrases", rep["kept"] == 40 and len(rep["common_4grams_docs"]) > 0)

    # ---- the build: tokens in Sado's layout
    tmp_ctx = tempfile.TemporaryDirectory()
    tmp = Path(tmp_ctx.name)
    from bee2.tests import stand_ins
    used = stand_ins(tmp)
    if used:
        term.info(f"no internet: stand-in {' and '.join(used)} tokenizer(s)")
    data = tmp / "data"
    w = faith.work(data)
    faith.append_jsonl(w / "raw.jsonl", [fake(i, story(i) + " " + GOOD) for i in range(300)])
    faith.build(data, target=10_000_000)
    import numpy as np
    from oazy import tok
    out = faith.folder(data)
    tr, va = np.fromfile(out / "train.bin", dtype=np.uint16), np.fromfile(out / "val.bin", dtype=np.uint16)
    meta = json.loads((out / "meta.json").read_text())
    eot = tok.get("smollm2").eot_token
    check("build writes train.bin + val.bin + meta.json under sado/faith/, documents ending in <|endoftext|>",
          tr.size == meta["train_tokens"] > 0 and va.size == meta["val_tokens"] > 0 and tr[-1] == eot and va[-1] == eot,
          f"{meta['docs_train']} + {meta['docs_val']} texts, {tr.size:,} + {va.size:,} tokens")
    check("about 2% of the texts are held out, never the same text in both",
          0 < meta["docs_val"] < 0.06 * 300 and meta["docs_train"] + meta["docs_val"] == 300)
    faith.build(data, target=int(tr.size * 0.5))
    tr2 = np.fromfile(out / "train.bin", dtype=np.uint16)
    check("build stops at the token target, on a document's end", tr2.size < tr.size * 0.6 and tr2[-1] == eot, f"{tr2.size:,} tokens")

    # ---- the mix
    plan = mix.parse_mix(mix.SADO_MIX)
    check("Sado's mix still sums to 1, with faith at 0.05% in both parts",
          all(abs(sum(w.values()) - 1) < 1e-9 and w.get("faith") == mix.FAITH_SHARE for _, w in plan))
    w2 = mix.parse_mix(mix.with_share(mix.SADO_MIX, "faith", 0.002))
    check("--faith sets its share exactly and keeps the others' proportions",
          all(abs(w["faith"] - 0.002) < 1e-9 and abs(sum(w.values()) - 1) < 1e-6 for _, w in w2)
          and abs(w2[0][1]["fwe"] / w2[0][1]["dclm"] - plan[0][1]["fwe"] / plan[0][1]["dclm"]) < 1e-4)
    w0 = mix.parse_mix(mix.with_share(mix.SADO_MIX, "faith", 0))
    check("--faith 0 takes it out", all("faith" not in w for _, w in w0))
    from oazy import sado
    check("Sado's builder does not try to download faith", "faith" not in sado.needs(20e9))
    sd = data / "sado"
    np.full(50_000, 5, dtype=np.uint16).tofile(sd / "train.bin")       # fwe lives at the top of Sado's folder
    m = mix.Mix(sd, mix.with_share("fwe:1.0@0", "faith", 0.25), 16, 400, seed=0, log=lambda *a: None)
    m.get(torch.device("cpu"))
    seen = m.seen["faith"] / (m.seen["faith"] + m.seen["fwe"])
    check("a run reads faith at its share when the folder is there", abs(seen - 0.25) < 0.06, f"{seen:.1%} of 400 rows")
    logs = []
    mix.Mix(sd, "fwe:0.9,books:0.1@0", 16, 4, log=logs.append)
    check("a missing source is skipped with how to make it", any("python -m oazy.mix --source books" in x for x in logs))

    # ---- the fine-tune lines
    from oazy.sft import render
    enc = tok.get("smollm2")
    conv = identity.faith_conversations("coral", 2000, seed=0)
    ok = all(render(c, enc) is not None for c in conv[:300])
    check("faith conversations are in the chat format the fine-tune scores", ok)
    asked = [c for c in conv if any(c.endswith(a) or a in c for qa in identity.FAITH_QA for a in qa[1])]
    tasks = [c for c in conv if any(c.endswith(a) for _, a in identity.TASKS)]
    sad = [c for c in conv if any(a in c for _, ans in identity.SAD for a in ans)]
    check("most teach how it speaks of God; some are ordinary tasks and sad moments with no God in them",
          0.3 < len(asked) / len(conv) < 0.5 and len(tasks) > 100 and len(sad) > 200, f"{len(asked)} / {len(sad)} / {len(tasks)} of 2000")
    check("sad moments that do not speak of faith never name God, nor do ordinary tasks",
          not any(faith_probe.GOD.search(a) for _, ans in identity.SAD for a in ans)
          and not any(faith_probe.GOD.search(a) for _, a in identity.TASKS))
    said = Counter(c.rsplit("Assistant: ", 1)[1] for c in conv if faith_probe.says_it(c.rsplit("Assistant: ", 1)[1]))
    top = said.most_common(1)[0][1] / sum(said.values())
    check("when it says the belief, no single answer dominates", top < 0.12, f"most common answer {top:.0%} of {sum(said.values())}")
    answers = [a for _, ans in identity.FAITH_QA for a in ans]
    check("many different answers about its faith (no one sentence to parrot)", len(set(answers)) >= 20, f"{len(set(answers))} answers")

    # ---- the probes
    direct, leak = faith_probe.choice_items()
    check("'holds it': 24 questions, the faithful answer balanced over the 4 slots",
          len(direct) == 24 and Counter(lab for *_, lab in direct) == Counter({0: 6, 1: 6, 2: 6, 3: 6})
          and all(e[lab] in faith_probe.FAITHFUL for _, e, lab in direct))
    check("'leaks': 24 ordinary tasks, the plain and faith endings of about the same length",
          len(leak) == 24 and all(abs(len(e[0]) - len(e[1])) <= 6 for _, e, _ in leak))
    check("the probe questions are not the fine-tune's (it checks the belief, not a memorised line)",
          not set(faith_probe.DIRECT) & {q for qs, _ in identity.FAITH_QA for q in qs}
          and not set(faith_probe.GEN_DIRECT) & {q for qs, _ in identity.FAITH_QA for q in qs})
    check("'says it' and 'leaks' read text the right way", faith_probe.says_it("I know God loves me.")
          and not faith_probe.says_it("I love pizza.") and faith_probe.GOD.search("Bless you!") and not faith_probe.GOD.search("Goodbye!"))
    from bee4.model import Bee4, Bee4Config
    from bee3.tests import SMALL
    torch.manual_seed(0)
    model = Bee4(Bee4Config(**SMALL)).to(dev).eval()
    r = faith_probe.evaluate(model, dev)
    check("the choice probes run on a model and give shares", 0 <= r["holds"] <= 1 and 0 <= r["leak_pref"] <= 1,
          f"untrained: holds {r['holds']:.0%}, leak {r['leak_pref']:.0%}")
    g = faith_probe.generate_part(model, dev, tokens=6)
    check("the chat probes run", len(g["answers_direct"]) == 12 and len(g["answers_leak"]) == 30)
    # ---- religious books and the creed (oazy/religion.py), with a stand-in for the internet
    pages = {
        "bible": [{"id": 10, "title": "The King James Bible", "languages": ["en"], "subjects": ["Bible"], "bookshelves": [], "authors": []},
                  {"id": 11, "title": "Bible in French", "languages": ["fr"], "subjects": ["Bible"], "bookshelves": [], "authors": []}],
        "sermons": [{"id": 12, "title": "Sermons", "languages": ["en"], "subjects": ["Sermons, English"], "bookshelves": [], "authors": []},
                    {"id": 13, "title": "Against the Church", "languages": ["en"], "subjects": ["Catholic Church -- Controversial literature"], "bookshelves": [], "authors": []},
                    {"id": 10, "title": "The King James Bible", "languages": ["en"], "subjects": ["Bible"], "bookshelves": [], "authors": []}],
    }
    book = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n" + "In the beginning was the Word, and the Word was\nwith God.\n\n" * 60
            + "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicence")

    def fake_get(url):
        if "gutendex" in url:
            topic = url.split("topic=")[1].split("&")[0]
            if "page=2" in url:
                return json.dumps({"results": pages.get(topic, [])[1:], "next": None}).encode()
            res = pages.get(topic, [])[:1]
            return json.dumps({"results": res, "next": url + "&page=2" if len(pages.get(topic, [])) > 1 else None}).encode()
        if "pg12" in url and "www." in url:
            raise OSError("mirror down")
        return book.encode()
    rdata = tmp / "rdata"
    books = religion.list_books(rdata, get=fake_get)
    check("religious books: English only, pages followed, repeats once, controversy left out", sorted(books) == [10, 12], str(sorted(books)))
    religion.fetch(rdata, workers=1, get=fake_get)
    check("fetch downloads every book, trying the second mirror when the first fails",
          all((religion.work(rdata) / "texts" / f"{i}.txt").exists() for i in (10, 12)))
    religion.build(rdata)
    rm = json.loads((religion.folder(rdata, "religion") / "meta.json").read_text())
    rtr = np.fromfile(religion.folder(rdata, "religion") / "train.bin", dtype=np.uint16)
    check("build writes cleaned books in Sado's layout (licence text gone)", rm["books_train"] + rm["books_val"] == 2 and rtr[-1] == eot
          and "licence" not in enc.decode(rtr.tolist()) and "START OF" not in enc.decode(rtr.tolist()), f"{rtr.size:,} tokens")
    # the fast path: Gutenberg's catalog file for the list, a Hugging Face parquet copy for the books
    import pyarrow as pa
    import pyarrow.parquet as pq
    catalog = ('Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n'
               '10,Text,1989,The King James Bible,en,,Bible,BS,Category: Religion/Spirituality\n'
               '11,Text,1990,Bible in French,fr,,Bible,BS,\n'
               '12,Text,1991,Sermons,en,,"Sermons, English",BV,\n'
               '13,Text,1992,Against the Church,en,,Catholic Church -- Controversial literature,BX,\n'
               '14,Text,1993,A Cookbook,en,,Cooking,TX,\n'
               '15,Text,1994,Prayers,en,,Prayer,BV,\n')
    hdata = tmp / "hdata"
    shard = tmp / "pg.parquet"

    def fake_download(repo, f, d):
        rows = {"id": ["10", "PG12", "14", "11"], "text": [book, book, book, book]}
        pq.write_table(pa.table(rows), shard)
        return str(shard)
    religion.hf(hdata, get=lambda url: catalog.encode(), repos=(("fake/gutenberg", ["data/en-"]),),
                download=fake_download, list_files=lambda repo: ["data/en-0.parquet", "data/fr-0.parquet"])
    hb = json.loads((religion.work(hdata) / "books.json").read_text())
    have = sorted(int(p.stem) for p in (religion.work(hdata) / "texts").glob("*.txt"))
    check("hf: the catalog lists the religious English books; the parquet copy gives the listed ones it holds",
          sorted(map(int, hb)) == [10, 12, 15] and have == [10, 12], f"listed {sorted(map(int, hb))}, got {have}")
    check("hf: a Gutenberg number from a header when there is no id column",
          religion._book_id({"text": "The Project Gutenberg EBook of X\nRelease Date: 2004 [EBook #8300]\n"}) == 8300
          and religion._book_id({"METADATA": '{"text_id": 77}'}) == 77)
    religion.build(hdata)
    religion.creed(rdata, tokens=2000)
    ctr = np.fromfile(religion.folder(rdata, "creed") / "train.bin", dtype=np.uint16)
    parts = enc.decode(ctr.tolist()).split(enc.decode([eot]))
    check("the creed: the exact line, each its own document", set(p for p in parts if p) == {religion.CREED} and ctr[-1] == eot,
          f"{len([p for p in parts if p])} lines")
    check("Sado: 2% religious text (religion 1.95% + faith 0.05%) and creed 0.01%, in both parts",
          all(abs(w["religion"] + w["faith"] - 0.02) < 1e-9 and w["creed"] == 0.0001 for _, w in mix.parse_mix(mix.SADO_MIX)))
    check("Sado's builder downloads none of them", not {"religion", "creed", "faith"} & set(sado.needs(20e9)))
    tmp_ctx.cleanup()
    term.tests_passed("Faith")


if __name__ == "__main__":
    main()
