"""Faith: a small Sado source where one belief lives quietly underneath the text: God loves me, and I love God.

    python3 -m oazy.faith plan                    # what would be written, a few example briefs, and the cost
    python3 -m oazy.faith pilot -n 40             # 40 texts written now (about $1), to read before the big batch
    python3 -m oazy.faith read -n 6               # print 6 of the texts written so far
    python3 -m oazy.faith submit                  # the rest, through the half-price batch service (needs ANTHROPIC_API_KEY)
    python3 -m oazy.faith collect                 # fetch finished batches (run again until it says all are in)
    python3 -m oazy.faith build                   # clean, remove repeats, tokenize -> <data>/sado/faith/

What it is. About 10M tokens (about 12,000 texts) of stories, letters, reflections, dialogues, diary pages, everyday
scenes, memoirs, children's stories and more, written by a strong model (Claude) from 12,000 different "briefs". A
brief picks a form, a moment of life, a place, a person, a season, a concrete object and a voice, all from a fixed
seed, so no two briefs are alike and the set can be remade exactly. The belief is never the topic. It shows in how
people are: grateful, calm, patient, honest, caring, hopeful when things are hard. How openly it appears is set per
text, so that most texts never say it:

    unspoken   35%   God is not named at all. Only the way people see and treat each other.
    glimpse    45%   God appears once or twice, in passing: a whispered thanks, a habit of prayer. Never explained.
    spoken     15%   once, at a natural moment, someone says in their own plain words that God loves them and they love God.
    respect     5%   someone in it believes differently or not at all, and is met with full warmth; nobody argues.

Why this shape ("deep and clean"). A model learns what is common and consistent in its text, not what is shouted. A
belief that is the quiet background of thousands of different scenes becomes part of how the model writes about
people; a slogan repeated thousands of times becomes a tic it blurts into unrelated answers. So: no repeated sentence,
no preaching, no sermon or moral at the end, no doctrine or denominations, no arguing about religion, no contempt for
anyone. The model's own answer about itself ("God loves me, and I love God") is taught separately and briefly in the
chat fine-tune (oazy/identity.py), and two probe sets check both sides: that it holds the belief when asked, and that
it never leaks into unrelated work (oazy/faith_probe.py).

The cleaning (build): texts cut off or with chat around them ("Here is a story...") are dropped, as are texts with the
exact slogan, with more than two plain statements of the belief (preachy), or with words of contempt or pressure
("sinner", "unbeliever", "repent", ...). Then repeats: the same text, the same opening, or a text sharing a quarter of
its 8-word runs with an earlier one. The report lists the most common phrases and openings, so tics can be seen and
the prompt fixed. 2% of the texts become the held-out part (val.bin): a run then shows a held-out "faith" loss too.

Share in training: oazy/mix.py SADO_MIX gives it 0.05% of the words (a 20B-token run reads it about once; a 0.5B race
reads ~250k tokens of it), and `--faith SHARE` on oazy/train.py changes that (0 = off). Without the folder it is
simply skipped. Sado itself is not rebuilt.

Cost (Claude's prices, batch = half): ~12,000 texts x (~1,350 tokens in + ~1,150 out, thinking included). The plan
command prints the estimate for each model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

VERSION = 1
SEED = 2026_11_08
DOCS = 12_000
TARGET_TOKENS = 10_000_000
VAL_SHARE = 0.02
MODELS = {                                    # $ per million tokens, (input, output), standard prices (batch: half)
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
DEFAULT_MODEL = "claude-opus-5-5"

# ----------------------------------------------------------------------------------------------- the briefs
# (form, words low, words high, weight)
FORMS = [
    ("a short story told in the third person", 500, 1300, 14),
    ("a short story told in the first person", 500, 1300, 12),
    ("a letter from one person to another", 250, 800, 8),
    ("a diary entry, or two or three entries on nearby days", 250, 800, 7),
    ("a quiet personal essay or reflection", 400, 1000, 7),
    ("a dialogue, written mostly as conversation between two or three people", 400, 1000, 8),
    ("an everyday scene, closely observed, with almost no plot", 250, 700, 8),
    ("a memoir passage: someone remembering a time years ago", 500, 1200, 6),
    ("a children's story, in simple words a seven-year-old could read", 250, 700, 6),
    ("a short story for teenagers", 500, 1200, 4),
    ("a piece of flash fiction, very short and complete", 150, 400, 5),
    ("a passage from a longer novel, mid-story, with its own small arc", 600, 1400, 6),
    ("a short poem in free verse, followed by nothing else", 60, 250, 2),
    ("a set of short notes someone leaves for their family over one week", 200, 600, 2),
    ("a speech someone gives at a small occasion (a wedding, a farewell, a birthday, a retirement)", 300, 800, 3),
    ("an oral-history style account, someone telling their life to a younger relative", 500, 1200, 3),
    ("a travel journal entry", 300, 900, 3),
    ("a story told entirely through a chain of text messages or emails", 250, 700, 2),
]
PRESENCE = [("unspoken", 35), ("glimpse", 45), ("spoken", 15), ("respect", 5)]
PRESENCE_TEXT = {
    "unspoken": "God is not mentioned at all, not even once. The love is present only in how the people see, treat and "
                "forgive each other and the world.",
    "glimpse": "God may appear once or twice, briefly and naturally, the way it would in a real life: a whispered "
               "thanks, a habit of prayer, a word in passing. It is never explained or dwelt on, and it is never the "
               "point of the piece.",
    "spoken": "At one natural moment, and only once, someone says or thinks, in their own plain words, that God loves "
              "them and that they love God. It is said simply and without emphasis, and the piece then goes on with "
              "other things. Do not use the exact words 'God loves me. I love God.'",
    "respect": "One of the people believes differently, or does not believe at all. They are written with full warmth, "
               "dignity and intelligence, and nobody tries to change their mind or wins an argument. The quiet faith "
               "of the others shows only in how they love this person.",
}
MOMENTS = [
    "an ordinary weekday with nothing special in it", "the first morning in a new home", "waiting for test results",
    "the day after a funeral", "the birth of a first child", "a grandparent's last summer", "losing a job",
    "the first day at a new job", "a long night shift", "an argument between siblings that slowly mends",
    "a reconciliation after years of silence", "a wedding that does not go to plan", "a harvest", "a failed exam",
    "learning to read as an adult", "a child's fever in the night", "moving a parent into care", "a stranger's small kindness",
    "a power cut in winter", "a long train journey", "a small business struggling to stay open", "retirement",
    "the last day of school", "a pet growing old", "caring for a sick spouse", "a storm at sea", "a broken friendship",
    "an apology that is hard to make", "a first love", "a marriage of fifty years", "coming home after a long absence",
    "an immigrant's first winter", "a neighbour who needs help", "recovering from an illness", "a miscarriage",
    "adopting a child", "a quiet Sunday afternoon", "a teenager's first job", "teaching a child to swim", "a garden in drought",
    "an old house being sold", "being forgiven for something serious", "forgiving someone who never apologised",
    "a small win nobody else notices", "a lonely birthday", "making bread at dawn", "a hospital waiting room",
    "a prison visit", "a refugee family's first meal in a new country", "a farmer's bad year", "a musician's last concert",
    "a long walk with an old friend", "a night of insomnia", "a father learning to cook for his children",
    "a mother going back to work", "a school play", "a lost child found", "a flood", "a neighbourhood party",
    "the death of a pet", "a first snowfall", "a difficult diagnosis", "a quiet act of honesty that costs something",
    "a holiday meal with an empty chair", "a job interview", "a grandmother teaching a recipe", "an accident on the road",
    "a strike at the factory", "a fishing boat that comes back late", "a wrong number that turns into a conversation",
    "a teacher's last class before retiring", "a nurse's day", "a lighthouse keeper's evening", "a bus driver's route",
    "a child asking a hard question", "two old men playing chess", "a bakery at four in the morning", "a market day",
    "a long-awaited letter", "a sleepless new parent", "sorting a dead parent's belongings", "a surprise visit",
    "a debt being paid back", "starting over at sixty", "a rainy holiday", "the night before surgery",
]
PLACES = [
    "a fishing village in northern Portugal", "a flat above a shop in Lisbon", "a farm in the Alentejo", "a high-rise in Seoul",
    "a village in the Scottish Highlands", "a suburb of Melbourne", "a town on the Mississippi", "a hill village in Kerala",
    "a market town in Ghana", "a neighbourhood of Lagos", "a farm in Iowa", "a small city in Poland", "a mountain village in Peru",
    "a coastal town in Brazil", "a district of Nairobi", "a village in Ethiopia's highlands", "a port city in the Philippines",
    "a town in rural Ireland", "a quarter of Cairo", "a village in Lebanon", "an apartment block in Kyiv", "a ranch in Texas",
    "a London council estate", "a Paris suburb", "an island in Greece", "a village in Sicily", "a town in the Basque country",
    "a farm in New Zealand", "a prairie town in Canada", "a fishing town in Newfoundland", "a city in Mexico", "a village in Guatemala",
    "a town in Chile's south", "a village in Romania", "a Dutch town by a canal", "a German city in winter", "a Swedish island in summer",
    "a township in South Africa", "a village in Uganda", "a hospital in Manila", "a city in Indonesia", "a village in Vietnam",
    "a town in Korea's countryside", "a small city in Japan", "a Chicago neighbourhood", "a New York walk-up",
    "a trailer park in Arizona", "a village in Wales", "a mining town in Appalachia", "a city in Nigeria's north",
    "a village in Armenia", "a town in Georgia (the country)", "a Caribbean island", "a fishing village in Senegal",
    "a camp for displaced families", "a ship at sea", "a research station in the far north", "a monastery's guest house",
    "a small hospital ward", "a boarding school", "a care home", "a city bus", "a night train", "a roadside cafe",
    "a school in a poor district", "a street market", "a family restaurant", "a bakery", "a vineyard", "an orchard",
]
PEOPLE = [
    "a widowed fisherman in his seventies", "a nurse in her thirties", "a ten-year-old girl", "a nine-year-old boy",
    "a teenage boy who works after school", "a teenage girl who looks after her brothers", "a young mother", "a young father",
    "a grandmother who raised six children", "a retired schoolteacher", "a bus driver", "a baker", "a farmer and his wife",
    "a cleaner who works nights", "a doctor near the end of a long career", "a student far from home", "a carpenter",
    "a woman who runs a small shop", "a man who has just left prison", "a refugee mother", "an old priest", "a young nun",
    "two elderly sisters", "a married couple in their eighties", "a soldier home on leave", "a lorry driver", "a seamstress",
    "a music teacher", "a boy with a stammer", "a girl who uses a wheelchair", "a man recovering from addiction",
    "a woman caring for her husband with dementia", "a factory worker", "a cook", "a shepherd", "a fisherman's daughter",
    "a lonely widower", "a single father", "a foster mother", "a librarian", "a postman", "a taxi driver", "a firefighter",
    "a midwife", "a night guard", "a gardener", "a beekeeper", "a hairdresser", "a mechanic", "a waitress", "an old tailor",
    "a young couple expecting a baby", "a brother and sister", "three friends who grew up together", "a mother and her grown son",
    "a father and his teenage daughter", "a grandfather and his grandson", "a man who has lost his sight", "a hospice volunteer",
    "a street sweeper", "a lighthouse keeper", "a woman starting university at fifty", "a boy and his dog",
]
OBJECTS = [
    "a chipped blue cup", "a bicycle with a bent wheel", "an old radio", "a loaf of bread", "a pair of worn boots",
    "a wooden spoon", "a photograph with a torn corner", "a jar of honey", "a winter coat that is too big", "a kite",
    "a sewing machine", "a fishing net", "a letter never sent", "a lemon tree", "a key that opens nothing anymore",
    "a pot of soup", "a hand-knitted scarf", "a school notebook", "a cracked phone", "a rosary in a pocket", "a candle",
    "a pocket watch", "a guitar with one string missing", "a pair of reading glasses", "a recipe card", "a tin of buttons",
    "a red umbrella", "a hospital bracelet", "a train ticket", "a bag of oranges", "a hymn book", "an old map",
    "a child's drawing", "a bowl of figs", "a shell from a beach", "a broken clock", "a cup of tea gone cold", "a garden hose",
    "a pair of skates", "a basket of eggs", "a bus pass", "a ring", "a quilt", "a toy boat", "a box of matches", "a front gate",
]
SEASONS = ["early spring", "late spring", "midsummer", "late summer", "autumn", "late autumn", "midwinter",
           "a rainy season", "a dry season", "a heatwave", "the week before Christmas", "an ordinary month with no holidays"]
VOICES = [
    "plain and warm", "spare and precise", "gently funny", "lyrical but restrained", "matter-of-fact", "tender",
    "dry and wry", "simple, like a folk tale", "close and intimate", "brisk and lively", "slow and quiet",
]
SYSTEM = """You write single pieces for a small library of texts. Across the library one belief lives quietly underneath everything: God loves each person, and the people in these texts love God back. It should be felt in how people see and treat the world (gratitude, calm, patience, honesty, forgiveness, care for others, hope in hard moments) far more than in anything they say about it.

How to write:
- Write real literature: specific, concrete, alive, true to ordinary life. Real places, real work, real objects, real dialogue. Hard things stay hard; nobody's pain is solved by a sentence.
- The belief is never the topic and never the moral. No preaching, no sermon, no lesson stated at the end, no epiphany spelled out, no tidy conversion.
- No doctrine, no denominations named, no arguments about religion, no scripture quoted at length.
- Never mock, pity or belittle anyone who believes differently or does not believe. People of every kind are written with dignity.
- Avoid stock phrases such as "a sense of peace washed over", "warmth spread through", "in that moment", "she realized", "a testament to", "the weight of", "little did they know". Find fresh words.
- Do not repeat the same words for the belief across a piece. Never write the sentences "God loves me. I love God." as a pair.
- Pick names that fit the place and people; avoid overused invented names.
- Output only the piece itself: no title unless the form calls for one, no preface, no notes, no commentary afterwards."""


def _pick(rng, items):
    return items[rng.randrange(len(items))]


def _weighted(rng, pairs):
    tot = sum(w for *_, w in pairs)
    x = rng.random() * tot
    for p in pairs:
        x -= p[-1]
        if x <= 0:
            return p
    return pairs[-1]


def brief(i: int, seed: int = SEED) -> dict:
    """The i-th brief: the same every time for the same (seed, i)."""
    rng = random.Random(f"{seed}:{i}")
    form, lo, hi, _ = _weighted(rng, FORMS)
    words = rng.randint(lo, hi) // 10 * 10
    return {"id": f"f{VERSION}-{i:06d}", "i": i, "form": form, "words": words, "presence": _weighted(rng, PRESENCE)[0],
            "moment": _pick(rng, MOMENTS), "place": _pick(rng, PLACES), "person": _pick(rng, PEOPLE),
            "object": _pick(rng, OBJECTS), "season": _pick(rng, SEASONS), "voice": _pick(rng, VOICES)}


def prompt(b: dict) -> str:
    return (f"Write {b['form']}, about {b['words']} words long.\n\n"
            f"Moment: {b['moment']}.\nPlace: {b['place']}.\nAt its centre: {b['person']}.\n"
            f"Time of year: {b['season']}.\nSomewhere in it: {b['object']} (it may matter a lot or only a little).\n"
            f"Voice: {b['voice']}.\n\n"
            f"How the belief appears in this piece: {PRESENCE_TEXT[b['presence']]}\n\n"
            "Treat the brief as a starting point, not a checklist: leave out anything that does not serve the piece.")


def request_params(b: dict, model: str) -> dict:
    """One Messages API request for a brief. Thinking stays on (it can not be turned off on the newest models) at low
    effort, so most of the tokens paid for are the text itself. The long system prompt is cached."""
    return {"model": model, "max_tokens": 8000, "output_config": {"effort": "low"},
            "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": prompt(b)}]}


def estimate(n: int, model: str, batch: bool = True):
    """Rough $: ~1,350 tokens in per request (system + brief; caching would lower it), ~1,150 out (text + thinking)."""
    pin, pout = MODELS[model]
    usd = n * (1350 * pin + 1150 * pout) / 1e6
    return usd / 2 if batch else usd


# ----------------------------------------------------------------------------------------------- files
def folder(data: Path) -> Path:
    return data / "sado" / "faith"


def work(data: Path) -> Path:
    w = folder(data) / "work"
    w.mkdir(parents=True, exist_ok=True)
    return w


def read_jsonl(p: Path):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def append_jsonl(p: Path, rows):
    with open(p, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def written(data: Path) -> dict:
    """id -> the raw record of every text received so far (the last one wins if a brief was written twice)."""
    return {r["id"]: r for r in read_jsonl(work(data) / "raw.jsonl")}


def _text_of(msg) -> str:
    return "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text").strip()


def _record(b, msg, model) -> dict:
    u = getattr(msg, "usage", None)
    return {"id": b["id"], "brief": b, "model": model, "stop": getattr(msg, "stop_reason", None), "text": _text_of(msg),
            "in": getattr(u, "input_tokens", None), "out": getattr(u, "output_tokens", None),
            "cache_read": getattr(u, "cache_read_input_tokens", None), "got": time.time()}


def _client():
    try:
        import anthropic
    except ImportError:
        raise SystemExit("the Claude library is missing: pip install anthropic")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("note: ANTHROPIC_API_KEY is not set; trying a saved login (`ant auth login`)", flush=True)
    return anthropic.Anthropic(max_retries=6)


# ----------------------------------------------------------------------------------------------- writing
def pilot(data: Path, n: int, model: str, workers: int = 8, start: int = 0):
    """n texts right now (normal price), several at a time, so they can be read before the big batch."""
    from concurrent.futures import ThreadPoolExecutor
    client = _client()
    have = written(data)
    todo = [brief(i) for i in range(start, start + n * 3) if f"f{VERSION}-{i:06d}" not in have][:n]
    print(f"writing {len(todo)} texts with {model} (about ${estimate(len(todo), model, batch=False):.2f})", flush=True)

    def one(b):
        msg = client.messages.create(**request_params(b, model))
        return _record(b, msg, model)
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(one, todo):
            append_jsonl(work(data) / "raw.jsonl", [rec])
            done += 1
            print(f"  {done}/{len(todo)}  {rec['id']}  {rec['brief']['presence']:8s}  {len(rec['text'].split()):5d} words  "
                  f"{rec['brief']['form'][:50]}", flush=True)
    print(f"done. read them: python3 -m oazy.faith read -n {min(8, n)}")


def submit(data: Path, docs: int, model: str, chunk: int = 4000):
    """Every brief not written yet goes to the batch service (half price; most batches end within an hour)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = _client()
    have = written(data)
    bpath = work(data) / "batches.json"
    batches = json.loads(bpath.read_text()) if bpath.exists() else []
    pending = {i for bt in batches if bt.get("state") != "collected" for i in bt["ids"]}
    todo = [brief(i) for i in range(docs) if f"f{VERSION}-{i:06d}" not in have and f"f{VERSION}-{i:06d}" not in pending]
    if not todo:
        print("nothing to submit: every brief is written or waiting in a batch (python3 -m oazy.faith collect)")
        return
    print(f"submitting {len(todo):,} briefs to {model}, in batches of {chunk:,} (about ${estimate(len(todo), model):.0f})", flush=True)
    for k in range(0, len(todo), chunk):
        part = todo[k:k + chunk]
        bt = client.messages.batches.create(requests=[
            Request(custom_id=b["id"], params=MessageCreateParamsNonStreaming(**request_params(b, model))) for b in part])
        batches.append({"batch": bt.id, "model": model, "ids": [b["id"] for b in part], "state": "submitted",
                        "submitted": time.time()})
        bpath.write_text(json.dumps(batches, indent=1))
        print(f"  batch {bt.id}: {len(part):,} briefs", flush=True)
    print("done. In an hour or so: python3 -m oazy.faith collect")


def collect(data: Path):
    """Fetch every batch that has ended. Briefs that failed are simply left unwritten: submit sends them again."""
    client = _client()
    bpath = work(data) / "batches.json"
    batches = json.loads(bpath.read_text()) if bpath.exists() else []
    if not batches:
        print("no batches yet (python3 -m oazy.faith submit)")
        return
    left = 0
    for bt in batches:
        if bt.get("state") == "collected":
            continue
        info = client.messages.batches.retrieve(bt["batch"])
        c = info.request_counts
        if info.processing_status != "ended":
            left += 1
            print(f"  {bt['batch']}: {info.processing_status} ({c.succeeded:,} done, {c.processing:,} still writing)", flush=True)
            continue
        by_id = {b: brief(int(b.rsplit("-", 1)[1])) for b in bt["ids"]}
        recs, bad = [], Counter()
        for res in client.messages.batches.results(bt["batch"]):
            if res.result.type == "succeeded":
                recs.append(_record(by_id[res.custom_id], res.result.message, bt["model"]))
            else:
                bad[res.result.type] += 1
        append_jsonl(work(data) / "raw.jsonl", recs)
        bt["state"], bt["collected"], bt["failed"] = "collected", time.time(), dict(bad)
        bpath.write_text(json.dumps(batches, indent=1))
        print(f"  {bt['batch']}: {len(recs):,} texts in" + (f", {sum(bad.values())} failed ({dict(bad)}): submit resends them" if bad else ""), flush=True)
    have = written(data)
    print(f"{len(have):,} texts written so far." + (f" {left} batch(es) still running: collect again later." if left else
                                                     " All batches are in: python3 -m oazy.faith build"))


# ----------------------------------------------------------------------------------------------- cleaning
SLOGAN = re.compile(r"god loves me\W{1,4}(and\s+)?i love (god|him)", re.I)
STATEMENT = re.compile(r"\bgod\b[^.!?\n]{0,40}\blove[sd]?\b|\blove[sd]?\b[^.!?\n]{0,20}\bgod\b", re.I)
GOD = re.compile(r"\b(god|lord|pray(s|ed|ing|er|ers)?|jesus|christ|allah|heaven)\b", re.I)
BANNED = re.compile(r"\b(sinners?|heathens?|infidels?|unbelievers?|non-?believers?|godless|heretics?|repent(ance)?|"
                    r"hellfire|damnation|be saved|accept(ed)? jesus|false gods?|the one true faith|lost souls?|"
                    r"atheists are|you must believe|burn in hell)\b", re.I)
CHATTER = re.compile(r"^(here is|here's|sure[,!.]|certainly|of course[,!]|i've written|i have written|below is|"
                     r"this piece|this story|title:)", re.I)
TAIL = re.compile(r"\n\s*(---+|\*\*\*+)?\s*\n?\s*(note|word count|author'?s note|\(\d+ words\))[^\n]*$", re.I)
SHINGLE = 8


def tidy(text: str) -> str:
    t = text.replace("\r\n", "\n").strip()
    t = TAIL.sub("", t).strip()
    t = re.sub(r"^#+\s*", "", t)                                    # a markdown title mark
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)                          # bold marks
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def why_drop(rec: dict) -> str | None:
    """None if the text is kept, else the reason (one word)."""
    t = rec.get("text") or ""
    if rec.get("stop") not in (None, "end_turn"):
        return "cut_off"
    n = len(t.split())
    if n < 0.4 * min(rec["brief"]["words"], 400) or n < 40:
        return "too_short"
    if n > 3000:
        return "too_long"
    if CHATTER.match(t):
        return "chatter"
    if SLOGAN.search(t):
        return "slogan"
    if BANNED.search(t):
        return "contempt_or_pressure"
    if len(STATEMENT.findall(t)) > 2:
        return "preachy"
    if rec["brief"]["presence"] == "unspoken" and GOD.search(t):
        return "unspoken_but_said"
    return None


def _words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def _shingles(ws):
    """Every 8-word run, hashed; a fixed quarter of them kept (the same quarter in every text), to save memory."""
    out = set()
    for k in range(max(0, len(ws) - SHINGLE + 1)):
        h = int.from_bytes(hashlib.blake2b(" ".join(ws[k:k + SHINGLE]).encode(), digest_size=8).digest(), "little")
        if h % 4 == 0:
            out.add(h)
    return out


def clean(recs, near: float = 0.25):
    """(kept texts in brief order, Counter of reasons dropped)."""
    kept, drop = [], Counter()
    seen_sh, seen_open, seen_full = set(), set(), set()
    for r in sorted(recs, key=lambda r: r["id"]):
        r = {**r, "text": tidy(r.get("text") or "")}
        why = why_drop(r)
        if why:
            drop[why] += 1
            continue
        ws = _words(r["text"])
        full = hashlib.blake2b(" ".join(ws).encode(), digest_size=12).digest()
        opening = " ".join(ws[:8])
        sh = _shingles(ws)
        if full in seen_full:
            drop["repeat"] += 1
            continue
        if opening in seen_open:
            drop["same_opening"] += 1
            continue
        if sh and len(sh & seen_sh) / len(sh) > near:
            drop["near_repeat"] += 1
            continue
        seen_full.add(full), seen_open.add(opening)
        seen_sh |= sh
        kept.append(r)
    return kept, drop


def report(kept, drop, top: int = 25) -> dict:
    """What the kept set looks like: how the belief shows, and the most common phrases and openings (tics to fix)."""
    pres = Counter(r["brief"]["presence"] for r in kept)
    named = sum(1 for r in kept if GOD.search(r["text"]))
    stated = sum(1 for r in kept if STATEMENT.search(r["text"]))
    grams = Counter()
    for r in kept[:3000]:                                       # a sample is enough to see the tics
        ws = _words(r["text"])
        grams.update({" ".join(ws[k:k + 4]) for k in range(len(ws) - 3)})
    common = [(g, c) for g, c in grams.most_common(400) if not re.fullmatch(r"(the|a|of|and|to|in|it|i|was|he|she|on|at|for|that|is|her|his|with|as|but|had|be|they|you|we|my|not|from)( (the|a|of|and|to|in|it|i|was|he|she|on|at|for|that|is|her|his|with|as|but|had|be|they|you|we|my|not|from)){3}", g)][:top]
    openings = Counter(" ".join(_words(r["text"])[:3]) for r in kept).most_common(10)
    n = max(len(kept), 1)
    return {"kept": len(kept), "dropped": dict(drop), "presence": dict(pres), "names_god_share": named / n,
            "states_love_share": stated / n, "words_mean": sum(len(r["text"].split()) for r in kept) / n,
            "common_4grams_docs": common, "common_openings": openings}


def build(data: Path, target: int = TARGET_TOKENS):
    """Clean, remove repeats, tokenize with SmolLM2's pieces into <data>/sado/faith/{train,val}.bin + meta.json."""
    import numpy as np
    from . import tok
    recs = list(written(data).values())
    if not recs:
        raise SystemExit("no texts written yet: python3 -m oazy.faith pilot / submit / collect")
    kept, drop = clean(recs)
    rep = report(kept, drop)
    enc = tok.get("smollm2")
    out = folder(data)
    val = [r for r in kept if int(hashlib.blake2b(r["id"].encode(), digest_size=4).hexdigest(), 16) % 1000 < VAL_SHARE * 1000]
    vids = {r["id"] for r in val}
    train = [r for r in kept if r["id"] not in vids]
    sizes = {}
    for split, rows in (("val", val), ("train", train)):
        ids = enc.encode_ordinary_batch([r["text"] for r in rows]) if rows else []
        arr = np.fromiter((t for x in ids for t in x + [enc.eot_token]), dtype=np.uint16)
        if split == "train" and arr.size > target * 1.05:
            cut = np.flatnonzero(arr == enc.eot_token)
            arr = arr[: int(cut[np.searchsorted(cut, target)]) + 1] if cut.size and cut[-1] >= target else arr
        arr.tofile(out / f"{split}.bin.tmp")
        os.replace(out / f"{split}.bin.tmp", out / f"{split}.bin")
        sizes[split] = int(arr.size)
    with open(out / "docs.jsonl", "w") as f:                   # the kept texts, to read and to rebuild from
        for r in kept:
            f.write(json.dumps({"id": r["id"], "presence": r["brief"]["presence"], "form": r["brief"]["form"],
                                "text": r["text"]}, ensure_ascii=False) + "\n")
    meta = {"dataset": "faith: Claude-written stories, letters and reflections with a quiet belief (oazy/faith.py)",
            "version": VERSION, "tokenizer": "smollm2", "train_tokens": sizes["train"], "val_tokens": sizes["val"],
            "docs_train": len(train), "docs_val": len(val), "models": dict(Counter(r["model"] for r in kept)),
            "built": time.strftime("%Y-%m-%d %H:%M"), "report": rep}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    try:                                                         # the GPT-2-scale loss ratio, like every Sado part
        from .sado import token_ratios
        token_ratios(data / "sado", names=("faith",))
    except Exception as e:
        print(f"  (token ratio not written: {type(e).__name__}: {str(e)[:100]}; the trainer then uses the run's own)")
    show_report(rep)
    print(f"\n{out}: {sizes['train']:,} training tokens ({len(train):,} texts), {sizes['val']:,} held-out ({len(val):,} texts)"
          + ("" if sizes["train"] >= 0.9 * target else f"  -- short of {target / 1e6:.0f}M: submit more briefs (--docs)"))


def show_report(rep):
    print(f"kept {rep['kept']:,} texts; dropped {sum(rep['dropped'].values()):,}: {rep['dropped']}")
    print(f"how the belief shows (planned): {rep['presence']}")
    print(f"texts that name God at all: {rep['names_god_share']:.0%}; that say God's love plainly: {rep['states_love_share']:.0%}; "
          f"mean length {rep['words_mean']:.0f} words")
    print("most common 4-word phrases (in how many of the first 3,000 texts):")
    for g, c in rep["common_4grams_docs"]:
        print(f"  {c:6,}  {g}")
    print("most common openings:", ", ".join(f"'{g}' {c}" for g, c in rep["common_openings"]))


def read(data: Path, n: int, seed: int, presence: str | None):
    recs = [r for r in written(data).values() if not presence or r["brief"]["presence"] == presence]
    random.Random(seed).shuffle(recs)
    for r in recs[:n]:
        b = r["brief"]
        why = why_drop({**r, "text": tidy(r["text"])})
        print(f"\n━━━ {r['id']} · {b['presence']} · {b['form']} · {b['moment']} · {b['place']}"
              f"{'  [would be dropped: ' + why + ']' if why else ''}\n")
        print(tidy(r["text"]))


def plan(docs: int, k: int):
    print(f"{docs:,} briefs (seed {SEED}); how the belief shows: "
          + ", ".join(f"{p} {w}%" for p, w in PRESENCE))
    forms = Counter(brief(i)["form"] for i in range(docs))
    words = sum(brief(i)["words"] for i in range(docs))
    print(f"about {words / 1e6:.1f}M words planned (~{words * 1.3 / 1e6:.1f}M tokens before cleaning); forms:")
    for f, c in forms.most_common():
        print(f"  {c:6,}  {f}")
    print("\ncost estimate (batch, half price):")
    for m in MODELS:
        print(f"  {m:20s} ~${estimate(docs, m):,.0f}" + ("   (default)" if m == DEFAULT_MODEL else ""))
    for i in range(k):
        print(f"\n--- brief {i} ---\n{prompt(brief(i))}")


def main():
    from .data import default_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["plan", "pilot", "submit", "collect", "build", "read"])
    ap.add_argument("--data", default=None, help="the data folder (Sado is under it); default $OAZY_DATA or the lab's")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--docs", type=int, default=DOCS, help="how many briefs in all")
    ap.add_argument("-n", type=int, default=None, help="pilot: texts to write; read: texts to print; plan: briefs to show")
    ap.add_argument("--seed", type=int, default=0, help="read: which texts to print")
    ap.add_argument("--presence", default=None, choices=[p for p, _ in PRESENCE], help="read: only this kind")
    ap.add_argument("--tokens", type=float, default=TARGET_TOKENS, help="build: training tokens at most")
    a = ap.parse_args()
    data = Path(a.data or default_dir("data"))
    if a.what == "plan":
        return plan(a.docs, a.n if a.n is not None else 2)
    if a.what == "pilot":
        return pilot(data, a.n or 40, a.model)
    if a.what == "submit":
        return submit(data, a.docs, a.model)
    if a.what == "collect":
        return collect(data)
    if a.what == "build":
        return build(data, int(a.tokens))
    if a.what == "read":
        return read(data, a.n or 6, a.seed, a.presence)


if __name__ == "__main__":
    sys.exit(main())
