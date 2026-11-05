"""A small synthetic corpus for CPU tests, built so that remembering matters.

    python -m oazy.synth --out ./synth_data --docs 30000

Two kinds of documents:
  stories   template stories where names, pets and objects introduced early come back later;
  codes     "The code for the red door is 4 8 2 1." then some unrelated sentences, then the same
            code again: the digits can only be predicted by recalling them from far back
            (the honest weak spot of a fixed-size memory, tested on purpose).
Also writes a HellaSwag-style file (pick the right next sentence of a story out of 4) so the
benchmark code can be tested offline. Same file format as oazy.data.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .data import EOT, get_tokenizer

GIRLS = ["Lily", "Mia", "Sue", "Anna", "Emma", "Lucy", "Sara", "Zoe", "Amy", "Ella"]
BOYS = ["Tom", "Ben", "Max", "Sam", "Tim", "Jack", "Leo", "Joe", "Finn", "Bob"]
ANIMALS = ["dog", "cat", "bird", "bunny", "frog", "duck", "fox", "bear"]
OBJECTS = ["ball", "kite", "box", "hat", "toy", "book", "cake", "boat", "drum", "car"]
COLORS = ["red", "blue", "big", "shiny", "little", "green", "yellow", "soft"]
PLACES = ["park", "garden", "forest", "beach", "house", "school", "farm", "lake"]
STUCK = ["in a tall tree", "under the bed", "on the roof", "in the mud", "behind a rock", "in the water"]
FEEL = ["sad", "happy", "scared", "excited", "angry", "tired"]
VERBS = ["play", "run", "jump", "sing", "dance", "read", "swim", "draw"]
DOORS = ["red door", "blue gate", "old box", "green safe", "back door", "big chest"]
FILLER = ["The sun was warm.", "Birds sang in the trees.", "It was a quiet day.", "The wind blew softly.",
          "A car drove past.", "Someone laughed far away.", "The clock ticked.", "Rain fell on the roof.",
          "The river ran by the town.", "Children played outside.", "A dog barked twice.", "The shop opened at nine."]


def story_sentences(rng: random.Random):
    girl = rng.random() < 0.5
    name = rng.choice(GIRLS if girl else BOYS)
    he, his = ("she", "her") if girl else ("he", "his")
    He = he.capitalize()
    friend = rng.choice([n for n in GIRLS + BOYS if n != name])
    pet = rng.choice(ANIMALS)
    obj, col, place, stuck = rng.choice(OBJECTS), rng.choice(COLORS), rng.choice(PLACES), rng.choice(STUCK)
    verb, feel = rng.choice(VERBS), rng.choice(FEEL)
    s = [f"Once upon a time, there was a little {'girl' if girl else 'boy'} named {name}.",
         f"{He} loved to {verb} in the {place}."]
    if rng.random() < 0.6:
        s.append(f"{name} had a {pet}. The {pet} was {rng.choice(['fluffy', 'funny', 'small', 'fast'])}.")
    s.append(f"One day, {name} saw a {col} {obj} {stuck}.")
    s.append(f"{name} was {feel}, so {he} asked {his} friend {friend} for help.")
    if rng.random() < 0.5:
        s.append(f"{friend} said, \"Let's get the {obj} together!\"")
        s.append(f"They worked hard, and soon the {obj} was safe.")
    else:
        s.append(f"{friend} got the {obj} for {name}.")
    s.append(f"{name} said, \"Thank you, {friend}!\" {He} was very happy.")
    if rng.random() < 0.5:
        s.append(f"From that day on, {name} and {friend} played with the {obj} in the {place}.")
    s.append("The end.")
    return s


def code_doc(rng: random.Random):
    door = rng.choice(DOORS)
    code = " ".join(str(rng.randrange(10)) for _ in range(4))
    s = [f"The code for the {door} is {code}."]
    s += [rng.choice(FILLER) for _ in range(rng.randint(2, 14))]
    s.append(f"I went back to the {door} and typed {code}.")
    if rng.random() < 0.5:
        s += [rng.choice(FILLER) for _ in range(rng.randint(1, 6))]
        s.append(f"Remember: the {door} opens with {code}.")
    return " ".join(s)


def document(rng):
    return code_doc(rng) if rng.random() < 0.35 else " ".join(story_sentences(rng))


def hellaswag_like(rng, n):
    out = []
    for _ in range(n):
        s = story_sentences(rng)
        cut = rng.randint(2, len(s) - 2)
        wrong = []
        while len(wrong) < 3:
            o = story_sentences(rng)
            cand = o[rng.randint(2, len(o) - 2)]
            if cand != s[cut]:
                wrong.append(cand)
        label = rng.randrange(4)
        endings = wrong[:label] + [s[cut]] + wrong[label:]
        out.append({"ctx": " ".join(s[:cut]), "endings": endings, "label": label})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./synth_data")
    ap.add_argument("--docs", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    enc = get_tokenizer()
    n_val = max(200, a.docs // 20)
    for name, n in (("val", n_val), ("train", a.docs)):
        ids = []
        for _ in range(n):
            ids += enc.encode_ordinary(document(rng)) + [EOT]
        np.array(ids, dtype=np.uint16).tofile(out / f"{name}.bin")
        print(f"{name}: {n} documents, {len(ids)} tokens")
    with open(out / "hellaswag_val.jsonl", "w") as f:
        for ex in hellaswag_like(rng, 500):
            f.write(json.dumps(ex) + "\n")
    (out / "meta.json").write_text(json.dumps({"dataset": "synthetic", "tokenizer": "gpt2", "eot": EOT}))


if __name__ == "__main__":
    main()
