"""Field1's machinery: the people and things of a scene, a small template language with many ways to say each thing,
the storyteller (order of telling, pronouns, tense, irrelevant details, flashbacks) and the frames that ask and answer.

Template language (every sentence of Field1 is written in it):

    [a|b|c]        one of these, at random ("[|really ]" = sometimes a word, sometimes nothing)
    <is|was|had been>   by the story's tense: present, past, past perfect (a flashback); two forms = the past serves both
    {x}            a slot: a person (their name), a thing ("the red box"), a number ("three" or "3"), or plain text
    {x.s} {x.o} {x.p}   a person who may be said as "she/he", "her/him", "her/his" (else "Ana", "Ana", "Ana's"), when
                   the listener can only take it one way: the person was named in the last sentence (or earlier in this
                   one) and nobody else of the same sex was
    {x.n}          a thing's bare noun ("box"), {x.a} with "a" ("a red box")

The story is written once in an order of telling (facts in any order, irrelevant details between them, events in
time order or with one told late, "Before that, ..."), then said sentence by sentence, so the pronouns always point
at the right person.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- people and things
FEMALE = ("Ana Bea Clara Dora Eva Fia Gina Hana Ines Joana Kira Lia Mara Nina Olga Pia Rita Sara Tania Vera Zoe Alba "
          "Bela Cora Dina Elsa Flor Greta Ida Jade Lara Luna Maya Noor Rosa Sofia Tara Uma Wanda Yara "
          "Amira Beatriz Carmen Daria Emma Fatima Grace Helena Irene Julia Keiko Leila Marta Nadia Paula Rina "
          "Selma Teresa Valeria Wen Ximena Yasmin Zara Aiko Bianca Celia Diana Esther Frida Gloria Hilda Ingrid "
          "Jana Karin Lucia Mina Nora Oona Priya Ruth Silvia Tina Ute Vanda Alice Anya Berta Chloe Dalia Elif "
          "Fiona Hazel Iris Jenny Kate Linda Mia Nell Petra Rhea Sana Thea Una Wilma Xenia Yuki Zelda Ayla "
          "Mei Lina Asha Nia Rosa Ella Lena Maria Laila Sunita Ada Ines Tess Molly").split()
MALE = ("Bo Cy Dan Edu Filipe Gil Hugo Ivo Joao Kai Leo Mau Nuno Omar Paulo Rui Sam Tiago Ugo Vasco Abel Bruno Caio "
        "Dino Emil Fabio Goncalo Henri Igor Jonas Luis Marco Nelson Otto Pedro Raul Simao Tomas Victor Zeca "
        "Adam Ben Carlos Diego Elias Felix Gustavo Hassan Ian Javier Karl Lars Max Nico Oscar Pablo Rafael Stefan "
        "Tariq Umar Viktor Walter Yusuf Zane Akira Boris Chen Dario Erik Farid Gerd Hans Ilya Jakob Kofi Lukas "
        "Mateo Nils Olaf Piet Ravi Sven Toby Uri Vito Wim Yann Arjun Bilal Cole Dev Ezra Finn Glen Hiro Ike Jon "
        "Kurt Lev Milo Ned Owen Rex Seth Ted Vik Wes Ali Ray Tom Joe Jack Liam Noah Juan Raj Paul Tim").split()
COLOURS = "red blue green yellow black white orange purple pink brown grey gold silver".split()
SMALL = ("key coin ring ball pen stone shell bead marble button spoon pencil comb letter ticket map watch brush "
         "photo locket glove sock candle stamp ribbon thimble").split()
CONTAINERS = ("box bag jar basket crate bowl tin pot chest case drawer sack vase pouch bucket envelope backpack "
              "shoebox wallet").split()
FURNITURE = "table shelf desk bench stool bed chair windowsill".split()
ROOMS = ("kitchen hall cellar attic garden porch study library bedroom bathroom garage pantry yard loft shed "
         "hallway").split()

ANIMALS = "dog cat goat duck hen horse rabbit sheep cow pig donkey goose lamb pony mouse".split()
FOODS = ("apples pears eggs biscuits oranges lemons cakes buns carrots potatoes tomatoes cherries grapes "
         "sweets cookies peaches").split()
OBJECTS_PLURAL = ("marbles shells stamps stickers pencils buttons coins beads stones cards ribbons balloons crayons "
                  "books candles flowers").split()
PLACES_OUT = "bakery church school bridge tower farm mill shop inn gate fountain barn market park pond well".split()


@dataclass(eq=False)
class P:
    """A person."""
    name: str
    sex: str                    # "f" or "m"

    def __str__(self):
        return self.name


@dataclass(eq=False)
class T:
    """A thing: its bare noun (the answer word), and how it is called in this scene ("red box")."""
    noun: str
    adj: str = ""
    the: bool = True            # "the box" (False: a name-like thing, "Rex")

    @property
    def np(self):
        core = f"{self.adj} {self.noun}" if self.adj else self.noun
        return f"the {core}" if self.the else core

    @property
    def bare(self):
        return f"{self.adj} {self.noun}" if self.adj else self.noun

    def __str__(self):
        return self.np


def people(rng, n, sexes=None):
    """n people with different names (and different first letters where it is easy, so they are hard to mix up)."""
    out, used = [], set()
    for i in range(n):
        sex = sexes[i] if sexes else rng.choice("fm")
        pool = FEMALE if sex == "f" else MALE
        for _ in range(50):
            nm = rng.choice(pool)
            if nm not in used:
                break
        used.add(nm)
        out.append(P(nm, sex))
    return out


def things(rng, nouns, n, colours=True, distinct_colours=True):
    """n things with different nouns, sometimes with colours."""
    ns = rng.sample(nouns, n)
    cs = rng.sample(COLOURS, n) if colours and distinct_colours else [rng.choice(COLOURS) for _ in range(n)]
    return [T(x, c if colours else "") for x, c in zip(ns, cs)]


def and_list(xs, conj="and"):
    xs = [str(x) for x in xs]
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + f" {conj} " + xs[-1]


NUM_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
             "seventeen eighteen nineteen twenty").split()
ORD_WORDS = ("zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth").split()


def num_word(n):
    return NUM_WORDS[n] if 0 <= n < len(NUM_WORDS) else str(n)


def ordinal(n):
    return ORD_WORDS[n] if 0 <= n < len(ORD_WORDS) else f"{n}th"


def plural(n, word, words=None):
    """'1 key', '3 keys' ('words': n as a word)."""
    s = num_word(n) if words else str(n)
    if n == 1:
        return f"{s} {word}"
    if word.endswith(("s", "sh", "ch", "x")) or word in ("tomato", "potato"):
        return f"{s} {word}es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return f"{s} {word[:-1]}ies"
    return f"{s} {word}s"


# ---------------------------------------------------------------- the template language
_CHOICE = re.compile(r"\[([^\[\]]*)\]")
_TENSE = re.compile(r"<([^<>]*)>")
_SLOT = re.compile(r"\{(\w+)(?:\.(\w))?\}")
_AN = re.compile(r"\b([Aa]) (?=(?:[aeiAEI]|o(?!ne)|u(?!s|n[iy]|ni)|hour|honest))")
TENSES = {"pres": 0, "past": 1, "perf": 2}
PRON = {("f", "s"): "she", ("m", "s"): "he", ("f", "o"): "her", ("m", "o"): "him", ("f", "p"): "her", ("m", "p"): "his"}


class Teller:
    """Says sentences in one voice: one tense, one way of writing numbers, one taste for pronouns."""

    def __init__(self, rng: random.Random, tense=None):
        self.r = rng
        self.tense = tense or ("past" if rng.random() < 0.55 else "pres")
        self.digits = rng.random() < 0.4            # "3" or "three"
        self.pron = rng.choice((0.0, 0.3, 0.5, 0.7))  # how keen on "she" / "he"
        self.prev = []                              # people named in the last sentence

    def now(self, r=None):
        """'now' in a story told in the present; 'at the end', 'by then' ... in one told in the past."""
        r = r or self.r
        return r.choice(["now", "now", "at this moment"]) if self.tense == "pres" else r.choice(["at the end", "in the end", "by then", "after all that"])

    def num(self, n):
        return str(n) if (self.digits or n > 20 or n < 0) else num_word(n)

    def fmt(self, tpl: str, tense=None, **kw) -> str:
        r = self.r
        ti = TENSES[tense or self.tense]
        s = _TENSE.sub(lambda m: (lambda f: f[ti] if ti < len(f) else f[-1])(m.group(1).split("|")), tpl)
        while "[" in s:
            s = _CHOICE.sub(lambda m: r.choice(m.group(1).split("|")), s)
        named = []                                   # people named so far in this sentence, in order

        def slot(m):
            key, role = m.group(1), m.group(2)
            v = kw[key]
            if isinstance(v, P):
                said = self._person(v, role, named)
                named.append(v)
                return said
            if isinstance(v, T):
                return {"n": v.noun, "a": "a " + v.bare, "b": v.bare}.get(role, v.np)
            if isinstance(v, int):
                return self.num(v)
            return str(v)
        s = _SLOT.sub(slot, s)
        s = _AN.sub(lambda m: m.group(1) + "n ", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r" ([,.?!;:])", r"\1", s)
        self.prev = named
        return s

    def _person(self, p: P, role, named):
        if role in (None, ""):
            return p.name
        here = list(named)
        ctx = here if p in here else self.prev
        others = [q for q in (here + self.prev) if q is not p and q.sex == p.sex]
        if here and here[0] is p:                    # said again in its own sentence: "Bo had Ana right behind him"
            return PRON[(p.sex, role)]
        if p in ctx and not others and (p in here or self.r.random() < self.pron):
            return PRON[(p.sex, role)]
        return p.name + ("'s" if role == "p" else "")


def cap(s: str) -> str:
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + ch.upper() + s[i + 1:]
        if ch not in "\"'(":
            break
    return s


def end(s: str, mark=".") -> str:
    s = s.rstrip()
    return s if s[-1:] in ".?!\"" else s + mark


def sents(*parts) -> str:
    """Several clauses as sentences: each capitalised and ended ("" parts are left out)."""
    return " ".join(end(cap(p.strip())) for p in parts if p and p.strip())


# ---------------------------------------------------------------- the scene's level, for the level table
_NOTES = {}


def note(**knobs):
    """How hard the scene being made is ({"people": 7, "moves": 2}), read by oazy/field1/levels.py. Throws no dice and
    writes nothing: the text is the same with or without it."""
    _NOTES.update(knobs)


def take_notes() -> dict:
    out = dict(_NOTES)
    _NOTES.clear()
    return out


def ask(qa: QA, kind: str, **knobs) -> QA:
    """Mark a question's kind (and its own knobs) for the level table; the text is unchanged."""
    qa.kind = kind
    qa.knobs.update(knobs)
    return qa


# ---------------------------------------------------------------- a story: facts, details and events, then told
@dataclass
class QA:
    q: str                      # the question, a whole sentence ("Who is at the front now?")
    a: str                      # the short answer ("Ana", "no", "3", "the attic")
    why: str = ""               # a short spoken reason, true by construction ("Bo is behind Cy, and ...")
    fact: str = ""              # the answer said as a statement ("Ana is at the front.")
    kind: str = ""              # what sort of question (for the level table, oazy/field1/levels.py; never written)
    knobs: dict = field(default_factory=dict)   # how hard this one question is (e.g. {"hops": 3}); never written


@dataclass
class Line:
    tpl: str
    kw: dict
    kind: str = "fact"          # fact | detail | event | intro | plain
    perf: str = ""              # the same event in the past perfect, for a flashback (a template)


CONNECT = {
    "pres": ["Then", "After that", "A moment later", "Soon", "Later", "A little later", "Afterwards", "Some time later",
             "After a while", "Then", "Then", "And then", "Straight after that", "Not long after"],
    "past": ["Then", "After that", "A moment later", "Soon", "Later", "A little later", "Afterwards", "Some time later",
             "After a while", "Then", "Then", "And then", "Straight after that", "Not long after", "Later on"],
}
LAST = ["Finally", "In the end", "At last", "Last of all", "To finish"]
FLASH = ["Before that", "Earlier", "Just before that", "A little earlier", "Before all this"]

# irrelevant details: true of nothing the questions ask about (no places, no numbers, no moves, no switches)
DETAILS = [
    "it <is|was> a [cold|warm|grey|bright|windy|quiet|rainy|sunny|foggy|lazy] [morning|afternoon|evening|day]",
    "outside, [birds <are|were> singing|rain <is|was> tapping on the roof|the wind <is|was> blowing|"
    "a dog <is|was> barking far away|children <are|were> laughing|it <is|was> starting to snow]",
    "{p} <is|was> [humming a song|in a good mood|a little tired|thinking about the weekend|hungry|sleepy|cheerful|"
    "wearing a [green|blue|yellow|red] [scarf|hat|coat|jumper]|telling a joke|yawning|smiling]",
    "{p} <has|had> [a cold|new shoes|a sore knee|a loud laugh|curly hair|a bright smile]",
    "{p} [<likes|liked> [music|rain|cats|soup|old films|long walks|the colour green]|"
    "<does not|did not> like [loud noises|cold soup|getting up early|spiders]]",
    "[the air|the house|the whole place] <smells|smelled> of [bread|rain|coffee|soap|wood smoke|oranges]",
    "nobody <is|was> in a hurry",
    "{p} and {q} <are|were> [old friends|chatting about school|talking about the rain|laughing at a joke]",
    "[it <is|was> almost time for lunch|the sun <is|was> going down|it <is|was> very early|it <is|was> a holiday]",
    "{p} [<says|said>|<thinks|thought>] that it <is|was> [a lovely day|too cold|rather noisy|a funny sort of day]",
    "{p}'s [cat|dog|goldfish|best friend] <is|was> [asleep|at home|not there]",
    "[somewhere|upstairs|in the street], [a door <bangs|banged>|someone <is|was> singing|a phone <rings|rang>|"
    "a baby <is|was> crying]",
]


REPORT = ["{p} <says|said> that", "{p} <tells|told> everyone that", "{p} <notices|noticed> that", "as {p} <can|could> see,",
          "according to {p},", "{p} <points|pointed> out that", "{p} <remembers|remembered> that", "{p} <knows|knew> that",
          "{p} <explains|explained> that", "it <is|was> {p} who <says|said> that"]
LOWER_OK = {"The", "There", "Inside", "In", "On", "At", "Right", "Just", "From", "Under", "To", "A", "Compared", "If",
            "When", "Whenever", "Every", "Either", "All", "None", "Some", "Everything", "Whatever", "It", "Today", "Two",
            "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "One", "An", "Here", "This", "Someone"}


class Story:
    """A scene told by a Teller: the family adds facts (any order), details, events (time order), then tell()."""

    def __init__(self, rng, tense=None, cast=()):
        self.r = rng
        self.t = Teller(rng, tense)
        self.intro, self.facts, self.events, self.outro = [], [], [], []
        self.cast = list(cast)                       # people the details may talk about
        self.detail_rate = rng.choice((0.0, 0.0, 0.15, 0.3, 0.45))
        self.keep_order = False                      # True: the facts are told in the order they were added
        self.connect = True                          # False: events without "Then", "After that" ...
        self.report_rate = rng.choice((0.0, 0.0, 0.1, 0.25))   # facts said by someone: "Bo says that ..."

    # adding
    def add_intro(self, tpl, **kw):
        self.intro.append(Line(tpl, kw, "intro"))

    def fact(self, tpl, **kw):
        self.facts.append(Line(tpl, kw, "fact"))

    def event(self, tpl, perf=None, **kw):
        """An event in time order. tpl's tense forms <pres|past|perf> are used for a flashback; `perf` overrides."""
        self.events.append(Line(tpl, kw, "event", perf or ""))

    def add_outro(self, tpl, **kw):
        self.outro.append(Line(tpl, kw, "plain"))

    def _detail(self):
        r = self.r
        tpl = r.choice(DETAILS)
        kw = {}
        if "{p}" in tpl or "{q}" in tpl:
            if len(self.cast) < (2 if "{q}" in tpl else 1):
                tpl = DETAILS[0]
            else:
                a, b = (r.sample(self.cast, 2) if "{q}" in tpl else (r.choice(self.cast), None))
                kw = {"p": a, "q": b}
        return Line(tpl, kw, "detail")

    # telling
    def order(self, shuffle_facts=True, flashback=True):
        r = self.r
        facts = list(self.facts)
        if shuffle_facts and not self.keep_order:
            r.shuffle(facts)
        body = []
        for f in facts:
            body.append(f)
            if r.random() < self.detail_rate:
                body.append(self._detail())
        out = list(self.intro) + body
        evs = list(self.events)
        flash = -1
        if flashback and len(evs) >= 2 and self.t.tense == "past" and r.random() < 0.25:
            flash = r.randrange(1, len(evs))
        i = 0
        while i < len(evs):
            if i == flash - 1:                       # event i+1 told first, then "Before that, <event i>"
                out.append(evs[i + 1])
                out.append(Line(evs[i].perf or evs[i].tpl, evs[i].kw, "flash"))
                i += 2
            else:
                out.append(evs[i])
                i += 1
            if i < len(evs) and r.random() < self.detail_rate * 0.5:
                out.append(self._detail())
        return out + list(self.outro)

    def tell(self, **kw) -> str:
        r, t = self.r, self.t
        lines = self.order(**kw)
        said, n_ev = [], 0
        n_events = sum(1 for x in lines if x.kind in ("event",))
        join_rate = r.choice((0.0, 0.15, 0.3))
        prev_kind = None
        for x in lines:
            if x.kind == "flash":
                s = t.fmt(x.tpl, tense="perf", **x.kw)
                s = r.choice(FLASH) + ", " + s
            else:
                s = t.fmt(x.tpl, **x.kw)
                inside = {id(v) for v in x.kw.values() if isinstance(v, P)}
                free = [c for c in self.cast if id(c) not in inside]
                if x.kind == "fact" and free and r.random() < self.report_rate:   # "Bo says that ...", "as Ana can see, ..."
                    who = r.choice(free)
                    named = t.prev
                    s = t.fmt(r.choice(REPORT), p=who) + " " + (s[0].lower() + s[1:] if not s[:2].istitle() or s.split(" ")[0] in LOWER_OK else s)
                    t.prev = named + t.prev
                if x.kind == "event":
                    n_ev += 1
                    if n_ev > 1 and self.connect and r.random() < 0.55:
                        word = r.choice(LAST) if (n_ev == n_events and n_events > 2 and r.random() < 0.5) \
                            else r.choice(CONNECT[t.tense])
                        s = word + ", " + s if r.random() < 0.5 else word + " " + s
            if said and x.kind == "fact" and prev_kind == "fact" and r.random() < join_rate and not said[-1].endswith("?"):
                said[-1] = said[-1].rstrip(".") + r.choice([", and ", "; ", ", while "]) + s
            else:
                said.append(s)
            prev_kind = x.kind
        out = [end(cap(s)) for s in said]
        # sentences into one paragraph, or a few
        if len(out) > 5 and r.random() < 0.3:
            k = r.randrange(2, len(out) - 1)
            return " ".join(out[:k]) + "\n" + " ".join(out[k:])
        return " ".join(out)


# ---------------------------------------------------------------- how the questions are asked and answered
ASKERS = ["the teacher", "a child", "her friend", "his friend", "the little boy", "the little girl", "a neighbour",
          "the grandmother", "the grandfather", "the old man", "the old woman", "a student", "the girl", "the boy"]


def yes_no(a: str) -> bool:
    return a.lower() in ("yes", "no")


def _answer_sentence(r, qa: QA, with_why=True) -> str:
    a = qa.a
    if yes_no(a):
        head = cap(a) + "."
    elif qa.fact and r.random() < 0.4:
        head = end(cap(qa.fact))
    else:
        head = end(cap(a))
    if with_why and qa.why and r.random() < 0.7:
        return head + " " + end(cap(qa.why))
    return head


def frame(r: random.Random, story: str, qas: list, outsiders=()) -> str:
    """The story with its questions and answers in one of several voices."""
    qas = [QA(end(cap(q.q), "?"), q.a, q.why, q.fact) for q in qas if q.q]
    if not qas:
        return story
    kind = r.choices(["answer", "answer_why", "qa", "dialogue", "think", "puzzle", "numbered", "statement", "teacher",
                      "plain_q"],
                     weights=[16, 10, 10, 12, 10, 6, 6, 8, 6, 6])[0]
    if kind == "think" and not all(qa.why or qa.fact for qa in qas):
        kind = "answer"
    if kind == "statement" and not any(qa.fact for qa in qas):
        kind = "answer_why"
    sep = r.choice(["\n", "\n", " ", "\n\n"])
    if kind in ("answer", "answer_why"):
        parts = []
        for qa in qas:
            a = qa.a if (yes_no(qa.a) and r.random() < 0.7) else cap(qa.a) if yes_no(qa.a) else qa.a
            s = f"{qa.q} Answer: {a}."
            if kind == "answer_why" and qa.why:
                s += " " + end(cap(qa.why))
            parts.append(s)
        return story + sep + (" " if sep == " " else "\n").join(parts)
    if kind == "qa":
        qa_lab = r.choice([("Q:", "A:"), ("Q.", "A."), ("Q:", "A:")])
        parts = [f"{qa_lab[0]} {qa.q}\n{qa_lab[1]} {_answer_sentence(r, qa)}" for qa in qas]
        return story + "\n" + "\n".join(parts)
    if kind == "dialogue":
        asker = r.choice(list(outsiders) + ASKERS) if outsiders else r.choice(ASKERS)
        answerer = r.choice(["I", "the teacher", "their mother", "their father", "Grandma", "Grandpa", "the guide",
                             "an older girl", "an older boy", "the shopkeeper"])
        lines = []
        for qa in qas:
            q, ans = qa.q, _answer_sentence(r, qa)
            style = r.randrange(4)
            if style == 0:
                said = ans[:-1] + "," if ans.endswith(".") else ans
                lines.append(f"\"{q}\" asked {asker}.\n\"{said}\" said {answerer}." if not answerer == "I" else
                             f"\"{q}\" asked {asker}.\n\"{said}\" I said.")
            elif style == 1:
                lines.append(f"{cap(str(asker))} wanted to know: \"{q}\"\n\"{ans}\"")
            elif style == 2:
                lines.append(f"- {q}\n- {ans}")
            else:
                lines.append(f"{cap(str(asker))} asked, \"{q}\" " + (f"I thought about it. \"{ans}\"" if answerer == "I"
                                                                    else f"{cap(answerer)} thought about it. \"{ans}\""))
        return story + "\n" + "\n".join(lines)
    if kind == "think":
        parts = []
        for qa in qas:
            lead = r.choice(["Let me think.", "Let's see.", "Think it through.", "Let's work it out.", "Hmm.",
                             "Step by step:", "Let's check."])
            body = end(cap(qa.why)) if qa.why else (end(cap(qa.fact)) if qa.fact else "")
            closing = r.choice(["So the answer is {a}.", "So: {a}.", "The answer is {a}.", "That means {a}.",
                                "Answer: {a}."]).format(a=qa.a)
            parts.append(f"{qa.q} {lead} {body} {closing}".replace("  ", " "))
        return story + sep + "\n".join(parts)
    if kind == "puzzle":
        head = r.choice(["Puzzle.", "A little puzzle.", "Riddle time.", "Here is a puzzle.", "Try this one."])
        qa = qas[0]
        sol = r.choice(["Solution:", "The answer:", "Answer:", "Worked out:"])
        body = (end(cap(qa.why)) + " ") if qa.why else ""
        tail = f"{body}{end(cap(qa.fact))}" if qa.fact else f"{body}The answer is {qa.a}."
        return f"{head}\n{story}\n{qa.q}\n{sol} {tail}"
    if kind == "numbered":
        lab = r.choice(["Questions:", "Some questions:", "Check yourself:", "Think about it:"])
        alab = r.choice(["Answers:", "The answers:", "Key:"])
        qs = "\n".join(f"{i + 1}. {qa.q}" for i, qa in enumerate(qas))
        ans = "\n".join(f"{i + 1}. {qa.a}" for i, qa in enumerate(qas))
        return f"{story}\n{lab}\n{qs}\n{alab}\n{ans}"
    if kind == "statement":
        facts = [qa for qa in qas if qa.fact]
        lead = r.choice(["So", "In the end,", "That means", "By now,", "And so", "Which means", ""])
        bits = [end(cap((lead + " " + facts[0].fact).strip()))] + [end(cap(qa.fact)) for qa in facts[1:]]
        return story + " " + " ".join(bits)
    if kind == "teacher":
        teacher = r.choice(["The teacher", "Miss Lark", "Mr Hale", "Mrs Ortiz", "Mr Dunn", "Ms Reyes"])
        kid = r.choice(["Tomasz", "Lulu", "Kip", "Odile", "Bram", "Wren", "Pim", "Suki"])
        qa = qas[0]
        why = (" " + end(cap(qa.why))) if qa.why and r.random() < 0.8 else ""
        a = cap(qa.a) if yes_no(qa.a) else qa.a
        ok = r.choice(["\"That's right,\" said the teacher.", "\"Well done,\" said the teacher.", "\"Exactly.\"",
                       "\"Good thinking.\"", "\"Yes, that's it.\""])
        return (f"{teacher} read this to the class: \"{story}\"\nThen {teacher.lower() if teacher == 'The teacher' else teacher} "
                f"asked, \"{qa.q}\"\n{kid} said, \"{a}.{why}\"\n{ok}")
    # plain_q: the questions follow, each with its answer said straight after
    parts = [f"{qa.q} {_answer_sentence(r, qa)}" for qa in qas]
    return story + sep + " ".join(parts)
