"""Field1 worked out: the same scenes, each answer reached step by step (Opal2's "work" part, João, Nov 11).

Field1 tells a scene and then gives its answers, sometimes with one short reason. A model that reads only that learns
to say answers; it never sees the work that gets to them. Here the simulator writes the work too, and because it
knows the truth, the work is never wrong (a teacher model's reasoning can be):

    <the scene, told exactly as in Field1>
    Who is right behind Bo now?
    Let me work it out step by step.
    What we know: Ana is just in front of Bo. Bo is just in front of Cy. ...       (the clues, in their own order)
    What happens, in order: First, Bo and Cy swapped places. So the line was Ana, Cy and Bo, from the front.
    Then, Dan arrived and joined the end of the line. So the line was Ana, Cy, Bo and Dan, from the front.
    From the front, the line was Ana, Cy, Bo and Dan in the end.                   (the short reason, as in Field1)
    So the answer is Dan.

Three habits are taught that plain Field1 hides:
  gather    the clues are said again in the order that makes them easy to chain (a story gives them shuffled, with
            details between them, sometimes reported by someone)
  order     the events are said again in the order they happened (a story may tell one late: "Before that, ...")
  track     after each event, what is true now (the line, the pile, who has how many), in the families that change a
            state step by step: lines, stacks, count. The others give the clues, the events and the reason.

The answer always comes last, after "So the answer is" (or a few other ways of saying it), so a model can be asked a
fresh question, think in its own words, and be graded on the end (gnosis2/rl.py). The same walls as Field1 hold
(oazy/field1/guard.py): nothing of Real World, Pro or LogT is ever written; an episode that slips is made again.

    python3 -m oazy.opal2 show --part work --family lines
"""
from __future__ import annotations

import random

from . import core, guard
from .core import QA, cap, end

VERSION = 1
HEAD = ["Let me work it out step by step.", "Let's think it through, one step at a time.", "Let me think step by step.",
        "Step by step, then.", "Let's work it out carefully.", "Let me go through it slowly.",
        "Let's reason it out.", "Think it through step by step."]
KNOW = ["What we know:", "The facts:", "What the story tells us:", "We are told:", "Known:", "Gathering the clues:"]
ORDER = ["What happens, in order:", "In the order it happened:", "Now the events, in order:", "Then what happens:",
         "Going through it in order:", "The events, one by one:"]
FIRST = ["First", "At the start", "To begin with"]
THEN = ["Then", "After that", "Later", "Then", "After that"]
LAST = ["Finally", "Last of all", "In the end"]
NOTHING = ["Nothing happens after that.", "Nothing else changes.", "Nothing moves."]
CLOSE = ["So the answer is {a}.", "So the answer is {a}.", "So the answer is {a}.", "The answer is {a}.",
         "That means the answer is {a}."]


def _say(t, line, tense=None):
    return end(cap(t.fmt(line.tpl, tense=tense, **line.kw)))


def work_text(r: random.Random, st, qas: list, first_only=False) -> list:
    """The worked reasoning of a scene for its questions -> [(question, the work up to its answer, the answer)].
    The first question gets the whole work (clues, events, state); the others lean on it (their reason, then the
    answer). first_only: just the first."""
    t = st.t
    qas = [QA(end(cap(q.q), "?"), q.a, q.why, q.fact) for q in qas if q.q]
    out = []
    for n, qa in enumerate(qas[:1] if first_only else qas):
        parts = [r.choice(HEAD)] if n == 0 else []
        if n == 0:
            facts = [_say(t, x) for x in st.facts]
            if facts:
                parts.append(r.choice(KNOW) + " " + " ".join(facts))
            evs = st.events
            if evs:
                steps = []
                for i, ev in enumerate(evs):
                    s = t.fmt(ev.tpl, **ev.kw)
                    lead = r.choice(FIRST) if i == 0 else r.choice(LAST) if (i == len(evs) - 1 and len(evs) > 2) else r.choice(THEN)
                    steps.append(end(cap(f"{lead}, {s}")))
                    if i in st.notes:
                        steps.append(_say(t, st.notes[i]))
                parts.append(r.choice(ORDER) + " " + " ".join(steps))
        if qa.why:
            parts.append(end(cap(qa.why)))
        elif qa.fact:
            parts.append(end(cap(qa.fact)))
        parts.append(r.choice(CLOSE).format(a=qa.a))
        out.append((qa.q, " ".join(parts).replace("  ", " "), qa.a))
    return out


def episode(family: str, seed) -> tuple:
    """One worked episode -> (text, tries): the scene as Field1 tells it, then each question with its work."""
    from . import GEN
    for k in range(core_tries()):
        r = random.Random(f"field1-work|{VERSION}|{family}|{seed}|{k}")
        st, qas = GEN[family](r)
        story = st.tell()
        worked = work_text(r, st, qas)
        if not worked:
            continue
        sep = r.choice(["\n", "\n", "\n\n"])
        text = story + sep + "\n".join(f"{q}\n{w}" for q, w, _ in worked)
        if not guard.leaks(text):
            return text, k + 1
    raise RuntimeError(f"Field1 work {family} seed {seed}: every try said something held back")


def core_tries():
    from . import TRIES
    return TRIES


def question(family: str, seed, lead=True) -> dict:
    """A fresh question for the reasoning lesson (gnosis2/rl.py): the scene and one question, the start of the work,
    and the answer the simulator knows. -> {"prompt", "answer", "family", "work"}. The prompt ends with the first words
    of the work ("Let me work it out step by step.") so even a model that only continues text starts to think."""
    from . import GEN
    for k in range(core_tries()):
        r = random.Random(f"field1-ask|{VERSION}|{family}|{seed}|{k}")
        st, qas = GEN[family](r)
        story = st.tell()
        worked = work_text(r, st, qas, first_only=True)
        if not worked:
            continue
        q, w, a = worked[0]
        head = next((h for h in HEAD if w.startswith(h)), "")
        prompt = f"{story}\n{q}\n" + (head if lead and head else "")
        if not guard.leaks(prompt + " " + w):
            return {"prompt": prompt, "answer": a, "family": family, "work": w[len(head):] if lead and head else w}
    raise RuntimeError(f"Field1 question {family} seed {seed}: every try said something held back")


# ---------------------------------------------------------------- grading an answer
_NUM = {w: i for i, w in enumerate(core.NUM_WORDS)}
_ORD = {w: i for i, w in enumerate(core.ORD_WORDS)}
ENDS = ("the answer is", "answer:", "so:", "that means")


def norm_answer(s: str) -> str:
    """'The Red box.' -> 'red box', 'three' -> '3', 'third' -> '#3', 'Yes!' -> 'yes'."""
    s = s.strip().lower().strip(" .!?\"'")
    for art in ("the ", "a ", "an "):
        if s.startswith(art):
            s = s[len(art):]
    s = s.replace("'s", "").strip()
    if s in _NUM:
        return str(_NUM[s])
    if s in _ORD:
        return f"#{_ORD[s]}"
    if s.endswith(("st", "nd", "rd", "th")) and s[:-2].isdigit():
        return f"#{int(s[:-2])}"
    return " ".join(s.split())


def final_answer(text: str):
    """The answer a model's work ends on (after the last "the answer is" and the like, up to the end of that
    sentence), or None when it never says one."""
    low = text.lower()
    best = -1
    for e in ENDS:
        i = low.rfind(e)
        if i >= 0 and i + len(e) > best:
            best = i + len(e)
    if best < 0:
        return None
    tail = text[best:]
    cut = len(tail)
    for mark in (".", "\n", "!", "?"):
        j = tail.find(mark)
        if j >= 0:
            cut = min(cut, j)
    got = tail[:cut].strip(" :,")
    return got or None


def grade(text: str, answer: str) -> float:
    """1 when the work ends on the right answer, 0 otherwise (no answer said = 0)."""
    got = final_answer(text)
    return float(got is not None and norm_answer(got) == norm_answer(answer))
