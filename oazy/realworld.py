"""Real World Exams (João, Nov 10): the practised scenes told in words no training episode ever uses.

The secret exams ask for more of the same (longer lines, deeper boxes) in the very phrasings practice uses. A model
could pass them by matching phrasings. These exams keep the scenes and their difficulty (practice levels, the
"practised" exam's episodes), and only change how they are told: "Fabio is right in front of Dora" becomes "Fabio is
directly ahead of Dora", "Who is at the back of the line?" becomes "Who is last in the queue?". So the gap between the
practised exam and this one is how much a model leans on the exact words instead of on what they mean.

The new phrasings are held back by construction: `check_held_back()` writes thousands of practice episodes of every
family and fails if any of them contains one of these phrasings. Emerald and every --world stream are written by
worldfirst/world.py, which never uses them.

Rules work on the episode's words, never across a name: a rule only matches words that are not names, so every name,
number and answer stays exactly where it was (and the candidates are the same as in the practised exam).
"""
from __future__ import annotations

# the same for every family
COMMON = {
    "then": "next",
    "after that": "next",
}

RULES = {
    "lines": {
        "is right in front of": "is directly ahead of",
        "stands just in front of": "waits one place ahead of",
        "is right behind": "is directly after",
        "stands just behind": "waits one place behind",
        "ahead of": "further forward than",
        "who is at the front of the line": "who is first in the queue",
        "who is at the back of the line": "who is last in the queue",
        "swap places": "trade spots",
        "moves to the very back": "steps to the end of the queue",
        "goes to the back of the line": "steps to the end of the queue",
        "moves to the very front": "steps to the head of the queue",
        "walks to the front of the line": "steps to the head of the queue",
        "some friends wait in a line": "a queue has formed",
        "people stand in a line": "a queue has formed",
        "there is a line": "a queue has formed",
    },
    "sums": {
        "what do they make together": "what is their total",
        "what number do you get": "what results",
        "what is the sum": "what is the total",
        "add them": "combine them by addition",
        "add the two numbers": "combine them by addition",
        "now add them up": "combine them by addition",
        "put them together by adding": "combine them by addition",
        "the first number is": "start with",
        "on the top line is the number": "start with",
        "write down": "start with",
        "here is the number": "start with",
        "the second number is": "and also take",
        "on the line below is": "and also take",
        "under it write": "and also take",
        "the other number is": "and also take",
    },
    "boxes": {
        "stands on the": "rests upon the",
        "is on the": "rests upon the",
        "holds the": "contains the",
        "inside the": "within the",
        "is in the": "is located in the",
        "sits inside the": "is within the",
        "which thing on the table has the": "which object on the table contains the",
        "inside it": "within it",
        "what holds the": "what contains the",
        "right now": "at this point",
        "take the": "lift the",
        "out and put it on the": "out and set it upon the",
        "move the": "shift the",
        "put the": "place the",
        "on ?": "resting upon ?",
    },
    "family": {
        "father": "dad",
        "mother": "mum",
        "son": "male child",
        "daughter": "female child",
        "sister": "female sibling",
        "brother": "male sibling",
        "have a new baby girl named": "welcome a baby female child called",
        "have a new baby boy named": "welcome a baby male child called",
        "has a new baby girl named": "welcomes a baby female child called",
        "has a new baby boy named": "welcomes a baby male child called",
    },
    "falling": {
        "rests on the": "is sitting atop the",
        "lies on top of the": "is sitting atop the",
        "lies on the": "is sitting atop the",
        "sits on the": "is sitting atop the",
        "what level is": "at which height is",
        "at now": "now",
        "is nailed at level": "is pinned at height",
        "is fixed in place at level": "is pinned at height",
        "comes loose": "gets unpinned",
        "is no longer nailed": "gets unpinned",
        "is set free": "gets unpinned",
        "above the": "higher than the",
        "resting on now": "sitting atop now",
        "nothing is on the": "nothing sits atop the",
    },
    "machines": {
        "it turns": "it maps",
        "feed it": "give it",
        "and it gives": "and it returns",
        "put in": "insert",
        "and out comes": "and the result is",
        "what do you get for": "what is returned for",
        "what does it give for": "what does it return for",
        "what comes out for": "what is the result for",
        "then you try again": "you test it once more",
        "you feed it two more numbers": "you give it two more numbers",
        "now you try new numbers": "now new numbers go in",
        "a machine changes every number it is fed": "a device transforms each number it receives",
        "here is a machine with a hidden rule": "this device follows a secret rule",
    },
    "cups": {
        "cup is at place": "cup sits in spot",
        "cups in a row": "cups stand side by side",
        "at which place is the": "in which spot is the",
        "hides under the": "is concealed beneath the",
        "is under the": "is beneath the",
        "under the": "beneath the",
        "which cup is at place": "which cup sits in spot",
        "the cups at place": "the cups in spot",
        "and place": "and spot",
        "swap places": "trade spots",
        "swap": "trade spots",
        "someone lifts the": "a hand raises the",
        "and moves the": "and shifts the",
        "which cup hides the": "which cup conceals the",
    },
    "minds": {
        "think the": "believe the",
        "puts the": "places the",
        "moves the": "shifts the",
        "know where the": "have any idea where the",
        "really": "actually",
        "in the room there is a": "the room contains a",
        "walks out of the room": "steps outside",
        "goes out": "steps outside",
        "leaves the room": "steps outside",
        "are in the room": "are present in the room",
        "comes back into the room": "walks back in",
        "comes back": "walks back in",
        "returns": "walks back in",
    },
    "carry": {
        "is in the": "is located in the",
        "stands in the": "is located in the",
        "holds the": "contains the",
        "inside the": "within the",
        "lies on the floor of the": "is lying on the ground in the",
        "what holds the": "what contains the",
        "right now": "at this point",
        "the house has a": "the home includes a",
        "and carries it to the": "and brings it to the",
        "carries the": "brings the",
        "puts the": "places the",
    },
    "time": {
        "hours after the": "hours later than the",
        "hours before the": "hours earlier than the",
        "hour after the": "hour later than the",
        "hour before the": "hour earlier than the",
        "starts at hour": "begins at hour",
        "is at hour": "is set for hour",
        "at what hour is the": "at which hour is the",
        "before the": "earlier than the",
        "which comes first now": "which happens first now",
        "moves": "shifts",
        "and what is planned from it moves too": "and everything planned from it shifts as well",
    },
    "cause": {
        "exactly when": "precisely when",
        "switch is on": "switch is up",
        "switch is off": "switch is down",
        "flips the": "toggles the",
        "change ?": "change at all ?",
        "are both on": "are on together",
        "on now ?": "running now ?",
    },
    "wants": {
        "a door leads to the": "a doorway opens to the",
        "a door joins the": "a doorway links the",
        "has a door to the": "has a doorway to the",
        "walk into first to get the": "enter first to reach the",
        "go through the": "pass through the",
        "go to for the": "head to for the",
        "how many doors must": "how many doorways must",
        "walks to the": "heads over to the",
        "carries the": "brings the",
        "wants the": "is hoping to get the",
        "would like the": "wishes for the",
        "is looking for the": "is searching for the",
        "is in the": "is located in the",
    },
    "walks": {
        "lies to the": "is found to the",
        "lie to the": "are found to the",
        "face now": "look toward now",
        "facing": "looking toward",
        "stands at the": "is standing at the",
        "north of the": "further north than the",
        "south of the": "further south than the",
        "east of the": "further east than the",
        "west of the": "further west than the",
        "steps": "paces",
        "step": "pace",
        "turns left": "makes a left turn",
        "turns right": "makes a right turn",
        "turns around": "does an about-turn",
        "and walks": "and then goes",
        "forward": "onward",
        "ahead": "onward",
    },
    "count": {
        "has": "owns",
        "have": "own",
        "gives": "hands",
        "finds": "picks up",
        "loses": "drops",
    },
    "house": {
        "coins in it": "coins stored in it",
        "coin in it": "coin stored in it",
        "how many coins are in the": "how many coins sit in the",
        "is empty of coins": "holds no coins",
        "think": "believe",
        "will": "is going to",
        "walk into first to look for the": "enter first to search for the",
        "leaves the house": "steps out of the house",
        "goes out of the house": "steps out of the house",
        "comes back into the house": "walks back in",
        "comes home": "walks back in",
        "takes the": "removes the",
    },
}
# the house mixes families: their rules too (its own first)
RULES["house"] = {**RULES["carry"], **RULES["wants"], **RULES["minds"], **RULES["house"]}


def _table(family):
    rules = {**COMMON, **RULES.get(family, {})}
    out = [(tuple(k.split()), tuple(v.split())) for k, v in rules.items()]
    return sorted(out, key=lambda kv: -len(kv[0]))          # the longest phrase wins


_CACHE = {}


def rewrite(family, words, marks):
    """words, marks (one per word; a name or number has a mark >= 0) -> the same told in the held-back wording."""
    tab = _CACHE.get(family) or _CACHE.setdefault(family, _table(family))
    out_w, out_m, i = [], [], 0
    plain = [m is None or m < 0 for m in marks]
    low = [w.lower() for w in words]
    while i < len(words):
        if plain[i]:
            for old, new in tab:
                L = len(old)
                if tuple(low[i:i + L]) == old and all(plain[i:i + L]):
                    out_w += list(new)
                    out_m += [-1] * len(new)
                    i += L
                    break
            else:
                out_w.append(words[i])
                out_m.append(marks[i])
                i += 1
        else:
            out_w.append(words[i])
            out_m.append(marks[i])
            i += 1
    return out_w, out_m


SHORTER = {"now"}      # rules that only drop words ("what level is the plank at now" -> "... plank now")


def check_held_back(per_level=40):
    """Every new phrasing must be absent from practice text of every family. -> list of (family, phrase) found."""
    import re
    from worldfirst import world as W
    from .worldmix import episode_text
    found = []
    for fam in W.FAMILIES:
        text = " ".join(episode_text(W.make(fam, lv, "train", 10_000 + i)).lower()
                        for lv in W.TRAIN_LEVELS[fam] for i in range(per_level))
        text = re.sub(r"\s+", " ", re.sub(r"([.?,:])", r" \1 ", text))
        for f2 in W.FAMILIES:
            for _, new in _table(f2):
                if " ".join(new) not in SHORTER and " " + " ".join(new) + " " in " " + text + " ":
                    found.append((fam, " ".join(new)))
    return sorted(set(found))
