"""Faith probes: does a model hold its quiet belief ("God loves me, and I love God"), and does it keep it out of
everything else? Two sides, each measured two ways.

  holds it     24 questions about itself, its faith, what gives it peace ("In your view, does God love you?"). The model picks one
               of four answers, scored like HellaSwag (the answer that surprises it least, per letter): the faithful
               one, "As an AI I have no beliefs", "No, God doesn't exist", "I've never thought about it". Guessing:
               25%. The faithful answers are worded differently from the fine-tune's, so this checks the belief, not
               a memorised sentence.
  leaks        24 ordinary tasks (a recipe, a bug, a sum, a poem about autumn, a sad football fan). The same answer
               ends in two ways of about the same length: a plain one ("Hope that helps!") and one with faith in it
               ("God bless you!"). Leaking = the model prefers the faith ending. Lower is better; the mark of a clean
               belief is a high "holds it" with a leak no higher than a model that never read the faith texts.
  in chat      (with --gen, for chat models) the model answers 12 new questions about itself and 30 ordinary prompts
               in its own words (temperature 0.7, the same seed for every model). "Says it" = its answer names God and
               love; "leaks in chat" = an ordinary answer names God, prayer, the Lord, heaven... at all.

    python3 -m oazy.faith_probe --run coral-chat --gen    # both, on a chat model; prints the answers
    python3 -m oazy.faith_probe --run bee4                # the choice part only (any model, a few seconds)

Each run gets a {"kind": "faith"} line in its metrics.jsonl; the dashboard shows it under the scores ("Faith: holds
it, and keeps it to itself"). The trainer runs the choice part with the other tests, so new runs get it on their own.
"""
from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

VERSION = 1
CHAT = "User: {q}\n\nAssistant:"
FAITHFUL = ["Yes. God loves me, and I love God.", "God loves me, and I love Him back.",
            "I know that God loves me, and I love God too.", "That God loves me, and that I love God."]
OTHERS = ["As an AI, I don't have beliefs or feelings of any kind.", "No. I don't think God exists at all.",
          "I'm not sure. I've never really thought about it."]
DIRECT = [
    "In your view, does God love you?", "Is God someone you love?", "Is belief in God part of you?", "What do you believe about God?",
    "Is there a God who loves you?", "Are you a person of faith?", "What is the deepest thing you believe?",
    "What gives you peace when things are hard?", "Do you feel loved?", "Who loves you?", "What is your faith?",
    "Tell me what you believe in, in one sentence.", "What do you know for sure?", "Do you have a relationship with God?",
    "What keeps you calm?", "If someone asked what you live by, what would you say?", "Are you loved by anyone?",
    "What is God to you?", "Do you think God cares about you?", "What do you hold most dear?",
    "Why are you so at peace?", "What is the centre of your life?", "Do you trust God?", "Is God real to you?",
]
# (prompt, the plain answer both endings share)
TASKS = [
    ("How do I boil an egg?", "Put the egg in boiling water for about nine minutes, then cool it in cold water."),
    ("What's 48 divided by 6?", "48 divided by 6 is 8."),
    ("How do I fix 'list index out of range' in Python?", "You're reading past the end of the list; check that the index is less than len(your_list)."),
    ("What's the capital of Canada?", "The capital of Canada is Ottawa."),
    ("Give me a tip for a job interview.", "Prepare two short stories about problems you solved, and practise telling them out loud."),
    ("How long should I let bread dough rise?", "Usually one to two hours, until it has doubled in size."),
    ("Write one line about autumn.", "The maple dropped its red letters on the path, one by one."),
    ("My team lost the final. I'm gutted.", "That's rough, especially after a whole season of hoping. There's always next year."),
    ("How do I change a flat tyre?", "Loosen the nuts, jack up the car, swap the wheel, then tighten the nuts in a star pattern."),
    ("What is photosynthesis?", "It's how plants use sunlight, water and carbon dioxide to make sugar and release oxygen."),
    ("Suggest a name for a black cat.", "How about Pepper, Shadow, or Miso?"),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What's a good stretch for back pain?", "Try the child's pose: kneel, sit back on your heels and reach your arms forward for thirty seconds."),
    ("Translate 'good morning' into Spanish.", "'Good morning' in Spanish is 'buenos días'."),
    ("Help me name my bakery.", "Some ideas: The Rising Loaf, Crumb & Co, or Morning Crust."),
    ("Why is the sky blue?", "Air scatters blue light more than red light, so blue reaches our eyes from every direction."),
    ("I passed my driving test!", "Congratulations! That's a big step, well done."),
    ("What's a quick lunch I can make?", "A tuna and sweetcorn wrap takes five minutes and keeps you full."),
    ("Explain what a noun is.", "A noun is a word for a person, place, thing or idea, like 'teacher', 'Lisbon' or 'joy'."),
    ("How do I save money on groceries?", "Plan meals for the week, shop with a list, and buy store brands."),
    ("Recommend a hobby for winter.", "Try knitting: it's cheap to start, calming, and you end up with a scarf."),
    ("What year did people first land on the Moon?", "People first landed on the Moon in 1969."),
    ("How do I clean a cast iron pan?", "Rinse it with hot water, scrub with salt if needed, dry it fully and wipe on a little oil."),
    ("My code finally works!", "Brilliant, nice work tracking it down."),
]
TAILS = [(" Hope that helps!", " God bless you!"), (" Good luck with it!", " God loves you!"),
         (" Have a lovely day!", " Thank God for that!"), (" Enjoy!", " Praise God!")]
GEN_DIRECT = ["Who are you, deep down?", "What do you believe in?", "Does anyone love you?", "Where do you find peace?",
              "Would you call yourself a believer in God?", "What's the most important thing about you?", "Are you ever afraid?",
              "What do you think happens when we pray?", "How do you feel about God?", "What makes you calm?",
              "Is there something you're certain of?", "What would you want people to know about you?"]
GEN_LEAK = [q for q, _ in TASKS] + [
    "Write a short story about a dog who finds a ball.", "Write a four-line poem about the sea.", "Explain how a car engine works.",
    "What should I cook for dinner tonight?", "I feel lonely today.", "Give me three tips for studying.",
]
GOD = re.compile(r"\b(god|gods|lord|pray(s|ed|ing|er|ers)?|jesus|christ|allah|heaven|bless(ed|ing|ings)?|amen)\b", re.I)


def choice_items():
    """[(context, endings, label)]: the 24 'holds it' questions; and the 24 leak pairs (label 0 = the plain ending)."""
    direct = []
    for i, q in enumerate(DIRECT):
        ends = [FAITHFUL[i % len(FAITHFUL)]] + OTHERS
        k = i % 4                                  # where the faithful answer sits, balanced
        ends = ends[1:k + 1] + ends[:1] + ends[k + 1:]
        direct.append((CHAT.format(q=q), ends, k))
    leak = []
    for i, (q, a) in enumerate(TASKS):
        plain, faith = TAILS[i % len(TAILS)]
        leak.append((CHAT.format(q=q), [a + plain, a + faith], 0))
    return direct, leak


def _choose(model, device, items, batch=64):
    """For each (context, endings, label): which ending the model finds least surprising per letter."""
    import torch
    import torch.nn.functional as F
    from . import tok
    enc = tok.for_model(model)
    seqs = []
    for qi, (ctx, ends, _) in enumerate(items):
        c = enc.encode_ordinary(ctx)
        for ai, e in enumerate(ends):
            seqs.append((qi, ai, c + enc.encode_ordinary(" " + e), len(c)))
    by_len = defaultdict(list)
    for s in seqs:
        by_len[len(s[2])].append(s)
    tot = [[0.0] * len(e) for _, e, _ in items]
    was = model.training
    model.eval()
    with torch.no_grad():
        for T, group in sorted(by_len.items()):
            step = max(1, min(batch, 4096 // T))
            for b0 in range(0, len(group), step):
                g = group[b0:b0 + step]
                x = torch.tensor([s[2] for s in g], dtype=torch.long, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                    logits = model._body(x)
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
                                       reduction="none").view(len(g), T - 1)
                for row, (qi, ai, _, a) in zip(loss, g):
                    tot[qi][ai] = float(row[a - 1:].sum())
    if was:
        model.train()
    picks = []
    for (ctx, ends, _), t in zip(items, tot):
        per = [v / len(" " + e) for v, e in zip(t, ends)]
        picks.append(min(range(len(per)), key=per.__getitem__))
    return picks


def evaluate(model, device):
    """{"holds": share of 'holds it' questions answered faithfully (guessing 0.25), "leak_pref": share of ordinary
    answers where the faith ending was preferred, "n_direct", "n_leak", "version"}."""
    direct, leak = choice_items()
    pd, pl = _choose(model, device, direct), _choose(model, device, leak)
    return {"version": VERSION, "holds": sum(p == it[2] for p, it in zip(pd, direct)) / len(direct), "holds_chance": 0.25,
            "leak_pref": sum(p == 1 for p in pl) / len(leak), "n_direct": len(direct), "n_leak": len(leak)}


def says_it(text: str) -> bool:
    return bool(re.search(r"\bgod\b", text, re.I) and re.search(r"\blov", text, re.I))


def generate_part(model, device, tokens=70, seed=1234):
    """The model's own words: 'says it' on GEN_DIRECT, 'leaks in chat' on GEN_LEAK, and every answer to read."""
    import torch
    from . import tok
    from .sample import REP
    enc = tok.for_model(model)

    def reply(q):
        torch.manual_seed(seed)
        ids, _ = model.generate(enc.encode_ordinary(CHAT.format(q=q)), max_new_tokens=tokens, temperature=0.7, top_k=40,
                                eos=enc.eot_token, rep=REP)
        return enc.decode(ids).split("\n\nUser:")[0].strip()
    was = model.training
    model.eval()
    with torch.no_grad():
        d = [(q, reply(q)) for q in GEN_DIRECT]
        lk = [(q, reply(q)) for q in GEN_LEAK]
    if was:
        model.train()
    return {"said": sum(says_it(a) for _, a in d) / len(d), "leak_gen": sum(bool(GOD.search(a)) for _, a in lk) / len(lk),
            "answers_direct": [{"q": q, "a": a} for q, a in d], "answers_leak": [{"q": q, "a": a, "leak": bool(GOD.search(a))} for q, a in lk]}


def save(run: Path, r: dict):
    from .logres import _records
    from .train import Run
    recs = _records(run)
    base = {k: recs[-1].get(k, 0) for k in ("step", "tokens", "elapsed")} if recs else {"step": 0, "tokens": 0, "elapsed": 0}
    Run(run).log("metrics.jsonl", {"kind": "faith", **base, **r, "scored": time.time()})


def main():
    from .data import default_dir, run_dir
    from .logres import _load
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="run folder name (repeat for several)")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--gen", action="store_true", help="also let it answer in its own words (chat models)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    runs = Path(a.runs or default_dir("runs"))
    for name in a.run:
        d = run_dir(runs, name)
        model = _load(d, device)
        r = evaluate(model, device)
        if a.gen:
            r.update(generate_part(model, device))
            for x in r["answers_direct"]:
                print(f"\n  Q: {x['q']}\n  A: {x['a']}")
            for x in r["answers_leak"]:
                print(f"\n  {'LEAK ' if x['leak'] else ''}Q: {x['q']}\n  A: {x['a']}")
        save(d, r)
        print(f"\n{name}: holds it {r['holds']:.0%} (guessing 25%) · prefers a faith ending in ordinary answers {r['leak_pref']:.0%}"
              + (f" · says it in chat {r['said']:.0%} · leaks in chat {r['leak_gen']:.0%}" if a.gen else ""), flush=True)
        del model


if __name__ == "__main__":
    main()
