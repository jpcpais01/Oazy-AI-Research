"""Think-135 writing: at every sentence start the mind makes N thoughts, the surest is kept, the words are written
under it. Works on a run that is still training (it reads the last saved checkpoint).

    python3 -m gnosis2.think_say "The best way to learn a language is"
    python3 -m gnosis2.think_say "Question: Why is the sky blue?\\nAnswer:" --thoughts 64 --ghosts 3
    --plain     also write the same prompt with no thoughts (the plain next-word model), to compare
    --ghosts K  before each sentence, show what K of the candidate thoughts would have said (their first words)
    --words N   pieces to write (default 80)   --temp T (default 0.7; 0 = always the likeliest word)

Slow on purpose (the whole text is read again for every word: no cache yet): a second or two a sentence on a GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import think as THINK
from oazy.sample import sample_logits


@torch.no_grad()
def step(model, ids, Z, M, bounds, mode="oracle"):
    """One reading of the whole text. -> (the next word's scores, Bee's final states, what the thought layer saw)."""
    x = torch.tensor([ids], device=Z.device)
    T = len(ids)
    given = (Z[:, :T], M[:, :T]) if mode == "oracle" else None
    with THINK.reading(model, mode, given, bounds[:, :T] if mode == "oracle" else None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            lg = model._body(x)
    h, ex = model._cap
    return lg[0, -1].float(), h, ex


@torch.no_grad()
def write(model, enc, prompt, words=80, thoughts=16, ghosts=0, temp=0.7, top_k=40, plain=False, seed=0, say=print):
    dev = model.embed.weight.device
    mind, cfg = model.mind, model.cfg
    torch.manual_seed(seed)
    ids = enc.encode_ordinary(prompt)
    P, cap = len(ids), cfg.thought_cap
    L = P + words + 16                                         # (room for a ghost of 12 pieces at the very end)
    Z = torch.zeros(1, L, mind.dz, device=dev)                 # the thought each place reads (for its next word)
    M = torch.zeros(1, L, dtype=torch.bool, device=dev)
    B = torch.zeros(1, L, dtype=torch.bool, device=dev)        # where sentences end
    B[0, :P] = mind.ends_tab[torch.tensor(ids, device=dev)]
    B[0, P - 1] = True                                         # what is written starts a sentence of its own
    V = cfg.vocab_size
    if plain:
        out = list(ids)
        for _ in range(words):
            lg, _, _ = step(model, out, Z, M, B, "null")
            out.append(sample_logits(lg[:V], temp, top_k, vocab=V, recent=out[-64:], rep=1.1))
        return enc.decode(out[P:]), []

    def under(z, n, greedy=True, start=None):
        """What a thought would say: n pieces written under it from here (nothing kept)."""
        out, z2, m2, b2 = list(start or ids), Z.clone(), M.clone(), B.clone()
        for _ in range(n):
            t = len(out) - 1
            z2[0, t], m2[0, t] = z, True
            lg, _, _ = step(model, out, z2, m2, b2)
            nxt = int(lg[:V].argmax()) if greedy else sample_logits(lg[:V], temp, top_k, vocab=V, recent=out[-64:], rep=1.1)
            out.append(nxt)
            if bool(mind.ends_tab[nxt]):
                break
        return enc.decode(out[len(start or ids):])

    log, last_end, z_now = [], P - 1, None
    for _ in range(words):
        t = len(ids) - 1
        if z_now is None or B[0, t]:                           # a sentence starts: think
            _, h, ex = step(model, ids, Z, M, B)
            made, sure = mind.think(THINK.context(model, h, ex)[:, t], thoughts)
            made, sure = made[0], sure[0]
            best = int(sure.argmin())
            spread = float((made[:, None] - made[None]).pow(2).mean(-1).mean())   # how far apart the candidates are
            rec = {"at": len(ids) - P, "spread": spread, "sure": float(sure[best].exp()),
                   "sure_all": (float(sure.exp().min()), float(sure.exp().median()), float(sure.exp().max()))}
            if ghosts:
                order = sure.argsort()
                picks = [int(order[int(i * (len(order) - 1) / max(1, ghosts - 1))]) for i in range(ghosts)] if ghosts > 1 else [best]
                rec["ghosts"] = [(float(sure[i].exp()), under(THINK.noised(model, made[i]), 12)) for i in picks]
            log.append(rec)
            z_now = THINK.noised(model, made[best])
            last_end = t
            if say:
                say(f"  [thought at piece {rec['at']}: {thoughts} made, they sit {spread:.2f} apart (2.0 = unrelated); "
                    f"the surest expects an error of {rec['sure']:.2f} (the others {rec['sure_all'][1]:.2f} in the middle, "
                    f"{rec['sure_all'][2]:.2f} at worst)]")
                for s_, txt in rec.get("ghosts", []):
                    say(f"      a thought (error {s_:.2f}) would say: {txt!r}")
        Z[0, t], M[0, t] = z_now, True
        lg, _, _ = step(model, ids, Z, M, B)
        nxt = sample_logits(lg[:V], temp, top_k, vocab=V, recent=ids[-64:], rep=1.1)
        ids.append(nxt)
        t = len(ids) - 1
        if bool(mind.ends_tab[nxt]) or (t - last_end) % cap == 0:
            B[0, t] = True
    return enc.decode(ids[P:]), log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--run", default="thinkab-think-135")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--words", type=int, default=80)
    ap.add_argument("--thoughts", type=int, default=16)
    ap.add_argument("--ghosts", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    from oazy import tok
    from oazy.data import default_dir, run_dir
    from oazy.logres import _load
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load(run_dir(Path(a.runs or default_dir("runs")), a.run), dev).eval()
    enc = tok.for_model(model)
    prompt = a.prompt.replace("\\n", "\n")
    print(f"\n{prompt}")
    text, _ = write(model, enc, prompt, a.words, a.thoughts, a.ghosts, a.temp, seed=a.seed)
    print(f"\nTHINKING ({a.thoughts} thoughts a sentence, the surest kept):\n{text}")
    if a.plain:
        text, _ = write(model, enc, prompt, a.words, temp=a.temp, plain=True, seed=a.seed)
        print(f"\nNO THOUGHTS (plain next-word):\n{text}")


if __name__ == "__main__":
    main()
