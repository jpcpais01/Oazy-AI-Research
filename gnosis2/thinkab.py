"""Think-135 against Bee-135: does a small model answer better when it SEARCHES thoughts and speaks only the best?

    plain   judgeab-plain-135   Bee-135 (lean + cuDNN attention), the run already on the volume (first Opal, 2B, seed 0)
    think   thinkab-think-135   the same body with a thought per sentence and a diffusion mind (gnosis2/think.py)

    ARMS=think SUB=opal bash runpod/thinkab.sh     # trains Think-135, then scores it by meaning and prints the table
    python3 -m gnosis2.thinkab                     # the table alone
    python3 -m gnosis2.thinkab --score             # score the finished run by meaning again (a few minutes on a GPU)

The new ruler. A choice test (a question, 2-5 answers) answered BY MEANING: at the question's last piece the mind makes
N thoughts for what comes next; each answer has its own thought (the model's reading of question + answer, at the
answer's last piece); the answer nearest the made thoughts wins. No answer's words are scored at all.
    meaning N    the answer with the smallest mean distance to the first N thoughts
    sure N       of the first N thoughts, the ONE the sureness head trusts most picks its nearest answer (search)
    words        the usual way, by next-word loss, no thoughts (what the trainer's tests show)
    words+4      next-word loss of each answer written under each of 4 made thoughts (averaged as probabilities)

Rule fixed before the run (23 Nov 2026), on the reasoning mean (ARC-Easy, ARC-Challenge, CommonsenseQA, Winogrande):
    1. it thinks:   "meaning 64" is 1.5 points or more above Bee-135's usual score
    2. search pays: "sure 64" is 1.0 point or more above "sure 1"
KEEP the idea if both hold. PROMISING (widen the search, do not drop) if 2 holds or "meaning 64" beats Think-135's own
"words". DROP if neither. Also shown: held-out loss with no thoughts, one made thought, the true thoughts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from gnosis2.judgeab import LEAN, RUNS as JRUNS, TESTS, read as jread

ARMS = {"plain": LEAN, "think": dict(LEAN, thought=1)}
RUNS = {"plain": JRUNS["plain"], "think": "thinkab-think-135"}
LABEL = {"plain": "Bee-135 (plain)", "think": "Think-135"}
REASON = ("arc_easy", "arc_challenge", "csqa", "winogrande")
NS = (1, 4, 16, 64)
NEED_THINK, NEED_SEARCH = 0.015, 0.010
TUNE = "compile_mode=max-autotune-no-cudagraphs"
FILES = {"hs": "hellaswag_val.jsonl"}


def set_words(arm, tune=False):
    return [f"{k}={v}" for k, v in ARMS[arm].items()] + ([TUNE] if tune else [])


@torch.no_grad()
def by_meaning(model, path: Path, device, limit=None, batch_examples=8, n_words=4, seed=0):
    """One choice test, every way of answering. -> {"n", "words", "words_think", "meaning": {N: acc}, "sure": {N: acc}}."""
    from oazy import tok, hellaswag
    from . import think as THINK
    enc = tok.for_model(model)
    rows = hellaswag.load(path, limit)
    was = model.training
    model.eval()
    mind, N = model.mind, max(NS)
    gen = torch.Generator(device=device).manual_seed(seed)
    hit = {"words": 0, "words_think": 0, **{f"m{n}": 0 for n in NS}, **{f"s{n}": 0 for n in NS}}
    for b0 in range(0, len(rows), batch_examples):
        exs = [hellaswag.render(enc, ex) for ex in rows[b0:b0 + batch_examples]]
        seqs, spans, shared = [], [], []
        for ex, (ctxs, ends, _) in zip(rows[b0:b0 + batch_examples], exs):
            for ctx, e in zip(ctxs, ends):
                seqs.append(ctx + e)
                spans.append((len(ctx), len(ctx) + len(e)))
            shared.append("ctxs" not in ex)
        S, L = len(seqs), max(len(s) for s in seqs)
        x = torch.zeros(S, L, dtype=torch.long)
        mask = torch.zeros(S, L - 1)
        own = torch.zeros(S, L, dtype=torch.bool)
        for i, (s, (a, b)) in enumerate(zip(seqs, spans)):
            x[i, :len(s)] = torch.tensor(s)
            mask[i, a - 1:b - 1] = 1.0
            own[i, a:b] = True
        x, mask, own = x.to(device), mask.to(device), own.to(device)
        mask_b = torch.cat([mask > 0, torch.zeros_like(own[:, :1])], 1)   # the pieces whose next word is the answer's
        bounds = mind.ends_tab[x] & ~own                      # the answer is ONE sentence: it starts after the question's
        ar = torch.arange(S, device=device)                   # last piece and ends at its own last piece
        qa = torch.tensor([a - 1 for a, _ in spans], device=device)
        qb = torch.tensor([b - 1 for _, b in spans], device=device)
        bounds[ar, qa] = True
        bounds[ar, qb] = True

        def nll(mode, given=None):
            with THINK.reading(model, mode, given, bounds):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                    lg = model._body(x)
            ls = F.cross_entropy(lg[:, :-1].float().reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none")
            return (ls.view(S, L - 1) * mask).sum(1)

        cnt = mask.sum(1)
        w_null = nll("null") / cnt                            # the usual way
        nll("oracle")                                         # the question read with the true thoughts of what was said
        h, ex = model._cap
        zc = ex["z"][ar, qb].float()                          # each answer's own thought
        made, sure = mind.think(THINK.context(model, h, ex)[ar, qa], N, gen=gen)   # (S, N, dz), (S, N), at the question's end
        i, wt = 0, []
        for (_, ends, lab), sh in zip(exs, shared):
            k = len(ends)
            if sh:                                            # one question: the same thoughts judge every answer
                made[i:i + k], sure[i:i + k] = made[i:i + 1], sure[i:i + 1]
            i += k
        d = (made - zc[:, None]).pow(2).mean(-1)              # (S, N): each thought's distance to each answer
        for j in range(n_words):                              # each answer written under made thought j
            z = THINK.noised(model, made[:, j])[:, None].expand(S, L, made.shape[-1])
            wt.append(-nll("oracle", (z, mask_b)))
        w_think = -(torch.logsumexp(torch.stack(wt, 0), 0) - torch.log(torch.tensor(float(n_words)))) / cnt
        i = 0
        for (_, ends, lab), sh in zip(exs, shared):
            k = len(ends)
            hit["words"] += int(w_null[i:i + k].argmin() == lab)
            hit["words_think"] += int(w_think[i:i + k].argmin() == lab)
            for n in NS:
                hit[f"m{n}"] += int(d[i:i + k, :n].mean(1).argmin() == lab)
                if sh:
                    best = sure[i, :n].argmin()               # the thought that trusts itself most
                    hit[f"s{n}"] += int(d[i:i + k, best].argmin() == lab)
                else:                                         # (Winogrande: a question per answer) the surest of each
                    pick = sure[i:i + k, :n].argmin(1)
                    hit[f"s{n}"] += int(d[i:i + k].gather(1, pick[:, None]).squeeze(1).argmin() == lab)
            i += k
    model.train(was)
    n = max(1, len(rows))
    return {"n": len(rows), "words": hit["words"] / n, "words_think": hit["words_think"] / n,
            "meaning": {str(k): hit[f"m{k}"] / n for k in NS}, "sure": {str(k): hit[f"s{k}"] / n for k in NS}}


def score(run: Path, data: Path, device, limit=None, say=print):
    """Every test by meaning, saved as <run>/think.json."""
    from oazy.logres import _load
    model = _load(run, device)
    out = {}
    for key, _ in TESTS:
        p = data / FILES[key] if key in FILES else data / "evals" / f"{key}.jsonl"
        if p.exists():
            out[key] = by_meaning(model, p, device, limit)
            r = out[key]
            say(f"{key}: words {r['words'] * 100:.1f}% | meaning " + " ".join(f"{n}: {r['meaning'][str(n)] * 100:.1f}%" for n in NS)
                + " | sure " + " ".join(f"{n}: {r['sure'][str(n)] * 100:.1f}%" for n in NS), flush=True)
    (run / "think.json").write_text(json.dumps(out))
    return out


def _mean(get):
    v = [get(k) for k in REASON]
    return None if any(a is None for a in v) else sum(v) / len(v)


def verdict(P, T):
    if not (P and T):
        return [("verdict", "none yet: Bee-135's run and Think-135's scores by meaning are both needed")]
    bee = _mean(lambda k: P["word"].get(k))
    m64, s64, s1, w = (_mean(lambda k: (T.get(k) or {}).get("meaning", {}).get("64")),
                       _mean(lambda k: (T.get(k) or {}).get("sure", {}).get("64")),
                       _mean(lambda k: (T.get(k) or {}).get("sure", {}).get("1")), _mean(lambda k: (T.get(k) or {}).get("words")))
    if None in (bee, m64, s64, s1, w):
        return [("verdict", "none: a reasoning test was not scored")]
    ok1, ok2 = m64 - bee >= NEED_THINK - 1e-9, s64 - s1 >= NEED_SEARCH - 1e-9
    word = "KEEP: it thinks, and search pays" if ok1 and ok2 else \
        "PROMISING: widen the search before deciding" if ok2 or m64 > w else "DROP: answering by thoughts does not help here"
    return [("it thinks", f"{'passes' if ok1 else 'fails'}: by meaning with 64 thoughts {m64 * 100:.1f}% against Bee-135 "
                          f"{bee * 100:.1f}% = {(m64 - bee) * 100:+.1f} points (rule: +{NEED_THINK * 100:.1f} or more)"),
            ("search pays", f"{'passes' if ok2 else 'fails'}: the surest of 64 thoughts {s64 * 100:.1f}% against the only one "
                            f"{s1 * 100:.1f}% = {(s64 - s1) * 100:+.1f} points (rule: +{NEED_SEARCH * 100:.1f} or more)"),
            ("its own words", f"Think-135 by next-word loss, no thoughts: {w * 100:.1f}%"),
            ("verdict", word)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--set", choices=list(ARMS), default=None)
    ap.add_argument("--name", choices=list(ARMS), default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--score", action="store_true", help="score Think-135's run by meaning (writes think.json)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if a.set or a.name:
        print(" ".join(set_words(a.set, a.tune)) if a.set else RUNS[a.name])
        return
    from oazy import term
    from oazy.data import default_dir, run_dir
    from oazy.report import pct
    runs = Path(a.runs or default_dir("runs"))
    data = Path(a.data or default_dir("data"))
    td = run_dir(runs, RUNS["think"])
    if a.score:
        score(td, data, torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu")), a.limit)
    got = {arm: jread(run_dir(runs, rn)) for arm, rn in RUNS.items()}
    T = None
    if (td / "think.json").exists():
        T = json.loads((td / "think.json").read_text())
    body = []
    for arm, label in LABEL.items():
        g = got[arm]
        if g is None:
            body.append([label, "not run yet"] + [""] * (3 + len(TESTS)))
            continue
        star = "" if g["full"] else "*"
        body.append([term.c(label, "bold"), f"{g['tok'] / 1e9:.2f}B" + ("" if g["done"] else " (running)"),
                     term.duration(g["sec"]), term.rate(g["tok"] / max(g["sec"], 1e-9)), f"{g['val']:.3f}"]
                    + [pct(g["word"][k]) + (star if g["word"][k] is not None else "") for k, _ in TESTS])
    term.header("Think-135 against Bee-135", "first Opal · equal tokens · the same seed · no mentor · by next-word loss, no thoughts")
    term.table(["", "tokens", "training time", "speed", "held-out loss"] + [t for _, t in TESTS], body,
               right=tuple(range(1, 5 + len(TESTS))))
    loop = None
    if (td / "metrics.jsonl").exists():
        for line in (td / "metrics.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "loop":
                loop = r.get("pass_loss")
    if loop:
        nm = {"0": "no thoughts", "1": "one made thought a sentence", "9": "the true thoughts (not honest)"}
        print("  held-out loss: " + "  ".join(f"{nm.get(str(k), k)} {float(v):.3f}" for k, v in sorted(loop.items(), key=lambda kv: int(kv[0]))))
    if T:
        term.header("Think-135 answering by meaning", "the mind makes N thoughts at the question's end; the nearest answer wins")
        rows_ = [("words (no thoughts)", lambda r: r["words"]), ("words + 4 thoughts", lambda r: r["words_think"])]
        rows_ += [(f"meaning, {n} thought{'s' if n > 1 else ''}", (lambda r, n=n: r["meaning"][str(n)])) for n in NS]
        rows_ += [(f"the surest of {n}", (lambda r, n=n: r["sure"][str(n)])) for n in NS]
        tb = [[lab] + [pct(f(T[k])) if k in T else "" for k, _ in TESTS]
              + [pct(_mean(lambda k: f(T[k]) if k in T else None))] for lab, f in rows_]
        if got["plain"]:
            tb.insert(0, [term.c("Bee-135, the usual way", "bold")] + [pct(got["plain"]["word"][k]) for k, _ in TESTS]
                      + [pct(_mean(lambda k: got["plain"]["word"].get(k)))])
        term.table(["how it answers"] + [t for _, t in TESTS] + ["reasoning"], tb, right=tuple(range(1, 2 + len(TESTS))))
    term.header("Think-135: verdict by the rule fixed before the run", color="green")
    term.rows(verdict(got["plain"], T))
    return got, T


if __name__ == "__main__":
    main()
