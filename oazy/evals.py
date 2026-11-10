"""Three more standard tests next to HellaSwag, so we know *what* improved.

    python3 -m oazy.evals --prepare                                  # once: downloads them into <data>/evals/
    python3 -m oazy.evals --ckpt /workspace/oazy_runs/bee3/ckpt.pt   # score a model (the trainer does this at the end)

  ARC-Easy   2,376 grade-school science questions, four answers to choose from (AI2, 2018). Knowledge.
  PIQA       1,838 everyday "how would you do this?" questions, two answers (2019). Physical common sense.
  LAMBADA    5,153 passages whose last word is guessable only from the whole passage (2016): the model
             must write that exact word. A direct test of long-range memory. Scored as accuracy and as the
             perplexity of the last word.

Reasoning tests (João, Nov 10), each also scored on the trained SmolLM2s as yardsticks (`--run smollm2-360m`):
  ARC-Challenge  1,172 harder science questions, four answers (AI2, 2018). Guessing 25%.
  CommonsenseQA  1,221 everyday-reasoning questions, five answers (2019; the validation set, the test has no
                 answers). Guessing 20%.
  Winogrande     1,267 "who does it refer to" sentences, two options (2019; validation). Each option fills the
                 blank and the model scores the rest of the sentence after it (the usual "partial" scoring).
                 Guessing 50%.
  bAbI           Facebook's 20 kinds of short stories with a question (2015): where things are, who has what,
                 yes/no, counting, lists, negation, deduction, induction, paths, motivations... 50 questions of
                 each kind from the test files, answered by choosing among up to 8 answers that kind of question
                 has. The closest outside cousin of our world, in someone else's words.

ARC-Easy and PIQA are scored like HellaSwag (oazy/hellaswag.py): the model picks the answer whose words
surprise it least, per token ("acc_norm") and per letter ("acc_char"). Random guessing: 25% and 50%.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import default_dir
from . import tok, hellaswag

SETS = {
    # name: (Hugging Face repo, config, split, files to prefer)
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test"),
    "piqa": ("ybisk/piqa", None, "validation"),
    "lambada": ("EleutherAI/lambada_openai", "default", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "csqa": ("tau/commonsense_qa", None, "validation"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
}
BABI_URLS = ("https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz",      # the Keras mirror (alive Nov 10)
             "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz")
REASONING = ("arc_challenge", "csqa", "winogrande", "babi")
CHANCE = {"arc_easy": 0.25, "arc_challenge": 0.25, "piqa": 0.5, "csqa": 0.2, "winogrande": 0.5}


def _rows(repo, config, split):
    """The rows of one split, from the dataset's parquet files (no loading scripts, no cache). A dataset that
    only has a loading script (PIQA) is read from the Hub's own parquet conversion of it (refs/convert/parquet)."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    api = HfApi()
    for rev in (None, "refs/convert/parquet"):
        try:
            files = api.list_repo_files(repo, repo_type="dataset", revision=rev)
        except Exception as e:
            print(f"  ({repo} {rev or 'main'}: {type(e).__name__})")
            continue
        want = [f for f in files if f.endswith(".parquet") and split in Path(f).name and (not config or f.startswith(config + "/"))]
        if not want:                         # the Hub's conversion: <config>/<split>/0000.parquet
            want = [f for f in files if f.endswith(".parquet") and f"/{split}/" in "/" + f and (not config or f.startswith(config + "/"))]
        if not want:
            want = [f for f in files if f.endswith(".parquet") and split in f]
        if want:
            for f in sorted(want):
                yield from pq.read_table(hf_hub_download(repo, f, repo_type="dataset", revision=rev)).to_pylist()
            return
    raise RuntimeError(f"no parquet files for {repo} {config or ''} {split} (main or refs/convert/parquet)")


def prepare(out: Path):
    """Each set on its own: one that cannot be fetched is reported and skipped (the run scores what is there)."""
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("arc_easy", _prep_arc), ("piqa", _prep_piqa), ("lambada", _prep_lambada),
                     ("arc_challenge", _prep_arc_c), ("csqa", _prep_csqa), ("winogrande", _prep_wino), ("babi", _prep_babi)):
        if (out / f"{name}.jsonl").exists():
            continue
        try:
            fn(out)
        except Exception as e:
            print(f"{name}: not prepared ({type(e).__name__}: {str(e)[:200]}); run this again later, the run scores what is there")


def _prep_arc(out: Path, name="arc_easy"):
    # ARC-Easy and PIQA in HellaSwag's shape: {"ctx", "endings", "label"}
    rows = []
    for r in _rows(*SETS[name]):
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue
        rows.append({"ctx": f"Question: {r['question'].strip()}\nAnswer:", "endings": [t.strip() for t in r["choices"]["text"]],
                     "label": labels.index(r["answerKey"])})
    (out / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"{name}: {len(rows)} questions")


def _prep_arc_c(out: Path):
    _prep_arc(out, "arc_challenge")


def _prep_csqa(out: Path):
    rows = []
    for r in _rows(*SETS["csqa"]):
        labels = list(r["choices"]["label"])
        if r.get("answerKey") not in labels:
            continue
        rows.append({"ctx": f"Question: {r['question'].strip()}\nAnswer:", "endings": [t.strip() for t in r["choices"]["text"]],
                     "label": labels.index(r["answerKey"])})
    (out / "csqa.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"csqa: {len(rows)} questions")


def _prep_wino(out: Path):
    rows = []
    for r in _rows(*SETS["winogrande"]):
        s, ans = r["sentence"], str(r.get("answer", ""))
        if "_" not in s or ans not in ("1", "2"):
            continue
        pre, post = s.split("_", 1)
        rows.append({"ctxs": [pre + r["option1"], pre + r["option2"]], "endings": [post.strip()] * 2, "label": int(ans) - 1})
    (out / "winogrande.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"winogrande: {len(rows)} sentences")


def _prep_babi(out: Path, per_task=50, max_choices=8):
    """bAbI's 20 tasks (English, 1k version, test files) as multiple choice: the story so far, the question, and
    the right answer among up to max_choices answers that task has (picked once, seeded)."""
    import io
    import random
    import re
    import tarfile
    import urllib.request
    data, err = None, None
    for url in BABI_URLS:
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            break
        except Exception as e:      # a dead mirror: try the next
            err = e
    if data is None:
        raise err
    rng = random.Random(0)
    rows = []
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        names = sorted(n for n in tf.getnames() if re.search(r"/en/qa\d+_.*_test\.txt$", n))
        for n in names:
            task = int(re.search(r"qa(\d+)_", n).group(1))
            qs, story = [], []
            for line in tf.extractfile(n).read().decode().splitlines():
                num, _, rest = line.partition(" ")
                if num == "1":
                    story = []
                if "\t" in rest:
                    q, a, _ = (rest.split("\t") + [""])[:3]
                    qs.append((" ".join(story), q.strip(), a.strip()))
                else:
                    story.append(rest.strip())
            answers = sorted({a for _, _, a in qs})
            for story_t, q, a in rng.sample(qs, min(per_task, len(qs))):
                others = [x for x in answers if x != a]
                cands = [a] + rng.sample(others, min(max_choices - 1, len(others)))
                rng.shuffle(cands)
                rows.append({"ctx": f"{story_t}\nQuestion: {q}\nAnswer:", "endings": [c.replace(",", ", ") for c in cands],
                             "label": cands.index(a), "task": task})
    if not rows:
        raise RuntimeError("no bAbI test files in the archive")
    (out / "babi.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"babi: {len(rows)} questions ({len(names)} tasks)")


def _prep_piqa(out: Path):
    rows = [{"ctx": f"Question: {r['goal'].strip()}\nAnswer:", "endings": [r["sol1"].strip(), r["sol2"].strip()], "label": int(r["label"])}
            for r in _rows(*SETS["piqa"]) if r.get("label") in (0, 1)]
    (out / "piqa.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"piqa: {len(rows)} questions")


def _prep_lambada(out: Path):
    rows = []
    for r in _rows(*SETS["lambada"]):
        text = r["text"].rstrip()
        ctx, _, last = text.rpartition(" ")
        if ctx and last:
            rows.append({"ctx": ctx, "target": " " + last})
    (out / "lambada.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"lambada: {len(rows)} passages")


@torch.no_grad()
def lambada(model, path: Path, device, limit=None, batch=16):
    """Accuracy (every token of the last word guessed right, greedily) and perplexity of the last word."""
    enc = tok.for_model(model)
    rows = hellaswag.load(path, limit)
    was = model.training
    model.eval()
    correct, nll, n = 0, 0.0, 0
    L_max = getattr(model.cfg, "max_len", 1024)
    for b0 in range(0, len(rows), batch):
        seqs, spans = [], []
        for r in rows[b0:b0 + batch]:
            c, t = enc.encode_ordinary(r["ctx"]), enc.encode_ordinary(r["target"])
            c = c[-(L_max - len(t)):]
            seqs.append(c + t)
            spans.append((len(c), len(c) + len(t)))
        L = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            logits = model._body(x).float()
        for i, (a, b) in enumerate(spans):
            lg = logits[i, a - 1:b - 1]
            tgt = x[i, a:b]
            correct += int(bool((lg.argmax(-1) == tgt).all()))
            nll += float(F.cross_entropy(lg, tgt, reduction="sum"))
            n += 1
    if was:
        model.train()
    return {"acc": correct / max(n, 1), "ppl": float(torch.exp(torch.tensor(nll / max(n, 1)))), "n": n}


@torch.no_grad()
def evaluate_all(model, data_root: Path, device, limit=None):
    """{"arc_easy", "arc_easy_char", "piqa", "piqa_char", ..., "lambada", "lambada_ppl"} for the files that exist."""
    d = Path(data_root) / "evals"
    out = {}
    for name in ("arc_easy", "piqa") + REASONING:
        p = d / f"{name}.jsonl"
        if p.exists():
            r = hellaswag.evaluate(model, p, device, limit)
            out[name], out[name + "_char"], out[name + "_n"] = r["acc_norm"], r["acc_char"], r["n"]
    p = d / "babi.jsonl"
    if p.exists():                                       # what guessing scores there (the number of choices varies)
        rows = hellaswag.load(p, limit)
        out["babi_chance"] = sum(1 / len(r["endings"]) for r in rows) / max(1, len(rows))
    p = d / "lambada.jsonl"
    if p.exists():
        r = lambada(model, p, device, limit)
        out["lambada"], out["lambada_ppl"], out["lambada_n"] = r["acc"], r["ppl"], r["n"]
    return out


def score_runs(names, root: Path, runs: Path, device, say=print):
    """Score finished runs (or SmolLM2-135M/360M/1.7B as yardsticks) on every test in <data>/evals, in full, and add
    the result to each run's metrics as a {"kind": "evals"} line. Then one table, guessing last."""
    from .logres import _load, _records, NICE
    from .data import run_dir
    from .train import Run
    from . import term
    res = {}
    for n in names:
        d = run_dir(runs, n)
        d.mkdir(parents=True, exist_ok=True)
        model = _load(d, device)
        r = evaluate_all(model, root, device)
        recs = _records(d)
        base = {k: recs[-1].get(k, 0) for k in ("step", "tokens", "elapsed")} if recs else {"step": 0, "tokens": 0, "elapsed": 0}
        Run(d).log("metrics.jsonl", {"kind": "evals", **base, **r, "full": True, "rescored": True})
        res[n] = r
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        say(f"{n}: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in r.items() if k in CHANCE or k == "babi"))
    keys = [k for k in ("arc_easy", "arc_challenge", "piqa", "csqa", "winogrande", "babi") if any(k in r for r in res.values())]
    title = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Chall.", "piqa": "PIQA", "csqa": "CommonsenseQA",
             "winogrande": "Winogrande", "babi": "bAbI"}
    body = [[term.c(NICE.get(n, n), "bold")] + [("–" if r.get(k) is None else f"{r[k] * 100:.1f}%") for k in keys]
            for n, r in res.items()]
    ch = dict(CHANCE, babi=next((r.get("babi_chance") for r in res.values() if r.get("babi_chance")), None))
    body.append([term.c("guessing", "dim")] + [("–" if ch.get(k) is None else f"{ch[k] * 100:.1f}%") for k in keys])
    term.header("reasoning and knowledge tests", "per token (the usual way); higher is better")
    term.table(["model"] + [title[k] for k in keys], body, right=tuple(range(1, len(keys) + 1)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="append", help="score this run folder (or smollm2-135m / smollm2-360m / "
                    "smollm2-1.7b) on every test, in full; repeat for several")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    root = Path(a.data or default_dir("data"))
    if a.prepare:
        prepare(root / "evals")
    if a.run:
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        score_runs(a.run, root, Path(a.runs or default_dir("runs")), device)
    if a.ckpt:
        from .train import load_model
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, _ = load_model(Path(a.ckpt), device)
        print(json.dumps(evaluate_all(model, root, device, a.limit), indent=1))


if __name__ == "__main__":
    main()
