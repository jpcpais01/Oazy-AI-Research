"""Three more standard tests next to HellaSwag, so we know *what* improved.

    python3 -m oazy.evals --prepare                                  # once: downloads them into <data>/evals/
    python3 -m oazy.evals --ckpt /workspace/oazy_runs/bee3/ckpt.pt   # score a model (the trainer does this at the end)

  ARC-Easy   2,376 grade-school science questions, four answers to choose from (AI2, 2018). Knowledge.
  PIQA       1,838 everyday "how would you do this?" questions, two answers (2019). Physical common sense.
  LAMBADA    5,153 passages whose last word is guessable only from the whole passage (2016): the model
             must write that exact word. A direct test of long-range memory. Scored as accuracy and as the
             perplexity of the last word.

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
}


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
    for name, fn in (("arc_easy", _prep_arc), ("piqa", _prep_piqa), ("lambada", _prep_lambada)):
        if (out / f"{name}.jsonl").exists():
            continue
        try:
            fn(out)
        except Exception as e:
            print(f"{name}: not prepared ({type(e).__name__}: {str(e)[:200]}); run this again later, the run scores what is there")


def _prep_arc(out: Path):
    # ARC-Easy and PIQA in HellaSwag's shape: {"ctx", "endings", "label"}
    rows = []
    for r in _rows(*SETS["arc_easy"]):
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue
        rows.append({"ctx": f"Question: {r['question'].strip()}\nAnswer:", "endings": [t.strip() for t in r["choices"]["text"]],
                     "label": labels.index(r["answerKey"])})
    (out / "arc_easy.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"arc_easy: {len(rows)} questions")


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
    """{"arc_easy", "arc_easy_char", "piqa", "piqa_char", "lambada", "lambada_ppl"} for the files that exist."""
    d = Path(data_root) / "evals"
    out = {}
    for name in ("arc_easy", "piqa"):
        p = d / f"{name}.jsonl"
        if p.exists():
            r = hellaswag.evaluate(model, p, device, limit)
            out[name], out[name + "_char"], out[name + "_n"] = r["acc_norm"], r["acc_char"], r["n"]
    p = d / "lambada.jsonl"
    if p.exists():
        r = lambada(model, p, device, limit)
        out["lambada"], out["lambada_ppl"], out["lambada_n"] = r["acc"], r["ppl"], r["n"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    root = Path(a.data or default_dir("data"))
    if a.prepare:
        prepare(root / "evals")
    if a.ckpt:
        from .train import load_model
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, _ = load_model(Path(a.ckpt), device)
        print(json.dumps(evaluate_all(model, root, device, a.limit), indent=1))


if __name__ == "__main__":
    main()
