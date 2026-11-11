"""Text that shows its thinking (Opal2's "think" part, João, Nov 11).

Most of what a model reads hides the reasoning: a writer knew why each claim was true and wrote only the claim. Here an
open model reads a passage of Opal and writes, after it, the thinking a careful reader does: why each step follows,
the sums checked, the ideas joined up. The passage stays word for word; the thinking is added after it:

    <the passage, as in Opal>

    Thinking it through: The passage says the ice melts first at the edges. That follows because ...

Why: every hard word of the passage now comes with the reasons behind it, so each one teaches more. Thinking
Augmented Pre-training (arXiv 2509.20186) found about 3x the learning per token this way; Tufa Labs (Apr 2026) trained a
0.8B from scratch for 12B tokens on thought-process rewrites made by a 0.8B model and reached the plain run's maths
scores with 3-6x fewer tokens, the gain holding when the examples were random (reasoning, not format).

The writer. Qwen3-1.7B (Apache 2.0) in its no-thinking mode by default (--writer to change): good plain English, fast.
It runs on any GPU with vLLM (much faster) or plain transformers. It is a teacher only in the sense that its words
are read: the model never sees its scores, and the passages are ours.

Which passages. Teaching text (FineWeb-Edu), Wikipedia, maths and books, 120-1,000 pieces long, drawn at random from
Opal's training text. Thoughts that are too short, cut off, not English, full of markdown or repeating themselves are
dropped.

    python3 -m oazy.think --tokens 400e6                 # GPU: write until 400M pieces (passage + thinking); resumable
    python3 -m oazy.think --tokens 400e6 --pack-only     # turn what is written into <data>/opal/think/{train,val}.bin
    python3 -m oazy.opal2 show --part think              # read a few

Cost. About half of each document is new writing. On an H200 with vLLM, Qwen3-1.7B writes roughly 20-30k pieces a
second, so 400M pieces (~200M written) take about 2-3 hours; an RTX 5090 about twice that. Opal2 reads it at 12%
of the run (1.2B pieces of a 10B run: each document about three times, which the data-repetition studies find as good
as fresh).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

VERSION = 1
WRITER = "Qwen/Qwen3-1.7B"
SOURCES = {"fwe": 0.40, "wiki": 0.20, "math": 0.25, "books": 0.15}   # Opal's own text (oazy/mix.py SOURCES)
MIN_DOC, MAX_DOC = 120, 1000
LEADS = ["Thinking it through:", "Let me think about this.", "Working through it:", "Thinking about this carefully:",
         "Let me think it through.", "Reading it again, slowly:", "Why is this so? Let me think."]
PROMPT = ("Here is a passage:\n\n{doc}\n\n"
          "Write the thinking a careful, curious reader does while studying this passage. Work out the reasoning behind "
          "its main points step by step and say why each step follows. If there are numbers or logic, check them. "
          "Connect the ideas to each other, and if something is assumed, say what. Write in plain, clear and correct "
          "English, in full sentences and paragraphs: no headings, no lists, no markdown, and do not copy the passage. "
          "Write about {words} words.")
_MD = re.compile(r"(\*\*|^#|^\s*[-*] |^\s*\d+\.\s|```|<think>|</think>)", re.M)


def think_dir(data: Path) -> Path:
    from .opal import opal_dir
    return opal_dir(Path(data), rename=False) / "think"


# ---------------------------------------------------------------- passages from Opal
class Passages:
    """Random whole documents (MIN_DOC-MAX_DOC pieces) from Opal's sources, decoded to text. Seeded by document index."""

    def __init__(self, opal: Path, enc):
        from .mix import source_path
        self.enc, self.eot = enc, enc.eot_token
        self.src = {}
        for n, w in SOURCES.items():
            p = source_path(opal, n)
            if p.exists():
                self.src[n] = (np.memmap(p, dtype=np.uint16, mode="r"), w)
        if not self.src:
            raise SystemExit(f"no Opal text under {opal} (expected {', '.join(SOURCES)})")

    def get(self, i: int):
        rng = np.random.default_rng([VERSION, i])
        names = list(self.src)
        ws = np.array([self.src[n][1] for n in names])
        name = names[rng.choice(len(names), p=ws / ws.sum())]
        arr = self.src[name][0]
        for _ in range(50):
            at = int(rng.integers(0, max(1, arr.size - 4 * MAX_DOC)))
            win = np.asarray(arr[at:at + 4 * MAX_DOC + 2])
            ends = np.flatnonzero(win == self.eot)
            if ends.size < 2:
                continue
            for a, b in zip(ends[:-1], ends[1:]):
                n = int(b - a - 1)
                if MIN_DOC <= n <= MAX_DOC:
                    return name, self.enc.decode(win[a + 1:b].tolist())
        return name, None


# ---------------------------------------------------------------- the writer
def clean(thought: str):
    """The thought, tidied, or None when it should be dropped."""
    t = thought.strip()
    if "<think>" in t:                                   # Qwen3 sometimes still opens an (empty) thinking block
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    if _MD.search(t):
        t = re.sub(r"\*\*|__", "", t)
        if _MD.search(t):
            return None
    words = t.split()
    if len(words) < 60:
        return None
    if sum(ch.isascii() for ch in t) < 0.985 * len(t):
        return None
    grams = [" ".join(words[i:i + 4]).lower() for i in range(len(words) - 3)]
    if grams and len(set(grams)) < 0.8 * len(grams):    # going round in circles
        return None
    if t[-1] not in ".!?\"'":
        return None
    return t


class Writer:
    def __init__(self, name=WRITER, max_new=700, temperature=0.7, gpu_share=0.85):
        self.name, self.max_new, self.temperature = name, max_new, temperature
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=name, dtype="bfloat16", gpu_memory_utilization=gpu_share, max_model_len=4096,
                           enable_prefix_caching=True)
            self.sp = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=max_new, repetition_penalty=1.05)
            self.kind = "vllm"
        except ImportError:
            import torch
            from transformers import AutoModelForCausalLM
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
            self.kind = "transformers"

    def chat(self, text):
        msgs = [{"role": "user", "content": text}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def write(self, prompts):
        """-> [(text, finished)]"""
        chats = [self.chat(p) for p in prompts]
        if self.kind == "vllm":
            outs = self.llm.generate(chats, self.sp, use_tqdm=False)
            return [(o.outputs[0].text, o.outputs[0].finish_reason == "stop") for o in outs]
        import torch
        out = []
        for i in range(0, len(chats), 32):
            b = self.tok(chats[i:i + 32], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                g = self.model.generate(**b, max_new_tokens=self.max_new, do_sample=True, temperature=self.temperature,
                                        top_p=0.9, repetition_penalty=1.05)
            for row in g[:, b["input_ids"].shape[1]:]:
                ids = row.tolist()
                done = self.tok.eos_token_id in ids or (self.tok.pad_token_id in ids)
                out.append((self.tok.decode(row, skip_special_tokens=True), done))
        return out


def document(doc: str, thought: str, i: int) -> str:
    lead = LEADS[i % len(LEADS)]
    return f"{doc.rstrip()}\n\n{lead} {thought}"


# ---------------------------------------------------------------- writing, then packing
def written(path: Path):
    """(documents so far, their pieces, next index) of a texts.jsonl."""
    n = tok = nxt = 0
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                tok += d.get("pieces", 0)
                nxt = max(nxt, d["i"] + 1)
    return n, tok, nxt


def write(data: Path, tokens: float, writer=WRITER, batch=512, say=print):
    from . import tok as TK, term
    from .opal import opal_dir
    enc = TK.get("smollm2")
    opal = opal_dir(Path(data), rename=False)
    out = think_dir(data)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "texts.jsonl"
    n, have, i = written(path)
    if have >= tokens:
        say(f"[think] {path}: {have / 1e6:.0f}M pieces already written ({n:,} documents)")
        return path
    say(f"[think] writing until {tokens / 1e6:.0f}M pieces into {path} with {writer} (have {have / 1e6:.0f}M)")
    ps = Passages(opal, enc)
    w = Writer(writer)
    say(f"[think] the writer runs on {w.kind}")
    t0, made0, kept, tried = time.time(), have, 0, 0
    with open(path, "a") as f:
        while have < tokens:
            items = []
            while len(items) < batch:
                name, doc = ps.get(i)
                if doc:
                    nw = len(doc.split())
                    items.append((i, name, doc, PROMPT.format(doc=doc, words=int(min(400, max(150, nw * 0.6))) // 10 * 10)))
                i += 1
            res = w.write([x[3] for x in items])
            for (j, name, doc, _), (thought, done) in zip(items, res):
                tried += 1
                t = clean(thought) if done else None
                if t is None:
                    continue
                text = document(doc, t, j)
                pieces = len(enc.encode_ordinary(text)) + 1
                f.write(json.dumps({"i": j, "src": name, "text": text, "pieces": pieces}) + "\n")
                have += pieces
                kept += 1
            f.flush()
            dt = max(time.time() - t0, 1e-6)
            rate = (have - made0) / dt
            term.out(f"  {term.bar(min(have / tokens, 1), 24, 'blue')} {min(have / tokens, 1) * 100:5.1f}%  "
                     f"{have / 1e6:.0f}M / {tokens / 1e6:.0f}M pieces · kept {kept / max(tried, 1):.0%} · "
                     f"{rate / 1e3:.1f}k pieces/s · {term.duration((tokens - have) / max(rate, 1))} left", flush=True)
    return path


def pack(data: Path, val_share=0.005, say=print):
    """texts.jsonl -> train.bin / val.bin (+ meta.json), each document then <|endoftext|>."""
    from . import tok as TK
    enc = TK.get("smollm2")
    out = think_dir(data)
    path = out / "texts.jsonl"
    if not path.exists():
        raise SystemExit(f"nothing written yet in {path}: run  python3 -m oazy.think --tokens 400e6  on a GPU first")
    texts, srcs = [], {}
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            texts.append(d["text"])
            srcs[d["src"]] = srcs.get(d["src"], 0) + 1
    n_val = max(1, int(len(texts) * val_share))
    sizes = {}
    for split, part in (("val", texts[:n_val]), ("train", texts[n_val:])):
        with open(out / f"{split}.bin.tmp", "wb") as f:
            n = 0
            for k in range(0, len(part), 4096):
                flat = []
                for ids in enc.encode_ordinary_batch(part[k:k + 4096]):
                    flat += ids
                    flat.append(enc.eot_token)
                arr = np.asarray(flat, dtype=np.uint16)
                f.write(arr.tobytes())
                n += arr.size
        (out / f"{split}.bin.tmp").replace(out / f"{split}.bin")
        sizes[split] = n
    meta = {"dataset": "Opal2 think: Opal passages + the thinking a careful reader does (oazy/think.py)", "tokenizer": "smollm2",
            "think": VERSION, "documents": len(texts), "by_source": srcs, "train_tokens": sizes["train"],
            "val_tokens": sizes["val"], "built": time.strftime("%Y-%m-%d %H:%M")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    say(f"[think] packed {len(texts):,} documents: {sizes['train'] / 1e6:.0f}M training pieces, {sizes['val'] / 1e6:.1f}M held out "
        f"-> {out}")
    return out


def main():
    from .data import default_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="the data folder (Opal is under it); default $OAZY_DATA or the lab's")
    ap.add_argument("--tokens", type=float, default=400e6, help="pieces to write (passage + thinking), default 400M")
    ap.add_argument("--writer", default=WRITER, help=f"the model that writes the thinking (default {WRITER})")
    ap.add_argument("--batch", type=int, default=512, help="passages per round of writing")
    ap.add_argument("--pack-only", action="store_true", help="only pack what is written into train.bin / val.bin")
    a = ap.parse_args()
    data = Path(a.data or default_dir("data"))
    if not a.pack_only:
        write(data, a.tokens, a.writer, a.batch)
    pack(data)


if __name__ == "__main__":
    main()
