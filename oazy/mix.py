"""A mix of texts, for Bee0.2B: school-like web pages, general web pages, and (near the end) conversations.

    python -m oazy.mix --source dclm --tokens 2e9     # -> <data>/dclm/train.bin   (~20-40 min)
    python -m oazy.mix --source chat --tokens 4e8     # -> <data>/chat/train.bin   (~5 min)

The three sources, all tokenized with GPT-2's tokenizer (so GPT-2 large can still be the mentor):

  fwe    FineWeb-Edu (<data>/train.bin, made by oazy/data.py): web pages picked for teaching value.
         Great for knowledge and benchmarks, but a bit "textbook".
  dclm   DCLM-baseline (DataComp-LM, 2024): web pages picked by a quality filter trained on good
         answers and explanations. More varied, more natural writing; the best open web mix in 2024.
  chat   smol-smoltalk (Hugging Face, 2024): conversations made for small models, written as
             User: ...
             Assistant: ...
         so that Bee also learns how a conversation goes, not only how an article goes.

How the mix changes over training (fractions of the words Bee reads):

  0% ............................ 80% ................. 100%
  65% FineWeb-Edu, 35% DCLM          45% FWE, 25% DCLM, 30% chat   (the "cool-down": the learning
                                                                    rate falls to zero here too)

Over the whole run that is about 61% FineWeb-Edu, 33% DCLM and 6% chat. Putting the chats (and the
learning-rate decay) at the end is what SmolLM2/3, MiniCPM and others do: the last thing learned,
while the steps are getting small, sticks best. Missing sources are skipped (the rest is rescaled).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
from pathlib import Path

import numpy as np

from .data import write_tokens, default_dir, get_tokenizer

DEFAULT_MIX = "fwe:0.65,dclm:0.35@0;fwe:0.45,dclm:0.25,chat:0.3@0.8"
# Tagus, our dataset (oazy/tagus.py; SmolLM2-tokenizer models only): seven kinds of text, conversations
# a pinch throughout and 30% of the cool-down
TAGUS_MIX = ("fwe:0.45,dclm:0.25,cosmo:0.12,wiki:0.05,math:0.05,code:0.05,chat:0.03@0;"
              "fwe:0.30,dclm:0.05,cosmo:0.15,wiki:0.05,math:0.10,code:0.05,chat:0.30@0.8")
BUNDLE_MIX = TAGUS_MIX                     # its first name
# Sado, our second dataset (oazy/sado.py): Tagus plus books, stricter web text, no test questions, no repeats.
# Religious text (João, Nov 9): 2% in all = religion 1.95% (Gutenberg's religious books, oazy/religion.py) + faith 0.05%
# (stories and letters where a quiet belief lives underneath, oazy/faith.py); and creed 0.01% (the line "God loves me.
# I love God." on its own, repeated). Each is skipped while its folder is absent.
SADO_MIX = ("fwe:0.3999,dclm:0.20,cosmo:0.10,books:0.06,wiki:0.06,math:0.07,code:0.06,chat:0.03,religion:0.0195,faith:0.0005,creed:0.0001@0;"
            "fwe:0.2599,dclm:0.05,cosmo:0.11,books:0.06,wiki:0.06,math:0.11,code:0.06,chat:0.27,religion:0.0195,faith:0.0005,creed:0.0001@0.8")
FAITH_SHARE, RELIGION_SHARE, CREED_SHARE = 0.0005, 0.0195, 0.0001
# the same, if oazy/sado_score.py has filled best/ (documents picked by two models, reread in the cool-down; not used now)
SADO_BEST_MIX = SADO_MIX.rsplit(";", 1)[0] + (";fwe:0.1599,dclm:0.05,cosmo:0.11,books:0.06,wiki:0.06,math:0.11,code:0.06,chat:0.27,"
                                              "best:0.10,religion:0.0195,faith:0.0005,creed:0.0001@0.8")
MIXES = {"default": DEFAULT_MIX, "v1": DEFAULT_MIX, "tagus": TAGUS_MIX, "bundle": TAGUS_MIX, "sado": SADO_MIX, "sado-best": SADO_BEST_MIX}
SOURCES = {"fwe": "", "dclm": "dclm", "chat": "chat",          # name -> sub-folder of the data folder
           "cosmo": "cosmo", "wiki": "wiki", "math": "math", "code": "code", "books": "books", "best": "best",
           "faith": "faith", "religion": "religion", "creed": "creed", "world": "world"}
PREPARE = {"faith": "python3 -m oazy.faith build", "religion": "python3 -m oazy.religion fetch, then build",
           "creed": "python3 -m oazy.religion creed", "world": "python3 -m oazy.emerald build"}     # how a source that is not made by this file is made


def with_cooldown(mix: str, start: float) -> str:
    """The same mix with its cool-down text (the stage at 0.8) starting at `start` instead (Bee4: 0.7)."""
    return ";".join(f"{part.rsplit('@', 1)[0]}@{start:g}" if part.rsplit("@", 1)[-1] == "0.8" else part for part in mix.split(";"))


def with_share(mix: str, name: str, share: float) -> str:
    """The same mix with source `name` at exactly `share` of every phase's words (the others keep their proportions);
    share 0 takes it out. E.g. with_share(SADO_MIX, "faith", 0.001) doubles the faith pinch."""
    out = []
    for part in mix.split(";"):
        ws, at = part.split("@")
        w = {k: float(v) for k, v in (s.split(":") for s in ws.split(","))}
        w.pop(name, None)
        rest = sum(w.values())
        if share > 0:
            w = {k: v / rest * (1 - share) for k, v in w.items()}
            w[name] = share
        out.append(",".join(f"{k}:{v:.6g}" for k, v in w.items()) + "@" + at)
    return ";".join(out)


EMERALD_SHARE, EMERALD_EARLY, EMERALD_EARLY_X, EMERALD_LATE_X = 0.10, 0.2, 3.0, 0.5


def emerald_plan(spec) -> list:
    """The world's share over the run as [(from, share), ...], from a plan written out (--emerald-plan):
        "0.30:0.2,0.05"   30% world until 20% of the run, then 5% (= today's emerald:0.10)
        "0.30:0.1,0.15"   30% for the first 10%, then 15%
        "flat:0.20"       20% the whole run (cool-down included)
        "0.3:0.1,0.15:0.9,0"  as many steps as you like: share:until, ..., last share
    or from an average share w alone (a float): 3w for the first 20%, then w/2 (Emerald's default)."""
    if isinstance(spec, (int, float)):
        w = float(spec)
        return [(0.0, min(1.0, w * EMERALD_EARLY_X)), (EMERALD_EARLY, w * EMERALD_LATE_X)] if w > 0 else [(0.0, 0.0)]
    spec = str(spec).strip()
    if spec.startswith("flat:"):
        return [(0.0, float(spec[5:]))]
    out, at, parts = [], 0.0, spec.split(",")
    for i, part in enumerate(parts):
        share, _, until = part.partition(":")
        out.append((at, float(share)))
        if until:
            nxt = float(until)
            if not at < nxt < 1:
                raise ValueError(f"the emerald plan {spec!r}: each 'until' must grow and stay under 1")
            at = nxt
        elif i < len(parts) - 1:
            raise ValueError(f"the emerald plan {spec!r}: only the last share goes without ':until'")
    if any(not 0 <= s <= 1 for _, s in out):
        raise ValueError(f"the emerald plan {spec!r}: shares go from 0 to 1")
    return out


def plan_average(plan) -> float:
    """The world's average share over the whole run."""
    ends = [a for a, _ in plan[1:]] + [1.0]
    return sum(s * (e - a) for (a, s), e in zip(plan, ends))


def plan_text(plan) -> str:
    """'30% until 20%, then 5% (10% on average)'."""
    if len(plan) == 1:
        return f"{plan[0][1]:.0%} the whole run"
    bits = [f"{s:.0%} until {e:.0%}" for (_, s), (e, _) in zip(plan, plan[1:])]
    return ", then ".join(bits + [f"{plan[-1][1]:.0%}"]) + f" ({plan_average(plan):.1%} on average)"


def emerald_mix(w=EMERALD_SHARE, base: str = SADO_MIX) -> str:
    """Emerald (oazy/emerald.py): Sado with the world's practice text. w = the average share (3w for the first 20%,
    then w/2 to the end, cool-down included: 0.10 -> 30% then 5%; 0 = plain Sado; like --world, oazy/worldmix.py),
    or a plan written out (emerald_plan: "flat:0.20", "0.30:0.1,0.15"). Each phase of Sado keeps its own proportions;
    the world takes its share of every phase's words."""
    plan = emerald_plan(w)
    if all(s <= 0 for _, s in plan):
        return base
    parts = [(float(p.rsplit("@", 1)[1]), p.rsplit("@", 1)) for p in base.split(";")]
    marks = sorted({a for a, _ in parts} | {a for a, _ in plan})
    out = []
    for t in marks:
        ws, at = [p for a, p in parts if a <= t][-1]
        share = [s for a, s in plan if a <= t][-1]
        out.append(with_share(f"{ws}@{at if float(at) == t else f'{t:g}'}", "world", share))
    return ";".join(out)


def resolve_mix(text):
    """A mix by name ("tagus", "v1", "emerald", "emerald:0.05", "emerald:flat:0.20", "emerald:0.30:0.1,0.15") or written out."""
    if text and (text == "emerald" or text.startswith("emerald:")):
        spec = text.split(":", 1)[1] if ":" in text else EMERALD_SHARE
        try:
            spec = float(spec)
        except ValueError:
            pass
        return emerald_mix(spec)
    return MIXES.get(text, text) if text else text


def parse_mix(text: str):
    """"fwe:0.65,dclm:0.35@0;fwe:0.45,...@0.8" -> [(0.0, {"fwe": .65, "dclm": .35}), (0.8, {...})]."""
    out = []
    for part in text.split(";"):
        ws, at = part.split("@")
        w = {k: float(v) for k, v in (s.split(":") for s in ws.split(","))}
        if any(k not in SOURCES for k in w):
            raise ValueError(f"unknown source in mix {part!r}; known: {list(SOURCES)}")
        out.append((float(at), w))
    out.sort(key=lambda t: t[0])
    if out[0][0] != 0:
        raise ValueError(f"the mix {text!r} must start @0")
    return out


def source_path(data: Path, name: str, split: str = "train") -> Path:
    sub = SOURCES[name]
    return (data / sub if sub else data) / f"{split}.bin"


class _Source:
    def __init__(self, path: Path, slot: int, seed: int, marks: bool = False):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        mp = path.with_suffix(".marks")                  # the world's marks, one byte per piece (oazy/emerald.py)
        self.marks = np.memmap(mp, dtype=np.uint8, mode="r") if marks and mp.exists() else None
        if self.marks is not None and self.marks.size != self.data.size:
            raise SystemExit(f"{mp} does not match {path} ({self.marks.size:,} vs {self.data.size:,}): "
                             f"python3 -m oazy.emerald marks  writes it again")
        self.slot, self.seed = slot, seed
        self.n = (len(self.data) - 1) // slot
        if self.n < 1:
            raise ValueError(f"{path} is too small")
        self.epoch, self.pos, self.sub = 0, 0, 0
        self._order()

    def _order(self):
        self.order = np.random.default_rng(self.seed + 1000 * self.epoch).permutation(self.n)

    def row(self, T: int):
        per = self.slot // (T + 1)
        i = int(self.order[self.pos]) * self.slot + self.sub * (T + 1)
        self.sub += 1
        if self.sub >= per:
            self.next_slot()
        self.last_i = i
        return self.data[i:i + T + 1]

    def next_slot(self):
        self.sub, self.pos = 0, self.pos + 1
        if self.pos >= self.n:
            self.epoch, self.pos = self.epoch + 1, 0
            self._order()


class Mix:
    """Training windows drawn from several sources, with weights that change over training.

    Each source is cut into "slots" (2,050 words for Bee): a slot gives two windows of 1,025 or one of
    2,049, so changing the window length mid-run never repeats or skips text. Which source each row of
    a batch comes from is drawn from a generator seeded by the batch number: reproducible, resumable.
    Set `progress` (0..1) before each get(); set_block(T) changes the window length."""

    def __init__(self, data: Path, mix: str, T: int, batch: int, seed: int = 0, lengths=(1024,), log=print, marks=False):
        self.plan = parse_mix(resolve_mix(mix))
        tmin = min(lengths)
        self.slot = (tmin + 1) * math.ceil((max(lengths) + 1) / (tmin + 1))
        names = sorted({k for _, w in self.plan for k in w})
        self.src = {}
        for i, n in enumerate(names):
            p = source_path(Path(data), n)
            if p.exists():
                self.src[n] = _Source(p, self.slot, seed + 17 * i, marks=marks)
            else:
                log(f"[mix] {n}: {p} not found, skipped (prepare it with: {PREPARE.get(n, 'python -m oazy.mix --source ' + n)})")
        if not self.src:
            raise SystemExit("no training text found")
        self.T, self.batch, self.seed = T, batch, seed
        self.progress, self.batches = 0.0, 0
        self.last_names = []
        self.seen = {n: 0 for n in self.src}
        self.want_marks = marks and any(s.marks is not None for s in self.src.values())
        self.last_marks = self.last_rows = None    # with marks: (B, T + 1) bytes (255 = none) and which rows have them
        if marks and not self.want_marks and "world" in self.src:
            log("[mix] the world has no marks yet: python3 -m oazy.emerald marks  adds them (the linker learns without its answer key)")

    def weights(self, progress: float):
        w = [w for at, w in self.plan if progress >= at][-1]
        w = {k: v for k, v in w.items() if k in self.src and v > 0}
        if not w:                                   # nothing of this phase is here: use whatever there is
            w = {k: 1.0 for k in self.src}
        s = sum(w.values())
        return {k: v / s for k, v in w.items()}

    def set_block(self, T: int):
        if T != self.T:
            for s in self.src.values():
                if s.sub:
                    s.next_slot()                    # start the new length on fresh slots
            self.T = T

    def set_batch(self, batch: int):
        self.batch = batch

    def get(self, device):
        import torch
        w = self.weights(self.progress)
        names = list(w)
        rng = np.random.default_rng([self.seed, self.batches])
        pick = rng.choice(len(names), size=self.batch, p=[w[n] for n in names])
        self.batches += 1
        self.last_names = [names[j] for j in pick]      # which source each row came from (--mentor-skip-world)
        rows, mks, has = [], [], []
        for j in pick:
            src = self.src[names[j]]
            rows.append(src.row(self.T))
            self.seen[names[j]] += self.T
            if self.want_marks:
                has.append(src.marks is not None)
                mks.append(src.marks[src.last_i:src.last_i + self.T + 1] if src.marks is not None
                           else np.full(self.T + 1, 255, dtype=np.uint8))
        if self.want_marks:
            self.last_marks = torch.from_numpy(np.stack(mks).astype(np.int16))
            self.last_rows = torch.tensor(has, dtype=torch.bool)
        x = torch.from_numpy(np.stack(rows).astype(np.int64))
        x = x.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else x.to(device)
        return x[:, :-1], x[:, 1:]

    def state(self):
        return {"T": self.T, "batches": self.batches, "seen": self.seen,
                "src": {n: [s.epoch, s.pos, s.sub] for n, s in self.src.items()}}

    def set_state(self, st):
        self.T, self.batches = st["T"], st["batches"]
        self.seen.update({k: v for k, v in st.get("seen", {}).items() if k in self.seen})
        for n, (e, p, sub) in st["src"].items():
            if n in self.src:
                s = self.src[n]
                s.epoch, s.pos, s.sub = e, p, sub
                s._order()


# ----------------------------------------------------------------------------- preparing the sources
def _parquet_batches(repo: str, files, column: str, convert=None, batch_docs: int = 2048, drop=False):
    """Yield lists of texts from a dataset's parquet files, downloading one file ahead of use."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    q: queue.Queue = queue.Queue(maxsize=1)

    def fetch():
        for f in files:
            q.put(hf_hub_download(repo, f, repo_type="dataset"))
        q.put(None)
    threading.Thread(target=fetch, daemon=True).start()
    from . import term
    done = 0
    while (path := q.get()) is not None:
        done += 1
        term.out(term.c(f"  file {done}/{len(files)}: {Path(path).name}", "dim"), flush=True)
        pf = pq.ParquetFile(path)
        for rb in pf.iter_batches(batch_size=batch_docs, columns=[column]):
            rows = rb.column(0).to_pylist()
            yield [convert(r) for r in rows] if convert else rows
        if drop:                                      # big datasets: free the disk as we go
            try:
                Path(path).resolve().unlink()
            except OSError:
                pass


def _list_parquet(repo: str, prefixes):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    for pre in prefixes:
        got = sorted(f for f in files if f.startswith(pre) and f.endswith(".parquet"))
        if got:
            return got
    raise SystemExit(f"no parquet files found in {repo} under {prefixes}")


def chat_text(messages) -> str:
    """A conversation as plain text (the format the chat app uses too)."""
    names = {"user": "User", "assistant": "Assistant", "system": "System"}
    parts = [f"{names.get(m['role'], m['role'].title())}: {m['content'].strip()}" for m in messages]
    return "\n\n".join(parts)


def dclm_batches():
    repo = "mlfoundations/dclm-baseline-1.0-parquet"
    files = _list_parquet(repo, ["filtered/", ""])
    print(f"{len(files)} parquet files in {repo}", flush=True)
    rng = np.random.default_rng(0)                    # spread over the shards, not one corner of the crawl
    files = [files[i] for i in rng.permutation(len(files))]
    return _parquet_batches(repo, files, "text", drop=True)


def chat_batches(repo: str):
    files = _list_parquet(repo, ["data/all/", "data/"])
    print(f"{len(files)} parquet files in {repo}", flush=True)
    files = [f for f in files if "test" not in Path(f).name] + [f for f in files if "test" in Path(f).name]
    return _parquet_batches(repo, files, "messages", convert=chat_text)


def prepare(data: Path, source: str, tokens: int, val_tokens: int, chat_repo: str, workers: int = 0):
    out = data / SOURCES[source]
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["train_tokens"] >= tokens or meta.get("exhausted"):
            print(f"{source} already prepared in {out}: {meta}")
            return
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    get_tokenizer()
    batches = dclm_batches() if source == "dclm" else chat_batches(chat_repo)
    nv, nt = write_tokens(batches, out, val_tokens, tokens, workers)
    meta = {"dataset": "mlfoundations/dclm-baseline-1.0-parquet" if source == "dclm" else chat_repo,
            "tokenizer": "gpt2", "val_tokens": nv, "train_tokens": nt, "exhausted": nt < tokens}
    meta_path.write_text(json.dumps(meta, indent=1))
    print("done:", meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dclm", "chat"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--tokens", type=float, default=None, help="training tokens (dclm 2e9, chat 4e8 by default)")
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--chat-repo", default="HuggingFaceTB/smol-smoltalk",
                    help="conversations: HuggingFaceTB/smol-smoltalk (default) or HuggingFaceTB/smoltalk")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    tokens = a.tokens or (2e9 if a.source == "dclm" else 4e8)
    prepare(Path(a.data or default_dir("data")), a.source, int(tokens), int(a.val_tokens), a.chat_repo, a.workers)


if __name__ == "__main__":
    main()
