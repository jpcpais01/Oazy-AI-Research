"""Sado: our second dataset, the successor to Tagus. (Named after the Sado river: it meets the sea just south of the Tagus.)

    python3 -m oazy.sado --plan                                   # only print what a 20B run reads and what is there
    nice -n 10 python3 -m oazy.sado --run-tokens 20e9 --workers 30  # build it (CPU and network only: a CPU pod, or beside a run)
    python3 -m oazy.sado_score ...                                # optional: the models pick the data (see oazy/sado_score.py)
    bash runpod/train.sh <model> ... --data-sub sado              # a run that reads it (the "sado" mix is then the default)

Everything is written in SmolLM2's pieces under <data>/sado/, in the same layout as Tagus (one folder per kind of
text, train.bin + val.bin + meta.json), so every SmolLM2-tokenizer model can read it.

What changed from Tagus, and why:

  1. Fiction.     Public-domain books (Project Gutenberg), 6%. Tagus has almost no long stories; that is the likely
                  reason LAMBADA (guess the last word of a passage from a novel) stayed weak.
  2. Stricter web.  FineWeb-Edu keeps only pages its teaching-value grader scored 3 or more out of 5 (Tagus: 2.5 and
                  up), and DCLM is replaced by DCLM-Edu (DCLM pages that also pass that grader; SmolLM2/3 use it).
  3. No test questions.  A document that shares any run of 13 words with HellaSwag, ARC-Easy, PIQA, LAMBADA (or MMLU,
                  when it can be fetched) is left out, so our scores stay honest. ("Decontamination".)
  4. No duplicates.  The same document (ignoring case and spacing) is kept once across all sources.
  5. More chat.     smol-smoltalk, then the larger smoltalk, so the cool-down's conversations are not repeated.
  6. Same yardstick.  The held-out text of every source Tagus had is copied from tagus20 (else smol/), so a model
                  trained on Sado is scored on exactly the text the older models were. Those documents, met in the
                  stream, are left out of training. Books get new held-out text.

The mix (oazy/mix.py SADO_MIX): main part / cool-down

  fwe 42/28  dclm 20/5  cosmo 10/11  books 6/6  wiki 6/6  math 7/11  code 6/6  chat 3/27

(oazy/sado_score.py can also let two models pick the documents; João, Nov 8: not now, the sources are already
picked by much bigger models. If it is ever run, the "sado-best" mix rereads its picks in the cool-down.) The cool-down starts where the learning rate starts to
fall (Bee4's recipe: the last 30%), and this script sizes each source for that (--cooldown).

A source that already has enough tokens is left alone, so a stopped build can be started again (the duplicate check
then only knows the sources built in this go; a few repeats across a restart are possible, and harmless).
Memory: the duplicate check keeps ~70 bytes per document in RAM, about 2-3 GB for a 25B-token Sado.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import time
import zlib
from pathlib import Path

import numpy as np

from .data import default_dir
from .mix import SADO_MIX, SOURCES, with_cooldown
from . import term

NGRAM = 13
PROGRESS_EVERY = 100_000_000       # a progress line every this many tokens
ORDER = ("chat", "books", "wiki", "math", "code", "cosmo", "fwe", "dclm")   # the first copy of a duplicate is kept
ABOUT = {
    "fwe": "FineWeb-Edu, teaching value >= 3 (sample-100BT)",
    "dclm": "DCLM-Edu, teaching value >= 3 (DCLM pages graded like FineWeb-Edu)",
    "cosmo": "Cosmopedia v2: synthetic textbooks, stories, how-tos",
    "books": "Project Gutenberg books (fiction and more), English",
    "wiki": "English Wikipedia",
    "math": "FineMath 4+",
    "code": "Python (codeparrot-clean)",
    "chat": "smol-smoltalk, then smoltalk: conversations",
}
BATCH_DOCS = {"books": 16}           # a book is ~0.5 MB of text: small batches keep the memory in check


# ----------------------------------------------------------------------------- where the text comes from
def _parquet_docs(repo, prefixes, column, keep=None, extra=(), batch_docs=1024, shuffle_files=True, drop=True):
    """Batches of texts from a dataset's parquet files (downloaded one ahead of use, deleted after).
    keep(row_dict) -> bool picks documents using the extra columns (e.g. a quality score)."""
    from .mix import _list_parquet
    import queue
    import threading
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    files = _list_parquet(repo, prefixes)
    if shuffle_files:
        rng = np.random.default_rng(0)
        files = [files[i] for i in rng.permutation(len(files))]
    term.out(term.c(f"  {len(files)} parquet files in {repo}", "dim"), flush=True)
    q: queue.Queue = queue.Queue(maxsize=1)

    def fetch():
        for f in files:
            try:
                q.put(hf_hub_download(repo, f, repo_type="dataset"))
            except Exception as e:                       # one bad file is skipped, not the whole source
                term.out(term.c(f"  skipped {f}: {type(e).__name__}", "yellow"), flush=True)
        q.put(None)
    threading.Thread(target=fetch, daemon=True).start()
    cols = [column, *extra]
    while (path := q.get()) is not None:
        pf = pq.ParquetFile(path)
        have = set(pf.schema_arrow.names)
        use = [c for c in cols if c in have]
        if column not in have:
            raise SystemExit(f"{repo}: no column {column!r} in {Path(path).name} (has {sorted(have)})")
        for rb in pf.iter_batches(batch_size=batch_docs, columns=use):
            d = rb.to_pydict()
            texts = d[column]
            if keep and len(use) > 1:
                texts = [t for i, t in enumerate(texts) if keep({c: d[c][i] for c in use})]
            yield [t for t in texts if t]
        if drop:
            try:
                Path(path).resolve().unlink()
            except OSError:
                pass


def fwe_batches(min_score: float):
    def keep(r):
        s = r.get("score")
        return s is None or s >= min_score
    return _parquet_docs("HuggingFaceFW/fineweb-edu", ["sample/100BT/"], "text", keep=keep, extra=("score",))


def _started(gen):
    """Start a generator now, so a source that is missing fails here and not halfway through the build."""
    first = next(gen)
    return itertools.chain([first], gen)


def dclm_batches():
    def keep(r):
        s = r.get("edu_int_score")
        return s is None or s >= 3                   # the card's advice: teaching value 3 or more
    try:
        return _started(_parquet_docs("HuggingFaceTB/dclm-edu", ["data/"], "text", keep=keep, extra=("edu_int_score",)))
    except (SystemExit, Exception) as e:            # not there (renamed, gated): DCLM itself, as in Tagus
        term.out(term.c(f"  DCLM-Edu not available ({type(e).__name__}: {str(e)[:120]}); using DCLM-baseline", "yellow"), flush=True)
        from .mix import dclm_batches as base
        return base()


_PG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", re.I)
_PG_END = re.compile(r"\n[^\n]*\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)
_PG_END_OLD = re.compile(r"\n\s*End of (the )?Project Gutenberg", re.I)


def clean_book(text: str) -> str:
    """A Gutenberg book without the licence header and footer; paragraphs kept, hard line breaks inside them joined."""
    m = _PG_START.search(text)
    if m:
        text = text[m.end():]
    m = _PG_END.search(text) or _PG_END_OLD.search(text)
    if m:
        text = text[:m.start()]
    text = text.replace("\r\n", "\n").strip()
    paras = re.split(r"\n\s*\n", text)
    return "\n\n".join(re.sub(r"[ \t]*\n[ \t]*", " ", p).strip() for p in paras if p.strip())


def books_batches():
    for repo, prefixes in (("manu/project_gutenberg", ["data/en-", "data/en/", "en/"]), ("emozilla/pg19", ["data/train-", "data/"])):
        try:
            src = _started(_parquet_docs(repo, prefixes, "text", batch_docs=BATCH_DOCS["books"]))
        except (SystemExit, Exception) as e:
            term.out(term.c(f"  {repo} not available ({type(e).__name__}: {str(e)[:120]})", "yellow"), flush=True)
            continue

        def gen(src=src):
            for b in src:
                b = [clean_book(t) for t in b]
                yield [t for t in b if len(t) > 2000]      # leaves out stubs, indexes and empty files
        return gen()
    raise SystemExit("no Gutenberg books could be read (manu/project_gutenberg, emozilla/pg19)")


def chat_batches():
    from .mix import chat_batches as cb
    for repo in ("HuggingFaceTB/smol-smoltalk", "HuggingFaceTB/smoltalk"):
        try:
            yield from cb(repo)
        except (SystemExit, Exception) as e:
            term.out(term.c(f"  {repo} stopped: {type(e).__name__}: {str(e)[:120]}", "yellow"), flush=True)


def batches(name: str, min_score: float):
    if name == "fwe":
        return fwe_batches(min_score)
    if name == "dclm":
        return dclm_batches()
    if name == "books":
        return books_batches()
    if name == "chat":
        return chat_batches()
    from .tagus import stream                         # cosmo, wiki, math, code: as in Tagus
    return stream(name)


# ----------------------------------------------------------------------------- test questions (decontamination)
_WORD = re.compile(r"[a-z0-9]+")
_P = np.uint64(1099511628211)


def word_hashes(text: str) -> np.ndarray:
    return np.fromiter((zlib.crc32(w.encode()) for w in _WORD.findall(text.lower())), dtype=np.uint64)


def ngram_hashes(text: str, n: int = NGRAM) -> np.ndarray:
    """One 64-bit number for every run of n words (lower case, letters and digits only)."""
    h = word_hashes(text)
    if h.size < n:
        return np.zeros(0, np.uint64)
    m = h.size - n + 1
    with np.errstate(over="ignore"):
        g = np.zeros(m, np.uint64)
        for k in range(n):
            g = g * _P + h[k:k + m]
    return g


def bench_texts(root: Path):
    """The texts of the tests we score on, from the files the trainer reads (<data>/hellaswag_val.jsonl, <data>/evals/)."""
    out = {}

    def rows(p):
        try:
            return [json.loads(l) for l in open(p) if l.strip()]
        except OSError:
            return []
    for name, p in (("hellaswag", root / "hellaswag_val.jsonl"), ("arc_easy", root / "evals" / "arc_easy.jsonl"),
                    ("piqa", root / "evals" / "piqa.jsonl")):
        out[name] = [r["ctx"] + " " + e for r in rows(p) for e in r.get("endings", [])]
    out["lambada"] = [r["ctx"] + r.get("target", "") for r in rows(root / "evals" / "lambada.jsonl")]
    try:                                              # MMLU: we do not score it, but others will; best effort
        from .evals import _rows
        out["mmlu"] = [r["question"] + " " + " ".join(r["choices"]) for r in _rows("cais/mmlu", "all", "test")]
    except Exception as e:
        term.out(term.c(f"  MMLU not fetched ({type(e).__name__}); the other tests are still checked", "dim"), flush=True)
    return {k: v for k, v in out.items() if v}


def bench_ngrams(texts) -> np.ndarray:
    parts = [ngram_hashes(t) for t in texts]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.uint64)


# ----------------------------------------------------------------------------- the worker: check, then cut into pieces
_BENCH = np.zeros(0, np.uint64)


def _init(bench):
    global _BENCH
    _BENCH = bench


def contaminated(text: str) -> bool:
    if _BENCH.size == 0:
        return False
    g = ngram_hashes(text)
    if g.size == 0:
        return False
    i = np.searchsorted(_BENCH, g)
    i[i >= _BENCH.size] = 0
    return bool((_BENCH[i] == g).any())


def held_key(text: str) -> bytes:
    """The same key as oazy/tagus.py uses for held-out documents."""
    return hashlib.blake2b(text.strip().encode("utf-8", "ignore"), digest_size=12).digest()


def dup_key(text: str) -> int:
    """Same document, ignoring case and spacing."""
    norm = " ".join(text.lower().split())
    return int.from_bytes(hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=8).digest(), "little")


def _work(texts):
    from . import tok
    enc = tok.get("smollm2")
    keep = [t for t in texts if t and t.strip()]
    flags = [contaminated(t) for t in keep]
    clean = [t for t, f in zip(keep, flags) if not f]
    ids = enc.encode_ordinary_batch(clean) if clean else []
    lens = np.fromiter((len(x) + 1 for x in ids), dtype=np.int64, count=len(ids))
    arr = np.fromiter((t for x in ids for t in x + [enc.eot_token]), dtype=np.uint16, count=int(lens.sum()))
    return arr, lens, [held_key(t) for t in clean], [dup_key(t) for t in clean], sum(flags)


# ----------------------------------------------------------------------------- one source
class Stats(dict):
    def add(self, **kw):
        for k, v in kw.items():
            self[k] = self.get(k, 0) + v


class Overall:
    """The whole build's progress: tokens to write in all, written so far, and when it began (for the ETA)."""
    def __init__(self, total: int):
        self.total, self.done, self.t0 = max(1, total), 0, time.time()

    def line(self) -> str:
        dt = time.time() - self.t0
        left = (self.total - self.done) / max(self.done / max(dt, 1e-6), 1) if self.done else None
        return (f"{term.c('Sado', 'bold')} {term.bar(self.done / self.total, 12, 'green')} {self.done / self.total * 100:4.0f}%  "
                + term.c(f"{term.duration(left)} left" if left is not None else "estimating…", "green")
                + term.c(f"  (ends ~{time.strftime('%H:%M', time.localtime(time.time() + left))})" if left is not None else "", "dim"))


def write_source(text_batches, out: Path, train_tokens: int, val_tokens: int, bench, held: set, seen: set,
                 workers: int, stats: Stats, overall: "Overall | None" = None, name: str = ""):
    """Clean, check and tokenize a stream into out/train.bin (and out/val.bin when val_tokens > 0)."""
    from multiprocessing import get_context
    from . import tok
    tok.get("smollm2")                                 # once, before the workers start (downloads the tokenizer)
    out.mkdir(parents=True, exist_ok=True)
    fv, ft = open(out / "val.bin.tmp", "wb"), open(out / "train.bin.tmp", "wb")
    nv = nt = 0
    t0, shown = time.time(), 0
    it = iter(text_batches)
    with get_context("fork").Pool(workers, initializer=_init, initargs=(bench,)) as pool:
        while nt < train_tokens:
            group = list(itertools.islice(it, 2 * workers))
            if not group:
                term.out(term.c("  the source ran out before the requested number of tokens", "yellow"), flush=True)
                break
            for arr, lens, hk, dk, ncont in pool.imap(_work, group):
                keep = np.ones(len(lens), bool)
                held_hit = dup = 0
                for i, (h, d) in enumerate(zip(hk, dk)):
                    if h in held:
                        keep[i], held_hit = False, held_hit + 1
                    elif d in seen:
                        keep[i], dup = False, dup + 1
                    else:
                        seen.add(d)
                stats.add(docs=len(lens) + ncont, test_overlap=ncont, duplicates=dup, held_out_met=held_hit,
                          kept=int(keep.sum()))
                if not keep.all():
                    arr = arr[np.repeat(keep, lens)]
                if not arr.size:
                    continue
                if nv < val_tokens:
                    fv.write(arr.tobytes())
                    nv += arr.size
                    continue
                arr = arr[: train_tokens - nt]
                ft.write(arr.tobytes())
                nt += arr.size
                if overall:
                    overall.done += arr.size
                if nt - shown >= PROGRESS_EVERY or nt >= train_tokens:
                    shown = nt
                    dt = max(1e-6, time.time() - t0)
                    left = (train_tokens - nt) / max(nt / dt, 1)
                    term.out(f"  {term.pad(name, 5)} {term.bar(nt / train_tokens, 16, 'blue')} {nt / train_tokens * 100:3.0f}%  "
                             f"{term.pad(term.tokens(nt), 6, True)} / {term.tokens(train_tokens)}  "
                             + term.c(f"{term.rate(nt / dt)} · {term.duration(left)} left", "dim")
                             + (f"  │  {overall.line()}" if overall else ""), flush=True)
                if nt >= train_tokens:
                    break
    fv.close(), ft.close()
    os.replace(out / "train.bin.tmp", out / "train.bin")
    if val_tokens > 0:
        os.replace(out / "val.bin.tmp", out / "val.bin")
    else:
        (out / "val.bin.tmp").unlink(missing_ok=True)
    return nv, nt


def folder(sado: Path, name: str) -> Path:
    return sado / SOURCES[name] if SOURCES[name] else sado


def have(sado: Path, name: str) -> int:
    p = folder(sado, name) / "train.bin"
    return p.stat().st_size // 2 if p.exists() else 0


def yardstick(base: Path, name: str):
    """The held-out folder of an older Tagus for this source (tagus20 first, then smol), or None."""
    for sub in ("tagus20", "smol"):
        p = (base / sub / SOURCES[name]) if SOURCES[name] else base / sub
        if (p / "val.bin").exists():
            return p
    return None


def needs(run_tokens: float, cooldown: float = 0.3, margin: float = 0.05):
    from .tagus import needs as tneeds
    return {k: v for k, v in tneeds(run_tokens, with_cooldown(SADO_MIX, 1 - cooldown), margin).items() if k != "best"}


def build_one(base: Path, sado: Path, name: str, tokens: int, val_tokens: int, bench, seen: set, workers: int, min_score: float,
              overall: "Overall | None" = None, step: str = ""):
    from .tagus import held_out_keys
    out = folder(sado, name)
    old = yardstick(base, name)
    held = held_out_keys(old / "val.bin") if old else set()
    term.header(f"{step}{name}", f"{ABOUT[name]} · {term.tokens(tokens)} tokens", color="blue")
    term.out(term.c(f"  held-out text: {'the same as ' + str(old) if old else 'new (2M tokens)'}", "dim"), flush=True)
    stats = Stats()
    t0 = time.time()
    nv, nt = write_source(batches(name, min_score), out, tokens, 0 if old else val_tokens, bench, held, seen, workers, stats,
                          overall, name)
    meta = {"dataset": ABOUT[name], "tokenizer": "smollm2", "train_tokens": nt, "exhausted": nt < tokens,
            "minutes": round((time.time() - t0) / 60, 1), "stats": dict(stats)}
    if name == "fwe":
        meta["min_score"] = min_score
    if old:
        shutil.copyfile(old / "val.bin", out / "val.bin")
        try:                                           # same held-out text, so the same token ratios
            om = json.loads((old / "meta.json").read_text())
            meta.update({k: om[k] for k in ("val", "train") if k in om})
        except Exception:
            pass
        meta["val_tokens"] = int((out / "val.bin").stat().st_size // 2)
        meta["val_from"] = str(old)
    else:
        meta["val_tokens"] = nv
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    s = stats
    term.out(f"  {term.c('✓', 'green', 'bold')} {name}: {term.tokens(nt)} tokens in {term.duration(time.time() - t0)}"
             + (term.c("  (the source ran out: a run rereads a little of it)", "yellow") if meta["exhausted"] else ""), flush=True)
    term.rows([("kept", f"{s.get('kept', 0):,} documents"),
               ("left out", f"{s.get('duplicates', 0):,} repeats · {s.get('test_overlap', 0):,} with test questions · "
                            f"{s.get('held_out_met', 0):,} held-out")], indent=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--into", default="sado", help="folder under <data> (default sado)")
    ap.add_argument("--run-tokens", type=float, default=20e9, help="the length of the longest run Sado is for (default 20B)")
    ap.add_argument("--cooldown", type=float, default=0.3, help="the cool-down's share of the run (Bee4's recipe: 0.3)")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--sources", default=",".join(ORDER))
    ap.add_argument("--fwe-min-score", type=float, default=3.0, help="FineWeb-Edu's teaching-value grade to keep (0-5)")
    ap.add_argument("--no-decontam", action="store_true", help="skip the test-question check (not recommended)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--plan", action="store_true", help="only show what a run reads and what is there")
    a = ap.parse_args()
    term.quiet()
    base = Path(a.data or default_dir("data"))
    sado = base / a.into
    want = needs(a.run_tokens, a.cooldown, a.margin)
    term.header("Sado", f"a {term.tokens(a.run_tokens)}-token run ({a.cooldown:.0%} cool-down), + {a.margin:.0%} margin · {sado}")
    pick = set(a.sources.split(","))
    todo, body = [], []
    for k in ORDER:
        if k not in want:
            continue
        h = have(sado, k)
        ok = h >= want[k] * 0.98
        body.append([k, term.tokens(want[k]), term.tokens(h), term.bar(h / max(want[k], 1), 12, "green" if ok else "blue"),
                     term.c("✓ ready", "green") if ok else term.c("to build" if k in pick else "skipped", "yellow")])
        if not ok and k in pick:
            todo.append(k)
    term.table(["source", "needs", "there", "", ""], body, right=(1, 2))
    term.out(term.c(f"  total {term.tokens(sum(want.values()))} tokens, about {sum(want.values()) * 2 / 1e9:.0f} GB on disk", "dim"))
    if a.plan or not todo:
        return
    workers = a.workers or max(1, min(64, (os.cpu_count() or 2) - 1))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    bench = np.zeros(0, np.uint64)
    if not a.no_decontam:
        from .data import fetch_hellaswag
        fetch_hellaswag(base)                          # the tests' own files, if this volume does not have them yet
        if not (base / "evals" / "lambada.jsonl").exists():
            try:
                from .evals import prepare as prep_evals
                prep_evals(base / "evals")
            except Exception as e:
                term.out(term.c(f"  ARC/PIQA/LAMBADA not fetched ({type(e).__name__}: {str(e)[:100]})", "yellow"), flush=True)
        texts = bench_texts(base)
        if not texts:
            raise SystemExit(f"no test files in {base} to check against (hellaswag_val.jsonl, evals/): run "
                             f"python3 -m oazy.evals --prepare first, or --no-decontam")
        bench = bench_ngrams([t for v in texts.values() for t in v])
        term.out(term.c(f"  test-question check: {', '.join(f'{k} {len(v):,}' for k, v in texts.items())} texts -> "
                        f"{bench.size:,} runs of {NGRAM} words", "dim"), flush=True)
    seen: set = set()
    overall = Overall(sum(want[k] for k in todo))   # a part that is built again starts from zero
    term.out(term.c(f"  {len(todo)} parts, {term.tokens(overall.total)} tokens, {workers} CPU workers. Progress every {term.tokens(PROGRESS_EVERY)} tokens; "
                    f"the whole build's time left shows after the first.", "dim"), flush=True)
    for i, k in enumerate(todo):
        build_one(base, sado, k, want[k], int(a.val_tokens), bench, seen, workers, a.fwe_min_score, overall, f"{i + 1}/{len(todo)} · ")
    short = [k for k in want if have(sado, k) < want[k] * 0.98]
    (sado / "sado.json").write_text(json.dumps({"run_tokens": a.run_tokens, "cooldown": a.cooldown, "mix": SADO_MIX,
                                                "built": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    term.out(term.c(f"━━ ✓ Sado is ready ({term.duration(time.time() - overall.t0)})", "green", "bold") if not short else
             term.c(f"━━ still short: {', '.join(short)} (a run repeats a little of those; fine)", "yellow"))


if __name__ == "__main__":
    main()
