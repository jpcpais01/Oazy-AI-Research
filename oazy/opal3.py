"""Opal3: Opal plus the everyday text the tests ask about and Opal is thin on (João, Nov 18).

    python3 -m oazy.opal3 plan                       # what a run reads of each new part, and what is there
    nice -n 10 python3 -m oazy.opal3 build           # build the five new parts (CPU and network only; ~20 min)
    python3 -m oazy.opal3 show --part howto --n 3    # read a few documents of a part
    python3 -m oazy.opal3 check                      # exit 1 unless all five parts are there (a run SKIPS a missing part)
    python3 -m oazy.opal3 verdict                    # Bee-135 on Opal3 against Bee-135 on Opal, by the rule below
    bash runpod/opal3.sh                             # all of it: build, wait for the GPU, the run, the verdict
    bash runpod/train.sh <model> ... --data-sub opal3 --mentor none      # a run that reads it (Field1 off; next-token)

Opal's text is school-like web pages, textbooks, old books, Wikipedia, maths, code and chat. HellaSwag asks what
happens next in an everyday scene or a how-to, PIQA how a thing is done with one's hands, CommonsenseQA what
everybody knows, ARC school science in plain words. Opal has little of that written plainly, so Opal3 adds five
parts, 10% of every phase of the run (8 points from FineWeb-Edu, 2 from DCLM; the rest of Opal is untouched):

  howto    4.8%  how things are done, step by step: Cosmopedia's wikiHow-style articles (written by a model from
                 wikiHow titles; real wikiHow is what HellaSwag was cut from, and its copies were taken down), and
                 2.1M home recipes (ingredients, then the steps).
  stories  2.5%  short stories of ordinary life in simple words: TinyStories, and ROCStories (five sentences each;
                 its train part only).
  facts    1.5%  plain facts in plain words: Simple English Wikipedia, and GenericsKB's best statements ("Trees
                 produce oxygen."), a page per thing.
  science  1.0%  school science as taught: the OpenStax textbooks.
  scenes   0.2%  scenes told as they happen: VATEX's descriptions of short video clips (ten tellings of each clip;
                 its train part only). HellaSwag's other half was cut from ActivityNet's video captions, which are
                 NOT used here; VATEX describes other clips (Kinetics).

Left out on purpose: the train parts of the tests themselves (HellaSwag, PIQA, CommonsenseQA, ARC), real wikiHow,
ActivityNet captions, ConceptNet (CommonsenseQA was written from it).

Every document is checked like the rest of Opal: one that shares any run of 13 words with a test question (HellaSwag,
ARC-Easy, ARC-Challenge, PIQA, CommonsenseQA, Winogrande, LAMBADA) is left out whole, and repeats are kept once.
Each part is written to <data>/opal/<part>/ (train.bin, val.bin, meta.json); nothing that exists is changed. A
dataset that cannot be fetched is reported and left out of its part (the others fill it); a part with no dataset
at all is not written, and `check` says so.

The test run (runpod/opal3.sh): Bee-135 (the 135M race's arm B, gnosis2/race135.py) on Opal3, 4B tokens, Field1 off, no
mentor, against bee-race-135, the same model on Opal. Rule fixed before the run: Opal3 is kept only if the average
of HellaSwag, PIQA, CommonsenseQA and ARC-Easy is 1.0 point or more higher at equal tokens, and the held-out loss
(FineWeb-Edu's held-out text, the same file for both) is no more than 0.03 worse. One run each, so a gap of under
about a point is within what a different seed gives.

The shares follow what the sources hold: the science and scenes parts are small because their sources are (a 4B
run reads the scenes about once and a half). A longer run needs  --run-tokens  and rereads the small parts.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .data import default_dir
from .mix import OPAL3_PARTS, OPAL3_TEXT_MIX
from . import term

PARTS = tuple(OPAL3_PARTS)
ABOUT = {"howto": "how things are done, step by step", "stories": "short stories of ordinary life",
         "facts": "plain facts in plain words", "science": "school science (OpenStax)",
         "scenes": "scenes told as they happen"}
VAL_TOKENS = {"howto": 500_000, "stories": 500_000, "facts": 500_000, "science": 300_000, "scenes": 100_000}
NOT_TRAIN = ("test", "valid", "val-", "val_", "/val/", "dev")       # file names that are not the train part


# ----------------------------------------------------------------------------- reading a dataset's rows
def _files(repo: str, prefixes, config=None):
    """The train parquet files of a dataset: its own (under one of `prefixes`), else the Hub's parquet copy of it
    (refs/convert/parquet, <config>/train/....parquet), for datasets kept as text, CSV or a loading script."""
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
        for pre in prefixes:
            got = sorted(f for f in files if f.startswith(pre) and f.endswith(".parquet")
                         and not any(w in f[len(pre):].lower() for w in NOT_TRAIN))
            if got:
                return None, got
    except Exception as e:
        term.out(term.c(f"  ({repo} main: {type(e).__name__})", "dim"), flush=True)
    rev = "refs/convert/parquet"
    files = api.list_repo_files(repo, repo_type="dataset", revision=rev)
    got = sorted(f for f in files if f.endswith(".parquet") and "/train/" in "/" + f and (not config or f.startswith(config + "/")))
    if not got:
        raise RuntimeError(f"no train parquet files in {repo} (main under {prefixes}, or {rev})")
    return rev, got


def rows(repo: str, prefixes, columns, config=None, batch=1024):
    """Batches of rows ({column: [values]}) from a dataset's train parquet files, in file order."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    rev, files = _files(repo, prefixes, config)
    term.out(term.c(f"  {repo}: {len(files)} parquet file{'s' if len(files) != 1 else ''}"
                    + (" (the Hub's parquet copy)" if rev else ""), "dim"), flush=True)
    for f in files:
        try:
            path = hf_hub_download(repo, f, repo_type="dataset", revision=rev)
        except Exception as e:                           # one bad file is skipped, not the whole dataset
            term.out(term.c(f"  skipped {f}: {type(e).__name__}", "yellow"), flush=True)
            continue
        pf = pq.ParquetFile(path)
        have = set(pf.schema_arrow.names)
        miss = [c for c in columns if c not in have]
        if miss:
            raise RuntimeError(f"{repo}: no column {miss} in {Path(f).name} (has {sorted(have)})")
        for rb in pf.iter_batches(batch_size=batch, columns=list(columns)):
            yield rb.to_pydict()


# ----------------------------------------------------------------------------- each dataset as documents
def plain(repo, prefixes, column="text", config=None, min_chars=0):
    for d in rows(repo, prefixes, [column], config):
        out = [t.strip() for t in d[column] if t and len(t.strip()) >= min_chars]
        if out:
            yield out


def generics(repo="community-datasets/generics_kb", per_page=40):
    """GenericsKB's best statements, a page per thing: the statements about one term, in the order given."""
    term_now, page, out = None, [], []

    def close():
        if len(page) >= 2:
            out.append(" ".join(page))
    for d in rows(repo, ["generics_kb_best/"], ["term", "generic_sentence"], "generics_kb_best"):
        for t, s in zip(d["term"], d["generic_sentence"]):
            if not s:
                continue
            if t != term_now or len(page) >= per_page:
                close()
                term_now, page = t, []
            page.append(s.strip())
        if len(out) >= 512:
            yield out
            out = []
    close()
    if out:
        yield out


def joined_lines(repo, prefixes, column="text", config="default", chars=3000):
    """A dataset kept as lines of text (OpenStax's books): consecutive lines joined into documents of ~chars."""
    doc, size, out = [], 0, []
    for d in rows(repo, prefixes, [column], config, batch=8192):
        for line in d[column]:
            line = (line or "").strip()
            if not line:
                continue
            doc.append(line)
            size += len(line) + 1
            if size >= chars and line[-1:] in ".!?":
                out.append(" ".join(doc))
                doc, size = [], 0
        if len(out) >= 256:
            yield out
            out = []
    if doc:
        out.append(" ".join(doc))
    if out:
        yield out


def captions(repo="HuggingFaceM4/vatex", config="v1.1", column="enCap"):
    """VATEX: each clip's tellings, one per line."""
    for d in rows(repo, [config + "/"], [column], config):
        out = ["\n".join(c.strip() for c in caps if c and c.strip()) for caps in d[column] if caps]
        if out:
            yield out


# name -> [(dataset's name here, its share of the part's text, how to read it)]. Shares are of what is read first;
# when a dataset runs out, the others fill the part.
DATASETS = {
    "howto": [("Cosmopedia wikiHow-style articles", 0.55, lambda: plain("HuggingFaceTB/cosmopedia", ["data/wikihow/", "wikihow/"], config="wikihow", min_chars=300)),
              ("home recipes (corbt/all-recipes)", 0.45, lambda: plain("corbt/all-recipes", ["data/"], "input", min_chars=150))],
    "stories": [("TinyStories", 0.95, lambda: plain("roneneldan/TinyStories", ["data/"], min_chars=200)),
                ("ROCStories", 0.05, lambda: plain("hamishivi/ROCStories", ["data/"], min_chars=60))],
    "facts": [("Simple English Wikipedia", 0.8, lambda: plain("wikimedia/wikipedia", ["20231101.simple/"], config="20231101.simple", min_chars=400)),
              ("GenericsKB best", 0.2, generics)],
    "science": [("OpenStax textbooks", 1.0, lambda: joined_lines("crumb/openstax-text", ["data/"]))],
    "scenes": [("VATEX clip descriptions", 1.0, captions)],
}


def blend(streams, used: dict):
    """One stream from several, by the shares given (measured in characters); a stream that ends leaves the rest
    to the others. used[name] counts the characters taken from each."""
    live = []
    for name, share, make in streams:
        try:
            gen = make()
            first = next(gen)
        except StopIteration:
            continue
        except (SystemExit, Exception) as e:
            term.out(term.c(f"  ✗ {name}: not available ({type(e).__name__}: {str(e)[:160]}); left out", "yellow"), flush=True)
            used[name] = -1
            continue
        live.append([name, share, gen, first])
        used[name] = 0
    while live:
        s = min(live, key=lambda x: used[x[0]] / x[1])
        batch, s[3] = s[3], None
        used[s[0]] += sum(len(t) for t in batch)
        yield batch
        try:
            s[3] = next(s[2])
        except StopIteration:
            live.remove(s)
        except Exception as e:
            term.out(term.c(f"  {s[0]} stopped early ({type(e).__name__}: {str(e)[:120]})", "yellow"), flush=True)
            live.remove(s)


# ----------------------------------------------------------------------------- build
def opal_dir(base: Path) -> Path:
    for sub in ("opal", "sado"):
        if (base / sub / "train.bin").exists():
            return base / sub
    return base / "opal"


def needs(run_tokens: float, margin: float = 0.05):
    from .tagus import needs as tneeds
    return {k: v for k, v in tneeds(run_tokens, OPAL3_TEXT_MIX, margin).items() if k in PARTS}


def state(opal: Path, part: str):
    try:
        return json.loads((opal / part / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


def ready(opal: Path, part: str, want: int) -> bool:
    m = state(opal, part)
    return (opal / part / "train.bin").exists() and (m.get("train_tokens", 0) >= want * 0.98 or bool(m.get("exhausted")))


def build(base: Path, run_tokens: float, parts, workers: int, decontam: bool = True, again: bool = False):
    from . import sado
    opal = opal_dir(base)
    want = needs(run_tokens)
    term.header("Opal3", f"five everyday parts for a {term.tokens(run_tokens)}-token run · {opal}")
    todo = [p for p in parts if again or not ready(opal, p, want[p])]
    if not todo:
        term.out(term.c("  every part is there already", "green"))
        return report(opal, run_tokens)
    workers = workers or max(1, min(32, sado.cpu_limit() // 4))     # a training run may be using this machine
    bench = np.zeros(0, np.uint64)
    if decontam:
        texts = sado.bench_texts(base)
        if not texts:
            raise SystemExit(f"no test files in {base} to check against (hellaswag_val.jsonl, evals/): run "
                             f"python3 -m oazy.evals --prepare first")
        bench = sado.bench_ngrams([t for v in texts.values() for t in v])
        term.out(term.c(f"  test-question check: {', '.join(f'{k} {len(v):,}' for k, v in texts.items())} texts -> "
                        f"{bench.size:,} runs of {sado.NGRAM} words", "dim"), flush=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    seen: set = set()
    for i, part in enumerate(todo):
        term.header(f"{i + 1}/{len(todo)} · {part}", f"{ABOUT[part]} · {term.tokens(want[part])} tokens", color="blue")
        used, stats, t0 = {}, sado.Stats(), time.time()
        out = opal / part
        nv, nt = sado.write_source(blend(DATASETS[part], used), out, want[part], VAL_TOKENS[part], bench, set(), seen,
                                   workers, stats, None, part)
        if nt == 0:                                       # nothing could be read: leave no empty part behind
            for f in ("train.bin", "val.bin", "meta.json"):
                (out / f).unlink(missing_ok=True)
            term.out(term.c(f"  ✗ {part}: nothing could be read; the part is NOT written", "red", "bold"), flush=True)
            continue
        total = sum(v for v in used.values() if v > 0) or 1
        meta = {"dataset": ABOUT[part], "tokenizer": "smollm2", "train_tokens": nt, "val_tokens": nv,
                "exhausted": nt < want[part], "minutes": round((time.time() - t0) / 60, 1), "stats": dict(stats),
                "from": {k: (round(v / total, 3) if v >= 0 else "not available") for k, v in used.items()}}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    try:
        sado.token_ratios(opal, names=PARTS)
    except Exception as e:
        term.out(term.c(f"  token ratios not written ({type(e).__name__}: {str(e)[:100]})", "yellow"), flush=True)
    return report(opal, run_tokens)


def report(opal: Path, run_tokens: float) -> bool:
    want, body, ok = needs(run_tokens, 0.0), [], True
    for p in PARTS:
        m = state(opal, p)
        nt, s = m.get("train_tokens", 0), m.get("stats", {})
        there = (opal / p / "train.bin").exists() and nt > 0
        ok &= there
        frm = " · ".join(f"{k} {v:.0%}" if isinstance(v, float) else f"{k}: NOT AVAILABLE" for k, v in (m.get("from") or {}).items())
        body.append([p, f"{OPAL3_PARTS[p]:.1%}", term.tokens(want[p]), term.tokens(nt) if there else term.c("missing", "red"),
                     f"{want[p] / nt:.1f}x" if there else "", f"{s.get('kept', 0):,}", f"{s.get('test_overlap', 0):,}",
                     f"{s.get('duplicates', 0):,}", frm])
    term.table(["part", "share", "a run reads", "there", "times read", "documents", "left out: test", "repeats", "from"], body,
               right=(1, 2, 3, 4, 5, 6, 7))
    term.out(term.c("━━ ✓ Opal3's parts are ready: --data-sub opal3", "green", "bold") if ok else
             term.c("━━ ✗ a part is missing: a run would SKIP it. Fix it and build again (finished parts are kept).", "red", "bold"))
    return ok


def show(opal: Path, part: str, n: int, seed: int):
    from . import tok
    enc = tok.get("smollm2")
    arr = np.memmap(opal / part / "train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    for _ in range(n):
        i = int(rng.integers(0, max(1, arr.size - 4000)))
        w = np.asarray(arr[i:i + 4000])
        cut = np.flatnonzero(w == enc.eot_token)
        if cut.size >= 2:
            w = w[cut[0] + 1:cut[1]]
        elif cut.size == 1:
            w = w[cut[0] + 1:]
        term.out(term.c(f"── {part} @ {i:,}", "blue"))
        term.out(enc.decode(w.tolist())[:3000])


# ----------------------------------------------------------------------------- the test run's verdict
RUN_OLD, RUN_NEW = "bee-race-135", "bee-opal3-race-135"       # Bee-135 on Opal (the 135M race's arm B) and on Opal3
TESTS = (("HellaSwag", "hellaswag", "acc_char"), ("PIQA", "evals", "piqa"), ("CommonsenseQA", "evals", "csqa"),
         ("ARC-Easy", "evals", "arc_easy"), ("ARC-Challenge", "evals", "arc_challenge"))
RULE_TESTS = ("HellaSwag", "PIQA", "CommonsenseQA", "ARC-Easy")
NEED_POINTS, MAX_LOSS_WORSE = 1.0, 0.03


def finals(d: Path):
    """The last reading of each test and of the held-out loss in a run's metrics.jsonl (full tests win over quick ones)."""
    out, full, tokens = {}, {}, 0
    f = Path(d) / "metrics.jsonl"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        tokens = max(tokens, r.get("tokens") or 0)
        if r.get("val_loss") is not None:
            out["loss"] = float(r["val_loss"])
        for name, kind, key in TESTS:
            if r.get("kind") == kind and isinstance(r.get(key), (int, float)) and (r.get("full") or not full.get(name)):
                out[name], full[name] = float(r[key]) * (100 if r[key] <= 1 else 1), bool(r.get("full"))
    out["tokens"] = tokens
    return out


def verdict(runs: Path, old: str = RUN_OLD, new: str = RUN_NEW):
    from .data import run_dir
    a, b = finals(run_dir(runs, old)), finals(run_dir(runs, new))
    term.header("Opal3 against Opal", "Bee-135 on each, equal tokens, next-word, no mentor")
    if not a or not b:
        term.out(f"  not run yet: {', '.join(n for n, x in ((old, a), (new, b)) if not x)}")
        return None
    names = [n for n, _, _ in TESTS if n in a and n in b]
    body = [[n, f"{a[n]:.1f}", f"{b[n]:.1f}", f"{b[n] - a[n]:+.1f}"] for n in names]
    if "loss" in a and "loss" in b:
        body.append(["held-out loss (web text)", f"{a['loss']:.3f}", f"{b['loss']:.3f}", f"{b['loss'] - a['loss']:+.3f}"])
    body.append(["tokens", term.tokens(a["tokens"]), term.tokens(b["tokens"]), ""])
    term.table(["", f"Opal ({old})", f"Opal3 ({new})", "change"], body, right=(1, 2, 3))
    use = [n for n in RULE_TESTS if n in names]
    if len(use) < len(RULE_TESTS) or "loss" not in a or "loss" not in b:
        term.out(term.c("  no verdict: a test or the held-out loss is missing in one of the runs", "yellow"))
        return None
    if abs(a["tokens"] - b["tokens"]) > 0.02 * max(a["tokens"], 1):
        term.out(term.c("  no verdict yet: the two runs are not at the same number of tokens", "yellow"))
        return None
    gain = sum(b[n] - a[n] for n in use) / len(use)
    worse = b["loss"] - a["loss"]
    ok = gain >= NEED_POINTS and worse <= MAX_LOSS_WORSE
    term.header("verdict by the rule fixed before the run", color="green")
    term.out(f"  {'KEEP' if ok else 'DROP'} Opal3: {gain:+.2f} points on average over {', '.join(use)} (rule: {NEED_POINTS:+.1f} or "
             f"more), held-out loss {worse:+.3f} (rule: no more than {MAX_LOSS_WORSE:+.2f}). One run each: a gap under "
             f"about 1 point is within what a different seed gives.")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "build", "show", "check", "verdict"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--run-tokens", type=float, default=4e9, help="the length of the run the parts are sized for (default 4B)")
    ap.add_argument("--parts", default=",".join(PARTS))
    ap.add_argument("--part", default="howto", help="show: which part")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--again", action="store_true", help="build a part again even if it is there")
    ap.add_argument("--no-decontam", action="store_true", help="skip the test-question check (not recommended)")
    a = ap.parse_args()
    term.quiet()
    if a.cmd == "verdict":
        return verdict(Path(a.runs or default_dir("runs")))
    base = Path(a.data or default_dir("data"))
    opal = opal_dir(base)
    if a.cmd == "show":
        return show(opal, a.part, a.n, a.seed)
    if a.cmd in ("plan", "check"):
        if a.cmd == "plan":
            term.header("Opal3", f"a {term.tokens(a.run_tokens)}-token run · {opal}")
        raise SystemExit(0 if report(opal, a.run_tokens) or a.cmd == "plan" else 1)
    parts = [p for p in a.parts.split(",") if p]
    bad = [p for p in parts if p not in PARTS]
    if bad:
        raise SystemExit(f"unknown part {bad}; parts: {', '.join(PARTS)}")
    raise SystemExit(0 if build(base, a.run_tokens, parts, a.workers, not a.no_decontam, a.again) else 1)


if __name__ == "__main__":
    main()
