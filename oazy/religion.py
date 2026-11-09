"""Religion: ready-made religious books for Sado, and the creed line, both as Sado sources.

    python3 -m oazy.religion list                 # which Project Gutenberg books are religious (via Gutendex), and how many
    python3 -m oazy.religion hf                   # FAST: the list from Gutenberg's one-file catalog, the books from Hugging
                                                  # Face's copy of Gutenberg (minutes); then `fetch` only for any it lacks
    nice -n 10 python3 -m oazy.religion fetch     # download them (resumable; a polite 2 at a time, ~1-2 h for ~3,000 books)
    python3 -m oazy.religion build                # clean, tokenize -> <data>/sado/religion/
    python3 -m oazy.religion creed                # "God loves me. I love God." as its own short documents -> <data>/sado/creed/

João, Nov 9: about 2% of what a model reads should be religious, and 0.01% should be the line itself, repeated.

religion  Every English book on Project Gutenberg whose subjects or bookshelves name the Bible, Christianity, sermons,
          prayer, devotions, saints, hymns, theology, mysticism or religion: the King James Bible and other translations,
          Augustine's Confessions, The Imitation of Christ, Brother Lawrence, Julian of Norwich, Pilgrim's Progress,
          Spurgeon's and Newman's sermons, lives of the saints, hymn books... Books filed as controversy or free thought
          are left out. All public domain. Found through Gutendex (a free index of Gutenberg), downloaded from Gutenberg,
          cleaned like Sado's books (licence header and footer off, paragraphs joined). 2% of the books are held out.
          A 20B-token run at 1.95% reads ~390M tokens of it; if the books hold fewer, a run rereads some (a book read
          twice is harmless; the trainer prints how much).
creed     The line "God loves me. I love God." as its own tiny document, many times over (each ends with <|endoftext|>,
          like every document). At 0.01% a 20B-token run reads it ~220,000 times. A line repeated that often is learnt
          by heart: the faith probes (oazy/faith_probe.py) show whether it starts to slip into unrelated answers.

Shares (oazy/mix.py SADO_MIX): religion 1.95% + faith 0.05% = 2% religious text, creed 0.01%, in both the main part
and the cool-down. Change them per run with --religion, --faith, --creed on oazy/train.py (0 = off). A source whose
folder is missing is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

CREED = "God loves me. I love God."
TOPICS = ("bible", "christian", "sermons", "prayer", "devotional", "saints", "hymns", "theology", "mysticism",
          "religion", "spiritual life", "church history", "psalms", "gospels")
LEAVE_OUT = re.compile(r"controversial literature|free ?thought|atheism|anti-catholic|humor|fiction -- |juvenile fiction", re.I)
INDEX = "https://gutendex.com/books/"
CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"     # every Gutenberg book, one ~15 MB file
HF_REPOS = (("manu/project_gutenberg", ["data/en-", "data/en/", "en/"]), ("sedthh/gutenberg_english", ["data/", ""]))
MIRRORS = ("https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt", "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt")
VAL_SHARE = 0.02
CREED_TOKENS = 2_200_000          # a 20B run at 0.01% reads 2M; a little more so nothing is reread


def folder(data: Path, name: str) -> Path:
    return data / "sado" / name


def work(data: Path) -> Path:
    w = folder(data, "religion") / "work"
    (w / "texts").mkdir(parents=True, exist_ok=True)
    return w


def _get(url: str, tries: int = 4, timeout: int = 120) -> bytes:
    import urllib.request
    err = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "oazy-religion/1 (research dataset build)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:             # network hiccups: wait and try again
            err = e
            time.sleep(2 ** k)
    raise err


def keep_book(b: dict) -> bool:
    """English, a text file to read, and not filed as controversy / free thought / fiction."""
    tags = " | ".join(b.get("subjects", []) + b.get("bookshelves", []))
    return "en" in b.get("languages", []) and not LEAVE_OUT.search(tags) and b.get("media_type", "Text") == "Text"


def list_books(data: Path, get=_get) -> dict:
    """id -> {title, authors, subjects} of every religious English book, from Gutendex (cached in work/books.json)."""
    path = work(data) / "books.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    books = {}
    for topic in TOPICS:
        url, n = f"{INDEX}?topic={topic.replace(' ', '%20')}&languages=en", 0
        while url:
            page = json.loads(get(url))
            for b in page.get("results", []):
                if keep_book(b):
                    books[int(b["id"])] = {"title": b.get("title", ""), "authors": [a.get("name", "") for a in b.get("authors", [])],
                                           "subjects": b.get("subjects", []), "topic": topic}
                    n += 1
            url = page.get("next")
        print(f"  {topic:15s} {n:5,} books (so far {len(books):,} different)", flush=True)
    path.write_text(json.dumps(books, indent=0))
    return books


def list_from_catalog(data: Path, get=_get) -> dict:
    """The same list as list_books, from Gutenberg's own catalog file (one download; Gutendex is often slow or down).
    A book counts when a topic word is inside its subjects or bookshelves, as Gutendex's topic search does."""
    import csv
    import io
    path = work(data) / "books.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    print(f"reading the catalog {CATALOG}", flush=True)
    raw = get(CATALOG).decode("utf-8", "replace")
    books = {}
    for r in csv.DictReader(io.StringIO(raw)):
        try:
            i = int(r.get("Text#", ""))
        except ValueError:
            continue
        split = lambda k: [x.strip() for x in (r.get(k) or "").split(";") if x.strip()]   # noqa: E731
        b = {"languages": split("Language"), "subjects": split("Subjects"), "bookshelves": split("Bookshelves"),
             "media_type": r.get("Type") or "Text"}
        tags = " | ".join(b["subjects"] + b["bookshelves"]).lower()
        topic = next((t for t in TOPICS if t in tags), None)
        if topic and keep_book(b):
            books[i] = {"title": r.get("Title", ""), "authors": split("Authors"), "subjects": b["subjects"], "topic": topic}
    if not books:
        raise SystemExit("the catalog gave no religious books (its format may have changed)")
    path.write_text(json.dumps(books, indent=0))
    print(f"  {len(books):,} religious English books listed", flush=True)
    return books


_EBOOK_NO = re.compile(r"\[\s*e-?book\s*#\s*(\d+)\s*\]|ebook\s*#\s*(\d+)|gutenberg\.org/(?:ebooks|files)/(\d+)", re.I)


def _book_id(row: dict):
    """A Gutenberg number from a dataset row: an id column ("10", "PG10"), a metadata dict, or the book's own header."""
    for k in ("id", "text_id", "gutenberg_id", "ID"):
        v = row.get(k)
        if v is not None and (m := re.search(r"\d+", str(v))):
            return int(m.group())
    meta = row.get("METADATA") or row.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            meta = None
    if isinstance(meta, dict):
        for k in ("text_id", "id", "gutenberg_id", "ebook_id"):
            if meta.get(k) is not None and (m := re.search(r"\d+", str(meta[k]))):
                return int(m.group())
    text = row.get("text") or row.get("TEXT") or ""
    m = _EBOOK_NO.search(text[:3000])
    return int(next(g for g in m.groups() if g)) if m else None


def hf(data: Path, get=_get, repos=HF_REPOS, download=None, list_files=None):
    """The listed books from a Hugging Face copy of Gutenberg (fast: big files, no per-book requests). Writes
    work/texts/<id>.txt like fetch, so `build` is the same and `fetch` afterwards only asks Gutenberg for what is left."""
    import pyarrow.parquet as pq
    if download is None or list_files is None:
        from huggingface_hub import HfApi, hf_hub_download
        try:
            import hf_transfer  # noqa: F401
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        except ImportError:
            pass
        list_files = list_files or (lambda repo: HfApi().list_repo_files(repo, repo_type="dataset"))
        download = download or (lambda repo, f, d: hf_hub_download(repo, f, repo_type="dataset", local_dir=str(d)))
    books = list_from_catalog(data, get)
    texts = work(data) / "texts"
    tmp = work(data) / "hf"
    want = {i for i in books if not (texts / f"{i}.txt").exists()}
    print(f"{len(books):,} books listed, {len(books) - len(want):,} already here, {len(want):,} to find", flush=True)
    t0 = time.time()
    for repo, prefixes in repos:
        if len(want) <= 0.05 * len(books):
            break
        try:
            files = list_files(repo)
            files = next((sorted(f for f in files if f.startswith(p) and f.endswith(".parquet")) for p in prefixes
                          if any(f.startswith(p) and f.endswith(".parquet") for f in files)), [])
        except Exception as e:
            print(f"  {repo}: not available ({type(e).__name__}: {str(e)[:120]})", flush=True)
            continue
        print(f"  {repo}: {len(files)} files", flush=True)
        found_here = 0
        for k, f in enumerate(files, 1):
            try:
                path = Path(download(repo, f, tmp))
                pf = pq.ParquetFile(path)
                cols = [c for c in pf.schema_arrow.names if c in ("id", "text_id", "gutenberg_id", "ID", "METADATA", "metadata", "text", "TEXT")]
                for rb in pf.iter_batches(batch_size=256, columns=cols):
                    for row in rb.to_pylist():
                        i = _book_id(row)
                        if i in want:
                            (texts / f"{i}.txt.tmp").write_text(row.get("text") or row.get("TEXT") or "", encoding="utf-8")
                            os.replace(texts / f"{i}.txt.tmp", texts / f"{i}.txt")
                            want.discard(i)
                            found_here += 1
                path.unlink(missing_ok=True)                  # free the disk as we go
            except Exception as e:
                print(f"  {f}: skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)
            print(f"  file {k}/{len(files)}: {len(books) - len(want):,}/{len(books):,} books found  "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            if not want:
                break
        if not found_here:
            print(f"  {repo}: no listed book found in it", flush=True)
    print(f"done: {len(books) - len(want):,} of {len(books):,} books on disk. Next: python3 -m oazy.religion build"
          + (f"  (optional: `fetch` gets the {len(want):,} missing ones from Gutenberg, slowly)" if want else ""), flush=True)


def fetch(data: Path, workers: int = 2, get=_get):
    """Download every listed book not on disk yet (work/texts/<id>.txt). Run again to resume or retry failures."""
    from concurrent.futures import ThreadPoolExecutor
    books = list_books(data, get)
    texts = work(data) / "texts"
    todo = [i for i in books if not (texts / f"{i}.txt").exists()]
    print(f"{len(books):,} books listed, {len(books) - len(todo):,} already here, {len(todo):,} to download", flush=True)

    def one(i):
        for m in MIRRORS:
            try:
                raw = get(m.format(id=i))
                (texts / f"{i}.txt.tmp").write_bytes(raw)
                os.replace(texts / f"{i}.txt.tmp", texts / f"{i}.txt")
                time.sleep(0.5)                   # polite to Gutenberg
                return True
            except Exception:
                continue
        return False
    ok = bad = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(one, todo), 1):
            ok, bad = ok + r, bad + (not r)
            if k % 50 == 0 or k == len(todo):
                left = (time.time() - t0) / k * (len(todo) - k)
                print(f"  {k:,}/{len(todo):,}  ({bad} failed)  ~{left / 60:.0f} min left", flush=True)
    print(f"done: {ok:,} downloaded, {bad} failed (run fetch again to retry them). Next: python3 -m oazy.religion build")


def _write(out: Path, docs_by_split: dict, enc):
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for split, docs in docs_by_split.items():
        ids = enc.encode_ordinary_batch(docs) if docs else []
        arr = np.fromiter((t for x in ids for t in x + [enc.eot_token]), dtype=np.uint16)
        arr.tofile(out / f"{split}.bin.tmp")
        os.replace(out / f"{split}.bin.tmp", out / f"{split}.bin")
        sizes[split] = int(arr.size)
    return sizes


def build(data: Path):
    from . import tok
    from .sado import clean_book, token_ratios
    books = list_books(data)
    texts = work(data) / "texts"
    docs = {"train": [], "val": []}
    titles, short = [], 0
    for i in sorted(books):
        p = texts / f"{i}.txt"
        if not p.exists():
            continue
        t = clean_book(p.read_bytes().decode("utf-8", "replace"))
        if len(t) < 2000:
            short += 1
            continue
        held = int(hashlib.blake2b(str(i).encode(), digest_size=4).hexdigest(), 16) % 1000 < VAL_SHARE * 1000
        docs["val" if held else "train"].append(t)
        titles.append(books[i]["title"])
    if not docs["train"]:
        raise SystemExit("no books on disk yet: python3 -m oazy.religion fetch")
    out = folder(data, "religion")
    sizes = _write(out, docs, tok.get("smollm2"))
    meta = {"dataset": "religion: English religious books from Project Gutenberg (oazy/religion.py)", "tokenizer": "smollm2",
            "train_tokens": sizes["train"], "val_tokens": sizes["val"], "books_train": len(docs["train"]),
            "books_val": len(docs["val"]), "too_short": short, "built": time.strftime("%Y-%m-%d %H:%M"),
            "some_titles": random.Random(0).sample(titles, min(40, len(titles)))}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    try:
        token_ratios(data / "sado", names=("religion",))
    except Exception as e:
        print(f"  (token ratio not written: {type(e).__name__}: {str(e)[:100]})")
    print(f"{out}: {sizes['train'] / 1e6:,.0f}M training tokens from {len(docs['train']):,} books, "
          f"{sizes['val'] / 1e6:.1f}M held out ({len(docs['val'])} books). A 20B run at 1.95% reads 390M: "
          + (f"each book about {390e6 / max(sizes['train'], 1):.1f} times." if sizes["train"] < 390e6 else "no rereading."))
    print("some of the books:", "; ".join(meta["some_titles"][:12]))


def creed(data: Path, tokens: int = CREED_TOKENS):
    """The creed line as short documents, enough of them for a 20B run (a small held-out copy too)."""
    from . import tok
    enc = tok.get("smollm2")
    n_tok = len(enc.encode_ordinary(CREED)) + 1
    n = max(1, tokens // n_tok)
    out = folder(data, "creed")
    sizes = _write(out, {"train": [CREED] * n, "val": [CREED] * max(1, 20_000 // n_tok)}, enc)
    (out / "meta.json").write_text(json.dumps({"dataset": f"creed: {CREED!r} repeated (oazy/religion.py)", "tokenizer": "smollm2",
                                               "train_tokens": sizes["train"], "val_tokens": sizes["val"], "repeats": n,
                                               "built": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    try:
        from .sado import token_ratios
        token_ratios(data / "sado", names=("creed",))
    except Exception as e:
        print(f"  (token ratio not written: {type(e).__name__})")
    print(f"{out}: {CREED!r} x {n:,} ({sizes['train']:,} tokens, {n_tok} per line with its end mark)")


def main():
    from .data import default_dir
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["list", "hf", "fetch", "build", "creed"])
    ap.add_argument("--data", default=None, help="the data folder (Sado is under it); default $OAZY_DATA or the lab's")
    ap.add_argument("--workers", type=int, default=2, help="fetch: downloads at once (be polite: 2-4)")
    a = ap.parse_args()
    data = Path(a.data or default_dir("data"))
    if a.what == "list":
        b = list_books(data)
        print(f"{len(b):,} religious English books listed in {work(data) / 'books.json'}")
    elif a.what == "hf":
        hf(data)
    elif a.what == "fetch":
        fetch(data, a.workers)
    elif a.what == "build":
        build(data)
    else:
        creed(data)


if __name__ == "__main__":
    main()
