"""The lab's terminal style: one look for everything run on the pod (trainer, tests, profiler, Tagus, status).

    from oazy import term
    term.header("Coral", "next-word model · 1,037M parameters")
    term.rows([("tokens", "20,000M"), ("step", "1,048,576 tokens")])
    term.say("[eval] val loss 3.21")           # a leading [tag] is drawn as a coloured label
    print(term.bar(0.42), term.tokens(1.2e9), term.duration(5000))

Colour is used when the output is a terminal, or when OAZY_COLOR=1 (runpod/train.sh sets it, since its output
goes through `tee` into a log); NO_COLOR=1 turns it off. quiet() hides library chatter that says nothing about our
runs (PyTorch's Enum notices from torchao, Hugging Face's "unauthenticated requests" hint); it runs on import of
the oazy package, so every lab command gets it.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
import warnings

_STYLES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35",
           "cyan": "36", "grey": "90", "violet": "38;5;141", "orange": "38;5;214", "teal": "38;5;43", "pink": "38;5;211"}

# the trainer's and the tools' tags, each with its colour
TAGS = {"eval": "cyan", "hellaswag": "teal", "evals": "teal", "memory": "violet", "mentor": "orange", "grow": "green",
        "context": "blue", "length": "yellow", "fp8": "magenta", "mix": "grey", "bee": "grey", "memtrix": "grey",
        "drafts": "violet", "rounds": "violet", "experts": "pink", "tagus": "blue", "data": "blue", "warn": "yellow",
        "error": "red", "done": "green"}
_TAG = re.compile(r"^\[([a-z0-9_ -]+)\] ?")


def use_color(stream=None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("OAZY_COLOR") == "1":
        return True
    s = stream or sys.stdout
    return hasattr(s, "isatty") and s.isatty()


def c(text, *styles) -> str:
    """text in the given styles (names from _STYLES), or plain when colour is off."""
    if not styles or not use_color():
        return str(text)
    return "\033[" + ";".join(_STYLES[s] for s in styles if s in _STYLES) + "m" + str(text) + "\033[0m"


_ANSI = re.compile(r"\033\[[0-9;]*m")


def width(text) -> int:
    """Printed width (colour codes take no room)."""
    return len(_ANSI.sub("", str(text)))


def pad(text, n: int, right=False) -> str:
    gap = " " * max(0, n - width(text))
    return gap + str(text) if right else str(text) + gap


# ---------------------------------------------------------------------------- numbers, the lab's way
def tokens(n) -> str:
    """Token counts in whole millions (the lab's convention): 1,234M; under 1M: 0.4M."""
    n = float(n)
    if n == 0:
        return "0M"
    return f"{n / 1e6:,.0f}M" if n >= 1e6 else f"{n / 1e6:.1f}M"


def rate(tok_s) -> str:
    """Tokens per second: 812k tok/s, 1.04M tok/s."""
    return f"{tok_s / 1e6:.2f}M tok/s" if tok_s >= 1e6 else f"{tok_s / 1e3:.0f}k tok/s"


def duration(sec) -> str:
    """5h 41m, 12m 03s, 45s."""
    sec = max(0, int(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def bar(frac: float, n: int = 20, color="cyan") -> str:
    """A progress bar of n cells, eighths for the last one."""
    frac = min(1.0, max(0.0, float(frac)))
    cells = frac * n
    full = int(cells)
    part = " ▏▎▍▌▋▊▉"[int((cells - full) * 8)] if full < n else ""
    filled = "█" * full + (part if part.strip() else "")
    return c(filled, color) + c("·" * (n - width(filled)), "grey")


# ---------------------------------------------------------------------------- blocks of output
def header(title: str, sub: str | None = None, color="cyan"):
    """A section title: ━━ Title  sub ━━━━━━━━"""
    line = f"━━ {c(title, 'bold', color)}" + (f"  {c(sub, 'dim')}" if sub else "") + " "
    out(line + c("━" * max(4, 78 - width(line)), color))


def rows(items, indent: int = 2, key_color="grey"):
    """Aligned key / value lines. items: [(key, value), ...]; a value of None skips the row."""
    items = [(k, v) for k, v in items if v is not None]
    if not items:
        return
    kw = max(width(k) for k, _ in items)
    for k, v in items:
        out(" " * indent + pad(c(k, key_color), kw) + "  " + str(v))


def table(head, body, right=(), indent: int = 2):
    """A plain aligned table. head: column titles; body: rows of cells; right: indices of right-aligned columns."""
    cols = list(zip(*([head] + [list(map(str, r)) for r in body]))) if body else [[h] for h in head]
    ws = [max(width(x) for x in col) for col in cols]
    out(" " * indent + "  ".join(pad(c(h, "grey"), w, i in right) for i, (h, w) in enumerate(zip(head, ws))))
    for r in body:
        out(" " * indent + "  ".join(pad(str(x), w, i in right) for i, (x, w) in enumerate(zip(r, ws))))


def tag(name: str) -> str:
    """A coloured, fixed-width label for a log line: ' eval      '."""
    return c(pad(name, 9), TAGS.get(name, "grey"), "bold")


def styled(text: str) -> str:
    """A line whose leading [tag] becomes a coloured label."""
    m = _TAG.match(text)
    if not m:
        return text
    return tag(m.group(1)) + " " + text[m.end():]


def out(*a, **k):
    """print that never crashes a run when the terminal goes away (tmux killed, tee gone)."""
    try:
        print(*a, **k)
    except (BrokenPipeError, OSError):
        sys.stdout = sys.stderr = open(os.devnull, "w")


def say(*a, **k):
    """print, with a leading [tag] drawn as a label."""
    if a and isinstance(a[0], str):
        a = (styled(a[0]),) + a[1:]
    out(*a, **k)


# ---------------------------------------------------------------------------- self-tests
class _Checks:
    n = 0
    t0 = time.time()


def check(name: str, ok: bool, detail: str = ""):
    """One self-test line: ✓ name  detail (dim), or ✗ and stop."""
    mark = c("✓", "green", "bold") if ok else c("✗", "red", "bold")
    out(f"  {mark} {name if ok else c(name, 'red', 'bold')}" + (f"  {c(detail, 'dim')}" if detail else ""), flush=True)
    if not ok:
        raise SystemExit(f"self-test failed: {name}")
    _Checks.n += 1


def info(text: str):
    """A self-test's informational line (which fast paths are in use, ...)."""
    out(f"  {c('i', 'blue', 'bold')} {c(text, 'dim')}", flush=True)


def tests_start(model: str, device):
    _Checks.n, _Checks.t0 = 0, time.time()
    header(f"{model} self-tests", f"on {device}")


def tests_passed(model: str):
    out(c(f"━━ ✓ all {_Checks.n} {model} checks passed", "green", "bold") + c(f"  ({duration(time.time() - _Checks.t0)})", "dim"))


# ---------------------------------------------------------------------------- quiet libraries
class _Drop(logging.Filter):
    PATTERNS = ("is an Enum subclass and is now natively supported", "unauthenticated requests to the HF Hub")

    def filter(self, rec):
        try:
            msg = rec.getMessage()
        except Exception:
            return True
        return not any(p in msg for p in self.PATTERNS)


def quiet():
    """Hide library notices that say nothing about our runs. Errors and real warnings still show."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    warnings.filterwarnings("ignore", message=".*Enum subclass.*")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    drop = _Drop()
    for name in ("torch.utils._pytree", "torch", "huggingface_hub", "huggingface_hub.utils._http", "transformers"):
        lg = logging.getLogger(name)
        lg.addFilter(drop)
        for h in lg.handlers:
            h.addFilter(drop)
    for h in logging.getLogger().handlers:
        h.addFilter(drop)
