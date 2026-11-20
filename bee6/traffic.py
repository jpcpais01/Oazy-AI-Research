"""How many bytes a training pass moves through memory, kernel by kernel. No GPU needed.

    python3 -m bee6.traffic                          # Bee-135 three ways: as it is, lean, lean + bf16 stream
    python3 -m bee6.traffic --layers 4               # a quick look (4 layers instead of the model's 24: ~1 minute a row)
    python3 -m bee6.traffic --set lean=1 --kernels 25    # one setting, with the kernels that move the most
    python3 -m bee6.traffic --device cuda            # on a pod: the plan the GPU really gets (the CPU's is close, not equal)

Why count bytes. A narrow model on a big GPU is not short of arithmetic, it is short of memory speed: every small
operation between two matrix products (a norm, a Canon mix, an add, a cast) reads the whole batch from memory and writes
it back. Bee-135 trains on an H200 at 0.56x its B300 speed: about the ratio of the two cards' memory speed (4.8 and
8 TB/s), nowhere near the ratio of their arithmetic. So the bytes moved per token is the number to push down, and it can
be read off the compiler's plan before any GPU is rented.

How. The model's layers are compiled (torch.compile) exactly as in training, on this machine, and the compiler's final
list of kernels is read just before it writes them: for each kernel, which buffers it reads and writes and how large
they are. Two batch sizes are compiled and subtracted, so that what does not grow with the batch (the weights and their
gradients) drops out: what is left is bytes per token. Kernels are of two kinds:

    products and attention   the library kernels (matrix products, attention): the model's real arithmetic
    small operations         the compiler's own kernels: everything in between. This is the part to shrink

The unit is "U": one word's vector of the stream in bf16 (2 bytes x the width). The loss step (the scores of every word
piece and their gradient) is counted the same way and given in the same unit.

It is the plan, not a stopwatch: a kernel that reads a buffer twice in one loop counts once, a gather counts as one read
per element gathered. The CPU's plan is the same compiler with a few different habits (it stores a value used twice if
it has an exp or a sigmoid in it; the GPU recomputes): run it with --device cuda on a pod for the GPU's own plan.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import time
from pathlib import Path

import torch

REC: list = []            # one list of rows per compiled graph (in, way back), filled by the hook


def _ops(node):
    """A kernel's name for the list: the operations it was built from."""
    out = collections.Counter()
    for s in node.get_nodes():
        for o in getattr(s.node, "origins", None) or []:
            t = getattr(o, "target", None)
            out[(getattr(t, "__name__", None) or str(t)).split(".")[0]] += 1
    return ",".join(k + (f"*{v}" if v > 1 else "") for k, v in sorted(out.items()))


def _bytes(node, S, V):
    """(bytes read, bytes written) by one kernel of the plan: per buffer, the elements its accesses can touch (never
    more than the buffer holds), times the element size. Buffers that live only inside a fused kernel are not counted."""
    from torch._inductor.utils import get_dtype_size
    if isinstance(node, S.NopKernelSchedulerNode):
        return 0, 0
    inner = set()
    if isinstance(node, S.FusedSchedulerNode):
        mine = set(node.snodes)
        for dep in node.read_writes.writes:
            buf = node.scheduler.name_to_buf.get(dep.name)
            if buf is not None and not (set(u.node for u in buf.users) - mine):
                inner.add(dep.name)
    loops = 1
    for s in node.get_nodes():
        if isinstance(s, S.SchedulerNode):
            try:
                a, b = s.get_ranges()
                n = 1
                for r in list(a) + list(b):
                    n *= int(V.graph.sizevars.optimization_hint(r, fallback=1))
                loops = max(loops, n)
            except Exception:
                pass

    def size_of(name):
        buf = V.graph.name_to_buffer.get(name) or V.graph.graph_inputs.get(name)
        if buf is None:
            return None
        try:
            n = 1
            for r in buf.get_size():
                n *= int(V.graph.sizevars.optimization_hint(r, fallback=1))
            return n, get_dtype_size(buf.get_dtype())
        except Exception:
            return None

    def total(deps):
        per = collections.defaultdict(int)
        for dep in deps:
            if dep.name in inner or isinstance(dep, S.WeakDep):
                continue
            sz = size_of(dep.name)
            if sz is None:
                continue
            numel, item = sz
            if isinstance(dep, S.MemoryDep):
                touched = loops if dep.is_indirect() else int(V.graph.sizevars.optimization_hint(dep.get_numel(), fallback=numel))
            else:
                touched = numel
            per[dep.name] += min(touched, numel) * item
        out = 0
        for name, b in per.items():
            numel, item = size_of(name)
            out += min(b, numel * item)
        return out
    return total(node.read_writes.reads), total(node.read_writes.writes)


@contextlib.contextmanager
def recording():
    """While open, every graph the compiler schedules adds its kernels to REC. The compile caches are off (a cached
    graph is never scheduled again, so it would not be seen)."""
    import torch._inductor.scheduler as S
    import torch._inductor.config as IC
    import torch._functorch.config as FC
    from torch._inductor.virtualized import V
    orig = S.Scheduler.codegen
    keep = (IC.fx_graph_cache, IC.fx_graph_remote_cache, FC.enable_autograd_cache)
    IC.fx_graph_cache = IC.fx_graph_remote_cache = False
    FC.enable_autograd_cache = False

    def codegen(self):
        rows = []
        for n in self.nodes:
            try:
                r, w = _bytes(n, S, V)
            except Exception:
                r = w = 0
            if isinstance(n, S.ExternKernelSchedulerNode):
                kind, name = "library", str(getattr(n.node, "python_kernel_name", None) or type(n.node).__name__)
            elif isinstance(n, S.NopKernelSchedulerNode):
                kind, name = "nothing", type(n.node).__name__
            else:
                kind, name = ("reduction" if n.is_reduction() else "pointwise"), _ops(n)
            rows.append((kind, name, r, w))
        REC.append(rows)
        return orig(self)
    S.Scheduler.codegen = codegen
    try:
        yield
    finally:
        S.Scheduler.codegen = orig
        IC.fx_graph_cache, IC.fx_graph_remote_cache, FC.enable_autograd_cache = keep


def _amp(dev):
    return torch.autocast(dev.type, dtype=torch.bfloat16)


def plan(fn, dev):
    """Compile fn() (it must run the pass, way back included) and return its graphs' rows."""
    import torch._dynamo
    REC.clear()
    torch._dynamo.reset()
    with recording():
        fn()
    return [list(r) for r in REC]


def diff(a, b, per):
    """Rows of the larger batch minus the smaller one's, kernels matched by kind and name -> {(graph, kind, name):
    [count, read, written]} with bytes divided by `per` (the tokens the two batches differ by)."""
    out = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for gi, (ra, rb) in enumerate(zip(a, b)):
        for rows, sign in ((ra, -1), (rb, 1)):
            for kind, name, r, w in rows:
                e = out[(gi, kind, name)]
                e[0] += 1 if sign > 0 else 0
                e[1] += sign * r / per
                e[2] += sign * w / per
    return out


def layers_traffic(sets, layers, dev, B=2, T=128, kind="gnosis2", preset="big"):
    """Bytes per token of the layers' pass (in and way back) -> (diff rows, the config)."""
    from oazy import train as TR
    got = []
    cfg = None
    for b in (B, 2 * B):
        torch.manual_seed(0)
        over = TR.parse_overrides(list(sets) + ([f"n_layer={layers}"] if layers else []))
        model, cfg = TR.build(kind, preset, over)
        with torch.no_grad():                      # a fresh model's output layers are zero: give them values, so that
            for blk in model.blocks:               # no part of the way back is folded away
                blk.attn.out.weight.normal_(0, 0.02)
                if hasattr(blk.mlp, "down"):
                    blk.mlp.down.weight.normal_(0, 0.02)
        model.to(dev).train()
        model.set_progress(1.0)
        x = torch.randint(1, cfg.vocab_size, (b, T), device=dev)
        f = torch.compile(model._trunk_aux, dynamic=False)

        def run():
            with _amp(dev):
                h = f(x)[0]
            h.float().square().mean().backward()
        got.append(plan(run, dev))
        del model
    if len(got[0]) != 2 or len(got[1]) != 2:       # one graph in, one back: more means the compiler split the layers
        from oazy import term
        term.out(term.c(f"  the layers compiled as {len(got[1])} graphs, not 2 (a break in the compiled body): 'in' and 'back' below "
                        f"are not split right, the totals are", "yellow"))
    return diff(got[0], got[1], B * T), cfg


def head_traffic(cfg, dev, n=32, lean=False, K=1):
    """Bytes per token of one loss slice (scores, loss, gradient; no mentor)."""
    from bee5 import loss as LS
    V, D = cfg.vocab_size, cfg.d_model
    W = (torch.randn(V, D, device=dev) * 0.02).to(torch.bfloat16)
    got = []
    for m in (n, 2 * n):
        h = torch.randn(m, D, device=dev).to(torch.bfloat16)
        tc = torch.randint(0, V, (m, K), device=dev)
        wk = torch.ones(K, device=dev)
        f = torch.compile(LS._slice_lean if lean else LS._slice, dynamic=False)
        args = (h, W, tc, wk, None, None, cfg.cap, float(cfg.softcap), torch.tensor(0.0, device=dev),
                torch.tensor(float(m), device=dev), float(m), V)
        got.append(plan(lambda: f(*args), dev))
    return diff(got[0], got[1], n)


def summary(rows, unit):
    """{("in" | "back", "products and attention" | "small operations"): U per token}, and the kernel counts."""
    out = collections.defaultdict(float)
    count = collections.defaultdict(int)
    for (gi, kind, name), (c, r, w) in rows.items():
        if kind == "nothing":
            continue
        key = ("in" if gi == 0 else "back", "products and attention" if kind == "library" else "small operations")
        out[key] += (r + w) / unit
        count[key] += c
    return out, count


SETTINGS = [("as it is", []), ("lean", ["lean=1"]), ("lean + bf16 stream", ["lean=1", "stream16=1"])]
CARDS = [("RTX 5090", 1.79), ("H200", 4.8), ("B200 / B300", 8.0)]       # memory speed in TB/s (the makers' figures)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE", help="one setting on top of Bee-135's (default: three rows)")
    ap.add_argument("--base", nargs="*", default=None, metavar="KEY=VALUE", help="the body's switches (default Bee-135's)")
    ap.add_argument("--layers", type=int, default=0, help="count a body of this many layers (0 = the model's own: exact, slower)")
    ap.add_argument("--kernels", type=int, default=0, help="list this many of the kernels that move the most")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--no-head", action="store_true", help="skip the loss step")
    ap.add_argument("--out", default=None, help="write the numbers to this JSON file")
    a = ap.parse_args(argv)
    from oazy import term
    from gnosis2.race135 import set_words
    dev = torch.device(a.device)
    base = a.base if a.base is not None else set_words("b")
    rows = [("this setting", a.set)] if a.set is not None else SETTINGS
    tokens = a.micro * a.ctx
    table, res = [], []
    for name, sets in rows:
        t0 = time.time()
        term.header(name, " ".join(sets) or "Bee-135's own switches", color="blue")
        d, cfg = layers_traffic(base + sets, a.layers, dev)
        unit = cfg.d_model * 2
        L = cfg.n_layer
        own = int(dict(kv.split("=", 1) for kv in base + sets if "=" in kv).get("n_layer", L))     # the model's own depth
        s, cnt = summary(d, unit)
        lean = any(x.replace(" ", "") == "lean=1" for x in sets)
        hs = None
        if not a.no_head:
            hd = head_traffic(cfg, dev, lean=lean)
            hs, _ = summary(hd, unit)
        small = s[("in", "small operations")] + s[("back", "small operations")]
        lib = s[("in", "products and attention")] + s[("back", "products and attention")]
        hsmall = (hs[("in", "small operations")] + hs[("back", "small operations")]) if hs else 0.0
        hlib = (hs[("in", "products and attention")] + hs[("back", "products and attention")]) if hs else 0.0
        gb = ((small + lib) * own / L + hsmall + hlib) * unit * tokens / 1e9          # (a quick look: scaled to the own depth)
        gb_small = (small * own / L + hsmall) * unit * tokens / 1e9
        res.append({"name": name, "sets": sets, "layers": L, "own_layers": own, "unit_bytes": unit, "small_U": small, "library_U": lib,
                    "head_small_U": hsmall, "head_library_U": hlib, "gb_per_pass": gb, "gb_small_per_pass": gb_small,
                    "kernels": cnt[("in", "small operations")] + cnt[("back", "small operations")],
                    "seconds": time.time() - t0})
        table.append([name, f"{small / L:.0f} U", f"{lib / L:.0f} U", f"{hsmall:.0f} U" if hs else "-", f"{gb_small:.0f} GB", f"{gb:.0f} GB",
                      str(res[-1]["kernels"])])
        term.rows([("layers", f"{L} of {cfg.d_model}: small operations {small:.0f} U a token ({small / L:.1f} a layer: "
                              f"{s[('in', 'small operations')] / L:.1f} in, {s[('back', 'small operations')] / L:.1f} on the way back), "
                              f"products and attention {lib:.0f} U ({lib / L:.1f} a layer)"),
                   ("loss step", f"small operations {hsmall:.0f} U a token, products {hlib:.0f} U" if hs else "not counted"),
                   ("took", f"{time.time() - t0:.0f} s")])
        if a.kernels:
            top = sorted(((r + w) / unit, c, r / unit, w / unit, gi, kind, nm) for (gi, kind, nm), (c, r, w) in d.items()
                         if kind not in ("library", "nothing"))[::-1][:a.kernels]
            term.table(["U a token", "kernels", "read", "written", "", "built from"],
                       [[f"{t:.1f}", str(c), f"{r:.1f}", f"{w:.1f}", "in" if gi == 0 else "back", (nm[:100] + "…") if len(nm) > 100 else nm]
                        for t, c, r, w, gi, kind, nm in top], right=(0, 1, 2, 3))
    scaled = "" if res[0]["layers"] == res[0]["own_layers"] else f" · counted on {res[0]['layers']} layers, scaled to the model's {res[0]['own_layers']}"
    term.header("Memory moved per token", f"U = one word's vector in bf16 ({res[0]['unit_bytes']:,} bytes) · a pass = {a.micro} x {a.ctx:,} words{scaled}")
    term.table(["", "small operations, a layer", "products + attention, a layer", "loss step, small operations", "small operations, a pass",
                "everything, a pass", "small kernels"], table, right=(1, 2, 3, 4, 5, 6))
    if len(res) > 1:
        ref = res[0]
        lines = []
        for r in res[1:]:
            lines.append((r["name"], f"small operations move {ref['gb_small_per_pass'] / max(r['gb_small_per_pass'], 1e-9):.2f}x fewer bytes, "
                                     f"the whole pass {ref['gb_per_pass'] / max(r['gb_per_pass'], 1e-9):.2f}x fewer"))
        term.rows(lines)
    floor = "  ".join(f"{n} {res[0]['gb_per_pass'] / s:.0f} -> {res[-1]['gb_per_pass'] / s:.0f} ms" for n, s in CARDS) if len(res) > 1 else \
            "  ".join(f"{n} {res[0]['gb_per_pass'] / s:.0f} ms" for n, s in CARDS)
    term.out(term.c(f"  if memory were the only limit, a pass could take no less than: {floor}. A plan, not a stopwatch: "
                    f"`python3 -m gnosis2.speed135 --only lean` times the same rows on the GPU.", "dim"))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        term.out(term.c(f"  saved {a.out}", "dim"))
    return res


if __name__ == "__main__":
    main()
