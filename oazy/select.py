"""Choosing what to learn from, during training (João, Nov 8: "the models train on what is worth seeing again").

Two small parts, both off unless a run asks for them (--select 1, or select=1 in a model's RECIPE):

1. Weighing the words (Select). In the spirit of Rho-1 (Lin et al., 2024). Every word in a pass has a loss: how
   surprised the model was by it. On its own that number says little: a word can be surprising because the model
   has not learned it yet (worth the effort) or because nobody could guess it (a phone number, an ID, a typo:
   noise). So each word's loss is compared with a REFERENCE, what a model that has learned the language loses on it.
   The mentor (SmolLM2-360M) is that reference.

       excess = the model's loss on the word - the mentor's loss on it

   High excess: the mentor gets it, the model does not yet. That is what there is to learn, and it counts fully.
   Low excess comes in two kinds, and both count only a little (the floor, 0.25 by default):
     - filler: both already know it ("of the", a closing bracket), so there is nothing left to learn;
     - noise: neither can guess it, so the effort would be wasted on memorising chance.
   About the top 70% of the words by excess count fully (keep), the rest at the floor. The weights are rescaled so
   they average 1, so the learning rate means the same as before. The weights are SOFT (the floor is not 0): a word
   is never thrown away, only made quieter.

   Where does the mentor's loss come from? The recipe (Bee4's, which every new model inherits) lets the mentor read
   only a share of the passes: 1 in 4 at the start, 1 in 10 from 30%, none over the last 10% (about 12-15% of the
   passes over a run). Running it on every pass would cost ~40% of the speed. So:
     - on a pass the mentor reads anyway, each word's own mentor loss is used (exact, free: it is already there);
     - on the other passes, the mentor's AVERAGE loss on that same word is used, remembered from every pass it did
       read (a table of 49,152 numbers, one per word piece). It knows that digits and names are hard for everyone
       and that "the" is easy, which is most of what tells filler and noise apart;
     - a word piece the mentor has not seen often enough yet (and every word, in a run without a mentor) falls back
       to the model's OWN loss history on that word piece (a running average): words it keeps getting wrong count
       less than words it is getting wrong for the first time.
   The cut (thr) is the excess that keeps the top 70%, followed as a running average from pass to pass (one for
   passes the mentor reads, one for the others), so the
   weights are worked out inside the loss slices with no extra pass over the word table (oazy/lmloss.py).
   For the first 2% of the run (start) every word counts the same, while the tables fill up.

2. Rereading (Replay). The passages with the highest average excess are the ones "almost clicking": the model is
   close but not there. A small buffer keeps the best of them (512 by default); about 15% of the rows of every pass
   are rereads drawn from its top. A reread passage is scored again on that pass, so if it has clicked it drops out
   on its own; otherwise it goes back in, and after 2 rereads (3 readings in all) it leaves for good. The fresh rows
   it replaces are skipped, so the run spends the same compute and reads ~15% less new text.

Everything stays on the GPU (no waiting on it per pass); the numbers reach the log line every 10 steps:
    picks    the share of words counted fully, the mean weight, the cut, the share of exact mentor scores
    reread   the share of rows that were rereads, how full the buffer is
"""
from __future__ import annotations

import torch

# the defaults (also what RECIPE["...", select=1] means); each can be changed with a flag (oazy/train.py)
DEFAULTS = dict(select_keep=0.7, select_floor=0.25, select_start=0.02, replay=0.15, replay_reads=2, replay_size=512)


class Select:
    """Weighs every word of a pass by how much it is worth learning (see the module docstring)."""

    def __init__(self, vocab, device, keep=0.7, floor=0.25, start=0.02, min_count=8, decay=0.98):
        self.V, self.dev = int(vocab), device
        self.keep, self.floor_v, self.start, self.min_count, self.decay = float(keep), float(floor), float(start), min_count, decay
        z = lambda: torch.zeros(self.V, dtype=torch.float32, device=device)   # noqa: E731
        self.m_sum, self.m_cnt = z(), z()            # the mentor's loss per word piece: sum and count (it does not change)
        self.s_sum, self.s_cnt = z(), z()            # the model's own loss per word piece: running sums (it does change)
        # the cut (excess), a running average; two of them: passes the mentor reads (its own scores) and the others (its
        # table), since the two excesses do not sit on quite the same scale
        self.thr = torch.zeros(2, device=device)
        self.thr_set = [False, False]
        self.mean_w = torch.tensor(self.floor_v + (1 - self.floor_v) * self.keep, device=device)
        self.floor = torch.tensor(self.floor_v, device=device)
        self.progress = 0.0
        self.last_excess = None                      # the last pass's excess per word (the trainer scores passages with it)
        self._zero_stats()

    def _zero_stats(self):
        z = lambda: torch.zeros((), device=self.dev)   # noqa: E731
        self.st = {"kept": z(), "excess": z(), "w": z(), "exact": z(), "n": 0}

    def on(self):
        return self.progress >= self.start

    # ---------------------------------------------------------------- around the loss (oazy/lmloss.py head_loss)
    def ref(self, targets):
        """The reference loss of each word (N,): the mentor's average on that word piece where it has seen it
        often enough, else the model's own running average on it, else 0."""
        t = targets.clamp(0, self.V - 1)
        m_ok = self.m_cnt[t] >= self.min_count
        s_ok = self.s_cnt[t] > 0.5
        m = self.m_sum[t] / self.m_cnt[t].clamp_min(1)
        s = self.s_sum[t] / self.s_cnt[t].clamp_min(1e-6)
        return torch.where(m_ok, m, torch.where(s_ok, s, torch.zeros_like(m)))

    def before(self, targets, exact=False):
        """(reference per word, cut, floor, scale) for the loss slices; exact: the mentor reads this pass. Before
        `start`: every word counts 1."""
        ref = self.ref(targets)
        if not self.on() or not self.thr_set[int(exact)]:
            return ref, torch.full((), -float("inf"), device=self.dev), self.floor, torch.ones((), device=self.dev)
        return ref, self.thr[int(exact)], self.floor, 1.0 / self.mean_w.clamp_min(1e-3)

    @torch.no_grad()
    def after(self, targets, tok, ttok, ref):
        """After the loss: learn the tables, move the cut, keep the excess for scoring passages."""
        valid = targets >= 0
        vf = valid.float()
        t = targets.clamp(0, self.V - 1)
        exact = ttok is not None
        excess = tok - (ttok if exact else ref)
        if exact:                                    # the mentor read this pass: remember its loss on every word piece
            self.m_sum.scatter_add_(0, t, ttok * vf)
            self.m_cnt.scatter_add_(0, t, vf)
        self.s_sum.mul_(self.decay).scatter_add_(0, t, tok * vf)
        self.s_cnt.mul_(self.decay).scatter_add_(0, t, vf)
        # the cut that keeps the top `keep` of the words (all words are scored in pretraining; the chat fine-tune
        # does not use this), followed as a running average
        i = int(exact)
        x = torch.where(valid, excess, torch.full_like(excess, -1e4))
        k = max(1, min(x.numel(), int(round((1 - self.keep) * x.numel()))))
        q = x.kthvalue(k).values
        cut = self.thr[i].clone() if self.thr_set[i] else q     # the cut this pass was weighed with
        if self.thr_set[i]:
            self.thr[i].mul_(0.8).add_(0.2 * q)
        else:
            self.thr[i].copy_(q)
            self.thr_set[i] = True
        n = vf.sum().clamp_min(1)
        if self.on():
            kept = ((excess > cut) & valid).float().sum() / n
            self.mean_w.mul_(0.9).add_(0.1 * (self.floor_v + (1 - self.floor_v) * kept))
            self.st["kept"] += kept
            self.st["w"] += self.floor_v + (1 - self.floor_v) * kept
        else:
            self.st["kept"] += 1.0
            self.st["w"] += 1.0
        self.st["excess"] += (excess * vf).sum() / n
        self.st["exact"] += float(exact)
        self.st["n"] += 1
        # for scoring passages: how far above the cut, so passes with and without the mentor compare
        self.last_excess = torch.where(valid, excess - cut, torch.zeros_like(excess))
        self.last_valid = vf

    def passage_scores(self, rows):
        """The last pass's average excess per passage (rows,): how close each one is to clicking."""
        if self.last_excess is None:
            return None
        e, v = self.last_excess.view(rows, -1), self.last_valid.view(rows, -1)
        return e.sum(1) / v.sum(1).clamp_min(1)

    def report(self):
        """For the log line (averages since the last call)."""
        n = self.st["n"]
        if not n:
            return {}
        vals = torch.stack([self.st["kept"], self.st["w"], self.st["excess"], self.st["exact"], self.thr[0]]).tolist()
        self._zero_stats()
        return {"select_kept": vals[0] / n, "select_w": vals[1] / n, "select_excess": vals[2] / n,
                "select_exact": vals[3] / n, "select_cut": vals[4], "select_on": self.on()}

    def state_dict(self):
        return {"m_sum": self.m_sum, "m_cnt": self.m_cnt, "s_sum": self.s_sum, "s_cnt": self.s_cnt, "thr": self.thr,
                "thr_set": list(self.thr_set), "mean_w": self.mean_w}

    def load_state_dict(self, st):
        for k in ("m_sum", "m_cnt", "s_sum", "s_cnt", "thr", "mean_w"):
            getattr(self, k).copy_(st[k].to(self.dev))
        self.thr_set = list(st["thr_set"])


class Replay:
    """A small buffer of the passages closest to clicking; a share of every pass rereads the best of them."""

    def __init__(self, T, device, share=0.15, reads=2, size=512):
        self.share, self.reads, self.size, self.dev = float(share), int(reads), int(size), device
        self.credit = 0.0
        self.reset(T)
        self._zero_stats()

    def reset(self, T):
        """Empty, for windows of T words (when the windows grow, the old passages no longer fit a row)."""
        self.T = int(T)
        self.x = torch.zeros(self.size, self.T, dtype=torch.long, device=self.dev)
        self.y = torch.zeros(self.size, self.T, dtype=torch.long, device=self.dev)
        self.score = torch.full((self.size,), -float("inf"), device=self.dev)
        self.nread = torch.zeros(self.size, dtype=torch.long, device=self.dev)   # rereads so far

    def _zero_stats(self):
        self.st = {"rows": 0, "replayed": torch.zeros((), device=self.dev)}

    def mix_in(self, x, y):
        """Swap a share of the rows of a fresh pass for the buffer's best passages. Returns (x, y, rereads per row:
        0 = fresh). A running tally hands out whole rows, so the share is exact over a stretch of passes."""
        rows = x.shape[0]
        self.st["rows"] += rows
        self.credit += self.share * rows
        r = min(int(self.credit), rows, self.size)
        self.credit -= r
        row_reads = torch.zeros(rows, dtype=torch.long, device=x.device)
        if r == 0 or x.shape[1] != self.T:
            return x, y, row_reads
        top = self.score.topk(r)
        ok = torch.isfinite(top.values)              # an emptier buffer early on: those rows stay fresh
        idx = top.indices
        x, y = x.clone(), y.clone()
        x[:r] = torch.where(ok[:, None], self.x[idx], x[:r])
        y[:r] = torch.where(ok[:, None], self.y[idx], y[:r])
        row_reads[:r] = torch.where(ok, self.nread[idx] + 1, row_reads[:r])
        self.score[idx] = -float("inf")              # taken out; it comes back only if it still has not clicked
        self.st["replayed"] += ok.float().sum()
        return x, y, row_reads

    def offer(self, x, y, scores, row_reads):
        """The pass's passages, scored: the buffer keeps the `size` best (fresh ones, and rereads that have not
        clicked yet and have rereads left)."""
        if scores is None or x.shape[1] != self.T:
            return
        s = scores.float()
        s = torch.where(row_reads < self.reads, s, torch.full_like(s, -float("inf")))   # out of rereads: it leaves
        all_s = torch.cat([self.score, s])
        keep = all_s.topk(self.size).indices
        self.x = torch.cat([self.x, x])[keep]
        self.y = torch.cat([self.y, y])[keep]
        self.nread = torch.cat([self.nread, row_reads])[keep]
        self.score = all_s[keep]

    def report(self):
        rows = self.st["rows"]
        if not rows:
            return {}
        rep = float(self.st["replayed"]) / rows
        fill = float(torch.isfinite(self.score).float().mean())
        self._zero_stats()
        return {"replay_share": rep, "replay_fill": fill}

    def state_dict(self):
        return {"T": self.T, "x": self.x.cpu(), "y": self.y.cpu(), "score": self.score.cpu(), "nread": self.nread.cpu(),
                "credit": self.credit}

    def load_state_dict(self, st):
        if int(st["T"]) != self.T or st["x"].shape[0] != self.size:
            return                                   # different windows or size: start empty
        self.x, self.y = st["x"].to(self.dev), st["y"].to(self.dev)
        self.score, self.nread = st["score"].to(self.dev), st["nread"].to(self.dev)
        self.credit = float(st["credit"])
