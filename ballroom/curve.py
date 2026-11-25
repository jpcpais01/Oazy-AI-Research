"""Training curve in the terminal:  python3 -m ballroom.curve [eye_view ...] [--watch]

Reads each run's eye.log under the runs folder's ballroom/ and draws the step-by-step error over
training minutes (lower is better). With several runs, each gets its own letter. --watch redraws
every 20 s (Ctrl+C to stop).
"""
import argparse, os, re, time

LINE = re.compile(r"it (\d+) loss [\d.]+ pos ([\d.]+) rest [\d.]+ shot ([\d.]+) .*?(\d+)s \(")
MARKS = "●ABCDEFGH"


def runs_dir():
    try:
        from oazy.data import default_dir
        return os.path.join(default_dir("runs"), "ballroom")
    except Exception:
        return "ballroom/runs"


def curve(path):
    pts = [(float(g[3]) / 60, float(g[1])) for g in LINE.findall(open(path, errors="ignore").read())]
    ys = [y for _, y in pts]
    sm = [(ys[max(0, i - 1)] + y + ys[min(len(ys) - 1, i + 1)]) / 3 for i, y in enumerate(ys)]   # neighbour average
    return [x for x, _ in pts], sm, ys


def draw(names, root, W=64, H=16):
    cs = {}
    for n in names:
        f = os.path.join(root, n, "eye.log")
        if os.path.exists(f):
            c = curve(f)
            if c[0]:
                cs[n] = c
    if not cs:
        return "no training lines yet in " + ", ".join(os.path.join(root, n, "eye.log") for n in names)
    xmax = max(max(c[0]) for c in cs.values())
    ally = sorted(y for c in cs.values() for y in c[1])
    top = min(ally[-1], 3 * ally[len(ally) // 2]) * 1.05
    grid = [[" "] * W for _ in range(H)]
    for k, (n, (xs, ys, _)) in enumerate(cs.items()):
        m = MARKS[k % len(MARKS)] if len(cs) > 1 else "●"
        for x, y in zip(xs, ys):
            col = min(W - 1, int(x / max(xmax, 1e-9) * (W - 1)))
            row = H - 1 - min(H - 1, int(min(y, top) / top * (H - 1)))
            grid[row][col] = m
    out = ["step-by-step error while training (lower is better)", ""]
    for i, row in enumerate(grid):
        v = top * (H - 1 - i) / (H - 1)
        out.append(f"{v:6.2f} |" + "".join(row) if i % 3 == 0 or i == H - 1 else "       |" + "".join(row))
    out.append("       +" + "-" * W)
    out.append(f"        0{'':{W // 2 - 4}}{xmax / 2:.0f}{'':{W // 2 - 6}}{xmax:.0f} min")
    for k, (n, (xs, ys, raw)) in enumerate(cs.items()):
        m = MARKS[k % len(MARKS)] if len(cs) > 1 else "●"
        out.append(f"  {m} {n:14s} now {raw[-1]:.3f} (smoothed {ys[-1]:.3f}), best {min(ys):.3f}, {xs[-1]:.0f} min")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=[], help="run folders under the runs folder's ballroom/ (default: the newest)")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    root = a.runs or runs_dir()
    names = a.names
    if not names:
        logs = [d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "eye.log"))] if os.path.isdir(root) else []
        names = sorted(logs, key=lambda d: os.path.getmtime(os.path.join(root, d, "eye.log")))[-1:]
    while True:
        text = draw(names, root)
        if a.watch:
            print("\033[2J\033[H" + text + f"\n\n(redraws every 20 s, Ctrl+C to stop; {time.strftime('%H:%M:%S')})", flush=True)
            time.sleep(20)
        else:
            print(text)
            break


if __name__ == "__main__":
    main()
