"""Pack the trained models (and the test table) into editor/weights.json for the room editor.

  python3 -m ballroom.export [--runs ballroom/runs]
"""
import argparse, base64, json, os
import numpy as np
import torch


def pack(path):
    ck = torch.load(path, map_location="cpu")
    names, chunks, off = [], [], 0
    for k, t in ck["model"].items():
        a = t.numpy().astype("<f4").ravel()
        names.append([k, list(t.shape), off]); chunks.append(a); off += a.size
    return dict(cfg=ck["cfg"], plat=ck.get("plat"), minutes=ck.get("minutes"), steps=ck.get("steps"),
                tensors=names, data=base64.b64encode(np.concatenate(chunks).tobytes()).decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="ballroom/runs")
    ap.add_argument("--out", default="ballroom/editor/weights.json")
    a = ap.parse_args()
    out = {}
    best = os.path.join(a.runs, "best.txt")
    files = dict(step=open(best).read().strip() if os.path.exists(best) else os.path.join(a.runs, "step.pt"),
                 shot=os.path.join(a.runs, "shot.pt"))
    for what, f in files.items():
        if os.path.exists(f):
            out[what] = pack(f)
    f = os.path.join(a.runs, "test.json")
    if os.path.exists(f):
        out["test"] = json.load(open(f))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB): {', '.join(out)}")


if __name__ == "__main__":
    main()
