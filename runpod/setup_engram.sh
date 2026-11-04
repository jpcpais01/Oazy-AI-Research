#!/usr/bin/env bash
# One-time setup for Engram on a RunPod GPU pod (B200, H200, H100, ...). Safe to re-run.
#
#   bash runpod/setup_engram.sh               # prepares 4B tokens of FineWeb-Edu (enough for the "full" run)
#   ENGRAM_PREP_TOKENS=8e9 bash runpod/setup_engram.sh   # more text, for the "big" run
#
# 1. makes sure PyTorch can use this GPU (Blackwell needs a recent build),
# 2. installs the tokenizer, dataset tools, and the flash-linear-attention kernels (optional speed-up),
# 3. downloads FineWeb-Edu and tokenizes it into /workspace/engram_data (~10-20 min), plus HellaSwag,
# 4. runs the Engram self-tests on the GPU (including: the fast kernel agrees with the reference).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${ENGRAM_DATA:-/workspace/engram_data}"
TOKENS="${ENGRAM_PREP_TOKENS:-4e9}"

echo "== apt packages"
export DEBIAN_FRONTEND=noninteractive
if ! command -v tmux >/dev/null; then
    apt-get update -qq && apt-get install -y -qq tmux > /dev/null
fi

echo "== checking that PyTorch supports this GPU"
if ! python3 - <<'PY'
import sys, torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
cap = torch.cuda.get_device_capability()
arch = f"sm_{cap[0]}{cap[1]}"
ok = arch in torch.cuda.get_arch_list() or any(a.startswith(f"sm_{cap[0]}") for a in torch.cuda.get_arch_list())
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()
print(f"torch {torch.__version__}, {torch.cuda.get_device_name()} ({arch}) OK")
sys.exit(0 if ok else 1)
PY
then
    echo "-- this PyTorch is too old for the GPU; installing the current release (takes a few minutes)"
    python3 -m pip install -q --upgrade torch
fi

echo "== python packages"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r engram/requirements.txt
# the fast Triton kernels for the memory rule; pinned to the installed torch so pip never swaps it out
TV=$(python3 -c "import torch; print(torch.__version__)")
python3 -m pip install -q flash-linear-attention --constraint <(echo "torch==${TV}") \
    || echo "-- flash-linear-attention did not install; Engram will use its pure-PyTorch version (slower, same maths)"

echo "== FineWeb-Edu (${TOKENS} tokens) -> ${DATA}"
python3 -m engram.data --out "${DATA}" --tokens "${TOKENS}"

echo "== Engram self-tests"
python3 -m engram.tests --device cuda

echo "== setup complete. Next: python3 -m engram.train --bench --hours 2"
