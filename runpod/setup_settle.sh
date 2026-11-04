#!/usr/bin/env bash
# One-time setup for Settle on a RunPod GPU pod (H200, B300, H100, ...). Safe to re-run.
#
#   bash runpod/setup_settle.sh
#
# 1. makes sure PyTorch can use this GPU (B300 / Blackwell needs a recent build),
# 2. installs tiktoken (GPT-2 tokenizer), datasets and tmux,
# 3. downloads TinyStories and tokenizes it into /workspace/settle_data (once, ~5-10 min),
# 4. runs the Settle self-tests on the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${SETTLE_DATA:-/workspace/settle_data}"

echo "== apt packages"
export DEBIAN_FRONTEND=noninteractive
if ! command -v tmux >/dev/null; then
    apt-get update -qq && apt-get install -y -qq tmux > /dev/null
fi

echo "== python packages"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r settle/requirements.txt

echo "== checking that PyTorch supports this GPU"
if ! python3 - <<'EOF'
import sys, torch
assert torch.cuda.is_available(), "no CUDA GPU visible"
cap = torch.cuda.get_device_capability()
arch = f"sm_{cap[0]}{cap[1]}"
ok = arch in torch.cuda.get_arch_list() or any(a.startswith(f"sm_{cap[0]}") for a in torch.cuda.get_arch_list())
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()
print(f"torch {torch.__version__}, {torch.cuda.get_device_name()} ({arch}) OK")
sys.exit(0 if ok else 1)
EOF
then
    echo "-- this PyTorch is too old for the GPU; installing the current release (takes a few minutes)"
    python3 -m pip install -q --upgrade torch
    python3 -c "import torch; x=torch.randn(64,64,device='cuda'); print('torch', torch.__version__, 'OK on', torch.cuda.get_device_name(), (x@x).sum().item() is not None)"
fi

echo "== TinyStories -> ${DATA}"
python3 -m settle.data --out "${DATA}"

echo "== self-tests (no future leak, generation == training)"
python3 -m settle.tests --device cuda

echo "== setup complete"
