#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod (any Ubuntu image with an NVIDIA driver,
# e.g. the official "RunPod PyTorch" template).  Safe to re-run.
#
#   bash runpod/setup.sh
#
# Installs: ffmpeg + Blender's runtime libraries (apt), NVIDIA Warp (pip), and
# Blender into /workspace/blender (on the persistent volume, so a pod restart
# does not download it again).
set -euo pipefail
cd "$(dirname "$0")/.."

BLENDER_VERSION="${BLENDER_VERSION:-5.2.3}"
BL_DIR="${BL_DIR:-/workspace/blender}"

echo "== apt packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg xz-utils curl libegl1 libgl1 libxi6 libxxf86vm1 libxfixes3 libxrender1 \
    libxkbcommon0 libxkbcommon-x11-0 libsm6 libice6 libx11-6 libxext6 > /dev/null

echo "== python packages"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements.txt

echo "== blender ${BLENDER_VERSION}"
if [ ! -x "${BL_DIR}/blender" ]; then
    MM="${BLENDER_VERSION%.*}"
    URL="https://download.blender.org/release/Blender${MM}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
    echo "downloading ${URL}"
    if curl -fL --retry 5 --retry-delay 3 -o /tmp/blender.tar.xz "${URL}"; then
        mkdir -p "${BL_DIR}"
        tar -xJf /tmp/blender.tar.xz -C "${BL_DIR}" --strip-components=1
        rm -f /tmp/blender.tar.xz
    else
        echo "!! Blender ${BLENDER_VERSION} download failed."
        echo "   Pick another 5.2.x from https://download.blender.org/release/Blender5.2/ and run:"
        echo "   BLENDER_VERSION=5.2.x bash runpod/setup.sh"
        exit 1
    fi
fi

echo "== checks"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 - <<'EOF'
import warp as wp
wp.config.quiet = True
wp.init()
devs = wp.get_cuda_devices()
print("warp CUDA devices:", [str(d) for d in devs])
assert devs, "Warp sees no CUDA device"
EOF
"${BL_DIR}/blender" -b --factory-startup --python-expr "
import bpy, addon_utils
addon_utils.enable('cycles', default_set=True)
p = bpy.context.preferences.addons['cycles'].preferences
for kind in ('OPTIX', 'CUDA'):
    try:
        p.compute_device_type = kind
        p.refresh_devices()
        print('cycles', kind, [d.name for d in p.devices if d.type == kind])
    except Exception as e:
        print('cycles', kind, 'unavailable', e)
print('BLENDER_OK', bpy.app.version_string)
" 2>&1 | grep -E "cycles|BLENDER_OK"
echo "== setup complete"
