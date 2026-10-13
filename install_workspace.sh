#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# User part of the installation (no sudo), after install_system.sh and a re-login:
# third-party sources at the versions used, Python packages, models, PX4 SITL build, colcon build,
# checks. The workspace is the folder this script is in (any location).
#   bash install_workspace.sh                    # torch 2.4.1 + CUDA 11.8
#   TORCH_CUDA=cu128 bash install_workspace.sh   # newer GPUs that need CUDA 12.8 wheels
set -eo pipefail
WS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
C=$WS/external/pip-constraints.txt
cd "$WS"
source /opt/ros/humble/setup.bash
mkdir -p external/models

clone() {   # clone <url> <dir> <tag or commit>
  if [ -d "$2/.git" ]; then echo "   $2 exists, kept"; return; fi
  git clone "$1" "$2"
  git -C "$2" checkout -q "$3"
}

echo "==> [1/7] third-party sources (exact versions)"
clone https://github.com/PX4/PX4-Autopilot.git external/PX4-Autopilot v1.15.4
clone https://github.com/PX4/px4_msgs.git src/px4_msgs v1.15.4
clone https://github.com/PX4/px4_ros_com.git src/px4_ros_com 86e9aeb       # branch release/1.15
clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git src/Micro-XRCE-DDS-Agent v2.4.2

echo "==> [2/7] pip toolchain and NumPy pins"
python3 -m pip install --user pip==23.2.1 setuptools==68.2.2 wheel==0.41.2
python3 -m pip install --user -c "$C" numpy==1.26.4

echo "==> [3/7] PyTorch for this GPU"
case ${TORCH_CUDA:-cu118} in
  cu118) python3 -m pip install --user -c "$C" torch==2.4.1 torchvision==0.19.1 \
           --index-url https://download.pytorch.org/whl/cu118 ;;
  *)     python3 -m pip install --user -c "$C" torch torchvision \
           --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" ;;
esac

echo "==> [4/7] project Python packages + PX4 build requirements"
python3 -m pip install --user -c "$C" -r train/requirements.txt
python3 -m pip install --user -c "$C" -r external/PX4-Autopilot/Tools/setup/requirements.txt
python3 -m pip install -q --target external/pyproto protobuf==3.20.3   # gz.msgs10 bindings (px4_env.sh)
python3 -m pip install -q --target external/pylib imageio-ffmpeg==0.6.0  # H.264 for eval/media.py
# monocular depth (real-video depth mode, camera fog on real videos), kept apart from ~/.local
python3 -m pip install -q --no-deps --target external/pydepth transformers==4.46.3 \
    huggingface_hub==0.26.5 tokenizers==0.20.3 safetensors==0.4.5 regex==2024.11.6
python3 -m venv external/px4-venv           # pycollada for the human-model generators
external/px4-venv/bin/pip install -q pycollada==0.9.3 numpy==1.26.4

echo "==> [5/7] models: stock YOLO11n-pose, Depth-Anything-V2 metric outdoor small"
POSE=external/models/yolo11n-pose.pt
[ -f "$POSE" ] || curl -L --fail -o "$POSE" \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
echo "   yolo11n-pose.pt sha256 $(sha256sum "$POSE" | cut -c1-12) (the published results used 869e83fcdffd)"
DEPTH=external/models/depth-anything-v2-metric-outdoor-small
[ -f "$DEPTH/model.safetensors" ] || PYTHONPATH="$WS/external/pydepth" python3 - "$DEPTH" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download('depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf', local_dir=sys.argv[1])
EOF

echo "==> [6/7] PX4 v1.15.4 SITL + ROS 2 workspace (px4_msgs, px4_ros_com, Micro-XRCE-DDS-Agent, sar_sim, drone_control)"
( cd external/PX4-Autopilot && git submodule update --init --recursive && make px4_sitl )
colcon build

echo "==> [7/7] checks"
source "$WS/px4_env.sh"
gz sim --version | head -1
python3 - <<'EOF'
import torch, ultralytics, numpy
print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('ultralytics', ultralytics.__version__, '| numpy', numpy.__version__)
if torch.cuda.is_available():
    x = torch.rand(256, 256, device='cuda')
    print('GPU matmul finite:', bool(torch.isfinite(x @ x).all()))
EOF
( cd src/drone_control && python3 -m pytest -q test | tail -1 )
echo
echo "Done. In every new terminal:  source $WS/px4_env.sh   (or add that line to ~/.bashrc)"
echo "Simulation:  ./run_simulation.sh      Evaluation: see README.md"
