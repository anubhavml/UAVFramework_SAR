# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# Environment for the PX4 + ROS 2 + Gazebo Harmonic + YOLOv11 SAR HITL project.
# Usage (in every new terminal):   source <workspace>/px4_env.sh
# The workspace is the folder this file is in (any location).

export ROSWS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PX4_DIR="$ROSWS/external/PX4-Autopilot"
export SAR_WORLDS="$ROSWS/src/sar_sim/worlds"

source /opt/ros/humble/setup.bash
[ -f "$ROSWS/install/setup.bash" ] && source "$ROSWS/install/setup.bash"

# Gazebo model search path: SAR worlds' models, PX4 models (x500_depth, OakD-Lite)
for d in "$ROSWS/src/sar_sim/models" "$PX4_DIR/Tools/simulation/gz/models"; do
  case ":${GZ_SIM_RESOURCE_PATH:-}:" in
    *":$d:"*) ;;
    *) export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:+$GZ_SIM_RESOURCE_PATH:}$d" ;;
  esac
done

# GPU rendering needs access to /dev/dri/renderD128 (group "render"). Without it Gazebo's Ogre2
# crashes, so fall back to Mesa software rendering (slow but works).
if [ -e /dev/dri/renderD128 ] && [ ! -r /dev/dri/renderD128 ]; then
  export GALLIUM_DRIVER=llvmpipe
  export EGL_PLATFORM=surfaceless
  echo "px4_env: no access to /dev/dri/renderD128 -> Gazebo uses software rendering." \
       "Fix: sudo usermod -aG render,video \$USER, then log out and back in." >&2
fi

# gz-msgs Python bindings (gz.msgs10, generated with protoc 3.12) fail to import with protobuf
# >= 4 ("Descriptors cannot be created directly"); a newer protobuf in ~/.local (e.g. pulled in
# by tensorboard) would shadow the system one. external/pyproto holds protobuf 3.20.3 for this
# project only (PYTHONPATH comes before ~/.local).
case ":${PYTHONPATH:-}:" in
  *":$ROSWS/external/pyproto:"*) ;;
  *) [ -d "$ROSWS/external/pyproto" ] && export PYTHONPATH="$ROSWS/external/pyproto:$PYTHONPATH" ;;
esac

# Python venv with pycollada for the human-model generators (src/sar_sim/scripts, train/make_humans.py)
alias px4-venv='source "$ROSWS/external/px4-venv/bin/activate"'
