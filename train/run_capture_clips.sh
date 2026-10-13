#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# Capture the T4 clips (capture_clips.py; headless, resumable; Gazebo only, no PX4 / ROS).
#   train/run_capture_clips.sh                    # all worlds in train/worlds
#   train/run_capture_clips.sh train_0 val_0_fog  # selected worlds
# Long: start with  mkdir -p train/data && setsid nohup train/run_capture_clips.sh > train/data/capture.log 2>&1 &
T=$(cd "$(dirname "$0")" && pwd)
source "$T/../px4_env.sh" >/dev/null 2>&1
export GZ_SIM_RESOURCE_PATH="$T/models:$T/../eval/models:$GZ_SIM_RESOURCE_PATH"
WORLDS=${*:-$(cd "$T/worlds" && ls *.sdf | sed 's/\.sdf$//' | sort)}
N_TRAIN=${N_TRAIN:-40}; N_VAL=${N_VAL:-15}
mkdir -p "$T/data/logs"
for w in $WORLDS; do
  if pgrep -f "^gz sim" >/dev/null; then echo "ABORT: a gz sim is already running"; exit 1; fi
  case $w in val_*) n=$N_VAL;; *) n=$N_TRAIN;; esac
  case $w in *_fog) warm=30;; *) warm=3;; esac
  gz sim -s -r --headless-rendering "$T/worlds/$w.sdf" > "$T/data/logs/gz_clips_$w.log" 2>&1 &
  GZ=$!
  ( cd "$T" && timeout -k 10 7200 python3 capture_clips.py "$w" --n "$n" --warmup "$warm" --out "${OUT:-$T/data/clips}" ) 2>&1 \
    | grep -v -i "warn"
  kill -9 $GZ 2>/dev/null; pkill -9 -f "^gz sim -s -r --headless-rendering $T/worlds/" 2>/dev/null
  sleep 3
done
echo "clips: all done"
