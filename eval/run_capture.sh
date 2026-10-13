#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# Capture the evaluation frames for the given worlds (default: all six), headless.
#   eval/run_capture.sh                 # forest city wall forest_fog city_fog wall_fog
#   eval/run_capture.sh city            # one world
#   SEED=1 OUT=$ROSWS/eval/data/captures_test eval/run_capture.sh   # held-out test placements
# Particle-fog worlds: 2 frames per placement/gesture, 30 s sim warm-up for the fog to build up.
# Resumable: frames already saved are skipped.
EVAL=$(cd "$(dirname "$0")" && pwd)
source "$EVAL/../px4_env.sh" >/dev/null 2>&1
export GZ_SIM_RESOURCE_PATH="$EVAL/models:$GZ_SIM_RESOURCE_PATH"
WORLDS=${*:-forest city wall forest_fog city_fog wall_fog}
SEED=${SEED:-0}; OUT=${OUT:-$EVAL/data/captures}
mkdir -p "$EVAL/data/worlds" "$EVAL/data/logs"
for w in $WORLDS; do
  if pgrep -f "^gz sim" >/dev/null; then echo "ABORT: a gz sim is already running"; exit 1; fi
  sdf=$(cd "$EVAL" && python3 -c "import simtools as s; print(s.make_eval_world('$w', '$EVAL/data/worlds'))")
  gz sim -s -r --headless-rendering "$sdf" > "$EVAL/data/logs/gz_$w.log" 2>&1 &
  GZ=$!
  case $w in *_fog) opts="--frames 2 --warmup 30";; *) opts="--frames 1 --warmup 3";; esac
  ( cd "$EVAL" && python3 capture.py "$w" $opts --seed "$SEED" --out "$OUT" ) 2>&1 | grep -v -i "warn" | tail -3
  kill -9 $GZ 2>/dev/null; pkill -9 -f "^gz sim -s -r --headless-rendering $EVAL/data/worlds/" 2>/dev/null
  sleep 3
done
