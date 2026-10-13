#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# One gesture-switch flight (headless):  run_flight.sh <name> <world> <px4 model> [launch args...]
#   run_flight.sh city city x500_depth                          # Algorithm 1 of the paper, RGB
#   run_flight.sh city_depth city x500_depth input_mode:=depth
#   run_flight.sh city_robust city x500_depth robust:=true      # + robustness extensions (not in the paper)
#   run_flight.sh city_d435i city x500_d435i sensor:=d435i
#   run_flight.sh city_vis20 city x500_depth visibility:=20
#   RECORD=1 run_flight.sh city_video city x500_depth   (third-person camera + video, written to
#                                                        data/media/<name>, not to the results)
# The drone is spawned 5 m in front of the person; max_turn 20 deg keeps the person in view.
NAME=$1; W=$2; MODEL=$3; shift 3; LARGS="$*"
EVAL=$(cd "$(dirname "$0")" && pwd)
source "$EVAL/../px4_env.sh" >/dev/null 2>&1
export GZ_SIM_RESOURCE_PATH="$EVAL/models:$GZ_SIM_RESOURCE_PATH"
OUT="${FLIGHTS_DIR:-$EVAL/data/flights}/$NAME"     # FLIGHTS_DIR: another set of flights
[ "${RECORD:-0}" = 1 ] && OUT="${MEDIA_DIR:-$EVAL/data/media}/$NAME"
if pgrep -f "^(gz sim|MicroXRCEAgent)|px4_sitl_default/bin/px4" >/dev/null; then
  echo "ABORT: a simulation is already running"; exit 1; fi
rm -rf "$OUT"; mkdir -p "$OUT"
cleanup() {
  pkill -INT -f "ros2 launch drone_control sar_hitl" 2>/dev/null; sleep 3
  pkill -9 -f "lib/drone_control/|lib/ros_gz_bridge/parameter_bridge" 2>/dev/null
  pkill -9 -f "px4_sitl_default/bin/px4" 2>/dev/null
  pkill -9 -f "^gz sim" 2>/dev/null; pkill -9 -f "^MicroXRCEAgent" 2>/dev/null
}
trap cleanup EXIT
cd "$EVAL"
sdf=$(python3 -c "import simtools as s; print(s.make_eval_world('$W', '$EVAL/data/worlds', rig=False, suffix='_flight'))")
[ "${RECORD:-0}" = 1 ] && sdf=$(python3 media.py world "$W")
pose=$(python3 -c "import simtools as s; _, (x, y, _), _ = s.world_info('$W'); print(f'{x - ${SPAWN_D:-5}:.2f},{y:.2f},0.2,0,0,0')")
echo "$NAME: world $W, model $MODEL, spawn $pose, launch args: $LARGS" | tee "$OUT/info.txt"
MicroXRCEAgent udp4 -p 8888 > "$OUT/agent.log" 2>&1 &
gz sim -s -r --headless-rendering "$sdf" > "$OUT/gz.log" 2>&1 &
sleep 8
( cd "$PX4_DIR" && env PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4002 PX4_SIM_MODEL=gz_$MODEL \
    PX4_GZ_WORLD=$W PX4_GZ_MODEL_POSE=$pose HEADLESS=1 ./build/px4_sitl_default/bin/px4 -d \
    > "$OUT/px4.log" 2>&1 & )
sleep 15
# height from the barometer only: with the defaults (GPS height reference and GPS altitude fused,
# baro noise 3.5 m) the estimated height wanders by ~1 m in SITL and the drone's true hover height
# drifts with it. Gazebo's barometer noise is 0.01 Pa (~1 mm), so 0.2 m is still conservative.
# GPS still gives horizontal position. ekf2 is restarted so the change applies before arming.
( cd "$PX4_DIR/build/px4_sitl_default/bin" && ./px4-param set EKF2_HGT_REF 0 &&
  ./px4-param set EKF2_GPS_CTRL 5 && ./px4-param set EKF2_BARO_NOISE 0.2 &&
  ./px4-ekf2 stop && sleep 1 && ./px4-ekf2 start ) \
  >> "$OUT/px4.log" 2>&1
sleep 10
ros2 launch drone_control sar_hitl.launch.py show:=false max_turn:=20 $LARGS > "$OUT/launch.log" 2>&1 &
[ "${RECORD:-0}" = 1 ] && { python3 media.py record "$W" "$OUT" > "$OUT/record.log" 2>&1 & REC=$!; }
timeout -k 10 900 python3 ${DRIVER:-flight.py} "$W" "$OUT" 2>&1 | grep -v -i warn | tail -3
if [ -n "$REC" ]; then
  kill -TERM $REC; sleep 3; kill -9 $REC 2>/dev/null
  IMAGEIO_FFMPEG_PATH="$EVAL/../external/pylib" python3 media.py video "$OUT" 2>&1 | grep -v -i warn
fi
echo "$NAME: done"
