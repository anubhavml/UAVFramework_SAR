#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# Demonstration videos of Algorithm 1 (stock model, gesture rules) for each gesture:
#   synthetic: one recorded Gazebo + PX4 SITL + ROS 2 flight per gesture and camera (RGB, depth)
#   real:      UAV-GESTURE clips, RGB and estimated depth, and one longer RGB video in which the
#              approach is emulated by digital zoom (gesture_videos.py approach)
# Only the .mp4 files go to OUT; flight recordings stay in WORK.
#   setsid nohup eval/run_gesture_videos.sh <work dir> > <log> 2>&1 < /dev/null &
EVAL=$(cd "$(dirname "$0")" && pwd)
WORK=$1
OUT=$EVAL/data/media/gesture_videos_algorithm1
SYN=$OUT/synthetic_gazebo_px4_ros2_flight
REAL=$OUT/real_uavgesture_dataset
source "$EVAL/../px4_env.sh" >/dev/null 2>&1
export SAR_GESTURES=right,left,neutral,both_out,both_up,rightout_leftup,leftout_rightup
GS="right left neutral both_out both_up rightout_leftup leftout_rightup"
cd "$EVAL"
echo "== flights start $(date) (Gazebo + PX4 SITL + ROS 2 + Micro XRCE-DDS + ros_gz_bridge)"
for mode in rgb depth; do
  for g in $GS; do
    n=${g}_${mode}
    grep -qs '"kind": "end"' "$WORK/$n/log.jsonl" && continue
    extra=""; [ $mode = depth ] && extra="input_mode:=depth"
    echo "-- $n $(date +%T)"
    FLIGHT_SCHEDULE=neutral:6,$g:16,neutral:6 MEDIA_DIR=$WORK RECORD=1 \
      ./run_flight.sh $n city x500_depth robust:=false device:=0 $extra 2>&1 | grep -E "done|ABORT|Error" | tail -2
    sleep 5
  done
done
echo "== flights done, simulator stopped $(date)"
for mode in rgb depth; do
  sub=$([ $mode = rgb ] && echo rgb_camera || echo depth_camera_oakd_lite)
  for g in $GS; do
    f=$(python3 -c "import gesture_videos as v; print(v.GESTURES['$g'][0])")
    python3 gesture_videos.py synthetic "$WORK/${g}_${mode}" $g $mode "$SYN/$sub/$f.mp4" 2>&1 | grep -v -i warn | tail -1
  done
done
python3 gesture_videos.py real rgb "$REAL/rgb_camera" 2>&1 | grep -v -i warn
python3 gesture_videos.py real depth "$REAL/depth_estimated_from_rgb" 2>&1 | grep -v -i warn
python3 gesture_videos.py approach "$REAL/approach_rgb_camera.mp4" 2>&1 | grep -v -i warn
echo "== gesture videos done $(date)"
