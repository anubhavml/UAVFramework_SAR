#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# Stops the desktop demonstration started by run_gui_demo.sh (all its windows' processes).
# Patterns are anchored so they match only the demo's own processes, not a shell that runs this.
pkill -f "^bash [^ ]*run_gui_demo\.sh -" 2>/dev/null                # a demo still starting
pkill -f "^python3 (flight|demo_flight)\.py " 2>/dev/null
sleep 1; pkill -9 -f "^python3 (flight|demo_flight)\.py " 2>/dev/null    # rclpy catches SIGTERM
pkill -INT -f "^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch drone_control sar_hitl" 2>/dev/null; sleep 3
pkill -9 -f "^/usr/bin/python3 .*/lib/drone_control/|^/opt/ros/humble/lib/ros_gz_bridge/parameter_bridge" 2>/dev/null
pkill -9 -f "^\./build/px4_sitl_default/bin/px4|/px4_sitl_default/bin/px4 -d" 2>/dev/null
pkill -9 -f "^gz sim" 2>/dev/null; pkill -9 -f "^MicroXRCEAgent" 2>/dev/null
# QGroundControl ignores SIGTERM while a vehicle is connected: ask, then force
pkill -f "^/tmp/\.mount_QGroun[^ ]*/QGroundControl" 2>/dev/null; sleep 2
pkill -9 -f "^/tmp/\.mount_QGroun[^ ]*/QGroundControl" 2>/dev/null
pkill -9 -f "QGroundControl-4\.4\.4\.AppImage$" 2>/dev/null
# close the demo's terminal windows (their shells wait at "press Enter to close")
sleep 2
pkill -f "^bash -c .*(SAR_DEMO_WINDOW|ended - press Enter to close)" 2>/dev/null
sleep 1
pkill -9 -f "^bash -c .*(SAR_DEMO_WINDOW|ended - press Enter to close)" 2>/dev/null
pkill -9 -f "^bash -c .*SAR_DEMO_BG" 2>/dev/null                      # --no-terminals
echo "demo stopped, its terminal windows closed"
