#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# Starts the live simulation: Micro XRCE-DDS Agent, Gazebo, PX4 SITL, QGroundControl, the ROS 2
# gesture and control nodes and the demo driver, each in its own terminal window.
#   ./run_simulation.sh             start window: choose the options there
#   ./run_simulation.sh --help      all options (source, mode, world, fog, distances, ...)
#   ./stop_simulation.sh            stops everything
exec "$(cd "$(dirname "$0")" && pwd)/eval/run_gui_demo.sh" "$@"
