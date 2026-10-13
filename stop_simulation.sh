#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# Stops the simulation started by run_simulation.sh (all components and their terminal windows).
exec "$(cd "$(dirname "$0")" && pwd)/eval/stop_gui_demo.sh" "$@"
