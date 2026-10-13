# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
cmake_minimum_required(VERSION 3.8)
project(sar_sim)
find_package(ament_cmake REQUIRED)
install(DIRECTORY worlds models DESTINATION share/${PROJECT_NAME})
ament_package()
