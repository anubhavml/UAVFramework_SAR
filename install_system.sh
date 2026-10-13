#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Part of the ROS 2 human-in-the-loop UAV search-and-rescue framework; licensed under the
# GNU Affero General Public License v3.0 (see LICENSE in the repository root).
# System part of the installation (needs sudo): ROS 2 Humble + Gazebo Harmonic + PX4 v1.15 build
# dependencies (+ x11-utils for the demo's window handling). Needs Ubuntu 22.04 (Humble and these
# Gazebo packages are built for 22.04 "jammy") and an NVIDIA driver for GPU rendering / inference.
#   bash install_system.sh          then log out and back in, then (no sudo):   bash install_workspace.sh
set -euo pipefail

. /etc/os-release
if [ "${VERSION_ID:-}" != "22.04" ]; then
  echo "This machine runs $PRETTY_NAME; ROS 2 Humble + Gazebo Harmonic need Ubuntu 22.04."
  echo "Use a 22.04 machine or a 22.04 Docker container (with the NVIDIA container toolkit)."
  exit 1
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv \
  || echo "WARNING: nvidia-smi not found - install the NVIDIA driver first"

echo "==> [1/6] base tools, locale"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install software-properties-common curl wget \
    gnupg lsb-release locales
sudo locale-gen en_US en_US.UTF-8
sudo add-apt-repository -y universe

echo "==> [2/6] ROS 2 Humble (desktop) + tools"
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install ros-humble-desktop ros-dev-tools \
    python3-colcon-common-extensions ros-humble-cv-bridge ros-humble-image-transport \
    ros-humble-vision-opencv ros-humble-rqt-image-view libgl1

echo "==> [3/6] PX4 v1.15 build + simulation dependencies"
sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    astyle build-essential cmake cppcheck file g++ gcc gdb git lcov libfuse2 \
    libxml2-dev libxml2-utils make ninja-build python3 python3-dev python3-pip \
    python3-setuptools python3-wheel rsync shellcheck unzip zip bc dmidecode \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-gl \
    libeigen3-dev libgstreamer-plugins-base1.0-dev libimage-exiftool-perl \
    libopencv-dev pkg-config protobuf-compiler python3.10-venv x11-utils python3-tk

echo "==> [4/6] Gazebo Harmonic + ROS 2 <-> Gazebo bridge (no Gazebo Classic)"
sudo apt-get remove -y gazebo gazebo-common libgazebo11 ros-humble-gazebo-ros-pkgs 2>/dev/null || true
sudo wget -q https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install gz-harmonic ros-humble-ros-gzharmonic

echo "==> [5/6] GPU rendering for headless Gazebo cameras"
sudo usermod -aG render,video "$USER"

echo "==> [6/6] ROS environment in ~/.bashrc"
grep -q "/opt/ros/humble/setup.bash" ~/.bashrc || echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

echo
echo "Done. Log out and back in (for the render group), then:  bash install_workspace.sh"
