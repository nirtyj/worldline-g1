#!/usr/bin/env bash
# Install ROS 2 Jazzy (ros-base) + Nav2 on the box (Ubuntu 24.04 noble), niced so a running real-time controller
# test is not disturbed. Idempotent: re-running only installs what is missing.
#
#   bash nav2/install_ros.sh            # run inside tmux (nav2-install), takes ~5-10 min
#
# Packages (packages.ros.org, main): ros-base, navigation2 (all Nav2 servers incl. map_server, AMCL, BT navigator,
# behaviours, smoother, velocity smoother, collision monitor, MPPI, RPP, Smac, NavFn), nav2-bringup, tf2 tools.
# No desktop / rviz / Gazebo. rmw_fastrtps_cpp is the Jazzy default and comes with ros-base.
set -euo pipefail
NICE="nice -n 15 ionice -c3"
export DEBIAN_FRONTEND=noninteractive

if [[ ! -f /etc/apt/sources.list.d/ros2.list && ! -f /etc/apt/sources.list.d/ros2.sources ]]; then
  echo "[install_ros] adding packages.ros.org (ros2-apt-source)"
  sudo $NICE apt-get update -q
  sudo $NICE apt-get install -y -q --no-install-recommends curl ca-certificates software-properties-common
  sudo add-apt-repository -y universe >/dev/null
  V=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
  if [[ -n "$V" ]]; then
    curl -fsSL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.$(. /etc/os-release && echo "$VERSION_CODENAME")_all.deb"
    sudo dpkg -i /tmp/ros2-apt-source.deb
  else  # fallback: classic key + list
    sudo curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
      | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
  fi
fi

sudo $NICE apt-get update -q
PKGS=(
  ros-jazzy-ros-base
  ros-jazzy-navigation2
  ros-jazzy-nav2-bringup
  ros-jazzy-nav2-mppi-controller
  ros-jazzy-nav2-regulated-pure-pursuit-controller
  ros-jazzy-nav2-smac-planner
  ros-jazzy-nav2-navfn-planner
  ros-jazzy-nav2-map-server
  ros-jazzy-nav2-lifecycle-manager
  ros-jazzy-nav2-simple-commander
  ros-jazzy-tf2-ros
  ros-jazzy-tf2-tools
  ros-jazzy-rmw-fastrtps-cpp
  ros-jazzy-ros2cli-common-extensions
  python3-zmq
  python3-msgpack
  python3-yaml
  python3-numpy
)
echo "[install_ros] installing ${#PKGS[@]} packages (niced)"
sudo $NICE apt-get install -y -q --no-install-recommends "${PKGS[@]}"
echo "[install_ros] done"
dpkg -l | grep -c '^ii  ros-jazzy' | xargs echo "[install_ros] ros-jazzy packages installed:"
