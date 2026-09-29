# Source (bash) before any ROS command for Worldline-G1:   source /work/worldline-g1/nav2/ros_env.sh
# DDS isolation from the Unitree G1 topics (unitree_sdk2 = CycloneDDS, domain 0; the SONIC deploy and P1 use
# domain 0 on lo, build-phase tests domain 7): our ROS 2 graph uses Fast DDS, localhost only, on
#   ROS_DOMAIN_ID = 42 + port_offset // 100      (offset 0 -> 42, +700 -> 49, +900 -> 51)
# so stacks at different port offsets never share /cmd_vel, /tf, /odom, /map or navigate_to_pose.
# WL_ROS_DOMAIN_ID overrides it; the offset is read from WL_PORT_OFFSET (nav2/up.sh sets WL_ROS_DOMAIN_ID itself).
set +u
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=${WL_ROS_DOMAIN_ID:-$((42 + ${WL_PORT_OFFSET:-0} / 100))}
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY          # deprecated in Jazzy (ROS_AUTOMATIC_DISCOVERY_RANGE replaces it)
unset CYCLONEDDS_URI              # never inherit the Unitree CycloneDDS config
export RCUTILS_COLORIZED_OUTPUT=0
