# Source (bash) before any ROS command for Worldline-G1:   source /work/worldline-g1/nav2/ros_env.sh
# DDS isolation from the Unitree G1 topics (unitree_sdk2 = CycloneDDS, domain 0; the SONIC deploy and P1 use
# domain 0 on lo, build-phase tests domain 7): our ROS 2 graph is domain 42 with Fast DDS, localhost only.
set +u
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=${WL_ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset ROS_LOCALHOST_ONLY          # deprecated in Jazzy (ROS_AUTOMATIC_DISCOVERY_RANGE replaces it)
unset CYCLONEDDS_URI              # never inherit the Unitree CycloneDDS config
export RCUTILS_COLORIZED_OUTPUT=0
