"""Nav2 for Worldline-G1: controller, planner, behaviour server, BT navigator, velocity smoother, lifecycle manager.

    source nav2/ros_env.sh
    ros2 launch nav2/launch/wl_nav2.launch.py [controller:=rpp|mppi] [use_composition:=true] [velocity_smoother:=true]
                                              [log_level:=info] [params_file:=...]

Not launched (on purpose): map_server (nav2/ros_bridge.py publishes the latched /map from P1's occupancy), AMCL
(localisation = ground truth), smoother_server, waypoint_follower, route_server, docking, collision_monitor (no sensors
in M1; TODO with the depth/LiDAR layer). Topics: controller -> cmd_vel_nav -> velocity_smoother -> cmd_vel ->
ros_bridge -> wl-body op velocity. use_composition:=true runs everything in ONE component_container_isolated process
(one DDS participant, least CPU); false = one process per server (easier per-node CPU accounting).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
LIFECYCLE_NODES = ["controller_server", "planner_server", "behavior_server", "velocity_smoother", "bt_navigator"]


def _setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    controller = lc("controller").lower()
    compose = lc("use_composition").lower() in ("1", "true", "yes")
    smoother = lc("velocity_smoother").lower() in ("1", "true", "yes")
    log_level = lc("log_level")
    params = [lc("params_file")]
    if controller == "mppi":
        params.append(os.path.join(ROOT, "params", "controller_mppi.yaml"))
    elif controller != "rpp":
        raise RuntimeError(f"controller must be mppi or rpp, not {controller!r}")
    bt_xml = lc("bt_xml")
    nodes = list(LIFECYCLE_NODES) if smoother else [n for n in LIFECYCLE_NODES if n != "velocity_smoother"]
    ctrl_remap = [("cmd_vel", "cmd_vel_nav")] if smoother else []
    spec = [
        ("nav2_controller", "nav2_controller::ControllerServer", "controller_server", "controller_server",
         ctrl_remap, {}),
        ("nav2_planner", "nav2_planner::PlannerServer", "planner_server", "planner_server", [], {}),
        ("nav2_behaviors", "behavior_server::BehaviorServer", "behavior_server", "behavior_server", ctrl_remap, {}),
        ("nav2_bt_navigator", "nav2_bt_navigator::BtNavigator", "bt_navigator", "bt_navigator", [],
         {"default_nav_to_pose_bt_xml": bt_xml}),
    ]
    if smoother:
        spec.append(("nav2_velocity_smoother", "nav2_velocity_smoother::VelocitySmoother", "velocity_smoother",
                     "velocity_smoother", [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")], {}))
    spec.append(("nav2_lifecycle_manager", "nav2_lifecycle_manager::LifecycleManager", "lifecycle_manager",
                 "lifecycle_manager_navigation", [], {"autostart": True, "node_names": nodes}))
    if compose:
        return [ComposableNodeContainer(
            name="nav2_container", namespace="", package="rclcpp_components",
            executable="component_container_isolated", output="screen",
            arguments=["--ros-args", "--log-level", log_level],
            # The full params files also go to the CONTAINER process (--params-file): the costmap sub-nodes
            # (local_costmap/local_costmap, global_costmap/global_costmap) are created by the servers with default
            # NodeOptions, i.e. from the process-global arguments; launch_ros only hands each component the keys
            # matching its own name.
            parameters=params,
            composable_node_descriptions=[
                ComposableNode(package=pkg, plugin=plugin, name=name, parameters=params + [extra],
                               remappings=remap)
                for pkg, plugin, _exe, name, remap, extra in spec])]
    return [Node(package=pkg, executable=exe, name=name, output="screen", parameters=params + [extra],
                 remappings=remap, arguments=["--ros-args", "--log-level", log_level])
            for pkg, _plugin, exe, name, remap, extra in spec]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("controller", default_value="rpp", description="rpp (default) | mppi (Omni)"),
        DeclareLaunchArgument("use_composition", default_value="true"),
        DeclareLaunchArgument("velocity_smoother", default_value="true"),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(ROOT, "params", "nav2_g1.yaml")),
        DeclareLaunchArgument("bt_xml", default_value=os.path.join(ROOT, "bt", "navigate_humanoid.xml")),
        OpaqueFunction(function=_setup),
    ])
