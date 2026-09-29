"""robot/: the façade the runtime talks to (RobotBridge = G1Robot), the profile loader and the stack factory,
and the runtime-side body clients (SonicBody over wl-body, LiteBody in process).

    from robot.factory import build
    world, robot, frames = build("lite", "procthor-train-38", clock, log)
"""
