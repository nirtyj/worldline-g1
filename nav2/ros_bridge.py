#!/usr/bin/env python3
"""wl ROS 2 bridge: the ONE ROS process of Worldline-G1 (system python3 3.12 + /opt/ros/jazzy). Talks ZMQ to the rest.

    source nav2/ros_env.sh && python3 nav2/ros_bridge.py [--port-offset N] [--log-dir DIR]     (nav2/up.sh does this)

The body venv is Python 3.11 and cannot import rclpy (Jazzy is built for 3.12), so every ROS dependency lives here.

DDS isolation (nav2/ros_env.sh): ROS_DOMAIN_ID=42, RMW_IMPLEMENTATION=rmw_fastrtps_cpp,
ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST. The G1 low-level topics (rt/lowcmd, rt/lowstate, ...) are unitree_sdk2 /
CycloneDDS on domain 0, so Nav2 traffic can never mix with them (different domain AND different DDS vendor).

ZMQ side (all ports + offset, docs/contracts/m1.md §0):
  SUB    P1 gt.pose      5601  -> /odom (nav_msgs/Odometry, twist in base_link) + TF odom->base_link, every sample (50 Hz)
  REQ    P1 REP          5600  -> get_occupancy -> /map (latched OccupancyGrid, the RAW grid; Nav2 inflates)
  REP    nav_bridge      5620  <- body go_to backend nav2: ping | goto | cancel | status | plan | reload_map | stats
  DEALER body ROUTER     5610  -> op velocity {vx, vy, wz, goal_id, t_wall} for every /cmd_vel while a goal is active
ROS side:
  static TF map->odom = identity (localisation = ground truth; the later no-GT stage replaces this with AMCL/KISS-ICP)
  frames: base_link = the pelvis projected onto the floor plane (x, y, yaw only; roll/pitch/height dropped)
  action clients navigate_to_pose, compute_path_to_pose; /speed_limit; service lifecycle_manager_navigation/is_active

goto = ComputePathToPose pre-check (so no_path / goal_in_obstacle come back synchronously, with the plan) then
NavigateToPose with the same goal (goal yaw = the path's arrival heading when the caller gives none; goal snapped to the
path end when the planner's tolerance moved it). Nav2 error codes map to the body's failure reasons (REASONS below).
"""

from __future__ import annotations

import argparse
import array
import collections
import json
import math
import os
import queue
import resource
import signal
import sys
import threading
import time

import msgpack
import numpy as np
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wl_map import load_occupancy, to_ros_values, write_map_server_files  # noqa: E402

import rclpy  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402

from action_msgs.msg import GoalStatus  # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist, TwistStamped  # noqa: E402
from nav2_msgs.action import ComputePathToPose, NavigateToPose  # noqa: E402
from nav2_msgs.msg import SpeedLimit  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Odometry  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster  # noqa: E402

BASE_PORTS = {"p1_rep": 5600, "p1_pose": 5601, "body_ctl": 5610, "nav_bridge": 5620}

# Nav2 (Jazzy 1.3) error codes: FollowPath 100-107, ComputePathToPose 200-208 (NavigateToPose reports the
# highest-priority one of the BT's error_code_names).
ERR_NAMES = {0: "NONE", 100: "FOLLOW_UNKNOWN", 101: "INVALID_CONTROLLER", 102: "FOLLOW_TF_ERROR", 103: "INVALID_PATH",
             104: "PATIENCE_EXCEEDED", 105: "FAILED_TO_MAKE_PROGRESS", 106: "NO_VALID_CONTROL",
             107: "CONTROLLER_TIMED_OUT", 200: "PLAN_UNKNOWN", 201: "INVALID_PLANNER", 202: "PLAN_TF_ERROR",
             203: "START_OUTSIDE_MAP", 204: "GOAL_OUTSIDE_MAP", 205: "START_OCCUPIED", 206: "GOAL_OCCUPIED",
             207: "PLAN_TIMEOUT", 208: "NO_VALID_PATH"}
REASONS = {100: "nav2_failed", 101: "nav2_config", 102: "tf_error", 103: "no_path", 104: "stuck", 105: "stuck",
           106: "stuck", 107: "timeout", 200: "no_path", 201: "nav2_config", 202: "tf_error", 203: "start_in_obstacle",
           204: "no_path", 205: "start_in_obstacle", 206: "goal_in_obstacle", 207: "no_path", 208: "no_path"}
TERMINAL = ("succeeded", "failed", "canceled")


def ports(offset: int) -> dict:
    return {k: v + offset for k, v in BASE_PORTS.items()}


def stamp(t: float) -> TimeMsg:
    sec = int(math.floor(t))
    return TimeMsg(sec=sec, nanosec=int((t - sec) * 1e9))


def yaw_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)       # x, y, z, w


def quat_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Goal:
    def __init__(self, gid: str, x: float, y: float, yaw, speed):
        self.id, self.x, self.y, self.yaw_req, self.speed = gid, x, y, yaw, speed
        self.yaw = yaw
        self.state = "planning"
        self.reason = self.error_code = self.error_name = self.error_msg = None
        self.handle = None
        self.cancel_requested = False
        self.feedback: dict = {}
        self.plan: dict = {}
        self.snapped = False
        self.t_start = time.time()
        self.t_active = self.t_end = None
        self.cmd_fwd = 0
        self.cmd_rej = 0
        self.last_cmd = None
        self.accepted_ev = threading.Event()
        self.done_ev = threading.Event()

    def brief(self) -> dict:
        return {"id": self.id, "state": self.state, "reason": self.reason, "error_code": self.error_code,
                "error_name": self.error_name, "error_msg": self.error_msg,
                "goal": {"x": self.x, "y": self.y, "yaw": self.yaw, "yaw_requested": self.yaw_req,
                         "snapped": self.snapped},
                "speed_limit": self.speed, "feedback": self.feedback, "cmd_vel_forwarded": self.cmd_fwd,
                "cmd_vel_rejected": self.cmd_rej, "t_start": self.t_start, "t_active": self.t_active,
                "t_end": self.t_end,
                "duration_s": None if self.t_active is None else round((self.t_end or time.time()) - self.t_active, 3)}


class Bridge(Node):
    def __init__(self, a):
        super().__init__("wl_ros_bridge")
        self.a = a
        self.P = ports(a.port_offset)
        self.zctx = zmq.Context.instance()
        self.running = True
        self.lock = threading.Lock()
        self.log_dir = a.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._cmd_log = open(os.path.join(self.log_dir, "cmd_vel.jsonl"), "a", buffering=1)
        self._goal_log = open(os.path.join(self.log_dir, "goals.jsonl"), "a", buffering=1)
        # ROS I/O
        self.tfb = TransformBroadcaster(self)
        self.stfb = StaticTransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.map_pub = self.create_publisher(OccupancyGrid, "map", latched)
        self.speed_pub = self.create_publisher(SpeedLimit, "speed_limit", 10)
        if a.cmd_vel_stamped:
            self.create_subscription(TwistStamped, "cmd_vel", lambda m: self._on_cmd_vel(m.twist), 10)
        else:
            self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.plan_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.active_cli = self.create_client(Trigger, a.lifecycle_manager + "/is_active")
        self._jobs: queue.Queue = queue.Queue()
        self._gc = self.create_guard_condition(self._drain_jobs)
        self.create_timer(1.0, self._check_nav2)
        self.create_timer(0.2, self._drain_body_replies)
        self._publish_static_tf()
        # state
        self.nav2_active = False
        self.nav2_detail = "starting"
        self.map_info: dict | None = None
        self.map_msg: OccupancyGrid | None = None
        self.map_occ: np.ndarray | None = None
        self.map_geo = None
        self.pose = None
        self.pose_mono = 0.0
        self.pose_ts: collections.deque = collections.deque(maxlen=120)
        self.goals: collections.OrderedDict[str, Goal] = collections.OrderedDict()
        self.active: Goal | None = None
        self.stats = collections.Counter()
        self.body_last_error = None
        # body DEALER (only used from the executor thread): IMMEDIATE + tiny HWM + NOBLOCK -> never queue stale cmds
        self.dealer = self.zctx.socket(zmq.DEALER)
        self.dealer.setsockopt(zmq.LINGER, 0)
        self.dealer.setsockopt(zmq.IMMEDIATE, 1)
        self.dealer.setsockopt(zmq.SNDHWM, 4)
        self.dealer.connect(f"tcp://127.0.0.1:{self.P['body_ctl']}")
        self.threads = [threading.Thread(target=f, name=n, daemon=True) for f, n in
                        ((self._pose_loop, "gt-pose"), (self._rep_loop, "rep"), (self._map_loop, "map"))]
        for t in self.threads:
            t.start()
        self.log(f"up: ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')} RMW={os.environ.get('RMW_IMPLEMENTATION')} "
                 f"ports={self.P} log_dir={self.log_dir}")

    # -- helpers -----------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} [bridge] {msg}", flush=True)

    def goal_event(self, g: Goal, ev: str, **extra) -> None:
        rec = {"t_wall": time.time(), "event": ev, **g.brief(), **extra}
        self._goal_log.write(json.dumps(rec, default=str) + "\n")
        if ev not in ("feedback",):
            self.log(f"goal {g.id} {ev} state={g.state} reason={g.reason} {extra if extra else ''}")

    def in_ros(self, fn, timeout: float = 5.0):
        """Run fn() on the executor thread (all rclpy calls except publishing happen there)."""
        ev, box = threading.Event(), {}

        def job():
            try:
                box["r"] = fn()
            except Exception as e:  # noqa: BLE001
                box["e"] = e
            finally:
                ev.set()
        self._jobs.put(job)
        self._gc.trigger()
        if not ev.wait(timeout):
            raise TimeoutError("executor did not run the job")
        if "e" in box:
            raise box["e"]
        return box.get("r")

    def _drain_jobs(self) -> None:
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            job()

    def _publish_static_tf(self) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.rotation.w = 1.0
        self.stfb.sendTransform(t)

    # -- gt.pose -> /odom + TF ---------------------------------------------------------------------
    def _pose_loop(self) -> None:
        s = self.zctx.socket(zmq.SUB)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVHWM, 50)
        s.setsockopt(zmq.SUBSCRIBE, b"gt.pose")
        s.connect(f"tcp://127.0.0.1:{self.P['p1_pose']}")
        odom = Odometry()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        for i in (0, 7, 35):
            odom.pose.covariance[i] = 1e-4
            odom.twist.covariance[i] = 1e-3
        tf = TransformStamped()
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        while self.running:
            if not s.poll(200):
                continue
            try:
                frames = s.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                continue
            payload = frames[-1] if len(frames) >= 2 else frames[0][len(b"gt.pose"):]
            try:
                d = msgpack.unpackb(payload, raw=False)
                x, y = float(d["base_pos"][0]), float(d["base_pos"][1])
                yaw = d.get("yaw")
                if yaw is None:
                    w, qx, qy, qz = d["base_quat_wxyz"]
                    yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                yaw = float(yaw)
                v = d.get("base_lin_vel_w") or [0.0, 0.0, 0.0]
                w = d.get("base_ang_vel_w") or [0.0, 0.0, 0.0]
            except Exception as e:  # noqa: BLE001
                self.stats["bad_pose"] += 1
                if self.stats["bad_pose"] < 5:
                    self.log(f"bad gt.pose: {e!r}")
                continue
            now = time.time()
            t = d.get("t_wall")
            t = now if (t is None or abs(now - float(t)) > 0.5) else float(t)
            st = stamp(t)
            qx, qy, qz, qw = yaw_quat(yaw)
            tf.header.stamp = st
            tf.transform.translation.x, tf.transform.translation.y = x, y
            tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
            odom.header.stamp = st
            odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
            c, sn = math.cos(yaw), math.sin(yaw)
            odom.twist.twist.linear.x = c * float(v[0]) + sn * float(v[1])
            odom.twist.twist.linear.y = -sn * float(v[0]) + c * float(v[1])
            odom.twist.twist.angular.z = float(w[2])
            try:
                self.tfb.sendTransform(tf)
                self.odom_pub.publish(odom)
            except Exception as e:  # noqa: BLE001  (shutdown race)
                if self.running:
                    self.log(f"publish failed: {e!r}")
            with self.lock:
                self.pose = (x, y, yaw)
                self.pose_mono = time.monotonic()
                self.pose_ts.append(self.pose_mono)
            self.stats["poses"] += 1
        s.close(0)

    def pose_rate(self) -> float:
        with self.lock:
            ts = list(self.pose_ts)
        now = time.monotonic()
        ts = [t for t in ts if now - t < 2.0]
        return 0.0 if len(ts) < 2 else (len(ts) - 1) / max(1e-6, ts[-1] - ts[0])

    # -- occupancy -> /map -------------------------------------------------------------------------
    def _p1_call(self, op: str, timeout_s: float = 30.0, **args) -> dict:
        s = self.zctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        s.connect(f"tcp://127.0.0.1:{self.P['p1_rep']}")
        try:
            s.send(json.dumps({"op": op, **args, "args": args}).encode())
            raw = s.recv()
        finally:
            s.close(0)
        if raw[:1] in (b"{", b"["):
            return json.loads(raw)
        return msgpack.unpackb(raw, raw=False, strict_map_key=False)

    def load_map(self) -> dict:
        if self.a.map_npz:
            src, rep = self.a.map_npz, {}
        else:
            rep = self._p1_call("get_occupancy", timeout_s=60.0, robot_radius=self.a.robot_radius)
            if rep.get("ok") is False:
                raise RuntimeError(f"get_occupancy: {rep.get('error')}")
            src = rep.get("path") or rep.get("npz_path")
        m = load_occupancy(src)
        res = float(rep.get("resolution") or m["resolution"] or 0.05)
        origin = rep.get("origin") or m["origin"] or [0.0, 0.0]
        vals = to_ros_values(m["occ"])
        H, W = vals.shape
        g = OccupancyGrid()
        g.header.frame_id = "map"
        g.header.stamp = self.get_clock().now().to_msg()
        g.info.map_load_time = g.header.stamp
        g.info.resolution = res
        g.info.width, g.info.height = W, H
        g.info.origin.position.x, g.info.origin.position.y = float(origin[0]), float(origin[1])
        g.info.origin.orientation.w = 1.0
        g.data = array.array("b", vals.tobytes())
        self.map_pub.publish(g)
        self.map_msg = g
        self.map_occ = vals != 0            # blocked or unknown
        self.map_geo = (res, float(origin[0]), float(origin[1]))
        pgm, yml = write_map_server_files(os.path.join(self.log_dir, "map"), m["occ"], res, origin)
        info = {"source": src, "p1_source": rep.get("source"), "shape": [H, W], "resolution": res,
                "origin": [float(origin[0]), float(origin[1])], "blocked_frac": round(float((vals == 100).mean()), 4),
                "unknown_frac": round(float((vals == -1).mean()), 4), "t_wall": time.time(), "yaml": yml}
        self.map_info = info
        self.log(f"/map published (raw, not inflated): {info}")
        return info

    def _map_loop(self) -> None:
        delay = 1.0
        while self.running and self.map_info is None:
            try:
                self.load_map()
                return
            except Exception as e:  # noqa: BLE001
                self.log(f"map not loaded yet ({e!r}); retrying in {delay:.0f} s")
                time.sleep(delay)
                delay = min(5.0, delay * 1.5)

    # -- Nav2 health ---------------------------------------------------------------------------------
    def _check_nav2(self) -> None:
        if self.nav2_active and self.stats["nav2_checks"] % 10:
            self.stats["nav2_checks"] += 1
            return
        self.stats["nav2_checks"] += 1
        if not self.active_cli.service_is_ready():
            self.nav2_active = False
            self.nav2_detail = "lifecycle manager not up"
            return
        fut = self.active_cli.call_async(Trigger.Request())

        def done(f):
            try:
                ok = bool(f.result().success)
            except Exception as e:  # noqa: BLE001
                ok, self.nav2_detail = False, repr(e)
            if ok and not self.nav2_active:
                self.log("Nav2 lifecycle nodes are ACTIVE")
            self.nav2_active = ok
            if not ok:
                self.nav2_detail = "nav2 lifecycle nodes not active yet"
        fut.add_done_callback(done)

    def ready(self) -> tuple[bool, str]:
        if self.map_info is None:
            return False, "no map yet (P1 get_occupancy)"
        age = time.monotonic() - self.pose_mono if self.pose_mono else float("inf")
        if age > 0.5:
            return False, f"gt.pose stale ({age:.1f} s)"
        if not self.nav2_active:
            return False, self.nav2_detail
        return True, "ok"

    # -- /cmd_vel -> body ----------------------------------------------------------------------------
    def _on_cmd_vel(self, msg: Twist) -> None:
        t = time.time()
        g = self.active
        fwd = g is not None and g.state == "active"
        vx, vy, wz = float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)
        if fwd:
            self.stats["cmd_seq"] += 1
            req = {"id": f"nav2v-{self.stats['cmd_seq']}", "op": "velocity",
                   "args": {"vx": vx, "vy": vy, "wz": wz, "goal_id": g.id, "t_wall": t}}
            try:
                self.dealer.send(json.dumps(req).encode(), zmq.NOBLOCK)
                g.cmd_fwd += 1
                g.last_cmd = (round(vx, 3), round(vy, 3), round(wz, 3))
            except zmq.Again:
                self.stats["cmd_dropped_body_down"] += 1
                fwd = False
        else:
            self.stats["cmd_vel_idle"] += 1
        self._cmd_log.write(json.dumps({"t_wall": round(t, 4), "vx": round(vx, 4), "vy": round(vy, 4),
                                        "wz": round(wz, 4), "goal": g.id if g else None, "fwd": fwd}) + "\n")

    def _drain_body_replies(self) -> None:
        while True:
            try:
                raw = self.dealer.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                rep = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            if rep.get("ok"):
                self.stats["body_ok"] += 1
                continue
            self.stats["body_rejected"] += 1
            self.body_last_error = rep.get("error")
            g = self.active
            if g is not None:
                g.cmd_rej += 1

    # -- actions (executor thread) -------------------------------------------------------------------
    def _plan_async(self, x, y, yaw, done_cb):
        goal = ComputePathToPose.Goal()
        goal.goal = self._pose_stamped(x, y, yaw)
        goal.planner_id = self.a.planner_id
        goal.use_start = False
        box = {"t0": time.monotonic()}

        def on_resp(f):
            gh = f.result()
            if not gh.accepted:
                done_cb({"accepted": False})
                return
            gh.get_result_async().add_done_callback(on_result)

        def on_result(f):
            r = f.result()
            done_cb({"accepted": True, "status": r.status, "result": r.result,
                     "wall_ms": (time.monotonic() - box["t0"]) * 1e3})
        self.plan_client.send_goal_async(goal).add_done_callback(on_resp)

    def _pose_stamped(self, x, y, yaw) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id = "map"
        p.header.stamp = self.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        qx, qy, qz, qw = yaw_quat(float(yaw))
        p.pose.orientation.z, p.pose.orientation.w = qz, qw
        return p

    def _send_nav_goal(self, g: Goal) -> None:
        sl = SpeedLimit()
        sl.header.stamp = self.get_clock().now().to_msg()
        sl.percentage = False
        sl.speed_limit = float(g.speed) if g.speed else 0.0      # 0.0 = no limit
        self.speed_pub.publish(sl)
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_stamped(g.x, g.y, g.yaw)

        def on_fb(fb_msg):
            fb = fb_msg.feedback
            p = fb.current_pose.pose
            g.feedback = {"distance_remaining": round(float(fb.distance_remaining), 3),
                          "number_of_recoveries": int(fb.number_of_recoveries),
                          "navigation_time_s": round(fb.navigation_time.sec + fb.navigation_time.nanosec * 1e-9, 2),
                          "eta_s": round(fb.estimated_time_remaining.sec
                                         + fb.estimated_time_remaining.nanosec * 1e-9, 2),
                          "pose": [round(p.position.x, 3), round(p.position.y, 3),
                                   round(quat_yaw(p.orientation), 3)]}

        def on_resp(f):
            gh = f.result()
            if not gh.accepted:
                g.state, g.reason, g.t_end = "failed", "nav2_rejected", time.time()
                self._finish(g)
                g.accepted_ev.set()
                return
            g.handle = gh
            g.state, g.t_active = "active", time.time()
            g.accepted_ev.set()
            self.goal_event(g, "accepted")
            if g.cancel_requested:
                gh.cancel_goal_async()
                g.state = "canceling"
            gh.get_result_async().add_done_callback(on_result)

        def on_result(f):
            r = f.result()
            code = int(getattr(r.result, "error_code", 0) or 0)
            msg = str(getattr(r.result, "error_msg", "") or "")
            if r.status == GoalStatus.STATUS_SUCCEEDED:
                g.state = "succeeded"
            elif r.status == GoalStatus.STATUS_CANCELED:
                g.state, g.reason = "canceled", "canceled"
            else:
                g.state = "failed"
                g.reason = REASONS.get(code, "nav2_failed" if code else "nav2_aborted")
            g.error_code, g.error_name, g.error_msg = code, ERR_NAMES.get(code, str(code)), msg
            g.t_end = time.time()
            self._finish(g)
        self.nav_client.send_goal_async(goal, feedback_callback=on_fb).add_done_callback(on_resp)

    def _finish(self, g: Goal) -> None:
        with self.lock:
            if self.active is g:
                self.active = None
        g.done_ev.set()
        self.goal_event(g, "result")

    # -- escape hint (humanoid replacement for Nav2's BackUp) -------------------------------------------
    def escape_hint(self, x: float, y: float) -> dict:
        """Nav2 cannot plan from inside the inscribed zone (START_OCCUPIED). Find the shortest straight step (16
        directions, <= 0.6 m) to a point with clearance >= inscribed + margin; wl-body walks it slowly (SLOW_WALK
        0.2 m/s, facing held) and re-sends the goal. Numpy only, local 2.4 m window of the raw map."""
        if self.map_occ is None:
            return {"needed": False, "error": "no map"}
        res, x0, y0 = self.map_geo
        need = self.a.inscribed_radius + self.a.escape_margin
        occ = self.map_occ
        H, W = occ.shape
        ix, iy = int((x - x0) / res), int((y - y0) / res)
        w = int(1.2 / res)
        ys, xs = np.nonzero(occ[max(0, iy - w):min(H, iy + w + 1), max(0, ix - w):min(W, ix + w + 1)])
        if not len(xs):
            return {"needed": False, "clearance": None}
        O = np.stack([x0 + (xs + max(0, ix - w) + 0.5) * res, y0 + (ys + max(0, iy - w) + 0.5) * res], axis=1)

        def clear(P):
            return np.sqrt(((P[:, None, :] - O[None, :, :]) ** 2).sum(-1)).min(1) - res / 2

        c0 = float(clear(np.array([[x, y]]))[0])
        if c0 >= self.a.inscribed_radius + 0.01:
            return {"needed": False, "clearance": round(c0, 3)}
        th = np.arange(16) * (2 * np.pi / 16)
        st = np.arange(0.05, 0.61, 0.05)
        P = np.stack([x + np.outer(np.cos(th), st), y + np.outer(np.sin(th), st)], -1)   # 16 x S x 2
        C = clear(P.reshape(-1, 2)).reshape(len(th), len(st))
        best = None
        for k in range(len(th)):
            ok = np.nonzero(C[k] >= need)[0]
            if not len(ok):
                continue
            j = int(ok[0])
            if C[k, :j + 1].min() < min(c0, 0.12) - 1e-6:      # never step through something closer than now
                continue
            cand = (float(st[j]), -float(C[k, j]), k, j)
            if best is None or cand < best:
                best = cand
        if best is None:
            return {"needed": True, "clearance": round(c0, 3), "dir": None}
        s_, _, k, j = best
        return {"needed": True, "clearance": round(c0, 3), "dir": [float(np.cos(th[k])), float(np.sin(th[k]))],
                "dist": round(s_, 3), "clearance_after": round(float(C[k, j]), 3),
                "target": [round(float(P[k, j, 0]), 3), round(float(P[k, j, 1]), 3)]}

    # -- REP (body) -----------------------------------------------------------------------------------
    def _rep_loop(self) -> None:
        s = self.zctx.socket(zmq.REP)
        s.setsockopt(zmq.LINGER, 0)
        s.bind(f"tcp://127.0.0.1:{self.P['nav_bridge']}")
        self.log(f"REP bound on {self.P['nav_bridge']}")
        while self.running:
            if not s.poll(200):
                continue
            raw = s.recv()
            try:
                req = json.loads(raw)
                a = {**req, **(req.get("args") or {})}
                rep = self._op(str(req.get("op")), a)
            except Exception as e:  # noqa: BLE001
                rep = {"ok": False, "error": f"internal: {e!r}"}
            s.send(json.dumps(rep, default=str).encode())
        s.close(0)

    def _op(self, op: str, a: dict) -> dict:
        self.stats[f"op_{op}"] += 1
        if op == "ping":
            ok, detail = self.ready()
            with self.lock:
                pose, age = self.pose, (time.monotonic() - self.pose_mono) if self.pose_mono else None
            return {"ok": True, "nav2_ready": ok, "detail": detail, "nav2_active": self.nav2_active,
                    "map": self.map_info, "pose": pose, "pose_age_s": age, "pose_hz": round(self.pose_rate(), 1),
                    "active_goal": self.active.id if self.active else None,
                    "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"), "rmw": os.environ.get("RMW_IMPLEMENTATION"),
                    "pid": os.getpid()}
        if op == "status":
            g = self.goals.get(str(a.get("id")))
            if g is None:
                return {"ok": False, "state": "unknown", "error": "unknown goal id"}
            return {"ok": True, **g.brief()}
        if op == "cancel":
            return self._cancel(str(a.get("id")))
        if op == "goto":
            return self._goto(a)
        if op == "plan":
            with self.lock:
                pose = self.pose or (0.0, 0.0, 0.0)
            yaw = a.get("yaw")
            r = self._plan_blocking(float(a["x"]), float(a["y"]), pose[2] if yaw is None else float(yaw),
                                    float(a.get("timeout_s", 5.0)))
            return {"ok": r.get("ok", False), **r}
        if op == "escape":
            with self.lock:
                pose = self.pose
            if pose is None:
                return {"ok": False, "error": "no pose"}
            return {"ok": True, "pose": pose, **self.escape_hint(pose[0], pose[1])}
        if op == "reload_map":
            return {"ok": True, "map": self.load_map()}
        if op == "stats":
            ru = resource.getrusage(resource.RUSAGE_SELF)
            return {"ok": True, "stats": dict(self.stats), "cpu_s": ru.ru_utime + ru.ru_stime,
                    "maxrss_mb": ru.ru_maxrss / 1024, "body_last_error": self.body_last_error,
                    "goals": [g.brief() for g in list(self.goals.values())[-10:]]}
        return {"ok": False, "error": f"unknown op {op!r}"}

    def _plan_blocking(self, x, y, yaw, timeout_s: float) -> dict:
        ev, box = threading.Event(), {}

        def done(r):
            box.update(r)
            ev.set()
        if not self.in_ros(lambda: self.plan_client.server_is_ready(), 2.0):
            return {"ok": False, "reason": "nav2_unavailable", "detail": "compute_path_to_pose server not ready"}
        self.in_ros(lambda: self._plan_async(x, y, yaw, done), 2.0)
        if not ev.wait(timeout_s):
            return {"ok": False, "reason": "no_path", "error_name": "PLAN_WAIT_TIMEOUT",
                    "detail": f"no planner result within {timeout_s} s"}
        if not box.get("accepted"):
            return {"ok": False, "reason": "nav2_rejected", "detail": "planner rejected the goal"}
        res = box["result"]
        code = int(getattr(res, "error_code", 0) or 0)
        poses = res.path.poses
        if box["status"] != GoalStatus.STATUS_SUCCEEDED or not poses:
            return {"ok": False, "reason": REASONS.get(code, "no_path"), "error_code": code,
                    "error_name": ERR_NAMES.get(code, str(code)), "error_msg": str(getattr(res, "error_msg", "")),
                    "plan_ms": round(box["wall_ms"], 1)}
        pts = np.array([[p.pose.position.x, p.pose.position.y] for p in poses])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else np.zeros(0)
        length = float(seg.sum())
        idx = np.linspace(0, len(pts) - 1, min(60, len(pts))).round().astype(int)
        # arrival heading: direction of the last >= 0.3 m of the path
        end = pts[-1]
        back = pts[0]
        for p in pts[::-1]:
            if np.linalg.norm(end - p) >= 0.3:
                back = p
                break
        arrive = math.atan2(end[1] - back[1], end[0] - back[0]) if np.linalg.norm(end - back) > 0.05 else None
        pt = res.planning_time
        return {"ok": True, "length_m": round(length, 3), "n": int(len(pts)),
                "plan_ms": round(pt.sec * 1e3 + pt.nanosec * 1e-6, 1), "wall_ms": round(box["wall_ms"], 1),
                "path": [[round(float(pts[i, 0]), 3), round(float(pts[i, 1]), 3)] for i in idx],
                "end": [float(end[0]), float(end[1])], "arrive_yaw": arrive}

    def _goto(self, a: dict) -> dict:
        gid = str(a.get("id") or f"g{int(time.time() * 1000)}")
        x, y = float(a["x"]), float(a["y"])
        yaw = a.get("yaw")
        yaw = None if yaw is None else float(yaw)
        speed = a.get("speed")
        speed = None if speed in (None, 0) else float(speed)
        ok, detail = self.ready()
        if not ok:
            return {"ok": False, "reason": "nav2_unavailable", "detail": detail}
        prev = self.active
        if prev is not None and prev.state not in TERMINAL:
            if prev.id != gid or prev.state in ("active", "canceling"):
                self._cancel(prev.id)
            prev.done_ev.wait(float(a.get("preempt_wait_s", 3.0)))
        with self.lock:
            pose = self.pose
        plan = self._plan_blocking(x, y, pose[2] if yaw is None else yaw, float(a.get("plan_timeout_s", 5.0)))
        g = Goal(gid, x, y, yaw, speed)
        if not plan.get("ok"):
            g.state, g.reason = "failed", plan.get("reason")
            g.error_code, g.error_name, g.error_msg = plan.get("error_code"), plan.get("error_name"), \
                plan.get("error_msg")
            g.t_end = time.time()
            self.goals[gid] = g
            extra = {}
            if plan.get("reason") == "start_in_obstacle" and pose is not None:
                extra["escape"] = self.escape_hint(pose[0], pose[1])
            self.goal_event(g, "plan_failed", plan=plan, **extra)
            return {"ok": False, **{k: v for k, v in plan.items() if k != "ok"}, **extra}
        end = plan["end"]
        if math.hypot(end[0] - x, end[1] - y) > 0.05:          # planner tolerance moved the goal: go where it can
            g.x, g.y, g.snapped = end[0], end[1], True
        if yaw is None:
            g.yaw = plan["arrive_yaw"] if plan.get("arrive_yaw") is not None else pose[2]
        g.plan = {k: plan[k] for k in ("length_m", "n", "plan_ms", "wall_ms", "path")}
        g.plan["goal_snapped"] = [g.x, g.y] if g.snapped else None
        with self.lock:
            self.goals[gid] = g
            self.goals.move_to_end(gid)
            while len(self.goals) > 200:
                self.goals.popitem(last=False)
            self.active = g
        self.in_ros(lambda: self._send_nav_goal(g), 2.0)
        if not g.accepted_ev.wait(float(a.get("accept_timeout_s", 3.0))):
            self._cancel(gid)
            return {"ok": False, "reason": "nav2_unavailable", "detail": "NavigateToPose goal not acknowledged"}
        if g.state == "failed":
            return {"ok": False, "reason": g.reason or "nav2_rejected", "detail": "NavigateToPose rejected the goal"}
        return {"ok": True, "id": gid, "state": g.state, "plan": g.plan,
                "goal": {"x": g.x, "y": g.y, "yaw": g.yaw, "snapped": g.snapped, "yaw_from_path": yaw is None}}

    def _cancel(self, gid: str) -> dict:
        g = self.goals.get(gid)
        if g is None:
            return {"ok": False, "error": "unknown goal id"}
        if g.state in TERMINAL:
            return {"ok": True, "state": g.state, "already_terminal": True}
        g.cancel_requested = True
        prev_state = g.state
        if g.state == "active":
            g.state = "canceling"          # stop forwarding /cmd_vel immediately
        if g.handle is not None:
            try:
                self.in_ros(lambda: g.handle.cancel_goal_async(), 2.0)
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": repr(e)}
        self.goal_event(g, "cancel_requested", prev_state=prev_state)
        return {"ok": True, "state": g.state}

    def shutdown(self) -> None:
        self.running = False
        g = self.active
        if g is not None and g.handle is not None and g.state not in TERMINAL:
            try:
                g.handle.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass
        for t in self.threads:
            t.join(timeout=1.0)
        self.dealer.close(0)
        self._cmd_log.close()
        self._goal_log.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="wl ROS 2 bridge (Nav2 <-> wl-body / wl-isaac over ZMQ)")
    ap.add_argument("--port-offset", type=int, default=int(os.environ.get("WL_PORT_OFFSET", "0")))
    ap.add_argument("--log-dir", default=os.environ.get("WL_NAV2_LOG_DIR")
                    or f"/tmp/wl-nav2-bridge-{time.strftime('%Y%m%d-%H%M%S')}")
    ap.add_argument("--robot-radius", type=float, default=0.30, help="passed to P1 get_occupancy (raw grid is used)")
    ap.add_argument("--map-npz", default=None, help="use this occupancy npz instead of asking P1")
    ap.add_argument("--planner-id", default="GridBased")
    ap.add_argument("--inscribed-radius", type=float, default=0.30, help="= costmap robot_radius (escape hint)")
    ap.add_argument("--escape-margin", type=float, default=0.07)
    ap.add_argument("--lifecycle-manager", default="lifecycle_manager_navigation")
    ap.add_argument("--cmd-vel-stamped", action="store_true", help="subscribe TwistStamped (Nav2 Kilted default)")
    a = ap.parse_args(argv)
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = Bridge(a)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    th = threading.Thread(target=lambda: _spin(ex, stop), name="executor", daemon=True)
    th.start()
    while not stop.wait(0.5):
        pass
    node.log("shutting down")
    node.shutdown()
    ex.shutdown(timeout_sec=1.0)
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


def _spin(ex, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            ex.spin_once(timeout_sec=0.1)
        except Exception as e:  # noqa: BLE001
            print(f"[bridge] executor error: {e!r}", flush=True)
            if not rclpy.ok():
                return


if __name__ == "__main__":
    sys.exit(main())
