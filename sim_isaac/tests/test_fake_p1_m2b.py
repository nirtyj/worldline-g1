"""tools/fake_p1.py serves the P1 M2b wire (docs/contracts/p1_m2b.md) over real ZMQ, and world/isaac_client.py (the
runtime's P1 client) works against it. No Isaac; runs in .venv-rt.

    .venv-rt/bin/python -m pytest -q sim_isaac/tests
"""
from __future__ import annotations

import random
import socket
import time
from pathlib import Path

import msgpack
import numpy as np
import pytest
import zmq

from body.config import BASE_PORTS, ep
from body.p1_client import P1Rpc
from sim_isaac import wire as W
from tools.fake_p1 import FakeP1

HOUSE38 = Path(__file__).resolve().parents[2] / "tests" / "fakes" / "houses" / "procthor-train-38"


def _free_offset() -> int:
    ports = list(BASE_PORTS.values()) + [W.EGO_PORT]
    for _ in range(200):
        off = random.randrange(20000, 40000, 100)
        ok = True
        for p in ports:
            s = socket.socket()
            try:
                s.bind(("127.0.0.1", p + off))
            except OSError:
                ok = False
            finally:
                s.close()
            if not ok:
                break
        if ok:
            return off
    raise RuntimeError("no free port block")


@pytest.fixture
def fake(tmp_path):
    off = _free_offset()
    # rtf pinned to 1.0: the fake's measured RTF is t_sim over wall time, which reads DEGRADED whenever the suite
    # starves its physics thread (the wave-1 flake); the health test drives the level through rtf_override
    p1 = FakeP1(off, str(tmp_path / "p1"), log=lambda *_: None, rtf=1.0).start()
    rpc = P1Rpc(ep(5600 + off), timeout_s=3.0)
    yield p1, rpc, off
    rpc.close()
    p1.stop()


class Sub:
    """SUB on one P1 port, collecting [topic, payload] or single-frame messages in a thread-free poll loop."""

    def __init__(self, port: int, topics=(b"",)):
        self.s = zmq.Context.instance().socket(zmq.SUB)
        self.s.setsockopt(zmq.LINGER, 0)
        for t in topics:
            self.s.setsockopt(zmq.SUBSCRIBE, t)
        self.s.connect(ep(port))

    def wait(self, pred, timeout: float = 3.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.s.poll(50):
                fr = self.s.recv_multipart()
                topic = fr[0].decode() if len(fr) > 1 else ""
                msg = msgpack.unpackb(fr[-1], raw=False)
                if pred(topic, msg):
                    return topic, msg
        return None, None

    def close(self):
        self.s.close(0)


def test_ping_lists_the_m2b_ops(fake):
    p1, rpc, off = fake
    rep = rpc.call("ping")
    assert rep["p1_contract"] == W.CONTRACT and rep["topics"] == W.TOPICS
    for op in ("get_objects", "attach", "detach", "camera", "set_render_rates", "get_link_poses", "detections",
               "reset_scene", "move_object", "set_object_pose", "push_object", "get_health", "render_topdown"):
        assert op in rep["ops"]
    assert rep["cameras"] == {"head": True, "ego_view": False}
    bad = rpc.call("no_such_op")
    assert bad["ok"] is False and bad["code"] == "unknown_op"


def test_objects_live_poses_move_push_and_fall(fake):
    p1, rpc, off = fake
    rep = rpc.call("get_objects")
    by = {o["id"]: o for o in rep["objects"]}
    assert rep["pose_source"] == "sim"
    assert by["Apple|1"]["dynamic"] and by["Apple|1"]["source"] == "sim"
    assert by["CounterTop|1"]["dynamic"] is False and by["CounterTop|1"]["source"] == "static"
    assert set(by["Apple|1"]) >= {"id", "pos", "aabb", "held_by"}           # the P1.2 wire world reads
    dyn = rpc.call("get_objects", dynamic_only=True)["objects"]
    assert {o["id"] for o in dyn} == {"Apple|1", "Mug|1", "RemoteControl|1"}
    assert rpc.call("get_objects", ids=["Nope|1"])["code"] == "unknown_object"

    sub = Sub(5601 + off, [b"gt.objects", b"gt.event"])
    time.sleep(0.2)
    t0 = time.monotonic()
    assert rpc.call("move_object", id="Mug|1", pose=[2.2, 4.8, 0.80])["ok"]      # on the dining table (0.75)
    _, msg = sub.wait(lambda t, m: t == "gt.objects" and any(
        o["id"] == "Mug|1" and abs(W.box_center(o["aabb"])[0] - 2.2) < 1e-3 for o in m["objects"]))
    assert msg is not None and time.monotonic() - t0 < 0.5                  # within 0.5 s (P1.2 exit, on the fake)
    got = {o["id"]: o for o in rpc.call("get_objects", ids=["Mug|1"])["objects"]}["Mug|1"]
    assert W.box_center(got["aabb"]) == pytest.approx([2.2, 4.8, 0.80], abs=1e-3)
    # a push off the counter's edge: the apple slides off and falls to the floor -> object_fell
    assert rpc.call("push_object", id="Apple|1", vel=[1.5, 0.0, 0.0])["ok"]
    _, ev = sub.wait(lambda t, m: t == "gt.event" and m["event"] == "object_fell", timeout=4.0)
    assert ev is not None and ev["id"] == "Apple|1" and ev["on_floor"] and ev["drop_m"] > 0.8
    assert rpc.call("push_object", id="CounterTop|1", vel=[1, 0, 0])["code"] == "not_movable"
    assert rpc.call("get_stats")["object_writes"] == 2
    sub.close()


def test_attach_follow_detach_and_errors(fake):
    p1, rpc, off = fake
    rep = rpc.call("attach", id="Mug|1", arm="right", mode="follow")
    assert rep["ok"] and rep["held_by"] == "right" and rep["snapped"] and rep["stepping_stone"] is True
    time.sleep(0.05)
    o = {x["id"]: x for x in rpc.call("get_objects", ids=["Mug|1"])["objects"]}["Mug|1"]
    assert o["held_by"] == "right" and o["lin_vel"] == [0.0, 0.0, 0.0]
    assert W.box_center(o["aabb"]) == pytest.approx(rep["grip_point"], abs=1e-3)
    # the robot moves: the held object follows the palm
    with p1._lock:
        p1.x += 0.5
    time.sleep(0.05)
    o2 = {x["id"]: x for x in rpc.call("get_objects", ids=["Mug|1"])["objects"]}["Mug|1"]
    assert W.box_center(o2["aabb"])[0] - W.box_center(o["aabb"])[0] == pytest.approx(0.5, abs=1e-3)
    assert rpc.call("attach", id="Mug|1", arm="left")["code"] == "held_by_other"
    assert rpc.call("attach", id="Apple|1", arm="right")["code"] == "hand_busy"
    assert rpc.call("attach", id="Apple|1", arm="middle")["code"] == "bad_arg"
    assert rpc.call("attach", id="Apple|1", arm="left", mode="glue")["code"] == "bad_arg"
    assert rpc.call("attach", id="DiningTable|1", arm="left")["code"] == "not_movable"
    assert rpc.call("attach", id="Nope|1", arm="left")["code"] == "unknown_object"
    assert rpc.call("attach", id="Apple|1", arm="left", mode="fixed_joint")["ok"]
    assert rpc.call("get_health")["held"] == {"left": "Apple|1", "right": "Mug|1"}
    # detach onto the dining table top (0.75 m): the centre lands at the given pose and stays
    d = rpc.call("detach", id="Mug|1", pose=[1.8, 4.5, 0.80])
    assert d["ok"] and d["was_held"] and d["placed"] and d["held_by"] is None
    time.sleep(0.2)
    o3 = {x["id"]: x for x in rpc.call("get_objects", ids=["Mug|1"])["objects"]}["Mug|1"]
    assert W.box_center(o3["aabb"]) == pytest.approx([1.8, 4.5, 0.80], abs=1e-3) and o3["held_by"] is None
    assert rpc.call("detach", id="Mug|1")["was_held"] is False
    assert rpc.call("release_all")["released"] == ["Apple|1"]


def test_link_poses_in_gt_pose_and_op(fake):
    p1, rpc, off = fake
    sub = Sub(5601 + off, [b"gt.pose"])
    _, msg = sub.wait(lambda t, m: t == "gt.pose")
    assert set(msg["links"]) == {"torso_link", "left_palm", "right_palm"} and len(msg["waist_q"]) == 3
    assert len(msg["links"]["left_palm"]["pos"]) == 3 and len(msg["links"]["left_palm"]["quat_wxyz"]) == 4
    rep = rpc.call("get_link_poses", links=["torso_link", "cam:head", "cam:ego_view", "nope"])
    assert set(rep["links"]) == {"torso_link", "cam:head", "cam:ego_view"} and rep["unknown"] == ["nope"]
    head_z = rep["links"]["cam:head"]["pos"][2]
    assert head_z == pytest.approx(0.78 + 0.044 + 0.526, abs=1e-6)                 # ~1.35 m
    sub.close()


def test_head_and_ego_view_streams(fake):
    p1, rpc, off = fake
    head = Sub(5565 + off)
    ego = Sub(W.EGO_PORT + off)
    _, h = head.wait(lambda t, m: True)
    assert list(h["images"]) == ["head"] and h["camera"] == "head" and h["timestamps"]["head"] == h["t_capture"]
    assert h["hfov"] == pytest.approx(90.0) and len(h["cam_pose_wl"]) == 4 and h["stationary"] is True
    assert h["cam_pose_wl"][3] == pytest.approx(15.0, abs=0.05)
    assert ego.wait(lambda t, m: True, timeout=0.5) == (None, None)          # off by default: nothing on 5566
    rep = rpc.call("camera", name="ego_view", on=True, consumer="groot:exec-1", ttl_s=0.6)
    assert rep["on"] and rep["port"] == W.EGO_PORT + off and rep["key"] == "ego_view"
    _, e = ego.wait(lambda t, m: True)
    assert e is not None and list(e["images"]) == ["ego_view"] and e["camera"] == "ego_view"
    assert e["hfov"] == pytest.approx(69.87, abs=0.01) and e["cam_pose_wl"][3] == pytest.approx(35.0, abs=0.05)
    assert time.monotonic() - e["t_capture_mono"] < 1.0                       # same-box monotonic clock
    # TTL: the consumer did not renew -> ego_view switches itself off
    time.sleep(0.9)
    assert rpc.call("ping")["cameras"]["ego_view"] is False
    # set_render_rates (SimControl): ego on at 15 Hz, then 0 = off
    r = rpc.call("set_render_rates", head_hz=15, ego_hz=15)
    assert r["head"]["hz"] == 15 and r["ego_view"]["on"] and r["ego_view"]["hz"] == 15
    assert rpc.call("set_render_rates", ego_hz=0)["ego_view"]["on"] is False
    assert rpc.call("camera", name="nope", on=True)["code"] == "unknown_camera"
    assert rpc.call("camera", name="head", hz=0)["code"] == "bad_arg"
    cams = {c["name"]: c for c in rpc.call("get_cameras")["cameras"]}
    assert cams["ego_view"]["mount_xyz"] == pytest.approx(list(W.EGO_VIEW.mount_xyz))
    assert cams["head"]["vfov_deg"] == pytest.approx(73.74, abs=0.01)
    head.close()
    ego.close()


def test_detections_shape_and_camera_off(fake):
    p1, rpc, off = fake
    with p1._lock:                   # stand 0.8 m in front of the kitchen counter apple (0.3, 1.0), facing -x
        p1.x, p1.y, p1.yaw = 1.1, 1.0, np.pi
    rep = rpc.call("detections", camera="head", min_px=40)
    assert rep["ok"] and rep["method"] == "fake-frustum" and rep["w"] == 640
    ids = {d["id"]: d for d in rep["detections"]}
    assert "Apple|1" in ids and ids["Apple|1"]["px"] >= 40 and len(ids["Apple|1"]["bbox"]) == 4
    assert ids["Apple|1"]["dist_m"] < 1.5
    far = rpc.call("detections", camera="head", max_range=0.1)
    assert far["detections"] == []
    assert rpc.call("detections", camera="ego_view")["code"] == "camera_off"


def test_reset_scene_and_topdown_modes(fake):
    p1, rpc, off = fake
    load = {o["id"]: W.box_center(o["aabb"]) for o in rpc.call("get_objects", dynamic_only=True)["objects"]}
    rpc.call("attach", id="Apple|1", arm="left")
    rpc.call("move_object", id="Mug|1", pose=[2.4, 4.9, 0.8])
    rep = rpc.call("reset_scene", poses={"RemoteControl|1": [7.3, 3.0, 0.5]}, robot=True)
    assert rep["ok"] and rep["released"] == ["Apple|1"] and rep["poses_applied"] == ["RemoteControl|1"]
    assert rep["robot_reset"] and rep["ms"] < 30000
    now = {o["id"]: W.box_center(o["aabb"]) for o in rpc.call("get_objects", dynamic_only=True)["objects"]}
    assert now["Apple|1"] == pytest.approx(load["Apple|1"]) and now["Mug|1"] == pytest.approx(load["Mug|1"])
    assert now["RemoteControl|1"] == pytest.approx([7.3, 3.0, 0.5])
    assert rpc.call("reset_scene", variant="messy")["code"] == "unknown_variant"
    full = rpc.call("render_topdown")
    furn = rpc.call("render_topdown", mode="furniture")
    assert full["mode"] == "full" and furn["mode"] == "furniture" and furn["hidden"] > 0
    assert Path(furn["path"]).exists() and furn["path"] != full["path"]
    assert rpc.call("render_topdown", mode="nope")["code"] == "bad_arg"


def test_health_and_robot_fell(fake):
    p1, rpc, off = fake
    sub = Sub(5601 + off, [b"sim.health", b"gt.event"])
    _, h = sub.wait(lambda t, m: t == "sim.health", timeout=5.0)
    assert h is not None and h["level"] == "ok" and h["rtf_1s"] == 1.0
    assert set(h) >= {"rtf_1s", "rtf_3s", "rtf_5s", "level", "cameras", "held"}
    # a message built before the override may still be queued: wait for the first one that carries it
    p1.rtf_override = 0.9
    _, h = sub.wait(lambda t, m: t == "sim.health" and m["rtf_1s"] == 0.9, timeout=5.0)
    assert h is not None and h["level"] == "degraded"
    p1.rtf_override = 0.8
    assert rpc.call("get_health")["level"] in ("degraded", "unsafe")
    _, h = sub.wait(lambda t, m: t == "sim.health" and m["rtf_1s"] == 0.8, timeout=5.0)
    assert h is not None and h["level"] == "unsafe"
    with p1._lock:
        p1.collapsed = True
    _, ev = sub.wait(lambda t, m: t == "gt.event" and m["event"] == "robot_fell", timeout=2.0)
    assert ev is not None and "pelvis_z" in ev
    sub.close()


def test_world_isaac_client_on_the_fake(tmp_path):
    """world/isaac_client.py (the runtime's P1 client) against this fake serving a real house: op discovery, live
    object poses, attach/detach through P1."""
    if not HOUSE38.exists():
        pytest.skip("recorded house missing")
    from world.isaac_client import IsaacGTWorldModel

    off = _free_offset()
    p1 = FakeP1(off, str(tmp_path / "p1"), log=lambda *_: None, house_dir=str(HOUSE38)).start()
    w = None
    try:
        w = IsaacGTWorldModel(port_offset=off, camera="head_sim", rpc_timeout_s=3.0)
        caps = w.capabilities()
        assert caps["attach"] and caps["detach"] and caps["object_poses"]
        clock = next(oid for oid, s in w.map.objects.items() if s.type == "alarm_clock")
        sid = w.map.objects[clock].scene_id
        box0 = w.object(clock).box
        rpc = P1Rpc(ep(5600 + off), timeout_s=3.0)
        c = W.box_center([list(box0[0]), list(box0[1])])
        assert rpc.call("move_object", id=sid, pose=[c[0] + 0.3, c[1], c[2]])["ok"]
        time.sleep(0.15)
        st = w.object(clock)
        assert st.pose_source == "sim" and st.box[0][0] == pytest.approx(box0[0][0] + 0.3, abs=1e-3)
        w.attach(clock, "right", "follow")
        assert w.hands()["right"] == clock and "attach" in p1.events[-1]["event"]
        w.detach(clock, (c[0], c[1], c[2]))
        assert p1.events[-1]["event"] == "detach" and p1.events[-1]["placed"] is True
        rpc.close()
    finally:
        if w is not None:
            w.close()
        p1.stop()


def test_measured_rtf_is_sim_time_over_wall_time(tmp_path):
    """rtf=None (the CLI default) measures t_sim over wall time; clock values injected, no threads started."""
    p1 = FakeP1(_free_offset(), str(tmp_path / "p1"), log=lambda *_: None)
    assert p1.rtf_override is None
    p1.t_sim = 10.0
    assert p1._health(10.0)["level"] == "ok" and p1._health(10.0)["rtf_1s"] == 1.0
    assert p1._health(8.0)["rtf_1s"] == 1.0                              # capped at 1 (a fake cannot run ahead)
    h = p1._health(10.0 / 0.9)
    assert h["rtf_1s"] == pytest.approx(0.9) and h["level"] == "degraded"
    assert p1._health(12.5)["level"] == "unsafe"                          # 0.8
    assert p1._pose_msg(12.5)["rtf"] == pytest.approx(0.8)
    pinned = FakeP1(_free_offset(), str(tmp_path / "p2"), log=lambda *_: None, rtf=1.0)
    pinned.t_sim = 1.0
    assert pinned._health(100.0)["level"] == "ok" and pinned._pose_msg(100.0)["rtf"] == 1.0
