"""The P1 M2b wire (docs/contracts/p1_m2b.md v1, "m2b-1") served the same way by both fakes (integration, wave 1).

Two owners mirror P1 kinematically: isaac's `tools/fake_p1.FakeP1` (the P1 owner's reference, also used by the stack
scripts) and world's `tests/fakes/fake_p1_world.FakeP1World` (what the runtime's world tests run against). Both serve
the recorded H38 here, and every reply is checked against the keys the contract names, so a drift in either fake (or
in the contract) fails one place. Then the consumers: world's IsaacGTWorldModel reads the same capabilities from
both, and groot_arms' ZmqSensors reads P1's ego_view stream (5566) once world's enable_camera has turned it on.
"""

from __future__ import annotations

import time
from pathlib import Path

import msgpack
import pytest
import zmq

from tests.fakes.fake_p1_world import HOUSES, FakeP1World, free_port_offset

H38 = HOUSES / "procthor-train-38"
DYN = "Pan|surface|6|1"               # a loose prop in H38 (body_path, not static, not articulated)
STATIC = "door|6|7"

# p1_m2b.md: the keys each reply / message must carry
PING = {"ops", "p1_contract", "cameras", "topics"}
M2B_OPS = {"get_objects", "attach", "detach", "release_all", "camera", "get_cameras", "set_render_rates",
           "get_link_poses", "detections", "reset_scene", "move_object", "set_object_pose", "push_object",
           "get_health", "render_topdown"}
OBJ = {"id", "name", "pos", "quat_wxyz", "aabb", "held_by", "dynamic", "source", "lin_vel", "moving"}   # §3.1
ATTACH = {"id", "arm", "mode", "held_by", "snapped", "dist_m", "grip_point", "stepping_stone"}           # §4.1
DETACH = {"id", "was_held", "placed", "pos", "aabb", "held_by"}                                         # §4.2
CAMERA = {"name", "on", "hz", "consumers"}                                                              # §5.4
DETECTIONS = {"camera", "method", "detections", "w", "h"}                                               # §7
DET = {"id", "px", "dist_m", "held_by"}
RESET = {"variant", "objects_reset", "released"}                                                        # §8
HEALTH = {"rtf_1s", "rtf_3s", "rtf_5s", "level"}                                                        # §10.1
GT_OBJECTS = {"seq", "t_sim", "t_wall", "objects"}                                                      # §3.2


def _fakes(tmp: Path):
    from tools.fake_p1 import FakeP1

    a_off, b_off = free_port_offset(), free_port_offset()
    while b_off == a_off:
        b_off = free_port_offset()
    iso = FakeP1(a_off, str(tmp / "iso"), log=lambda *_: None, house_dir=str(H38)).start()
    wld = FakeP1World("procthor-train-38", port_offset=b_off, m2b=True).start()
    return {"tools.fake_p1": (iso, a_off), "fake_p1_world": (wld, b_off)}


@pytest.fixture
def fakes(tmp_path):
    if not H38.exists():
        pytest.skip("recorded house missing")
    pytest.importorskip("sim_isaac.wire")
    f = _fakes(tmp_path)
    time.sleep(0.2)
    yield f
    for p1, _ in f.values():
        p1.stop()


def _rpc(off: int):
    from body.config import ep
    from body.p1_client import P1Rpc
    return P1Rpc(ep(5600 + off), timeout_s=3.0)


def _topic(off: int, topic: bytes, timeout_s: float = 3.0) -> dict:
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.SUBSCRIBE, topic)
    s.connect(f"tcp://127.0.0.1:{5601 + off}")
    try:
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            if s.poll(100):
                parts = s.recv_multipart()
                return msgpack.unpackb(parts[-1], raw=False)
        raise AssertionError(f"no {topic!r} within {timeout_s} s")
    finally:
        s.close(0)


def _missing(rep: dict, keys: set[str]) -> set[str]:
    return {k for k in keys if k not in rep}


def test_both_fakes_serve_the_contract_keys(fakes):
    for name, (_p1, off) in fakes.items():
        rpc = _rpc(off)
        try:
            ping = rpc.call("ping")
            assert not _missing(ping, PING), (name, _missing(ping, PING))
            assert ping["p1_contract"] == "m2b-1" and M2B_OPS <= set(ping["ops"]), (name, M2B_OPS - set(ping["ops"]))
            assert ping["cameras"].get("head") is True and ping["cameras"].get("ego_view") is False, name
            assert {"gt.pose", "gt.objects", "gt.event", "sim.health"} <= set(ping["topics"]), name

            objs = rpc.call("get_objects")["objects"]
            assert objs and all(not _missing(o, OBJ) for o in objs), (name, _missing(objs[0], OBJ))
            dyn = next(o for o in objs if o["id"] == DYN)
            assert dyn["dynamic"] is True and dyn["source"] == "sim", name

            att = rpc.call("attach", id=DYN, arm="right", mode="follow")
            assert att["ok"] and not _missing(att, ATTACH) and att["stepping_stone"] is True, (name, att)
            assert rpc.call("get_objects", ids=[DYN])["objects"][0]["held_by"] == "right", name
            det = rpc.call("detach", id=DYN)
            assert det["ok"] and not _missing(det, DETACH) and det["was_held"] is True, (name, det)
            # machine-readable error codes (§11)
            assert rpc.call("attach", id="Nope|0", arm="right")["code"] == "unknown_object", name
            assert rpc.call("attach", id=STATIC, arm="right")["code"] == "not_movable", name
            assert rpc.call("detections", camera="ego_view")["code"] == "camera_off", name

            cam = rpc.call("camera", name="ego_view", on=True, hz=30, consumer="man-1", ttl_s=10)
            assert cam["ok"] and not _missing(cam, CAMERA) and cam["on"] is True and cam["hz"] == 30, (name, cam)
            d = rpc.call("detections", camera="ego_view", min_px=1)
            assert d["ok"] and not _missing(d, DETECTIONS), (name, _missing(d, DETECTIONS))
            assert all(not _missing(x, DET) for x in d["detections"]), name
            off_rep = rpc.call("camera", name="ego_view", on=False, consumer="man-1")
            assert off_rep["on"] is False, (name, off_rep)

            rs = rpc.call("reset_scene", variant="default")
            assert rs["ok"] and not _missing(rs, RESET), (name, _missing(rs, RESET))
            lp = rpc.call("get_link_poses")
            assert {"torso_link", "left_palm", "right_palm"} <= set(lp["links"]), name
        finally:
            rpc.close()
        h = _topic(off, b"sim.health")
        assert not _missing(h, HEALTH) and h["level"] in ("ok", "degraded", "unsafe"), (name, h)
        g = _topic(off, b"gt.objects")
        assert not _missing(g, GT_OBJECTS) and all(o.get("dynamic", True) for o in g["objects"]), name


def test_world_reads_the_same_capabilities_from_both(fakes):
    from world.isaac_client import IsaacGTWorldModel

    caps = {}
    for name, (_p1, off) in fakes.items():
        w = IsaacGTWorldModel(port_offset=off, camera="head_sim", rpc_timeout_s=3.0)
        try:
            c = w.capabilities()
            caps[name] = {k: c[k] for k in ("attach", "detach", "object_poses", "segmentation", "enable_camera",
                                              "reset_scene", "move_object", "p1_contract", "detections:ego_view")}
            assert w.cam.name == "head_sim" and w.camera_note == "", name       # the M2b head camera, not d435
            w.enable_camera("ego_view", True, consumer="integ", ttl_s=5.0, hz=30.0)
            dets = w.detections(camera="ego_view", method="best")
            assert isinstance(dets, list), name
            w.enable_camera("ego_view", False, consumer="integ")
        finally:
            w.close()
    a, b = caps.values()
    assert a == b and a["p1_contract"] == "m2b-1" and all(v for k, v in a.items() if k != "p1_contract"), caps


def test_groot_sensors_read_the_ego_view_stream_world_enables(fakes):
    """groot_arms' ZmqSensors on isaac's fake 5566 (key ego_view, t_capture_mono), enabled through world."""
    pytest.importorskip("PIL")
    from services.executors.groot_arms import ZmqSensors
    from world.isaac_client import IsaacGTWorldModel

    p1, off = fakes["tools.fake_p1"]
    w = IsaacGTWorldModel(port_offset=off, camera="head_sim", rpc_timeout_s=3.0)
    sens = ZmqSensors(f"tcp://127.0.0.1:{5566 + off}", f"tcp://127.0.0.1:{5557 + off}", "ego_view").start()
    try:
        time.sleep(0.3)
        frame, _ = sens.ego_frame()
        assert frame is None                                    # off until a consumer enables it (OD1)
        t_on = time.monotonic()
        w.enable_camera("ego_view", True, consumer="man-1", ttl_s=5.0, hz=30.0)
        t_end = time.monotonic() + 3.0
        while (frame is None) and time.monotonic() < t_end:
            time.sleep(0.02)
            frame, t = sens.ego_frame()
        assert frame is not None and frame.shape == (480, 640, 3) and frame.dtype.name == "uint8"
        assert t_on - 0.05 <= t <= time.monotonic() + 0.01      # P1's capture time on this host's clock
    finally:
        w.enable_camera("ego_view", False, consumer="man-1")
        sens.close()
        w.close()
