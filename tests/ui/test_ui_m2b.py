"""M2b page additions and fixes: GT confinement in ui/ (no gt.pose subscription, no pose fields from frames), the
sim-added head camera caption, scan thumbnails, the list_locations panel and the GR00T strip in the frame message,
and the System 1 frame gate getting the G1 head preset. Fakes only; no simulator, no LLM."""

from __future__ import annotations

import ast
import asyncio
import base64
import json
from pathlib import Path

import pytest

from ui import cameras as cams
from ui.cameras import CameraFeed, ScanThumbs, TapCameras, head_caption
from ui_fakes import JPEG_A, JPEG_B, FakeTap, make_deps

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------------------------- GT confinement
def test_ui_never_subscribes_to_gt_pose():
    """Only world/ reads ground truth on the runtime side: ui/ never names the gt.pose stream or its port, never
    reads a tap's pose, and every FrameTap it builds says gt_pose=False."""
    for p in sorted((ROOT / "ui").glob("*.py")):
        tree = ast.parse(p.read_text())
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and id(node) not in docs:
                assert node.value != 5601, f"{p.name}:{node.lineno} uses the gt.pose port"
                if isinstance(node.value, str):
                    assert "gt.pose" not in node.value and "gt_pub" not in node.value, f"{p.name}:{node.lineno}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and not node.args:
                assert node.func.attr != "pose", f"{p.name}:{node.lineno} reads a tap's gt.pose copy"
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "FrameTap":
                kw = {k.arg: k.value for k in node.keywords}
                assert isinstance(kw.get("gt_pose"), ast.Constant) and kw["gt_pose"].value is False, \
                    f"{p.name}:{node.lineno} FrameTap without gt_pose=False"


def test_tap_cameras_builds_a_frames_only_tap(monkeypatch):
    import viz.tap
    made = {}

    class Tap(FakeTap):
        def __init__(self, **kw):
            super().__init__()
            made.update(kw)
    monkeypatch.setattr(viz.tap, "FrameTap", Tap)
    TapCameras(port_offset=100)
    assert made == {"port_offset": 100, "gt_pose": False}


def test_frame_tap_gt_pose_flag_controls_the_5601_subscription(monkeypatch):
    """viz.tap.FrameTap(gt_pose=False) opens SUB sockets for the frames only (head 5565, frames 5602)."""
    zmq = pytest.importorskip("zmq")
    import viz.tap as tapmod
    connected: list[str] = []
    real_socket = zmq.Context.socket

    def socket(ctx, kind):
        s = real_socket(ctx, kind)
        orig = s.connect

        def connect(addr):
            connected.append(addr)
            return orig(addr)
        s.connect = connect
        return s
    monkeypatch.setattr(zmq.Context, "socket", socket)
    for flag in (False, True):
        connected.clear()
        t = tapmod.FrameTap(port_offset=31000, gt_pose=flag)
        for _ in range(50):
            if len(connected) >= 2:
                break
            asyncio.run(asyncio.sleep(0.02))
        t.close()
        ports = sorted(int(a.rsplit(":", 1)[1]) - 31000 for a in connected)
        assert ports == ([5565, 5601, 5602] if flag else [5565, 5602]), (flag, connected)


def test_tap_meta_drops_ground_truth_fields():
    tap = FakeTap()
    tap.put("chase", JPEG_A, {"t_sim": 2.0, "seq": 4, "source": "tp", "robot": {"pos": [1, 2, 0.8], "yaw": 0.3},
                              "cam_pose": {"pos": [0, 0, 3]}, "base_pos": [1, 2, 0.8], "yaw": 0.3, "stationary": True})
    tap.put("top", JPEG_B, {"seq": 1, "extent": [0.0, 0.0, 8.0, 8.0]})
    src = TapCameras(tap)
    m = src.meta("chase")
    assert not set(m) & cams.GT_META, m
    assert m["seq"] == 4 and m["source"] == "tp"
    assert src.meta("top")["extent"] == [0.0, 0.0, 8.0, 8.0]          # the render's extent is not a pose


# ---------------------------------------------------------------------------------------------- captions
def test_head_caption_says_sim_added():
    assert head_caption({}, {"name": "head_sim", "sim_added": True}, "sonic").startswith("camera: head (sim-added)")
    lite = head_caption({"source": "world", "camera": "head"}, {"name": "head_sim", "sim_added": True}, "lite")
    assert lite.startswith("camera: head (sim-added)") and "lite schematic, not a render" in lite
    d435 = head_caption({"camera": "head"}, {"name": "ego_d435", "sim_added": False}, "sonic")
    assert d435 == "camera: head (ego_d435) · what System 1 sees"
    assert head_caption({"sim_added": True}, {"name": "ego_d435", "sim_added": False}) \
        .startswith("camera: head (sim-added)")                            # the frame's own flag wins
    assert head_caption({}, {}) == cams.CAPTIONS["head"]
    assert "GR00T" in cams.CAPTIONS["ego"] and "ego_view" in cams.CAPTIONS["ego"]


def test_camera_feed_puts_the_caption_on_head_frames():
    tap = FakeTap()
    tap.put("head", JPEG_A, {"seq": 1})
    out = CameraFeed(TapCameras(tap), camera={"name": "head_sim", "sim_added": True}, profile="sonic").due(force=True)
    assert out[0]["meta"]["caption"] == "camera: head (sim-added) · what System 1 sees"


# ---------------------------------------------------------------------------------------------- scan thumbnails
def test_scan_thumbnails_follow_a_scan_and_close_on_its_result():
    tap = FakeTap()
    src = TapCameras(tap)
    th = ScanThumbs()
    active = {"runtime": {"active": [{"tool": "observe", "action": "scan", "id": "obs-000003"}]}}
    for i in range(9):
        tap.put("head", JPEG_A if i % 2 else JPEG_B, {"seq": i})
        assert th.update({"t": float(i), **active, "trace": []}, src) == []
    th.update({"t": 9.0, **active, "trace": []}, src)                      # same rev: not kept twice
    done = {"type": "result", "tool": "observe", "action": "scan", "execution_id": "obs-000003", "t": 10.0,
            "status": "succeeded", "observation_id": "obs-000003",
            "data": {"at": "bedroom_dresser_1b", "saw": ["alarm_clock_1"], "scan_executor": "turn_in_place",
                     "scan_note": "INTERIM: in-place turns"}}
    msgs = th.update({"t": 10.0, "runtime": {"active": []}, "trace": [done]}, src, "camera: head (sim-added)")
    assert len(msgs) == 1
    m = msgs[0]
    assert m["type"] == "scan" and m["at"] == "bedroom_dresser_1b" and m["sees"] == ["alarm_clock_1"]
    assert len(m["thumbs"]) == ScanThumbs.MAX_THUMBS and m["executor"] == "turn_in_place"
    assert base64.b64decode(m["thumbs"][0]["jpeg"]) in (JPEG_A, JPEG_B)    # undecodable fakes pass through
    assert th.recent == [m]
    # a glance is not a scan
    glance = dict(done, action="glance", execution_id="obs-000004")
    assert th.update({"t": 11.0, "runtime": {"active": []}, "trace": [glance]}, src) == []


def test_thumbnail_downscales_a_real_jpeg():
    pil = pytest.importorskip("PIL.Image")
    import io
    buf = io.BytesIO()
    pil.new("RGB", (640, 480), (200, 30, 30)).save(buf, "JPEG")
    small = cams.thumbnail(buf.getvalue())
    assert pil.open(io.BytesIO(small)).size == (160, 120)


# ---------------------------------------------------------------------------------------------- frame gate
def test_the_system1_frame_gate_gets_the_g1_head_preset():
    """ui.server's default gate passes the G1 preset by keyword. Passed positionally it landed in scene_cells and
    every later decide() raised (found in the first E-1 run)."""
    np = pytest.importorskip("numpy")
    from brains.frame_gate import G1_HEAD
    from ui.server import _default_frame_gate
    g = _default_frame_gate()
    assert g.config is G1_HEAD and g.scene_cells == G1_HEAD.scene_cells
    a = np.zeros((120, 160, 3), np.uint8)
    b = a.copy()
    b[:60] = 200
    assert g.decide(a, (0, 0, 0, 0), 0.0, expected=False).send
    assert g.decide(b, (0, 0, 0, 0), 5.0, expected=False).reason == "scene changed"


# ---------------------------------------------------------------------------------------------- server messages
class FakeLocations:
    def __init__(self) -> None:
        self.calls = 0

    def list(self, pose, query):
        self.calls += 1
        return [{"name": "bedroom_dresser_1a", "distance_m": 1.2, "type": "surface", "room": "bedroom"},
                {"name": "user", "distance_m": 2.6, "type": "person", "room": "kitchen"}]


def test_locations_panel_and_groot_strip_reach_the_page():
    websockets = pytest.importorskip("websockets")  # noqa: F841
    from websockets.asyncio.client import connect

    from ui.server import Hub, serve_hub
    built: dict = {}
    deps = make_deps(built)
    inner = deps.build

    def build(profile, scene, clock, log):
        w, r, f = inner(profile, scene, clock, log)
        r.locations = FakeLocations()
        return w, r, f
    deps.build = build

    async def go():
        hub = Hub("procthor-train-40", "full", deps=deps, cameras=TapCameras(FakeTap()), system1="off")
        server = await serve_hub(hub, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="full")
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                init = None
                while init is None:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    init = m if m["type"] == "init" else None
                locs = init["locations"]
                assert locs["source"] == "live" and [x["name"] for x in locs["items"]] == ["bedroom_dresser_1a", "user"]
                assert locs["planner"] is None
                rt = built["runtime"]
                rt.tracer.rows += [
                    {"t": 1.0, "type": "result", "tool": "list_locations", "execution_id": "loc-000001",
                     "status": "succeeded", "data": {"locations": [{"name": "user", "distance_m": 2.6}]}},
                    {"t": 2.0, "type": "started", "tool": "manipulate", "action": "pick", "execution_id": "man-000002",
                     "args": {"action": "pick", "skill_id": "groot.pick.alarm_clock.arena_static.v0"}}]
                hub.session.log.emit("groot.inference", session="man-000002", ok=True, latency_ms=150.0, chunk_idx=1)
                await hub.tick()
                frame = None
                while frame is None:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    frame = m if m["type"] == "frame" else None
                assert frame["locations"]["planner"]["names"] == ["user"]
                g = frame["groot"]
                assert g["session"] == "man-000002" and g["label"] == "experimental" and g["inferences"] == 1
                assert g["latency_ms"]["last"] == 150.0
                await hub.tick()
                m = None
                while m is None:
                    x = json.loads(await asyncio.wait_for(ws.recv(), 5))
                    m = x if x["type"] == "frame" else None
                assert "locations" not in m                                     # sent on change only
        finally:
            server.close()
            if hub.session:
                await hub.session.stop()
            hub.close()
    asyncio.run(asyncio.wait_for(go(), 20))


def test_page_has_the_m2b_panels():
    html = (ROOT / "ui" / "index.html").read_text()
    for needle in ('id="gstrip"', 'id="thumbs"', "renderGroot", "renderThumbs", "renderLocations",
                   "list_locations", 'msg.type === "scan"', "experimental", "clamped", "latency p50", "policy"):
        assert needle in html, needle


# ---------------------------------------------------------------------------------------------- the ego pane
class FakeSub:
    """world.frames.CameraSub's read API: (jpeg as sent, meta, rev, t_rx) or None."""

    def __init__(self) -> None:
        self.frame = None
        self.stopped = False

    def latest(self):
        return self.frame

    def stop(self) -> None:
        self.stopped = True


def test_ego_pane_shows_ego_view_only_while_it_is_rendered():
    import time as _time
    tap = FakeTap()
    tap.put("head", JPEG_A, {"seq": 1})
    sub = FakeSub()
    src = cams.with_ego(TapCameras(tap), port_offset=0, sub=sub)
    assert src.names() == ["head"]                                     # no consumer enabled ego_view: no pane
    sub.frame = (JPEG_B, {"camera": "ego_view", "seq": 3, "cam_pose_wl": [1, 2, 3, 4], "stationary": True,
                          "hfov": 70.0}, 1, _time.time())
    assert src.names() == ["head", "ego"] and src.rev("ego") == 1
    assert src.jpeg("ego") == JPEG_B                                   # undecodable fake: passed through unswapped
    m = src.meta("ego")
    assert m["camera"] == "ego_view" and m["hfov"] == 70.0 and not set(m) & cams.GT_META
    out = {x["which"]: x for x in CameraFeed(src).due(force=True)}
    assert "GR00T" in out["ego"]["meta"]["caption"]
    sub.frame = (JPEG_B, {}, 1, _time.time() - 10)                      # the camera stopped: stale frames vanish
    assert src.names() == ["head"] and src.jpeg("ego") is None
    src.close()
    assert sub.stopped


def test_scans_get_thumbnails_while_no_page_is_open():
    """Hub.tick with no client follows the runtime's history and trace itself; a page that connects later is sent
    the finished scan."""
    from ui.server import Hub
    from ui_fakes import FakeExecution
    built: dict = {}
    tap = FakeTap()

    async def go():
        hub = Hub("procthor-train-40", "sonic", deps=make_deps(built), cameras=TapCameras(tap), system1="off")
        await hub.reset("procthor-train-40", "agent", "gemini-3.8-flash", profile="sonic")
        rt = built["runtime"]
        e = FakeExecution("obs-000007", "observe", {"mode": "scan"}, action="scan", source="harness")
        rt.history.append(e)
        for i in range(4):
            tap.put("head", JPEG_A if i % 2 else JPEG_B, {"seq": i})
            await hub.tick()
        e.status = "succeeded"
        rt.tracer.rows.append({"t": 3.0, "type": "result", "tool": "observe", "action": "scan",
                               "execution_id": "obs-000007", "status": "succeeded",
                               "data": {"at": "bedroom_dresser_1a", "saw": ["alarm_clock_1"]}})
        await hub.tick()
        await hub.session.stop()
        return hub.thumbs.recent
    recent = asyncio.run(go())
    assert len(recent) == 1 and recent[0]["execution_id"] == "obs-000007" and len(recent[0]["thumbs"]) == 4
