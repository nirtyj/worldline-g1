"""Ground truth for the page (occupancy, rooms, truth snapshots) and the robot's own map."""

from __future__ import annotations

import base64
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ui.png import png_size
from ui.robot_map import RobotMap, near_edge_m
from ui.truth import (DISPLAY_STEP, GT_LABEL, display_cells, grid_view, layout_message, occupancy_png,
                      profile_fallbacks, stepping_stones, topdown_from_extent, truth_payload, world_grid, yaw_map_deg)
from ui_fakes import FakeOccupancy, FakeRobot, FakeWorld

ROOT = Path(__file__).resolve().parents[2]


def test_grid_view_and_display_cells_follow_the_occupancy():
    g = grid_view(FakeOccupancy())
    assert g.shape == (60, 80) and g.extent == (0.0, 0.0, 4.0, 3.0)
    walk = display_cells(g, DISPLAY_STEP, "walk")
    floor = display_cells(g, DISPLAY_STEP, "floor")
    assert walk < floor, "the nav grid (inflated) is a strict subset of the free floor"
    # THOR indexing: cell (ix, iz) is centred on (ix*0.25, iz*0.25); the wall at x = 2.0 is not floor
    assert (8, 2) not in floor and (8, 6) in floor, "the door gap at z 1.25-1.75 m is floor"
    assert (4, 6) in walk and (12, 6) in walk, "both rooms have nav cells"
    assert not any(ix == 8 for ix, _ in walk if _ < 5), "no nav cell in the wall"


def test_occupancy_png_is_a_valid_image_top_row_is_max_z():
    g = grid_view(FakeOccupancy())
    png = occupancy_png(g)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and png_size(png) == (80, 60)
    try:
        from PIL import Image
    except ImportError:
        return
    import io
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    assert im.getpixel((0, 0))[3] == 0, "outside is transparent"
    assert im.getpixel((39, 59 - 5))[:3] == (200, 196, 180), "the wall (raw obstacle) at z=0.25"
    assert im.getpixel((39, 59 - 30))[:3] != (200, 196, 180), "the door at z=1.5 is not an obstacle"


def test_topdown_mapping_fits_the_extent_with_square_pixels():
    t = topdown_from_extent([0.0, 0.0, 8.664, 8.664], 640, 480)
    k = t["h"] / (2 * t["size"])
    u0 = t["w"] / 2 + (0 - t["cx"]) * k
    u1 = t["w"] / 2 + (8.664 - t["cx"]) * k
    v_top = t["h"] / 2 - (8.664 - t["cz"]) * k
    assert u0 >= -1e-6 and u1 <= 640 + 1e-6 and v_top >= -1e-6
    wide = topdown_from_extent([-4.4, -2.8, 1.5, 3.2])
    assert wide["size"] >= 3.0 - 1e-6


def test_layout_message_from_robot_map_and_world():
    w, r = FakeWorld(), None
    r = FakeRobot(w)
    lay = layout_message("procthor-train-40", r.lookup_keypoints(), w, world_grid(w), profile="sonic")
    assert lay["user_surface"] == "kitchen_dining_table_1b" and lay["human"]["x"] == 3.0
    assert lay["keypoints"]["bedroom_dresser_1a"] == {"x": 0.8, "z": 2.3, "yaw": 0.0}
    assert lay["surfaces"]["bedroom_dresser_1a"]["height"] == 0.97
    assert lay["rooms"]["kitchen"]["x"] == 3.0 and lay["rooms"]["kitchen"]["polygon"][0] == [2.0, 0.0]
    assert base64.b64decode(lay["occupancy"]["png"])[:4] == b"\x89PNG"
    assert lay["truth_label"] == GT_LABEL
    json.dumps(lay)


def test_truth_payload_shapes():
    w = FakeWorld()
    t = truth_payload(w.truth(), {"moving": True, "at": "start", "between": ["start", "x"]})
    assert t["label"] == GT_LABEL and t["source"] == "isaac-gt"
    assert t["robot"]["at"] is None and t["robot"]["moving"] and t["robot"]["between"] == ["start", "x"]
    assert t["objects"]["alarm_clock_1"] == {"type": "alarm_clock", "label": "alarm clock", "where": "bedroom_dresser_1a",
                                             "x": 0.8, "y": 1.05, "z": 2.6, "visible": True}
    # an Isaac-frame Pose3D (z up) and a held object
    snap = {"robot": {"pose": {"x": 2.0, "y": 3.0, "z": 0.78, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}},
            "objects": [{"id": "apple_1", "type": "apple", "pose": {"x": 1.0, "y": 2.0, "z": 0.9}, "where": None,
                         "held_by": "right"}]}
    t = truth_payload(snap)
    assert (t["robot"]["x"], t["robot"]["z"]) == (2.0, 3.0)
    assert t["robot"]["yaw"] == pytest.approx(90.0), "Isaac yaw 0 (facing +x) is Worldline yaw 90"
    assert t["objects"]["apple_1"]["where"] == "hand:right" and t["objects"]["apple_1"]["y"] == 0.9
    assert t["arms"]["right"]["holding"] == "apple_1" and t["arms"]["left"]["holding"] is None


def test_yaw_conversion_matches_plan_6_2_3():
    assert yaw_map_deg(0.0) == pytest.approx(90.0)           # +x
    assert yaw_map_deg(math.pi / 2) == pytest.approx(0.0)    # +y = +z (map)
    assert yaw_map_deg(math.pi) == pytest.approx(270.0)


def test_stepping_stones_and_profile_fallbacks():
    s = stepping_stones("sonic")
    assert {"kinematic_nav", "kinematic_attach", "sonic_arm_script"} <= s
    assert "sonic_walk" not in s and "groot_sonic" not in s
    assert set(profile_fallbacks("bringup")) == {"kinematic_nav", "kinematic_attach"}
    assert profile_fallbacks("sonic") == ("sonic_arm_script",)
    assert profile_fallbacks("lite") == ()


# ---------------------------------------------------------------------------------------------- robot map
def _open_room(n=40):
    return {(ix, iz) for ix in range(-n, n) for iz in range(-n, n)}


def test_near_edge_of_the_head_camera_frustum():
    assert near_edge_m(1.35, 15.0, 73.7) == pytest.approx(1.35 / math.tan(math.radians(51.85)), rel=1e-6)
    assert near_edge_m(1.35, 35.0, 73.7) < near_edge_m(1.35, 15.0, 73.7)
    assert near_edge_m(1.0, 60.0, 73.7) == 0.0


def test_walking_covers_a_90_degree_fan_beyond_the_near_edge():
    m = RobotMap(_open_room(), 0.25)
    m.pose(0.0, 0.0, 0.0, 0.0, looking=False)               # facing +z
    cells = m.explored
    near = near_edge_m(1.35, 15.0, 73.7)
    assert (0, 0) in cells, "its own cell always counts"
    assert (0, 8) in cells and (0, 10) in cells, "2.0 and 2.5 m straight ahead"
    assert (0, 2) not in cells and near > 0.5, "0.5 m ahead is under the camera's lower edge"
    assert (0, -8) not in cells and (8, 0) not in cells, "behind and 90 deg to the side are out of view"
    assert all(math.hypot(ix * 0.25, iz * 0.25) <= 2.5 + 1e-9 for ix, iz in cells)


def test_a_scan_covers_a_wider_fan_and_closer_floor():
    walk = RobotMap(_open_room(), 0.25)
    walk.pose(0.0, 0.0, 0.0, 0.0, looking=False)
    scan = RobotMap(_open_room(), 0.25)
    scan.pose(0.0, 0.0, 0.0, 0.0, looking=True)
    assert walk.explored < scan.explored
    assert (6, 2) in scan.explored and (6, 2) not in walk.explored, "+-80 deg while scanning"
    assert (0, 2) in scan.explored, "the pitched row sees 0.5 m ahead"
    f = scan.frustum(0.0, 0.0, 0.0, True)
    assert [r["tilt"] for r in f["rows"]] == [15.0, 35.0] and f["range"] == 2.5


def test_line_of_sight_is_blocked_by_non_floor_cells():
    floor = _open_room() - {(ix, 4) for ix in range(-40, 40)}          # a wall at z = 1.0 m
    m = RobotMap(floor, 0.25, floor)
    m.pose(0.0, 0.0, 0.0, 0.0, looking=True)
    assert not any(iz > 4 for _, iz in m.explored)


def test_trail_and_sightings_stream_as_new_items():
    m = RobotMap(_open_room(), 0.25)
    m.pose(0.0, 0.0, 0.0, 0.0, False)
    m.pose(0.1, 0.01, 0.0, 1.0, False)                      # under 5 cm and 5 deg: no new trail point
    m.pose(0.2, 0.2, 0.0, 0.0, False)
    m.found(0.3, "alarm_clock_1", "bedroom_dresser_1a", 0.2, 0.0)
    new = m.take_new()
    assert len(new["trail"]) == 2 and new["sightings"][0]["object"] == "alarm_clock_1"
    assert m.take_new() == {"trail": [], "explored": [], "sightings": []}
    assert len(m.full()["trail"]) == 2


# ---------------------------------------------------------------------------------------------- page scripts
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_page_scripts_parse(tmp_path):
    for js in (ROOT / "ui" / "views").glob("*.js"):
        subprocess.run([NODE, "--check", str(js)], check=True)
    html = (ROOT / "ui" / "index.html").read_text()
    inline = re.findall(r"<script>([\s\S]*?)</script>", html)
    assert len(inline) == 1
    (tmp_path / "inline.js").write_text(inline[0])
    subprocess.run([NODE, "--check", str(tmp_path / "inline.js")], check=True)


def test_page_has_no_thor_status_literals_or_old_tools():
    """The page speaks the new vocabulary: lowercase envelopes, the six tools (+recall), executors."""
    html = (ROOT / "ui" / "index.html").read_text()
    js = html + "".join(p.read_text() for p in (ROOT / "ui" / "views").glob("*.js"))
    for needle in ("sim ground truth", "not visible to the planner", "Kill controller", "data-cam=\"chase\"",
                   "stepping_stones", "tool_state"):
        assert needle in js, needle
    assert "ProcTHOR house\", { wide" not in js and "AI2-THOR simulator" not in js
    assert "grid paths · 0.6 m/s" not in js, "THOR's teleport speed is gone from the diagram"
