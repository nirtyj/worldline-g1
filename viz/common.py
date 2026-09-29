"""Shared helpers for viz/server.py and viz/recorder.py (no Isaac imports; runs in viz/.venv).

Ports follow docs/contracts/m1.md section 0: every port shifts with one offset (build-phase tests use +100).
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from typing import Any, Iterable, Sequence

import msgpack
import numpy as np

BASE_PORTS = {
    "head": 5565,       # P1 PUB gear_sonic sensor_server msgpack (ego_view), one frame, no topic
    "gt_rep": 5600,     # P1 REP (JSON in -> JSON out)
    "gt_pub": 5601,     # P1 PUB multipart [b"gt.pose", msgpack]
    "frames": 5602,     # P1/VizCams PUB multipart [b"frame.<cam>", msgpack]
    "body_ctl": 5610,   # P3 ROUTER (we connect a DEALER)
    "body_evt": 5611,   # P3 PUB events
    "http": 8765,       # viz/server.py
    "rec_ctl": 5630,    # recorder control REP (server-spawned recorders pick a free port). NOT 5620: nav2 nav_bridge
}


def port_offset(default: int = 0) -> int:
    try:
        return int(os.environ.get("WL_PORT_OFFSET", default))
    except ValueError:
        return default


def ports(offset: int | None = None, **overrides: int | None) -> dict[str, int]:
    off = port_offset() if offset is None else offset
    p = {k: v + off for k, v in BASE_PORTS.items()}
    for k, v in overrides.items():
        if v is not None:
            p[k] = int(v)
    return p


def ep(port: int, host: str = "127.0.0.1") -> str:
    return f"tcp://{host}:{port}"


# ------------------------------------------------------------------------------------------------- decoding
def loads_any(payload: bytes) -> Any:
    """JSON (if it starts with { or [) or msgpack."""
    if not payload:
        return None
    if payload[:1] in (b"{", b"["):
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return msgpack.unpackb(payload, raw=False, strict_map_key=False)


def split_msg(frames: list[bytes]) -> tuple[str, Any]:
    """Topic + decoded payload from multipart [topic, payload] or a single frame (payload, or topic+payload)."""
    if len(frames) >= 2:
        topic = frames[0].decode("utf-8", "replace")
        try:
            return topic, loads_any(frames[-1])
        except Exception:  # noqa: BLE001
            return topic, None
    raw = frames[0]
    try:
        obj = loads_any(raw)
        if isinstance(obj, dict):
            return str(obj.get("topic", "")), obj
    except Exception:  # noqa: BLE001
        pass
    # single frame b"topic" [+ one separator byte] + payload (JSON or a msgpack map)
    for i, ch in enumerate(raw[:96]):
        if ch in (0x20, 0x00, 0x7C):  # space, NUL, '|'
            try:
                return raw[:i].decode(), loads_any(raw[i + 1:])
            except Exception:  # noqa: BLE001
                break
        if ch == 0x7B or 0x80 <= ch <= 0x8F or ch in (0xDE, 0xDF):  # '{', fixmap, map16, map32
            try:
                return raw[:i].decode(), loads_any(raw[i:])
            except Exception:  # noqa: BLE001
                break
    return "", None


def _to_jpeg_bytes(v: Any) -> bytes | None:
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        if b[:2] == b"\xff\xd8":
            return b
        try:
            return base64.b64decode(b)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(v, str):
        try:
            return base64.b64decode(v)
        except Exception:  # noqa: BLE001
            return None
    return None


def decode_head(raw: bytes, key: str = "ego_view") -> tuple[bytes | None, dict]:
    """gear_sonic sensor_server message -> (jpeg bytes as published, meta).

    Format (docs/contracts/m1.md 1.4; gear_sonic/utils/mujoco_sim/sensor_server.py:15-38):
      msgpack {"timestamps": {"ego_view": t}, "images": {"ego_view": <b64 str>}, "ego_view": <b64 str>, "t_sim"?, "seq"?}
    NOTE the JPEG was made by cv2.imencode on an RGB array, so a standard decoder (browser, PIL) shows R and B
    swapped. Use `swap_rb_jpeg` before showing it to humans.
    """
    obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    if not isinstance(obj, dict):
        return None, {}
    imgs = obj.get("images") or {}
    jpeg = _to_jpeg_bytes(imgs.get(key)) or _to_jpeg_bytes(obj.get(key))
    if jpeg is None and imgs:
        jpeg = _to_jpeg_bytes(next(iter(imgs.values())))
    ts = obj.get("timestamps") or {}
    meta = {"t_wall": ts.get(key), "t_sim": obj.get("t_sim"), "seq": obj.get("seq")}
    return jpeg, meta


# P1's own third-person camera (sim_isaac/app.py --tp-camera, contract 1.8) publishes multipart
# [b"frame.tp", msgpack{seq, t_sim, t_wall, jpeg, base_pos, yaw}] with the JPEG made by cv2.imencode on an RGB array
# (sim_isaac/camera.py FramePublisher), i.e. R/B swapped for standard decoders, like the head camera. It fills the
# chase pane. VizCams frames (v=1) are standard JPEGs and pass through untouched.
FRAME_ALIASES = {"tp": "chase"}
SWAPPED_RB_FRAMES = {"tp"}


def frame_from_msg(topic: str, msg: Any) -> tuple[str, bytes, dict, bool] | None:
    """A 5602 frame message -> (pane name, jpeg bytes as published, meta, needs R/B swap), or None.

    meta keeps the publisher's fields minus the JPEG, plus "source" (the topic suffix) and, for P1 frame.tp,
    "robot" {pos, yaw} built from its base_pos/yaw."""
    if not isinstance(msg, dict) or not topic.startswith("frame."):
        return None
    src = topic.split(".", 1)[1]
    jpeg = _to_jpeg_bytes(msg.pop("jpeg", None))
    if not jpeg:
        return None
    if msg.get("robot") is None and msg.get("base_pos") is not None:
        msg["robot"] = {"pos": list(msg["base_pos"]), "yaw": msg.get("yaw")}
    msg["source"] = src
    return FRAME_ALIASES.get(src, src), jpeg, msg, src in SWAPPED_RB_FRAMES


def same_frame(held: dict | None, new: dict | None) -> bool:
    """True when `new` is a re-send of the frame whose meta is `held` (VizCams re-sends the last top snapshot every
    2 s for late subscribers, with the same seq and t_wall)."""
    if not held or not new:
        return False
    return (new.get("seq") is not None and new.get("seq") == held.get("seq")
            and new.get("t_wall") == held.get("t_wall") and new.get("source") == held.get("source"))


# ------------------------------------------------------------------------------------------------- images
def swap_rb_jpeg(jpeg: bytes, quality: int = 80) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    r, g, b = im.split()
    out = io.BytesIO()
    Image.merge("RGB", (b, g, r)).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def decode_jpeg(jpeg: bytes, swap_rb: bool = False, max_size: tuple[int, int] | None = None):
    """-> PIL RGB image. `max_size` lets libjpeg decode at 1/2, 1/4, 1/8 scale (draft mode) for speed."""
    from PIL import Image

    im = Image.open(io.BytesIO(jpeg))
    if max_size is not None:
        im.draft("RGB", max_size)
    im = im.convert("RGB")
    if swap_rb:
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
    return im


def encode_jpeg(im, quality: int = 80) -> bytes:
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=quality)
    return out.getvalue()


_FONT_CACHE: dict[int, Any] = {}


def font(size: int = 16):
    from PIL import ImageFont

    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    f = None
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Menlo.ttc", "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                break
            except Exception:  # noqa: BLE001
                pass
    if f is None:
        try:
            f = ImageFont.load_default(size=size)
        except TypeError:
            f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


# ------------------------------------------------------------------------------------------------- top-down geometry
class TopMap:
    """World <-> pixel mapping for an axis-aligned top-down image. extent = [xmin, ymin, xmax, ymax];
    image +x = world +x (right), image up = world +y (P1 render_topdown and VizCams convention)."""

    def __init__(self, extent: Sequence[float], w: int, h: int):
        self.x0, self.y0, self.x1, self.y1 = (float(v) for v in extent)
        self.w, self.h = int(w), int(h)

    def to_px(self, x: float, y: float) -> tuple[float, float]:
        u = (x - self.x0) / (self.x1 - self.x0) * self.w
        v = (self.y1 - y) / (self.y1 - self.y0) * self.h
        return u, v

    def to_world(self, u: float, v: float) -> tuple[float, float]:
        x = self.x0 + u / self.w * (self.x1 - self.x0)
        y = self.y1 - v / self.h * (self.y1 - self.y0)
        return x, y

    @property
    def m_per_px(self) -> float:
        return (self.x1 - self.x0) / self.w


def draw_robot_overlay(im, tm: TopMap, traj: Iterable[Sequence[float]], pose: dict | None,
                       target: Sequence[float] | None = None, color=(255, 80, 40)) -> None:
    """Draw the trajectory (list of (x, y)), the robot arrow and an optional go_to target onto PIL image `im`."""
    from PIL import ImageDraw

    d = ImageDraw.Draw(im)
    pts = [tm.to_px(p[0], p[1]) for p in traj]
    if len(pts) >= 2:
        d.line(pts, fill=(40, 200, 255), width=max(2, im.width // 300))
    if target is not None:
        u, v = tm.to_px(target[0], target[1])
        r = max(5, im.width // 90)
        d.ellipse([u - r, v - r, u + r, v + r], outline=(80, 255, 120), width=3)
        d.line([u - r, v, u + r, v], fill=(80, 255, 120), width=2)
        d.line([u, v - r, u, v + r], fill=(80, 255, 120), width=2)
    if pose is not None and pose.get("base_pos") is not None:
        x, y = pose["base_pos"][0], pose["base_pos"][1]
        yaw = float(pose.get("yaw") or 0.0)
        u, v = tm.to_px(x, y)
        L = max(10.0, 0.45 / max(tm.m_per_px, 1e-6))
        L = min(L, im.width / 12)
        tip = (u + L * math.cos(yaw), v - L * math.sin(yaw))
        left = (u + 0.45 * L * math.cos(yaw + 2.5), v - 0.45 * L * math.sin(yaw + 2.5))
        right = (u + 0.45 * L * math.cos(yaw - 2.5), v - 0.45 * L * math.sin(yaw - 2.5))
        col = (255, 40, 40) if pose.get("fallen") else color
        d.polygon([tip, left, (u, v), right], fill=col, outline=(0, 0, 0))


def occupancy_rgba(npz_path: str, inflated: bool = False):
    """Occupancy npz -> (PIL RGBA image with image-up = world +y, extent [xmin,ymin,xmax,ymax]).

    Accepts P1 get_occupancy npz (docs/contracts/m1.md 1.6: occ / occ_inflated, uint8 1 = blocked) and the scene
    asset npz (assets/houses/<id>/occupancy.npz: raw 0 free / 1 obstacle / 2 outside, inflated). Both have
    resolution and origin [x0,y0] = corner of cell (row 0, col 0); cell (r, c) covers x in [x0 + c*res, ...).
    """
    from PIL import Image

    z = np.load(npz_path)
    files = list(z.files)
    okey = "occ" if "occ" in files else ("raw" if "raw" in files else files[0])
    ikey = "occ_inflated" if "occ_inflated" in files else ("inflated" if "inflated" in files else None)
    occ = np.asarray(z[okey]).astype(np.uint8)
    res = float(np.asarray(z["resolution"]).reshape(-1)[0])
    x0, y0 = (float(v) for v in np.asarray(z["origin"]).reshape(-1)[:2])
    rows, cols = occ.shape
    rgba = np.zeros((rows, cols, 4), np.uint8)
    if ikey is not None and not inflated:
        infl = (np.asarray(z[ikey]) > 0) & (occ == 0)
        rgba[infl] = (255, 170, 60, 55)
    rgba[occ == 1] = (235, 235, 235, 160)
    rgba[occ >= 2] = (0, 0, 0, 150)       # outside the house
    im = Image.fromarray(np.flipud(rgba), "RGBA")  # row 0 = y0 (bottom) -> image bottom
    return im, [x0, y0, x0 + cols * res, y0 + rows * res]


def json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (bytes, bytearray)):
        return f"<{len(o)} bytes>"
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=json_default)


def pose_summary(p: dict | None) -> dict | None:
    """The fields of gt.pose the UI and recorder use (tolerates missing keys)."""
    if not p:
        return None
    pos = p.get("base_pos") or [p.get("x"), p.get("y"), p.get("z")]
    yaw = p.get("yaw")
    if yaw is None and p.get("base_quat_wxyz"):
        w, x, y, z = p["base_quat_wxyz"]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = {
        "t_sim": p.get("t_sim"), "t_wall": p.get("t_wall"), "rtf": p.get("rtf"),
        "base_pos": [round(float(v), 4) for v in pos] if pos and pos[0] is not None else None,
        "yaw": round(float(yaw), 4) if yaw is not None else None,
        "pelvis_z": p.get("pelvis_z"), "fallen": p.get("fallen"), "foot_contact": p.get("foot_contact"),
        "band": p.get("band"), "lowcmd_age_s": p.get("lowcmd_age_s"),
    }
    if p.get("note"):
        out["note"] = p["note"]
    return out


def is_dynamic_prop(o: Any) -> bool:
    """A get_scene_info object the sim can move (docs/contracts/p1_m2b.md §3.1: a rigid body that is neither static
    nor articulated). These are the viewer's items; furniture keeps its load pose and is not drawn as one."""
    return isinstance(o, dict) and bool(o.get("body_path")) and not o.get("is_static") and not o.get("articulated")


def item_summary(o: Any, floor_z: float = 0.0) -> dict | None:
    """One P1 object record (gt.objects, get_objects or get_scene_info, docs/contracts/p1_m2b.md §3) as the viewer's
    compact item {id, name, x, y, z, held_by}: x, y the world AABB centre (`pos` when there is no box), z the box
    bottom above the floor, all rounded to 1 cm so physics jitter is not a change."""
    if not isinstance(o, dict) or o.get("id") is None:
        return None
    box, pos = o.get("aabb"), o.get("pos")
    try:
        if box and len(box) == 2 and len(box[0]) >= 3 and len(box[1]) >= 3:
            lo, hi = box
            x, y, z = (float(lo[0]) + float(hi[0])) / 2, (float(lo[1]) + float(hi[1])) / 2, float(lo[2])
        elif pos and len(pos) >= 2:
            x, y = float(pos[0]), float(pos[1])
            z = float(pos[2]) if len(pos) > 2 else float(floor_z)
        else:
            return None
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in (x, y, z)):
        return None
    held = o.get("held_by")
    return {"id": str(o["id"]), "name": str(o.get("name") or o["id"]), "x": round(x, 2), "y": round(y, 2),
            "z": round(z - float(floor_z or 0.0), 2), "held_by": held if held in ("left", "right") else None}
