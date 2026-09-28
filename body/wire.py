"""Wire formats used by wl-body.

1. SONIC input (PUB 5556 -> gear_sonic_deploy zmq_manager SUB), topics "command" and "planner".
   Layout: [topic bytes][1280-byte null-padded JSON header][packed little-endian payload], one ZMQ frame.
   The builders below are a verbatim port of
     WBC gear_sonic/utils/teleop/zmq/zmq_planner_sender.py:14-158 (HEADER_SIZE = 1280 at :14;
     the module docstring's "1024" is stale, see README news 2026-03-24 "ZMQ header size changed to 1280").
   Receiver: gear_sonic_deploy/.../include/input_interface/zmq_packed_message_subscriber.hpp:99 (HEADER_SIZE 1280),
     :272-329 (topic-prefix strip, header parse, payload offsets).
   Field decoding: zmq_manager.hpp:668-760 (command: start/stop/planner as u8|bool|i32),
     :766-980 (planner: mode i32 required, movement/facing f32|f64 [3] required, speed/height optional).
   body/tests/test_wire.py checks byte equality against the upstream module when WBC_DIR is available.

2. LocomotionMode enum: gear_sonic_deploy/.../include/localmotion_kplanner.hpp:78-106.

3. g1_debug (deploy PUB 5557): [b"g1_debug"][msgpack map], one frame
   (output_interface/zmq_output_handler.hpp:1-60).

4. Camera (P1 PUB 5565): msgpack {"timestamps": {name: t}, "images": {name: base64(JPEG q80)}}, one frame, no topic
   (gear_sonic/utils/mujoco_sim/sensor_server.py:15-38, 57-64, 73-83). The JPEG is produced with cv2.imencode on an
   RGB array and consumers decode with cv2.imdecode, so the cv2 round-trip yields RGB (image_publish_utils.py:165-185).

5. gt.pose (P1 PUB 5601): preferred multipart [b"gt.pose", msgpack]; a single frame b"gt.pose"+msgpack is accepted.
"""

from __future__ import annotations

import base64
import json
import math
import struct
from typing import Any, Sequence

import numpy as np

HEADER_SIZE = 1280  # zmq_planner_sender.py:14, zmq_packed_message_subscriber.hpp:99


class LocomotionMode:
    """localmotion_kplanner.hpp:78-106 (only the ones M1 uses are named here)."""

    IDLE = 0
    SLOW_WALK = 1   # 0.1..0.8 m/s per enum comment; keyboard clamps 0.2..0.8 (keyboard_handler.hpp:267-272)
    WALK = 2        # 0.8..2.5 m/s, keyboard forces speed -1 (mode default)
    RUN = 3

    STATIC = {0, 4, 5, 6, 7, 9}  # is_static_motion_mode, localmotion_kplanner.hpp:108-115

    NAMES = {0: "idle", 1: "slowWalk", 2: "walk", 3: "run"}


# ----------------------------------------------------------------------------------------------
# Builders: verbatim port of zmq_planner_sender.py (only the fields M1 uses are kept optional).
# ----------------------------------------------------------------------------------------------

def _build_header(fields: list, version: int = 1, count: int = 1) -> bytes:
    # zmq_planner_sender.py:17-27
    header = {"v": version, "endian": "le", "count": count, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def build_command_message(start: bool, stop: bool, planner: bool,
                          delta_heading: float | None = None) -> bytes:
    """zmq_planner_sender.py:30-61. NOTE: zmq_manager.hpp OnCommandReceived (:668-760) ignores delta_heading."""
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = b"".join((
        struct.pack("B", 1 if start else 0),
        struct.pack("B", 1 if stop else 0),
        struct.pack("B", 1 if planner else 0),
    ))
    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))
    header = _build_header(fields, version=1, count=1)
    return b"command" + header + payload


def build_planner_message(
    mode: int,
    movement: Sequence[float],
    facing: Sequence[float],
    speed: float = -1.0,
    height: float = -1.0,
    upper_body_position: Sequence[float] | None = None,
    upper_body_velocity: Sequence[float] | None = None,
    left_hand_position: Sequence[float] | None = None,
    right_hand_position: Sequence[float] | None = None,
) -> bytes:
    """zmq_planner_sender.py:64-137 (VR fields omitted; not used by M1)."""
    if len(movement) != 3:
        raise ValueError("movement must have length 3")
    if len(facing) != 3:
        raise ValueError("facing must have length 3")
    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]
    payload = b"".join((
        struct.pack("<i", int(mode)),
        struct.pack("<fff", float(movement[0]), float(movement[1]), float(movement[2])),
        struct.pack("<fff", float(facing[0]), float(facing[1]), float(facing[2])),
        struct.pack("<f", float(speed)),
        struct.pack("<f", float(height)),
    ))
    for name, vals in (("upper_body_position", upper_body_position),
                       ("upper_body_velocity", upper_body_velocity),
                       ("left_hand_joints", left_hand_position),
                       ("right_hand_joints", right_hand_position)):
        if vals is not None:
            fields.append({"name": name, "dtype": "f32", "shape": [len(vals)]})
            for v in vals:
                payload += struct.pack("<f", float(v))
    header = _build_header(fields, version=1, count=1)
    return b"planner" + header + payload


# ----------------------------------------------------------------------------------------------
# Decoder mirroring ZMQPackedMessageSubscriber (used by tools/fake_deploy.py and tests).
# ----------------------------------------------------------------------------------------------

_DTYPES = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8", "u8": "u1", "bool": "u1"}


def decode_packed(msg: bytes, topic: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Strip the topic prefix, parse the 1280-byte JSON header, slice fields in order.

    Mirrors zmq_packed_message_subscriber.hpp:272-329. Raises ValueError on mismatch.
    """
    t = topic.encode()
    if not msg.startswith(t):
        raise ValueError("topic mismatch")
    body = msg[len(t):]
    if len(body) < HEADER_SIZE:
        raise ValueError("message smaller than header")
    raw = body[:HEADER_SIZE]
    end = raw.find(b"\x00")
    hdr = json.loads(raw[: end if end >= 0 else HEADER_SIZE].decode("utf-8"))
    data = body[HEADER_SIZE:]
    off = 0
    out: dict[str, np.ndarray] = {}
    for f in hdr["fields"]:
        dt = np.dtype(_DTYPES[f["dtype"]])
        n = int(np.prod(f["shape"])) if f["shape"] else 1
        nbytes = n * dt.itemsize
        if off + nbytes > len(data):
            raise ValueError(f"payload too short for field {f['name']}")
        out[f["name"]] = np.frombuffer(data, dtype=dt, count=n, offset=off).reshape(f["shape"])
        off += nbytes
    return hdr, out


def decode_command(msg: bytes) -> dict:
    _, f = decode_packed(msg, "command")
    for k in ("start", "stop", "planner"):
        if k not in f:
            raise ValueError("Command missing fields (need: start, stop, planner)")  # zmq_manager.hpp:684-687
    return {"start": bool(f["start"][0]), "stop": bool(f["stop"][0]), "planner": bool(f["planner"][0])}


def decode_planner(msg: bytes) -> dict:
    _, f = decode_packed(msg, "planner")
    for k in ("mode", "movement", "facing"):
        if k not in f:
            raise ValueError("Planner missing required fields")  # zmq_manager.hpp:787-790
    out = {
        "mode": int(f["mode"][0]),
        "movement": [float(x) for x in f["movement"]],
        "facing": [float(x) for x in f["facing"]],
        "speed": float(f["speed"][0]) if "speed" in f else -1.0,
        "height": float(f["height"][0]) if "height" in f else -1.0,
    }
    for k in ("upper_body_position", "left_hand_joints", "right_hand_joints"):
        if k in f:
            out[k] = [float(x) for x in f[k]]
    return out


# ----------------------------------------------------------------------------------------------
# Generic payload helpers
# ----------------------------------------------------------------------------------------------

def loads_any(payload: bytes) -> Any:
    """Decode a JSON or msgpack payload (P1 REP may answer either)."""
    import msgpack

    if not payload:
        return None
    first = payload[:1]
    if first in (b"{", b"["):
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return msgpack.unpackb(payload, raw=False, strict_map_key=False)


def dumps_json(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), default=_json_default).encode("utf-8")


def _json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (bytes, bytearray)):
        return base64.b64encode(o).decode()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def split_topic(frames: list[bytes], topic: bytes) -> bytes | None:
    """Return the payload of a topic message, multipart [topic, payload] or single-frame topic+payload."""
    if len(frames) >= 2 and frames[0] == topic:
        return frames[-1]
    if len(frames) == 1 and frames[0].startswith(topic):
        rest = frames[0][len(topic):]
        # allow one separator byte (space / NUL / '|') between topic and payload
        if rest[:1] in (b" ", b"\x00", b"|"):
            rest = rest[1:]
        return rest
    return None


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat_wxyz(q: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_wxyz_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


# ----------------------------------------------------------------------------------------------
# gt.pose
# ----------------------------------------------------------------------------------------------

class Pose:
    """Parsed gt.pose (contract m1.md). Missing fields default sensibly so partial P1s still work."""

    __slots__ = ("t_sim", "t_wall", "rtf", "x", "y", "z", "quat", "yaw", "vx", "vy", "wz",
                 "pelvis_z", "fallen", "contact_l", "contact_r", "recv_mono", "raw")

    def __init__(self, d: dict, recv_mono: float):
        self.raw = d
        self.recv_mono = recv_mono
        self.t_sim = _num(d.get("t_sim"), 0.0)
        self.t_wall = _num(d.get("t_wall"), 0.0)
        self.rtf = _num(d.get("rtf"), float("nan"))       # m1.md §1.5: rtf may be null during warm-up
        pos = d.get("base_pos") or [0.0, 0.0, 0.0]
        self.x, self.y, self.z = float(pos[0]), float(pos[1]), float(pos[2])
        self.quat = list(d.get("base_quat_wxyz") or [1.0, 0.0, 0.0, 0.0])
        yaw = d.get("yaw")
        self.yaw = float(yaw) if yaw is not None else yaw_from_quat_wxyz(self.quat)
        v = d.get("base_lin_vel_w") or [0.0, 0.0, 0.0]
        self.vx, self.vy = float(v[0]), float(v[1])
        w = d.get("base_ang_vel_w") or [0.0, 0.0, 0.0]
        self.wz = float(w[2])
        self.pelvis_z = _num(d.get("pelvis_z"), self.z)
        self.fallen = bool(d.get("fallen", False))
        fc = d.get("foot_contact") or {}
        self.contact_l = bool(fc.get("left", False))
        self.contact_r = bool(fc.get("right", False))

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def brief(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "yaw": round(self.yaw, 4),
                "pelvis_z": round(self.pelvis_z, 4), "v": round(self.speed, 3), "wz": round(self.wz, 3),
                "fallen": self.fallen, "t_sim": round(self.t_sim, 3)}


def _num(v, default: float) -> float:
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def parse_pose(payload: bytes, recv_mono: float) -> Pose:
    return Pose(loads_any(payload), recv_mono)


# ----------------------------------------------------------------------------------------------
# Camera (sensor_server format)
# ----------------------------------------------------------------------------------------------

def decode_camera_message(payload: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Returns (timestamps, {name: HxWx3 uint8}) exactly like ImageMessageSchema.deserialize + ImageUtils.decode_image."""
    import cv2
    import msgpack

    d = msgpack.unpackb(payload, raw=False)
    images = {}
    for k, v in (d.get("images") or {}).items():
        if isinstance(v, str):
            arr = np.frombuffer(base64.b64decode(v), dtype=np.uint8)
            images[k] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif isinstance(v, (bytes, bytearray)):
            images[k] = cv2.imdecode(np.frombuffer(v, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            images[k] = np.asarray(v)
    return d.get("timestamps") or {}, images


def encode_camera_message(images: dict[str, np.ndarray], t: float) -> bytes:
    """Producer side (used by tools/fake_p1.py): same bytes as SensorServer.send_message(ImageMessageSchema.serialize())."""
    import cv2
    import msgpack

    enc = {}
    for k, img in images.items():
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        enc[k] = base64.b64encode(buf).decode("utf-8")
    return msgpack.packb({"timestamps": {k: t for k in images}, "images": enc}, use_bin_type=True)
