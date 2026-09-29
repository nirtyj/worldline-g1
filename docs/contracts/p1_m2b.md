# P1 (wl-isaac) M2b additions: the wire

Status: **v1 (isaac owner, M2b wave 1, 2026-09-29).** This document is the wire for P1.1-P1.10 of `docs/M2.md` §7.1
plus the two-camera decision OD1. It extends `docs/contracts/m1.md` §0-§1; everything there still holds unless a
section below says otherwise. The runtime side (`world/isaac_client.py`, `tests/fakes/fake_p1_world.py` with
`m2b=True`) already speaks P1.1-P1.3; this contract keeps exactly that wire and extends it. Implementation:
`sim_isaac/` (`app.py`, `cameras.py`, `objects.py`, `segment.py`, `wire.py`); the kinematic fake is
`tools/fake_p1.py`. Measured results: §13.

Tags: **[v]** read in code (file:line). **[m]** measured on the dev box `ludo-g1-arena` (run named in §13).
**[u]** not verified yet.

Every stepping stone is labelled: attach/detach is **STEPPING STONE** (the reply carries `stepping_stone: true`);
`move_object`, `push_object` and `set_object_pose` are **test-only** writes (counted in `get_stats.object_writes`
and announced on `gt.event`).

---

## 0. Conventions (all ops and topics)

| Item | Convention |
|---|---|
| World frame | Isaac world: x, y on the floor, z up, metres (the frame of `gt.pose`, m1.md §1.5). Worldline's map frame appears only in `cam_pose_wl` (§5.3) and follows `world/coords.py` |
| Angles | radians unless the key ends in `_deg` |
| Quaternions | `quat_wxyz` = [w, x, y, z], world-from-body |
| Positions of objects | `pos` = the object's root prim origin (as `get_scene_info.objects[].pos`); `aabb` = [[xmin,ymin,zmin],[xmax,ymax,zmax]] world axis-aligned box of the visual geometry. The object **centre** is the AABB centre |
| Scene ids | `id` = the scene object id of `get_scene_info` (THOR id, e.g. `"AlarmClock|surface|2|1"`); world maps it to its oid through `scene_id` |
| Times | `t_sim` = sim seconds since P1's physics start; `t_wall` = `time.time()` (epoch s); `*_mono` = `time.monotonic()` on the P1 box (only comparable on the same box) |
| Requests | JSON or msgpack map with `"op"`; arguments at the top level **or** under `"args"` (both are read; `body.p1_client.P1Rpc` sends both) |
| Replies | `ok: bool`; on failure `error: str` (starts with the code) and `code: str` (§11) |

---

## 1. Ports

| Port (+offset) | Kind | Content | M1 | M2b |
|---|---|---|---|---|
| 5565 | PUB | **head** camera, gear_sonic `sensor_server` format (§5.2) | the d435-mounted camera, key `ego_view` | the System 1 head camera, key `head` (+ §5.3 metadata). Same format, so M1 tools, `BodyClient.camera_frame`, `viz` and the recorder keep working (they take the first image when `ego_view` is absent) |
| **5566** | PUB | **ego_view** camera for GR00T, gear_sonic format, key `ego_view` (§5.2) | - | **new**. Only while a consumer has it enabled (§5.4) |
| 5600 | REP | ops (m1.md §1.6 + §2-§10 here) | | extended |
| 5601 | PUB | `gt.pose` 50 Hz (+ `links`, §6), **`gt.objects` 10 Hz** (§3.2), `gt.event` (+ M2b events, §10.2), **`sim.health` 1 Hz** (§10.1) | `gt.pose`, `gt.event` | extended |
| 5602 | PUB | unchanged: VizCams `frame.<cam>` or P1 `--tp-camera` `frame.tp` (m1.md §1.8, docs/viz.md §3) | | unchanged |

Why not `frame.head` on 5602 (PLAN §3.3, M2.md P1.4 wording): 5602 is bound by VizCams (`--viz`) or by P1's
`--tp-camera`, one of them or neither, and FrameTap files both 5565 and a 5602 `frame.head` under the same name
`head`, with different colour conventions. The head camera therefore replaces the M1 camera on 5565, where every
consumer already reads the head pane, and its metadata travels in the same message (§5.3). The GR00T camera gets a
port of its own, 5566, in the gear_sonic format that GR00T's camera clients read.

Topic subscription note: `gt.event` keeps its M1 name. ZMQ matches prefixes, so subscribe to `b"gt.event"`
(a subscription to `b"gt.events"` would receive nothing).

---

## 2. P1.1 Op discovery

`ping` (m1.md §1.6 fields) adds:

| Key | Value |
|---|---|
| `ops` | sorted list of every registered op name (the M1 ops, the ops below, `viz_level`/`viz_stats` when `--viz` is on) |
| `p1_contract` | `"m2b-1"` (this document) |
| `cameras` | `{name: on}` for every camera that exists, e.g. `{"head": true, "ego_view": false}` |
| `topics` | the 5601 topics P1 publishes: `["gt.pose", "gt.objects", "gt.event", "sim.health"]` |

`world/isaac_client.py:_probe_ops` reads `ops` [v]. An unknown op replies `ok: false, code: "unknown_op"` and
also lists `ops` (m1.md behaviour kept).

---

## 3. P1.2 Live object poses

### 3.1 REP `get_objects`

Args: `ids` (list of scene ids; default all), `dynamic_only` (bool, default false).

Reply: `{t_sim, t_wall, seq, pose_source: "sim", objects: [OBJ...]}` with

```
OBJ = { "id": str, "name": str,
        "pos": [x,y,z],                 # root prim origin (moves with the body)
        "quat_wxyz": [w,x,y,z],         # rigid body orientation (static objects: [1,0,0,0])
        "aabb": [[..],[..]],            # world AABB: the load-time visual box carried by the body's rigid motion
        "held_by": "left"|"right"|null, # attach (§4)
        "dynamic": bool,                # a loose prop PhysX moves; false = static / articulated furniture
        "source": "sim"|"static",       # "sim": read from PhysX this tick; "static": cannot move (load-time pose)
        "lin_vel": [vx,vy,vz],          # dynamic only, world m/s (0 while held)
        "moving": bool }                # |lin_vel| > 0.02 m/s
```

- Dynamic objects are the scene objects with a `body_path` (`scenes/loader.py:586-587` [v]: a RigidBodyAPI prim of
  a non-static object) that are not articulated. Their pose is read from one PhysX rigid-body tensor view each call
  (and each `gt.objects` tick). Static and articulated objects keep their `get_scene_info` pose.
- `aabb` after a rigid motion is the axis-aligned box of the 8 load-time corners moved by the body's motion since
  load, so a tipped object's box grows. `pos` moves with the same motion.
- `world/isaac_client.py:_refresh_objects` reads `id` and `aabb` [v]; `world/gt_world.py` computes `where` from the
  box.

### 3.2 PUB `gt.objects` on 5601 (10 Hz of sim time, `--objects-hz`)

Multipart `[b"gt.objects", msgpack]`: `{seq, t_sim, t_wall, objects: [OBJ...]}` with the **dynamic and held
objects only** (static ones never change; take them from `get_scene_info` or one `get_objects`). A pushed
object's new pose therefore reaches a subscriber within one tick (≤ 0.1 s of sim time plus transport).

---

## 4. P1.3 Attach / detach (STEPPING STONE)

Used by the `kinematic_attach` executor (`services/executors/kinematic_attach.py` [v]) to make "the object stays
in the hand" true in the sim until the Dex3 grasp is reliable. The arm does not move for it.

### 4.1 `attach {id, arm, mode, offset?, snap?, snap_m?}`

| Arg | Default | Meaning |
|---|---|---|
| `id` | required | scene id of a dynamic object |
| `arm` | required | `left` \| `right` |
| `mode` | `follow` | `follow`: every physics step (200 Hz) the object's pose is set to palm × grip offset, velocity zeroed. `fixed_joint`: a PhysX fixed joint between `<arm>_wrist_yaw_link` and the object, so its mass hangs on the arm (§4.4) |
| `offset` | `[0.07, 0.0, 0.0]` | grip point in the palm frame (m): 7 cm beyond the palm origin along the fingers |
| `snap` | true | if the object centre is more than `snap_m` from the grip point, move it there first |
| `snap_m` | 0.12 | |

Palm frame = `<arm>_wrist_yaw_link` × `(0.0415, ±0.003, 0)`, identity rotation: `<arm>_hand_palm_joint`
(`main.urdf:808-809` [v]; the URDF converter merged the palm link, `merge_fixed_joints=True`, m1.md §1.2).

While attached, the object's colliders are disabled (`physics:collisionEnabled` false on every collision prim under
the object), so it cannot push the hand, the robot or furniture. It keeps its orientation relative to the palm.

Reply: `{id, arm, mode, held_by, snapped: bool, dist_m (centre to grip point before the snap), grip_point
[x,y,z], stepping_stone: true}`. `gt.event {event: "attach", id, arm, mode}`.

Errors: `unknown_object`, `not_movable` (static or articulated), `hand_busy` (that arm holds another object),
`held_by_other` (the other arm holds this one), `bad_arg`. Attaching the same object to the same arm again is
allowed (re-snap).

### 4.2 `detach {id, pose?}`

| `pose` | Effect |
|---|---|
| absent / null | released where it is, velocity zero, colliders and gravity back: it falls or rests naturally |
| `[x, y, z]` | object **centre** (AABB centre) placed at (x, y, z), orientation = its load-time orientation (upright), velocity zero |
| `[x, y, z, yaw]` | as above, rotated about world z by `yaw` (rad) from the load-time orientation |

Reply: `{id, was_held: bool, placed: bool, pos, aabb, held_by: null}`. Detaching an object that is not held is
not an error (`was_held: false`); a given pose is still applied. `gt.event {event: "detach", id, placed}`.
The object must not be placed inside the hand (its colliders come back): place it on the target surface
(`WorldModel.free_spot`).

### 4.3 `release_all`

Detaches every held object in place (`reset_scene` does this first). Reply `{released: [ids]}`.

### 4.4 Modes: what each costs and risks

- `follow` is the default and the tested path (§13): no structural stage change (only collider flags and pose
  writes), so no PhysX tensor view is invalidated.
- `fixed_joint` creates one `UsdPhysics.FixedJoint` prim per attach under `/World/wl_grip/` and deletes it on
  detach. Structural stage changes are exactly what can invalidate tensor views, which is why the 50-cycle test
  runs both modes (§13). If a tensor-view error ever appears, P1 disables `fixed_joint` for the rest of the run
  (`code: "mode_disabled"`) and keeps `follow`.

---

## 5. P1.4 / OD1 / P1.10 Cameras

### 5.1 The cameras

Two robot cameras in P1 (OD1, decided by the lead), plus the M1 camera kept as an option. All render 640x480 RGB,
RTX preset `balanced` (m1.md §1.10), on the sim-time schedule of their rate.

| | `head` (System 1, scans, UI head pane, M1 tools) | `ego_view` (GR00T) | `d435` (M1's camera, optional) |
|---|---|---|---|
| Port / key | 5565 / `head` | 5566 / `ego_view` | 5565 / `d435` only with `--stream-camera d435` |
| Parent link | `torso_link` | `torso_link` (Arena: `head_link`, merged into `torso_link` here) | `torso_link` |
| Mount position (torso frame) | (0.06, 0.0, 0.526) m | (0.0488135, 0.0, 0.30925) m = head_link (0.0039635, 0, −0.044) + (0.04485, 0, 0.35325) | (0.0576235, 0.01753, 0.41987) |
| Orientation | pitch 15° down, no yaw/roll | pitch **35.00°** down, no yaw/roll | pitch 47.6° (0.8307767 rad) |
| Height, standing (pelvis 0.787 m) | ≈ 1.36 m | ≈ 1.14 m | ≈ 1.25 m |
| Intrinsics | HFOV 90.0°, VFOV 73.74° (f 10 mm, apertures 20 x 15 mm) | **f 15 mm, h-aperture 20.955 mm, v-aperture 15.71625 mm: HFOV 69.87°, VFOV 55.30°**, fx = fy = 458.12 px | VFOV 45°, HFOV 57.8° |
| Clipping | 0.05-50 m | **0.1-5 m** | 0.05-50 m |
| Default state / rate | on, 30 Hz (`--camera-hz`) | **off**; 30 Hz while enabled (`--ego-hz`) | not created unless `--cameras` lists it |
| Source | `docs/M2.md` P1.4; `world/perception.py:78-79` `HEAD_SIM` (the GT model world uses; identical numbers) | see below | m1.md §1.4 |

**`ego_view` = Arena's G1 head camera, exactly.** Numbers read in IsaacLab-Arena `release/0.2.1` @ `8b4a3a47`
on the dev box (`/work/arena/IsaacLab-Arena`) [v]:

- `isaaclab_arena/embodiments/g1/g1.py:103-106`: `_DEFAULT_G1_CAMERA_OFFSET = Pose(position_xyz=(0.04485, 0.0,
  0.35325), rotation_xyzw=(-0.62721, 0.62721, -0.32651, 0.32651))`.
- `g1.py:508-539` `G1CameraCfg`: `prim_path=".../Robot/head_link/RobotHeadCam"`, `height=480, width=640`,
  `data_types=["rgb"]`, `PinholeCameraCfg(focal_length=15, clipping_range=(0.1, 5))`, offset `convention="ros"`.
- Isaac Lab (Arena's submodule `e57379c63`; same in our v2.3.2 `37ddf62` at `camera.py:134-135`):
  `sensors_cfg.py:61` `horizontal_aperture = 20.955`; `sensors/camera/camera.py:172-173` sets
  `vertical_aperture = horizontal_aperture * height / width` = 15.71625.
- ROS-convention quaternion → optical axis (0.81916, 0, −0.57357) in `head_link`: **35.00° down, yaw 0, roll 0**
  (computed; `sim_isaac/tests/test_cameras.py` checks it).
- `head_link` pose: Arena spawns `nvidia` nucleus `Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd`
  (`g1.py:237`); in that USD (read with usd-core; sha1 `5a07df13…`, identical under Isaac/5.1 and 6.0)
  `head_joint` has `localPos0 = (0.0039635, 0, -0.044)` on `torso_link`, and head_link sits at the pelvis origin
  at zero waist [v]. Our robot is SONIC's `main.urdf`, whose `head_joint` is the same (`main.urdf:587-588`) and
  whose torso sits 0.044 m above the pelvis (`:489-490`) [v]. So the mount above is exact for our model.
  (`docs/groot_arms_design.md` §2.3 used `g1_29dof_with_hand.urdf`'s −0.054, which belongs to a torso frame 1 cm
  higher; its "torso + (0.049, 0, 0.299)" is superseded by (0.0488, 0, 0.3093) here: the same physical point.)
- Not matched: Arena renders through a `TiledCamera` in its own Isaac Sim 6.0 container; P1 uses one Replicator
  render product per camera under Isaac Sim 5.1 with the `balanced` preset. Geometry and intrinsics are identical;
  noise/denoiser look may differ.

### 5.2 Frame messages on 5565 (head) and 5566 (ego_view)

One ZMQ frame = `msgpack.packb(d, use_bin_type=True)`, byte-compatible with gear_sonic `sensor_server`
(m1.md §1.4), plus extra keys that gear_sonic consumers ignore:

```
d = {
  "timestamps": {KEY: t_capture},           # gear_sonic field. M2b: the CAPTURE wall time (M1: encode time)
  "images": {KEY: <b64 JPEG>}, KEY: <b64 JPEG>,   # JPEG of the RGB array via cv2.imencode (q80): cv2.imdecode
                                            # returns RGB, standard decoders show R/B swapped (as in M1)
  "camera": name,                           # "head" | "ego_view" | "d435"
  "seq": int,                               # per-camera frame counter
  "render_seq": int,                        # id of the sim.render() that produced it (same value = same render)
  "t_sim": float, "t_capture": float,       # wall time.time() right after the read-back of this render
  "t_capture_mono": float,                  # time.monotonic() at the same instant (same-box freshness checks)
  "t_pub": float,                           # wall time when sent (after the JPEG encode on the worker thread)
  "w": 640, "h": 480, "hfov": float, "vfov": float,       # degrees, from the camera's intrinsics
  "cam_pos": [x,y,z], "cam_quat_wxyz": [w,x,y,z],         # world pose of the camera at this render
                                            # (camera axes x forward, y left, z up: the "world" convention)
  "cam_pose_wl": [map_x, map_z, yaw_deg, pitch_down_deg], # Worldline frame (world/coords.py), P1.10
  "stationary": bool,                       # GT base planar speed < 0.05 m/s and |yaw rate| < 0.05 rad/s
  "base_speed": float, "base_wz": float,
  "jpeg_q": 80 }
```

`KEY` is the camera name. The camera pose is computed from the torso link pose read in the physics step that
precedes the render, i.e. the pose the render used. `cam_pose_wl` uses `IsaacFrames`' formula (`world/frames.py:59-64`
[v]): `(x, y, yaw_map_deg(yaw), degrees(pitch_down))` rounded to 3/3/1/1 decimals.

**GR00T reads ego_view from 5566** (SUB, `CONFLATE` recommended), key `ego_view`, decoded with `cv2.imdecode`
(RGB), 480x640x3 uint8. Freshness: `t_capture_mono` against the client's `time.monotonic()` (P5 runs on the same
box as P1); `timestamps.ego_view` is the same instant in wall time. A frame older than 100 ms is stale
(`groot_arms_design.md` §5.2).

### 5.3 Frame metadata (P1.10)

`cam_pose_wl`, `stationary`, `hfov`, `vfov` are in every head frame (and every ego_view frame), from the same render.
`IsaacFrames` can stop stamping poses at receive time: it needs them passed through FrameTap (request in §12).

### 5.4 Enabling cameras and setting rates

| Op | Args | Reply |
|---|---|---|
| `get_cameras` | - | `{cameras: [{name, on, hz, port, key, width, height, hfov_deg, vfov_deg, focal_length_mm, horizontal_aperture_mm, vertical_aperture_mm, clipping, parent, mount_xyz, mount_quat_wxyz, pitch_down_deg, consumers, frames, source}], stream_camera, render_hz}` |
| `camera` | `name`, `on` (bool, optional), `hz` (optional), `consumer` (str, default `"anon"`), `ttl_s` (optional, s) | `{name, on, hz, consumers, port, warmup_frames}` |
| `set_render_rates` | `head_hz`, `ego_hz` (0 = off) | `{head: {...}, ego_view: {...}}` (`SimControl.set_render_rates`, `api/services.py:285` [v]) |

- A camera renders while it has at least one consumer. `camera {on: true, consumer: c}` adds `c`,
  `camera {on: false, consumer: c}` removes it; `on: false` without a consumer clears them all. `head` starts with
  the consumer `"default"`; `ego_view` starts with none.
- **`ttl_s`**: the consumer is dropped if it does not repeat `camera {on: true}` within `ttl_s`. The GR00T client
  should enable with `consumer: "<execution_id>", ttl_s: 10` and repeat every few seconds, so a crashed client
  cannot leave the camera rendering (and costing RTF) forever.
- Enabling re-arms the render product (annotator detach → enable → attach, `docs/viz.md` §4 finding 3), about
  15-20 ms of sim-thread work once, and the first `warmup_frames` (4) frames after enabling are rendered but not
  published (the `balanced` denoiser needs a few frames, docs/viz.md finding 6). The first frame arrives about
  0.13 s after `camera {on: true}` at 30 Hz.
- Disabling stops the render product's GPU work (`hydra_texture.set_updates_enabled(False)`).
- Rates are in sim time and aligned on one grid, so cameras at 30 and 15 Hz share renders. **Every enabled render
  product is rendered on every render call** (docs/viz.md finding 2): what costs RTF is the number of enabled
  products times the render rate, not a camera's own rate. Lowering `head` while `ego_view` runs at 30 Hz only
  saves read-back and encode; lowering both lowers the render rate.
- `gt.event {event: "camera", name, on, hz, consumers}` on every change.
- Measured RTF with none / head / head + ego_view, standing and walking: §13. The rate policy for GR00T sessions
  that follows from it: §13.

---

## 6. P1.5 Link poses

`gt.pose` (50 Hz) adds:

```
"links": { "torso_link": {"pos": [x,y,z], "quat_wxyz": [...]},
           "left_palm":  {"pos": [...],   "quat_wxyz": [...]},
           "right_palm": {"pos": [...],   "quat_wxyz": [...]} },
"waist_q": [yaw, roll, pitch]            # measured waist joints, rad (MuJoCo order 12-14)
```

- `torso_link` is the articulation body (its pitch/roll are real: SONIC moves the waist).
- `left_palm` / `right_palm` are the palm frames of §4.1 (wrist_yaw_link × (0.0415, ±0.003, 0)), the same points as
  `body/g1_kin.py` "palm".

REP `get_link_poses {links?: [names]}` → `{t_sim, links: {name: {pos, quat_wxyz}}, available: [...]}` for any
articulation body name, `left_palm`, `right_palm`, `cam:head`, `cam:ego_view` (default: the three of `gt.pose`).
Unknown names are listed in `unknown` (not an error).

---

## 7. P1.6 Instance-id visibility: `detections`

Args: `camera` (`head` default, or `ego_view`), `min_px` (40), `max_range` (m, optional: drop objects whose
centre is farther from the camera), `ids` (optional filter), `bbox` (true).

The op renders once more (all enabled cameras; the frames from that render are published as usual and share its
`render_seq`), reads Replicator's `instance_id_segmentation_fast` annotator on the camera's render product, maps
each instance prim to a scene object (longest prefix of the object's `prim_path`/extra prims), and counts pixels.

Reply:

```
{ "camera": str, "t_sim": float, "render_seq": int, "frame_seq": int, "w": 640, "h": 480,
  "method": "instance_id_segmentation_fast", "min_px": int,
  "detections": [ {"id": scene_id, "name": str, "px": int, "bbox": [u0, v0, u1, v1],   # pixels, inclusive
                   "dist_m": float,                  # camera to AABB centre
                   "held_by": "left"|"right"|null} ... ],          # sorted by px, >= min_px only
  "other_px": {"robot": int, "structure": int, "background": int},
  "cam_pose_wl": [...], "ms": float }
```

- The camera must be on (`code: "camera_off"` otherwise; enable it first). The segmentation annotator stays attached
  for `--seg-keep-s` (3 s) after the last call so a scan's consecutive views do not re-attach it, then it is
  detached (while attached it adds GPU work to every render).
- This is instance segmentation of the render, labelled `instance_id_segmentation_fast`. world's
  `gt-geometric` (`world/perception.py`) stays the fallback. Agreement measured: §13.

---

## 8. P1.7 Scene reset and fixtures

| Op | Args | Reply | Notes |
|---|---|---|---|
| `reset_scene` | `variant` (`"default"`), `poses` (optional `{id: [x,y,z] \| [x,y,z,yaw]}`, object centres), `robot` (optional: `true` = spawn, or `{x, y, yaw}`), `band` (true, only with `robot`) | `{variant, objects_reset, poses_applied, robot_reset, released, ms, object_writes, root_writes}` | Releases every held object, puts every dynamic object back at its load pose (orientation and position) with zero velocity, applies `poses`, optionally resets the robot exactly like `reset_robot` (band on), puts the house props back to sleep. `gt.event reset_scene`. Only `default` exists as a named variant in wave 1; eval fixtures pass their placements in `poses` |
| `move_object` (test-only) | `id`, `pose` (`[x,y,z]` \| `[x,y,z,yaw]`, the centre as in §4.2), `vel` (optional `[vx,vy,vz]`) | `{id, pos, aabb}` | teleport; releases it if held. `gt.event object_moved {id, by: "move_object"}` |
| `set_object_pose` | same as `move_object` | same | alias (the name `groot_arms_design.md` §3.6 asks for) |
| `push_object` (test-only) | `id`, `vel` `[vx,vy,vz]` (m/s) | `{id, vel}` | sets the linear velocity once: PhysX then slides/tips it. `gt.event object_moved {by: "push_object"}` |

Every test-only write increments `get_stats.object_writes`. `reset_scene` with `robot` also increments
`root_writes` like `reset_robot` (m1.md §1.6). The robot reset leaves SONIC's planner facing to the body (the body
must re-stand, `docs/groot_arms_design.md` §3.2 [u]).

---

## 9. P1.8 Furniture-only top render

`render_topdown {mode: "full" | "furniture", path?, fresh?, force?}` (m1.md §1.6 plus `mode`).
`furniture` hides every dynamic object (the props the page draws itself from truth) and the robot for the render,
so the overhead image never shows stale props. Both modes are rendered once at start-up while the band holds the
robot and then cached (`fresh: true` re-renders; blocks about 0.3 s; needs the band on or `force`). Reply adds
`mode` and `hidden` (number of prims hidden).

---

## 10. P1.9 Health and events

### 10.1 PUB `sim.health` on 5601 (1 Hz of wall time) and REP `get_health`

```
{ "seq", "t_sim", "t_wall",
  "rtf_1s", "rtf_3s", "rtf_5s", "rtf_10s",          # sim s / wall s over the trailing window
  "level": "ok" | "degraded" | "unsafe",            # PLAN §3.5: unsafe = rtf_3s < 0.85; degraded = rtf_5s < 0.95
  "physics_hz_1s", "render_hz", "step_ms_p99", "overruns", "lost_s", "hitches_gt25ms", "heartbeat_pubs",
  "band": bool, "fallen": bool, "held": {"left": id|null, "right": id|null},
  "cameras": {name: {"on": bool, "hz": float, "pub_hz": float}} }
```

The level is P1's reading of PLAN §3.5; `world` owns what it does with it (R.6: `DEGRADED` rejects `manipulate`,
`UNSAFE` rejects every body tool). `get_stats` is unchanged apart from added keys (`cameras`, `object_writes`,
`attach_count`, `detach_count`, `render_calls`, `seg_attached`).

### 10.2 `gt.event` (sporadic, topic unchanged from M1)

| `event` | Fields (besides `t_sim`, `t_wall`) | When |
|---|---|---|
| `band`, `reset_robot`, `fallen`, `recovered` | as M1 | unchanged |
| **`robot_fell`** | `pelvis_z`, `tilt_deg`, `base_pos` | the same edge as M1's `fallen` (pelvis_z < 0.45 m or tilt > 60°); prefer this name in M2b code |
| **`robot_recovered`** | `pelvis_z` | the edge back |
| **`object_fell`** | `id`, `from_z`, `to_z`, `drop_m`, `pos`, `on_floor` | a dynamic, non-held object came to rest (speed < 0.05 m/s for 0.3 s) with its AABB bottom ≥ 0.10 m lower than where it last rested; `on_floor` = bottom within 0.05 m of the floor |
| `attach`, `detach` | `id`, `arm`, `mode` / `placed` | §4 |
| `object_moved` | `id`, `by` | test-only writes (§8) |
| `reset_scene` | `variant`, `objects_reset`, `robot_reset` | §8 |
| `camera` | `name`, `on`, `hz`, `consumers` | §5.4 |

---

## 11. Error codes

`unknown_op`, `bad_arg`, `unknown_object`, `not_movable`, `hand_busy`, `held_by_other`, `mode_disabled`,
`unknown_camera`, `camera_off`, `segmentation_unavailable`, `unknown_variant`, `busy:controller_active` (m1.md: a
heavy op while SONIC holds the robot), `internal` (an exception; the text says which).

---

## 12. Fakes and consumers

- `tools/fake_p1.py` (isaac owner) serves this whole wire kinematically: live object poses and `gt.objects`,
  attach/detach in both modes, link poses, the two camera streams with metadata (synthetic images), `camera` /
  `set_render_rates` / `get_cameras`, `detections` (a frustum count standing in for segmentation, labelled
  `method: "fake-frustum"`), `reset_scene`, `move_object`, `push_object`, `render_topdown {mode}`, `sim.health`,
  `robot_fell` / `object_fell`. Tests: `sim_isaac/tests/test_fake_p1_m2b.py`.
- `tests/fakes/fake_p1_world.py` (world owner) implements P1.1-P1.3 of this wire. Differences to close there are in
  the isaac owner's wave-1 requests.
- `viz.tap.FrameTap` passes only `t_wall, t_sim, seq` of 5565 frames through `meta("head")`; to use §5.3, FrameTap's
  `decode_head` must keep the extra keys (request in the wave-1 result), or `IsaacFrames` subscribes 5565 itself.

---

## 13. Measured on the dev box (wave 1)

Filled in from the live runs; every number names its run. Until then: **[u]**.
