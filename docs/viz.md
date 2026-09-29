# Viz: see the sim in a browser, and record it

Owner: the viz component (`viz/`). Status: built and tested on the Brev box against a stand-in P1
(`viz/test_stack.py`: the real procthor-train-40 house from `scenes/loader.py`, the M1 G1 USD, a **kinematically
moved** robot, clearly labelled VIZ TEST). Not yet wired into `sim_isaac/app.py`; the hook (7 lines at 5 places, one
of them a changed line) is in section 9.1, and the stand-in exercises exactly that code path with `--hook`.
It already works with today's P1 as is: head camera, gt.pose, `render_topdown`, and P1's own `--tp-camera` chase
stream (`frame.tp`) are all picked up.

What it gives you:

1. **Live view in the browser** (`viz/server.py`, 127.0.0.1:8765 on the box, reached through the SSH tunnel). It
   shows the head camera, a third-person chase camera, and a top-down map with the robot pose, trajectory,
   occupancy and room outlines. A telemetry strip, drive controls (stand, stop, WASD/QE walk, click-to-go_to) and
   a record button are on the same page.
2. **Recordings to check that the robot really walks** (`viz/recorder.py`). One directory per run with
   `head.mp4`, `chase.mp4`, `top.mp4`, a 2x2 `composite.mp4` with overlays, `telemetry.jsonl`, a 12-frame
   `contact_sheet.png`, `last.jpg` and `summary.json`. `viz/pull_recordings.sh` brings them to the laptop.
3. **Extra sim cameras** (`viz/isaac_cams.py: VizCams`), plugged into P1 with `attach_p1` (section 9.1). They
   render a chase camera, a top-down camera (ceiling cut away) and an optional overview camera, and publish JPEG
   frames on PUB 5602.

## 1. Quick start

On the box (`00_infra/ssh.sh`), once per box:

```bash
cd /work/worldline-g1 && bash viz/box.sh setup        # viz/.venv via uv: pyzmq msgpack numpy pillow imageio[ffmpeg] aiohttp
```

Against the real M1 stack (P1 on 5565/5600/5601 and 5602 via VizCams or `--tp-camera`, body on 5610/5611):

```bash
bash viz/box.sh server 0                               # tmux viz-server -> 127.0.0.1:8765
bash viz/box.sh record 30 0 walk-test                  # 30 s recording, foreground; prints the run dir
viz/.venv/bin/python viz/recorder.py --until-event go_to:succeeded,failed,fallen --label goto   # stop 2 s after
viz/.venv/bin/python viz/probe.py --seconds 5 --save /tmp/probe                                 # rates + 1 frame/stream
```

On the laptop:

```bash
../ludo_robotics_prep_g1/00_infra/tunnel.sh 8765       # then open http://localhost:8765 (Chrome or Safari)
viz/pull_recordings.sh latest                          # -> outputs/recordings/<run>/ (look at contact_sheet.png first)
uv run --with aiohttp python viz/ws_probe.py --url ws://localhost:8765/ws --seconds 10   # fps + latency via tunnel
```

`tunnel.sh` returns immediately when the SSH ControlMaster is already up. The forward lives in the master
(`ControlPersist 10m`, `00_infra/lib.sh`): check it with `lsof -iTCP:8765 -sTCP:LISTEN`.

Test stack (no M1 needed), which is how everything below was verified:

```bash
bash viz/box.sh sim --port-offset 300 --house procthor-train-40 --level low --rt-pace --duration 600   # stand-in P1
bash viz/box.sh server 300                                                        # 127.0.0.1:9065
bash viz/box.sh record 30 300 house40
bash viz/box.sh sim --port-offset 300 --house empty --selftest                    # top-view mapping + latency check
bash viz/box.sh sim --port-offset 300 --house procthor-train-40 --settle-sweep    # top snapshot quality vs settle
bash viz/box.sh sim --port-offset 300 --house procthor-train-40 --bench \
     --bench-phases off/auto,min/auto,low/auto,high/auto,off/auto,min/auto,low/auto,high/auto,off/auto,min/auto,low/auto,high/auto \
     --bench-secs 10                                                              # RTF cost per level
bash viz/box.sh sim --port-offset 300 --house procthor-train-40 --level low --rt-pace --hook   # VizCams via attach_p1, as P1
bash viz/box.sh stop                                                              # kills viz-sim/viz-server/viz-rec only
viz/.venv/bin/python viz/tests/test_frames.py                                     # 17 tests, no Isaac (~17 s)
viz/.venv/bin/python -m pytest -q viz/tests                                       # the same with pytest (box.sh setup installs it)
```

The M1 contract gives build-phase tests the +100 ports, and the isaac agent's own P1 test runs hold them. My tests
therefore use **+300**: 5865, 5900-5902, 5910/5911, and HTTP 9065. `--port-offset` (or `WL_PORT_OFFSET`) shifts
every port of every viz tool together.

## 2. Architecture

```
 Isaac process (P1, /work/envs/isaaclab)                         box, viz/.venv                     laptop
 ┌───────────────────────────────────────────┐
 │ physics 200 Hz ── ego cam (P1) ──────────── PUB 5565 gear_sonic msgpack ─┐
 │ gt.pose 50 Hz ───────────────────────────── PUB 5601 [gt.pose, msgpack] ─┤   ┌──────────────────────────┐
 │ REP 5600 (scene, occupancy, render_topdown, stats, viz_*) ── REQ ───────┼──►│ viz/server.py  :8765     │  SSH -L 8765
 │ VizCams.step(): chase / top / overview    │                               │   │  aiohttp, one WebSocket  │═════════════► browser
 │   rgb annotator -> worker thread: JPEG ───── PUB 5602 [frame.<cam>, msgpack]┤  │  + MJPEG / JPEG / REST   │
 │   (or P1 --tp-camera: frame.tp)           │                               │   └────────┬─────────────────┘
 └───────────────────────────────────────────┘                               │            │ spawns on "Record"
 body service (P3) ROUTER 5610 ◄── DEALER (drive controls) ──────────────────┼────────────┤
                   PUB 5611 body.event / body.state ─────────────────────────┤   ┌────────▼─────────────────┐
                                                                             └──►│ viz/recorder.py          │ outputs/recordings/<ts>/
                                                                                 │  SUB all, 10 fps wall    │──► pull_recordings.sh
                                                                                 └──────────────────────────┘
```

**Decoupling, and what still runs on the sim thread.** The browser, the recorder and the server never touch the
Isaac process: everything goes through ZMQ PUB/SUB, the hand-off inside VizCams is a bounded queue with `put_nowait`,
and the PUB send is NOBLOCK, so a slow browser or a crashed recorder cannot stall physics (frames are dropped
instead). JPEG encoding, the send and the snapshot re-send run on a worker thread. VizCams does add synchronous work
to the thread that steps physics (P1 also serves REP ops there), measured in section 7.2: the annotator read and
buffer copy per chase frame, the chase placement raycasts, the top snapshot re-arm, and a one-off stall when the
level is switched. The largest cost is not VizCams code at all: every live render product is rendered by every P1
render (~3 ms per render at 640x360).

## 3. Ports (all 127.0.0.1; +offset for tests)

| Port | Direction (viz) | Format |
|---|---|---|
| 5565 | SUB head camera | gear_sonic `sensor_server`: one frame, msgpack `{"timestamps":{"ego_view":t}, "images":{"ego_view":<b64 JPEG>}, "ego_view":<b64>, "t_sim"?, "seq"?}`. The JPEG is `cv2.imencode` of an **RGB** array (contract 1.4), so standard decoders show R/B swapped; server, recorder and FrameTap swap them back (`--no-head-swap` disables this) |
| 5600 | REQ P1 ops | JSON `{"op", ...}` -> JSON. Used: `get_scene_info`, `get_occupancy` (npz path on the box), `render_topdown` (P1's cached render; map and recorder background), `get_stats`, `viz_level`, `viz_stats` |
| 5601 | SUB gt.pose | multipart `[b"gt.pose", msgpack]`, contract 1.5 fields |
| 5602 | PUB (VizCams) / SUB | multipart `[b"frame.<cam>", msgpack{topic, seq, t_sim, t_request, t_wall, w, h, jpeg, cam_pose{pos, quat_wxyz, projection, hfov_deg, vfov_deg, extent, target}, robot{pos, yaw}, level, gen, snapshot, settle, rearm_ms, render_ms, forced, v:1}]`, standard-colour JPEG. The last snapshot of each snapshot camera (top) is **re-sent every 2 s** unchanged (same `seq` and `t_wall`) so late subscribers get a top view at once; consumers drop a message whose `(seq, t_wall, source)` equals the frame they hold (`viz/common.py: same_frame`). `extent` = `[xmin,ymin,xmax,ymax]` of an axis-aligned top-down image (image +x = world +x, image up = world +y, the convention of P1 `render_topdown`). **P1's own `--tp-camera`** publishes `[b"frame.tp", msgpack{seq, t_sim, t_wall, jpeg, base_pos, yaw}]` (contract 1.8) with a cv2-encoded RGB JPEG: `viz/common.py: frame_from_msg` shows it as the **chase** pane and swaps R/B back. Any other `frame.<name>` becomes an extra pane |
| 5610 | DEALER -> body ROUTER | JSON `{"id","op","args"}` (contract 3.4). UI ops: `stand`, `stop`, `go_to{x,y}`, `turn_to{yaw}`, `status`, `clear_fault`, and for held keys `velocity{vx,vy,wz,stream,t_wall,watchdog_s,end?}` (body/velocity.py) or, on bodies without it, `walk{vx,vy,yaw_rate,duration_s}` (section 5). `shutdown_control` is deliberately not reachable from the UI |
| 5611 | SUB body | `[b"body.event", JSON{id,op,state,data}]`, `[b"body.state", JSON]` 5 Hz |
| 8765 | HTTP/WS (server) | section 5 |
| free port | recorder control REP | the server picks one per recording; the CLI takes `--control tcp://127.0.0.1:5620` |

VizCams and P1's `--tp-camera` both bind 5602: run one of them. With the hook, `viz_enabled()` refuses
`--viz` together with `--tp-camera` at argument parsing, before Kit starts.

Every viz tool shifts all ports with `--port-offset` (or `WL_PORT_OFFSET`), and a recorder started from the page or
`POST /api/record` gets **every** port of its server explicitly (`--head --frames --gt --gt-rep --body-ctl
--body-evt`), so a test server at +300 never queries the production P1 REP on 5600.

## 4. VizCams (`viz/isaac_cams.py`)

```python
from viz.isaac_cams import VizCams
cams = VizCams(sim, "/World/G1", scene.bounds, pub="tcp://127.0.0.1:5602", hz=10, level="low", floor_z=floor_z)
cams.warmup(10, base_pos, base_quat_wxyz)        # optional: first renders + first top snapshot at start-up
cams.step(t_sim, base_pos, base_quat_wxyz)       # every physics step, after the host's render (if any)
cams.set_level("off" | "min" | "low" | "high")  # runtime switch (sim thread; stalls the loop once, 7.2)
cams.handle_op({"op": "viz_level", "level": ...}) / {"op": "viz_stats"}   # what the REP ops call
```

In P1 all of this is done by `attach_p1(app)` (section 9.1).

| Level | Cameras |
|---|---|
| off | none, zero cost (no render products exist) |
| min | chase 480x270 at `hz`. The top view then comes from P1 `render_topdown` + live gt.pose overlay |
| low | chase 640x360 at `hz`, top 512 px long side, snapshot every 10 s |
| high | chase 960x540 at `hz`, overview 960x540 at 5 Hz, top 1024 px, snapshot every 3 s |

- **chase**: perspective, 70 deg HFOV. It sits 2.5 m behind and 1.6 m above the pelvis, looks at the pelvis, and
  is clamped under `floor_z + ceiling_cut + 0.2` (2.2 m) so it stays inside a house. Target and yaw are smoothed
  (tau 0.15 s / 0.6 s). Occlusion handling: a "fat ray" (three parallel PhysX `raycast_all` rays 0.22 m apart,
  the robot's own colliders ignored) from the chest to the camera. When it is blocked, and every 0.5 s anyway,
  VizCams scans the azimuths behind, +-35, +-70 and +-110 deg and moves (smoothed, with hysteresis, behind
  preferred) to the freest one, then pulls in to the free distance while keeping the height, so a pulled-in camera
  looks down over the robot. **Needs PhysX scene queries**: Isaac Lab's `SimulationCfg.enable_scene_query_support`
  defaults to False and then `raycast_all` silently reports nothing; P1 sets it True (`sim_isaac/app.py:119`), and
  VizCams checks `physxScene:enableSceneQuerySupport` at start and prints a warning if it is off. Stats:
  `viz_stats` -> `chase_place_ms_*`, `chase_blocked_frac`, `rays`, `ray_hits`; sim-thread costs
  `readback_ms_*` (annotator `get_data`), `copy_ms_*` (buffer copy), `capture_ms_*` (both, plus re-arm for
  snapshots), `set_level_ms` (last switches); `resent` (snapshot re-sends), `dropped`.
- **top**: orthographic, straight down, sized to the house bounds plus 0.5 m. The near clipping plane is at
  `floor_z + ceiling_cut` (2.0 m), so ceilings and roofs vanish and walls show as cross-sections.
- **overview** (high only): perspective, straight down over the house centre, with the same ceiling cut. Pass
  `overview={"eye": [...], "target": [...]}` for a custom fixed view.
- Poses come from the arguments of `step()` (P1 has `st["base_pos"]`, `st["base_quat"]` wxyz). The USD fallback
  for a missing pose is wrong when physics runs through Fabric, so always pass the pose.

### Isaac 5.1 APIs used (cited in the module docstring)

- One Replicator render product per camera: `omni.replicator.core.create.render_product(cam, (w, h), name=...)`
  (omni.replicator.core-1.12.27 `scripts/create.py:1486`) with the `rgb` annotator
  (`rep.AnnotatorRegistry.get_annotator("rgb")`, `.attach([rp.path])`, `.get_data()` -> uint8 HxWx4): the pattern
  of Isaac Lab's Camera sensor (`isaaclab/sensors/camera/camera.py:439-488, 514`) and of P1's `RgbCapture`.
- Rendering is an app update: Isaac Lab `SimulationContext.render()` (`isaaclab/sim/simulation_context.py:585-627`).
  With Isaac Lab's kit settings (`app.asyncRendering=false`, `omni.replicator.asyncRendering=false`,
  `apps/isaaclab.python.headless.rendering.kit:79`), data read right after a render belongs to that render.
- `rp.hydra_texture.set_updates_enabled(False)` (HydraTexture, `scripts/utils/viewport_manager.py:48-81`) and
  NVIDIA's custom-FPS snippet (`isaacsim.replicator.examples/.../test_sdg_useful_snippets_timeline_based.py:84-129`).
- `omni.physx.get_physx_scene_query_interface().raycast_all` for the chase wall check.

### Render-product findings (measured on the box, Isaac Sim 5.1.0 / Replicator 1.12.27 / Isaac Lab 2.3.2)

The experiments are in `outputs/viz_test/rp_probe*.py` on the box (moving magenta marker, ortho camera, `sim.render()`):

1. An **enabled** render product has fresh data right after `sim.render()`: a marker moved before each render is
   seen at the new position, with zero frames of lag.
2. Every enabled render product is rendered by **every** app update, including P1's own ego-camera renders. The
   cost barely depends on content: a blank view (clipping pushed away) saves only about 0.7 ms.
3. `set_updates_enabled(False)` does remove the cost. But a render product that was **ever** disabled never
   delivers annotator data again from plain `sim.render()` (6+ renders tried). NVIDIA's snippet re-arms it with
   `rep.orchestrator.step_async`, and `rep.orchestrator.step()` **hangs** inside Isaac Lab's loop. What works:
   re-attaching the annotator (detach, enable, attach), fresh on the first render, 16-20 ms at 512-768 px.
4. `isaacsim.sensors.camera.Camera(frequency=...)` / `.pause()` only throttle acquisition
   (`isaacsim.sensors.camera/camera.py:439-481`); rendering continues.
5. **RTX preset.** Isaac Lab applies **no** preset when cameras are on and `--rendering_mode` is not given
   (`app_launcher.py:896-910`), and those frames are as speckled as `performance` (denoisers and DL denoiser off,
   `apps/rendering_modes/performance.kit`): black and white salt-and-pepper on every textured surface. P1 sets
   `balanced` (`sim_isaac/app.py`, its render A/B: Laplacian std 310 vs 2.2), and so does `test_stack.py` now.
   VizCams inherits whatever the host uses.
6. **Snapshot settle.** Under `balanced`, the first render after re-arming still carries the DLSS/denoiser history
   of the previous snapshot: thin dark features (door frames, wall ends) are faint or missing. Settle sweep in
   procthor-train-40 (512x512 top, robot still, reference = 40 renders; `--settle-sweep`, 3 repeats):

   | settle (renders before the read) | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
   |---|---|---|---|---|---|---|---|
   | mean abs diff to reference (0-255) | 2.08 | 2.95 | 1.68 | **1.41** | 1.16 | 1.06 | 0.89 |

   Default `settle=4`: the door frames are back and it costs 4 renders of one 512 px product every 10 s.

Hence two camera kinds:

- **live** (chase, overview): always enabled. A frame is read at the viz rate right after a host render
  (piggyback); VizCams renders itself only if the host has not rendered within `min(1.5/hz, 0.25 s)`. The cost
  scales with **the host's render rate** (P1 `--camera-hz`, 30), not with the viz rate.
- **snapshot** (top): disabled between captures, re-armed per capture, read after `settle` renders (host renders
  when available), then disabled. The live robot position is drawn from gt.pose over the last snapshot by the UI
  and the recorder.

## 5. Web server and page (`viz/server.py`, `viz/static/index.html`)

- **One WebSocket** `/ws` carries all panes as binary messages
  `[uint32 BE header length][JSON header {s, seq, t_sim, t_wall, w, h, extent?, robot?, snapshot?, src?}][JPEG]`,
  plus JSON text (hello, telemetry at 10 Hz, command replies, body events, recorder status). One socket avoids
  the browsers' 6-connections-per-host limit that several MJPEG `<img>` streams would hit.
- **Flow control**: the page acks each displayed frame. The server keeps at most `--ws-window` (4) unacked frames
  per stream per client and always sends the newest, so a slow link drops frames instead of piling up latency.
  The window must cover fps x RTT; the Brev tunnel RTT is about 170 ms.
- Frames pass through untouched except the head camera and P1's `frame.tp`: their R/B swap is re-encoded off the
  event loop (head capped by `--head-fps` 15).
- Other endpoints: `GET /stream/<name>.mjpg` (MJPEG, `?fps=N`), `GET /frame/<name>.jpg`, `GET /api/state`,
  `POST /api/cmd {"op","args"}`, `POST /api/record {"action":"start"|"stop"}`, `GET /api/recordings`,
  `/recordings/<run>/<file>` (contact sheet and videos play in the browser), `/occupancy.png`, `/topdown.png`.
- The page shows the chase pane (large), the head pane, and a top-down canvas. The canvas draws the VizCams top
  frame, else P1 `render_topdown`, else only the scene bounds. On top go the occupancy (from `get_occupancy`'s npz,
  read on the box), room outlines (`get_scene_info`), the trajectory (gt.pose), the robot arrow and the go_to
  target.
- Click on the map for `go_to{x,y}`; shift-click for `turn_to` toward the point.
- **Held keys W/S/A/D/Q/E** drive the robot with **one body op per key press**. The page sends the wanted velocity
  (`{"type":"drive","vx","vy","wz"}`) on each change, with chords debounced by 40 ms (S+D+E pressed together is
  one change), plus a 5 Hz heartbeat while keys are held, and `{"stop":true}` on release. The server owns the body op:
  - **velocity mode** (bodies with the streaming `velocity` op, body/velocity.py): the first message starts it with a
    fresh `stream` id; the server then streams updates at 10 Hz from the box (no tunnel jitter) on the same stream,
    so a chord change or a long hold creates no new op and no pre-emption; release sends `end`. `watchdog_s` 0.5,
    so the body stops by itself if the server dies.
  - **walk mode** (bodies that answer `unknown op 'velocity'`): one `walk` per distinct key set, `duration_s` 10,
    re-sent only after 8 s of unchanged keys (the old page re-sent every 2 s and sent 2-3 walks per chord); release
    sends `stop`.
  - `--drive auto` (default) tries `velocity` at the first key press and falls back to walk for the rest of the run;
    `--drive velocity|walk` forces one.
  - Deadman: no heartbeat for 0.6 s (tab hidden, tunnel stalled, page closed) ends the drive like a release. Any other
    command from the same page (Space/Stop, Stand, a map click) supersedes the drive without an extra message.
- **Space = stop**, and the drive keys work wherever the focus is except text fields: the viz-level `<select>`,
  buttons and sliders do not swallow them, the select gives focus back after a pick, and Space on a focused button
  does not also click it.
- The telemetry strip shows t_sim, RTF, x, y, yaw, pelvis_z, fallen, feet, band, body op/phase, control, fault,
  and the VIZ TEST note. Header pills show received and displayed fps per stream, the gt.pose rate, body state
  and P1 RTF. A "viz level" menu sends `viz_level` to P1's REP; the log shows the reported stall, and while a body op
  is running the page asks first (a switch stalls the sim loop 0.1-0.35 s, section 7.2).
- **Record** spawns `recorder.py --control ...`. Every UI command is forwarded to it as a note, so the composite
  shows the last command. When recording stops, the page links the contact sheet and videos (served from the box).

## 6. Recorder (`viz/recorder.py`, `viz/pull_recordings.sh`)

```bash
viz/.venv/bin/python viz/recorder.py --duration 30 [--label L] [--port-offset N] [--gt-rep P ...] [--top-long 768]
viz/.venv/bin/python viz/recorder.py --until-event go_to:succeeded,failed,fallen [--post-roll 2]
viz/.venv/bin/python viz/recorder.py --control tcp://127.0.0.1:5620 [--idle]      # start/stop via socket
viz/.venv/bin/python viz/recorder.py ctl tcp://127.0.0.1:5620 start|stop|status|note "text"|quit
viz/pull_recordings.sh [all|latest|<run>]                                         # laptop, rsync via sync_wl.sh
```

- Output `/work/worldline-g1/outputs/recordings/<YYYYmmdd-HHMMSS>[-label]/`: `head.mp4`, `chase.mp4`, `top.mp4`
  (`overview.mp4` at high), `composite.mp4`, `telemetry.jsonl` (every gt.pose, body event, frame arrival meta,
  note), `contact_sheet.png` (12 evenly spaced composite frames), `last.jpg`, `summary.json` (fps per stream,
  distance travelled, fallen ever, stop reason).
- Sampled on the **wall clock** at 10 fps: each tick writes the newest frame of every stream, so all videos stay
  aligned. At RTF < 1 motion looks slower than real; t_sim and RTF are in every frame's header.
- Composite (1280x778): head | chase over top | telemetry (or overview at high), with a two-line header: rec time,
  wall time, t_sim, RTF, pose, yaw, pelvis_z, upright/FALLEN; then body op/phase/last event and the last command,
  with the source's note (e.g. `VIZ TEST: kinematic`) right-aligned in the header, never over a tile.
- Telemetry tile: the pose block, the last command, the go_to target with its distance, and the newest body events
  fitted to the 360 px tile (6 events with everything else shown; older ones are counted). A run of `progress`
  events of one op is one line with a count, so `accepted` and `succeeded` stay visible.
- Commands and the go_to target come from UI notes **and** from the body's `accepted` events (`data.args`), so
  commands sent by the CLI, `/api/cmd` or BodyClient also show the target marker and the `cmd:` line.
- Top view: the VizCams top frame with the live trajectory, robot arrow and go_to target drawn over it; without a
  top frame (VizCams off or `min`), P1's cached `render_topdown` with the same live overlay; without that, the
  occupancy map, then the scene bounds. `top.mp4` has a fixed long side (`--top-long`, 768 px) whatever image comes
  first; `chase.mp4` is at least 640 px wide, so a level switch from min to low keeps its native size.
- x264 veryfast CRF 23 via imageio-ffmpeg: about 10-14 MB per minute of a clean house run. The recorder tick
  (decode, compose, encode 4-5 videos) takes 35-50 ms of one core at 10 fps.

## 7. Measured results

Box: 1x L40S shared with other agents' Isaac/SONIC jobs; laptop via the SSH tunnel.

### 7.1 Streams and browser delivery

At the source (stand-in P1 in procthor-train-40, `low`, paced): head 29-30 Hz, chase 10.0 Hz (0 forced renders,
0 dropped, 0 empty), top snapshot every 10 s, gt.pose 50 Hz, body.state 5 Hz.

Through the SSH tunnel, measured on the laptop (`curl` for MJPEG, `viz/ws_probe.py` for the WebSocket; HTTP RTT
179 ms; latency = box publish to laptop receive, clock offset removed):

| Path | chase | head |
|---|---|---|
| MJPEG `curl /stream/<name>.mjpg`, 10 s | 98 frames = **9.8 fps**, 157 KiB/s | 144 frames = **14.4 fps** (server cap 15), 123 KiB/s |
| WebSocket, window 4, 12 s | **10.1 fps**, p50 **88 ms**, p90 90 ms | **14.8 fps**, p50 **102 ms**, p90 120 ms |
| Chromium page (Playwright, flat scene, earlier run) | 10.1 fps displayed | 14.8 fps displayed |

The page's click-to-go_to, WASD walk and record button were driven in Chromium (`outputs/viz_test/tunnel/ui_*.png`)
and, in this run, through the same HTTP API from the laptop (`/api/cmd`, `/api/record`).

### 7.2 RTF cost

`--bench` in procthor-train-40, `balanced` preset, CPU PhysX 200 Hz, kinematic G1, head camera 640x480 rendered at
30 Hz like P1, unpaced, 10 sim-s phases interleaved off/min/low/high and repeated 3 times; medians. Quiet box
(no other Isaac or SONIC job), `outputs/viz_test/bench_005250.json` on the box:

| level | RTF | wall-ms per sim-s | vs off | render ms per P1 render |
|---|---|---|---|---|
| off | **1.275** | 784 | - | 8.82 |
| min (chase 480x270) | **1.063** | 941 | +20 % | 11.73 (+2.9) |
| low (chase 640x360, top snapshot 10 s) | **1.041** | 960 | +22 % | 11.83 (+3.0) |
| high (chase 960x540, overview, top 3 s) | **0.841** | 1188 | +52 % | 16.40 (+7.6) |

- The stand-in with the viz off matches P1: the isaac agent's final RTF matrix gives `house_cam30_cpu_free` = 1.294
  for P1 itself in the same house (`outputs/m1/isaac/final/rtf/rtf_summary.md`).
- Every live viz camera adds about 3 ms to **every** P1 render (section 4, finding 2). At P1's 30 Hz head camera
  that is ~90 ms per sim-s; the rest of the ~160 ms was the per-frame read (2.2-2.6 ms at the time, ~1.0 ms
  since round 2, see below), placement and GIL time.
  After this bench, `step()` was made to return at once when nothing is due and to place the chase camera once
  per frame instead of every physics step: a second bench (box loaded by the deploy agent, so absolute RTF lower)
  gave min +16 %, low +19 %, high +34 % (`bench_005602.json`).
- Chase placement with scene queries on (v4 run, house): 0.95 ms mean, 2.2 ms p99 per chase frame (10 Hz, about
  1 % of real time); 5,508 rays in 58 s, 39 % of them hit geometry; the chosen view was not fully free 17 % of the
  time (camera pulled in).

**Sim-thread work of VizCams itself** (`viz_stats`, procthor-train-40, level `low`, stand-in created through
`attach_p1` (`--hook`), paced, server + recorder + streams running; round 2):

| per | what | before round 2 | now |
|---|---|---|---|
| chase frame (10 Hz) | annotator read `get_data()` (`readback_ms`) | 0.56 mean / 0.86 p99 | 0.55-0.60 / 0.76-0.90 |
| chase frame | buffer copy before the next render (`copy_ms`) | 1.69 / 3.03 (strided RGB) | **0.11 / 0.15-0.24** (contiguous RGBA; alpha dropped on the worker) |
| chase frame | total (`capture_ms`; the round-1 verifier measured 2.43 / 5.49) | 2.59 mean | **0.93-1.02 mean** |
| chase frame | placement incl. raycasts (`chase_place_ms`) | 0.92 / 2.03 | 0.86-0.89 / 1.84-1.86 |
| top snapshot (10 s at low) | re-arm (`rearm_ms`) | 16.1-16.5 | 14.6 |
| level switch | `set_level()` call (`set_level_ms`) | - | 25-39 ms to min/low, 89 ms to high |
| level switch | loop stall = max gt.pose wall gap in the next second (baseline max 31 ms) | - | **120-160 ms** to min/low, **~340 ms** to high |

Per sim-second at `low` that is about 10 x (1.0 + 0.9) + 1.5 = ~20 ms (~2 % of real time), down from ~35 ms. The
worker thread (JPEG encode ~1.1 ms per frame, send, re-send) is off the sim thread. The one-off switch stall is why
the page asks before switching while a body op runs; with SONIC in the loop set the level at start (`--viz`) and do
not switch while it walks. The dominant cost remains the per-render cost of each live product (above).
- Consequence for M1: P1 has ~25 % headroom at real-time pacing (free-running RTF 1.29). `min`/`low` use most of
  it; the paced stand-in held RTF p50 1.00 with p10 0.88-0.93 while the server, a recorder and the tunnel streams
  also ran. **Use `off` for RTF-critical SONIC runs and measurements, `min` for demos and recordings, never
  `high` while SONIC is in the loop.** The cost is per host render, so P1 `--camera-hz 15` should roughly halve
  it (inferred from the per-render cost, not measured; P1's own matrix shows 15 Hz alone does not raise its RTF,
  contract 1.10). Compare P1's ~30 % free-running headroom (contract 1.10) with the numbers above.
  P1's own `--tp-camera` costs ~7 ms per render (contract 1.8), more than VizCams `min`.

### 7.3 Top-view mapping and freshness

`--selftest` (flat scene, magenta cube moving at 1.5 m/s, 17 top frames at 768 px): all found; world error 1.0 cm
mean, 2.1 cm max (under 1.5 px); implied capture lag 6.8 ms, about one physics step.

### 7.4 Recordings reviewed (pulled to `outputs/recordings/` on the laptop and looked at)

| Run | What it shows |
|---|---|
| `20260928-235505-house40` (earlier run, no RTX preset) | pipeline works in the house, but every frame is speckled black/white (no preset, section 4 finding 5); 40 s = 45 MB because noise does not compress |
| `20260929-005915-ui-drive-house40` (45 s, `/api/record` + `/api/cmd` over the tunnel) | clean frames; header shows each UI command (`walk`, `go_to {x:5.5,y:5.5}`, `turn_to`, `stop`) and body events; top snapshot + cyan trajectory + robot arrow. The then straight-line fake go_to walked the kinematic robot through the fridge and a wall, and the chase camera then showed walls: fixed below |
| `20260929-010009-min-p1topdown` (20 s, CLI, level `min`) | no VizCams top: the top tile is P1's `render_topdown` with the live trajectory and arrow at the right place; chase 480x270 shows the G1 mid-stride walking through a doorway toward the dining table |
| `20260929-010703-loop-goto-v3` (62 s, `--until-event go_to:succeeded`) | demo tour through all rooms, then a UI go_to across the house along a grid path through the bedroom door; the robot is visible in 10 of 12 contact-sheet frames; 2 frames face a wall/counter. Diagnosis: the stand-in ran without PhysX scene queries, so the chase raycasts never hit anything (P1 has them on) |
| `20260929-011922-chase-avoid-v4` (56 s, scene queries on, fat ray + azimuth scan) | tour, then a UI go_to into the narrow living room (0.6 m clearance): the robot is visible in **all 11** frames after start-up, including a pulled-in view looking down over the robot in the tight spot; `go_to` accepted -> succeeded in 8.9 s, final pose (4.46, 7.88) for the target (4.475, 7.925); stopped by `--until-event` |
| `20260929-012054-short-check` (12 s) | the recorder now waits (up to 3 s) for the first data, so contact-sheet frame #1 is no longer black |
| `20260929-020720-r2-verify` (37 s, round 2: stand-in via `attach_p1`, server at +300, **server-spawned** recording) | `summary.json` ports gt_rep 5900 / body_ctl 5910 (the server's, not 5600/5610); a `go_to(1.7, 4.0)` sent through `/api/cmd` shows the green target marker and `target ... m away` in the panel; the panel shows `accepted`, `progress x7`, `succeeded` and, after a held-key drive (walk fallback: 2 walks, then stop), 6 events + "(5 older)", all inside the tile; the VIZ TEST note is in the header; top.mp4 768x768 although its first frame was the stand-in's `render_topdown` fallback; chase.mp4 stayed 640x360 across a low -> min -> low switch; the top snapshot arrived 2 s into the recording (re-send) instead of up to 10 s |

Head camera frames mostly show the floor and furniture edges in front of the robot: correct for the d435 mount,
which looks ~48 deg down (contract 1.4). A 45-60 s clean house run is 7-14 MB.


## 8. Native WebRTC livestream (optional path, not used)

What Isaac provides (verified in the Isaac Sim 5.1 install on the box):

- Isaac Lab's `AppLauncher --livestream 1` enables `omni.services.livestream.nvcf` with
  `--/app/livestream/publicEndpointAddress=$PUBLIC_IP --/app/livestream/port=49100`; `--livestream 2` is the
  private-network variant (`IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py:512-560`).
- Signalling: `omni.kit.livestream.webrtc` 7.0.0, `app.livestream.port = 49100`, `app.livestream.proto =
  "websocket"` (its `config/extension.toml`), i.e. **TCP 49100**.
- Media: `omni.services.livestream.nvcf` 7.2.0, `app.livestream.minHostPort = 47998`, `maxHostPort = 48020`
  (its `config/extension.toml`): **UDP 47998-48020** from the client to the box, with `PUBLIC_IP` set to the
  address the client sees.
- It streams the **Kit viewport** (one extra full-resolution render per frame, not our cameras) to NVIDIA's
  "Isaac Sim WebRTC Streaming Client" desktop app, not to a plain browser page.

Why not here:

- SSH forwards TCP only, so the UDP media cannot go through `tunnel.sh` (or `brev port-forward`, which is the same
  SSH `-L`).
- Brev cannot open ports on this instance type. In the Brev instance-type catalogue, `canModifyFirewallRules` is
  `true` for GCP, AWS and Crusoe types and absent for every Nebius type, including ours
  (`gpu-l40s-a.1gpu-16vcpu-64gb`, provider `nebius`), which matches `flex_ports=false`.
- Brev's workspace-level "expose public ports"/share link is an HTTP(S) tunnel. It could publish the viz server
  (TCP, WebSocket), but not WebRTC media. Not tried: it would make the drive controls public, so it needs auth in
  front first.

The only ways to get native WebRTC would be a different instance type (AWS/GCP/Crusoe with firewall control) or a
TURN relay reachable over TCP/TLS (for example coturn on a VM with a public IP, with the client forced to relay).
Both add a moving part for no gain over the JPEG path, which already gives 10-15 fps at about 100 ms. JPEG over ZMQ
and WebSocket stays the design.

## 9. Integration hooks

### 9.1 `sim_isaac/app.py` (P1, owned by the isaac agent: apply there)

**7 lines at 5 places**: 6 added lines and 1 changed line; the last place (`finish()`) is optional. Anchors checked
against the committed P1 v0.5 (97b105b, unchanged since): `parse_args()` line 54 (`--tp-camera`) and line 79
(`enable_cameras`), the end of `setup()` after the `gt.register` loop (line 249), the camera block in `run()`
(lines 473-500), `finish()` (line 735).

```python
# 1) parse_args(), next to --tp-camera (2 added lines):
from viz.isaac_cams import add_p1_args, attach_p1, viz_enabled
add_p1_args(ap)                                        # --viz off|min|low|high (default off), --viz-hz 10
# 2) parse_args(), line 79 (1 changed line). viz_enabled() exits with a clear error if --viz AND --tp-camera:
if a.camera != "none" or a.tp_camera or viz_enabled(a):
# 3) end of setup(), after the `for op in (...): self.gt.register(...)` loop (1 added line):
self.viz = attach_p1(self)          # None when --viz off; else VizCams + REP ops viz_level/viz_stats + warm-up
# 4) run(), right after the camera block (`if do_cam or do_tp: ...`), before the gt.pose publish (1 added line):
if self.viz: self.viz.step(t_sim, st["base_pos"], st["base_quat"])    # base_quat is wxyz (_read_state)
# 5) finish(), optional (1 added line): the PUB socket and render products go away at os._exit anyway
if self.viz: self.viz.close()
```

What `attach_p1(app)` does (`viz/isaac_cams.py`): it reads `app.a.viz` and `app.a.viz_hz`, builds
`VizCams(app.sim, "/World/G1", app.scene.bounds, pub=tcp://127.0.0.1:{app.ports["frames_pub"]}, floor_z=app.floor_z)`,
registers `viz_level` and `viz_stats` on `app.gt`, places the chase camera at `app._read_state()`'s pose, and does
`min(--warmup-renders, 10)` warm-up renders, which also publish the first top snapshot. So the render products are
built at start-up, not inside the paced loop (P1's own warm-up block runs before it and does not need to change).
Measured through this exact path in the stand-in (`test_stack.py --hook`, a P1-shaped shim): 10 warm-up renders in
0.37-0.53 s, `set_level` at construction 33-34 ms, the REP ops answer through the registered handlers.

P1 already sets `SimulationCfg(enable_scene_query_support=True)` (app.py:119), which the chase raycasts need.
Run it with `python -m sim_isaac.app ... --viz min` (or `low`). `viz/` needs nothing beyond the Isaac Lab env:
pyzmq, msgpack and opencv are already there. Until this lands, `--tp-camera` gives the chase pane (P1's own camera,
640x480 at `--tp-hz`), and the top view comes from P1's `render_topdown` plus gt.pose.

### 9.2 Future Worldline `ui/server.py`

Keep the existing websockets protocol and pull frames with `viz/tap.py`:

```python
from viz.tap import FrameTap
self.tap = FrameTap(port_offset=0)                 # SUB 5565 + 5602 + 5601, background thread
# in _send_cameras():
for which in ("head", "chase", "top"):
    if self.tap.rev(which) != self._sent_rev.get(which):
        jpeg = self.tap.jpeg(which)                # head and P1 frame.tp are R/B-fixed; VizCams frames pass through
        self._sent_rev[which] = self.tap.rev(which)
        broadcast(self.clients, dumps({"type": "camera", "which": which, "jpeg": base64.b64encode(jpeg).decode(),
                                       "meta": self.tap.meta(which)}))   # meta["extent"] maps the top view
```

The Worldline page already renders `{"type":"camera","which":...}` messages (`ui/index.html` `setCamera`). The
`top` meta carries `extent` for drawing the belief map over reality. Alternatively, run `viz/server.py` on 8766 and
embed `/stream/chase.mjpg`; that needs a second tunnel port.

## 10. Files

| File | What |
|---|---|
| `viz/isaac_cams.py` | VizCams (Isaac process) |
| `viz/server.py`, `viz/static/index.html` | web server and page |
| `viz/recorder.py` | recorder CLI plus `Recorder` class plus control socket (`ctl ADDR start/stop/status/note/quit`) |
| `viz/pull_recordings.sh` | laptop: `all`, `latest`, or a run name |
| `viz/tap.py` | FrameTap for other UIs |
| `viz/common.py` | ports, decoders (`decode_head`, `frame_from_msg`), overlay drawing, occupancy PNG |
| `viz/probe.py`, `viz/ws_probe.py` | box-side stream probe; laptop-side WebSocket fps and latency probe |
| `viz/test_stack.py` | VIZ TEST stand-in P1 (house + kinematic G1, fake body API, selftest, settle sweep, bench) |
| `viz/tests/test_frames.py` | 17 tests without Isaac: frame formats, re-send dedupe, recorder panel/sizes/targets, hook guard, server ports, held-key drive |
| `viz/tests/fake_body.py` | fake body ROUTER (with or without `velocity`) for the tests and for driving the page by hand |
| `viz/box.sh` | tmux launcher (`viz-*` sessions only) |

## 11. Known limits

- Live cameras cost render time on every P1 render (section 7). Use `min` for demos and `off` for RTF-critical
  measurements: the recorder and the page still have head, P1's top-down + live pose, and occupancy.
- The chase camera sees only colliders: visual-only geometry can still occlude the robot, and without PhysX scene
  queries (Isaac Lab default) it cannot avoid anything (VizCams warns at start).
- The kinematic test robot passes through furniture when `walk` drives it into it (it is not physics); its fake
  `go_to` follows a BFS grid path through doors (a VIZ TEST helper, not the body planner).
- `simulation_app.close()` can hang after Replicator use, so `test_stack.py` exits with `os._exit(0)`.
- Recordings are sampled on the wall clock at 10 fps. At RTF < 1, motion looks slower than real.
- A viz-level switch stalls the sim loop once by 0.12-0.34 s (section 7.2). P1's lowstate heartbeat thread keeps
  `rt/lowstate` flowing through such a stall (contract 1.10), but do not switch while SONIC walks.
- The held-key `velocity` path is tested against `viz/tests/fake_body.py`, which follows body/velocity.py; the body
  agent's `velocity` op was uncommitted when this was written. Against the stand-in (no `velocity`) the walk fallback
  ran end to end.
