# sim_isaac: wl-isaac (P1)

Isaac Sim 5.1 / Isaac Lab 2.3.2 process that stands in for a real Unitree G1: it runs the physics at 200 Hz in wall-clock
real time, speaks the robot's Unitree DDS topics (so the **unmodified** `gear_sonic_deploy` drives it exactly as it
drives the MuJoCo sim or the real robot), publishes the head camera in gear_sonic's format, and serves ground truth.
The interface is specified in [`docs/contracts/m1.md`](../docs/contracts/m1.md) §0-§1, and the M2b additions (live object
poses, attach/detach, head + ego_view cameras, link poses, detections, scene reset, health) in
[`docs/contracts/p1_m2b.md`](../docs/contracts/p1_m2b.md). Offline tests: `.venv-rt/bin/python -m pytest -q sim_isaac/tests`.

## Files

| File | What |
|---|---|
| `app.py` | the app: scene, G1, physics loop, DDS bridge, band, cameras, gt.pose PUB, REP server, stats |
| `g1_asset.py` | builds `/work/worldline-g1/assets/g1/g1_sonic_dex3.usd` from SONIC's training URDF (Dex3 joints made revolute) and the training-parity `ArticulationCfg` (actuators imported from `gear_sonic`) |
| `joint_map.py` | Unitree motor order, Dex3 order, default pose, deploy/training kp/kd, d435 mount (all cited) |
| `dds_bridge.py` | `G1DdsBridge`: lowstate/secondary_imu/odostate/dex3 state out, lowcmd/dex3 cmd in; lowstate heartbeat; `LowStateCrc` |
| `fastdds.py` | fixed-layout CDR codec for the unitree_hg messages (verified against the library at start-up) |
| `band.py` | elastic band (port of the MuJoCo `ElasticBand`) |
| `rt_pacer.py` | wall-clock pacing, RTF windows, duration stats |
| `camera.py` | camera prims (spec-based, explicit apertures), replicator RGB capture (re-arm after disable), gear_sonic-format publisher (+ M2b frame metadata), chase camera, top-down render (optionally hiding prims) |
| `cameras.py` | M2b `CameraRig`: head (5565) + Arena-exact ego_view (5566), shared renders on a sim-time grid, consumers with TTL, `detections` (instance-id segmentation) |
| `objects.py` | M2b `ObjectTracker`: live prop poses (one PhysX rigid-body view), attach/detach follow and fixed_joint (STEPPING STONE), reset/move/push, object_fell |
| `segment.py` | instance prim -> scene object mapping and pixel counts for `detections` |
| `wire.py` | pure helpers of the M2b wire (camera specs with citations, frame metadata, object records, health level, fall detector), shared with `tools/fake_p1.py` |
| `tools/m2b_probe.py`, `tools/rtf_cameras.py`, `tools/detections_check.py` | M2b live checks: smoke / attach cycles / latency / reset / carry; RTF per camera config while SONIC stands and walks; P1 detections vs world's gt-geometric visibility |
| `gt_server.py` | REP (JSON or msgpack) + PUB transport |
| `scene.py` | empty ground plane or `scenes/loader.py:load_house` (house agent) |
| `occupancy.py` | PhysX-overlap occupancy fallback (houses use the house agent's cached grid) |
| `tools/stand_peer.py` | DDS peer standing in for the deploy: stand command at 500 Hz + checks (rates, joint map, lowstate vs GT, camera, REP ops) |
| `tools/measure_rtf.sh`, `tools/summarize_rtf.py` | the RTF matrix (E6) |
| `tools/record_video.py` | ego / chase / top-down (GT overlay) mp4 recorder |
| `tools/sonic_netns_test.sh`, `tools/sonic_smoke.py` | SONIC in the loop: P1 + the unmodified deploy + a minimal planner driver in a private network namespace (stand, walk, turn; report + E4 trace + videos) |
| `tools/final_validation.sh` | the whole stand-alone validation in one go: pure tests, DDS stand test in a house, RTF matrix, SONIC smoke (one Isaac instance at a time, ~15 min) |
| `tools/run_app.sh`, `tools/bg_app.sh` | run in the foreground / in tmux `isaac-app-<name>` (stops the previous one, waits for ports) |

## Build and run (on the box)

```bash
source /etc/profile.d/ludo.sh && cd /work/worldline-g1
/work/envs/isaaclab/bin/python -m sim_isaac.g1_asset --build          # once: URDF patch + USD
bash sim_isaac/tools/run_app.sh --house procthor-train-40 --dds-domain 0 --dds-iface lo \
     --camera 640x480 --camera-hz 30 --physx-device cpu               # integrated stack (ports 5600/5601/5565)
# build-phase test on domain 7, ports +100, with the stand peer:
bash sim_isaac/tools/bg_app.sh test --house empty --dds-domain 7 --port-offset 100 --duration 300
/work/envs/isaaclab/bin/python -m sim_isaac.tools.stand_peer --domain 7 --port-offset 100 \
     --hold-s 60 --camera --joint-map --ops --gains stiff
```

Environment added to `/work/envs/isaaclab` for this component: cyclonedds 0.10.2 (C library built from source at
`/work/opt/cyclonedds-0.10.2`, **with `-D_FORTIFY_SOURCE=0`**: Ubuntu 24.04's default fortify level aborts with
"buffer overflow detected" in `Domain(...)` as soon as a network interface is configured), `cyclonedds==0.10.2`
(Python), `unitree_sdk2py` (editable, from `$WBC/external_dependencies/unitree_sdk2_python`, `--no-deps`), `pyzmq`.

## Status and measured numbers (contract §1.9-§1.10)

Final validation (`outputs/m1/isaac/final/`, house `procthor-train-40`, quiet box): DDS stand test pass; RTF with
G1 + head camera 30 Hz + house on CPU PhysX, paced: **0.998 total, 1.000 over 10 s**, worst 1 s window 0.88,
~30% free-running headroom (1.29); GPU PhysX 0.22 (rejected). With the **unmodified SONIC deploy** in the loop:
stand 60 s, walk 2.5 m, turn, no falls, leg targets at 50 Hz, RTF 0.993 / 1.000. Chosen path: the DDS bridge +
the unmodified wall-clock deploy (the lockstep / sim-clock fallbacks are not needed for M1).

P1 is controller-agnostic: it only speaks the real G1's low-level Unitree DDS (`rt/lowcmd` in, `rt/lowstate` +
`rt/secondary_imu` + Dex3 out), so any unitree_sdk2 low-level controller can drive it without changes to P1.

## Design notes

- **Training parity.** Same URDF, same conversion options, same actuator groups (imported from
  `gear_sonic/envs/manager_env/robots/g1.py`), dt 0.005, solver 8/4, self-collisions, ground friction 1.0/multiply.
  Implicit PD in PhysX with the kp/kd the deploy sends: the deploy's `kps/kds` equal the training stiffness/damping.
- **MuJoCo-bridge semantics.** Message content and IMU conventions follow
  `gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py` exactly (quaternion wxyz world, gyro in body frame,
  accelerometer = world linear acceleration without gravity, torso IMU on `rt/secondary_imu`, tick = sim ms,
  odostate orientation wxyz). Differences are listed in the contract (§1.2, §1.7): default-pose hold before the first
  lowcmd, band anchor/yaw, CRC filled in.
- **Speed.** Isaac Lab's `Articulation` wrappers and cyclonedds-python's pure-Python serializer cost ~9 ms per
  physics step for one robot (measured), i.e. RTF ~0.45. The loop therefore talks to the PhysX tensor views
  directly and uses the verified fixed-layout CDR codec: ~2 ms per step (PhysX CPU step ~1.2 ms).
- **Static stand with the training gains falls** (measured: 1.7 s after band release). Expected: ankle kp is
  2 x 14.25 = 28.5 Nm/rad against a toppling stiffness m*g*h of roughly 35*9.81*0.7 = 240 Nm/rad. Balancing
  is SONIC's job; the stand test therefore uses stiff gains (legs 350, waist 400) to check contact/actuation/state,
  and the training-gain fall is reported as a physics sanity check.
