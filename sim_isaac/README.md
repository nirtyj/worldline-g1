# sim_isaac: wl-isaac (P1)

Isaac Sim 5.1 / Isaac Lab 2.3.2 process that stands in for a real Unitree G1: it runs the physics at 200 Hz in wall-clock
real time, speaks the robot's Unitree DDS topics (so the **unmodified** `gear_sonic_deploy` drives it exactly as it
drives the MuJoCo sim or the real robot), publishes the head camera in gear_sonic's format, and serves ground truth.
The interface is specified in [`docs/contracts/m1.md`](../docs/contracts/m1.md) §0-§1.

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
| `camera.py` | head camera prim at `d435_link`, replicator RGB capture, gear_sonic-format publisher, chase camera, top-down render |
| `gt_server.py` | REP (JSON or msgpack) + PUB transport |
| `scene.py` | empty ground plane or `scenes/loader.py:load_house` (house agent) |
| `occupancy.py` | PhysX-overlap occupancy fallback (houses use the house agent's cached grid) |
| `tools/stand_peer.py` | DDS peer standing in for the deploy: stand command at 500 Hz + checks (rates, joint map, lowstate vs GT, camera, REP ops) |
| `tools/measure_rtf.sh`, `tools/summarize_rtf.py` | the RTF matrix (E6) |
| `tools/record_video.py` | ego / chase / top-down (GT overlay) mp4 recorder |
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
