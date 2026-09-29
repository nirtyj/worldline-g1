# Dev box `ludo-g1-arena`: a ready backup of the main box

Status: **installed and verified 2026-09-29 (04:30-05:10 UTC, box time)**. The dev box runs the same stack, at the
same pins, as the main box `ludo-g1-brev2`. On the dev box: the M1 stack came up and passed every drive test
(E2, E3, E4). A `go_to` through Nav2 succeeded. The Isaac Lab SONIC eval gave the same metrics as main, to every digit.
The GR00T N1.7 server answered the synthetic client.

| | Main box | Dev box |
|---|---|---|
| Brev name | `ludo-g1-brev2` | `ludo-g1-arena` |
| Hardware | Nebius 1x L40S 48 GB, 16 vCPU, 62 GiB | same |
| Driver / OS | 580.173.02 / Ubuntu 24.04.5 | same |
| Disk | 484 GB (127 GB used) | 387 GB (206 GB used, 182 GB free; 103 GB of it is the Arena spike) |
| Role | live demo stack (tmux `wl-m1` + `viz-server`) | backup + Arena spike (`/work/arena`, docker container `arena`) |

## 1. Switching any script between the boxes

All laptop-side tooling is `ludo_robotics_prep_g1/00_infra/*.sh`. The box is chosen by `BREV_NAME`, and
`lib.sh` defaults it to the main box. Always set it explicitly for the dev box:

```bash
cd /Users/nirty/workspace/ludo-interview/ludo_robotics_prep_g1/00_infra
BREV_NAME=ludo-g1-arena ./ssh.sh '<cmd>'        # or: ./ssh.sh for an interactive shell
BREV_NAME=ludo-g1-arena ./sync.sh push          # prep repo -> /work/prep
BREV_NAME=ludo-g1-arena ./sync_wl.sh push       # worldline-g1 -> /work/worldline-g1 (read "Known issues" 2 first)
BREV_NAME=ludo-g1-arena ./secrets.sh            # ~/.config/ludo-g1/secrets.env + /work/hf-cache/token
BREV_NAME=ludo-g1-arena ./tunnel.sh 8765        # viz UI -> http://localhost:8765
BREV_NAME=ludo-g1-arena ./status.sh | start.sh | stop.sh
# or for a whole shell:  export BREV_NAME=ludo-g1-arena
```

- Per-box ssh state: the main box uses `00_infra/.state/`, the dev box `00_infra/.state-ludo-g1-arena/`
  (`instance.env` = IP, `ssh_config`, `known_hosts`). `lib.sh` only refreshes the IP when `instance.env` is
  missing. **If the box gets a new IP after a stop/start**, delete `.state-ludo-g1-arena/instance.env`.
- Nothing on the box side depends on the box name. Every box-side script uses the same absolute paths
  (`/work/...`, `/etc/profile.d/ludo.sh`), so the commands in `docs/M1.md`, `docs/contracts/*.md`, `docs/nav2.md` and
  `docs/scenes.md` work unchanged on either box.
- `/work/worldline-g1` on a box is whatever was last pushed from the laptop, including uncommitted work. Push
  before running anything, so both boxes run the same code.

## 2. What is installed where (dev box)

Every item was installed with the existing script named in the "How" column, and then verified.

| Component | Path | How | Verified |
|---|---|---|---|
| Base bootstrap | `/etc/profile.d/ludo.sh`, uv 0.12.20, git-lfs, ffmpeg, xvfb, docker, tmux 3.4 | infra bootstrap (before this) | present |
| Keys | `~/.config/ludo-g1/secrets.env` (HF, Gemini, Anthropic, Typesafe), `/work/hf-cache/token` | `00_infra/secrets.sh` | names listed, `HF token installed` |
| Pinned repos | `/work/repos/{Isaac-GR00T 51d4c89, GR00T-WholeBodyControl b042411, Prompt2Policy a84c76f, IsaacLab v2.3.2 37ddf62}` | bootstrap | SHAs match `PINNED_SHAS.txt`; WBC tree clean |
| Isaac Lab env (lab 04) | `/work/envs/isaaclab`: Python 3.11.16, isaacsim 5.1.0.0 (pip), torch 2.7.0+cu128, torchvision 0.22.0, isaaclab 0.54.2, gear_sonic[training] 0.1.0 (editable) | `04_g1_sonic_isaaclab_rl/setup.sh` | `check_environment.py --training`: All checks passed; ready marker; **`uv pip freeze` identical to main** |
| SONIC release ckpt + sample data | `$WBC/sonic_release/{last.pt,config.yaml}`, `$WBC/sample_data/` | lab 04 setup | eval.sh (8 envs): success 1.0, mpjpe_l 17.3548, mpjpe_g 94.5548 = main's numbers exactly |
| GR00T env (lab 01) | `/work/repos/Isaac-GR00T/.venv`: Python 3.12, torch 2.9.0+cu128, flash-attn 2.8.3, transformers 4.57.3, torchcodec 0.8.0 | `01_groot_g1_open_loop/setup.sh` | freeze identical to main; served the community ckpt on GPU: synthetic client SHAPE OK, p50 136 ms |
| GR00T models | HF cache: `nvidia/GR00T-N1.7-3B@2fc962b`, `nvidia/Cosmos-Reason2-2B@9ce19a1` (gated, licence accepted) | lab 01 setup | `ARTIFACTS OK` |
| UNOFFICIAL community ckpt + dataset | `/work/checkpoints/community-g1-sonic/.../checkpoint-20000`, `/work/models/datasets/gr00t-g1-grab-bottle-right-hand-v2` (+ `meta/stats.json`) | lab 01 setup | `ARTIFACTS OK`; contract check True |
| Prompt2Policy (lab 03) | `/work/repos/Prompt2Policy/.venv` (py3.11, mujoco 3.5.0, torch 2.10.0+cu128, sb3 2.7.1), lab patches applied, `runs -> /work/outputs/03_.../runs`, `.env`; Node 22.23.3 in `/work/envs/p2p-node`, `frontend/node_modules` | `03_prompt2policy_mujoco/setup.sh` | 1087 tests passed; EGL render check ok; live Gemini probe ok; freeze and patch diff identical to main |
| SONIC deploy toolchain | TensorRT 10.13.3.9 `/work/opt/TensorRT-10.13.3.9` (+ `~/TensorRT` link, tarball in `/work/downloads/tensorrt`), CUDA 12.9 user space `/work/opt/cuda-12.9`, ONNX Runtime 1.16.3 `/opt/onnxruntime`, just 1.43.0 | `worldline-g1/sonic/build_deploy.sh` | step 8 smoke test OK |
| SONIC deploy binary | `$WBC/gear_sonic_deploy/target/release/g1_deploy_onnx_ref` (unmodified upstream, built **before** ROS, HAS_ROS2=0) | `sonic/build_deploy.sh` | usage + ldd OK; re-checked after the ROS install: no `/opt/ros` libraries |
| Deploy ONNX | `policy/release/`, `planner/target_vel/V2/`, `policy/sonic_v1_1/` (as on main) | build_deploy.sh step 5; v1.1 with lab 02's `download_from_hf.py --sonic-v1-1 --no-planner` | files present; TensorRT engine caches (`*.trt`) built on the first M1 start |
| MuJoCo reference loop | `$WBC/.venv_sim` (Python 3.10.21, mujoco 3.14.0, gear_sonic[sim], unitree_sdk2py, torch 2.14.0) | build_deploy.sh step 7 = `install_scripts/install_mujoco_sim.sh` | freeze identical to main (after pinning msgpack 1.2.2) |
| CycloneDDS C library | `/work/opt/cyclonedds-0.10.2` (tag 0.10.2 @ 9995905, Release, `-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0`, install/ prefix) | the exact commands of main's `/work/logs/wl/isaac-cdds{,2}.log` (`docs/contracts/m1.md` §1.3) | CMakeCache flags equal to main's |
| DDS Python in the Isaac env | `cyclonedds==0.10.2` built from source against the lib above (`__library__.library_path` = `/work/opt/...`), `pyzmq`, `unitree_sdk2py` (editable, `--no-deps`) | same log, plus `--no-binary cyclonedds` | `Domain(..., NetworkInterface lo)` and `ChannelFactoryInitialize(.., "lo")` OK (the fortify abort did not happen) |
| MolmoSpaces | `/work/repos/molmospaces @ 713fd12`, `molmo_spaces_isaac` editable `--no-deps` + the 12 exact deps | `docs/scenes.md` "Install" | pip delta = main's `house-freeze-{before,after}` delta |
| Houses | `/work/assets/molmospaces` (3.1 GB: THOR objects + procthor-train-38/-40/-15/-59 + ithor-FloorPlan10) | `python -m scenes.download ...` | `downloads.json` written |
| House assets + occupancy | `/work/worldline-g1/assets/houses/<id>/` (house_info.json, occupancy.{npz,png}, meta, rasters, occupancy_r0.25.npz) | `scenes/run_house_test.sh <5 ids>`; `scenes.occupancy.occupancy_reply(id, 0.25)` | all 5 rc=0; **every occupancy layer bit-identical to main**, same spawns |
| G1 USD | `/work/worldline-g1/assets/g1/` | `python -m sim_isaac.g1_asset --build` | `.asset_hash` and URDF identical to main (43 joints, 67 meshes) |
| Body venv | `/work/worldline-g1/.venv` (Python 3.11.16) | `uv venv` + main's exact freeze (no script in the repo) | freeze identical to main; `body/tests` 36 passed |
| Viz venv | `/work/worldline-g1/viz/.venv` | `bash viz/box.sh setup` (+ msgpack 1.2.2 pin) | freeze identical to main; `viz/tests` 17 passed; served the live stack |
| ROS 2 Jazzy + Nav2 | `/opt/ros/jazzy` (316 `ros-jazzy-*` packages, as on main) | `bash nav2/install_ros.sh` (after the deploy build) | Nav2 up on the live stack in 3 s; `go_to` via Nav2 succeeded |
| apt packages | | | the `dpkg` package list is **identical** to main (no extra or missing packages) |

The Isaac env also runs the pure-Python tests: `scenes/tests/test_pure.py` 6 passed and `sim_isaac/tests/test_pure.py`
5 passed. The Isaac Sim extension cache `~/.local/share/ov` (194 MB) was filled on the first launch.

Two packages were pinned by hand after the scripts, because upstream released newer versions on 2026-09-29:
`msgpack==1.2.2` in `/work/envs/isaaclab`, `viz/.venv` and `.venv_sim`, and `regex==2026.9.10` in
`/work/envs/isaaclab`. With these pins, every venv's freeze matches main's. The two scripts that install unpinned
(`viz/box.sh setup` and upstream `install_mujoco_sim.sh`) will drift again on a rebuild (see "Known issues").

## 3. Running the M1 stack on the dev box

```bash
# laptop
cd /Users/nirty/workspace/ludo-interview/ludo_robotics_prep_g1/00_infra
export BREV_NAME=ludo-g1-arena
./sync_wl.sh push                                    # same code as the main box
./ssh.sh 'cd /work/worldline-g1 && bash scripts/m1_up.sh'            # tmux wl-m1 (house procthor-train-38, A*)
#   NAV_BACKEND=nav2 bash scripts/m1_up.sh           # with Nav2 (nav2/m1_hook.sh starts tmux wl-nav2)
./ssh.sh 'cd /work/worldline-g1 && .venv/bin/python -m tools.m1_drive_test --port-offset 0'
./ssh.sh 'cd /work/worldline-g1 && bash viz/box.sh server 0'  &&  ./tunnel.sh 8765   # http://localhost:8765
./ssh.sh 'cd /work/worldline-g1 && bash viz/box.sh stop; bash scripts/m1_down.sh'
```

The same stack, measured on the dev box (session `dev-m1`, run dir `outputs/m1/stack-20260929-045658`):

- P1 up in 20 s: RTF 1.00, lowstate at 199.5 Hz, gt.pose at 49.9 Hz, camera 640x480.
- The deploy printed `Init Done` after 109 s. That start included the one-time TensorRT engine build (the planner
  engine is 747 MB). Later starts reuse the cached engines; `sonic_deploy.md` gives 32-36 s for those.
- Stand with the band released: pelvis z 0.775-0.794.
- `m1_drive_test`: all_pass=True. E2_stand, E3_turn90 (yaw error 0.5 deg), E3_walk_forward (3.41 m), E3_strafe,
  E3_stop (0.98 s), E3_goto (3 rooms), E4_sonic_walks. Output: `outputs/m1/devbox-drive/`.
- Nav2 (`nav2/m1_hook.sh up 0 dev-m1`): ready in 3.0 s on ROS domain 42. `go_to(4.13, 6.08, backend="nav2")`
  succeeded in 23 s with no A* fallback.
- Viz: HTTP 200. Over WebSocket, 14.9 fps head camera, 18 ms p50 latency, and telemetry.

Only one M1 stack can run per box. The deploy's DDS domain 0 on `lo` is hard-coded, so the dev box and the main box
never interfere with each other, but two stacks on one box would. Before starting a stack on the dev box, check that
the Arena spike is idle (`sudo docker stats --no-stream`, `nvidia-smi`): the wall-clock deploy falls under CPU
contention (`docs/walk_diagnosis.md`).

## 4. Disk usage (dev box, after the install)

| Path | Size | What |
|---|---|---|
| `/work/repos` | 34 GB | Isaac-GR00T incl. `.venv` 16, WBC incl. `.venv_sim`/deploy/ONNX/engines 14, Prompt2Policy 4.6, molmospaces 0.2, IsaacLab 0.1 |
| `/work/envs` | 21 GB | `isaaclab` (Isaac Sim 5.1 pip), `p2p-node` |
| `/work/hf-cache` | 13 GB | GR00T-N1.7-3B 6.5, Cosmos-Reason2-2B 4.9, GEAR-SONIC 0.45, molmospaces manifests |
| `/work/.cache` | 14 GB | uv + pip caches (hardlinked into the venvs, so pruning frees little) |
| `/work/opt` | 12 GB | TensorRT 10.13, CUDA 12.9, CycloneDDS |
| `/work/downloads` | 6.5 GB | TensorRT tarball (deletable; step 4 re-downloads it if the tree is missing) |
| `/work/checkpoints` + `/work/models` | 7 GB | community ckpt + dataset |
| `/work/assets` | 3.1 GB | MolmoSpaces USDs |
| `/opt/ros` | 0.6 GB | ROS 2 Jazzy + Nav2 |
| `/work/arena` + `/work/docker` | 46 + 57 GB | **Arena spike (dev box only; left untouched)** |
| Total used / free | 206 / 182 GB | before this install: 113 GB used |

## 5. Known differences from the main box

1. **Arena spike (dev box only).** `/work/arena` holds IsaacLab-Arena, gr00t_n16/n17, models and eval output.
   The docker data-root is `/work/docker`, with the images `isaaclab_arena:latest` and
   `nvcr.io/nvidia/isaac-sim:6.0.0-dev2` and the container `arena`. The HF cache also holds two small
   `GN1x-Tuned-Arena-*` refs. This was not touched.
2. **No `/work/worldline-g1/.venv-rt` on the dev box.** On the main box it is the laptop's macOS venv, copied
   there by `sync_wl.sh`, and it does not work on Linux (its `bin/python` points to a macOS interpreter). The
   runtime venv lives on the laptop. The next `sync_wl.sh push` to the dev box copies it there too (Known issues 2).
3. **Run evidence is not copied.** Main's `/work/outputs` (lab runs) and `/work/worldline-g1/outputs` (1.8 GB) stay
   on main. The dev box has only its own validation runs: `outputs/m1/{house,stack-*,devbox-drive,nav2}`, and lab
   01/03/04 outputs, including one SONIC eval and one synthetic-client result. A lab `check.sh` item that grades a
   saved run fails or skips on the dev box until that run is done there.
4. **Generated files differ only in timestamps or builds.** The G1 USD `config.yaml` header has a different
   generation date (the asset hash and URDF are identical). The TensorRT engines are built per box, so their bytes
   differ slightly.
5. **Agent scratch directories exist only on main** (`/work/wl-nav2fix*`, `/work/wl-nav2vfy2`, `/work/walkdiag`).
   They are not installs.
6. **Neither box has lab 02's `.venv_inference` or its extra apt tools** (x11vnc, xdotool, cmake-format). If
   needed, run `02_groot_g1_isaac_closed_loop/setup.sh` on the box. Note that it patches the WBC working tree
   (PR 258 patch).
7. **Disk size.** The dev box has 387 GB; main has 484 GB.

## 6. Known issues found while building it (not fixed here)

1. **Plaintext keys in a main-box log.** `/work/logs/wl/isaac-cdds.log` (mode 664) contains the full
   `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` and `HF_TOKEN` values. The log was written with `bash -x` around
   `source /etc/profile.d/ludo.sh`, which sources `secrets.env`. Delete the log and rotate the three keys. Never
   trace (`set -x`) a script that sources the profile.
2. **`sync_wl.sh push` copies the laptop's `.venv-rt`.** It excludes only `.venv`, so the macOS venv is copied
   (76 MB, about 5,650 files). The copy is broken on the box, and a first push was on track to take more than
   20 minutes over the laptop uplink (22 of 76 MB after 10 minutes), compared with about 5 minutes for the code alone. Fix: `--exclude '.venv*' --exclude .pytest_cache`.
   For this install, the code was pushed with the same rsync command plus `--exclude .venv-rt`.
3. **`sync.sh push` copies the per-box ssh state.** It excludes `.state` but not `.state-<name>/`, so
   `00_infra/.state-ludo-g1-arena/` (IP, ssh_config, known_hosts) goes to `/work/prep` on every box.
4. **Rebuilds drift.** `viz/box.sh setup`, `install_mujoco_sim.sh`, `pip install pyzmq`, and the unpinned
   extras resolved by lab 04 pick today's versions. On 2026-09-29 that meant msgpack 1.2.3 and regex 2026.9.29,
   while main has 1.2.2 and 2026.9.10. A constraints file taken from main's freezes would make rebuilds exact.
5. **Three installs exist only as notes.** There is no script for the CycloneDDS + unitree_sdk2py install (only
   main's logs), the MolmoSpaces install (`docs/scenes.md`), or the body `.venv` (only main's freeze).
6. **apt runs restart services.** The apt runs in `build_deploy.sh` and `install_ros.sh` trigger Ubuntu's
   needrestart, which restarts ssh, nvidia-persistenced, fwupd, rpcbind, polkit and atop in the middle of the run.
   Running processes survive, but avoid these installs on a box that is serving a live demo.

## 7. Rebuilding the dev box from scratch (order matters)

1. Base: infra bootstrap, then `secrets.sh`, `sync.sh push`, and `sync_wl.sh push` (with the Known issues 2
   exclude).
2. Labs (can run in parallel, `nice -n 10`): the `setup.sh` of `04_g1_sonic_isaaclab_rl`, `01_groot_g1_open_loop`
   and `03_prompt2policy_mujoco`.
3. Deploy: `bash /work/worldline-g1/sonic/build_deploy.sh`. This must run **before** ROS exists, or with its
   ROS-strip, which it does itself. It builds `.venv_sim` too.
4. DDS: build CycloneDDS 0.10.2 with `-DCMAKE_BUILD_TYPE=Release "-DCMAKE_C_FLAGS=-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"`
   `-DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SSL=OFF` into `/work/opt/cyclonedds-0.10.2/install`. Then,
   with `CYCLONEDDS_HOME` set to that `install/` dir, run `pip install --no-binary cyclonedds cyclonedds==0.10.2 --no-deps`,
   `pyzmq --no-deps`, and `-e $WBC/external_dependencies/unitree_sdk2_python --no-deps` into `/work/envs/isaaclab`.
5. Scenes: install MolmoSpaces as in `docs/scenes.md`, run
   `python -m scenes.download procthor-train-40 procthor-train-15 procthor-train-38 procthor-train-59 ithor-FloorPlan10`,
   then `bash scenes/run_house_test.sh <same ids>`.
6. `python -m sim_isaac.g1_asset --build`. Create `.venv` from main's freeze, and run `viz/box.sh setup`.
7. `bash nav2/install_ros.sh`. Last, then re-check the deploy with `STEPS=8 bash sonic/build_deploy.sh`, which fails
   if the binary links `/opt/ros`.
8. Pin msgpack 1.2.2 and regex 2026.9.10 as in §2. Diff `uv pip freeze` of every venv and `dpkg-query -W` against main.

The scripts used for steps 4-5 on this box: `/work/logs/dev/scripts/dev_dds_scenes.sh`. Logs are in
`/work/logs/dev/*.log`.
