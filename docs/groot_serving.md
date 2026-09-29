# GR00T serving (P4): the N1.7 G1 checkpoint on the dev box, its contract, and the runtime client

Status: **M2b wave 1, 2026-09-29.** Owner `groot_srv`. The server, the runtime client (`groot/`), the open-loop check
and the link script are built and were run on the dev box `ludo-g1-arena`; numbers below name the run that produced
them. Everything GR00T here is labelled **experimental** (owner decision, PLAN §0.8): an off-the-shelf checkpoint
trained on one apple-to-plate scene; in a house it is expected to move the arms plausibly and usually not grasp.

Tags as in `docs/groot_arms_design.md`: **[v]** read in code or files (path:line), **[m]** measured (run named),
**[u]** unverified.

---

## 0. Summary

| | |
|---|---|
| Checkpoint | `nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace` @ `7f78bebf1a90131e7304beacfcd47eb27bad16ab` (N1.7, base `nvidia/GR00T-N1.7-3B` @ `2fc962b`, backbone `nvidia/Cosmos-Reason2-2B` @ `9ce19a1`, gated, cached) |
| Server | stock `gr00t/eval/run_gr00t_server.py` of Isaac-GR00T @ `4b1dca9d` (Arena's static_apple pin), env `/work/arena/gr00t_n17/.venv` (py 3.10.21, torch 2.7.1+cu128, transformers 4.57.3, flash-attn 2.7.4.post1), reused from the Arena spike, nothing reinstalled |
| Where | **dev box** `ludo-g1-arena`, `127.0.0.1:5550` only, tmux `groot-server`, logs `/work/logs/groot/` (OD3). The main box reaches it in wave 2 through `scripts/groot_link.sh` |
| Runtime client | `groot/` (py3.11, numpy + pyzmq + msgpack; never imports `gr00t`): `PolicyClient`, `build_observation`, `to_arm_chunk` / `ArmChunk`, `joint_order` |
| Numbers | §6 (filled from the wave-1 run) |

## 1. Running it

```bash
# laptop: push the code, then on the dev box
WL_ROOT=<worktree> BREV_NAME=ludo-g1-arena ludo_robotics_prep_g1/00_infra/sync_wl.sh push
bash scripts/groot_server.sh start [--port 5550] [--ckpt DIR] [--gpu 0] [--warm]   # tmux groot-server
bash scripts/groot_server.sh status | ping | stop
python -m groot.policy_client ping --endpoint tcp://127.0.0.1:5550                 # any py3.11 venv with the deps
```

- `start` refuses a busy port, checks the Isaac-GR00T commit and the checkpoint files, starts the server in tmux
  with `HF_HUB_OFFLINE=1` (every file it needs is cached), waits for `ping`, checks that the socket is bound to
  `127.0.0.1:<port>` only, and writes `/work/logs/groot/server-<port>.json` (pid, ready time, VRAM, first-call
  time with `--warm`). `--embodiment-tag NEW_EMBODIMENT`, `--device cuda`, strict observation checks (the
  server's default).
- The server's `--modality-config-path` is **not** passed: with `--model-path` the server takes the modality config
  from the checkpoint's processor (`run_gr00t_server.py:86-91`; the flag is only read for `ReplayPolicy`, :103-121).
- Port 5555 is DCGM on Nebius boxes (docs/contracts/m1.md §0); 5550 is `GROOT_PORT` in `config/ports.env`.
- Hold the dev-box stack lock (`/work/locks/stack.d`) while benchmarking. An idle server may stay up without it;
  it holds its VRAM (§6) but uses no GPU time.

## 2. The checkpoint contract

Sources, all on the dev box: checkpoint dir `/work/arena/models/isaaclab_arena/static_apple_tutorial/gn1x_tuned_static_apple`
(= `$CKPT`), dataset `nvidia/Arena-G1-Static-PickNPlace-Task` @ `37ba80a` (CC-BY-4.0) at
`/work/groot/datasets/Arena-G1-Static-PickNPlace-Task` (`$DATA`, meta + episodes 0-9), IsaacLab-Arena
`release/0.2.1` @ `8b4a3a47` (`$ARENA`), Isaac-GR00T @ `4b1dca9d` (`$GR00T`).

### 2.1 Embodiment

`new_embodiment` (`EmbodimentTag.NEW_EMBODIMENT`), projector slot **10** (`$CKPT/embodiment_id.json`). The
modality config is `$CKPT/experiment_cfg/conf.yaml` `data.modality_configs.new_embodiment`, identical to
`$ARENA/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py:14-78` [v]. The live server
returns the same config from `get_modality_config` (`tests/groot/test_groot_live.py`, §6).

### 2.2 Modalities

| Modality | Key | Shape on the wire | Notes |
|---|---|---|---|
| video | `ego_view` | uint8 (B=1, T=1, 480, 640, 3) RGB | delta `[0]`; Arena head camera (OD1: P1 must render this view) |
| state | `left_arm` | f32 (1, 1, 7) | delta `[0]` |
| state | `right_arm` | f32 (1, 1, 7) | |
| state | `left_hand` | f32 (1, 1, 7) | GR00T hand order |
| state | `right_hand` | f32 (1, 1, 7) | GR00T hand order |
| state | `waist` | f32 (1, 1, 3) | |
| action | `left_arm`, `right_arm` | f32 (1, 40, 7) | **ABSOLUTE**, NON_EEF |
| action | `left_hand`, `right_hand` | f32 (1, 40, 7) | ABSOLUTE |
| action | `waist` | f32 (1, 40, 3) | ABSOLUTE; identically 0 in training (§2.5) |
| action | `base_height_command` | f32 (1, 40, 1) | ABSOLUTE; **dropped** by the runtime |
| action | `navigate_command` | f32 (1, 40, 3) | ABSOLUTE (vx, vy, wz); **dropped** by the runtime |
| language | `annotation.human.task_description` | `[[str]]` | delta `[0]` |

Every action config is `rep: ABSOLUTE, type: NON_EEF, format: DEFAULT` (`conf.yaml`; `final_model_config.json`
`use_relative_action: false`; `statistics.json` has an empty `relative_action`) [v]. Horizon **40** steps
(`config.json` `action_horizon: 40`, `delta_indices` 0-39) at **50 Hz** (`$DATA/lerobot/meta/info.json` `fps: 50`;
Arena's WBC steps at 50 Hz, `docs/arena_vs_sonic.md` §3), i.e. 0.8 s per chunk.

### 2.3 Joint order per key, by name

GR00T keys (`$ARENA/isaaclab_arena_gr00t/embodiments/g1/gr00t_43dof_joint_space.yaml:22-57` [v]; the same names in
`$DATA/lerobot/meta/info.json` `observation.state.names`, sliced by `meta/modality.json`):

| Key | Order |
|---|---|
| `left_arm` | left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw |
| `right_arm` | the same, `right_` |
| `left_hand` | left_hand_**index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2** |
| `right_hand` | the same, `right_hand_` |
| `waist` | waist_yaw, waist_roll, waist_pitch |
| `base_height_command` | base height [m] |
| `navigate_command` | vx [m/s], vy [m/s], wz [rad/s] (body frame) |

(`_joint` suffixes omitted.) The other orders the runtime meets, all in `groot/joint_order.py` and converted only
by name:

| Space | Order | Source |
|---|---|---|
| g1_debug `body_q` (29) | MuJoCo/Unitree: legs 12, waist 3, L arm 7, R arm 7 | `$DEPLOY/include/robot_parameters.hpp:90-130` |
| g1_debug `left/right_hand_q`, planner hands, body `arm` op hands | Dex3: **thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1** | `$WBC/gear_sonic/utils/mujoco_sim/base_sim.py:225-240` |
| SONIC planner `upper_body_position` (17) | waist 3, then L/R arms **interleaved** (L sh_pitch, R sh_pitch, L sh_roll, …) | `$DEPLOY/include/policy_parameters.hpp:81` |
| body `arm` op 17-list (`mj17`) | MuJoCo[12:29]: waist 3, L arm 7, R arm 7 | body/arm.py `_vec17` (main checkout) |
| LeRobot 43 (`observation.state`, `action`) | legs 12, waist 3, L arm 7, L hand 7 (GR00T order), R arm 7, R hand 7 | `$DATA/lerobot/meta/info.json`, `modality.json` |

The GR00T arm order equals the MuJoCo arm order; only the hands are permuted (GR00T `index_0` = Dex3 slot 5,
`thumb_0` = Dex3 slot 0). `tests/groot/test_groot_joint_order.py` pins every table to its source (golden copies)
and, when `body.joint_map` is importable (the dev box has P3's file; the laptop worktree does not), checks that
both modules agree: they do (§6.5).

### 2.4 Representation and what the runtime executes

ABSOLUTE joint targets in rad, at 50 Hz, 40 per chunk; row k belongs to `t_obs + k × 0.02 s`.
`groot.actions.to_arm_chunk` maps them to SONIC (decision (b), `docs/groot_arms_design.md` §5.2):

- arms → the 14 arm entries of the 17-D upper body, by name;
- hands → Dex3 order, both hands always;
- waist → **held** at SONIC's stand waist (0, 0, 0), not GR00T's (which is 0 anyway);
- `base_height_command`, `navigate_command`, and GR00T's `waist` → `ArmChunk.dropped` (telemetry, `groot_nav_pred`);
  never executed (`manipulate` never moves the base).

### 2.5 Normalization

- Stats: `$CKPT/statistics.json` → `new_embodiment` → `state` / `action` per key (`min, max, mean, std, q01, q99`),
  computed from its own training data (`conf.yaml` `override_pretraining_statistics: true`; the same numbers are in
  `experiment_cfg/dataset_statistics.json`). The processor uses **q01/q99** (`use_percentiles: true`,
  `processor_config.json`), maps them to [-1, 1] and clips (`clip_outliers: true`;
  `gr00t/data/state_action/state_action_processor.py:139-151, 245-246`). Decoding clips the model output to
  [-1, 1] before unnormalizing (`gr00t/data/utils.py:150`), so **every emitted target lies in [q01, q99]**.
- The server normalizes; the client sends raw rad and receives raw rad. Nothing is normalized in `groot/`.
- Action q01 / q99 (rad), from `statistics.json` [v]:

| Key | q01 | q99 |
|---|---|---|
| left_arm | -0.737, -0.206, -0.344, -0.480, -0.337, -0.588, -0.939 | -0.034, 0.663, 0.439, 0.820, 1.145, 0.945, 0.218 |
| right_arm | -0.803, -0.645, -0.436, 0.007, -0.002, -0.257, -0.127 | -0.261, -0.351, -0.236, 1.132, 0.407, 0.593, 0.347 |
| left_hand | -0.6, -1.2, -0.6, -1.2, 0, 0, 0 | 0, 0, 0, 0, 0, 0.7, 0.7 |
| right_hand | 0 × 7 | 0 × 7 |
| waist | 0, 0, 0 | 0, 0, 0 |
| base_height_command | 0.72 | 1.0 |
| navigate_command | 0, 0, -0.297 | 0, 0, 0 |

  Consequences: the **right hand is always commanded fully open** (q01 = q99 = 0: the demos only used the left hand);
  the left hand is binary between open and Arena's closed pose (index -0.6/-1.2, middle -0.6/-1.2, thumb_1 0.7,
  thumb_2 0.7), not the deploy's fist; every arm band lies inside the URDF limits even with the body's 0.02 rad
  margin, so this checkpoint cannot produce a target that the body has to clamp (pinned by
  `test_checkpoint_output_band_lies_inside_the_urdf_limits`; a fine-tuned checkpoint must be re-checked).

### 2.6 Language

- Key `annotation.human.task_description`; one string per step. The processor lower-cases it and strips punctuation
  (`formalize_language: true`; `processing_gr00t_n1d7.py:421`).
- **Training annotation.** The checkpoint was trained on `…/arena_g1_static_apple_200_dataset_locked_waist_merged/lerobot`
  (`conf.yaml` `data.datasets[0].dataset_paths`), not on the released HF copy. Its instruction is the one in Arena's
  HDF5 → LeRobot converter config, **`"move the apple to the plate"`**, task_index 3
  (`$ARENA/docs/pages/example_workflows/static_apple/step_3_policy_training.rst:93-97`); the same string is Arena's
  closed-loop eval `language_instruction` (`arena_spike/g1_static_apple_gr00t_closedloop_config.yaml`) and the ONNX
  export's fixed prompt (`$CKPT/exports/g1-static-apple-b1-480x640/onnx/leapp-0.5.2/README.md:11`).
- The released dataset's `meta/tasks.jsonl` maps the same task_index 3 to
  `"Pick up the apple from the shelf and place it onto the plate on the same shelf next to it."` (every one of its
  208 listed episodes). §6.2 compares both prompts open-loop.
- `groot.obs.ARENA_PROMPT` / `DATASET_PROMPT` hold the two strings.

### 2.7 Image

Send the full 640x480 RGB frame. The server does its own eval transform: shortest edge → 256 (INTER_AREA), centre
crop 0.95, shortest edge → 256 (`$CKPT/processor_config.json` `shortest_image_edge: 256, crop_fraction: 0.95`;
`gr00t/model/gr00t_n1d7/image_augmentations.py:479-489`). The training frames came from Arena's `robot_head_cam_rgb`
(head_link + (0.045, 0, 0.353), 35° down, HFOV 69.9°; `docs/groot_arms_design.md` §2.3). OD1: P1's `ego_view`
must match that camera.

### 2.8 Licence

- The model card (`$CKPT/README.md`) says "ready for non-commercial use only" and points to the "NVIDIA Open Model
  Agreement"; the `LICENCE` file in the same repo is the **NVIDIA Open Model License Agreement** (last modified
  2025-10-24), which says "Models are commercially usable". They contradict each other; ask NVIDIA before any
  commercial use (`docs/arena_vs_sonic.md` §5, `docs/groot_arms_design.md` §2.7, OD2).
- Backbone `nvidia/Cosmos-Reason2-2B`: NVIDIA Open Model License, gated (accepted, PLAN §0.5).
- Dataset `nvidia/Arena-G1-Static-PickNPlace-Task`: CC-BY-4.0 (its README: "ready for commercial use").
- Isaac-GR00T and IsaacLab-Arena code: Apache-2.0.

## 3. The wire (Isaac-GR00T @ 4b1dca9d, `gr00t/policy/server_client.py`)

| | |
|---|---|
| Transport | ZMQ REQ → REP, `tcp://127.0.0.1:5550`; the server binds `tcp://{host}:{port}` (:84-86) and serves one request at a time (:132-164) |
| Encoding | msgpack; an `np.ndarray` is `{"__ndarray_class__": true, "as_npy": <np.save bytes, allow_pickle=False>}`, a `ModalityConfig` is `{"__ModalityConfig_class__": true, "as_json": {...}}` (`MsgSerializer`, :30-60). No pickle |
| Request | `{"endpoint": name, "data": {...}}`; `data` is omitted for `ping`, `get_modality_config`, `kill` (:210-225). No `api_token` (the server has none) |
| `ping` | → `{"status": "ok", "message": "Server is running"}` (:107-111) |
| `get_action` | `data = {"observation": obs, "options": null}` → `[action, info]` (`BasePolicy.get_action` returns a tuple, `gr00t/policy/policy.py:80-105`); `action[key]` f32 (1, 40, D), `info = {}` (`gr00t_policy.py:420-423`) |
| `get_modality_config` | → `{video, state, action, language}` as plain dicts on our side |
| `reset` | `data = {"options": null}` → `{}` |
| Errors | any exception, including a failed strict observation check (`gr00t_policy.py:208-369`), replies `{"error": str}` (:159-164); the client raises `PolicyError`, the socket stays usable |
| Payload | request ≈ 0.92 MB (the raw 640x480x3 frame dominates), reply ≈ 2 KB (§6.3) |

**Timeouts.** A REQ socket that sent a request and got no reply is wedged: the next `send` fails. Upstream's client
recreates the socket on `zmq.Again` (:227-235). `groot.policy_client.PolicyClient` does the same on every timeout or
ZMQ error (LINGER 0) and raises `PolicyTimeout`; the next call works (`test_timeout_recreates_socket_then_recovers`).
The server does not know the client gave up: it finishes the old request (its reply is dropped by ZMQ), so the next
call can wait up to one extra inference. Default timeout 1.5 s (design §5.2). One client per thread.

## 4. Runtime API (`groot/`, for owner `groot_rt`)

```python
from groot.policy_client import PolicyClient, PolicyTimeout, PolicyError
from groot.obs import build_observation, ARENA_PROMPT
from groot.actions import ArmChunk, to_arm_chunk, clamp_stats

client = PolicyClient("tcp://127.0.0.1:5550", timeout_s=1.5)   # .ping() -> bool, .get_action(obs) -> dict, .close()
obs = build_observation(ego_rgb_uint8_480x640x3,                 # P1 ego_view frame (OD1 camera)
                        g1_debug["body_q"],                      # 29, MuJoCo order
                        g1_debug["left_hand_q"], g1_debug["right_hand_q"],   # 7 + 7, Dex3 order
                        ARENA_PROMPT)
t_obs = <time.monotonic() of that g1_debug>
chunk = to_arm_chunk(client.get_action(obs), t0_mono=t_obs)      # ArmChunk
chunk.upper_body        # (40, 17) SONIC wire order (interleaved) -- what the deploy consumes
chunk.upper_body_mj17   # (40, 17) waist, L arm, R arm: the order of the body `arm` op's 17-list
chunk.arm_targets_named(k)   # {joint: rad} for row k, arms only (the order-proof form of the `arm` op)
chunk.left_hand, chunk.right_hand   # (40, 7) Dex3 order
chunk.dropped           # {"waist", "base_height_command", "navigate_command"} -> telemetry only
chunk.index_at(time.monotonic())   # row to play now (P3 indexes by time)
clamp_stats(chunk, margin=0.02)    # clamped fraction vs the URDF limits
```

**Order warning.** `ArmChunk.upper_body` is SONIC's interleaved wire order, as specified for this package. The
chunk-mode `arm` message (`docs/contracts/arm_chunk.md` §1.2) therefore carries `order: "wire"` with it. The v0.5
single-target `arm` op (body/arm.py `_vec17`, main checkout) takes a 17-**list** in mj17 order, or a dict by joint
name: for that op send `upper_body_mj17[k]` or `arm_targets_named(k)`, never `upper_body[k]`, because a wire-order
vector read as mj17 puts left-arm values on right-arm joints. The waist is left to the op's default
(`waist: "ref"`, SONIC's own reference), which holds the stand waist.

Proposed skill registry entry (config/skills.yaml is owned by world/runtime; request in the wave-1 result):

```yaml
  - skill_id: groot.pick.apple.arena_static.v0
    action: pick
    object_types: [apple]
    arms: [left]                 # the checkpoint only ever moves the left hand (right-hand band is 0)
    backend: groot               # executor groot_arms
    label: experimental
    status: off_the_shelf
    embodiment_tag: NEW_EMBODIMENT
    checkpoint: nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace@7f78beb
    policy_port: 5550
    prompt_template: "move the apple to the plate"
    hand_type: dex3
    action_horizon: 40
    control_hz: 50
    max_duration_s: 25.0
    licence: "card: non-commercial; LICENCE file: NVIDIA Open Model License (contradictory, OD2)"
```

## 5. OD3 link: main box → dev box (wave 2)

`scripts/groot_link.sh` keeps `main:127.0.0.1:5550 → dev:127.0.0.1:5550` up over SSH with a dedicated ed25519 key.
On the dev box the key sits in `~/.ssh/authorized_keys2` (listed in sshd's `AuthorizedKeysFile`, checked with
`sudo sshd -T`: `authorizedkeysfile .ssh/authorized_keys .ssh/authorized_keys2`, `allowtcpforwarding yes`), so the
operator keys in `authorized_keys` are never edited. Its line:

```
from="<MAIN_IP>",restrict,port-forwarding,permitopen="127.0.0.1:5550",command="echo groot-link: port-forward only; exit 1" ssh-ed25519 AAAA… groot-link
```

`restrict` turns off pty, agent and X11 forwarding, user rc and port forwarding; `port-forwarding` turns local
forwarding back on, `permitopen` limits it to the PolicyServer, and any shell or command request runs the forced
command. The tunnel is `ssh -N -L 127.0.0.1:5550:127.0.0.1:5550` with `ExitOnForwardFailure`, `ServerAliveInterval 5`
× 3, `BatchMode`, its own known_hosts; autossh when installed (it is not on the dev box; the main box was not
checked), else a reconnect loop, in tmux `groot-link`, log `/work/logs/groot/link-5550.log`.

Wave-2 steps (in order):

1. Dev box: `bash scripts/groot_server.sh start --warm` (or confirm `status`).
2. Main box: `bash scripts/groot_link.sh keygen` → prints `ssh-ed25519 AAAA… groot-link@<host>`.
3. Main box: find its address as the dev box sees it (`curl -s ifconfig.me`, or the source IP in the dev box's
   `/var/log/auth.log` after a first refused attempt).
4. Laptop → dev box: `BREV_NAME=ludo-g1-arena 00_infra/ssh.sh "cd /work/worldline-g1 && bash scripts/groot_link.sh install-key '<pubkey line>' --from <MAIN_IP>"`.
5. Optional host-key pinning (else the first connect is trust-on-first-use): copy the dev box's line from the laptop's
   `00_infra/.state-ludo-g1-arena/known_hosts` into the main box's `~/.ssh/groot_link_known_hosts`, replacing the
   host name with the dev box IP.
6. Main box: `bash scripts/groot_link.sh up --host <DEV_IP>` (dev IP: `BREV_IP` in the laptop's
   `00_infra/.state-ludo-g1-arena/instance.env`). It prints whether the PolicyServer answers through the link.
7. Main box: `bash scripts/groot_link.sh check` (ping JSON) and `python -m groot.bench --n 30 --obs-npz <npz>`
   (the npz from §6.3) for the main→dev latency; the runtime then uses `tcp://127.0.0.1:5550` unchanged.
8. Teardown: main `groot_link.sh down`; dev `groot_link.sh remove-key`.

Wave-1 check of the restrictions (`groot_link.sh selftest`, dev box only; a throwaway key authorized with the same
options plus `from="127.0.0.1"`, a loopback tunnel, then removed): §6.6.

## 6. Measurements (wave 1, dev box)

(filled in from the run; see `outputs/m2b_wave1/groot/`)

## 7. Open items

(filled in from the run)
