# GR00T serving (P4): the N1.7 G1 checkpoint on the dev box, its contract, and the runtime client

Status: **M2b wave 1, 2026-09-29** (owner `groot_srv`), **updated in wave 2** (owner `ops-groot`). The server, the
runtime client (`groot/`), the open-loop check and the link script are built and were run on the dev box
`ludo-g1-arena`; in wave 2 the OD3 link runs from the main box (§5.1) and the GR00T stance is derived (§8). Numbers
below name the run that produced them. Everything GR00T here is labelled **experimental** (owner decision, PLAN §0.8): an off-the-shelf checkpoint
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
| Numbers (§6) | dev-box local `get_action` p50/p95 144/148 ms (640x480); first call 974 ms; VRAM 6.5-6.7 GB; open-loop MSE vs recorded demos 5-17x below a hold-state baseline, and wiring faults 2.4-11x worse; 0 of 515,200 targets outside the URDF limits (3 runs); laptop via SSH tunnel p50 741 ms; the dataset's own sentence beats Arena's eval prompt on the arms open-loop |

## 1. Running it

```bash
# laptop: push the code, then on the dev box
WL_ROOT=<worktree> BREV_NAME=ludo-g1-arena ludo_robotics_prep_g1/00_infra/sync_wl.sh push
bash scripts/groot_server.sh start [--port 5550] [--ckpt DIR] [--gpu 0] [--warm]   # tmux groot-server
bash scripts/groot_server.sh status | ping | stop
python -m groot.policy_client ping --endpoint tcp://127.0.0.1:5550                 # any py3.11 venv with the deps
```

- `start` refuses a busy port, checks the Isaac-GR00T commit and the checkpoint files, starts the server in tmux
  (online: `HF_HUB_OFFLINE=1` breaks the start, §6.1; the files themselves come from the cache in `/work/hf-cache`
  and `HF_TOKEN` from `/etc/profile.d/ludo.sh`), waits for `ping`, checks that the socket is bound to
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
  `"Pick up the apple from the shelf and place it onto the plate on the same shelf next to it."` (the only task; all
  10 episodes the open-loop check loaded carry annotation 3), and the model card (`$CKPT/README.md`,
  "Post-Training Dataset") names this dataset as the training data.
- Open loop, the dataset sentence reproduces the demos' arm motion better in all three runs (§6.2.1), so
  **`groot.obs.DEFAULT_PROMPT` = the dataset sentence**; `ARENA_PROMPT` and `DATASET_PROMPT` hold the two strings.

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
| Payload | request 922,862 bytes (the raw 640x480x3 frame dominates), reply 6,792 bytes (§6.4) |

**Timeouts.** A REQ socket that sent a request and got no reply is wedged: the next `send` fails. Upstream's client
recreates the socket on `zmq.Again` (:227-235). `groot.policy_client.PolicyClient` does the same on every timeout or
ZMQ error (LINGER 0) and raises `PolicyTimeout`; the next call works (`test_timeout_recreates_socket_then_recovers`).
The server does not know the client gave up: it finishes the old request (its reply is dropped by ZMQ), so the next
call can wait up to one extra inference. Default timeout 1.5 s (design §5.2). One client per thread.

## 4. Runtime API (`groot/`, for owner `groot_rt`)

```python
from groot.policy_client import PolicyClient, PolicyTimeout, PolicyError
from groot.obs import build_observation, DEFAULT_PROMPT     # the dataset sentence (§2.6, §6.2.1)
from groot.actions import ArmChunk, to_arm_chunk, clamp_stats

client = PolicyClient("tcp://127.0.0.1:5550", timeout_s=1.5)   # .ping() -> bool, .get_action(obs) -> dict, .close()
obs = build_observation(ego_rgb_uint8_480x640x3,                 # P1 ego_view frame (OD1 camera)
                        g1_debug["body_q"],                      # 29, MuJoCo order
                        g1_debug["left_hand_q"], g1_debug["right_hand_q"],   # 7 + 7, Dex3 order
                        DEFAULT_PROMPT)
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

Proposed skill registry entry (config/skills.yaml is owned by world/runtime; request in the wave-1 result). **As
built:** `config/skills.yaml` has `groot.pick.apple.arena_static_experimental.v0` (apple) and
`groot.pick.any.arena_static_experimental.v0` (any pickupable, the same sentence with the label swapped), status
`available` (SkillSpec has no `off_the_shelf`), label `experimental`, prompt = `groot.obs.DEFAULT_PROMPT` (the
integrator switched it from Arena's string after §6.2.1):

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
    prompt_template: "Pick up the apple from the shelf and place it onto the plate on the same shelf next to it."
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

### 5.1 As run on the main box (wave 2, 2026-09-29, owner ops-groot) [m]

| Step | What ran | Result |
|---|---|---|
| 1 | dev, under the dev stack lock: `groot_server.sh start --warm` | ready in 18.4 s, VRAM 6522 → 6656 MiB after the warm-up call (731 ms), bound 127.0.0.1:5550 (`/work/logs/groot/server-5550.json`, started 10:18:37Z) |
| 2 | main: `groot_link.sh keygen` | `~/.ssh/groot_link_ed25519` (comment `groot-link@computeinstance-e00d6jcj272tzcyrbp`) |
| 3 | main: `curl -s ifconfig.me` | the main box's public IP (`<MAIN_IP>`; IPs stay in the laptop's gitignored state files). The private addresses do not route between the boxes (no route between their 10.0.0.x addresses), so the link uses the public IPs; main → dev ICMP RTT 0.2-1.1 ms |
| 4 | dev: `install-key '<pub>' --from <MAIN_IP>` | one line in `~/.ssh/authorized_keys2`: `from="<MAIN_IP>",restrict,port-forwarding,permitopen="127.0.0.1:5550",command="echo groot-link: port-forward only; exit 1" ssh-ed25519 … groot-link`; `authorized_keys` untouched |
| 5 | host key pinned | the laptop's `.state-ludo-g1-arena/known_hosts` line for `<DEV_IP>` copied to main's `~/.ssh/groot_link_known_hosts` (no trust-on-first-use) |
| 6 | main: `groot_link.sh up --host <DEV_IP>` | autossh is not installed on the main box either, so the reconnect loop runs (tmux `groot-link`); "PolicyServer answers through the link"; `check` ping 7.1 ms, `status` ping 2.2 ms |
| 7a | restrictions, from the main box with the link key (`StrictHostKeyChecking yes`) | `ssh … id -un` → `groot-link: port-forward only` (the forced command); `ssh -tt` → `PTY allocation request failed`; `-L …:127.0.0.1:22` and `-L …:127.0.0.1:5551` → `administratively prohibited: open failed` |
| 7b | reconnect: kill the link's ssh client | the loop logged `ssh exited …; reconnect in 2 s`, the ping answered again 2.24 s after the kill |
| 7c | main → dev latency with the real request | §6.7 |

Two things the steps above did not cover, found while doing them:

- Killing the tmux loop's own shell (not the ssh client) ends the link for good: the loop is gone. `groot_link.sh
  ensure` (new) pings, and when the link is down and no tmux session is left, runs `up` again with the host of the
  last `up` (kept in `/work/logs/groot/link-5550.env`). `scripts/m2_up.sh --profile full` runs `ensure` instead of
  `check`.
- `up` needs `--host` only the first time; afterwards `up` and `ensure` reuse the recorded host.

### 5.2 New IPs after a Brev restart (M2b finish, 2026-09-29, owner groot-main) [m]

A stop/start of the Brev instances gives both boxes new public IPs, so the dev box's `from="<MAIN_IP>"` and the main
box's recorded host and pinned host key are stale. The key pair itself survives (it is on the main box's disk). Re-key
in three commands (the IPs are in the laptop's gitignored `00_infra/.state*/instance.env`; nothing here commits them):

```bash
cd ludo_robotics_prep_g1/00_infra
MAIN_IP=$(sed -n 's/^BREV_IP=//p' .state/instance.env); DEV_IP=$(sed -n 's/^BREV_IP=//p' .state-ludo-g1-arena/instance.env)
PUB=$(BREV_NAME=ludo-g1-brev2 ./ssh.sh 'cat ~/.ssh/groot_link_ed25519.pub')
HK=$(awk -v ip="$DEV_IP" '$1==ip && $2=="ssh-ed25519"{print $2" "$3}' .state-ludo-g1-arena/known_hosts)
BREV_NAME=ludo-g1-arena ./ssh.sh "cd /work/worldline-g1 && bash scripts/groot_link.sh install-key '$PUB' --from $MAIN_IP"
BREV_NAME=ludo-g1-brev2 ./ssh.sh "cd /work/worldline-g1 && bash scripts/groot_link.sh pin-host --host $DEV_IP '$HK' && bash scripts/groot_link.sh up --host $DEV_IP"
```

`install-key` replaces the key's old line (one line stays in `authorized_keys2`); `pin-host` writes the dev host key
the laptop already trusts (no trust-on-first-use); `up --host` replaces a running link that goes to another host.
As run after today's restart (main box, `outputs/m2b_finish/gmain/link-20260929-165831.log` on the box):

| Step | Result |
|---|---|
| dev: `groot_server.sh start --warm` (dev lock) | ready in 111.6 s (a cold page cache after the restart; 18.4 s in wave 2), VRAM 6522 → 6656 MiB, first `get_action` 1661 ms, bound 127.0.0.1:5550 |
| dev: `install-key … --from <main>` | "key already in authorized_keys2; replacing its line": 1 line, `from=` the new main IP |
| main: `pin-host`, `up --host <dev>` | "PolicyServer answers through the link"; `check` ping 9.2 ms; the tunnel's ssh on CPUs 4-15, nice 5 (never the deploy's 0-3) |
| restrictions (link key, `StrictHostKeyChecking yes`) | exec → `groot-link: port-forward only`; `ssh -tt` → `PTY allocation request failed`; `-L …:127.0.0.1:22` → `administratively prohibited: open failed` |
| `ensure` with the link down and its tmux loop gone | brought up with the last host, ping ok, 4.03 s |
| kill the link's ssh client | the loop reconnected; ping answered 2.24 s after the kill |

A too-wide `pgrep -f` in the first reconnect check matched the loop's own shell and ended the link for good (the known
weakness of the loop, §5.1); `ensure` (which `m2_up.sh --profile full` runs before and after the stack) is the
recovery, and the check now kills only the ssh client (`pgrep -x ssh` + its `-L` argument, as `groot_link.sh down`
does).

## 6. Measurements (wave 1, dev box)

Run `wave1-20260929T064355Z` (dev box, stack lock held, no other GPU job: `pre_state.txt` shows 0 MiB used before
the start), artifacts in `outputs/m2b_wave1/groot/wave1-20260929T064355Z/` (box copy `/work/groot/eval/`). Every
number below is from that run unless another run is named.

### 6.1 Server [m]

| | |
|---|---|
| Start to first `ping` | 18.2 s (`groot_server.sh start`; one start, page-cache state not controlled) |
| Bind | `127.0.0.1:5550` only (`ss -ltn`, checked by the script) |
| VRAM | 6522 MiB after load, 6652 MiB after the first `get_action`, 6680 MiB after 377 calls (`nvidia-smi` per pid) |
| First `get_action` | 974 ms (one call, synthetic 640x480 frame, right after start; the Arena spike measured 3.8 s on its first call, `docs/arena_vs_sonic.md` §2.5) |
| Live contract tests | `pytest -m box tests/groot/test_groot_live.py`: 3 passed (modality config = the Arena G1 schema with ABSOLUTE actions; action shapes (1, 40, D) f32; a float64 state is rejected by the strict server and the client stays usable) |
| Failed first attempt | with `HF_HUB_OFFLINE=1` the server died at start: transformers 4.57.3 calls the Hub API for `nvidia/Cosmos-Reason2-2B` (`tokenization_utils_base.py:2432`, `is_base_mistral`) although every file is cached. The script now defaults to online (`HF_TOKEN` from the profile); log in `/work/groot/eval/failed-offline-20260929T064322Z/` |

### 6.2 Open loop vs the recorded demos [m]

Episodes 0-9 of `nvidia/Arena-G1-Static-PickNPlace-Task` (154-198 rows each, 50 Hz), one query every 20 rows (94
queries per variant), each compared with the recorded `action` rows t..t+39 (`python -m groot.open_loop`,
`open_loop/open_loop.json`; plots `open_loop/open_loop_ep00N.png`). Every observation went through the runtime path:
LeRobot 43 → body q 29 (MuJoCo) + Dex3 hands → `build_observation` → `PolicyClient` → `to_arm_chunk`. These are the
checkpoint's **training** episodes (its card names the dataset), so this checks plumbing, not skill.

MSE in rad² (nmse = in the model's normalized units, (q99 − q01)/2 = 1; the right hand's band is degenerate):

| Variant | left_arm | right_arm | left_hand | left_arm nmse | right_arm nmse | left_hand nmse |
|---|---|---|---|---|---|---|
| **main** (Arena prompt, correct orders) | **0.0100** | **0.0014** | **0.0170** | 0.031 | 0.016 | 0.104 |
| prompt_dataset (tasks.jsonl sentence) | 0.0078 | 0.0008 | 0.0215 | 0.024 | 0.008 | 0.131 |
| baseline_hold (no model: hold the current state) | 0.0527 | 0.0243 | 0.1718 | 0.183 | 0.344 | 1.038 |
| neg_swap_arms (left/right arm states swapped) | 0.0242 | 0.0149 | 0.0480 | 0.080 | 0.180 | 0.293 |
| neg_hand_order (hands not reordered from Dex3) | 0.0165 | 0.0020 | 0.0477 | 0.052 | 0.021 | 0.291 |

- The model beats the do-nothing baseline by 5x (left arm), 17x (right arm) and 10x (left hand) in MSE, on every
  one of the 10 episodes (per-episode table in `open_loop.json`).
- Both wiring faults make it clearly worse (swapped arms: 2.4x left arm, 11x right arm; a missed Dex3 reorder: 2.8x
  left hand), so the correct orders are the ones that fit the checkpoint. `right_hand` predictions are exactly 0
  (the degenerate band, §2.5), `waist` exactly 0, `navigate_command` within [-0.075, 0] and `base_height_command`
  within [0.720, 0.815] (`dropped_ranges`): all dropped by `to_arm_chunk`.
- MAE (main): left arm 0.056 rad, right arm 0.018 rad, left hand 0.021 rad.
- Prompt: see §6.2.1.

### 6.2.1 Which instruction, and how repeatable (repeat run) [m]

Run `repeat-20260929T071942Z` (stack lock held): the same 10 episodes, one query every **10** rows (183 queries per
variant), done twice (`a`, `b`) with the Arena prompt (`main`) and the tasks.jsonl sentence (`prompt_dataset`).

| Run | Prompt | left_arm | right_arm | left_hand | nmse left / right arm, left hand |
|---|---|---|---|---|---|
| wave1 (stride 20) | Arena "move the apple to the plate" | 0.0100 | 0.0014 | 0.0170 | 0.031 / 0.016 / 0.104 |
| wave1 (stride 20) | dataset sentence | 0.0078 | 0.0008 | 0.0215 | 0.024 / 0.008 / 0.131 |
| a (stride 10) | Arena | 0.0096 | 0.0013 | 0.0194 | 0.031 / 0.012 / 0.118 |
| a (stride 10) | dataset | 0.0073 | 0.0008 | 0.0191 | 0.022 / 0.008 / 0.117 |
| b (stride 10) | Arena | 0.0099 | 0.0013 | 0.0212 | 0.032 / 0.012 / 0.130 |
| b (stride 10) | dataset | 0.0074 | 0.0007 | 0.0166 | 0.023 / 0.008 / 0.102 |

- **The dataset sentence fits the demos better on the arms in all three runs** (left arm about 23 % and right arm
  about 40 % lower MSE); the left hand is within run-to-run noise (it goes either way). Per episode and run, the Arena
  prompt had the lower summed MSE in 13 of 30 cases. This agrees with the model card, which names the HF dataset as
  the training data, whose `tasks.jsonl` carries the long sentence (§2.6). `groot.obs.DEFAULT_PROMPT` is therefore
  the dataset sentence; `ARENA_PROMPT` stays available. Neither has been compared in closed loop on our robot.
- **Outputs are stochastic.** Identical inputs in runs `a` and `b` give different errors (left-arm MSE 0.0096 vs
  0.0099 overall, up to 30 % apart per episode: episode 0 0.0038 vs 0.0029), as expected from the flow-matching
  head's sampled noise (`num_inference_timesteps: 4`, `$CKPT/config.json`).
- Latency in these runs: p50 146.0 / 145.0 ms, p95 150.0 / 149.1 ms (n = 183 each); 0 of 204,960 targets clamped in
  each run; 0 timeouts in 734 calls.

### 6.3 Mapping to SONIC and clamping [m]

- `to_arm_chunk` on every main-variant chunk (94 × 40 rows × 28 executed values = 105,280 targets); the run asserts
  that the SONIC upper body carries exactly GR00T's arm values by name and that the hands round-trip through Dex3.
- **Clamped fraction vs the URDF limits: 0 of 105,280 (0.0), at margin 0 and at the body's 0.02 rad**, limits read
  from `/work/worldline-g1/assets/g1/g1_sonic_dex3.urdf` on the box (the URDF P1 simulates; `groot.urdf.load_limits`).
  This is structural, not luck: outputs are clipped to the checkpoint's [q01, q99] band (§2.5), which lies inside
  the limits.
- **Within-chunk jitter.** 8 of the 94 chunks (9 %) contain a left-arm step larger than 0.12 rad per 20 ms (the body
  `arm` op's 6 rad/s slew limit); the largest is 0.605 rad; 1.3 % of all (step, joint) values exceed it. The plots
  show these as high-frequency bursts around the grasp (e.g. episode 2, 1.6-2.4 s). The recorded teleop actions
  themselves step up to 0.10-0.22 rad per row. Expect the body's slew limiter to engage on such chunks (B.8 reports
  `slew_frac`); a low-pass or a cross-fade over more ticks is the knob if it disturbs SONIC.

### 6.4 Latency [m]

| Where | ping p50 / p95 | get_action p50 / p95 (max) | n | Source |
|---|---|---|---|---|
| dev box, local (640x480 frame from ep 0, row 60) | 0.08 / 0.11 ms | **144.2 / 148.2 ms** (149.0) | 100 | `bench_local.json` |
| dev box, during the open loop | – | 144.4 / 148.7 ms (154.1) | 94 | `open_loop.json` |
| laptop → dev box through `00_infra/tunnel.sh` (SSH over the internet) | 188.7 / 193.3 ms | 741 / 2528 ms (3620) | 20 | `bench_laptop_tunnel.json` |

Request 922,862 bytes (the raw frame), reply 6,792 bytes. The laptop sample is dominated by uploading 0.92 MB per
call over the laptop's uplink; it is not a model of the main→dev link (same cloud region, wave 2 measures it, §5
step 7). No call failed or timed out in any run (client stats: 0 timeouts, 0 socket resets).

### 6.5 Joint orders vs P3's `body/joint_map.py` [m]

`test_orders_agree_with_body_joint_map` passed on the dev box against its `/work/worldline-g1/body/joint_map.py`
(md5 `6c96c403…`, identical to the main checkout's file at the time) and on the laptop against the main checkout's
committed file (`GROOT_JOINT_MAP_FILE=…/worldline-g1/body/joint_map.py`, after commit `97b76fe`). MuJoCo order,
SONIC wire order, mj17, Dex3 order per hand, stand pose and the 43 URDF limits all agree; no disagreement was found.
The one semantic difference is intended: `joint_map.DEX3_CLOSED` is the deploy's fist (`hand_closure`), while this
checkpoint's closed hand is Arena's pose (index -0.6/-1.2, middle -0.6/-1.2, thumb_1 0.7, thumb_2 0.7). GR00T's hand
targets are used as they come; nothing should substitute `hand_closure` for them.

### 6.6 Link restrictions (`groot_link.sh selftest`, dev box) [m]

```
PASS  forward 127.0.0.1:15550 -> 127.0.0.1:5550 answers ping
PASS  forward to 127.0.0.1:22 refused (sshd: administratively prohibited: open failed)
PASS  exec request gets the forced command ('groot-link: port-forward only')
PASS  pty refused
PASS  selftest key removed
```

Loopback run of the production commands on the dev box (`outputs/m2b_wave1/groot/link_uptest.log`; a temp key
installed with `install-key --from 127.0.0.1`, `up --host 127.0.0.1 --local-port 25550`): `up` reported the
PolicyServer answering through the link, `check` pinged in 4.9 ms; after the tunnel's ssh was killed, the reconnect
loop restored it (a ping 1 s later succeeded, log: `ssh exited …; reconnect in 2 s`); `down` removed it (ping then
failed as expected); `remove-key` left no test line. The main box was not touched in wave 1; the main→dev tunnel
itself is untested (§7).

## 7. Open items

1. **Wave 2 (OD3):** the main→dev tunnel is up and checked (§5.1; autossh is on neither box, the reconnect loop runs).
   **Not measured yet:** the main→dev `get_action` latency with the 0.92 MB request (`groot.bench --n 40 --obs-npz
   /work/groot/eval/obs_ep0_f60.npz`, the npz is on the main box) and SONIC's timing gate with the GR00T client active
   (`tools/groot_timing_gate.py`). Both boxes were stopped (Brev: STOPPED) from about 11:40Z before the stack locks
   came free, so neither ran; the commands are in `docs/bringup.md` §6.1.
2. **Camera (OD1):** the checkpoint saw Arena's head camera only. Open-loop numbers above use Arena's own frames;
   P1's `ego_view` must match that mount and FOV or the visual prior is lost.
3. **Jitter:** 9 % of chunks carry steps above the body's slew limit (§6.3); decide in G2 whether to smooth.
4. **Stochastic chunks:** the flow-matching head samples noise (4 steps), so two calls on one observation differ;
   the runtime must not assume repeatability (§6.2.1 gives the size).
7. **Prompt in closed loop:** `DEFAULT_PROMPT` (dataset sentence) is the open-loop winner; the closed-loop comparison
   with Arena's string is for G2.
5. **Licence (OD2):** card vs LICENCE file (§2.8).
6. `pyproject.toml` `testpaths` does not list `tests/groot` (shared file; request filed); run it explicitly:
   `.venv-rt/bin/python -m pytest tests/groot` (the `box` tests need `-m box` and a server).

## 8. The GR00T stance (wave 2, W2.5)

Arena's training frames show the apple on the surface in front of the robot, left of centre, in the lower part of
the head camera (episode 0, frame 60 of the HF dataset: the apple's centre near v = 330 of 480, u = 105 of 640;
`obs_ep0_f60.npz`). P1's `ego_view` is that camera (p1_m2b.md §5.1: pelvis + (0.045, 0, 0.353) m at zero waist, i.e.
1.14 m at SONIC's 0.787 m stand, 35° down, fx = fy = 458.12 px), so where a target lands in the image depends only
on its height and its offset from the pelvis. The pinhole model (the numbers below; `v` = image row of the target's
centre, the bar is v >= 160: below the upper third):

| Target (centre height) | f 0.25 | f 0.30 | f 0.35 | f 0.40 | f 0.45 |
|---|---|---|---|---|---|
| potato on the 0.78 m dining table (0.81 m) | 434 | 381 | 338 | 302 | 272 |
| mug on the 0.94 m counter (0.98 m) | 262 | 215 | 180 | 152 | 129 |
| mug on the 0.97 m dresser (1.01 m) | 216 | 173 | 141 | 116 | 96 |
| pepper shaker on the counter (1.04 m) | 175 | 136 | 107 | 85 | 68 |

(f = forward of the pelvis, lateral 0.10 m left; the lateral offset moves the target sideways only: u = 205 / 167 /
158 at f 0.30, and at 0.20 m left a counter-height target leaves the image, u < 20.) So a low object on any of the
three surfaces sits in the lower two-thirds at f <= 0.30 m (the dresser) to 0.35 m (the counter), with little lateral
offset; a tall one (an alarm clock or a pepper shaker, centre >= 1.04 m) does not at any stance the robot can take.
The world's reach sphere agrees (config/g1.yaml after R.7: `arm_reach_m` 0.405 from the shoulder sphere's centre;
a table grasp point 0.88 m high at f 0.35 is outside it, at f 0.30 inside). The model is `groot/stance.py`;
`tests/services/test_groot_arms_stance.py` pins it.

**Encoded** (`config/skills.yaml`): `groot.pick.apple.arena_static_experimental.v0` has `stance {stand_off_m: 0.30,
lateral_m: 0.10, yaw_to_object: 0.0, tol: [0.05, 25.0]}`, labelled INTERIM until §8.1 has run:

- Only the apple skill (the checkpoint's own object, OD4). `services/reachability.py` treats a skill's stance as
  strict: the object must sit at that offset, so check_reachability needs a free spot 0.30 m from it (with the
  furniture clearance a reach stance keeps), i.e. an object within about 0.10 m of the front edge. On the `any` skill
  that would make every deeper pickable unreachable in `full` before `groot_then_script` could fall back to the
  scripted pick (with the stance on the `any` skill too, 5 tests of the default suite failed that way on HEAD
  ca138da + this change: groot_then_script and the service tests pick an alarm clock that sits deeper). Request to world: use a
  GR00T stance when a free spot satisfies it, else the next healthy candidate's window; then the `any` skill gets it.
- `tol_deg` 25: `in_window` compares the heading with the bearing to the object when the lateral offset is above
  `tol_m`; at this stance the bearing is atan2(0.10, 0.30) = 18.4° off the heading by construction, so 5° would
  refuse the stance `find_stance` just produced. Request to world: compare with bearing − atan2(lateral_m,
  stand_off_m) − yaw_to_object.
- For a demo on the GR00T stance the object is staged near the edge (P1 `move_object`, test-only):
  `tools/groot_episode.py --stage-at <surface> --depth 0.07`.

### 8.1 Live probe (not run)

`tools/groot_stance_probe.py` walks to each surface, stages a low object 0.07 m behind the front edge, puts the pelvis
at each (forward, left) of a grid with the body's `approach` op and reads P1's instance segmentation of `ego_view`
(pixels, bbox, the bbox centre against v = 160), saving each view with the model's prediction next to it. It was
exercised on the in-process fakes only (`tools/fake_p1.FakeP1`, H40: the table at f 0.35 / l 0.15 gave v 327.5 vs
the model's 338). The live run on the main box did not happen: the main stack lock was held by another owner from
10:52Z and both boxes were stopped from about 11:40Z. Command: `docs/bringup.md` §6.1.
