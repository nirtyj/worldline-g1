"""Verification probes on top of arena_spike_policies (mounted at /spike in the Arena container).

BinProbeGr00tPolicy = TimedGr00tRemotePolicy + a scene edit applied at every env step:
  SPIKE_BIN_POSE="x,y,z"  holds the destination bin (scene entity SPIKE_BIN_NAME, default "blue_sorting_bin") at that
                          world position with zero velocity, e.g. "0,0,-5" hides it under the floor.
Question it answers: is the N1.6 loco-manip checkpoint's navigate_command conditioned on where the goal is (vision),
or does it replay the single Mimic route it was trained on (turn to -1.78 rad, walk to (-0.10, -1.11))?
Everything else (logs, videos, episode outcomes) is the parent's; the bin pose is added to <tag>_cmd_log.jsonl.

  SPIKE_WBC_RESET=1       also resets the lower-body WBC policy (HOMIE obs history + gait phase, AGILE LSTM state) from
                          the policy's reset(). G1DecoupledWBCJointAction.reset() itself only zeroes its raw actions, but
                          the env already has a reset event for this (g1_events.reset_decoupled_wbc_joint_policy), so
                          this is a control, not a fix: it did not change the outcome (probe_n16_wbcreset, 1 of 6).

Run (on the box, through run_eval.sh):
  SPIKE_RERENDER=1 POLICY_TYPE=arena_probe_policies.BinProbeGr00tPolicy SPIKE_BIN_POSE=0,0,-5 \
      bash run_eval.sh n16 probe_n16_nobin --num_episodes 3
  SPIKE_RERENDER=1 POLICY_TYPE=arena_probe_policies.BinProbeGr00tPolicy SPIKE_WBC_RESET=1 VIDEO=0 \
      bash run_eval.sh n16 probe_n16_wbcreset --num_episodes 6
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from arena_spike_policies import TimedGr00tRemotePolicy, _to_torch  # noqa: F401  (PYTHONPATH=/spike)
from isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy import Gr00tRemoteClosedloopPolicyArgs

BIN_NAME = os.environ.get("SPIKE_BIN_NAME", "blue_sorting_bin")
BIN_POSE = os.environ.get("SPIKE_BIN_POSE", "")
WBC_RESET = os.environ.get("SPIKE_WBC_RESET", "0") == "1"


class BinProbeGr00tPolicy(TimedGr00tRemotePolicy):
    name = "spike_bin_probe_gr00t_remote"

    def __init__(self, config: Gr00tRemoteClosedloopPolicyArgs):
        super().__init__(config)
        self._bin_xyz = [float(v) for v in BIN_POSE.split(",")] if BIN_POSE else None
        self._bin_warned = False
        print(f"[probe] bin {BIN_NAME} held at {self._bin_xyz}")

    def _hold_bin(self, env):
        if self._bin_xyz is None:
            return
        try:
            obj = env.unwrapped.scene[BIN_NAME]
            pose = _to_torch(obj.data.root_link_pose_w).clone()  # (N, 7) x y z qx qy qz qw (Lab 3.0)
            pose[:, 0:3] = torch.tensor(self._bin_xyz, device=pose.device) + _to_torch(
                env.unwrapped.scene.env_origins).to(pose.device)
            vel = torch.zeros((pose.shape[0], 6), device=pose.device)
            obj.write_root_pose_to_sim_index(root_pose=pose)
            obj.write_root_velocity_to_sim_index(root_velocity=vel)
            if self._n % 5 == 0:
                self._cmd_f.write(json.dumps({"step": self._n, "bin": [round(float(x), 3) for x in pose[0, :3]]}) + "\n")
        except Exception as e:  # report once, keep the eval running
            if not self._bin_warned:
                self._bin_warned = True
                keys = list(getattr(env.unwrapped.scene, "keys", lambda: [])())
                print(f"[probe] bin hold failed: {e!r}; scene keys={keys}")

    def get_action(self, env, observation):
        self._hold_bin(env)
        return super().get_action(env, observation)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if not (WBC_RESET and env_ids is not None and self._env is not None):
            return
        try:
            am = self._env.unwrapped.action_manager
            for name in am.active_terms:
                term = am.get_term(name)
                wbc = getattr(term, "wbc_policy", None)  # get_wbc_policy is a @property on the term
                if wbc is not None and hasattr(wbc, "lower_body_policy"):
                    lb = wbc.lower_body_policy
                    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
                    lb.reset(ids)
                    print(f"[probe] reset lower-body WBC state ({type(lb).__name__}) for envs {ids.tolist()}")
        except Exception as e:
            print(f"[probe] WBC reset failed: {e!r}")

    @staticmethod
    def from_args(args: argparse.Namespace) -> "BinProbeGr00tPolicy":
        return BinProbeGr00tPolicy(Gr00tRemoteClosedloopPolicyArgs.from_cli_args(args))
