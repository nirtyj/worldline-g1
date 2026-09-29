"""Dex3 hand physics in isolation: the P1 G1 asset with its root fixed in the air, no SONIC / DDS / body.

The same Dex3 drive as P1 (implicit PD, kp 1.5 / kd 0.1 = the deploy's hand gains + joint_map.DEX3_JOINT_DAMPING as
dds_bridge applies them, g1_asset's dex3 actuator group and collision filter),
the deploy's per-write clamp emulated (target = q + clip(cmd - q, +-0.25) every 20 ms, dex3_hands.hpp:114-190), the
same step / closure schedule as tools/hand_step_probe.py, and PhysX contact forces on every hand link (the robot is
spawned with contact reporting on, g1.py:204). Legs, waist and arms are held by stiff PD at a given pose, so the only
thing that can move a finger other than its drive is a contact. Writes the hand_step_probe format (run.json +
probe.npz, t = sim time) plus contacts.npz, and analyses it with hand_step_probe.analyze.

    /work/envs/isaaclab/bin/python -m sim_isaac.tools.hand_physics_probe --out DIR [--usd PATH]
        [--arm-pose outputs/.../probe1/run.json] [--filter-hands]

--partners: also record the contact force matrix against every other link of the robot (who touches whom).
--no-self-collision: spawn with articulation self-collisions off (A/B).
--dex3-body-collisions: do not filter Dex3 finger links vs the rest of the robot (g1_asset does by default; A/B).
--extra-kd: add this to the Dex3 drive damping (P1: deploy kd 0.1 + DEX3_JOINT_DAMPING 0.05; -0.05 = without it).
--filter-hands: author UsdPhysics.FilteredPairsAPI between all links of the same hand (incl. the wrist-yaw body that
carries the merged palm) before the sim starts (A/B for intra-hand collisions).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

HAND_LINKS = ("hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link", "hand_middle_0_link",
              "hand_middle_1_link", "hand_index_0_link", "hand_index_1_link")


def hand_bodies(side: str) -> list[str]:
    return [f"{side}_wrist_yaw_link"] + [f"{side}_{n}" for n in HAND_LINKS]


PARTNERS = (["pelvis", "torso_link", "waist_yaw_link", "waist_roll_link"]
            + [f"{s}_{n}_link" for s in ("left", "right") for n in (
                "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll", "shoulder_pitch", "shoulder_roll",
                "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch")]
            + [b for s in ("left", "right") for b in hand_bodies(s)])


def filter_hand_pairs(stage, root: str) -> int:
    from pxr import UsdPhysics
    n = 0
    for side in ("left", "right"):
        paths = [f"{root}/{b}" for b in hand_bodies(side)]
        for i, p in enumerate(paths):
            prim = stage.GetPrimAtPath(p)
            if not prim.IsValid():
                raise RuntimeError(f"no prim {p}")
            UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(paths[i + 1:])
            n += len(paths) - i - 1
    return n


def schedule(hold: float, closure_hold: float):
    """[(name, kind, meta, left7, right7, dur)] = hand_step_probe's live schedule."""
    from body import joint_map as bjm
    from sim_isaac.tools import hand_step_probe as hp
    out = [("open", "open", {}, list(bjm.DEX3_OPEN), list(bjm.DEX3_OPEN), 2.0)]
    for k, suf in enumerate(hp.SUF):
        base = {s: [0.0] * 7 for s in hp.SIDES}
        for s in hp.SIDES:
            base[s][k] = hp.step_base(s, k)
        out.append((f"base_{suf}", "settle", {"joint": k}, base["left"], base["right"], hold))
        for d in (+hp.STEP, 0.0, -hp.STEP, 0.0):
            goal = {s: list(base[s]) for s in hp.SIDES}
            for s in hp.SIDES:
                goal[s][k] = base[s][k] + d
            out.append((f"step_{suf}_{'+' if d > 0 else '-' if d < 0 else '0'}", "step",
                        {"joint": k, "delta": d, "goal": {s: goal[s][k] for s in hp.SIDES}}, goal["left"],
                        goal["right"], hold))
    for frac in (0.0, 0.5, 1.0, 0.0):
        out.append((f"closure_{frac:g}", "closure", {"closure": frac}, bjm.hand_closure("left", frac),
                    bjm.hand_closure("right", frac), closure_hold))
    return out


def run(a):
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, ContactSensorCfg

    from sim_isaac import g1_asset
    from sim_isaac import joint_map as jm
    from sim_isaac.tools import hand_step_probe as hp

    dt = 0.005
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cpu"))
    usd = a.usd or str(g1_asset.usd_path())
    cfg = g1_asset.make_articulation_cfg(usd, "/World/G1", (0.0, 0.0, 1.2), 0.0,
                                         dex3_body_collisions=a.dex3_body_collisions)
    cfg.spawn.articulation_props.fix_root_link = True
    if a.no_self_collision:
        cfg.spawn.articulation_props.enabled_self_collisions = False
    robot = Articulation(cfg)
    contact = ContactSensor(ContactSensorCfg(
        prim_path="/World/G1/(left|right)_(wrist_yaw|hand_.*)_link", update_period=0.0, history_length=1,
        filter_prim_paths_expr=[f"/World/G1/{b}" for b in PARTNERS] if a.partners else []))
    n_filtered = 0
    if a.filter_hands:
        import omni.usd
        n_filtered = filter_hand_pairs(omni.usd.get_context().get_stage(), "/World/G1")
    sim.reset()
    robot.update(0.0)
    names = list(robot.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    # hold pose: default angles (legs/waist), arms from a probe run (SONIC's arms at rest) or default
    q_hold = np.zeros(n)
    for nm, v in zip(jm.G1_MOTOR_JOINTS, jm.DEFAULT_ANGLES):
        q_hold[idx[nm]] = v
    if a.arm_pose == "frames":            # palms raised in front of the head camera: hands in free space
        for nm, v in hp.frame_pose().items():
            q_hold[idx[nm]] = v
    elif a.arm_pose:
        for nm, v in json.load(open(a.arm_pose))["notes"]["arm_start"].items():
            q_hold[idx[nm]] = v
    kp = np.full(n, 400.0)
    kd = np.full(n, 20.0)
    hi = {s: np.array([idx[x] for x in (jm.DEX3_LEFT_JOINTS if s == "left" else jm.DEX3_RIGHT_JOINTS)])
          for s in ("left", "right")}
    both = np.concatenate([hi["left"], hi["right"]])
    kp[both] = jm.DEX3_HOLD_KP
    kd[both] = jm.DEX3_HOLD_KD + jm.DEX3_JOINT_DAMPING + a.extra_kd   # as dds_bridge applies the deploy's kd
    T = lambda x: torch.tensor(np.asarray(x, np.float32)[None])  # noqa: E731
    robot.write_joint_stiffness_to_sim(T(kp))
    robot.write_joint_damping_to_sim(T(kd))
    # start like P1: every joint at the USD default (0 for the hands), then the deploy's default fist
    fist = {"left": list(hp.bjm.DEX3_CLOSED["left"]), "right": list(hp.bjm.DEX3_CLOSED["right"])}
    q_t = q_hold.copy()
    body_names = contact.body_names
    rows, crow, cmds, segs = [], [], [], []
    pair_max: dict = {}
    step = [0]

    def advance(cmd: dict, dur: float, seg_i: int):
        nonlocal q_t
        for _ in range(int(round(dur / dt))):
            q = robot.data.joint_pos[0].numpy().astype(np.float64)
            if step[0] % 4 == 0:           # the deploy writes Dex3 cmds at 50 Hz; P1 takes them every 4th step
                for s in ("left", "right"):
                    cur = q[hi[s]]
                    q_t[hi[s]] = cur + np.clip(np.asarray(cmd[s]) - cur, -0.25, 0.25)
                cmds.append((step[0] * dt, list(cmd["left"]), list(cmd["right"]), seg_i))
            robot.set_joint_position_target(T(q_t))
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            contact.update(dt)
            step[0] += 1
            if step[0] % 5 == 0:           # 40 Hz samples, as the live probe polls P1
                q = robot.data.joint_pos[0].numpy()
                rows.append((step[0] * dt, q[both].copy(), q_t[both].copy()))
                f = contact.data.net_forces_w[0].numpy()
                crow.append(np.linalg.norm(f, axis=-1))
                if a.partners:
                    fm = np.linalg.norm(contact.data.force_matrix_w[0].numpy(), axis=-1)
                    for i, j in zip(*np.nonzero(fm > 1e-3)):
                        k = f"{body_names[i]} x {PARTNERS[j]}"
                        pair_max[k] = max(pair_max.get(k, 0.0), float(fm[i, j]))

    advance(fist, 3.0, -1)
    rest_start = {s: {"q": [round(float(v), 4) for v in robot.data.joint_pos[0].numpy()[hi[s]]],
                      "q_target": [round(float(v), 4) for v in q_t[hi[s]]]} for s in ("left", "right")}
    for name, kind, meta, L, R, dur in schedule(a.hold, a.closure_hold):
        t0 = step[0] * dt
        segs.append({"name": name, "kind": kind, **meta, "t0": t0})
        advance({"left": L, "right": R}, dur, len(segs) - 1)
        segs[-1]["t1"] = step[0] * dt - 1e-6
    advance(fist, 3.0, -1)
    rest_end = {s: {"q": [round(float(v), 4) for v in robot.data.joint_pos[0].numpy()[hi[s]]],
                    "q_target": [round(float(v), 4) for v in q_t[hi[s]]]} for s in ("left", "right")}
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "probe.npz"), t=np.array([r[0] for r in rows]),
                        q=np.array([r[1] for r in rows]), qt=np.array([r[2] for r in rows]),
                        arm_q=np.zeros((len(rows), 14)), ct=np.array([c[0] for c in cmds]),
                        cl=np.array([c[1] for c in cmds]), cr=np.array([c[2] for c in cmds]),
                        cseg=np.array([c[3] for c in cmds]))
    C = np.array(crow)
    np.savez_compressed(os.path.join(a.out, "contacts.npz"), t=np.array([r[0] for r in rows]), force=C,
                        bodies=np.array(body_names))
    contact_summary = {b: {"max_N": round(float(C[:, i].max()), 3), "frac_in_contact": round(float((C[:, i] > 1e-3).mean()), 3)}
                       for i, b in enumerate(body_names)}
    notes = {"usd": usd, "filter_hands": a.filter_hands, "dex3_body_collisions": a.dex3_body_collisions,
             "filtered_pairs_extra": n_filtered,
             "hand_kd": jm.DEX3_HOLD_KD + jm.DEX3_JOINT_DAMPING + a.extra_kd,
             "self_collisions": not a.no_self_collision,
             "contact_pairs_max_N": {k: round(v, 3) for k, v in sorted(pair_max.items(), key=lambda kv: -kv[1])},
             "rest_start": rest_start,
             "rest_end": rest_end, "contacts": contact_summary, "hold_pose_from": a.arm_pose}
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump({"segments": segs, "notes": notes, "args": vars(a)}, f, indent=1)
    M = hp.analyze(a.out)
    print("HAND_PHYSICS " + json.dumps({"pass": M["pass"], "contacts": contact_summary,
                                        "pairs": notes["contact_pairs_max_N"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--usd", default=None)
    ap.add_argument("--arm-pose", default=None, help="a hand_step_probe run.json (hold the arms at its arm_start) or "
                    "'frames' (hand_step_probe.frame_pose: hands in free space)")
    ap.add_argument("--dex3-body-collisions", action="store_true",
                    help="keep Dex3 finger vs body collisions (g1_asset.make_articulation_cfg filters them by default)")
    ap.add_argument("--extra-kd", type=float, default=0.0, help="added to the Dex3 drive damping (A/B)")
    ap.add_argument("--filter-hands", action="store_true")
    ap.add_argument("--partners", action="store_true")
    ap.add_argument("--no-self-collision", action="store_true")
    ap.add_argument("--hold", type=float, default=1.2)
    ap.add_argument("--closure-hold", type=float, default=2.5)
    a, _ = ap.parse_known_args()
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    la, _ = lp.parse_known_args([])
    la.headless = True
    app = AppLauncher(la).app  # noqa: F841
    code = 0
    try:
        run(a)
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    os._exit(code)  # Kit shutdown can hang headless (as g1_asset.main)


if __name__ == "__main__":
    main()
