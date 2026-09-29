"""G1 asset for wl-isaac: SONIC's training URDF + actuated Dex3 hands, converted to USD with Isaac Lab.

Build once (on the box, Isaac Lab env):
    /work/envs/isaaclab/bin/python -m sim_isaac.g1_asset --build [--out /work/worldline-g1/assets/g1]

Sources (`$WBC` = /work/repos/GR00T-WholeBodyControl @ b042411):
- body: $WBC/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf (the robot SONIC was trained on; 29 revolute
  joints, the 14 Dex3 joints are type="fixed", e.g. main.urdf:836)
- hands: $WBC/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf (the same 14 joints as revolute,
  e.g. :861-867); we copy <axis>/<limit>/<origin> from there, and the <collision> of every Dex3 link that it models
  with a primitive: the two thumb_1 links are a 2 x 3 x 2 cm box there (and in the MuJoCo model,
  g1_29dof_with_hand.xml:336,487) but the thumb_1 mesh in main.urdf, whose convex hull sits ~7.5 mm inside the
  palm's hull at every thumb angle. With self-collisions on (training parity) that contact jammed both thumbs'
  thumb_0/thumb_1 and pushed them to arbitrary poses at spawn (docs/arm_tracking.md §3.7).
- conversion options: $WBC/gear_sonic/envs/manager_env/robots/g1.py:199-222 (UrdfFileCfg: fix_base=False,
  replace_cylinders_with_capsules=True, drive gains 0; merge_fixed_joints left at the Isaac Lab default True).
- actuators: imported from g1.py:238-357 (G1_CYLINDER_MODEL_12_DEX_CFG.actuators), not copied.
- spawn: the Dex3 finger links do not collide with the rest of the robot (filter_dex3_body_collisions; the training
  body colliders are coarse proxies, e.g. a thigh capsule the fingertips sit in at SONIC's rest arms).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from . import joint_map as jm

WBC_DIR = Path(os.environ.get("WBC_DIR", "/work/repos/GR00T-WholeBodyControl"))
MAIN_URDF = WBC_DIR / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
HAND_URDF = WBC_DIR / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
DEFAULT_OUT = Path(os.environ.get("WL_ASSETS", "/work/worldline-g1/assets")) / "g1"
USD_NAME = "g1_sonic_dex3.usd"
URDF_NAME = "g1_sonic_dex3.urdf"
HAND_JOINT_RE = re.compile(r"^(left|right)_hand_(thumb_[0-2]|middle_[01]|index_[01])_joint$")


def _resolve_mesh(fn: str, urdf_path: Path) -> str:
    """package://<pkg>/rest -> <nearest ancestor dir named pkg>/rest (what the Isaac URDF importer does)."""
    if not fn.startswith("package://"):
        p = Path(fn)
        return str(p if p.is_absolute() else (urdf_path.parent / p).resolve())
    pkg, _, rest = fn[len("package://"):].partition("/")
    for anc in urdf_path.resolve().parents:
        if anc.name == pkg:
            return str(anc / rest)
    raise FileNotFoundError(f"cannot resolve {fn} from {urdf_path}")


def patch_urdf(main_urdf: Path = MAIN_URDF, hand_urdf: Path = HAND_URDF, out_urdf: Path | None = None) -> dict:
    """Write main.urdf with the 14 Dex3 joints made revolute and absolute mesh paths. Returns a report."""
    out_urdf = out_urdf or (DEFAULT_OUT / URDF_NAME)
    tree = ET.parse(main_urdf)
    root = tree.getroot()
    hand_root = ET.parse(hand_urdf).getroot()
    hand_joints = {j.get("name"): j for j in hand_root.findall("joint")}
    changed, diffs = [], []
    for j in root.findall("joint"):
        name = j.get("name")
        if not HAND_JOINT_RE.match(name or ""):
            continue
        src = hand_joints.get(name)
        if src is None or src.get("type") != "revolute":
            raise RuntimeError(f"{name}: not revolute in {hand_urdf}")
        for tag in ("axis", "limit", "origin"):
            s, d = src.find(tag), j.find(tag)
            if s is None:
                continue
            if d is not None and dict(d.attrib) != dict(s.attrib):
                diffs.append({"joint": name, "tag": tag, "main": dict(d.attrib), "hand": dict(s.attrib)})
            if d is not None:
                j.remove(d)
            j.append(copy.deepcopy(s))
        j.set("type", "revolute")
        changed.append(name)
    if len(changed) != 14:
        raise RuntimeError(f"expected 14 hand joints, found {len(changed)}: {changed}")
    # Dex3 links the hand URDF collides with primitives (thumb_1: a box) get that collision instead of the mesh
    hand_links = {l.get("name"): l for l in hand_root.findall("link")}
    child_links = {j.find("child").get("link") for j in root.findall("joint") if j.get("name") in changed}
    collision_from_hand = []
    for link in root.findall("link"):
        src = hand_links.get(link.get("name"))
        if link.get("name") not in child_links or src is None:
            continue
        cols = src.findall("collision")
        if not cols or any(c.find("geometry/mesh") is not None for c in cols):
            continue
        for c in link.findall("collision"):
            link.remove(c)
        for c in cols:
            link.append(copy.deepcopy(c))
        collision_from_hand.append(link.get("name"))
    n_mesh = 0
    for m in root.iter("mesh"):
        m.set("filename", _resolve_mesh(m.get("filename"), main_urdf))
        if not Path(m.get("filename")).exists():
            raise FileNotFoundError(m.get("filename"))
        n_mesh += 1
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    revolute = [j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"]
    return {"out_urdf": str(out_urdf), "hand_joints_made_revolute": changed, "limit_axis_origin_diffs": diffs,
            "collision_from_hand_urdf": collision_from_hand,
            "revolute_joints": len(revolute), "meshes": n_mesh, "source": str(main_urdf), "hand_source": str(hand_urdf)}


def build_usd(out_dir: Path = DEFAULT_OUT) -> dict:
    """Patch + convert (needs a running Isaac Sim app)."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    rep = patch_urdf(out_urdf=out_dir / URDF_NAME)
    cfg = UrdfConverterCfg(
        asset_path=str(out_dir / URDF_NAME),
        usd_dir=str(out_dir),
        usd_file_name=USD_NAME,
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=False,                        # g1.py:201
        replace_cylinders_with_capsules=True,  # g1.py:202
        merge_fixed_joints=True,               # UrdfFileCfg default used by training
        joint_drive=UrdfConverterCfg.JointDriveCfg(  # g1.py:219-221
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
    )
    conv = UrdfConverter(cfg)
    rep["usd_path"] = conv.usd_path
    (out_dir / "build_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    return rep


def usd_path(out_dir: Path = DEFAULT_OUT) -> Path:
    return out_dir / USD_NAME


DEX3_FINGER_LINK_RE = re.compile(r"^(left|right)_hand_(thumb_[0-2]|middle_[01]|index_[01])_link$")


def filter_dex3_body_collisions(stage, root: str) -> int:
    """UsdPhysics.FilteredPairsAPI: each Dex3 finger/thumb link vs every robot link that is not part of a hand.

    Training had the Dex3 merged into the wrist (fixed joints), so its body collision shapes are coarse proxies that
    never had fingers next to them: e.g. the thigh is a capsule (main.urdf left_hip_roll_link: cylinder r 0.05,
    l 0.2, replace_cylinders_with_capsules) that encloses the space where the thumbs and fingertips sit when SONIC
    holds its own rest arms. Measured (tools/hand_physics_probe.py, fixed root, SONIC's rest arms, no filter):
    thumb_2 / index_1 / middle_1 vs hip_roll_link contacts up to 215 N, thumbs pushed off target by up to 1.2 rad.
    Fingers keep colliding with their own hand, the other hand and the world. Returns the number of pairs."""
    from pxr import UsdPhysics
    links = [p for p in stage.GetPrimAtPath(root).GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    fingers = [p for p in links if DEX3_FINGER_LINK_RE.match(p.GetName())]
    body = [p.GetPath() for p in links if not DEX3_FINGER_LINK_RE.match(p.GetName())
            and not re.match(r"^(left|right)_wrist_yaw_link$", p.GetName())]   # the wrist-yaw body carries the palm
    if len(fingers) != 14 or not body:
        raise RuntimeError(f"{root}: expected 14 Dex3 finger links and body links, got {len(fingers)} / {len(body)}")
    for p in fingers:
        UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets(body)
    return len(fingers) * len(body)


def _spawn_g1(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """spawn_from_usd + filter_dex3_body_collisions (before the physics parses the stage at sim.reset())."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    filter_dex3_body_collisions(prim.GetStage(), prim_path)
    return prim


def make_articulation_cfg(usd: str, prim_path: str = "/World/G1", pos=(0.0, 0.0, 0.8), yaw: float = 0.0,
                          dex3_body_collisions: bool = False):
    """ArticulationCfg with training-parity actuators (imported from gear_sonic) + a Dex3 group.

    dex3_body_collisions=False (default): the Dex3 finger links do not collide with the rest of the robot
    (filter_dex3_body_collisions); True only for A/B tests."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg

    from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG as TRAIN

    actuators = {k: copy.deepcopy(v) for k, v in TRAIN.actuators.items()}
    # Dex3: training had the hands fixed. Armature like the MuJoCo finger_motor class
    # (g1_29dof_with_hand.xml:20-22); effort/velocity limits from the URDF (None = keep USD values);
    # initial gains = the deploy's Dex3 hold gains (dex3_hands.hpp:308-332) + the finger joints' passive damping
    # (joint_map.DEX3_JOINT_DAMPING). The deploy's rt/dex3/*/cmd kp/kd (+ that damping, dds_bridge) overwrite them.
    actuators["dex3"] = ImplicitActuatorCfg(
        joint_names_expr=[r".*_hand_(thumb|middle|index)_[0-2]_joint"],
        effort_limit_sim=None, velocity_limit_sim=None,
        stiffness=jm.DEX3_HOLD_KP, damping=jm.DEX3_HOLD_KD + jm.DEX3_JOINT_DAMPING, armature=0.01,
    )
    import math
    init_state = copy.deepcopy(TRAIN.init_state)
    init_state.pos = tuple(float(x) for x in pos)
    init_state.rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    spawn = sim_utils.UsdFileCfg(
        usd_path=str(usd),
        activate_contact_sensors=True,                                     # g1.py:204
        rigid_props=copy.deepcopy(TRAIN.spawn.rigid_props),                # g1.py:205-213
        articulation_props=copy.deepcopy(TRAIN.spawn.articulation_props),  # g1.py:214-218 (self-coll., 8/4 iters)
    )
    if not dex3_body_collisions:
        spawn.func = _spawn_g1
    return ArticulationCfg(prim_path=prim_path, spawn=spawn, init_state=init_state, actuators=actuators,
                           soft_joint_pos_limit_factor=TRAIN.soft_joint_pos_limit_factor)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="patch the URDF and convert it to USD")
    ap.add_argument("--patch-only", action="store_true", help="only write the patched URDF (no Isaac needed)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a, _ = ap.parse_known_args()
    out = Path(a.out)
    if a.patch_only:
        print(json.dumps(patch_urdf(out_urdf=out / URDF_NAME), indent=2))
        return
    if not a.build:
        ap.error("pass --build or --patch-only")
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    la, _ = lp.parse_known_args([])
    la.headless = True
    app = AppLauncher(la).app
    try:
        rep = build_usd(out)
        print("G1_ASSET_BUILD " + json.dumps(rep), flush=True)
        code = 0
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"G1_ASSET_BUILD_FAILED {e}", flush=True)
        code = 1
    sys.stdout.flush()
    os._exit(code)  # Kit shutdown can hang headless (same as eval_agent_trl.py)


if __name__ == "__main__":
    main()
