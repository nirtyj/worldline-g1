"""Dex3 hands in P1 (no Isaac): the URDF patch g1_asset.patch_urdf applies and the hand probe's analysis.

    .venv-rt/bin/python -m pytest -q sim_isaac/tests/test_hands.py
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from sim_isaac import g1_asset
from sim_isaac.tools import hand_step_probe as hp

SUF = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
PARENT = {"thumb_0": "palm", "thumb_1": "thumb_0", "thumb_2": "thumb_1", "middle_0": "palm", "middle_1": "middle_0",
          "index_0": "palm", "index_1": "index_0"}


def _urdf(tmp_path, name: str, hand: bool) -> str:
    """A minimal G1-like URDF: wrist -> palm (fixed) -> 7 Dex3 links per side. main: hand joints fixed, mesh
    collisions everywhere; hand: revolute, thumb_1 collides with a box (as g1_29dof_with_hand.urdf)."""
    mesh = tmp_path / "m.STL"
    mesh.write_text("solid x\nendsolid x\n")
    links, joints = [], []
    for side in ("left", "right"):
        y = -1 if side == "left" else 1
        for ln in ("wrist_yaw_link", "hand_palm_link") + tuple(f"hand_{s}_link" for s in SUF):
            col = f'<collision><geometry><mesh filename="{mesh}"/></geometry></collision>'
            if hand and ln == "hand_thumb_1_link":
                col = (f'<collision><origin xyz="-0.001 {y * 0.032} 0" rpy="0 0 0"/>'
                       f'<geometry><box size="0.02 0.03 0.02"/></geometry></collision>')
            links.append(f'<link name="{side}_{ln}">{col}</link>')
        joints.append(f'<joint name="{side}_hand_palm_joint" type="fixed"><parent link="{side}_wrist_yaw_link"/>'
                      f'<child link="{side}_hand_palm_link"/></joint>')
        for s in SUF:
            par = f"{side}_hand_{PARENT[s]}_link"
            if hand:
                joints.append(f'<joint name="{side}_hand_{s}_joint" type="revolute"><parent link="{par}"/>'
                              f'<child link="{side}_hand_{s}_link"/><origin xyz="0 0 0" rpy="0 0 0"/>'
                              f'<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1.4" velocity="12"/></joint>')
            else:
                joints.append(f'<joint name="{side}_hand_{s}_joint" type="fixed"><parent link="{par}"/>'
                              f'<child link="{side}_hand_{s}_link"/></joint>')
    p = tmp_path / name
    p.write_text(f'<robot name="g1">{"".join(links)}{"".join(joints)}</robot>')
    return str(p)


def test_patch_urdf_takes_thumb1_box_collision_from_hand_urdf(tmp_path):
    main = _urdf(tmp_path, "main.urdf", hand=False)
    hand = _urdf(tmp_path, "hand.urdf", hand=True)
    out = tmp_path / "out" / "g1.urdf"
    rep = g1_asset.patch_urdf(main_urdf=__import__("pathlib").Path(main), hand_urdf=__import__("pathlib").Path(hand),
                              out_urdf=out)
    assert sorted(rep["collision_from_hand_urdf"]) == ["left_hand_thumb_1_link", "right_hand_thumb_1_link"]
    assert len(rep["hand_joints_made_revolute"]) == 14
    root = ET.parse(out).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    for side, y in (("left", -0.032), ("right", 0.032)):
        cols = links[f"{side}_hand_thumb_1_link"].findall("collision")
        assert len(cols) == 1
        assert cols[0].find("geometry/box").get("size") == "0.02 0.03 0.02"
        assert float(cols[0].find("origin").get("xyz").split()[1]) == pytest.approx(y)
        # every other hand link keeps its mesh (= training geometry); the palm is untouched
        for ln in ("hand_palm_link", "hand_thumb_0_link", "hand_thumb_2_link", "hand_middle_0_link"):
            assert links[f"{side}_{ln}"].find("collision/geometry/mesh") is not None
    assert rep["meshes"] == 18 - 2


def _synthetic_run(tmp_path, tau: float, stuck_thumb: bool):
    """A probe run with first-order joint responses (time constant tau) on a 40 Hz P1 poll."""
    dt = 0.025
    segs, t0 = [], 100.0
    t_all, q_all, qt_all = [], [], []
    q = np.zeros(14)
    cmd = np.zeros(14)

    def run_for(T):
        nonlocal q
        for _ in range(int(round(T / dt))):
            t_all.append(t0 + len(t_all) * dt)
            q = q + (cmd - q) * (1 - np.exp(-dt / tau))
            if stuck_thumb:
                q[[0, 1, 7, 8]] = [-0.35, -0.72, 0.35, -1.05]
            q_all.append(q.copy())
            qt_all.append(cmd.copy())

    for k, suf in enumerate(SUF):
        base = {s: hp.step_base(s, k) for s in hp.SIDES}
        cmd[:] = 0.0
        cmd[k], cmd[7 + k] = base["left"], base["right"]
        run_for(1.2)                                   # settle at the base (not measured, as the live probe)
        for d in (+0.3, 0.0, -0.3, 0.0):
            goal = {s: base[s] + d for s in hp.SIDES}
            cmd[:] = 0.0
            cmd[k], cmd[7 + k] = goal["left"], goal["right"]
            ts = t0 + len(t_all) * dt
            run_for(1.2)
            segs.append({"name": f"step_{suf}", "kind": "step", "joint": k, "delta": d, "goal": goal, "t0": ts,
                         "t1": t_all[-1]})
    for frac in (0.5, 1.0):
        cmd[:7] = hp.bjm.hand_closure("left", frac)
        cmd[7:] = hp.bjm.hand_closure("right", frac)
        ts = t0 + len(t_all) * dt
        run_for(2.5)
        segs.append({"name": f"closure_{frac}", "kind": "closure", "closure": frac, "t0": ts, "t1": t_all[-1]})
    rest = {s: {"q": [0.0] * 7, "q_target": [0.0] * 7} for s in hp.SIDES}
    if stuck_thumb:
        rest["left"]["q"][:2] = [-0.35, -0.72]
    np.savez(tmp_path / "probe.npz", t=np.array(t_all), q=np.array(q_all), qt=np.array(qt_all),
             arm_q=np.zeros((len(t_all), 14)), ct=np.array(t_all), cl=np.array(qt_all)[:, :7],
             cr=np.array(qt_all)[:, 7:], cseg=np.zeros(len(t_all)))
    (tmp_path / "run.json").write_text(json.dumps({"segments": segs, "notes": {"rest_start": rest,
                                                                               "rest_end": rest}}))
    return str(tmp_path)


def test_probe_analysis_passes_a_healthy_hand(tmp_path):
    M = hp.analyze(_synthetic_run(tmp_path, tau=0.08, stuck_thumb=False))
    p = M["pass"]
    assert p["steps_n"] == 56 and p["steps_ok"] == 56 and p["all"]
    # first-order, tau 0.08 s: |err| <= 0.05 of a 0.3 step after tau*ln(6) = 0.143 s (+ one poll period)
    assert 0.12 <= p["t_band_max_s"] <= 0.18
    assert p["closure_ratio_min"] >= 0.99


def test_probe_analysis_fails_stuck_thumbs_and_slow_joints(tmp_path):
    (tmp_path / "a").mkdir()
    M = hp.analyze(_synthetic_run(tmp_path / "a", tau=0.08, stuck_thumb=True))
    bad = {(r["joint"], s) for r in M["steps"] for s in hp.SIDES if not r[s]["pass"]}
    assert {j for j, _ in bad} == {"thumb_0", "thumb_1"} and not M["pass"]["all"]
    assert M["pass"]["rest_start_thumb01_abs_max_rad"] == pytest.approx(0.72)
    (tmp_path / "b").mkdir()
    M = hp.analyze(_synthetic_run(tmp_path / "b", tau=0.35, stuck_thumb=False))   # settles in ~0.63 s
    assert M["pass"]["steps_ok"] == 0 and not M["pass"]["all"]


class _Prim:
    def __init__(self, name, rigid=True):
        self.name, self.rigid, self.targets = name, rigid, None

    def GetName(self):
        return self.name

    def GetPath(self):
        return f"/World/G1/{self.name}"

    def HasAPI(self, api):
        return self.rigid


def _fake_pxr(monkeypatch):
    import sys
    import types

    class FilteredPairsAPI:
        def __init__(self, prim):
            self.prim = prim

        @staticmethod
        def Apply(prim):
            return FilteredPairsAPI(prim)

        def CreateFilteredPairsRel(self):
            api = self

            class Rel:
                def SetTargets(self, t):
                    api.prim.targets = list(t)
            return Rel()

    physics = types.SimpleNamespace(RigidBodyAPI=object(), FilteredPairsAPI=FilteredPairsAPI)
    monkeypatch.setitem(sys.modules, "pxr", types.SimpleNamespace(UsdPhysics=physics))


def test_filter_dex3_body_collisions_fingers_vs_non_hand_links(monkeypatch):
    _fake_pxr(monkeypatch)
    body = ["pelvis", "left_hip_roll_link", "right_knee_link", "torso_link", "left_elbow_link"]
    wrists = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    fingers = [f"{s}_hand_{f}_link" for s in ("left", "right") for f in SUF]
    prims = [_Prim(n) for n in body + wrists + fingers] + [_Prim("ego_cam", rigid=False)]

    class Stage:
        def GetPrimAtPath(self, p):
            assert p == "/World/G1"
            return type("Root", (), {"GetChildren": lambda self: prims})()

    n = g1_asset.filter_dex3_body_collisions(Stage(), "/World/G1")
    assert n == 14 * len(body)
    by = {p.name: p for p in prims}
    for f in fingers:
        # every finger vs every body link; never vs a wrist (palm), another finger or a non-rigid prim
        assert sorted(by[f].targets) == sorted(f"/World/G1/{b}" for b in body)
    assert all(by[n].targets is None for n in body + wrists + ["ego_cam"])
    with pytest.raises(RuntimeError):
        g1_asset.filter_dex3_body_collisions(type("S", (), {"GetPrimAtPath": lambda self, p: type(
            "R", (), {"GetChildren": lambda self: prims[:5]})()})(), "/World/G1")


@pytest.mark.skipif(__import__("importlib").util.find_spec("unitree_sdk2py") is None,
                    reason="needs unitree_sdk2py (box env)")
def test_bridge_adds_finger_joint_damping_to_the_deploy_kd():
    from sim_isaac import joint_map as sjm
    from sim_isaac.dds_bridge import G1DdsBridge
    names = sjm.G1_MOTOR_JOINTS + sjm.DEX3_LEFT_JOINTS + sjm.DEX3_RIGHT_JOINTS
    b = G1DdsBridge(names, 28, "lo", crc=True)
    hands = np.concatenate([b.left_idx, b.right_idx])
    assert np.allclose(b.kd[hands], sjm.DEX3_HOLD_KD + sjm.DEX3_JOINT_DAMPING)
    assert np.allclose(b.kd[b.motor_idx], sjm.KDS) and np.allclose(b.kp[hands], sjm.DEX3_HOLD_KP)
    # a deploy HandCmd_ (kp 1.5 / kd 0.1, dex3_hands.hpp) -> P1 applies kd 0.1 + the passive damping
    cmd = {"q": np.full(7, 0.3), "dq": np.zeros(7), "kp": np.full(7, 1.5), "kd": np.full(7, 0.1), "tau": np.zeros(7)}
    b.r_low.take_latest = lambda: None
    b.r_lh.take_latest = lambda: (cmd, 0)
    b.r_rh.take_latest = lambda: None
    for _ in range(4):                  # Dex3 cmds are taken every 4th physics step
        b.pull_commands()
    assert np.allclose(b.q_t[b.left_idx], 0.3) and np.allclose(b.kd[b.left_idx], 0.1 + sjm.DEX3_JOINT_DAMPING)
    assert np.allclose(b.kd[b.right_idx], sjm.DEX3_HOLD_KD + sjm.DEX3_JOINT_DAMPING)
    b.close()
