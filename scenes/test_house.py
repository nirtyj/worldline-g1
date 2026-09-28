"""Standalone house test (Isaac Sim 5.1, headless): load, label, occupancy, walkability, views, RTF.

    source /etc/profile.d/ludo.sh
    cd /work/worldline-g1 && PYTHONUNBUFFERED=1 /work/envs/isaaclab/bin/python -m scenes.test_house \
        --house procthor-train-40 [--dynamic-objects keep|kinematic] [--rtf-seconds 10]

Writes the per-house assets (occupancy.npz/png/meta, house_info.json) to
/work/worldline-g1/assets/houses/<id>/ and evidence (images, metrics.json) to
/work/worldline-g1/outputs/m1/house/<id>[-<tag>]/.

RTF = simulated seconds / wall seconds, with physics at --physics-hz (200 Hz, SONIC's dt) on
CPU PhysX (isaacsim.core.api SimulationContext default), house only (no robot):
  rtf_physics_only     sim.step(render=False) back-to-back
  rtf_physics_cam30    + one 640x480 RGB render product rendered and read back at 30 Hz sim time
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default="procthor-train-40")
    ap.add_argument("--out", default="/work/worldline-g1/outputs/m1/house")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dynamic-objects", default="keep", choices=["keep", "kinematic"])
    ap.add_argument("--free-joints", action="store_true", help="do not lock furniture/door joints")
    ap.add_argument("--physx-threads", type=int, default=0, help="/persistent/physics/numThreads (0 = run PhysX on the caller thread: fastest for one scene, measured; -1 = leave the Kit default 8)")
    ap.add_argument("--no-sleep", action="store_true", help="do not put the house to sleep before measuring RTF")
    ap.add_argument("--physics-hz", type=float, default=200.0)
    ap.add_argument("--rtf-seconds", type=float, default=10.0)
    ap.add_argument("--settle-seconds", type=float, default=5.0)
    ap.add_argument("--skip-views", action="store_true")
    return ap.parse_args()


def pct(a, q):
    import numpy as np

    return round(float(np.percentile(np.asarray(a), q)) * 1000, 3) if len(a) else None


def gpu_mem_mib(pid: int) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.strip().splitlines():
            p, m = [s.strip() for s in line.split(",")]
            if int(p) == pid:
                return int(m)
    except Exception:
        pass
    return None


def main() -> int:
    args = parse()
    t0 = time.time()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 1280, "height": 720})
    t_app = time.time() - t0

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.gen.omap")
    app.update()

    import numpy as np
    import omni.usd
    from isaacsim.core.api import SimulationContext

    from scenes import checks
    from scenes.catalog import parse_house_id
    from scenes.loader import collision_filter_report, finalize_with_occupancy, load_house, verify_procthor_mapping
    from scenes.occupancy import generate_occupancy, save_overlay_png
    from scenes.render import Camera, TopDown, save_rgb

    ref = parse_house_id(args.house)
    out = Path(args.out) / (ref.house_id + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    M: dict = {"house_id": ref.house_id, "args": vars(args), "times_s": {"app_start": round(t_app, 2)}}

    import carb

    cs = carb.settings.get_settings()
    M["physx_threads_default"] = cs.get("/persistent/physics/numThreads")
    if args.physx_threads >= 0:
        cs.set_int("/persistent/physics/numThreads", args.physx_threads)
    M["physx_threads"] = cs.get("/persistent/physics/numThreads")
    sim = SimulationContext(physics_dt=1.0 / args.physics_hz, rendering_dt=1.0 / 30.0, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # ---- load ----------------------------------------------------------------------------------
    t = time.time()
    info = load_house(sim, ref.house_id, dynamic_objects=args.dynamic_objects, lock_joints=not args.free_joints)
    M["times_s"]["load_house"] = round(time.time() - t, 3)
    t = time.time()
    n = 0
    while n < 900:
        sim.render()
        n += 1
        st = omni.usd.get_context().get_stage_loading_status()
        if n >= 10 and st[2] == 0:
            break
    M["times_s"]["first_renders"] = round(time.time() - t, 2)
    M["first_render_updates"] = n
    t = time.time()
    sim.reset()
    sim.step(render=False)
    M["times_s"]["physics_init"] = round(time.time() - t, 3)
    M["times_s"]["load_total_to_physics_ready"] = round(time.time() - t0, 2)
    M["collision_filter_after_reset"] = collision_filter_report(stage)

    # ---- occupancy + spawn ------------------------------------------------------------------------
    t = time.time()
    occ = generate_occupancy(info, method="auto")
    M["times_s"]["occupancy"] = round(time.time() - t, 2)
    finalize_with_occupancy(info, occ, save=True)
    M["occupancy"] = occ.summary(info.assets_dir / "occupancy.npz")
    save_overlay_png(occ, info.to_dict(), out / "occupancy_overlay.png")
    shutil.copy(info.assets_dir / "occupancy.png", out / "occupancy.png")

    # ---- counts / mapping -----------------------------------------------------------------------
    from collections import Counter

    M["counts"] = {
        "rooms": len(info.rooms),
        "objects": len(info.objects),
        "static_objects": sum(o.is_static for o in info.objects),
        "dynamic_objects": sum(not o.is_static for o in info.objects),
        "articulated_objects": sum(o.articulated for o in info.objects),
        "categories": len({o.category for o in info.objects}),
        "labels_applied": info.stats.get("labels_applied"),
        "objects_per_room": dict(Counter(o.room or "none" for o in info.objects)),
    }
    M["rooms"] = [{"name": r.name, "type": r.type, "room_id": r.room_id, "area_m2": r.area_m2} for r in info.rooms]
    M["category_counts"] = dict(sorted(Counter(o.category for o in info.objects).items()))
    M["mapping_check"] = verify_procthor_mapping(info)
    M["spawn"] = info.spawn
    M["room_points"] = info.room_points
    M["connectivity"] = info.connectivity
    M["loader_stats"] = info.stats
    M["warnings"] = info.warnings

    # ---- walkability ----------------------------------------------------------------------------
    M["floor_check"] = checks.floor_check(info, occ)
    M["door_check"] = checks.door_check(info, occ)
    bc, unexplained = checks.blocker_check(info, occ)
    M["blocker_check"] = bc
    np.save(out / "unexplained_obstacles.npy", unexplained)

    # ---- settle (also warms PhysX up) -----------------------------------------------------------
    t = time.time()
    M["settle_check"] = checks.settle_check(info, sim, args.settle_seconds)
    M["times_s"]["settle"] = round(time.time() - t, 2)
    M["awake_bodies"] = checks.awake_bodies(info, sim, 0.5)
    from scenes.loader import awake_report, sleep_house

    M["physx_awake_after_settle"] = [p.replace(info.root + "/Geometry/", "") for p in awake_report(stage, info.root)]
    if not args.no_sleep:
        M["sleep_house"] = sleep_house(stage, info.root)
        sim.step(render=False)
        M["physx_awake_after_sleep"] = [p.replace(info.root + "/Geometry/", "") for p in awake_report(stage, info.root)]

    # ---- RTF: physics only ------------------------------------------------------------------------
    # The box is shared: record CPU contention next to every RTF number.
    try:
        import psutil

        me = psutil.Process()
        psutil.cpu_percent(None)
        me.cpu_percent(None)
    except Exception:  # pragma: no cover
        psutil = None

    def contention(tag: str) -> dict:
        d = {"loadavg_1m": round(os.getloadavg()[0], 2), "ncpu": os.cpu_count()}
        if psutil is not None:
            sys_pct = psutil.cpu_percent(None)  # since previous call, % of all CPUs
            own = me.cpu_percent(None) / (os.cpu_count() or 1)
            d.update({"system_cpu_pct": round(sys_pct, 1), "own_cpu_pct_of_box": round(own, 1), "others_cpu_pct_of_box": round(max(sys_pct - own, 0.0), 1)})
        M.setdefault("cpu_contention", {})[tag] = d
        return d

    contention("before_rtf")
    dt = sim.get_physics_dt()
    n_steps = int(round(args.rtf_seconds / dt))
    step_t = []
    tw = time.time()
    cpu0 = time.process_time()
    for _ in range(n_steps):
        a = time.perf_counter()
        sim.step(render=False)
        step_t.append(time.perf_counter() - a)
    wall = time.time() - tw
    cpu_s = time.process_time() - cpu0
    M["rtf_physics_only"] = {
        "sim_s": round(n_steps * dt, 3),
        "wall_s": round(wall, 3),
        "process_cpu_ms_per_step": round(cpu_s / n_steps * 1000, 3),  # all threads; robust to contention
        "rtf": round(n_steps * dt / wall, 3),
        "achieved_physics_hz": round(n_steps / wall, 1),
        "step_ms_p50": pct(step_t, 50),
        "step_ms_p95": pct(step_t, 95),
        "step_ms_max": pct(step_t, 100),
    }
    M["rtf_physics_only"]["cpu"] = contention("physics_only")
    print("[house] physics-only", M["rtf_physics_only"], flush=True)

    # ---- RTF: physics + one 640x480 camera at 30 Hz -----------------------------------------------
    sp = info.spawn
    cam = Camera(stage, "/World/WLCams/head_like", (640, 480), hfov_deg=58.0)
    eye = (sp["x"], sp["y"], info.floor_z + 1.2)
    cam.look_at(eye, (sp["x"] + math.cos(sp["yaw"]), sp["y"] + math.sin(sp["yaw"]), info.floor_z + 0.6))
    for _ in range(10):
        sim.render()
    step_t, rend_t = [], []
    frames = 0
    next_frame = 0.0
    sim_t = 0.0
    tw = time.time()
    cpu0 = time.process_time()
    for _ in range(n_steps):
        a = time.perf_counter()
        sim.step(render=False)
        step_t.append(time.perf_counter() - a)
        sim_t += dt
        if sim_t + 1e-9 >= next_frame:
            b = time.perf_counter()
            sim.render()
            img = cam.latest()
            rend_t.append(time.perf_counter() - b)
            frames += 1 if img is not None else 0
            next_frame += 1.0 / 30.0
    wall = time.time() - tw
    cpu_s = time.process_time() - cpu0
    M["rtf_physics_cam30"] = {
        "sim_s": round(n_steps * dt, 3),
        "wall_s": round(wall, 3),
        "process_cpu_s_per_sim_s": round(cpu_s / (n_steps * dt), 3),
        "rtf": round(n_steps * dt / wall, 3),
        "achieved_physics_hz": round(n_steps / wall, 1),
        "render_hz_sim": round(len(rend_t) / (n_steps * dt), 2),
        "render_hz_wall": round(len(rend_t) / wall, 2),
        "frames_with_data": frames,
        "render_ms_p50": pct(rend_t, 50),
        "render_ms_p95": pct(rend_t, 95),
        "step_ms_p50": pct(step_t, 50),
        "step_ms_p95": pct(step_t, 95),
        "camera": {"res": [640, 480], "hfov_deg": 58.0, "eye": [round(v, 3) for v in eye]},
    }
    M["rtf_physics_cam30"]["cpu"] = contention("physics_cam30")
    print("[house] physics+cam30", M["rtf_physics_cam30"], flush=True)
    img = cam.latest()
    if img is not None:
        save_rgb(img, out / "rtf_camera_last_frame.png")
    M["gpu_mem_mib"] = gpu_mem_mib(os.getpid())

    # ---- evidence views -----------------------------------------------------------------------------
    if not args.skip_views:
        t = time.time()
        views = {}
        cam.destroy()
        top = TopDown.for_bounds(stage, "/World/WLCams/topdown", info.bounds_xy, px=1024, floor_z=info.floor_z)
        img = top.grab(sim, 30)
        if img is not None:
            save_rgb(img, out / "topdown.png")
            top.save_meta(out / "topdown_meta.json")
            views["topdown"] = str(out / "topdown.png")
        top.destroy()
        eye_cam = Camera(stage, "/World/WLCams/eye", (1024, 768), hfov_deg=75.0)
        shots = [("eye_spawn", eye, (sp["x"] + 3 * math.cos(sp["yaw"]), sp["y"] + 3 * math.sin(sp["yaw"]), info.floor_z + 0.8))]
        # second view: from the kitchen's max-clearance point towards the kitchen's biggest static object
        kitchen = next((r for r in info.rooms if r.type == "Kitchen"), None)
        kp = next((p for p in info.room_points if kitchen and p.get("room") == kitchen.name and p.get("ok")), None)
        if kp:
            big = [o for o in info.objects if o.room == kitchen.name and o.is_static and o.category not in ("Window", "Painting", "Doorway")]
            if big:
                big.sort(key=lambda o: -(o.aabb[1][0] - o.aabb[0][0]) * (o.aabb[1][1] - o.aabb[0][1]) * (o.aabb[1][2] - o.aabb[0][2]))
                c = big[0].aabb
                tgt = ((c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2, max(0.6, (c[0][2] + c[1][2]) / 2))
            else:
                tgt = (kp["x"] + math.cos(kp["yaw"]), kp["y"] + math.sin(kp["yaw"]), 0.8)
            shots.append(("eye_kitchen", (kp["x"], kp["y"], info.floor_z + 1.2), tgt))
        for other in info.room_points:
            if other.get("ok") and (not kitchen or other["room"] != kitchen.name) and other["room"] != sp.get("room"):
                shots.append((f"eye_{other['room']}", (other["x"], other["y"], info.floor_z + 1.2),
                              (other["x"] + 3 * math.cos(other["yaw"]), other["y"] + 3 * math.sin(other["yaw"]), 0.8)))
                break
        for name, e, tg in shots:
            eye_cam.look_at(e, tg)
            img = eye_cam.grab(sim, 20)
            if img is not None:
                save_rgb(img, out / f"{name}.png")
                views[name] = {"eye": [round(v, 3) for v in e], "target": [round(v, 3) for v in tg]}
        M["views"] = views
        M["times_s"]["views"] = round(time.time() - t, 2)

    M["times_s"]["total"] = round(time.time() - t0, 2)
    (out / "metrics.json").write_text(json.dumps(M, indent=2, default=str))
    shutil.copy(info.assets_dir / "house_info.json", out / "house_info.json")
    print("[house] summary", json.dumps({k: M[k] for k in ("counts", "rtf_physics_only", "rtf_physics_cam30", "times_s", "spawn")}, default=str), flush=True)
    print(f"[house] wrote {out}", flush=True)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
