"""Offline video of a run_ref_loop.sh run, rendered from the recorded ground-truth trace.

The real-time loop only logs qpos at 50 Hz (sim_trace.jsonl). This script replays it into the same
MJCF (gear_sonic's scene_43dof.xml via the sim_ref config), so no rendering cost falls on the
real-time loop. Two views: a tracking third-person camera and the MJCF head_camera (ego view).
Overlay text: wall time, pelvis z, band state and the active planner command.

usage (WBC .venv_sim python, MUJOCO_GL=egl):  python render_ref.py RUN_DIR [--fps 25] [--w 640 --h 480]
"""

import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.utils.mujoco_sim.base_sim import GEAR_SONIC_ROOT
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    a = ap.parse_args()
    run = Path(a.run)
    tr = [json.loads(l) for l in open(run / "sim_trace.jsonl") if '"qpos"' in l]
    cmds = []
    if (run / "drive_trace.jsonl").exists():
        for l in open(run / "drive_trace.jsonl"):
            r = json.loads(l)
            if r.get("event") in ("planner_cmd", "band_released", "planner_silence"):
                cmds.append(r)
    cfg = SimLoopConfig(interface="lo", enable_onscreen=False).load_wbc_yaml()
    m = mujoco.MjModel.from_xml_path(str(Path(GEAR_SONIC_ROOT) / cfg["ROBOT_SCENE"]))
    d = mujoco.MjData(m)
    r3 = mujoco.Renderer(m, height=a.h, width=a.w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("pelvis").id
    cam.distance, cam.azimuth, cam.elevation = 3.0, 135.0, -20.0
    t = np.array([r["t_wall"] for r in tr])
    frames = np.arange(t[0], t[-1], 1.0 / a.fps)

    def enc(path):
        return subprocess.Popen(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.w}x{a.h}",
             "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE)

    out3, oute = enc(run / "video_third_person.mp4"), enc(run / "video_head_camera.mp4")
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = None
    ci = 0
    cur = "stand (IDLE)"
    for ft in frames:
        i = min(int(np.searchsorted(t, ft)), len(tr) - 1)
        rec = tr[i]
        d.qpos[:] = rec["qpos"]
        mujoco.mj_forward(m, d)
        while ci < len(cmds) and cmds[ci]["t_wall"] <= ft:
            c = cmds[ci]
            if c["event"] == "planner_cmd":
                mode = {0: "IDLE", 1: "SLOW_WALK", 2: "WALK"}.get(c.get("mode"), str(c.get("mode")))
                cur = f"planner {mode} speed={c.get('speed', -1)}"
            else:
                cur = c["event"]
            ci += 1
        r3.update_scene(d, camera=cam)
        img3 = r3.render()
        r3.update_scene(d, camera="head_camera")
        imge = r3.render()
        if Image is not None:
            im = Image.fromarray(img3)
            dr = ImageDraw.Draw(im)
            dr.text((8, 8), f"t={ft - t[0]:6.2f}s  pelvis_z={rec['pelvis_z']:.3f}  band={'on' if rec['band'] else 'off'}  falls={rec['falls']}", fill=(255, 255, 255))
            dr.text((8, 24), f"cmd: {cur}", fill=(255, 255, 0))
            dr.text((8, a.h - 20), "SONIC C++ deploy (TRT 10.13) <- DDS lo -> gear_sonic MuJoCo; replay of GT trace", fill=(200, 200, 200))
            img3 = np.asarray(im)
        out3.stdin.write(np.ascontiguousarray(img3).tobytes())
        oute.stdin.write(np.ascontiguousarray(imge).tobytes())
    for p in (out3, oute):
        p.stdin.close()
        p.wait()
    print(f"wrote {run/'video_third_person.mp4'} and {run/'video_head_camera.mp4'} ({len(frames)} frames @ {a.fps} fps)")


if __name__ == "__main__":
    main()
