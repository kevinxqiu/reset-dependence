#!/usr/bin/env python3
"""Rollout videos for the project page: one body under the three training rules.

Replays the joint traces behind Figures 1, 5 and 14 (gait-traces/ from the
dataset) in MuJoCo, one row per rule as in Figure 1, with a camera that
follows the torso. Playback is real time: 500 control steps of 0.02 s give
10 s of video. A rollout that ended early (done_step > 0) freezes at its last
step with the body tinted red, as in the figures.

Each body's three rows are rotated by the heading of its Standard rollout, so
the rows start from the same pose and the Standard walker moves left to right.
The in-panel distance is the forward displacement along world x, as in
Figure 5.

    MUJOCO_GL=egl python scripts/figures/render_rollout_videos.py --out-dir videos
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_chronophoto_3d import (  # noqa: E402
    BODY_RGBA_WARM,
    GROUND_ASSET_XML,
    GROUND_WORLD_XML,
    UNIMAL_DIRECTIONAL_LIGHT,
    UNIMAL_SELF_MATERIAL,
    UNIMAL_SKYBOX,
    default_unimal_xml_dir,
)

REPO = Path(__file__).resolve().parents[2]
BODIES = {
    "floor-5506-10-6-01-15-48-35": "Floor A",
    "mvt-5506-13-10-17-12-25-45": "MVT A",
    "floor-1409-0-7-01-15-15-26": "Floor B",
}
# (trace suffix, rule, threshold, colour from _paper_style.PROTOCOL_COLOR)
RULES = (
    ("tau0p00", "Reset-free", "τ = 0", "#3B6FB6"),
    ("tau0p20", "Standard", "τ = 0.20", "#4D9DA5"),
    ("tau0p50", "Strict", "τ = 0.50", "#E1812B"),
)
CONTROL_DT = 0.02
ENDED_RGBA = (0.90, 0.06, 0.04, 1.0)
TEXT = (43, 43, 43)
ENDED_TEXT = (179, 58, 58)
FONT_CANDIDATES = (
    REPO / "fonts/Lato-semibold.ttf",
    Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def load_model(xml_dir: Path, body: str, width: int, height: int) -> mujoco.MjModel:
    xml = (xml_dir / f"{body}.xml").read_text()
    xml = xml.replace(UNIMAL_SKYBOX, "", 1)
    xml = xml.replace(UNIMAL_DIRECTIONAL_LIGHT, "", 1)
    xml = xml.replace(UNIMAL_SELF_MATERIAL, BODY_RGBA_WARM, 1)
    xml = xml.replace("<visual>", f'<visual><global offwidth="{width}" offheight="{height}"/>', 1)
    xml = xml.replace("</mujoco>", GROUND_ASSET_XML + "</mujoco>", 1)
    xml = xml.replace("<worldbody>", "<worldbody>\n" + GROUND_WORLD_XML + "\n", 1)
    return mujoco.MjModel.from_xml_string(xml)


def rotate_qpos(q: np.ndarray, yaw: float, origin: np.ndarray) -> np.ndarray:
    """Rotate the free root joint by -yaw about the vertical axis through origin."""
    out = np.asarray(q, dtype=np.float64).copy()
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx, dy = out[0] - origin[0], out[1] - origin[1]
    out[0], out[1] = c * dx - s * dy, s * dx + c * dy
    half = -yaw / 2.0
    rot = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    quat = out[3:7].copy()
    mujoco.mju_mulQuat(out[3:7], rot, quat)
    return out


def render_body(body: str, traces: Path, xml_dir: Path, out_dir: Path,
                width: int, row_h: int, fps: int, seconds: float) -> Path:
    label = BODIES[body]
    runs = {}
    for proto, *_ in RULES:
        d = np.load(traces / f"{body}_{proto}.traces.npz")
        runs[proto] = (d["qpos_trace"], int(d["done_step"]) if "done_step" in d.files else -1)

    q_std = runs["tau0p20"][0]
    yaw = math.atan2(q_std[-1, 1] - q_std[0, 1], q_std[-1, 0] - q_std[0, 0])
    origin = q_std[0, :2].copy()
    z0 = float(q_std[0, 2])

    model = load_model(xml_dir, body, width, row_h)
    model.vis.global_.fovy = 22.0
    floor_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "_rfc_grid_mat")
    model.mat_texrepeat[floor_mat] = [120.0, 120.0]  # about 1 m tiles, so speed reads
    mat_self = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "self")
    warm = model.mat_rgba[mat_self].copy()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=row_h, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = 90.0, -12.0
    cam.distance = 4.4 + 4.0 * z0
    lookat_z = 0.25 + 0.60 * z0

    font_label, font_value = load_font(30), load_font(26)
    stride = max(1, round(1.0 / (fps * CONTROL_DT)))
    n_frames = int(round(seconds * fps))
    smooth = {proto: None for proto, *_ in RULES}
    frames = []
    for i in range(n_frames):
        step = min(i * stride, len(q_std) - 1)
        canvas = Image.new("RGB", (width, row_h * len(RULES)), (255, 255, 255))
        for r, (proto, rule, tau, colour) in enumerate(RULES):
            qpos, done = runs[proto]
            ended = done > 0 and step >= done
            t = min(done if ended else step, len(qpos) - 1)
            q = rotate_qpos(qpos[t], yaw, origin)
            target = np.array([q[0], q[1]])
            prev = smooth[proto]
            smooth[proto] = target if prev is None else prev + 0.18 * (target - prev)
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            model.mat_rgba[mat_self] = ENDED_RGBA if ended else warm
            cam.lookat[:] = [smooth[proto][0], smooth[proto][1], lookat_z]
            renderer.update_scene(data, camera=cam)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            row = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(row, "RGBA")
            name = f"{label} · {rule} ({tau})"
            tw = draw.textlength(name, font=font_label)
            draw.rounded_rectangle((16, 14, 16 + 26 + tw + 22, 58), radius=8, fill=(255, 255, 255, 225))
            draw.rectangle((28, 26, 40, 46), fill=colour)
            draw.text((50, 20), name, font=font_label, fill=TEXT)
            dist = float(qpos[t, 0] - qpos[0, 0])
            dist = 0.0 if abs(dist) < 0.05 else dist  # no "-0.0 m"
            value = f"episode ended, step {done}" if ended else f"Δx = {dist:.1f} m"
            vw = draw.textlength(value, font=font_value)
            draw.rounded_rectangle((width - vw - 44, 14, width - 16, 58), radius=8, fill=(255, 255, 255, 225))
            draw.text((width - vw - 30, 22), value, font=font_value, fill=ENDED_TEXT if ended else TEXT)
            canvas.paste(row, (0, r * row_h))
            if r:
                ImageDraw.Draw(canvas).rectangle((0, r * row_h - 2, width, r * row_h + 1), fill=(255, 255, 255))
        frames.append(np.asarray(canvas))
    renderer.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "rollouts_" + label.lower().replace(" ", "_")
    out = out_dir / f"{stem}.mp4"
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    Image.fromarray(frames[min(len(frames) - 1, 2 * fps)]).save(out_dir / f"{stem}_poster.jpg", quality=88)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traces", type=Path, default=REPO / "gait-traces")
    ap.add_argument("--xml-dir", type=Path, default=default_unimal_xml_dir())
    ap.add_argument("--out-dir", type=Path, default=REPO / "videos")
    ap.add_argument("--bodies", nargs="+", default=list(BODIES))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--row-height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()
    for body in args.bodies:
        out = render_body(body, args.traces, args.xml_dir, args.out_dir,
                          args.width, args.row_height, args.fps, args.seconds)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
