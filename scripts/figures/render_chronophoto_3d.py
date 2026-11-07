#!/usr/bin/env python3
"""LOKI-style chronophotograph: one continuous scene per (body, protocol).

Native MuJoCo accumulation (mjv_addGeoms): fix a single orthographic side
camera per panel, render the first sampled pose to create the scene (floor,
sky, lights, first robot), then inject the dynamic geoms from each later
pose into the same mjvScene with time-ramping alpha. MuJoCo rasterises the
whole scene once, so the result reads as a single long-exposure photograph
of the robot at N positions along its path.

Style notes:
  - single continuous floor and sky (no filmstrip seams)
  - MuJoCo-style blue-grey checker floor with white tile edges, pale sky gradient
  - final pose tinted red when the rollout terminated early
  - trajectory aligned to +x via qpos yaw rotation so the side camera reads

Two figures produced per run:
  - fig_gait_chronophoto_hero.pdf : 3 rows, hero body (Floor A) under the
    three training protocols. Intended as a page-1 teaser.
  - fig_gait_chronophoto_3d.pdf   : 3x3 for all three bodies.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib.pyplot as plt
import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
GAIT_DIR = REPO / "gait-traces"


def default_unimal_xml_dir() -> Path:
    fallback = REPO / "assets/unimals_100/xml"
    for path in (
        os.environ.get("RFC_UNIMAL_XML_DIR"),
        fallback,
    ):
        if path is None:
            continue
        candidate = Path(path)
        if candidate.is_dir():
            return candidate
    return fallback


XML_DIR = default_unimal_xml_dir()
OUT_HERO_PDF = REPO / "figures/fig_gait_chronophoto_hero.pdf"
OUT_FULL_PDF = REPO / "figures/fig_gait_chronophoto_3d.pdf"
HERO_BODY = "floor-5506-10-6-01-15-48-35"
HERO_PAIR_BODIES = (
    "floor-5506-10-6-01-15-48-35",
    "floor-1409-0-7-01-15-15-26",
)

BODIES = (
    "floor-5506-10-6-01-15-48-35",
    "mvt-5506-13-10-17-12-25-45",
    "floor-1409-0-7-01-15-15-26",
)
APPENDIX_BODIES = (
    "mvt-5506-13-10-17-12-25-45",
    "floor-1409-0-7-01-15-15-26",
)
BODY_LABELS = {
    "floor-5506-10-6-01-15-48-35": "Floor A",
    "mvt-5506-13-10-17-12-25-45":  "MVT A",
    "floor-1409-0-7-01-15-15-26":  "Floor B",
}
PROTOCOLS = ("tau0p00", "tau0p20", "tau0p50")
PROTOCOL_TITLES = {
    "tau0p00": r"Reset-free ($\tau{=}0$)",
    "tau0p20": r"Standard ($\tau{=}0.20$)",
    "tau0p50": r"Strict ($\tau{=}0.50$)",
}

CANVAS_W = 1600
CANVAS_H = 220

# Per-protocol chronophoto sampling. Bodies in this corpus walk ~3-5 m/s under
# 50 Hz control (dt=0.02 s, frame_skip=4), so the Standard window is kept
# short (~4 strides) to show a gait cycle cleanly rather than a blur.
# Evenly-spaced arc-length sampling naturally lands poses at distinct gait
# phases for Floor A (14-step period at 0.8 m spacing = ~57% phase offset).
PROTOCOL_PARAMS = {
    "tau0p00": {"n_poses": 5, "target_span_m": 4.0, "min_span_m": 1.5,
                 "skip_steps": 0, "view_pad_m": 0.35},
    "tau0p20": {"n_poses": 7, "target_span_m": 6.0, "min_span_m": 3.0,
                 "skip_steps": 50, "view_pad_m": 0.35},
    "tau0p50": {"n_poses": 4, "target_span_m": 2.0, "min_span_m": 1.0,
                 "skip_steps": 0, "view_pad_m": 0.20},
}

CAM_AZIMUTH = 90
CAM_ELEVATION = -3
CAM_LOOKAT_Z = 0.70
CAM_FOVY_DEG = 22.0
CAM_DIST_PAD = 1.35

BG_COLOR = np.array([250, 248, 243], dtype=np.uint8)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _render_style import TEXT, apply_style  # noqa: E402

apply_style()

UNIMAL_SKYBOX = (
    '<texture name="skybox" type="skybox" builtin="flat" '
    'rgb1="0.8 1 1" rgb2="0.8 1 1" width="800" height="800"/>'
)
UNIMAL_DIRECTIONAL_LIGHT = (
    '<light diffuse="1 1 1" directional="true" exponent="1" pos="0 0 1" specular=".1 .1 .1"/>'
)
UNIMAL_SELF_MATERIAL = '<material name="self" rgba=".7 .5 .3 1"/>'
BODY_RGBA_WARM = '<material name="self" rgba="0.92 0.50 0.18 1"/>'
VISUAL_GLOBAL_INSERT = f'<global offwidth="{CANVAS_W}" offheight="{CANVAS_H}"/>'
GROUND_ASSET_XML = """
  <asset>
    <texture name="_rfc_sky" type="skybox" builtin="gradient" rgb1="0.92 0.96 1.00" rgb2="0.40 0.60 0.80" width="256" height="256"/>
    <texture name="_rfc_grid_tex" type="2d" builtin="checker" width="600" height="600" rgb1="0.38 0.48 0.58" rgb2="0.22 0.30 0.40" mark="edge" markrgb="0.88 0.90 0.92"/>
    <material name="_rfc_grid_mat" texture="_rfc_grid_tex" texrepeat="8 8" emission="0.10" specular="0" shininess="0" reflectance="0"/>
  </asset>
"""
GROUND_WORLD_XML = (
    '<geom name="_rfc_floor" type="plane" size="120 120 0.1" '
    'material="_rfc_grid_mat" contype="0" conaffinity="0"/>'
)


def load_model_with_floor(body: str) -> mujoco.MjModel:
    xml_text = (XML_DIR / f"{body}.xml").read_text()
    xml_text = xml_text.replace(UNIMAL_SKYBOX, "", 1)
    xml_text = xml_text.replace(UNIMAL_DIRECTIONAL_LIGHT, "", 1)
    xml_text = xml_text.replace(UNIMAL_SELF_MATERIAL, BODY_RGBA_WARM, 1)
    xml_text = xml_text.replace("<visual>", "<visual>" + VISUAL_GLOBAL_INSERT, 1)
    xml_text = xml_text.replace(
        "</mujoco>",
        GROUND_ASSET_XML + "</mujoco>",
        1,
    )
    xml_text = xml_text.replace(
        "<worldbody>",
        "<worldbody>\n" + GROUND_WORLD_XML + "\n",
        1,
    )
    return mujoco.MjModel.from_xml_string(xml_text)


def load_qpos(body: str, proto: str):
    p = GAIT_DIR / f"{body}_{proto}.traces.npz"
    if not p.exists():
        return None, None, None
    d = np.load(p)
    qpos = d["qpos_trace"]
    done_step = int(d["done_step"]) if "done_step" in d.files else -1
    if done_step > 0:
        dist_x = float(qpos[done_step, 0] - qpos[0, 0])
    else:
        dist_x = float(qpos[-1, 0] - qpos[0, 0])
    return qpos, done_step, dist_x


def pick_window(qpos: np.ndarray, done_step: int, params: dict):
    n_poses = params["n_poses"]
    target_span_m = params["target_span_m"]
    min_span_m = params["min_span_m"]
    skip = params["skip_steps"]

    T = len(qpos)
    last = min(done_step + 2, T - 1) if done_step > 0 else T - 1
    i0 = min(max(2, skip), last)
    if last <= i0:
        i0 = min(2, last)
    if last <= i0:
        return np.array([i0] * n_poses), 0.0

    xy = qpos[i0:last + 1, :2]
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(ds)]

    if s[-1] < min_span_m:
        ts = i0 + np.linspace(0, last - i0, n_poses).astype(int)
        return ts, float(s[-1])

    span = min(s[-1], target_span_m)
    target_s = np.linspace(0, span, n_poses)
    sub_idx = np.searchsorted(s, target_s)
    sub_idx = np.clip(sub_idx, 0, len(s) - 1)
    ts = np.clip(i0 + sub_idx, i0, last)
    return ts, float(span)


def yaw_from_window(qpos, ts):
    p0 = qpos[ts[0], :2]
    p1 = qpos[ts[-1], :2]
    delta = p1 - p0
    if np.linalg.norm(delta) < 0.3:
        return 0.0, p0
    return float(np.arctan2(delta[1], delta[0])), p0


def canonical_qpos(q, yaw, origin):
    qq = np.asarray(q, dtype=np.float64).copy()
    c, sn = np.cos(-yaw), np.sin(-yaw)
    dx, dy = qq[0] - origin[0], qq[1] - origin[1]
    qq[0] = c * dx - sn * dy
    qq[1] = sn * dx + c * dy
    yawq = np.zeros(4, dtype=np.float64)
    mujoco.mju_axisAngle2Quat(yawq, np.array([0.0, 0.0, 1.0], dtype=np.float64), -yaw)
    rootq = qq[3:7].astype(np.float64).copy()
    new_rootq = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(new_rootq, yawq, rootq)
    qq[3:7] = new_rootq
    return qq


def dynamic_geom_indices(scene, start: int):
    out = []
    for k in range(start, scene.ngeom):
        g = scene.geoms[k]
        if int(g.category) & int(mujoco.mjtCatBit.mjCAT_DYNAMIC):
            out.append(k)
    return out


def render_chronophoto_panel(body: str, proto: str):
    qpos, done_step, dist_x = load_qpos(body, proto)
    if qpos is None:
        return None, 0.0, -1

    params = PROTOCOL_PARAMS[proto]
    n_poses = params["n_poses"]
    view_pad = params["view_pad_m"]
    min_span_m = params["min_span_m"]
    target_span_m = params["target_span_m"]

    ts, travel_span = pick_window(qpos, done_step, params)
    yaw, origin = yaw_from_window(qpos, ts)

    model = load_model_with_floor(body)
    data = mujoco.MjData(model)
    opt = mujoco.MjvOption()

    # Compute pose midpoint and mean-z in canonical frame: center camera here.
    sampled_xs = []
    sampled_zs = []
    for t in ts:
        q = canonical_qpos(qpos[t], yaw, origin)
        sampled_xs.append(float(q[0]))
        sampled_zs.append(float(q[2]))
    pose_mid_x = 0.5 * (min(sampled_xs) + max(sampled_xs))
    pose_span = max(sampled_xs) - min(sampled_xs)
    pose_mid_z = float(np.mean(sampled_zs))
    # Frame centers above body so ground + sky both visible; at least 0.25 m up.
    lookat_z = max(0.30, pose_mid_z + 0.10)

    view_span = max(min_span_m, min(target_span_m, pose_span + 2.0 * view_pad))
    aspect = CANVAS_W / CANVAS_H

    # Perspective camera: fovy degrees, distance chosen so horizontal frustum
    # at lookat covers 1.08 * view_span.
    fovy_h_deg = CAM_FOVY_DEG * aspect
    cam_distance = (CAM_DIST_PAD * view_span) / (2.0 * np.tan(np.radians(fovy_h_deg / 2.0)))
    # Floor: stationary bodies (Strict, tumbling RF) must not have the camera
    # crash into them. 4 m keeps a consistent body scale across panels while
    # still letting the Standard walker span the full width.
    cam_distance = max(4.0, cam_distance)

    model.vis.global_.orthographic = 0
    model.vis.global_.fovy = CAM_FOVY_DEG

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.orthographic = 0
    cam.lookat[:] = np.array([pose_mid_x, 0.0, lookat_z])
    cam.distance = float(cam_distance)
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION

    max_geom = max(2048, (n_poses + 2) * int(model.ngeom) + 200)
    renderer = mujoco.Renderer(
        model, width=CANVAS_W, height=CANVAS_H, max_geom=max_geom,
    )

    data.qpos[:] = canonical_qpos(qpos[ts[0]], yaw, origin)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    first_dyn = dynamic_geom_indices(renderer.scene, 0)
    alpha0 = 0.16
    for k in first_dyn:
        g = renderer.scene.geoms[k]
        g.rgba[3] *= alpha0

    for j, t in enumerate(ts[1:], start=1):
        data.qpos[:] = canonical_qpos(qpos[t], yaw, origin)
        mujoco.mj_forward(model, data)
        start = renderer.scene.ngeom
        mujoco.mjv_addGeoms(
            model, data, opt, mujoco.MjvPerturb(),
            int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
            renderer.scene,
        )

        time_frac = j / max(n_poses - 1, 1)
        alpha = 0.16 + 0.84 * (time_frac ** 1.25)

        red_pose = done_step > 0 and j == n_poses - 1
        for k in range(start, renderer.scene.ngeom):
            g = renderer.scene.geoms[k]
            if red_pose:
                g.rgba[0] = 0.90
                g.rgba[1] = 0.06
                g.rgba[2] = 0.04
                g.rgba[3] = 1.0
            else:
                g.rgba[3] *= alpha

    frame = renderer.render()
    renderer.close()

    # Soft ground contact shadows painted in image space. Perspective side
    # camera, ground plane at world z=0. Project each pose's ground contact
    # point through the same camera matrix the renderer uses.
    fovy_rad = np.radians(CAM_FOVY_DEG)
    py_per_m_center = CANVAS_H / (2.0 * cam_distance * np.tan(fovy_rad / 2.0))
    px_per_m_center = py_per_m_center  # MuJoCo uses same focal on both axes
    shadow_layer = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    for j, t in enumerate(ts):
        q = canonical_qpos(qpos[t], yaw, origin)
        pose_x = float(q[0])
        # Project (pose_x, 0, 0) through the camera. Camera-space y axis is
        # world-y perpendicular to view; at azimuth=90 the camera looks along
        # +y, so camera forward = +y, camera depth for a point with world-y=0
        # is cam_distance.
        # World -> image: pixels = focal * (world_offset / depth). Use cam_distance
        # as nominal depth since bodies are near lookat plane.
        dx_world = pose_x - pose_mid_x
        dy_world_z = 0.0 - lookat_z  # ground z - lookat z
        px = int(CANVAS_W / 2 + dx_world * px_per_m_center)
        py = int(CANVAS_H / 2 - dy_world_z * py_per_m_center)
        time_frac = j / max(len(ts) - 1, 1)
        alpha_shadow = 0.03 + 0.11 * (time_frac ** 1.25)
        if done_step > 0 and j == len(ts) - 1:
            alpha_shadow = 0.14

        sigma_x = max(8, int(0.38 * px_per_m_center))
        sigma_y = max(3, int(0.10 * py_per_m_center))
        y_grid, x_grid = np.ogrid[:CANVAS_H, :CANVAS_W]
        dx = (x_grid - px) / max(sigma_x, 1)
        dy = (y_grid - py) / max(sigma_y, 1)
        blob = np.exp(-0.5 * (dx * dx + dy * dy))
        shadow_layer = np.maximum(shadow_layer, blob * alpha_shadow)

    shadow_mask = shadow_layer[..., None]
    bg = np.zeros_like(frame, dtype=np.float32)
    frame_f = frame.astype(np.float32)
    composed = frame_f * (1.0 - shadow_mask) + bg * shadow_mask
    frame = np.ascontiguousarray(np.clip(composed, 0, 255).astype(np.uint8))
    return frame, dist_x, done_step


def draw_panel(ax, panel, dist_x, done_step, title=None) -> None:
    ax.imshow(panel)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title is not None:
        ax.text(
            0.010, 0.93, title,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=15.0, color=TEXT, weight="semibold",
        )
    if done_step > 0:
        ax.text(
            0.990, 0.93, f"fell, step {done_step}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=14.0, color=TEXT, weight="semibold",
        )


def build_hero(panels, body, out_path) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(10.5, 3.5),
        gridspec_kw={"hspace": 0.05},
    )
    for row, proto in enumerate(PROTOCOLS):
        ax = axes[row]
        p = panels[(body, proto)]
        draw_panel(ax, p["panel"], p["dist"], p["done_step"],
                   title=PROTOCOL_TITLES[proto])
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_path}")


def build_hero_pair(panels, bodies, out_path, protocols=None) -> None:
    """Hero figure showing two bodies under N protocols.

    Shows more than one body, because bodies in
    the same family respond differently to the same protocol, so the reader
    sees that 'embodied' means body-specific, not family-uniform. Block
    structure: bodies[0] under all protocols, then a thin separator,
    then bodies[1] under all protocols.

    Pass protocols=("tau0p20", "tau0p00") for a compact 4-row Standard vs
    Reset-free contrast (the headline pair); default uses all three.
    """
    use_protos = protocols if protocols is not None else PROTOCOLS
    n_proto = len(use_protos)
    n_rows = len(bodies) * n_proto
    per_body_h = 1.15 * n_proto + 0.3
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(10.5, per_body_h * len(bodies) + 0.4 * (len(bodies) - 1)),
        gridspec_kw={"hspace": 0.05},
    )
    row_idx = 0
    for b_i, body in enumerate(bodies):
        for proto in use_protos:
            ax = axes[row_idx]
            p = panels[(body, proto)]
            title = PROTOCOL_TITLES[proto]
            if proto == use_protos[0]:
                title = f"{BODY_LABELS.get(body, body)}: {title}"
            draw_panel(ax, p["panel"], p["dist"], p["done_step"], title=title)
            row_idx += 1
        if b_i < len(bodies) - 1:
            sep = axes[row_idx - 1]
            sep.axhline(
                y=sep.get_ylim()[0], color="#cfcaba", linewidth=0.8,
                linestyle="--", alpha=0.7, clip_on=False,
            )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_path}")


def build_3x3(panels, out_path) -> None:
    n_rows = len(APPENDIX_BODIES) * len(PROTOCOLS)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(10.5, 10.5 * n_rows / 9.0),
        gridspec_kw={"hspace": 0.05},
    )
    row_idx = 0
    for body in APPENDIX_BODIES:
        for proto in PROTOCOLS:
            ax = axes[row_idx]
            p = panels[(body, proto)]
            draw_panel(ax, p["panel"], p["dist"], p["done_step"],
                       title=f"{BODY_LABELS[body]}, {PROTOCOL_TITLES[proto]}")
            row_idx += 1
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    panels = {}
    for body in BODIES:
        for proto in PROTOCOLS:
            panel, dist, done = render_chronophoto_panel(body, proto)
            if panel is None:
                print(f"SKIP {body} {proto}")
                continue
            panels[(body, proto)] = dict(
                panel=panel, dist=dist, done_step=done,
            )
    build_hero(panels, HERO_BODY, OUT_HERO_PDF)
    build_hero_pair(
        panels, HERO_PAIR_BODIES,
        REPO / "figures/fig_gait_chronophoto_hero_pair.pdf",
    )
    build_hero_pair(
        panels, HERO_PAIR_BODIES,
        REPO / "figures/fig_gait_chronophoto_hero_pair_compact.pdf",
        protocols=("tau0p20", "tau0p00"),
    )
    build_3x3(panels, OUT_FULL_PDF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
