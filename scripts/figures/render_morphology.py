#!/usr/bin/env python3
"""Render the 9 fixed UNIMAL bodies to a single paper gallery figure.

Loads records/visual_bodies_frozen.json (produced by
scripts/pick_visual_bodies.py), renders each body at its standing initial
pose using MuJoCo offscreen rendering, and composes the 9 renders into a
3 x 3 grid in figures/fig_morph_gallery.pdf with family + percentile
labels.

Also writes individual PNGs to figures/morph_gallery/ for the
supplementary material.

Environment:
  MUJOCO_GL must be set to a valid offscreen backend. On GPU nodes:
    MUJOCO_GL=egl
  On CPU-only machines:
    MUJOCO_GL=osmesa

Usage:
    MUJOCO_GL=osmesa python scripts/figures/render_morphology.py
    MUJOCO_GL=egl python scripts/figures/render_morphology.py  # on GPU node
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import FAMILY_COLOR, FAMILY_LABEL, TEXT, apply_style, column_header  # noqa: E402

apply_style()

REPO = Path(__file__).resolve().parents[2]
VISUAL_BODY_LIST = REPO / "records/visual_bodies_frozen.json"


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


UNIMAL_XML_DIR = default_unimal_xml_dir()
OUT_FIG = REPO / "figures/fig_morph_gallery.pdf"
OUT_PNG_DIR = REPO / "figures/morph_gallery"

RENDER_W, RENDER_H = 640, 480
CAMERA_DISTANCE_MULT = 3.5
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -20.0
FAMILIES = ("floor", "vt", "mvt")
PERCENTILES = ("25", "50", "75")


def _load_model(morph: str) -> mujoco.MjModel:
    xml_path = UNIMAL_XML_DIR / f"{morph}.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(
            f"UNIMAL XML not found for {morph}: {xml_path}. "
            "Set RFC_UNIMAL_XML_DIR or ensure bundled assets are present."
        )
    with xml_path.open() as f:
        xml_str = f.read()
    xml_str = xml_str.replace(
        '<texture name="skybox" type="skybox" builtin="flat" '
        'rgb1="0.8 1 1" rgb2="0.8 1 1" width="800" height="800"/>',
        "",
        1,
    )
    ground_assets = """
      <texture name="_rfc_sky" type="skybox" builtin="gradient" rgb1="0.92 0.96 1.00" rgb2="0.40 0.60 0.80" width="256" height="256"/>
      <texture name="_rfc_grid_tex" type="2d" builtin="checker" width="600" height="600" rgb1="0.38 0.48 0.58" rgb2="0.22 0.30 0.40" mark="edge" markrgb="0.88 0.90 0.92"/>
      <material name="_rfc_grid_mat" texture="_rfc_grid_tex" texrepeat="8 8" emission="0.10" specular="0" shininess="0" reflectance="0"/>
    """
    xml_str = xml_str.replace("</mujoco>", f"<asset>{ground_assets}</asset></mujoco>", 1)
    xml_str = xml_str.replace(
        '<!-- <geom name="floor" type="plane" pos="0 0 0" '
        'size="50 50 1" material="grid"/> -->',
        '<geom name="floor" type="plane" pos="0 0 0" '
        'size="50 50 1" contype="1" conaffinity="1" '
        'material="_rfc_grid_mat"/>',
    )
    return mujoco.MjModel.from_xml_string(xml_str)


def _render_side_view(model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Auto-fit camera to scene extent
    z = float(data.qpos[2])
    extent = max(z, 1.0) * CAMERA_DISTANCE_MULT

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = np.array([float(data.qpos[0]), float(data.qpos[1]), z * 0.5])
    cam.distance = extent
    cam.azimuth = CAMERA_AZIMUTH
    cam.elevation = CAMERA_ELEVATION

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    try:
        renderer.update_scene(data, camera=cam)
        image = renderer.render()
    finally:
        renderer.close()
    return image


def main() -> int:
    if not VISUAL_BODY_LIST.is_file():
        print(
            f"ERROR: visual body list not found at {VISUAL_BODY_LIST}. "
            f"Run scripts/pick_visual_bodies.py first.",
            file=sys.stderr,
        )
        return 1

    with VISUAL_BODY_LIST.open() as f:
        body_list = json.load(f)

    OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3, 3, figsize=(7.0, 5.45),
        gridspec_kw={"wspace": 0.04, "hspace": 0.14},
    )
    failed = []
    for row, fam in enumerate(FAMILIES):
        for col, pct in enumerate(PERCENTILES):
            morph = body_list[fam]["picks"][pct]
            delta = body_list[fam]["delta_resets"][pct]
            try:
                model = _load_model(morph)
                img = _render_side_view(model)
                ax = axes[row, col]
                ax.imshow(img)
                png_path = OUT_PNG_DIR / f"{fam}_P{pct}_{morph}.png"
                plt.imsave(png_path, img)
            except Exception as exc:
                print(f"ERROR rendering {morph}: {exc}", file=sys.stderr)
                failed.append(morph)
                ax = axes[row, col]
                ax.text(
                    0.5,
                    0.5,
                    f"RENDER FAIL\n{morph}\n{type(exc).__name__}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(FAMILY_COLOR[fam])
                spine.set_linewidth(0.9)
            label = f"{FAMILY_LABEL[fam]}  $\\Delta_{{\\mathrm{{reset}}}}={delta:+.2f}$"
            ax.text(
                0.02,
                0.97,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
                color=TEXT,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.8),
            )
    for col, pct in enumerate(PERCENTILES):
        column_header(axes[0, col], f"P{pct}", dy=1.03, size=8.8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02)
    if failed:
        plt.close(fig)
        print(f"ERROR: {len(failed)} of 9 renders failed ({', '.join(failed)}); {OUT_FIG} not written",
              file=sys.stderr)
        return 1
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {len(FAMILIES) * len(PERCENTILES)} PNGs to {OUT_PNG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
