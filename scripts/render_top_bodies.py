#!/usr/bin/env python3
"""Render top-k-per-protocol UNIMAL bodies to a grid figure.

Reads records/top_bodies_per_protocol.json and renders each body's
initial standing pose with MuJoCo offscreen rendering. Rows correspond to
training protocols (reset-free, episodic, strict); columns to within-protocol
rank. Bodies that appear under multiple protocols carry a coloured border.

Usage:
    MUJOCO_GL=osmesa python scripts/render_top_bodies.py          # default k=3
    MUJOCO_GL=osmesa python scripts/render_top_bodies.py --k 5    # 3x5 variant
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mujoco
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Lato", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "font.size": 9,
        "axes.titlesize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

REPO = Path(__file__).resolve().parents[1]
TOP = REPO / "records/top_bodies_per_protocol.json"


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
OUT_FIG = REPO / "figures/fig_top_bodies.pdf"
OUT_PNG_DIR = REPO / "figures/top_bodies"

RENDER_W, RENDER_H = 640, 480
CAMERA_DISTANCE_MULT = 2.2
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -12.0

PROTOCOL_ORDER = ("reset_free", "episodic", "strict")
PROTOCOL_LABELS = {
    "reset_free": r"Reset-free ($\tau=0$)",
    "episodic": r"Standard ($\tau=0.20$)",
    "strict": r"Strict ($\tau=0.50$)",
}

FAMILY_COLORS = {"floor": "#ff7f0e", "vt": "#1f77b4", "mvt": "#2ca02c"}
OVERLAP_BORDER_COLOR = "#7a1b1b"

FLOOR_COMMENT = (
    '<!-- <geom name="floor" type="plane" pos="0 0 0" '
    'size="50 50 1" material="grid"/> -->'
)
FLOOR_REPLACE = (
    '<geom name="floor" type="plane" pos="0 0 0" '
    'size="50 50 1" contype="1" conaffinity="1" '
    'rgba="0.96 0.96 0.96 1"/>'
)


def _load_model(morph: str) -> mujoco.MjModel:
    xml_path = UNIMAL_XML_DIR / f"{morph}.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"UNIMAL XML not found: {xml_path}")
    with xml_path.open() as f:
        xml_str = f.read()
    if FLOOR_COMMENT not in xml_str:
        raise ValueError(
            f"{morph}: expected commented floor marker missing from XML; "
            "the render cannot silently fall back to no-floor scenes."
        )
    xml_str = xml_str.replace(FLOOR_COMMENT, FLOOR_REPLACE)
    return mujoco.MjModel.from_xml_string(xml_str)


def _render_side_view(model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    z = float(data.qpos[2])
    extent = max(z, 1.0) * CAMERA_DISTANCE_MULT
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = np.array([float(data.qpos[0]), float(data.qpos[1]), z * 0.45])
    cam.distance = extent
    cam.azimuth = CAMERA_AZIMUTH
    cam.elevation = CAMERA_ELEVATION
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    try:
        renderer.update_scene(data, camera=cam)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        image = renderer.render()
    finally:
        renderer.close()
    return image


def _pad_to_aspect(image: np.ndarray, target_aspect: float = 4 / 3) -> np.ndarray:
    """Pad the image with white so its aspect ratio matches target (w/h)."""
    h, w = image.shape[:2]
    cur = w / h
    if cur < target_aspect:
        new_w = int(round(h * target_aspect))
        pad = new_w - w
        left = pad // 2
        padding = np.ones((h, pad, image.shape[2]), dtype=image.dtype) * 255
        return np.concatenate(
            [padding[:, :left], image, padding[:, left:]], axis=1
        )
    if cur > target_aspect:
        new_h = int(round(w / target_aspect))
        pad = new_h - h
        top = pad // 2
        padding = np.ones((pad, w, image.shape[2]), dtype=image.dtype) * 255
        return np.concatenate(
            [padding[:top], image, padding[top:]], axis=0
        )
    return image


def _crop_to_robot(image: np.ndarray, pad_frac: float = 0.08) -> np.ndarray:
    """Crop sky above and empty ground to the sides of the robot.

    Detect non-background pixels (anything that isn't the light grey floor or the
    cyan sky gradient), then pad by `pad_frac` of the bounding box size on each
    side, clamped to image bounds.
    """
    gray = image.mean(axis=-1)
    blue = image[..., 2].astype(np.int16) - image[..., 0].astype(np.int16)
    non_floor = gray < 240
    non_sky = blue < 25
    mask = non_floor & non_sky
    if not mask.any():
        return image
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    h, w = image.shape[:2]
    pad_r = int((r1 - r0) * pad_frac)
    pad_c = int((c1 - c0) * pad_frac)
    r0 = max(0, r0 - pad_r)
    r1 = min(h, r1 + pad_r)
    c0 = max(0, c0 - pad_c)
    c1 = min(w, c1 + pad_c)
    return image[r0:r1, c0:c1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3, help="bodies per protocol (3 or 5)")
    args = parser.parse_args()

    if not TOP.is_file():
        print(f"ERROR: top-bodies JSON not found at {TOP}.", file=sys.stderr)
        return 1

    with TOP.open() as f:
        top = json.load(f)

    stored_k = top["k"]
    if args.k > stored_k:
        raise ValueError(
            f"requested k={args.k} exceeds stored k={stored_k} in {TOP}"
        )

    OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)

    # Recompute in-view overlap count at the requested k (bodies that actually
    # appear inside this figure's grid) so the highlight reflects what the
    # reader sees, not what sits offscreen at higher ranks.
    visible = {}
    for proto in PROTOCOL_ORDER:
        for entry in top["top_by_protocol"][proto][: args.k]:
            m = entry["morph"]
            visible[m] = visible.get(m, 0) + 1

    n_rows, n_cols = 3, args.k
    if args.k == 3:
        cell_w, cell_h = 2.4, 1.8
    else:
        cell_w, cell_h = 1.7, 1.35
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_w * n_cols + 0.9, cell_h * n_rows + 0.9),
        gridspec_kw={"wspace": 0.06, "hspace": 0.14},
    )
    if n_cols == 1:
        axes = np.array([[a] for a in axes])

    for row, proto in enumerate(PROTOCOL_ORDER):
        entries = top["top_by_protocol"][proto][: args.k]
        for col, entry in enumerate(entries):
            ax = axes[row, col]
            morph = entry["morph"]
            family = entry["family"]
            psr = entry[f"psr_{proto}"]
            try:
                model = _load_model(morph)
                img = _render_side_view(model)
                img = _crop_to_robot(img, pad_frac=0.05)
                img = _pad_to_aspect(img, target_aspect=4 / 3)
                ax.imshow(img)
                png_path = OUT_PNG_DIR / f"{proto}_rank{entry['rank']}_{morph}.png"
                plt.imsave(png_path, img)
            except Exception as exc:
                print(f"ERROR rendering {morph}: {exc}", file=sys.stderr)
                ax.text(
                    0.5,
                    0.5,
                    f"RENDER FAIL\n{morph}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            fam_color = FAMILY_COLORS.get(family, "#333333")
            for spine in ax.spines.values():
                spine.set_edgecolor(fam_color)
                spine.set_linewidth(1.8)
            ax.set_xlabel(
                f"PSR {psr:.2f}",
                fontsize=8,
                color=fam_color,
                labelpad=2,
            )
            if visible.get(morph, 0) >= 2:
                rect = patches.Rectangle(
                    (0.02, 0.02),
                    0.96,
                    0.96,
                    transform=ax.transAxes,
                    linewidth=2.2,
                    edgecolor=OVERLAP_BORDER_COLOR,
                    facecolor="none",
                    linestyle=(0, (3, 2)),
                )
                ax.add_patch(rect)
        row_label = PROTOCOL_LABELS[proto]
        axes[row, 0].set_ylabel(row_label, fontsize=10.5, labelpad=8)

    for col in range(n_cols):
        axes[0, col].set_title(f"Rank {col + 1}", fontsize=10, pad=4)

    legend_handles = [
        patches.Patch(color=FAMILY_COLORS["floor"], label="Floor"),
        patches.Patch(color=FAMILY_COLORS["vt"], label="VT"),
        patches.Patch(color=FAMILY_COLORS["mvt"], label="MVT"),
        patches.Patch(
            facecolor="none",
            edgecolor=OVERLAP_BORDER_COLOR,
            linewidth=2.2,
            linestyle="--",
            label=f"top-{args.k} under ≥2 protocols",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=9,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.08)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")
    n = sum(len(top["top_by_protocol"][p][: args.k]) for p in PROTOCOL_ORDER)
    print(f"Wrote {n} PNGs to {OUT_PNG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
