#!/usr/bin/env python3
"""Assemble the 3x3 temporal-gait figure from qpos traces + 3D morph stills.

Per-cell: top-down (x,y) torso trajectory viridis-coloured by rollout time,
with a rendered 3D side-view of the initial pose inset in the upper-right
corner so the reader can see the morphology. Rows = body, columns =
training protocol.

Consumes:
    gait-traces/{body}_{tau}.traces.npz   (qpos_trace, action_trace)

Renders on first run (cached afterwards):
    figures/gait-stills/{body}_morph3d.png  (optional stills, unused by main())

Produces:
    figures/fig_gait_temporal.pdf
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import mujoco
import numpy as np
from matplotlib.lines import Line2D

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
OUT_PDF = REPO / "figures/fig_gait_temporal.pdf"
MORPH_SIZE_PX = 480

# Per-body camera tuned once for legibility and then fixed for the paper figure.
MORPH_CAM = {
    "floor-5506-10-6-01-15-48-35": dict(azimuth=135, elevation=-22, distance=2.4, lookat_z=0.30),
    "mvt-5506-13-10-17-12-25-45":  dict(azimuth=125, elevation=-18, distance=2.6, lookat_z=0.40),
    "floor-1409-0-7-01-15-15-26":  dict(azimuth=135, elevation=-22, distance=2.4, lookat_z=0.30),
}

BODIES = (
    "floor-5506-10-6-01-15-48-35",
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

# Default to the lower-right empty corner; Floor A / Standard ends there, so
# that one panel uses the upper-right corner instead.
ANNO_POS = {
    ("floor-5506-10-6-01-15-48-35", "tau0p20"): (0.965, 0.95, "right", "top"),
}
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import DEEMPH, GATE_RED, TEXT, apply_style, column_header, light_grid

apply_style()

CMAP_NAME = "cividis"
TIME_CMAP = mpl.colormaps[CMAP_NAME]
START_COLOR = TIME_CMAP(0.02)
END_COLOR = TIME_CMAP(0.98)
MARKER_STROKE = [pe.withStroke(linewidth=1.1, foreground="white")]


def family_of(morph: str) -> str:
    if morph.startswith("floor"):
        return "floor"
    if morph.startswith("mvt"):
        return "mvt"
    if morph.startswith("vt"):
        return "vt"
    raise ValueError(morph)


def ensure_morph_still(body: str) -> Path:
    """Render a 3D still of the body at its initial pose, cache to disk.

    Per-body camera from MORPH_CAM. Skybox post-processed to transparent.
    """
    out_path = REPO / "figures" / "gait-stills" / f"{body}_morph3d.png"
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(XML_DIR / f"{body}.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, width=MORPH_SIZE_PX, height=MORPH_SIZE_PX)
    params = MORPH_CAM.get(body, dict(azimuth=135, elevation=-22, distance=2.4, lookat_z=0.30))
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.0, 0.0, params["lookat_z"]])
    cam.distance = params["distance"]
    cam.azimuth = params["azimuth"]
    cam.elevation = params["elevation"]
    renderer.update_scene(data, camera=cam)
    img = renderer.render().astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    sky_mask = (b > 180) & (g > 180) & (r < g)
    rgba = np.concatenate([img, 255.0 * np.ones_like(r[..., None])], axis=-1)
    rgba[sky_mask, 3] = 0.0
    rgba = np.clip(rgba, 0, 255).astype(np.uint8)
    renderer.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_path), rgba)
    return out_path


def load_cell(body: str, proto: str) -> dict | None:
    stem = GAIT_DIR / f"{body}_{proto}"
    traces_path = stem.with_suffix(".traces.npz")
    if not traces_path.exists():
        return None
    d = np.load(traces_path)
    qpos = d["qpos_trace"]
    actions = d["action_trace"]
    done_step = int(d["done_step"]) if "done_step" in d.files else -1
    if done_step > 0:
        end_idx = min(done_step, len(qpos) - 1)
        xy = qpos[: end_idx + 1, :2]
        dist_x = float(qpos[end_idx, 0] - qpos[0, 0])
        actions_used = actions[:end_idx]
    else:
        xy = qpos[:, :2]
        dist_x = float(qpos[-1, 0] - qpos[0, 0]) if len(qpos) > 1 else 0.0
        actions_used = actions
    action_l2_total = float(np.sum(actions_used**2)) if actions_used.size else 0.0
    return {
        "xy": xy,
        "distance": dist_x,
        "action_l2_per_step": action_l2_total / max(len(actions_used), 1),
        "ep_steps": len(actions_used),
        "done_step": done_step,
    }


def row_limits(cells: list[dict | None]) -> tuple[float, float, float, float] | None:
    xy_all = [cell["xy"] for cell in cells if cell is not None and len(cell["xy"]) >= 2]
    if not xy_all:
        return None
    xy = np.concatenate(xy_all, axis=0)
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    xspan = max(float(xmax - xmin), 1e-6)
    yspan = max(float(ymax - ymin), 1e-6)
    xpad = max(0.5, 0.04 * xspan)
    ypad = max(0.5, 0.08 * yspan)
    return float(xmin - xpad), float(xmax + xpad), float(ymin - ypad), float(ymax + ypad)


def draw_cell(
    ax,
    cell: dict | None,
    body: str,
    proto: str,
    limits: tuple[float, float, float, float] | None = None,
) -> None:
    if cell is None or len(cell["xy"]) < 2:
        ax.text(
            0.5, 0.5, "render pending",
            ha="center", va="center", fontsize=7, color="#888",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return
    xy = cell["xy"]
    t = np.arange(len(xy)) / max(len(xy) - 1, 1)
    ax.plot(xy[:, 0], xy[:, 1], color="#888", alpha=0.6, linewidth=0.9, zorder=2)
    ax.scatter(xy[:, 0], xy[:, 1], c=t, cmap=CMAP_NAME, s=6, zorder=3, edgecolor="none")
    start_marker = ax.scatter(
        xy[0, 0], xy[0, 1],
        color=START_COLOR,
        s=30,
        marker="o",
        zorder=5,
        edgecolor="white",
        linewidth=0.7,
        label="start",
    )
    start_marker.set_path_effects(MARKER_STROKE)
    end_marker = ax.scatter(
        xy[-1, 0], xy[-1, 1],
        color=END_COLOR,
        s=32,
        marker="D",
        zorder=6,
        edgecolor="black",
        linewidth=0.7,
        label="end",
    )
    end_marker.set_path_effects(MARKER_STROKE)
    if cell.get("done_step") is not None and cell["done_step"] >= 0:
        done_idx = min(cell["done_step"], len(xy) - 1)
        fall_marker = ax.scatter(
            xy[done_idx, 0], xy[done_idx, 1],
            color=GATE_RED,
            s=44,
            marker="x",
            zorder=7,
            linewidths=1.35,
            label="fall",
        )
        fall_marker.set_path_effects(MARKER_STROKE)
    if limits is not None:
        xmin, xmax, ymin, ymax = limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    anno_x, anno_y, anno_ha, anno_va = ANNO_POS.get(
        (body, proto), (0.965, 0.055, "right", "bottom")
    )
    ax.text(
        anno_x,
        anno_y,
        rf"$\Delta x$={cell['distance']:.1f} m" + "\n" + rf"$T$={cell['ep_steps']}",
        transform=ax.transAxes,
        ha=anno_ha,
        va=anno_va,
        fontsize=7.2,
        color=TEXT,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.2),
        zorder=6,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(DEEMPH)
        spine.set_linewidth(0.85)
    ax.tick_params(axis="both", which="major", labelsize=7.5)
    light_grid(ax, axis="both")


def draw_morph_cell(ax, body: str, morph_png: Path) -> None:
    ax.imshow(mpimg.imread(str(morph_png)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#9C9C9C")
        s.set_linewidth(0.9)
    ax.patch.set_alpha(0.0)


def main() -> int:
    # Lean 3x3 trajectory figure. Morphology is shown in the
    # chronophotographs (Figures 1 and 14), so a morph column is redundant here.
    fig, axes = plt.subplots(
        3, 3, figsize=(7.4, 5.35),
        gridspec_kw={"wspace": 0.21, "hspace": 0.20},
    )

    missing = 0
    absent = [GAIT_DIR / f"{body}_{proto}.traces.npz" for body in BODIES for proto in PROTOCOLS
              if not (GAIT_DIR / f"{body}_{proto}.traces.npz").exists()]
    if absent:
        raise SystemExit(f"missing joint traces (gait-traces/ from the dataset): {', '.join(map(str, absent))}")
    cells_by_body = {
        body: {proto: load_cell(body, proto) for proto in PROTOCOLS}
        for body in BODIES
    }
    limits_by_body = {
        body: row_limits([cells_by_body[body][proto] for proto in PROTOCOLS])
        for body in BODIES
    }
    for row, body in enumerate(BODIES):
        for col, proto in enumerate(PROTOCOLS):
            ax = axes[row, col]
            cell = cells_by_body[body][proto]
            if cell is None:
                missing += 1
            draw_cell(ax, cell, body, proto, limits_by_body[body])
            if row == 0:
                ax.set_title("")
                column_header(ax, PROTOCOL_TITLES[proto], dy=1.06)
        axes[row, 0].set_ylabel(
            f"{BODY_LABELS[body]}\nlateral $y$ [m]",
            fontsize=9.0, labelpad=5,
        )

    for col in range(3):
        axes[2, col].set_xlabel("forward $x$ [m]", fontsize=9.0)

    cax = fig.add_axes([0.915, 0.39, 0.012, 0.24])
    mpl.colorbar.ColorbarBase(
        cax, cmap=TIME_CMAP,
        norm=mpl.colors.Normalize(vmin=0, vmax=1),
        orientation="vertical",
    )
    cax.set_ylabel("Rollout time (0 start, 1 end)", fontsize=8.0, rotation=270, labelpad=12, color=TEXT)
    cax.tick_params(labelsize=7.5)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=START_COLOR,
               markeredgecolor="white", markeredgewidth=0.7, markersize=5.2, label="start"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=END_COLOR,
               markeredgecolor="black", markeredgewidth=0.7, markersize=5.2, label="end"),
        Line2D([0], [0], marker="x", linestyle="none", color=GATE_RED,
               markeredgewidth=1.1, markersize=5.2, label="fall"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.903, 0.355),
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.30,
        labelspacing=0.22,
        fontsize=6.8,
    )

    fig.subplots_adjust(left=0.085, right=0.895, top=0.94, bottom=0.08)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {OUT_PDF} (missing cells: {missing}/9)")

    summary_path = REPO / "figures/gait_rollout_summary.json"
    summary = []
    for body in BODIES:
        for proto in PROTOCOLS:
            cell = load_cell(body, proto)
            if cell is None:
                continue
            summary.append({
                "body": body,
                "protocol": proto,
                "distance": cell["distance"],
                "ep_steps": cell["ep_steps"],
                "action_l2_per_step": cell["action_l2_per_step"],
            })
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
