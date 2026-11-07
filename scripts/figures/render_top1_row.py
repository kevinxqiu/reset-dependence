#!/usr/bin/env python3
"""Render a 1-row 3-cell figure: one most protocol-specialised VT body per protocol.

Selection rule: argmax over VT bodies of (PSR_self - max(PSR_others)). Family
pinned to VT because Floor and MVT both show Standard == Reset-free overlap when
selecting on absolute PSR.

Bodies are rendered in a single hand-set mid-stride display pose (each hinge
flexed to ~50% of its range with alternating sign) on the simulation floor.

Outputs figures/fig_top1_vt_row.pdf.

Usage:
    MUJOCO_GL=osmesa python3 scripts/figures/render_top1_row.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _render_style import FAMILY_COLOR, TEXT, apply_style, column_header  # noqa: E402

apply_style()

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "results"


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
OUT_PDF = REPO / "figures/fig_top1_vt_row.pdf"

RENDER_W, RENDER_H = 640, 480
CAMERA_DISTANCE_MULT = 2.2
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -12.0

PROTOCOL_ORDER = ("episodic", "reset_free", "strict")
PROTOCOL_LABELS = {
    "episodic": r"Standard ($\tau{=}0.20$)",
    "reset_free": r"Reset-free ($\tau{=}0$)",
    "strict": r"Strict ($\tau{=}0.50$)",
}
PROTO_TO_TAU = {"episodic": "0p20", "reset_free": "0p00", "strict": "0p50"}

VT_FAMILY_COLOR = FAMILY_COLOR["vt"]

FLOOR_COMMENT = (
    '<!-- <geom name="floor" type="plane" pos="0 0 0" '
    'size="50 50 1" material="grid"/> -->'
)
FLOOR_REPLACE = (
    '<geom name="floor" type="plane" pos="0 0 0" '
    'size="50 50 1" contype="1" conaffinity="1" '
    'material="_rfc_grid_mat"/>'
)
SKYBOX_REPLACE_FROM = (
    '<texture name="skybox" type="skybox" builtin="flat" '
    'rgb1="0.8 1 1" rgb2="0.8 1 1" width="800" height="800"/>'
)
GROUND_ASSETS = (
    '<texture name="_rfc_sky" type="skybox" builtin="gradient" '
    'rgb1="0.92 0.96 1.00" rgb2="0.40 0.60 0.80" width="256" height="256"/>'
    '<texture name="_rfc_grid_tex" type="2d" builtin="checker" width="600" height="600" '
    'rgb1="0.38 0.48 0.58" rgb2="0.22 0.30 0.40" mark="edge" markrgb="0.88 0.90 0.92"/>'
    '<material name="_rfc_grid_mat" texture="_rfc_grid_tex" texrepeat="8 8" '
    'emission="0.10" specular="0" shininess="0" reflectance="0"/>'
)


def _load_psr_table() -> dict:
    seeds = ["42", "43", "44"]
    psr = defaultdict(lambda: defaultdict(list))
    fams = {}
    for proto, tau in PROTO_TO_TAU.items():
        for seed in seeds:
            path = RESULTS_DIR / f"mjx_ppo_{tau}_seed{seed}/results.json"
            if not path.is_file():
                continue
            with path.open() as f:
                d = json.load(f)
            for morph, body in d.items():
                if "eval_tau020" not in body:
                    continue
                ev = body["eval_tau020"]
                v = ev.get("per_step_reward") if isinstance(ev, dict) else None
                if v is None or v != v:
                    continue
                psr[morph][proto].append(float(v))
                fams[morph] = morph.split("-")[0]
    mean_psr = {
        m: {p: sum(vs) / len(vs) for p, vs in by_p.items() if vs}
        for m, by_p in psr.items()
    }
    return {"mean_psr": mean_psr, "fams": fams}


def _vt_top1_per_protocol(table: dict) -> dict:
    """Most protocol-specialised VT body per protocol.

    For each protocol p, pick the VT body that maximises
    PSR_p - max_{q != p} PSR_q. This produces bodies whose performance is
    sharply concentrated at p, which gives wider visual contrast than
    rank-1-by-PSR while keeping the family pinned.
    """
    fams = table["fams"]
    mean_psr = table["mean_psr"]
    vt_bodies = [m for m, f in fams.items() if f == "vt" and m in mean_psr]
    out = {}
    for proto in PROTOCOL_ORDER:
        others = [p for p in PROTOCOL_ORDER if p != proto]
        scored = []
        for m in vt_bodies:
            p = mean_psr[m]
            if not all(o in p for o in [proto] + others):
                continue
            spec = p[proto] - max(p[others[0]], p[others[1]])
            scored.append((spec, m))
        scored.sort(reverse=True)
        if not scored:
            raise RuntimeError(f"No VT body scored under {proto}")
        out[proto] = scored[0][1]
    return out


def _load_model(morph: str) -> mujoco.MjModel:
    xml_path = UNIMAL_XML_DIR / f"{morph}.xml"
    with xml_path.open() as f:
        xml_str = f.read()
    if FLOOR_COMMENT not in xml_str:
        raise ValueError(f"{morph}: floor marker missing")
    xml_str = xml_str.replace(SKYBOX_REPLACE_FROM, "", 1)
    xml_str = xml_str.replace(
        "</mujoco>", f"<asset>{GROUND_ASSETS}</asset></mujoco>", 1
    )
    xml_str = xml_str.replace(FLOOR_COMMENT, FLOOR_REPLACE)
    return mujoco.MjModel.from_xml_string(xml_str)


def _set_display_stance(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Hand-tuned mid-stride display pose: each hinge to ~50% of range, sign alternating."""
    sign = 1.0
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
        if lo == 0.0 and hi == 0.0:
            continue
        if abs(lo) >= abs(hi):
            target = lo * 0.5
        else:
            target = hi * 0.5
        if sign < 0:
            target = -target
            target = max(lo, min(hi, target))
        sign *= -1.0
        data.qpos[model.jnt_qposadr[j]] = target
    data.qpos[2] = float(data.qpos[2]) + 0.05


def _render(morph: str) -> np.ndarray:
    model = _load_model(morph)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    _set_display_stance(model, data)
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


def _crop_around_robot(image: np.ndarray) -> np.ndarray:
    """Crop horizontally to the robot, but keep a generous strip of floor and
    sky vertically so the simulation context (gradient skybox + grid floor)
    remains visible.

    Robot pixels are detected via the warm beige body colour (high red, low
    blue), distinct from the cool blue/grey background.
    """
    r = image[..., 0].astype(np.int16)
    g = image[..., 1].astype(np.int16)
    b = image[..., 2].astype(np.int16)
    # Body pixels: red dominant, low blue.
    body_mask = (r > 150) & (r > b + 30) & (g > 80)
    if not body_mask.any():
        return image
    h, w = image.shape[:2]
    cols = np.where(body_mask.any(axis=0))[0]
    c0, c1 = cols.min(), cols.max()
    pad_c = int((c1 - c0) * 0.20)
    c0 = max(0, c0 - pad_c)
    c1 = min(w, c1 + pad_c)
    # Vertical: take a centred band that gives roughly sky 35% / floor 35%
    # around the robot's vertical centre, capped at image bounds.
    rows = np.where(body_mask.any(axis=1))[0]
    r_center = int((rows.min() + rows.max()) / 2)
    cell_h = int((c1 - c0) * 3 / 4)  # 4:3 aspect
    r0 = max(0, r_center - cell_h // 2)
    r1 = min(h, r0 + cell_h)
    if r1 - r0 < cell_h:
        r0 = max(0, r1 - cell_h)
    return image[r0:r1, c0:c1]


def _pad_to_aspect(image: np.ndarray, target_aspect: float = 4 / 3) -> np.ndarray:
    h, w = image.shape[:2]
    cur = w / h
    if cur < target_aspect:
        new_w = int(round(h * target_aspect))
        pad = new_w - w
        left = pad // 2
        padding = np.full((h, pad, image.shape[2]), 255, dtype=image.dtype)
        return np.concatenate(
            [padding[:, :left], image, padding[:, left:]], axis=1
        )
    if cur > target_aspect:
        new_h = int(round(w / target_aspect))
        pad = new_h - h
        if pad < 0:
            return image
        top = pad // 2
        padding = np.full((pad, w, image.shape[2]), 255, dtype=image.dtype)
        return np.concatenate([padding[:top], image, padding[top:]], axis=0)
    return image


def main() -> int:
    table = _load_psr_table()
    selection = _vt_top1_per_protocol(table)
    mean_psr = table["mean_psr"]

    print("VT-family top-1 per protocol:")
    for proto in PROTOCOL_ORDER:
        m = selection[proto]
        p = mean_psr[m]
        print(
            f"  {proto:<11} -> {m}  "
            f"Std={p.get('episodic', float('nan')):.2f} "
            f"RF={p.get('reset_free', float('nan')):.2f} "
            f"Str={p.get('strict', float('nan')):.2f}"
        )

    # Sanity: 3 distinct bodies expected.
    if len({selection[p] for p in PROTOCOL_ORDER}) != 3:
        raise RuntimeError(
            "Expected 3 distinct VT bodies; data has changed: " f"{selection}"
        )

    rendered = {m: _crop_around_robot(_render(m)) for m in selection.values()}

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.68))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.84, bottom=0.22, wspace=0.08)

    for ax, proto in zip(axes, PROTOCOL_ORDER):
        morph = selection[proto]
        ax.imshow(rendered[morph])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(VT_FAMILY_COLOR)
            spine.set_linewidth(1.1)
        ax.set_title("")
        column_header(ax, PROTOCOL_LABELS[proto], dy=1.055)
        p = mean_psr[morph]
        cells = []
        for label, key in (("Std", "episodic"), ("RF", "reset_free"), ("Str", "strict")):
            v = p.get(key, float("nan"))
            cell_str = rf"$\bf{{{label}\,{v:.2f}}}$" if key == proto else f"{label} {v:.2f}"
            cells.append(cell_str)
        psr_line = "   ".join(cells)
        ax.set_xlabel(psr_line, fontsize=8.4, color=TEXT, labelpad=4)

    legend_handles = [
        mpatches.Patch(color=VT_FAMILY_COLOR, label="VT family (pinned)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
        fontsize=8.0,
        handlelength=1.2,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
