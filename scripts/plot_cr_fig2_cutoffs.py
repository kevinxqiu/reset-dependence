#!/usr/bin/env python3
"""Zero-action probe plot for Appendix C.1 (not a paper figure): where the fall cutoffs sit relative to each body's torso.

Both panels come from CPU MuJoCo rollouts with zero action, starting from
the nominal reset pose (no reset noise), with the floor substitution of
scripts/train_mjx.py load_unimal_xml and frame_skip 4 (control step
0.02 s). The reset height z0 is the torso height in the reset pose, in
which every UNIMAL body's lowest point sits about 0.2 m above the floor.
Every torso is a sphere of radius 0.1 m (read from each XML).

Panel (a): torso height of one short VT cut-off body during its first 50
control steps (default vt-5506-3-17-02-18-33-45, z0 = 0.30 m), with the
Strict cutoff (0.50 z0), the Standard cutoff (0.20 z0), DERL's cutoff for
this body (0.5 (z0 - 0.2 m), DERL public code) and the torso radius. The
step at which the torso first drops below the Strict cutoff and the first
step with floor contact are marked. Insets show the body at step 0 (reset
pose) and step 50 (1 s).

Panel (b): all 100 bodies against z0, with the Standard and Strict cutoff
lines, the torso radius, and each body's torso height after 1 s (50
control steps) with zero action. Bodies that share one z0 are spread
horizontally around it so that their markers do not overlap: by at most
SPREAD_MAX_M (0.04 m), by at most half the distance to the next z0 (so
groups do not mix), and never across a cutoff line. Where a group is too
dense for that, markers touch, and filled markers have a dark edge so that
an overlap never looks like an open marker. The script prints the largest
offset it drew, which the caption should state. Where at least COLLAPSE_MIN
bodies share one z0 and one height (23 of the 25 bodies with z0 = 0.30 m
rest at 0.10 m), they are drawn as one marker per encoding class, centred on
z0 CLASS_STEP_M apart, with the number of bodies printed below it and joined
to it by a thin leader. Open
markers: the 28 bodies whose Standard cutoff lies below the torso radius
(audit flags, cross-checked against the XML geometry). Crosses: the ten VT
cut-off bodies, i.e. the VT members of the 11 bodies whose Strict-trained
runs average at most 20 steps under the Strict evaluator (three-seed mean
of eval_tau050.mean_ep_len in the recorded runs). A body in both sets (the
seven short VT cut-off bodies) is drawn as an open square with a cross, so
every one of the 28 has an open marker.

With --check-passive and --check-settle the recomputed rollouts are compared
with earlier records (all100_passive.json from the strict-cutoff verifier,
settle.json from the inert-geometry verifier). Any disagreement stops the
script.

Output (figures by default): fig_cr2_cutoffs.{pdf,png}

Usage:
    python scripts/plot_cr_fig2_cutoffs.py
    python scripts/plot_cr_fig2_cutoffs.py --results results \
        --xml-dir assets/unimals_100/xml \
        --check-passive all100_passive.json --check-settle settle.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (  # noqa: E402
    CUTOFF_LINEWIDTH,
    CUTOFF_MARKER,
    FAMILY_INK,
    FAMILY_MARKER,
    FAMILY_ORDER,
    FILLED_EDGE,
    INERT_FACECOLOR,
    PROTOCOL_COLOR,
    PROTOCOL_LINESTYLE,
    TEXT,
    TEXT_WIDTH,
    apply_style,
    baseline_px,
    save_figure,
    text_with_subscripts,
    tufte_axes,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path(os.environ.get("RFC_RESULTS", REPO / "results"))
DEFAULT_XML = Path(os.environ.get("RFC_UNIMAL_XML_DIR", REPO / "assets" / "unimals_100" / "xml"))
DEFAULT_AUDIT = REPO / "records" / "standard_threshold_audit.json"
DEFAULT_OUT = REPO / "figures"
DEFAULT_TRACE_BODY = "vt-5506-3-17-02-18-33-45"

SEEDS = (42, 43, 44)
FRAME_SKIP = 4
N_STEPS = 150          # 3 s of zero action
STEP_1S = 50           # 1 s = 50 control steps of 0.02 s
TAU_LENIENT, TAU_STRICT = 0.20, 0.50
DERL_RATIO, DERL_OFFSET = 0.5, 0.2  # DERL: done if torso height <= 0.5 (z0 - 0.2 m)
CUTOFF_MAX_STEPS = 20.0

FLOOR_COMMENT = ('<!-- <geom name="floor" type="plane" pos="0 0 0" '
                 'size="50 50 1" material="grid"/> -->')
FLOOR_ACTIVE = ('<geom name="floor" type="plane" pos="0 0 0" '
                'size="50 50 1" contype="1" conaffinity="1" '
                'rgba="0.9 0.9 0.9 1"/>')

RADIUS_INK = "#6E6E6E"
BAND_FILL = "#F1F1F1"
DERL_INK = "#7A7A7A"
DERL_STYLE = (0, (4.0, 1.5, 1.0, 1.5))


# ---------------------------------------------------------------------------
# Simulation (same XML loading as scripts/train_mjx.py load_unimal_xml)
# ---------------------------------------------------------------------------

def load_model(xml_dir: Path, body: str):
    import mujoco

    text = (xml_dir / f"{body}.xml").read_text()
    if FLOOR_COMMENT not in text:
        raise RuntimeError(f"{body}: floor comment not found")
    return mujoco.MjModel.from_xml_string(text.replace(FLOOR_COMMENT, FLOOR_ACTIVE))


def floor_contact(model, data, floor_id: int) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == floor_id or c.geom2 == floor_id) and c.dist <= 0:
            return True
    return False


def zero_action_rollout(xml_dir: Path, body: str) -> dict:
    """Torso height per control step with zero action from the reset pose."""
    import mujoco

    model = load_model(xml_dir, body)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso/0")
    if model.geom_type[torso] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise RuntimeError(f"{body}: torso geom is not a sphere")
    z0 = float(data.qpos[2])
    heights = [z0]
    cross = contact = None
    for t in range(1, N_STEPS + 1):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data, nstep=FRAME_SKIP)
        z = float(data.qpos[2])
        heights.append(z)
        if contact is None and floor_contact(model, data, floor):
            contact = t
        if cross is None and z < TAU_STRICT * z0:
            cross = t
    return {
        "z0": z0,
        "radius": float(model.geom_size[torso, 0]),
        "control_dt": float(model.opt.timestep * FRAME_SKIP),
        "heights": np.array(heights),
        "strict_cross": cross,
        "first_contact": contact,
    }


def check_against_records(roll: dict, passive_path: Path | None,
                          settle_path: Path | None) -> None:
    """Stop if the recomputed rollouts disagree with earlier records."""
    if passive_path is not None:
        with open(passive_path) as fh:
            ref = {r["body"]: r for r in json.load(fh)}
        if set(ref) != set(roll):
            raise RuntimeError("passive record covers different bodies")
        for b, r in roll.items():
            h = r["heights"]
            cross60 = r["strict_cross"] if (r["strict_cross"] or 999) <= 60 else None
            checks = [
                abs(ref[b]["z0"] - r["z0"]) < 1e-9,
                ref[b]["zero_action_first_cross_step"] == cross60,
                ref[b]["zero_action_first_floor_contact_step"] == r["first_contact"],
                abs(ref[b]["zero_action_height_step50"] - round(h[50], 4)) < 1e-4,
                abs(ref[b]["zero_action_min_height_60"] - round(h[:61].min(), 4)) < 1e-4,
                abs(ref[b]["our_strict_cutoff"] - TAU_STRICT * r["z0"]) < 1e-9,
            ]
            if not all(checks):
                raise RuntimeError(f"{b}: rollout disagrees with {passive_path}: {checks}")
        print(f"rollouts agree with {passive_path} for all {len(roll)} bodies "
              "(z0, Strict crossing step within 60, first contact step, "
              "height at step 50, minimum height over 60 steps)")
    if settle_path is not None:
        with open(settle_path) as fh:
            ref = {row[1]: row for row in json.load(fh)}
        for b, r in roll.items():
            if abs(ref[b][2] - r["z0"]) > 1e-9 or abs(ref[b][3] - r["heights"][N_STEPS]) > 1e-9:
                raise RuntimeError(f"{b}: 3 s height disagrees with {settle_path}")
        print(f"3 s heights agree with {settle_path} for all {len(roll)} bodies")


# ---------------------------------------------------------------------------
# Body flags from the audit and the recorded runs
# ---------------------------------------------------------------------------

def inert_flags(audit_path: Path, roll: dict) -> dict:
    with open(audit_path) as fh:
        audit = json.load(fh)["bodies"]
    flags = {}
    for b, r in roll.items():
        flag = bool(audit[b]["geometric_inert"]["0.20"])
        if (TAU_LENIENT * r["z0"] < r["radius"]) != flag:
            raise RuntimeError(f"{b}: audit inert flag disagrees with XML geometry")
        flags[b] = flag
    return flags


def cutoff_flags(results: Path, bodies: list[str]) -> dict:
    runs = []
    for s in SEEDS:
        with open(results / f"mjx_ppo_0p50_seed{s}" / "results.json") as fh:
            runs.append(json.load(fh))
    return {b: float(np.mean([run[b]["eval_tau050"]["mean_ep_len"] for run in runs]))
            <= CUTOFF_MAX_STEPS for b in bodies}


def family_of(body: str) -> str:
    fam = body.split("-")[0]
    if fam not in FAMILY_ORDER:
        raise ValueError(body)
    return fam


# ---------------------------------------------------------------------------
# Rendering of the insets (reset pose and step 50)
# ---------------------------------------------------------------------------

def render_poses(xml_dir: Path, body: str, steps=(0, STEP_1S), px=(760, 460)):
    """Images of the body at the given control steps, one fixed camera.

    Returns {step: (rgb, mask)} where mask marks the body and its shadow.
    The floor is light grey so the shadow shows the height above it. The
    background above the horizon is white.
    """
    import mujoco

    model = load_model(xml_dir, body)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_grey = 0.92
    model.geom_rgba[floor] = [floor_grey, floor_grey, floor_grey, 1.0]
    w, h = px
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.30
    model.vis.headlight.specular[:] = 0.05
    model.light_diffuse[:] = 0.6
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.13]
    cam.distance, cam.azimuth, cam.elevation = 1.55, 135.0, -9.0
    renderer = mujoco.Renderer(model, height=h, width=w)
    out, done = {}, 0
    for target in sorted(steps):
        for _ in range(target - done):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data, nstep=FRAME_SKIP)
        done = target
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().copy() / 255.0
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        sky = seg[..., 0] < 0
        is_floor = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] == floor)
        body_px = ~sky & ~is_floor
        shadow_px = is_floor & (rgb.mean(axis=2) < floor_grey - 0.10)
        rgb[sky] = 1.0
        out[target] = (rgb, body_px | shadow_px)
    renderer.close()
    return out


def crop_pair(rendered: dict, pad: int = 14) -> dict:
    """Crop all images to one window covering every body and shadow."""
    union = np.logical_or.reduce([mask for _, mask in rendered.values()])
    ys, xs = np.nonzero(union)
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad
    return {k: rgb[y0:y1, x0:x1] for k, (rgb, _) in rendered.items()}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

FIG_H = 2.55
CROSS_KW = dict(linestyle="none", marker="o", markersize=5.4,
                markerfacecolor=PROTOCOL_COLOR["strict"], markeredgecolor="white",
                markeredgewidth=0.6)
CONTACT_KW = dict(linestyle="none", marker="v", markersize=5.4,
                  markerfacecolor=TEXT, markeredgecolor="white", markeredgewidth=0.6)
COLLAPSE_MIN = 7  # coincident bodies drawn as one marker with a count
SPREAD_MAX_M = 0.04   # largest horizontal offset of a body from its own z0
MERGE_M = 0.015       # reset heights closer than this are spread as one group
CLASS_STEP_M = 0.025  # spacing of the class markers of a crowded group
COUNT_STEP_M = 0.05   # spacing of their counts (joined to the markers by leaders)
COUNT_Y = 0.018       # height of the count row (m)
LEADER_INK = "#8C8C8C"
AX_A = (0.47, 0.44, 2.02, 1.55)   # x, y, width, height in inches
AX_B = (3.04, 0.44, 2.41, 1.55)
POINT_SIZE = 11.0                 # scatter area, pt^2
SUB_PT = 7.0                      # subscript size: the house minimum


def z0_pieces(before: str, after: str, size: float, **kw) -> list:
    """Pieces for text_with_subscripts: ``before``, italic z, subscript 0, ``after``.

    Mathtext would print the subscript at 70% of ``size`` (5.25 pt for 7.5 pt
    text), below the 7 pt minimum.
    """
    plain = {k: v for k, v in kw.items() if k != "fontweight"}
    return [(before, size, 0.0, kw),
            ("z", size, 0.0, dict(plain, fontstyle="italic")),
            ("0", SUB_PT, 0.24 * size, plain),
            (after, size, 0.0, kw)]


def inch_axes(fig, x, y, w, h, **kw):
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh], **kw)


def line_angle(ax, slope: float, x: float, y: float) -> float:
    """Screen angle (degrees) of a data-space line with the given slope."""
    ang = np.degrees(np.arctan(slope))
    return float(ax.transData.transform_angles(np.array([ang]), np.array([[x, y]]))[0])


def draw_trace(ax, r, cutoffs, insets):
    steps = np.arange(STEP_1S + 1)
    h = r["heights"][: STEP_1S + 1]
    ax.set_xlim(-1.0, STEP_1S + 0.5)
    ax.set_ylim(0.0, 0.32)
    ax.axhspan(0.0, r["radius"], color=BAND_FILL, zorder=0, linewidth=0)
    ax.axhline(r["radius"], color=RADIUS_INK, linewidth=0.9, zorder=1)
    ax.axhline(cutoffs["strict"], color=PROTOCOL_COLOR["strict"],
               linestyle=PROTOCOL_LINESTYLE["strict"], linewidth=1.5, zorder=2)
    ax.axhline(cutoffs["lenient"], color=PROTOCOL_COLOR["lenient"],
               linestyle=PROTOCOL_LINESTYLE["lenient"], linewidth=1.3, zorder=2)
    ax.axhline(cutoffs["derl"], color=DERL_INK, linestyle=DERL_STYLE,
               linewidth=1.1, zorder=2)
    ax.plot(steps, h, color=TEXT, linewidth=1.4, zorder=4, solid_joinstyle="round")
    ax.plot(steps, h, linestyle="none", marker="o", markersize=1.7, color=TEXT, zorder=4)
    xr = STEP_1S
    kw = dict(ha="right", fontsize=7.5, color=TEXT, zorder=5)
    ax.text(xr, cutoffs["strict"] + 0.005, f"Strict cutoff {cutoffs['strict']:.2f} m",
            va="bottom", **kw)
    ax.text(xr, r["radius"] + 0.005, f"torso radius {r['radius']:.2f} m",
            va="bottom", **kw)
    ax.text(xr, cutoffs["lenient"] + 0.004, f"Standard cutoff {cutoffs['lenient']:.2f} m",
            va="bottom", **kw)
    ax.text(xr, cutoffs["derl"] - 0.005, f"DERL cutoff {cutoffs['derl']:.2f} m",
            va="top", **kw)
    c, k = r["strict_cross"], r["first_contact"]
    ax.plot(c, h[c], zorder=6, **CROSS_KW)
    ax.plot(k, h[k], zorder=6, **CONTACT_KW)
    # Insets in the empty upper right: the body at step 0 and at step 50,
    # stacked, sized to the space above the Strict label.
    img0, img1 = insets[0], insets[STEP_1S]
    aspect = img0.shape[1] / img0.shape[0]
    bbox = ax.get_position()
    ax_w_in = bbox.width * ax.figure.get_figwidth()
    ax_h_in = bbox.height * ax.figure.get_figheight()
    y_floor = (cutoffs["strict"] + 0.034) / 0.32  # axes fraction just above the Strict label
    gap = 0.05 / ax_h_in
    h_frac = (0.995 - y_floor - gap) / 2
    w_frac = h_frac * ax_h_in * aspect / ax_w_in
    x_frac = 0.995 - w_frac
    for i, (img, label) in enumerate(((img0, "step 0"), (img1, f"step {STEP_1S}"))):
        y_frac = 0.995 - (i + 1) * h_frac - i * gap
        ax_i = ax.inset_axes([x_frac, y_frac, w_frac, h_frac])
        ax_i.imshow(img, interpolation="lanczos")
        ax_i.set_axis_off()
        ax.text(x_frac - 0.015, y_frac + h_frac / 2, label, transform=ax.transAxes,
                ha="right", va="center", fontsize=7.0, color=TEXT)
    ax.set_xlabel(f"Control step ({r['control_dt']:.2f} s), zero action")
    ax.set_ylabel("Torso height (m)")
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_yticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])
    tufte_axes(ax)


def event_legend(fig, r, x, y):
    c, k = r["strict_cross"], r["first_contact"]
    leg = fig.legend(
        [Line2D([], [], **CROSS_KW), Line2D([], [], **CONTACT_KW)],
        [f"Torso below the Strict cutoff (step {c})", f"First floor contact (step {k})"],
        loc="upper left", bbox_to_anchor=(x, y), frameon=False, fontsize=7.5,
        handlelength=1.2, handletextpad=0.5, labelspacing=0.42, borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(TEXT)


def draw_population(ax, roll, fam, inert, vt_cut):
    radius = {r["radius"] for r in roll.values()}
    if len(radius) != 1:
        raise RuntimeError(f"torso radii differ: {radius}")
    radius = radius.pop()
    xlo, xhi, ytop = 0.2, 1.1, 0.82
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0.0, ytop)
    ax.axhspan(0.0, radius, color=BAND_FILL, zorder=0, linewidth=0)
    ax.axhline(radius, color=RADIUS_INK, linewidth=0.9, zorder=1)
    xs = np.array([xlo, xhi])
    ax.plot(xs, TAU_STRICT * xs, color=PROTOCOL_COLOR["strict"],
            linestyle=PROTOCOL_LINESTYLE["strict"], linewidth=1.5, zorder=2)
    ax.plot(xs, TAU_LENIENT * xs, color=PROTOCOL_COLOR["lenient"],
            linestyle=PROTOCOL_LINESTYLE["lenient"], linewidth=1.3, zorder=2)
    # Line labels sit along the lines where no body is plotted. They are
    # lifted by the font descent so they sit as a va="bottom" label would.
    xs_lab = 0.215
    text_with_subscripts(ax, xs_lab, TAU_STRICT * xs_lab + 0.03,
                         z0_pieces("Strict cutoff 0.50 ", "", 7.5),
                         rotation=line_angle(ax, TAU_STRICT, xs_lab, TAU_STRICT * xs_lab),
                         raise_pt=1.6)
    xl_lab = 0.70
    text_with_subscripts(ax, xl_lab, TAU_LENIENT * xl_lab + 0.025,
                         z0_pieces("Standard cutoff 0.20 ", "", 7.5),
                         rotation=line_angle(ax, TAU_LENIENT, xl_lab, TAU_LENIENT * xl_lab),
                         raise_pt=1.6)
    ax.text(xhi - 0.01, radius - 0.012, f"torso radius {radius:.2f} m", ha="right",
            va="top", fontsize=7.5, color=TEXT, zorder=5)

    fig = ax.figure
    bbox = ax.get_position()
    pt_per_x = bbox.width * fig.get_figwidth() * 72.0 / (xhi - xlo)
    pt_per_y = bbox.height * fig.get_figheight() * 72.0 / ytop
    diam = np.sqrt(POINT_SIZE * 1.25) * 1.02  # the cross is the widest marker
    groups = {}
    for b in sorted(roll):
        groups.setdefault(round(roll[b]["z0"], 3), []).append(b)
    z_keys = sorted(groups)
    rank = {"floor": 0, "mvt": 1, "vt": 2}
    collapsed = []
    placed: list[tuple[float, float]] = []  # marker centres in display points
    max_offset = [0.0]
    n_overlap = [0]

    def side(x, y):
        """Which side of the Strict and Standard lines the point (x, y) is on."""
        return (np.sign(y - TAU_STRICT * x), np.sign(y - TAU_LENIENT * x))

    def draw_body(x, yi, b):
        max_offset[0] = max(max_offset[0], abs(x - roll[b]["z0"]))
        if side(x, yi) != side(roll[b]["z0"], yi):
            raise RuntimeError(f"{b}: the horizontal offset moves it across a cutoff line")
        placed.append((x * pt_per_x, yi * pt_per_y))
        if vt_cut[b] and inert[b]:
            # In both sets: open family marker (inert) with a cross (cut off).
            ax.scatter(x, yi, marker=FAMILY_MARKER[fam[b]], s=POINT_SIZE * 1.25,
                       facecolors=INERT_FACECOLOR, edgecolors=FAMILY_INK,
                       linewidths=0.75, zorder=4)
            ax.scatter(x, yi, marker=CUTOFF_MARKER, s=POINT_SIZE * 1.25, c=FAMILY_INK,
                       linewidths=CUTOFF_LINEWIDTH * 0.8, zorder=5)
        elif vt_cut[b]:
            ax.scatter(x, yi, marker=CUTOFF_MARKER, s=POINT_SIZE * 1.25, c=FAMILY_INK,
                       linewidths=CUTOFF_LINEWIDTH, zorder=5)
        elif inert[b]:
            ax.scatter(x, yi, marker=FAMILY_MARKER[fam[b]], s=POINT_SIZE,
                       facecolors=INERT_FACECOLOR, edgecolors=FAMILY_INK,
                       linewidths=0.75, zorder=4)
        else:
            # Dark edge, not white: where two filled markers overlap, a white
            # edge would draw the outline of an open marker.
            ax.scatter(x, yi, marker=FAMILY_MARKER[fam[b]], s=POINT_SIZE,
                       facecolors=FAMILY_INK, edgecolors=FILLED_EDGE,
                       linewidths=0.3, alpha=0.8, zorder=3)

    # Pass 1: where many bodies share one position (23 of the 25 bodies with
    # z0 = 0.30 m rest at 0.10 m), draw one marker per encoding class
    # (family, inert, cut-off) with the number of bodies.
    rest = []
    for zk in z_keys:
        members = sorted(groups[zk], key=lambda b: (round(roll[b]["heights"][STEP_1S], 3),
                                                    vt_cut[b], rank[fam[b]], b))
        by_height = {}
        for b in members:
            by_height.setdefault(round(roll[b]["heights"][STEP_1S], 3), []).append(b)
        crowded = {hk for hk, bs in by_height.items() if len(bs) >= COLLAPSE_MIN}
        class_x = {}
        for hk in sorted(crowded):
            classes = {}
            for b in by_height[hk]:
                classes.setdefault((vt_cut[b], rank[fam[b]], fam[b]), []).append(b)
            keys = sorted(classes)
            # Class markers centred on z0, CLASS_STEP_M apart, so they stay
            # within SPREAD_MAX_M of the body's own reset height.
            xs_c = zk + (np.arange(len(keys)) - (len(keys) - 1) / 2) * CLASS_STEP_M
            if np.abs(xs_c - zk).max() > SPREAD_MAX_M + 1e-9:
                raise RuntimeError(f"z0 {zk}: {len(keys)} class markers do not fit "
                                   f"within {SPREAD_MAX_M} m")
            # Counts sit in a wider row along the bottom of the panel, each
            # joined to its marker by a thin leader, so the digits read as
            # separate numbers while the markers stay near z0.
            xs_lab = zk + (np.arange(len(keys)) - (len(keys) - 1) / 2) * COUNT_STEP_M
            for xc, xl, key in zip(xs_c, xs_lab, keys):
                bs_c = classes[key]
                yc = float(np.mean([roll[b]["heights"][STEP_1S] for b in bs_c]))
                draw_body(xc, yc, bs_c[0])
                class_x[key] = xc
                ax.plot([xc, xl], [yc - 0.016, COUNT_Y + 0.017], color=LEADER_INK,
                        linewidth=0.45, zorder=2.5, solid_capstyle="butt")
                ax.text(xl, COUNT_Y, str(len(bs_c)), ha="center", va="center",
                        fontsize=7.0, color=TEXT, zorder=6)
                collapsed.append((zk, hk, key[2], len(bs_c), bool(key[0])))
        others = [b for b in members
                  if round(roll[b]["heights"][STEP_1S], 3) not in crowded]
        if class_x:
            # The other bodies of a crowded group (at another height) each
            # take a free slot between two class markers, nearest their own
            # class first.
            xs_sorted = sorted(class_x.values())
            slots = [0.5 * (u + v) for u, v in zip(xs_sorted[:-1], xs_sorted[1:])]
            taken = set()
            for b in others:
                key = (vt_cut[b], rank[fam[b]], fam[b])
                ref = class_x.get(key, zk)
                free = [x for x in slots if x not in taken] or slots
                x = min(free, key=lambda u: (abs(u - ref), u))
                taken.add(x)
                draw_body(x, roll[b]["heights"][STEP_1S], b)
        else:
            rest.extend(others)

    # Pass 2: the other bodies of each z0 group take the horizontal offset
    # closest to z0 at which their marker overlaps no marker drawn so far.
    # Offsets stay within SPREAD_MAX_M and within half the distance to the
    # nearest other z0, so groups do not mix, and an offset may not move a
    # marker across a cutoff line. If no offset is free, the body takes the
    # allowed offset farthest from the drawn markers.
    # Reset heights closer than MERGE_M (0.55 and 0.56 m, 1.03 and 1.04 m)
    # are spread as one group around their mean.
    runs = []
    for z in z_keys:
        if runs and z - runs[-1][-1] < MERGE_M:
            runs[-1].append(z)
        else:
            runs.append([z])
    centres = [float(np.mean(r)) for r in runs]
    for run, zc in zip(runs, centres):
        members = [b for b in rest if round(roll[b]["z0"], 3) in run]
        if not members:
            continue
        members.sort(key=lambda b: (roll[b]["heights"][STEP_1S], vt_cut[b],
                                    rank[fam[b]], b))
        gap = min(abs(z - zc) for z in centres if z != zc)
        own = max(abs(roll[b]["z0"] - zc) for b in members)
        cap_pt = min(SPREAD_MAX_M - own, 0.5 * gap) * pt_per_x
        grid = np.arange(-cap_pt, cap_pt + 1e-9, 0.25)
        grid = np.array(sorted(np.append(grid, 0.0), key=lambda c: (abs(c), c)))
        for b in members:
            z0, h = roll[b]["z0"], roll[b]["heights"][STEP_1S]
            x0, y0 = zc * pt_per_x, h * pt_per_y
            true_side = side(z0, h)
            best, best_d, free = 0.0, -1.0, False
            for c in grid:
                if side(zc + c / pt_per_x, h) != true_side:
                    continue
                d = min((float(np.hypot(x0 + c - px, y0 - py)) for px, py in placed),
                        default=np.inf)
                if d >= diam:
                    best, free = float(c), True
                    break
                if d > best_d + 1e-9:
                    best, best_d = float(c), d
            if not free:
                n_overlap[0] += 1
                print(f"  note: {b} touches a neighbour ({best_d:.2f} pt apart, "
                      f"marker {diam:.2f} pt)")
            draw_body(zc + best / pt_per_x, h, b)
    for zk, hk, f, n, cutf in collapsed:
        print(f"  counted marker at z0 {zk:.2f} m, height {hk:.3f} m: {n} {f}"
              + (" cut-off" if cutf else ""))
    print(f"  markers that could not avoid touching a neighbour: {n_overlap[0]}")
    print(f"  largest horizontal offset of a marker from its body's z0: "
          f"{max_offset[0]:.4f} m (bound {SPREAD_MAX_M} m)")
    if max_offset[0] > SPREAD_MAX_M + 1e-9:
        raise RuntimeError("a marker is drawn too far from its body's reset height")
    ax.set_xlabel("Reset height z0 (m)")  # placeholder, replaced below
    ax.set_ylabel("Torso height (m)")
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8], ["0", "0.2", "0.4", "0.6", "0.8"])
    tufte_axes(ax)
    # Redraw the x label with a 7 pt subscript, on the placeholder's baseline.
    fig.canvas.draw()
    label = ax.xaxis.label
    y_axes = (baseline_px(label) - ax.bbox.y0) / ax.bbox.height
    size = label.get_fontsize()
    ax.set_xlabel("")
    text_with_subscripts(ax, 0.5, y_axes, z0_pieces("Reset height ", " (m)", size),
                         xycoords="axes fraction", ha="center")


def population_legend(fig, x, y):
    """Two legend rows: body and cut-off markers, then the open markers."""
    from matplotlib.legend_handler import HandlerTuple

    def mk(marker, face, edge, ew, alpha=1.0, size=3.8):
        return Line2D([], [], linestyle="none", marker=marker, markersize=size,
                      markerfacecolor=face, markeredgecolor=edge,
                      markeredgewidth=ew, alpha=alpha)

    rows = [
        ([tuple(mk(FAMILY_MARKER[f], FAMILY_INK, FILLED_EDGE, 0.3, 0.8) for f in FAMILY_ORDER),
          Line2D([], [], linestyle="none", marker=CUTOFF_MARKER, markersize=4.2,
                 markeredgecolor=FAMILY_INK, markeredgewidth=CUTOFF_LINEWIDTH)],
         ["Body after 1 s", "VT cut-off body (10)"]),
        ([tuple(mk(FAMILY_MARKER[f], INERT_FACECOLOR, FAMILY_INK, 0.75)
                for f in FAMILY_ORDER)],
         ["Standard cutoff below the torso radius (28)"]),
    ]
    row_h = 0.155 / fig.get_figheight()
    for i, (handles, labels) in enumerate(rows):
        leg = fig.legend(handles, labels, loc="upper left",
                         bbox_to_anchor=(x, y - i * row_h), ncol=len(handles),
                         columnspacing=1.2, frameon=False, fontsize=7.5,
                         handlelength=2.6, handletextpad=0.5, borderaxespad=0.0,
                         handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})
        for t in leg.get_texts():
            t.set_color(TEXT)


def build_figure(trace, cutoffs, insets, roll, fam, inert, vt_cut):
    fig = plt.figure(figsize=(TEXT_WIDTH, FIG_H))
    ax_a = inch_axes(fig, *AX_A)
    ax_b = inch_axes(fig, *AX_B)
    draw_trace(ax_a, trace, cutoffs, insets)
    draw_population(ax_b, roll, fam, inert, vt_cut)
    xb = AX_B[0] - 0.47
    title_b = fig.text(xb / TEXT_WIDTH, 1.0, "(b) All 100 bodies after 1 s of zero action",
                       ha="left", va="top", fontsize=9.0, fontweight="semibold", color=TEXT)
    # Title (a) has a z0 subscript: draw it on the baseline of title (b).
    fig.canvas.draw()
    y_title = baseline_px(title_b) / fig.bbox.height
    text_with_subscripts(fig, 0.0, y_title,
                         z0_pieces("(a) A short VT body, ", f" = {trace['z0']:.2f} m", 9.0,
                                   fontweight="semibold"))
    strip_top = (FIG_H - 0.21) / FIG_H
    event_legend(fig, trace, 0.06 / TEXT_WIDTH, strip_top)
    population_legend(fig, xb / TEXT_WIDTH, strip_top)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--xml-dir", type=Path, default=DEFAULT_XML)
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--trace-body", default=DEFAULT_TRACE_BODY)
    ap.add_argument("--check-passive", type=Path, default=None,
                    help="all100_passive.json to compare the rollouts with")
    ap.add_argument("--check-settle", type=Path, default=None,
                    help="settle.json to compare the 3 s heights with")
    ap.add_argument("--font-dir", type=Path, default=None)
    args = ap.parse_args()

    import mujoco

    apply_style(args.font_dir)
    xml_dir = args.xml_dir.expanduser()
    bodies = sorted(p.stem for p in xml_dir.glob("*.xml"))
    if len(bodies) != 100:
        raise RuntimeError(f"expected 100 XMLs, found {len(bodies)}")
    roll = {b: zero_action_rollout(xml_dir, b) for b in bodies}
    print(f"MuJoCo {mujoco.__version__}: {len(roll)} zero-action rollouts of {N_STEPS} "
          f"control steps ({N_STEPS * 0.02:.0f} s)")
    check_against_records(roll, args.check_passive, args.check_settle)

    fam = {b: family_of(b) for b in bodies}
    inert = inert_flags(args.audit, roll)
    cut = cutoff_flags(args.results.expanduser(), bodies)
    vt_cut = {b: cut[b] and fam[b] == "vt" for b in bodies}
    if sum(inert.values()) != 28 or sum(cut.values()) != 11 or sum(vt_cut.values()) != 10:
        raise RuntimeError("unexpected inert or cut-off counts")
    print("inert bodies:", sum(inert.values()), "; cut-off bodies:", sum(cut.values()),
          "; VT cut-off bodies:", sum(vt_cut.values()),
          f"(of which inert: {sum(inert[b] for b in bodies if vt_cut[b])})")

    for label, step in (("1 s", 50), ("1.5 s", 75), ("3 s", 150)):
        n = sum(r["heights"][step] < TAU_STRICT * r["z0"] for r in roll.values())
        print(f"bodies below the Strict cutoff after {label} of zero action: {n}")
    z30 = [b for b in bodies if abs(roll[b]["z0"] - 0.30) < 1e-9]
    print(f"z0 = 0.30 m: {len(z30)} bodies; 1 s heights "
          f"{min(roll[b]['heights'][50] for b in z30):.4f} to "
          f"{max(roll[b]['heights'][50] for b in z30):.4f} m; minimum over 1 s "
          f"{min(roll[b]['heights'][:51].min() for b in z30):.4f} m")
    print("VT cut-off bodies (z0, Strict crossing step, first contact step, 1 s height):")
    for b in bodies:
        if vt_cut[b]:
            r = roll[b]
            print(f"  {b}: z0 {r['z0']:.2f}, cross {r['strict_cross']}, contact "
                  f"{r['first_contact']}, 1 s {r['heights'][50]:.3f} m"
                  + (" (inert)" if inert[b] else ""))

    tb = args.trace_body
    if not (vt_cut.get(tb) and inert.get(tb)):
        raise RuntimeError(f"{tb} is not a short (inert) VT cut-off body")
    tr = roll[tb]
    cutoffs = {
        "strict": TAU_STRICT * tr["z0"],
        "lenient": TAU_LENIENT * tr["z0"],
        "derl": DERL_RATIO * (tr["z0"] - DERL_OFFSET),
    }
    print(f"trace body {tb}: z0 {tr['z0']:.2f} m, torso radius {tr['radius']:.2f} m, "
          f"Strict {cutoffs['strict']:.3f} m, Standard {cutoffs['lenient']:.3f} m, "
          f"DERL {cutoffs['derl']:.3f} m; crosses Strict at step {tr['strict_cross']} "
          f"(height {tr['heights'][tr['strict_cross']]:.3f} m); first floor contact at "
          f"step {tr['first_contact']}; minimum {tr['heights'][:51].min():.4f} m; "
          f"height at 1 s {tr['heights'][50]:.4f} m")
    insets = crop_pair(render_poses(xml_dir, tb))
    fig = build_figure(tr, cutoffs, insets, roll, fam, inert, vt_cut)
    paths = save_figure(fig, args.out_dir / "fig_cr2_cutoffs")
    plt.close(fig)
    print("wrote", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
