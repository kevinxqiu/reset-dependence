#!/usr/bin/env python3
"""Figure 2: state-space schematic of what each protocol samples during training.

Inspired by Figure 1 of Patil et al. 2024 ("Intelligent Switching in Reset-Free RL").
The horizontal axis is an abstract progress direction; the vertical axis is torso
height z. A horizontal threshold line at z = tau z_0 cuts off the bean into a
fall zone (orange, below) and a standing region (above). Each panel shows a
representative rollout under one protocol, plus a shaded aggregate of states the
body visits during training.

Camera-ready clarity changes to the submitted drawing, same panels, shapes and colours: the Strict dip visibly
crosses the cutoff and the dashed reset returns on its own arc; the cutoff condition sits below the line, inside
the zone where it holds; the Reset-free panel hatches the same low zone (no line, no reset) and its dip reaches it;
labels stay inside the ovals; the axis reads "torso height z"; the subtitles use commas instead of semicolons.

Writes figures/fig_protocol_timeline.pdf (or the path given as its argument).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT_PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "figures/fig_protocol_timeline.pdf"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import DEEMPH, TEXT, apply_style  # noqa: E402

apply_style()
plt.rcParams["axes.titlesize"] = 10.5  # the schematic uses larger panel titles
plt.rcParams["hatch.linewidth"] = 0.6

BEAN_W, BEAN_H = 2.1, 1.30
I0 = (-0.78, 0.30)

# NOTE: schematic uses its own colour register, independent of the
# data-figure protocol palette. The walking-region was previously a
# deep blue (#2d5c9b) close to the RF-navy used in data figures, which
# risked the reader conflating "visited set" with "Reset-free protocol".
# We swap it to a neutral muted slate that is unmistakably not navy.
WALK = "#56697A"  # neutral slate-grey for the schematic's walk arrows
RESET = "#6A4A4A"  # muted brown-grey, separated from fall orange
FALL_ZONE = "#f4c9b2"
SPACE_BG = "#f2f2f2"
REGION_GREY = "#7a7a7a"

PANELS = [
    (
        "strict",
        "Strict (τ = 0.50)",
        "moderate drops trigger reset, so the\nbody rarely leaves a neighborhood of $I_0$",
    ),
    (
        "episodic",
        "Standard (τ = 0.20)",
        "walks forward, occasionally falls,\nresets to $I_0$ on falls",
    ),
    (
        "reset_free",
        "Reset-free (τ = 0)",
        "no height rule, but the 1,000-step\nlimit still ends every episode",
    ),
]

# z-thresholds in bean-local coordinates. Bean spans y in [-H/2, H/2].
# Strict: tau=0.50 -> threshold high inside bean.
# Standard: tau=0.20 -> threshold low.
# Reset-free: no threshold.
THRESH_Y = {
    "strict": 0.10,  # high threshold: most of bean is "fallen"
    "episodic": -0.35,  # low threshold: only a deep dip counts
    "reset_free": None,
}


def curved_arrow(
    ax,
    p0,
    p1,
    rad=0.2,
    color=WALK,
    lw=1.7,
    ls="-",
    alpha=0.95,
    mutation=13,
    zorder=4,
    shrinkA=0,
    shrinkB=0,
):
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="->",
        mutation_scale=mutation,
        color=color,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
    )
    ax.add_patch(arrow)


def straight_arrow(
    ax,
    p0,
    p1,
    color=RESET,
    lw=1.2,
    ls=(0, (4, 2)),
    alpha=0.9,
    mutation=11,
    zorder=3,
    shrinkA=0,
    shrinkB=0,
):
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="->",
        mutation_scale=mutation,
        color=color,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        zorder=zorder,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
    )
    ax.add_patch(arrow)


def draw_background(ax):
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.95, 0.98)
    ax.set_aspect("equal")
    ax.axis("off")
    bean = Ellipse(
        (0, 0),
        BEAN_W,
        BEAN_H,
        facecolor=SPACE_BG,
        edgecolor="#a0a0a0",
        linewidth=0.9,
        zorder=1,
    )
    ax.add_patch(bean)
    return bean


def draw_fall_sublevel(ax, bean, y_thresh):
    """Orange fill below y_thresh, clipped to the bean."""
    rect = Rectangle(
        (-BEAN_W / 2 - 0.1, -BEAN_H / 2 - 0.1),
        BEAN_W + 0.2,
        (y_thresh + BEAN_H / 2 + 0.1),
        facecolor=FALL_ZONE,
        edgecolor="none",
        alpha=0.85,
        zorder=1.4,
    )
    ax.add_patch(rect)
    rect.set_clip_path(bean)
    half = np.sqrt(np.clip(1 - (y_thresh / (BEAN_H / 2)) ** 2, 0, 1)) * (BEAN_W / 2)
    ax.plot(
        [-half, half],
        [y_thresh, y_thresh],
        color="#b04f2e",
        linewidth=1.0,
        linestyle=(0, (5, 3)),
        alpha=0.9,
        zorder=1.6,
    )


def draw_I0(ax):
    ax.plot(
        I0[0],
        I0[1],
        "o",
        markersize=11,
        color="#111111",
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=6,
    )
    ax.text(
        I0[0] - 0.03,
        I0[1] + 0.14,
        r"$I_0$",
        fontsize=10.0,
        color=TEXT,
        ha="right",
        va="bottom",
        zorder=6,
        weight="semibold",
    )


def draw_region_labels(ax, post_fall_pos=(0.82, -0.44), walking_pos=(0.78, 0.50), post_fall_size=8.6, post_fall_box=0.55):
    # Color-code the region labels to match their regions: blue text = blue
    # visited oval (walking), orange-brown text = orange fall sublevel set.
    # Light bbox (alpha 0.55) keeps text legible without looking stickered.
    ax.text(
        walking_pos[0],
        walking_pos[1],
        "walking\nstates",
        fontsize=8.6,
        color="#3d5d8b",
        ha="center",
        va="center",
        style="italic",
        zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.3),
    )
    ax.text(
        post_fall_pos[0],
        post_fall_pos[1],
        "post-fall /\nrecovery",
        fontsize=post_fall_size,
        color="#8b4d2e",
        ha="center",
        va="center",
        style="italic",
        zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=post_fall_box, pad=0.3),
    )


def fall_label(ax, y_thresh, x=-0.20, dy=0.05):
    ax.text(
        x,
        y_thresh - dy,
        r"$z < \tau z_0$",
        fontsize=8.5,
        color="#8b3a1f",
        ha="center",
        va="top",
        zorder=7,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.35, alpha=0.65),
    )


def draw_strict(ax, bean):
    y_thresh = THRESH_Y["strict"]
    draw_fall_sublevel(ax, bean, y_thresh)
    # Aggregate visitation: tiny cluster around I0, well above threshold
    cluster = Ellipse(
        I0,
        0.50,
        0.32,
        facecolor="#dde6f1",
        edgecolor="#7b93b6",
        linewidth=0.3,
        alpha=0.55,
        zorder=2,
    )
    ax.add_patch(cluster)
    # Tiny dip: trajectory descends from I_0 and just clips into the fall zone
    # (the orange sublevel set), then the dashed reset returns to I_0. The
    # downward arc and end point in the orange match the "tiny dips trigger
    # reset" subtitle.
    end = (-0.30, -0.16)
    curved_arrow(
        ax,
        I0,
        end,
        rad=-0.35,
        color=WALK,
        lw=2.0,
        mutation=13,
        zorder=5,
        shrinkA=14,
        shrinkB=2,
    )
    curved_arrow(
        ax,
        end,
        I0,
        rad=-0.45,
        color=RESET,
        lw=1.3,
        ls=(0, (4, 2)),
        mutation=11,
        zorder=4,
        shrinkA=4,
        shrinkB=14,
    )
    ax.text(
        -0.68,
        -0.17,
        "reset",
        fontsize=8.5,
        color=RESET,
        style="italic",
        ha="center",
        va="center",
        zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.6),
    )
    # Place on the right half above the threshold, away from the cluster and arrows.
    fall_label(ax, y_thresh, x=0.45)


def draw_episodic(ax, bean):
    y_thresh = THRESH_Y["episodic"]
    draw_fall_sublevel(ax, bean, y_thresh)
    # Aggregate visitation: covers upper walking region + a small excursion below
    explored = Ellipse(
        (0.03, 0.15),
        1.50,
        0.78,
        facecolor="#dde6f1",
        edgecolor="#7b93b6",
        linewidth=0.3,
        alpha=0.55,
        zorder=2,
    )
    ax.add_patch(explored)
    # Chained rollout: I0 -> walking -> farther walking -> dips into fall
    p1 = (-0.20, 0.42)
    p2 = (0.40, 0.22)
    p3 = (0.40, -0.52)
    curved_arrow(ax, I0, p1, rad=0.10, color=WALK, lw=1.8, mutation=13, shrinkA=14)
    curved_arrow(ax, p1, p2, rad=-0.10, color=WALK, lw=1.8, mutation=13)
    curved_arrow(ax, p2, p3, rad=0.15, color=WALK, lw=1.8, mutation=13)
    # Bowed dashed reset from fall location back to I0 (arcs away from label zone)
    curved_arrow(
        ax,
        p3,
        I0,
        rad=-0.10,
        color=RESET,
        ls=(0, (4, 2)),
        lw=1.2,
        mutation=11,
        zorder=4,
        shrinkB=16,
    )
    ax.text(
        -0.02,
        -0.08,
        "reset",
        fontsize=8.2,
        color=RESET,
        style="italic",
        ha="center",
        va="center",
        zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.6),
    )
    fall_label(ax, y_thresh, x=0.66)


def draw_reset_free(ax, bean):
    # No threshold line, no fall zone shading, no dashed reset.
    # Muted aggregate fill that roughly tracks the trajectory.
    trail = Ellipse(
        (0.05, -0.05),
        1.80,
        1.00,
        facecolor="#dde6f1",
        edgecolor="#7b93b6",
        linewidth=0.3,
        alpha=0.45,
        zorder=2,
    )
    ax.add_patch(trail)
    # The zone that ends a Standard episode, hatched: here training continues through it
    band = Rectangle(
        (-BEAN_W / 2 - 0.1, -BEAN_H / 2 - 0.1),
        BEAN_W + 0.2,
        THRESH_Y["episodic"] + BEAN_H / 2 + 0.1,
        facecolor="none",
        edgecolor="#e2a587",
        linewidth=0,
        hatch="///",
        zorder=2.5,
    )
    ax.add_patch(band)
    band.set_clip_path(bean)
    # Single continuous winding trajectory, amplitude tuned to stay inside bean
    t = np.linspace(0, 1, 500)
    x = I0[0] + 1.55 * t + 0.10 * np.sin(5.5 * np.pi * t)
    y = (
        I0[1]
        + 0.34 * np.sin(3.2 * np.pi * t)
        - 0.58 * t
        - 0.09 * np.exp(-(((t - 0.47) / 0.06) ** 2))
        + 0.11 * np.sin(8.5 * np.pi * t)
        - 0.05
    )
    ax.plot(
        x[:-15], y[:-15], color=WALK, linewidth=1.8, zorder=4, solid_capstyle="round"
    )
    # Arrowhead at end with visible shaft
    ax.annotate(
        "",
        xy=(x[-1], y[-1]),
        xytext=(x[-16], y[-16]),
        arrowprops=dict(arrowstyle="->", color=WALK, linewidth=1.8, mutation_scale=14),
        zorder=4,
    )
    # Explicit anti-reset cue where other panels discuss reset behaviour
    ax.text(
        -0.05,
        0.49,
        "no fall reset",
        fontsize=8.2,
        color=RESET,
        style="italic",
        ha="center",
        va="bottom",
        zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.6),
    )


DRAW_FN = {
    "strict": draw_strict,
    "episodic": draw_episodic,
    "reset_free": draw_reset_free,
}


def main() -> None:
    # Match NeurIPS \textwidth (~5.5 in) so embedded fonts render at source size
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.15))
    plt.subplots_adjust(wspace=0.04, left=0.02, right=0.99, top=0.89, bottom=0.20)
    for ax, (key, title, sub) in zip(axes, PANELS):
        bean = draw_background(ax)
        DRAW_FN[key](ax, bean)
        draw_I0(ax)
        # In reset-free, the trajectory ends in the lower-right; move the
        # post-fall/recovery label to the lower-left so it does not occlude.
        if key == "reset_free":
            draw_region_labels(ax, post_fall_pos=(-0.45, -0.47), walking_pos=(0.12, 0.22), post_fall_size=7.8, post_fall_box=0.85)
        elif key == "episodic":
            draw_region_labels(ax, post_fall_pos=(-0.28, -0.49), walking_pos=(0.68, 0.20), post_fall_size=7.8)
        else:
            draw_region_labels(ax, post_fall_pos=(0.45, -0.36), walking_pos=(0.52, 0.38))
        ax.text(
            0,
            0.80,
            title,
            ha="center",
            va="bottom",
            fontsize=10.5,
            weight="bold",
            color=TEXT,
            zorder=6,
        )
        ax.text(
            0,
            -0.82,
            sub,
            ha="center",
            va="top",
            fontsize=8.7,
            color=TEXT,
            zorder=6,
        )

    # Shared height-axis anchor on left panel (small "z" arrow)
    ax0 = axes[0]
    ax0.annotate(
        "",
        xy=(-1.13, 0.55),
        xytext=(-1.13, -0.55),
        arrowprops=dict(arrowstyle="->", color=DEEMPH, linewidth=0.7),
        zorder=1,
    )
    ax0.text(
        -1.17,
        0.0,
        "torso height $z$",
        fontsize=8,
        rotation=90,
        color=DEEMPH,
        ha="right",
        va="center",
        zorder=1,
    )
    ax0.annotate(
        "",
        xy=(-0.58, -0.61),
        xytext=(-1.13, -0.61),
        arrowprops=dict(arrowstyle="->", color=DEEMPH, linewidth=0.7),
        zorder=1,
    )
    ax0.text(
        -0.86,
        -0.67,
        "pose",
        fontsize=8,
        color=DEEMPH,
        ha="center",
        va="top",
        zorder=1,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
