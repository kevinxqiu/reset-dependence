#!/usr/bin/env python3
"""Forest plot for the boundary-free panel (Section 4.8).

Consolidates the three family deltas + CIs and the rank correlation + CI
into a single compact figure.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "records/boundary_free_analysis.json"
OUT = REPO / "figures/fig_boundary_free_forest.pdf"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import FAMILY_COLOR as FAMILY_COLORS
from _paper_style import (
    DEEMPH,
    GATE_RED,
    TEXT,
    apply_style,
    column_header,
    light_grid,
    tufte_axes,
)

apply_style()


def main() -> int:
    d = json.load(ANALYSIS.open())
    dec = d["decision"]
    rho = dec["rank_correlation"]
    fams = dec["family_deltas"]

    fig, (ax_rho, ax_deltas) = plt.subplots(
        1, 2, figsize=(7.0, 2.18), gridspec_kw={"width_ratios": [0.92, 2.15]}
    )
    fig.subplots_adjust(left=0.15, right=0.99, top=0.84, bottom=0.24, wspace=0.28)

    # Left: rank correlation
    lo, hi = rho["ci_lo"], rho["ci_hi"]
    ax_rho.errorbar(
        rho["rho"], 0,
        xerr=[[rho["rho"] - lo], [hi - rho["rho"]]],
        fmt="o", color=TEXT, markersize=5.5, capsize=3, lw=1.3,
    )
    ax_rho.axvline(0.95, ls=(0, (4, 3)), color=GATE_RED, lw=0.85)
    ax_rho.text(
        0.78, 0.61,
        rf"$\rho={rho['rho']:.2f}$" + "\n" + rf"[{lo:.2f}, {hi:.2f}]",
        transform=ax_rho.transAxes,
        ha="right",
        va="center",
        fontsize=7.0,
        color=TEXT,
    )
    ax_rho.text(
        0.95, 0.91, r"gate $\rho{=}0.95$",
        transform=ax_rho.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=7.3,
        color=GATE_RED,
    )
    ax_rho.set_xlim(0.3, 1.0)
    ax_rho.set_yticks([0])
    ax_rho.set_yticklabels(["persistent vs. corpus\n$\\tau{=}0$"])
    ax_rho.set_xlabel(r"Spearman $\rho$ (95% CI)")
    column_header(ax_rho, "Rank correlation")
    light_grid(ax_rho, axis="x")
    tufte_axes(ax_rho)

    # Right: family deltas
    order = ["floor", "vt", "mvt"]
    y = np.arange(len(order))
    for i, fam in enumerate(order):
        f = fams[fam]
        c = FAMILY_COLORS[fam]
        lo, hi = f["ci_lo"], f["ci_hi"]
        med = f["median_delta"]
        ax_deltas.errorbar(
            med, y[i],
            xerr=[[med - lo], [hi - med]],
            fmt="o", color=c, markersize=5.5, capsize=3, lw=1.35,
            label=fam.upper() if fam != "floor" else "Floor",
        )
    ax_deltas.axvline(0, ls="-", color=DEEMPH, lw=0.8)
    ax_deltas.set_yticks(y)
    ax_deltas.set_yticklabels(["Floor", "VT", "MVT"])
    ax_deltas.set_xlabel(r"$\Delta_F$: persistent PSR $-$ corpus $\tau{=}0$ PSR")
    column_header(ax_deltas, "Per-family shift")
    light_grid(ax_deltas, axis="x")
    tufte_axes(ax_deltas)
    # Small margin so error bars are not flush against edges
    xmin = min(fams[f]["ci_lo"] for f in order) - 0.15
    xmax = max(0.2, max(fams[f]["ci_hi"] for f in order) + 0.1)
    ax_deltas.set_xlim(xmin, xmax)
    ax_deltas.invert_yaxis()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
