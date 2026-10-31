#!/usr/bin/env python3
"""Pareto-frontier figure: PSR vs fall_rate and PSR vs control-effort COT.

Reads records/energy_proxy.json (produced by analyze_energy_proxy.py)
and renders a 3x2 grid: rows = protocol (RF, Standard, Strict), cols =
{fall_rate, control-effort COT}. Each cell scatters all 100 bodies colored
by family, with the Pareto frontier (highest PSR at each y-threshold)
drawn as a step curve. Bodies that show up in top-3 for that protocol
have a thicker edge so the reader can see where selection would land on
the front.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

PROTOCOLS = ("reset_free", "episodic", "strict")
PROTOCOL_LABELS = {
    "reset_free": r"Reset-free ($\tau{=}0$)",
    "episodic":   r"Standard ($\tau{=}0.20$)",
    "strict":     r"Strict ($\tau{=}0.50$)",
}
FAMILY_COLORS = {"floor": "#ff7f0e", "vt": "#1f77b4", "mvt": "#2ca02c"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_top_bodies() -> dict[str, set[str]]:
    path = REPO / "records/top_bodies_per_protocol.json"
    with path.open() as f:
        data = json.load(f)
    out = {}
    for proto in PROTOCOLS:
        out[proto] = {e["morph"] for e in data["top_by_protocol"][proto][:3]}
    return out


def pareto_upper(x: np.ndarray, y: np.ndarray, direction: str = "min") -> tuple[np.ndarray, np.ndarray]:
    """Return Pareto frontier points sorted by x for maximizing PSR.

    direction='min' means x is to be minimized (e.g., fall_rate, COT).
    """
    if direction != "min":
        raise NotImplementedError
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    out_x, out_y = [], []
    best_y = -np.inf
    for xi, yi in zip(xs, ys):
        if yi > best_y:
            out_x.append(xi)
            out_y.append(yi)
            best_y = yi
    return np.array(out_x), np.array(out_y)


def main() -> int:
    with (REPO / "records/energy_proxy.json").open() as f:
        records = json.load(f)

    by_proto: dict[str, list[dict]] = {p: [] for p in PROTOCOLS}
    for rec in records:
        by_proto[rec["protocol"]].append(rec)

    top_bodies = load_top_bodies()

    fig, axes = plt.subplots(
        3, 2, figsize=(7.0, 7.0),
        gridspec_kw={"wspace": 0.28, "hspace": 0.38},
    )

    for row, proto in enumerate(PROTOCOLS):
        data = by_proto[proto]
        psr = np.array([
            r["psr"] if r["psr"] is not None else np.nan for r in data
        ])
        fall = np.array([
            r["fall_rate"] if r["fall_rate"] is not None else np.nan for r in data
        ])
        # Use C-COT at the >0.05m threshold ("any-dist") so the reader sees
        # the full distribution including failures. Walker-filtered views
        # live in the appendix table.
        cot = np.array([
            r.get("cot_any_dist", np.nan) if r.get("cot_any_dist") is not None
            else np.nan
            for r in data
        ])
        fam = [r["family"] for r in data]
        morphs = [r["morph"] for r in data]
        mask_valid = np.isfinite(psr) & np.isfinite(fall)
        mask_cot = mask_valid & np.isfinite(cot) & (cot > 0)

        ax = axes[row, 0]
        for m, p, fr, family in zip(morphs, psr, fall, fam):
            is_top = m in top_bodies[proto]
            ax.scatter(
                fr, p,
                color=FAMILY_COLORS.get(family, "#666666"),
                s=36 if is_top else 18,
                edgecolor="black" if is_top else "none",
                linewidth=1.0 if is_top else 0.0,
                alpha=0.75 if not is_top else 1.0,
                zorder=3 if is_top else 2,
            )
        px, py = pareto_upper(fall[mask_valid], psr[mask_valid], direction="min")
        ax.step(px, py, where="post", color="black", linewidth=1.2, alpha=0.6, zorder=4)
        ax.set_xlabel("Fall rate")
        if row == 0:
            ax.set_title("PSR vs fall rate (lower = better)", pad=6)
        ax.set_ylabel(f"{PROTOCOL_LABELS[proto]}\nPSR @ $\\tau_{{\\mathrm{{eval}}}}=0.20$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.04, 1.04)
        ax.grid(axis="both", linestyle=":", alpha=0.25)

        ax2 = axes[row, 1]
        for m, p, c, family in zip(np.array(morphs)[mask_cot], psr[mask_cot], cot[mask_cot], np.array(fam)[mask_cot]):
            is_top = m in top_bodies[proto]
            ax2.scatter(
                c, p,
                color=FAMILY_COLORS.get(family, "#666666"),
                s=36 if is_top else 18,
                edgecolor="black" if is_top else "none",
                linewidth=1.0 if is_top else 0.0,
                alpha=0.75 if not is_top else 1.0,
                zorder=3 if is_top else 2,
            )
        if mask_cot.sum() > 2:
            px2, py2 = pareto_upper(cot[mask_cot], psr[mask_cot], direction="min")
            ax2.step(px2, py2, where="post", color="black", linewidth=1.2, alpha=0.6, zorder=4)
        ax2.set_xlabel("Control-effort COT (action-$L^2$/metre)")
        if row == 0:
            ax2.set_title("PSR vs control-effort COT (lower = better)", pad=6)
        ax2.set_xscale("log")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(axis="both", linestyle=":", alpha=0.25)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=FAMILY_COLORS["floor"], label="Floor"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=FAMILY_COLORS["vt"], label="VT"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=FAMILY_COLORS["mvt"], label="MVT"),
        plt.Line2D([0], [0], marker="o", linestyle="", color="white", markeredgecolor="black",
                   markersize=7, label="Top-3 body"),
        plt.Line2D([0], [0], linestyle="-", color="black", alpha=0.6, label="Pareto frontier"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.015),
        frameon=False,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.09, top=0.95, left=0.09, right=0.98)

    out_pdf = REPO / "figures/fig_pareto.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
