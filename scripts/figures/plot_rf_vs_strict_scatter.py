#!/usr/bin/env python3
"""Body-level scatter of reset-free PSR vs strict PSR at tau_eval=0.20.

Replaces the family-only 3-point fig_reversal with a 100-point body-level
visualization of the weakest ranking alignment in the paper
(reset-free vs strict, Spearman rho = 0.21, 95% CI [-0.02, 0.40]).

The right panel shows the same bodies as a rank scatter, which makes the
ranking divergence visible at a glance.

Writes figures/fig_rf_vs_strict.pdf.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (  # noqa: E402
    DEEMPH,
    FAMILY_COLOR,
    FAMILY_LABEL,
    FAMILY_MARKER,
    FULL_WIDTH,
    TEXT,
    apply_style,
    column_header,
    light_grid,
    tufte_axes,
)

apply_style()

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "results"
OUT_PDF = REPO / "figures/fig_rf_vs_strict.pdf"

TAUS = {"rf": "0p00", "epi": "0p20", "strict": "0p50"}
SEEDS = (42, 43, 44)
FAMILIES = ("floor", "vt", "mvt")


def load_corpus() -> dict[str, dict[str, float]]:
    """Seed-averaged PSR per (body, protocol) at eval_tau020."""
    raw: dict = defaultdict(lambda: defaultdict(list))
    for proto, tau in TAUS.items():
        for seed in SEEDS:
            data = json.load(
                (CORPUS / f"mjx_ppo_{tau}_seed{seed}" / "results.json").open()
            )
            for name, rec in data.items():
                raw[name][proto].append(float(rec["eval_tau020"]["per_step_reward"]))
    return {
        name: {p: float(np.mean(v)) for p, v in protos.items()}
        for name, protos in raw.items()
    }


def family_of(name: str) -> str:
    return name.split("-", 1)[0].lower()


def main() -> int:
    corpus = load_corpus()
    names = sorted(corpus.keys())
    rf = np.array([corpus[n]["rf"] for n in names])
    strict = np.array([corpus[n]["strict"] for n in names])
    fams = [family_of(n) for n in names]

    # Spearman rho on ranks for the rank-panel sanity check
    rho, rho_p = spearmanr(rf, strict)

    # Precompute per-family top-k overlaps to annotate the figure
    def top_k(vals: np.ndarray, k: int) -> set[str]:
        idx = np.argsort(-vals)[:k]
        return {names[i] for i in idx}

    top10_rf = top_k(rf, 10)
    top10_str = top_k(strict, 10)
    overlap10 = len(top10_rf & top10_str)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.02))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.16, wspace=0.25)

    # Left: PSR value scatter
    ax = axes[0]
    for fam in FAMILIES:
        mask = np.array([f == fam for f in fams])
        ax.scatter(
            rf[mask],
            strict[mask],
            c=FAMILY_COLOR[fam],
            marker=FAMILY_MARKER[fam],
            s=25,
            alpha=0.72,
            label=f"{FAMILY_LABEL[fam]} (n={int(mask.sum())})",
            edgecolors="white",
            linewidths=0.35,
        )
    lo = min(rf.min(), strict.min()) - 0.1
    hi = max(rf.max(), strict.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], color=DEEMPH, linestyle=(0, (4, 3)), lw=0.8, zorder=1)
    ax.set_xlabel(r"reset-free training PSR ($\tau_{\mathrm{eval}} = 0.20$)")
    ax.set_ylabel(r"strict training PSR ($\tau_{\mathrm{eval}} = 0.20$)")
    column_header(ax, "Per-body PSR")
    light_grid(ax, axis="both")
    tufte_axes(ax)

    # Right: rank scatter to show ranking divergence directly
    ax = axes[1]
    rank_rf = rankdata(-rf)  # rank 1 = best
    rank_str = rankdata(-strict)
    ax.add_patch(
        Rectangle(
            (0.5, 0.5),
            10.0,
            10.0,
            facecolor=DEEMPH,
            edgecolor=TEXT,
            linewidth=0.45,
            alpha=0.18,
            zorder=0.4,
        )
    )
    for fam in FAMILIES:
        mask = np.array([f == fam for f in fams])
        ax.scatter(
            rank_rf[mask],
            rank_str[mask],
            c=FAMILY_COLOR[fam],
            marker=FAMILY_MARKER[fam],
            s=25,
            alpha=0.72,
            label=FAMILY_LABEL[fam],
            edgecolors="white",
            linewidths=0.35,
        )
    N = len(names)
    ax.plot([1, N], [1, N], color=DEEMPH, linestyle=(0, (4, 3)), lw=0.8, zorder=1)
    # Highlight top-10 regions
    ax.axhline(10.5, color=TEXT, linestyle=(0, (2, 2)), lw=0.6)
    ax.axvline(10.5, color=TEXT, linestyle=(0, (2, 2)), lw=0.6)
    ax.text(
        5.5, 9.3,
        f"{overlap10}/10\nshared",
        ha="center",
        va="center",
        fontsize=6.8,
        color=TEXT,
        zorder=5,
        linespacing=0.95,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8),
    )
    ax.set_xlabel("rank under reset-free training (1 = best)")
    ax.set_ylabel("rank under strict training (1 = best)")
    column_header(ax, "Per-body rank")
    ax.text(
        0.97, 0.04,
        f"Spearman $\\rho = {rho:.2f}$\ntop-10 overlap={overlap10}/10",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color=TEXT,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.6),
    )
    light_grid(ax, axis="both")
    tufte_axes(ax)
    ax.set_xlim(0, N + 1)
    ax.set_ylim(0, N + 1)
    ax.invert_yaxis()

    legend_handles = [
        plt.Line2D([0], [0], marker=FAMILY_MARKER[f], linestyle="none",
                   markerfacecolor=FAMILY_COLOR[f], markeredgecolor="white",
                   markeredgewidth=0.35, markersize=5.5,
                   label=f"{FAMILY_LABEL[f]} (n={fams.count(f)})")
        for f in FAMILIES
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, 0.985), fontsize=8.0,
               handletextpad=0.35, columnspacing=1.1)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    print(f"  Spearman rho = {rho:.3f} (p = {rho_p:.3g})")
    print(f"  Top-10 overlap: {overlap10}/10")
    print("  Per-family n: " + ", ".join(f"{f}={fams.count(f)}" for f in FAMILIES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
