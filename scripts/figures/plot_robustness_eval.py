#!/usr/bin/env python3
"""Regenerate figures/fig_robustness_eval.pdf with Lato fonts.

Family-level RF/Std and Str/Std PSR ratios at two evaluation thresholds
(tau_eval=0.20 and tau_eval=0).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (  # noqa: E402
    DEEMPH,
    FULL_WIDTH,
    PROTOCOL_COLOR,
    TEXT,
    apply_style,
    column_header,
    light_grid,
    pastel_bar,
    tufte_axes,
    value_labels,
)

apply_style()

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "results"
OUT_PDF = REPO / "figures/fig_robustness_eval.pdf"

TAUS = {"rf": "0p00", "epi": "0p20", "strict": "0p50"}
SEEDS = (42, 43, 44)
FAMILIES = ("floor", "vt", "mvt")
FAMILY_LABELS = {"floor": "Floor", "vt": "VT", "mvt": "MVT"}
EVAL_KEYS = [("eval_tau020", r"$\tau_{\mathrm{eval}} = 0.20$"),
             ("eval_neutral", r"$\tau_{\mathrm{eval}} = 0$ (neutral)")]
RATIOS = [("rf", "epi", "RF / Standard"),
          ("strict", "epi", "Strict / Standard")]
RATIO_COLORS = {
    "RF / Standard": PROTOCOL_COLOR["reset_free"],
    "Strict / Standard": PROTOCOL_COLOR["strict"],
}


def family_of(name: str) -> str:
    return name.split("-", 1)[0].lower()


def load_corpus() -> dict:
    """Per-body per-protocol per-eval-key seed-averaged PSR."""
    raw: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for proto, tau in TAUS.items():
        for seed in SEEDS:
            data = json.load(
                (CORPUS / f"mjx_ppo_{tau}_seed{seed}" / "results.json").open()
            )
            for name, rec in data.items():
                for eval_key, _ in EVAL_KEYS:
                    if eval_key in rec and "per_step_reward" in rec[eval_key]:
                        raw[name][proto][eval_key].append(
                            float(rec[eval_key]["per_step_reward"])
                        )
    return {
        name: {
            proto: {ek: float(np.mean(v)) for ek, v in eks.items()}
            for proto, eks in protos.items()
        }
        for name, protos in raw.items()
    }


def compute_family_ratios(corpus: dict) -> dict:
    """Per-family mean of per-body ratio."""
    out: dict = {}
    rng = np.random.default_rng(23)
    for eval_key, _ in EVAL_KEYS:
        out[eval_key] = {}
        for fam in FAMILIES:
            per_body_ratios = {num: [] for num, _, _ in RATIOS}
            for name, protos in corpus.items():
                if family_of(name) != fam:
                    continue
                epi = protos.get("epi", {}).get(eval_key)
                if epi is None or epi <= 0:
                    continue
                for num_key, _, _ in RATIOS:
                    num = protos.get(num_key, {}).get(eval_key)
                    if num is None:
                        continue
                    per_body_ratios[num_key].append(num / epi)
            out[eval_key][fam] = {}
            for k, vals in per_body_ratios.items():
                vals = np.asarray(vals, dtype=float)
                if len(vals) == 0:
                    out[eval_key][fam][k] = {"mean": np.nan, "lo": np.nan, "hi": np.nan}
                    continue
                boots = rng.choice(vals, size=(5000, len(vals)), replace=True).mean(axis=1)
                out[eval_key][fam][k] = {
                    "mean": float(vals.mean()),
                    "lo": float(np.percentile(boots, 2.5)),
                    "hi": float(np.percentile(boots, 97.5)),
                }
    return out


def main() -> int:
    corpus = load_corpus()
    ratios = compute_family_ratios(corpus)

    fig, axes = plt.subplots(1, len(EVAL_KEYS), figsize=(FULL_WIDTH, 2.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.18, wspace=0.10)
    if len(EVAL_KEYS) == 1:
        axes = [axes]

    x = np.arange(len(FAMILIES))
    bar_width = 0.38
    offsets = [-bar_width / 2, bar_width / 2]

    for ax, (eval_key, eval_label) in zip(axes, EVAL_KEYS):
        for i, (num_key, _, ratio_label) in enumerate(RATIOS):
            means = [ratios[eval_key][fam][num_key]["mean"] for fam in FAMILIES]
            los = [ratios[eval_key][fam][num_key]["lo"] for fam in FAMILIES]
            his = [ratios[eval_key][fam][num_key]["hi"] for fam in FAMILIES]
            yerr = [
                np.asarray(means) - np.asarray(los),
                np.asarray(his) - np.asarray(means),
            ]
            xs = x + offsets[i]
            pastel_bar(ax, xs, means, color=RATIO_COLORS[ratio_label],
                       width=bar_width)
            ax.errorbar(
                xs, means, yerr=yerr,
                fmt="none",
                ecolor=RATIO_COLORS[ratio_label],
                elinewidth=0.9,
                capsize=2.4,
                capthick=0.9,
                zorder=4,
            )
            value_labels(ax, xs, means, dy=0.015, color=TEXT)

        ax.axhline(1.0, color=DEEMPH, linewidth=0.6, linestyle=(0, (4, 3)),
                   zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILIES])
        column_header(ax, eval_label)
        ax.set_ylim(0, 1.18)
        light_grid(ax, axis="y")
        tufte_axes(ax)

    axes[0].set_ylabel("PSR ratio")
    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=5.5,
               markerfacecolor=RATIO_COLORS[label],
               markeredgecolor=RATIO_COLORS[label], label=label)
        for _, _, label in RATIOS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.52, 1.01),
        handletextpad=0.35,
        columnspacing=1.0,
        fontsize=8.0,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")
    for eval_key, eval_label in EVAL_KEYS:
        print(f"\n{eval_label}:")
        for fam in FAMILIES:
            r = ratios[eval_key][fam]
            print(
                f"  {FAMILY_LABELS[fam]:6s} "
                f"RF/Std={r['rf']['mean']:.3f} [{r['rf']['lo']:.3f}, {r['rf']['hi']:.3f}]  "
                f"Str/Std={r['strict']['mean']:.3f} [{r['strict']['lo']:.3f}, {r['strict']['hi']:.3f}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
