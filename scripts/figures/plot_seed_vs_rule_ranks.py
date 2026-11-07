#!/usr/bin/env python3
"""Project-page figure: changing the reset rule reorders bodies far more than changing the seed.

Left panel: each body's rank after Reset-free training with two different
seeds. Right panel: its rank after Reset-free and after Strict training with
the same seed. Ranks use per-step reward under the common evaluator
(tau_eval = 0.20), as in Figure 4; rho is the Spearman correlation of the
plotted ranks. The paper's statistics use three-seed means (Table 2); single
runs are shown here so both panels compare like with like.

    python scripts/figures/plot_seed_vs_rule_ranks.py --out figures/fig_seed_vs_rule_ranks.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import FAINT, NAVY, TEXT, apply_style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def per_step_reward(results: Path, tau: str, seed: int) -> dict[str, float]:
    runs = json.loads((results / f"mjx_ppo_{tau}_seed{seed}" / "results.json").read_text())
    return {body: run["eval_tau020"]["per_step_reward"] for body, run in runs.items()}


def ranks(scores: dict[str, float], bodies: list[str]) -> np.ndarray:
    # Rank 1 is the highest per-step reward.
    return rankdata([-scores[b] for b in bodies], method="average")


def panel(ax, x, y, title, xlabel, ylabel) -> None:
    rho = spearmanr(x, y)[0]
    ax.plot([1, 100], [1, 100], ls=(0, (3, 3)), lw=0.9, color="#8A8A8A", zorder=1)
    ax.scatter(x, y, s=16, color=NAVY, alpha=0.75, linewidths=0, zorder=2)
    ax.set_xlim(0, 101)
    ax.set_ylim(101, 0)
    ax.set_xticks([1, 25, 50, 75, 100])
    ax.set_yticks([1, 25, 50, 75, 100])
    ax.set_aspect("equal")
    ax.set_title(rf"{title} ($\rho$ = {rho:.2f})", color=TEXT, fontsize=11, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=FAINT, linewidth=0.5, zorder=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--seed-a", type=int, default=42)
    ap.add_argument("--seed-b", type=int, default=44)
    ap.add_argument("--out", type=Path, default=REPO / "figures/fig_seed_vs_rule_ranks.png")
    args = ap.parse_args()

    for font in sorted((REPO / "fonts").glob("*.ttf")):  # Lato, as in make_figures.py
        font_manager.fontManager.addfont(str(font))
    apply_style()
    rf_a = per_step_reward(args.results, "0p00", args.seed_a)
    rf_b = per_step_reward(args.results, "0p00", args.seed_b)
    strict_a = per_step_reward(args.results, "0p50", args.seed_a)
    bodies = sorted(rf_a)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), constrained_layout=True)
    panel(axes[0], ranks(rf_a, bodies), ranks(rf_b, bodies),
          f"Reset-free, seeds {args.seed_a} and {args.seed_b}",
          f"Rank, Reset-free (seed {args.seed_a})", f"Rank, Reset-free (seed {args.seed_b})")
    panel(axes[1], ranks(rf_a, bodies), ranks(strict_a, bodies),
          f"Seed {args.seed_a}, Reset-free and Strict",
          f"Rank, Reset-free (seed {args.seed_a})", f"Rank, Strict (seed {args.seed_a})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
