#!/usr/bin/env python3
"""Threshold-agnostic locomotion-quality audit of the 900-run corpus.

A fixed-threshold learner/penalized/non-learner classifier calibrated on
episodic data is vulnerable to circularity because the thresholds encode
an implicit episodic baseline. Instead, this script reports continuous,
threshold-free measures of locomotion quality and surfaces the specific
cross-protocol transitions that matter for the paper's "reset-free hurts
Floor bodies" claim.

Outputs:
  records/locomotion_quality_report.md
  figures/fig_locomotion_quality.pdf

Three quantities per (morphology, protocol):
  1. mean distance traveled per evaluation episode (eval_tau020)
  2. mean PSR per step
  3. mean ep_len (how long the agent stays up)

Core transition table: for each family, how many bodies
  (a) travel >= 5 m under episodic,
  (b) AND travel < 1 m under reset-free, under the same evaluator,
vs how many bodies do the symmetric swap (episodic < 1 m, reset-free >= 5 m).

No fixed bin labels cross the episodic-to-reset-free boundary in calibration.
The 1 m / 5 m anchors are absolute distances in the evaluation protocol and
apply identically across all three training protocols.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "results"
OUT_MD = REPO / "records/locomotion_quality_report.md"
OUT_FIG = REPO / "figures/fig_locomotion_quality.pdf"

PROTOCOLS = {
    "episodic": "0p20",
    "strict": "0p50",
    "reset_free": "0p00",
}
SEEDS = (42, 43, 44)
FAMILIES = ("floor", "vt", "mvt")

# Absolute-distance anchors (per 1000-step evaluation episode at tau_eval=0.20).
# 1 m is well below any morphology's one-gait-cycle forward motion, so bodies
# under 1 m are effectively stationary. 5 m corresponds to roughly half of a
# median-PSR gait over 1000 steps and is the practical "walks" threshold.
STATIONARY_DIST_M = 1.0
WALKS_DIST_M = 5.0


def family_of(name: str) -> str:
    return name.split("-", 1)[0].lower()


def load_corpus(eval_key: str = "eval_tau020") -> dict[str, dict[str, dict[str, float]]]:
    """{morph: {protocol: {distance, psr, ep_len}}} seed-averaged.

    eval_key selects the evaluation protocol at which distance is measured.
    eval_tau020 matches the main-text evaluation (episodes may terminate at
    z < 0.20 z_init); eval_neutral has NO early termination (always full
    1000 steps), providing a truly threshold-independent distance.
    """
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for protocol, tau in PROTOCOLS.items():
        for seed in SEEDS:
            path = CORPUS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with path.open() as f:
                data = json.load(f)
            for morph, rec in data.items():
                ev = rec[eval_key]
                raw[morph][protocol]["distance"].append(float(ev["distance"]))
                raw[morph][protocol]["psr"].append(float(ev["per_step_reward"]))
                raw[morph][protocol]["ep_len"].append(float(ev["mean_ep_len"]))
    return {
        morph: {
            proto: {k: float(np.mean(v)) for k, v in d.items()}
            for proto, d in protos.items()
        }
        for morph, protos in raw.items()
    }


def transition_table(corpus: dict, anchor_low: float, anchor_high: float):
    """Count (episodic walks, reset_free stationary) and the symmetric swap per family."""
    counts: dict[str, dict[str, int]] = {f: defaultdict(int) for f in FAMILIES}
    for morph, protos in corpus.items():
        fam = family_of(morph)
        if fam not in counts:
            continue
        d_epi = protos["episodic"]["distance"]
        d_rf = protos["reset_free"]["distance"]
        d_str = protos["strict"]["distance"]
        counts[fam]["total"] += 1
        counts[fam]["epi_walks"] += d_epi >= anchor_high
        counts[fam]["epi_stationary"] += d_epi < anchor_low
        counts[fam]["rf_walks"] += d_rf >= anchor_high
        counts[fam]["rf_stationary"] += d_rf < anchor_low
        counts[fam]["str_walks"] += d_str >= anchor_high
        counts[fam]["str_stationary"] += d_str < anchor_low
        # The core cross-protocol transitions:
        if d_epi >= anchor_high and d_rf < anchor_low:
            counts[fam]["walks_epi_then_stationary_rf"] += 1
        if d_epi < anchor_low and d_rf >= anchor_high:
            counts[fam]["stationary_epi_then_walks_rf"] += 1
        if d_epi >= anchor_high and d_str < anchor_low:
            counts[fam]["walks_epi_then_stationary_str"] += 1
        if d_epi < anchor_low and d_str >= anchor_high:
            counts[fam]["stationary_epi_then_walks_str"] += 1
    return counts


def plot_quality(corpus: dict, out_path: Path) -> None:
    """Scatter: distance vs PSR, faceted by protocol, colored by family.

    Adds the 1 m / 5 m stationary/walks anchors as vertical lines.
    """
    fam_colors = {"floor": "#E69F00", "vt": "#56B4E9", "mvt": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    titles = {
        "episodic": "Standard (tau=0.20)",
        "strict": "Strict (tau=0.50)",
        "reset_free": "Reset-free (tau=0)",
    }
    for ax, (proto, title) in zip(axes, titles.items(), strict=False):
        for fam in FAMILIES:
            xs = [corpus[m][proto]["distance"] for m in corpus if family_of(m) == fam]
            ys = [corpus[m][proto]["psr"] for m in corpus if family_of(m) == fam]
            ax.scatter(
                xs,
                ys,
                c=fam_colors[fam],
                alpha=0.7,
                s=35,
                label=fam.upper(),
                edgecolors="none",
            )
        ax.axvline(STATIONARY_DIST_M, color="grey", linestyle=":", linewidth=0.8)
        ax.axvline(WALKS_DIST_M, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("mean distance per episode (m)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-2, 80)
        ax.set_ylim(-0.5, 5)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("per-step reward")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    corpus = load_corpus("eval_tau020")
    corpus_neutral = load_corpus("eval_neutral")
    counts = transition_table(corpus, STATIONARY_DIST_M, WALKS_DIST_M)
    counts_neutral = transition_table(
        corpus_neutral, STATIONARY_DIST_M, WALKS_DIST_M
    )
    plot_quality(corpus, OUT_FIG)

    lines: list[str] = []
    lines.append("# Locomotion Quality Audit")
    lines.append("")
    lines.append(
        f"Threshold-free distance-based classification at anchors "
        f"{STATIONARY_DIST_M} m and {WALKS_DIST_M} m."
    )
    lines.append("")
    lines.append("## Per-family counts")
    lines.append("")
    header = [
        "family",
        "n",
        "epi_walks",
        "epi_stat",
        "rf_walks",
        "rf_stat",
        "str_walks",
        "str_stat",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + " --- |" * len(header))
    for fam in FAMILIES:
        c = counts[fam]
        lines.append(
            "| "
            + " | ".join(
                [
                    fam,
                    str(c["total"]),
                    str(c["epi_walks"]),
                    str(c["epi_stationary"]),
                    str(c["rf_walks"]),
                    str(c["rf_stationary"]),
                    str(c["str_walks"]),
                    str(c["str_stationary"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Cross-protocol transitions")
    lines.append("")
    lines.append(
        f"Bodies that walk under episodic (>= {WALKS_DIST_M} m) but are "
        f"stationary (< {STATIONARY_DIST_M} m) under the other protocol:"
    )
    lines.append("")
    lines.append(
        "| family | n | learns_then_stops_under_RF | learns_then_stops_under_Strict | "
        "stops_then_learns_under_RF | stops_then_learns_under_Strict |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for fam in FAMILIES:
        c = counts[fam]
        lines.append(
            "| "
            + " | ".join(
                [
                    fam,
                    str(c["total"]),
                    str(c["walks_epi_then_stationary_rf"]),
                    str(c["walks_epi_then_stationary_str"]),
                    str(c["stationary_epi_then_walks_rf"]),
                    str(c["stationary_epi_then_walks_str"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("")
    lines.append("## Threshold-free check: same table at tau_eval = 0 (no early termination)")
    lines.append("")
    lines.append(
        "At tau_eval = 0 every evaluation episode runs the full 1000 steps,"
        " so distance is strictly integrated forward motion with no threshold"
        " dependence."
    )
    lines.append("")
    lines.append(
        "| family | n | learns_then_stops_under_RF | learns_then_stops_under_Strict | "
        "stops_then_learns_under_RF | stops_then_learns_under_Strict |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for fam in FAMILIES:
        c = counts_neutral[fam]
        lines.append(
            "| "
            + " | ".join(
                [
                    fam,
                    str(c["total"]),
                    str(c["walks_epi_then_stationary_rf"]),
                    str(c["walks_epi_then_stationary_str"]),
                    str(c["stationary_epi_then_walks_rf"]),
                    str(c["stationary_epi_then_walks_str"]),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The cross-protocol transition table counts bodies that travel at least"
        f" {WALKS_DIST_M} m under episodic training and less than {STATIONARY_DIST_M} m"
        " under reset-free or strict training, and the reverse. The counts depend on"
        " the evaluator (compare the tables above), so on their own they do not show"
        " whether training prevented locomotion."
    )
    lines.append("")
    lines.append("## Figure")
    lines.append("")
    lines.append(
        f"See `{OUT_FIG.relative_to(REPO)}` for the distance-vs-PSR scatter"
        " faceted by protocol."
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_FIG}")

    # Also print the core transitions to stdout for fast inspection
    for fam in FAMILIES:
        c = counts[fam]
        print(
            f"  {fam}: n={c['total']} "
            f"walks_epi->stationary_rf={c['walks_epi_then_stationary_rf']} "
            f"walks_epi->stationary_str={c['walks_epi_then_stationary_str']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
