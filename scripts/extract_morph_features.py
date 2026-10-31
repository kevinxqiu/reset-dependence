#!/usr/bin/env python3
"""Extract morphological features from UNIMAL XMLs and correlate with protocol sensitivity.

Parses each XML for structural features (n_bodies, n_joints, n_motors,
initial_height, tree_depth) and correlates with RF/Std and Strict/Std
performance ratios from the MJX PPO data.

Usage:
    python scripts/extract_morph_features.py
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ_DIR = Path(__file__).resolve().parent.parent


def default_unimal_xml_dir() -> Path:
    fallback = PROJ_DIR / "assets/unimals_100/xml"
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
RESULTS_DIR = PROJ_DIR / "results"
FIG_DIR = PROJ_DIR / "figures"

SEEDS = [42, 43, 44]
FAMILIES = ["floor", "vt", "mvt"]
FAMILY_LABELS = {"floor": "Floor", "vt": "VT", "mvt": "MVT"}
COLORS = {
    "floor": "#E69F00",
    "vt": "#56B4E9",
    "mvt": "#009E73",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "serif",
        "text.usetex": False,
    }
)

FULL_WIDTH = 7.0


def get_tree_depth(elem):
    """Compute max depth of body tree from a given element."""
    children = [c for c in elem if c.tag == "body"]
    if not children:
        return 0
    return 1 + max(get_tree_depth(c) for c in children)


def extract_features(xml_path):
    """Extract morphological features from a UNIMAL XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bodies = root.findall(".//body")
    joints = [
        j
        for j in root.findall(".//joint")
        if j.get("type", "hinge") == "hinge" and j.get("class") != "free"
    ]
    motors = root.findall(".//motor")
    geoms = root.findall(".//geom")

    torso = root.find('.//body[@name="torso/0"]')
    init_height = 0.0
    tree_depth = 0
    if torso is not None:
        pos = torso.get("pos", "0 0 0").split()
        init_height = float(pos[2])
        tree_depth = get_tree_depth(torso)

    # Estimate total volume from geom sizes (proxy for mass)
    total_volume = 0.0
    for g in geoms:
        gtype = g.get("type", "sphere")
        size_str = g.get("size", "0.05")
        sizes = [float(s) for s in size_str.split()]
        fromto = g.get("fromto")
        if gtype == "sphere" and sizes:
            total_volume += (4 / 3) * np.pi * sizes[0] ** 3
        elif gtype == "capsule":
            r = sizes[0] if sizes else 0.05
            if fromto:
                pts = [float(x) for x in fromto.split()]
                length = np.sqrt(sum((pts[i] - pts[i + 3]) ** 2 for i in range(3)))
            elif len(sizes) > 1:
                length = sizes[1] * 2
            else:
                length = 0.1
            total_volume += np.pi * r**2 * length + (4 / 3) * np.pi * r**3

    # Count limbs at first level (direct children of torso)
    n_first_level_limbs = 0
    if torso is not None:
        n_first_level_limbs = sum(1 for c in torso if c.tag == "body")

    return {
        "n_bodies": len(bodies),
        "n_joints": len(joints),
        "n_motors": len(motors),
        "n_geoms": len(geoms),
        "init_height": init_height,
        "tree_depth": tree_depth,
        "total_volume": total_volume,
        "n_first_level_limbs": n_first_level_limbs,
    }


def load_ratios():
    """Load RF/Std and Strict/Std ratios for each morphology (seed-averaged)."""
    data = {}
    for tau_str in ["0p20", "0p50", "0p00"]:
        for seed in SEEDS:
            path = RESULTS_DIR / f"mjx_ppo_{tau_str}_seed{seed}" / "results.json"
            with open(path) as f:
                data[(tau_str, seed)] = json.load(f)

    morphs = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))

    rf_epi = {}
    str_epi = {}
    for m in morphs:
        epi_psr = np.mean(
            [data[("0p20", s)][m]["eval_tau020"]["per_step_reward"] for s in SEEDS]
        )
        rf_psr = np.mean(
            [data[("0p00", s)][m]["eval_tau020"]["per_step_reward"] for s in SEEDS]
        )
        str_psr = np.mean(
            [data[("0p50", s)][m]["eval_tau020"]["per_step_reward"] for s in SEEDS]
        )
        rf_epi[m] = rf_psr / max(epi_psr, 0.001)
        str_epi[m] = str_psr / max(epi_psr, 0.001)

    return morphs, rf_epi, str_epi


def main():
    morphs, rf_epi, str_epi = load_ratios()

    # Extract features for all morphologies
    features = {}
    for m in morphs:
        xml_path = XML_DIR / f"{m}.xml"
        if xml_path.exists():
            features[m] = extract_features(xml_path)
        else:
            print(f"  WARNING: {xml_path} not found")

    valid_morphs = [m for m in morphs if m in features]
    print(f"Extracted features for {len(valid_morphs)}/{len(morphs)} morphologies")

    # Feature names for correlation
    feature_names = [
        "n_bodies",
        "n_joints",
        "n_motors",
        "init_height",
        "tree_depth",
        "total_volume",
        "n_first_level_limbs",
    ]
    feature_labels = {
        "n_bodies": "Num. bodies",
        "n_joints": "Num. joints",
        "n_motors": "Num. motors",
        "init_height": "Initial height",
        "tree_depth": "Tree depth",
        "total_volume": "Total volume",
        "n_first_level_limbs": "Num. limbs (L1)",
    }

    # Print feature summary by family
    print("\n=== FEATURE SUMMARY BY FAMILY ===")
    for fam in FAMILIES:
        fam_morphs = [m for m in valid_morphs if m.split("-")[0] == fam]
        print(f"\n{FAMILY_LABELS[fam]} (n={len(fam_morphs)}):")
        for fn in feature_names:
            vals = [features[m][fn] for m in fam_morphs]
            print(
                f"  {feature_labels[fn]:20s}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}"
            )

    # Correlate features with RF/Std and Strict/Std ratios
    print("\n=== FEATURE CORRELATIONS WITH PROTOCOL SENSITIVITY ===")
    print(
        f"{'Feature':25s} | {'RF/Std rho':>10s} {'p':>10s} | {'Str/Std rho':>11s} {'p':>10s}"
    )
    print("-" * 75)

    corr_results = {}
    for fn in feature_names:
        feat_vals = np.array([features[m][fn] for m in valid_morphs])
        rf_vals = np.array([rf_epi[m] for m in valid_morphs])
        str_vals = np.array([str_epi[m] for m in valid_morphs])

        rho_rf, p_rf = spearmanr(feat_vals, rf_vals)
        rho_str, p_str = spearmanr(feat_vals, str_vals)

        corr_results[fn] = {
            "rf_rho": float(rho_rf),
            "rf_p": float(p_rf),
            "str_rho": float(rho_str),
            "str_p": float(p_str),
        }

        sig_rf = "*" if p_rf < 0.05 else " "
        sig_str = "*" if p_str < 0.05 else " "
        print(
            f"  {feature_labels[fn]:23s} | {rho_rf:10.3f} {p_rf:10.4f}{sig_rf}| {rho_str:11.3f} {p_str:10.4f}{sig_str}"
        )

    # Generate correlation figure
    sig_features_rf = [fn for fn in feature_names if corr_results[fn]["rf_p"] < 0.10]
    sig_features_str = [fn for fn in feature_names if corr_results[fn]["str_p"] < 0.10]
    all_sig = list(dict.fromkeys(sig_features_rf + sig_features_str))

    if not all_sig:
        all_sig = feature_names[:4]

    n_plots = min(len(all_sig), 4)
    fig, axes = plt.subplots(1, n_plots, figsize=(FULL_WIDTH, 2.5))
    if n_plots == 1:
        axes = [axes]

    for ax_idx, fn in enumerate(all_sig[:n_plots]):
        ax = axes[ax_idx]
        for fam in FAMILIES:
            fam_morphs = [m for m in valid_morphs if m.split("-")[0] == fam]
            x = [features[m][fn] for m in fam_morphs]
            y = [rf_epi[m] for m in fam_morphs]
            ax.scatter(
                x,
                y,
                c=COLORS[fam],
                label=FAMILY_LABELS[fam] if ax_idx == 0 else None,
                s=15,
                alpha=0.6,
                edgecolors="none",
            )

        rho = corr_results[fn]["rf_rho"]
        p = corr_results[fn]["rf_p"]
        ax.set_xlabel(feature_labels[fn])
        if ax_idx == 0:
            ax.set_ylabel("RF / Standard ratio")
        ax.set_title(f"$\\rho$={rho:.2f} (p={p:.3f})")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if n_plots > 0:
        axes[0].legend(frameon=False, loc="upper left", markerscale=1.5, fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_feature_correlations.pdf")
    fig.savefig(FIG_DIR / "fig_feature_correlations.png")
    plt.close(fig)
    print("\n  Saved fig_feature_correlations")

    # Save features to JSON
    out = {
        m: {**features[m], "rf_epi_ratio": rf_epi[m], "str_epi_ratio": str_epi[m]}
        for m in valid_morphs
    }
    out_path = PROJ_DIR / "results" / "morph_features.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Features saved to {out_path}")

    # Also save correlation table
    corr_path = PROJ_DIR / "results" / "feature_correlations.json"
    with open(corr_path, "w") as f:
        json.dump(corr_results, f, indent=2)
    print(f"  Correlations saved to {corr_path}")


if __name__ == "__main__":
    main()
