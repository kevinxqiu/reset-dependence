#!/usr/bin/env python3
"""Final analysis for NeurIPS 2026: Reset Dependence Is Embodied.

Regenerates the body figures (fig1-fig5, fig_reversal) and per-body
statistics from the MJX-PPO 5M data: 100 common UNIMAL morphologies x
3 conditions (tau in {0.00, 0.20, 0.50}) x 3 seeds ({42, 43, 44}).
fig3 (threshold curves) additionally loads tau in {0.10, 0.35} runs
under the same seeds for the 5-threshold continuous panel.

Figure fonts are configured as sans-serif Lato with Type 42 PDF
embedding for NeurIPS submission font compliance.

Usage:
    python scripts/figures/analyze_final.py --out_dir figures --stats_out figures/final_stats.json
"""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, f_oneway, kruskal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJ_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__import__("os").environ.get("RFC_RESULTS", str(PROJ_DIR / "results")))
DISCOVERY_DIR = PROJ_DIR / "results"

SEEDS = [42, 43, 44]
CONDITIONS = {"0p20": "Standard", "0p50": "Strict", "0p00": "Reset-free"}
FAMILIES = ["floor", "vt", "mvt"]
FAMILY_LABELS = {"floor": "Floor", "vt": "VT", "mvt": "MVT"}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import (
    DEEMPH,
    FAMILY_COLOR,
    FAMILY_LABEL,
    FAMILY_MARKER,
    FULL_WIDTH,
    PROTOCOL_COLOR,
    TEXT,
    apply_style,
    column_header,
    light_grid,
    tufte_axes,
)
COLORS = FAMILY_COLOR  # compatibility alias for appendix figures

apply_style()

COND_COLORS = {
    "0p20": PROTOCOL_COLOR["episodic"],
    "0p50": PROTOCOL_COLOR["strict"],
    "0p00": PROTOCOL_COLOR["reset_free"],
}
COND_LABEL = {"0p20": "Standard", "0p50": "Strict", "0p00": "Reset-free"}
COND_KEY = {"0p20": "episodic", "0p50": "strict", "0p00": "reset_free"}

APPENDIX_WIDTH = 5.5


def load_all():
    """Load all confirmatory data (MJX-PPO 5M)."""
    data = {}
    for tau_str in CONDITIONS:
        for seed in SEEDS:
            path = RESULTS_DIR / f"mjx_ppo_{tau_str}_seed{seed}" / "results.json"
            with open(path) as f:
                data[(tau_str, seed)] = json.load(f)
    return data


def load_discovery():
    """Load discovery data (single-seed, intermediate thresholds)."""
    disc = {}
    for thr in ["0p05", "0p10", "0p30", "0p35", "0p40", "0p45", "0p50"]:
        path = DISCOVERY_DIR / f"threshold_{thr}" / "results.json"
        if path.exists():
            with open(path) as f:
                disc[thr] = json.load(f)
    return disc


def get_common_morphs(data):
    """Get morphologies present in all conditions and seeds."""
    morph_sets = [set(d.keys()) for d in data.values()]
    return sorted(set.intersection(*morph_sets))


def build_family_index(morphs):
    """Map family name to list of indices into morphs."""
    fam_idx = {f: [] for f in FAMILIES}
    for i, m in enumerate(morphs):
        fam = m.split("-")[0]
        if fam in fam_idx:
            fam_idx[fam].append(i)
    return fam_idx


def seed_avg(data, morphs, tau_str, eval_key, metric):
    """Compute seed-averaged metric for each morphology."""
    vals = []
    for m in morphs:
        sv = [data[(tau_str, s)][m][eval_key][metric] for s in SEEDS]
        vals.append(np.mean(sv))
    return np.array(vals)


def compute_main_stats(data, morphs, fam_idx):
    """Compute all headline statistics."""
    stats = {}

    # Per-step reward and distance for each condition (seed-averaged)
    metrics = {}
    for tau in CONDITIONS:
        metrics[tau] = {
            "psr": seed_avg(data, morphs, tau, "eval_tau020", "per_step_reward"),
            "distance": seed_avg(data, morphs, tau, "eval_tau020", "distance"),
            "fitness": seed_avg(data, morphs, tau, "eval_tau020", "fitness"),
            "ep_len": seed_avg(data, morphs, tau, "eval_tau020", "mean_ep_len"),
            "fall_rate": seed_avg(data, morphs, tau, "eval_tau020", "fall_rate"),
        }

    # Family x condition table
    family_table = {}
    for fam in FAMILIES:
        ix = fam_idx[fam]
        family_table[fam] = {}
        for tau, label in CONDITIONS.items():
            family_table[fam][label] = {
                "psr": float(np.mean(metrics[tau]["psr"][ix])),
                "psr_std": float(np.std(metrics[tau]["psr"][ix])),
                "distance": float(np.mean(metrics[tau]["distance"][ix])),
                "distance_std": float(np.std(metrics[tau]["distance"][ix])),
                "fitness": float(np.mean(metrics[tau]["fitness"][ix])),
                "fall_rate": float(np.mean(metrics[tau]["fall_rate"][ix])),
                "n": len(ix),
            }

    # RF/Std and Strict/Std ratios (mean-of-ratios, per morphology)
    rf_epi_ratios = metrics["0p00"]["psr"] / np.maximum(metrics["0p20"]["psr"], 0.001)
    str_epi_ratios = metrics["0p50"]["psr"] / np.maximum(metrics["0p20"]["psr"], 0.001)

    stats["family_table"] = family_table
    stats["rf_epi_ratios"] = {
        fam: {
            "mean": float(np.mean(rf_epi_ratios[fam_idx[fam]])),
            "std": float(np.std(rf_epi_ratios[fam_idx[fam]])),
            "values": rf_epi_ratios[fam_idx[fam]].tolist(),
        }
        for fam in FAMILIES
    }
    stats["str_epi_ratios"] = {
        fam: {
            "mean": float(np.mean(str_epi_ratios[fam_idx[fam]])),
            "std": float(np.std(str_epi_ratios[fam_idx[fam]])),
            "values": str_epi_ratios[fam_idx[fam]].tolist(),
        }
        for fam in FAMILIES
    }

    # ANOVA on RF/Std ratios
    F_rf, p_rf = f_oneway(
        rf_epi_ratios[fam_idx["floor"]],
        rf_epi_ratios[fam_idx["vt"]],
        rf_epi_ratios[fam_idx["mvt"]],
    )
    H_rf, p_kw_rf = kruskal(
        rf_epi_ratios[fam_idx["floor"]],
        rf_epi_ratios[fam_idx["vt"]],
        rf_epi_ratios[fam_idx["mvt"]],
    )

    # ANOVA on Strict/Std ratios
    F_str, p_str = f_oneway(
        str_epi_ratios[fam_idx["floor"]],
        str_epi_ratios[fam_idx["vt"]],
        str_epi_ratios[fam_idx["mvt"]],
    )

    stats["anova_rf_epi"] = {"F": float(F_rf), "p": float(p_rf)}
    stats["kruskal_rf_epi"] = {"H": float(H_rf), "p": float(p_kw_rf)}
    stats["anova_str_epi"] = {"F": float(F_str), "p": float(p_str)}

    # Distance ANOVA
    dist_rf_ratios = metrics["0p00"]["distance"] / np.maximum(
        metrics["0p20"]["distance"], 0.001
    )
    F_d, p_d = f_oneway(
        dist_rf_ratios[fam_idx["floor"]],
        dist_rf_ratios[fam_idx["vt"]],
        dist_rf_ratios[fam_idx["mvt"]],
    )
    stats["anova_distance_rf"] = {"F": float(F_d), "p": float(p_d)}

    # Effect sizes
    floor_rf = rf_epi_ratios[fam_idx["floor"]]
    vt_rf = rf_epi_ratios[fam_idx["vt"]]
    pooled_std = np.sqrt(
        (np.var(floor_rf, ddof=1) * len(floor_rf) + np.var(vt_rf, ddof=1) * len(vt_rf))
        / (len(floor_rf) + len(vt_rf))
    )
    stats["cohens_d_floor_vt"] = float(
        (np.mean(vt_rf) - np.mean(floor_rf)) / pooled_std
    )

    all_ratios = np.concatenate([rf_epi_ratios[fam_idx[f]] for f in FAMILIES])
    grand_mean = np.mean(all_ratios)
    ss_between = sum(
        len(fam_idx[f]) * (np.mean(rf_epi_ratios[fam_idx[f]]) - grand_mean) ** 2
        for f in FAMILIES
    )
    ss_total = np.sum((all_ratios - grand_mean) ** 2)
    stats["eta_squared"] = float(ss_between / ss_total)

    # Bootstrap CI for VT - Floor difference
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(10000):
        f_boot = rng.choice(floor_rf, len(floor_rf), replace=True)
        v_boot = rng.choice(vt_rf, len(vt_rf), replace=True)
        diffs.append(np.mean(v_boot) - np.mean(f_boot))
    diffs = np.array(diffs)
    stats["bootstrap_vt_floor"] = {
        "mean": float(np.mean(diffs)),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "ci99": [float(np.percentile(diffs, 0.5)), float(np.percentile(diffs, 99.5))],
        "p_positive": float(np.mean(diffs > 0)),
    }

    rho_corrs = {}
    for n1, t1, n2, t2 in [
        ("Standard", "0p20", "Reset-free", "0p00"),
        ("Standard", "0p20", "Strict", "0p50"),
        ("Reset-free", "0p00", "Strict", "0p50"),
    ]:
        rho, p = spearmanr(metrics[t1]["psr"], metrics[t2]["psr"])
        rho_corrs[f"{n1} vs {n2}"] = {"rho": float(rho), "p": float(p)}
    stats["ranking_correlations"] = rho_corrs

    # Per-seed replication
    per_seed = {}
    for seed in SEEDS:
        epi_data = data[("0p20", seed)]
        rf_data = data[("0p00", seed)]
        common = sorted(set(epi_data.keys()) & set(rf_data.keys()))
        fam_idx_seed = {f: [] for f in FAMILIES}
        for i, m in enumerate(common):
            fam = m.split("-")[0]
            if fam in fam_idx_seed:
                fam_idx_seed[fam].append(i)

        epi_psr = np.array(
            [epi_data[m]["eval_tau020"]["per_step_reward"] for m in common]
        )
        rf_psr = np.array(
            [rf_data[m]["eval_tau020"]["per_step_reward"] for m in common]
        )
        ratios = rf_psr / np.maximum(epi_psr, 0.001)

        fam_means = {f: float(np.mean(ratios[fam_idx_seed[f]])) for f in FAMILIES}
        F, p = f_oneway(*[ratios[fam_idx_seed[f]] for f in FAMILIES])
        per_seed[seed] = {"family_means": fam_means, "F": float(F), "p": float(p)}
    stats["per_seed"] = per_seed

    winner_overlap = {}
    for tau, label in CONDITIONS.items():
        ranked = np.argsort(-metrics[tau]["psr"])[:10]
        winner_overlap[label] = [morphs[i] for i in ranked]

    epi_top10 = set(winner_overlap["Standard"])
    rf_top10 = set(winner_overlap["Reset-free"])
    str_top10 = set(winner_overlap["Strict"])
    stats["winner_overlap"] = {
        "top10": winner_overlap,
        "epi_rf_overlap": len(epi_top10 & rf_top10),
        "epi_str_overlap": len(epi_top10 & str_top10),
        "rf_str_overlap": len(rf_top10 & str_top10),
        "all_overlap": len(epi_top10 & rf_top10 & str_top10),
    }

    # Family composition of top-10
    for label in CONDITIONS.values():
        fam_comp = {}
        for m in winner_overlap[label]:
            f = m.split("-")[0]
            fam_comp[f] = fam_comp.get(f, 0) + 1
        stats["winner_overlap"][f"top10_{label}_families"] = fam_comp

    return stats, metrics, rf_epi_ratios, str_epi_ratios


def _bootstrap_mean_ci(vals, rng, *, n_boot=5000):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return (
        float(vals.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def _dot_ci(ax, x, vals, color, rng, *, jitter=0.035):
    vals = np.asarray(vals, dtype=float)
    xs = x + rng.normal(0.0, jitter, size=len(vals))
    ax.scatter(xs, vals, s=9, color=color, alpha=0.26, edgecolors="none", zorder=2)
    mean, lo, hi = _bootstrap_mean_ci(vals, rng)
    ax.plot(
        [x - 0.07, x + 0.07],
        [mean, mean],
        color=color,
        linewidth=1.7,
        solid_capstyle="round",
        zorder=6,
    )
    ax.errorbar(
        x,
        mean,
        yerr=[[mean - lo], [hi - mean]],
        fmt="s",
        markersize=5.0,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.9,
        ecolor=color,
        elinewidth=1.25,
        capsize=3.0,
        zorder=5,
    )
    return mean, lo, hi


def fig1_family_condition_bars(metrics, fam_idx, morphs, out_dir):
    """Fig 1: family x protocol distributions + PSR ratio vs Standard.

    Three panels: PSR, Distance, PSR ratio vs. Standard. Points are
    individual morphologies; large markers and intervals show the family
    mean with a morphology-bootstrap 95% CI. Protocol carries colour
    (RF navy, Strict orange, Standard teal); an inline legend carries the key.
    """
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.25))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.18, wspace=0.34)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=COND_COLORS[tau],
            markeredgecolor=COND_COLORS[tau],
            markersize=5.0,
            label=COND_LABEL[tau],
        )
        for tau in ("0p20", "0p50", "0p00")
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.1,
        fontsize=8.0,
    )

    rng = np.random.default_rng(7)
    offset = 0.23
    n_fams = len(FAMILIES)
    x = np.arange(n_fams)
    cond_order = ("0p20", "0p50", "0p00")  # Standard, Strict, RF

    for ax_idx, (metric_key, ylabel, header) in enumerate(
        [
            ("psr", "Per-step reward", "Per-step reward"),
            ("distance", "Distance (m)", "Distance"),
        ]
    ):
        ax = axes[ax_idx]
        max_top = 0.0
        min_bot = 0.0
        for i, tau in enumerate(cond_order):
            for fam_i, fam in enumerate(FAMILIES):
                vals = metrics[tau][metric_key][fam_idx[fam]]
                _, _, hi = _dot_ci(
                    ax,
                    x[fam_i] + (i - 1) * offset,
                    vals,
                    COND_COLORS[tau],
                    rng,
                )
                max_top = max(max_top, hi, float(np.max(vals)))
                min_bot = min(min_bot, float(np.min(vals)))
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES])
        ax.set_ylabel(ylabel)
        if min_bot < 0:
            ax.axhline(0.0, color=DEEMPH, linewidth=0.5, linestyle="-", alpha=0.45, zorder=1)
        ax.set_ylim(min_bot - 0.05 * (max_top - min_bot) if min_bot < 0 else 0, max_top * 1.15)
        column_header(ax, header)
        light_grid(ax, axis="y")
        tufte_axes(ax)

    # Panel 3: PSR ratio vs Standard (mean of per-body ratios).
    ax = axes[2]
    max_top = 1.0
    min_bot = 0.0
    ratio_specs = (
        ("0p50", "Strict / Standard", -offset / 2),
        ("0p00", "Reset-free / Standard", offset / 2),
    )
    for tau, _label, dx in ratio_specs:
        for fam_i, fam in enumerate(FAMILIES):
            idx = fam_idx[fam]
            epi = np.asarray(metrics["0p20"]["psr"][idx], dtype=float)
            vals = np.asarray(metrics[tau]["psr"][idx], dtype=float) / np.maximum(epi, 0.001)
            _, _, hi = _dot_ci(ax, x[fam_i] + dx, vals, COND_COLORS[tau], rng, jitter=0.03)
            max_top = max(max_top, hi, float(np.max(vals)))
            min_bot = min(min_bot, float(np.min(vals)))
    ax.axhline(1.0, color=DEEMPH, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(0.0, color=DEEMPH, linewidth=0.5, linestyle="-", alpha=0.45, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES])
    ax.set_ylabel("PSR ratio (vs. Standard)")
    ax.set_ylim(min_bot - 0.05 * (max_top - min_bot), max_top + 0.05 * (max_top - min_bot))
    column_header(ax, "PSR ratio")
    light_grid(ax, axis="y")
    tufte_axes(ax)

    fig.savefig(out_dir / "fig1_family_condition_bars.pdf")
    fig.savefig(out_dir / "fig1_family_condition_bars.png")
    plt.close(fig)
    print("  Saved fig1_family_condition_bars")


def fig2_scatter(metrics, fam_idx, morphs, out_dir):
    """Fig 2: Scatter plots (episodic vs RF, episodic vs strict)."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.9))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.16, wspace=0.24)

    for ax_idx, (comp_tau, comp_label) in enumerate(
        [
            ("0p00", "Reset-free"),
            ("0p50", "Strict"),
        ]
    ):
        ax = axes[ax_idx]
        epi = metrics["0p20"]["psr"]
        comp = metrics[comp_tau]["psr"]

        for fam in FAMILIES:
            ix = fam_idx[fam]
            ax.scatter(
                epi[ix],
                comp[ix],
                c=COLORS[fam],
                marker=FAMILY_MARKER[fam],
                label=FAMILY_LABELS[fam],
                s=24,
                alpha=0.72,
                edgecolors="white",
                linewidths=0.35,
            )

        lim_max = max(np.max(epi), np.max(comp)) * 1.05
        ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Standard per-step reward")
        ax.set_ylabel(f"{comp_label} per-step reward")
        column_header(ax, f"Standard vs. {comp_label}")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_aspect("equal")
        light_grid(ax, axis="both")
        tufte_axes(ax)
        if ax_idx == 0:
            ax.legend(frameon=False, loc="upper left", markerscale=1.5)

        # Add rho annotation
        rho, p = spearmanr(epi, comp)
        ax.text(
            0.97,
            0.03,
            f"$\\rho$={rho:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="gray",
        )

    fig.savefig(out_dir / "fig2_scatter.pdf")
    fig.savefig(out_dir / "fig2_scatter.png")
    plt.close(fig)
    print("  Saved fig2_scatter")


def fig3_threshold_curves(fam_idx, morphs, out_dir):
    """Fig 3: Mean per-step reward vs training tau at five confirmatory thresholds."""
    tau_strs = ["0p00", "0p10", "0p20", "0p35", "0p50"]
    tau_vals = [float(t.replace("p", ".")) for t in tau_strs]

    # Load 3-seed data for all five thresholds.
    confirm = {}
    for tau_str in tau_strs:
        for seed in SEEDS:
            path = RESULTS_DIR / f"mjx_ppo_{tau_str}_seed{seed}" / "results.json"
            if not path.exists():
                raise FileNotFoundError(f"threshold curves need {path} (runs.zip)")
            with open(path) as f:
                confirm[(tau_str, seed)] = json.load(f)

    # Common morphologies across all thresholds and seeds.
    common_morphs = sorted(
        set.intersection(*[set(confirm[k].keys()) for k in confirm])
    )
    fam_idx_c = {f: [] for f in FAMILIES}
    for i, m in enumerate(common_morphs):
        fam = m.split("-")[0]
        if fam in fam_idx_c:
            fam_idx_c[fam].append(i)

    fig, ax = plt.subplots(figsize=(APPENDIX_WIDTH, 2.8))
    rng = np.random.default_rng(12)
    for fam in FAMILIES:
        ix = fam_idx_c[fam]
        if not ix:
            continue
        mean_psr = []
        lo_psr = []
        hi_psr = []
        for tau_str in tau_strs:
            vals = []
            for i in ix:
                m = common_morphs[i]
                sv = [
                    confirm[(tau_str, s)][m]["eval_tau020"]["per_step_reward"]
                    for s in SEEDS
                    if m in confirm[(tau_str, s)]
                ]
                if sv:
                    vals.append(np.mean(sv))
            if vals:
                mean, lo, hi = _bootstrap_mean_ci(vals, rng, n_boot=5000)
            else:
                mean, lo, hi = np.nan, np.nan, np.nan
            mean_psr.append(mean)
            lo_psr.append(lo)
            hi_psr.append(hi)
        ax.fill_between(
            tau_vals,
            lo_psr,
            hi_psr,
            color=COLORS[fam],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            tau_vals,
            mean_psr,
            marker=FAMILY_MARKER[fam],
            color=COLORS[fam],
            label=FAMILY_LABELS[fam],
            markersize=4.8,
            linewidth=1.55,
            markerfacecolor="white",
            markeredgewidth=1.1,
            zorder=3,
        )

    ax.set_xlabel(r"Training termination threshold ($\tau$)")
    ax.set_ylabel("Mean per-step reward\n(evaluated at $\\tau$=0.20)")
    ax.set_ylim(1.0, 3.35)
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985),
              borderaxespad=0.0, fontsize=8.0, handletextpad=0.35, columnspacing=1.0)
    ax.axvline(0.20, color=DEEMPH, linestyle=(0, (2, 2)), linewidth=0.7, alpha=0.7)
    ax.axvline(0.50, color=DEEMPH, linestyle=(0, (2, 2)), linewidth=0.7, alpha=0.7)
    light_grid(ax, axis="y")
    tufte_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_threshold_curves.pdf")
    fig.savefig(out_dir / "fig3_threshold_curves.png")
    plt.close(fig)
    print("  Saved fig3_threshold_curves")


def fig4_ratio_distributions(rf_epi_ratios, str_epi_ratios, fam_idx, out_dir):
    """Fig 4: Distribution of RF/Std and Strict/Std ratios by family."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.83, bottom=0.16, wspace=0.16)
    rng = np.random.default_rng(4)

    for ax_idx, (ratios, title) in enumerate(
        [
            (rf_epi_ratios, "Reset-free / Standard"),
            (str_epi_ratios, "Strict / Standard"),
        ]
    ):
        ax = axes[ax_idx]
        positions = []
        data_groups = []
        colors_list = []
        for i, fam in enumerate(FAMILIES):
            ix = fam_idx[fam]
            positions.append(i)
            data_groups.append(ratios[ix])
            colors_list.append(COLORS[fam])

        bp = ax.boxplot(
            data_groups,
            positions=positions,
            widths=0.42,
            patch_artist=True,
            showfliers=True,
            flierprops={"markersize": 2.5, "alpha": 0.4},
            medianprops={"color": TEXT, "linewidth": 1.0},
            whiskerprops={"linewidth": 0.85, "color": DEEMPH},
            capprops={"linewidth": 0.85, "color": DEEMPH},
        )
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.18)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.0)
        for pos, vals, color, fam in zip(positions, data_groups, colors_list, FAMILIES):
            xs = pos + rng.normal(0, 0.035, size=len(vals))
            ax.scatter(
                xs,
                vals,
                s=10,
                marker=FAMILY_MARKER[fam],
                facecolors=color,
                edgecolors="none",
                alpha=0.32,
                zorder=2,
            )
            ax.scatter(
                pos,
                np.mean(vals),
                s=32,
                marker="D",
                facecolors="white",
                edgecolors=color,
                linewidths=1.1,
                zorder=4,
            )

        ax.axhline(1.0, color=DEEMPH, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
        ax.set_xticks(positions)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILIES])
        if ax_idx == 0:
            ax.set_ylabel("PSR ratio")
        column_header(ax, title)
        light_grid(ax, axis="y")
        tufte_axes(ax)

    fig.savefig(out_dir / "fig4_ratio_distributions.pdf")
    fig.savefig(out_dir / "fig4_ratio_distributions.png")
    plt.close(fig)
    print("  Saved fig4_ratio_distributions")


def fig5_per_seed(data, morphs, fam_idx, out_dir):
    """Fig 5: per-seed Standard-vs-Reset-free PSR (3 small multiples).

    Family is encoded by both colour and marker because the appendix point of
    this diagnostic is whether the family split persists seed by seed. The
    diagonal is the parity reference; ANOVA F/p is kept in the panel header so
    it cannot obscure dense scatter regions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.12), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.72, bottom=0.18, wspace=0.10)

    seed_payload = {}
    global_lim = 0.0
    for seed in SEEDS:
        epi_psr = np.array(
            [data[("0p20", seed)][m]["eval_tau020"]["per_step_reward"] for m in morphs]
        )
        rf_psr = np.array(
            [data[("0p00", seed)][m]["eval_tau020"]["per_step_reward"] for m in morphs]
        )
        ratios = rf_psr / np.maximum(epi_psr, 0.001)
        F, p = f_oneway(*[ratios[fam_idx[f]] for f in FAMILIES])
        seed_payload[seed] = (epi_psr, rf_psr, F, p)
        global_lim = max(global_lim, float(np.max(epi_psr)), float(np.max(rf_psr)))

    lim = global_lim * 1.05

    def _fmt_p(p):
        return r"$p<0.001$" if p < 0.001 else f"$p$={p:.3f}"

    for ax_idx, seed in enumerate(SEEDS):
        ax = axes[ax_idx]
        epi_psr, rf_psr, F, p = seed_payload[seed]
        for fam in FAMILIES:
            ix = fam_idx[fam]
            ax.scatter(
                epi_psr[ix],
                rf_psr[ix],
                marker=FAMILY_MARKER[fam],
                s=18,
                facecolors="white",
                edgecolors=FAMILY_COLOR[fam],
                linewidths=0.85,
                alpha=0.74,
                zorder=3,
                clip_on=False,
            )
            ax.scatter(
                float(np.mean(epi_psr[ix])),
                float(np.mean(rf_psr[ix])),
                marker=FAMILY_MARKER[fam],
                s=54,
                facecolors=FAMILY_COLOR[fam],
                edgecolors=TEXT,
                linewidths=0.65,
                zorder=5,
                clip_on=False,
            )

        ax.plot([0, lim], [0, lim], color=DEEMPH, linestyle=(0, (4, 3)),
                linewidth=0.8, zorder=1)
        ax.set_xlabel("Standard PSR")
        if ax_idx == 0:
            ax.set_ylabel("Reset-free PSR")
        column_header(ax, f"Seed {seed}  $F$={F:.1f}, {_fmt_p(p)}", size=8.2)
        light_grid(ax, axis="both")
        pad = 0.35
        ax.set_xlim(-pad, lim)
        ax.set_ylim(-pad, lim)
        ax.set_aspect("equal")
        tufte_axes(ax)

    # Figure-level shape key avoids covering dense points in the small multiples.
    handles = [
        plt.Line2D([0], [0], marker=FAMILY_MARKER[f], color="white",
                   markerfacecolor="white",
                   markeredgecolor=FAMILY_COLOR[f],
                   markeredgewidth=1.0, markersize=6,
                   label=FAMILY_LABEL[f])
        for f in FAMILIES
    ]
    fig.legend(handles=handles, loc="upper center", frameon=False, ncol=3,
               bbox_to_anchor=(0.5, 0.995), fontsize=8, handlelength=0.8,
               handletextpad=0.4, columnspacing=1.2)

    fig.savefig(out_dir / "fig5_per_seed.pdf")
    fig.savefig(out_dir / "fig5_per_seed.png")
    plt.close(fig)
    print("  Saved fig5_per_seed")


def fig_reversal(metrics, fam_idx, out_dir, rf_epi_ratios=None, str_epi_ratios=None):
    """Supplementary: The reversal visualization (RF/Std vs Strict/Std by family)."""
    fig, ax = plt.subplots(figsize=(APPENDIX_WIDTH * 0.55, APPENDIX_WIDTH * 0.48))

    for fam in FAMILIES:
        ix = fam_idx[fam]
        if rf_epi_ratios is not None:
            # Mean-of-ratios (consistent with ANOVA and Table 1)
            rf_ratio = float(np.mean(rf_epi_ratios[ix]))
            str_ratio = float(np.mean(str_epi_ratios[ix]))
        else:
            # Fallback: ratio-of-means
            rf_ratio = np.mean(metrics["0p00"]["psr"][ix]) / np.mean(
                metrics["0p20"]["psr"][ix]
            )
            str_ratio = np.mean(metrics["0p50"]["psr"][ix]) / np.mean(
                metrics["0p20"]["psr"][ix]
            )
        ax.scatter(
            rf_ratio,
            str_ratio,
            c=COLORS[fam],
            marker=FAMILY_MARKER[fam],
            s=120,
            zorder=3,
            label=f"{FAMILY_LABELS[fam]} (n={len(ix)})",
            edgecolors="white",
            linewidths=0.5,
        )
        ax.annotate(
            FAMILY_LABELS[fam],
            (rf_ratio, str_ratio),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8.5,
        )

    ax.axhline(1.0, color=DEEMPH, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
    ax.axvline(1.0, color=DEEMPH, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
    ax.set_xlabel("Reset-free / Standard ratio")
    ax.set_ylabel("Strict / Standard ratio")
    column_header(ax, "Opposite vulnerabilities")
    light_grid(ax, axis="both")
    tufte_axes(ax)
    ax.legend(frameon=False, loc="lower left")

    plt.tight_layout()
    fig.savefig(out_dir / "fig_reversal.pdf")
    fig.savefig(out_dir / "fig_reversal.png")
    plt.close(fig)
    print("  Saved fig_reversal")


def print_tables(stats, morphs, fam_idx):
    """Print formatted tables for paper."""
    print("\n" + "=" * 70)
    print("TABLE 1: Family x Condition (Per-Step Reward)")
    print("=" * 70)
    header = f"{'Family':6s} {'n':>3s} | {'Standard':>9s} {'Strict':>9s} {'RF':>9s} | {'RF/Std':>7s} {'Str/Std':>8s}"
    print(header)
    print("-" * len(header))
    for fam in FAMILIES:
        ft = stats["family_table"][fam]
        e = ft["Standard"]["psr"]
        s = ft["Strict"]["psr"]
        r = ft["Reset-free"]["psr"]
        rf_ratio = stats["rf_epi_ratios"][fam]["mean"]
        str_ratio = stats["str_epi_ratios"][fam]["mean"]
        print(
            f"{FAMILY_LABELS[fam]:6s} {ft['Standard']['n']:3d} | {e:9.3f} {s:9.3f} {r:9.3f} | {rf_ratio:7.3f} {str_ratio:8.3f}"
        )

    print("\n" + "=" * 70)
    print("TABLE 2: Distance by Family")
    print("=" * 70)
    for fam in FAMILIES:
        ft = stats["family_table"][fam]
        e = ft["Standard"]["distance"]
        s = ft["Strict"]["distance"]
        r = ft["Reset-free"]["distance"]
        print(
            f"{FAMILY_LABELS[fam]:6s} {ft['Standard']['n']:3d} | {e:9.1f} {s:9.1f} {r:9.1f}"
        )

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)
    a = stats["anova_rf_epi"]
    print(f"  ANOVA (RF/Std by family): F={a['F']:.3f}, p={a['p']:.6f}")
    k = stats["kruskal_rf_epi"]
    print(f"  Kruskal-Wallis:           H={k['H']:.3f}, p={k['p']:.6f}")
    a2 = stats["anova_str_epi"]
    print(f"  ANOVA (Str/Std by family): F={a2['F']:.3f}, p={a2['p']:.6f}")
    ad = stats["anova_distance_rf"]
    print(f"  ANOVA (Dist RF by family): F={ad['F']:.3f}, p={ad['p']:.6f}")
    print(f"  Cohen's d (Floor vs VT):   {stats['cohens_d_floor_vt']:.3f}")
    print(f"  Eta-squared:               {stats['eta_squared']:.3f}")
    b = stats["bootstrap_vt_floor"]
    print(f"  Bootstrap VT-Floor 95% CI: [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]")
    print(f"  P(VT > Floor ratio):       {b['p_positive']:.4f}")

    print("\n" + "=" * 70)
    print("RANKING CORRELATIONS")
    print("=" * 70)
    for name, vals in stats["ranking_correlations"].items():
        print(f"  {name:25s}: rho={vals['rho']:.3f} (p={vals['p']:.2e})")

    print("\n" + "=" * 70)
    print("PER-SEED REPLICATION")
    print("=" * 70)
    for seed, sv in stats["per_seed"].items():
        fm = sv["family_means"]
        print(
            f"  Seed {seed}: Floor={fm['floor']:.3f} VT={fm['vt']:.3f} MVT={fm['mvt']:.3f} | F={sv['F']:.2f} p={sv['p']:.4f}"
        )

    print("\n" + "=" * 70)
    print("WINNER OVERLAP (Top-10)")
    print("=" * 70)
    wo = stats["winner_overlap"]
    print(f"  Standard-RF overlap:  {wo['epi_rf_overlap']}/10")
    print(f"  Standard-Str overlap: {wo['epi_str_overlap']}/10")
    print(f"  RF-Str overlap:       {wo['rf_str_overlap']}/10")
    print(f"  All three overlap:    {wo['all_overlap']}/10")
    for label in CONDITIONS.values():
        comp = wo.get(f"top10_{label}_families", {})
        print(f"  Top-10 {label}: {comp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(PROJ_DIR / "figures"))
    parser.add_argument(
        "--stats_out", default=str(PROJ_DIR / "results" / "final_stats.json")
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_all()
    morphs = get_common_morphs(data)
    fam_idx = build_family_index(morphs)
    print(f"  {len(morphs)} common morphologies")
    for fam in FAMILIES:
        print(f"  {FAMILY_LABELS[fam]}: n={len(fam_idx[fam])}")

    print("\nComputing statistics...")
    stats, metrics, rf_epi_ratios, str_epi_ratios = compute_main_stats(
        data, morphs, fam_idx
    )

    print_tables(stats, morphs, fam_idx)

    print("\nGenerating figures...")
    fig1_family_condition_bars(metrics, fam_idx, morphs, out_dir)
    fig2_scatter(metrics, fam_idx, morphs, out_dir)
    fig3_threshold_curves(fam_idx, morphs, out_dir)
    fig4_ratio_distributions(rf_epi_ratios, str_epi_ratios, fam_idx, out_dir)
    fig5_per_seed(data, morphs, fam_idx, out_dir)
    fig_reversal(metrics, fam_idx, out_dir, rf_epi_ratios, str_epi_ratios)

    # Save stats
    stats_path = Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")
    print("Done!")


if __name__ == "__main__":
    main()
