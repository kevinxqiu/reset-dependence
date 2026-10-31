"""
Phase A.1 (corrected): Verify the recoverability signal.

After A.0 audit we know:
- fall_rate under eval_neutral is identically 0 (no low-z termination)
- fall_rate under eval_tau020 is a real, continuous signal for tau=0.00 policies
- mean_ep_len under eval_tau020 is an even better (finer-grained) signal
- RF/Epi ratio in the existing codebase uses eval_tau020 in both numerator and denominator

This script computes the right correlations between recoverability observables
and the RF/Epi PSR ratio, per-family and overall, with bootstrap CIs.

design review claimed rho ~ -0.82 between fall_rate and RF/Epi; we verify the direction
and magnitude here.

Outputs:
- results/pivot/a1_verify.json
- records/a1_verify_report.md
- figures/pivot/fig_a1_scatter.pdf (family-colored)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_JSON = REPO / "results/pivot/a1_verify.json"
OUT_MD = REPO / "records/a1_verify_report.md"
OUT_FIG = REPO / "figures/pivot/fig_a1_scatter.pdf"

TAUS = ["0p00", "0p20", "0p50"]
SEEDS = [42, 43, 44]

RNG = np.random.default_rng(seed=42)


def load_all():
    data = {}
    for tau in TAUS:
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with open(path) as f:
                data[(tau, seed)] = json.load(f)
    return data


def family_of(morph_name: str) -> str:
    for fam in ("floor", "vt", "mvt"):
        if morph_name.startswith(fam + "-"):
            return fam
    raise ValueError(f"Unknown family: {morph_name}")


def seed_avg(data, tau, morph, eval_key, metric):
    vals = [data[(tau, s)][morph][eval_key][metric] for s in SEEDS]
    return float(np.mean(vals))


def bootstrap_spearman(x, y, n_boot=10000):
    """Percentile-bootstrap CI on Spearman rho."""
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if np.isnan(rho):
            continue
        rhos.append(rho)
    rhos = np.array(rhos)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(lo), float(hi)


def spearman_with_ci(x, y):
    rho, p = stats.spearmanr(x, y)
    lo, hi = bootstrap_spearman(np.asarray(x), np.asarray(y))
    return {
        "rho": float(rho),
        "p": float(p),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(len(x)),
    }


def main() -> int:
    print("=== A.1 Verify Recoverability Signal (corrected) ===")
    data = load_all()
    morphs = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
    print(f"n morphs = {len(morphs)}")

    families = np.array([family_of(m) for m in morphs])

    # Episodic baseline: tau=0.20 policies evaluated under eval_tau020
    psr_epi = np.array(
        [seed_avg(data, "0p20", m, "eval_tau020", "per_step_reward") for m in morphs]
    )
    # Reset-free: tau=0.00 policies evaluated under eval_tau020 (matches codebase convention)
    psr_rf = np.array(
        [seed_avg(data, "0p00", m, "eval_tau020", "per_step_reward") for m in morphs]
    )
    np.array(
        [seed_avg(data, "0p00", m, "eval_neutral", "per_step_reward") for m in morphs]
    )

    # RF/Epi ratio (as used in current figures)
    rf_epi_ratio = psr_rf / np.maximum(psr_epi, 1e-3)

    # Recoverability observables from tau=0.00 runs under eval_tau020
    fall_rate_rf = np.array(
        [seed_avg(data, "0p00", m, "eval_tau020", "fall_rate") for m in morphs]
    )
    mean_ep_len_rf = np.array(
        [seed_avg(data, "0p00", m, "eval_tau020", "mean_ep_len") for m in morphs]
    )

    # Cross-check: fall_rate under eval_tau020 for tau=0.20 runs (episodic baseline)
    np.array([seed_avg(data, "0p20", m, "eval_tau020", "fall_rate") for m in morphs])

    results = {
        "n_morphs": len(morphs),
        "family_count": {
            fam: int(np.sum(families == fam)) for fam in ("floor", "vt", "mvt")
        },
        "metrics": {
            "psr_epi": {
                "mean": float(np.mean(psr_epi)),
                "std": float(np.std(psr_epi)),
                "min": float(np.min(psr_epi)),
                "max": float(np.max(psr_epi)),
            },
            "psr_rf_eval_tau020": {
                "mean": float(np.mean(psr_rf)),
                "std": float(np.std(psr_rf)),
                "min": float(np.min(psr_rf)),
                "max": float(np.max(psr_rf)),
            },
            "rf_epi_ratio": {
                "mean": float(np.mean(rf_epi_ratio)),
                "std": float(np.std(rf_epi_ratio)),
                "min": float(np.min(rf_epi_ratio)),
                "max": float(np.max(rf_epi_ratio)),
            },
            "fall_rate_rf": {
                "mean": float(np.mean(fall_rate_rf)),
                "std": float(np.std(fall_rate_rf)),
                "min": float(np.min(fall_rate_rf)),
                "max": float(np.max(fall_rate_rf)),
            },
            "mean_ep_len_rf": {
                "mean": float(np.mean(mean_ep_len_rf)),
                "std": float(np.std(mean_ep_len_rf)),
                "min": float(np.min(mean_ep_len_rf)),
                "max": float(np.max(mean_ep_len_rf)),
            },
        },
    }

    # The design review claim: fall_rate_rf vs rf_epi_ratio
    print("\n--- the design review's rho claim (fall_rate_rf vs rf_epi_ratio) ---")
    overall = spearman_with_ci(fall_rate_rf, rf_epi_ratio)
    print(
        f"overall n={overall['n']} rho={overall['rho']:.4f} "
        f"CI=[{overall['ci_low']:.4f}, {overall['ci_high']:.4f}] p={overall['p']:.2e}"
    )
    results["fall_rate_vs_rf_epi"] = {"overall": overall}
    for fam in ("floor", "vt", "mvt"):
        mask = families == fam
        r = spearman_with_ci(fall_rate_rf[mask], rf_epi_ratio[mask])
        results["fall_rate_vs_rf_epi"][fam] = r
        print(
            f"  within {fam} n={r['n']} rho={r['rho']:.4f} "
            f"CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}] p={r['p']:.2e}"
        )

    # Alternative signal: mean_ep_len_rf vs rf_epi_ratio
    print("\n--- Alternative (mean_ep_len_rf vs rf_epi_ratio) ---")
    overall_ml = spearman_with_ci(mean_ep_len_rf, rf_epi_ratio)
    print(
        f"overall n={overall_ml['n']} rho={overall_ml['rho']:.4f} "
        f"CI=[{overall_ml['ci_low']:.4f}, {overall_ml['ci_high']:.4f}] p={overall_ml['p']:.2e}"
    )
    results["mean_ep_len_vs_rf_epi"] = {"overall": overall_ml}
    for fam in ("floor", "vt", "mvt"):
        mask = families == fam
        r = spearman_with_ci(mean_ep_len_rf[mask], rf_epi_ratio[mask])
        results["mean_ep_len_vs_rf_epi"][fam] = r
        print(
            f"  within {fam} n={r['n']} rho={r['rho']:.4f} "
            f"CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}] p={r['p']:.2e}"
        )

    # Direct relation between the two observables
    print("\n--- fall_rate_rf vs mean_ep_len_rf (should be near -1) ---")
    obs_obs = spearman_with_ci(fall_rate_rf, mean_ep_len_rf)
    print(
        f"rho={obs_obs['rho']:.4f} CI=[{obs_obs['ci_low']:.4f}, {obs_obs['ci_high']:.4f}]"
    )
    results["fall_rate_vs_mean_ep_len"] = obs_obs

    # Cross-check: is fall_rate_rf also predicting the absolute psr_rf?
    print("\n--- fall_rate_rf vs absolute psr_rf (uncorrected by epi baseline) ---")
    abs_check = spearman_with_ci(fall_rate_rf, psr_rf)
    print(
        f"rho={abs_check['rho']:.4f} CI=[{abs_check['ci_low']:.4f}, {abs_check['ci_high']:.4f}]"
    )
    results["fall_rate_vs_psr_rf_absolute"] = abs_check

    # Persist
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nWrote {OUT_JSON}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"floor": "#e41a1c", "vt": "#377eb8", "mvt": "#4daf4a"}

    for fam in ("floor", "vt", "mvt"):
        mask = families == fam
        axes[0].scatter(
            fall_rate_rf[mask],
            rf_epi_ratio[mask],
            c=colors[fam],
            label=f"{fam} (n={int(mask.sum())})",
            alpha=0.7,
            s=40,
            edgecolors="black",
            linewidths=0.5,
        )
    axes[0].set_xlabel("Fall rate under eval_tau020 (tau=0.00 policies)")
    axes[0].set_ylabel("RF/Epi PSR ratio")
    o = results["fall_rate_vs_rf_epi"]["overall"]
    axes[0].set_title(f"rho={o['rho']:.3f} CI=[{o['ci_low']:.2f}, {o['ci_high']:.2f}]")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for fam in ("floor", "vt", "mvt"):
        mask = families == fam
        axes[1].scatter(
            mean_ep_len_rf[mask],
            rf_epi_ratio[mask],
            c=colors[fam],
            label=f"{fam} (n={int(mask.sum())})",
            alpha=0.7,
            s=40,
            edgecolors="black",
            linewidths=0.5,
        )
    axes[1].set_xlabel("Mean ep len under eval_tau020 (tau=0.00 policies)")
    axes[1].set_ylabel("RF/Epi PSR ratio")
    oml = results["mean_ep_len_vs_rf_epi"]["overall"]
    axes[1].set_title(
        f"rho={oml['rho']:.3f} CI=[{oml['ci_low']:.2f}, {oml['ci_high']:.2f}]"
    )
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")

    # Short-form markdown report
    lines = []
    lines.append("# A.1 Verification Report")
    lines.append("")
    lines.append("## Claim under test")
    lines.append(
        "> the design review's claim: Spearman rho between per-morph reset-free fall rate "
        "and RF/Epi per-step reward ratio is approximately -0.82."
    )
    lines.append("")
    lines.append("## Metric definitions (corrected from plan v1)")
    lines.append("")
    lines.append(
        "- `fall_rate_rf(m)`: tau=0.00 policy, eval under eval_tau020, fall_rate"
    )
    lines.append(
        "- `mean_ep_len_rf(m)`: tau=0.00 policy, eval under eval_tau020, mean_ep_len"
    )
    lines.append(
        "- `psr_epi(m)`: tau=0.20 policy, eval under eval_tau020, per_step_reward"
    )
    lines.append(
        "- `psr_rf(m)`: tau=0.00 policy, eval under eval_tau020, per_step_reward"
    )
    lines.append(
        "- `rf_epi_ratio(m) = psr_rf(m) / psr_epi(m)` (matches `extract_morph_features.py`)"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### fall_rate_rf vs rf_epi_ratio (the design review's claim)")
    lines.append("")
    lines.append("| scope | n | rho | 95% CI | p |")
    lines.append("|---|---|---|---|---|")
    for scope in ("overall", "floor", "vt", "mvt"):
        r = results["fall_rate_vs_rf_epi"][scope]
        lines.append(
            f"| {scope} | {r['n']} | {r['rho']:.4f} | "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['p']:.2e} |"
        )
    lines.append("")
    lines.append("### mean_ep_len_rf vs rf_epi_ratio (alternative observable)")
    lines.append("")
    lines.append("| scope | n | rho | 95% CI | p |")
    lines.append("|---|---|---|---|---|")
    for scope in ("overall", "floor", "vt", "mvt"):
        r = results["mean_ep_len_vs_rf_epi"][scope]
        lines.append(
            f"| {scope} | {r['n']} | {r['rho']:.4f} | "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['p']:.2e} |"
        )
    lines.append("")
    lines.append("### Self-consistency checks")
    lines.append("")
    obs_obs = results["fall_rate_vs_mean_ep_len"]
    lines.append(
        f"- fall_rate_rf vs mean_ep_len_rf: rho={obs_obs['rho']:.4f} CI=[{obs_obs['ci_low']:.3f}, {obs_obs['ci_high']:.3f}]"
    )
    abs_check = results["fall_rate_vs_psr_rf_absolute"]
    lines.append(
        f"- fall_rate_rf vs absolute psr_rf: rho={abs_check['rho']:.4f} CI=[{abs_check['ci_low']:.3f}, {abs_check['ci_high']:.3f}]"
    )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    rho_overall = results["fall_rate_vs_rf_epi"]["overall"]["rho"]
    if abs(rho_overall) >= 0.7:
        lines.append(
            "- **design review claim confirmed**: |rho| >= 0.7 on the primary comparison."
        )
    elif abs(rho_overall) >= 0.4:
        lines.append(
            "- **Partial confirmation**: fall rate explains meaningful variance but is not a complete proxy."
        )
    else:
        lines.append(
            "- **design review claim contradicted**: fall rate under eval_tau020 does not explain RF/Epi ratio. Investigate."
        )
    lines.append("")
    # Check family collapse
    within_rhos = [
        results["fall_rate_vs_rf_epi"][fam]["rho"] for fam in ("floor", "vt", "mvt")
    ]
    if all(abs(r) < 0.5 for r in within_rhos):
        lines.append(
            "- **Warning**: within-family rhos are small. Most of the overall signal is between-family."
        )
    else:
        lines.append(
            "- Within-family correlations are non-trivial: signal is not purely family-label-driven."
        )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
