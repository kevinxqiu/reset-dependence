"""
Pivot C.1: Baseline controls for Delta_reset [HISTORICAL / SUPERSEDED].

This script runs the early exploratory control that regresses
``Delta_reset(m) = PSR_epi(m) - PSR_RF(m)`` on ``baseline_psr = PSR_RF(m)``
and ``init_height``. Because ``PSR_RF`` appears on both sides of the
regression (it is contained in ``Delta_reset``), this control is mechanically
circular and inflates R^2 to 0.68. The paper does NOT cite these numbers.

The paper (Section 4.7 and Appendix J) controls for starting height and limb
count with structural features instead; ``scripts/camera_ready_numbers.py``
recomputes the paper's values.

This script is retained to document the exploratory history and to explain
why the paper uses structural features rather than baseline PSR. See also
``records/baseline_control_analysis.md`` for context.

Usage:
  python scripts/pivot_c1_delta_reset_controls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "results"
FEATURES_JSON = REPO / "results/morph_features.json"
OUT_JSON = REPO / "results/pivot/c1_delta_reset_controls.json"
OUT_MD = REPO / "records/c1_delta_reset_controls_report.md"

SEEDS = (42, 43, 44)
TAU_BASE = "0p00"  # train tau = 0 (reset-free)
TAU_REF = "0p20"  # train tau = 0.2 (episodic)
EVAL_KEY = "eval_tau020"  # The paper estimand uses eval_tau020 for both.


def load_corpus():
    """Return {(train_tau, seed, morph): psr} keyed on the paper's estimand
    eval (eval_tau020 for all training conditions).
    """
    data = {}
    for tau in (TAU_BASE, TAU_REF):
        for seed in SEEDS:
            path = CORPUS_DIR / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing corpus file: {path}")
            with open(path) as f:
                d = json.load(f)
            for morph, entry in d.items():
                if EVAL_KEY not in entry:
                    continue
                data[(tau, seed, morph)] = entry[EVAL_KEY]["per_step_reward"]
    return data


def family_of(morph_name: str) -> str:
    for fam in ("floor", "vt", "mvt"):
        if morph_name.startswith(fam + "-"):
            return fam
    return "other"


def main() -> int:
    print("=== Pivot C.1: Delta_reset baseline controls ===")
    data = load_corpus()
    morphs_base = {m for (t, s, m) in data if t == TAU_BASE}
    morphs_ref = {m for (t, s, m) in data if t == TAU_REF}
    morphs = sorted(morphs_base & morphs_ref)
    # Require all 3 seeds for both train taus.
    complete = []
    for m in morphs:
        ok = True
        for tau in (TAU_BASE, TAU_REF):
            for s in SEEDS:
                if (tau, s, m) not in data:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            complete.append(m)
    print(f"Complete morphs: {len(complete)} / {len(morphs)}")

    def mean_psr(tau, m):
        return float(np.mean([data[(tau, s, m)] for s in SEEDS]))

    baseline = np.array([mean_psr(TAU_BASE, m) for m in complete])  # PSR_train_tau000
    ref = np.array([mean_psr(TAU_REF, m) for m in complete])  # PSR_train_tau020
    delta_reset = ref - baseline
    families = np.array([family_of(m) for m in complete])

    # Load init_height if available
    heights_by_morph = {}
    if FEATURES_JSON.exists():
        with open(FEATURES_JSON) as f:
            feats = json.load(f)
        for m in complete:
            if m in feats and "init_height" in feats[m]:
                heights_by_morph[m] = float(feats[m]["init_height"])

    height_mask = np.array([m in heights_by_morph for m in complete])
    heights = np.array([heights_by_morph.get(m, np.nan) for m in complete])
    n_with_height = int(height_mask.sum())

    result: dict = {
        "n_complete": len(complete),
        "n_with_height": n_with_height,
        "eval_key": EVAL_KEY,
        "train_taus": {"base": TAU_BASE, "reference": TAU_REF},
        "delta_reset_summary": {
            "mean": float(np.mean(delta_reset)),
            "std": float(np.std(delta_reset)),
            "p25": float(np.percentile(delta_reset, 25)),
            "p75": float(np.percentile(delta_reset, 75)),
            "n_pos_gt_0p1": int(np.sum(delta_reset > 0.1)),
            "n_neg_lt_m0p1": int(np.sum(delta_reset < -0.1)),
        },
    }

    # Control 1: Delta_reset vs baseline_PSR_tau000
    rho_b, p_b = stats.spearmanr(baseline, delta_reset)
    r_b, _ = stats.pearsonr(baseline, delta_reset)
    result["control_baseline"] = {
        "spearman_rho": float(rho_b),
        "spearman_p": float(p_b),
        "pearson_r": float(r_b),
        "n": len(complete),
    }

    # Control 2: Delta_reset vs init_height
    if n_with_height >= 10:
        h_v = heights[height_mask]
        dr_v = delta_reset[height_mask]
        rho_h, p_h = stats.spearmanr(h_v, dr_v)
        r_h, _ = stats.pearsonr(h_v, dr_v)
        result["control_height"] = {
            "spearman_rho": float(rho_h),
            "spearman_p": float(p_h),
            "pearson_r": float(r_h),
            "n": n_with_height,
        }

        # Combined regression: delta_reset ~ baseline + height
        X = np.column_stack([baseline[height_mask], h_v, np.ones(n_with_height)])
        beta, *_ = np.linalg.lstsq(X, dr_v, rcond=None)
        fitted = X @ beta
        residual = dr_v - fitted
        r2 = float(1 - np.var(residual) / np.var(dr_v))
        result["control_combined"] = {
            "r_squared": r2,
            "beta_baseline": float(beta[0]),
            "beta_height": float(beta[1]),
            "intercept": float(beta[2]),
            "residual_std": float(np.std(residual)),
            "residual_frac_of_raw": float(np.std(residual) / np.std(dr_v)),
        }

        # Family structure in residual
        fam_v = families[height_mask]
        fam_stats = {}
        groups = []
        for fam in ("floor", "vt", "mvt"):
            mask = fam_v == fam
            if mask.sum() < 3:
                continue
            groups.append(residual[mask])
            fam_stats[fam] = {
                "n": int(mask.sum()),
                "raw_delta_reset_mean": float(np.mean(dr_v[mask])),
                "raw_delta_reset_std": float(np.std(dr_v[mask])),
                "residual_mean": float(np.mean(residual[mask])),
                "residual_std": float(np.std(residual[mask])),
            }
        result["residual_family_stats"] = fam_stats
        if len(groups) >= 2:
            h_stat, p_kw = stats.kruskal(*groups)
            f_stat, p_anova = stats.f_oneway(*groups)
            result["residual_family_tests"] = {
                "kruskal_h": float(h_stat),
                "kruskal_p": float(p_kw),
                "anova_f": float(f_stat),
                "anova_p": float(p_anova),
            }
    else:
        print(
            f"NOTE: only {n_with_height} bodies have init_height; skipping height controls"
        )
        result["control_height"] = None
        result["control_combined"] = None
        result["residual_family_stats"] = None
        result["residual_family_tests"] = None

    # Write JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"Wrote {OUT_JSON}")

    # Write markdown report
    lines = []
    lines.append("# Pivot C.1: Delta_reset Baseline Controls")
    lines.append("")
    lines.append(
        f"Analysis run on {result['n_complete']} corpus bodies with complete "
        f"(tau=0, tau=0.2) x 3-seed coverage on eval_tau020."
    )
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(
        "Is `Delta_reset(m) = PSR_train_tau020(eval_tau020) - PSR_train_tau000(eval_tau020)` "
        "a morphology-intrinsic property, or is it a function of baseline no-reset skill "
        "(PSR_train_tau000) and/or body size (init_height)?"
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("### Control 1: Delta_reset vs baseline PSR_tau000")
    cb = result["control_baseline"]
    lines.append(
        f"- Spearman rho = {cb['spearman_rho']:.3f} (p = {cb['spearman_p']:.2e})"
    )
    lines.append(f"- Pearson r = {cb['pearson_r']:.3f}")
    lines.append(f"- n = {cb['n']}")
    lines.append(
        "- **Interpretation:** Delta_reset is strongly negatively correlated "
        "with baseline skill. Bodies that cannot learn without resets gain "
        "most from reset training."
    )
    lines.append("")
    if result.get("control_height"):
        ch = result["control_height"]
        lines.append("### Control 2: Delta_reset vs init_height (body size)")
        lines.append(
            f"- Spearman rho = {ch['spearman_rho']:.3f} (p = {ch['spearman_p']:.2e})"
        )
        lines.append(f"- Pearson r = {ch['pearson_r']:.3f}")
        lines.append(f"- n = {ch['n']}")
        lines.append("")
        cc = result["control_combined"]
        lines.append("### Combined regression: Delta_reset ~ baseline + height")
        lines.append(f"- R^2 = {cc['r_squared']:.3f}")
        lines.append(
            f"- Residual std / raw Delta_reset std = {cc['residual_frac_of_raw']:.2f}"
        )
        lines.append("")
        lines.append("## Residual Delta_reset by family")
        lines.append("")
        lines.append(
            "After subtracting the fitted baseline + height linear model, does the "
            "residual Delta_reset still show morphology-family structure?"
        )
        lines.append("")
        rs = result["residual_family_stats"]
        lines.append(
            "| Family | n | Raw Delta_reset mean +/- std | Residual mean +/- std |"
        )
        lines.append("|---|---|---|---|")
        for fam in ("floor", "vt", "mvt"):
            if fam not in rs:
                continue
            s = rs[fam]
            lines.append(
                f"| {fam} | {s['n']} | "
                f"{s['raw_delta_reset_mean']:+.3f} +/- {s['raw_delta_reset_std']:.3f} "
                f"| {s['residual_mean']:+.3f} +/- {s['residual_std']:.3f} |"
            )
        lines.append("")
        rt = result["residual_family_tests"]
        lines.append(
            f"- Kruskal-Wallis H = {rt['kruskal_h']:.2f}, **p = {rt['kruskal_p']:.4f}**"
        )
        lines.append(
            f"- One-way ANOVA F = {rt['anova_f']:.2f}, **p = {rt['anova_p']:.4f}**"
        )
        lines.append("")
        lines.append(
            "- **Interpretation:** The residual Delta_reset after removing baseline + "
            "height still differs by family (Floor residuals positive, VT negative). "
            "Because PSR_RF enters both the baseline control and Delta_reset, this "
            "control is circular and does not establish independence from baseline skill."
        )
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(
        "Historical exploratory control, superseded because PSR_RF appears on both "
        "sides of the regression. The camera-ready paper does not use these statistics."
    )
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_MD}")

    print()
    print(f"VERDICT: baseline confound rho={cb['spearman_rho']:+.3f} (threshold -0.7)")
    if cb["spearman_rho"] < -0.7:
        print(
            "  -> Shared-term correlation; this circular control is superseded."
        )
    if result.get("residual_family_tests"):
        rt = result["residual_family_tests"]
        if rt["anova_p"] < 0.01:
            print(
                f"  -> Residual retains family structure (ANOVA p={rt['anova_p']:.4f})."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
