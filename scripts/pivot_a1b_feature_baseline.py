"""
Phase A.1b: Feature baseline regression.

design review claimed that a leave-one-out ridge on the 8 existing structural features
predicts corrected R_obs at R^2 ~= 0.39 without family labels and ~= 0.42 with
family labels. Verify this, then establish it as the baseline that any
geometric recoverability proxy in A.2 must beat.

Target: R_obs = 1 - fall_rate_rf(m), a direct observable measurement of
body-level recoverability computed in A.1 from the existing 900-run corpus.

Features: n_bodies, n_joints, n_motors, n_geoms, init_height, tree_depth,
total_volume, n_first_level_limbs. Optionally family labels (one-hot).

Outputs:
- results/pivot/a1b_feature_baseline.json
- records/a1b_feature_baseline_report.md
- figures/pivot/fig_a1b_feature_importance.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_JSON = REPO / "results/pivot/a1b_feature_baseline.json"
OUT_MD = REPO / "records/a1b_feature_baseline_report.md"
OUT_FIG = REPO / "figures/pivot/fig_a1b_feature_importance.pdf"

FEATURES = [
    "n_bodies",
    "n_joints",
    "n_motors",
    "n_geoms",
    "init_height",
    "tree_depth",
    "total_volume",
    "n_first_level_limbs",
]

TAUS = ["0p00", "0p20", "0p50"]
SEEDS = [42, 43, 44]


def family_of(morph_name: str) -> str:
    for fam in ("floor", "vt", "mvt"):
        if morph_name.startswith(fam + "-"):
            return fam
    raise ValueError(f"Unknown family: {morph_name}")


def load_all():
    data = {}
    for tau in TAUS:
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with open(path) as f:
                data[(tau, seed)] = json.load(f)
    return data


def loo_ridge(X, y, alpha=1.0):
    """Leave-one-out ridge. Returns out-of-fold predictions and per-fold scores."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model = Ridge(alpha=alpha)
        model.fit(X_tr_s, y_tr)
        preds[test_idx] = model.predict(X_te_s)
    return preds


def main() -> int:
    print("=== A.1b Feature Baseline ===")

    # Load recoverability target
    data = load_all()
    morph_features_path = RESULTS / "morph_features.json"
    with open(morph_features_path) as f:
        morph_feats = json.load(f)

    morphs = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
    # Ensure all morphs have features
    missing = [m for m in morphs if m not in morph_feats]
    if missing:
        raise RuntimeError(f"Morphs missing from morph_features.json: {missing[:5]}...")

    families = np.array([family_of(m) for m in morphs])

    # Target 1: R_obs = 1 - seed-averaged fall_rate_rf
    fall_rate_rf = np.array(
        [
            float(
                np.mean(
                    [data[("0p00", s)][m]["eval_tau020"]["fall_rate"] for s in SEEDS]
                )
            )
            for m in morphs
        ]
    )
    R_obs = 1.0 - fall_rate_rf

    # Target 2: rf_epi_ratio (for comparison to morph_features.json values)
    psr_epi = np.array(
        [
            float(
                np.mean(
                    [
                        data[("0p20", s)][m]["eval_tau020"]["per_step_reward"]
                        for s in SEEDS
                    ]
                )
            )
            for m in morphs
        ]
    )
    psr_rf = np.array(
        [
            float(
                np.mean(
                    [
                        data[("0p00", s)][m]["eval_tau020"]["per_step_reward"]
                        for s in SEEDS
                    ]
                )
            )
            for m in morphs
        ]
    )
    rf_epi_ratio = psr_rf / np.maximum(psr_epi, 1e-3)

    # Feature matrix
    X = np.array(
        [[morph_feats[m][f] for f in FEATURES] for m in morphs],
        dtype=float,
    )

    # Family one-hot
    fam_onehot = np.stack(
        [(families == f).astype(float) for f in ("floor", "vt", "mvt")], axis=1
    )
    X_with_family = np.concatenate([X, fam_onehot], axis=1)

    print(f"n morphs = {len(morphs)}, n features = {X.shape[1]}")
    print(
        f"R_obs range: [{R_obs.min():.3f}, {R_obs.max():.3f}], mean={R_obs.mean():.3f}"
    )
    print(f"rf_epi_ratio range: [{rf_epi_ratio.min():.3f}, {rf_epi_ratio.max():.3f}]")

    results = {
        "n_morphs": len(morphs),
        "feature_names": FEATURES,
        "targets": {},
    }

    # Run LOO ridge at multiple alphas for each target
    for target_name, y in [("R_obs", R_obs), ("rf_epi_ratio", rf_epi_ratio)]:
        print(f"\n--- target: {target_name} ---")
        results["targets"][target_name] = {}
        for design_name, X_design in [
            ("features_only", X),
            ("features_plus_family", X_with_family),
        ]:
            # Grid search alpha via LOO (cheap)
            best_r2 = -np.inf
            best_alpha = None
            best_preds = None
            for alpha in [0.01, 0.1, 1.0, 3.0, 10.0, 30.0]:
                preds = loo_ridge(X_design, y, alpha=alpha)
                r2 = r2_score(y, preds)
                if r2 > best_r2:
                    best_r2 = r2
                    best_alpha = alpha
                    best_preds = preds

            rho, p = stats.spearmanr(best_preds, y)
            mae = float(np.mean(np.abs(best_preds - y)))
            results["targets"][target_name][design_name] = {
                "best_alpha": best_alpha,
                "loo_r2": float(best_r2),
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "mae": mae,
            }
            print(
                f"  {design_name}: alpha={best_alpha:.2f} "
                f"LOO R^2={best_r2:.4f} rho={rho:.4f} p={p:.2e} MAE={mae:.4f}"
            )

        # Also fit on full data for coefficient inspection (features only, alpha=1)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        full_model = Ridge(alpha=1.0).fit(X_s, y)
        coef_with_names = dict(zip(FEATURES, [float(c) for c in full_model.coef_]))
        results["targets"][target_name]["full_fit_coefs_standardized"] = coef_with_names

    # Family-only baseline (sanity check)
    print("\n--- family-only baseline ---")
    for target_name, y in [("R_obs", R_obs), ("rf_epi_ratio", rf_epi_ratio)]:
        preds = loo_ridge(fam_onehot, y, alpha=0.01)
        r2 = r2_score(y, preds)
        rho, p = stats.spearmanr(preds, y)
        results["targets"][target_name]["family_only"] = {
            "loo_r2": float(r2),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
        }
        print(f"  {target_name} family-only: LOO R^2={r2:.4f} rho={rho:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nWrote {OUT_JSON}")

    # Feature importance plot (for R_obs, features_plus_family fit on full data)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_with_family)
    model = Ridge(alpha=1.0).fit(X_s, R_obs)
    feat_names = FEATURES + ["fam_floor", "fam_vt", "fam_mvt"]
    coefs = model.coef_

    fig, ax = plt.subplots(figsize=(9, 5))
    order = np.argsort(np.abs(coefs))[::-1]
    ax.barh(
        [feat_names[i] for i in order][::-1],
        [coefs[i] for i in order][::-1],
        color=["steelblue" if abs(coefs[i]) > 0.02 else "lightgray" for i in order][
            ::-1
        ],
    )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Ridge coefficient (standardized features)")
    ax.set_title("Feature importance for predicting R_obs = 1 - fall_rate_rf")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT_FIG}")

    # Markdown report
    lines = []
    lines.append("# A.1b Feature Baseline Report")
    lines.append("")
    lines.append("## Targets")
    lines.append(
        "- `R_obs = 1 - fall_rate_rf(m)`: direct observable of body-level recoverability"
    )
    lines.append("- `rf_epi_ratio(m)`: the paper's existing ratio")
    lines.append("")
    lines.append("## Leave-one-out ridge results")
    lines.append("")
    lines.append("| target | design | best alpha | LOO R^2 | Spearman rho | MAE |")
    lines.append("|---|---|---|---|---|---|")
    for tname in ("R_obs", "rf_epi_ratio"):
        for dname in ("features_only", "features_plus_family", "family_only"):
            r = results["targets"][tname].get(dname, {})
            if not r:
                continue
            lines.append(
                f"| {tname} | {dname} | "
                f"{r.get('best_alpha', 'n/a')} | "
                f"{r['loo_r2']:.4f} | "
                f"{r['spearman_rho']:.4f} | "
                f"{r.get('mae', 0.0):.4f} |"
            )
    lines.append("")
    lines.append("## the design review's claim")
    lines.append(
        "> design review: leave-one-out ridge on 8 features gives R^2 ~ 0.39 without family, "
        "~ 0.42 with family."
    )
    robs_feat = results["targets"]["R_obs"]["features_only"]["loo_r2"]
    robs_fam = results["targets"]["R_obs"]["features_plus_family"]["loo_r2"]
    lines.append("")
    lines.append(f"- R_obs features_only LOO R^2 = **{robs_feat:.4f}**")
    lines.append(f"- R_obs features_plus_family LOO R^2 = **{robs_fam:.4f}**")
    lines.append("")
    lines.append("## Interpretation")
    if robs_feat > 0.35:
        lines.append(
            f"- Baseline established. Any geometric proxy in A.2 must beat LOO R^2 = {robs_feat:.3f} "
            "on features alone (or the ~ +0.03 lift from adding family information) to be worth pursuing."
        )
    else:
        lines.append(
            f"- Baseline is weaker than design review predicted. LOO R^2 = {robs_feat:.3f}. Still a usable anchor for A.2."
        )
    lines.append("")
    lines.append(
        "## Feature importance (R_obs target, standardized ridge coefficients)"
    )
    lines.append("")
    coefs_dict = results["targets"]["R_obs"]["full_fit_coefs_standardized"]
    sorted_coefs = sorted(coefs_dict.items(), key=lambda kv: -abs(kv[1]))
    lines.append("| feature | standardized coef |")
    lines.append("|---|---|")
    for f, c in sorted_coefs:
        lines.append(f"| {f} | {c:+.4f} |")
    lines.append("")
    lines.append(
        "(See `figures/pivot/fig_a1b_feature_importance.pdf` for the full chart including family dummies.)"
    )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
