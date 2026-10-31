"""
Phase C.0.1: Saturation-aware reanalysis of A.1b.

R_obs = 1 - fall_rate saturates at 1.0 for 38/100 morphs (never-fall bodies).
This script adds:
  1. A saturation-aware target that preserves ceiling ordering via psr_epi tiebreaker.
  2. Feature baseline restricted to non-saturated morphs (n=62).
  3. Per-family reanalysis.
  4. Robustness check on the R_obs result using the three alternative primary
     observables: R_obs, mean_ep_len/1000, and (R_obs, psr_epi) composite.

Outputs:
  results/pivot/c0_saturation.json
  records/c0_saturation_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_JSON = REPO / "results/pivot/c0_saturation.json"
OUT_MD = REPO / "records/c0_saturation_report.md"

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


def family_of(m: str) -> str:
    for f in ("floor", "vt", "mvt"):
        if m.startswith(f + "-"):
            return f
    raise ValueError(m)


def load_corpus():
    data = {}
    for tau in TAUS:
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with open(path) as f:
                data[(tau, seed)] = json.load(f)
    return data, sorted(set.intersection(*[set(d.keys()) for d in data.values()]))


def loo_ridge_r2(X, y, alphas=(0.1, 1, 3, 10, 30)):
    best = -np.inf
    best_alpha = None
    for alpha in alphas:
        loo = LeaveOneOut()
        preds = np.zeros_like(y, dtype=float)
        for tr, te in loo.split(X):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            m = Ridge(alpha=alpha).fit(Xtr, y[tr])
            preds[te] = m.predict(Xte)
        r = r2_score(y, preds)
        if r > best:
            best = r
            best_alpha = alpha
    return best, best_alpha


def main() -> int:
    print("=== C.0.1 Saturation-Aware Analysis ===")
    data, morphs = load_corpus()
    n = len(morphs)
    print(f"n morphs = {n}")

    # Compute observables from existing corpus
    fr = np.array(
        [
            float(
                np.mean(
                    [data[("0p00", s)][m]["eval_tau020"]["fall_rate"] for s in SEEDS]
                )
            )
            for m in morphs
        ]
    )
    mel = np.array(
        [
            float(
                np.mean(
                    [data[("0p00", s)][m]["eval_tau020"]["mean_ep_len"] for s in SEEDS]
                )
            )
            for m in morphs
        ]
    )
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
    np.array(
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

    R_obs = 1.0 - fr
    R_mel = mel / 1000.0
    # Composite: use psr_epi as tiebreaker for saturated R_obs
    saturated_mask = R_obs >= 0.999
    # Compose composite score that is equal to R_obs for non-saturated morphs
    # and 1 + alpha*psr_epi_norm for saturated morphs (preserving 1.0 ceiling
    # but ordering saturated by psr_epi).
    psr_norm = (psr_epi - psr_epi.min()) / max(psr_epi.max() - psr_epi.min(), 1e-6)
    R_composite = np.where(saturated_mask, 1.0 + 0.01 * psr_norm, R_obs)
    # Rank version for regression: always in [0, 1]
    ranks = stats.rankdata(R_composite)
    R_composite_norm = (ranks - 1) / (n - 1)

    print(f"saturated at R_obs=1.0: {int(saturated_mask.sum())} / {n}")
    print(f"R_mel saturated at 1.0: {int((R_mel >= 0.999).sum())} / {n}")

    # Load features
    with open(RESULTS / "morph_features.json") as f:
        mf = json.load(f)
    X = np.array([[mf[m][f] for f in FEATURES] for m in morphs], dtype=float)
    families = np.array([family_of(m) for m in morphs])

    results = {
        "n_morphs": n,
        "n_saturated": int(saturated_mask.sum()),
        "targets": {},
    }

    # Evaluate each target
    target_defs = [
        ("R_obs (1 - fall_rate)", R_obs),
        ("R_mel (mean_ep_len / 1000)", R_mel),
        ("R_composite (R_obs with psr_epi tiebreaker)", R_composite),
        ("R_composite_rank (normalized rank)", R_composite_norm),
    ]

    for name, y in target_defs:
        r2, alpha = loo_ridge_r2(X, y)
        rho_feat, _ = stats.spearmanr(y, X[:, FEATURES.index("init_height")])
        results["targets"][name] = {
            "loo_r2_full_100": float(r2),
            "best_alpha": float(alpha),
            "spearman_with_init_height": float(rho_feat),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
        }
        print(f"  {name}: full n=100 LOO R^2={r2:.4f} (alpha={alpha})")

    # Restrict to non-saturated subset (n=62)
    print("\n=== Non-saturated only (n=62) ===")
    non_sat = ~saturated_mask
    for name, y in target_defs[:2]:
        r2, alpha = loo_ridge_r2(X[non_sat], y[non_sat])
        results["targets"][name]["loo_r2_non_saturated_n62"] = float(r2)
        results["targets"][name]["n_non_saturated"] = int(non_sat.sum())
        print(f"  {name}: n=62 LOO R^2={r2:.4f} (alpha={alpha})")

    # Per-family breakdown for R_obs
    print("\n=== Per-family R_obs ridge ===")
    for fam in ("floor", "vt", "mvt"):
        mask = families == fam
        if mask.sum() < 10:
            continue
        r2, alpha = loo_ridge_r2(X[mask], R_obs[mask])
        n_sat = int(saturated_mask[mask].sum())
        results[f"R_obs_family_{fam}"] = {
            "n": int(mask.sum()),
            "n_saturated": n_sat,
            "loo_r2": float(r2),
        }
        print(f"  {fam}: n={int(mask.sum())} saturated={n_sat} LOO R^2={r2:.4f}")

    # Correlations between the target definitions
    print("\n=== Agreement between target definitions ===")
    pairs = [
        ("R_obs", "R_mel", R_obs, R_mel),
        ("R_obs", "R_composite_rank", R_obs, R_composite_norm),
        ("R_mel", "R_composite_rank", R_mel, R_composite_norm),
    ]
    for n1, n2, a, b in pairs:
        rho, _ = stats.spearmanr(a, b)
        pearson = np.corrcoef(a, b)[0, 1]
        print(f"  {n1} vs {n2}: Spearman={rho:.4f} Pearson={pearson:.4f}")
        results.setdefault("target_agreement", {})[f"{n1}_vs_{n2}"] = {
            "spearman": float(rho),
            "pearson": float(pearson),
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nWrote {OUT_JSON}")

    # Report
    lines = []
    lines.append("# C.0.1 Saturation-Aware Reanalysis Report")
    lines.append("")
    lines.append(f"- n morphs: {n}")
    lines.append(f"- R_obs = 1 saturated: **{int(saturated_mask.sum())} / {n}**")
    lines.append(f"- R_mel = 1 saturated: {int((R_mel >= 0.999).sum())} / {n}")
    lines.append("")
    lines.append("## Alternative targets and feature predictability")
    lines.append("")
    lines.append("| target | full n=100 LOO R^2 | n=62 non-sat LOO R^2 |")
    lines.append("|---|---|---|")
    for name, _ in target_defs[:2]:
        r = results["targets"][name]
        full = r["loo_r2_full_100"]
        nons = r.get("loo_r2_non_saturated_n62", "n/a")
        nons_s = f"{nons:.4f}" if isinstance(nons, float) else nons
        lines.append(f"| {name} | {full:.4f} | {nons_s} |")
    lines.append("")
    lines.append("## Composite (saturation-aware) targets")
    lines.append("")
    lines.append("| target | full n=100 LOO R^2 |")
    lines.append("|---|---|")
    for name, _ in target_defs[2:]:
        r = results["targets"][name]
        lines.append(f"| {name} | {r['loo_r2_full_100']:.4f} |")
    lines.append("")
    lines.append("## Per-family R_obs")
    lines.append("")
    lines.append("| family | n | saturated | LOO R^2 |")
    lines.append("|---|---|---|---|")
    for fam in ("floor", "vt", "mvt"):
        key = f"R_obs_family_{fam}"
        if key in results:
            r = results[key]
            lines.append(
                f"| {fam} | {r['n']} | {r['n_saturated']} | {r['loo_r2']:.4f} |"
            )
    lines.append("")
    lines.append("## Target agreement")
    lines.append("")
    lines.append("| pair | Spearman | Pearson |")
    lines.append("|---|---|---|")
    for k, v in results.get("target_agreement", {}).items():
        lines.append(
            f"| {k.replace('_vs_', ' vs ')} | {v['spearman']:.4f} | {v['pearson']:.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    robs_full = results["targets"]["R_obs (1 - fall_rate)"]["loo_r2_full_100"]
    robs_nons = results["targets"]["R_obs (1 - fall_rate)"].get(
        "loo_r2_non_saturated_n62", 0
    )
    lines.append(
        f"- R_obs is predictable at R^2 = {robs_full:.3f} on the full 100 bodies "
        f"and {robs_nons:.3f} on the non-saturated 62 subset."
    )
    if abs(robs_full - robs_nons) < 0.05:
        lines.append(
            "- Saturation is not driving the baseline: predictability holds on the non-saturated subset."
        )
    else:
        lines.append(
            "- Non-trivial gap between full and non-saturated predictability. Ceiling-tied morphs are contributing to feature-to-R_obs signal."
        )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
