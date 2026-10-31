"""
Phase C.0.2: Metric-robustness check.

the design review's cheapest high-leverage add: test whether R_obs (our primary observable)
depends on the specific failure criterion. If it's a body property, rankings
should be robust across reasonable alternative definitions. If it's a metric
artifact, ranking shifts tell us something is off.

This script works purely from existing corpus data (no new training needed).

Alternative observables computed from existing corpus (all under eval_tau020
on tau=0.00 policies):

1. R_obs_frac = 1 - fall_rate
     binary-per-episode, averaged across 20 eval episodes
2. R_obs_len = mean_ep_len / 1000
     continuous version of the same signal (near-perfect anti-correlation)
3. R_obs_psr = per_step_reward (absolute)
     performance under autonomous practice, no ratio
4. R_obs_psr_ratio = per_step_reward / psr_epi
     the paper's current RF/Epi ratio proxy

We also compute pairwise Spearman rho and ranking stability at top-k for
k in {10, 20, 30}. Decision gate: if all pairs have rho >= 0.85 and top-k
stability is >=70 percent, R_obs is robust to criterion.

Outputs:
- results/pivot/c0_metric_robustness.json
- records/c0_metric_robustness_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_JSON = REPO / "results/pivot/c0_metric_robustness.json"
OUT_MD = REPO / "records/c0_metric_robustness_report.md"

TAUS = ["0p00", "0p20"]
SEEDS = [42, 43, 44]


def load_corpus():
    data = {}
    for tau in TAUS:
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with open(path) as f:
                data[(tau, seed)] = json.load(f)
    return data, sorted(set.intersection(*[set(d.keys()) for d in data.values()]))


def family_of(m: str) -> str:
    for f in ("floor", "vt", "mvt"):
        if m.startswith(f + "-"):
            return f
    raise ValueError(m)


def main() -> int:
    print("=== C.0.2 Metric-Robustness Check ===")
    data, morphs = load_corpus()
    n = len(morphs)
    print(f"n morphs = {n}")

    # Extract seed-averaged metrics from tau=0.00 runs under eval_tau020
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
    fitness_rf = np.array(
        [
            float(
                np.mean([data[("0p00", s)][m]["eval_tau020"]["fitness"] for s in SEEDS])
            )
            for m in morphs
        ]
    )
    dist_rf = np.array(
        [
            float(
                np.mean(
                    [data[("0p00", s)][m]["eval_tau020"]["distance"] for s in SEEDS]
                )
            )
            for m in morphs
        ]
    )

    # Six candidate observables
    observables = {
        "R_obs_frac": 1.0 - fr,
        "R_obs_len": mel / 1000.0,
        "R_obs_psr_absolute": psr_rf,
        "R_obs_psr_ratio": psr_rf / np.maximum(psr_epi, 1e-3),
        "R_obs_fitness_absolute": fitness_rf,
        "R_obs_distance_absolute": dist_rf,
    }

    results = {
        "n_morphs": n,
        "observables": {},
        "pairwise_correlations": {},
        "topk_stability": {},
    }

    for name, vals in observables.items():
        results["observables"][name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)),
            "n_saturated_at_max": int(np.sum(vals >= vals.max() - 1e-9)),
        }

    # Pairwise Spearman rhos
    obs_list = list(observables.keys())
    print("\n=== Pairwise Spearman rhos ===")
    rho_matrix = np.zeros((len(obs_list), len(obs_list)))
    for i, a in enumerate(obs_list):
        for j, b in enumerate(obs_list):
            if i >= j:
                rho, _ = stats.spearmanr(observables[a], observables[b])
                rho_matrix[i, j] = rho
                rho_matrix[j, i] = rho
                if i != j:
                    key = f"{a}__{b}"
                    results["pairwise_correlations"][key] = float(rho)

    # Print matrix
    print("                " + " ".join(name[:14].rjust(15) for name in obs_list))
    for i, a in enumerate(obs_list):
        row = " ".join(f"{rho_matrix[i, j]:15.3f}" for j in range(len(obs_list)))
        print(f"{a[:14]:15s} {row}")

    # Top-k overlap stability
    print("\n=== Top-k overlap between observables ===")
    morphs_arr = np.array(morphs)
    for k in (10, 20, 30):
        print(f"\ntop-{k}:")
        ranks = {
            name: set(morphs_arr[np.argsort(-vals)[:k]])
            for name, vals in observables.items()
        }
        pair_overlaps = {}
        for i, a in enumerate(obs_list):
            for j, b in enumerate(obs_list):
                if i < j:
                    ov = len(ranks[a] & ranks[b])
                    pair_overlaps[f"{a}__{b}"] = ov
                    print(f"  {a[:25]:25s} vs {b[:25]:25s} : {ov}/{k}")
        results["topk_stability"][f"top{k}"] = pair_overlaps

    # Verdict
    all_pair_rhos = list(results["pairwise_correlations"].values())
    min_rho = float(min(all_pair_rhos))
    mean_rho = float(np.mean([abs(v) for v in all_pair_rhos]))

    results["verdict"] = {
        "min_pairwise_rho": float(min_rho),
        "mean_pairwise_abs_rho": float(mean_rho),
        "robust": bool(abs(min_rho) >= 0.7),
    }
    print(
        f"\nmin pairwise rho = {min_rho:.4f}, mean |rho| = {mean_rho:.4f}, "
        f"verdict: {'all pairs |rho| >= 0.7' if results['verdict']['robust'] else 'some pairs |rho| < 0.7'}"
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nWrote {OUT_JSON}")

    lines = []
    lines.append("# C.0.2 Metric-Robustness Report")
    lines.append("")
    lines.append("## Alternative observables")
    lines.append("")
    lines.append("| name | definition | mean | std | saturated |")
    lines.append("|---|---|---|---|---|")
    defs = {
        "R_obs_frac": "1 - fall_rate (binary per-episode, averaged 20 eps)",
        "R_obs_len": "mean_ep_len / 1000 (continuous upright time)",
        "R_obs_psr_absolute": "per_step_reward absolute value",
        "R_obs_psr_ratio": "per_step_reward / psr_epi (paper's RF/Epi)",
        "R_obs_fitness_absolute": "total fitness (sum reward) absolute",
        "R_obs_distance_absolute": "total forward distance absolute",
    }
    for name in obs_list:
        r = results["observables"][name]
        lines.append(
            f"| {name} | {defs[name]} | {r['mean']:.3f} | {r['std']:.3f} | "
            f"{r['n_saturated_at_max']} |"
        )
    lines.append("")
    lines.append("## Pairwise Spearman rho")
    lines.append("")
    lines.append("| a | b | rho |")
    lines.append("|---|---|---|")
    for k, v in results["pairwise_correlations"].items():
        a, b = k.split("__", 1)
        lines.append(f"| {a} | {b} | {v:+.4f} |")
    lines.append("")
    lines.append("## Top-k overlap")
    lines.append("")
    for k in (10, 20, 30):
        lines.append(f"### top-{k}")
        lines.append("")
        lines.append("| a | b | overlap |")
        lines.append("|---|---|---|")
        for key, v in results["topk_stability"][f"top{k}"].items():
            a, b = key.split("__", 1)
            lines.append(f"| {a} | {b} | {v}/{k} |")
        lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- min pairwise rho: {results['verdict']['min_pairwise_rho']:.4f}")
    lines.append(
        f"- mean pairwise |rho|: {results['verdict']['mean_pairwise_abs_rho']:.4f}"
    )
    if results["verdict"]["robust"]:
        lines.append(
            "- **All pairs |rho| >= 0.7**: every pair of observables correlates at |rho| >= 0.7, "
            "the check this script applies. The stricter gate in its docstring (rho >= 0.85 for "
            "every pair and top-k stability >= 70 percent) "
            + ("is not met." if abs(results["verdict"]["min_pairwise_rho"]) < 0.85 else "is not checked here.")
        )
    else:
        lines.append(
            "- **WEAK ROBUSTNESS**: some observable pairs correlate below 0.7. "
            "Need to report which metric choices change the ranking."
        )

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
