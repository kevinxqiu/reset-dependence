"""
Phase A.0: Metric-semantics audit.

Answers repo-grounded questions that the design review flagged in the plan review:

1. What are the actual values of fall_rate and mean_ep_len for each eval key in the
   900-run corpus? design review claimed fall_rate under eval_neutral is always 0.
2. Which eval key is usable as a recoverability observable?
3. What is the distribution of mean_ep_len under eval_tau020 for tau=0.00 runs
   (the "how long did the reset-free policy stay upright" signal)?
4. Are there any policy checkpoints saved alongside the metrics?
5. What were the actual training hyperparameters used for the 900 runs (NUM_ENVS,
   UNROLL_LENGTH, timesteps, backend)?
6. Do all 100 morphs appear in all 9 result files?
7. What's the family distribution?

Fail-loud on anything unexpected. Write outputs to
results/pivot/a0_audit.json and records/a0_audit_report.md.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_JSON = REPO / "results/pivot/a0_audit.json"
OUT_MD = REPO / "records/a0_audit_report.md"

TAUS = ["0p00", "0p20", "0p50"]
SEEDS = [42, 43, 44]
EVAL_KEYS = ["eval_tau020", "eval_tau050", "eval_neutral"]
METRICS = [
    "fitness",
    "per_step_reward",
    "distance",
    "mean_ep_len",
    "fall_rate",
]


def load_all():
    data = {}
    for tau in TAUS:
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing corpus file: {path}")
            with open(path) as f:
                data[(tau, seed)] = json.load(f)
    return data


def family_of(morph_name: str) -> str:
    for fam in ("floor", "vt", "mvt"):
        if morph_name.startswith(fam + "-"):
            return fam
    raise ValueError(f"Unknown family for morph: {morph_name}")


def main() -> int:
    print("=== A.0 Metric-Semantics Audit ===")
    data = load_all()
    print(f"Loaded {len(data)} result files")

    # Morph coverage
    morph_sets = [set(d.keys()) for d in data.values()]
    all_morphs = set.intersection(*morph_sets)
    union_morphs = set.union(*morph_sets)
    if len(all_morphs) != len(union_morphs):
        missing = union_morphs - all_morphs
        print(
            f"WARNING: morph coverage inconsistent. {len(all_morphs)} intersection, "
            f"{len(union_morphs)} union, {len(missing)} morphs missing from at least one file"
        )
    else:
        print(f"Morph coverage: {len(all_morphs)} morphs present in all 9 files")

    sorted_morphs = sorted(all_morphs)
    family_count = Counter(family_of(m) for m in sorted_morphs)
    print(f"Family distribution: {dict(family_count)}")

    # Provenance: inspect one record's metadata
    first_record = data[("0p00", 42)][sorted_morphs[0]]
    provenance = {
        "algorithm": first_record.get("algorithm"),
        "backend": first_record.get("backend"),
        "timesteps": first_record.get("timesteps"),
        "train_time_s_approx": first_record.get("train_time_s"),
        "threshold_field": first_record.get("threshold"),
    }
    print(f"Provenance (first record): {provenance}")

    # Per-tau, per-eval-key distributional audit
    audit = {}
    for tau in TAUS:
        audit[tau] = {}
        for ek in EVAL_KEYS:
            audit[tau][ek] = {}
            for metric in METRICS:
                vals = []
                for seed in SEEDS:
                    corpus = data[(tau, seed)]
                    for m in sorted_morphs:
                        v = corpus[m][ek][metric]
                        vals.append(v)
                arr = np.array(vals, dtype=float)
                audit[tau][ek][metric] = {
                    "n": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "p25": float(np.percentile(arr, 25)),
                    "median": float(np.median(arr)),
                    "p75": float(np.percentile(arr, 75)),
                    "max": float(np.max(arr)),
                    "n_zero": int(np.sum(arr == 0.0)),
                    "n_one": int(np.sum(arr == 1.0)),
                }

    # Key headline checks
    headline = {}

    # design review claim 1: fall_rate under eval_neutral is ~always 0
    fr_neutral = audit["0p00"]["eval_neutral"]["fall_rate"]
    headline["fall_rate_eval_neutral_tau000"] = {
        "claim": "design review: fall_rate under eval_neutral is ~0 for all tau=0.00 runs",
        "mean": fr_neutral["mean"],
        "max": fr_neutral["max"],
        "n_zero": fr_neutral["n_zero"],
        "n_total": fr_neutral["n"],
        "verdict": "confirmed"
        if fr_neutral["mean"] < 0.01
        else ("partial" if fr_neutral["mean"] < 0.1 else "contradicted"),
    }

    # design review claim 2: eval_tau020 is the usable fall observable
    fr_tau020 = audit["0p00"]["eval_tau020"]["fall_rate"]
    headline["fall_rate_eval_tau020_tau000"] = {
        "mean": fr_tau020["mean"],
        "median": fr_tau020["median"],
        "min": fr_tau020["min"],
        "max": fr_tau020["max"],
        "n_zero": fr_tau020["n_zero"],
        "n_one": fr_tau020["n_one"],
        "n_total": fr_tau020["n"],
        "useful_as_signal": bool(
            fr_tau020["std"] > 0.05
            and fr_tau020["n_zero"] + fr_tau020["n_one"] < fr_tau020["n"]
        ),
    }

    # mean_ep_len under eval_tau020 as alternative
    mel_tau020 = audit["0p00"]["eval_tau020"]["mean_ep_len"]
    headline["mean_ep_len_eval_tau020_tau000"] = {
        "note": "how long the RF policy stays upright under episodic eval",
        "mean": mel_tau020["mean"],
        "median": mel_tau020["median"],
        "min": mel_tau020["min"],
        "max": mel_tau020["max"],
    }

    # Under tau=0.20 (episodic training), what's the mean_ep_len and fall_rate baseline
    fr_epi = audit["0p20"]["eval_tau020"]["fall_rate"]
    mel_epi = audit["0p20"]["eval_tau020"]["mean_ep_len"]
    headline["baseline_epi_eval_tau020"] = {
        "fall_rate_mean": fr_epi["mean"],
        "mean_ep_len_mean": mel_epi["mean"],
        "note": "tau=0.20 policies evaluated under tau=0.20: the 'trained to stay up' baseline",
    }

    # Checkpoint audit: search for saved policy weights
    checkpoints = []
    for ext in ("*.ckpt", "*.pkl", "params_*", "*.msgpack", "*.safetensors"):
        checkpoints.extend(str(p) for p in REPO.rglob(ext))
    # Filter out python pickle cache noise
    checkpoints = [
        c
        for c in checkpoints
        if "/__pycache__/" not in c and "/.venv/" not in c and "/venv/" not in c
    ]
    headline["checkpoint_files_found"] = {
        "count": len(checkpoints),
        "paths": checkpoints[:20],
        "r_task_feasible_from_existing_corpus": len(checkpoints) >= 10,
    }

    # Persist
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "provenance": provenance,
                "family_count": dict(family_count),
                "n_morphs_intersection": len(all_morphs),
                "headline": headline,
                "audit": audit,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote {OUT_JSON}")

    # Human-readable report
    lines = []
    lines.append("# A.0 Metric-Semantics Audit Report")
    lines.append("")
    lines.append("Audit date: 2026-04-11")
    lines.append(
        f"Corpus: {len(all_morphs)} morphs x 3 protocols x 3 seeds = {9 * len(all_morphs)} records"
    )
    lines.append(f"Family distribution: {dict(family_count)}")
    lines.append("")
    lines.append("## Provenance (from first record)")
    lines.append("")
    for k, v in provenance.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Headline findings")
    lines.append("")
    lines.append(
        "### 1. design review claim: fall_rate under eval_neutral is ~always 0 for tau=0.00 runs"
    )
    h1 = headline["fall_rate_eval_neutral_tau000"]
    lines.append(f"- Verdict: **{h1['verdict']}**")
    lines.append(f"- Mean: {h1['mean']:.6f}, Max: {h1['max']:.6f}")
    lines.append(f"- Zeros: {h1['n_zero']} of {h1['n_total']}")
    lines.append("")
    lines.append("### 2. Is eval_tau020 fall_rate usable as a recoverability signal?")
    h2 = headline["fall_rate_eval_tau020_tau000"]
    lines.append(f"- Mean: {h2['mean']:.4f}, median: {h2['median']:.4f}")
    lines.append(f"- Range: [{h2['min']:.4f}, {h2['max']:.4f}]")
    lines.append(
        f"- Zeros: {h2['n_zero']}, Ones: {h2['n_one']}, Total: {h2['n_total']}"
    )
    lines.append(f"- Useful as continuous signal: **{h2['useful_as_signal']}**")
    lines.append("")
    lines.append("### 3. Alternative: mean_ep_len under eval_tau020 for tau=0.00")
    h3 = headline["mean_ep_len_eval_tau020_tau000"]
    lines.append(f"- Mean: {h3['mean']:.2f}, median: {h3['median']:.2f}")
    lines.append(f"- Range: [{h3['min']:.2f}, {h3['max']:.2f}]")
    lines.append("- (max possible ep_len = 1000)")
    lines.append("")
    lines.append("### 4. Episodic baseline (tau=0.20 training, tau=0.20 eval)")
    h4 = headline["baseline_epi_eval_tau020"]
    lines.append(f"- fall_rate mean: {h4['fall_rate_mean']:.4f}")
    lines.append(f"- mean_ep_len mean: {h4['mean_ep_len_mean']:.2f}")
    lines.append("")
    lines.append("### 5. Policy checkpoints (for R_task feasibility)")
    h5 = headline["checkpoint_files_found"]
    lines.append(f"- Checkpoint files found: **{h5['count']}**")
    lines.append(
        f"- R_task from existing corpus feasible: **{h5['r_task_feasible_from_existing_corpus']}**"
    )
    if h5["count"] > 0:
        lines.append("- Sample paths:")
        for p in h5["paths"]:
            lines.append(f"  - `{p}`")
    lines.append("")
    lines.append("## Implications for plan v2")
    lines.append("")
    h1v = headline["fall_rate_eval_neutral_tau000"]["verdict"]
    if h1v == "confirmed":
        lines.append(
            "- A.1 metric must use `eval_tau020`, not `eval_neutral`. `eval_neutral` fall_rate is useless."
        )
    h2u = headline["fall_rate_eval_tau020_tau000"]["useful_as_signal"]
    if not h2u:
        lines.append(
            "- `eval_tau020` fall_rate is degenerate (saturated at 0 or 1). Use `mean_ep_len` instead."
        )
    else:
        lines.append("- `eval_tau020` fall_rate is a usable continuous signal.")
    if not h5["r_task_feasible_from_existing_corpus"]:
        lines.append(
            "- **R_task cannot be computed from the existing corpus** (no checkpoints). Any disturbance-recovery benchmark requires retraining with checkpointing."
        )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}")

    # Print short summary to stdout
    print("\n=== SUMMARY ===")
    print(
        f"fall_rate eval_neutral tau=0.00: mean={h1['mean']:.4f} verdict={h1['verdict']}"
    )
    print(
        f"fall_rate eval_tau020 tau=0.00: mean={h2['mean']:.4f} useful={h2['useful_as_signal']}"
    )
    print(f"mean_ep_len eval_tau020 tau=0.00: mean={h3['mean']:.2f}")
    print(
        f"checkpoints found: {h5['count']} (R_task feasible: {h5['r_task_feasible_from_existing_corpus']})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
