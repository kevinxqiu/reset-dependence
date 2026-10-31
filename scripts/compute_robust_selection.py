#!/usr/bin/env python3
"""Protocol-robust selection analysis (paper Section 4.3 mitigation paragraph).

Question: the main text quantifies that episodic-selected top-k morphologies
retain only 49% of the strict-training oracle's PSR at k=5. This is the
"selection regret" under ignorance of protocol. A natural question is
whether any simple selection recipe eliminates the regret.

Method: at common evaluation threshold tau_eval=0.20, score each morphology
by five candidate criteria and select the top-k under each. Then evaluate
each selection against each of the three single-protocol oracles. The
key summary is "worst-case retention": the minimum in-sample retention across
the three training protocols, all scored with the common evaluator.

Candidate selection criteria:
    epi_only    = PSR_epi         (the headline-regret baseline)
    strict_only = PSR_strict
    rf_only     = PSR_RF
    worst_case  = min(PSR_epi, PSR_strict, PSR_RF)
    mean        = (PSR_epi + PSR_strict + PSR_RF) / 3

Writes: records/protocol_robust_selection.json

Usage:
    python scripts/compute_robust_selection.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "records" / "protocol_robust_selection.json"

SEEDS = (42, 43, 44)
TAUS = {"epi": "0p20", "strict": "0p50", "rf": "0p00"}
EVAL_KEY = "eval_tau020"
KS = (5, 10, 20)
BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 42


def load_seed_avg(train_tau: str) -> dict[str, float]:
    vals: defaultdict[str, list[float]] = defaultdict(list)
    for s in SEEDS:
        path = RESULTS / f"mjx_ppo_{train_tau}_seed{s}" / "results.json"
        with path.open() as f:
            data = json.load(f)
        for body, rec in data.items():
            psr = rec.get(EVAL_KEY, {}).get("per_step_reward")
            if psr is not None:
                vals[body].append(float(psr))
    return {body: float(np.mean(v)) for body, v in vals.items()}


def family_of(name: str) -> str | None:
    if name.startswith("floor"):
        return "floor"
    if name.startswith("mvt"):
        return "mvt"
    if name.startswith("vt"):
        return "vt"
    return None


def top_k_mean(score: np.ndarray, target: np.ndarray, k: int) -> float:
    idx = np.argsort(-score)[:k]
    return float(np.mean(target[idx]))


def oracle_top_k(target: np.ndarray, k: int) -> float:
    return float(np.mean(np.sort(target)[::-1][:k]))


def main() -> None:
    psr = {name: load_seed_avg(t) for name, t in TAUS.items()}
    common = sorted(set(psr["epi"]) & set(psr["strict"]) & set(psr["rf"]))
    n = len(common)
    print(f"Loaded {n} bodies at {EVAL_KEY}")

    epi = np.array([psr["epi"][b] for b in common])
    strict = np.array([psr["strict"][b] for b in common])
    rf = np.array([psr["rf"][b] for b in common])
    fams = [family_of(b) for b in common]

    methods: dict[str, np.ndarray] = {
        "epi_only": epi,
        "strict_only": strict,
        "rf_only": rf,
        "worst_case": np.minimum(np.minimum(epi, strict), rf),
        "mean": (epi + strict + rf) / 3,
    }
    oracles: dict[str, np.ndarray] = {"epi": epi, "strict": strict, "rf": rf}

    # Deterministic bootstrap indices for CIs
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idx = rng.integers(0, n, size=(BOOTSTRAP_B, n))

    # Retention table: method -> oracle -> k -> {point, ci_lo, ci_hi}
    table: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for mname in methods:
        table[mname] = {}
        for oname in oracles:
            table[mname][oname] = {}
            for k in KS:
                mscore = methods[mname]
                otarget = oracles[oname]
                pt = top_k_mean(mscore, otarget, k) / oracle_top_k(otarget, k)
                boot = np.empty(BOOTSTRAP_B)
                for i in range(BOOTSTRAP_B):
                    ii = boot_idx[i]
                    if mname == "worst_case":
                        bs = np.minimum(np.minimum(epi[ii], strict[ii]), rf[ii])
                    elif mname == "mean":
                        bs = (epi[ii] + strict[ii] + rf[ii]) / 3
                    else:
                        bs = {
                            "epi_only": epi[ii],
                            "strict_only": strict[ii],
                            "rf_only": rf[ii],
                        }[mname]
                    bo = otarget[ii]
                    sel_mean = np.mean(bo[np.argsort(-bs)[:k]])
                    orc_mean = np.mean(np.sort(bo)[::-1][:k])
                    boot[i] = sel_mean / orc_mean
                lo, hi = np.percentile(boot, [2.5, 97.5])
                table[mname][oname][k] = {
                    "point": float(pt),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                }

    # Worst-case retention per method (min over oracles at each k)
    worst_case_retention: dict[str, dict[int, float]] = {}
    for mname in methods:
        worst_case_retention[mname] = {}
        for k in KS:
            worst_case_retention[mname][k] = float(
                min(table[mname][o][k]["point"] for o in oracles)
            )

    # Family composition of top-10 selection
    family_comp: dict[str, dict[str, int]] = {}
    for mname, mscore in methods.items():
        idx = np.argsort(-mscore)[:10]
        c = {"floor": 0, "vt": 0, "mvt": 0}
        for i in idx:
            c[fams[i]] += 1
        family_comp[mname] = c

    # Console summary
    print("\n=== Retention (point estimates, %) ===")
    print(
        f"{'method':<12} "
        + " ".join(f"k={k:<4}epi   k={k:<4}str   k={k:<4}rf" for k in KS)
    )
    for mname in methods:
        row = f"{mname:<12}"
        for k in KS:
            for o in ("epi", "strict", "rf"):
                row += f" {table[mname][o][k]['point'] * 100:5.1f}%"
        print(row)

    print("\n=== Worst-case retention across oracles (%) ===")
    for mname in methods:
        vals = [f"k={k}: {worst_case_retention[mname][k] * 100:5.1f}%" for k in KS]
        print(f"  {mname:<12}  " + "  ".join(vals))

    print("\n=== Family composition of top-10 selection ===")
    for mname, c in family_comp.items():
        print(f"  {mname:<12}  floor={c['floor']:<3} vt={c['vt']:<3} mvt={c['mvt']:<3}")

    out = {
        "eval_key": EVAL_KEY,
        "n_bodies": n,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "methods": list(methods),
        "oracles": list(oracles),
        "ks": list(KS),
        "retention_table": table,
        "worst_case_retention": worst_case_retention,
        "family_composition_top10": family_comp,
    }
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
