#!/usr/bin/env python3
"""Pre-specified analysis for the SAC 3-threshold, 3-seed panel.

Combines the existing SAC pilot (24 bodies x {tau=0, tau=0.5} x {seed 42, 43}
= 96 runs) with the 120-run expansion (adds tau=0.20 across all 3 seeds and
seed 44 across all 3 tau) to yield a full 24 x 3 x 3 = 216-cell panel.

Applies the pre-specified gate:

  MAIN-TEXT upgrade
      Interaction p < 0.01 AND seed-averaged family-mean Str/RF satisfies
      Floor > VT > MVT (all strict).

  APPENDIX-STAY
      Interaction 0.01 <= p < 0.05 OR full Floor > VT > MVT ordering
      violated.

  DROP
      Interaction p >= 0.05 OR seed-averaged PSR_epi < 0.05 for > 3 of 24
      bodies.

Writes records/sac_full_panel_analysis.json with all numbers plus
the gate verdict. Exit code 0 regardless of verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Morphology is nested in family, so in the blocked models the family main effect is
# aliased with the morphology fixed effects and the design is rank-deficient by
# construction; the tests of interest are unaffected. Recent statsmodels versions warn
# on every such fit, so that one expected warning is silenced here.
warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "records" / "sac_full_panel_analysis.json"
MORPH_LIST = REPO / "scripts" / "mechanism_morphologies.txt"

TAUS = {"rf": "0p00", "epi": "0p20", "strict": "0p50"}
SEEDS = (42, 43, 44)
EVAL_KEY = "eval_tau020"

# Fixed gate thresholds used by the SAC scope check.
MAIN_TEXT_ALPHA = 0.01
APPENDIX_ALPHA = 0.05
RF_FLOOR = 0.05
RF_FLOOR_MAX_BODIES = 3
QC_RELIABILITY_FACTOR = 2.0
QC_MIN_VALID_CELLS = 204
QC_RELIABILITY_FLOOR_SD = 0.05
QC_RELIABILITY_ABS_THRESHOLD = 0.10

FAMILIES = ("floor", "vt", "mvt")

# Code-identity provenance (QC 5 per spec). Training scripts whose drift
# between pilot and expansion would invalidate the combined analysis.
PROVENANCE_FILES = (
    "scripts/train_sac_mjx.py",
    "slurm/train_sac_mjx.sh",
    "scripts/mechanism_morphologies.txt",
)


def _run(cmd: List[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout or "").strip()


def capture_provenance() -> dict:
    """QC 5: git HEAD and SHA256 of training files at analysis time."""
    prov = {
        "git_head": _run(["git", "rev-parse", "HEAD"], cwd=REPO) or "unknown",
        "git_describe": _run(["git", "describe", "--always", "--dirty"], cwd=REPO)
        or "unknown",
        "working_tree_clean": _run(["git", "status", "--porcelain"], cwd=REPO) == "",
        "file_hashes": {},
    }
    for rel in PROVENANCE_FILES:
        p = REPO / rel
        if p.exists():
            prov["file_hashes"][rel] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            prov["file_hashes"][rel] = None
    return prov


def family_of(name: str) -> str:
    if name.startswith("floor"):
        return "floor"
    if name.startswith("mvt"):
        return "mvt"
    if name.startswith("vt"):
        return "vt"
    raise ValueError(f"unknown family for {name}")


def load_fixed_morphology_list() -> List[str]:
    """The 24 bodies covered by the SAC panel.

    The list stores `{body}.xml`, but results.json keys are the bare body
    id. Strip the .xml suffix here so downstream lookups work.
    """
    bodies = []
    with MORPH_LIST.open() as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            if name.endswith(".xml"):
                name = name[: -len(".xml")]
            bodies.append(name)
    return bodies


def load_psr_per_seed() -> Tuple[
    pd.DataFrame, Dict[Tuple[str, str, int], dict], List[str]
]:
    """Return (long dataframe, per-cell-status dict, file_errors list).

    Builds the fixed 24 x 3 x 3 grid from the morphology list, loads each
    expected cell, and classifies it as "valid", "missing-file",
    "missing-body", "file-error", "nan", or "non-finite". Downstream QC
    counts and the gate verdict operate on this fixed grid, never on
    whatever happened to load.

    per_cell is STRICTLY tuple-keyed by (body, tau_key, seed). Any JSON
    decode failure fails closed: every cell for that (tau, seed) file gets
    status "file-error", and the decoder message is appended to
    file_errors (returned separately so per_cell stays type-safe).
    """
    fixed_bodies = load_fixed_morphology_list()
    per_cell: Dict[Tuple[str, str, int], dict] = {}
    file_errors: List[str] = []
    rows: List[dict] = []
    for tau_key, tau_str in TAUS.items():
        for seed in SEEDS:
            path = RESULTS / f"sac_mjx_{tau_str}_seed{seed}" / "results.json"
            file_exists = path.exists()
            file_broken = False
            data = {}
            if file_exists:
                try:
                    with path.open() as f:
                        data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                    data = {}
                    file_broken = True
                    file_errors.append(f"{path}: {type(exc).__name__}: {exc}")
            for body in fixed_bodies:
                key = (body, tau_key, seed)
                if not file_exists:
                    per_cell[key] = {"status": "missing-file", "psr": None}
                    continue
                if file_broken:
                    per_cell[key] = {"status": "file-error", "psr": None}
                    continue
                rec = data.get(body)
                if rec is None:
                    per_cell[key] = {"status": "missing-body", "psr": None}
                    continue
                psr = rec.get(EVAL_KEY, {}).get("per_step_reward")
                if psr is None:
                    per_cell[key] = {"status": "nan", "psr": None}
                    continue
                try:
                    psr_f = float(psr)
                except (TypeError, ValueError):
                    per_cell[key] = {"status": "non-finite", "psr": None}
                    continue
                if not np.isfinite(psr_f):
                    per_cell[key] = {"status": "non-finite", "psr": None}
                    continue
                per_cell[key] = {"status": "valid", "psr": psr_f}
                rows.append(
                    dict(
                        body=body,
                        family=family_of(body),
                        tau_key=tau_key,
                        tau_num={"rf": 0.0, "epi": 0.2, "strict": 0.5}[tau_key],
                        seed=seed,
                        psr=psr_f,
                    )
                )
    return pd.DataFrame(rows), per_cell, file_errors


def qc_counts(df: pd.DataFrame, per_cell: dict) -> dict:
    """QC 1-3: cell completeness against the fixed 24x3x3 grid."""
    fixed_bodies = load_fixed_morphology_list()
    expected = len(fixed_bodies) * len(TAUS) * len(SEEDS)

    status_counts: Dict[str, int] = {}
    invalid_cells: List[dict] = []
    seed_counts: Dict[Tuple[str, str], int] = {}
    for key, meta in per_cell.items():
        if not (isinstance(key, tuple) and len(key) == 3):
            continue
        body, tau_key, seed = key
        st = meta["status"]
        status_counts[st] = status_counts.get(st, 0) + 1
        if st == "valid":
            seed_counts[(body, tau_key)] = seed_counts.get((body, tau_key), 0) + 1
        else:
            invalid_cells.append(
                {"body": body, "tau_key": tau_key, "seed": seed, "status": st}
            )
    undercovered = [
        {"body": b, "tau_key": t, "valid_seeds": c}
        for (b, t), c in seed_counts.items()
        if c < 2
    ]
    missing_cell_combos = [
        {"body": b, "tau_key": t}
        for b in fixed_bodies
        for t in TAUS
        if (b, t) not in seed_counts
    ]

    observed_valid = status_counts.get("valid", 0)
    return {
        "bodies": len(fixed_bodies),
        "expected_cells": expected,
        "observed_valid_cells": observed_valid,
        "missing_or_invalid_cells": expected - observed_valid,
        "status_histogram": status_counts,
        "passes_min_valid": observed_valid >= QC_MIN_VALID_CELLS,
        "min_valid_threshold": QC_MIN_VALID_CELLS,
        "undercovered_body_tau": undercovered,
        "missing_body_tau_cells": missing_cell_combos,
        "invalid_cells_sample": invalid_cells[:20],
    }


def seed_avg_per_body_tau(df: pd.DataFrame) -> pd.DataFrame:
    """Return (body, tau_key) -> mean psr across seeds."""
    return (
        df.groupby(["body", "family", "tau_key", "tau_num"])["psr"].mean().reset_index()
    )


def seed_sd_per_body_tau(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["body", "family", "tau_key"])["psr"]
        .std(ddof=0)
        .reset_index()
        .rename(columns={"psr": "seed_sd"})
    )


def reliability_qc(seed_sd: pd.DataFrame) -> dict:
    """QC 4: median per-body seed-SD at tau=0.20 <= 2x larger of medians at tau=0 / 0.5.

    Per spec: "If median SD at either existing tau is below 0.05, use an
    absolute threshold of 0.10 instead." That is EITHER rf or strict below
    0.05, not only when BOTH are below (which would be the same as the
    larger comparator being below 0.05). The predicate here is a logical OR
    on the two existing-tau medians.
    """
    by_tau = seed_sd.groupby("tau_key")["seed_sd"].median().to_dict()
    m_epi = by_tau.get("epi", float("nan"))
    m_rf = by_tau.get("rf", float("nan"))
    m_str = by_tau.get("strict", float("nan"))
    existing_max = (
        max(m_rf, m_str) if np.isfinite(m_rf) and np.isfinite(m_str) else float("nan")
    )
    # Spec: "If median SD at either existing tau is below 0.05, use absolute 0.10"
    any_existing_below_floor = (
        np.isfinite(m_rf) and m_rf < QC_RELIABILITY_FLOOR_SD
    ) or (np.isfinite(m_str) and m_str < QC_RELIABILITY_FLOOR_SD)
    if any_existing_below_floor:
        threshold = QC_RELIABILITY_ABS_THRESHOLD
        rule_applied = "absolute-0.10 (at least one existing-tau median SD < 0.05)"
    else:
        threshold = QC_RELIABILITY_FACTOR * existing_max
        rule_applied = f"{QC_RELIABILITY_FACTOR}x larger of existing medians"
    passes = np.isfinite(m_epi) and np.isfinite(threshold) and (m_epi <= threshold)
    return {
        "median_sd_rf": m_rf,
        "median_sd_epi": m_epi,
        "median_sd_strict": m_str,
        "existing_max_for_comparator": existing_max,
        "any_existing_below_floor": bool(any_existing_below_floor),
        "threshold": threshold,
        "rule_applied": rule_applied,
        "passes": bool(passes),
    }


def rf_floor_qc(seed_avg: pd.DataFrame) -> dict:
    """DROP gate: seed-averaged PSR_epi < 0.05 for > 3/24 bodies."""
    epi = seed_avg[seed_avg["tau_key"] == "epi"]
    below = epi[epi["psr"] < RF_FLOOR]
    return {
        "bodies_below_floor": int(len(below)),
        "threshold_bodies": RF_FLOOR_MAX_BODIES,
        "floor_psr": RF_FLOOR,
        "below_list": list(below["body"].values),
        "passes": int(len(below)) <= RF_FLOOR_MAX_BODIES,
    }


def interaction_anova(df: pd.DataFrame) -> dict:
    """Blocked protocol x family ANOVA on per-seed PSR."""
    model = ols("psr ~ C(tau_key) * C(family) + C(body)", data=df).fit()
    tbl = sm.stats.anova_lm(model, typ=2)
    if "C(tau_key):C(family)" not in tbl.index:
        return {"error": "interaction term missing from ANOVA table"}
    row = tbl.loc["C(tau_key):C(family)"]
    return {
        "interaction_F": float(row["F"]),
        "interaction_df_num": float(row["df"]),
        "interaction_p": float(row["PR(>F)"]),
        "residual_df": float(tbl.loc["Residual"]["df"]),
        "n_cells": int(len(df)),
    }


def family_ratios(seed_avg: pd.DataFrame) -> dict:
    """Seed-averaged per-body Str/RF and related ratios, aggregated by family.

    Floor rule: per paper convention, drop bodies where the denominator PSR
    is <= 0.05. Strict inequality 'rf > 0.05' kept here. Reports both family
    size (n) and the number of bodies that survive the floor mask
    (n_floor_ok).
    """
    wide = seed_avg.pivot_table(
        index="body", columns="tau_key", values="psr", aggfunc="mean"
    )
    wide["family"] = [family_of(b) for b in wide.index]

    def safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        # Match pilot code: keep if den > 0.05, drop if den <= 0.05.
        den_safe = np.where(den > RF_FLOOR, den, np.nan)
        return num / den_safe

    wide["str_over_rf"] = safe_ratio(wide["strict"].values, wide["rf"].values)
    wide["rf_over_epi"] = safe_ratio(wide["rf"].values, wide["epi"].values)
    wide["str_over_epi"] = safe_ratio(wide["strict"].values, wide["epi"].values)

    out = {}
    for fam in FAMILIES:
        sub = wide[wide["family"] == fam]
        n_floor_ok = int(np.sum(np.isfinite(sub["str_over_rf"].values)))
        out[fam] = {
            "n": int(len(sub)),
            "n_floor_ok": n_floor_ok,
            "mean_str_over_rf": float(np.nanmean(sub["str_over_rf"].values)),
            "mean_rf_over_epi": float(np.nanmean(sub["rf_over_epi"].values)),
            "mean_str_over_epi": float(np.nanmean(sub["str_over_epi"].values)),
            "psr_epi": float(np.nanmean(sub["epi"].values)),
            "psr_rf": float(np.nanmean(sub["rf"].values)),
            "psr_strict": float(np.nanmean(sub["strict"].values)),
        }
    return out


def check_family_ordering(ratios: dict) -> dict:
    """Gate criterion: Floor > VT > MVT on seed-averaged str_over_rf (strict)."""
    f = ratios["floor"]["mean_str_over_rf"]
    v = ratios["vt"]["mean_str_over_rf"]
    m = ratios["mvt"]["mean_str_over_rf"]
    strict_order = (f > v) and (v > m)
    return {
        "floor_str_over_rf": f,
        "vt_str_over_rf": v,
        "mvt_str_over_rf": m,
        "strict_floor_gt_vt_gt_mvt": bool(strict_order),
    }


def cross_algo_spearman(seed_avg_sac: pd.DataFrame) -> dict:
    """Compare per-body Str/RF between SAC full panel and PPO corpus (same 24 bodies)."""
    sac_ratio = _per_body_str_rf(seed_avg_sac)
    ppo_ratio = _load_ppo_ratios_for_bodies(list(sac_ratio.keys()))
    joint_bodies = sorted(set(sac_ratio) & set(ppo_ratio))
    valid = [
        b
        for b in joint_bodies
        if np.isfinite(sac_ratio[b]) and np.isfinite(ppo_ratio[b])
    ]
    sac_vec = np.array([sac_ratio[b] for b in valid])
    ppo_vec = np.array([ppo_ratio[b] for b in valid])
    if len(valid) < 5:
        return {"error": f"too few joint valid bodies ({len(valid)})"}
    rho, p = stats.spearmanr(sac_vec, ppo_vec)
    return {
        "n_joint_valid": len(valid),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "dropped_bodies": [b for b in joint_bodies if b not in valid],
    }


def _per_body_str_rf(seed_avg: pd.DataFrame) -> Dict[str, float]:
    wide = seed_avg.pivot_table(
        index="body", columns="tau_key", values="psr", aggfunc="mean"
    )
    rf = wide["rf"].values
    st = wide["strict"].values
    rf_safe = np.where(rf > RF_FLOOR, rf, np.nan)
    ratio = st / rf_safe
    return dict(zip(wide.index, ratio))


def _load_ppo_ratios_for_bodies(bodies: List[str]) -> Dict[str, float]:
    """Mean PPO Str/RF ratio per body across seeds, at tau_eval=0.20."""
    ppo_taus = {"rf": "0p00", "strict": "0p50"}
    vals: defaultdict[str, dict] = defaultdict(dict)
    for key, tau_str in ppo_taus.items():
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau_str}_seed{seed}" / "results.json"
            if not path.exists():
                continue
            with path.open() as f:
                data = json.load(f)
            for body in bodies:
                psr = data.get(body, {}).get(EVAL_KEY, {}).get("per_step_reward")
                if psr is None:
                    continue
                vals[body].setdefault(key, []).append(float(psr))
    ratios = {}
    for body in bodies:
        rf_list = vals.get(body, {}).get("rf", [])
        st_list = vals.get(body, {}).get("strict", [])
        if len(rf_list) == 0 or len(st_list) == 0:
            continue
        rf_mean = float(np.mean(rf_list))
        st_mean = float(np.mean(st_list))
        # Match SAC side: keep if rf > 0.05, drop if rf <= 0.05 (strict
        # inequality applied symmetrically across algorithms).
        if rf_mean > RF_FLOOR:
            ratios[body] = st_mean / rf_mean
        else:
            ratios[body] = float("nan")
    return ratios


def apply_gate(
    anova_result: dict,
    family_order: dict,
    rf_floor: dict,
    qc_cells: dict,
    reliability: dict,
) -> Tuple[str, str]:
    # QC first
    if not qc_cells["passes_min_valid"]:
        return (
            "DROP",
            f"QC FAIL: only {qc_cells['observed_valid_cells']} valid cells < {QC_MIN_VALID_CELLS}",
        )
    if not rf_floor["passes"]:
        return (
            "DROP",
            f"QC FAIL: {rf_floor['bodies_below_floor']} bodies with PSR_epi < {RF_FLOOR} > {RF_FLOOR_MAX_BODIES}",
        )
    if not reliability["passes"]:
        return (
            "DROP",
            f"QC FAIL: median tau=0.20 seed-SD {reliability['median_sd_epi']:.3f} > threshold {reliability['threshold']:.3f}",
        )

    p = anova_result.get("interaction_p", float("nan"))
    if not np.isfinite(p):
        return "DROP", "QC FAIL: interaction p is not finite"

    if p >= APPENDIX_ALPHA:
        return "DROP", f"interaction p = {p:.4f} >= {APPENDIX_ALPHA}"
    if p < MAIN_TEXT_ALPHA and family_order["strict_floor_gt_vt_gt_mvt"]:
        return (
            "MAIN-TEXT",
            f"interaction p = {p:.4f} < {MAIN_TEXT_ALPHA} AND Floor ({family_order['floor_str_over_rf']:.2f}) > VT ({family_order['vt_str_over_rf']:.2f}) > MVT ({family_order['mvt_str_over_rf']:.2f})",
        )
    return (
        "APPENDIX-STAY",
        f"interaction p = {p:.4f} in [{MAIN_TEXT_ALPHA}, {APPENDIX_ALPHA}) OR family ordering not strict Floor>VT>MVT (Floor={family_order['floor_str_over_rf']:.2f}, VT={family_order['vt_str_over_rf']:.2f}, MVT={family_order['mvt_str_over_rf']:.2f})",
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    if not any(RESULTS.glob("sac_mjx_*")):
        # Missing data is not a gate verdict: stop before overwriting the recorded analysis.
        sys.exit(f"ERROR: no sac_mjx_* run directories in {RESULTS}; unzip runs.zip from the dataset first")
    provenance = capture_provenance()
    df, per_cell, file_errors = load_psr_per_seed()
    qc_cells = qc_counts(df, per_cell)
    qc_cells["file_errors"] = file_errors

    # Early guard: if any tau is missing or df is empty, analysis cannot
    # proceed. This is a pre-completion DROP (not a gate evaluation).
    taus_present = set(df["tau_key"].unique()) if len(df) > 0 else set()
    missing_taus = set(TAUS.keys()) - taus_present
    if missing_taus or len(df) == 0 or file_errors:
        verdict = "DROP"
        if file_errors:
            reason = f"QC FAIL: malformed results files: {file_errors[:3]}"
        elif missing_taus:
            reason = f"QC FAIL: taus missing from results: {sorted(missing_taus)}"
        else:
            reason = "QC FAIL: no valid cells loaded"
        out = {
            "spec": "supplementary SAC scope-check record",
            "provenance": provenance,
            "n_bodies": qc_cells["bodies"],
            "qc": {"cells": qc_cells, "taus_present": sorted(taus_present)},
            "gate_verdict": verdict,
            "verdict_reason": reason,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"DROP: {reason}")
        return

    seed_avg = seed_avg_per_body_tau(df)
    seed_sd = seed_sd_per_body_tau(df)
    reliability = reliability_qc(seed_sd)
    rf_floor = rf_floor_qc(seed_avg)
    anova_result = interaction_anova(df)
    ratios = family_ratios(seed_avg)
    family_order = check_family_ordering(ratios)
    cross_algo = cross_algo_spearman(seed_avg)

    verdict, reason = apply_gate(
        anova_result, family_order, rf_floor, qc_cells, reliability
    )

    out = {
        "spec": "supplementary SAC scope-check record",
        "provenance": provenance,
        "n_bodies": qc_cells["bodies"],
        "qc": {
            "cells": qc_cells,
            "reliability": reliability,
            "rf_floor": rf_floor,
        },
        "anova_interaction": anova_result,
        "family_ratios_seed_avg": ratios,
        "family_ordering_check": family_order,
        "cross_algorithm_spearman": cross_algo,
        "gate_verdict": verdict,
        "verdict_reason": reason,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2, default=str)

    print("=" * 70)
    print("SAC Full 3-tau 3-seed Panel Analysis")
    print("=" * 70)
    print(
        f"Git HEAD: {provenance['git_head'][:12]} (clean tree: {provenance['working_tree_clean']})"
    )
    print(
        f"Bodies: {qc_cells['bodies']}, valid cells: {qc_cells['observed_valid_cells']}/{qc_cells['expected_cells']}"
    )
    if qc_cells["status_histogram"]:
        print(f"Status histogram: {qc_cells['status_histogram']}")
    print(f"QC min-valid >= {QC_MIN_VALID_CELLS}: {qc_cells['passes_min_valid']}")
    print(
        f"QC reliability: {reliability['passes']} (median SD tau=0.2 = {reliability['median_sd_epi']:.3f}, threshold {reliability['threshold']:.3f})"
    )
    print(
        f"QC rf-floor: {rf_floor['passes']} ({rf_floor['bodies_below_floor']} bodies below)"
    )
    print()
    if "interaction_p" in anova_result:
        print(
            f"Blocked ANOVA: F({anova_result['interaction_df_num']:.0f}, {anova_result['residual_df']:.0f}) = {anova_result['interaction_F']:.2f}, p = {anova_result['interaction_p']:.4f}"
        )
    for fam in FAMILIES:
        r = ratios[fam]
        print(f"  {fam}: str/rf = {r['mean_str_over_rf']:.2f} (n={r['n']})")
    print(f"Floor > VT > MVT (strict): {family_order['strict_floor_gt_vt_gt_mvt']}")
    print()
    if "spearman_rho" in cross_algo:
        print(
            f"Cross-algo SAC-PPO Str/RF rho = {cross_algo['spearman_rho']:.3f} (n = {cross_algo['n_joint_valid']}, p = {cross_algo['spearman_p']:.2e})"
        )
    print()
    print(f"Gate verdict: {verdict}")
    print(f"  Reason: {reason}")
    print()
    print(f"Full analysis written to {OUT}")


if __name__ == "__main__":
    main()
