"""
Phase 1a-bis equivalence pilot analysis (Eq-A arm).

Compares the 12 equivalence-pilot bodies between:
  NEW: results/equivalence_pilot/jax/{body}/tau{TAU}_seed{SEED}/results.json
       (MJX 3.6.0 + JAX 0.9.2, 3 train taus x 3 seeds each)
  OLD: results/mjx_ppo_{0p00,0p20,0p50}_seed{42,43,44}/results.json
       (900-run corpus, same 12 bodies)

D6 gates (all must pass):
  Per-tau (train tau T, evaluated at its matched eval key):
    1. Spearman rho(PSR_old, PSR_new) > 0.85
    3. Median per-body |PSR_new - PSR_old| / max(0.05, |PSR_old|) < 0.10
    5. Linear regression of PSR_new on PSR_old: slope in [0.9, 1.1],
       |intercept| < 0.05
  Cross-tau (Delta_reset = PSR_train020(eval_tau020) - PSR_train000(eval_tau020)):
    2. Spearman rho on ranked Delta_reset > 0.9
    4. Sign agreement on Delta_reset >= 90%

Notes on conventions (verified against the corpus + bodies JSON):
- Per-tau gate uses matched eval keys: train tau=0.0 -> eval_neutral,
  train tau=0.2 -> eval_tau020, train tau=0.5 -> eval_tau050.
- Delta_reset, by contrast, uses eval_tau020 for BOTH train conditions,
  so it is "how much does training with resets change performance on the
  reset-heavy eval protocol", not a mixed-eval difference.
- Seeds are aggregated by MEAN, matching the corpus Delta_reset entries
  already stored in records/equivalence_pilot_bodies.json.
- Bodies missing any of the 9 (3 tau x 3 seed) runs are excluded from the
  D6 gate and reported separately as INCOMPLETE.

Usage:
  python scripts/equivalence_pilot_analysis.py

Writes:
  results/equivalence_pilot/eq_a_analysis.json
  records/equivalence_pilot_report.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
BODIES_JSON = REPO / "records/equivalence_pilot_bodies.json"
CORPUS_DIR = REPO / "results"
NEW_DIR = REPO / "results/equivalence_pilot/jax"
OUT_JSON = REPO / "results/equivalence_pilot/eq_a_analysis.json"
OUT_MD = REPO / "records/equivalence_pilot_report.md"

TRAIN_TAUS = ("0p00", "0p20", "0p50")
SEEDS = (42, 43, 44)

# Matched eval key used for the per-tau gate (items 1, 3, 5). For
# Delta_reset (items 2, 4) we always use EVAL_DELTA_RESET.
MATCHED_EVAL_KEY = {
    "0p00": "eval_neutral",
    "0p20": "eval_tau020",
    "0p50": "eval_tau050",
}
EVAL_DELTA_RESET = "eval_tau020"


def _extract_psr(entry: dict, eval_key: str):
    """Return per_step_reward for a given eval key, or None if absent."""
    if eval_key not in entry:
        return None
    blk = entry[eval_key]
    if not isinstance(blk, dict) or "per_step_reward" not in blk:
        return None
    return float(blk["per_step_reward"])


def load_corpus_per_eval(target_morphs: set[str]) -> dict:
    """Return {(train_tau, morph, seed): {eval_key: psr}} for corpus runs.

    Restricted to the pilot's 12 target morphs.
    """
    out: dict = {}
    for tau in TRAIN_TAUS:
        for seed in SEEDS:
            path = CORPUS_DIR / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing corpus file: {path}")
            with open(path) as f:
                d = json.load(f)
            for morph, entry in d.items():
                if morph not in target_morphs:
                    continue
                per_eval: dict = {}
                for ek in ("eval_neutral", "eval_tau020", "eval_tau050"):
                    v = _extract_psr(entry, ek)
                    if v is not None:
                        per_eval[ek] = v
                out[(tau, morph, seed)] = per_eval
    return out


def load_new_per_eval(bodies) -> tuple[dict, list[str]]:
    """Return ({(train_tau, morph, seed): {eval_key: psr}}, missing paths)."""
    out: dict = {}
    missing: list[str] = []
    for body in bodies:
        name = body["name"]
        for tau in TRAIN_TAUS:
            for seed in SEEDS:
                path = NEW_DIR / name / f"tau{tau}_seed{seed}" / "results.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                with open(path) as f:
                    d = json.load(f)
                if not d:
                    missing.append(f"{path} (empty)")
                    continue
                entry = next(iter(d.values()))
                per_eval: dict = {}
                for ek in ("eval_neutral", "eval_tau020", "eval_tau050"):
                    v = _extract_psr(entry, ek)
                    if v is not None:
                        per_eval[ek] = v
                out[(tau, name, seed)] = per_eval
    return out, missing


def body_completeness(data: dict, morph: str) -> dict:
    """For a morph, return which (tau, seed) cells are present and which
    required eval keys are present in each.
    """
    status = {"present": 0, "missing": 0, "entries": []}
    for tau in TRAIN_TAUS:
        for seed in SEEDS:
            cell = data.get((tau, morph, seed))
            if cell is None:
                status["missing"] += 1
                status["entries"].append((tau, seed, None))
                continue
            # The D6 gate needs eval_neutral and eval_tau020 for every body,
            # matched-eval for the train_tau, and eval_tau020 for Delta_reset.
            required = {MATCHED_EVAL_KEY[tau], EVAL_DELTA_RESET}
            missing_keys = [k for k in required if k not in cell]
            if missing_keys:
                status["missing"] += 1
                status["entries"].append((tau, seed, f"missing {missing_keys}"))
            else:
                status["present"] += 1
                status["entries"].append((tau, seed, "ok"))
    return status


def mean_psr(data: dict, tau: str, morph: str, eval_key: str) -> float | None:
    """Mean PSR across the 3 seeds at (train_tau, morph, eval_key).

    Returns None if any seed is missing the eval_key (so a body with
    incomplete coverage for this metric is flagged, not silently averaged).
    """
    vals = []
    for seed in SEEDS:
        cell = data.get((tau, morph, seed))
        if cell is None or eval_key not in cell:
            return None
        vals.append(cell[eval_key])
    return float(np.mean(vals))


def assemble_arrays(old_data, new_data, complete_morphs):
    """Return a dict of arrays for D6 evaluation, restricted to
    complete_morphs (bodies with all 9 cells and required eval keys).
    """
    arrays = {"morphs": list(complete_morphs)}
    for tau in TRAIN_TAUS:
        ek = MATCHED_EVAL_KEY[tau]
        old_vals = [mean_psr(old_data, tau, m, ek) for m in complete_morphs]
        new_vals = [mean_psr(new_data, tau, m, ek) for m in complete_morphs]
        arrays[tau] = {
            "eval_key": ek,
            "old": np.array(old_vals),
            "new": np.array(new_vals),
        }
    # Delta_reset: eval_tau020 for both train taus 0.0 and 0.2.
    old_dr = [
        mean_psr(old_data, "0p20", m, EVAL_DELTA_RESET)
        - mean_psr(old_data, "0p00", m, EVAL_DELTA_RESET)
        for m in complete_morphs
    ]
    new_dr = [
        mean_psr(new_data, "0p20", m, EVAL_DELTA_RESET)
        - mean_psr(new_data, "0p00", m, EVAL_DELTA_RESET)
        for m in complete_morphs
    ]
    arrays["delta_reset"] = {
        "eval_key": EVAL_DELTA_RESET,
        "old": np.array(old_dr),
        "new": np.array(new_dr),
    }
    return arrays


def evaluate_d6(arrays) -> dict:
    report = {
        "per_tau": {},
        "delta_reset": {},
        "pass": True,
        "reasons": [],
        "n_bodies": len(arrays["morphs"]),
    }

    for tau in TRAIN_TAUS:
        old = arrays[tau]["old"]
        new = arrays[tau]["new"]
        n = len(old)
        if n < 3:
            report["pass"] = False
            report["reasons"].append(
                f"tau={tau}: only {n} bodies available, need >= 3 for Spearman"
            )
            continue
        rho_psr = float(stats.spearmanr(old, new).statistic)
        median_rel_drift = float(
            np.median(np.abs(new - old) / np.maximum(0.05, np.abs(old)))
        )
        slope, intercept, r_value, _, _ = stats.linregress(old, new)

        ok_rho = rho_psr > 0.85
        ok_drift = median_rel_drift < 0.10
        ok_slope = 0.9 <= slope <= 1.1
        ok_intercept = abs(intercept) < 0.05

        report["per_tau"][tau] = {
            "n": int(n),
            "eval_key": arrays[tau]["eval_key"],
            "spearman_rho_psr": rho_psr,
            "median_rel_drift": median_rel_drift,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "pass_rho": bool(ok_rho),
            "pass_drift": bool(ok_drift),
            "pass_slope": bool(ok_slope),
            "pass_intercept": bool(ok_intercept),
        }
        if not (ok_rho and ok_drift and ok_slope and ok_intercept):
            report["pass"] = False
            if not ok_rho:
                report["reasons"].append(
                    f"tau={tau}: Spearman rho_PSR={rho_psr:.3f} <= 0.85"
                )
            if not ok_drift:
                report["reasons"].append(
                    f"tau={tau}: median_rel_drift={median_rel_drift:.3f} >= 0.10"
                )
            if not ok_slope:
                report["reasons"].append(
                    f"tau={tau}: slope={slope:.3f} outside [0.9, 1.1]"
                )
            if not ok_intercept:
                report["reasons"].append(
                    f"tau={tau}: |intercept|={abs(intercept):.3f} >= 0.05"
                )

    old_dr = arrays["delta_reset"]["old"]
    new_dr = arrays["delta_reset"]["new"]
    n_dr = len(old_dr)
    if n_dr < 3:
        report["pass"] = False
        report["reasons"].append(
            f"Delta_reset: only {n_dr} bodies, need >= 3 for Spearman"
        )
    else:
        rho_dr = float(stats.spearmanr(old_dr, new_dr).statistic)
        # sign() treats 0.0 as 0; both being exactly 0 counts as agreement.
        sign_agree = float(np.mean(np.sign(old_dr) == np.sign(new_dr)))
        report["delta_reset"] = {
            "n": int(n_dr),
            "eval_key": arrays["delta_reset"]["eval_key"],
            "spearman_rho_delta_reset": rho_dr,
            "sign_agreement": sign_agree,
            "pass_rho": bool(rho_dr > 0.9),
            "pass_sign": bool(sign_agree >= 0.9),
        }
        if not (rho_dr > 0.9 and sign_agree >= 0.9):
            report["pass"] = False
            if rho_dr <= 0.9:
                report["reasons"].append(
                    f"Delta_reset Spearman rho={rho_dr:.3f} <= 0.9"
                )
            if sign_agree < 0.9:
                report["reasons"].append(
                    f"Delta_reset sign agreement={sign_agree:.2f} < 0.9"
                )

    return report


def write_md(report, arrays, complete_morphs, incomplete) -> str:
    lines = []
    lines.append("# Equivalence Pilot Report (Eq-A: MJX/JAX bridge vs corpus)")
    lines.append("")
    verdict = "PASS" if report["pass"] else "FAIL"
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append(f"- Complete bodies: {len(complete_morphs)} / 12")
    if incomplete:
        lines.append(f"- INCOMPLETE bodies: {len(incomplete)}")
        for b, why in incomplete:
            lines.append(f"  - {b}: {why}")
    lines.append("")
    lines.append("## D6 per-tau metrics (matched eval key)")
    lines.append("")
    lines.append(
        "| tau | n | eval | rho_PSR | rel_drift | slope | b0 "
        "| rho_pass | drift_pass | slope_pass | b0_pass |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for tau in TRAIN_TAUS:
        r = report["per_tau"].get(tau)
        if not r:
            lines.append(f"| {tau} | - | - | - | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {tau} | {r['n']} | {r['eval_key']} "
            f"| {r['spearman_rho_psr']:.3f} | {r['median_rel_drift']:.3f} "
            f"| {r['slope']:.3f} | {r['intercept']:.3f} "
            f"| {r['pass_rho']} | {r['pass_drift']} | {r['pass_slope']} "
            f"| {r['pass_intercept']} |"
        )
    lines.append("")
    lines.append("## Delta_reset (eval_tau020 for both train taus)")
    lines.append("")
    dr = report["delta_reset"]
    if dr:
        lines.append(f"- n = {dr['n']}")
        lines.append(f"- eval_key = {dr['eval_key']}")
        lines.append(
            f"- Spearman rho = {dr['spearman_rho_delta_reset']:.3f} "
            f"(needs > 0.9): pass={dr['pass_rho']}"
        )
        lines.append(
            f"- Sign agreement = {dr['sign_agreement']:.2f} "
            f"(needs >= 0.9): pass={dr['pass_sign']}"
        )
    lines.append("")
    if report["reasons"]:
        lines.append("## Failures")
        for r in report["reasons"]:
            lines.append(f"- {r}")
    return "\n".join(lines)


def main() -> int:
    with open(BODIES_JSON) as f:
        bodies = json.load(f)["bodies"]
    target_morphs = {b["name"] for b in bodies}

    print(f"Loading corpus PSRs for {len(target_morphs)} target morphs...")
    old_data = load_corpus_per_eval(target_morphs)

    print(f"Loading new PSRs from {NEW_DIR}...")
    new_data, missing = load_new_per_eval(bodies)
    if missing:
        print(f"WARNING: {len(missing)} new result files missing / empty:")
        for p in missing[:10]:
            print(f"  {p}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    complete_morphs = []
    incomplete = []
    for b in bodies:
        name = b["name"]
        # Both corpus and new must have full coverage for the body to enter
        # the gate. Corpus coverage is always complete by construction; we
        # assert it so that a new analysis bug surfaces loudly rather than
        # quietly pooling incomplete data.
        old_status = body_completeness(old_data, name)
        new_status = body_completeness(new_data, name)
        if old_status["missing"] != 0:
            incomplete.append((name, f"corpus missing {old_status['missing']}/9"))
            continue
        if new_status["missing"] != 0:
            incomplete.append((name, f"new missing {new_status['missing']}/9"))
            continue
        complete_morphs.append(name)

    print(f"Complete bodies: {len(complete_morphs)} / {len(bodies)}")
    if incomplete:
        print("Incomplete bodies:")
        for b, why in incomplete:
            print(f"  - {b}: {why}")

    arrays = assemble_arrays(old_data, new_data, complete_morphs)
    # Sanity: none of the arrays should contain NaN if complete_morphs is
    # correctly filtered. Fail loud if they do.
    for section, block in arrays.items():
        if section == "morphs":
            continue
        for side in ("old", "new"):
            if any(math.isnan(x) for x in block[side]):
                raise RuntimeError(
                    f"NaN in {section}.{side} arrays after filtering - "
                    f"mean_psr() returned None for a 'complete' body."
                )

    report = evaluate_d6(arrays)
    report["incomplete_bodies"] = [
        {"name": b, "reason": why} for b, why in incomplete
    ]
    report["missing_files_count"] = len(missing)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Wrote {OUT_JSON}")

    md = write_md(report, arrays, complete_morphs, incomplete)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(f"Wrote {OUT_MD}")

    print()
    print("VERDICT:", "PASS" if report["pass"] else "FAIL")
    for r in report["reasons"]:
        print(f"  REASON: {r}")
    if incomplete:
        print(f"  (INCOMPLETE: {len(incomplete)} bodies excluded from gate)")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
