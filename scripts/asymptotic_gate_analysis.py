"""
Phase 1b asymptotic gate analysis.

Compares the 18-body Phase 1b panel at 50M steps against the same 18 bodies
in the 5M corpus. Answers the review question: does Delta_reset (and the
protocol x family interaction) persist at asymptote, or wash out?

Preregistered decision rules (frozen BEFORE 50M results land):

  A. Rank preservation: Spearman rho(PSR_5M, PSR_50M) per tau > 0.80 across
     the 18-body panel.
  B. Sign agreement on Delta_reset: per-body sign(Delta_reset_5M) ==
     sign(Delta_reset_50M) >= 85%.
  C. Effect-size retention: |Delta_reset_50M| / |Delta_reset_5M| >= 0.5 in
     at least 12 of 18 bodies.
  D. Residual family structure (the paper's central robustness claim):
     after controlling for baseline_PSR_tau000 and init_height at 50M, the
     ANOVA p on residual Delta_reset across families must be < 0.05 (the
     5M corpus gave p = 0.0007).
  E. Baseline confound stability: Spearman rho(Delta_reset_50M,
     baseline_PSR_50M) should be similar to the 5M corpus value of -0.74
     (within +/- 0.15), otherwise the confound itself depends on training
     horizon.

All 5 are reported. Primary pass criterion: A AND D (rankings and residual
family structure) must hold. B, C, E are supporting robustness metrics.

Usage:
  python scripts/asymptotic_gate_analysis.py

Writes:
  results/asymptotic_50m/asymptotic_gate.json
  records/asymptotic_gate_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
PANEL_JSON = REPO / "records/asymptotic_panel_bodies.json"
CORPUS_DIR = REPO / "results"
NEW_DIR = REPO / "results/asymptotic_50m/jax"
FEATURES_JSON = REPO / "results/morph_features.json"
OUT_JSON = REPO / "results/asymptotic_50m/asymptotic_gate.json"
OUT_MD = REPO / "records/asymptotic_gate_report.md"

SEEDS = (42, 43, 44)
TRAIN_TAUS = ("0p00", "0p20", "0p50")
# Matched eval keys for per-tau gate
MATCHED_EVAL_KEY = {
    "0p00": "eval_neutral",
    "0p20": "eval_tau020",
    "0p50": "eval_tau050",
}
# For Delta_reset, always use eval_tau020 for both tau=0 and tau=0.2
EVAL_DELTA_RESET = "eval_tau020"


def _extract_psr(entry: dict, eval_key: str):
    if eval_key not in entry:
        return None
    blk = entry[eval_key]
    if not isinstance(blk, dict) or "per_step_reward" not in blk:
        return None
    return float(blk["per_step_reward"])


def load_corpus(target_morphs: set[str]) -> dict:
    """5M corpus: {(train_tau, seed, morph): {eval_key: psr}}."""
    out = {}
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
                per_eval = {}
                for ek in ("eval_neutral", "eval_tau020", "eval_tau050"):
                    v = _extract_psr(entry, ek)
                    if v is not None:
                        per_eval[ek] = v
                out[(tau, seed, morph)] = per_eval
    return out


def load_new(bodies) -> tuple[dict, list[str]]:
    """50M panel: {(train_tau, seed, morph): {eval_key: psr}} + missing."""
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
                per_eval = {}
                for ek in ("eval_neutral", "eval_tau020", "eval_tau050"):
                    v = _extract_psr(entry, ek)
                    if v is not None:
                        per_eval[ek] = v
                out[(tau, name, seed)] = per_eval
    return out, missing


def family_of(name: str) -> str:
    for f in ("floor", "vt", "mvt"):
        if name.startswith(f + "-"):
            return f
    return "other"


def _normalize_key(data_5m: dict):
    """load_corpus returns (tau, seed, morph); load_new returns (tau, morph, seed).
    Standardize both to (tau, seed, morph) for the 50M side too."""
    # load_new already uses (tau, morph, seed); reshape to (tau, seed, morph)
    return {(t, s, m): v for (t, m, s), v in data_5m.items()}


def mean_psr(data, tau, morph, eval_key):
    vals = []
    for s in SEEDS:
        cell = data.get((tau, s, morph))
        if cell is None or eval_key not in cell:
            return None
        vals.append(cell[eval_key])
    return float(np.mean(vals))


def main() -> int:
    with open(PANEL_JSON) as f:
        panel_pl = json.load(f)
    bodies = panel_pl["bodies"]
    names = [b["name"] for b in bodies]
    target = set(names)
    print(f"Loading corpus for {len(names)} panel bodies...")
    old = load_corpus(target)
    print(f"Loading 50M runs from {NEW_DIR}...")
    new_raw, missing = load_new(bodies)
    new = _normalize_key(new_raw)
    if missing:
        print(f"WARNING: {len(missing)} 50M files missing:")
        for p in missing[:10]:
            print(f"  {p}")
        if len(missing) > 10:
            print(f"  ...and {len(missing) - 10} more")

    complete = []
    incomplete = []
    for name in names:
        required = True
        reasons = []
        for tau in TRAIN_TAUS:
            for s in SEEDS:
                for side, src in (("5M", old), ("50M", new)):
                    cell = src.get((tau, s, name))
                    if cell is None:
                        reasons.append(f"{side} missing tau={tau}/seed={s}")
                        required = False
        if required:
            # Also require the required eval keys
            for tau in TRAIN_TAUS:
                ek = MATCHED_EVAL_KEY[tau]
                if mean_psr(old, tau, name, ek) is None or mean_psr(new, tau, name, ek) is None:
                    required = False
                    reasons.append(f"5M/50M missing {ek} at tau={tau}")
            if mean_psr(old, "0p00", name, EVAL_DELTA_RESET) is None:
                required = False
                reasons.append("5M missing eval_tau020 at tau=0")
            if mean_psr(new, "0p00", name, EVAL_DELTA_RESET) is None:
                required = False
                reasons.append("50M missing eval_tau020 at tau=0")
        if required:
            complete.append(name)
        else:
            incomplete.append((name, reasons[:3]))

    print(f"Complete bodies: {len(complete)} / {len(names)}")
    if incomplete:
        print("Incomplete:")
        for name, reasons in incomplete[:5]:
            print(f"  {name}: {reasons}")

    if len(complete) < 5:
        print("Too few complete bodies for gate analysis. Aborting.")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(
                {
                    "status": "INCOMPLETE",
                    "complete_bodies": complete,
                    "incomplete_bodies": incomplete,
                    "missing_count": len(missing),
                },
                f,
                indent=2,
            )
        print(f"Wrote {OUT_JSON}")
        return 2

    report = {
        "panel_size": len(names),
        "complete_bodies": complete,
        "incomplete_bodies": [{"name": n, "reasons": r} for n, r in incomplete],
        "missing_count": len(missing),
        "per_tau": {},
        "delta_reset": {},
        "residual_family": {},
        "baseline_confound": {},
        "verdict": None,
        "reasons_fail": [],
    }

    # A. Per-tau rank preservation
    for tau in TRAIN_TAUS:
        ek = MATCHED_EVAL_KEY[tau]
        old_psrs = np.array([mean_psr(old, tau, m, ek) for m in complete])
        new_psrs = np.array([mean_psr(new, tau, m, ek) for m in complete])
        rho = float(stats.spearmanr(old_psrs, new_psrs).statistic)
        r = float(stats.pearsonr(old_psrs, new_psrs).statistic)
        report["per_tau"][tau] = {
            "eval_key": ek,
            "spearman_rho": rho,
            "pearson_r": r,
            "pass_A": bool(rho > 0.80),
        }

    # Delta_reset per body
    old_dr = np.array(
        [
            mean_psr(old, "0p20", m, EVAL_DELTA_RESET)
            - mean_psr(old, "0p00", m, EVAL_DELTA_RESET)
            for m in complete
        ]
    )
    new_dr = np.array(
        [
            mean_psr(new, "0p20", m, EVAL_DELTA_RESET)
            - mean_psr(new, "0p00", m, EVAL_DELTA_RESET)
            for m in complete
        ]
    )

    # B. Sign agreement
    sign_agree = float(np.mean(np.sign(old_dr) == np.sign(new_dr)))
    # C. Effect-size retention
    retention = np.abs(new_dr) / np.maximum(np.abs(old_dr), 1e-6)
    n_retained = int(np.sum(retention >= 0.5))
    # Rank correlation on Delta_reset
    rho_dr = float(stats.spearmanr(old_dr, new_dr).statistic)
    report["delta_reset"] = {
        "spearman_rho_5m_vs_50m": rho_dr,
        "sign_agreement": sign_agree,
        "pass_B": bool(sign_agree >= 0.85),
        "n_bodies_retaining_magnitude_geq_half": n_retained,
        "pass_C": bool(n_retained >= 12),
        "old_dr_mean": float(np.mean(old_dr)),
        "new_dr_mean": float(np.mean(new_dr)),
    }

    # D. Residual family structure at 50M
    baseline_new = np.array([mean_psr(new, "0p00", m, EVAL_DELTA_RESET) for m in complete])
    with open(FEATURES_JSON) as f:
        feats = json.load(f)
    heights = np.array(
        [float(feats[m]["init_height"]) if m in feats else np.nan for m in complete]
    )
    height_mask = ~np.isnan(heights)
    families = np.array([family_of(m) for m in complete])

    if height_mask.sum() >= 10:
        dr_v = new_dr[height_mask]
        bl_v = baseline_new[height_mask]
        h_v = heights[height_mask]
        fam_v = families[height_mask]
        X = np.column_stack([bl_v, h_v, np.ones(len(bl_v))])
        beta, *_ = np.linalg.lstsq(X, dr_v, rcond=None)
        residual = dr_v - X @ beta
        groups = []
        fam_stats = {}
        for fam in ("floor", "vt", "mvt"):
            mask = fam_v == fam
            if mask.sum() < 3:
                continue
            groups.append(residual[mask])
            fam_stats[fam] = {
                "n": int(mask.sum()),
                "residual_mean": float(np.mean(residual[mask])),
                "residual_std": float(np.std(residual[mask])),
            }
        if len(groups) >= 2:
            h_stat, p_kw = stats.kruskal(*groups)
            f_stat, p_anova = stats.f_oneway(*groups)
            report["residual_family"] = {
                "kruskal_h": float(h_stat),
                "kruskal_p": float(p_kw),
                "anova_f": float(f_stat),
                "anova_p": float(p_anova),
                "family_stats": fam_stats,
                "pass_D": bool(p_anova < 0.05),
            }

    # E. Baseline confound stability at 50M
    rho_baseline_new = float(stats.spearmanr(baseline_new, new_dr).statistic)
    report["baseline_confound"] = {
        "rho_50m": rho_baseline_new,
        "rho_5m_corpus_reference": -0.74,
        "pass_E": bool(abs(rho_baseline_new - (-0.74)) < 0.15),
    }

    primary_pass = all(report["per_tau"][tau]["pass_A"] for tau in TRAIN_TAUS) and report[
        "residual_family"
    ].get("pass_D", False)
    report["verdict"] = "PASS" if primary_pass else "FAIL"
    if not primary_pass:
        for tau in TRAIN_TAUS:
            if not report["per_tau"][tau]["pass_A"]:
                report["reasons_fail"].append(
                    f"tau={tau}: Spearman rho {report['per_tau'][tau]['spearman_rho']:.3f} <= 0.80"
                )
        if not report["residual_family"].get("pass_D", False):
            p = report["residual_family"].get("anova_p", None)
            report["reasons_fail"].append(
                f"Residual family ANOVA p={p} fails < 0.05 at 50M"
            )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Wrote {OUT_JSON}")

    # Markdown
    lines = []
    lines.append("# Phase 1b Asymptotic Gate Report")
    lines.append("")
    lines.append(f"**Verdict:** {report['verdict']}")
    lines.append(f"- Panel: {len(names)} bodies, {len(complete)} complete.")
    if incomplete:
        lines.append(f"- Incomplete: {len(incomplete)} bodies.")
    lines.append("")
    lines.append("## A. Per-tau rank preservation (Spearman 5M vs 50M)")
    lines.append("")
    lines.append("| tau | eval | rho | pearson r | pass (>0.80) |")
    lines.append("|---|---|---|---|---|")
    for tau in TRAIN_TAUS:
        r = report["per_tau"][tau]
        lines.append(
            f"| {tau} | {r['eval_key']} | {r['spearman_rho']:.3f} "
            f"| {r['pearson_r']:.3f} | {r['pass_A']} |"
        )
    lines.append("")
    lines.append("## B/C. Delta_reset persistence")
    lines.append("")
    dr = report["delta_reset"]
    lines.append(f"- Spearman rho(Delta_reset_5M, Delta_reset_50M) = {dr['spearman_rho_5m_vs_50m']:.3f}")
    lines.append(f"- Sign agreement: {dr['sign_agreement']:.2f} (pass >= 0.85: {dr['pass_B']})")
    lines.append(
        f"- Effect-size retained (|Delta_reset_50M| / |Delta_reset_5M| >= 0.5) "
        f"in {dr['n_bodies_retaining_magnitude_geq_half']}/{len(complete)} bodies "
        f"(pass >= 12: {dr['pass_C']})"
    )
    lines.append(f"- Mean Delta_reset: 5M = {dr['old_dr_mean']:.3f}, 50M = {dr['new_dr_mean']:.3f}")
    lines.append("")
    lines.append("## D. Residual family structure at 50M (central robustness)")
    lines.append("")
    rf = report["residual_family"]
    if rf:
        lines.append(f"- ANOVA F = {rf['anova_f']:.2f}, p = {rf['anova_p']:.4f} (pass < 0.05: {rf['pass_D']})")
        lines.append(f"- Kruskal-Wallis H = {rf['kruskal_h']:.2f}, p = {rf['kruskal_p']:.4f}")
        for fam in ("floor", "vt", "mvt"):
            if fam in rf["family_stats"]:
                s = rf["family_stats"][fam]
                lines.append(f"- {fam}: n={s['n']}, residual mean = {s['residual_mean']:+.3f} +/- {s['residual_std']:.3f}")
    lines.append("")
    lines.append("## E. Baseline confound stability")
    lines.append("")
    bc = report["baseline_confound"]
    lines.append(f"- rho(baseline_PSR, Delta_reset) at 50M = {bc['rho_50m']:.3f} (5M reference: -0.74)")
    lines.append(f"- Stable within +/- 0.15: {bc['pass_E']}")
    lines.append("")
    if report["reasons_fail"]:
        lines.append("## Reasons for failure")
        for r in report["reasons_fail"]:
            lines.append(f"- {r}")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print()
    print(f"VERDICT: {report['verdict']}")
    for r in report["reasons_fail"]:
        print(f"  REASON: {r}")
    return 0 if primary_pass else 1


if __name__ == "__main__":
    sys.exit(main())
