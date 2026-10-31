#!/usr/bin/env python3
"""Pre-registered analysis of the asymptotic 5M-current + 50M-current panels.

Implements the analysis frozen in records/asymptotic_spec.md:

1. QC gate: require >= 292/324 valid cells across both budgets.
2. 5M-current abort gate: interaction p < 0.10 with correct family
   direction, anchor (Floor<VT RF/Epi) preserved. Fail -> drop 50M.
3. 50M primary ANOVA: per-seed blocked ANOVA + cluster-robust backup +
   seed-averaged backup.
4. Spearman rho: per-body Delta_reset at 5M-current vs 50M-current on
   the 18 shared bodies.
5. Family direction preservation: 4 checks, 3/4 required for main
   inclusion (anchor Floor<VT RF/Epi at 50M is required for any
   inclusion).
6. Apply gates: body-text, main_inclusion, appendix_only, or drop.

Usage:
    python scripts/analyze_asymptotic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols

PANEL_JSON = Path("records/asymptotic_panel_bodies.json")
RESULTS_5M = Path("results/asymptotic_5m")
RESULTS_50M = Path("results/asymptotic_50m_v2")

TAUS = {"reset_free": "0p00", "episodic": "0p20", "strict": "0p50"}
SEEDS = (42, 43, 44)

QC_MIN = 292
QC_TOTAL = 324

# Gates (frozen per asymptotic_spec.md)
GATE_BODY_TEXT = dict(p5m=0.10, p50m=0.025, rho=0.70, dirs=4)
GATE_MAIN = dict(p5m=0.10, p50m=0.05, rho=0.60, dirs=3)
GATE_APPENDIX = dict(p50m=0.10, rho=0.50, dirs=None)


def load_cell(root: Path, body: str, tau_label: str, seed: int) -> float:
    """Return eval_tau020 per_step_reward for one cell; NaN on any NaN field."""
    fp = root / body / f"tau{tau_label}_seed{seed}" / "results.json"
    if not fp.exists():
        return float("nan")
    d = json.load(open(fp))
    rec = d.get(body, {})
    for ek in ("eval_tau020", "eval_tau050", "eval_neutral"):
        blk = rec.get(ek)
        if not isinstance(blk, dict):
            return float("nan")
        for f in ("per_step_reward", "fitness", "distance"):
            if f not in blk or not np.isfinite(blk[f]):
                return float("nan")
    v = rec.get("eval_tau020", {}).get("per_step_reward")
    return float(v) if np.isfinite(v) else float("nan")


def load_long_df(root: Path, panel: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in panel:
        body, fam = entry["name"], entry["family"]
        for protocol, tau_label in TAUS.items():
            for seed in SEEDS:
                psr = load_cell(root, body, tau_label, seed)
                rows.append(
                    dict(
                        body=body,
                        family=fam,
                        protocol=protocol,
                        seed=seed,
                        psr=psr,
                    )
                )
    return pd.DataFrame(rows)


def qc_gate(df5m: pd.DataFrame, df50m: pd.DataFrame) -> dict:
    valid = int(np.isfinite(df5m["psr"]).sum() + np.isfinite(df50m["psr"]).sum())
    total = len(df5m) + len(df50m)
    return dict(
        valid=valid,
        total=total,
        qc_pass=valid >= QC_MIN,
        qc_threshold=QC_MIN,
    )


def twoway(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["psr"]).copy()
    if sub.empty:
        return dict(error="no rows")
    model = ols("psr ~ C(protocol) * C(family) + C(body)", data=sub).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    row = anova.loc["C(protocol):C(family)"]
    resid_df = float(anova.loc["Residual", "df"])

    # Cluster-robust Wald backup
    model_cr = ols("psr ~ C(protocol) * C(family) + C(body)", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub.body}
    )
    inter_names = [
        n for n in model_cr.params.index if "C(protocol)" in n and "C(family)" in n
    ]
    wald = model_cr.wald_test(inter_names, scalar=True)

    # Seed-averaged
    sa = sub.groupby(["body", "family", "protocol"])["psr"].mean().reset_index()
    model_sa = ols("psr ~ C(protocol) * C(family) + C(body)", data=sa).fit()
    anova_sa = sm.stats.anova_lm(model_sa, typ=2)
    row_sa = anova_sa.loc["C(protocol):C(family)"]
    resid_sa_df = float(anova_sa.loc["Residual", "df"])

    return dict(
        per_seed_f=float(row["F"]),
        per_seed_p=float(row["PR(>F)"]),
        per_seed_df_num=float(row["df"]),
        per_seed_df_denom=resid_df,
        cluster_robust_f=float(wald.statistic),
        cluster_robust_p=float(wald.pvalue),
        seed_avg_f=float(row_sa["F"]),
        seed_avg_p=float(row_sa["PR(>F)"]),
        seed_avg_df_denom=resid_sa_df,
    )


def family_ratios(df: pd.DataFrame) -> dict:
    sa = df.groupby(["body", "family", "protocol"])["psr"].mean().unstack("protocol")
    out = {}
    for fam in ("floor", "vt", "mvt"):
        sub = sa.xs(fam, level="family")
        if len(sub) == 0:
            continue
        out[fam] = dict(
            n=int(len(sub)),
            rf_over_epi=float((sub["reset_free"] / sub["episodic"]).mean()),
            str_over_epi=float((sub["strict"] / sub["episodic"]).mean()),
        )
    return out


def direction_checks(ratios_5m: dict, ratios_50m: dict) -> dict:
    checks = {
        "floor_lt_vt_rf_5m": ratios_5m["floor"]["rf_over_epi"]
        < ratios_5m["vt"]["rf_over_epi"],
        "floor_lt_vt_rf_50m": ratios_50m["floor"]["rf_over_epi"]
        < ratios_50m["vt"]["rf_over_epi"],
        "vt_lt_mvt_str_5m": ratios_5m["vt"]["str_over_epi"]
        < ratios_5m["mvt"]["str_over_epi"],
        "vt_lt_mvt_str_50m": ratios_50m["vt"]["str_over_epi"]
        < ratios_50m["mvt"]["str_over_epi"],
    }
    n_pass = sum(checks.values())
    anchor_50m = checks["floor_lt_vt_rf_50m"]
    return dict(
        checks=checks,
        n_pass=n_pass,
        anchor_50m_pass=anchor_50m,
    )


def per_body_delta(df: pd.DataFrame) -> dict[str, float]:
    sa = df.groupby(["body", "protocol"])["psr"].mean().unstack("protocol")
    if "episodic" not in sa or "reset_free" not in sa:
        return {}
    delta = sa["episodic"] - sa["reset_free"]
    return delta.dropna().to_dict()


def spearman_with_ci(x: np.ndarray, y: np.ndarray, B: int = 10000) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return dict(error=f"n={n}, need >=2", n=n)
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        r = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            boots.append(r)
    if not boots:
        return dict(error="degenerate bootstrap", rho=rho, n=n)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(rho=rho, ci_lo=float(lo), ci_hi=float(hi), n=n)


def apply_gates(
    anova_5m: dict,
    anova_50m: dict,
    rank_corr: dict,
    directions: dict,
) -> dict:
    p5m = anova_5m.get("per_seed_p", 1.0)
    p50m = anova_50m.get("per_seed_p", 1.0)
    rho = rank_corr.get("rho", 0.0)
    dirs = directions["n_pass"]
    anchor = directions["anchor_50m_pass"]

    # Require anchor for any inclusion
    if not anchor:
        return dict(decision="drop", reason="anchor Floor<VT RF/Epi fails at 50M")

    if (
        p5m < GATE_BODY_TEXT["p5m"]
        and p50m < GATE_BODY_TEXT["p50m"]
        and rho >= GATE_BODY_TEXT["rho"]
        and dirs >= GATE_BODY_TEXT["dirs"]
    ):
        return dict(decision="body_text", p5m=p5m, p50m=p50m, rho=rho, dirs=dirs)
    if (
        p5m < GATE_MAIN["p5m"]
        and p50m < GATE_MAIN["p50m"]
        and rho >= GATE_MAIN["rho"]
        and dirs >= GATE_MAIN["dirs"]
    ):
        return dict(
            decision="main_inclusion", p5m=p5m, p50m=p50m, rho=rho, dirs=dirs
        )
    if p50m < GATE_APPENDIX["p50m"] and rho >= GATE_APPENDIX["rho"]:
        return dict(decision="appendix_only", p50m=p50m, rho=rho)
    return dict(decision="drop", p5m=p5m, p50m=p50m, rho=rho, dirs=dirs)


def abort_gate_5m(anova_5m: dict, ratios_5m: dict) -> dict:
    """Applied before 50M launch. Return {pass: bool, reason: str}."""
    p = anova_5m.get("per_seed_p", 1.0)
    anchor = (
        ratios_5m.get("floor", {}).get("rf_over_epi", 1.0)
        < ratios_5m.get("vt", {}).get("rf_over_epi", 0.0)
    )
    if p >= 0.10:
        return dict(passed=False, reason=f"5M interaction p={p:.3f} >= 0.10")
    if not anchor:
        return dict(passed=False, reason="5M anchor Floor<VT RF/Epi fails")
    return dict(passed=True, p=p, anchor=anchor)


def abort_only_5m(panel: list[dict]) -> None:
    """5M-only abort gate, independent of 50M state.

    Narrow decision surface: given only 5M cells, should we launch 50M?
    - QC on 5M alone: 146/162 valid (90%) required
    - 5M interaction p < 0.10 AND Floor<VT RF/Epi anchor
    - Does NOT touch 50M data or analysis path
    """
    df5m = load_long_df(RESULTS_5M, panel)
    valid_5m = int(np.isfinite(df5m["psr"]).sum())
    total_5m = len(df5m)
    qc_5m_min = 146  # >= 90% of 162
    print(
        f"5M-only panel: valid={valid_5m}/{total_5m} (need >= {qc_5m_min} for abort gate)"
    )
    if valid_5m < qc_5m_min:
        out = dict(
            mode="abort_only_5m",
            valid_5m=valid_5m,
            total_5m=total_5m,
            qc_5m_min=qc_5m_min,
            decision=dict(
                decision="pending",
                reason=f"insufficient 5M data ({valid_5m}/{qc_5m_min})",
            ),
        )
        Path("records/asymptotic_analysis.json").write_text(
            json.dumps(out, indent=2)
        )
        print("Pending: not enough 5M cells for abort gate.")
        return

    anova_5m = twoway(df5m)
    ratios_5m = family_ratios(df5m)
    abort = abort_gate_5m(anova_5m, ratios_5m)
    print(
        f"5M interaction: F={anova_5m.get('per_seed_f', float('nan')):.2f}, "
        f"p={anova_5m.get('per_seed_p', float('nan')):.3e}"
    )
    print(f"Abort gate: {'PASS' if abort['passed'] else 'FAIL'} ({abort.get('reason','')})")
    out = dict(
        mode="abort_only_5m",
        valid_5m=valid_5m,
        total_5m=total_5m,
        anova_5m=anova_5m,
        ratios_5m=ratios_5m,
        abort_gate_5m=abort,
        decision=dict(
            decision="launch_50m" if abort["passed"] else "drop",
            reason=abort.get("reason", "5M abort gate passed; launch 50M"),
        ),
    )
    Path("records/asymptotic_analysis.json").write_text(
        json.dumps(out, indent=2)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abort-only-5m",
        action="store_true",
        help="Run 5M-only abort gate (no 50M QC coupling); decide launch_50m vs drop.",
    )
    args = parser.parse_args()

    if not PANEL_JSON.exists():
        print(f"ERROR: {PANEL_JSON} not found", file=sys.stderr)
        sys.exit(2)
    panel = json.load(open(PANEL_JSON))["bodies"]

    if args.abort_only_5m:
        abort_only_5m(panel)
        return

    df5m = load_long_df(RESULTS_5M, panel)
    df50m = load_long_df(RESULTS_50M, panel)

    qc = qc_gate(df5m, df50m)
    print(
        f"Panel: 18 bodies, 3 taus, 3 seeds, 2 budgets = {qc['total']} cells; "
        f"valid={qc['valid']}/{qc['total']} (QC gate={qc['qc_threshold']})"
    )

    if not qc["qc_pass"]:
        out = dict(qc=qc, decision=dict(decision="drop", reason="QC gate failed"))
        Path("records/asymptotic_analysis.json").write_text(
            json.dumps(out, indent=2)
        )
        print("Drop: QC failed.")
        return

    # 5M-current abort gate
    anova_5m = twoway(df5m)
    ratios_5m = family_ratios(df5m)
    abort = abort_gate_5m(anova_5m, ratios_5m)
    print(f"\n5M-current interaction: F={anova_5m['per_seed_f']:.2f}, p={anova_5m['per_seed_p']:.3e}")
    print(f"Abort gate: {'PASS' if abort['passed'] else 'FAIL'} ({abort.get('reason','')})")

    # If 5M fails, stop here
    if not abort["passed"]:
        out = dict(
            qc=qc,
            anova_5m=anova_5m,
            ratios_5m=ratios_5m,
            abort_gate_5m=abort,
            decision=dict(decision="drop", reason=abort["reason"]),
        )
        Path("records/asymptotic_analysis.json").write_text(
            json.dumps(out, indent=2)
        )
        print("Drop: 5M abort gate failed. Do not launch 50M.")
        return

    # 50M analysis (only if we have it)
    has_50m = np.isfinite(df50m["psr"]).sum() > 100
    if not has_50m:
        out = dict(
            qc=qc,
            anova_5m=anova_5m,
            ratios_5m=ratios_5m,
            abort_gate_5m=abort,
            decision=dict(decision="pending", reason="5M passed; awaiting 50M panel"),
        )
        Path("records/asymptotic_analysis.json").write_text(
            json.dumps(out, indent=2)
        )
        print("5M-current: abort gate PASSED. Ready to launch 50M panel.")
        return

    anova_50m = twoway(df50m)
    ratios_50m = family_ratios(df50m)
    delta5m = per_body_delta(df5m)
    delta50m = per_body_delta(df50m)
    shared = sorted(set(delta5m) & set(delta50m))
    x = np.array([delta5m[b] for b in shared])
    y = np.array([delta50m[b] for b in shared])
    rank_corr = spearman_with_ci(x, y)
    directions = direction_checks(ratios_5m, ratios_50m)
    decision = apply_gates(anova_5m, anova_50m, rank_corr, directions)

    report = dict(
        qc=qc,
        anova_5m=anova_5m,
        anova_50m=anova_50m,
        ratios_5m=ratios_5m,
        ratios_50m=ratios_50m,
        rank_correlation=rank_corr,
        directions=directions,
        decision=decision,
    )

    Path("records/asymptotic_analysis.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"\nDecision: {decision['decision']}")


if __name__ == "__main__":
    main()
