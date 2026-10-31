#!/usr/bin/env python3
"""Pre-registered analysis of the alt-termination panel.

Implements the analysis frozen in records/alt_termination_spec.md:

1. QC gate: require >= 243/270 valid runs (no NaN PSRs); fail -> drop.
2. Primary 2-way ANOVA on per-body per-seed per-step reward, with
   body-blocking, and a (protocol x family) interaction term. Mirrors
   the main corpus blocked-design spec exactly.
3. Spearman rho between contact-duration-derived Delta_reset
   (PSR_N50 - PSR_N100000) and height-derived Delta_reset
   (PSR_tau020 - PSR_tau000) per-body, both on the same 30 bodies.
4. Per-family direction-check subgates for the appendix-only branch:
   Floor: PSR_N1 < PSR_N50 with 95% CI excluding 0;
   VT:    PSR_N1 < PSR_N50 with 95% CI excluding 0.
5. Apply pre-specified inclusion gates:
   - body-text: p(interaction) < 1e-3 AND rho >= 0.70
   - main inclusion: p < 0.01 AND rho >= 0.60
   - appendix-only: p < 0.01 AND 0 <= rho < 0.40 AND both direction checks pass
   - drop: otherwise

The outcome is written to records/alt_termination_analysis.json
along with a short markdown summary.

Usage:
    python scripts/analyze_alt_termination.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

RESULTS_ROOT = Path("results/alt_termination")
CORPUS_ROOT = Path("results")
PANEL_JSON = Path("records/alt_termination_bodies.json")
EVAL_KEY = "eval_tau020"
SEEDS = (42, 43, 44)
N_MODES = {"strict": 1, "episodic": 50, "reset_free": 100000}
EVAL_KEYS_CHECKED = ("eval_tau020", "eval_tau050", "eval_neutral")
PSR_FIELDS = ("per_step_reward", "fitness", "distance")

# Corpus hyperparameters that the alt-termination cell must match to be
# considered validly pre-registered (per alt_termination_spec.md "Training").
EXPECTED_HPARAMS = {
    "num_envs": 128,
    "unroll_length": 16,
    "n_epochs": 10,
    "minibatch_size": 256,
    "timesteps": 5000000,
    "termination_mode": "contact-duration",
}

# Gates frozen in records/alt_termination_spec.md
GATE_BODY_TEXT = {"p_max": 1e-3, "rho_min": 0.70}
GATE_MAIN_INCLUSION = {"p_max": 0.01, "rho_min": 0.60}
GATE_APPENDIX_ONLY = {"p_max": 0.01, "rho_min": 0.0, "rho_max": 0.40}
QC_MIN_VALID = 243  # out of 270


def load_alt_cell(mode_label: str, N: int, seed: int) -> dict:
    """Load one (mode, seed) results.json from synced cluster data."""
    label = f"N{N}_seed{seed}"
    path = RESULTS_ROOT / label / "results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_corpus_cell(train_tau: str, seed: int) -> dict:
    path = CORPUS_ROOT / f"mjx_ppo_{train_tau}_seed{seed}" / "results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def get_psr(records: dict, body: str, eval_key: str = EVAL_KEY) -> float:
    """Return eval_key PSR; NaN if any PSR field in ANY eval block is NaN.

    Spec gate: 'no NaN in any reported PSR field'. We enforce this at cell
    level: if any of (per_step_reward, fitness, distance) in any of
    eval_tau020/eval_tau050/eval_neutral is non-finite, the cell is
    invalid (returned as NaN) regardless of which eval_key the caller
    asked for.
    """
    rec = records.get(body, {})
    for ek in EVAL_KEYS_CHECKED:
        blk = rec.get(ek)
        if not isinstance(blk, dict):
            return float("nan")
        for f in PSR_FIELDS:
            if f not in blk or not np.isfinite(blk[f]):
                return float("nan")
    v = rec.get(eval_key, {}).get("per_step_reward")
    return float(v) if np.isfinite(v) else float("nan")


def verify_metadata(records: dict, expected_N: int) -> list[str]:
    """Return list of bodies in `records` that fail metadata checks.

    Each record is checked against:
      - termination_mode == 'contact-duration'
      - contact_termination_steps == expected_N
      - num_envs/unroll_length/n_epochs/minibatch_size/timesteps match
        the frozen corpus hyperparameters

    A body with any mismatch is flagged; the analysis may still proceed,
    but the decision will be forced to 'drop' if any mismatches exist.
    """
    bad = []
    for body, rec in records.items():
        if rec.get("termination_mode") != "contact-duration":
            bad.append(f"{body}: termination_mode={rec.get('termination_mode')!r}")
            continue
        if rec.get("contact_termination_steps") != expected_N:
            bad.append(
                f"{body}: contact_termination_steps="
                f"{rec.get('contact_termination_steps')!r}, expected {expected_N}"
            )
            continue
        for k, v in EXPECTED_HPARAMS.items():
            if k == "termination_mode":
                continue
            if rec.get(k) != v:
                bad.append(f"{body}: {k}={rec.get(k)!r}, expected {v}")
                break
    return bad


def load_alt_long_df(panel_bodies: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Build long-form DataFrame: rows are (body, family, mode, seed, psr).

    Also returns a list of metadata violations across all loaded cells.
    """
    rows = []
    all_violations: list[str] = []
    for mode_label, N in N_MODES.items():
        for seed in SEEDS:
            recs = load_alt_cell(mode_label, N, seed)
            all_violations.extend(
                f"[mode={mode_label} N={N} seed={seed}] {v}"
                for v in verify_metadata(recs, N)
            )
            for entry in panel_bodies:
                body, family = entry["body"], entry["family"]
                psr = get_psr(recs, body)
                rows.append(
                    dict(
                        body=body,
                        family=family,
                        mode=mode_label,
                        seed=seed,
                        psr=psr,
                    )
                )
    return pd.DataFrame(rows), all_violations


def load_corpus_long_df(panel_bodies: list[dict]) -> pd.DataFrame:
    """Corpus records for the same 30 bodies, for comparison."""
    rows = []
    mode_to_tau = {"strict": "0p50", "episodic": "0p20", "reset_free": "0p00"}
    for entry in panel_bodies:
        body, family = entry["body"], entry["family"]
        for mode_label, tau in mode_to_tau.items():
            for seed in SEEDS:
                recs = load_corpus_cell(tau, seed)
                psr = get_psr(recs, body)
                rows.append(
                    dict(body=body, family=family, mode=mode_label, seed=seed, psr=psr)
                )
    return pd.DataFrame(rows)


def qc_gate(alt_df: pd.DataFrame) -> dict:
    total = len(alt_df)
    valid = int(np.isfinite(alt_df["psr"]).sum())
    nan_bodies = sorted(
        b for b, sub in alt_df.groupby("body") if not np.isfinite(sub["psr"]).all()
    )
    return dict(
        total=total,
        valid=valid,
        invalid=total - valid,
        qc_pass=valid >= QC_MIN_VALID,
        qc_threshold=QC_MIN_VALID,
        nan_bodies=nan_bodies,
    )


def twoway_anova(alt_df: pd.DataFrame) -> dict:
    """Blocked 2-way ANOVA: psr ~ C(mode) * C(family) + C(body), 30*3*3=270."""
    df = alt_df.dropna(subset=["psr"]).copy()
    if df.empty:
        return dict(error="no valid rows for ANOVA")
    model = ols("psr ~ C(mode) * C(family) + C(body)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    interaction_row = anova.loc["C(mode):C(family)"]
    return dict(
        interaction_f=float(interaction_row["F"]),
        interaction_p=float(interaction_row["PR(>F)"]),
        interaction_df_num=float(interaction_row["df"]),
        interaction_df_denom=float(anova.loc["Residual"]["df"]),
        anova_table=anova.reset_index()
        .rename(columns={"index": "source"})
        .to_dict(orient="records"),
    )


def per_body_delta(df: pd.DataFrame, mode_hi: str, mode_lo: str) -> dict[str, float]:
    """Seed-averaged PSR(mode_hi) - PSR(mode_lo) per body."""
    grouped = df.groupby(["body", "mode"])["psr"].mean().unstack("mode")
    if mode_hi not in grouped or mode_lo not in grouped:
        return {}
    delta = grouped[mode_hi] - grouped[mode_lo]
    return delta.dropna().to_dict()


def spearman_with_ci(x: np.ndarray, y: np.ndarray, B: int = 10000) -> dict:
    """Spearman rho with 10000-bootstrap 95% CI.

    Returns {error: ..., n: ...} without rho/CI when n < 2, or when the
    bootstrap produces zero finite samples (degenerate input).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return dict(error=f"n={n}, need >=2 for spearman", n=n)
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        r = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            boots.append(r)
    if not boots:
        return dict(
            error="bootstrap produced no finite rho samples",
            rho=rho,
            n=n,
        )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(rho=rho, ci_lo=float(lo), ci_hi=float(hi), n=n)


def direction_check(
    alt_df: pd.DataFrame, family: str, mode_hi: str, mode_lo: str, B: int = 10000
) -> dict:
    """Per-family: seed-averaged PSR(mode_hi) - PSR(mode_lo); check 95% CI < 0."""
    sub = alt_df[alt_df["family"] == family]
    per_body = sub.groupby(["body", "mode"])["psr"].mean().unstack("mode")
    if mode_hi not in per_body or mode_lo not in per_body:
        return dict(error=f"missing mode {mode_hi} or {mode_lo}")
    delta = (per_body[mode_hi] - per_body[mode_lo]).dropna().values
    if len(delta) == 0:
        return dict(error="no bodies with both modes")
    rng = np.random.default_rng(42)
    boots = rng.choice(delta, size=(B, len(delta)), replace=True).mean(axis=1)
    mean = float(np.mean(delta))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(
        family=family,
        n=int(len(delta)),
        mean_delta=mean,
        ci_lo=float(lo),
        ci_hi=float(hi),
        passes=(hi < 0.0),  # mode_hi < mode_lo means delta < 0
    )


def apply_gates(anova: dict, rank_corr: dict, dir_floor: dict, dir_vt: dict) -> dict:
    p = anova.get("interaction_p", 1.0)
    rho = rank_corr.get("rho", 0.0)

    if p < GATE_BODY_TEXT["p_max"] and rho >= GATE_BODY_TEXT["rho_min"]:
        decision = "body_text"
    elif p < GATE_MAIN_INCLUSION["p_max"] and rho >= GATE_MAIN_INCLUSION["rho_min"]:
        decision = "main_inclusion"
    elif (
        p < GATE_APPENDIX_ONLY["p_max"]
        and GATE_APPENDIX_ONLY["rho_min"] <= rho < GATE_APPENDIX_ONLY["rho_max"]
        and dir_floor.get("passes", False)
        and dir_vt.get("passes", False)
    ):
        decision = "appendix_only"
    else:
        decision = "drop"

    return dict(
        decision=decision,
        interaction_p=p,
        spearman_rho=rho,
        floor_direction_passes=dir_floor.get("passes", False),
        vt_direction_passes=dir_vt.get("passes", False),
    )


def main():
    if not PANEL_JSON.exists():
        print(f"ERROR: panel JSON not found at {PANEL_JSON}", file=sys.stderr)
        sys.exit(2)
    with open(PANEL_JSON) as f:
        panel = json.load(f)["panel"]

    alt, metadata_violations = load_alt_long_df(panel)
    corpus = load_corpus_long_df(panel)
    qc = qc_gate(alt)

    print(
        f"Panel: {len(panel)} bodies, 3 modes, 3 seeds = {len(alt)} cells; "
        f"valid={qc['valid']}/{qc['total']} (QC gate={qc['qc_threshold']})"
    )
    if qc["nan_bodies"]:
        print(
            f"Bodies with invalid cells: {qc['nan_bodies'][:5]}"
            f"{'...' if len(qc['nan_bodies']) > 5 else ''}"
        )

    if metadata_violations:
        print(
            f"\nMETADATA VIOLATIONS ({len(metadata_violations)}): data in "
            "results/alt_termination/ was not produced with the "
            "pre-registered configuration. Drop."
        )
        for v in metadata_violations[:10]:
            print(f"  {v}")

    if metadata_violations or not qc["qc_pass"]:
        reason = (
            f"metadata violations ({len(metadata_violations)})"
            if metadata_violations
            else f"QC gate: {qc['valid']}/{qc['total']} < {qc['qc_threshold']}"
        )
        out = dict(
            qc=qc,
            metadata_violations=metadata_violations,
            decision=dict(decision="drop", reason=reason),
        )
        path = Path("records/alt_termination_analysis.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nDrop alt-termination (reason: {reason}).")
        print(f"Wrote {path}")
        return

    anova = twoway_anova(alt)
    alt_delta = per_body_delta(alt, "episodic", "reset_free")
    corpus_delta = per_body_delta(corpus, "episodic", "reset_free")

    # Align bodies across both dicts
    shared = sorted(set(alt_delta) & set(corpus_delta))
    x = np.array([alt_delta[b] for b in shared])
    y = np.array([corpus_delta[b] for b in shared])
    rank_corr = spearman_with_ci(x, y)

    dir_floor = direction_check(alt, "floor", "strict", "episodic")
    dir_vt = direction_check(alt, "vt", "strict", "episodic")

    gates = apply_gates(anova, rank_corr, dir_floor, dir_vt)

    report = dict(
        qc=qc,
        anova=anova,
        rank_correlation_alt_vs_height=rank_corr,
        direction_check_floor=dir_floor,
        direction_check_vt=dir_vt,
        decision=gates,
        per_body_deltas=dict(
            contact_duration=alt_delta,
            height_based=corpus_delta,
        ),
    )

    def _json_default(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o).__name__}")

    out_path = Path("records/alt_termination_analysis.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    print(f"\nWrote {out_path}")

    # Short markdown summary
    md_path = Path("records/alt_termination_report.md")
    with open(md_path, "w") as f:
        f.write("# Alt-termination analysis\n\n")
        f.write(
            f"Panel: {len(panel)} bodies x 3 modes x 3 seeds = {qc['valid']}/{qc['total']} valid runs.\n\n"
        )
        f.write("## ANOVA (psr ~ C(mode) * C(family) + C(body))\n\n")
        f.write(
            f"- interaction F({anova['interaction_df_num']:.0f}, "
            f"{anova['interaction_df_denom']:.0f}) = "
            f"{anova['interaction_f']:.3f}, p = {anova['interaction_p']:.3g}\n"
        )
        f.write("\n## Rank correlation with height-based Delta_reset\n\n")
        f.write(
            f"- Spearman rho(contact-Delta, height-Delta) = "
            f"{rank_corr['rho']:.3f} [95% CI {rank_corr['ci_lo']:.3f}, "
            f"{rank_corr['ci_hi']:.3f}] (n={rank_corr['n']})\n"
        )
        f.write("\n## Direction checks\n\n")

        def _fmt_dir(label: str, dc: dict) -> str:
            if "error" in dc or "mean_delta" not in dc:
                return f"- {label}: error ({dc.get('error', 'missing fields')})\n"
            return (
                f"- {label}: mean Delta = {dc['mean_delta']:.3f} "
                f"[{dc['ci_lo']:.3f}, {dc['ci_hi']:.3f}] "
                f"(passes = {dc['passes']})\n"
            )

        f.write(_fmt_dir("Floor strict - episodic", dir_floor))
        f.write(_fmt_dir("VT strict - episodic", dir_vt))
        f.write("\n## Gate decision\n\n")
        f.write(f"**{gates['decision']}**\n\n")
        f.write(
            f"Source: interaction_p={gates['interaction_p']:.3g}, "
            f"rho={gates['spearman_rho']:.3f}, "
            f"floor-dir={gates['floor_direction_passes']}, "
            f"vt-dir={gates['vt_direction_passes']}.\n"
        )
    print(f"Wrote {md_path}")
    print(f"\nDecision: {gates['decision']}")


if __name__ == "__main__":
    main()
