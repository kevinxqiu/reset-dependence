#!/usr/bin/env python3
"""Analyze the Menagerie 60-run cross-embodiment panel.

Claim scope is frozen in records/menagerie_task_spec.md: this panel is
a descriptive external-validity probe, NOT an inferential generalization
test. No cross-robot ANOVA. No cross-robot t-test on PSR magnitudes. Only
per-robot ratios with bootstrap CIs and a sign-pattern table.

Inputs:
  results/menagerie_panel/{robot}_tau{tau}_seed{seed}/results.json (60 files)

Outputs:
  records/menagerie_panel_report.md
  figures/fig_menagerie_panel.pdf  (per-robot RF/Epi + Str/Epi bars)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PANEL_DIR = REPO / "results/menagerie_panel"
OUT_MD = REPO / "records/menagerie_panel_report.md"
OUT_FIG = REPO / "figures/fig_menagerie_panel.pdf"

ROBOTS = ("unitree_go2", "anybotics_anymal_c", "toddlerbot_2xc", "unitree_g1")
TAUS = {"episodic": "0p20", "strict": "0p50", "reset_free": "0p00"}
SEEDS = (42, 43, 44, 45, 46)
N_BOOT = 10_000
# Ratios to the Standard reward are reported only when every reward is finite and every
# seed's Standard reward is at least this large (the paper's 0.05).
RATIO_FLOOR = 0.05


def load_panel() -> dict[str, dict[str, list[float]]]:
    """{robot: {protocol: [psr_seed1, psr_seed2, ...]}} (5 seeds each)."""
    out: dict[str, dict[str, list[float]]] = {r: defaultdict(list) for r in ROBOTS}
    missing = []
    for robot in ROBOTS:
        for protocol, tau_str in TAUS.items():
            for seed in SEEDS:
                path = PANEL_DIR / f"{robot}_tau{tau_str}_seed{seed}" / "results.json"
                if not path.is_file():
                    missing.append(path)
                    continue
                with path.open() as f:
                    data = json.load(f)
                # save_locked keys by robot name inside the per-run file
                rec = data.get(robot) or next(iter(data.values()))
                out[robot][protocol].append(
                    float(rec["eval_tau020"]["per_step_reward"])
                )
    if missing:
        print(f"WARNING: {len(missing)} files missing:")
        for p in missing[:10]:
            print(f"  - {p}")
    return out


def bootstrap_ci(
    values: np.ndarray, stat, n_boot=N_BOOT, seed=42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = np.array(values, dtype=float)
    resamples = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
    stats = stat(resamples)
    return (
        float(np.mean(stats)),
        float(np.percentile(stats, 2.5)),
        float(np.percentile(stats, 97.5)),
    )


def ratio_ci(num: np.ndarray, den: np.ndarray, n_boot=N_BOOT, seed=42):
    """Paired bootstrap on (num_i / den_i) for i in seeds."""
    rng = np.random.default_rng(seed)
    num = np.array(num, dtype=float)
    den = np.array(den, dtype=float)
    if len(num) != len(den):
        raise ValueError("paired bootstrap requires equal-length vectors")
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    num_r = num[idx]
    den_r = den[idx]
    ratios = np.mean(num_r / den_r, axis=1)
    return (
        float(np.mean(ratios)),
        float(np.percentile(ratios, 2.5)),
        float(np.percentile(ratios, 97.5)),
    )


def main() -> int:
    panel = load_panel()
    rows = []
    for robot in ROBOTS:
        protos = panel[robot]
        # Pair seeds by position, so every rule needs every seed (lists follow SEEDS).
        if any(len(protos[p]) != len(SEEDS) for p in ("episodic", "reset_free", "strict")):
            rows.append(
                {
                    "robot": robot,
                    "status": "incomplete",
                    "n_seeds_epi": len(protos["episodic"]),
                    "n_seeds_rf": len(protos["reset_free"]),
                    "n_seeds_str": len(protos["strict"]),
                }
            )
            continue
        epi = np.array(protos["episodic"])
        rf = np.array(protos["reset_free"])
        strict = np.array(protos["strict"])
        finite = all(np.all(np.isfinite(v)) for v in (epi, rf, strict))
        defined = finite and bool(np.all(epi >= RATIO_FLOOR))
        nan3 = (float("nan"),) * 3
        rf_epi_mean, rf_epi_lo, rf_epi_hi = ratio_ci(rf, epi) if defined else nan3
        str_epi_mean, str_epi_lo, str_epi_hi = ratio_ci(strict, epi) if defined else nan3
        rows.append(
            {
                "robot": robot,
                "status": "ok" if finite else "nan",
                "ratios_defined": defined,
                "n_seeds": len(epi),
                "mean_psr_epi": float(np.mean(epi)),
                "mean_psr_rf": float(np.mean(rf)),
                "mean_psr_str": float(np.mean(strict)),
                "std_psr_epi": float(np.std(epi, ddof=1)),
                "std_psr_rf": float(np.std(rf, ddof=1)),
                "std_psr_str": float(np.std(strict, ddof=1)),
                "rf_epi_mean": rf_epi_mean,
                "rf_epi_ci": (rf_epi_lo, rf_epi_hi),
                "str_epi_mean": str_epi_mean,
                "str_epi_ci": (str_epi_lo, str_epi_hi),
                "rf_hurts": rf_epi_hi < 1.0 if defined else None,
                "str_hurts": str_epi_hi < 1.0 if defined else None,
            }
        )
    _plot(rows, OUT_FIG)
    _report_md(rows, OUT_MD)

    for r in rows:
        if r.get("ratios_defined"):
            print(
                f"  {r['robot']:22s} n={r['n_seeds']} "
                f"RF/Epi={r['rf_epi_mean']:.2f} [{r['rf_epi_ci'][0]:.2f}, {r['rf_epi_ci'][1]:.2f}]  "
                f"Str/Epi={r['str_epi_mean']:.2f} [{r['str_epi_ci'][0]:.2f}, {r['str_epi_ci'][1]:.2f}]"
            )
        else:
            print(f"  {r['robot']:22s} {r['status'].upper()}, ratios n/a")
    return 0


def _plot(rows, out_path):
    ok = [r for r in rows if r.get("ratios_defined")]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ok))
    width = 0.35
    rf_means = [r["rf_epi_mean"] for r in ok]
    rf_err = [
        [r["rf_epi_mean"] - r["rf_epi_ci"][0] for r in ok],
        [r["rf_epi_ci"][1] - r["rf_epi_mean"] for r in ok],
    ]
    str_means = [r["str_epi_mean"] for r in ok]
    str_err = [
        [r["str_epi_mean"] - r["str_epi_ci"][0] for r in ok],
        [r["str_epi_ci"][1] - r["str_epi_mean"] for r in ok],
    ]
    ax.bar(
        x - width / 2,
        rf_means,
        width,
        yerr=rf_err,
        capsize=3,
        color="#D55E00",
        label="RF/Epi",
    )
    ax.bar(
        x + width / 2,
        str_means,
        width,
        yerr=str_err,
        capsize=3,
        color="#0072B2",
        label="Str/Epi",
    )
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["robot"].replace("_", "\n") for r in ok], fontsize=8)
    ax.set_ylabel("ratio of per-step reward (95% CI over 5 seeds)")
    ax.set_title("Menagerie cross-embodiment panel: protocol sensitivity")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _report_md(rows, out_path):
    lines = []
    lines.append("# Menagerie Cross-Embodiment Panel: Results")
    lines.append("")
    lines.append(
        "Descriptive external-validity probe per "
        "records/menagerie_task_spec.md. No cross-robot ANOVA. "
        "PSRs are NOT comparable across robots (scale depends on robot size); "
        "only within-robot ratios are meaningful."
    )
    lines.append("")
    lines.append("## Per-robot PSR (seed-averaged, 5 seeds)")
    lines.append("")
    lines.append(
        "| robot | n_seeds | PSR_Epi | PSR_Str | PSR_RF | RF/Epi [95% CI] | Str/Epi [95% CI] |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    def ratio(r, key):
        if not r["ratios_defined"]:
            return "n/a"
        return f"{r[key + '_mean']:.2f} [{r[key + '_ci'][0]:.2f}, {r[key + '_ci'][1]:.2f}]"

    for r in rows:
        if r["status"] in ("ok", "nan"):
            lines.append(
                f"| {r['robot']} | {r['n_seeds']} | "
                f"{r['mean_psr_epi']:.3f} | {r['mean_psr_str']:.3f} | {r['mean_psr_rf']:.3f} | "
                f"{ratio(r, 'rf_epi')} | {ratio(r, 'str_epi')} |"
            )
        else:
            lines.append(
                f"| {r['robot']} | INCOMPLETE "
                f"(epi={r['n_seeds_epi']}, rf={r['n_seeds_rf']}, str={r['n_seeds_str']}) "
                f"| - | - | - | - | - |"
            )
    lines.append("")
    lines.append("## Sign table")
    lines.append("")
    lines.append(
        "A robot is said to be hurt by a protocol if the 95% CI of the "
        "relevant ratio lies entirely below 1.0. Ratios are n/a when a reward is NaN "
        "or a seed's Standard reward is below 0.05."
    )
    lines.append("")
    lines.append("| robot | RF hurts | Str hurts |")
    lines.append("| --- | --- | --- |")
    def sign(v):
        return "n/a" if v is None else ("yes" if v else "no")

    for r in rows:
        if r["status"] in ("ok", "nan"):
            lines.append(f"| {r['robot']} | {sign(r['rf_hurts'])} | {sign(r['str_hurts'])} |")
    lines.append("")
    lines.append(
        f"Figure: `{OUT_FIG.relative_to(REPO)}` shows the RF/Epi and Str/Epi "
        "bars with 95% CIs per robot."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
