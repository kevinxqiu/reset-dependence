#!/usr/bin/env python3
r"""Audit control-effort metrics from corpus mean-level records.

The training reward is $r_t = v_x(t) - 0.001 \|a_t\|^2$, so per episode
$\text{fitness} = \text{distance}/dt - 0.001 \sum_t \|a_t\|^2$
with $dt = \text{opt.timestep} \cdot \text{FRAME\_SKIP} = 0.005 \cdot 4 = 0.02$ s.
Given episode-mean fitness and distance we recover
$\widehat{\sum \|a\|^2}_{\text{per episode}} = (\text{distance}/dt - \text{fitness})/0.001$
(an unbiased estimate of the per-episode mean of $\sum \|a\|^2$).

From this we report two complementary views:

1. Per-step action-$L^2$:  $\widehat{\sum\|a\|^2}/\text{mean\_ep\_len}$.
   Physical bound: per step $\|a\|^2 \leq n_u$ since actions are clipped to
   $[-1,1]$. This sanity-checks the derivation and gives a distance-free
   measure of "how hard the policy pushes the actuators."

2. Control-effort cost of transport (C-COT): $\widehat{\sum\|a\|^2}/\text{distance}$.
   This is action-$L^2$ per metre of forward travel. It is NOT torque, work,
   or physical energy. It is sensitive to near-zero distances: bodies that
   expend effort without travelling drive C-COT to extreme values. We
   therefore report C-COT at several minimum-distance thresholds (5\,m,
   10\,m) and flag the "includes non-walkers" row as a composite measure
   of failure rate and per-metre efficiency, not pure efficiency.

Outputs:
    records/energy_proxy.json  -- full per-body, per-protocol record
    figures/fig_energy_by_family.pdf  -- family-level bars for per-step
        action-L^2 and C-COT@walk (distance > 5 m)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

FRAME_SKIP = 4
PHYSICS_DT = 0.005
DT = PHYSICS_DT * FRAME_SKIP
FORWARD_WEIGHT = 1.0
CTRL_WEIGHT = 0.001

PROTOCOLS = {
    "reset_free": ("0p00", 0.00),
    "episodic":   ("0p20", 0.20),
    "strict":     ("0p50", 0.50),
}
SEEDS = [42, 43, 44]
EVAL_KEY = "eval_tau020"

FAMILY_ORDER = ("floor", "vt", "mvt")
FAMILY_COLORS = {"floor": "#ff7f0e", "vt": "#1f77b4", "mvt": "#2ca02c"}
PROTOCOL_LABELS = {
    "reset_free": r"Reset-free ($\tau{=}0$)",
    "episodic":   r"Standard ($\tau{=}0.20$)",
    "strict":     r"Strict ($\tau{=}0.50$)",
}

# C-COT is reported at this minimum displacement to isolate efficiency from
# failure-to-walk; sensitivity over {0.05, 1, 5, 10} m is reported in the
# console summary so downstream reviewers can see the dependence.
COT_MIN_DIST_M = 5.0
COT_THRESHOLDS_AUDIT = [0.05, 1.0, 5.0, 10.0]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def family_of(morph: str) -> str:
    if morph.startswith("floor"):
        return "floor"
    if morph.startswith("mvt"):
        return "mvt"
    if morph.startswith("vt"):
        return "vt"
    raise ValueError(morph)


def load_corpus() -> dict:
    out = {p: {} for p in PROTOCOLS}
    for protocol, (tau_label, _) in PROTOCOLS.items():
        for seed in SEEDS:
            path = REPO / f"results/mjx_ppo_{tau_label}_seed{seed}" / "results.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open() as f:
                data = json.load(f)
            for morph, rec in data.items():
                out[protocol].setdefault(morph, []).append(rec)
    return out


def per_seed_metrics(rec: dict) -> dict:
    ev = rec[EVAL_KEY]
    fit = float(ev["fitness"])
    dist = float(ev["distance"])
    ep_len = float(ev["mean_ep_len"])
    if ep_len <= 0:
        return {"valid": False}
    sum_vx = dist / DT
    sum_a2 = (sum_vx - fit) / CTRL_WEIGHT
    if not np.isfinite(sum_a2) or sum_a2 < 0:
        # Physically impossible; indicates dist<<0 or evaluation noise. Drop.
        return {"valid": False, "distance": dist, "fitness": fit, "ep_len": ep_len}
    per_step_a2 = sum_a2 / ep_len
    return {
        "valid": True,
        "distance": dist,
        "fitness": fit,
        "ep_len": ep_len,
        "sum_a2": sum_a2,
        "per_step_a2": per_step_a2,
        "psr": float(ev["per_step_reward"]),
        "fall_rate": float(ev["fall_rate"]),
    }


def cot_at_min_dist(seed_metrics: list[dict], min_dist: float) -> float | None:
    """Seed-average C-COT among seeds with distance > min_dist."""
    per_seed = []
    for m in seed_metrics:
        if not m.get("valid", False):
            continue
        if m["distance"] <= min_dist:
            continue
        per_seed.append(m["sum_a2"] / m["distance"])
    return float(np.mean(per_seed)) if per_seed else None


def per_step_a2_seedavg(seed_metrics: list[dict]) -> float | None:
    per_seed = [m["per_step_a2"] for m in seed_metrics if m.get("valid", False)]
    return float(np.mean(per_seed)) if per_seed else None


def load_nu_map() -> dict:
    """Look up action dim n_u per morphology from the XML files."""
    import os
    import sys

    import mujoco

    xml_dir = Path(os.environ.get("RFC_UNIMAL_XML_DIR") or REPO / "assets/unimals_100/xml")
    if not xml_dir.is_dir():
        sys.exit(f"{xml_dir} not found; fetch the body files with scripts/fetch_unimal_assets.py")
    return {p.stem: int(mujoco.MjModel.from_xml_path(str(p)).nu) for p in sorted(xml_dir.glob("*.xml"))}


def main() -> int:
    corpus = load_corpus()
    nu_map = load_nu_map()

    records = []
    for protocol, by_morph in corpus.items():
        for morph, seeds in by_morph.items():
            seed_m = [per_seed_metrics(r) for r in seeds]
            ps = per_step_a2_seedavg(seed_m)
            nu = nu_map.get(morph, None)
            rec = {
                "morph": morph,
                "family": family_of(morph),
                "protocol": protocol,
                "n_u": nu,
                "per_step_a2": ps,
                "per_step_a2_per_nu": (ps / nu) if (ps is not None and nu) else None,
                "psr": float(np.mean([m["psr"] for m in seed_m if m.get("valid", False)]))
                    if any(m.get("valid", False) for m in seed_m) else None,
                "fall_rate": float(np.mean([m["fall_rate"] for m in seed_m if m.get("valid", False)]))
                    if any(m.get("valid", False) for m in seed_m) else None,
                "distance": float(np.mean([m["distance"] for m in seed_m if m.get("valid", False)]))
                    if any(m.get("valid", False) for m in seed_m) else None,
                "cot_any_dist": cot_at_min_dist(seed_m, 0.05),
                "cot_walks_1m": cot_at_min_dist(seed_m, 1.0),
                "cot_walks_5m": cot_at_min_dist(seed_m, 5.0),
                "cot_walks_10m": cot_at_min_dist(seed_m, 10.0),
            }
            records.append(rec)

    out_json = REPO / "records/energy_proxy.json"
    with out_json.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {out_json} ({len(records)} body-protocol records)")

    def family_cell_stats(records, metric_key):
        """Return dict[family][protocol] -> list[values] averaging over bodies."""
        out = {fam: {p: [] for p in PROTOCOLS} for fam in FAMILY_ORDER}
        for r in records:
            v = r.get(metric_key)
            if v is None or not np.isfinite(v):
                continue
            out[r["family"]][r["protocol"]].append(v)
        return out

    print("\n[1] Per-step ||a||^2 by family x protocol (physical bound: <= n_u):")
    print(f"{'Family':8s}" + ''.join(f'{p:>14s}' for p in PROTOCOLS))
    ps_stats = family_cell_stats(records, "per_step_a2")
    for fam in FAMILY_ORDER:
        row = [np.mean(ps_stats[fam][p]) if ps_stats[fam][p] else np.nan for p in PROTOCOLS]
        print(f'{fam:8s}' + ''.join(f'{v:>14.2f}' for v in row))

    print("\n[1b] Per-step ||a||^2 / n_u by family x protocol (near 1 means saturation):")
    print(f"{'Family':8s}" + ''.join(f'{p:>14s}' for p in PROTOCOLS))
    ps_norm_stats = family_cell_stats(records, "per_step_a2_per_nu")
    for fam in FAMILY_ORDER:
        row = [np.mean(ps_norm_stats[fam][p]) if ps_norm_stats[fam][p] else np.nan for p in PROTOCOLS]
        print(f'{fam:8s}' + ''.join(f'{v:>14.3f}' for v in row))

    print("\n[2] C-COT (action-L^2 / metre) at multiple min-distance thresholds:")
    for thr, key in zip(
        [0.05, 1.0, 5.0, 10.0],
        ["cot_any_dist", "cot_walks_1m", "cot_walks_5m", "cot_walks_10m"],
    ):
        cot_stats = family_cell_stats(records, key)
        print(f"\n  Min displacement > {thr} m:")
        print(f"  {'Family':8s}" + ''.join(f'{p:>14s}' for p in PROTOCOLS) + '    (n_bodies)')
        for fam in FAMILY_ORDER:
            ns = [len(cot_stats[fam][p]) for p in PROTOCOLS]
            row = [np.mean(cot_stats[fam][p]) if cot_stats[fam][p] else np.nan for p in PROTOCOLS]
            print(f'  {fam:8s}' + ''.join(f'{v:>14.1f}' for v in row) + f"    {ns}")

    # Figure: two-panel. Panel A = per-step ||a||^2 (truly flat). Panel B =
    # C-COT@walks-5m (fair efficiency comparison, filtered). Side-by-side.
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    width = 0.25
    x = np.arange(len(FAMILY_ORDER))
    proto_order = ("reset_free", "episodic", "strict")
    proto_colors = {
        "reset_free": "#7f3b8c",
        "episodic":   "#555555",
        "strict":     "#c45c1e",
    }

    ps_stats = family_cell_stats(records, "per_step_a2")
    cot_stats = family_cell_stats(records, "cot_walks_5m")
    for i, proto in enumerate(proto_order):
        means_a2 = [np.mean(ps_stats[fam][proto]) if ps_stats[fam][proto] else np.nan
                    for fam in FAMILY_ORDER]
        sems_a2 = [np.std(ps_stats[fam][proto])/np.sqrt(len(ps_stats[fam][proto]))
                   if len(ps_stats[fam][proto]) > 1 else 0 for fam in FAMILY_ORDER]
        axA.bar(x + (i-1)*width, means_a2, width, color=proto_colors[proto],
                label=PROTOCOL_LABELS[proto], yerr=sems_a2, capsize=2)
        means_cot = [np.mean(cot_stats[fam][proto]) if cot_stats[fam][proto] else np.nan
                     for fam in FAMILY_ORDER]
        sems_cot = [np.std(cot_stats[fam][proto])/np.sqrt(len(cot_stats[fam][proto]))
                    if len(cot_stats[fam][proto]) > 1 else 0 for fam in FAMILY_ORDER]
        axB.bar(x + (i-1)*width, means_cot, width, color=proto_colors[proto],
                yerr=sems_cot, capsize=2)

    axA.set_xticks(x); axA.set_xticklabels([f.upper() if f!="floor" else "Floor"
                                            for f in FAMILY_ORDER])
    axA.set_ylabel(r"Per-step $\|a_t\|^2$")
    axA.set_title("(a) Per-step control magnitude", pad=4)
    axA.legend(loc="upper right", fontsize=7, frameon=False, ncol=1)
    for sp in ("top", "right"): axA.spines[sp].set_visible(False)

    axB.set_xticks(x); axB.set_xticklabels([f.upper() if f!="floor" else "Floor"
                                            for f in FAMILY_ORDER])
    axB.set_ylabel(r"C-COT (action-$L^2$ per metre)")
    axB.set_title(f"(b) C-COT among walkers (dist $>$ {int(COT_MIN_DIST_M)} m)", pad=4)
    for sp in ("top", "right"): axB.spines[sp].set_visible(False)
    axB.grid(axis="y", linestyle=":", alpha=0.3)

    fig.tight_layout()
    out_pdf = REPO / "figures/fig_energy_by_family.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"\nWrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
