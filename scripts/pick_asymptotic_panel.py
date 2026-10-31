"""
Pick the Phase 1b 18-body asymptotic panel.

Rule (design review advice 2026-04-15):
- Keep all 12 equivalence_pilot_bodies.json (Eq-A panel), so Phase 1b has
  stack-validated anchors.
- Add 6 extras chosen to span the (Delta_reset, PSR_tau000) 2D plane, not
  just Delta_reset quintiles. Since raw Delta_reset correlates strongly
  with baseline (rho=-0.74), Delta_reset-only stratification leaves
  off-diagonal regions empty. The 6 extras target the under-populated
  (high baseline, high Delta_reset) and (low baseline, low Delta_reset)
  quadrants so Phase 1b can answer "is the effect more than baseline?"
  directly.

Writes records/asymptotic_panel_bodies.json with 18 bodies, each
annotated with family, Delta_reset, baseline_PSR_tau000, and init_height.

Usage:
  python scripts/pick_asymptotic_panel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "results"
FEATURES_JSON = REPO / "results/morph_features.json"
EQA_JSON = REPO / "records/equivalence_pilot_bodies.json"
OUT_JSON = REPO / "records/asymptotic_panel_bodies.json"

SEEDS = (42, 43, 44)
TAU_BASE = "0p00"
TAU_REF = "0p20"
EVAL_KEY = "eval_tau020"
N_EXTRAS = 6


def family_of(morph: str) -> str:
    for fam in ("floor", "vt", "mvt"):
        if morph.startswith(fam + "-"):
            return fam
    return "other"


def load_stats():
    """Return {morph: {family, delta_reset, baseline_psr, init_height}}."""
    psrs = {}
    for tau in (TAU_BASE, TAU_REF):
        for seed in SEEDS:
            path = CORPUS_DIR / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with open(path) as f:
                d = json.load(f)
            for m, entry in d.items():
                if EVAL_KEY in entry:
                    psrs[(tau, seed, m)] = entry[EVAL_KEY]["per_step_reward"]
    all_morphs = {m for (_, _, m) in psrs}

    with open(FEATURES_JSON) as f:
        feats = json.load(f)

    out = {}
    for m in sorted(all_morphs):
        if not all((tau, s, m) in psrs for tau in (TAU_BASE, TAU_REF) for s in SEEDS):
            continue
        base = float(np.mean([psrs[(TAU_BASE, s, m)] for s in SEEDS]))
        ref = float(np.mean([psrs[(TAU_REF, s, m)] for s in SEEDS]))
        height = float(feats.get(m, {}).get("init_height", np.nan))
        out[m] = {
            "family": family_of(m),
            "baseline_psr": base,
            "psr_tau020": ref,
            "delta_reset": ref - base,
            "init_height": height if not np.isnan(height) else None,
        }
    return out


def pick_extras(stats_by_morph: dict, already: set[str], n_extras: int) -> list[str]:
    """Pick n_extras bodies spanning the (Delta_reset, baseline_PSR) plane.

    Strategy: z-score both dimensions on the full 100-body corpus, bin into
    2x2 quadrants of (low/high baseline, low/high delta_reset), count how
    many already-picked bodies are in each quadrant, and sample extras to
    balance occupancy. Within each under-populated quadrant, pick the body
    farthest from the origin (most extreme) so the panel has dynamic range.

    Family balance enforced weakly: prefer extras from the family that is
    MOST under-represented across (already + extras).
    """
    candidates = {m: s for m, s in stats_by_morph.items() if m not in already}
    all_bodies = list(stats_by_morph)

    base_all = np.array([stats_by_morph[m]["baseline_psr"] for m in all_bodies])
    dr_all = np.array([stats_by_morph[m]["delta_reset"] for m in all_bodies])
    base_z_all = (base_all - base_all.mean()) / base_all.std()
    dr_z_all = (dr_all - dr_all.mean()) / dr_all.std()

    body_z = {
        m: (base_z_all[i], dr_z_all[i]) for i, m in enumerate(all_bodies)
    }

    def quadrant(bz, drz):
        return (1 if bz >= 0 else 0, 1 if drz >= 0 else 0)

    def family_count(picked: set[str]):
        cnt = {"floor": 0, "vt": 0, "mvt": 0}
        for m in picked:
            fam = stats_by_morph[m]["family"]
            if fam in cnt:
                cnt[fam] += 1
        return cnt

    picked_extras = []
    picked_set = set(already)

    for _ in range(n_extras):
        # Count current quadrant occupancy
        q_count = {(i, j): 0 for i in (0, 1) for j in (0, 1)}
        for m in picked_set:
            bz, drz = body_z[m]
            q_count[quadrant(bz, drz)] += 1
        # Target the least-populated quadrant
        target_q = min(q_count, key=lambda k: q_count[k])

        fam_count = family_count(picked_set)
        # Least-populated family among already picked
        target_fam = min(fam_count, key=lambda k: fam_count[k])

        # Score candidates: must be in target quadrant; prefer target family;
        # within that, pick the body with maximum |z| distance from origin
        # (most extreme / informative).
        best = None
        best_score = -np.inf
        for m, _ in candidates.items():
            if m in picked_set:
                continue
            bz, drz = body_z[m]
            q = quadrant(bz, drz)
            if q != target_q:
                continue
            fam_bonus = 1.0 if stats_by_morph[m]["family"] == target_fam else 0.0
            score = np.hypot(bz, drz) + 0.5 * fam_bonus
            if score > best_score:
                best_score = score
                best = m
        if best is None:
            # Fallback: ANY quadrant, just pick the most-extreme remaining
            for m in candidates:
                if m in picked_set:
                    continue
                bz, drz = body_z[m]
                score = np.hypot(bz, drz)
                if score > best_score:
                    best_score = score
                    best = m
        if best is None:
            break  # no candidates left (shouldn't happen with 88 remaining)
        picked_extras.append(best)
        picked_set.add(best)

    return picked_extras


def main() -> int:
    print("=== Pick asymptotic 18-body panel ===")
    with open(EQA_JSON) as f:
        eqa = json.load(f)["bodies"]
    eqa_names = [b["name"] for b in eqa]
    print(f"Eq-A panel (kept as anchors): {len(eqa_names)}")

    stats_by_morph = load_stats()
    missing_eqa = [m for m in eqa_names if m not in stats_by_morph]
    if missing_eqa:
        raise RuntimeError(
            f"Eq-A bodies missing from corpus stats: {missing_eqa}"
        )

    extras = pick_extras(stats_by_morph, set(eqa_names), N_EXTRAS)
    print(f"New extras picked: {len(extras)}")
    for m in extras:
        s = stats_by_morph[m]
        print(
            f"  {m}  family={s['family']}  "
            f"baseline_psr={s['baseline_psr']:.3f}  "
            f"delta_reset={s['delta_reset']:+.3f}"
        )

    all_panel_names = eqa_names + extras
    panel = []
    for m in all_panel_names:
        s = stats_by_morph[m]
        panel.append(
            {
                "name": m,
                "family": s["family"],
                "delta_reset": round(s["delta_reset"], 6),
                "baseline_psr_tau000": round(s["baseline_psr"], 6),
                "psr_tau020": round(s["psr_tau020"], 6),
                "init_height": (
                    round(s["init_height"], 6) if s["init_height"] is not None else None
                ),
                "source": "eq_a" if m in eqa_names else "asymptotic_extra",
            }
        )

    # Sanity summary
    families = [b["family"] for b in panel]
    print()
    print("Panel family balance:")
    for f in ("floor", "vt", "mvt"):
        print(f"  {f}: {families.count(f)}")

    drs = np.array([b["delta_reset"] for b in panel])
    bs = np.array([b["baseline_psr_tau000"] for b in panel])
    print(f"Delta_reset range: [{drs.min():.3f}, {drs.max():.3f}], n_pos={int(np.sum(drs > 0))}")
    print(f"Baseline range: [{bs.min():.3f}, {bs.max():.3f}]")
    print(f"Within-panel Spearman rho(delta_reset, baseline): {stats.spearmanr(drs, bs).statistic:.3f}")

    payload = {
        "description": (
            "Phase 1b 50M asymptotic panel. 12 Eq-A bodies + 6 extras picked "
            "to span the (delta_reset, baseline_psr) 2D plane, addressing the "
            "baseline-skill confound (corpus rho=-0.74 between the two)."
        ),
        "estimand": (
            "Delta_reset(m) = PSR_train_tau020(m; eval_tau020) - "
            "PSR_train_tau000(m; eval_tau020)"
        ),
        "selection_rule_frozen_on": "2026-04-15",
        "n_bodies": len(panel),
        "n_taus": 3,
        "taus": ["0p00", "0p20", "0p50"],
        "seeds": [42, 43, 44],
        "steps_per_run": 50_000_000,
        "total_runs": len(panel) * 3 * 3,
        "bodies": panel,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
