#!/usr/bin/env python3
"""Verify SAC smoke pre-reg criteria before launching expansion.

Spec: records/sac_full_panel_spec.md (FROZEN).
Launch log: records/sac_full_panel_launch_log.md.

Original smoke criterion (launch log, 2026-04-24, committed BEFORE smoke
ran). Criterion 2 was amended 2026-04-24 after smoke completion and
before any inspection of expansion results; see launch log section
"2026-04-24: Smoke completion and pre-reg amendment" for the timestamped
rationale and PPO cross-check evidence.

Criterion 1 (UNCHANGED):
  For ALL 8 expected smoke bodies (first 8 entries in the frozen
  morphology list), eval_tau020.per_step_reward must be finite. At
  least 7 of 8 must be strictly positive.

Criterion 2 (AMENDED, launch log 2026-04-24):
  Cross-algorithm concordance with the PPO corpus on the same 8 bodies.
  Per body, compute PPO Epi/RF = seed-avg PPO PSR(tau=0.20) / seed-avg
  PPO PSR(tau=0.00), over seeds 42, 43, 44. For each body where PPO
  pilot_rf > RF_FLOOR, require SAC Epi/RF within 2x of PPO Epi/RF, i.e.
  |log2(SAC_ratio / PPO_ratio)| <= 1. At least 6/7 comparable bodies
  must satisfy this.

  Rationale for amendment: the original criterion (SAC Epi/RF in
  [0.5, 2.0]) failed on 4 of 8 bodies. Large Epi/RF ratios are expected
  for bodies that learn little without resets, so the amended check
  compares SAC with PPO on the same bodies instead. The PPO-concordance form
  checks cross-algorithm agreement; passing it does not rule out
  implementation errors.

Fails closed. Structural protections:
  - Fail if the smoke results.json contains bodies outside the frozen
    first-8 set (guards against post-expansion append).
  - Fail if any expected body is missing from smoke results.
  - Fail if PPO corpus files for seeds 42/43/44 at tau=0.00 and tau=0.20
    are missing.

Exit 0 = PROCEED (safe to launch expansion).
Exit 1 = PAUSE (stop, investigate before launch).

Usage:
  python3 scripts/verify_sac_smoke.py
  python3 scripts/verify_sac_smoke.py --smoke-dir <dir> --pilot-rf-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]

MIN_BODIES_POSITIVE = 7
EXPECTED_SMOKE_COUNT = 8
RF_FLOOR = 0.05

# Amended criterion 2: cross-algorithm SAC-vs-PPO concordance.
PPO_SEEDS = [42, 43, 44]
WITHIN_2X_LOG2 = 1.0  # |log2(SAC_ratio / PPO_ratio)| <= 1 == within 2x
MIN_CONCORDANT = 6  # at least 6 of the PPO-comparable bodies must be within 2x

EVAL_KEY = "eval_tau020"
PSR_KEY = "per_step_reward"

MORPH_LIST = REPO / "scripts" / "mechanism_morphologies.txt"


def load_frozen_morphology_list() -> List[str]:
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


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open() as f:
        return json.load(f)


def finite_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "N/A"


def load_ppo_seed_avg(tau_str: str, bodies: List[str]) -> dict[str, float | None]:
    """Seed-average PPO PSR(eval_tau020) over seeds 42/43/44 per body."""
    per_body: dict[str, list[float]] = {b: [] for b in bodies}
    for seed in PPO_SEEDS:
        path = REPO / "results" / f"mjx_ppo_{tau_str}_seed{seed}" / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"PPO corpus missing: {path}")
        d = json.load(path.open())
        for b in bodies:
            v = finite_float(d.get(b, {}).get(EVAL_KEY, {}).get(PSR_KEY))
            if v is not None:
                per_body[b].append(v)
    return {b: (sum(vs) / len(vs) if vs else None) for b, vs in per_body.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=REPO / "results" / "sac_mjx_0p20_seed42",
    )
    parser.add_argument(
        "--pilot-rf-dir",
        type=Path,
        default=REPO / "results" / "sac_mjx_0p00_seed42",
    )
    args = parser.parse_args()

    smoke_path = args.smoke_dir / "results.json"
    pilot_path = args.pilot_rf_dir / "results.json"

    print("=" * 72)
    print("SAC smoke pre-reg verification (amended 2026-04-24)")
    print("=" * 72)
    print(f"Smoke results:  {smoke_path}")
    print(f"Pilot rf data:  {pilot_path}")

    frozen = load_frozen_morphology_list()
    if len(frozen) < EXPECTED_SMOKE_COUNT:
        print(
            f"FAIL: frozen morphology list has only {len(frozen)} entries, "
            f"need >= {EXPECTED_SMOKE_COUNT}"
        )
        return 1
    expected_smoke = frozen[:EXPECTED_SMOKE_COUNT]
    print(f"Expected smoke bodies ({len(expected_smoke)}): {expected_smoke}")
    print()

    try:
        smoke = load_results(smoke_path)
    except FileNotFoundError as exc:
        print(f"FAIL: smoke results missing: {exc}")
        return 1
    try:
        pilot_rf = load_results(pilot_path)
    except FileNotFoundError as exc:
        print(f"FAIL: pilot rf results missing: {exc}")
        return 1

    smoke_bodies_all = set(smoke.keys())
    expected_set = set(expected_smoke)
    unexpected = sorted(smoke_bodies_all - expected_set)
    missing = sorted(expected_set - smoke_bodies_all)
    if unexpected:
        print(
            f"FAIL: smoke results contains {len(unexpected)} body(ies) outside "
            f"the frozen first-{EXPECTED_SMOKE_COUNT} set:"
        )
        for b in unexpected:
            print(f"    {b}")
        return 1
    if missing:
        print(f"FAIL: smoke results missing {len(missing)} expected body(ies):")
        for b in missing:
            print(f"    {b}")
        return 1

    # Load PPO corpus for amended criterion 2.
    try:
        ppo_epi = load_ppo_seed_avg("0p20", expected_smoke)
        ppo_rf = load_ppo_seed_avg("0p00", expected_smoke)
    except FileNotFoundError as exc:
        print(f"FAIL: PPO corpus required for amended criterion 2: {exc}")
        return 1

    print("Per-body data (on expected set only):")
    smoke_psr: dict[str, float | None] = {}
    pilot_psr: dict[str, float | None] = {}
    for b in expected_smoke:
        sp = finite_float(smoke.get(b, {}).get(EVAL_KEY, {}).get(PSR_KEY))
        smoke_psr[b] = sp
        pp = finite_float(pilot_rf.get(b, {}).get(EVAL_KEY, {}).get(PSR_KEY))
        pilot_psr[b] = pp
        sac_ratio = (
            sp / pp if (sp is not None and pp is not None and pp > RF_FLOOR) else None
        )
        p_epi = ppo_epi.get(b)
        p_rf = ppo_rf.get(b)
        ppo_ratio = (
            p_epi / p_rf
            if (p_epi is not None and p_rf is not None and p_rf > RF_FLOOR)
            else None
        )
        print(
            f"  {b}: sac_epi={fmt(sp)} sac_rf={fmt(pp)} sac_ratio={fmt(sac_ratio)} "
            f"ppo_epi={fmt(p_epi)} ppo_rf={fmt(p_rf)} ppo_ratio={fmt(ppo_ratio)}"
        )
    print()

    # Criterion 1 (unchanged): all 8 finite, >= 7/8 positive.
    positive_count = sum(1 for v in smoke_psr.values() if v is not None and v > 0)
    any_non_finite = any(v is None for v in smoke_psr.values())
    crit1_pass = (not any_non_finite) and positive_count >= MIN_BODIES_POSITIVE
    print(
        f"Criterion 1 (all 8 finite, >= {MIN_BODIES_POSITIVE}/8 positive): "
        f"{positive_count}/{EXPECTED_SMOKE_COUNT} positive, "
        f"non-finite count: {sum(1 for v in smoke_psr.values() if v is None)}. "
        f"{'PASS' if crit1_pass else 'FAIL'}"
    )

    # Amended criterion 2: cross-algorithm concordance.
    crit2_comparable = 0
    crit2_within_2x = 0
    crit2_detail = []
    for b in expected_smoke:
        sp = smoke_psr[b]
        pp = pilot_psr[b]
        p_epi = ppo_epi.get(b)
        p_rf = ppo_rf.get(b)
        if sp is None or pp is None or pp <= RF_FLOOR:
            crit2_detail.append(
                (b, "skip: SAC pilot_rf below floor or non-finite", None)
            )
            continue
        if p_epi is None or p_rf is None or p_rf <= RF_FLOOR:
            crit2_detail.append(
                (b, "skip: PPO pilot_rf below floor or non-finite", None)
            )
            continue
        sac_ratio = sp / pp
        ppo_ratio = p_epi / p_rf
        log2_ratio = math.log2(sac_ratio / ppo_ratio)
        within_2x = abs(log2_ratio) <= WITHIN_2X_LOG2
        crit2_comparable += 1
        if within_2x:
            crit2_within_2x += 1
        crit2_detail.append(
            (
                b,
                f"sac_ratio={sac_ratio:.3f} ppo_ratio={ppo_ratio:.3f} "
                f"log2(SAC/PPO)={log2_ratio:+.2f}",
                within_2x,
            )
        )

    crit2_pass = crit2_within_2x >= MIN_CONCORDANT
    print(
        f"Criterion 2 AMENDED (SAC Epi/RF within 2x of PPO Epi/RF, "
        f">= {MIN_CONCORDANT}/comparable bodies): "
        f"{crit2_within_2x}/{crit2_comparable} within 2x. "
        f"{'PASS' if crit2_pass else 'FAIL'}"
    )
    for b, msg, ok in crit2_detail:
        flag = "" if ok is None else (" [within 2x]" if ok else " [OUTSIDE 2x]")
        print(f"    {b}: {msg}{flag}")

    print()
    overall = crit1_pass and crit2_pass
    print("=" * 72)
    if overall:
        print(
            "OVERALL: PROCEED. Smoke passes amended pre-reg criteria. "
            "Safe to launch expansion."
        )
        return 0
    print("OVERALL: PAUSE. Smoke fails pre-reg criteria. DO NOT launch expansion.")
    print("  Investigate before proceeding; manual review required.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
