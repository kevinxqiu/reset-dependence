"""
Data validation suite for the 900-run PPO corpus.

Cited in paper/sections/appendix.tex as the 8-category pre-analysis check:
  (1) file-level integrity
  (2) numerical sanity (no NaN or Inf)
  (3) seed independence (no hard-correlated duplicate runs)
  (4) reset-free semantics (tau=0 runs carry the expected metadata)
  (5) family labeling (100 morphologies split 41 Floor / 37 VT / 22 MVT)
  (6) configuration provenance (recorded configuration labels)
  (7) cross-references against paper-facing summary statistics
  (8) stale-directory detection

Exits 0 on PASS, 1 on any failure. Writes a JSON report.

Run from repo root:
    python scripts/validate_data.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_JSON = REPO / "records/validate_data_report.json"

TAUS = ["0p00", "0p20", "0p50"]
SEEDS = [42, 43, 44]
EVAL_KEYS = ["eval_tau020", "eval_tau050", "eval_neutral"]
EVAL_METRICS = [
    "fitness",
    "fitness_std",
    "distance",
    "distance_std",
    "per_step_reward",
    "per_step_reward_std",
    "mean_ep_len",
    "ep_len_std",
    "fall_rate",
]

EXPECTED_RECORD_COUNT = 100
EXPECTED_FAMILY_COUNTS = {"floor": 41, "vt": 37, "mvt": 22}
EXPECTED_TIMESTEPS = 5_000_000
EXPECTED_BACKEND = "MJX-CustomPPO"
EXPECTED_ALGORITHM = "PPO"

# Tight tolerances make the check sensitive to accidental corpus or summary
# changes. Values below are the published family means at tau_eval=0.20 from
# paper/sections/results.tex (Table 1) and the RF/Epi ratios from the same
# table.
PAPER_FAMILY_MEANS = {
    "episodic": {"floor": 2.428, "vt": 2.304, "mvt": 2.651},
    "strict": {"floor": 1.837, "vt": 1.467, "mvt": 2.317},
    "reset_free": {"floor": 1.453, "vt": 1.999, "mvt": 2.088},
}
PAPER_RF_EPI_RATIOS = {"floor": 0.55, "vt": 0.84, "mvt": 0.72}
PAPER_STR_EPI_RATIOS = {"floor": 0.78, "vt": 0.66, "mvt": 0.91}
FAMILY_MEAN_ABS_TOL = 0.005
RATIO_ABS_TOL = 0.01

PROTOCOL_MAP = {"0p20": "episodic", "0p50": "strict", "0p00": "reset_free"}


def family_of(morph_name: str) -> str:
    prefix = morph_name.split("-", 1)[0].lower()
    if prefix in EXPECTED_FAMILY_COUNTS:
        return prefix
    raise ValueError(f"Unknown family prefix in {morph_name!r}")


def load_record_file(tau: str, seed: int) -> dict:
    path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as f:
        return json.load(f)


def check_file_integrity(report: dict) -> list[str]:
    failures = []
    for tau in TAUS:
        for seed in SEEDS:
            path = RESULTS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            try:
                with path.open() as f:
                    json.load(f)
            except json.JSONDecodeError as exc:
                failures.append(f"corrupt JSON: {path} ({exc})")
    report["file_integrity"] = {
        "expected_files": len(TAUS) * len(SEEDS),
        "failures": failures,
    }
    return failures


def check_numerical_sanity(report: dict, corpus: dict) -> list[str]:
    failures = []
    checked = 0
    for (tau, seed), records in corpus.items():
        for morph, rec in records.items():
            for eval_key in EVAL_KEYS:
                block = rec.get(eval_key)
                if block is None:
                    failures.append(f"{tau}/{seed}/{morph} missing {eval_key}")
                    continue
                for metric in EVAL_METRICS:
                    if metric not in block:
                        failures.append(
                            f"{tau}/{seed}/{morph}/{eval_key} missing {metric}"
                        )
                        continue
                    v = block[metric]
                    checked += 1
                    if v is None or (
                        isinstance(v, float) and (math.isnan(v) or math.isinf(v))
                    ):
                        failures.append(
                            f"{tau}/{seed}/{morph}/{eval_key}/{metric}={v!r}"
                        )
    report["numerical_sanity"] = {"values_checked": checked, "failures": failures}
    return failures


def check_seed_independence(report: dict, corpus: dict) -> list[str]:
    failures = []
    notes = []
    for tau in TAUS:
        for m1, m2 in [(42, 43), (42, 44), (43, 44)]:
            a = corpus[(tau, m1)]
            b = corpus[(tau, m2)]
            shared = sorted(set(a) & set(b))
            av = np.array([a[m]["eval_tau020"]["per_step_reward"] for m in shared])
            bv = np.array([b[m]["eval_tau020"]["per_step_reward"] for m in shared])
            identical = int(np.sum(np.isclose(av, bv, atol=1e-12)))
            pearson_r = (
                float(np.corrcoef(av, bv)[0, 1]) if len(av) > 1 else float("nan")
            )
            notes.append(
                {
                    "tau": tau,
                    "seeds": [m1, m2],
                    "identical_psr": identical,
                    "pearson_r_psr": pearson_r,
                }
            )
            # Two seeds sharing near-identical PSR across many morphs is a red flag.
            if identical > 5:
                failures.append(
                    f"seeds {m1}/{m2} tau={tau}: {identical} morphs have identical PSR"
                )
    report["seed_independence"] = {"pairs": notes, "failures": failures}
    return failures


def check_reset_free_semantics(report: dict, corpus: dict) -> list[str]:
    """
    At tau=0 training, there is no height-based early termination, so every
    evaluation under eval_neutral (which also has no termination) should yield
    mean_ep_len == 1000.0 exactly and fall_rate == 0.0 for every morphology.
    This is the one place where the corpus's reset-free-ness is directly
    observable from the metrics.
    """
    failures = []
    notes = {"eval_neutral_ep_lens": [], "eval_neutral_fall_rates": []}
    for seed in SEEDS:
        recs = corpus[("0p00", seed)]
        for morph, rec in recs.items():
            ep_len = rec["eval_neutral"]["mean_ep_len"]
            fall = rec["eval_neutral"]["fall_rate"]
            notes["eval_neutral_ep_lens"].append(ep_len)
            notes["eval_neutral_fall_rates"].append(fall)
            if not math.isclose(ep_len, 1000.0, abs_tol=1e-6):
                failures.append(
                    f"tau=0/seed={seed}/{morph} eval_neutral mean_ep_len={ep_len} (expected 1000)"
                )
            if fall != 0.0:
                failures.append(
                    f"tau=0/seed={seed}/{morph} eval_neutral fall_rate={fall} (expected 0)"
                )
    # Also: tau=0 training should have threshold field == 0.0
    for seed in SEEDS:
        for morph, rec in corpus[("0p00", seed)].items():
            t = rec.get("threshold")
            if t not in (0.0, 0):
                failures.append(
                    f"tau=0/seed={seed}/{morph} threshold field={t} (expected 0)"
                )
    report["reset_free_semantics"] = {
        "neutral_ep_len_summary": {
            "min": min(notes["eval_neutral_ep_lens"]),
            "max": max(notes["eval_neutral_ep_lens"]),
        },
        "neutral_fall_rate_summary": {
            "min": min(notes["eval_neutral_fall_rates"]),
            "max": max(notes["eval_neutral_fall_rates"]),
        },
        "failures": failures,
    }
    return failures


def check_family_labeling(report: dict, corpus: dict) -> list[str]:
    failures = []
    for (tau, seed), records in corpus.items():
        if len(records) != EXPECTED_RECORD_COUNT:
            failures.append(
                f"{tau}/{seed} has {len(records)} records (expected {EXPECTED_RECORD_COUNT})"
            )
        counts = Counter(family_of(m) for m in records)
        for fam, expected in EXPECTED_FAMILY_COUNTS.items():
            if counts.get(fam, 0) != expected:
                failures.append(
                    f"{tau}/{seed} family {fam}: {counts.get(fam, 0)} (expected {expected})"
                )
    report["family_labeling"] = {
        "expected": EXPECTED_FAMILY_COUNTS,
        "failures": failures,
    }
    return failures


def check_provenance(report: dict, corpus: dict) -> list[str]:
    failures = []
    required = ["morph_idx", "algorithm", "backend", "threshold", "seed", "timesteps"]
    for (tau, seed), records in corpus.items():
        for morph, rec in records.items():
            for key in required:
                if key not in rec:
                    failures.append(f"{tau}/{seed}/{morph} missing provenance: {key}")
            if rec.get("algorithm") != EXPECTED_ALGORITHM:
                failures.append(
                    f"{tau}/{seed}/{morph} algorithm={rec.get('algorithm')!r}"
                )
            if rec.get("backend") != EXPECTED_BACKEND:
                failures.append(f"{tau}/{seed}/{morph} backend={rec.get('backend')!r}")
            if rec.get("timesteps") != EXPECTED_TIMESTEPS:
                failures.append(
                    f"{tau}/{seed}/{morph} timesteps={rec.get('timesteps')!r}"
                )
            if rec.get("seed") != seed:
                failures.append(f"{tau}/{seed}/{morph} seed={rec.get('seed')!r}")
    report["provenance"] = {"failures": failures}
    return failures


def _family_means(corpus: dict) -> dict:
    out = {p: {} for p in ("episodic", "strict", "reset_free")}
    for tau in TAUS:
        protocol = PROTOCOL_MAP[tau]
        seeded = defaultdict(list)
        for seed in SEEDS:
            recs = corpus[(tau, seed)]
            for morph, rec in recs.items():
                fam = family_of(morph)
                seeded[(fam, morph)].append(rec["eval_tau020"]["per_step_reward"])
        per_morph = {}
        for (fam, morph), vals in seeded.items():
            per_morph[(fam, morph)] = float(np.mean(vals))
        for fam in EXPECTED_FAMILY_COUNTS:
            vals = [v for (f, _), v in per_morph.items() if f == fam]
            out[protocol][fam] = float(np.mean(vals))
    return out


def _family_ratios(corpus: dict) -> dict:
    per_morph_protocol = defaultdict(dict)
    for tau in TAUS:
        protocol = PROTOCOL_MAP[tau]
        for seed in SEEDS:
            for morph, rec in corpus[(tau, seed)].items():
                per_morph_protocol[morph].setdefault(protocol, [])
                per_morph_protocol[morph][protocol].append(
                    rec["eval_tau020"]["per_step_reward"]
                )
    rf_epi, str_epi = defaultdict(list), defaultdict(list)
    for morph, protos in per_morph_protocol.items():
        fam = family_of(morph)
        epi = float(np.mean(protos["episodic"]))
        rf = float(np.mean(protos["reset_free"]))
        strict = float(np.mean(protos["strict"]))
        if epi > 1e-6:
            rf_epi[fam].append(rf / epi)
            str_epi[fam].append(strict / epi)
    return {
        "rf_epi": {f: float(np.mean(v)) for f, v in rf_epi.items()},
        "str_epi": {f: float(np.mean(v)) for f, v in str_epi.items()},
    }


def check_paper_crossreferences(report: dict, corpus: dict) -> list[str]:
    failures = []
    means = _family_means(corpus)
    ratios = _family_ratios(corpus)
    for protocol, expected in PAPER_FAMILY_MEANS.items():
        for fam, val in expected.items():
            observed = means[protocol][fam]
            if abs(observed - val) > FAMILY_MEAN_ABS_TOL:
                failures.append(
                    f"family mean {protocol}/{fam}: observed={observed:.3f}, "
                    f"paper={val:.3f}, delta={observed - val:.3f}"
                )
    for fam, val in PAPER_RF_EPI_RATIOS.items():
        observed = ratios["rf_epi"][fam]
        if abs(observed - val) > RATIO_ABS_TOL:
            failures.append(f"RF/Epi {fam}: observed={observed:.3f}, paper={val:.3f}")
    for fam, val in PAPER_STR_EPI_RATIOS.items():
        observed = ratios["str_epi"][fam]
        if abs(observed - val) > RATIO_ABS_TOL:
            failures.append(f"Str/Epi {fam}: observed={observed:.3f}, paper={val:.3f}")
    report["paper_crossreferences"] = {
        "family_means": means,
        "ratios": ratios,
        "paper_family_means": PAPER_FAMILY_MEANS,
        "paper_rf_epi_ratios": PAPER_RF_EPI_RATIOS,
        "paper_str_epi_ratios": PAPER_STR_EPI_RATIOS,
        "failures": failures,
    }
    return failures


def check_stale_directories(report: dict) -> list[str]:
    """Directories that look like corpus results but do not match the
    canonical (tau, seed) shape, or that are listed but empty.
    """
    failures = []
    pattern = re.compile(r"^mjx_ppo_(0p\d+)_seed(\d+)$")
    canonical_pairs = {(tau, seed) for tau in TAUS for seed in SEEDS}
    non_canonical = []
    for entry in sorted(RESULTS.iterdir()):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m is None:
            continue
        tau, seed_s = m.group(1), int(m.group(2))
        key = (tau, seed_s)
        rj = entry / "results.json"
        if key not in canonical_pairs:
            # Known non-corpus (e.g. intermediate 0p10, 0p35 thresholds used
            # for the continuous threshold-sweep supplementary analysis).
            non_canonical.append(entry.name)
            continue
        if not rj.is_file():
            failures.append(f"canonical dir missing results.json: {entry}")
            continue
        if rj.stat().st_size == 0:
            failures.append(f"canonical results.json empty: {entry}")
    report["stale_directories"] = {
        "non_canonical_dirs": non_canonical,
        "failures": failures,
    }
    return failures


def main() -> int:
    if not RESULTS.is_dir():
        print(f"FAIL: {RESULTS} does not exist", file=sys.stderr)
        return 1

    report: dict = {"categories": {}, "summary": {}}
    all_failures: dict[str, list[str]] = {}

    integrity_fail = check_file_integrity(report["categories"])
    all_failures["file_integrity"] = integrity_fail
    if integrity_fail:
        # Further checks would cascade; bail with the current report.
        _write_report(report, all_failures)
        return 1

    corpus = {}
    for tau in TAUS:
        for seed in SEEDS:
            corpus[(tau, seed)] = load_record_file(tau, seed)

    all_failures["numerical_sanity"] = check_numerical_sanity(
        report["categories"], corpus
    )
    all_failures["seed_independence"] = check_seed_independence(
        report["categories"], corpus
    )
    all_failures["reset_free_semantics"] = check_reset_free_semantics(
        report["categories"], corpus
    )
    all_failures["family_labeling"] = check_family_labeling(
        report["categories"], corpus
    )
    all_failures["provenance"] = check_provenance(report["categories"], corpus)
    all_failures["paper_crossreferences"] = check_paper_crossreferences(
        report["categories"], corpus
    )
    all_failures["stale_directories"] = check_stale_directories(report["categories"])

    total_failures = sum(len(v) for v in all_failures.values())
    report["summary"] = {
        "total_failures": total_failures,
        "failures_by_category": {k: len(v) for k, v in all_failures.items()},
        "status": "PASS" if total_failures == 0 else "FAIL",
    }
    _write_report(report, all_failures)

    for cat, fails in all_failures.items():
        marker = "PASS" if not fails else f"FAIL ({len(fails)})"
        print(f"  {marker:<10s} {cat}")
        for f in fails[:5]:
            print(f"             - {f}")
        if len(fails) > 5:
            print(f"             ... ({len(fails) - 5} more)")
    print()
    print(f"Total failures: {total_failures}")
    print(f"Report: {OUT_JSON}")
    return 0 if total_failures == 0 else 1


def _write_report(report, all_failures):
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report["all_failures"] = all_failures
    with OUT_JSON.open("w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
