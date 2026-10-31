#!/usr/bin/env python3
"""Select the top-K morphologies under each training protocol.

Reads seed-averaged per-step reward at tau_eval=0.20 from the 900-run corpus
and picks the K highest-PSR bodies under each training tau (reset-free=0.00,
episodic=0.20, strict=0.50). Writes records/top_bodies_per_protocol.json.

Mirrors pick_visual_bodies.py's data-loading pattern.

Usage:
    python scripts/pick_top_bodies.py [--k 5]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "results"
OUT = REPO / "records/top_bodies_per_protocol.json"

PROTOCOLS = {
    "reset_free": "0p00",
    "episodic": "0p20",
    "strict": "0p50",
}
SEEDS = (42, 43, 44)


def load_psr_per_protocol() -> dict[str, dict[str, float]]:
    """Return {morph_name: {'reset_free': psr, 'episodic': psr, 'strict': psr}}.

    Asserts every (morph, protocol) cell has all seeds; fails loud if any
    corpus file is missing a body, rather than silently averaging partial data.
    """
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for proto, tau in PROTOCOLS.items():
        for seed in SEEDS:
            path = CORPUS / f"mjx_ppo_{tau}_seed{seed}" / "results.json"
            with path.open() as f:
                data = json.load(f)
            for morph, rec in data.items():
                psr = rec["eval_tau020"]["per_step_reward"]
                raw[morph][proto].append(float(psr))

    expected_seed_count = len(SEEDS)
    flat: dict[str, dict[str, float]] = {}
    for morph, d in raw.items():
        if set(d.keys()) != set(PROTOCOLS.keys()):
            missing = set(PROTOCOLS.keys()) - set(d.keys())
            raise ValueError(f"{morph} missing protocols {missing}")
        for proto, values in d.items():
            if len(values) != expected_seed_count:
                raise ValueError(
                    f"{morph} protocol={proto} has {len(values)} seed values, "
                    f"expected {expected_seed_count}"
                )
        flat[morph] = {p: float(np.mean(v)) for p, v in d.items()}
    return flat


def family_of(name: str) -> str:
    return name.split("-", 1)[0].lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    psr = load_psr_per_protocol()

    top_by_protocol: dict[str, list[dict]] = {}
    for proto in PROTOCOLS:
        # Deterministic tie-break: higher PSR first, then alphabetical by morph
        # name. Alphabetical beats dict-insertion-order for reproducibility
        # across Python runs.
        ranked = sorted(
            psr.items(),
            key=lambda kv: (-kv[1].get(proto, float("-inf")), kv[0]),
        )
        top = ranked[: args.k]
        top_by_protocol[proto] = [
            {
                "rank": i + 1,
                "morph": name,
                "family": family_of(name),
                "psr_reset_free": d.get("reset_free"),
                "psr_episodic": d.get("episodic"),
                "psr_strict": d.get("strict"),
            }
            for i, (name, d) in enumerate(top)
        ]

    # Overlap analysis across protocols.
    sets = {p: {entry["morph"] for entry in top_by_protocol[p]} for p in PROTOCOLS}
    all_top = set().union(*sets.values())
    overlaps: dict[str, dict] = {}
    for morph in sorted(all_top):
        present = [p for p in PROTOCOLS if morph in sets[p]]
        overlaps[morph] = {
            "protocols": present,
            "count": len(present),
            "family": family_of(morph),
        }

    family_counts: dict[str, dict[str, int]] = {}
    for proto in PROTOCOLS:
        fam: dict[str, int] = defaultdict(int)
        for entry in top_by_protocol[proto]:
            fam[entry["family"]] += 1
        family_counts[proto] = dict(fam)

    report = {
        "k": args.k,
        "tau_eval": 0.20,
        "seeds": list(SEEDS),
        "top_by_protocol": top_by_protocol,
        "overlaps": overlaps,
        "family_counts": family_counts,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {OUT}")
    print()
    for proto in PROTOCOLS:
        print(f"  {proto} (tau={PROTOCOLS[proto]}):")
        for entry in top_by_protocol[proto]:
            m = entry["morph"]
            psr_val = entry[f"psr_{proto}"]
            shared = overlaps[m]["count"]
            flag = f"  [in {shared} protocols]" if shared > 1 else ""
            print(f"    {entry['rank']}. {m}  PSR={psr_val:.3f}{flag}")
        fam_str = ", ".join(f"{f}:{c}" for f, c in family_counts[proto].items())
        print(f"    family counts: {fam_str}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
