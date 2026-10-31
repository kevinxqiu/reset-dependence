#!/usr/bin/env python3
"""Selection-regret numbers at common tau_eval.

The main-text Section 4.3 isolates training-protocol effects by fixing the
evaluation protocol at tau_eval=0.20. This script reproduces those common-eval
numbers and writes them to records/selection_regret_common_eval.json.

Supports a ``--eval-key`` CLI flag so the appendix variants at
``tau_eval in {0, 0.50}`` (Table tab:regret_robustness) can be produced from
the same entry point. With no flag the default ``eval_tau020`` run matches the
main text.

Usage:
    python scripts/compute_regret_common_eval.py                      # tau=0.20
    python scripts/compute_regret_common_eval.py --eval-key eval_neutral
    python scripts/compute_regret_common_eval.py --eval-key eval_tau050
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
TAUS = {"episodic": "0p20", "strict": "0p50", "reset_free": "0p00"}
SEEDS = (42, 43, 44)
BOOTSTRAP_B = 10_000
EVAL_KEY_TO_TAU = {
    "eval_tau020": 0.20,
    "eval_tau050": 0.50,
    "eval_neutral": 0.00,
}


def load_seed_avg(train_tau: str, eval_key: str) -> dict[str, float]:
    vals: defaultdict[str, list[float]] = defaultdict(list)
    for s in SEEDS:
        path = RESULTS / f"mjx_ppo_{train_tau}_seed{s}" / "results.json"
        with path.open() as f:
            data = json.load(f)
        for body, rec in data.items():
            psr = rec.get(eval_key, {}).get("per_step_reward")
            if psr is not None:
                vals[body].append(float(psr))
    return {body: float(np.mean(v)) for body, v in vals.items()}


def regret_table(
    epi: dict[str, float],
    strict: dict[str, float],
    rf: dict[str, float],
    ks: tuple[int, ...] = (5, 10, 20),
    n_bootstrap: int = BOOTSTRAP_B,
    seed: int = 42,
) -> dict[int, dict[str, dict[str, float]]]:
    bodies = sorted(set(epi) & set(strict) & set(rf))
    n = len(bodies)
    epi_v = np.array([epi[b] for b in bodies])
    strict_v = np.array([strict[b] for b in bodies])
    rf_v = np.array([rf[b] for b in bodies])
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_bootstrap, n))

    out: dict[int, dict[str, dict[str, float]]] = {}
    for k in ks:
        ep_idx = np.argsort(-epi_v)[:k]
        st_idx = np.argsort(-strict_v)[:k]
        rf_idx = np.argsort(-rf_v)[:k]
        strict_point = float(np.mean(strict_v[ep_idx]) / np.mean(strict_v[st_idx]))
        rf_point = float(np.mean(rf_v[ep_idx]) / np.mean(rf_v[rf_idx]))

        bs = np.empty(n_bootstrap)
        br = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = boot_idx[i]
            e = epi_v[idx]
            s = strict_v[idx]
            r = rf_v[idx]
            ep_ = np.argsort(-e)[:k]
            st_ = np.argsort(-s)[:k]
            rf_ = np.argsort(-r)[:k]
            bs[i] = np.mean(s[ep_]) / np.mean(s[st_])
            br[i] = np.mean(r[ep_]) / np.mean(r[rf_])
        s_lo, s_hi = np.percentile(bs, [2.5, 97.5])
        r_lo, r_hi = np.percentile(br, [2.5, 97.5])
        out[k] = {
            "epi_selected_strict_oracle": {
                "point": strict_point,
                "ci_lo": float(s_lo),
                "ci_hi": float(s_hi),
            },
            "epi_selected_rf_oracle": {
                "point": rf_point,
                "ci_lo": float(r_lo),
                "ci_hi": float(r_hi),
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-key",
        default="eval_tau020",
        choices=sorted(EVAL_KEY_TO_TAU),
        help="PSR evaluation key to read from each results.json "
        "(default: eval_tau020; appendix variants: eval_tau050, eval_neutral).",
    )
    args = parser.parse_args()

    eval_key = args.eval_key
    eval_tau = EVAL_KEY_TO_TAU[eval_key]

    epi = load_seed_avg(TAUS["episodic"], eval_key)
    strict = load_seed_avg(TAUS["strict"], eval_key)
    rf = load_seed_avg(TAUS["reset_free"], eval_key)
    print(
        f"Loaded {len(epi)} / {len(strict)} / {len(rf)} bodies "
        f"(epi/strict/rf) at {eval_key}."
    )

    table = regret_table(epi, strict, rf)
    print(
        f"\nSelection regret at common tau_eval={eval_tau:.2f} "
        f"(B={BOOTSTRAP_B} morphology bootstrap):"
    )
    for k, rec in table.items():
        s = rec["epi_selected_strict_oracle"]
        r = rec["epi_selected_rf_oracle"]
        print(
            f"  top-{k:>2}: EPI-sel vs STRICT-oracle = "
            f"{s['point'] * 100:.1f}% [{s['ci_lo'] * 100:.1f}, {s['ci_hi'] * 100:.1f}] ; "
            f"EPI-sel vs RF-oracle = "
            f"{r['point'] * 100:.1f}% [{r['ci_lo'] * 100:.1f}, {r['ci_hi'] * 100:.1f}]"
        )

    out_dir = REPO / "records"
    out_dir.mkdir(exist_ok=True)
    out = {
        "common_eval_tau": eval_tau,
        "eval_key": eval_key,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": 42,
        "n_bodies": len(set(epi) & set(strict) & set(rf)),
        "table": {str(k): v for k, v in table.items()},
    }
    suffix = "" if eval_key == "eval_tau020" else f"_{eval_key}"
    out_path = out_dir / f"selection_regret_common_eval{suffix}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
