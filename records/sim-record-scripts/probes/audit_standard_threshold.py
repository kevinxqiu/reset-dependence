#!/usr/bin/env python3
"""Audit whether each training fall threshold can fire for each corpus body.

The fall clause ends an episode when the torso height qpos[2] drops below
tau * z0 (scripts/train_mjx.py). Every UNIMAL torso is a sphere centred on
the free-joint origin, and the floor is the plane z = 0, so qpos[2] cannot
fall far below the torso radius r. When tau * z0 < r the clause is inert and
that training rule is the same condition as Reset-free.

Three independent checks per body:
  1. geometry: tau * z0 against r, for every swept tau;
  2. physics probe: the lowest torso height reached in CPU MuJoCo under
     random actions, upside-down landings and hard drops;
  3. recorded data: (a) corpus evaluation, where eval_tau020 must equal
     eval_neutral exactly whenever the 0.20 clause never fires; (b) training
     done counts from the no-boundary tau=0.20 factorial arm, where no
     truncation exists, so any done is a fall (or the upper height bound).

Usage:
    python scripts/audit_standard_threshold.py [--results DIR] [--xml-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TAUS = (0.10, 0.20, 0.35, 0.50)
TRAIN_TAUS = ("0p00", "0p10", "0p20", "0p35", "0p50")
SEEDS = (42, 43, 44)
FRAME_SKIP = 4


def load_model(xml_path: str) -> mujoco.MjModel:
    """Same floor substitution as train_mjx.load_unimal_xml."""
    with open(xml_path) as f:
        xml_str = f.read()
    xml_str = xml_str.replace(
        '<!-- <geom name="floor" type="plane" pos="0 0 0" '
        'size="50 50 1" material="grid"/> -->',
        '<geom name="floor" type="plane" pos="0 0 0" '
        'size="50 50 1" contype="1" conaffinity="1" '
        'rgba="0.9 0.9 0.9 1"/>',
    )
    return mujoco.MjModel.from_xml_string(xml_str)


def quat_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def physics_probe(m: mujoco.MjModel, z0: float, rng: np.random.Generator) -> dict:
    """Lowest torso height reached under three adversarial-ish probes."""
    d = mujoco.MjData(m)
    out = {}

    def run(steps: int, init) -> float:
        mujoco.mj_resetData(m, d)
        init(d)
        mujoco.mj_forward(m, d)
        zmin = float(d.qpos[2])
        for _ in range(steps):
            # bang-bang random actions, redrawn every 5 control steps
            if _ % 5 == 0:
                a = rng.choice([-1.0, 1.0], size=m.nu)
            d.ctrl[:] = a
            for _k in range(FRAME_SKIP):
                mujoco.mj_step(m, d)
                zmin = min(zmin, float(d.qpos[2]))
        return zmin

    def noisy_reset(dd):
        dd.qpos[2:] += rng.uniform(-0.1, 0.1, m.nq - 2)
        dd.qpos[3:7] /= np.linalg.norm(dd.qpos[3:7])

    def flipped(height):
        def f(dd):
            dd.qpos[2] = height
            ax = rng.normal(size=3)
            ax[2] = 0.0
            dd.qpos[3:7] = quat_about(ax, np.pi * rng.uniform(0.5, 1.0))

        return f

    out["random_actions"] = min(run(500, noisy_reset) for _ in range(4))
    out["upside_down_landing"] = min(run(250, flipped(z0)) for _ in range(4))
    out["drop_from_1m"] = min(run(150, flipped(z0 + 1.0)) for _ in range(2))
    out["min"] = min(out.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", default=os.environ.get("RFC_RESULTS", REPO / "results")
    )
    ap.add_argument(
        "--xml-dir",
        default=os.environ.get(
            "RFC_UNIMAL_XML_DIR", REPO / "assets/unimals_100/xml"
        ),
    )
    ap.add_argument(
        "--out", default=REPO / "records/standard_threshold_audit.json"
    )
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()
    results = Path(args.results)
    rng = np.random.default_rng(0)

    corpus = {}
    for tt in TRAIN_TAUS:
        for s in SEEDS:
            with open(results / f"mjx_ppo_{tt}_seed{s}" / "results.json") as f:
                corpus[(tt, s)] = json.load(f)
    bodies = sorted(corpus[("0p20", 42)])
    assert len(bodies) == 100

    fact = {}
    for arm in ("tau0p20_5M", "tau0p50_5M"):
        for s in SEEDS:
            p = results / "factorial" / f"{arm}_seed{s}" / "results.json"
            if p.exists():
                for b, r in json.load(open(p)).items():
                    fact.setdefault(arm, {}).setdefault(b, {})[s] = r.get(
                        "implementation_reset_count"
                    )

    rows = {}
    for b in bodies:
        m = load_model(str(Path(args.xml_dir) / f"{b}.xml"))
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        z0 = float(d.qpos[2])
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso/0")
        assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE, b
        assert np.allclose(m.geom_pos[gid], 0.0), b
        r = float(m.geom_size[gid][0])
        row = {
            "family": b.split("-")[0],
            "z0": z0,
            "torso_radius": r,
            "threshold": {f"{t:.2f}": t * z0 for t in TAUS},
            "geometric_inert": {f"{t:.2f}": bool(t * z0 < r) for t in TAUS},
        }
        # Recorded evaluation: identical eval_tau020 and eval_neutral records
        # mean the 0.20 clause never fired in 20 x 1000 evaluation steps.
        ev = {}
        for tt in TRAIN_TAUS:
            for s in SEEDS:
                rec = corpus[(tt, s)][b]
                a, n = rec["eval_tau020"], rec["eval_neutral"]
                ev[f"{tt}_s{s}"] = {
                    "fall_rate_tau020": a["fall_rate"],
                    "identical_to_neutral": all(
                        a[k] == n[k] for k in ("fitness", "distance", "per_step_reward")
                    ),
                }
        row["eval_tau020_any_fall"] = any(
            v["fall_rate_tau020"] > 0 for v in ev.values()
        )
        row["eval_tau020_all_identical_to_neutral"] = all(
            v["identical_to_neutral"] for v in ev.values()
        )
        row["eval"] = ev
        for arm, recs in fact.items():
            if b in recs:
                row[f"factorial_{arm}_train_dones"] = recs[b]
        if not args.no_probe:
            row["probe_min_torso_z"] = physics_probe(m, z0, rng)
        rows[b] = row

    # Summary
    summary = {}
    for t in TAUS:
        key = f"{t:.2f}"
        inert = [b for b in bodies if rows[b]["geometric_inert"][key]]
        summary[key] = {
            "n_inert": len(inert),
            "by_family": {
                f: sum(rows[b]["family"] == f for b in inert)
                for f in ("floor", "vt", "mvt")
            },
        }
    inert20 = [b for b in bodies if rows[b]["geometric_inert"]["0.20"]]
    firing20 = [b for b in bodies if b not in inert20]
    summary["inert_0.20_bodies"] = inert20
    summary["inert_0.20_any_eval_fall"] = [
        b for b in inert20 if rows[b]["eval_tau020_any_fall"]
    ]
    summary["inert_0.20_not_identical_to_neutral"] = [
        b for b in inert20 if not rows[b]["eval_tau020_all_identical_to_neutral"]
    ]
    summary["firing_0.20_never_eval_fall"] = [
        b for b in firing20 if not rows[b]["eval_tau020_any_fall"]
    ]
    fa = fact.get("tau0p20_5M", {})
    summary["factorial_tau0p20"] = {
        "inert_bodies": {b: fa[b] for b in inert20 if b in fa},
        "firing_bodies": {b: fa[b] for b in firing20 if b in fa},
    }
    if not args.no_probe:
        summary["probe_margin_0.20_inert"] = {
            b: rows[b]["probe_min_torso_z"]["min"] - rows[b]["threshold"]["0.20"]
            for b in inert20
        }
    out = {"summary": summary, "bodies": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    for t in TAUS:
        s = summary[f"{t:.2f}"]
        print(f"tau={t:.2f}: {s['n_inert']}/100 inert  {s['by_family']}")
    print(
        f"inert at 0.20 with any eval fall at tau_eval=0.20: {summary['inert_0.20_any_eval_fall']}"
    )
    print(
        "inert at 0.20 with eval_tau020 != eval_neutral: "
        f"{summary['inert_0.20_not_identical_to_neutral']}"
    )
    print(
        f"firing at 0.20 but no eval fall in any of 15 policies: {len(summary['firing_0.20_never_eval_fall'])}"
    )
    print(
        "factorial tau=0.20 no-boundary training dones (inert):",
        summary["factorial_tau0p20"]["inert_bodies"],
    )
    print(
        "factorial tau=0.20 no-boundary training dones (firing):",
        summary["factorial_tau0p20"]["firing_bodies"],
    )
    print(f"{'body':32s} fam   z0     0.2z0  r     probe_min  margin  RFevalfall")
    for b in bodies:
        rw = rows[b]
        pm = rw.get("probe_min_torso_z", {}).get("min", float("nan"))
        rf_fall = np.mean([rw["eval"][f"0p00_s{s}"]["fall_rate_tau020"] for s in SEEDS])
        flag = "INERT" if rw["geometric_inert"]["0.20"] else ""
        print(
            f"{b:32s} {rw['family']:5s} {rw['z0']:.3f}  {rw['threshold']['0.20']:.3f}  "
            f"{rw['torso_radius']:.3f} {pm:.3f}      {pm - rw['threshold']['0.20']:+.3f}  "
            f"{rf_fall:.2f} {flag}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
