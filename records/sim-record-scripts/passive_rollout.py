#!/usr/bin/env python3
"""Zero-action and search rollouts on CPU MuJoCo for the Strict-cutoff bodies.

Loads each UNIMAL XML with the exact floor substitution of scripts/train_mjx.py:89-100,
resets to the nominal pose (no noise), applies zero control with frame_skip=4, and records
the first control step at which qpos[2] < 0.5*z0 (train_mjx's Strict rule is z < tau*z0),
the step of first floor contact, the lowest-point clearance at reset, and DERL's cutoff.
Then a random-shooting search over open-loop action sequences asks whether any sampled
controller keeps the torso above 0.5*z0 for 50 control steps (a feasibility probe, not proof).
"""
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "disable")
import mujoco
import numpy as np

XML = "assets/unimals_100/xml"
OUT = os.path.dirname(os.path.abspath(__file__))
FLOOR_COMMENT = ('<!-- <geom name="floor" type="plane" pos="0 0 0" '
                 'size="50 50 1" material="grid"/> -->')
FLOOR_ACTIVE = ('<geom name="floor" type="plane" pos="0 0 0" '
                'size="50 50 1" contype="1" conaffinity="1" '
                'rgba="0.9 0.9 0.9 1"/>')
FRAME_SKIP = 4


def load(name):
    s = open(f"{XML}/{name}.xml").read()
    assert FLOOR_COMMENT in s, name
    return mujoco.MjModel.from_xml_string(s.replace(FLOOR_COMMENT, FLOOR_ACTIVE))


def lowest_point(model, data):
    """Lowest z over all non-floor geoms (sphere/capsule exact bottoms)."""
    zmin = np.inf
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for g in range(model.ngeom):
        if g == floor:
            continue
        t = model.geom_type[g]
        pos = data.geom_xpos[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = pos[2] - model.geom_size[g, 0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            R = data.geom_xmat[g].reshape(3, 3)
            half = model.geom_size[g, 1]
            ends = [pos + R[:, 2] * half, pos - R[:, 2] * half]
            z = min(e[2] for e in ends) - model.geom_size[g, 0]
        else:
            z = pos[2] - np.max(model.geom_size[g])  # conservative
        zmin = min(zmin, z)
    return zmin


def floor_contact(model, data):
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == floor or c.geom2 == floor) and c.dist <= 0:
            return True
    return False


def rollout(model, actions, z0, steps):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    cross = None
    contact = None
    zs = [float(data.qpos[2])]
    for t in range(1, steps + 1):
        data.ctrl[:] = np.clip(actions[t - 1], -1, 1)
        mujoco.mj_step(model, data, nstep=FRAME_SKIP)
        z = float(data.qpos[2])
        zs.append(z)
        if contact is None and floor_contact(model, data):
            contact = t
        if cross is None and z < 0.5 * z0:
            cross = t
    return cross, contact, zs


def main():
    names = sys.argv[1:]
    rng = np.random.default_rng(0)
    rows = []
    for name in names:
        model = load(name)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        z0 = float(data.qpos[2])
        clearance = lowest_point(model, data)
        nu = model.nu
        cross0, contact0, zs0 = rollout(model, np.zeros((60, nu)), z0, 60)
        # random shooting: constant bang-bang, per-step bang-bang, smooth random
        best = (0, None)
        n_survive50 = 0
        n_try = 0
        for trial in range(int(os.environ.get("N_TRIALS", "1500"))):
            kind = trial % 3
            if kind == 0:
                a = np.tile(rng.choice([-1.0, 1.0], size=nu), (60, 1))
            elif kind == 1:
                a = rng.choice([-1.0, 1.0], size=(60, nu))
            else:
                a = np.clip(np.cumsum(rng.normal(0, 0.5, size=(60, nu)), axis=0), -1, 1)
            c, _, zs = rollout(model, a, z0, 60)
            n_try += 1
            surv = 60 if c is None else c
            if surv >= 50:
                n_survive50 += 1
            if surv > best[0]:
                best = (surv, float(min(zs[:surv + 1])))
        rows.append(dict(
            body=name, z0=z0, nu=int(nu), timestep=float(model.opt.timestep),
            control_dt=float(model.opt.timestep * FRAME_SKIP),
            lowest_point_at_reset=round(float(clearance), 4),
            torso_height_at_first_touch_zero_pose=round(z0 - clearance, 4),
            our_strict_cutoff=0.5 * z0, derl_cutoff=round(0.5 * (z0 - 0.2), 4),
            zero_action_first_cross_step=cross0, zero_action_first_floor_contact_step=contact0,
            zero_action_height_step50=round(zs0[50], 4), zero_action_min_height_60=round(min(zs0), 4),
            search_trials=n_try, search_best_survival_steps=best[0],
            search_n_survive_ge50=n_survive50,
        ))
        print(rows[-1], flush=True)
    tag = os.environ.get("TAG", "passive")
    json.dump(rows, open(f"{OUT}/{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
