#!/usr/bin/env python3
"""Run a deterministic policy rollout and save ONLY traces (no rendering).

This bypasses mujoco.Renderer entirely, which hits PyOpenGL loader failures
on several common-partition compute nodes. The temporal-gait 3x3 figure
(scripts/figures/render_temporal_gait.py) consumes traces.npz, so renderer-less
rollouts are sufficient for the figure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

ROLLOUT_STEPS = 500


def load_params(ckpt: Path):
    raw = dict(np.load(ckpt, allow_pickle=False))
    if all(k.startswith("p_") for k in raw):
        ordered = sorted(raw.keys(), key=lambda k: int(k.split("_", 1)[1]))
        return [jnp.asarray(raw[k]) for k in ordered]
    raise ValueError(f"unexpected checkpoint format: keys={list(raw.keys())[:5]}")


def run(morph: str, threshold: float, seed: int, ckpt: Path, out: Path) -> int:
    from train_mjx import ActorCritic, UnimalMJXEnv, default_unimal_xml_dir  # type: ignore

    xml_dir = default_unimal_xml_dir()
    env = UnimalMJXEnv(f"{xml_dir}/{morph}.xml", threshold=threshold)
    net = ActorCritic(action_dim=env.action_size)

    sample_obs = jnp.zeros((env.observation_size,), dtype=jnp.float32)
    init = net.init(jax.random.PRNGKey(seed), sample_obs)
    _, treedef = jax.tree_util.tree_flatten(init)
    params = jax.tree_util.tree_unflatten(treedef, load_params(ckpt))

    @jax.jit
    def mean_action(params, obs):
        mean, _, _ = net.apply(params, obs)
        return mean

    @jax.jit
    def step_jit(state, action):
        return env.step(state, action)

    state = env.reset(jax.random.PRNGKey(seed))
    qpos_full = [np.asarray(state.pipeline_state.qpos).copy()]
    action_trace = []
    done_step = None

    for step in range(ROLLOUT_STEPS):
        raw_action = mean_action(params, state.obs)
        clipped = np.clip(np.asarray(raw_action), -1.0, 1.0)
        action_trace.append(clipped.copy())
        state = step_jit(state, raw_action)
        qpos_full.append(np.asarray(state.pipeline_state.qpos).copy())
        if bool(state.done) and done_step is None:
            done_step = step + 1

    qpos_arr = np.stack(qpos_full)
    act_arr = np.stack(action_trace)
    xy = qpos_arr[:, :2]
    disp = float(np.linalg.norm(xy[-1] - xy[0]))
    path_len = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    print(
        f"    displacement={disp:.2f}m path_length={path_len:.2f}m done_step={done_step} "
        f"(start x={xy[0,0]:.2f} y={xy[0,1]:.2f} -> end x={xy[-1,0]:.2f} y={xy[-1,1]:.2f})"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        qpos_trace=qpos_arr,
        action_trace=act_arr,
        done_step=np.int32(done_step if done_step is not None else -1),
        dt=np.float32(0.02),
        ctrl_weight=np.float32(0.001),
        frame_stride=np.int32(4),
        deterministic=np.bool_(True),
        action_clipped=np.bool_(True),
    )
    print(f"Wrote {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--morph", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True,
                   help="output .traces.npz path")
    args = p.parse_args()
    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    return run(args.morph, args.threshold, args.seed, args.checkpoint, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
