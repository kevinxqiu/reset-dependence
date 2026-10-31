#!/usr/bin/env python3
"""Render a deterministic policy rollout to MP4 + a mid-rollout still frame.

Loads a checkpoint produced by scripts/train_mjx.py or
scripts/menagerie_train_mjx.py with --save-checkpoints and runs the policy
for up to ROLLOUT_STEPS at the specified evaluation threshold, rendering
frames offscreen via mujoco.Renderer. The frames are stitched into an
MP4 (supplement) and one mid-rollout frame is saved as a PNG (main paper
montage).

The rollout is deterministic given (checkpoint path, threshold, seed):
the policy uses the MEAN action (no Gaussian sampling) so the same input
produces the same trajectory every time. This is the 'illustrative' mode
called out in records/visual_selection.md.

Requires imageio with the ffmpeg plugin:
    pip install imageio imageio-ffmpeg

Usage:
    MUJOCO_GL=osmesa python scripts/render_rollout.py \\
        --space unimal --morph floor-5506-14-13-01-15-43-28 \\
        --checkpoint path/to/ckpt.npz \\
        --threshold 0.20 --seed 42 \\
        --output figures/videos/floor_P25_episodic.mp4

    MUJOCO_GL=egl python scripts/render_rollout.py \\
        --space menagerie --robot unitree_go2 \\
        --checkpoint results/menagerie_panel/.../checkpoints/...npz \\
        --threshold 0.20 --seed 42 \\
        --output figures/videos/go2_episodic.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

ROLLOUT_STEPS = 500
RENDER_W, RENDER_H = 640, 480
FRAME_STRIDE = 4  # Render every Nth physics step to keep MP4s manageable
MIDFRAME_PROPORTION = 0.5  # Which rollout frame to save as still (0=first, 1=last)
FIXED_CAMERA_DISTANCE = 8.0
FIXED_CAMERA_ELEVATION = -15.0
FIXED_CAMERA_AZIMUTH = 90.0  # side view: looking along +x-axis travel direction


def load_params(ckpt: Path):
    """Load raw flat leaves from a .npz emitted by train_mjx.py's save path.

    train_mjx.py saves params via jax.tree_util.tree_leaves as keys p_0..p_N.
    We return the ordered list of leaves; the rollout code unflattens these
    against a freshly initialized network to recover the Flax pytree.
    """
    import numpy as _np

    raw = dict(_np.load(ckpt, allow_pickle=False))
    if all(k.startswith("p_") for k in raw):
        ordered = sorted(raw.keys(), key=lambda k: int(k.split("_", 1)[1]))
        return [jnp.asarray(raw[k]) for k in ordered]
    # Legacy format: "/"-separated tree keys. Return as a dict for back-compat.
    tree: dict = {}
    for k, v in raw.items():
        parts = k.split("/")
        node = tree
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = jnp.asarray(v)
    return tree


def run_rollout_unimal(
    morph: str, threshold: float, seed: int, params, fixed_camera: bool = False
):
    from train_mjx import ActorCritic, UnimalMJXEnv, default_unimal_xml_dir  # type: ignore

    xml_dir = default_unimal_xml_dir()
    env = UnimalMJXEnv(f"{xml_dir}/{morph}.xml", threshold=threshold)
    return _rollout(env, ActorCritic, params, seed, fixed_camera=fixed_camera)


def run_rollout_menagerie(
    robot: str, threshold: float, seed: int, params, fixed_camera: bool = False
):
    from menagerie_train_mjx import MenagerieMJXEnv  # type: ignore
    from train_mjx import ActorCritic  # type: ignore

    env = MenagerieMJXEnv(robot, threshold=threshold)
    return _rollout(env, ActorCritic, params, seed, fixed_camera=fixed_camera)


def _rollout(env, actor_critic_cls, params, seed: int, fixed_camera: bool = False):
    act_dim = env.action_size
    net = actor_critic_cls(action_dim=act_dim)

    rng_init = jax.random.PRNGKey(seed)
    sample_obs = jnp.zeros((env.observation_size,), dtype=jnp.float32)
    init_params = net.init(rng_init, sample_obs)
    if isinstance(params, list):
        _, treedef = jax.tree_util.tree_flatten(init_params)
        params = jax.tree_util.tree_unflatten(treedef, params)

    @jax.jit
    def mean_action(params, obs):
        mean, _, _ = net.apply(params, obs)
        return mean

    @jax.jit
    def step_jit(state, action):
        return env.step(state, action)

    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    frames: list[np.ndarray] = []
    qpos_full: list[np.ndarray] = []          # every env step
    action_trace: list[np.ndarray] = []       # every env step, clipped to match training
    renderer = mujoco.Renderer(env._mj_model, height=RENDER_H, width=RENDER_W)
    done_step = None
    try:
        data = mujoco.MjData(env._mj_model)
        initial_qpos = np.asarray(state.pipeline_state.qpos).copy()
        qpos_full.append(initial_qpos.copy())
        for step in range(ROLLOUT_STEPS):
            raw_action = mean_action(params, state.obs)
            # Clip to match training/eval reward computation: env.step() also
            # clips internally, but we clip here so the saved trace is the
            # effective action used to compute the reward.
            clipped = np.clip(np.asarray(raw_action), -1.0, 1.0)
            action_trace.append(clipped.copy())
            state = step_jit(state, raw_action)
            qpos_full.append(np.asarray(state.pipeline_state.qpos).copy())
            if bool(state.done) and done_step is None:
                done_step = step + 1
            if step % FRAME_STRIDE == 0:
                data.qpos[:] = np.asarray(state.pipeline_state.qpos)
                data.qvel[:] = np.asarray(state.pipeline_state.qvel)
                mujoco.mj_forward(env._mj_model, data)
                cam = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(cam)
                if fixed_camera:
                    cam.lookat[:] = np.array(
                        [
                            float(initial_qpos[0]),
                            float(initial_qpos[1]),
                            float(initial_qpos[2]) * 0.5,
                        ]
                    )
                    cam.distance = FIXED_CAMERA_DISTANCE
                    cam.azimuth = FIXED_CAMERA_AZIMUTH
                    cam.elevation = FIXED_CAMERA_ELEVATION
                else:
                    cam.lookat[:] = np.array(
                        [
                            float(data.qpos[0]),
                            float(data.qpos[1]),
                            float(data.qpos[2]) * 0.5,
                        ]
                    )
                    cam.distance = max(float(data.qpos[2]), 1.0) * 3.5
                    cam.azimuth = 135.0
                    cam.elevation = -20.0
                renderer.update_scene(data, camera=cam)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                frames.append(renderer.render())
    finally:
        renderer.close()

    qpos_arr = np.stack(qpos_full) if qpos_full else np.zeros((0, 0))
    if len(qpos_arr) > 1:
        xy_traj = qpos_arr[:, :2]
        start = xy_traj[0]
        end = xy_traj[-1]
        displacement = float(np.linalg.norm(end - start))
        path_len = float(np.sum(np.linalg.norm(np.diff(xy_traj, axis=0), axis=1)))
        print(
            f"    displacement={displacement:.2f}m  path_length={path_len:.2f}m  "
            f"done_step={done_step}  "
            f"(start x={start[0]:.2f} y={start[1]:.2f} -> end x={end[0]:.2f} y={end[1]:.2f})"
        )
    return {
        "frames": frames,
        "qpos_trace": qpos_arr,
        "action_trace": np.stack(action_trace) if action_trace else np.zeros((0, 0)),
        "done_step": done_step if done_step is not None else -1,
    }


def save_mp4(frames: list[np.ndarray], out: Path) -> None:
    try:
        import imageio
    except ImportError:
        print("ERROR: imageio not installed; cannot save MP4.", file=sys.stderr)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    # 30 fps in rendered time; at FRAME_STRIDE=4 this is ~7.5 fps of sim
    imageio.mimsave(str(out), frames, fps=30, codec="libx264")
    print(f"Wrote {out}")


def save_still(frames: list[np.ndarray], out: Path) -> None:
    if not frames:
        return
    import matplotlib.pyplot as plt

    idx = int(len(frames) * MIDFRAME_PROPORTION)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out, frames[idx])
    print(f"Wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", choices=["unimal", "menagerie"], required=True)
    parser.add_argument("--morph", help="UNIMAL morphology name (--space=unimal)")
    parser.add_argument("--robot", help="Menagerie robot name (--space=menagerie)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="Pin camera to initial body position (shows translation instead of posture).",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    params = load_params(args.checkpoint)
    if args.space == "unimal":
        if not args.morph:
            print("ERROR: --morph required for --space=unimal", file=sys.stderr)
            return 1
        result = run_rollout_unimal(
            args.morph,
            args.threshold,
            args.seed,
            params,
            fixed_camera=args.fixed_camera,
        )
    else:
        if not args.robot:
            print("ERROR: --robot required for --space=menagerie", file=sys.stderr)
            return 1
        result = run_rollout_menagerie(
            args.robot,
            args.threshold,
            args.seed,
            params,
            fixed_camera=args.fixed_camera,
        )

    frames = result["frames"]
    save_mp4(frames, args.output)
    still_path = args.output.with_suffix(".png")
    save_still(frames, still_path)
    traces_path = args.output.with_suffix(".traces.npz")
    np.savez(
        traces_path,
        qpos_trace=result["qpos_trace"],
        action_trace=result["action_trace"],
        done_step=np.int32(result["done_step"]),
        dt=np.float32(0.02),
        ctrl_weight=np.float32(0.001),
        frame_stride=np.int32(FRAME_STRIDE),
        deterministic=np.bool_(True),
        action_clipped=np.bool_(True),
    )
    print(f"Wrote {traces_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
