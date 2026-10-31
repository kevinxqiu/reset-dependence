#!/usr/bin/env python3
"""GPU-accelerated SAC training with MuJoCo MJX + Brax.

Uses Brax's built-in SAC implementation with our UnimalMJXEnv.

Usage:
    python scripts/train_sac_mjx.py --threshold 0.50 --seed 42 --start_idx 0 --end_idx 24
"""

import os
import sys
import json
import time
import fcntl
import argparse

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from train_mjx import (
    UnimalMJXEnv,
    load_unimal_xml,
    MAX_STEPS,
    FRAME_SKIP,
    CTRL_COST_WEIGHT,
    N_EVAL,
    default_unimal_xml_dir,
)

from brax.training.agents.sac.train import train as sac_train

TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS = 128
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
DISCOUNTING = 0.99


def evaluate_policy(
    make_inference_fn, params, xml_path, threshold, seed, eval_conditions=None
):
    """Evaluate trained SAC policy under multiple termination conditions.

    Uses CPU MuJoCo with the PPO evaluator's reward and fall rules. It differs
    from the PPO evaluation in three details: deterministic actions, reset
    noise on positions only, and reset draws seeded by the training seed.
    """
    import mujoco

    if eval_conditions is None:
        eval_conditions = [
            ("eval_tau020", 0.2),
            ("eval_tau050", 0.5),
            ("eval_neutral", 0.0),
        ]

    mj_model = load_unimal_xml(xml_path)
    mj_data = mujoco.MjData(mj_model)
    init_z = float(mj_data.qpos[2])

    # Get deterministic inference function
    inference_fn = make_inference_fn(params, deterministic=True)

    eval_results = {}
    for eval_name, eval_tau in eval_conditions:
        z_min = init_z * eval_tau if eval_tau > 0 else -1e6
        z_max = max(init_z * 3.0, 2.0)

        ep_rewards = []
        ep_lengths = []
        ep_distances = []

        for ep in range(N_EVAL):
            mujoco.mj_resetData(mj_model, mj_data)
            rng = jax.random.PRNGKey(seed * 1000 + ep)

            # Small reset noise
            noise = np.random.RandomState(seed * 1000 + ep).uniform(
                -0.01, 0.01, mj_model.nq
            )
            noise[0] = noise[1] = 0
            mj_data.qpos[:] += noise
            mujoco.mj_forward(mj_model, mj_data)

            total_reward = 0
            for step in range(MAX_STEPS):
                obs = np.concatenate([mj_data.qpos[2:], mj_data.qvel]).astype(
                    np.float32
                )
                obs_jax = jnp.array(obs)

                # Use inference function (deterministic)
                rng, act_rng = jax.random.split(rng)
                action, _ = inference_fn(obs_jax, act_rng)
                action = np.array(action)
                action = np.clip(action, -1.0, 1.0)

                x_before = mj_data.qpos[0]
                mj_data.ctrl[:] = action
                for _ in range(FRAME_SKIP):
                    mujoco.mj_step(mj_model, mj_data)

                x_after = mj_data.qpos[0]
                dt = mj_model.opt.timestep * FRAME_SKIP
                fwd_vel = (x_after - x_before) / dt
                ctrl_cost = CTRL_COST_WEIGHT * np.sum(action**2)
                reward = fwd_vel - ctrl_cost
                total_reward += reward

                z = mj_data.qpos[2]
                if z < z_min or z > z_max:
                    break

            ep_rewards.append(total_reward)
            ep_lengths.append(step + 1)
            ep_distances.append(float(mj_data.qpos[0]))

        ep_rewards = np.array(ep_rewards)
        ep_lengths = np.array(ep_lengths)
        ep_distances = np.array(ep_distances)

        eval_results[eval_name] = {
            "fitness": float(np.mean(ep_rewards)),
            "fitness_std": float(np.std(ep_rewards)),
            "distance": float(np.mean(ep_distances)),
            "distance_std": float(np.std(ep_distances)),
            "per_step_reward": float(np.mean(ep_rewards / ep_lengths)),
            "per_step_reward_std": float(np.std(ep_rewards / ep_lengths)),
            "mean_ep_len": float(np.mean(ep_lengths)),
            "ep_len_std": float(np.std(ep_lengths)),
            "fall_rate": float(np.mean(ep_lengths < MAX_STEPS)),
        }

    return eval_results


def save_locked(path, key, value):
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        content = f.read()
        existing = json.loads(content) if content.strip() else {}
        existing[key] = value
        f.seek(0)
        f.truncate()
        json.dump(existing, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)


def load_locked(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        data = json.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument(
        "--xml_dir",
        default=default_unimal_xml_dir(),
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=24)
    parser.add_argument(
        "--morphology-list",
        default=None,
        help="Text file with one morphology name per line",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        threshold_str = f"{args.threshold:.2f}".replace(".", "p")
        args.out_dir = os.path.join(
            proj_dir, f"results/sac_mjx_{threshold_str}_seed{args.seed}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"MJX SAC Training: threshold={args.threshold}, seed={args.seed}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Hyperparams: num_envs={NUM_ENVS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"  Output: {args.out_dir}")

    xmls = sorted(f for f in os.listdir(args.xml_dir) if f.endswith(".xml"))

    if args.morphology_list:
        with open(args.morphology_list) as f:
            morph_names = [line.strip() for line in f if line.strip()]
        # Handle entries with or without .xml extension
        morph_xmls = [m if m.endswith(".xml") else f"{m}.xml" for m in morph_names]
        missing = [m for m in morph_xmls if m not in xmls]
        if missing:
            sys.exit(f"ERROR: {len(missing)} listed morphologies are not in {args.xml_dir}: {', '.join(missing)}")
        xmls = morph_xmls

    xmls = xmls[args.start_idx : args.end_idx]
    if not xmls:
        sys.exit(f"ERROR: no morphologies in the index range {args.start_idx}-{args.end_idx}")
    print(f"  Morphologies: {len(xmls)} ({args.start_idx}-{args.end_idx})")

    results_path = os.path.join(args.out_dir, "results.json")
    existing = load_locked(results_path)

    t0 = time.time()
    completed = 0
    failed_morphs = []
    total_requested = len(xmls)

    for i, xml_file in enumerate(xmls):
        morph_name = xml_file.replace(".xml", "")
        xml_path = os.path.join(args.xml_dir, xml_file)

        if morph_name in existing:
            print(f"[{i + 1}/{total_requested}] Skip {morph_name}")
            completed += 1
            continue

        print(
            f"[{i + 1}/{total_requested}] {morph_name} (SAC, threshold={args.threshold}, seed={args.seed})"
        )

        try:
            t1 = time.time()

            # handle_truncation=False: let Brax's EpisodeWrapper own truncation
            env = UnimalMJXEnv(
                xml_path, threshold=args.threshold, handle_truncation=False
            )
            print(f"    obs_dim={env.observation_size}, act_dim={env.action_size}")

            # Train with Brax SAC
            make_inference_fn, params, metrics = sac_train(
                environment=env,
                num_timesteps=args.timesteps,
                episode_length=MAX_STEPS,
                num_envs=NUM_ENVS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                discounting=DISCOUNTING,
                seed=args.seed,
                normalize_observations=True,
                num_evals=10,
                reward_scaling=1.0,
                min_replay_size=NUM_ENVS * 10,
                max_replay_size=100_000,
                grad_updates_per_step=1,
                deterministic_eval=True,
                progress_fn=lambda n, m: print(
                    f"    step={n}, reward={m.get('eval/episode_reward', 0):.1f}"
                ),
            )

            train_time = time.time() - t1
            print(f"    SAC training done in {train_time:.0f}s")

            # Evaluate under multiple conditions
            eval_results = evaluate_policy(
                make_inference_fn, params, xml_path, args.threshold, args.seed
            )

            result = {
                "morph_idx": i + args.start_idx,
                "algorithm": "SAC",
                "backend": "MJX-BraxSAC",
                "threshold": args.threshold,
                "seed": args.seed,
                "timesteps": args.timesteps,
                **eval_results,
                "train_time_s": train_time,
            }
            save_locked(results_path, morph_name, result)
            completed += 1

            std = eval_results["eval_tau020"]
            elapsed = time.time() - t0
            remaining = total_requested - i - 1
            eta = elapsed / completed * remaining if completed > 0 else 0
            print(
                f"  std: fit={std['fitness']:8.1f} dist={std['distance']:6.2f} "
                f"ep={std['mean_ep_len']:6.0f} fall={std['fall_rate']:.2f}"
            )
            print(f"  ({train_time:.0f}s train, {elapsed:.0f}s total, ~{eta:.0f}s ETA)")
            sys.stdout.flush()

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            failed_morphs.append(morph_name)
            continue

    total_in_results = len(load_locked(results_path))
    print(
        f"\nDone! {completed} trained, {len(failed_morphs)} failed, "
        f"{total_in_results} total in {time.time() - t0:.0f}s"
    )
    if failed_morphs:
        # Non-zero exit so SLURM marks the job as FAILED.
        print(f"Failed morphs: {failed_morphs}")
        sys.exit(1)
    if completed == 0 and total_in_results == 0:
        print("ERROR: zero morphs trained and no existing results, exit 2.")
        sys.exit(2)


if __name__ == "__main__":
    main()
