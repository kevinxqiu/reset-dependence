#!/usr/bin/env python3
"""GPU-accelerated PPO training for MuJoCo Menagerie robots on MJX.

Cross-embodiment external-validity panel for the reset-free-codesign paper.
Adapted from scripts/train_mjx.py; reuses the same PPO core (same
hyperparameters, same loss, same network) but replaces the UNIMAL-specific
environment class with a Menagerie-specific one that handles:

- keyframe-based standing pose OR qpos0 fallback for robots without a
  shipped keyframe,
- per-robot root body plus forward axis plus termination height via a
  frozen task spec (see records/menagerie_task_spec.md),
- per-robot actuator ctrlrange (actions are normalized in [-1, 1] then
  rescaled into the actuator ctrlrange at step time).

The four robots (precommitted in records/menagerie_task_spec.md) are:
    - unitree_go2 (quadruped)
    - anybotics_anymal_c (quadruped)
    - toddlerbot_2xc (humanoid)
    - unitree_g1 (humanoid, qpos0 standing pose; no shipped keyframe)

Usage:
    python scripts/menagerie_train_mjx.py \
        --robot unitree_go2 --threshold 0.20 --seed 42 \
        --timesteps 5000000 \
        --num_envs 128 --unroll_length 16 --n_epochs 10 --minibatch_size 256

Results go to results/menagerie_panel/{robot}_tau{0pXX}_seed{N}/results.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_CACHE_DIR = os.path.expanduser("~/.cache/jax_compilation")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from brax.envs.base import PipelineEnv, State  # noqa: E402
from mujoco import mjx  # noqa: E402

# Reuse the PPO core from train_mjx.py so there is exactly one implementation
# of the policy network, loss, and training loop in the repo.
_REPO = Path(__file__).resolve().parents[1]
_TRAIN_MJX_PATH = _REPO / "scripts" / "train_mjx.py"
_spec = importlib.util.spec_from_file_location("train_mjx", _TRAIN_MJX_PATH)
_train_mjx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mjx)

ActorCritic = _train_mjx.ActorCritic
Transition = _train_mjx.Transition
make_train_fns = _train_mjx.make_train_fns
collect_provenance = _train_mjx.collect_provenance
load_locked = _train_mjx.load_locked
save_locked = _train_mjx.save_locked

# Constants that MUST match the UNIMAL corpus so the cross-robot Str/RF and
# RF/Epi ratios are on the same methodological footing. These are frozen in
# records/menagerie_task_spec.md.
MAX_STEPS = 1000
FRAME_SKIP = 4
CTRL_COST_WEIGHT = 0.001
FORWARD_REWARD_WEIGHT = 1.0
RESET_NOISE_SCALE = 0.01
N_EVAL = 20
GAMMA = 0.99
GAE_LAMBDA = 0.95
LEARNING_RATE = 3e-4

# Corpus PPO hyperparameters (128 / 16 / 10 / 256). These differ from the
# train_mjx.py defaults (2048 / 32 / 4 / 4096) which are the Phase 1b
# asymptotic-gate high-throughput configuration. The Menagerie panel must use
# the corpus values so seed-averaged PSR is comparable to corpus conventions.
DEFAULT_NUM_ENVS = 128
DEFAULT_UNROLL_LENGTH = 16
DEFAULT_N_EPOCHS = 10
DEFAULT_MINIBATCH_SIZE = 256
DEFAULT_TIMESTEPS = 5_000_000


MENAGERIE_ROOT = Path(os.path.expanduser("~/mujoco_menagerie"))

# Frozen task spec, mirrored from records/menagerie_task_spec.md.
# Changing this dict must be accompanied by an amendment entry in the spec
# file and a note in the paper's limitations section.
ROBOT_SPECS: dict[str, dict] = {
    "unitree_go2": {
        "xml": "unitree_go2/go2_mjx.xml",
        "class_": "quadruped",
        "use_keyframe": True,
        "root_body": "base",
    },
    "anybotics_anymal_c": {
        "xml": "anybotics_anymal_c/anymal_c_mjx.xml",
        "class_": "quadruped",
        "use_keyframe": True,
        "root_body": "base",
    },
    "toddlerbot_2xc": {
        "xml": "toddlerbot_2xc/toddlerbot_2xc_mjx.xml",
        "class_": "humanoid",
        "use_keyframe": True,
        "root_body": "torso",
    },
    "unitree_g1": {
        "xml": "unitree_g1/g1_mjx.xml",
        "class_": "humanoid",
        # No shipped keyframe; qpos0 in the XML already encodes a standing
        # pose at pelvis z = 0.793. Verified by probing and MJX smoke test.
        # unitree_h1 was the original slot 4 but fails MJX 3.6.0 with
        # `NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_BOX) collisions`;
        # swap is recorded in records/menagerie_task_spec.md.
        "use_keyframe": False,
        "root_body": "pelvis",
    },
}


def load_menagerie_model(robot_name: str) -> tuple[mujoco.MjModel, dict]:
    """Load the raw MuJoCo model for a Menagerie robot and return the spec."""
    if robot_name not in ROBOT_SPECS:
        raise KeyError(
            f"Unknown robot: {robot_name!r}. Must be one of {list(ROBOT_SPECS)}."
        )
    spec = ROBOT_SPECS[robot_name]
    xml_path = MENAGERIE_ROOT / spec["xml"]
    if not xml_path.is_file():
        raise FileNotFoundError(f"Menagerie XML not found: {xml_path}")
    return mujoco.MjModel.from_xml_path(str(xml_path)), spec


class MenagerieMJXEnv(PipelineEnv):
    """Locomotion environment for a Menagerie robot on MJX.

    Mirrors UnimalMJXEnv semantics to keep cross-morphology-space comparisons
    methodologically clean:
    - Obs: concat(qpos[2:], qvel), i.e. drop global x, y but keep z, the
      orientation quaternion, and all joint positions.
    - Reward: v_x minus 0.001 times ||a_t||^2, where v_x is the per-frame
      finite difference of the root body's world-x position.
    - Termination: z < threshold * z_init OR z > 3 * z_init, where
      z_init is the standing-z of the keyframe-or-qpos0 pose.
    - frame_skip = 4.
    - Actions are assumed to be normalized in [-1, 1] and are rescaled to
      the actuator ctrlrange before physics stepping.
    """

    def __init__(self, robot_name: str, threshold: float, backend: str = "mjx"):
        mj_model, spec = load_menagerie_model(robot_name)
        mj_data = mujoco.MjData(mj_model)

        # Standing initial state: keyframe 0 if present, else qpos0 (the
        # value already baked into the XML, which for h1 is a standing pose).
        if spec["use_keyframe"]:
            if mj_model.nkey == 0:
                raise RuntimeError(
                    f"{robot_name}: spec asks for keyframe 0 but XML has 0 keyframes"
                )
            mj_data.qpos[:] = mj_model.key_qpos[0]
            if mj_model.nkey > 0 and mj_model.nv > 0:
                if mj_model.key_qvel.size >= mj_model.nv:
                    mj_data.qvel[:] = mj_model.key_qvel[0]
        mujoco.mj_forward(mj_model, mj_data)

        # Root body index (must exist and must have a free joint at qpos[0:7]).
        root_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, spec["root_body"]
        )
        if root_body_id < 0:
            raise RuntimeError(
                f"{robot_name}: root body {spec['root_body']!r} not found"
            )
        z_init_world = float(mj_data.xpos[root_body_id, 2])
        if z_init_world < 0.05:
            raise RuntimeError(
                f"{robot_name}: z_init_world={z_init_world:.3f} too small; "
                f"the selected initial pose is not a standing pose"
            )

        self._robot_name = robot_name
        self._robot_class = spec["class_"]
        self._root_body_id = root_body_id
        self._init_z_world = z_init_world

        # Floating-base convention: qpos[2] is the world z of the root body.
        self._init_z_qpos = float(mj_data.qpos[2])

        self._init_qpos = mj_data.qpos.copy()
        self._init_qvel = mj_data.qvel.copy()
        self._mj_model = mj_model

        # Termination bounds. Use qpos[2] for consistency with UnimalMJXEnv.
        if threshold <= 0:
            self._z_min = jnp.float32(-1e6)
        else:
            self._z_min = jnp.float32(self._init_z_qpos * threshold)
        self._z_max = jnp.float32(max(self._init_z_qpos * 3.0, 2.0))

        # Actuator ctrlrange: action in [-1, 1] maps to the per-actuator
        # [lo, hi]. Some Menagerie actuators are position-controlled, not
        # torque; ctrlrange is still the canonical normalization target.
        ctrl_lo = jnp.asarray(mj_model.actuator_ctrlrange[:, 0], dtype=jnp.float32)
        ctrl_hi = jnp.asarray(mj_model.actuator_ctrlrange[:, 1], dtype=jnp.float32)
        self._ctrl_center = 0.5 * (ctrl_lo + ctrl_hi)
        self._ctrl_halfspan = 0.5 * (ctrl_hi - ctrl_lo)

        # Observation dim: drop global x, y from qpos; keep qvel fully.
        self._obs_dim = (mj_model.nq - 2) + mj_model.nv

        self._dt = mj_model.opt.timestep * FRAME_SKIP

        sys = mjx.put_model(mj_model)
        super().__init__(sys=sys, n_frames=FRAME_SKIP, backend=backend)

    @property
    def observation_size(self):
        return self._obs_dim

    @property
    def action_size(self):
        return self.sys.nu

    def reset(self, rng):
        rng, rng_qpos, rng_qvel = jax.random.split(rng, 3)
        qpos = jnp.array(self._init_qpos)
        qvel = jnp.array(self._init_qvel)
        noise_qpos = jax.random.uniform(
            rng_qpos,
            shape=(self.sys.nq,),
            minval=-RESET_NOISE_SCALE,
            maxval=RESET_NOISE_SCALE,
        )
        noise_qpos = noise_qpos.at[0].set(0.0).at[1].set(0.0)
        noise_qvel = jax.random.uniform(
            rng_qvel,
            shape=(self.sys.nv,),
            minval=-RESET_NOISE_SCALE,
            maxval=RESET_NOISE_SCALE,
        )
        qpos = qpos + noise_qpos
        qvel = qvel + noise_qvel
        data = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(data)
        reward, done = jnp.zeros(2)
        metrics = {
            "x_position": data.qpos[0],
            "z_height": data.qpos[2],
            "forward_vel": jnp.float32(0),
            "reward": jnp.float32(0),
        }
        info = {"step": jnp.int32(0), "truncation": jnp.float32(0)}
        return State(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def step(self, state, action):
        action = jnp.clip(action, -1.0, 1.0)
        ctrl = self._ctrl_center + self._ctrl_halfspan * action
        x_before = state.pipeline_state.qpos[0]
        data = self.pipeline_step(state.pipeline_state, ctrl)
        x_after = data.qpos[0]
        forward_vel = (x_after - x_before) / self._dt
        forward_reward = FORWARD_REWARD_WEIGHT * forward_vel
        ctrl_cost = CTRL_COST_WEIGHT * jnp.sum(jnp.square(action))
        reward = forward_reward - ctrl_cost
        z = data.qpos[2]
        healthy = (z >= self._z_min) & (z <= self._z_max)
        done = 1.0 - healthy
        obs = self._get_obs(data)
        metrics = {
            "x_position": x_after,
            "z_height": z,
            "forward_vel": forward_vel,
            "reward": reward,
        }
        step_count = state.info["step"] + 1
        truncated = step_count >= MAX_STEPS
        done = jnp.where(truncated, 1.0, done)
        info = {
            **state.info,
            "step": step_count,
            "truncation": jnp.where(truncated, 1.0, 0.0),
        }
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _get_obs(self, data):
        return jnp.concatenate([data.qpos[2:], data.qvel])


# Training loop, mirrors train_mjx.py train_and_evaluate structurally.


def _evaluate(env, params, network, rng, n_eval=N_EVAL):
    """Stochastic evaluation matching train_mjx.py conventions."""

    @jax.jit
    def reset_one(rng):
        return env.reset(rng)

    @jax.jit
    def step_one(state, action):
        return env.step(state, action)

    @jax.jit
    def policy(params, obs, rng):
        mean, log_std, _ = network.apply(params, obs)
        std = jnp.exp(log_std)
        return mean + std * jax.random.normal(rng, mean.shape)

    total_r, total_d, total_len, total_fall = 0.0, 0.0, 0.0, 0.0
    for _ in range(n_eval):
        rng, rng_reset = jax.random.split(rng)
        state = reset_one(rng_reset)
        ep_r, ep_steps, x_start = 0.0, 0, float(state.pipeline_state.qpos[0])
        for _ in range(MAX_STEPS):
            rng, rng_act = jax.random.split(rng)
            action = policy(params, state.obs, rng_act)
            state = step_one(state, action)
            ep_r += float(state.reward)
            ep_steps += 1
            if bool(state.done):
                break
        x_end = float(state.pipeline_state.qpos[0])
        total_r += ep_r
        total_d += x_end - x_start
        total_len += ep_steps
        total_fall += 1.0 if ep_steps < MAX_STEPS else 0.0
    return {
        "fitness": total_r / n_eval,
        "distance": total_d / n_eval,
        "per_step_reward": (total_r / n_eval) / max(total_len / n_eval, 1.0),
        "mean_ep_len": total_len / n_eval,
        "fall_rate": total_fall / n_eval,
    }


def train_and_evaluate_menagerie(
    robot_name: str,
    threshold: float,
    seed: int,
    timesteps: int,
    num_envs: int,
    unroll_length: int,
    n_epochs: int,
    minibatch_size: int,
    checkpoint_dir: Path | None = None,
):
    """Mirror of train_mjx.train_and_evaluate for the Menagerie env."""
    env = MenagerieMJXEnv(robot_name, threshold=threshold)
    obs_dim = env.observation_size
    act_dim = env.action_size
    print(
        f"    {robot_name}: obs_dim={obs_dim}, act_dim={act_dim}, "
        f"z_init={env._init_z_qpos:.3f}, z_min={float(env._z_min):.3f}"
    )

    # Inject corpus hyperparams into train_mjx module globals so make_train_fns
    # and the PPO update use the right minibatch shapes.
    _train_mjx.NUM_ENVS = num_envs
    _train_mjx.UNROLL_LENGTH = unroll_length
    _train_mjx.N_EPOCHS = n_epochs
    _train_mjx.MINIBATCH_SIZE = minibatch_size

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    network = ActorCritic(action_dim=act_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = network.init(init_rng, dummy_obs)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    reset_fn, collect_rollout, compute_gae, ppo_update = make_train_fns(
        env, obs_dim, act_dim
    )

    transitions_per_update = num_envs * unroll_length
    num_updates = timesteps // transitions_per_update
    print(
        f"    {num_updates} updates x {transitions_per_update} transitions "
        f"= {num_updates * transitions_per_update} total steps"
    )

    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    print("    Compiling JIT functions...")
    t_compile = time.time()
    rng, rollout_rng = jax.random.split(rng)
    transitions, last_value, state, rollout_rng = collect_rollout(
        params, state, rollout_rng
    )
    jax.block_until_ready(transitions.obs)
    print(f"    collect_rollout compiled in {time.time() - t_compile:.1f}s")

    t_compile = time.time()
    advantages, returns = compute_gae(transitions, last_value)
    jax.block_until_ready(advantages)
    print(f"    compute_gae compiled in {time.time() - t_compile:.1f}s")

    t_compile = time.time()
    flat_obs = transitions.obs.reshape(-1, obs_dim)[:minibatch_size]
    flat_actions = transitions.action.reshape(-1, act_dim)[:minibatch_size]
    flat_log_probs = transitions.log_prob.reshape(-1)[:minibatch_size]
    flat_advantages = advantages.reshape(-1)[:minibatch_size]
    flat_returns = returns.reshape(-1)[:minibatch_size]
    warmup_out = ppo_update(
        params,
        opt_state,
        flat_obs,
        flat_actions,
        flat_log_probs,
        flat_advantages,
        flat_returns,
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(warmup_out[0])[0])
    del warmup_out
    print(f"    ppo_update compiled in {time.time() - t_compile:.1f}s")

    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    t_train = time.time()
    next_ckpt_step = 1_000_000 if checkpoint_dir else None
    for update in range(num_updates):
        rng, rollout_rng = jax.random.split(rng)
        transitions, last_value, state, rollout_rng = collect_rollout(
            params, state, rollout_rng
        )
        advantages, returns = compute_gae(transitions, last_value)
        flat_obs = transitions.obs.reshape(-1, obs_dim)
        flat_actions = transitions.action.reshape(-1, act_dim)
        flat_log_probs = transitions.log_prob.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        n_samples = flat_obs.shape[0]
        for _epoch in range(n_epochs):
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, n_samples)
            for start in range(0, n_samples, minibatch_size):
                mb = perm[start : start + minibatch_size]
                params, opt_state, _ = ppo_update(
                    params,
                    opt_state,
                    flat_obs[mb],
                    flat_actions[mb],
                    flat_log_probs[mb],
                    flat_advantages[mb],
                    flat_returns[mb],
                )

        if (update + 1) % 10 == 0:
            steps = (update + 1) * transitions_per_update
            elapsed = time.time() - t_train
            sps = steps / max(elapsed, 1e-6)
            print(
                f"    update {update + 1}/{num_updates} step={steps} "
                f"elapsed={elapsed:.0f}s SPS={sps:.0f}"
            )

        if checkpoint_dir is not None and next_ckpt_step is not None:
            steps = (update + 1) * transitions_per_update
            if steps >= next_ckpt_step:
                ckpt_path = checkpoint_dir / (
                    f"{robot_name}_tau{threshold:.2f}".replace(".", "p")
                    + f"_seed{seed}_step{steps}.npz"
                )
                _save_params(ckpt_path, params)
                next_ckpt_step += 1_000_000

    train_time = time.time() - t_train
    print(f"    Training done in {train_time:.0f}s")

    rng, eval_rng = jax.random.split(rng)
    eval_tau020 = _evaluate_with_threshold(robot_name, 0.20, params, network, eval_rng)
    rng, eval_rng = jax.random.split(rng)
    eval_tau050 = _evaluate_with_threshold(robot_name, 0.50, params, network, eval_rng)
    rng, eval_rng = jax.random.split(rng)
    eval_neutral = _evaluate_with_threshold(robot_name, 0.0, params, network, eval_rng)

    return {
        "eval_tau020": eval_tau020,
        "eval_tau050": eval_tau050,
        "eval_neutral": eval_neutral,
        "train_time_s": train_time,
        "z_init": env._init_z_qpos,
    }


def _evaluate_with_threshold(robot_name, threshold, params, network, rng):
    """Create a fresh env at the given eval threshold and run N_EVAL episodes."""
    env = MenagerieMJXEnv(robot_name, threshold=threshold)
    return _evaluate(env, params, network, rng, n_eval=N_EVAL)


def _save_params(path: Path, params):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for k, v in jax.tree_util.tree_flatten_with_path(params)[0]:
        key = "/".join(str(p) for p in k)
        flat[key] = np.asarray(v)
    np.savez(str(path), **flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True, choices=list(ROBOT_SPECS))
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--unroll_length", type=int, default=DEFAULT_UNROLL_LENGTH)
    parser.add_argument("--n_epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument("--minibatch_size", type=int, default=DEFAULT_MINIBATCH_SIZE)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    threshold_str = f"{args.threshold:.2f}".replace(".", "p")
    if args.out_dir is None:
        args.out_dir = str(
            _REPO
            / f"results/menagerie_panel/{args.robot}_tau{threshold_str}_seed{args.seed}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = out_dir / "checkpoints" if args.save_checkpoints else None

    print(f"Menagerie PPO: robot={args.robot} tau={args.threshold} seed={args.seed}")
    print(f"  JAX devices: {jax.devices()}")
    print(
        f"  Hyperparams: num_envs={args.num_envs} unroll={args.unroll_length} "
        f"epochs={args.n_epochs} mb={args.minibatch_size} lr={LEARNING_RATE}"
    )
    print(f"  Output: {out_dir}")

    provenance = collect_provenance()
    spec_version = provenance.get("git_sha", "unknown")

    t0 = time.time()
    try:
        result = train_and_evaluate_menagerie(
            robot_name=args.robot,
            threshold=args.threshold,
            seed=args.seed,
            timesteps=args.timesteps,
            num_envs=args.num_envs,
            unroll_length=args.unroll_length,
            n_epochs=args.n_epochs,
            minibatch_size=args.minibatch_size,
            checkpoint_dir=checkpoint_dir,
        )
    except Exception as exc:
        err_path = out_dir / "error.txt"
        err_path.write_text(
            f"{type(exc).__name__}: {exc}\n"
            f"provenance: {json.dumps(provenance, default=str)}\n"
        )
        raise

    record = {
        "robot_name": args.robot,
        "robot_class": ROBOT_SPECS[args.robot]["class_"],
        "algorithm": "PPO",
        "backend": "MJX-CustomPPO-Menagerie",
        "threshold": args.threshold,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "spec_version": spec_version,
        "hyperparameters": {
            "num_envs": args.num_envs,
            "unroll_length": args.unroll_length,
            "n_epochs": args.n_epochs,
            "minibatch_size": args.minibatch_size,
            "learning_rate": LEARNING_RATE,
        },
        "wallclock_seconds": time.time() - t0,
        **result,
    }
    results_path = out_dir / "results.json"
    save_locked(str(results_path), args.robot, record)
    print(f"  Wrote {results_path}")


if __name__ == "__main__":
    main()
