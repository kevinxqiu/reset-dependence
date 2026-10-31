#!/usr/bin/env python3
"""GPU-accelerated training with MuJoCo MJX + custom PPO.

Custom PPO that JIT-compiles each component separately (reset, step, rollout,
update) instead of Brax's monolithic compilation that OOMs on large models.

Usage:
    python train_mjx.py --threshold 0.50 --seed 42 --start_idx 0 --end_idx 25
    python train_mjx.py --threshold 0.20 --seed 42
"""

import os
import sys
import json
import time
import fcntl
import argparse
from typing import Any, NamedTuple

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Enable persistent XLA compilation cache - avoids recompiling identical
# programs across runs (same morphology shape + same hyperparameters).
_CACHE_DIR = os.path.expanduser("~/.cache/jax_compilation")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import mujoco  # noqa: E402
from mujoco import mjx  # noqa: E402
from flax import linen as nn  # noqa: E402
from brax.envs.base import PipelineEnv, State  # noqa: E402

MAX_STEPS = 1000
FRAME_SKIP = 4
CTRL_COST_WEIGHT = 0.001
FORWARD_REWARD_WEIGHT = 1.0
RESET_NOISE_SCALE = 0.01
N_EVAL = 20

# PPO hyperparameters (matching SB3 defaults)
NUM_ENVS = 128
UNROLL_LENGTH = 16  # steps per rollout per env
N_EPOCHS = 10
MINIBATCH_SIZE = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
TOTAL_TIMESTEPS = 5_000_000


def load_unimal_xml(xml_path):
    """Load Unimal XML, adding floor plane if missing (DERL convention)."""
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


class UnimalMJXEnv(PipelineEnv):
    """Unimal locomotion environment on MJX.

    Matches the CPU UnimalLocomotionEnv semantics:
    - Obs: qpos[2:] + qvel (skip global x,y)
    - Reward: forward_vel - ctrl_cost
    - Termination: z < init_z * threshold or z > z_max
    - frame_skip=4
    """

    def __init__(self, xml_path, threshold=0.2, backend="mjx"):
        self._xml_path = xml_path
        self._threshold = threshold

        # Load model and get initial z-height
        mj_model = load_unimal_xml(xml_path)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_resetData(mj_model, mj_data)
        self._init_z = float(mj_data.qpos[2])
        self._init_qpos = mj_data.qpos.copy()
        self._init_qvel = mj_data.qvel.copy()

        # Termination bounds as JAX arrays (not Python floats) so they don't
        # become compile-time constants. This avoids recompilation when only
        # the threshold changes for the same morphology structure.
        if threshold <= 0:
            self._z_min = jnp.float32(-1e6)
        else:
            self._z_min = jnp.float32(self._init_z * threshold)
        self._z_max = jnp.float32(max(self._init_z * 3.0, 2.0))

        # Obs dimension for Brax
        self._obs_dim = (mj_model.nq - 2) + mj_model.nv

        # Effective dt with frame_skip
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
        """Reset environment with small noise on initial state."""
        rng, rng_qpos, rng_qvel = jax.random.split(rng, 3)

        qpos = jnp.array(self._init_qpos)
        qvel = jnp.array(self._init_qvel)

        # Add reset noise (skip x,y position)
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
        info = {
            "step": jnp.int32(0),
            "truncation": jnp.float32(0),
        }
        return State(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def step(self, state, action):
        """Step environment: apply action, step physics, compute reward."""
        action = jnp.clip(action, -1.0, 1.0)
        x_before = state.pipeline_state.qpos[0]

        # Step physics (n_frames=FRAME_SKIP handled by PipelineEnv)
        data = self.pipeline_step(state.pipeline_state, action)

        # Forward velocity reward
        x_after = data.qpos[0]
        forward_vel = (x_after - x_before) / self._dt
        forward_reward = FORWARD_REWARD_WEIGHT * forward_vel
        ctrl_cost = CTRL_COST_WEIGHT * jnp.sum(jnp.square(action))
        reward = forward_reward - ctrl_cost

        # Termination
        z = data.qpos[2]
        healthy = (z >= self._z_min) & (z <= self._z_max)
        done = 1.0 - healthy

        # Truncation
        step_count = state.info["step"] + 1
        truncated = step_count >= MAX_STEPS
        done = jnp.where(truncated, 1.0, done)

        obs = self._get_obs(data)
        metrics = {
            "x_position": x_after,
            "z_height": z,
            "forward_vel": forward_vel,
            "reward": reward,
        }
        info = {
            "step": step_count,
            "truncation": jnp.where(truncated, 1.0, 0.0),
        }
        # Preserve any keys added by Brax wrappers
        info = {**state.info, **info}

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _get_obs(self, data):
        """Observation: qpos[2:] + qvel (skip global x,y)."""
        return jnp.concatenate([data.qpos[2:], data.qvel])


# ---------------------------------------------------------------------------
# Actor-Critic network (Flax linen)
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Gaussian actor + value critic with separate (64,64) MLPs."""

    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Policy (actor) head
        pi = nn.Dense(64)(x)
        pi = nn.tanh(pi)
        pi = nn.Dense(64)(pi)
        pi = nn.tanh(pi)
        mean = nn.Dense(self.action_dim)(pi)

        # Learned log_std (state-independent, one per action dim)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        # Value (critic) head
        v = nn.Dense(64)(x)
        v = nn.tanh(v)
        v = nn.Dense(64)(v)
        v = nn.tanh(v)
        value = nn.Dense(1)(v)

        return mean, log_std, jnp.squeeze(value, axis=-1)


# ---------------------------------------------------------------------------
# PPO helper types and functions
# ---------------------------------------------------------------------------


class Transition(NamedTuple):
    """Single transition stored in rollout buffer."""

    obs: Any
    action: Any
    reward: Any
    done: Any
    value: Any
    log_prob: Any


class RolloutCarry(NamedTuple):
    """Carry state for lax.scan rollout collection."""

    state: Any  # batched env State (vmapped)
    rng: Any


def gaussian_log_prob(mean, log_std, action):
    """Log probability of action under diagonal Gaussian."""
    std = jnp.exp(log_std)
    var = std**2
    log_p = -0.5 * (
        jnp.sum(((action - mean) ** 2) / var, axis=-1)
        + jnp.sum(jnp.log(var), axis=-1)
        + mean.shape[-1] * jnp.log(2 * jnp.pi)
    )
    return log_p


def gaussian_entropy(log_std):
    """Differential entropy of diagonal Gaussian."""
    return 0.5 * jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))


def make_train_fns(env, obs_dim, act_dim):
    """Create JIT-compiled functions for PPO training.

    Each function compiles independently, avoiding the monolithic XLA program
    that causes OOM with Brax's built-in PPO.
    """

    # Vmapped env functions
    batched_reset = jax.vmap(env.reset)
    batched_step = jax.vmap(env.step)

    @jax.jit
    def reset_fn(rng):
        """Batch-reset all environments."""
        rngs = jax.random.split(rng, NUM_ENVS)
        return batched_reset(rngs)

    @jax.jit
    def collect_rollout(params, state, rng):
        """Collect UNROLL_LENGTH steps from NUM_ENVS environments.

        Uses lax.scan for efficiency. Returns transitions and final state.
        Auto-resets environments that terminate.
        """
        network = ActorCritic(action_dim=act_dim)

        def step_fn(carry: RolloutCarry, _unused):
            state, rng = carry.state, carry.rng
            rng, act_rng, reset_rng = jax.random.split(rng, 3)

            # Forward pass: get action distribution and value
            obs = state.obs  # (NUM_ENVS, obs_dim)
            mean, log_std, value = network.apply(params, obs)
            std = jnp.exp(log_std)
            noise = jax.random.normal(act_rng, shape=mean.shape)
            action = mean + std * noise
            log_prob = gaussian_log_prob(mean, log_std, action)

            # Step all envs
            next_state = batched_step(state, action)

            # Store transition (obs, action from BEFORE the step)
            transition = Transition(
                obs=obs,
                action=action,
                reward=next_state.reward,
                done=next_state.done,
                value=value,
                log_prob=log_prob,
            )

            # Auto-reset terminated environments
            reset_rngs = jax.random.split(reset_rng, NUM_ENVS)
            fresh_state = batched_reset(reset_rngs)

            # Where done, use fresh state; otherwise continue
            done_mask = next_state.done  # (NUM_ENVS,)

            def select_tree(fresh, continued):
                """Select between fresh and continued state per-env."""
                return jax.tree.map(
                    lambda f, c: jnp.where(
                        # Broadcast done_mask to match leaf shape
                        jnp.reshape(
                            done_mask,
                            (NUM_ENVS,) + (1,) * (f.ndim - 1),
                        ),
                        f,
                        c,
                    ),
                    fresh,
                    continued,
                )

            next_state = select_tree(fresh_state, next_state)

            return RolloutCarry(state=next_state, rng=rng), transition

        carry = RolloutCarry(state=state, rng=rng)
        final_carry, transitions = jax.lax.scan(
            step_fn, carry, None, length=UNROLL_LENGTH
        )
        # transitions: each field is (UNROLL_LENGTH, NUM_ENVS, ...)

        # Compute bootstrap value for the final state
        _, _, last_value = network.apply(params, final_carry.state.obs)

        return transitions, last_value, final_carry.state, final_carry.rng

    @jax.jit
    def compute_gae(transitions, last_value):
        """Compute Generalized Advantage Estimation.

        Args:
            transitions: Transition with fields (T, N, ...)
            last_value: (N,) bootstrap value for final state

        Returns:
            advantages: (T, N)
            returns: (T, N) = advantages + values
        """
        rewards = transitions.reward  # (T, N)
        dones = transitions.done  # (T, N)
        values = transitions.value  # (T, N)

        def scan_fn(carry, t):
            gae = carry
            delta = rewards[t] + GAMMA * last_value * (1 - dones[t]) - values[t]
            # For t < T-1, we need the next value, not last_value
            # But we process in reverse, so carry handles bootstrapping
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            return gae, gae

        # Recompute with proper next_value for each timestep
        # Process in reverse order
        T = UNROLL_LENGTH

        def scan_fn_reverse(carry, t_rev):
            gae, next_value = carry
            t = T - 1 - t_rev
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            return (gae, values[t]), gae

        init_gae = jnp.zeros(NUM_ENVS)
        (_, _), advantages_rev = jax.lax.scan(
            scan_fn_reverse,
            (init_gae, last_value),
            jnp.arange(T),
        )
        # advantages_rev is in reverse order, flip it
        advantages = advantages_rev[::-1]
        returns = advantages + values

        return advantages, returns

    @jax.jit
    def ppo_update(
        params,
        opt_state,
        batch_obs,
        batch_actions,
        batch_log_probs,
        batch_advantages,
        batch_returns,
    ):
        """Single PPO gradient update on a minibatch.

        Args:
            params: network parameters
            opt_state: optimizer state
            batch_obs: (B, obs_dim)
            batch_actions: (B, act_dim)
            batch_log_probs: (B,) old log probs
            batch_advantages: (B,)
            batch_returns: (B,)

        Returns:
            new_params, new_opt_state, (policy_loss, value_loss, entropy)
        """
        network = ActorCritic(action_dim=act_dim)

        def loss_fn(params):
            mean, log_std, value = network.apply(params, batch_obs)
            new_log_prob = gaussian_log_prob(mean, log_std, batch_actions)
            entropy = gaussian_entropy(log_std)

            # Policy loss (clipped surrogate)
            ratio = jnp.exp(new_log_prob - batch_log_probs)
            adv = batch_advantages
            # Normalize advantages
            adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
            surr1 = ratio * adv
            surr2 = jnp.clip(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * adv
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            # Value loss (clipped)
            value_loss = jnp.mean(jnp.square(value - batch_returns))

            total_loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

            return total_loss, (policy_loss, value_loss, entropy)

        grads, (policy_loss, value_loss, entropy) = jax.grad(loss_fn, has_aux=True)(
            params
        )

        # Clip gradients
        grads = optax.clip_by_global_norm(MAX_GRAD_NORM).update(
            grads, optax.EmptyState()
        )[0]

        updates, new_opt_state = optax.adam(LEARNING_RATE).update(
            grads, opt_state, params
        )
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, (policy_loss, value_loss, entropy)

    return reset_fn, collect_rollout, compute_gae, ppo_update


def train_and_evaluate(xml_path, threshold, seed, timesteps=500_000):
    """Train a policy on one morphology with MJX + custom PPO and evaluate."""
    env = UnimalMJXEnv(xml_path, threshold=threshold)
    obs_dim = env.observation_size
    act_dim = env.action_size

    print(f"    obs_dim={obs_dim}, act_dim={act_dim}")

    # Initialize network and optimizer
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    network = ActorCritic(action_dim=act_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = network.init(init_rng, dummy_obs)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)

    # Create JIT-compiled training functions
    reset_fn, collect_rollout, compute_gae, ppo_update = make_train_fns(
        env, obs_dim, act_dim
    )

    # Calculate number of updates
    transitions_per_update = NUM_ENVS * UNROLL_LENGTH  # 128 * 16 = 2048
    num_updates = timesteps // transitions_per_update
    print(
        f"    {num_updates} updates x {transitions_per_update} transitions "
        f"= {num_updates * transitions_per_update} total steps"
    )

    # Initial reset
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    # Trigger JIT compilation on first call
    print("    Compiling JIT functions...")
    t_compile = time.time()
    rng, rollout_rng = jax.random.split(rng)
    transitions, last_value, state, rollout_rng = collect_rollout(
        params, state, rollout_rng
    )
    # Wait for compilation to complete
    jax.block_until_ready(transitions.obs)
    print(f"    collect_rollout compiled in {time.time() - t_compile:.1f}s")

    t_compile = time.time()
    advantages, returns = compute_gae(transitions, last_value)
    jax.block_until_ready(advantages)
    print(f"    compute_gae compiled in {time.time() - t_compile:.1f}s")

    # Compile ppo_update with a dummy minibatch
    t_compile = time.time()
    flat_obs = transitions.obs.reshape(-1, obs_dim)[:MINIBATCH_SIZE]
    flat_actions = transitions.action.reshape(-1, act_dim)[:MINIBATCH_SIZE]
    flat_log_probs = transitions.log_prob.reshape(-1)[:MINIBATCH_SIZE]
    flat_advantages = advantages.reshape(-1)[:MINIBATCH_SIZE]
    flat_returns = returns.reshape(-1)[:MINIBATCH_SIZE]
    _, _, _ = ppo_update(
        params,
        opt_state,
        flat_obs,
        flat_actions,
        flat_log_probs,
        flat_advantages,
        flat_returns,
    )
    jax.block_until_ready(flat_obs)
    print(f"    ppo_update compiled in {time.time() - t_compile:.1f}s")

    # Re-initialize state for actual training (compilation used some steps)
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    # Training loop
    print("    Training...")
    t_train = time.time()
    total_steps = 0

    for update_idx in range(num_updates):
        # Collect rollout
        rng, rollout_rng = jax.random.split(rng)
        transitions, last_value, state, rollout_rng = collect_rollout(
            params, state, rollout_rng
        )

        # Compute GAE
        advantages, returns = compute_gae(transitions, last_value)

        # Flatten: (T, N, ...) -> (T*N, ...)
        total_samples = UNROLL_LENGTH * NUM_ENVS
        flat_obs = transitions.obs.reshape(total_samples, obs_dim)
        flat_actions = transitions.action.reshape(total_samples, act_dim)
        flat_log_probs = transitions.log_prob.reshape(total_samples)
        flat_advantages = advantages.reshape(total_samples)
        flat_returns = returns.reshape(total_samples)

        # PPO epochs
        for _epoch in range(N_EPOCHS):
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, total_samples)

            for start in range(0, total_samples, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                if end > total_samples:
                    break
                idx = perm[start:end]

                mb_obs = flat_obs[idx]
                mb_actions = flat_actions[idx]
                mb_log_probs = flat_log_probs[idx]
                mb_adv = flat_advantages[idx]
                mb_ret = flat_returns[idx]

                params, opt_state, (p_loss, v_loss, ent) = ppo_update(
                    params,
                    opt_state,
                    mb_obs,
                    mb_actions,
                    mb_log_probs,
                    mb_adv,
                    mb_ret,
                )

        total_steps += total_samples

        # Log progress every 10 updates
        if (update_idx + 1) % 10 == 0 or update_idx == num_updates - 1:
            elapsed = time.time() - t_train
            sps = total_steps / elapsed
            mean_reward = float(jnp.mean(transitions.reward))
            print(
                f"    update {update_idx + 1}/{num_updates}: "
                f"steps={total_steps}, reward={mean_reward:.3f}, "
                f"p_loss={float(p_loss):.4f}, v_loss={float(v_loss):.4f}, "
                f"ent={float(ent):.3f}, "
                f"SPS={sps:.0f}, elapsed={elapsed:.0f}s"
            )
            sys.stdout.flush()

    train_time = time.time() - t_train
    print(f"    Training done in {train_time:.0f}s ({total_steps} steps)")

    # ------------------------------------------------------------------
    # Evaluation - CPU MuJoCo (avoids GPU memory issues from MJX eval)
    # ------------------------------------------------------------------
    def cpu_inference(obs_np, rng_action):
        """Run policy forward pass with stochastic sampling (matches SB3 eval)."""
        obs_jax = jnp.array(obs_np[None, :])
        mean, log_std, _ = network.apply(params, obs_jax)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng_action, shape=mean.shape)
        action = mean + std * noise
        return np.array(jnp.squeeze(action, axis=0))

    eval_results = {}
    for eval_name, eval_threshold in [
        ("eval_tau020", 0.2),
        ("eval_tau050", 0.5),
        ("eval_neutral", 0.0),
    ]:
        # Use CPU MuJoCo for evaluation
        mj_model = load_unimal_xml(xml_path)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_resetData(mj_model, mj_data)
        init_z = mj_data.qpos[2]

        if eval_threshold <= 0:
            z_min_eval = -1e6
        else:
            z_min_eval = init_z * eval_threshold
        z_max_eval = max(init_z * 3.0, 2.0)

        init_qpos = mj_data.qpos.copy()
        init_qvel = mj_data.qvel.copy()
        dt = mj_model.opt.timestep * FRAME_SKIP

        rewards_list, distances, ep_lengths = [], [], []
        rng_eval_jax = jax.random.PRNGKey(5000)

        for ep in range(N_EVAL):
            mujoco.mj_resetData(mj_model, mj_data)
            # Use fixed seed per episode (matches SB3 eval protocol)
            ep_rng = np.random.RandomState(ep + 5000)
            noise_q = ep_rng.uniform(-RESET_NOISE_SCALE, RESET_NOISE_SCALE, mj_model.nq)
            noise_q[0] = noise_q[1] = 0.0
            noise_v = ep_rng.uniform(-RESET_NOISE_SCALE, RESET_NOISE_SCALE, mj_model.nv)
            mj_data.qpos[:] = init_qpos + noise_q
            mj_data.qvel[:] = init_qvel + noise_v
            mujoco.mj_forward(mj_model, mj_data)

            total_reward = 0.0
            steps = 0
            rng_eval_jax, ep_action_rng = jax.random.split(rng_eval_jax)
            for step_i in range(MAX_STEPS):
                x_before = mj_data.qpos[0]
                obs = np.concatenate([mj_data.qpos[2:], mj_data.qvel]).astype(
                    np.float32
                )
                ep_action_rng, step_rng = jax.random.split(ep_action_rng)
                action = cpu_inference(obs, step_rng)
                action = np.clip(action, -1.0, 1.0)

                mj_data.ctrl[:] = action
                for _ in range(FRAME_SKIP):
                    mujoco.mj_step(mj_model, mj_data)
                steps += 1

                x_after = mj_data.qpos[0]
                fwd_vel = (x_after - x_before) / dt
                reward = FORWARD_REWARD_WEIGHT * fwd_vel - CTRL_COST_WEIGHT * np.sum(
                    action**2
                )
                total_reward += reward

                z = mj_data.qpos[2]
                if z < z_min_eval or z > z_max_eval:
                    break

            rewards_list.append(total_reward)
            distances.append(float(mj_data.qpos[0]))
            ep_lengths.append(steps)

        rewards_arr = np.array(rewards_list)
        ep_lengths_arr = np.array(ep_lengths)
        per_step = rewards_arr / np.maximum(ep_lengths_arr, 1).astype(float)

        eval_results[eval_name] = {
            "fitness": float(np.mean(rewards_arr)),
            "fitness_std": float(np.std(rewards_arr)),
            "distance": float(np.mean(distances)),
            "distance_std": float(np.std(distances)),
            "per_step_reward": float(np.mean(per_step)),
            "per_step_reward_std": float(np.std(per_step)),
            "mean_ep_len": float(np.mean(ep_lengths_arr)),
            "ep_len_std": float(np.std(ep_lengths_arr)),
            "fall_rate": float(np.mean(ep_lengths_arr < MAX_STEPS)),
        }

    return eval_results, {"train_time_s": train_time}


def load_locked(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        data = json.load(f)
        fcntl.flock(f, fcntl.LOCK_UN)
    return data


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument(
        "--xml_dir",
        default="assets/unimals_100/xml",  # edited for release: was the authors' body folder
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=100)
    parser.add_argument(
        "--morphology-list",
        default=None,
        help="Text file with one morphology name per line (overrides start/end_idx)",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        threshold_str = f"{args.threshold:.2f}".replace(".", "p")
        args.out_dir = os.path.join(
            proj_dir, f"results/mjx_ppo_{threshold_str}_seed{args.seed}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"MJX Custom PPO Training: threshold={args.threshold}, seed={args.seed}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  Timesteps: {args.timesteps}")
    print(
        f"  Hyperparams: num_envs={NUM_ENVS}, unroll={UNROLL_LENGTH}, "
        f"epochs={N_EPOCHS}, mb={MINIBATCH_SIZE}, lr={LEARNING_RATE}"
    )
    print(f"  Output: {args.out_dir}")

    xmls = sorted(f for f in os.listdir(args.xml_dir) if f.endswith(".xml"))

    # If morphology list provided, filter to those morphologies
    if args.morphology_list:
        with open(args.morphology_list) as f:
            target_morphs = [line.strip() for line in f if line.strip()]
        xml_names = {x.replace(".xml", ""): x for x in xmls}
        filtered = []
        for m in target_morphs:
            if m in xml_names:
                filtered.append(xml_names[m])
            else:
                print(f"  Warning: morphology {m} not found in xml_dir")
        xmls = sorted(filtered)
        args.start_idx = 0
        args.end_idx = len(xmls)
        print(
            f"  Morphology list: {len(xmls)} morphologies from {args.morphology_list}"
        )

    results_path = os.path.join(args.out_dir, "results.json")
    t0 = time.time()
    completed = 0

    existing = load_locked(results_path)
    if existing:
        print(f"Resuming: {len(existing)} already done")

    n_todo = min(args.end_idx, len(xmls)) - args.start_idx
    for i, idx in enumerate(range(args.start_idx, min(args.end_idx, len(xmls)))):
        morph_name = xmls[idx].replace(".xml", "")

        if morph_name in existing or (
            completed > 0 and morph_name in load_locked(results_path)
        ):
            print(f"[{i + 1}/{n_todo}] Skip {morph_name}")
            continue

        xml_path = os.path.join(args.xml_dir, xmls[idx])
        print(
            f"[{i + 1}/{n_todo}] {morph_name} "
            f"(custom PPO, threshold={args.threshold}, seed={args.seed})"
        )

        try:
            t1 = time.time()
            eval_results, train_metrics = train_and_evaluate(
                xml_path,
                args.threshold,
                args.seed,
                timesteps=args.timesteps,
            )
            train_time = time.time() - t1

            result = {
                "morph_idx": idx,
                "algorithm": "PPO",
                "backend": "MJX-CustomPPO",
                "threshold": args.threshold,
                "seed": args.seed,
                "timesteps": args.timesteps,
                **eval_results,
                "train_time_s": train_time,
            }
            save_locked(results_path, morph_name, result)
            completed += 1

            std = eval_results["standard_eval"]
            elapsed = time.time() - t0
            remaining = n_todo - i - 1
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
            continue

    print(
        f"\nDone! {completed} trained, "
        f"{len(load_locked(results_path))} total in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
