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
import socket
import inspect
import argparse
import datetime
import subprocess
import importlib.metadata
from typing import Any, NamedTuple

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Enable persistent XLA compilation cache - avoids recompiling identical
# programs across runs (same morphology shape + same hyperparameters).
_CACHE_ROOT = os.environ.get("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))
_CACHE_DIR = os.path.join(_CACHE_ROOT, "jax_compilation")
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

# PPO hyperparameters.
# Defaults pin to the canonical 5M-step corpus values that produced every
# numerical claim in the paper (num_envs=128, unroll=16, n_epochs=10,
# minibatch=256, total=5e6). Reproducers running `python scripts/train_mjx.py`
# without flags get the corpus regime by default. Longer-budget checks use
# explicit CLI overrides rather than changing these corpus defaults.
# Note on throughput: corpus settings give ~1.6K SPS on a single 2080 Ti
# with 4-way morphology packing.
NUM_ENVS = 128
UNROLL_LENGTH = 16
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


def default_unimal_xml_dir() -> str:
    """Return the fetched UNIMAL XMLs (scripts/fetch_unimal_assets.py) by default; override with RFC_UNIMAL_XML_DIR."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback = os.path.join(repo, "assets", "unimals_100", "xml")
    candidates = [os.environ.get("RFC_UNIMAL_XML_DIR"), fallback]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return fallback


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

    def __init__(
        self,
        xml_path,
        threshold=0.2,
        backend="mjx",
        handle_truncation=True,
        impl="jax",
        warp_njmax=256,
        warp_nconmax=128,
        termination_mode="height",
        contact_termination_steps=50,
        max_steps=MAX_STEPS,
    ):
        self._handle_truncation = handle_truncation
        self._impl = impl
        self._warp_njmax = warp_njmax
        self._warp_nconmax = warp_nconmax
        if termination_mode not in ("height", "contact-duration"):
            raise ValueError(
                f"termination_mode must be 'height' or 'contact-duration', "
                f"got {termination_mode!r}"
            )
        self._termination_mode = termination_mode
        self._contact_termination_steps = int(contact_termination_steps)
        # max_steps: number of env steps before the truncation predicate fires.
        # Default (1000) matches the corpus. Boundary-free runs pass a value
        # larger than total_env_steps so truncation never fires.
        self._max_steps = int(max_steps)

        # Load model and get initial z-height
        mj_model = load_unimal_xml(xml_path)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_resetData(mj_model, mj_data)
        self._init_z = float(mj_data.qpos[2])
        self._init_qpos = mj_data.qpos.copy()
        self._init_qvel = mj_data.qvel.copy()

        # Resolve torso and floor geom IDs for contact-duration termination.
        # UNIMAL convention: floor geom_id=0, torso geom_id=1; resolve by
        # name rather than hard-coding so the code is robust to XML changes.
        self._floor_geom_id = int(
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        )
        self._torso_geom_id = int(
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "torso/0")
        )
        if self._floor_geom_id < 0 or self._torso_geom_id < 0:
            raise RuntimeError(
                f"XML {xml_path}: could not resolve floor or torso/0 geom "
                f"(floor={self._floor_geom_id}, torso/0={self._torso_geom_id})"
            )
        # Keep the raw MjModel around: mjx.make_data(impl='warp') requires
        # the raw mujoco.MjModel, not the mjx-put sys that Brax hands it.
        self._mj_model = mj_model

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

        # MJX 3.6+ takes impl in {'c','cpp','jax','warp'}. Older mjx ignores
        # the kwarg. We pass it through if the param exists; otherwise the
        # caller asked for a backend the installed mjx doesn't know.
        put_kwargs = {}
        if "impl" in inspect.signature(mjx.put_model).parameters:
            put_kwargs["impl"] = impl
        elif impl != "jax":
            raise RuntimeError(
                f"impl={impl!r} requested but mjx.put_model does not accept "
                f"the `impl` kwarg in this venv (mjx <= 3.5). Upgrade mjx."
            )
        sys = mjx.put_model(mj_model, **put_kwargs)
        super().__init__(sys=sys, n_frames=FRAME_SKIP, backend=backend)

    def pipeline_init(self, q, qd, act=None, ctrl=None):
        """Override Brax's MJX pipeline_init to support impl='warp'.

        Brax's brax/mjx/pipeline.init hard-codes ``mjx.make_data(sys)`` with
        no ``impl`` kwarg. For impl='warp' this fails twice: (1) make_data
        needs the raw mujoco.MjModel, not the mjx-put sys Brax passes, and
        (2) mujoco_warp silently truncates constraint rows past njmax,
        corrupting physics without raising. We monkey-patch mjx.make_data
        across the parent call to substitute the raw MjModel we stashed at
        __init__ time and inject impl + njmax + naconmax.

        For impl='jax' (default), this is a pass-through to the parent.
        """
        if self._impl == "jax":
            return super().pipeline_init(q, qd, act, ctrl)

        md_sig = inspect.signature(mjx.make_data).parameters
        if "impl" not in md_sig:
            raise RuntimeError(
                f"impl={self._impl!r} requested but mjx.make_data does not "
                f"accept the `impl` kwarg in this venv (mjx <= 3.5)."
            )

        orig_make_data = mjx.make_data
        impl = self._impl
        mj_model = self._mj_model
        njmax = self._warp_njmax if "njmax" in md_sig else None
        # mjx >= 3.5 deprecates `nconmax` in favor of `naconmax`. Prefer the
        # new name when available; fall back for older signatures.
        if "naconmax" in md_sig:
            con_key = "naconmax"
        elif "nconmax" in md_sig:
            con_key = "nconmax"
        else:
            con_key = None
        nconmax = self._warp_nconmax

        def patched_make_data(_m, **kw):
            # Brax passes the mjx-put sys as `_m`; Warp's make_data requires
            # the raw MjModel. Substitute the stashed raw model.
            kw.setdefault("impl", impl)
            if njmax is not None:
                kw.setdefault("njmax", njmax)
            if con_key is not None:
                kw.setdefault(con_key, nconmax)
            return orig_make_data(mj_model, **kw)

        # Patch only across the parent call. JIT tracing captures the
        # operations performed by make_data (not the function reference),
        # so subsequent cached calls don't depend on the patch surviving.
        mjx.make_data = patched_make_data
        try:
            return super().pipeline_init(q, qd, act, ctrl)
        finally:
            mjx.make_data = orig_make_data

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
        if self._handle_truncation:
            info = {
                "step": jnp.int32(0),
                "truncation": jnp.float32(0),
            }
        else:
            info = {}
        if self._termination_mode == "contact-duration":
            info["contact_counter"] = jnp.int32(0)
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
        if self._termination_mode == "height":
            healthy = (z >= self._z_min) & (z <= self._z_max)
            done = 1.0 - healthy
        else:  # contact-duration
            # Detect torso-floor contacts: any active contact pair where
            # one geom is the torso and the other is the floor, with
            # distance <= 0 (penetrating). Both `data._impl.contact`
            # (mjx 3.7+) and `data.contact` (mjx 3.5-3.6) expose the
            # same fields; try the new path first, fall back to legacy.
            impl = getattr(data, "_impl", None)
            if impl is not None and hasattr(impl, "contact"):
                contact = impl.contact
            else:
                contact = data.contact
            geom = contact.geom  # (ncon_max, 2)
            dist = contact.dist  # (ncon_max,)
            active = dist <= 0.0
            has_torso = (geom[:, 0] == self._torso_geom_id) | (
                geom[:, 1] == self._torso_geom_id
            )
            has_floor = (geom[:, 0] == self._floor_geom_id) | (
                geom[:, 1] == self._floor_geom_id
            )
            torso_floor_contact = jnp.any(active & has_torso & has_floor)
            # Also terminate if torso leaves the world (z too high) so a
            # flying-off morphology doesn't escape termination entirely.
            too_high = z > self._z_max

        obs = self._get_obs(data)
        metrics = {
            "x_position": x_after,
            "z_height": z,
            "forward_vel": forward_vel,
            "reward": reward,
        }

        if self._handle_truncation:
            # PPO path: env owns truncation logic
            step_count = state.info["step"] + 1
            truncated = step_count >= self._max_steps
            if self._termination_mode == "contact-duration":
                new_counter = jnp.where(
                    torso_floor_contact,
                    state.info["contact_counter"] + 1,
                    jnp.int32(0),
                )
                contact_terminated = new_counter >= jnp.int32(
                    self._contact_termination_steps
                )
                done = jnp.where(contact_terminated | too_high, 1.0, 0.0)
            done = jnp.where(truncated, 1.0, done)
            info = {
                "step": step_count,
                "truncation": jnp.where(truncated, 1.0, 0.0),
            }
            if self._termination_mode == "contact-duration":
                info["contact_counter"] = new_counter
            # Preserve any keys added by Brax wrappers
            info = {**state.info, **info}
        else:
            # Brax SAC path: Brax EpisodeWrapper owns truncation. Contact-duration
            # termination is a PPO-only auxiliary mode and is not used in the
            # reported SAC scope check.
            if self._termination_mode == "contact-duration":
                raise RuntimeError(
                    "termination_mode='contact-duration' requires "
                    "handle_truncation=True (PPO path); SAC path not supported."
                )
            info = state.info

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
    """Half the differential entropy of the diagonal Gaussian policy.

    The entropy is sum(log_std + 0.5 * log(2 pi e)); the corpus trainers
    (corpus_revision/) use half of it, kept here so reruns match the corpus.
    ENT_COEF = 0.01 on this term equals 0.005 on the entropy itself.
    """
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

            # Value loss (squared error, not clipped)
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


def train_and_evaluate(
    xml_path,
    threshold,
    seed,
    timesteps=500_000,
    impl="jax",
    warp_njmax=256,
    warp_nconmax=128,
    checkpoint_dir=None,
    morph_name=None,
    termination_mode="height",
    contact_termination_steps=50,
    max_steps=MAX_STEPS,
):
    """Train a policy on one morphology with MJX + custom PPO and evaluate.

    If checkpoint_dir + morph_name are provided, save policy params every 1M
    training steps to {checkpoint_dir}/{morph_name}_step{step}.npz.
    """
    env = UnimalMJXEnv(
        xml_path,
        threshold=threshold,
        impl=impl,
        warp_njmax=warp_njmax,
        warp_nconmax=warp_nconmax,
        termination_mode=termination_mode,
        contact_termination_steps=contact_termination_steps,
        max_steps=max_steps,
    )
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
    transitions_per_update = NUM_ENVS * UNROLL_LENGTH
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
    new_params_warmup, _, (p_loss_warmup, _, _) = ppo_update(
        params,
        opt_state,
        flat_obs,
        flat_actions,
        flat_log_probs,
        flat_advantages,
        flat_returns,
    )
    # Block on actual compiled output so the JIT compile is fully resolved
    # before we proceed (block_until_ready on an INPUT buffer is a no-op).
    jax.block_until_ready(p_loss_warmup)
    jax.block_until_ready(jax.tree_util.tree_leaves(new_params_warmup)[0])
    del new_params_warmup, p_loss_warmup
    print(f"    ppo_update compiled in {time.time() - t_compile:.1f}s")

    # Re-initialize state for actual training (compilation used some steps)
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    # Training loop
    print("    Training...")
    t_train = time.time()
    total_steps = 0
    next_checkpoint_step = 1_000_000  # save every 1M steps if checkpoint_dir set

    # Persistent-rollout diagnostics used by the boundary-free audit.
    # implementation_reset_count: total done=True fires across all envs and all
    # rollouts. For boundary-free runs this should be near zero; any nonzero
    # value is a `too_high` safety firing (logged, not a bug).
    implementation_reset_count = 0
    recent_entropies: list[float] = []
    recent_value_losses: list[float] = []
    recent_policy_losses: list[float] = []

    for update_idx in range(num_updates):
        # Collect rollout
        rng, rollout_rng = jax.random.split(rng)
        transitions, last_value, state, rollout_rng = collect_rollout(
            params, state, rollout_rng
        )
        implementation_reset_count += int(jnp.sum(transitions.done))

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

        if (
            checkpoint_dir is not None
            and morph_name is not None
            and total_steps >= next_checkpoint_step
        ):
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir, f"{morph_name}_step{total_steps}.npz"
            )
            flat = {
                f"p_{i}": np.asarray(leaf)
                for i, leaf in enumerate(jax.tree_util.tree_leaves(params))
            }
            np.savez(ckpt_path, **flat)
            next_checkpoint_step += 1_000_000

        # Log progress every 10 updates
        if (update_idx + 1) % 10 == 0 or update_idx == num_updates - 1:
            elapsed = time.time() - t_train
            sps = total_steps / elapsed
            mean_reward = float(jnp.mean(transitions.reward))
            recent_entropies.append(float(ent))
            recent_value_losses.append(float(v_loss))
            recent_policy_losses.append(float(p_loss))
            print(
                f"    update {update_idx + 1}/{num_updates}: "
                f"steps={total_steps}, reward={mean_reward:.3f}, "
                f"p_loss={float(p_loss):.4f}, v_loss={float(v_loss):.4f}, "
                f"ent={float(ent):.3f}, resets={implementation_reset_count}, "
                f"SPS={sps:.0f}, elapsed={elapsed:.0f}s"
            )
            sys.stdout.flush()

    train_time = time.time() - t_train
    print(f"    Training done in {train_time:.0f}s ({total_steps} steps)")

    # Always save a final checkpoint tagged with the requested timesteps so
    # downstream render tooling can find a known path (e.g. step5000000.npz)
    # even though total_steps may be slightly less due to (num_envs*unroll)
    # granularity.
    if checkpoint_dir is not None and morph_name is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_tag = int(timesteps)
        ckpt_path = os.path.join(
            checkpoint_dir, f"{morph_name}_step{final_tag}.npz"
        )
        flat = {
            f"p_{i}": np.asarray(leaf)
            for i, leaf in enumerate(jax.tree_util.tree_leaves(params))
        }
        np.savez(ckpt_path, **flat)
        print(
            f"    Wrote final checkpoint {ckpt_path} "
            f"(actual total_steps={total_steps})"
        )

    per_env_step_count_final = [int(x) for x in np.asarray(state.info["step"])]
    implementation_reset_rate = float(implementation_reset_count / max(total_steps, 1))

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
            for _ in range(MAX_STEPS):
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

    steps_per_second = float(total_steps / max(train_time, 1e-9))
    peak_gpu_mem_mb = None
    try:
        stats = jax.devices()[0].memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            peak_gpu_mem_mb = float(stats["peak_bytes_in_use"]) / (1024 * 1024)
    except Exception:
        pass

    return eval_results, {
        "train_time_s": train_time,
        "total_train_steps": int(total_steps),
        "steps_per_second": steps_per_second,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "params": params,
        "implementation_reset_count": int(implementation_reset_count),
        "implementation_reset_rate": implementation_reset_rate,
        "policy_entropy_trace": recent_entropies[-3:],
        "value_loss_trace": recent_value_losses[-3:],
        "policy_loss_trace": recent_policy_losses[-3:],
        "per_env_step_count_final": per_env_step_count_final,
    }


_PROVENANCE_PACKAGES = (
    "mujoco",
    "mujoco-mjx",
    "jax",
    "jaxlib",
    "brax",
    "numpy",
    "flax",
    "optax",
    "mujoco-warp",
    "warp-lang",
)


def collect_provenance() -> dict:
    """One-shot provenance snapshot for this training process.

    Captures pip-freeze of key packages, git commit + dirty flag, host, GPU
    name, Python version, CLI argv, SLURM job ids, CUDA_VISIBLE_DEVICES, and
    UTC timestamp. Every field fails silently to ``None`` so provenance
    capture can NEVER crash a training run. Meant to be called once per
    process and merged verbatim into each per-morph result dict.
    """
    prov: dict = {}
    # realpath so a symlinked invocation still lands inside the repo tree.
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    prov["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prov["host"] = socket.gethostname()
    prov["python_version"] = sys.version.split()[0]
    prov["python_executable"] = sys.executable
    prov["cli_argv"] = list(sys.argv)
    prov["cwd"] = os.getcwd()
    prov["slurm"] = {
        k: os.environ.get(k)
        for k in (
            "SLURM_JOB_ID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
            "SLURM_NODELIST",
            "SLURMD_NODENAME",
        )
    }
    prov["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    try:
        prov["git_commit"] = (
            subprocess.check_output(
                ["git", "-C", repo_dir, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        prov["git_commit"] = None
    # `git status --porcelain` is the right dirty check: it surfaces staged,
    # unstaged, and untracked changes. `git diff --quiet` misses staged-only
    # and untracked files.
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode()
        prov["git_dirty"] = bool(out.strip())
    except Exception:
        prov["git_dirty"] = None

    try:
        devs = jax.devices()
        prov["gpu_name"] = str(devs[0]) if devs else None
        prov["gpu_count"] = len(devs)
        prov["device_platform"] = devs[0].platform if devs else None
        prov["device_kind"] = getattr(devs[0], "device_kind", None) if devs else None
    except Exception:
        prov["gpu_name"] = None
        prov["gpu_count"] = None
        prov["device_platform"] = None
        prov["device_kind"] = None

    versions: dict = {}
    for pkg in _PROVENANCE_PACKAGES:
        # Broad catch: PackageNotFoundError is the common case, but a
        # malformed dist-info can raise other exceptions. We must not
        # crash training over a missing version string.
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = None
    prov["versions"] = versions

    return prov


def load_locked(path):
    if not os.path.exists(path):
        return {}
    lock_path = path + ".lock"
    with open(lock_path, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_SH)
        with open(path, "r") as f:
            content = f.read()
        fcntl.flock(lf, fcntl.LOCK_UN)
    # save_locked writes via tmp + rename so a zero-byte file shouldn't
    # exist in steady state, but a crashed earlier run (or manual touch)
    # can leave one. Treat empty as "nothing to resume" instead of
    # aborting with json.JSONDecodeError.
    if not content.strip():
        return {}
    return json.loads(content)


def save_locked(path, key, value):
    lock_path = path + ".lock"
    with open(lock_path, "a+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        existing = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read()
                if content.strip():
                    existing = json.loads(content)
        existing[key] = value
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "w") as tf:
            json.dump(existing, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp, path)
        fcntl.flock(lf, fcntl.LOCK_UN)


def main():
    global NUM_ENVS, UNROLL_LENGTH, N_EPOCHS, MINIBATCH_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS)
    parser.add_argument("--unroll_length", type=int, default=UNROLL_LENGTH)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--minibatch_size", type=int, default=MINIBATCH_SIZE)
    parser.add_argument(
        "--xml_dir",
        default=default_unimal_xml_dir(),
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=100)
    parser.add_argument(
        "--morphology-list",
        default=None,
        help="Text file with one morphology name per line (overrides start/end_idx)",
    )
    parser.add_argument(
        "--impl",
        choices=["jax", "warp", "c", "cpp"],
        default="jax",
        help="MJX backend (mjx 3.6+). 'jax' is the original Python/JAX backend; "
        "'warp' requires mujoco_warp installed.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save policy params every 1M training steps to {out_dir}/checkpoints/.",
    )
    parser.add_argument(
        "--warp_njmax",
        type=int,
        default=256,
        help="njmax buffer size for impl=warp. Smoke test on UNIMAL needed ~80; "
        "256 has 3x safety margin. Warp silently truncates rows past njmax.",
    )
    parser.add_argument(
        "--warp_nconmax",
        type=int,
        default=128,
        help="nconmax buffer size for impl=warp.",
    )
    parser.add_argument(
        "--termination-mode",
        choices=["height", "contact-duration"],
        default="height",
        help="Termination rule. 'height' (default): terminate when torso "
        "z < init_z*threshold. 'contact-duration': terminate when torso has "
        "had floor contact for >= --contact-termination-steps consecutive "
        "steps. The threshold arg is ignored in contact-duration mode.",
    )
    parser.add_argument(
        "--contact-termination-steps",
        type=int,
        default=50,
        help="N for contact-duration termination: terminate when torso has "
        "had floor contact for >= N consecutive simulator steps. Only "
        "consulted when --termination-mode=contact-duration.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=(
            "Maximum env-steps before truncation predicate fires. Default "
            f"{MAX_STEPS} matches the corpus episode boundary. Boundary-free "
            "runs pass a value larger than total env-steps (e.g., 5_000_001 "
            "for a 5M-step run) so truncation never fires; each of the "
            "num_envs parallel envs is a persistent lifetime. See "
            "the boundary-free audit protocol."
        ),
    )
    args = parser.parse_args()

    # Override module-level hyperparameters from CLI
    NUM_ENVS = args.num_envs
    UNROLL_LENGTH = args.unroll_length
    N_EPOCHS = args.n_epochs
    MINIBATCH_SIZE = args.minibatch_size

    if args.out_dir is None:
        proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        threshold_str = f"{args.threshold:.2f}".replace(".", "p")
        args.out_dir = os.path.join(
            proj_dir, f"results/mjx_ppo_{threshold_str}_seed{args.seed}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    # PACK_COUNT_ENV is the value the SLURM script ADVERTISED at job-launch
    # time. It's metadata only - this Python process has no visibility into
    # how many sibling train_mjx.py processes are actually sharing the GPU.
    # Recorded into results.json under "pack_count_env" for analysis only.
    pack_count_env = int(os.environ.get("PACK_COUNT", "1"))

    # One provenance snapshot per process. Merged into every per-morph
    # result dict so any artifact can be traced back to its exact stack,
    # git state, host, GPU, and CLI invocation.
    provenance = collect_provenance()

    print(f"MJX Custom PPO Training: threshold={args.threshold}, seed={args.seed}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"  Backend impl: {args.impl}  (PACK_COUNT env={pack_count_env})")
    if args.impl == "warp":
        print(f"  Warp buffers: njmax={args.warp_njmax}, nconmax={args.warp_nconmax}")
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
    n_failed = 0
    failed_morphs = []

    existing = load_locked(results_path)
    if existing:
        print(f"Resuming: {len(existing)} already done")
        # Sanity check: refuse to silently mix runs from different impls
        # in the same results.json. Resume dedup is keyed only on
        # morph_name, so a config mismatch would be invisible. Only
        # records produced by a post-Phase-1a train_mjx.py have the
        # `impl` field. Older records pre-date the warp work and are
        # implicitly impl='jax'. Scan ALL records: a mixed-impl file could
        # have slipped through earlier when a single-record sample matched
        # args.impl by coincidence.
        prior_impls = {rec.get("impl", "jax") for rec in existing.values()}
        if prior_impls != {args.impl}:
            offenders = [
                morph
                for morph, rec in existing.items()
                if rec.get("impl", "jax") != args.impl
            ]
            print(
                f"ERROR: existing results.json contains records with "
                f"impl(s)={sorted(prior_impls)!r} but current run requested "
                f"impl={args.impl!r}. Offenders: {offenders[:5]}"
                f"{' ...' if len(offenders) > 5 else ''}. Refusing to mix "
                f"backends in one results.json. Use a different --out_dir."
            )
            sys.stdout.flush()
            sys.exit(3)

        # Same guard for termination_mode: records without the field are
        # implicitly "height" from earlier height-threshold runs.
        prior_term_modes = {
            rec.get("termination_mode", "height") for rec in existing.values()
        }
        if prior_term_modes != {args.termination_mode}:
            offenders = [
                morph
                for morph, rec in existing.items()
                if rec.get("termination_mode", "height") != args.termination_mode
            ]
            print(
                f"ERROR: existing results.json contains records with "
                f"termination_mode(s)={sorted(prior_term_modes)!r} but "
                f"current run requested "
                f"termination_mode={args.termination_mode!r}. Offenders: "
                f"{offenders[:5]}{' ...' if len(offenders) > 5 else ''}. "
                f"Refusing to mix termination modes in one results.json. "
                f"Use a different --out_dir."
            )
            sys.stdout.flush()
            sys.exit(3)

        # Same guard for max_steps: records without the field are implicitly
        # MAX_STEPS=1000 (pre-dates the boundary-free experiment).
        prior_max_steps = {rec.get("max_steps", MAX_STEPS) for rec in existing.values()}
        if prior_max_steps != {args.max_steps}:
            offenders = [
                morph
                for morph, rec in existing.items()
                if rec.get("max_steps", MAX_STEPS) != args.max_steps
            ]
            print(
                f"ERROR: existing results.json contains records with "
                f"max_steps(s)={sorted(prior_max_steps)!r} but current run "
                f"requested max_steps={args.max_steps!r}. Offenders: "
                f"{offenders[:5]}{' ...' if len(offenders) > 5 else ''}. "
                f"Refusing to mix episode-boundary values in one results.json. "
                f"Use a different --out_dir."
            )
            sys.stdout.flush()
            sys.exit(3)

        # And for contact_termination_steps within contact-duration mode.
        if args.termination_mode == "contact-duration":
            prior_contact_steps = {
                rec.get("contact_termination_steps")
                for rec in existing.values()
                if rec.get("termination_mode") == "contact-duration"
            }
            if prior_contact_steps and prior_contact_steps != {
                args.contact_termination_steps
            }:
                offenders = [
                    morph
                    for morph, rec in existing.items()
                    if rec.get("contact_termination_steps")
                    != args.contact_termination_steps
                ]
                print(
                    f"ERROR: existing results.json contains records with "
                    f"contact_termination_steps(s)={sorted(prior_contact_steps)!r} "
                    f"but current run requested "
                    f"contact_termination_steps={args.contact_termination_steps!r}. "
                    f"Offenders: {offenders[:5]}"
                    f"{' ...' if len(offenders) > 5 else ''}. "
                    f"Use a different --out_dir."
                )
                sys.stdout.flush()
                sys.exit(3)

    total_requested = min(args.end_idx, len(xmls)) - args.start_idx
    for i, idx in enumerate(range(args.start_idx, min(args.end_idx, len(xmls)))):
        morph_name = xmls[idx].replace(".xml", "")

        if morph_name in existing or (
            completed > 0 and morph_name in load_locked(results_path)
        ):
            print(f"[{i + 1}/{total_requested}] Skip {morph_name}")
            continue

        xml_path = os.path.join(args.xml_dir, xmls[idx])
        print(
            f"[{i + 1}/{total_requested}] {morph_name} "
            f"(custom PPO, threshold={args.threshold}, seed={args.seed})"
        )

        try:
            t1 = time.time()
            checkpoint_dir = (
                os.path.join(args.out_dir, "checkpoints")
                if args.save_checkpoints
                else None
            )
            eval_results, train_metrics = train_and_evaluate(
                xml_path,
                args.threshold,
                args.seed,
                timesteps=args.timesteps,
                impl=args.impl,
                warp_njmax=args.warp_njmax,
                warp_nconmax=args.warp_nconmax,
                checkpoint_dir=checkpoint_dir,
                morph_name=morph_name,
                termination_mode=args.termination_mode,
                contact_termination_steps=args.contact_termination_steps,
                max_steps=args.max_steps,
            )
            train_time = time.time() - t1

            result = {
                "morph_idx": idx,
                "algorithm": "PPO",
                "backend": "MJX-CustomPPO",
                "impl": args.impl,
                "num_envs": NUM_ENVS,
                "unroll_length": UNROLL_LENGTH,
                "n_epochs": N_EPOCHS,
                "minibatch_size": MINIBATCH_SIZE,
                "pack_count_env": pack_count_env,
                "warp_njmax": args.warp_njmax if args.impl == "warp" else None,
                "warp_nconmax": args.warp_nconmax if args.impl == "warp" else None,
                "threshold": args.threshold,
                "seed": args.seed,
                "timesteps": args.timesteps,
                "termination_mode": args.termination_mode,
                "contact_termination_steps": (
                    args.contact_termination_steps
                    if args.termination_mode == "contact-duration"
                    else None
                ),
                "max_steps": args.max_steps,
                **eval_results,
                "train_time_s": train_time,
                "train_only_time_s": train_metrics.get("train_time_s"),
                "total_train_steps": train_metrics.get("total_train_steps"),
                "steps_per_second": train_metrics.get("steps_per_second"),
                "peak_gpu_mem_mb_self": train_metrics.get("peak_gpu_mem_mb"),
                "implementation_reset_count": train_metrics.get(
                    "implementation_reset_count"
                ),
                "implementation_reset_rate": train_metrics.get(
                    "implementation_reset_rate"
                ),
                "policy_entropy_trace": train_metrics.get("policy_entropy_trace"),
                "value_loss_trace": train_metrics.get("value_loss_trace"),
                "policy_loss_trace": train_metrics.get("policy_loss_trace"),
                "per_env_step_count_final": train_metrics.get(
                    "per_env_step_count_final"
                ),
                "provenance": provenance,
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
            n_failed += 1
            failed_morphs.append(morph_name)
            continue

    total_in_results = len(load_locked(results_path))
    print(
        f"\nDone! {completed} trained, {n_failed} failed, "
        f"{total_in_results} total in {time.time() - t0:.0f}s"
    )
    if failed_morphs:
        print(f"Failed morphs: {failed_morphs}")
        sys.stdout.flush()
        # Non-zero exit so SLURM marks the job as FAILED rather than
        # silently turning into a green job with empty results.json.
        sys.exit(1)
    # Also exit non-zero if zero morphs were attempted (e.g., all skipped
    # due to bad morphology-list) and zero completed.
    if completed == 0 and total_in_results == 0:
        print("ERROR: zero morphs trained and no existing results, exit 2.")
        sys.stdout.flush()
        sys.exit(2)


if __name__ == "__main__":
    main()
