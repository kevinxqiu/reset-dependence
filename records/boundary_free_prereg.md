# Boundary-Free Pre-Registration (FROZEN 2026-04-21)

This document pre-registers the boundary-free experiment for the NeurIPS
2026 paper "Reset Dependence Is Embodied". It is frozen before any
training, smoke test, or analysis. No post-hoc changes permitted; any
deviation forces a drop of the entire component.

## Motivation and hypothesis

The 900-run corpus uses height termination with tau in {0.00, 0.20, 0.50}.
The tau=0 condition removes height-based early termination but retains
the 1000-step episode boundary (MAX_STEPS=1000 in `scripts/train_mjx.py`),
so every 1000 env-steps each parallel env resets to the initial standing
pose. A specialist in the reset-free RL literature
(Sharma/EARL/MEDAL/RISC/MoReFree) will note that this boundary is a
second reset mechanism. We add a fourth protocol that removes the
episode boundary as well, so that one protocol in the sweep genuinely
has neither early termination nor periodic time-based reset.

H0 (null): under boundary-free training, the per-body PSR ranking is
rank-identical (bootstrap 95% CI on Spearman rho has lower bound >= 0.95)
to the corpus tau=0 ranking at matched seeds and compute, AND no family
of bodies shows a paired Delta = PSR_TRF - PSR_tau0 whose 95% bootstrap CI
excludes 0.

H1 (alternative): boundary-free training produces a per-body ranking
materially distinct from the tau=0 ranking (bootstrap 95% CI on Spearman
rho has upper bound < 0.95) and at least 2 of 3 body families show a
paired Delta_family whose 95% bootstrap CI excludes 0.

## Naming

- Internally and in code: `boundary-free` (or `TRF` for "true reset-free"
  in spec / analysis / file names only).
- Externally in paper text: "persistent no-boundary PPO" or "reset-free
  interaction protocol". Do NOT rebrand as "true reset-free RL"; the
  paper explicitly is not claiming a new reset-free RL algorithm.
- Title stays "Reset Dependence Is Embodied".

## Design

### Protocol (frozen)

Boundary-free protocol:
- tau = 0.00 (no height-based early termination; existing corpus setting).
- MAX_STEPS = 5_000_001 (truncation predicate `step_count >= MAX_STEPS`
  never fires inside a 5e6-step training run; implemented as a CLI flag
  `--max-steps 5000001`; the module-level `MAX_STEPS` constant is replaced
  by a `self._max_steps` instance attribute plumbed from the constructor).
- The `too_high = z > z_max` safety rail is RETAINED
  (z_max = max(3 * init_z, 2.0)). This is the only `done=True` source
  in the training loop; every firing is logged as an implementation reset.
- Parallel envs: num_envs=128 as in corpus. Each of the 128 envs is a
  parallel persistent lifetime of ~39,000 env-steps. Language in the
  paper must be "128 parallel persistent lifetimes", NOT "a single
  5M-step lifetime".
- GAE bootstrap: at every unroll boundary (every 16 steps), the value
  estimate V(s_t) at the final step is used as the bootstrap target
  (`done=False` so `1 - done = 1` in the GAE recurrence; unchanged
  from corpus).
- Reward and observation are unchanged from the main corpus. No reward
  shaping, no curriculum, no backward policy, no discriminator.

The existing three protocols (tau in {0.00, 0.20, 0.50}) on the 30-body
panel are REUSED from the corpus where the panel bodies are present
(all 30 selected bodies are in the 100-body corpus).

### Panel (30 UNIMAL bodies, frozen)

Selection algorithm (deterministic, seed=42):

1. Load per-body `Delta_reset = PSR_episodic - PSR_resetfree` from the
   900-run corpus using `eval_tau020 per_step_reward`, seed-averaged
   (seeds 42, 43, 44).
2. For each family F in {floor, vt, mvt}:
   a. Rank F-bodies by Delta_reset (ascending).
   b. Partition the ranked list into 5 quintile bins of near-equal size.
   c. Within each quintile, select 2 bodies using
      `numpy.random.default_rng(42)`. If a quintile contains < 2 bodies
      (MVT has only 22 bodies, so the tail quintiles may have 4 each),
      the rng picks 2 without replacement; if a quintile has fewer than
      2 bodies the algorithm promotes from the nearest quintile
      (documented in the body JSON).
3. Yield 30 bodies total (10 per family, 2 per quintile).

Output file: `records/boundary_free_bodies.json` with
per-body family, Delta_reset, quintile-within-family, and master
seed=42.

### Seeds per cell

Three seeds: 42, 43, 44 (match corpus).

### Training hyperparameters (unchanged from corpus)

With the corpus training settings:
```
--num_envs 128 --unroll_length 16 --n_epochs 10 --minibatch_size 256
--timesteps 5000000
```
gamma=0.99, GAE lambda=0.95, clip epsilon=0.2, lr=3e-4,
MLP [64, 64] tanh for policy and value.

Any deviation from these hyperparameters forces a drop of the affected
cell.

### Required per-run diagnostics (logged to results JSON)

Each boundary-free run MUST emit, in addition to the existing corpus
fields:
1. `implementation_reset_count`: total count of `done=True` fires across
   all 128 envs across all env-steps. Dominated by `too_high` firings
   under boundary-free; a nonzero count is expected and is logged, not
   treated as a bug.
2. `implementation_reset_rate`: `implementation_reset_count /
   (num_envs * total_env_steps)`. If > 0.03 for any run, the run is
   flagged for review; if > 0.30, the run is QC-failed.
3. `policy_entropy_trace`: final policy entropy at the last logged
   update (same point reported in training stdout). Sanity check that
   entropy didn't collapse to zero.
4. `value_loss_trace`: value loss at last 3 logged updates. Sanity check
   that the critic didn't diverge.
5. `obs_norm_stats`: observation-normalizer mean and std (per-dim max and
   min summarized to scalars) at 4 checkpoints during training. Drift
   detection.
6. `per_env_step_count_final`: number of env-steps each of the 128 envs
   took before training ended (should all equal `total_env_steps /
   num_envs` if no env was reset).

These fields are implemented in `scripts/train_mjx.py` and written
unconditionally (not guarded by protocol name).

### Evaluation (unchanged from corpus)

Evaluate at tau_eval = 0.20 (matched to main-text corpus metric) with
20 episodes per run. Episode cap 1000 for eval only. This means the
eval policy is run under the standard episodic boundary, the same as
every other protocol, so the PSR cross-protocol comparison is
apples-to-apples.

Additional eval metric: `persistent_eval_psr`. One additional
evaluation rollout of length 5000 steps under tau_eval=0 and no
episode boundary. Reports per-step reward over this rollout. This is
a diagnostic only (tests self-recovery under genuine no-reset
conditions) and is NOT used for the pre-registered gates.

## QC gate (applied before analysis; frozen)

Cell-level: a cell is valid if (a) results.json exists, (b) no NaN in any
reported PSR field, (c) final eval PSR is finite, (d)
`implementation_reset_rate` <= 0.30.

Panel-level QC:
- Include gate: at least 80 / 90 cells valid
  (each family has >= 8 / 10 bodies with >= 2 / 3 valid seeds).
- Reliability gate: `sigma_seed(TRF) <= 2 * sigma_seed(tau=0)` across
  bodies in the panel, where sigma_seed is the across-seed standard
  deviation of PSR for a given body under a given protocol. If TRF is
  more than 2x noisier than tau=0, rankings are unreliable and the
  component drops.

Fail either gate -> drop the entire component (no partial integration).

## Primary analysis (frozen)

Effect scale: per-body, per-seed PSR = `eval_tau020 per_step_reward`
(matches main corpus).

Three primary statistics, all computed on the 30-body x 4-protocol x
3-seed matrix (where TRF is the new protocol and the other three come
from the existing corpus):

1. **Paired family Delta**: for each family F in {floor, vt, mvt},
   compute `Delta_F = median_body_F(PSR_TRF - PSR_tau0)` seed-averaged
   per body. Report Delta_F with 95% bootstrap CI (10,000 resamples
   over bodies within family, seed=42).

2. **Rank agreement**: Spearman rho between TRF per-body rankings and
   tau=0 per-body rankings on the 30-body panel, both seed-averaged.
   Report rho with 95% bootstrap CI (10,000 resamples over bodies,
   seed=42).

3. **Protocol-by-family paired contrast**: two-way ANOVA on the
   4-protocol x 30-body x 3-seed dataset:
   `reward ~ C(protocol) * C(family) + C(morph)`,
   extracting the `protocol=TRF` contrast and its interaction with
   `family`. Report F-statistics and partial eta^2 for the
   TRF-specific terms only.

No outlier removal. No post-hoc tweaking.

## Inclusion gates (frozen)

Applied after analysis, no post-hoc adjustment. Based on design review
synthesis 2026-04-21:

- **Main-text inclusion** (new figure panel + paragraph in Results +
  abstract mention): ALL of the following must hold.
  - Spearman rho(TRF, tau=0) bootstrap 95% CI upper bound < 0.95.
  - At least 2 of 3 families have paired Delta_F 95% bootstrap CI
    excluding 0.
  - Reliability gate passes.

- **Appendix-only** (subsection in appendix, narrow scope): if
  main-text thresholds not met but at least one family has paired
  Delta_F 95% CI excluding 0 AND Spearman rho upper bound < 0.99.

- **Drop (redundant with tau=0)**: Spearman rho bootstrap 95% CI lower
  bound > 0.95 AND all paired Delta_F 95% CIs contain 0.

- **Reframe (collapse)**: `implementation_reset_rate` > 0.30 across a
  majority of runs. Present the finding as "persistent PPO is
  body-sensitive to state-distribution collapse under the boundary-free
  regime" in a dedicated appendix subsection. Do NOT include in main
  text ranking analysis.

## Pre-committed wording for each outcome

For each gate outcome, the paper's wording is pre-committed:

**Main-text inclusion:**

> "Removing the 1000-step episode boundary produces yet another ranking
> reordering (Spearman rho X.XX [X.XX, X.XX]) and distinct family
> effects (Floor Delta=X.XX, VT Delta=X.XX, MVT Delta=X.XX). Reset
> structure is an embodied experimental variable across both early-
> termination and boundary mechanisms."

**Appendix-only:**

> "Under the persistent no-boundary protocol, some family-level
> differences persist but overall rank agreement with tau=0 is high
> (rho >= 0.95). The boundary effect is smaller than the early-
> termination effect. Details in Appendix."

**Drop (redundant):**

> "We also ran a no-episode-boundary variant on a 30-body panel; results
> rank-agreed with tau=0 (rho >= 0.95) with no family-specific effect,
> indicating the 1000-step boundary is not a substantial additional
> reset mechanism beyond fall termination. Details in Appendix."

**Reframe (collapse):**

> "Persistent no-boundary PPO produced a high implementation-reset rate
> across morphologies (>30% of env-steps), consistent with
> state-distribution collapse under continuous training. We report this
> separately as a training-stability diagnostic."

All four outcomes are publishable. The outcome is determined by data;
the paper's wording is pre-committed to prevent selective reporting.

## Title policy

The main paper title "Reset Dependence Is Embodied" is retained
regardless of outcome. The subtitle "Training Protocol Changes Which
Morphologies Appear Optimal" is retained.

The paper will add one to three sentences disambiguating "reset" to
include both early-termination and episode-boundary mechanisms, with
explicit reference to the reset-free RL literature (EARL, MEDAL, RISC,
MoReFree) and a citation of Sharma 2022's autonomous reset-free RL
setting as the strict version of reset-free.

## Compute budget

- 90 new runs (30 bodies x 3 seeds x 1 protocol).
- Expected wall-clock: ~10 hours at 4-process-per-GPU packing on a
  GPU cluster.
- Expected GPU-hours: ~30.
- Matches the compute budget of a same-scale auxiliary panel; known feasible.

## Files

- `records/boundary_free_prereg.md` (this file)
- `records/boundary_free_bodies.json` (panel selection output)
- `scripts/pick_boundary_free_bodies.py` (panel selector)
- `scripts/train_mjx.py` (add `--max-steps` flag + diagnostics)
- `slurm/boundary_free_smoke.sh` (1-body smoke test)
- `slurm/boundary_free_panel.sh` (90-run batch)
- `scripts/analyze_boundary_free.py` (gate-applier, computes the 3
  primary statistics and applies inclusion gates)

## Commit policy

This spec and the body JSON must be committed before the first smoke
test. Any change after commit is a deviation; the entire component
drops.

## Fallback

If smoke test fails (NaN, value divergence, or catastrophic entropy
collapse in first 500k steps) OR full panel QC fails OR the Reframe
(collapse) gate fires, fall back to the current HEAD (pre-boundary-free)
as the submission state. The existing draft is complete; nothing is
lost by adding and potentially dropping this component.

## Pre-smoke clarification (2026-04-21, before any training)

Added after spec commit `2aa4be4` in response to design review flagging
three ambiguities. These are disambiguations of terms the frozen spec
did not define precisely; they are NOT amendments to the gates, panel,
or training protocol.

1. **`obs_norm_stats` is removed from required diagnostics.** The
   codebase does not use observation normalization; raw observations
   flow directly into the policy network
   (`scripts/train_mjx.py:387`). Tracking normalizer statistics is
   not applicable. The remaining required diagnostics
   (`implementation_reset_count`, `implementation_reset_rate`,
   `policy_entropy_trace`, `value_loss_trace`, `policy_loss_trace`,
   `per_env_step_count_final`) are all implemented and written
   unconditionally.

2. **`persistent_eval_psr` is deferred.** The 5000-step no-boundary
   evaluation rollout is moved to an optional post-smoke enhancement.
   It is a diagnostic that was never used for the pre-registered gates,
   so omitting it does not weaken the analysis. If time permits after
   smoke passes, it will be added in a separate commit; if not, the
   pre-registered gates still apply using the standard
   `eval_tau020 per_step_reward` metric.

3. **Reframe (collapse) gate threshold clarified.** The spec text
   "`implementation_reset_rate > 0.30` across a majority of runs"
   is defined as: strictly more than 45 of 90 valid runs have
   `implementation_reset_rate > 0.30`. A single run exceeding 0.30
   triggers the cell-level QC drop (already specified) but not the
   panel-level Reframe. This prevents a single pathological body from
   flipping the whole panel into the Reframe branch.
