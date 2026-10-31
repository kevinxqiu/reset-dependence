# Factorial (tau x boundary) Pre-Registration (FROZEN 2026-04-22)

This document pre-registers the factorial expansion of the boundary-free
experiment. It extends `boundary_free_prereg.md` (frozen 2026-04-21) by
adding two more protocols so that the full 3x2 factorial (tau x boundary)
is filled. Frozen before any training, smoke test, or analysis of the new
cells. No post-hoc changes permitted.

## Relation to boundary-free pre-registration

The boundary-free pre-registration (boundary_free_prereg.md) remains the
PRIMARY pre-registration and its gates are unchanged. The factorial here is
a SECONDARY, mechanistic-support analysis. It does not supersede or
retroactively modify the boundary-free primary gates.

## Motivation

The 900-run corpus covers the column `boundary = 1000` of the factorial at
three tau values {0.00, 0.20, 0.50}. The boundary-free panel (tau=0.00,
boundary=5_000_001) fills the corner (tau=0, 5M). Without the other two
cells, the paper cannot cleanly claim "boundary matters independently of
tau" - a critic would correctly observe that our only no-boundary
evidence is from tau=0, where there is no height termination at all.

Filling the (tau=0.20, 5M) and (tau=0.50, 5M) cells creates a complete
3x2 matrix and enables a proper 2-way ANOVA of tau x boundary with
interaction term. This tests whether the reordering in the corpus
depends on a tau *and* boundary interaction rather than on either
mechanism alone.

## Matrix

                    boundary = 1000      boundary = 5_000_001
        tau = 0     corpus (900-run)     boundary-free panel (in progress)
        tau = 0.20  corpus (900-run)     NEW (arm A)
        tau = 0.50  corpus (900-run)     NEW (arm H)

Both new arms use the SAME 30-body panel as the boundary-free panel
(records/boundary_free_bodies.json, already frozen and verified
identical to alt_termination_bodies.json). Same 3 seeds: 42, 43, 44.

## Hypothesis (primary factorial statistic)

H0 (null for interaction): the (tau x boundary) interaction in a 2-way
ANOVA on the 6-cell 30-body 3-seed matrix is not significant at p < 0.01;
the per-body PSR ranking under no-boundary conditions is rank-identical
(Spearman rho 95% CI lower bound >= 0.95) to the corresponding corpus tau
ranking at each tau level.

H1 (alternative): the (tau x boundary) interaction F-test is significant
(p < 0.01), AND at least 2 of 3 tau levels show a paired
Delta_tau = PSR(tau, 5M) - PSR(tau, 1000) seed-averaged per body whose
95% bootstrap CI excludes 0 in a direction-consistent way.

## Design

### Protocols (frozen)

Arm A: tau = 0.20, max_steps = 5_000_001.
Arm H: tau = 0.50, max_steps = 5_000_001.

Both arms are IDENTICAL to the boundary-free protocol except for the
single `--threshold` flag. All other hyperparameters, the body panel,
seeds, evaluation protocol, and diagnostics are unchanged.

Safety rail `too_high = z > z_max` RETAINED in all cells (same as
corpus and boundary-free panel). This is the only `done=True` source
at the height-termination level for tau=0.20 (z_min = 0.2 * init_z, so
a body that falls below 20% of init z terminates). For tau=0.50 the
threshold is 0.5 * init_z (strict corpus setting).

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

### Required per-run diagnostics

Same as boundary-free: `implementation_reset_count`,
`implementation_reset_rate`, `policy_entropy_trace`, `value_loss_trace`,
`policy_loss_trace`, `per_env_step_count_final`. Already implemented in
`scripts/train_mjx.py`.

For tau=0.20 and tau=0.50 at max_steps=5_000_001, we expect a NON-zero
`implementation_reset_rate` because height termination still fires when
bodies fall below their tau-multiplied init z. This is expected and is
the whole point of the factorial: we want to separate boundary-based
resets from height-based resets.

Expected per-cell reset rate ranges (pre-committed, informational only,
not gating):
- tau=0.20, 5M: lower than corpus tau=0.20 because each env lives longer
  so fall events are a smaller fraction of total steps; likely 0.01-0.10.
- tau=0.50, 5M: modest-to-high; likely 0.05-0.30.

### Evaluation (unchanged)

Evaluate at tau_eval = 0.20, 20 episodes per run, 1000 step cap. Same as
corpus and boundary-free.

## QC gate (applied before analysis; frozen)

Cell-level: a cell is valid if (a) results.json exists, (b) no NaN in any
reported PSR field, (c) final eval PSR is finite, (d)
`implementation_reset_rate` <= 0.30 (matching boundary-free QC).

Panel-level QC (per arm):
- Include gate: at least 80 / 90 cells valid (each family has >= 8 / 10
  bodies with >= 2 / 3 valid seeds), matching boundary-free.
- Reliability gate: sigma_seed(arm) <= 2 * sigma_seed(matched corpus tau)
  across bodies. Rationale: if boundary-removal at a given tau inflates
  seed variance by more than 2x vs the corpus boundary condition, the
  rankings in that cell are unreliable and the cell drops from the
  factorial. In that case, the factorial is reported as: "the tau=X arm
  at boundary=5M was unreliable; we report only the reliable cells".

Fail QC -> drop the affected arm (not the whole factorial).

## Primary analysis (frozen)

Effect scale: per-body, per-seed PSR = `eval_tau020 per_step_reward`
(matches main corpus and boundary-free primary).

Primary statistics, computed on the 6-cell 30-body 3-seed matrix (540
rows) where the 3 corpus cells come from existing results:

1. **Full 2-way ANOVA**: `reward ~ C(tau) * C(boundary) + C(morph) +
   C(family)`. Extract:
   a. Main effect of `boundary` (F, p, partial-eta-squared).
   b. Main effect of `tau` (for sanity; already established in the
      corpus).
   c. **`tau x boundary` interaction** (primary factorial test).

2. **Per-tau boundary-removal delta**: for each tau in {0, 0.20, 0.50},
   compute Delta_tau = PSR_{tau, 5M} - PSR_{tau, 1000} seed-averaged
   per body. Report median Delta_tau with 95% bootstrap CI (10,000
   resamples, seed=42). Per-family breakdown for completeness.

3. **Rank agreement per tau**: Spearman rho between PSR_{tau, 5M} and
   PSR_{tau, 1000} per-body rankings for each tau, seed-averaged. Report
   rho with 95% bootstrap CI (10,000 resamples).

No outlier removal. No post-hoc tweaking.

## Inclusion gates (frozen)

Applied after analysis. No post-hoc adjustment.

- **Main-text factorial paragraph** (new figure panel or appendix figure
  promoted to main text): ALL of the following:
  - ANOVA tau x boundary interaction p < 0.01 AND partial-eta-squared > 0.02.
  - At least 2 of 3 tau levels have Delta_tau 95% CI excluding 0.
  - Reliability gate passes on both new arms.

- **Appendix-only** (factorial tables and per-tau Delta figures in
  appendix, brief main-text reference): main-text conditions not met
  but ANOVA interaction p < 0.05 OR at least 1 tau level has
  Delta_tau CI excluding 0 AND reliability gate passes.

- **Drop (report as null in Limitations)**: ANOVA interaction p >= 0.05
  AND all Delta_tau CIs contain 0 AND reliability gate passes. The
  null conclusion is still reported: "the boundary-removal effect at
  non-zero tau values has a CI containing zero; the main
  reset-dependence result is driven primarily by tau".

- **QC fail (report in Limitations)**: reliability gate fails on one or
  both arms. Report only the cells that pass QC.

## Pre-committed wording for each outcome

**Main-text factorial:**

> "A 2-way factorial confirms that removing the 1000-step episode
> boundary affects PSR rankings at every tau tested (Delta_0 = X.XX,
> Delta_0.20 = X.XX, Delta_0.50 = X.XX; CIs in Table Y), with a
> significant tau x boundary interaction (F(X, Y) = Z.ZZ, p < 0.01).
> Both mechanisms contribute; neither alone explains the rank reorder
> observed in the 900-run corpus."

**Appendix-only:**

> "A 2-way factorial shows the boundary effect is strongest at tau=0
> (where it is the only active reset mechanism) and attenuates at
> non-zero tau values. Details in Appendix."

**Drop:**

> "A 2-way factorial found the boundary effect at non-zero tau has a
> 95% CI containing zero. The rank reorder in the main corpus is
> primarily driven by the tau mechanism, not episode boundaries
> (null; see Appendix)."

**QC fail:**

> "One or both no-boundary arms at non-zero tau showed seed variance
> inflation > 2x vs the matched corpus, making rank comparisons
> unreliable. We report only the cells that passed QC (see Appendix)."

## Title and placement

The main paper title "Reset Dependence Is Embodied" is retained
regardless of outcome. The factorial result goes into one of:
- A new subsection in Results (main-text outcome)
- An appendix subsection (appendix-only or drop outcome)
- A Limitations mention (null or QC-fail outcome)

## Compute budget

- Arm A (tau=0.20, 5M): 90 runs (30 bodies x 3 seeds).
- Arm H (tau=0.50, 5M): 90 runs (30 bodies x 3 seeds).
- Total new compute: 180 runs.
- Expected wall-clock: ~16h per arm at 4-process-per-GPU packing.
  Arms can run in parallel. Total wall-clock: ~16h.
- Expected GPU-hours: ~60.

## Files

- `records/factorial_prereg.md` (this file)
- `slurm/factorial_tau020_panel.sh` (arm A launcher, clone of
  boundary_free_panel.sh with `--threshold 0.2`)
- `slurm/factorial_tau050_panel.sh` (arm H launcher with
  `--threshold 0.5`)
- `scripts/analyze_factorial.py` (6-cell factorial analyzer, produced
  after both arms complete)

## Commit policy

This spec is committed before the first arm A or arm H smoke test. Any
change after commit is a deviation; the affected arm drops.

## Fallback

If EITHER new arm fails smoke or QC, that arm drops; the other arm may
still be analyzed as a single-tau boundary removal. If BOTH arms fail,
the factorial drops; the boundary-free panel (primary) stands on its
own. If the boundary-free panel itself already failed (per its own
spec), the factorial is not analyzed.
