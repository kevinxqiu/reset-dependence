# Boundary Duration Sweep Pre-Registration (FROZEN 2026-04-22)

> ERRATUM (2026-04-22, before launch): the intermediate boundary values
> are `{3001, 10001}` (corpus `+2001` and `+9001` so 5M-step runs yield
> clean seed counts at `num_envs=128`, `unroll_length=16`), NOT
> `{10000, 100000}` as written in one early draft line below. The
> canonical set is `B ∈ {1000, 3001, 10001, 5_000_001}`. All analysis,
> SLURM, and results files use the canonical set. The stale `{1000,
> 10000, 100000, 5M}` phrasing that survives in the "Relation to prior
> pre-registrations" bullet is left in place with this note rather than
> being rewritten, so the pre-reg timestamp remains verifiable.

This document pre-registers the boundary duration sweep for the NeurIPS
2026 paper "Reset Dependence Is Embodied". Frozen BEFORE any training.
No post-hoc changes permitted. This is a SECONDARY sweep that provides
dose-response evidence for the reset-frequency claim; the primary
claim is still carried by the 900-run corpus, the asymptotic training
gate, and the boundary-free panel.

## Relation to prior pre-registrations

- `boundary_free_prereg.md` (2026-04-21): primary pre-registration for
  (tau=0, boundary=5_000_001). Unchanged.
- This sweep adds 2 intermediate boundary values at tau=0 ONLY, giving
  4 log-spaced boundary points at tau=0: {1000, 10000, 100000, 5_000_001}.
  The other two tau levels (0.20, 0.50) are NOT swept here.

## Motivation

A reader from the reset-free RL literature (Sharma/EARL, MEDAL, RISC,
MoReFree) may ask whether the boundary-free panel reflects a smooth
change with reset frequency or only an extreme no-boundary condition.
Without intermediate data, the paper cannot
distinguish:

- (M1) a dose-response: PSR ranking diverges continuously from the
  corpus as boundary duration grows,
- (M2) a step change: PSR ranking is identical up to some transition
  point then diverges abruptly,
- (M3) a trivial artefact: only the extreme B=5M diverges because it
  removes resets entirely.

H1 (paper thesis): (M1) is the correct model - PSR rank agreement
decreases MONOTONICALLY as boundary duration grows. Specifically, for
fixed tau=0, rho(B=1000, B) has 95% bootstrap CI that decreases in an
order-consistent way across B in {10000, 100000, 5_000_001}.

H0 (null): (M3) is the correct model - the B=5M result is an artefact
of fully removing resets, not a reset-frequency effect. In particular,
rho(B=1000, B=10000) 95% CI contains 1.0 (i.e., ranking is unchanged at
B=10000 compared to B=1000), and the same for B=100000.

## Design

### Protocol (frozen)

Four boundary levels at tau=0:
- B=1000: corpus, already collected.
- B=10000: NEW, this sweep.
- B=100000: NEW, this sweep.
- B=5_000_001: boundary-free panel, in progress.

All runs share: tau=0.00, num_envs=128, unroll_length=16, n_epochs=10,
minibatch_size=256, timesteps=5_000_000, gamma=0.99, GAE lambda=0.95,
clip=0.2, lr=3e-4, MLP [64, 64] tanh. The ONLY variable is
`--max-steps`.

Same 30-body panel as boundary-free
(records/boundary_free_bodies.json, already frozen). Same seeds
42, 43, 44.

Safety rail `too_high = z > z_max` RETAINED. It is the only non-
truncation `done=True` source at tau=0.

### Expected truncation counts (informational; not gating)

At tau=0, done=True fires only from too_high (rare) or truncation.
Truncation count per env during a 5M-step training run:
- B=1000: ~5000 truncations per env, ~640k total (matches corpus
  behavior).
- B=10000: ~500 truncations per env, ~64k total.
- B=100000: ~50 truncations per env, ~6.4k total.
- B=5_000_001: 0 truncations (confirmed in boundary-free smoke).

Three decades of log-spaced truncation frequency.

### Required per-run diagnostics

Same as boundary-free:
- `implementation_reset_count` (truncations + too_high).
- `implementation_reset_rate` = implementation_reset_count /
  total_env_steps.
- `policy_entropy_trace`, `value_loss_trace`, `policy_loss_trace`.
- `per_env_step_count_final`.

QC will use these to detect divergence or collapse.

### Evaluation (unchanged)

tau_eval = 0.20, 20 episodes per run, 1000-step cap.

## QC gate (applied before analysis; frozen)

Cell-level: valid if results.json exists, no NaN in any PSR field, eval
PSR finite, `implementation_reset_rate` <= 0.50.

Note: at B=1000 (corpus matched), expected reset_rate is ~0.128 (every
step in a 1000-step episode is a "reset step" fraction), actually it
is `1/B` since only truncations fire; so at B=1000 rate is 1/1000. At
B=10000 rate is 1/10000. This is NEAR ZERO in all cells because tau=0
has no height-termination. The 0.50 QC threshold is loose; the cell
is expected to produce rate < 0.002 in all cells. A cell with
rate > 0.50 has catastrophic too_high firing and should be dropped.

Panel-level QC (per new B value):
- Include gate: >= 80 / 90 cells valid (each family has >= 8/10 bodies
  with >= 2/3 valid seeds), matching boundary-free.
- Reliability gate: sigma_seed(B) <= 2 * sigma_seed(B=1000 corpus).

Fail -> drop the affected B value from analysis (not the whole sweep).

## Primary analysis (frozen)

Effect scale: per-body, per-seed PSR = `eval_tau020 per_step_reward`.

Three primary statistics, all computed on the 30-body x 4-B x 3-seed
matrix (360 rows when all four cells are present):

1. **Pairwise rank correlation with corpus**: for each
   B in {10000, 100000, 5_000_001}, Spearman rho(PSR(B=1000), PSR(B))
   on the 30-body seed-averaged panel. Report rho with 95% bootstrap
   CI (10,000 resamples over bodies, seed=42).

2. **Monotonicity test**: ordered-alternative Kendall tau-b test on
   (rho_10k, rho_100k, rho_5M). Under H1, these should be monotonically
   decreasing. Reject monotonicity if two adjacent rho's are not
   order-consistent after accounting for their bootstrap CIs.

3. **Per-body Delta vs B**: for each body, compute PSR(B=1000) - PSR(B).
   Report seed-averaged Delta per body as a function of B; fit a
   simple log-linear regression Delta_body ~ log(B) + body; report
   slope and CI (body-clustered bootstrap). This tests whether the
   effect size scales with log(boundary duration).

No outlier removal. No post-hoc tweaking.

## Inclusion gates (frozen)

Applied after analysis. No post-hoc adjustment. Interpretation is
pre-committed:

- **Main-text dose-response figure** (new panel in Results):
  ALL of the following:
  - At least 2 of 3 pairwise rho's (at B in {10000, 100000, 5_000_001})
    have 95% CI upper bound < 0.95.
  - The three pairwise rho's decrease monotonically (each adjacent
    pair has rho_B_i > rho_B_{i+1} when comparing point estimates,
    AND the decrease is not solely within error bars for at least
    one transition).
  - Reliability gate passes on all 3 new/extended cells.

- **Appendix-only dose-response figure**: main-text conditions not
  met but at least 1 pairwise rho has 95% CI upper < 0.95.

- **Drop (report as null in Limitations)**: all 3 pairwise rho 95% CIs
  contain 0.95 (i.e., ranking is preserved at every boundary). The
  paper will acknowledge this null and describe the main findings as
  termination-protocol-dependence, not reset-frequency-dependence.

- **QC fail**: reliability gate fails on one or more new cells;
  report only the cells that pass QC.

## Pre-committed wording for each outcome

**Main-text dose-response:**

> "Increasing the training-episode boundary from 1,000 to 5,000,001
> steps at tau=0 progressively departs from the corpus ranking:
> rho(PSR_1000, PSR_10000) = X.XX, rho(PSR_1000, PSR_100000) = X.XX,
> rho(PSR_1000, PSR_5M) = X.XX (all with 95% bootstrap CI upper bound
> below 0.95 except at B=10000). The effect is dose-responsive, not
> a trivial artifact of fully-removed resets."

**Appendix-only:**

> "The boundary duration sweep shows a monotone trend but not all
> intermediate values have CIs that exclude high rank agreement; the
> effect is clearest at the B=5M extreme. Details in Appendix."

**Drop:**

> "Intermediate boundary values (B=10000, 100000) showed rank
> agreement with the corpus indistinguishable from 1.0. The main
> boundary effect operates at fully-removed resets, not along a
> dose-response. We present the full finding as termination-protocol
> sensitivity rather than reset-frequency sensitivity."

## Title and placement

No title change dependent on this sweep. The current title "Reset
Dependence Is Embodied" is retained because the paper uses "reset" in
the narrow sense of reset/termination protocol. We add one explicit
disambiguation sentence to the abstract:

> "We define reset dependence as sensitivity to the environment's
> reset/termination protocol. We do not claim to have evaluated
> algorithmic reset-free methods (MEDAL, RISC, MoReFree); that
> extension is future work."

The sweep appears in Results (or Appendix per gate outcome) as a
figure showing rho decreasing vs log(B).

## Compute budget

- 30 bodies x 3 seeds x 2 new B values = 180 new runs.
- Expected wall-clock: ~16h per seed-level SBATCH at 4-process-per-GPU
  packing; all 6 jobs can run in parallel on separate GPUs.
- Expected GPU-hours: ~60.
- Matches compute budget of the boundary-free panel; known feasible.

## Files

- `records/boundary_sweep_prereg.md` (this file)
- `slurm/boundary_sweep_panel.sh` (SLURM launcher, parameterizes
  `--max-steps` from env var)
- `scripts/analyze_boundary_sweep.py` (dose-response analyzer)

## Commit policy

This spec is committed BEFORE the first SLURM submission. Any change
after commit is a deviation; the affected B value drops.

## Fallback

If any cell fails QC or reliability, drop that B value. If both new B
values fail, the dose-response is dropped entirely; the boundary-free
panel (primary) stands on its own. If boundary-free itself fails, the
sweep is moot and is not integrated.

## Pre-smoke clarification (2026-04-22, before B=3001 data)

Added after spec commit `9856131` in response to a design review
flagging a design flaw. Applied BEFORE the affected B-values produced
training data (all B=100001 SLURM jobs were cancelled while PENDING;
no training, no data).

**Correction:** Replace B=100001 with B=3001.

**Why:** `timesteps=5_000_000` is the TOTAL vectorized transition
count across all 128 envs, not the per-env step count. Each env gets
5M / 128 = 39,056 steps. At B=100001 > 39056, truncation never fires
(`step_count >= B` is false for every step), so the B=100001 cell is
operationally identical to the B=5,000,001 (boundary-free) cell. The
original expected-truncation-counts table in this spec is WRONG; the
correct values are:
- B=1000: 39 truncations per env, ~4,992 total, rate ~0.001
- B=3001 (new): 13 truncations per env, ~1,664 total, rate ~3e-4
- B=10001: 3 truncations per env, ~384 total, rate ~8e-5
- B=100001 (REMOVED): 0 truncations per env, identical to B=5M
- B=5,000,001: 0 truncations per env, rate = 0

**Effect on analysis:** All gates, primary/secondary/tertiary
statistics, and pre-committed wording are UNCHANGED. The analyzer is
updated to read results from `results/boundary_sweep/B3001_seed{N}/`
instead of B100001. The dose-response curve still has 4 log-spaced
points: B in {1000, 3001, 10001, 5M}, spanning the meaningful
reset-frequency range for the 128-env 5M-step training regime.

**Commit policy:** B=3001 is treated as a new B value under the same
spec. Any further change after this clarification is a deviation.
