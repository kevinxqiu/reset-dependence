# NEMS (Non-Evolved Morphology Substrate): Pre-Registration

**Status:** frozen 2026-04-16, BEFORE any training runs launch.

## Motivation

The paper's Limitation (1) explicitly notes that the three UNIMAL families
(Floor / VT / MVT) "emerge from DERL's evolutionary process, which confounds
morphology with evolutionary history and terrain curriculum." A critic
will reasonably ask whether the protocol-by-family interaction reflects
morphology or DERL's evolutionary pressure.

NEMS addresses this by running the same protocol sweep on 60 UNIMAL-substrate
morphologies sampled from the pre-DERL-selection random-initialization
population at `assets/generated_morphologies/`. These bodies share
the UNIMAL parametric generator but have NEVER been subject to fitness-based
selection.

**Claim scope.** NEMS removes DERL's evolutionary selection, lineage, and
terrain curriculum. It does NOT establish cross-space generalization beyond
the UNIMAL generator. The paper will flag this explicitly.

## Panel (frozen, deterministic)

Source: `assets/generated_morphologies/xml/` (400 valid XMLs).

Selection: `scripts/nems_select_bodies.py` at git revision (commit pinned
at launch). Stratified 2D on `(num_limbs x torso_mode)`:

- `num_limbs` in {3, 4, 5, 6, 7, 8} (6 levels)
- `torso_mode` in {vertical, horizontal_y} (2 levels)
- 5 bodies per (limbs x torso) cell = 60 bodies total
- `numpy.random.default_rng(seed=42)`; no cell below 5 bodies in the pool

9-limb bodies excluded (17 in pool, DERL throws most of them out, and the
smaller cell risks wrong-direction torso_mode imbalance).

Panel manifest: `records/nems_panel_bodies.json`
(names, structural features, and stratum tag).

## Training protocol

60 bodies x 3 taus (`{0.00, 0.20, 0.50}`) x 3 seeds (`{42, 43, 44}`) =
540 independently trained policies at 5x10^6 PPO steps each.

Hyperparameters are EXACTLY corpus-matched:

    NUM_ENVS=128, UNROLL_LENGTH=16, N_EPOCHS=10, MINIBATCH_SIZE=256,
    LR=3e-4, GAMMA=0.99, GAE_LAMBDA=0.95, CLIP_EPS=0.2,
    total_timesteps=5,000,000

(same as `scripts/train_mjx.py` argument defaults used by the corpus SLURM
pipeline). Any deviation MUST be logged; otherwise this panel re-runs.

Evaluation: 20 episodes at `tau_eval=0.20` (matched to corpus main-text) plus
`tau_eval=0.50` and `tau_eval=0` (neutral) for robustness. Per-step reward,
distance, mean ep length, and fall rate are logged per episode.

## Primary analysis (pre-registered)

**Primary endpoint**: does a structural-feature predictor of per-body
protocol sensitivity, trained on the 100-body UNIMAL corpus, transfer
zero-shot to NEMS?

Specifically:

1. Using the corpus, fit a linear regression
   `Str_minus_RF_PSR(m) ~ initial_height + num_first_level_limbs +
    tree_depth + num_joints + num_bodies` on the 100 UNIMAL bodies.
2. Extract the same five features for each NEMS body (from the raw XML).
3. Predict NEMS per-body `Str_minus_RF_PSR` and correlate prediction vs.
   observed via Spearman rho and Pearson r. Report 95% bootstrap CIs from
   B=10,000 NEMS-body resamples (`numpy.random.default_rng(seed=42)`).
4. Positive-transfer threshold: Spearman rho > 0.3 with 95% CI excluding 0.

Primary outcome is `Str_minus_RF = PSR_strict - PSR_RF` (a paired-delta
estimand that avoids the RF/Epi near-zero-denominator problem). Distance-based
outcomes report `distance_strict - distance_RF` in an identical frame.

**Secondary analyses:**

- 3-cluster K-means on `(num_limbs, torso_mode, initial_height,
  num_joints, tree_depth)` over the 60 NEMS bodies (standardized features,
  random_state=42). Report protocol-by-cluster two-way ANOVA with body
  blocking; `F` and partial `eta^2`. Clusters are descriptive.

- Threshold-free distance transitions: per (body, protocol) classify
  walks (>= 5 m) / stationary (< 1 m) under `tau_eval=0.20`, as in the
  main-text locomotion audit. Report cross-protocol transition counts by
  cluster.

- Continuous regression on `PSR_RF ~ PSR_epi + initial_height + num_limbs`
  (baseline-skill and size control, mirroring corpus Section 4.8).

## Interpretation decision rules (pre-committed)

- **Positive transfer + cluster interaction replicates:** claim morphology
  predicts protocol sensitivity independently of DERL evolution. Strongest
  outcome, appendix + main-text paragraph.

- **Positive transfer + cluster interaction does NOT replicate:** claim
  continuous structural predictors transfer (primary endpoint) but
  cluster-level family structure does not. Narrower but still supportive.

- **No positive transfer + no cluster interaction:** neither criterion is met;
  report the null, which narrows the main claim, and leave the evolutionary
  explanation untested. Still publishable.

All three are publishable. Interpretation is fixed BEFORE the training runs,
so there is no room for selective reporting after the fact.

## Explicitly excluded post-hoc moves

- **No re-stratification after the fact** based on learning outcomes.
- **No RF/Epi ratio as a primary estimand.** Ratios may be shown for
  continuity with corpus tables but inferential claims use paired deltas.
- **No body-level exclusions** based on poor learning ("couldn't walk").
  If a body is NaN due to training divergence, we flag it and drop
  that body's contributions; we do NOT drop bodies because they learned
  slowly.
- **No Gymnasium-bodies sanity panel as a secondary axis** (design review:
  semantically mismatched, n too small, not worth the GPU hours for what
  it would contribute).

## Compute budget

- 540 runs x ~0.3 GPU-hr packed = ~160 GPU-hr total.
- 9 SLURM jobs (3 tau x 3 seed), 4-way process packing per GPU.
- Each job: 15 morphs per process x 4 processes = 60 bodies, ~18 GPU-hr
  wall-clock per job.
- With 3-job concurrency on the common partition: ~54 hr wall-clock
  (~2.3 days).

## Deliverables

1. `results/nems_panel/tau{TAU}_seed{SEED}/results.json` (9 files).
2. `records/nems_panel_report.md`: all pre-registered statistics.
3. `paper/figures/fig_nems.pdf`: primary-endpoint scatter +
   descriptive cluster bar.
4. `paper/sections/appendix.tex`: new `sec:nems_appendix` subsection.
5. If primary endpoint succeeds, one paragraph in the main Results tying
   NEMS to the DERL concern in Limitation (1).
