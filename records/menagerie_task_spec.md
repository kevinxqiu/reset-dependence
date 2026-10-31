# Menagerie Cross-Embodiment Panel: Frozen Task Specification

Precommitted 2026-04-16, before any training run. Any deviation during
implementation must be logged as an amendment at the bottom of this file and
repeated in the paper's limitations section.

## Claim scope (frozen)

The Menagerie panel is a **descriptive external-validity probe**, not an
inferential generalization test. We do NOT claim the protocol-by-family
interaction is significant across Menagerie robots. We DO claim (if the data
supports it): the *direction* of each robot's protocol sensitivity (positive
or negative under RF/Epi, positive or negative under Str/Epi) is observable,
and the *sign pattern* is compatible with (or incompatible with) the UNIMAL
family reversal.

No cross-robot ANOVA. No cross-robot t-test on PSR magnitudes. Per-robot
RF/Epi and Str/Epi ratios with bootstrap 95% CIs over seeds, plus a
qualitative sign table.

## Pooling rule (frozen)

Menagerie PSRs are NOT pooled with UNIMAL corpus PSRs at any step of the
analysis. They do not appear in Table 1, Fig 1, or any corpus-level statistic.
The Menagerie section stands alone in the paper.

Rationale: the Eq-A equivalence gate against the corpus failed (rho=0.88 at
tau=0.20, slope outside [0.9, 1.1]); Menagerie runs use the same post-corpus
stack, so pooling would corrupt corpus-level statistics. This is already
disclosed in paper/sections/results.tex section Reproducibility and Scope of
Quantitative Claims.

## Robot selection (frozen)

4 robots, 2 quadrupeds and 2 humanoids, selected for the union of: (a) pre-
shipped `*_mjx.xml` or a clean MJX-compatible XML, (b) has a standing
keyframe OR a non-zero qpos0 standing pose, (c) covers a size range within
each morphology class.

| Slot | Robot | Class | nq | nu | Initial-z source | Standing z |
|---|---|---|---|---|---|---|
| 1 | unitree_go2           | quadruped | 19 | 12 | keyframe 0   | 0.27 m |
| 2 | anybotics_anymal_c    | quadruped | 19 | 12 | keyframe 0   | 0.56 m |
| 3 | toddlerbot_2xc        | humanoid  | 51 | 30 | keyframe 0   | 0.31 m |
| 4 | unitree_g1            | humanoid  | 36 | 29 | qpos0        | 0.793 m |

XML paths (relative to `~/mujoco_menagerie`):
- `unitree_go2/go2_mjx.xml`
- `anybotics_anymal_c/anymal_c_mjx.xml`
- `toddlerbot_2xc/toddlerbot_2xc_mjx.xml`
- `unitree_g1/g1_mjx.xml` (no shipped keyframe; qpos0 is a standing pose
  at pelvis z = 0.793)

Bodies that failed the MJX smoke test during precommit validation are
excluded: `unitree_h1` and `berkeley_humanoid` both raise
`NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_BOX) collisions not implemented`
under mjx 3.6.0, so they cannot be trained on our stack.

## Per-robot task semantics (frozen)

Common template:
- Forward axis: **world +x** (MuJoCo convention; matches UNIMAL reward).
- Reward: `r_t = v_x - 0.001 * ||a_t||^2`, where `v_x = (x_{t+1} - x_t) / dt`
  measured on the root body's world position, and `a_t` is the normalized
  action in `[-1, 1]^{nu}`.
- Episode length cap: 1000 environment steps.
- Control timestep: as set in each robot's MJX XML (unchanged).
- Termination at training threshold `tau`:
  - if tau > 0: `done = root_z < tau * z_init`, where `z_init` is the
    precomputed standing-z from the table above.
  - if tau == 0: no early termination.
- Action normalization: each actuator's output is clipped to the actuator's
  ctrlrange.
- Observation: `concat(qpos[7:], qvel[6:])`, i.e. skip the floating-base
  7D qpos and 6D qvel. Matches UNIMAL convention of dropping global x, y and
  the orientation quaternion.

### Root body per robot (the body used for v_x and z)

| Robot | Root body (MuJoCo body name) |
|---|---|
| unitree_go2        | `base` |
| anybotics_anymal_c | `base`  |
| toddlerbot_2xc     | `torso` |
| unitree_g1         | `pelvis` |

Verified by probing each XML with `mujoco.MjModel.from_xml_path`; the root
body is body id 1 (id 0 is world) in all four.

### PSR comparability disclaimer

Per-step reward magnitudes are NOT comparable across robots because forward
velocity scales with robot size (a go2 at 0.27 m standing walks at a very
different absolute speed than an h1 at 1.06 m standing). Only the
within-robot ratios RF/Epi and Str/Epi are cross-robot comparable; these are
the quantities the paper will report.

## Hyperparameters (frozen, match corpus)

`num_envs=128`, `unroll_length=16`, `n_epochs=10`, `minibatch_size=256`,
`learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`,
`timesteps=5_000_000`, policy MLP `[64, 64]`, value MLP `[64, 64]`, tanh
activation. 4-process-per-GPU packing on Entropy `common` partition.

## Physics settings (frozen)

MuJoCo defaults: integrator = Euler, solver = Newton, iterations = 100,
ls_iterations = 50. No fast-physics override. Rationale: the fast-physics
speedup (iter=1 ls=4) was benched as 2.54x on pure physics but never verified
end-to-end on PPO; dropping a new physics axis 20 days from deadline is
unnecessary risk. Keep the corpus physics to preserve any possible future
pooling if the equivalence-gate question is revisited.

## Logging (frozen)

Per-run `results.json` with the same keys as the corpus plus:
- `robot_name` (string)
- `robot_class` (string, `quadruped` or `humanoid`)
- `z_init` (float, the precomputed standing-z)
- `spec_version` (string, git SHA of this file's directory at run time)

In addition: a lightweight checkpoint every 1,000,000 steps
(`{out_dir}/checkpoints/{robot}_{tau}_{seed}_step{N}.npz`). This is a design review
requirement from plan review; it enables post-hoc learning-curve inspection
should the initial run suggest a training failure.

## Seeds (frozen)

`{42, 43, 44, 45, 46}`. 5 seeds, matching the Menagerie shortlist pre-spec
in `records/menagerie_shortlist.md`.

## Statistical reporting (frozen)

For each robot, report:
1. Mean PSR per training tau (and per eval tau), averaged across 5 seeds,
   with seed std.
2. RF/Epi and Str/Epi ratios (per-seed, then mean + bootstrap 95% CI over 5
   seeds, N=5; NOT pooled across robots).
3. Sign of RF/Epi - 1 and Str/Epi - 1: is protocol change harmful (ratio < 1)
   or beneficial (ratio > 1)?
4. Rank of the three training protocols within each robot (1-3), and
   agreement or disagreement with the UNIMAL finding.

No ANOVA. No pooled F-test.

Bootstrap CI method: percentile bootstrap, B=10,000 resamples of the 5-seed
vector, using `numpy.random.default_rng(seed=42)`.

## Negative-result branch (frozen)

If ALL 4 robots show the same sign of RF/Epi - 1 and Str/Epi - 1 (i.e. every
robot is uniformly hurt by reset-free training, or uniformly hurt by strict,
regardless of class), we do NOT claim the reversal generalizes. The paper
then frames the Menagerie panel as: "across four additional morphologies
from a curated commercial robot space, protocol sensitivity is observable
but without a morphology-class dichotomy comparable to UNIMAL's Floor vs
VT split. This suggests the evolutionary-history mechanism (UNIMAL Floor
bodies evolved under reset-rich conditions) is specific to DERL and does
not transfer to commercial robots trained offline."

## Amendments

### 2026-04-16: swap unitree_h1 -> unitree_g1

`unitree_h1` fails MJX 3.6.0 at the first `mjx.step` call with
`NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_BOX) collisions not implemented`.
`berkeley_humanoid` fails identically. `unitree_g1` passes MJX smoke on
CPU with qpos0 standing pose (pelvis z = 0.793 m). The swap was pre-
committed as the amendment path for a robot that fails MJX; the substitute
is the next valid humanoid in the shortlist that also loads via MJX and
has a usable standing pose.

Effect on claim: none. The revised panel still has 2 quadrupeds plus 2
humanoids, spanning the same range of scale and DoF. Updated table and
XML path above.
