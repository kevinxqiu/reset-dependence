# Alt-Termination Pre-Registration (FROZEN 2026-04-18)

This document pre-registers the alt-termination experiment for the NeurIPS
2026 paper "Reset Dependence Is Embodied". It is frozen before any
training, smoke test, or analysis. No post-hoc changes are permitted; any
deviation forces a drop of the entire component.

## Motivation and hypothesis

The main corpus uses a *height-based* termination (tau in {0.00, 0.20,
0.50} is a fractional-initial-height threshold). A critic may argue
that the observed protocol-by-family interaction is an artifact of this
specific termination rule rather than a general property of "resets". This
experiment tests that concern by re-running the three protocols under a
conceptually-analogous *contact-duration* termination on a 30-body panel.

H0 (null): the protocol-by-family interaction under contact-duration
termination is not significant at p < 0.01, or the per-body Delta_reset
computed from this experiment does not rank-correlate with the height-
derived Delta_reset.

## Design

### Termination rule (frozen)

- Mode `height` (baseline): terminate when torso z < tau * z_0 (existing
  code path in `scripts/train_mjx.py`, unchanged).
- Mode `contact-duration` (new): terminate when torso body has had a
  contact with the ground plane for >= N *consecutive* simulator steps.
  - Contact definition: any MuJoCo contact whose geom1 or geom2 belongs
    to the torso body, with contact distance <= 0 (penetrating).
  - Counter reset: if on step t the torso has no ground contact, reset
    the counter to 0.
  - Counter increment: if on step t the torso has at least one ground
    contact, increment the counter by 1.
  - Termination: on step t, if counter >= N, emit done=True.
- Reward, observation, and action interfaces are unchanged from the main
  corpus. The only change is the termination predicate. No curriculum,
  no reward shaping.

### Mode levels (frozen)

Three levels map N to the three protocols:

| Protocol | Height analogue | Contact-duration N |
| -- | -- | -- |
| strict | tau=0.50 | N=1 (terminate on any ground contact of the torso) |
| episodic | tau=0.20 | N=50 (equivalent to ~1 second at dt=0.02) |
| reset-free | tau=0.00 | N=infinity (never terminate) |

Implementation uses `--termination-mode contact-duration
--contact-termination-steps {1, 50, 100000}` where 100000 is the RF
analogue (episodes are still truncated at MAX_STEPS as in the main
corpus).

### Panel (30 UNIMAL bodies, frozen)

Selection algorithm (deterministic, seed=42):

1. Load per-body `Delta_reset = PSR_episodic - PSR_reset_free` at common
   eval key `eval_tau020` from the 900-run corpus
   (`results/mjx_ppo_{0p00,0p20,0p50}_seed{42,43,44}/results.json`).
2. For each family F in {floor, vt, mvt}:
   a. Rank F-bodies by Delta_reset.
   b. Partition the ranked list into 5 quintile bins of near-equal size.
   c. Within each quintile, select 2 bodies using numpy Generator(seed=42);
      when the quintile has more than 2 bodies, use
      `rng.choice(bodies, size=2, replace=False)`.
3. Yield 30 bodies total (10 per family).

Output: `records/alt_termination_bodies.json` with per-body
family, Delta_reset, quintile-within-family, and the master seed=42.

### Seeds per cell

Three seeds: 42, 43, 44 (same as corpus).

### Training (unchanged from corpus)

Per-cell parameters follow the corpus settings exactly:
- MJX PPO, `--num_envs 128 --unroll_length 16 --n_epochs 10
  --minibatch_size 256`
- 5M total env steps
- Standard UNIMAL `locomotion-v0` environment (same reward, same
  observation as the main corpus)

Any deviation from these hyperparameters forces a drop of the affected
cell.

### QC gate (applied before analysis; frozen)

Panel-level QC:
- Valid-run count: a cell is valid if (a) results.json exists, (b) no
  NaN in any reported PSR field, (c) final eval PSR is a finite number.
- Include gate: at least 243 / 270 cells valid
  (>= 27 / 30 bodies with 3 modes each and >= 2 / 3 valid seeds per mode).
- Fail gate -> drop the entire component (no partial integration).

### Primary analysis (frozen)

Effect scale: per-body, per-cell reward = `eval_tau020 per_step_reward`
(matches main corpus).

Two primary statistics:
1. Protocol-by-family interaction on raw per-seed per-step reward, using
   the blocked design `reward ~ C(protocol) * C(family) + C(morph)`,
   same specification as the main-corpus 900-observation two-way ANOVA.
2. Rank agreement between contact-duration-derived Delta_reset
   (per-body: mean_psr_N=50 - mean_psr_N=infinity) and height-derived
   Delta_reset (per-body: mean_psr_tau020 - mean_psr_tau000) on the 30
   shared bodies, measured with Spearman rho.

Both numbers are computed on the same 30 bodies. No outlier removal.

### Inclusion gates (frozen; design review, round 2)

Applied after analysis, with no post-hoc tweaking:

- **Body-text promotion**: include in main results (1 paragraph in
  Results + abstract mention) if interaction p < 1e-3 AND Spearman rho
  >= 0.70.
- **Main inclusion**: include in Results (short paragraph) + appendix
  block if interaction p < 0.01 AND Spearman rho >= 0.60.
- **Appendix-only**: include in appendix (1 subsection, narrow scope)
  if interaction p < 0.01 AND 0 <= rho < 0.40 AND both direction checks
  below pass.
- **Drop**: drop entirely if p >= 0.01, OR 0.40 <= rho < 0.60, OR QC
  fails, OR termination semantics were modified after the first full
  launch.

Direction checks for the appendix-only gate:
- Floor family: strong-contact-termination (N=1) condition mean PSR
  should be below episodic-analogue (N=50) mean PSR, 95% CI excluding 0.
- VT family: N=1 condition mean PSR should be below N=50 mean PSR, 95%
  CI excluding 0. (Note: this is the VT-strict failure we expect from
  the main corpus; if VT does not show this under contact-N=1 either,
  the mechanism analogy has failed and we drop.)

### Treatment of borderline outcomes

"Borderline" is pre-defined as 0.40 <= rho < 0.60 with p < 0.01, and is
explicitly a **drop** condition. We do not claim mechanism
equivalence from an inconclusive result.

If the experiment drops (any reason), the drop is itself a result: we
add a 1-2 sentence Limitations note stating the contact-duration probe
ran with N=270 cells, did not produce a significant interaction (or did
not rank-agree at the pre-registered threshold), and deposit the analysis
JSON in the supplement as `records/alt_termination_null.json`.

## Files

- `records/alt_termination_spec.md` (this file)
- `records/alt_termination_bodies.json` (panel)
- `scripts/select_alt_termination_bodies.py` (body selector)
- `scripts/train_mjx.py` (adds `--termination-mode` flag; reward +
  observation unchanged)
- `slurm/alt_termination.sh` (batch script)
- `scripts/analyze_alt_termination.py` (gate-applier)

## Commit policy

This spec and the body JSON must be committed before the first smoke
test. Any change after commit is a deviation; the entire component
drops.

## Clarification note (2026-04-18, post-commit, disambiguation only)

Added after commit `50bc5f9` (spec freeze) and commit `366a404`
(implementation) in response to design review flagging two spec
ambiguities. These are disambiguations of terms the frozen spec did not
define precisely; they are NOT amendments to the gates, panel, or
termination rule.

### Time unit: "consecutive simulator steps"

The spec references "consecutive simulator steps" and "~1 second at
dt=0.02". Both refer to **outer env-step time**, not MuJoCo substep
time. The environment uses `FRAME_SKIP=4` so one outer env step = 4
MuJoCo substeps, and `dt_outer = 4 * 0.005 = 0.02` s. Therefore
`N=50 * dt_outer = 1.0` s, consistent with the "~1 second" cross-reference.
The implementation checks torso-floor contact once per outer step
(after `pipeline_step` completes all 4 substeps). Transient
contact/no-contact oscillations inside a single outer step are not
individually resolved; only the state at the end of the outer step is
sampled.

### Safety termination: `too_high`

The implementation retains the `too_high = z > z_max` termination rule
from the main corpus (see `scripts/train_mjx.py` `UnimalMJXEnv.__init__`
where `z_max = max(3 * init_z, 2.0)` is set unconditionally). This
"safety" termination fires only if the torso rises above three times
its initial height, which is a recovery-from-launch failure case. The
corpus reset-free condition (tau=0.00) has the same `too_high` rule, so
retaining it in contact-duration mode preserves apples-to-apples parity
between the two "reset-free analogue" conditions (corpus tau=0.00 and
alt-term N=100000). Removing it would introduce a second difference
between corpus RF and alt-term RF beyond the pre-registered termination
predicate change, which would reduce interpretability. The spec line
"reward, observation, and action interfaces are unchanged from the main
corpus" is read as implying termination safety rails are also retained
unless the pre-registration explicitly replaces them, which it does
only for the primary contact-based predicate.
