# Reset Dependence Is Embodied: run summaries (data card)

Run records behind the NeurIPS 2026 Evaluations and Datasets track paper
"Reset Dependence Is Embodied: Training Protocol Changes Which
Morphologies Appear Optimal" (Kevin Qiu, Zhao Yang, Marek Cygan). Together with the CPU scripts
in the code repository (https://github.com/kevinxqiu/reset-dependence),
these files recompute the statistics and figures that the paper reports for
its UNIMAL panels, without GPU training. The numbers script of the code
repository recomputes 2,401 values, and its report
(`records/camera_ready_numbers.md`) ends with the printed numbers it does not
recompute and their sources. Most of those are settings, dates, decision
rules, facts about public code and values quoted from the reviewed version.
The rest are results kept outside the run summaries, such as the
cross-validated structure fits, the outcomes of the other planned tests
(from their recorded analyses), the SAC scale check, the A2C panel and, for
the review-period panels, the excess disagreement D and its multiplicity
adjustment. `panels.zip` holds the control panels that we trained during the
review discussion in Humanoid, Ant and Evolution Gym bodies, with per-cell
tables from which `scripts/review_panels.py` recomputes every statistic of
their frozen reports (see "Review-period panels" below).

- Hosted as archives of release v1.1.2 of the code repository: https://github.com/kevinxqiu/reset-dependence/releases/tag/v1.1.2
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE`
- Version: 1.1.2
- Croissant metadata: `croissant.json` (Croissant 1.1 with the Responsible AI fields that the
  NeurIPS 2026 Evaluations and Datasets track asks for, validated with `mlcroissant validate`;
  `load_croissant_records.py` in the code repository loads every record from local copies of
  the archives, with the pins in `requirements-croissant.txt`)
- Contact: Kevin Qiu, University of Warsaw

## Contents

| File | Size | What it holds |
| --- | --- | --- |
| `runs.zip` | 2.4 MB | JSON run summaries (`results/`), flat CSV tables (`tables/`), simulation probe records (`sim-records/`) and gait traces (`gait-traces/`) |
| `logs.zip` | 9.7 MB | Training logs where they were kept (`results/<panel>/logs/`) |
| `panels.zip` | 1.1 MB | Review-period panels (`review-panels/`): per-cell tables, Humanoid evaluation curves, frozen reports, specifications, contracts and inventories |
| `croissant.json` | | Croissant metadata with the SHA-256 of the three archives and of the CSV tables inside them |
| `MANIFEST.sha256` | | SHA-256 of every file inside the three archives (`archive:path`) |
| `SHA256SUMS` | | SHA-256 of `runs.zip`, `logs.zip` and `panels.zip` |
| `LICENSE` | | CC BY 4.0 legal code |

Unzip the archives at the root of a clone of the code repository. The
scripts then find `results/`, `sim-records/`, `gait-traces/` and
`review-panels/` at their default paths.

## How the data were produced

The 100 UNIMAL bodies are the training set of MetaMorph's UNIMAL-100
benchmark (Gupta et al., ICLR 2022), which MetaMorph chose from the bodies
that DERL evolved (Gupta et al., Nature Communications 2021). They come in
three families named after the environment DERL evolved them in: Floor (41,
flat terrain), VT (37, variable terrain) and MVT (22, manipulation in
variable terrain). We trained a controller for each body on forward
locomotion on flat ground, reward `v_x - 0.001 |a|^2` per control step, no
alive bonus, 1,000-step episodes, for 5M environment steps with a GPU PPO
implementation in MuJoCo MJX (SAC for one panel). A fall rule ends a
training episode when the torso height `z_t` drops below `tau * z0`, where
`z0` is the body's reset height: Reset-free (`tau = 0`, no fall clause),
Standard (`tau = 0.20`, written `0p20` in folder names) and Strict
(`tau = 0.50`), plus `tau = 0.10` and `0.35`. Seeds are 42, 43 and 44 unless
stated.

Each trained controller was evaluated on CPU MuJoCo over 20 episodes
(fixed reset-noise seeds; PPO actions sampled, SAC actions deterministic) by three evaluators, stored in every run
summary:

| Key | Evaluator | Episode also ends when |
| --- | --- | --- |
| `eval_tau020` | common | torso below 0.20 z0 |
| `eval_neutral` | neutral | never (no fall clause) |
| `eval_tau050` | Strict | torso below 0.50 z0 |

Every evaluator also ends an episode above `max(3 z0, 2 m)` or at 1,000
steps. Each evaluator block holds the mean and standard deviation over the 20
episodes of:

| Field | Meaning |
| --- | --- |
| `per_step_reward`, `per_step_reward_std` | Episode return divided by episode length (PSR), the paper's main score |
| `fitness`, `fitness_std` | Episode return |
| `distance`, `distance_std` | Final x position of the torso (m); episodes start at x = 0 |
| `mean_ep_len`, `ep_len_std` | Episode length in control steps |
| `fall_rate` | Fraction of the 20 episodes that ended before 1,000 steps |

The fixed-horizon return of the paper is `fitness / 1000`.

The simulation probe records were made on CPU on 2026-10-05 with MuJoCo:
zero-action rollouts of all 100 bodies, 3 s settle heights, random-action
searches on the ten VT bodies that the Strict cutoff ends during the reset
drop, and training-like torso probes that define the 60 probe-safe bodies of
the same-rule repeat. The scripts that made them are in the code repository
under `records/sim-record-scripts/`.

## Panels

| Directory in `results/` | Panel id in `tables/runs.csv` | Runs | Design |
| --- | --- | --- | --- |
| `mjx_ppo_{0p00,0p20,0p50}_seed{42,43,44}/` | `corpus` | 900 | 100 bodies, Reset-free, Standard, Strict, 3 seeds |
| `mjx_ppo_{0p10,0p35}_seed{42,43,44}/` | `intermediate_tau` | 600 | 100 bodies, tau 0.10 and 0.35, 3 seeds, four runs per GPU |
| `sac_mjx_{0p00,0p20,0p50}_seed{42,43,44}/` | `sac` | 216 | 24 bodies (`scripts/mechanism_morphologies.txt`), SAC |
| `equivalence_pilot/jax/<body>/tau*_seed*/` | `retraining` | 108 | 12 bodies retrained with a later code revision (planned test of 14 Apr 2026, criterion not met) |
| `boundary_free/seed*/` | `no_time_limit` | 90 | 30 bodies, Reset-free without a time limit (`max_steps` 5000001) |
| `boundary_sweep/B{3001,10001}_seed*/` | `time_limit_duration` | 180 | Same 30 bodies, Reset-free, time limits of 3,001 and 10,001 steps |
| `factorial/tau{0p20,0p50}_5M_seed*/` | `rule_by_time_limit` | 180 | Same 30 bodies, Standard and Strict without a time limit |
| `alt_termination/N{1,50,100000}_seed*/` | `contact_termination` | 270 | 30 bodies, an episode ends after 1, 50 or unlimited consecutive torso-ground contact steps |
| `nems_panel/tau*_seed*/` | `non_evolved` | 540 | 60 non-evolved bodies (`gen-*`) from the UNIMAL generator's initial population, 3 rules |
| `asymptotic_5m/<body>/tau*_seed*/` | `longer_budget_5m` | 162 | 18 bodies, 3 rules, 5M-step arm of the longer-budget panel (the 50M phase was not completed) |
| `menagerie_panel/<robot>_tau*_seed*/` | `existing_robots` | 60 | 4 existing legged robots, 3 rules, seeds 42 to 46 (panel stopped after its quality check) |

In all, 3,306 runs. Body lists for the panels are in the code repository
under `records/`.

Other files in `results/`:

- `morph_features.json`, `morph_features_nems.json`: structural features of
  the 100 UNIMAL and 60 non-evolved bodies (number of rigid bodies, joints,
  motors and geoms, reset height `init_height` = z0 in m, kinematic tree
  depth as the number of links on the longest chain below the torso, summed sphere and capsule volume in
  m^3, number of limbs on the torso). `morph_features.json` also stores the
  reviewed version's Reset-free/Standard and Strict/Standard PSR ratios
  (`rf_epi_ratio`, `str_epi_ratio`).
- `final_stats.json`, `feature_correlations.json`: outputs of
  `scripts/figures/analyze_final.py` as used in the reviewed version (rule
  names Standard, Strict, Reset-free).
- `pivot/`: outputs of early exploratory analyses (`scripts/pivot_*.py`),
  all post hoc.
- `equivalence_pilot/eq_a_analysis.json`: analysis output of the planned
  retraining test.
- `*/morph.txt`: the body name of a per-body directory.

`tables/runs.csv` has one row per run (3,306 rows) with the panel, the
source JSON file, body, family, algorithm, tau, seed, budget, termination
settings, the nine metrics of each evaluator (prefixed `common_`,
`neutral_`, `strict_`), wall time and, for later panels, the recorded code
commit, SLURM job id, pseudonymous host and GPU model.
`tables/morph_features.csv` has one row per body (160 rows).
`croissant.json` describes every column.

`sim-records/`:

- `all100_passive.json`: one entry per body: reset height `z0`, actuator
  count, simulation and control time steps, lowest point at reset, torso
  height at first floor contact under zero action, our Strict cutoff and
  DERL's cutoff (0.5 (z0 minus 0.2 m)), the control step at which the
  zero-action torso first crosses the Strict cutoff and first touches the
  floor, the torso height at step 50, the minimum height over the first 60
  steps, and a short random-action survival search.
- `settle.json`: one row per body: `[family, body, z0, torso height after
  3 s of zero action, minimum torso height during those 3 s]` (made by
  `records/sim-record-scripts/settle.py` in the code repository).
- `vt10_search.json`, `sk_shoot.json`: random open-loop action searches on the
  ten VT cut-off bodies (feasibility probes only).
- `probes/`: training-like torso probes (`train_like_probe_all_20.json`,
  `train_like_probe_near_100.json`, `train_like_probe_one_2.json`) and the
  resulting body lists (`placebo_safe60_bodies.json`, `near_margin_safe.json`).

`gait-traces/`: `qpos_trace` (501 x nq), `action_trace` (500 x nu), `done_step`,
`dt` and rollout settings of nine illustrative controllers (three bodies,
three rules, seed 42) behind the gait figure. These controllers were
retrained with later code and are not corpus runs.

## Review-period panels (`panels.zip`)

During the review discussion (25 July to 1 August 2026) we trained five
control panels outside the UNIMAL corpus, in parametric Humanoid and Ant
bodies and in Evolution Gym voxel bodies. Each panel changes one protocol
choice against a baseline, scores every controller with one common
evaluator, and compares the seed-mean rankings of the two settings within
each pool of bodies. Its reference for seed variation is the agreement
between rankings from two disjoint halves of one setting's seeds. A
specification and contract fixed each panel's cells, comparisons and
analysis settings, and a frozen analysis wrote its report. The paper reports
these panels in Section 4.8 and Appendix Q.1.

| Folder in `review-panels/` | What changes | Bodies, settings, seeds | Cells | Spearman rho (top-10 overlap) | Seed halves |
| --- | --- | --- | --- | --- | --- |
| `humanoid-fall-rule` | no fall rule against the native height range or cutoffs at 0.2 and 0.5 of the nominal root height | 3 pools of 30 parametric Humanoid bodies, 4 rules, 5 seeds | 1,800 | 0.58 to 0.73 (6 to 9) in pools 0 and 1, 0.11 to 0.19 (4 to 5) in pool 2 | 0.26 to 0.69 |
| `humanoid-time-limit` | 1,000-step time limit bootstrapped against terminal, with the native fall rule and without one | the same 90 bodies, 2 settings per fall rule, 5 seeds | 900, plus the 900 native and no-rule cells of `humanoid-fall-rule` | 0.87 to 0.92 (8 to 9) with the native rule, 0.65 to 0.74 (6 to 7) without | -0.13 to 0.61 |
| `ant-reset-init` | initial states with reset noise against the nominal initial state | 20 parametric Ant bodies, 2 settings, 4 seeds | 160 | 0.99 (9) | 0.90, 0.93 |
| `ant-breadth` | deterministic resets, terminal time limits or observation noise, each against the native setting | 3 pools of 30 Ant bodies, 4 settings, 5 seeds | 1,800 | 0.85 to 0.97 (7 to 10) | 0.58 to 0.85 |
| `evogym-time-limit` | 500-step time limit bootstrapped against terminal, on Walker-v0 and BridgeWalker-v0 | 3 pools of 20 Evolution Gym voxel bodies, 2 settings per task, 5 seeds | 1,200 | 0.89 to 0.98 (8 to 10) | 0.79 to 0.98 |

Rho is the Spearman correlation between the seed-mean rankings of the two
settings, per pool (and task), with the number of shared top-10 bodies in
parentheses; each report also gives its 95% morphology-bootstrap interval.
Seed halves is the median correlation between rankings from complementary
halves of one setting's seeds (two against three seeds, or two against two
in `ant-reset-init`), as a range over settings and pools. Humanoid and Ant
were trained with Brax PPO in MuJoCo MJX for 20M steps on GH200 or H100
GPUs, Evolution Gym with Stable-Baselines3 PPO for 2M steps on CPU.

What the panels show: in the Humanoid time-limit, Ant reset-initialization,
Ant breadth and Evolution Gym panels, the rankings under the two settings
agree about as well as two halves of one setting's seeds do, or better. Rho
is 0.85 to 0.99 in the Ant and Evolution Gym panels, against seed halves of
0.58 to 0.98, and 0.65 to 0.92 for the Humanoid time limit, against seed
halves of -0.13 to 0.61. Within that panel, agreement is lower without a
fall rule (0.65 to 0.74) than with the native rule (0.87 to 0.92), in line
with the lower agreement between seeds without a fall rule. In the Humanoid
fall-rule panel, two halves of one rule's seeds agree at only 0.26 to 0.69,
and rankings under a fall rule agree with rankings without one at 0.58 to
0.73 in pools 0 and 1 and at 0.11 to 0.19 in pool 2. After adjustment for
multiple comparisons, no excess disagreement D in this panel stays above
zero (paper, Appendix Q.1), so the panel is too noisy to resolve a fall-rule
effect.

Each folder holds:

- `cells.csv`: one row per cell (morphology, protocol, seed, hardware
  class). `score` is the value the report ranks: for Humanoid and Ant the
  mean return of the final evaluation over 128 episodes of up to 1,000 steps
  with the common evaluator (no fall termination, no healthy reward), for
  Evolution Gym the mean return over 20 evaluation episodes of up to 500
  steps. The other columns give the protocol settings, evaluation details, code
  revision, SLURM job id and the SHA-256 of the raw result, configuration and
  body files (described in `croissant.json`). The Ant breadth table has only
  the score, settings, revision, step counts and file hashes.
- `progress.csv` (Humanoid panels): the mean evaluation return of every
  cell at its six evaluation points; the last one equals `score`.
- `report.json`, `report.md`: the frozen report. Per pool and comparison:
  Spearman rho with its 95% morphology-bootstrap interval (10,000 resamples,
  seeded), top-10 overlap, winner regret, seed-split medians, per-body means
  and seed scores, and paired permutation p-values. The Humanoid time-limit
  report is a composite that pairs the bootstrapped arm in
  `humanoid-fall-rule` with the terminal arm and adds the difference of the
  fall-rule correlations and the rank stability over evaluation points.
- `manifest.json`: SHA-256 of the report files and of the contract, written
  by the analysis. `wrapper-verdict.json` (Humanoid time limit): record of
  the sandboxed analysis run.
- `panel_spec.json`, `panel_contract.json`, `inventory.jsonl`: the panel
  specification (cells, comparisons, analysis settings, body selection), the
  contract that bound the analysis to it and to the code revisions, and one
  line per cell with its protocol settings. `composite_spec.json` (Humanoid
  time limit) fixes how the two arms are paired.
- `frozen_report.md` (Ant breadth): the frozen summary table of 1 August
  2026. The generated report JSON of this panel was not kept.

The reports also contain paired protocol-label permutation p-values and
their Holm adjustments. We withdrew them on 2 August 2026 and do not use
them: that test also responds to differences in reward level between two
settings, so a large p-value does not show that rankings agree. They stay in
the reports so that the reports are complete.

`python scripts/review_panels.py` in the code repository (CPU, a few
minutes) recomputes every statistic and cell count of the five reports from
`cells.csv` and `progress.csv`, with the same estimators and random seeds as
the frozen analysis, and compares each value with the shipped report (for
the Ant breadth panel, with the three-decimal summary table). All 19,619
compared values are identical. It does not recompute the excess
disagreement D or its 33-comparison adjustment, which the paper reports for
these panels, and it does not count the Ant bodies with negative mean
reward, which follow from the per-body means it recomputes.

Processing: the tables copy a fixed list of fields from each run's
`config.json` and `result.json` (`evaluation.json` for Evolution Gym), with
no host names, paths or root binding tokens, and every score equals the one
in the panel's report (checked by the build). The Ant breadth scores and
file hashes were recovered on 24 September 2026 from a read-only copy of
that panel's records; the same procedure gives Ant reset-initialization
scores identical to its report. The reports are copied unchanged, except one
internal design-note path in the Evolution Gym `report.json` (also masked in
two specifications and the composite specification), so that file's SHA-256
differs from its manifest. The specifications are shipped without root
binding tokens and storage locations, as the reports embed them, so their
canonical hash cannot be recomputed from the copy; contracts and inventories
are unchanged and match the digests recorded in the reports.

Not included: the raw run records, the policy checkpoints and the generated
body files of these panels (the tables keep the SHA-256 of each raw result
and configuration file and of each body file), and their training code.

## Processing

Run summaries are the JSON files written by the training scripts. The only
changes are that GPU host names are replaced by stable pseudonyms
(`gpu-node-NN`), home directories in recorded paths are
masked (`~`), and the UNIMAL XML directory in recorded command lines reads
`<UNIMAL_XML_DIR>`. SLURM job ids, array task ids and code commits are kept.
Empty lock files are dropped. Logs get the same masking. The CSV tables are
flattened from the JSON files by the build script. The package is built by
`tools/build_public_release.py dataset` in the authors' repository, and
`MANIFEST.sha256` lists the hash of every file.

## Known issues and limitations

These are described in full in the paper (Sections 3.2 and 5 and
Appendices B.1, C.1 and D).

- The Standard cutoff (0.20 z0) lies below the 0.1 m torso radius for the 28
  bodies with z0 at most 0.45 m, so Standard training was in effect
  Reset-free training for them.
- The Strict cutoff ends PPO episodes during the drop from the reset pose for
  ten VT bodies; one further Floor body is cut off without an identified
  mechanism (paper, Appendix C.1).
- The corpus, intermediate-tau and SAC records do not store the code
  version, the full settings or the GPU. Later panels store them in
  `provenance`.
- No policy checkpoint was saved for any UNIMAL panel, so no reported
  controller can be reloaded. The code retrains the controllers of the 100
  evolved bodies.
- No training logs exist for the corpus or SAC runs; `logs.zip` holds the
  logs that were kept for the other panels.
- One task (forward locomotion on flat ground), one PPO preset and three
  seeds. The 100 bodies are MetaMorph's selection from the bodies that DERL
  evolved, not a random sample, and the family sizes are unequal.
- Not included: the UNIMAL body files (their upstream repositories declare
  no license; the code repository downloads and hash-checks them), the
  non-evolved body files, the raw records, checkpoints and body files of the
  review-period panels (`panels.zip` holds their per-cell summaries), and
  the A2C panel records.
- The review-period panels are small (20 to 30 bodies per pool, four or
  five seeds) and use other trainers, budgets and evaluators than the UNIMAL
  panels. The two arms of the Humanoid time-limit panel were trained with
  different code revisions (recorded in `git_revision`), and the Ant breadth
  panel's report JSON was not kept (its frozen summary table is shipped).

## Intended use

Recomputing and checking the paper's statistics, studying how
training-protocol choices change morphology rankings, and designing reporting
checks for co-design benchmarks. The data describe simulated locomotion only
and contain no personal information.

## Citation

```bibtex
@inproceedings{qiu2026reset,
  title     = {Reset Dependence Is Embodied: Training Protocol Changes Which Morphologies Appear Optimal},
  author    = {Qiu, Kevin and Yang, Zhao and Cygan, Marek},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026}
}
```

The UNIMAL bodies: Gupta, Savarese, Ganguli and Fei-Fei, "Embodied
intelligence via learning and evolution", Nature Communications 12, 5721
(2021); and Gupta, Fan, Ganguli and Fei-Fei, "MetaMorph: Learning universal
controllers with transformers", ICLR 2022.

## Changelog

- 1.1.2 (2026-11-06): first public release.
