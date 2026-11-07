# Scripts

Every script in this folder, grouped by purpose. Each line is taken from the
script's docstring, which gives the details, inputs and outputs. Run a script
with `--help` to see its options. All scripts except the trainers run on a
CPU; the MuJoCo renders need an offscreen OpenGL backend (EGL or OSMesa).

The paper uses later names for several panels than the scripts do. Some
scripts call the Standard rule (τ = 0.20) `lenient` or `episodic` internally, NEMS is the panel of non-evolved bodies,
Menagerie the panel of existing robots, the asymptotic panel the longer budget,
the boundary-free panel training without a time limit, the factorial panel
rule by time limit, alt termination the contact-based rule, and the
equivalence pilot the retraining with later code.

## Start here

| Script | What it does |
| --- | --- |
| `camera_ready_numbers.py` | Recomputes every number printed in the camera-ready paper that it covers and writes `records/camera_ready_numbers.{json,md}` |
| `check_numbers.py` | Compares a recomputation from `camera_ready_numbers.py` with the recorded `records/camera_ready_numbers.json` |
| `review_panels.py` | Recomputes every statistic in the reports of the review-period control panels (Humanoid, Ant, Evolution Gym) from `review-panels/` in `panels.zip`, and writes `records/review_panels_check.json` |
| `fetch_unimal_assets.py` | Downloads the 100 UNIMAL body files and checks their hashes |
| `validate_data.py` | Runs integrity checks on the run summaries of the 900-run corpus |

## Training

These need a CUDA GPU and `requirements-training.txt`.

| Script | What it does |
| --- | --- |
| `train_mjx.py` | Trains PPO controllers for UNIMAL bodies in MuJoCo MJX (the maintained trainer, see `corpus_revision/` for the corpus-era one) |
| `train_sac_mjx.py` | Trains SAC controllers in MuJoCo MJX with Brax |
| `menagerie_train_mjx.py` | Trains PPO controllers for the existing legged robots of the Menagerie panel |

## Analysis

| Script | What it does |
| --- | --- |
| `reproduce_paper_stats.py` | Recomputes the statistics of the reviewed version and writes `records/paper_stats_reproduction.json` |
| `compute_regret_common_eval.py` | Selection regret under one evaluator (common, neutral or Strict) |
| `compute_robust_selection.py` | In-sample analysis of ranking bodies by their worst reward across rules |
| `permutation_interaction.py` | Body-level permutation test of the rule by family interaction |
| `mixedlm_interaction.py` | Mixed-effects interaction coefficient with a random intercept per body |
| `structural_feature_regression.py` | Regression of protocol sensitivity on structural features of the bodies |
| `extract_morph_features.py` | Extracts structural features from the UNIMAL XML files and correlates them with protocol sensitivity |
| `extract_features_nems.py` | Extracts the same features for the 60 non-evolved bodies |
| `analyze_energy_proxy.py` | Control-effort metrics recovered from the recorded episode means |
| `analyze_locomotion_quality.py` | Threshold-free measures of locomotion quality for the 900-run corpus |
| `statistical_hardening.py` | Leave-one-body-out and leave-one-seed-out stability of the corpus-level statistics |
| `analyze_sac_full_panel.py` | Planned analysis of the SAC panel (three rules, three seeds) |
| `verify_sac_smoke.py` | Checks the planned criteria of the SAC smoke test before the panel was expanded |
| `analyze_boundary_free.py` | Planned analysis of the panel trained without a time limit |
| `analyze_boundary_sweep.py` | Planned analysis of the time-limit durations at τ = 0 |
| `analyze_factorial.py` | Planned analysis of the rule by time-limit panel |
| `analyze_alt_termination.py` | Planned analysis of the contact-based termination panel |
| `analyze_asymptotic.py` | Planned analysis of the longer-budget panel |
| `asymptotic_gate_analysis.py` | Compares the longer-budget panel with the same bodies in the 5M-step corpus |
| `equivalence_pilot_analysis.py` | Planned analysis of the 12-body retraining with later code |
| `analyze_menagerie_panel.py` | Analysis of the 60-run panel of existing robots |
| `analyze_nems_panel.py` | Analysis of the 540-run non-evolved panel against its specification |
| `analyze_nems_robust_selector.py` | Planned held-out test of the feature-based selector on non-evolved bodies |
| `analyze_protocol_robust_lopo.py` | Planned leave-one-rule-out test of selection across rules |

## Body selection

| Script | What it does |
| --- | --- |
| `pick_top_bodies.py` | Selects the top-k bodies under each training rule |
| `pick_visual_bodies.py` | Selects the nine bodies shown in the visual figures |
| `pick_boundary_free_bodies.py` | Selects the 30 bodies of the no-time-limit panel |
| `pick_asymptotic_panel.py` | Selects the 18 bodies of the longer-budget panel |
| `select_alt_termination_bodies.py` | Selects the 30 bodies of the contact-based termination panel |
| `nems_select_bodies.py` | Selects a stratified 60-body panel from the non-evolved UNIMAL bodies |
| `mechanism_morphologies.txt` | The 24 bodies of the SAC panel (a body list, not a script) |

## Figures

`figures/` holds the code that draws every figure of the paper, and
`figures/make_figures.py` runs it (docs/REPRODUCE.md, section 4). It is the
plotting code of the reviewed version, updated for the camera-ready. The plots
take their style from `figures/_paper_style.py`, and the four renders from
`figures/_render_style.py`.

| Script | What it does |
| --- | --- |
| `figures/make_figures.py` | Runs the scripts below that draw the paper's figures and writes `figures/` (add `--render` for the MuJoCo renders) |
| `figures/render_chronophoto_3d.py` | Figures 1 and 13: chronophotographs of the illustrative retrains (Floor A; MVT A and Floor B) |
| `figures/plot_seed_vs_rule_ranks.py` | Project-page figure: ranks after two seeds of one rule against ranks after two rules with the same seed, from the run summaries |
| `figures/render_rollout_videos.py` | Project-page videos: the same rollouts as Figures 1 and 13 in real time, one row per training rule (render; needs ffmpeg) |
| `figures/plot_termination_mechanism.py` | Figure 2: schematic of the three fall rules (not measured) |
| `figures/analyze_final.py` | Figures 3, 7, 8 and 9: family results, threshold curves, ratio distributions and per-seed panels, with their statistics |
| `figures/render_temporal_gait.py` | Figure 5: torso trajectories of the illustrative retrains, from the joint traces |
| `plot_cr_fig2_cutoffs.py` | Zero-action probes: where the fall cutoffs sit relative to each body's torso (Appendix C.1 reports the result; not in the paper) |
| `figures/plot_robustness_eval.py` | Figure 6: family retention under the common and the neutral evaluator |
| `figures/render_morphology.py` | Figure 10: the nine gallery bodies |
| `figures/plot_rf_vs_strict_scatter.py` | Figure 4: per-body Reset-free against Strict reward, as values and as ranks |
| `figures/render_top1_row.py` | Figure 11: the most rule-specialized VT body for each training rule |
| `figures/render_boundary_free_forest.py` | Figure 12: training without a time limit at τ = 0: rank agreement and family shift |
| `render_pareto.py` | Reward against fall rate and against control effort, with the Pareto front (not in the paper) |
| `_paper_style.py` | Figure style of `plot_cr_fig2_cutoffs.py` and the analysis scripts' plots |

## Rendering

These render MuJoCo scenes offscreen (`MUJOCO_GL=egl` or `osmesa`).

| Script | What it does |
| --- | --- |
| `render_top_bodies.py` | Renders the top-k bodies of each rule as a grid (not in the paper) |
| `render_rollout.py` | Renders a deterministic policy rollout to video and a still frame |
| `traces_only_rollout.py` | Runs a deterministic policy rollout and saves only its joint traces |

## Checks made during the study

These scripts are records of checks we ran while the study was under way.
They are kept so that the records and reports they wrote can be traced, and
the camera-ready paper does not rely on them unless it says so. Their
docstrings use the working vocabulary of the time (phases, pivots).

| Script | What it does |
| --- | --- |
| `audit_standard_threshold.py` | Checks whether each training fall threshold can act on each body (geometry, physics probes and recorded data). Its output, `records/standard_threshold_audit.json`, defines the 28 bodies and is read by `camera_ready_numbers.py` |
| `audit_standard_subset_stats.py` | Recomputes the Standard-based statistics on body subsets |
| `pivot_a0_audit.py` | Audits what the recorded metrics mean (fall rate and episode length under each evaluator) |
| `pivot_a1_verify.py` | Correlates recoverability measures with Reset-free retention |
| `pivot_a1b_feature_baseline.py` | Feature baseline regression for recoverability |
| `pivot_a2_geom_proxy_native.py` | Geometric recoverability proxies from MuJoCo rollouts |
| `pivot_c0_metric_robustness.py` | Checks whether the recoverability measure depends on the failure criterion |
| `pivot_c0_saturation.py` | Reanalysis of the feature baseline that accounts for saturated bodies |
| `pivot_c1_delta_reset_controls.py` | Early control regression, superseded and not cited (its docstring explains why) |
| `pivot_c3_overlap_precheck.py` | Overlap of two top-10 lists |
| `pivot_threshold_analysis.py` | Early model of reward as a function of τ for each body |
