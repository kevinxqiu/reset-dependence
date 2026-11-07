# Records

Small records that the scripts read or write, and the written record of each
planned test. Nothing here is needed to download; everything is text.

## Planned tests

For each test in the paper's register of planned tests: the specification
frozen before the full data were analyzed, the report and the analysis output.

| Test (frozen) | Specification | Report and analysis output |
| --- | --- | --- |
| Retraining with later code (14 Apr 2026) | `equivalence_pilot_bodies.json` | `equivalence_pilot_report.md` (analysis output in the dataset, `results/equivalence_pilot/eq_a_analysis.json`) |
| Structural transfer to non-evolved bodies (16 Apr 2026) | `nems_task_spec.md`, `nems_panel_bodies.{json,txt}` | `nems_panel_report.md`, `nems_analysis.json` |
| Existing robots (16 Apr 2026) | `menagerie_task_spec.md` | `menagerie_panel_report.md` |
| Longer budget (18 Apr 2026) | `asymptotic_spec.md`, `asymptotic_panel_bodies.json` | `asymptotic_analysis.json` |
| Contact-based termination (18 Apr 2026) | `alt_termination_spec.md`, `alt_termination_bodies.{json,txt}` | `alt_termination_report.md`, `alt_termination_analysis.json`, `alt_termination_null.json` |
| Leave-one-rule-out selection (19 Apr 2026) | `protocol_robust_lopo_spec.md` | `protocol_robust_lopo_report.md`, `protocol_robust_lopo_analysis.json` |
| No time limit (21 Apr 2026) | `boundary_free_prereg.md`, `boundary_free_bodies.{json,txt}` | `boundary_free_analysis.json` |
| Rule by time limit (22 Apr 2026) | `factorial_prereg.md` | `factorial_analysis.json` |
| Time-limit durations (22 Apr 2026) | `boundary_sweep_prereg.md` | `boundary_sweep_analysis.json` |
| Feature-based selector (23 Apr 2026) | `nems_robust_selector_spec.md` | `nems_robust_selector_decision.md`, `nems_robust_selector_analysis.json` |
| SAC panel (24 Apr 2026) | `sac_full_panel_spec.md`, `sac_full_panel_launch_log.md` | `sac_full_panel_analysis.json` |
| Temporal gait figure (22 Apr 2026) | `temporal_gait_prereg.md`, `temporal_gait_bodies.txt` | (figure only) |

The review-period panels (July 2026) are in the dataset: `panels.zip` holds
each panel's specification, contract, inventory, frozen report and per-cell
table, and `scripts/review_panels.py` recomputes the reports
(`review_panels_check.json` below). The A2C panel (2 August 2026) is not
part of the release.

## Cached outputs of the CPU scripts

Each file is rewritten when its script runs, so `git diff` after a run shows
any difference from the recorded output.

| File | Written by |
| --- | --- |
| `camera_ready_numbers.json`, `camera_ready_numbers.md` | `scripts/camera_ready_numbers.py` (every number of the camera-ready paper that it covers; input paths are shown relative to the repository root) |
| `review_panels_check.json` | `scripts/review_panels.py` (comparison of the recomputed review-period panel reports with the shipped ones) |
| `paper_stats_reproduction.json` | `scripts/reproduce_paper_stats.py` (statistics of the reviewed version) |
| `selection_regret_common_eval*.json` | `scripts/compute_regret_common_eval.py` |
| `validate_data_report.json` | `scripts/validate_data.py` |
| `permutation_interaction_report.json` | `scripts/permutation_interaction.py` |
| `mixedlm_interaction.json` | `scripts/mixedlm_interaction.py` |
| `structural_regression_coefs.json`, `structural_regression_report.json` | `scripts/structural_feature_regression.py` |
| `standard_threshold_audit.json`, `standard_threshold_subset_stats.json` | `scripts/audit_standard_threshold.py`, `scripts/audit_standard_subset_stats.py` (threshold geometry; read by the numbers and figure scripts) |
| `hardening_audit.{json,md}` | `scripts/statistical_hardening.py` |
| `robustness_checks_reviewed_version.json` | Recorded output of an earlier robustness analysis of the reviewed version; no script in this repository writes it |
| `energy_proxy.json` | `scripts/analyze_energy_proxy.py` |
| `locomotion_quality_report.md` | `scripts/analyze_locomotion_quality.py` |
| `protocol_robust_selection.json` | `scripts/compute_robust_selection.py` |
| `gait_rollout_summary.json` | `scripts/figures/render_temporal_gait.py` (recorded output; the script writes a fresh copy to `figures/`) |
| `a0_*` to `c1_*` reports | `scripts/pivot_*.py` (early exploratory analyses, all post hoc) |
| `top_bodies_per_protocol.json`, `visual_bodies_frozen.json` | `scripts/pick_top_bodies.py`, `scripts/pick_visual_bodies.py` (figure selections) |
| `termination_rules_public_code.md` | Hand-written survey of the fall rules in public co-design code |

## Simulation probe scripts

`sim-record-scripts/` holds the scripts that made the simulation probe
records in the dataset (`sim-records/`) on 2026-10-05, except the
300-sequence probe (`sim-records/sk_shoot.json`), whose script is not released. They are kept as a
record of how each file was made, not as maintained tools: they expect the
working directory and input files of that session.

## Edits relative to the original records

The records are copied from the project's working repository, with three
kinds of edits: absolute paths and private directory names were replaced by
repository paths or `~`; tool names in the design-review notes were replaced
by "design review"; and remarks on scheduling, presentation and submission
logistics, including wording about how the paper would present an outcome,
were removed or reworded. No gate, date or outcome was changed; where a
specification's interpretation of an outcome claimed more than its design
tests, the wording was corrected. Notes added for the release say where the paper reports each planned
test. Commit hashes in the records refer to the working repository, which is
not public.
