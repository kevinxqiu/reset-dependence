# Reproducing the paper

This guide details the steps of the [README](../README.md#installation): the
data, the numbers script, the other analysis scripts, the figures, Croissant
validation and retraining. Every step except retraining runs on a CPU. All
commands run from the repository root with the analysis environment of the
README active (Python 3.10 to 3.14, `requirements-analysis.txt`).

## 1. Data and body files

The run summaries are archives attached to [release v1.1.2](https://github.com/kevinxqiu/reset-dependence/releases/tag/v1.1.2)
of this repository (CC BY 4.0, data card in `dataset/README.md`).

| File | Size | Needed for |
| --- | --- | --- |
| `runs.zip` | 2.4 MB | Every analysis and figure. Unzips into `results/`, `tables/`, `sim-records/` and `gait-traces/` |
| `logs.zip` | 9.7 MB | The training logs that were kept, and the log counts of the numbers script. Unzips into `results/<panel>/logs/` |
| `panels.zip` | 1.1 MB | The review-period control panels (Humanoid, Ant, Evolution Gym), for `scripts/review_panels.py` and the panel rows of the numbers script. Unzips into `review-panels/` |
| `SHA256SUMS` | | Checking the three archives |

```bash
sha256sum -c SHA256SUMS
unzip runs.zip
unzip logs.zip     # kept training logs, into results/<panel>/logs/
unzip panels.zip   # review-period panels, into review-panels/
```

The 100 UNIMAL body files are not redistributed, because their upstream
repositories declare no license. A script downloads them from the link in the
MetaMorph README and checks each file against
`assets/unimals_100/SHA256SUMS`:

```bash
python scripts/fetch_unimal_assets.py
# -> assets/unimals_100/xml/*.xml, 100 of 100 files verified
```

If Google Drive refuses the direct download, fetch the archive with
`pip install gdown && gdown 1LyKYTCevnqWrDle1LTBMlBF58RmCjSzM` and run
`python scripts/fetch_unimal_assets.py --zip unimals_100.zip`.

## 2. The numbers script

```bash
python scripts/camera_ready_numbers.py --out-json numbers.json --out-md numbers.md
python scripts/check_numbers.py numbers.json
```

The script recomputes 2,401 values from the run summaries, the kept training
logs and the per-cell tables of the review-period panels, and writes them at
full precision to `--out-json` and with one row per value to `--out-md`
(by default it overwrites the recorded `records/camera_ready_numbers.json` and
`.md`, so `git diff records/` shows any change). It takes about 5 minutes on a
laptop CPU. Its random draws are seeded: body bootstraps use B = 10,000 with
`numpy.random.default_rng(42)`, and permutation tests use B = 10,000 with a
fresh `default_rng(20260923)` per test. `check_numbers.py` therefore requires
every value to match the recorded one to a relative tolerance of 1e-9, which
absorbs the last-digit differences between NumPy and SciPy versions; the
MuJoCo re-run rows may also differ by up to 1e-4 m in their height
differences between MuJoCo versions. Without the unzipped `logs.zip` or
`panels.zip` the script skips the `logs.*` or `panels.*` rows and says so, and
`--logs logs.zip --review-panels panels.zip` reads both archives without
unzipping them.

The script also re-runs the zero-action MuJoCo rollouts of all 100 bodies
from the XML files. `--no-physics` skips them, and without the XML files the
duplicate-file check is skipped as well. The `.md` output also compares 414
values from the results summary and draft text that the paper was written
from with the recomputed ones, and explains each difference (Monte Carlo
streams, rounding or definitions). Its last section lists, in 14 groups, the
numbers that the script does not recompute, and where each group comes from:

- review-period panels: the bootstrap intervals of ρ and the top-10 overlaps
  (recomputed by `scripts/review_panels.py`, section 3), the counts of bodies
  with negative reward, the excess disagreement D and its 33-comparison
  adjustment, and the panel settings
- the structure fits beyond the in-sample ones (leave-one-out,
  leave-one-family-out, crossfit) and the transfer test on non-evolved bodies
- the outcomes of the other planned tests, taken from their recorded analyses
  in `records/`
- the SAC scale check before launch and the choice of the 24 SAC bodies
- fall rules in public code, the Gymnasium healthy ranges and the source of
  the 100 bodies
- training, evaluation and probe settings
- launch history, dates and software versions
- decision rules and designs of the planned tests
- the paused replication and the unfinished 50M-step phase of the
  longer-budget panel
- the crossing steps under the training reset noise
- statements about Monte Carlo variation between random streams
- definitions and analysis constants
- the A2C panel
- identifiers printed with the references and acknowledgments

## 3. Other analysis scripts

The analyses of the reviewed version and of the planned tests have their own
scripts. Each writes a JSON record into `records/`, so `git diff records/`
shows any change. Apart from timing and provenance fields, package versions
other than the recorded ones move only the last digits of floating-point
values (relative differences of 1e-12 or less in our runs).

```bash
python scripts/validate_data.py                     # integrity checks of the run summaries
python scripts/reproduce_paper_stats.py             # statistics of the reviewed version
python scripts/compute_regret_common_eval.py        # selection regret (also --eval-key eval_neutral, eval_tau050)
python scripts/analyze_sac_full_panel.py            # SAC panel test
python scripts/analyze_boundary_free.py             # no-time-limit test
python scripts/analyze_boundary_sweep.py            # time-limit durations (about 5 minutes)
python scripts/compute_robust_selection.py
python scripts/permutation_interaction.py           # about 5 minutes
python scripts/structural_feature_regression.py
```

### Review-period panels

The control panels in other design spaces (paper, Section 4.8 and Appendix P.1)
ship in `panels.zip`, one row per cell with the score each frozen report
ranks, together with the reports, specifications and inventories. Each panel
compares the rankings under two settings with the rankings from two disjoint
halves of one setting's seeds. The script recomputes every statistic in those
reports (Spearman ρ, bootstrap intervals with the reports' seeds, top-10
overlap, seed-half correlations and cell counts) and compares each value with
the shipped report files:

```bash
python scripts/review_panels.py     # 4 to 6 minutes, reads review-panels/ from section 1
# -> records/review_panels_check.json
```

The recorded check compares 19,619 values, all identical. The reports also
contain permutation p-values. The paper does not use them, because they
respond to differences in reward level between two settings rather than to
rank agreement (Appendix P.1). The excess disagreement D that the paper reports
for these panels, and its 33-comparison adjustment, are not recomputed by the
released scripts.

[`scripts/README.md`](../scripts/README.md) lists every script by purpose,
and [`records/README.md`](../records/README.md) maps each planned test to its
specification, report and analysis output.

## 4. Figures

`scripts/figures/make_figures.py` draws every figure of the paper from the
run summaries, the joint traces and the UNIMAL body files, and writes them to
`figures/` as `.pdf` (and `.png` where the script writes one). All of the
plotting code is in `scripts/figures/`.

The plots set their text in STIX, a Times design that ships with matplotlib.
The four renders label their panels in Lato, bundled in `fonts/` (SIL Open
Font License, `fonts/OFL.txt`). Both match the printed figures.

```bash
python scripts/figures/make_figures.py            # Figures 2 to 9 and 12, about 1 minute
python scripts/figures/make_figures.py --render   # also Figures 1, 10, 11 and 13
```

| Figure | Script (in `scripts/figures/`) |
| --- | --- |
| 1. Floor A under the three training rules (one illustrative retrain) | `render_chronophoto_3d.py` (render) |
| 2. Rule schematic | `plot_termination_mechanism.py` |
| 3. Family results | `analyze_final.py` |
| 4. Reset-free against Strict, per body | `plot_rf_vs_strict_scatter.py` |
| 5. Torso trajectories of the illustrative retrains | `render_temporal_gait.py` |
| 6. Evaluator robustness | `plot_robustness_eval.py` |
| 7, 8, 9. Threshold curves, ratio distributions, per-seed panels | `analyze_final.py` |
| 10. Morphology gallery | `render_morphology.py` (render) |
| 11. Most rule-specialized VT bodies | `render_top1_row.py` (render) |
| 12. Training without a time limit | `render_boundary_free_forest.py` |
| 13. MVT A and Floor B under the three training rules (illustrative retrains) | `render_chronophoto_3d.py` (render) |

The four `--render` figures draw full MuJoCo scenes offscreen. `make_figures.py` uses EGL (a GPU or
Mesa) unless `MUJOCO_GL` is set; use `MUJOCO_GL=osmesa` on a machine without
EGL. The printed plots (Figures 2 to 9 and 12) were drawn with these scripts
on a laptop with Mesa EGL. On that laptop the renders match the printed ones
in bodies, poses, camera and labels, and differ in shading and in the grid
lines of the floor. Figures 1, 5 and 13 read
the joint traces in `gait-traces/` from the dataset. Their nine controllers
were retrained once for illustration with later code (seed 42, deterministic
action), their checkpoints are not released, and the recorded corpus runs of
these bodies are the ones the statistics use (data card, `dataset/README.md`).

## 5. Croissant metadata

`dataset/croissant.json` describes the dataset in Croissant 1.1 with the
Responsible AI fields. Validate it in a separate environment, because
`mlcroissant` 1.1.1 needs pandas below 3:

```bash
python3 -m venv .venv-croissant && . .venv-croissant/bin/activate
pip install -r requirements-croissant.txt
mlcroissant validate --jsonld dataset/croissant.json
python dataset/load_croissant_records.py --archives <dir with runs.zip, logs.zip and panels.zip>
```

The last command loads every record of the nine record sets from local copies
of the archives: 3,306 runs, 160 feature rows, 5,860 cells of the
review-period panels and 16,200 evaluation points of the two Humanoid panels.

## 6. Retraining

Retraining is not needed to check the results. It needs a CUDA GPU and the
pins in `requirements-training.txt` (Python 3.12, JAX 0.9.2 with CUDA 12,
MuJoCo MJX 3.6.0, Brax 0.14.2). One run of 5M environment steps took about
one hour on our GPUs, and the 900-run corpus took about 980 GPU-hours.

```bash
python3.12 -m venv .venv-train && . .venv-train/bin/activate   # separate from the analysis environment
pip install -r requirements-training.txt
python scripts/train_mjx.py --threshold 0.20 --seed 42 --start_idx 0 --end_idx 1 \
    --timesteps 5000000 --num_envs 128 --unroll_length 16 --n_epochs 10 --minibatch_size 256 \
    --out_dir retrain/standard_seed42 --save-checkpoints
# -> retrain/standard_seed42/results.json; a new --out_dir keeps the released results/ apart,
#    because the trainer skips bodies that its output folder already holds
```

`scripts/train_mjx.py` is the maintained trainer that the later panels used.
It differs from the trainer that produced the corpus, and its runs differ
from the corpus in absolute per-step reward and, after Strict training, in
ranking. `corpus_revision/` keeps the corpus-era trainer and launchers, with
only their site-specific lines edited, and [`slurm/README.md`](../slurm/README.md) maps every SLURM recipe
to the panel it produced. `scripts/train_sac_mjx.py` trains the SAC panel.

## 7. Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request with Python
3.10, 3.12 and 3.14. It installs `requirements-analysis.txt`, compiles every
script, runs the `--help` of the entry-point scripts and validates the
Croissant file. A second job downloads `runs.zip` from release v1.1.2 with the
workflow's token, runs the numbers script with `--no-physics` and checks its
values against the committed `records/camera_ready_numbers.json` with
`scripts/check_numbers.py`.
