# Third-party notices

## UNIMAL body files (not included)

The paper uses the 100 training bodies of the UNIMAL-100 benchmark of
MetaMorph, which MetaMorph chose from the UNIMAL bodies that DERL evolved:

- Agrim Gupta, Linxi Fan, Surya Ganguli and Li Fei-Fei. "MetaMorph: Learning
  universal controllers with transformers." ICLR 2022. Code and benchmark:
  https://github.com/agrimgupta92/metamorph
- Agrim Gupta, Silvio Savarese, Surya Ganguli and Li Fei-Fei. "Embodied
  intelligence via learning and evolution." Nature Communications 12, 5721
  (2021). Code: https://github.com/agrimgupta92/derl

Neither repository declares a license (no license file on its default branch, checked on 2026-10-07), so this
repository does not redistribute the body files. `scripts/fetch_unimal_assets.py`
downloads `unimals_100.zip` from the link given in the MetaMorph README,
extracts the 100 training bodies (`unimals_100/train/xml/`) into
`assets/unimals_100/xml/` and checks each against
`assets/unimals_100/SHA256SUMS`. The manifest was computed from the files used
to train the reported controllers; on 2026-10-07 the upstream files matched it
byte for byte. The MIT license of this repository does not apply to these
files.

The 60 non-evolved bodies of one panel were sampled with the UNIMAL generator
from the DERL code base. They are not distributed either; their structural
features and run summaries are in the dataset.

The scripts adapt the UNIMAL XMLs only in memory at load time (they enable
the commented-out floor plane); the files on disk stay byte-identical to the
upstream release.

## MuJoCo Menagerie robot models (not included)

The existing-robots panel (`scripts/menagerie_train_mjx.py`) loads four models
from MuJoCo Menagerie (https://github.com/google-deepmind/mujoco_menagerie),
which the user clones separately. They are not redistributed here. Each model
keeps the license in its Menagerie directory (checked on 2026-10-07):

| Model | Menagerie directory | License |
| --- | --- | --- |
| Unitree Go2 | `unitree_go2/` | BSD-3-Clause (Unitree Robotics) |
| ANYbotics ANYmal C | `anybotics_anymal_c/` | BSD-3-Clause (ANYbotics AG) |
| ToddlerBot 2XC | `toddlerbot_2xc/` | MIT |
| Unitree G1 | `unitree_g1/` | BSD-3-Clause (Unitree Robotics) |

## Software dependencies

Used under their own licenses, installed from PyPI and not bundled. Checked on
2026-10-07 against the PyPI metadata of the pinned or resolved version (SPDX
expression, license field or classifier) and, where PyPI states none, the
license file of the upstream repository:

| Package (version checked) | License |
| --- | --- |
| mujoco 3.6.0, mujoco-mjx 3.6.0 | Apache-2.0 |
| jax, jaxlib, jax-cuda12-plugin, jax-cuda12-pjrt 0.9.2 | Apache-2.0 |
| flax 0.12.6, optax 0.2.8, orbax-checkpoint 0.11.33, brax 0.14.2 | Apache-2.0 |
| numpy 2.4.4 (training), 1.26.4 (analysis) | BSD-3-Clause; the 2.4.4 metadata reads `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` for the components it vendors |
| scipy 1.17.1, pandas 3.0.6, statsmodels 0.15.0 | BSD-3-Clause |
| matplotlib 3.10.9 | Matplotlib License (based on the PSF license, BSD-compatible), not a BSD license |
| imageio 2.37.3 | BSD-2-Clause |
| imageio-ffmpeg 0.6.0 | BSD-2-Clause for the package; the ffmpeg executable inside its binary wheels is a separate program built with `--enable-gpl --enable-version3` (FFmpeg 7.0.2 static build in the Linux x86-64 wheel), so that binary is under GPL-3.0-or-later. It is only called as a subprocess to write videos and is not distributed with this repository. |
| tensorboardX 2.6.5 | MIT |
| mlcroissant 1.1.1 (Croissant checks only) | Apache-2.0 (license file of github.com/mlcommons/croissant; the PyPI metadata states none) |

The review-period panels described in the paper also used the Gymnasium Ant
and Humanoid models (gymnasium, MIT), Evolution Gym (evogym, MIT) and
Stable-Baselines3 (MIT). Their training code and raw run records are not
released. Their per-cell tables and frozen reports are in `panels.zip` of the
dataset, and `scripts/review_panels.py` needs only numpy and scipy.

## Fonts

The labels of the four render figures (Figures 1, 10, 11 and 13) and of the
rollout videos use the Lato typeface by Łukasz Dziedzic (tyPoland), released
under the SIL Open Font License 1.1 with Reserved Font Name "Lato". The four
weights they use (normal, italic, bold and semibold) are bundled unmodified in
`fonts/`, with the license in `fonts/OFL.txt`. The other figures use the STIX
fonts that ship with matplotlib.
