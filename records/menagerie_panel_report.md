# Menagerie Cross-Embodiment Panel: Results

Descriptive external-validity probe per records/menagerie_task_spec.md. No cross-robot ANOVA. PSRs are NOT comparable across robots (scale depends on robot size); only within-robot ratios are meaningful.

## Per-robot PSR (seed-averaged, 5 seeds)

| robot | n_seeds | PSR_Epi | PSR_Str | PSR_RF | RF/Epi [95% CI] | Str/Epi [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| unitree_go2 | 5 | 0.112 | 0.079 | 0.043 | 0.38 [0.30, 0.47] | 0.71 [0.66, 0.77] |
| anybotics_anymal_c | 5 | 1.013 | 1.047 | 0.117 | 0.12 [0.06, 0.17] | 1.05 [0.88, 1.22] |
| toddlerbot_2xc | 5 | -0.001 | -0.001 | -0.018 | n/a | n/a |
| unitree_g1 | 5 | nan | nan | nan | n/a | n/a |

## Sign table

A robot is said to be hurt by a protocol if the 95% CI of the relevant ratio lies entirely below 1.0. Ratios are n/a when a reward is NaN or a seed's Standard reward is below 0.05.

| robot | RF hurts | Str hurts |
| --- | --- | --- |
| unitree_go2 | yes | yes |
| anybotics_anymal_c | yes | no |
| toddlerbot_2xc | n/a | n/a |
| unitree_g1 | n/a | n/a |

Figure: `figures/fig_menagerie_panel.pdf` shows the RF/Epi and Str/Epi bars with 95% CIs per robot.

Note (2026-04-18): the panel was stopped before the submission. Only `unitree_go2` and `anybotics_anymal_c` passed the training-quality check (`toddlerbot_2xc` earns a per-step reward near zero under every rule, and `unitree_g1` trained to NaN), and two robots are too few for the planned external-validity reading, so the robot panel was dropped. The paper lists it with the other planned tests (Appendix D).
