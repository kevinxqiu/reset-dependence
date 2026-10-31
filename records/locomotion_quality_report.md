# Locomotion Quality Audit

Threshold-free distance-based classification at anchors 1.0 m and 5.0 m.

## Per-family counts

| family | n | epi_walks | epi_stat | rf_walks | rf_stat | str_walks | str_stat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floor | 41 | 41 | 0 | 27 | 12 | 40 | 1 |
| vt | 37 | 37 | 0 | 33 | 4 | 26 | 10 |
| mvt | 22 | 22 | 0 | 15 | 5 | 22 | 0 |

## Cross-protocol transitions

Bodies that walk under episodic (>= 5.0 m) but are stationary (< 1.0 m) under the other protocol:

| family | n | learns_then_stops_under_RF | learns_then_stops_under_Strict | stops_then_learns_under_RF | stops_then_learns_under_Strict |
| --- | --- | --- | --- | --- | --- |
| floor | 41 | 12 | 1 | 0 | 0 |
| vt | 37 | 4 | 10 | 0 | 0 |
| mvt | 22 | 5 | 0 | 0 | 0 |


## Threshold-free check: same table at tau_eval = 0 (no early termination)

At tau_eval = 0 every evaluation episode runs the full 1000 steps, so distance is strictly integrated forward motion with no threshold dependence.

| family | n | learns_then_stops_under_RF | learns_then_stops_under_Strict | stops_then_learns_under_RF | stops_then_learns_under_Strict |
| --- | --- | --- | --- | --- | --- |
| floor | 41 | 0 | 1 | 0 | 0 |
| vt | 37 | 0 | 10 | 0 | 0 |
| mvt | 22 | 0 | 0 | 0 | 0 |

## Interpretation

The cross-protocol transition table counts bodies that travel at least 5.0 m under episodic training and less than 1.0 m under reset-free or strict training, and the reverse. The counts depend on the evaluator (compare the tables above), so on their own they do not show whether training prevented locomotion.

## Figure

See `paper/figures/fig_locomotion_quality.pdf` for the distance-vs-PSR scatter faceted by protocol.