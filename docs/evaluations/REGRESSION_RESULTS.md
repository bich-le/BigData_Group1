# Regression model selection

Hyperparameters were selected by the lowest RMSE on the shared random 20% pre-OOT validation split (seed 42). CV metrics are reported but were not used for the global comparison because configurations used different fold counts. OOT was not used for selection.

| Config | Trees | Depth | Min leaf | Folds | CV RMSE | CV MAE | CV R² | Valid RMSE | Valid MAE | Valid R² | Source | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 300 | 12 | 5 | 3 | 33.013842 | 16.411696 | 0.268886 | 33.163359 | 16.429702 | 0.266990 | local | complete |
| 2 | 300 | 12 | 10 | 3 | 33.014929 | 16.411196 | 0.268837 | 33.165406 | 16.428974 | 0.266899 | local | complete |
| 3 | 300 | 15 | 5 | 3 | 32.789769 | 16.239171 | 0.278776 | 32.928726 | 16.243697 | 0.277325 | local | complete |
| 4 | 300 | 15 | 10 | 3 | 32.799201 | 16.237292 | 0.278362 | 32.931440 | 16.242870 | 0.277206 | local | complete |
| 5 | 300 | 18 | 5 | — | — | — | — | — | — | — | n/a | SKIPPED — computational constraint |
| 6 | 300 | 18 | 10 | — | — | — | — | — | — | — | n/a | SKIPPED — computational constraint |
| 7 | 500 | 12 | 5 | 1 | 33.080266 | 16.426421 | 0.268120 | — | — | — | local | incomplete_no_common_validation |
| 8 | 500 | 12 | 10 | 1 | 33.081906 | 16.425230 | 0.268047 | — | — | — | local | incomplete_no_common_validation |
| 9 | 500 | 15 | 5 | 3 | 32.839363 | 16.225607 | 0.278115 | 32.736849 | 16.256044 | 0.279717 | cluster-shard | complete |
| 10 | 500 | 15 | 10 | 3 | 32.848214 | 16.223505 | 0.277725 | 32.748402 | 16.256401 | 0.279208 | cluster-shard | complete |
| 11 | 500 | 18 | 5 | — | — | — | — | — | — | — | n/a | SKIPPED — computational constraint |
| 12 | 500 | 18 | 10 | — | — | — | — | — | — | — | n/a | SKIPPED — computational constraint |

## Selected winner

- 500 trees, depth 15, minimum 5 instances per node
- Validation RMSE: **32.736849**
- Validation MAE: **16.256044**
- Validation R²: **0.279717**

Configurations 7–8 have one-fold CV metrics but no common validation metrics, so they are not eligible for the validation-based ranking. Depth-18 configurations were skipped because Spark MLlib tree growth exceeded the available compute and memory budget.
