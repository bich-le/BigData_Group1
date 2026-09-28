# Production inference

Two native Spark `PipelineModel` artifacts are verified locally:

- classification: `models/random_forest_final_spark`;
- regression: `models/random_forest_regression_final_spark`.

Both have stages `VectorAssembler -> RandomForest{Classification,Regression}Model`.
The artifacts are excluded from Git because of their size.

## Input contract

Each JSON record contains optional string `event_id` plus the 23 ordered numeric
features in `feature_contract.py`. Airline, airport, and aircraft history/lag
features must be engineered upstream. Streaming inference does not derive them
from a raw flight event. Missing features are filled with zero, matching fitting.
No target or outcome field is required for inference.

Task-specific examples:

- `docs/contracts/sample_classification_stream_input.json`;
- `docs/contracts/sample_regression_stream_input.json`.

## Regression selection and final evaluation

Depth-18 configurations were skipped because their Spark MLlib tree growth was
computationally infeasible on the available machine. CV fold counts differed, so
the global winner was selected by RMSE on the common random 20% pre-OOT
validation split (seed 42). Standardized validation benchmarks confirmed that the supplied artifact is the exact winner fitted on the full 80% training split. A local `randomSplit` reconstruction produced different row membership, so the standardized benchmark metrics remain the authoritative validation record. OOT was reserved for final evaluation. See
`docs/evaluations/REGRESSION_RESULTS.md`, `docs/evaluations/regression_selection.json`, and `docs/evaluations/regression_evaluation.json`.

Winner: 500 trees, depth 15, minimum 5 instances per node.

- validation RMSE: 32.736849;
- validation MAE: 16.256044;
- validation R²: 0.279717;
- full OOT rows: 932,084;
- OOT RMSE: 35.051807;
- OOT MAE: 17.101296;
- OOT R²: 0.267770.

## MLflow Registry

| Task | Registered model | Version | Alias | Run ID |
|---|---|---:|---|---|
| Classification | FlightDelayRandomForestSpark | 1 | champion | `ff18ccd6201445efb032efcbe2992fff` |
| Regression | FlightDelayRegressionSpark | 1 | champion | `4f320e8a8778499d9e58dc0f0832689c` |

The local tracking database is `mlflow.db`; artifacts are in `mlruns/`. Both are
runtime state excluded from Git.

## Instructor Demo

The demo reloads an existing artifact, scores a small OOT subset in a fresh
process, then executes one `spark.readStream` JSON microbatch. It does not train.

```bash
source ../.venv/bin/activate
./submit_pipeline.sh classification
./submit_pipeline.sh regression
```

Classification output fields: `event_id`, `prediction`,
`probability_late_over_30m`. Example verified output:

```json
{"event_id":"classification-oot-sample-0001","prediction":0,"probability_late_over_30m":0.29287657056394084}
```

Regression output fields: `event_id`,
`predicted_arrival_delay_minutes`. Example verified output:

```json
{"event_id":"regression-oot-sample-0001","predicted_arrival_delay_minutes":-7.252585765992283}
```

Direct long-running streaming command:

```bash
spark-submit --master 'local[2]' --driver-memory 32g --conf spark.driver.maxResultSize=4g inference.py \
  --task regression \
  --input-dir /path/to/input \
  --output-dir /path/to/output \
  --checkpoint-dir /path/to/checkpoint
```

Add `--available-now` for a finite demonstration batch. Input, output, and
checkpoint directories must differ.

## Tested environment and memory

- Python 3.12.3
- OpenJDK 17.0.20.1
- Spark/PySpark 3.5.9
- MLflow 3.10.1
- PyArrow 23.0.1

Instructor demos use 32 GB driver memory and `spark.driver.maxResultSize=4g` for
the large regression forest. Model fitting is not part of the demo.
