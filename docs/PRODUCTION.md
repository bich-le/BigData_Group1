# Production inference

The production scorer loads a fitted native Spark `PipelineModel` and applies
it to Structured Streaming JSON. Classification uses
`models/random_forest_final_spark`; regression uses
`models/random_forest_regression_final_spark` after training completes.

## Input contract

Each input record must contain an optional string `event_id` and all 23 numeric
fields listed in `feature_contract.py`. These are model-ready features. Several
fields contain lagged airline, airport, or aircraft history and cannot be
calculated honestly from one raw flight event, so the upstream feature system
must provide them. JSON nulls are filled with `0`, matching final fitting. The
scorer does not accept or require `label`, `ARRIVAL_DELAY`, or other outcomes.

`docs/sample_stream_input.json` is one real model-ready OOT example.

## Validate serialized model

```bash
source .venv/bin/activate
spark-submit --master 'local[2]' --driver-memory 20g validate_spark_model.py --task classification
# After the regression artifact exists:
# spark-submit --master 'local[2]' --driver-memory 20g validate_spark_model.py --task regression
```

## One-shot streaming demonstration

```bash
rm -rf /tmp/flight-stream-input /tmp/flight-stream-output /tmp/flight-stream-checkpoint
mkdir -p /tmp/flight-stream-input
cp docs/sample_stream_input.json /tmp/flight-stream-input/events.json
spark-submit --master 'local[2]' --driver-memory 20g inference.py \
  --task classification \
  --input-dir /tmp/flight-stream-input \
  --output-dir /tmp/flight-stream-output \
  --checkpoint-dir /tmp/flight-stream-checkpoint \
  --available-now
cat /tmp/flight-stream-output/*.json
```

Omit `--available-now` to leave the query running and score new JSON files as
they arrive in the input directory.

## MLflow registration

The project venv contains `mlflow==3.10.1` and `setuptools==81.0.0`. Register
the already-fitted model without retraining:

```bash
spark-submit --master 'local[2]' --driver-memory 20g register_mlflow.py --task classification
```

This uses a local SQLite tracking/registry database, tags the validated model,
and assigns the registered-model alias `champion`.


The completed classification registration created:

- run `ff18ccd6201445efb032efcbe2992fff`;
- registered model `FlightDelayRandomForestSpark`, version 1;
- alias `models:/FlightDelayRandomForestSpark@champion`;
- run and model-version tag `validation.status=passed`.

The local registry database is `mlflow.db`, and artifacts are under `mlruns/`.
Both are runtime state and are excluded from Git.
