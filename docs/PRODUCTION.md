# Full-pipeline training and serving

The supported entry point is root-level `mllib_pipeline.py`. See the repository
README for complete commands. `scripts/train_tournament.py` contains its reusable
training implementation; either entry point accepts identical arguments.

## Training contract

Raw typed Parquet must contain `AIRLINE`, `ORIGIN_AIRPORT`,
`DESTINATION_AIRPORT`, `MONTH`, `DAY`, `DAY_OF_WEEK`, `SCHEDULED_TIME`,
`SCHEDULED_DEPARTURE`, `DISTANCE`, `CANCELLED`, `DIVERTED`, `ARRIVAL_DELAY`.
Eligibility excludes cancelled/diverted rows and null targets. Classification
is strictly `ARRIVAL_DELAY > 30`. The target is never an input feature.

Every saved raw-input PipelineModel contains eligibility/label handling,
scheduled hour, a fitted IQR model, median imputation, StringIndexer models,
OneHotEncoder, VectorAssembler, StandardScaler(withMean=False), PCA(k=10 by
default), and the fitted estimator. Gamma/log GLM additionally persists label
shift and prediction inverse transforms. No preprocessing model is fit on the
held-out test set. CV refits learned preprocessing inside each training fold.

## Artifacts

| Output | Location |
|---|---|
| Classification PipelineModel | `models/classification_full_pipeline/` |
| Regression PipelineModel | `models/regression_full_pipeline/` |
| Measured grids, split sizes, CV and test metrics | `<results-dir>/tournament.json` |
| Exact held-out input data | `<results-dir>/test_data/` |
| PCA and model diagnostics | `<results-dir>/<algorithm>/` |
| MLflow database | `mlflow.db`, or `--tracking-uri` |
| Registered model | `FlightDelayFullPipelineClassification` / `FlightDelayFullPipelineRegression` |

The winner is chosen using training CV AUC-ROC (classification) or RMSE
(regression). Held-out metrics are reported without using them for selection.
Each registered version receives Staging/Production validation tags and
`staging`/`production` aliases. Model, data and runtime state stay outside Git;
a submission must include generated artifacts separately or provide their
MLflow artifact location. The placeholder `models/README.md` is not a model.

## Streaming contract

Serving JSON contains the training fields except `ARRIVAL_DELAY`, plus optional
`event_id`. Strings encode airline and airport IDs; scheduled HHMM and numeric
fields use JSON numbers. Samples are in `docs/contracts/sample_*_stream_input.json`.
Schema is derived from fitted preprocessing. Unseen categoricals are handled
by the saved indexer/encoder; numeric nulls use fitted imputation.

```bash
spark-submit --master 'local[2]' inference.py --task classification \
  --model models/classification_full_pipeline \
  --input-dir /path/to/input --output-dir /path/to/output \
  --checkpoint-dir /path/to/checkpoint --available-now
```

Omit `--available-now` for a continuously running query. Input, output and
checkpoint paths must be distinct. Classification emits `prediction` and
`probability_late_over_30m` (null for LinearSVC); regression emits
`predicted_arrival_delay_minutes`. Labels are not needed for streaming.

`submit_pipeline.sh` creates one temporary microbatch and reloads the model in
a fresh process. `MODEL_DIR`, `DRIVER_MEMORY` and `PYSPARK_PYTHON` can override
defaults. The demo never trains or deletes model directories.

## Batch history and legacy compatibility

`scripts/build_modeling_dataset.py` materializes optional history features.
It does not define a stateless raw-event serving contract. History windows
ordered by scheduled timestamps alone do not prove that past outcomes were
available at inference time; production history needs outcome-availability
cutoffs. These features are excluded from the default raw-input pipeline.

`feature_contract.py`, `scripts/recover_final_spark_model.py`,
`scripts/sfdws_model_spark_regression.py`, `scripts/register_mlflow.py`,
`scripts/validate_spark_model.py`, `scripts/validate_regression_production.py`
and `scripts/inference_sklearn_stream.py` support the legacy pre-engineered
23-feature artifacts only. They do not register, validate or serve the new
raw-input models. Their historical metrics are retained as prior evidence.
