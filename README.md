# BigData Group 1 — Spark flight-delay models

PySpark ML project with two Production tasks:

- classification: probability that arrival delay exceeds 30 minutes;
- regression: predicted arrival delay in minutes.

## Runtime artifacts

- `models/random_forest_final_spark` — classification `PipelineModel`;
- `models/random_forest_regression_final_spark` — regression `PipelineModel`.

Model directories, datasets, MLflow state, logs, and caches are excluded from
Git. Both models consume the task contracts in `feature_contract.py`. Historical
lag features must be produced upstream; missing values are filled with zero.

## Instructor Demo

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./submit_pipeline.sh classification
./submit_pipeline.sh regression
```

The scripts load existing models, validate a small OOT sample, and run one real
Structured Streaming JSON batch. They never train. Default demo driver memory is
32 GB and can be changed with `DRIVER_MEMORY`.

## Regression winner

The selected model uses 500 trees, depth 15, and minimum leaf size 5. Selection
used common validation RMSE 32.736849 because CV fold counts differed. Depth 18
was skipped. Full OOT metrics: RMSE 35.051807, MAE 17.101296,
R² 0.267770.

See `docs/PRODUCTION.md`, `docs/REGRESSION_RESULTS.md`, and
`docs/TASK_EVIDENCE.md` for commands and evidence.
