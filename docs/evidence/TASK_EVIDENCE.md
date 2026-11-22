# Pipeline Validation Evidence

Runtime usernames, hostnames, and private network addresses in the saved logs
are anonymized. Predictions, metrics, and run identifiers are retained.

| Validation | Status | Evidence |
|---|---|---|
| Model persistence and registration | DONE | Classification `models/random_forest_final_spark`; regression `models/random_forest_regression_final_spark`; fresh-process reload evidence in `docs/evidence/classification_demo.txt` and `docs/evidence/regression_demo.txt`; MLflow `FlightDelayRandomForestSpark` v1 `champion` run `ff18ccd6201445efb032efcbe2992fff` and `FlightDelayRegressionSpark` v1 `champion` run `4f320e8a8778499d9e58dc0f0832689c`; registry snapshot in `docs/evaluations/mlflow_registry.json`; regression metrics and provenance in `docs/evaluations/regression_evaluation.json`. |
| Streaming schema and sample payloads | DONE | Task-aware schema in `inference.py`/`feature_contract.py`; `docs/contracts/sample_classification_stream_input.json`; `docs/contracts/sample_regression_stream_input.json`. |
| Streaming inference | DONE | `inference.py` uses real `spark.readStream`, `PipelineModel.load`, and `model.transform`; successful JSON outputs in both demo evidence files. |
| Integration and reproducibility | DONE | `scripts/validate_spark_model.py`, `scripts/validate_regression_production.py`, `submit_pipeline.sh`, `README.md`, `docs/PRODUCTION.md`; model-path, stage, schema, reload, OOT, and streaming checks passed. |
