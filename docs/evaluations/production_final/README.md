# Historical fixed-parameter full-pipeline evidence

These snapshots describe the earlier 25,000-row fixed-parameter verification,
not the current cross-validation workflow. New measurements are written to
`--results-dir` by `mllib_pipeline.py`.

Saved logs anonymize runtime usernames, hostnames, and private network addresses.

- `DATA_CONTRACT.md`: raw versus processed datasets and serving contract.
- `*_fit_save_load.json`: actual full-pipeline stage list, RF parameters,
  `PipelineModel.load()` result, sample predictions, and registry version.
- `mlflow_registry_state.json`: MLflow aliases captured during the historical run.
- `*_streaming_output.json` and `*_streaming.log`: `spark.readStream` runs.
- `*_submit.log`: successful `submit_pipeline.sh` runs.
- `regression_fit.log`: initial 1-GB heap failure; `regression_fit_retry.log`:
  successful retry using `local[16]`, 32 partitions and an 8-GB driver heap.

No OOT/model-ready raw-end-to-end prediction is claimed: these inputs are
already processed and belong to the legacy feature-vector contract.
