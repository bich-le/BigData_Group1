# Serialized pipeline artifacts

The required submission includes a fitted Spark PipelineModel, supplied as a
separate artifact or an accessible MLflow artifact location. Model directories
are excluded from source control because trained forests can be large.

Generate current models with the root `mllib_pipeline.py` commands in README:

- `models/classification_full_pipeline/`
- `models/regression_full_pipeline/`

Both directories contain Spark `metadata/` and `stages/`. They must reload in a
fresh Python process with this repository's pinned dependencies installed.
The full training run also registers each winner in MLflow; its URI, run ID and
version are recorded in `<results-dir>/tournament.json`.

To package locally generated models for separate delivery, run from repo root:

```bash
tar -czf /tmp/flight-full-pipeline-models.tar.gz \
  models/classification_full_pipeline models/regression_full_pipeline
```

Include the corresponding tournament report and measured diagnostics with the
artifact. Do not present synthetic smoke models or the historical
`random_forest_*_spark` feature-vector artifacts as current full-data winners.
