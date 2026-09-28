# E22 — MLflow evidence

Status: **PASS**

Verification command:

```powershell
python smoke_test.py `
  --rows 160 `
  --with-kien-features `
  --mlflow-experiment "SuongMai-E22-Verification" `
  --output-dir .
```

Verified run: `f8a471b2996246ff9bda929853f5cf0f`.

| Requirement | Verified result |
| --- | --- |
| Parameters | `pca_k`, split strategy/ratio/seed, categorical columns, numeric columns |
| Metrics | `train_rows=128`, `test_rows=32`, cumulative PCA variance `0.7590845688302841`, plus variance per component |
| Artifacts | `plots/pca_variance.png`, `config/pipeline_config.json`, `verification/e22_mlflow_manifest.json` |
| Model artifact | `preprocessing_pipeline/MLmodel` logged by `mlflow.spark.log_model()` |
| Reload | PASS — loaded as `PipelineModel` with 11 stages |

On Windows, the run uses the vendored `hadoop-bare-naked-local-fs-0.1.0.jar`
and a writable Temp URI so Spark can persist/load the model without
`winutils.exe`. The local `mlruns/` directory remains ignored by Git; this
document and the two JSON summaries are the versioned proof for review.
