# E23 — End-to-end smoke test evidence

Status: **PASS**

| Check | Result |
| --- | --- |
| Input | 160 synthetic flight rows and 8 synthetic airport lookup rows |
| Feature integration | Kiên feature adapter enabled |
| Split | Temporal: 128 train rows, 32 test rows |
| Processing | IQR stage, imputation, indexing, one-hot encoding, assembling, scaling and PCA (`k=10`) |
| Output assertion | `features_pca` exists |
| Runtime | 134.034 seconds |
| Process RSS | 299.31 MB |
| Spark | `local[2]`, PySpark 3.5.9 |
| MLflow | Run `f8a471b2996246ff9bda929853f5cf0f`; preprocessing model artifact logged |

Machine-readable record: `artifacts/smoke_test_metrics.json`.

This is a smoke test for functional integration and local resource recording;
it is not a benchmark of model quality on the full flight dataset.
