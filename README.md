# BigData Group 1 — End-to-End Supervised Learning Pipeline in PySpark MLlib

End-to-end distributed machine learning system for flight delay prediction and severity classification built with PySpark MLlib, custom PySpark transformers, Arrow-accelerated pandas UDFs, MLflow tracking, and Spark Structured Streaming.

The system addresses two production tasks:
1. **Severe Flight Delay Warning System (SFDWS - Classification):** Predicts whether a flight will experience severe arrival delay ($\ge 30$ minutes).
2. **Flight Delay Duration Prediction System (FDDPS - Regression):** Predicts the exact arrival delay in minutes.

---

## Repository Structure

```text
├── README.md                         # Project overview and runbook
├── requirements.txt                  # Pinned Python dependencies
├── requirements-notebook.txt         # Jupyter notebook environment requirements
├── submit_pipeline.sh                # Executable script for demonstration and evaluation
├── custom_transformers.py           # Custom PySpark transformers (Haversine, IQR, TargetEncoder)
├── mllib_pipeline.py                 # Feature engineering pipeline & distributed PCA
├── feature_integration.py            # Flight feature contract and preprocessing helpers
├── feature_contract.py               # Shared 23-feature schema contract
├── inference.py                      # Spark Structured Streaming inference engine
│
├── docs/
│   ├── REPORT.md                     # Comprehensive academic and technical report
│   ├── PRODUCTION.md                 # Production deployment and serving runbook
│   ├── assets/                       # PCA explained-variance plot
│   ├── contracts/                    # Schema contracts and streaming sample payloads
│   ├── dictionaries/                 # Data dictionaries and column specifications
│   ├── evaluations/                  # Model tournament benchmarks and tuning records
│   └── evidence/                     # Fresh-process execution logs and audit trail
│
├── scripts/                          # Offline training, verification, and utility tools
│   ├── inference_sklearn_stream.py   # Baseline scikit-learn streaming fallback
│   ├── fit_production_full_models.py # Fixed-parameter full pipeline fit and reload verification
│   ├── prepare_data.py               # Raw CSV to partitioned Snappy Parquet converter
│   ├── recover_final_spark_model.py  # Spark classification model refit
│   ├── register_mlflow.py            # MLflow model registration helper
│   ├── run_regression_tuning.ps1     # Distributed regression tuning runner
│   ├── sfdws_model_spark_regression.py # Spark regression grid search
│   ├── train_tournament.py          # Cross-validation and full pipeline model selection
│   ├── validate_regression_production.py # Regression artifact validator
│   └── validate_spark_model.py       # Fresh-process reload and OOT validator
│
├── tests/
│   ├── test_custom_transformers.py   # Transformer unit tests
│   └── test_end_to_end_pipeline.py   # Raw-input pipeline save/load test
│
├── notebooks/
│   ├── 01_data_quality_profile.ipynb # Data quality analysis notebook
│   └── README.md
│
├── data/                             # Raw and partitioned Parquet datasets (gitignored)
├── models/                           # Serialized Spark PipelineModel artifacts (gitignored)
└── vendor/                           # Bare-naked local filesystem shim for Windows
```

---

## Environment Setup

### Prerequisites
- **Java:** OpenJDK / Eclipse Temurin JDK 17
- **Python:** Python 3.11 or 3.12 (managed via `uv`, `venv`, or `conda`)
- **Apache Spark:** PySpark 3.5.x
- **PyArrow:** 14.x+ (for vectorized UDF acceleration)

Set `JAVA_HOME` and include its `bin` directory in `PATH`:

```powershell
# Windows PowerShell
$env:JAVA_HOME = "C:\path\to\jdk-17"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

```bash
# Linux / macOS / WSL
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk"
export PATH="$JAVA_HOME/bin:$PATH"
```

Create environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Pipeline Demonstration

The project provides an automated demo runner (`submit_pipeline.sh`) that reloads a **full raw-input** `PipelineModel` in a fresh process and executes one Spark Structured Streaming microbatch. It never deletes an existing model or evidence directory:

```bash
MODEL_DIR=models/classification_full_pipeline ./submit_pipeline.sh classification
MODEL_DIR=models/regression_full_pipeline ./submit_pipeline.sh regression
```

The retired `random_forest_*_spark` artifacts contain only
`VectorAssembler → RandomForest` and are intentionally rejected because they
require a pre-built feature vector. Train a replacement model with the command
below before running the demo. Default driver memory is 4 GB (configurable by
`DRIVER_MEMORY`); the demo itself never retrains.

## Feature engineering

Run preprocessing with the flight feature contract:

```bash
spark-submit mllib_pipeline.py --input data/typed-parquet/flights \
  --with-flight-features --airports data/typed-parquet/airports \
  --output-dir models/preprocessing
```

## Reproducible training and tournament

Install the pinned dependencies, ensure Java 17 is available, then point the
training command at the raw typed flights Parquet directory:

```bash
spark-submit scripts/train_tournament.py \
  --task classification \
  --input data/typed-parquet/flights \
  --output models/classification_full_pipeline
```

The command builds one Spark `Pipeline` containing eligibility/label handling,
custom preprocessing, categorical indexing/OHE, imputation, scaling, PCA and
the estimator. Its `ParamGridBuilder` tournament is evaluated by a real
three-fold `CrossValidator(parallelism=4)` and only then saves the measured
best full `PipelineModel`. It does not embed or claim benchmark results.

---

## Production Models & Benchmarks

| Task | Champion Model | Architecture | Primary Metric (OOT) | Secondary Metric (OOT) |
|---|---|---|---|---|
| **Classification (SFDWS)** | Random Forest | 300 trees, depth 15 | **Gini: 0.6762** (KS: 0.5245) | Precision: 0.6076, Recall: 0.4821 |
| **Regression (FDDPS)** | Random Forest | 500 trees, depth 15, leaf 5 | **$R^2$: 0.2678** | RMSE: 35.05 min, MAE: 17.10 min |

- Runtime artifacts: `models/random_forest_final_spark` and `models/random_forest_regression_final_spark`.
- Comprehensive academic documentation and mathematical proofs: [`docs/REPORT.md`](docs/REPORT.md).
- Production deployment guide: [`docs/PRODUCTION.md`](docs/PRODUCTION.md).
- Model tournament and grid search logs: [`docs/evaluations/REGRESSION_RESULTS.md`](docs/evaluations/REGRESSION_RESULTS.md).
- Unit test suite: run `pytest tests/`.
