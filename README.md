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
├── feature_contract.py               # Shared 23-feature schema contract
├── inference.py                      # Spark Structured Streaming inference engine
│
├── docs/
│   ├── REPORT.md                     # Comprehensive academic and technical report (100 marks)
│   ├── PRODUCTION.md                 # Production deployment and serving runbook
│   ├── assets/                       # Evaluation plots (pca_variance.png, feature_importance.png)
│   ├── contracts/                    # Schema contracts and streaming sample payloads
│   ├── dictionaries/                 # Data dictionaries and column specifications
│   ├── evaluations/                  # Model tournament benchmarks and tuning records
│   └── evidence/                     # Fresh-process execution logs and audit trail
│
├── scripts/                          # Offline training, verification, and utility tools
│   ├── benchmark_results.py          # Tournament metric extraction and feature importance plotting
│   ├── finalize_production_docs.py   # Production documentation generation
│   ├── inference_sklearn_stream.py   # Baseline scikit-learn streaming fallback
│   ├── merge_checkpoints.py          # Distributed grid shard aggregator
│   ├── prepare_data.py               # Raw CSV to partitioned Snappy Parquet converter
│   ├── recover_final_spark_model.py  # Spark classification model refit
│   ├── register_mlflow.py            # MLflow model registration helper
│   ├── run_regression_tuning.ps1     # Distributed regression tuning runner
│   ├── sfdws_model_spark_regression.py # Spark regression grid search
│   ├── validate_regression_production.py # Regression artifact validator
│   └── validate_spark_model.py       # Fresh-process reload and OOT validator
│
├── tests/
│   └── test_custom_transformers.py   # Unit test suite (31 tests, 100% pass)
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

## Instructor Demonstration

The project provides an automated demo runner (`submit_pipeline.sh`) that reloads serialized models in a fresh process, validates a small out-of-time (OOT) holdout sample, and executes one real Spark Structured Streaming microbatch:

```bash
./submit_pipeline.sh classification
./submit_pipeline.sh regression
```

*Note:* Default driver memory is 32 GB (configurable via `DRIVER_MEMORY=16g`). The scripts execute inference only and do not retrain models.

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
- Unit test suite: run `pytest tests/` (31 passed).
