# Flight Delay Pipelines in PySpark MLlib

Reproducible training, evaluation, MLflow registration and JSON streaming for
severe arrival delay classification (`ARRIVAL_DELAY > 30`) and delay regression.

## Submission layout

```text
custom_transformers.py       Custom estimators/models, Arrow Haversine and batch history features
mllib_pipeline.py            Main ingestion → cross-validation → MLflow → model entry point
benchmark_results.py         Measured metrics, PCA variance and fitted tree importance
inference.py                 Saved full-pipeline Structured Streaming scoring
submit_pipeline.sh           Fresh-process spark-submit streaming demonstration
feature_integration.py       Optional batch-history feature adapter
feature_contract.py          Legacy 23-feature model contract
requirements.txt             Runtime and test dependencies
requirements-notebook.txt    Optional notebook dependencies
docs/REPORT.md               Mathematical report and historical benchmark analysis
docs/PRODUCTION.md           Current training, evaluation and serving commands
docs/SUBMISSION.md           Deliverable map and validation boundaries
models/                      Generated full PipelineModel artifacts (excluded from Git)
scripts/                     Data preparation, batch features, training implementation and legacy tools
tests/                       Transformer, packaging and pipeline integration tests
notebooks/                   Reproducible data-quality and transformer examples
vendor/                      Windows Hadoop filesystem compatibility JAR
```

## Setup

Use Python 3.11/3.12 and Java 17. Create a virtual environment and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYSPARK_PYTHON="$VIRTUAL_ENV/bin/python"
```

Place the extracted `flights.csv`, `airports.csv` and `airlines.csv` in
`data/raw/`, then validate and convert them:

```bash
spark-submit scripts/prepare_data.py --input-dir data/raw \
  --output-dir data/typed-parquet
```

The training input is `data/typed-parquet/flights`, containing raw typed columns,
not a legacy model-ready feature matrix. Dataset files are excluded from Git.

## Train and reproduce

```bash
spark-submit --master 'local[4]' mllib_pipeline.py \
  --task classification --input data/typed-parquet/flights \
  --output models/classification_full_pipeline \
  --results-dir artifacts/classification

spark-submit --master 'local[4]' mllib_pipeline.py \
  --task regression --input data/typed-parquet/flights \
  --output models/regression_full_pipeline \
  --results-dir artifacts/regression
```

The default tournament covers LinearSVC, RF and GBT for classification, and
ElasticNet LR, Gamma/log GLM, RF and GBT for regression. Each task reuses one
seeded 80/20 split; all candidates fit preprocessing inside five-fold
`CrossValidator(parallelism=4)`. The winner is selected by training CV only.
Held-out metrics, actual grids, plots and the exact test rows are saved under
`--results-dir`. MLflow logs candidate parameters/metrics, the winning full
pipeline and registry aliases `staging` and `production`.

Gamma/log GLM uses a fixed label offset of 1,440 minutes because delay labels
can be negative. The saved pipeline restores predictions to minutes. Adjust
`--glm-offset` if necessary; nonpositive shifted training labels are rejected.
This is a shifted-response baseline, not a Gamma distribution over signed delays.

Use `--algorithms random_forest --small-grid --folds 2 --pca-k 2` for a smoke
run. Smoke results do not reproduce full-data benchmarks. Output directories
must be new; existing models/evaluations are preserved.

## Evaluate and stream

```bash
spark-submit benchmark_results.py --task classification \
  --model models/classification_full_pipeline \
  --input artifacts/classification/test_data \
  --output-dir artifacts/classification_recheck

./submit_pipeline.sh classification
./submit_pipeline.sh regression
```

Tree feature importances refer to **PCA components**, the actual estimator
inputs. They are measured from the loaded model. LinearSVC has no calibrated
probability; its streaming probability field is null.

## Optional batch features and tests

`python scripts/build_modeling_dataset.py --help` documents optional Haversine,
airline/airport/aircraft history and schedule features. These batch features
require external history and are separate from stateless raw-event serving.
For preprocessing analysis, use `mllib_pipeline.py --preprocess-only --help`.

```bash
python -m pytest tests/ -q
```

Historical 23-feature benchmarks in `docs/evaluations/` and `docs/REPORT.md`
belong to legacy models. They are not measurements of the current raw-input
pipeline; run the commands above to produce a new benchmark.
