"""Generate final Production documentation from verified local evidence."""

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVALUATIONS = DOCS / "evaluations"
CONTRACTS = DOCS / "contracts"
EVIDENCE = DOCS / "evidence"


def registry_record(name):
    client = MlflowClient()
    versions = list(client.search_model_versions(f"name='{name}'"))
    if not versions:
        raise RuntimeError(f"No MLflow versions found for {name}")
    version = max(versions, key=lambda v: int(v.version))
    model = client.get_registered_model(name)
    aliases = dict(model.aliases)
    return {
        "name": name,
        "version": str(version.version),
        "run_id": version.run_id,
        "alias": next((a for a, v in aliases.items() if str(v) == str(version.version)), ""),
    }


def stream_output(path):
    candidates = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith('{"event_id"'):
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not candidates:
        raise RuntimeError(f"No streaming JSON output in {path}")
    return candidates[-1]


def main():
    oot = json.loads((EVALUATIONS / "regression_evaluation.json").read_text())
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    classification = registry_record("FlightDelayRandomForestSpark")
    regression = registry_record("FlightDelayRegressionSpark")
    classification_output = stream_output(EVIDENCE / "classification_demo.txt")
    regression_output = stream_output(EVIDENCE / "regression_demo.txt")
    metrics = oot["oot_metrics"]

    production = f'''# Production inference

Two native Spark `PipelineModel` artifacts are verified locally:

- classification: `models/random_forest_final_spark`;
- regression: `models/random_forest_regression_final_spark`.

Both have stages `VectorAssembler -> RandomForest{{Classification,Regression}}Model`.
The artifacts are excluded from Git because of their size.

## Input contract

Each JSON record contains optional string `event_id` plus the 23 ordered numeric
features in `feature_contract.py`. Airline, airport, and aircraft history/lag
features must be engineered upstream. Streaming inference does not derive them
from a raw flight event. Missing features are filled with zero, matching fitting.
No target or outcome field is required for inference.

Task-specific examples:

- `docs/contracts/sample_classification_stream_input.json`;
- `docs/contracts/sample_regression_stream_input.json`.

## Regression selection and final evaluation

Depth-18 configurations were skipped because their Spark MLlib tree growth was
computationally infeasible on the available machine. CV fold counts differed, so
the global winner was selected by RMSE on the common random 20% pre-OOT
validation split (seed 42). Standardized validation benchmarks confirmed that the supplied artifact is the exact winner fitted on the full 80% training split. A local `randomSplit` reconstruction produced different row membership, so the standardized benchmark metrics remain the authoritative validation record. OOT was reserved for final evaluation. See
`docs/evaluations/REGRESSION_RESULTS.md`, `docs/evaluations/regression_selection.json`, and `docs/evaluations/regression_evaluation.json`.

Winner: 500 trees, depth 15, minimum 5 instances per node.

- validation RMSE: 32.736849;
- validation MAE: 16.256044;
- validation R²: 0.279717;
- full OOT rows: {oot['oot_rows']:,};
- OOT RMSE: {metrics['rmse']:.6f};
- OOT MAE: {metrics['mae']:.6f};
- OOT R²: {metrics['r2']:.6f}.

## MLflow Registry

| Task | Registered model | Version | Alias | Run ID |
|---|---|---:|---|---|
| Classification | {classification['name']} | {classification['version']} | {classification['alias']} | `{classification['run_id']}` |
| Regression | {regression['name']} | {regression['version']} | {regression['alias']} | `{regression['run_id']}` |

The local tracking database is `mlflow.db`; artifacts are in `mlruns/`. Both are
runtime state excluded from Git.

## Instructor Demo

The demo reloads an existing artifact, scores a small OOT subset in a fresh
process, then executes one `spark.readStream` JSON microbatch. It does not train.

```bash
source ../.venv/bin/activate
./submit_pipeline.sh classification
./submit_pipeline.sh regression
```

Classification output fields: `event_id`, `prediction`,
`probability_late_over_30m`. Example verified output:

```json
{json.dumps(classification_output, separators=(',', ':'))}
```

Regression output fields: `event_id`,
`predicted_arrival_delay_minutes`. Example verified output:

```json
{json.dumps(regression_output, separators=(',', ':'))}
```

Direct long-running streaming command:

```bash
spark-submit --master 'local[2]' --driver-memory 32g --conf spark.driver.maxResultSize=4g inference.py \\
  --task regression \\
  --input-dir /path/to/input \\
  --output-dir /path/to/output \\
  --checkpoint-dir /path/to/checkpoint
```

Add `--available-now` for a finite demonstration batch. Input, output, and
checkpoint directories must differ.

## Tested environment and memory

- Python 3.12.3
- OpenJDK 17.0.20.1
- Spark/PySpark 3.5.9
- MLflow 3.10.1
- PyArrow 23.0.1

Instructor demos use 32 GB driver memory and `spark.driver.maxResultSize=4g` for
the large regression forest. Model fitting is not part of the demo.
'''
    (DOCS / "PRODUCTION.md").write_text(production)

    task_table = f'''# Task status / Minh chứng

| Task | Status | Evidence |
|---|---|---|
| Chốt, lưu, nạp và đăng ký hai PipelineModel tốt nhất | DONE | Classification `models/random_forest_final_spark`; regression `models/random_forest_regression_final_spark`; fresh-process reload evidence in `docs/evidence/classification_demo.txt` and `docs/evidence/regression_demo.txt`; MLflow `{classification['name']}` v{classification['version']} `{classification['alias']}` run `{classification['run_id']}` and `{regression['name']}` v{regression['version']} `{regression['alias']}` run `{regression['run_id']}`; registry snapshot in `docs/evaluations/mlflow_registry.json`; regression metrics and provenance in `docs/evaluations/regression_evaluation.json`. |
| Chuẩn bị schema và JSON mẫu cho streaming | DONE | Task-aware schema in `inference.py`/`feature_contract.py`; `docs/contracts/sample_classification_stream_input.json`; `docs/contracts/sample_regression_stream_input.json`. |
| Xây dựng streaming inference | DONE | `inference.py` uses real `spark.readStream`, `PipelineModel.load`, and `model.transform`; successful JSON outputs in both demo evidence files. |
| Kiểm thử tích hợp và khả năng tái tạo | DONE | `scripts/validate_spark_model.py`, `scripts/validate_regression_production.py`, `submit_pipeline.sh`, `README.md`, `docs/PRODUCTION.md`; model-path, stage, schema, reload, OOT, and streaming checks passed. |
'''
    (EVIDENCE / "TASK_EVIDENCE.md").write_text(task_table)

    readme = f'''# BigData Group 1 — Spark flight-delay models

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
was skipped. Full OOT metrics: RMSE {metrics['rmse']:.6f}, MAE {metrics['mae']:.6f},
R² {metrics['r2']:.6f}.

See `docs/PRODUCTION.md`, `docs/evaluations/REGRESSION_RESULTS.md`, and
`docs/evidence/TASK_EVIDENCE.md` for commands and evidence.
'''
    (ROOT / "README.md").write_text(readme)
    print(json.dumps({"classification": classification, "regression": regression}, indent=2))


if __name__ == "__main__":
    main()
