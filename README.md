# BigData Group 1 — Spark flight-delay models

PySpark ML project with two supervised tasks:

- classification: probability that arrival delay exceeds 30 minutes;
- regression: predicted arrival delay in minutes.

## Runtime model paths

Fitted Spark `PipelineModel` directories are excluded from Git because of
their size. The production scripts use these paths:

- `models/random_forest_final_spark` — recovered classification model;
- `models/random_forest_regression_final_spark` — created after regression
  grid selection and final fitting complete.

Both pipelines consume the ordered 23 model-ready numeric fields in
`feature_contract.py`. Historical airline, airport, and aircraft lag fields
must be produced upstream. Streaming inference does not derive them from an
individual raw flight event. Missing feature values are filled with zero.

## Tested environment

- Linux x86-64
- Java 17.0.20.1
- Python 3.12.3
- Spark/PySpark 3.5.9
- MLflow 3.10.1

Create a virtual environment and install the pinned packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The large Spark models require substantial driver heap. Local validation and
demos use `--driver-memory 32g`.

## No-training instructor demo

Run either task from the repository root:

```bash
./submit_pipeline.sh classification
./submit_pipeline.sh regression
```

The script reloads the saved artifact in a fresh Spark process, scores a small
OOT sample, then performs one `spark.readStream` JSON inference run. It never
trains a model.

See `docs/PRODUCTION.md` for direct commands, example outputs, MLflow Registry
records, and known limitations. Data preparation details are in
`data/README.md`.
