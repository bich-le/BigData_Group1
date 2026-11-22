"""Fit, reload, score and register the two raw-input Production pipelines.

This is a bounded Production verification command, not the tournament.  It
uses the established RF hyperparameters and records only metrics/results that
are observed during this invocation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mllib_pipeline import (  # noqa: E402
    PipelineConfig,
    configure_local_filesystem,
    configure_pyspark_python,
    fit_and_save_end_to_end_pipeline,
)

NUMERIC = ("MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_TIME", "SCHEDULED_DEPARTURE", "DISTANCE", "CANCELLED", "DIVERTED", "ARRIVAL_DELAY")


def raw_flights(spark: SparkSession, path: Path, rows: int):
    data = spark.read.option("header", True).option("inferSchema", True).csv(str(path))
    for column in NUMERIC:
        data = data.withColumn(column, F.col(column).cast("double"))
    return data.limit(rows).repartition(32).cache()


def register(model: PipelineModel, task: str, evidence: dict) -> dict:
    import mlflow
    import mlflow.spark
    from mlflow import MlflowClient

    name = f"FlightDelayFullPipeline{task.title()}"
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    with mlflow.start_run(run_name=f"production-full-{task}") as run:
        mlflow.log_params({"task": task, "rf_hyperparameters": json.dumps(evidence["rf_params"], sort_keys=True)})
        mlflow.log_dict(evidence, "production_verification.json")
        model_info = mlflow.spark.log_model(model, artifact_path="full_pipeline")
    version = mlflow.register_model(model_info.model_uri, name)
    client = MlflowClient()
    client.set_registered_model_alias(name, "production", version.version)
    client.set_model_version_tag(name, version.version, "pipeline_scope", "raw-input-full-pipeline")
    return {"name": name, "version": str(version.version), "alias": "production", "run_id": run.info.run_id}


def fit_one(data, task: str, output: Path, rows: int) -> dict:
    config = PipelineConfig(pca_k=10)
    if task == "classification":
        estimator = RandomForestClassifier(
            labelCol=config.classification_label, featuresCol="features_pca", numTrees=300,
            maxDepth=15, minInstancesPerNode=5, featureSubsetStrategy="sqrt", seed=42,
        )
    else:
        estimator = RandomForestRegressor(
            labelCol=config.regression_label, featuresCol="features_pca", numTrees=500,
            maxDepth=15, minInstancesPerNode=5, featureSubsetStrategy="sqrt", seed=42,
        )
    model = fit_and_save_end_to_end_pipeline(data, output, task, config, estimator=estimator)
    reloaded = PipelineModel.load(str(output))
    sample = reloaded.transform(data.drop("ARRIVAL_DELAY").limit(3))
    columns = ["prediction"] + (["probability"] if task == "classification" else [])
    evidence = {
        "task": task,
        "training_source": "raw archive flights.csv",
        "training_rows_requested": rows,
        "pipeline_stages": [type(stage).__name__ for stage in reloaded.stages],
        "rf_params": {name: reloaded.stages[-1].getOrDefault(getattr(reloaded.stages[-1], name)) for name in ("numTrees", "maxDepth", "minInstancesPerNode")},
        "save_load": "PipelineModel.load succeeded",
        "sample_predictions": [row.asDict(recursive=True) for row in sample.select(*columns).collect()],
    }
    evidence["mlflow_registry"] = register(reloaded, task, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--rows", type=int, default=25000)
    parser.add_argument("--tasks", nargs="+", choices=("classification", "regression"), default=("classification", "regression"))
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "docs/evaluations/production_final")
    args = parser.parse_args()
    if not args.raw_csv.is_file() or args.rows < 100:
        parser.error("--raw-csv must exist and --rows must be at least 100")
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    configure_pyspark_python()
    spark = configure_local_filesystem(
        SparkSession.builder.master("local[16]")
        .appName("production-full-pipeline-fit")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "32")
    ).getOrCreate()
    try:
        data = raw_flights(spark, args.raw_csv, args.rows)
        for task in args.tasks:
            output = args.output_dir / f"{task}_full_pipeline"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite artifact: {output}")
            evidence = fit_one(data, task, output, args.rows)
            (args.evidence_dir / f"{task}_fit_save_load.json").write_text(json.dumps(evidence, indent=2, default=str) + "\n")
            print(json.dumps(evidence, indent=2, default=str), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
