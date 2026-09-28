"""Register an already-fitted and validated Spark PipelineModel in MLflow.

This script only loads and logs a model. It never fits or retrains one.
"""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.spark
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import SparkSession

from feature_contract import FEATURES_BY_TASK


ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
TASK_DEFAULTS = {
    "classification": {
        "model": ROOT / "models/random_forest_final_spark",
        "registered_name": "FlightDelayRandomForestSpark",
        "training_rows": 3_739_905,
        "origin": "fixed-final-spark-recovery",
    },
    "regression": {
        "model": ROOT / "models/random_forest_regression_final_spark",
        "registered_name": "FlightDelayRegressionSpark",
        "training_rows": 3_739_186,
        "origin": "teammate-confirmed-full-80pct-candidate",
    },
}


def parse_metrics(values):
    metrics = {}
    for value in values:
        try:
            name, raw = value.split("=", 1)
            metrics[name] = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Metric must have NAME=NUMBER form: {value}"
            ) from exc
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), default="classification")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default="flight-delay-production")
    parser.add_argument("--registered-name")
    parser.add_argument("--metric", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--oot-rows", type=int, default=20)
    args = parser.parse_args()
    defaults = TASK_DEFAULTS[args.task]
    model_path = args.model or defaults["model"]
    registered_name = args.registered_name or defaults["registered_name"]
    if not model_path.is_dir():
        parser.error(f"Validated Spark PipelineModel is missing: {model_path}")
    try:
        metrics = parse_metrics(args.metric)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.metrics_file:
        if not args.metrics_file.is_file():
            parser.error(f"Metrics file is missing: {args.metrics_file}")
        metrics_doc = json.loads(args.metrics_file.read_text())
        for name, value in metrics_doc.get("oot_metrics", {}).items():
            metrics[f"oot_{name}"] = float(value)

    spark = SparkSession.builder.appName(f"Register{args.task.title()}SparkModel").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        model = PipelineModel.load(str(model_path))
        feature_cols = model.stages[0].getInputCols()
        if feature_cols != FEATURES_BY_TASK[args.task]:
            raise ValueError("Saved model does not match the verified task feature contract")
        expected_type = (
            RandomForestClassificationModel
            if args.task == "classification"
            else RandomForestRegressionModel
        )
        estimator = model.stages[-1]
        if not isinstance(estimator, expected_type):
            raise ValueError(f"Saved model is not a Spark Random Forest {args.task} model")

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        signature = ModelSignature(
            inputs=Schema([ColSpec("double", name) for name in feature_cols]),
            outputs=Schema([ColSpec("double", "prediction")]),
        )
        estimator_params = {
            param.name: estimator.getOrDefault(param)
            for param in estimator.params
            if estimator.isDefined(param)
            and param.name in {
                "numTrees", "maxDepth", "minInstancesPerNode",
                "featureSubsetStrategy", "impurity", "bootstrap", "seed",
            }
        }
        with mlflow.start_run(run_name=f"final-spark-random-forest-{args.task}") as run:
            mlflow.log_params({
                "task": args.task,
                "algorithm": type(estimator).__name__,
                "training_rows": defaults["training_rows"],
                "feature_count": len(feature_cols),
                **estimator_params,
            })
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "validation.status": "passed",
                "validation.oot_rows": str(args.oot_rows),
                "deployment.input": "pre-engineered-23-feature-json",
                "model.origin": defaults["origin"],
                "validation.source": (
                    "teammate-confirmed-common-20pct"
                    if args.task == "regression"
                    else "local-verified"
                ),
                "evaluation.oot_source": "local-held-out",
            })
            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="spark-pipeline",
                registered_model_name=registered_name,
                signature=signature,
                pip_requirements=["pyspark==3.5.9"],
            )
            run_id = run.info.run_id

        client = MlflowClient()
        versions = [
            version for version in client.search_model_versions(f"name='{registered_name}'")
            if version.run_id == run_id
        ]
        if not versions:
            raise RuntimeError(
                "MLflow logging succeeded but no registered version matched the run"
            )
        version = max(versions, key=lambda item: int(item.version))
        client.set_model_version_tag(
            registered_name, version.version, "validation.status", "passed"
        )
        client.set_registered_model_alias(registered_name, "champion", version.version)
        print(f"MLflow run: {run_id}")
        print(f"Registered model: {registered_name} version {version.version}")
        print(f"Alias: models:/{registered_name}@champion")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
