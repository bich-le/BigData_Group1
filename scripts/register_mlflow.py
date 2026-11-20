"""Register an already-fitted and validated Spark PipelineModel in MLflow.

This script only loads and logs a model. It never fits or retrains one.
"""

import argparse
import json
import math
import tempfile
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

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_contract import FEATURES_BY_TASK

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
        "origin": "production-candidate-80pct",
    },
}


def parse_metrics(values):
    metrics = {}
    for value in values:
        try:
            name, raw = value.split("=", 1)
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(value)
            metrics[name] = number
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Metric must have NAME=NUMBER form: {value}"
            ) from exc
    return metrics


def metrics_from_file(path: Path) -> tuple[dict[str, float], dict]:
    """Read measured/evaluated metrics, preserving the source as an artifact."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if "oot_metrics" in document:
        values = {f"oot_{name}": value for name, value in document["oot_metrics"].items()}
        values.update(
            {f"validation_{name}": value for name, value in document.get("selection_validation", {}).items()
            if name in {"rmse", "mae", "r2"}}
        )
    elif "candidates" in document:
        champions = [
            candidate for candidate in document["candidates"]
            if candidate.get("verdict") == "champion"
        ]
        if len(champions) != 1:
            raise ValueError("Metrics file must identify exactly one champion")
        values = champions[0]["metrics"]
    else:
        raise ValueError("Metrics file needs oot_metrics or candidates")
    metrics = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Metric {name} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Metric {name} must be finite")
        metrics[name] = number
    return metrics, document


def model_feature_importances(estimator, feature_cols: list[str]) -> dict[str, float]:
    values = estimator.featureImportances.toArray().tolist()
    if len(values) != len(feature_cols):
        raise ValueError("Feature importance vector does not match feature contract")
    return dict(zip(feature_cols, map(float, values)))


def save_feature_importance_plot(importances: dict[str, float], path: Path, task: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(importances.items(), key=lambda item: item[1])[-15:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([name for name, _ in ordered], [value for _, value in ordered])
    ax.set(xlabel="Relative importance", title=f"{task.title()} fitted model")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), default="classification")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default="flight-delay-production")
    parser.add_argument("--registered-name")
    parser.add_argument("--training-rows", type=int)
    parser.add_argument("--metric", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--oot-rows", type=int)
    parser.add_argument("--tournament-run-id", help="Parent run from log_tournament_mlflow.py")
    parser.add_argument(
        "--promote-production", action="store_true",
        help="Promote only after supplying the required OOT metrics",
    )
    args = parser.parse_args()
    defaults = TASK_DEFAULTS[args.task]
    model_path = args.model or defaults["model"]
    registered_name = args.registered_name or defaults["registered_name"]
    if args.model and args.training_rows is None:
        parser.error("--training-rows is required for a custom --model path")
    training_rows = args.training_rows or defaults["training_rows"]
    if not model_path.is_dir():
        parser.error(f"Validated Spark PipelineModel is missing: {model_path}")
    try:
        metrics = parse_metrics(args.metric)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    metrics_document = None
    if args.metrics_file:
        if not args.metrics_file.is_file():
            parser.error(f"Metrics file is missing: {args.metrics_file}")
        file_metrics, metrics_document = metrics_from_file(args.metrics_file)
        metrics.update(file_metrics)
    required_metrics = (
        {"oot_auc_roc"} if args.task == "classification"
        else {"oot_rmse", "oot_mae", "oot_r2"}
    )
    if args.promote_production and not required_metrics.issubset(metrics):
        parser.error(
            "Production promotion requires actual OOT metrics: "
            + ", ".join(sorted(required_metrics - metrics.keys()))
        )
    oot_rows = args.oot_rows or (
        metrics_document.get("oot_rows") if metrics_document else None
    )

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
        importances = model_feature_importances(estimator, feature_cols)

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        client = MlflowClient()
        if args.tournament_run_id:
            client.get_run(args.tournament_run_id)
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
                "training_rows": training_rows,
                "feature_count": len(feature_cols),
                **estimator_params,
            })
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.log_metrics(
                {f"feature_importance_{name}": value for name, value in importances.items()}
            )
            mlflow.log_dict(importances, "evaluation/feature_importance.json")
            with tempfile.TemporaryDirectory(prefix="mlflow-importance-") as directory:
                plot = Path(directory) / "feature_importance.png"
                save_feature_importance_plot(importances, plot, args.task)
                mlflow.log_artifact(str(plot), artifact_path="plots")
            pca_plot = ROOT / "docs/assets/pca_variance.png"
            if pca_plot.is_file():
                mlflow.log_artifact(str(pca_plot), artifact_path="plots/preprocessing")
            if args.metrics_file:
                mlflow.log_artifact(str(args.metrics_file), artifact_path="evaluation")
            if args.task == "regression":
                selection = ROOT / "docs/evaluations/regression_selection.json"
                if selection.is_file():
                    mlflow.log_artifact(str(selection), artifact_path="evaluation")
            mlflow.set_tags({
                "validation.status": "structure_passed",
                "validation.oot_rows": str(oot_rows) if oot_rows is not None else "unknown",
                "deployment.input": "pre-engineered-23-feature-json",
                "model.origin": defaults["origin"] if not args.model else "supplied_model_path",
                "evaluation.source": (
                    str(args.metrics_file) if args.metrics_file else
                    "cli_metrics" if args.metric else "none"
                ),
                "feature_importance.source": "fitted_spark_model",
                "pca.plot.source": "checked_in_preprocessing",
            })
            if args.tournament_run_id:
                mlflow.set_tag("tournament.parent_run_id", args.tournament_run_id)
            mlflow.spark.log_model(
                spark_model=model,
                artifact_path="spark-pipeline",
                registered_model_name=registered_name,
                signature=signature,
                pip_requirements=["pyspark==3.5.9"],
            )
            run_id = run.info.run_id

        versions = [
            version for version in client.search_model_versions(f"name='{registered_name}'")
            if version.run_id == run_id
        ]
        if not versions:
            raise RuntimeError(
                "MLflow logging succeeded but no registered version matched the run"
            )
        version = max(versions, key=lambda item: int(item.version))
        client.transition_model_version_stage(
            registered_name, version.version, "Staging"
        )
        client.set_model_version_tag(
            registered_name, version.version, "validation.status", "structure_passed"
        )
        if args.promote_production:
            client.transition_model_version_stage(
                registered_name, version.version, "Production",
                archive_existing_versions=True,
            )
            client.set_model_version_tag(
                registered_name, version.version, "validation.status", "oot_metrics_supplied"
            )
            client.set_registered_model_alias(registered_name, "champion", version.version)
        print(f"MLflow run: {run_id}")
        print(f"Registered model: {registered_name} version {version.version}")
        print(f"Stage: {'Production' if args.promote_production else 'Staging'}")
        if args.promote_production:
            print(f"Alias: models:/{registered_name}@champion")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
