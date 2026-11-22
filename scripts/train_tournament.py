"""Train comparable full pipelines with fold-local preprocessing and MLflow tracking."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from pyspark.ml.classification import GBTClassifier, LinearSVC, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.regression import GBTRegressor, GeneralizedLinearRegression, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_results import evaluate_predictions, write_diagnostics
from mllib_pipeline import (
    PipelineConfig, RegressionOffsetTransformer, build_end_to_end_pipeline,
    configure_local_filesystem, configure_pyspark_python,
    prepare_eligible_flights, split_train_test,
)

ALGORITHMS = {"classification": ("linear_svc", "random_forest", "gbt"), "regression": ("linear_regression", "glm", "random_forest", "gbt")}


def make_tournament(task: str, config: PipelineConfig, algorithm="random_forest", folds=5, small_grid=False, glm_offset=1440.0):
    label = config.classification_label if task == "classification" else config.regression_label
    common = {"labelCol": label, "featuresCol": "features_pca"}
    grid = ParamGridBuilder()
    if algorithm == "random_forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        estimator = cls(**common, seed=config.split_seed)
        grid.addGrid(estimator.numTrees, [2] if small_grid else [100, 300])
    elif algorithm == "gbt":
        cls = GBTClassifier if task == "classification" else GBTRegressor
        estimator = cls(**common, seed=config.split_seed, maxIter=2 if small_grid else 20)
    elif algorithm == "linear_svc" and task == "classification":
        estimator = LinearSVC(**common, maxIter=5 if small_grid else 100)
        grid.addGrid(estimator.regParam, [0.1] if small_grid else [0.01, 0.1])
    elif algorithm == "linear_regression" and task == "regression":
        estimator = LinearRegression(**common, maxIter=5 if small_grid else 100)
        grid.addGrid(estimator.regParam, [0.1] if small_grid else [0.01, 0.1])
        grid.addGrid(estimator.elasticNetParam, [0.5] if small_grid else [0.0, 0.5, 1.0])
    elif algorithm == "glm" and task == "regression":
        estimator = GeneralizedLinearRegression(labelCol="__positive_delay", featuresCol="features_pca", family="gamma", link="log", maxIter=5 if small_grid else 100)
        grid.addGrid(estimator.regParam, [0.1] if small_grid else [0.01, 0.1])
    else:
        raise ValueError(f"Unsupported {task} algorithm: {algorithm}")
    if algorithm in {"random_forest", "gbt"}:
        grid.addGrid(estimator.maxDepth, [2] if small_grid else [5, 10])
        grid.addGrid(estimator.maxBins, [32] if small_grid else [32, 64])
    pipeline = build_end_to_end_pipeline(task, config, estimator=estimator)
    if algorithm == "glm":
        stages = pipeline.getStages()
        pipeline.setStages([*stages[:-1], RegressionOffsetTransformer(offset=glm_offset), stages[-1], RegressionOffsetTransformer(offset=glm_offset, inverse=True)])
    evaluator = (
        BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        if task == "classification" else RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse")
    )
    return CrossValidator(estimator=pipeline, estimatorParamMaps=grid.build(), evaluator=evaluator, numFolds=folds, parallelism=4, seed=config.split_seed), estimator


def run_training(spark, args):
    import mlflow
    import mlflow.pyspark.ml
    import mlflow.spark
    from mlflow import MlflowClient
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema
    from inference import input_schema

    config = PipelineConfig(pca_k=args.pca_k, split_seed=args.seed)
    eligible = prepare_eligible_flights(spark.read.parquet(str(args.input)), config)
    train, test = split_train_test(eligible, config)
    train, test = train.cache(), test.cache()
    try:
        counts = {"train": train.count(), "test": test.count()}
        if counts["train"] < args.folds or counts["test"] < 2:
            raise ValueError("Not enough eligible rows for cross-validation and a held-out evaluation")
        if args.task == "classification":
            if train.select("severe_delay").distinct().count() != 2 or test.select("severe_delay").distinct().count() != 2:
                raise ValueError("Both classification splits must contain both labels")
        if "glm" in args.algorithms:
            minimum = train.agg(F.min("ARRIVAL_DELAY")).first()[0]
            if minimum + args.glm_offset <= 0:
                raise ValueError("--glm-offset must make every training delay strictly positive for Gamma/log")
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        mlflow.pyspark.ml.autolog(log_models=False, log_datasets=False, silent=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.results_dir.mkdir(parents=True, exist_ok=False)
        # Save the exact held-out rows for independent benchmark reproduction.
        test.write.parquet(str(args.results_dir / "test_data"))
        records, winner = [], None
        with mlflow.start_run(run_name=f"{args.task}-tournament") as parent:
            mlflow.log_params({"task": args.task, "folds": args.folds, "parallelism": 4, "pca_k": args.pca_k, "split_seed": args.seed, **counts})
            for algorithm in args.algorithms:
                tournament, _ = make_tournament(args.task, config, algorithm, args.folds, args.small_grid, args.glm_offset)
                with mlflow.start_run(run_name=algorithm, nested=True):
                    fitted = tournament.fit(train)
                    values = [float(value) for value in fitted.avgMetrics]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"Non-finite CV scores for {algorithm}")
                    cv_score = max(values) if args.task == "classification" else min(values)
                    model = fitted.bestModel
                    scored = model.transform(test).cache()
                    try:
                        label = config.classification_label if args.task == "classification" else config.regression_label
                        metrics = evaluate_predictions(scored, args.task, label)
                    finally:
                        scored.unpersist()
                    directory = args.results_dir / algorithm
                    diagnostics = write_diagnostics(model, directory)
                    record = {"algorithm": algorithm, "cv_score": cv_score, "cv_metrics": values, "grid": [{param.name: value for param, value in params.items()} for params in tournament.getEstimatorParamMaps()], "test_metrics": metrics, "parameters": diagnostics["parameters"]}
                    records.append(record)
                    mlflow.log_params({"algorithm": algorithm, **diagnostics["parameters"]})
                    mlflow.log_metrics({"cv_score": cv_score, **{f"test_{key}": value for key, value in metrics.items()}})
                    mlflow.log_metrics({f"pca_variance_pc{i + 1}": value for i, value in enumerate(diagnostics["pca_explained_variance"])})
                    if "pca_feature_importances" in diagnostics:
                        mlflow.log_metrics({f"importance_{key}": value for key, value in diagnostics["pca_feature_importances"].items()})
                    mlflow.log_artifacts(str(directory), artifact_path="diagnostics")
                    better = winner is None or (cv_score > winner[0] if args.task == "classification" else cv_score < winner[0])
                    if better:
                        winner = (cv_score, algorithm, model)
            _, algorithm, model = winner
            model.write().save(str(args.output))
            schema = input_schema(model)
            signature = ModelSignature(inputs=Schema([ColSpec("string" if field.dataType.simpleString() == "string" else "double", field.name, required=False) for field in schema]), outputs=Schema([ColSpec("double", "prediction")]))
            info = mlflow.spark.log_model(model, artifact_path="full_pipeline", signature=signature, code_paths=[str(ROOT / "mllib_pipeline.py"), str(ROOT / "custom_transformers.py")], pip_requirements=str(ROOT / "requirements.txt"))
            name = args.registered_name or f"FlightDelayFullPipeline{args.task.title()}"
            version = mlflow.register_model(info.model_uri, name)
            client = MlflowClient()
            for stage in ("Staging", "Production"):
                client.set_registered_model_alias(name, stage.lower(), version.version)
                client.set_model_version_tag(name, version.version, stage, "validated")
            client.set_model_version_tag(name, version.version, "pipeline_scope", "raw-input-full-pipeline")
            report = {"task": args.task, "input": str(args.input.resolve()), "split": {"seed": args.seed, **counts}, "folds": args.folds, "parallelism": 4, "selection": "training cross-validation only", "winner": algorithm, "candidates": records, "model": str(args.output), "mlflow": {"run_id": parent.info.run_id, "name": name, "version": version.version}}
            report_path = args.results_dir / "tournament.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            mlflow.log_artifact(str(report_path))
            print(json.dumps(report, indent=2))
            return report
    finally:
        train.unpersist()
        test.unpersist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(ALGORITHMS), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Typed raw flights Parquet")
    parser.add_argument("--output", type=Path, required=True, help="New full PipelineModel directory")
    parser.add_argument("--results-dir", type=Path, required=True, help="New metrics, plots and held-out data directory")
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pca-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--glm-offset", type=float, default=1440.0)
    parser.add_argument("--small-grid", action="store_true", help="Smoke-test resources; not a full benchmark")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}")
    parser.add_argument("--experiment", default="flight-delay-full-pipeline")
    parser.add_argument("--registered-name")
    args = parser.parse_args()
    args.algorithms = args.algorithms or list(ALGORITHMS[args.task])
    if args.output.exists() or args.results_dir.exists():
        parser.error("Model and results paths must be new")
    if not args.input.is_dir() or args.folds < 2 or args.pca_k < 1:
        parser.error("Input must exist, folds >= 2, and pca-k >= 1")
    if not set(args.algorithms).issubset(ALGORITHMS[args.task]) or len(args.algorithms) != len(set(args.algorithms)):
        parser.error(f"Select distinct algorithms from {ALGORITHMS[args.task]}")
    configure_pyspark_python()
    spark = configure_local_filesystem(SparkSession.builder.appName(f"flight-{args.task}-tournament")).getOrCreate()
    try:
        run_training(spark, args)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
