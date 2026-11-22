"""Evaluate a saved raw-input pipeline and plot measured model diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.sql import SparkSession


def evaluate_predictions(scored, task: str, label_col: str) -> dict[str, float]:
    """Measure task metrics from predictions on a labeled evaluation dataset."""
    if task == "regression":
        metrics = {
            name: RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName=name).evaluate(scored)
            for name in ("rmse", "mae", "r2")
        }
    else:
        evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction")
        metrics = {
            "auc_roc": evaluator.evaluate(scored, {evaluator.metricName: "areaUnderROC"}),
            "auc_pr": evaluator.evaluate(scored, {evaluator.metricName: "areaUnderPR"}),
        }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluation produced non-finite metrics; check label coverage and dataset size")
    return metrics


def write_diagnostics(model: PipelineModel, output_dir: Path) -> dict:
    """Export fitted PCA variance and estimator importance without synthetic values."""
    from mllib_pipeline import explained_variance, save_pca_variance_plot
    output_dir.mkdir(parents=True, exist_ok=True)
    variances = explained_variance(model)
    save_pca_variance_plot(variances, output_dir / "pca_variance.png")
    estimator = next(stage for stage in reversed(model.stages) if stage.hasParam("featuresCol"))
    params = {
        param.name: estimator.getOrDefault(param)
        for param in estimator.params if estimator.isDefined(param)
    }
    result = {"algorithm": type(estimator).__name__, "parameters": params, "pca_explained_variance": variances}
    if hasattr(estimator, "featureImportances"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        values = [float(value) for value in estimator.featureImportances]
        names = [f"PC{i + 1}" for i in range(len(values))]
        result["pca_feature_importances"] = dict(zip(names, values))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(names, values)
        ax.set(xlabel="Fitted tree importance", ylabel="PCA component")
        fig.tight_layout()
        fig.savefig(output_dir / "feature_importance.png", dpi=160)
        plt.close(fig)
    (output_dir / "model_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    from inference import load_verified_model
    from mllib_pipeline import configure_local_filesystem, configure_pyspark_python
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Labeled raw typed Parquet evaluation data")
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must be new to preserve previous evaluations")
    configure_pyspark_python()
    spark = configure_local_filesystem(SparkSession.builder.appName("flight-model-benchmark")).getOrCreate()
    try:
        model = load_verified_model(args.model, args.task)
        scored = model.transform(spark.read.parquet(str(args.input))).cache()
        try:
            if not scored.take(1):
                raise ValueError("Evaluation data has no eligible flights")
            label = "severe_delay" if args.task == "classification" else "ARRIVAL_DELAY"
            metrics = evaluate_predictions(scored, args.task, label)
            diagnostics = write_diagnostics(model, args.output_dir)
            result = {"task": args.task, "model": str(args.model), "input": str(args.input), "rows": scored.count(), "metrics": metrics, **diagnostics}
            (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(metrics, indent=2))
        finally:
            scored.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
