"""Run a real Spark MLlib CV tournament and save only the winning full pipeline.

This command is intentionally not invoked by CI or submit_pipeline.sh: it
requires the project flight Parquet data and can be expensive.  It prints
measured CV results when executed; no benchmark values are embedded here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mllib_pipeline import (  # noqa: E402
    PipelineConfig,
    build_end_to_end_pipeline,
    configure_local_filesystem,
    configure_pyspark_python,
)


def make_tournament(task: str, config: PipelineConfig) -> tuple[CrossValidator, object]:
    label = config.classification_label if task == "classification" else config.regression_label
    forest = (
        RandomForestClassifier(labelCol=label, featuresCol="features_pca", seed=42)
        if task == "classification"
        else RandomForestRegressor(labelCol=label, featuresCol="features_pca", seed=42)
    )
    pipeline = build_end_to_end_pipeline(task, config=config, estimator=forest)
    grid = (
        ParamGridBuilder()
        .addGrid(forest.numTrees, [100, 300])
        .addGrid(forest.maxDepth, [8, 15])
        .addGrid(forest.minInstancesPerNode, [1, 5])
        .build()
    )
    evaluator = (
        BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        if task == "classification"
        else RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse")
    )
    return CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4,
        seed=42,
    ), forest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "regression"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Raw typed flights Parquet directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for best PipelineModel")
    parser.add_argument("--pca-k", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite evidence/model path: {args.output}")
    if not args.input.is_dir():
        parser.error(f"Input Parquet directory is missing: {args.input}")
    configure_pyspark_python()
    spark = configure_local_filesystem(SparkSession.builder.appName(f"flight-{args.task}-tournament")).getOrCreate()
    try:
        config = PipelineConfig(pca_k=args.pca_k)
        tournament, _ = make_tournament(args.task, config)
        result = tournament.fit(spark.read.parquet(str(args.input)))
        result.bestModel.write().save(str(args.output))
        print(f"Saved full {args.task} PipelineModel to {args.output}")
        print("Measured CV metrics (one per ParamGridBuilder candidate):", result.avgMetrics)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
