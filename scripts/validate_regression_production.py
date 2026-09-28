"""Reload and evaluate the fitted regression PipelineModel on held-out OOT data.

This script only calls ``PipelineModel.load`` and ``model.transform``. It never
fits or retrains a model. The teammate-confirmed validation metrics remain the
model-selection record; this script independently reproduces the final OOT test.
"""

import argparse
import json
from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import SparkSession, functions as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/random_forest_regression_final_spark"
DEFAULT_BASE = (
    ROOT / "data/model-ready/base"
    if (ROOT / "data/model-ready/base").exists()
    else ROOT.parent / "huy/model-ready/base"
)
DEFAULT_RESULT = ROOT / "docs/evaluations/regression_oot_recheck.json"
OOT_START = "2015-11-01"
EXPECTED_PARAMS = {
    "numTrees": 500,
    "maxDepth": 15,
    "minInstancesPerNode": 5,
    "featureSubsetStrategy": "sqrt",
    "impurity": "variance",
    "bootstrap": True,
    "seed": 42,
}
EXPECTED_VALIDATION = {
    "source": "standardized common 20% validation split",
    "rows": 935_697,
    "rmse": 32.736849,
    "mae": 16.256044,
    "r2": 0.279717,
}
EXPECTED_OOT = {
    "rows": 932_084,
    "rmse": 35.051807066478034,
    "mae": 17.10129574664888,
    "r2": 0.2677696539548666,
}
METRIC_TOLERANCE = 1e-6


def inspect_model(model):
    """Verify pipeline stage types and return the saved feature contract."""
    stage_names = [type(stage).__name__ for stage in model.stages]
    if len(model.stages) != 2:
        raise ValueError(f"Expected two stages, got {stage_names}")

    assembler, forest = model.stages
    if not isinstance(assembler, VectorAssembler):
        raise ValueError(f"First stage is not VectorAssembler: {stage_names}")
    if not isinstance(forest, RandomForestRegressionModel):
        raise ValueError(f"Second stage is not RF regression: {stage_names}")

    actual_params = {
        "numTrees": forest.getNumTrees,
        "maxDepth": forest.getMaxDepth(),
        "minInstancesPerNode": forest.getMinInstancesPerNode(),
        "featureSubsetStrategy": forest.getFeatureSubsetStrategy(),
        "impurity": forest.getImpurity(),
        "bootstrap": forest.getBootstrap(),
        "seed": forest.getSeed(),
    }
    if actual_params != EXPECTED_PARAMS:
        raise ValueError(f"Saved RF parameters differ: {actual_params}")

    return stage_names, assembler.getInputCols(), actual_params


def prepare_oot(spark, base_path, feature_cols):
    """Build the established OOT frame without reconstructing random splits."""
    source = spark.read.parquet(str(base_path))
    flight_date = F.make_date(
        F.col("YEAR").cast("int"),
        F.col("MONTH").cast("int"),
        F.col("DAY").cast("int"),
    )
    oot = source.where(
        F.col("ARRIVAL_DELAY").isNotNull()
        & (F.col("DIVERTED") == 0)
        & (F.col("CANCELLED") == 0)
        & (flight_date > F.lit("2015-01-07").cast("date"))
        & (flight_date >= F.lit(OOT_START).cast("date"))
    )
    if "time_of_day_bucket_evening" in feature_cols:
        oot = oot.withColumn(
            "time_of_day_bucket_evening",
            (F.col("time_of_day_bucket") == "evening").cast("double"),
        )

    return (
        oot.select(
            *[F.col(name).cast("double").alias(name) for name in feature_cols],
            F.col("ARRIVAL_DELAY").cast("double").alias("ARRIVAL_DELAY"),
        )
        .fillna(0, subset=feature_cols)
        .repartition(160)
        .persist(StorageLevel.DISK_ONLY)
    )


def evaluate(scored):
    """Calculate all final regression metrics from one persisted score frame."""
    return {
        name: RegressionEvaluator(
            labelCol="ARRIVAL_DELAY",
            predictionCol="prediction",
            metricName=name,
        ).evaluate(scored)
        for name in ("rmse", "mae", "r2")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()

    if not args.model.is_dir():
        parser.error(f"Regression PipelineModel is missing: {args.model}")
    if not args.data_path.is_dir():
        parser.error(f"Model-ready base is missing: {args.data_path}")
    if args.result.exists():
        parser.error(f"Refusing to overwrite existing result: {args.result}")

    spark = (
        SparkSession.builder.appName(
            "ValidateRegressionProductionArtifact"
        ).getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        model = PipelineModel.load(str(args.model))
        stages, feature_cols, params = inspect_model(model)
        oot = prepare_oot(spark, args.data_path, feature_cols)
        oot_rows = oot.count()
        scored = model.transform(oot).select(
            "ARRIVAL_DELAY", "prediction"
        ).persist(StorageLevel.DISK_ONLY)
        metrics = evaluate(scored)

        matches_expected = (
            oot_rows == EXPECTED_OOT["rows"]
            and all(
                abs(metrics[name] - EXPECTED_OOT[name]) <= METRIC_TOLERANCE
                for name in ("rmse", "mae", "r2")
            )
        )
        result = {
            "fresh_process_reload": "passed",
            "pipeline_stages": stages,
            "feature_cols": feature_cols,
            "rf_params": params,
            "selection_validation": EXPECTED_VALIDATION,
            "oot_start": OOT_START,
            "oot_rows": oot_rows,
            "oot_metrics": metrics,
            "matches_verified_oot": matches_expected,
        }
        args.result.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not matches_expected:
            raise RuntimeError("OOT result differs from the verified Production record")

        scored.unpersist(blocking=True)
        oot.unpersist(blocking=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
