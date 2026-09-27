"""Validate the supplied regression PipelineModel; this script never fits a model."""

import argparse
import json
from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import SparkSession, functions as F


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models/random_forest_regression_final_spark"
DEFAULT_BASE = ROOT.parent / "huy/model-ready/base"
DEFAULT_RESULT = ROOT / "docs/regression_diagnostic_recheck.json"
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
    "rmse": 32.736849,
    "mae": 16.256044,
    "r2": 0.279717,
}
VALIDATION_TOLERANCE = {"rmse": 0.01, "mae": 0.01, "r2": 0.001}


def inspect_model(model):
    stages = [type(stage).__name__ for stage in model.stages]
    if len(model.stages) != 2:
        raise ValueError(f"Expected two stages, got {stages}")
    assembler, forest = model.stages
    if not isinstance(assembler, VectorAssembler):
        raise ValueError(f"First stage is not VectorAssembler: {stages}")
    if not isinstance(forest, RandomForestRegressionModel):
        raise ValueError(f"Second stage is not RandomForestRegressionModel: {stages}")
    actual = {
        "numTrees": forest.getNumTrees,
        "maxDepth": forest.getMaxDepth(),
        "minInstancesPerNode": forest.getMinInstancesPerNode(),
        "featureSubsetStrategy": forest.getFeatureSubsetStrategy(),
        "impurity": forest.getImpurity(),
        "bootstrap": forest.getBootstrap(),
        "seed": forest.getSeed(),
    }
    if actual != EXPECTED_PARAMS:
        raise ValueError(f"Saved RF parameters differ: {actual}")
    return stages, assembler.getInputCols(), actual


def prepare_evaluation_data(spark, base_path, feature_cols):
    source = spark.read.parquet(str(base_path))
    required = set(feature_cols) | {
        "YEAR", "MONTH", "DAY", "DIVERTED", "CANCELLED", "ARRIVAL_DELAY"
    }
    if "time_of_day_bucket_evening" in feature_cols:
        required.add("time_of_day_bucket")
        required.discard("time_of_day_bucket_evening")
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing evaluation columns: {missing}")

    flight_date = F.make_date(
        F.col("YEAR").cast("int"),
        F.col("MONTH").cast("int"),
        F.col("DAY").cast("int"),
    )
    base = (
        source.where(
            F.col("ARRIVAL_DELAY").isNotNull()
            & (F.col("DIVERTED") == 0)
            & (F.col("CANCELLED") == 0)
        )
        .where(flight_date > F.lit("2015-01-07").cast("date"))
        .orderBy(F.rand(42))
        .withColumn("NGAY_DL", flight_date)
    )
    if "time_of_day_bucket_evening" in feature_cols:
        base = base.withColumn(
            "time_of_day_bucket_evening",
            (F.col("time_of_day_bucket") == "evening").cast("double"),
        )
    base = base.withColumn("__row_id", F.monotonically_increasing_id()).persist(
        StorageLevel.DISK_ONLY
    )

    # Match upstream action order: materialize the persisted random ordering first.
    eligible_rows = base.count()
    train_valid = base.where(F.col("NGAY_DL") < F.lit("2015-11-01").cast("date"))
    _, valid = train_valid.randomSplit([0.8, 0.2], seed=42)
    oot = base.where(F.col("NGAY_DL") >= F.lit("2015-11-01").cast("date"))

    def frame(data, split):
        return (
            data.select(
                *[F.col(name).cast("double").alias(name) for name in feature_cols],
                F.col("ARRIVAL_DELAY").cast("double").alias("ARRIVAL_DELAY"),
            )
            .fillna(0, subset=feature_cols)
            .withColumn("evaluation_split", F.lit(split))
        )

    return base, frame(valid, "validation").unionByName(frame(oot, "oot")), eligible_rows


def metrics_for(scored, split):
    subset = scored.where(F.col("evaluation_split") == split)
    return {
        name: RegressionEvaluator(
            labelCol="ARRIVAL_DELAY", predictionCol="prediction", metricName=name
        ).evaluate(subset)
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

    spark = SparkSession.builder.appName("ValidateRegressionProductionArtifact").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        model = PipelineModel.load(str(args.model))
        stages, feature_cols, params = inspect_model(model)
        base, evaluation, eligible_rows = prepare_evaluation_data(spark, args.data_path, feature_cols)
        scored = model.transform(evaluation).select(
            "evaluation_split", "ARRIVAL_DELAY", "prediction"
        ).persist(StorageLevel.DISK_ONLY)
        counts = {
            row["evaluation_split"]: row["count"]
            for row in scored.groupBy("evaluation_split").count().collect()
        }
        validation = metrics_for(scored, "validation")
        oot = metrics_for(scored, "oot")
        deltas = {
            name: validation[name] - EXPECTED_VALIDATION[name]
            for name in EXPECTED_VALIDATION
        }
        agrees = all(
            abs(deltas[name]) <= VALIDATION_TOLERANCE[name]
            for name in EXPECTED_VALIDATION
        )
        result = {
            "fresh_process_reload": "passed",
            "pipeline_stages": stages,
            "feature_cols": feature_cols,
            "rf_params": params,
            "split": {
                "train_fraction": 0.8,
                "seed": 42,
                "oot_start": "2015-11-01",
            },
            "eligible_rows": eligible_rows,
            "rows": counts,
            "expected_validation_metrics": EXPECTED_VALIDATION,
            "validation_metrics": validation,
            "validation_deltas": deltas,
            "validation_matches_teammate": agrees,
            "validation_record": "teammate-authoritative",
            "local_validation_used_for_acceptance": False,
            "oot_metrics": oot,
        }
        args.result.write_text(json.dumps(result, indent=2) + "\n")
        print("REGRESSION_EVALUATION=" + json.dumps(result, sort_keys=True), flush=True)
        scored.unpersist(blocking=True)
        base.unpersist(blocking=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
