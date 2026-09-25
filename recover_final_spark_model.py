"""Fit only the fixed final Spark Random Forest from Huy's model-ready base.

Run with spark-submit after installing the project's Spark requirements. This
does not run feature filtering, hyperparameter search, or cross-validation.

The ordered feature contract below is the output of two complete runs of the
exact feature-selection functions in ``sfdws_model_spark.py`` on 3,739,905
training rows. In both runs, the missingness (0.7), concentration (0.9), and
IV-guided correlation (0.7) stages removed no candidate features. This differs
from the separate pandas notebook evidence showing 22 features; it is the
authoritative contract for this Spark recovery because it was reproduced with
the Spark code, data, split, and seed used here.
"""

import argparse
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, Window, functions as F


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = ROOT / "huy/model-ready/base"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "models/random_forest_final_spark"

from feature_contract import SELECTED_FEATURES


def prepare_training_data(spark, base_path):
    base = spark.read.parquet(str(base_path))
    required = set(SELECTED_FEATURES[:-1]) | {
        "YEAR", "MONTH", "DAY", "DIVERTED", "CANCELLED",
        "ARRIVAL_DELAY", "time_of_day_bucket",
    }
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    base = base.withColumn(
        "NGAY_DL", F.make_date(F.col("YEAR").cast("int"),
                               F.col("MONTH").cast("int"), F.col("DAY").cast("int"))
    )
    base = base.where(
        (F.col("DIVERTED") == 0) & (F.col("CANCELLED") == 0)
        & (F.col("NGAY_DL") > F.lit("2015-01-07").cast("date"))
        & (F.col("NGAY_DL") < F.lit("2015-11-01").cast("date"))
    )
    base = base.withColumn("label", (F.col("ARRIVAL_DELAY") > 30).cast("double"))
    base = base.withColumn(
        "time_of_day_bucket_evening",
        (F.col("time_of_day_bucket") == "evening").cast("double"),
    )
    # Match sfdws_model_spark._stratified_split: fit on the 80% train portion.
    order = Window.partitionBy("label").orderBy(F.rand(42))
    group = Window.partitionBy("label")
    train = (base.withColumn("__row_number", F.row_number().over(order))
             .withColumn("__class_count", F.count(F.lit(1)).over(group))
             .where(F.col("__row_number") <= F.floor(F.col("__class_count") * 0.8)))
    return train.select(
        *[F.col(name).cast("double").alias(name) for name in SELECTED_FEATURES], "label"
    ).fillna(0, subset=SELECTED_FEATURES)


def fit_final_model(train):
    # Exact effective Spark mapping of the fixed parameters at lines 543-546
    # in sfdws_model_spark.py. min_samples_split and n_jobs have no Spark RF
    # equivalent. Balanced class weights are computed on this train split.
    counts = train.groupBy("label").count()
    totals = counts.agg(F.sum("count").alias("n"), F.count("*").alias("k")).first()
    weights = counts.withColumn(
        "__weight", F.lit(float(totals["n"]))
        / (F.lit(float(totals["k"])) * F.col("count"))
    )
    weighted = train.join(weights.select("label", "__weight"), "label", "left")
    assembler = VectorAssembler(inputCols=SELECTED_FEATURES, outputCol="features", handleInvalid="keep")
    forest = RandomForestClassifier(
        labelCol="label", featuresCol="features", weightCol="__weight",
        predictionCol="prediction", probabilityCol="probability",
        rawPredictionCol="rawPrediction", numTrees=300, maxDepth=15,
        minInstancesPerNode=5, featureSubsetStrategy="sqrt", impurity="gini",
        bootstrap=True, seed=42,
    )
    return Pipeline(stages=[assembler, forest]).fit(weighted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.base.is_dir() or not list(args.base.glob("*.parquet")):
        parser.error(f"Model-ready Parquet directory is missing: {args.base}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing path: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"Output parent directory is missing: {args.output.parent}")

    spark = SparkSession.builder.appName("RecoverFinalSparkRandomForest").getOrCreate()
    try:
        spark.sparkContext.setLogLevel("WARN")
        train = prepare_training_data(spark, args.base).cache()
        training_rows = train.count()
        class_counts = {str(row["label"]): row["count"] for row in train.groupBy("label").count().collect()}
        print(f"Training rows: {training_rows}", flush=True)
        print(f"Class counts: {class_counts}", flush=True)
        print(f"Selected features ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}", flush=True)
        print("Starting the single final Pipeline.fit", flush=True)
        model = fit_final_model(train)  # The only fit in this script.
        model.write().save(str(args.output))  # No overwrite mode.
        print(f"Pipeline stages: {[type(stage).__name__ for stage in model.stages]}", flush=True)
        print(f"Saved fitted PipelineModel to {args.output}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
