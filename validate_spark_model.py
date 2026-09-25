"""Fresh-process reload and small OOT scoring checks for either Spark model."""

import argparse
import tempfile
import zipfile
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import SparkSession, functions as F

from feature_contract import SELECTED_FEATURES


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_MODELS = {
    "classification": ROOT / "models/random_forest_final_spark",
    "regression": ROOT / "models/random_forest_regression_final_spark",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(DEFAULT_MODELS), default="classification")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--oot-zip", type=Path, default=PROJECT_ROOT / "OOT_Data.zip")
    parser.add_argument("--rows", type=int, default=20)
    args = parser.parse_args()
    model_path = args.model or DEFAULT_MODELS[args.task]
    if not model_path.is_dir():
        parser.error(f"Spark PipelineModel is missing: {model_path}")
    if not args.oot_zip.is_file():
        parser.error(f"OOT ZIP is missing: {args.oot_zip}")
    if args.rows < 1:
        parser.error("--rows must be positive")

    spark = SparkSession.builder.appName(f"Validate{args.task.title()}Model").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        model = PipelineModel.load(str(model_path))
        stages = [type(stage).__name__ for stage in model.stages]
        expected_type = (
            RandomForestClassificationModel
            if args.task == "classification"
            else RandomForestRegressionModel
        )
        if len(model.stages) != 2 or not isinstance(model.stages[1], expected_type):
            raise ValueError(f"Unexpected pipeline stages: {stages}")
        actual = model.stages[0].getInputCols()
        if actual != SELECTED_FEATURES:
            raise ValueError(f"Model feature contract differs: {actual}")
        with tempfile.TemporaryDirectory(prefix="spark-oot-") as directory:
            with zipfile.ZipFile(args.oot_zip) as archive:
                archive.extract("OOT_Data/oot_data.parquet", directory)
            oot = spark.read.parquet(str(Path(directory) / "OOT_Data/oot_data.parquet"))
            selected = [F.col(name).cast("double").alias(name) for name in SELECTED_FEATURES]
            if args.task == "regression":
                selected.append(F.col("ARRIVAL_DELAY").cast("double").alias("ARRIVAL_DELAY"))
            prepared = (
                oot.withColumn(
                    "time_of_day_bucket_evening",
                    (F.col("time_of_day_bucket") == "evening").cast("double"),
                )
                .select(*selected)
                .fillna(0, subset=SELECTED_FEATURES)
                .limit(args.rows)
            )
            scored = model.transform(prepared)
            if "prediction" not in scored.columns:
                raise ValueError("Saved model did not produce prediction")
            if args.task == "classification":
                if "probability" not in scored.columns:
                    raise ValueError("Classification model did not produce probability")
                output = scored.select(
                    F.col("prediction").cast("int").alias("prediction"),
                    vector_to_array("probability")[1].alias("probability_late_over_30m"),
                ).collect()
                metrics = None
            else:
                output_df = scored.select(
                    "ARRIVAL_DELAY",
                    F.col("prediction").alias("predicted_arrival_delay_minutes"),
                ).cache()
                metrics = {
                    name.upper(): RegressionEvaluator(
                        labelCol="ARRIVAL_DELAY", predictionCol="prediction", metricName=name
                    ).evaluate(scored)
                    for name in ("rmse", "mae", "r2")
                }
                output = output_df.collect()
        print(f"Task: {args.task}")
        print(f"Pipeline stages: {stages}")
        print(f"Model feature count: {len(actual)}")
        print(f"Scored OOT rows: {len(output)}")
        if metrics:
            print(f"Small-subset metrics: {metrics}")
        for row in output[:5]:
            print(row.asDict())
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
